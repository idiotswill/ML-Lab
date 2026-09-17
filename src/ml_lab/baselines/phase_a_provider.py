from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID, PHASE_A_CONTRACT_VERSION
from ml_lab.adapters.phase_a_provider import (
    PhaseALocalProviderError,
    PhaseALocalProviderRunner,
    validate_loopback_endpoint,
)
from ml_lab.adapters.phase_a_reference import (
    PhaseAReferenceValidator,
    ReferenceValidationReceipt,
)
from ml_lab.baselines.service import BaselineRunResult, BaselineService
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, ExperimentStatus, FailureSeverity
from ml_lab.datasets.leakage import canonical_json
from ml_lab.evaluation.phase_a import make_phase_a_case_evaluator
from ml_lab.evaluation.service import CaseEvaluator, CaseOutcome
from ml_lab.storage.workspace import Workspace

PHASE_A_LOCAL_PROVIDER_PREFIX = "phase-a-local-provider-v2"


def local_provider_baseline_id(
    *,
    model: str,
    endpoint: str,
    timeout_seconds: float,
) -> str:
    config = local_provider_config(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]
    return f"{PHASE_A_LOCAL_PROVIDER_PREFIX}-{digest}"


def local_provider_config(
    *,
    model: str,
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, object]:
    clean_model = model.strip()
    if not clean_model:
        raise ValueError("A local model name is required.")
    clean_endpoint = validate_loopback_endpoint(endpoint)
    timeout = max(1.0, min(float(timeout_seconds), 900.0))
    return {
        "provider_kind": "pinned-frankenhomie-local-chat-completions",
        "model": clean_model,
        "endpoint": clean_endpoint,
        "timeout_seconds": timeout,
        "max_tokens": 700,
        "network_policy": "LOOPBACK_HTTP_ONLY",
    }


def run_phase_a_local_provider_baseline(
    workspace: Workspace,
    *,
    project_id: str,
    dataset_id: str,
    contract_snapshot_id: str,
    model: str,
    endpoint: str,
    timeout_seconds: float = 120.0,
    cancelled: Callable[[], bool] | None = None,
) -> BaselineRunResult:
    config = local_provider_config(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )
    baseline_id = local_provider_baseline_id(
        model=str(config["model"]),
        endpoint=str(config["endpoint"]),
        timeout_seconds=float(config["timeout_seconds"]),
    )
    snapshots = ContractSnapshotService(workspace)
    snapshot = snapshots.get(contract_snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Pinned contract snapshot belongs to a different project.")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("Local provider baseline requires a Phase A contract snapshot.")
    if snapshot.contract_version != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Pinned Phase A contract version is incompatible.")

    baseline = BaselineService(workspace)
    experiment = baseline.create(
        project_id=project_id,
        dataset_id=dataset_id,
        baseline_id=baseline_id,
        contract_snapshot_id=contract_snapshot_id,
        config={
            **config,
            "adapter_id": PHASE_A_ADAPTER_ID,
            "contract_version": PHASE_A_CONTRACT_VERSION,
            "compatibility_signature": snapshot.compatibility_signature,
        },
        environment={
            "reference_commit_sha": snapshot.commit_sha,
            "reference_repository_identity": snapshot.repo_identity,
            "provider_source": "PINNED_COMMITTED_BYTES",
            "integration_gate": "NO_GO",
        },
    )
    runner = PhaseALocalProviderRunner(
        workspace,
        repository=Path(snapshot.repo_path),
        ref=snapshot.commit_sha,
        model=str(config["model"]),
        endpoint=str(config["endpoint"]),
        timeout_seconds=float(config["timeout_seconds"]),
    )
    reference = PhaseAReferenceValidator(workspace)
    evaluator = _provider_evaluator(
        runner,
        reference_check=lambda request, proposal: reference.validate(
            repository=Path(snapshot.repo_path),
            ref=snapshot.commit_sha,
            request=request,
            proposal=proposal,
        ),
    )
    return baseline.run(
        experiment.id,
        evaluator=evaluator,
        splits=(DatasetSplit.TEST, DatasetSplit.REDTEAM),
        cancelled=cancelled,
    )


def completed_local_provider_baseline(
    workspace: Workspace,
    *,
    project_id: str,
    dataset_id: str,
    contract_snapshot_id: str,
    model: str,
    endpoint: str,
    timeout_seconds: float,
) -> str | None:
    baseline_id = local_provider_baseline_id(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )
    trainer_id = f"baseline:{baseline_id}"
    for item in BaselineService(workspace).experiments.list_for_project(project_id):
        if (
            item.dataset_id == dataset_id
            and item.contract_snapshot_id == contract_snapshot_id
            and item.trainer_id == trainer_id
            and item.status is ExperimentStatus.COMPLETED
        ):
            return item.id
    return None


def _provider_evaluator(
    runner: PhaseALocalProviderRunner,
    *,
    reference_check: Callable[
        [Mapping[str, object], Mapping[str, object]],
        ReferenceValidationReceipt,
    ],
) -> CaseEvaluator:
    last_provider: dict[str, object] = {}

    def predict(request: Mapping[str, object]) -> Mapping[str, object]:
        result = runner.propose(request)
        last_provider.clear()
        last_provider.update(
            {
                "receipt_artifact_digest": result.receipt_artifact_digest,
                "provider_latency_ms": result.latency_ms,
                "model": runner.model,
                "endpoint": runner.endpoint,
                "network_policy": "LOOPBACK_HTTP_ONLY",
            }
        )
        return result.proposal

    base = make_phase_a_case_evaluator(
        predictor=predict,
        reference_check=reference_check,
    )

    def evaluate(row: Mapping[str, object]) -> CaseOutcome:
        last_provider.clear()
        try:
            outcome = base(row)
        except PhaseALocalProviderError as exc:
            observed = {
                "status": "LOCAL_PROVIDER_ERROR",
                "error_code": exc.error_code,
                "message": str(exc),
                "provider_receipt_artifact_digest": exc.receipt_artifact_digest,
                "model": runner.model,
                "endpoint": runner.endpoint,
            }
            return CaseOutcome(
                observed=observed,
                correct=False,
                latency_ms=-1.0,
                failure_kind="CONTRACT_FAILURE",
                failure_severity=FailureSeverity.VETO,
                evidence={
                    "provider_error_code": exc.error_code,
                    "provider_receipt_artifact_digest": exc.receipt_artifact_digest,
                    "network_policy": "LOOPBACK_HTTP_ONLY",
                    "authority_mutation_attempted": False,
                },
            )

        observed = outcome.observed
        if isinstance(observed, dict) and last_provider:
            observed = {**observed, "provider": dict(last_provider)}
        evidence = outcome.evidence
        if evidence is not None and last_provider:
            evidence = {**dict(evidence), "provider": dict(last_provider)}
        return replace(outcome, observed=observed, evidence=evidence)

    return evaluate
