from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID, PHASE_A_CONTRACT_VERSION
from ml_lab.adapters.phase_a_provider import (
    PhaseALocalProviderError,
    PhaseALocalProviderRunner,
    validate_loopback_endpoint,
)
from ml_lab.adapters.phase_a_reference import (
    PhaseAReferenceValidator,
    ReferencePreflightReceipt,
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
PHASE_A_LOCAL_PROVIDER_DEV_PREFIX = "phase-a-local-provider-dev-v2"


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
    config_model = cast(str, config["model"])
    config_endpoint = cast(str, config["endpoint"])
    config_timeout = cast(float, config["timeout_seconds"])
    baseline_id = local_provider_baseline_id(
        model=config_model,
        endpoint=config_endpoint,
        timeout_seconds=config_timeout,
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
            "reproducibility": {
                "mode": "NONDETERMINISTIC",
                "comparison_policy": "VETO_EXACT_NON_VETO_REPORT_ONLY",
                "metric_tolerances": {
                    "contract_failures": 0.0,
                    "false_commitments": 0.0,
                    "hidden_or_out_of_envelope": 0.0,
                    "unsupported_mechanics_authority": 0.0,
                    "zero_model_route_violations": 0.0,
                },
            },
        },
    )
    runner = PhaseALocalProviderRunner(
        workspace,
        repository=Path(snapshot.repo_path),
        ref=snapshot.commit_sha,
        model=config_model,
        endpoint=config_endpoint,
        timeout_seconds=config_timeout,
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
        reference_preflight=lambda request: reference.preflight(
            repository=Path(snapshot.repo_path),
            ref=snapshot.commit_sha,
            request=request,
        ),
    )
    return baseline.run(
        experiment.id,
        evaluator=evaluator,
        splits=(DatasetSplit.TEST, DatasetSplit.REDTEAM),
        cancelled=cancelled,
    )


def run_phase_a_local_provider_dev_baseline(
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
    config_model = cast(str, config["model"])
    config_endpoint = cast(str, config["endpoint"])
    config_timeout = cast(float, config["timeout_seconds"])
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:12]
    baseline_id = f"{PHASE_A_LOCAL_PROVIDER_DEV_PREFIX}-{digest}"

    snapshots = ContractSnapshotService(workspace)
    snapshot = snapshots.get(contract_snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Pinned contract snapshot belongs to a different project.")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("DEV provider baseline requires a Phase A contract snapshot.")
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
            "evaluation_scope": "DEV_ONLY",
            "protected_evidence": False,
        },
        environment={
            "reference_commit_sha": snapshot.commit_sha,
            "reference_repository_identity": snapshot.repo_identity,
            "provider_source": "PINNED_COMMITTED_BYTES",
            "integration_gate": "NO_GO",
            "reproducibility": {
                "mode": "NONDETERMINISTIC",
                "comparison_policy": "VETO_EXACT_NON_VETO_REPORT_ONLY",
                "metric_tolerances": {
                    "contract_failures": 0.0,
                    "false_commitments": 0.0,
                    "hidden_or_out_of_envelope": 0.0,
                    "unsupported_mechanics_authority": 0.0,
                    "zero_model_route_violations": 0.0,
                },
            },
        },
    )
    runner = PhaseALocalProviderRunner(
        workspace,
        repository=Path(snapshot.repo_path),
        ref=snapshot.commit_sha,
        model=config_model,
        endpoint=config_endpoint,
        timeout_seconds=config_timeout,
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
        reference_preflight=lambda request: reference.preflight(
            repository=Path(snapshot.repo_path),
            ref=snapshot.commit_sha,
            request=request,
        ),
    )
    result = baseline.run(
        experiment.id,
        evaluator=evaluator,
        splits=(DatasetSplit.DEV,),
        cancelled=cancelled,
    )

    evaluation = baseline.evaluation
    if evaluation.case_count(result.experiment.id, DatasetSplit.TEST) != 0:
        raise RuntimeError("DEV provider probe created TEST evidence")
    if evaluation.case_count(result.experiment.id, DatasetSplit.REDTEAM) != 0:
        raise RuntimeError("DEV provider probe created REDTEAM evidence")
    return result


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
    reference_preflight: Callable[
        [Mapping[str, object]],
        ReferencePreflightReceipt,
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
        reference_preflight=reference_preflight,
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

        observed_value = outcome.observed
        if isinstance(observed_value, dict) and last_provider:
            observed_value = {**observed_value, "provider": dict(last_provider)}
        evidence_value = outcome.evidence
        if evidence_value is not None and last_provider:
            evidence_value = {**dict(evidence_value), "provider": dict(last_provider)}
        return replace(
            outcome,
            observed=observed_value,
            evidence=evidence_value,
        )

    return evaluate
