from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_CONTRACT_VERSION,
    PhaseAResidualAdapter,
)
from ml_lab.adapters.phase_a_provider import (
    LocalProviderProposal,
    PhaseALocalProviderRunner,
)
from ml_lab.baselines.phase_a_provider import (
    _provider_evaluator,
    local_provider_config,
)
from ml_lab.baselines.service import BaselineRunResult, BaselineService
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.phase_a_candidate_sparse import (
    _ask_player,
    _declaration,
    _deterministic_abstention,
    _required_text,
    _slot_candidate_keys,
    _slot_specs,
    _string_values,
    _unique_explicit_candidate,
)

PHASE_A_BOUNDED_PROVIDER_SIGNAL_DEV_PREFIX = "phase-a-bounded-provider-signal-dev-v1"


def assemble_bounded_provider_signal(
    request: Mapping[str, object],
    raw_proposal: Mapping[str, object],
) -> dict[str, object]:
    adapter = PhaseAResidualAdapter()
    adapter.validate_exported_request(request)

    deterministic = _deterministic_abstention(request)
    if deterministic is not None:
        result = dict(deterministic)
        result["reason_code"] = "ML_LAB_BOUNDED_PROVIDER_SIGNAL_DETERMINISTIC_ABSTAIN"
        return result

    permitted = set(_string_values(request.get("permitted_decisions"), "permitted_decisions"))
    if raw_proposal.get("decision") != "RESOLVE" or "RESOLVE" not in permitted:
        result = _ask_player(request, missing_slots=("ACTION_FAMILY",))
        result["reason_code"] = "ML_LAB_BOUNDED_PROVIDER_SIGNAL_ABSTAIN"
        return result

    family = raw_proposal.get("action_family")
    allowed_families = set(
        _string_values(request.get("allowed_action_families"), "allowed_action_families")
    )
    if not isinstance(family, str) or family not in allowed_families:
        result = _ask_player(request, missing_slots=("ACTION_FAMILY",))
        result["reason_code"] = "ML_LAB_BOUNDED_PROVIDER_SIGNAL_ABSTAIN"
        return result

    raw_slots = raw_proposal.get("slots", [])
    provider_slots: dict[str, str] = {}
    if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)):
        for raw in raw_slots:
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            value = raw.get("value")
            if isinstance(name, str) and isinstance(value, str) and name not in provider_slots:
                provider_slots[name] = value

    declaration = _declaration(request)
    supplied: list[dict[str, str]] = []
    missing: list[str] = []
    for spec in _slot_specs(request, family):
        name = _required_text(spec.get("name"), "family_slots.slots.name")
        prebound = spec.get("prebound")
        if isinstance(prebound, str):
            supplied.append({"name": name, "value": prebound})
            continue

        candidate_keys = _slot_candidate_keys(request, spec)
        selected: str | None = None
        if candidate_keys:
            selected = _unique_explicit_candidate(request, declaration, candidate_keys)
        else:
            allowed_values = set(
                _string_values(spec.get("allowed_values", []), "allowed_values")
            )
            provider_value = provider_slots.get(name)
            if provider_value in allowed_values:
                selected = provider_value

        required = spec.get("required", True) is not False
        if selected is not None:
            supplied.append({"name": name, "value": selected})
        elif required:
            missing.append(name)

    if missing:
        result = _ask_player(
            request,
            action_family=family,
            slots=supplied,
            missing_slots=tuple(missing),
        )
        result["reason_code"] = "ML_LAB_BOUNDED_PROVIDER_SIGNAL_ABSTAIN"
        return result

    proposal: dict[str, object] = {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": supplied,
        "reason_code": "ML_LAB_BOUNDED_PROVIDER_SIGNAL",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }
    check = adapter.validate_proposal(request, proposal)
    if not check.accepted:
        result = _ask_player(request, missing_slots=("ACTION_FAMILY",))
        result["reason_code"] = "ML_LAB_BOUNDED_PROVIDER_SIGNAL_ABSTAIN"
        return result
    return proposal


class PhaseABoundedProviderSignalRunner(PhaseALocalProviderRunner):
    def propose(self, request: object) -> LocalProviderProposal:
        if not isinstance(request, Mapping):
            raise ValueError("Bounded provider signal request must be an object")
        raw = super().propose(request)
        bounded = assemble_bounded_provider_signal(request, raw.proposal)
        return LocalProviderProposal(
            proposal=bounded,
            receipt_artifact_digest=raw.receipt_artifact_digest,
            latency_ms=raw.latency_ms,
        )


def run_phase_a_bounded_provider_signal_dev(
    workspace: Workspace,
    *,
    project_id: str,
    dataset_id: str,
    contract_snapshot_id: str,
    model: str,
    endpoint: str,
    timeout_seconds: float = 120.0,
) -> BaselineRunResult:
    config = local_provider_config(
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )
    clean_model = cast(str, config["model"])
    clean_endpoint = cast(str, config["endpoint"])
    clean_timeout = cast(float, config["timeout_seconds"])

    snapshots = ContractSnapshotService(workspace)
    snapshot = snapshots.get(contract_snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Pinned contract snapshot belongs to a different project.")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("Bounded provider signal requires a Phase A contract snapshot.")
    if snapshot.contract_version != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Pinned Phase A contract version is incompatible.")
    baseline = BaselineService(workspace)
    digest = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
    experiment = baseline.create(
        project_id=project_id,
        dataset_id=dataset_id,
        baseline_id=f"{PHASE_A_BOUNDED_PROVIDER_SIGNAL_DEV_PREFIX}-{digest}",
        contract_snapshot_id=contract_snapshot_id,
        config={
            **config,
            "evaluation_scope": "DEV_ONLY",
            "provider_role": "SEMANTIC_SIGNAL_ONLY",
            "entity_binding": "CURRENT_VISIBLE_ENVELOPE_EXPLICIT_MENTION_ONLY",
            "authority_fields_from_provider": False,
        },
        environment={
            "reference_commit_sha": snapshot.commit_sha,
            "reference_repository_identity": snapshot.repo_identity,
            "provider_source": "PINNED_COMMITTED_BYTES",
            "integration_gate": "NO_GO",
        },
    )
    runner = PhaseABoundedProviderSignalRunner(
        workspace,
        repository=Path(snapshot.repo_path),
        ref=snapshot.commit_sha,
        model=clean_model,
        endpoint=clean_endpoint,
        timeout_seconds=clean_timeout,
    )
    from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator

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
    )
    if baseline.evaluation.case_count(result.experiment.id, DatasetSplit.TEST) != 0:
        raise RuntimeError("Bounded provider signal created TEST evidence")
    if baseline.evaluation.case_count(result.experiment.id, DatasetSplit.REDTEAM) != 0:
        raise RuntimeError("Bounded provider signal created REDTEAM evidence")
    return result
