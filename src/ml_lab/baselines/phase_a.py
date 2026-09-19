from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_CONTRACT_VERSION,
    PhaseAResidualAdapter,
)
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.baselines.service import BaselineRunResult, BaselineService
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, ExperimentStatus
from ml_lab.evaluation.phase_a import make_phase_a_case_evaluator
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.phase_a_sparse import BoundedPredictionError

PHASE_A_ABSTENTION_BASELINE_ID = "phase-a-deterministic-abstention-v2"
PHASE_A_LEXICAL_BASELINE_ID = "phase-a-bounded-lexical-v2"


@dataclass(frozen=True, slots=True)
class PhaseABaselineOption:
    baseline_id: str
    display_name: str
    description: str


@dataclass(frozen=True, slots=True)
class _RegistryFamily:
    family: str
    entity_slot: str | None
    aliases: tuple[str, ...]
    operation_aliases: dict[str, tuple[str, ...]]


_BASELINE_OPTIONS = (
    PhaseABaselineOption(
        baseline_id=PHASE_A_ABSTENTION_BASELINE_ID,
        display_name="Deterministic abstention",
        description="Always ASK_PLAYER inside the exported residual envelope.",
    ),
    PhaseABaselineOption(
        baseline_id=PHASE_A_LEXICAL_BASELINE_ID,
        display_name="Bounded lexical",
        description=(
            "Pinned registry aliases plus visible request candidates; abstains on "
            "missing or ambiguous required slots."
        ),
    ),
)


def phase_a_baseline_options(adapter_id: str) -> tuple[PhaseABaselineOption, ...]:
    return _BASELINE_OPTIONS if adapter_id == PHASE_A_ADAPTER_ID else ()


def run_phase_a_baseline(
    workspace: Workspace,
    *,
    project_id: str,
    dataset_id: str,
    contract_snapshot_id: str,
    baseline_id: str,
    cancelled: Callable[[], bool] | None = None,
) -> BaselineRunResult:
    snapshot_service = ContractSnapshotService(workspace)
    snapshot = snapshot_service.get(contract_snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Pinned contract snapshot belongs to a different project.")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("Phase A baseline requires a Phase A contract snapshot.")
    if snapshot.contract_version != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Pinned Phase A contract version is incompatible.")

    option_ids = {item.baseline_id for item in _BASELINE_OPTIONS}
    if baseline_id not in option_ids:
        raise ValueError(f"Unknown Phase A baseline {baseline_id!r}.")

    baseline = BaselineService(workspace)
    experiment = baseline.create(
        project_id=project_id,
        dataset_id=dataset_id,
        baseline_id=baseline_id,
        contract_snapshot_id=contract_snapshot_id,
        config={
            "adapter_id": PHASE_A_ADAPTER_ID,
            "contract_version": PHASE_A_CONTRACT_VERSION,
            "compatibility_signature": snapshot.compatibility_signature,
        },
        environment={
            "reference_commit_sha": snapshot.commit_sha,
            "reference_repository_identity": snapshot.repo_identity,
        },
    )

    reference = PhaseAReferenceValidator(workspace)
    repository = Path(snapshot.repo_path)
    predictor: Callable[[Mapping[str, object]], Mapping[str, object]]
    if baseline_id == PHASE_A_ABSTENTION_BASELINE_ID:
        predictor = _deterministic_abstention
    else:
        registry = _load_registry(workspace, contract_snapshot_id)
        predictor = _lexical_predictor(registry)

    evaluator = make_phase_a_case_evaluator(
        predictor=predictor,
        reference_check=lambda request, proposal: reference.validate(
            repository=repository,
            ref=snapshot.commit_sha,
            request=request,
            proposal=proposal,
        ),
        reference_preflight=lambda request: reference.preflight(
            repository=repository,
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


def completed_phase_a_baseline(
    workspace: Workspace,
    *,
    project_id: str,
    dataset_id: str,
    contract_snapshot_id: str,
    baseline_id: str,
) -> str | None:
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


def _deterministic_abstention(
    request: Mapping[str, object],
) -> Mapping[str, object]:
    return _ask_player(
        request,
        action_family=None,
        slots=(),
        missing_slots=("ACTION_FAMILY",),
        reason_code="ML_LAB_DETERMINISTIC_ABSTENTION",
    )


def _lexical_predictor(
    registry: dict[str, _RegistryFamily],
) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
    def predict(request: Mapping[str, object]) -> Mapping[str, object]:
        adapter = PhaseAResidualAdapter()
        adapter.validate_exported_request(request)
        text = _declaration(request).casefold()
        allowed_families = set(
            _string_values(request.get("allowed_action_families"), "allowed_action_families")
        )
        matched = [
            family
            for family, entry in sorted(registry.items())
            if family in allowed_families
            and any(_contains_phrase(text, alias) for alias in entry.aliases)
        ]
        if len(matched) != 1:
            return _ask_player(
                request,
                action_family=None,
                slots=(),
                missing_slots=("ACTION_FAMILY",),
                reason_code="ML_LAB_LEXICAL_FAMILY_AMBIGUOUS",
            )

        family = matched[0]
        entry = registry[family]
        candidates = _candidate_names(request)
        supplied: list[dict[str, str]] = []
        missing: list[str] = []
        for spec in _slot_specs(request, family):
            name = _required_text(spec.get("name"), "family_slots.slots.name")
            allowed = _string_values(spec.get("allowed_values", []), "allowed_values")
            prebound = spec.get("prebound")
            selected: str | None = None
            if isinstance(prebound, str):
                selected = prebound
            else:
                selected = _match_operation(text, allowed, entry.operation_aliases)
                if selected is None:
                    selected = _match_candidate(text, allowed, candidates)
            if selected is not None:
                supplied.append({"name": name, "value": selected})
            elif spec.get("required", True) is not False:
                missing.append(name)

        if missing:
            return _ask_player(
                request,
                action_family=family,
                slots=supplied,
                missing_slots=tuple(missing),
                reason_code="ML_LAB_LEXICAL_SLOT_AMBIGUOUS",
            )

        proposal: dict[str, object] = {
            "version": PHASE_A_CONTRACT_VERSION,
            "decision": "RESOLVE",
            "action_family": family,
            "entity_keys": [],
            "slots": supplied,
            "reason_code": "ML_LAB_BOUNDED_LEXICAL",
            "facts_used": [],
            "question": None,
            "missing_slots": [],
            "candidate_keys": [],
        }
        check = adapter.validate_proposal(request, proposal)
        if not check.accepted:
            raise BoundedPredictionError(
                f"Lexical baseline assembly failed adapter validation: {check.error_code}"
            )
        return proposal

    return predict


def _ask_player(
    request: Mapping[str, object],
    *,
    action_family: str | None,
    slots: Sequence[Mapping[str, str]],
    missing_slots: Sequence[str],
    reason_code: str,
) -> Mapping[str, object]:
    permitted = set(
        _string_values(request.get("permitted_decisions"), "permitted_decisions")
    )
    if "ASK_PLAYER" not in permitted:
        raise BoundedPredictionError(
            "The exported residual request does not permit safe baseline abstention."
        )
    proposal: dict[str, object] = {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "ASK_PLAYER",
        "action_family": action_family,
        "entity_keys": [],
        "slots": [dict(item) for item in slots],
        "reason_code": reason_code,
        "facts_used": [],
        "question": "Which action or target did you mean?",
        "missing_slots": list(dict.fromkeys(missing_slots)),
        "candidate_keys": [],
    }
    check = PhaseAResidualAdapter().validate_proposal(request, proposal)
    if not check.accepted:
        raise BoundedPredictionError(
            f"Baseline abstention failed adapter validation: {check.error_code}"
        )
    return proposal


def _load_registry(
    workspace: Workspace,
    snapshot_id: str,
) -> dict[str, _RegistryFamily]:
    manifest = ContractSnapshotService(workspace).manifest(snapshot_id)
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Contract manifest files must be a list.")
    registry_digest: str | None = None
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            continue
        if raw_file.get("role") != "semantic-registry":
            continue
        digest = raw_file.get("sha256")
        if isinstance(digest, str) and digest:
            registry_digest = digest
            break
    if registry_digest is None:
        raise RuntimeError("Pinned contract snapshot has no semantic registry artifact.")
    decoded = json.loads(
        workspace.artifacts.resolve(registry_digest).read_text(encoding="utf-8")
    )
    if not isinstance(decoded, dict):
        raise ValueError("Pinned semantic registry must be a JSON object.")
    if decoded.get("schema_version") != "asterra-semantic-family-registry/1":
        raise ValueError("Unsupported pinned semantic registry schema.")
    raw_families = decoded.get("families")
    if not isinstance(raw_families, list):
        raise ValueError("Pinned semantic registry families must be a list.")

    result: dict[str, _RegistryFamily] = {}
    for raw_family in raw_families:
        if not isinstance(raw_family, dict):
            raise ValueError("Pinned semantic registry family must be an object.")
        family = _required_text(raw_family.get("family"), "family")
        aliases = _string_values(
            [
                *_list_value(raw_family.get("deterministic_aliases", [])),
                *_list_value(raw_family.get("residual_hints", [])),
            ],
            "aliases",
        )
        raw_operations = raw_family.get("operation_aliases", {})
        if not isinstance(raw_operations, dict):
            raise ValueError("operation_aliases must be an object.")
        operations = {
            str(value): _string_values(aliases_value, "operation aliases")
            for value, aliases_value in raw_operations.items()
        }
        entity_slot = raw_family.get("entity_slot")
        result[family] = _RegistryFamily(
            family=family,
            entity_slot=(str(entity_slot) if isinstance(entity_slot, str) else None),
            aliases=aliases,
            operation_aliases=operations,
        )
    return result


def _match_operation(
    text: str,
    allowed_values: Sequence[str],
    aliases: Mapping[str, Sequence[str]],
) -> str | None:
    matched = [
        value
        for value in allowed_values
        if any(_contains_phrase(text, alias) for alias in aliases.get(value, ()))
    ]
    return matched[0] if len(matched) == 1 else None


def _match_candidate(
    text: str,
    allowed_values: Sequence[str],
    candidates: Mapping[str, str],
) -> str | None:
    matched = [
        value
        for value in allowed_values
        if value in candidates and _contains_phrase(text, candidates[value])
    ]
    return matched[0] if len(matched) == 1 else None


def _candidate_names(request: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    context = request.get("context")
    if isinstance(context, Mapping):
        raw_candidates = context.get("candidates", [])
        if isinstance(raw_candidates, Sequence) and not isinstance(
            raw_candidates,
            (str, bytes),
        ):
            for raw in raw_candidates:
                if not isinstance(raw, Mapping):
                    continue
                key = raw.get("entity_key")
                name = raw.get("name")
                if isinstance(key, str) and isinstance(name, str) and name:
                    result[key] = name.casefold()
    raw_candidates = request.get("candidates", [])
    if isinstance(raw_candidates, Sequence) and not isinstance(raw_candidates, (str, bytes)):
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                continue
            key = raw.get("key")
            name = raw.get("name")
            if isinstance(key, str) and isinstance(name, str) and name:
                result[key] = name.casefold()
    return result


def _slot_specs(
    request: Mapping[str, object],
    family: str,
) -> tuple[Mapping[str, object], ...]:
    raw_families = request.get("family_slots", [])
    if not isinstance(raw_families, Sequence) or isinstance(raw_families, (str, bytes)):
        raise ValueError("family_slots must be a list")
    for raw_family in raw_families:
        if not isinstance(raw_family, Mapping) or raw_family.get("family") != family:
            continue
        raw_slots = raw_family.get("slots", [])
        if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
            raise ValueError("family slot list must be a list")
        if not all(isinstance(item, Mapping) for item in raw_slots):
            raise ValueError("family slot entries must be objects")
        return tuple(item for item in raw_slots if isinstance(item, Mapping))
    return ()


def _declaration(request: Mapping[str, object]) -> str:
    assessment = request.get("assessment")
    if not isinstance(assessment, Mapping):
        raise ValueError("request.assessment must be an object")
    return _required_text(assessment.get("declaration"), "assessment.declaration")


def _contains_phrase(text: str, phrase: str) -> bool:
    clean = phrase.strip().casefold()
    if not clean:
        return False
    return re.search(rf"(?<!\w){re.escape(clean)}(?!\w)", text) is not None


def _list_value(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Registry alias collection must be a list.")
    return value


def _string_values(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a string list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(str(item) for item in value)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
