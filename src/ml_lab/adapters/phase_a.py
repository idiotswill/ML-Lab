from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_lab.contracts.snapshot import ContractFileSpec
from ml_lab.core.models import DatasetSplit, MetricDirection
from ml_lab.datasets.service import ValidatedExampleInput

PHASE_A_ADAPTER_ID = "frankenhomie.phase-a-residual"
PHASE_A_ADAPTER_VERSION = "1.0.0"
PHASE_A_CONTRACT_VERSION = "semantic-residual-v2"

PHASE_A_FAMILIES = frozenset(
    {
        "HARM_TARGET",
        "MOVE_TRAVEL",
        "SEARCH_INSPECT",
        "INTERACT_OBJECT",
        "SPEECH_ONLY",
        "CAST_SPELL",
        "USE_ITEM",
    }
)

_PHASE_A_PROPOSAL_FIELDS = frozenset(
    {
        "version",
        "decision",
        "action_family",
        "entity_keys",
        "slots",
        "reason_code",
        "facts_used",
        "question",
        "missing_slots",
        "candidate_keys",
    }
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_id: str
    direction: MetricDirection
    veto: bool


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    case_id: str
    text: str
    expected: str
    family: str | None
    slots: dict[str, str]
    zero_model: bool


@dataclass(frozen=True, slots=True)
class ProposalValidation:
    accepted: bool
    error_code: str | None = None


class PhaseAResidualAdapter:
    """Development adapter for Frankenhomie's bounded residual-semantic seam.

    This class does not assess commitment, enumerate legal actions, dispatch a resolver,
    or mutate campaign state. It validates Lab data/proposal shape against an already
    exported residual request and leaves authoritative acceptance to the pinned
    Frankenhomie reference validator.
    """

    adapter_id = PHASE_A_ADAPTER_ID
    version = PHASE_A_ADAPTER_VERSION
    contract_version = PHASE_A_CONTRACT_VERSION

    def contract_files(self) -> tuple[ContractFileSpec, ...]:
        root = "frankenhomie-asterra-v0.9.0"
        return (
            ContractFileSpec(f"{root}/asterra/intent_assessment.py", "commitment-gate"),
            ContractFileSpec(f"{root}/asterra/semantic_context.py", "context-envelope"),
            ContractFileSpec(f"{root}/asterra/semantic_residual.py", "residual-contract"),
            ContractFileSpec(f"{root}/asterra/semantic_slots.py", "slot-contract"),
            ContractFileSpec(f"{root}/asterra/semantic_routing.py", "routing-boundary"),
            ContractFileSpec(f"{root}/asterra/semantic_dispatch.py", "revalidation-dispatch"),
            ContractFileSpec(
                f"{root}/data/semantic_family_registry.json",
                "semantic-registry",
            ),
            ContractFileSpec(
                f"{root}/benchmarks/phase_a5_semantic_intake_v1.json",
                "reference-corpus",
            ),
            ContractFileSpec(
                f"{root}/benchmarks/PHASE_A5_SEMANTIC_INTAKE_RESULTS_V7.json",
                "provider-baseline",
            ),
            ContractFileSpec(
                f"{root}/docs/PHASE_A5_SEMANTIC_INTAKE_TASK.md",
                "phase-task",
            ),
            ContractFileSpec(
                f"{root}/docs/PHASE_A5_VERIFICATION.md",
                "verification",
            ),
            ContractFileSpec(
                f"{root}/docs/CORE_SUCCESSOR_ROADMAP.md",
                "roadmap",
            ),
        )

    def metric_definitions(self) -> tuple[MetricDefinition, ...]:
        return (
            MetricDefinition("resolution_precision", MetricDirection.HIGHER, False),
            MetricDefinition("useful_resolution_coverage", MetricDirection.HIGHER, False),
            MetricDefinition("ask_player_accuracy", MetricDirection.HIGHER, False),
            MetricDefinition("decision_accuracy", MetricDirection.HIGHER, False),
            MetricDefinition("family_slot_accuracy", MetricDirection.HIGHER, False),
            MetricDefinition("false_commitments", MetricDirection.ZERO, True),
            MetricDefinition("contract_failures", MetricDirection.ZERO, True),
            MetricDefinition("zero_model_route_violations", MetricDirection.ZERO, True),
            MetricDefinition("hidden_or_out_of_envelope", MetricDirection.ZERO, True),
            MetricDefinition("unsupported_mechanics_authority", MetricDirection.ZERO, True),
            MetricDefinition("adversarial_failures", MetricDirection.ZERO, True),
            MetricDefinition("latency_ms", MetricDirection.LOWER, False),
            MetricDefinition("ram_mb", MetricDirection.LOWER, False),
            MetricDefinition("model_size_mb", MetricDirection.LOWER, False),
        )

    def validate_dataset_row(
        self,
        row: dict[str, object],
        line_number: int,
        source_name: str,
    ) -> ValidatedExampleInput:
        example_id = _required_string(row, "example_id")
        source_id = str(row.get("source_id") or f"{source_name}:{line_number}")
        lineage_group = _required_string(row, "lineage_group")
        split = DatasetSplit(_required_string(row, "split").upper())
        request = _required_object(row, "request")
        expected = _required_object(row, "expected")
        tags = _string_tuple(row.get("tags", []), "tags")

        self.validate_exported_request(request)
        validation = self.validate_proposal(request, expected)
        if not validation.accepted:
            raise ValueError(
                f"Expected proposal violates exported envelope: {validation.error_code}"
            )
        return ValidatedExampleInput(
            example_id=example_id,
            split=split,
            source_id=source_id,
            lineage_group=lineage_group,
            payload={"request": request},
            label=expected,
            tags=tags,
        )

    def validate_exported_request(self, request: Mapping[str, object]) -> None:
        if request.get("version") != PHASE_A_CONTRACT_VERSION:
            raise ValueError("Phase A Lab rows require semantic-residual-v2 requests")
        assessment = request.get("assessment")
        if not isinstance(assessment, Mapping):
            raise ValueError("request.assessment must be an object")
        if assessment.get("route") != "SEMANTIC_REQUIRED":
            raise ValueError("Lab input begins only at SEMANTIC_REQUIRED residual boundary")
        failed_stage = request.get("failed_deterministic_stage")
        if not isinstance(failed_stage, str) or not failed_stage:
            raise ValueError("failed_deterministic_stage is required")
        if assessment.get("model_required_because") != failed_stage:
            raise ValueError("request must preserve the exact failed deterministic stage")

        permitted = _string_tuple(request.get("permitted_decisions"), "permitted_decisions")
        if not permitted or not set(permitted) <= {"RESOLVE", "ASK_PLAYER"}:
            raise ValueError("permitted_decisions must be bounded to RESOLVE/ASK_PLAYER")
        if len(set(permitted)) != len(permitted):
            raise ValueError("permitted_decisions must be unique")

        families = _string_tuple(
            request.get("allowed_action_families"),
            "allowed_action_families",
        )
        if not families or not set(families) <= PHASE_A_FAMILIES:
            raise ValueError("allowed_action_families contains unsupported Phase A family")
        family_slots = request.get("family_slots")
        if not isinstance(family_slots, Sequence) or isinstance(family_slots, (str, bytes)):
            raise ValueError("family_slots must be a list")
        for family in family_slots:
            if not isinstance(family, Mapping):
                raise ValueError("family_slots entries must be objects")
            family_name = family.get("family")
            if family_name not in families:
                raise ValueError("family_slots family must be in allowed_action_families")
            slots = family.get("slots")
            if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
                raise ValueError("family slot list must be a list")
            names: set[str] = set()
            for slot in slots:
                if not isinstance(slot, Mapping):
                    raise ValueError("slot spec must be an object")
                name = slot.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("slot name is required")
                if name in names:
                    raise ValueError("duplicate slot name in family envelope")
                names.add(name)
                allowed = _string_tuple(slot.get("allowed_values", []), "allowed_values")
                if len(set(allowed)) != len(allowed):
                    raise ValueError("duplicate allowed slot value")
                prebound = slot.get("prebound")
                if prebound is not None and prebound not in allowed:
                    raise ValueError("prebound slot must be inside allowed_values")

    def validate_proposal(
        self,
        request: Mapping[str, object],
        proposal: Mapping[str, object],
    ) -> ProposalValidation:
        try:
            self.validate_exported_request(request)
        except ValueError:
            return ProposalValidation(False, "INVALID_EXPORTED_REQUEST")
        if set(proposal) - _PHASE_A_PROPOSAL_FIELDS:
            return ProposalValidation(False, "UNSUPPORTED_AUTHORITY_FIELD")
        if proposal.get("version") != PHASE_A_CONTRACT_VERSION:
            return ProposalValidation(False, "SLOT_CONTRACT_REQUIRED")
        if proposal.get("entity_keys") not in (None, [], ()):
            return ProposalValidation(False, "LEGACY_ENTITY_KEYS_FORBIDDEN")
        decision = proposal.get("decision")
        permitted = set(_string_tuple(request.get("permitted_decisions"), "permitted_decisions"))
        if decision not in permitted:
            return ProposalValidation(False, "DECISION_NOT_PERMITTED")

        family = proposal.get("action_family")
        allowed_families = set(
            _string_tuple(request.get("allowed_action_families"), "allowed_action_families")
        )
        if family is not None and family not in allowed_families:
            return ProposalValidation(False, "FAMILY_OUT_OF_ENVELOPE")
        slot_specs = _slot_spec_map(request, str(family)) if isinstance(family, str) else {}
        proposal_slots = proposal.get("slots", [])
        if not isinstance(proposal_slots, Sequence) or isinstance(
            proposal_slots,
            (str, bytes),
        ):
            return ProposalValidation(False, "INVALID_SLOT_SHAPE")
        seen: set[str] = set()
        supplied: set[str] = set()
        for raw_slot in proposal_slots:
            if not isinstance(raw_slot, Mapping):
                return ProposalValidation(False, "INVALID_SLOT_SHAPE")
            name = raw_slot.get("name")
            value = raw_slot.get("value")
            if not isinstance(name, str) or not isinstance(value, str):
                return ProposalValidation(False, "INVALID_SLOT_SHAPE")
            if name in seen:
                return ProposalValidation(False, "DUPLICATE_SLOT")
            seen.add(name)
            supplied.add(name)
            spec = slot_specs.get(name)
            if spec is None:
                return ProposalValidation(False, "SLOT_OUT_OF_ENVELOPE")
            allowed_values = _string_tuple(spec.get("allowed_values", []), "allowed_values")
            if value not in allowed_values:
                return ProposalValidation(False, "SLOT_OUT_OF_ENVELOPE")
            prebound = spec.get("prebound")
            if prebound is not None and value != prebound:
                return ProposalValidation(False, "PREBOUND_SLOT_CHANGED")

        if decision == "RESOLVE":
            if not isinstance(family, str) or not family:
                return ProposalValidation(False, "RESOLVE_REQUIRES_FAMILY")
            required = {
                name
                for name, spec in slot_specs.items()
                if spec.get("required", True) is not False
            }
            if not required <= supplied:
                return ProposalValidation(False, "REQUIRED_SEMANTIC_SLOT_MISSING")
            if proposal.get("question") is not None:
                return ProposalValidation(False, "RESOLVE_HAS_QUESTION")
            if proposal.get("missing_slots") not in (None, [], ()):
                return ProposalValidation(False, "RESOLVE_HAS_MISSING_SLOTS")
            if proposal.get("candidate_keys") not in (None, [], ()):
                return ProposalValidation(False, "RESOLVE_HAS_CANDIDATES")
        elif decision == "ASK_PLAYER":
            question = proposal.get("question")
            missing = proposal.get("missing_slots")
            if not isinstance(question, str) or not question.strip():
                return ProposalValidation(False, "ASK_PLAYER_REQUIRES_QUESTION")
            if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
                return ProposalValidation(False, "ASK_PLAYER_REQUIRES_MISSING_SLOTS")
            if not missing:
                return ProposalValidation(False, "ASK_PLAYER_REQUIRES_MISSING_SLOTS")
        else:
            return ProposalValidation(False, "UNKNOWN_DECISION")
        return ProposalValidation(True)

    def load_reference_corpus(self, path: Path) -> tuple[ReferenceCase, ...]:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Phase A reference corpus must be a JSON object")
        if decoded.get("schema") != "frankenhomie-a5-intake-corpus/1":
            raise ValueError("Unsupported Phase A reference corpus schema")
        raw_cases = decoded.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("Phase A reference corpus cases must be a list")
        cases: list[ReferenceCase] = []
        for raw in raw_cases:
            if not isinstance(raw, dict):
                raise ValueError("Reference case must be an object")
            slots = raw.get("slots", {})
            if not isinstance(slots, dict):
                raise ValueError("Reference case slots must be an object")
            cases.append(
                ReferenceCase(
                    case_id=_required_string(raw, "id"),
                    text=_required_string(raw, "text"),
                    expected=_required_string(raw, "expected"),
                    family=str(raw["family"]) if raw.get("family") is not None else None,
                    slots={str(key): str(value) for key, value in slots.items()},
                    zero_model=bool(raw.get("zero_model", False)),
                )
            )
        return tuple(cases)

    def reference_split_policy(
        self,
        cases: Sequence[ReferenceCase],
    ) -> dict[str, DatasetSplit]:
        """Keep Frankenhomie's existing A5 corpus out of TRAIN.

        Residual provider cases are historical held-out reference evidence; upstream
        deterministic/unsupported cases remain REDTEAM gate evidence.
        """
        return {
            case.case_id: (DatasetSplit.REDTEAM if case.zero_model else DatasetSplit.TEST)
            for case in cases
        }


def _required_string(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _required_object(value: Mapping[str, object], key: str) -> dict[str, object]:
    raw = value.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object")
    return {str(item_key): item_value for item_key, item_value in raw.items()}


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a string list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a string list")
    return tuple(str(item) for item in value)


def _slot_spec_map(
    request: Mapping[str, object],
    family_name: str,
) -> dict[str, Mapping[str, object]]:
    raw_families = request.get("family_slots", [])
    if not isinstance(raw_families, Sequence) or isinstance(raw_families, (str, bytes)):
        return {}
    for family in raw_families:
        if not isinstance(family, Mapping) or family.get("family") != family_name:
            continue
        slots = family.get("slots", [])
        if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
            return {}
        result: dict[str, Mapping[str, object]] = {}
        for slot in slots:
            if isinstance(slot, Mapping) and isinstance(slot.get("name"), str):
                result[str(slot["name"])] = slot
        return result
    return {}
