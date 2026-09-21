from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION, PhaseAResidualAdapter
from ml_lab.trainers.phase_a_sparse import BoundedPredictionError
from ml_lab.trainers.sparse_nb import SparseNBBuilder, SparseNBModel

_MIN_DECISION_POSTERIOR = 0.60
_MIN_FAMILY_POSTERIOR = 0.65
_MIN_SLOT_POSTERIOR = 0.75


@dataclass(frozen=True, slots=True)
class PhaseACandidateSparseModel:
    format_version: int
    training_examples: int
    decision_model: SparseNBModel
    family_model: SparseNBModel | None
    non_entity_slot_models: dict[str, dict[str, SparseNBModel]]

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "kind": "phase-a-candidate-relative-sparse-v1",
            "training_examples": self.training_examples,
            "decision_model": self.decision_model.to_dict(),
            "family_model": (
                self.family_model.to_dict() if self.family_model is not None else None
            ),
            "non_entity_slot_models": {
                family: {
                    slot: model.to_dict()
                    for slot, model in sorted(family_models.items())
                }
                for family, family_models in sorted(
                    self.non_entity_slot_models.items()
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PhaseACandidateSparseModel:
        if value.get("kind") != "phase-a-candidate-relative-sparse-v1":
            raise ValueError("Unsupported Phase A candidate-relative model kind")
        decision_model = _model_from_object(
            value.get("decision_model"),
            "decision_model",
        )
        raw_family = value.get("family_model")
        family_model = (
            None
            if raw_family is None
            else _model_from_object(raw_family, "family_model")
        )
        raw_slots = value.get("non_entity_slot_models")
        if not isinstance(raw_slots, dict):
            raise ValueError("non_entity_slot_models must be an object")
        slot_models: dict[str, dict[str, SparseNBModel]] = {}
        for raw_family_name, raw_family_models in raw_slots.items():
            if not isinstance(raw_family_models, dict):
                raise ValueError("non_entity_slot_models family entries must be objects")
            family_name = str(raw_family_name)
            slot_models[family_name] = {
                str(slot_name): _model_from_object(
                    raw_model,
                    f"non_entity_slot_models.{family_name}.{slot_name}",
                )
                for slot_name, raw_model in raw_family_models.items()
            }
        return cls(
            format_version=_positive_int(value.get("format_version"), "format_version"),
            training_examples=_positive_int(
                value.get("training_examples"),
                "training_examples",
            ),
            decision_model=decision_model,
            family_model=family_model,
            non_entity_slot_models=slot_models,
        )

    def save(self, path: Path) -> None:
        path.write_bytes(
            (
                json.dumps(
                    self.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )

    @classmethod
    def load(cls, path: Path) -> PhaseACandidateSparseModel:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Phase A candidate-relative model must be a JSON object")
        return cls.from_dict({str(key): item for key, item in decoded.items()})


def train_phase_a_candidate_sparse(
    records: Iterable[Mapping[str, object]],
    *,
    feature_dim: int = 32768,
    alpha: float = 0.5,
) -> PhaseACandidateSparseModel:
    adapter = PhaseAResidualAdapter()
    decision_builder = SparseNBBuilder(
        text_key="declaration",
        label_key="decision",
        feature_dim=feature_dim,
        alpha=alpha,
    )
    family_builder = SparseNBBuilder(
        text_key="declaration",
        label_key="action_family",
        feature_dim=feature_dim,
        alpha=alpha,
    )
    slot_builders: dict[str, dict[str, SparseNBBuilder]] = {}
    examples = 0

    for record in records:
        request, expected = _training_row(record)
        adapter.validate_exported_request(request)
        proposal_check = adapter.validate_proposal(request, expected)
        if not proposal_check.accepted:
            raise ValueError(
                "Phase A candidate-relative training label violates its authority "
                f"envelope: {proposal_check.error_code}"
            )
        declaration = _declaration(request)
        decision = _required_text(expected.get("decision"), "expected.decision")
        decision_builder.add(declaration, decision)

        family = expected.get("action_family")
        if isinstance(family, str) and family:
            family_builder.add(declaration, family)
            specs = {
                _required_text(spec.get("name"), "family_slots.slots.name"): spec
                for spec in _slot_specs(request, family)
            }
            raw_slots = expected.get("slots", [])
            if not isinstance(raw_slots, Sequence) or isinstance(
                raw_slots,
                (str, bytes),
            ):
                raise ValueError("expected.slots must be a list")
            for raw_slot in raw_slots:
                if not isinstance(raw_slot, Mapping):
                    raise ValueError("expected.slots entries must be objects")
                name = _required_text(raw_slot.get("name"), "expected.slots.name")
                value = _required_text(raw_slot.get("value"), "expected.slots.value")
                spec = specs.get(name)
                if spec is None:
                    raise ValueError(
                        f"Training slot {family}.{name} is missing from request envelope"
                    )
                if value in _slot_candidate_keys(request, spec):
                    # Entity identity is deliberately not a learned class. It is rebound
                    # at prediction time from the current visible candidate envelope.
                    continue
                builder = slot_builders.setdefault(family, {}).setdefault(
                    name,
                    SparseNBBuilder(
                        text_key="declaration",
                        label_key=name,
                        feature_dim=feature_dim,
                        alpha=alpha,
                    ),
                )
                builder.add(declaration, value)
        examples += 1

    if examples == 0:
        raise ValueError("Phase A candidate-relative training data is empty")
    family_model = family_builder.finish() if family_builder.example_count else None
    return PhaseACandidateSparseModel(
        format_version=1,
        training_examples=examples,
        decision_model=decision_builder.finish(),
        family_model=family_model,
        non_entity_slot_models={
            family: {
                slot: builder.finish()
                for slot, builder in family_builders.items()
            }
            for family, family_builders in slot_builders.items()
        },
    )


def predict_phase_a_candidate_sparse(
    model: PhaseACandidateSparseModel,
    request: Mapping[str, object],
) -> dict[str, object]:
    adapter = PhaseAResidualAdapter()
    adapter.validate_exported_request(request)
    declaration = _declaration(request)

    deterministic = _deterministic_abstention(request)
    if deterministic is not None:
        return deterministic

    permitted = _string_values(
        request.get("permitted_decisions"),
        "permitted_decisions",
    )
    decision = _choose_bounded(
        model.decision_model,
        declaration,
        permitted,
        minimum_posterior=_MIN_DECISION_POSTERIOR,
    )
    if decision != "RESOLVE":
        return _ask_player(request, missing_slots=("ACTION_FAMILY",))

    allowed_families = _string_values(
        request.get("allowed_action_families"),
        "allowed_action_families",
    )
    if model.family_model is None:
        return _ask_player(request, missing_slots=("ACTION_FAMILY",))
    family = _choose_bounded(
        model.family_model,
        declaration,
        allowed_families,
        minimum_posterior=_MIN_FAMILY_POSTERIOR,
    )
    if family is None:
        return _ask_player(request, missing_slots=("ACTION_FAMILY",))

    supplied: list[dict[str, str]] = []
    missing: list[str] = []
    slot_models = model.non_entity_slot_models.get(family, {})
    for spec in _slot_specs(request, family):
        name = _required_text(spec.get("name"), "family_slots.slots.name")
        prebound = spec.get("prebound")
        if isinstance(prebound, str):
            supplied.append({"name": name, "value": prebound})
            continue

        candidate_keys = _slot_candidate_keys(request, spec)
        selected: str | None
        if candidate_keys:
            selected = _unique_explicit_candidate(
                request,
                declaration,
                candidate_keys,
            )
        else:
            allowed_values = _string_values(
                spec.get("allowed_values", []),
                "allowed_values",
            )
            slot_model = slot_models.get(name)
            selected = (
                _choose_bounded(
                    slot_model,
                    declaration,
                    allowed_values,
                    minimum_posterior=_MIN_SLOT_POSTERIOR,
                )
                if slot_model is not None
                else None
            )

        required = spec.get("required", True) is not False
        if selected is not None:
            supplied.append({"name": name, "value": selected})
        elif required:
            missing.append(name)

    if missing:
        return _ask_player(
            request,
            action_family=family,
            slots=supplied,
            missing_slots=tuple(missing),
        )

    proposal: dict[str, object] = {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": supplied,
        "reason_code": "ML_LAB_CANDIDATE_RELATIVE_SPARSE",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }
    check = adapter.validate_proposal(request, proposal)
    if not check.accepted:
        raise BoundedPredictionError(
            "Candidate-relative sparse assembly failed adapter validation: "
            f"{check.error_code}"
        )
    return proposal


def _deterministic_abstention(
    request: Mapping[str, object],
) -> dict[str, object] | None:
    permitted = _string_values(
        request.get("permitted_decisions"),
        "permitted_decisions",
    )
    failed_stage = _required_text(
        request.get("failed_deterministic_stage"),
        "failed_deterministic_stage",
    )
    if "RESOLVE" not in permitted or failed_stage == (
        "COMMITMENT:MULTIPLE_COMMITTED_ACTIONS"
    ):
        return _ask_player(
            request,
            missing_slots=("PRIMARY_ACTION",),
        )

    if failed_stage.startswith("IDENTITY:"):
        family = _single_assessed_family(request)
        if family is None:
            return _ask_player(
                request,
                missing_slots=("ACTION_FAMILY",),
            )
        missing = _required_candidate_slots(request, family)
        if not missing:
            return _ask_player(
                request,
                missing_slots=("ACTION_FAMILY",),
            )
        return _ask_player(
            request,
            action_family=family,
            missing_slots=missing,
        )
    return None


def _ask_player(
    request: Mapping[str, object],
    *,
    action_family: str | None = None,
    slots: Sequence[Mapping[str, str]] = (),
    missing_slots: Sequence[str],
) -> dict[str, object]:
    permitted = set(
        _string_values(
            request.get("permitted_decisions"),
            "permitted_decisions",
        )
    )
    if "ASK_PLAYER" not in permitted:
        raise BoundedPredictionError(
            "The current request does not permit ASK_PLAYER, so the scorer "
            "cannot abstain safely."
        )
    missing = list(dict.fromkeys(missing_slots)) or ["ACTION_FAMILY"]
    candidate_keys = (
        _candidate_keys_for_missing(request, action_family, missing)
        if action_family is not None
        else []
    )
    proposal: dict[str, object] = {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "ASK_PLAYER",
        "action_family": action_family,
        "entity_keys": [],
        "slots": [dict(item) for item in slots],
        "reason_code": "ML_LAB_CANDIDATE_RELATIVE_ABSTAIN",
        "facts_used": [],
        "question": _question_for_missing(missing),
        "missing_slots": missing,
        "candidate_keys": candidate_keys,
    }
    check = PhaseAResidualAdapter().validate_proposal(request, proposal)
    if not check.accepted:
        raise BoundedPredictionError(
            "Candidate-relative sparse abstention failed adapter validation: "
            f"{check.error_code}"
        )
    return proposal


def _required_candidate_slots(
    request: Mapping[str, object],
    family: str,
) -> tuple[str, ...]:
    names = []
    for spec in _slot_specs(request, family):
        if spec.get("required", True) is False:
            continue
        if _slot_candidate_keys(request, spec):
            names.append(_required_text(spec.get("name"), "family_slots.slots.name"))
    return tuple(names)


def _candidate_keys_for_missing(
    request: Mapping[str, object],
    family: str,
    missing: Sequence[str],
) -> list[str]:
    missing_set = set(missing)
    keys: set[str] = set()
    for spec in _slot_specs(request, family):
        name = _required_text(spec.get("name"), "family_slots.slots.name")
        if name in missing_set:
            keys.update(_slot_candidate_keys(request, spec))
    return sorted(keys)


def _slot_candidate_keys(
    request: Mapping[str, object],
    spec: Mapping[str, object],
) -> tuple[str, ...]:
    allowed = set(
        _string_values(
            spec.get("allowed_values", []),
            "allowed_values",
        )
    )
    context = request.get("context")
    if not isinstance(context, Mapping):
        return ()
    raw_candidates = context.get("candidates", [])
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates,
        (str, bytes),
    ):
        return ()
    keys = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        key = raw_candidate.get("entity_key")
        if isinstance(key, str) and key in allowed:
            keys.append(key)
    return tuple(sorted(set(keys)))


def _unique_explicit_candidate(
    request: Mapping[str, object],
    declaration: str,
    candidate_keys: Sequence[str],
) -> str | None:
    context = request.get("context")
    if not isinstance(context, Mapping):
        return None
    raw_candidates = context.get("candidates", [])
    if not isinstance(raw_candidates, Sequence) or isinstance(
        raw_candidates,
        (str, bytes),
    ):
        return None
    allowed = set(candidate_keys)
    matches: set[str] = set()
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, Mapping):
            continue
        key = raw_candidate.get("entity_key")
        if not isinstance(key, str) or key not in allowed:
            continue
        terms: list[str] = []
        name = raw_candidate.get("name")
        if isinstance(name, str) and name.strip():
            terms.append(name)
        aliases = raw_candidate.get("aliases", [])
        if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
            terms.extend(
                alias
                for alias in aliases
                if isinstance(alias, str) and alias.strip()
            )
        if any(_term_present(declaration, term) for term in terms):
            matches.add(key)
    return next(iter(matches)) if len(matches) == 1 else None


def _single_assessed_family(request: Mapping[str, object]) -> str | None:
    assessment = request.get("assessment")
    if not isinstance(assessment, Mapping):
        return None
    raw_clauses = assessment.get("clauses", [])
    if not isinstance(raw_clauses, Sequence) or isinstance(
        raw_clauses,
        (str, bytes),
    ):
        return None
    families = {
        str(raw_clause.get("family"))
        for raw_clause in raw_clauses
        if isinstance(raw_clause, Mapping)
        and raw_clause.get("family") not in (None, "UNKNOWN")
    }
    return next(iter(families)) if len(families) == 1 else None


def _question_for_missing(missing: Sequence[str]) -> str:
    if missing == ["PRIMARY_ACTION"]:
        return "Which action do you want to take first?"
    if missing == ["SUBJECT"]:
        return "Which visible subject do you mean?"
    if missing == ["OBJECT"]:
        return "Which visible object do you mean?"
    if missing == ["ADDRESSEE"]:
        return "Who are you speaking to?"
    if missing == ["DESTINATION"]:
        return "Which visible destination do you mean?"
    if missing == ["ACTION_FAMILY"]:
        return "What action do you want to take?"
    if missing == ["OPERATION"]:
        return "What do you want to do with it?"
    return "Please clarify the missing part of the action."


def _term_present(text: str, term: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(term.casefold())}(?!\w)",
            text.casefold(),
        )
    )


def _choose_bounded(
    model: SparseNBModel,
    text: str,
    allowed: Sequence[str],
    *,
    minimum_posterior: float,
) -> str | None:
    allowed_set = set(allowed)
    scores = {
        label: score
        for label, score in model.score(text).items()
        if label in allowed_set
    }
    if not scores:
        return None
    winner = max(scores, key=scores.__getitem__)
    maximum = scores[winner]
    denominator = sum(math.exp(score - maximum) for score in scores.values())
    posterior = 1.0 / denominator
    if posterior < minimum_posterior:
        return None
    return winner


def _training_row(
    record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    payload = record.get("payload")
    expected = record.get("label")
    if not isinstance(payload, dict) or not isinstance(expected, dict):
        raise ValueError("Phase A training rows require object payload and label")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("Phase A training row payload.request must be an object")
    return (
        {str(key): value for key, value in request.items()},
        {str(key): value for key, value in expected.items()},
    )


def _declaration(request: Mapping[str, object]) -> str:
    assessment = request.get("assessment")
    if not isinstance(assessment, Mapping):
        raise ValueError("request.assessment must be an object")
    return _required_text(
        assessment.get("declaration"),
        "assessment.declaration",
    )


def _slot_specs(
    request: Mapping[str, object],
    family: str,
) -> tuple[Mapping[str, object], ...]:
    raw_families = request.get("family_slots")
    if not isinstance(raw_families, Sequence) or isinstance(
        raw_families,
        (str, bytes),
    ):
        raise ValueError("family_slots must be a list")
    for raw_family in raw_families:
        if not isinstance(raw_family, Mapping) or raw_family.get("family") != family:
            continue
        raw_slots = raw_family.get("slots")
        if not isinstance(raw_slots, Sequence) or isinstance(
            raw_slots,
            (str, bytes),
        ):
            raise ValueError("family slot list must be a list")
        if not all(isinstance(item, Mapping) for item in raw_slots):
            raise ValueError("family slot entries must be objects")
        return tuple(
            item
            for item in raw_slots
            if isinstance(item, Mapping)
        )
    return ()


def _model_from_object(value: object, field: str) -> SparseNBModel:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return SparseNBModel.from_dict(
        {str(key): item for key, item in value.items()}
    )


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


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value
