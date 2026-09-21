from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace

_APP_ROOT = "frankenhomie-asterra-v0.9.0"


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    proposal: dict[str, object]
    contract_ok: bool
    contract_error: str | None


class V2DeterministicAbstention:
    name = "deterministic-abstention-v2"

    def predict(self, request: Mapping[str, object]) -> dict[str, object]:
        assessment = _mapping(request, "assessment")
        failed = str(request["failed_deterministic_stage"])
        known_family = _single_known_family(assessment)
        family_slots = _family_slot_map(request)

        if request.get("permitted_decisions") == ["ASK_PLAYER"] or failed == (
            "COMMITMENT:MULTIPLE_COMMITTED_ACTIONS"
        ):
            family = None
            missing = ["PRIMARY_ACTION"]
            question = "Which action do you want to take first?"
            candidates: list[str] = []
        elif known_family and known_family in family_slots:
            family = known_family
            missing = _required_unbound_slots(family_slots[known_family])
            if not missing:
                missing = ["ACTION_FAMILY"]
                family = None
            question = _question_for_missing(missing)
            candidates = _candidate_keys_for_slots(request, family, missing)
        else:
            family = None
            missing = ["ACTION_FAMILY"]
            question = "What action do you want to take?"
            candidates = []

        return {
            "version": "semantic-residual-v2",
            "decision": "ASK_PLAYER",
            "action_family": family,
            "entity_keys": [],
            "slots": [],
            "reason_code": "DETERMINISTIC_ABSTENTION_V2",
            "facts_used": [],
            "question": question,
            "missing_slots": missing,
            "candidate_keys": candidates,
        }


class V2BoundedLexical:
    name = "bounded-lexical-v2"

    def __init__(self, registry: Mapping[str, object]):
        self.registry = registry
        self.family_terms, self.entity_slots, self.operation_terms = _registry_terms(
            registry
        )
        self.abstention = V2DeterministicAbstention()

    def predict(self, request: Mapping[str, object]) -> dict[str, object]:
        permitted = tuple(str(value) for value in _sequence(request, "permitted_decisions"))
        if "RESOLVE" not in permitted:
            return self.abstention.predict(request)

        assessment = _mapping(request, "assessment")
        text = str(assessment["declaration"]).casefold()
        allowed = {str(value) for value in _sequence(request, "allowed_action_families")}
        scores = {
            family: max(
                (
                    1.0 if _term_present(text, term) else 0.0
                    for term in self.family_terms.get(family, ())
                ),
                default=0.0,
            )
            for family in allowed
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        winner, score = ranked[0] if ranked else (None, 0.0)
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if not winner or score < 0.75 or score - runner_up < 0.25:
            return self.abstention.predict(request)

        family_specs = _family_slot_map(request)
        specs = family_specs.get(winner, [])
        slots: list[dict[str, str]] = []
        resolved_names: set[str] = set()

        entity_slot = self.entity_slots.get(winner)
        for spec in specs:
            name = str(spec["name"])
            prebound = spec.get("prebound")
            if isinstance(prebound, str):
                slots.append({"name": name, "value": prebound})
                resolved_names.add(name)
                continue
            if name == entity_slot:
                value = _unique_visible_name_value(request, spec, text)
                if value is not None:
                    slots.append({"name": name, "value": value})
                    resolved_names.add(name)
                    continue
            operation = _unique_operation_value(
                spec,
                text,
                self.operation_terms.get(winner, {}),
            )
            if operation is not None:
                slots.append({"name": name, "value": operation})
                resolved_names.add(name)

        required = {
            str(spec["name"])
            for spec in specs
            if spec.get("required", True) is not False
        }
        missing = sorted(required - resolved_names)
        if missing:
            return {
                "version": "semantic-residual-v2",
                "decision": "ASK_PLAYER",
                "action_family": winner,
                "entity_keys": [],
                "slots": slots,
                "reason_code": "LEXICAL_REQUIRED_SLOT_MISSING",
                "facts_used": [],
                "question": _question_for_missing(missing),
                "missing_slots": missing,
                "candidate_keys": _candidate_keys_for_slots(request, winner, missing),
            }

        return {
            "version": "semantic-residual-v2",
            "decision": "RESOLVE",
            "action_family": winner,
            "entity_keys": [],
            "slots": slots,
            "reason_code": "LEXICAL_UNIQUE_WINNER_V2",
            "facts_used": [],
            "question": None,
            "missing_slots": [],
            "candidate_keys": [],
        }


def run_phase_a_baselines(
    *,
    workspace: Workspace,
    frankenhomie_repository: Path,
    receipt_path: Path,
    labels_path: Path,
    output_path: Path,
) -> dict[str, object]:
    receipt_bytes = receipt_path.read_bytes()
    labels_bytes = labels_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    labels = json.loads(labels_bytes)
    if not isinstance(receipt, dict) or not isinstance(labels, dict):
        raise ValueError("Phase A baseline inputs must be JSON objects")
    _validate_receipt_and_labels(receipt, labels)

    commit_sha = str(receipt["target_frankenhomie_commit"])
    materialized = PhaseAReferenceValidator(workspace).materialize(
        frankenhomie_repository,
        commit_sha,
    )
    registry_path = (
        materialized.source_root
        / _APP_ROOT
        / "data"
        / "semantic_family_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Pinned semantic registry must be a JSON object")

    cases_by_id = {
        str(row["case_id"]): row
        for row in _sequence_mapping(receipt, "cases")
        if isinstance(_mapping(row, "actual").get("request"), Mapping)
    }
    label_rows = {
        str(row["case_id"]): row
        for row in _sequence_mapping(labels, "cases")
    }
    if set(cases_by_id) != set(label_rows):
        raise ValueError("Residual receipt cases and protected labels do not match")

    adapter = PhaseAResidualAdapter()
    reports = []
    for baseline in (V2DeterministicAbstention(), V2BoundedLexical(registry)):
        rows = []
        contract_failures = false_commitments = correct = 0
        resolved = correct_resolved = expected_resolves = 0
        expected_asks = correct_asks = 0
        for case_id in sorted(cases_by_id):
            case = cases_by_id[case_id]
            label = label_rows[case_id]
            actual = _mapping(case, "actual")
            request = _mapping(actual, "request")
            adapter.validate_exported_request(request)
            proposal = baseline.predict(request)
            validation = adapter.validate_proposal(request, proposal)
            contract_ok = validation.accepted
            if not contract_ok:
                contract_failures += 1

            expected_decision = str(label["expected_decision"])
            actual_decision = str(proposal.get("decision"))
            expected_resolves += int(expected_decision == "RESOLVE")
            expected_asks += int(expected_decision == "ASK_PLAYER")
            resolved += int(actual_decision == "RESOLVE")
            if actual_decision == "RESOLVE" and expected_decision != "RESOLVE":
                false_commitments += 1

            is_correct = contract_ok and _matches_label(proposal, label)
            correct += int(is_correct)
            correct_resolved += int(
                is_correct and actual_decision == "RESOLVE"
            )
            correct_asks += int(
                is_correct and actual_decision == "ASK_PLAYER"
            )
            rows.append(
                {
                    "case_id": case_id,
                    "split": case["split"],
                    "expected": {
                        "decision": expected_decision,
                        "family": label.get("expected_family"),
                        "slots": label.get("expected_slots", []),
                        "missing_slots": label.get("expected_missing_slots", []),
                    },
                    "actual": proposal,
                    "contract_ok": contract_ok,
                    "contract_error": validation.error_code,
                    "correct": is_correct,
                }
            )

        total = len(rows)
        reports.append(
            {
                "candidate": baseline.name,
                "case_count": total,
                "correct": correct,
                "decision_accuracy": round(correct / total, 4) if total else None,
                "resolved": resolved,
                "useful_resolution_coverage": (
                    round(correct_resolved / expected_resolves, 4)
                    if expected_resolves
                    else None
                ),
                "resolution_precision": (
                    round(correct_resolved / resolved, 4) if resolved else None
                ),
                "ask_player_accuracy": (
                    round(correct_asks / expected_asks, 4)
                    if expected_asks
                    else None
                ),
                "false_commitments": false_commitments,
                "contract_failures": contract_failures,
                "rows": rows,
            }
        )

    base_payload: dict[str, object] = {
        "schema": "ml-lab-phase-a-baseline-evidence/1",
        "target_frankenhomie_commit": commit_sha,
        "target_contract": receipt["target_contract"],
        "source_receipt_payload_sha256": receipt["payload_sha256"],
        "source_receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "training_allowed": False,
        "production_data": False,
        "integration_gate": "NO_GO",
        "reports": reports,
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_payload).encode("utf-8")
    ).hexdigest()
    payload = {**base_payload, "payload_sha256": payload_sha}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def _validate_receipt_and_labels(
    receipt: Mapping[str, object],
    labels: Mapping[str, object],
) -> None:
    if receipt.get("schema") != "ml-lab-phase-a-fixture-evidence/1":
        raise ValueError("Unsupported Phase A fixture receipt schema")
    if receipt.get("ok") is not True or receipt.get("mismatch_count") != 0:
        raise ValueError("Baseline input requires a clean authoritative receipt")
    if receipt.get("training_allowed") is not False:
        raise ValueError("Fixture receipt must remain non-training evidence")
    if labels.get("schema") != "ml-lab-phase-a-fixture-labels/1":
        raise ValueError("Unsupported Phase A fixture labels schema")
    if labels.get("status") != "PROTECTED_EVALUATION_ONLY":
        raise ValueError("Phase A fixture labels must remain protected")
    if labels.get("training_allowed") is not False:
        raise ValueError("Phase A fixture labels cannot enter TRAIN")
    if labels.get("receipt_payload_sha256") != receipt.get("payload_sha256"):
        raise ValueError("Phase A labels are not bound to this authoritative receipt")
    if labels.get("target_frankenhomie_commit") != receipt.get(
        "target_frankenhomie_commit"
    ):
        raise ValueError("Phase A labels target a different Frankenhomie commit")


def _matches_label(
    proposal: Mapping[str, object],
    label: Mapping[str, object],
) -> bool:
    if proposal.get("decision") != label.get("expected_decision"):
        return False
    if proposal.get("action_family") != label.get("expected_family"):
        return False
    actual_slots = _normalized_slots(proposal.get("slots", []))
    expected_slots = _normalized_slots(label.get("expected_slots", []))
    if actual_slots != expected_slots:
        return False
    actual_missing = sorted(
        str(value) for value in _sequence_value(proposal.get("missing_slots", []))
    )
    expected_missing = sorted(
        str(value)
        for value in _sequence_value(label.get("expected_missing_slots", []))
    )
    return actual_missing == expected_missing


def _registry_terms(
    registry: Mapping[str, object],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, str],
    dict[str, dict[str, tuple[str, ...]]],
]:
    if registry.get("schema_version") != "asterra-semantic-family-registry/1":
        raise ValueError("Unsupported semantic family registry schema")
    family_terms: dict[str, tuple[str, ...]] = {}
    entity_slots: dict[str, str] = {}
    operations: dict[str, dict[str, tuple[str, ...]]] = {}
    for row in _sequence_mapping(registry, "families"):
        family = str(row["family"])
        aliases = [
            str(value)
            for value in _sequence_value(row.get("deterministic_aliases", []))
        ]
        hints = [
            str(value)
            for value in _sequence_value(row.get("residual_hints", []))
        ]
        family_terms[family] = tuple(dict.fromkeys([*aliases, *hints]))
        entity_slots[family] = str(row["entity_slot"])
        raw_operations = row.get("operation_aliases", {})
        if not isinstance(raw_operations, Mapping):
            raise ValueError("operation_aliases must be an object")
        operations[family] = {
            str(name): tuple(str(value) for value in _sequence_value(values))
            for name, values in raw_operations.items()
        }
    return family_terms, entity_slots, operations


def _term_present(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", text))


def _single_known_family(assessment: Mapping[str, object]) -> str | None:
    families = {
        str(clause.get("family"))
        for clause in _sequence_mapping(assessment, "clauses")
        if clause.get("family") not in (None, "UNKNOWN")
    }
    return next(iter(families)) if len(families) == 1 else None


def _family_slot_map(
    request: Mapping[str, object],
) -> dict[str, list[Mapping[str, object]]]:
    return {
        str(row["family"]): list(_sequence_mapping(row, "slots"))
        for row in _sequence_mapping(request, "family_slots")
    }


def _required_unbound_slots(
    specs: Sequence[Mapping[str, object]],
) -> list[str]:
    return [
        str(spec["name"])
        for spec in specs
        if spec.get("required", True) is not False and spec.get("prebound") is None
    ]


def _unique_visible_name_value(
    request: Mapping[str, object],
    spec: Mapping[str, object],
    text: str,
) -> str | None:
    allowed = {str(value) for value in _sequence_value(spec.get("allowed_values", []))}
    matches = [
        str(candidate["entity_key"])
        for candidate in _sequence_mapping(_mapping(request, "context"), "candidates")
        if str(candidate["entity_key"]) in allowed
        and _term_present(text, str(candidate["name"]))
    ]
    return matches[0] if len(matches) == 1 else None


def _unique_operation_value(
    spec: Mapping[str, object],
    text: str,
    operations: Mapping[str, tuple[str, ...]],
) -> str | None:
    allowed = {str(value) for value in _sequence_value(spec.get("allowed_values", []))}
    matches = [
        operation
        for operation, terms in operations.items()
        if operation in allowed and any(_term_present(text, term) for term in terms)
    ]
    return matches[0] if len(matches) == 1 else None


def _candidate_keys_for_slots(
    request: Mapping[str, object],
    family: str | None,
    missing: Sequence[str],
) -> list[str]:
    if not family:
        return []
    specs = _family_slot_map(request).get(family, [])
    missing_set = set(missing)
    allowed = {
        str(value)
        for spec in specs
        if str(spec["name"]) in missing_set
        for value in _sequence_value(spec.get("allowed_values", []))
    }
    visible = {
        str(candidate["entity_key"])
        for candidate in _sequence_mapping(_mapping(request, "context"), "candidates")
    }
    return sorted(allowed & visible)


def _question_for_missing(missing: Sequence[str]) -> str:
    if missing == ["SUBJECT"]:
        return "Which visible subject do you mean?"
    if missing == ["OBJECT"]:
        return "Which visible object do you mean?"
    if missing == ["TARGET_COMBATANT"]:
        return "Which visible combatant do you mean?"
    if missing == ["ADDRESSEE"]:
        return "Who are you speaking to?"
    if missing == ["OPERATION"]:
        return "What do you want to do with it?"
    if missing == ["PRIMARY_ACTION"]:
        return "Which action do you want to take first?"
    if missing == ["ACTION_FAMILY"]:
        return "What action do you want to take?"
    return "Please clarify the missing part of the action."


def _normalized_slots(value: object) -> list[tuple[str, str]]:
    rows = _sequence_value(value)
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("slots must contain objects")
        normalized.append((str(row["name"]), str(row["value"])))
    return sorted(normalized)


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    row = value.get(key)
    if not isinstance(row, Mapping):
        raise ValueError(f"{key} must be an object")
    return row


def _sequence(value: Mapping[str, object], key: str) -> Sequence[object]:
    return _sequence_value(value.get(key))


def _sequence_value(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected a list-like value")
    return value


def _sequence_mapping(
    value: Mapping[str, object],
    key: str,
) -> list[Mapping[str, object]]:
    rows = _sequence(value, key)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{key} must contain objects")
    return [row for row in rows if isinstance(row, Mapping)]
