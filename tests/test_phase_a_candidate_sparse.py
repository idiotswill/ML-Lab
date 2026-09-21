from pathlib import Path

import pytest

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION, PhaseAResidualAdapter
from ml_lab.trainers.phase_a_candidate_sparse import (
    PhaseACandidateSparseModel,
    predict_phase_a_candidate_sparse,
    train_phase_a_candidate_sparse,
)


def _request(
    declaration: str,
    *,
    family: str,
    slot_specs: list[dict[str, object]],
    candidates: list[dict[str, object]],
    failed_stage: str = "COMMITMENT:UNRESOLVED_DECLARATION",
    permitted: list[str] | None = None,
    assessed_family: str | None = None,
) -> dict[str, object]:
    clause_family = assessed_family or (
        "UNKNOWN"
        if failed_stage == "COMMITMENT:UNRESOLVED_DECLARATION"
        else family
    )
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "failed_deterministic_stage": failed_stage,
        "assessment": {
            "version": "input-assessment-v1",
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": failed_stage,
            "declaration": declaration,
            "clauses": [
                {
                    "index": 0,
                    "raw": declaration,
                    "commitment": "COMMITTED",
                    "family": clause_family,
                    "mention": None,
                    "entity_key": None,
                    "evidence": [],
                }
            ],
        },
        "context": {
            "version": "semantic-context-v1",
            "actor_id": "tester",
            "audience": "TABLE",
            "snapshot_revision": "test",
            "scene_facts": [],
            "discourse_facts": [],
            "campaign_facts": [],
            "combat_facts": [],
            "inventory_facts": [],
            "rules_facts": [],
            "candidates": candidates,
            "context": None,
        },
        "permitted_decisions": permitted or ["RESOLVE", "ASK_PLAYER"],
        "allowed_action_families": [family],
        "family_slots": [{"family": family, "slots": slot_specs}],
        "clarification_answer": None,
        "previous_question": None,
        "previous_reason_code": None,
        "unresolved_slots": [],
        "question_history": [],
        "selected_primary_action": None,
        "instruction": "INTERPRET_ENGLISH_ONLY_WITHIN_ENVELOPE",
    }


def _candidate(key: str, name: str, *aliases: str) -> dict[str, object]:
    return {
        "entity_key": key,
        "name": name,
        "aliases": list(aliases),
        "source_fact_keys": [f"campaign.entity.{key}"],
        "focus_rank": None,
    }


def _resolve(
    family: str,
    slots: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": slots,
        "reason_code": "TRAINING_LABEL",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _ask(
    family: str | None,
    missing: list[str],
    candidates: list[str],
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "ASK_PLAYER",
        "action_family": family,
        "entity_keys": [],
        "slots": [],
        "reason_code": "TRAINING_LABEL",
        "facts_used": [],
        "question": "Clarify",
        "missing_slots": missing,
        "candidate_keys": candidates,
    }


def _record(
    request: dict[str, object],
    expected: dict[str, object],
) -> dict[str, object]:
    return {"payload": {"request": request}, "label": expected}


def test_candidate_relative_model_rebinds_novel_entity_key(tmp_path: Path) -> None:
    train_request = _request(
        "I appraise Brass Hatch.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:hatch", "Brass Hatch")],
    )
    model = train_phase_a_candidate_sparse(
        [
            _record(
                train_request,
                _resolve(
                    "SEARCH_INSPECT",
                    [{"name": "SUBJECT", "value": "object:hatch"}],
                ),
            )
        ],
        feature_dim=1024,
    )
    assert model.non_entity_slot_models == {}

    current = _request(
        "I appraise Iron Shutter.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:shutter"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:shutter", "Iron Shutter")],
    )
    proposal = predict_phase_a_candidate_sparse(model, current)

    assert proposal["decision"] == "RESOLVE"
    assert proposal["action_family"] == "SEARCH_INSPECT"
    assert proposal["slots"] == [
        {"name": "SUBJECT", "value": "object:shutter"}
    ]
    assert "object:hatch" not in str(proposal)
    assert PhaseAResidualAdapter().validate_proposal(current, proposal).accepted

    first = tmp_path / "candidate-relative.json"
    second = tmp_path / "candidate-relative-roundtrip.json"
    model.save(first)
    PhaseACandidateSparseModel.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_candidate_relative_model_abstains_on_ambiguous_pronoun() -> None:
    train_request = _request(
        "I appraise Brass Hatch.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:hatch", "Brass Hatch")],
    )
    model = train_phase_a_candidate_sparse(
        [
            _record(
                train_request,
                _resolve(
                    "SEARCH_INSPECT",
                    [{"name": "SUBJECT", "value": "object:hatch"}],
                ),
            )
        ],
        feature_dim=1024,
    )
    current = _request(
        "I inspect it.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:shutter", "object:pedestal"],
                "prebound": None,
            }
        ],
        candidates=[
            _candidate("object:shutter", "Iron Shutter"),
            _candidate("object:pedestal", "Marble Pedestal"),
        ],
        failed_stage="IDENTITY:REFERENCE_NOT_UNIQUE",
        assessed_family="SEARCH_INSPECT",
    )

    proposal = predict_phase_a_candidate_sparse(model, current)

    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] == "SEARCH_INSPECT"
    assert proposal["missing_slots"] == ["SUBJECT"]
    assert proposal["candidate_keys"] == [
        "object:pedestal",
        "object:shutter",
    ]
    assert PhaseAResidualAdapter().validate_proposal(current, proposal).accepted


def test_candidate_relative_model_preserves_ask_only_compound_gate() -> None:
    train_request = _request(
        "I appraise Brass Hatch.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:hatch", "Brass Hatch")],
    )
    model = train_phase_a_candidate_sparse(
        [
            _record(
                train_request,
                _resolve(
                    "SEARCH_INSPECT",
                    [{"name": "SUBJECT", "value": "object:hatch"}],
                ),
            )
        ],
        feature_dim=1024,
    )
    current = _request(
        "I walk to West Landing and inspect Iron Shutter.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:shutter"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:shutter", "Iron Shutter")],
        failed_stage="COMMITMENT:MULTIPLE_COMMITTED_ACTIONS",
        permitted=["ASK_PLAYER"],
    )

    proposal = predict_phase_a_candidate_sparse(model, current)

    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] is None
    assert proposal["missing_slots"] == ["PRIMARY_ACTION"]
    assert proposal["candidate_keys"] == []
    assert PhaseAResidualAdapter().validate_proposal(current, proposal).accepted


def test_candidate_relative_model_learns_non_entity_operation_only() -> None:
    hatch = _candidate("object:hatch", "Brass Hatch")
    plinth = _candidate("object:plinth", "Stone Plinth")
    open_request = _request(
        "I swing Brass Hatch ajar.",
        family="INTERACT_OBJECT",
        slot_specs=[
            {
                "name": "OBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            },
            {
                "name": "OPERATION",
                "required": True,
                "allowed_values": ["OPEN", "CLOSE"],
                "prebound": None,
            },
        ],
        candidates=[hatch],
    )
    close_request = _request(
        "I ease Stone Plinth shut.",
        family="INTERACT_OBJECT",
        slot_specs=[
            {
                "name": "OBJECT",
                "required": True,
                "allowed_values": ["object:plinth"],
                "prebound": None,
            },
            {
                "name": "OPERATION",
                "required": True,
                "allowed_values": ["OPEN", "CLOSE"],
                "prebound": None,
            },
        ],
        candidates=[plinth],
    )
    model = train_phase_a_candidate_sparse(
        [
            _record(
                open_request,
                _resolve(
                    "INTERACT_OBJECT",
                    [
                        {"name": "OBJECT", "value": "object:hatch"},
                        {"name": "OPERATION", "value": "OPEN"},
                    ],
                ),
            ),
            _record(
                close_request,
                _resolve(
                    "INTERACT_OBJECT",
                    [
                        {"name": "OBJECT", "value": "object:plinth"},
                        {"name": "OPERATION", "value": "CLOSE"},
                    ],
                ),
            ),
        ],
        feature_dim=2048,
    )

    assert set(model.non_entity_slot_models["INTERACT_OBJECT"]) == {"OPERATION"}

    current = _request(
        "I swing Iron Shutter ajar.",
        family="INTERACT_OBJECT",
        slot_specs=[
            {
                "name": "OBJECT",
                "required": True,
                "allowed_values": ["object:shutter"],
                "prebound": None,
            },
            {
                "name": "OPERATION",
                "required": True,
                "allowed_values": ["OPEN", "CLOSE"],
                "prebound": None,
            },
        ],
        candidates=[_candidate("object:shutter", "Iron Shutter")],
    )
    proposal = predict_phase_a_candidate_sparse(model, current)

    assert proposal["decision"] == "RESOLVE"
    assert proposal["slots"] == [
        {"name": "OBJECT", "value": "object:shutter"},
        {"name": "OPERATION", "value": "OPEN"},
    ]
    assert PhaseAResidualAdapter().validate_proposal(current, proposal).accepted


def test_candidate_relative_model_never_reuses_missing_train_entity() -> None:
    train_request = _request(
        "I appraise Brass Hatch.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:hatch", "Brass Hatch")],
    )
    model = train_phase_a_candidate_sparse(
        [
            _record(
                train_request,
                _resolve(
                    "SEARCH_INSPECT",
                    [{"name": "SUBJECT", "value": "object:hatch"}],
                ),
            )
        ],
        feature_dim=1024,
    )
    current = _request(
        "I appraise it.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:shutter"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:shutter", "Iron Shutter")],
    )

    proposal = predict_phase_a_candidate_sparse(model, current)

    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] == "SEARCH_INSPECT"
    assert proposal["missing_slots"] == ["SUBJECT"]
    assert proposal["candidate_keys"] == ["object:shutter"]
    assert "object:hatch" not in str(proposal)


def test_candidate_relative_training_rejects_out_of_envelope_labels() -> None:
    request = _request(
        "I appraise Brass Hatch.",
        family="SEARCH_INSPECT",
        slot_specs=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:hatch"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:hatch", "Brass Hatch")],
    )
    escaped = _resolve(
        "SEARCH_INSPECT",
        [{"name": "SUBJECT", "value": "object:hidden"}],
    )

    with pytest.raises(ValueError, match="violates its authority envelope"):
        train_phase_a_candidate_sparse(
            [_record(request, escaped)],
            feature_dim=1024,
        )
