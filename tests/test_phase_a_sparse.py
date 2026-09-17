from pathlib import Path

import pytest

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION, PhaseAResidualAdapter
from ml_lab.trainers.phase_a_sparse import (
    BoundedPredictionError,
    PhaseASparseModel,
    predict_phase_a_sparse,
    train_phase_a_sparse,
)


def _request(
    declaration: str,
    family: str,
    slot_name: str,
    allowed_values: list[str],
    *,
    prebound: str | None = None,
    permitted: list[str] | None = None,
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "failed_deterministic_stage": "IDENTITY:AMBIGUOUS",
        "assessment": {
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "IDENTITY:AMBIGUOUS",
            "declaration": declaration,
        },
        "permitted_decisions": permitted or ["RESOLVE", "ASK_PLAYER"],
        "allowed_action_families": [family],
        "family_slots": [
            {
                "family": family,
                "slots": [
                    {
                        "name": slot_name,
                        "required": True,
                        "allowed_values": allowed_values,
                        "prebound": prebound,
                    }
                ],
            }
        ],
    }


def _resolve(
    family: str,
    slot_name: str,
    value: str,
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": [{"name": slot_name, "value": value}],
        "reason_code": "TRAINING_LABEL",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _record(
    request: dict[str, object],
    expected: dict[str, object],
) -> dict[str, object]:
    return {"payload": {"request": request}, "label": expected}


def test_bounded_sparse_resolves_and_round_trips_deterministically(tmp_path: Path) -> None:
    train_request = _request(
        "I clobber Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    model = train_phase_a_sparse(
        [_record(train_request, _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara"))],
        feature_dim=1024,
    )

    proposal = predict_phase_a_sparse(model, train_request)
    assert proposal["decision"] == "RESOLVE"
    assert proposal["action_family"] == "HARM_TARGET"
    assert proposal["slots"] == [
        {"name": "TARGET_COMBATANT", "value": "combatant:mara"}
    ]
    assert PhaseAResidualAdapter().validate_proposal(train_request, proposal).accepted

    first = tmp_path / "phase-a-sparse.json"
    second = tmp_path / "phase-a-sparse-roundtrip.json"
    model.save(first)
    PhaseASparseModel.load(first).save(second)
    assert first.read_bytes() == second.read_bytes()


def test_bounded_sparse_abstains_when_trained_family_is_not_currently_allowed() -> None:
    harm_request = _request(
        "I clobber Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    model = train_phase_a_sparse(
        [_record(harm_request, _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara"))],
        feature_dim=1024,
    )
    move_request = _request(
        "I head for camp",
        "MOVE_TRAVEL",
        "DESTINATION",
        ["location:camp"],
    )

    proposal = predict_phase_a_sparse(model, move_request)
    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] is None
    assert proposal["missing_slots"] == ["ACTION_FAMILY"]
    assert "HARM_TARGET" not in str(proposal)


def test_bounded_sparse_never_emits_trained_slot_value_outside_current_envelope() -> None:
    train_request = _request(
        "I clobber Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    model = train_phase_a_sparse(
        [_record(train_request, _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara"))],
        feature_dim=1024,
    )
    current_request = _request(
        "I clobber the guard",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:guard"],
    )

    proposal = predict_phase_a_sparse(model, current_request)
    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] == "HARM_TARGET"
    assert proposal["slots"] == []
    assert proposal["missing_slots"] == ["TARGET_COMBATANT"]
    assert "combatant:mara" not in str(proposal)
    assert PhaseAResidualAdapter().validate_proposal(current_request, proposal).accepted


def test_bounded_sparse_preserves_server_prebound_slot() -> None:
    request = _request(
        "I head for camp",
        "MOVE_TRAVEL",
        "DESTINATION",
        ["location:camp"],
        prebound="location:camp",
    )
    model = train_phase_a_sparse(
        [_record(request, _resolve("MOVE_TRAVEL", "DESTINATION", "location:camp"))],
        feature_dim=1024,
    )

    proposal = predict_phase_a_sparse(model, request)
    assert proposal["decision"] == "RESOLVE"
    assert proposal["slots"] == [{"name": "DESTINATION", "value": "location:camp"}]


def test_bounded_sparse_fails_closed_if_abstention_is_not_permitted() -> None:
    train_request = _request(
        "I clobber Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    model = train_phase_a_sparse(
        [_record(train_request, _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara"))],
        feature_dim=1024,
    )
    current_request = _request(
        "I head for camp",
        "MOVE_TRAVEL",
        "DESTINATION",
        ["location:camp"],
        permitted=["RESOLVE"],
    )

    with pytest.raises(BoundedPredictionError, match="cannot abstain safely"):
        predict_phase_a_sparse(model, current_request)


def test_phase_a_sparse_training_rejects_out_of_envelope_labels() -> None:
    request = _request(
        "I clobber Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    escaped = _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:hidden")

    with pytest.raises(ValueError, match="violates its exported authority envelope"):
        train_phase_a_sparse([_record(request, escaped)], feature_dim=1024)
