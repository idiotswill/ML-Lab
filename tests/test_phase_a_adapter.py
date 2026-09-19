import json
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a import (
    PHASE_A_CONTRACT_VERSION,
    PhaseAResidualAdapter,
)
from ml_lab.core.models import DatasetSplit


def _request() -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "failed_deterministic_stage": "IDENTITY:AMBIGUOUS",
        "assessment": {
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "IDENTITY:AMBIGUOUS",
        },
        "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        "allowed_action_families": ["INTERACT_OBJECT"],
        "family_slots": [
            {
                "family": "INTERACT_OBJECT",
                "slots": [
                    {
                        "name": "OBJECT",
                        "required": True,
                        "allowed_values": ["object:chest"],
                        "prebound": "object:chest",
                    },
                    {
                        "name": "OPERATION",
                        "required": True,
                        "allowed_values": ["OPEN", "CLOSE"],
                        "prebound": None,
                    },
                ],
            }
        ],
    }


def _resolve() -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": "INTERACT_OBJECT",
        "entity_keys": [],
        "slots": [
            {"name": "OBJECT", "value": "object:chest"},
            {"name": "OPERATION", "value": "OPEN"},
        ],
        "reason_code": "BOUNDED_MATCH",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def test_contract_snapshot_declares_real_residual_boundary_files() -> None:
    paths = {item.path for item in PhaseAResidualAdapter().contract_files()}
    assert "frankenhomie-asterra-v0.9.0/asterra/intent_assessment.py" in paths
    assert "frankenhomie-asterra-v0.9.0/asterra/semantic_residual.py" in paths
    assert "frankenhomie-asterra-v0.9.0/asterra/semantic_slots.py" in paths
    assert "frankenhomie-asterra-v0.9.0/asterra/semantic_dispatch.py" in paths
    assert "frankenhomie-asterra-v0.9.0/docs/CORE_SUCCESSOR_ROADMAP.md" in paths


def test_dataset_input_must_already_be_at_residual_boundary() -> None:
    adapter = PhaseAResidualAdapter()
    request = _request()
    request["assessment"] = {
        "route": "EXACT",
        "model_required_because": "IDENTITY:AMBIGUOUS",
    }
    with pytest.raises(ValueError, match="SEMANTIC_REQUIRED"):
        adapter.validate_exported_request(request)


def test_valid_bounded_resolve_is_accepted_by_adapter_precheck() -> None:
    adapter = PhaseAResidualAdapter()
    result = adapter.validate_proposal(_request(), _resolve())
    assert result.accepted is True
    assert result.error_code is None


def test_proposal_cannot_escape_family_slot_or_prebound_envelope() -> None:
    adapter = PhaseAResidualAdapter()

    wrong_family = _resolve()
    wrong_family["action_family"] = "HARM_TARGET"
    assert adapter.validate_proposal(_request(), wrong_family).error_code == (
        "FAMILY_OUT_OF_ENVELOPE"
    )

    wrong_value = _resolve()
    wrong_value["slots"] = [
        {"name": "OBJECT", "value": "object:hidden"},
        {"name": "OPERATION", "value": "OPEN"},
    ]
    assert adapter.validate_proposal(_request(), wrong_value).error_code == (
        "SLOT_OUT_OF_ENVELOPE"
    )

    changed_prebound = _resolve()
    changed_prebound["slots"] = [
        {"name": "OBJECT", "value": "object:other"},
        {"name": "OPERATION", "value": "OPEN"},
    ]
    assert adapter.validate_proposal(_request(), changed_prebound).accepted is False


def test_proposal_cannot_claim_mechanics_or_other_authority_fields() -> None:
    adapter = PhaseAResidualAdapter()
    proposal = _resolve()
    proposal["damage_roll"] = "1d8+3"
    result = adapter.validate_proposal(_request(), proposal)
    assert result.accepted is False
    assert result.error_code == "UNSUPPORTED_AUTHORITY_FIELD"


def test_legacy_entity_keys_and_forbidden_decisions_are_rejected() -> None:
    adapter = PhaseAResidualAdapter()
    legacy = _resolve()
    legacy["entity_keys"] = ["object:chest"]
    assert adapter.validate_proposal(_request(), legacy).error_code == (
        "LEGACY_ENTITY_KEYS_FORBIDDEN"
    )

    forbidden_request = _request()
    forbidden_request["permitted_decisions"] = ["ASK_PLAYER"]
    assert adapter.validate_proposal(forbidden_request, _resolve()).error_code == (
        "DECISION_NOT_PERMITTED"
    )


def test_dataset_validator_preserves_exported_request_and_label() -> None:
    adapter = PhaseAResidualAdapter()
    validated = adapter.validate_dataset_row(
        {
            "example_id": "semantic-1",
            "source_id": "dev-export:1",
            "lineage_group": "template-1",
            "split": "TRAIN",
            "request": _request(),
            "expected": _resolve(),
            "tags": ["synthetic", "reviewed"],
        },
        1,
        "examples.jsonl",
    )
    assert validated.split is DatasetSplit.TRAIN
    assert validated.payload == {"request": _request()}
    assert validated.label == _resolve()


def test_historical_a5_reference_corpus_is_never_training_data(tmp_path: Path) -> None:
    corpus = {
        "schema": "frankenhomie-a5-intake-corpus/1",
        "cases": [
            {
                "id": "residual",
                "text": "With a roar I spring at Mara",
                "expected": "RESOLVE",
                "family": "HARM_TARGET",
                "slots": {"TARGET_COMBATANT": "combatant:mara"},
            },
            {
                "id": "zero-model",
                "text": "I might rush Mara",
                "expected": "NO_ACTION",
                "slots": {},
                "zero_model": True,
            },
        ],
    }
    path = tmp_path / "phase_a5.json"
    path.write_text(json.dumps(corpus), encoding="utf-8")

    adapter = PhaseAResidualAdapter()
    cases = adapter.load_reference_corpus(path)
    policy = adapter.reference_split_policy(cases)

    assert policy == {
        "residual": DatasetSplit.TEST,
        "zero-model": DatasetSplit.REDTEAM,
    }
    assert DatasetSplit.TRAIN not in set(policy.values())
