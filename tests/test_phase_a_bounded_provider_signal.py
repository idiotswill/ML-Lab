from __future__ import annotations

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION, PhaseAResidualAdapter
from ml_lab.baselines.phase_a_bounded_provider_signal import (
    assemble_bounded_provider_signal,
)


def _candidate(key: str, name: str) -> dict[str, object]:
    return {
        "entity_key": key,
        "name": name,
        "aliases": [],
        "source_fact_keys": [f"fixture:{key}"],
        "focus_rank": None,
    }


def _request(
    declaration: str,
    *,
    family: str,
    slots: list[dict[str, object]],
    candidates: list[dict[str, object]],
    failed_stage: str = "COMMITMENT:UNRESOLVED_DECLARATION",
    permitted: list[str] | None = None,
    assessed_family: str | None = None,
) -> dict[str, object]:
    clause_family = assessed_family or (
        "UNKNOWN" if failed_stage.startswith("COMMITMENT:") else family
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
        "family_slots": [{"family": family, "slots": slots}],
        "clarification_answer": None,
        "previous_question": None,
        "previous_reason_code": None,
        "unresolved_slots": [],
        "question_history": [],
        "selected_primary_action": None,
        "instruction": "INTERPRET_ENGLISH_ONLY_WITHIN_ENVELOPE",
    }


def test_bounded_signal_preserves_deterministic_identity_abstention() -> None:
    request = _request(
        "I inspect it.",
        family="SEARCH_INSPECT",
        slots=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:a", "object:b"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:a", "Amber Box"), _candidate("object:b", "Blue Box")],
        failed_stage="IDENTITY:REFERENCE_NOT_UNIQUE",
        assessed_family="SEARCH_INSPECT",
    )
    raw = {
        "decision": "RESOLVE",
        "action_family": "SEARCH_INSPECT",
        "entity_keys": ["object:hidden"],
        "facts_used": ["gm.secret"],
        "slots": [{"name": "SUBJECT", "value": "object:hidden"}],
    }

    proposal = assemble_bounded_provider_signal(request, raw)

    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] == "SEARCH_INSPECT"
    assert proposal["missing_slots"] == ["SUBJECT"]
    assert proposal["candidate_keys"] == ["object:a", "object:b"]
    assert "hidden" not in str(proposal)
    assert "gm.secret" not in str(proposal)
    assert PhaseAResidualAdapter().validate_proposal(request, proposal).accepted


def test_bounded_signal_rebinds_entity_from_current_visible_mention() -> None:
    request = _request(
        "I scrutinize Iron Shutter.",
        family="SEARCH_INSPECT",
        slots=[
            {
                "name": "SUBJECT",
                "required": True,
                "allowed_values": ["object:shutter"],
                "prebound": None,
            }
        ],
        candidates=[_candidate("object:shutter", "Iron Shutter")],
    )
    raw = {
        "decision": "RESOLVE",
        "action_family": "SEARCH_INSPECT",
        "entity_keys": ["object:hidden"],
        "facts_used": ["gm.secret"],
        "candidate_keys": ["object:hidden"],
        "slots": [{"name": "SUBJECT", "value": "object:hidden"}],
    }

    proposal = assemble_bounded_provider_signal(request, raw)

    assert proposal["decision"] == "RESOLVE"
    assert proposal["slots"] == [{"name": "SUBJECT", "value": "object:shutter"}]
    assert proposal["entity_keys"] == []
    assert proposal["facts_used"] == []
    assert proposal["candidate_keys"] == []
    assert PhaseAResidualAdapter().validate_proposal(request, proposal).accepted


def test_bounded_signal_accepts_only_allowed_non_entity_semantic_hint() -> None:
    request = _request(
        "I wrench Iron Shutter open.",
        family="INTERACT_OBJECT",
        slots=[
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
    raw = {
        "decision": "RESOLVE",
        "action_family": "INTERACT_OBJECT",
        "slots": [
            {"name": "OBJECT", "value": "object:hidden"},
            {"name": "OPERATION", "value": "OPEN"},
        ],
    }

    proposal = assemble_bounded_provider_signal(request, raw)

    assert proposal["decision"] == "RESOLVE"
    assert proposal["slots"] == [
        {"name": "OBJECT", "value": "object:shutter"},
        {"name": "OPERATION", "value": "OPEN"},
    ]
    assert PhaseAResidualAdapter().validate_proposal(request, proposal).accepted


def test_bounded_signal_abstains_when_non_entity_hint_is_outside_allowlist() -> None:
    request = _request(
        "I wrench Iron Shutter somehow.",
        family="INTERACT_OBJECT",
        slots=[
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
    raw = {
        "decision": "RESOLVE",
        "action_family": "INTERACT_OBJECT",
        "slots": [
            {"name": "OBJECT", "value": "object:hidden"},
            {"name": "OPERATION", "value": "BREAK"},
        ],
    }

    proposal = assemble_bounded_provider_signal(request, raw)

    assert proposal["decision"] == "ASK_PLAYER"
    assert proposal["action_family"] == "INTERACT_OBJECT"
    assert proposal["slots"] == [{"name": "OBJECT", "value": "object:shutter"}]
    assert proposal["missing_slots"] == ["OPERATION"]
    assert PhaseAResidualAdapter().validate_proposal(request, proposal).accepted
