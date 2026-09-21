from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExporter
from ml_lab.storage.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "benchmarks" / "phase_a" / "authoritative_fixture_seed_v1.json"


def _seed() -> dict[str, object]:
    decoded = json.loads(SEED.read_text(encoding="utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def _case_fixture(seed: dict[str, object], case: dict[str, object]) -> dict[str, object]:
    return {
        "fixture_id": str(case["case_id"]),
        "fixture_kind": seed["fixture_kind"],
        "production_data": seed["production_data"],
        "declaration": case["declaration"],
        "actor_id": seed["actor_id"],
        "audience": seed["audience"],
        "snapshot_revision": seed["snapshot_revision"],
        "facts": seed["facts"],
    }


def test_authoritative_fixture_seed_is_protected_and_cannot_supply_authority() -> None:
    seed = _seed()
    assert seed["schema"] == "ml-lab-phase-a-authoritative-fixture-seed/1"
    assert seed["status"] == "PROTECTED_EVALUATION_ONLY"
    assert seed["target_contract"] == "semantic-residual-v2"
    assert seed["fixture_kind"] == "NON_CANON_FIXTURE"
    assert seed["production_data"] is False

    forbidden = {
        "allowed_action_families",
        "family_slots",
        "candidates",
        "permitted_decisions",
    }
    assert not forbidden.intersection(seed)

    cases = seed["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 7
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert all(case["split"] in {"TEST", "REDTEAM"} for case in cases)
    assert all(case["training_allowed"] is False for case in cases)
    assert all(not forbidden.intersection(case) for case in cases)

    facts = seed["facts"]
    assert isinstance(facts, list)
    assert all(str(fact["source"]).startswith("fixture:") for fact in facts)
    assert any(fact["visibility"] == "GM_ONLY" for fact in facts)


def test_authoritative_fixture_seed_contains_both_boundary_paths() -> None:
    cases = _seed()["cases"]
    assert isinstance(cases, list)
    statuses = {case["expected_status"] for case in cases}
    assert statuses == {"TERMINATED_BEFORE_RESIDUAL", "RESIDUAL_EXPORTED"}
    assert any(
        case["expected_failed_deterministic_stage"]
        == "COMMITMENT:MULTIPLE_COMMITTED_ACTIONS"
        and case["expected_permitted_decisions"] == ["ASK_PLAYER"]
        for case in cases
    )
    assert any(
        case["expected_failed_deterministic_stage"]
        == "COMMITMENT:UNRESOLVED_DECLARATION"
        and case["expected_permitted_decisions"] == ["RESOLVE", "ASK_PLAYER"]
        for case in cases
    )


def test_authoritative_fixture_seed_against_pinned_frankenhomie(tmp_path: Path) -> None:
    repository_value = os.environ.get("ML_LAB_FRANKENHOMIE_REPO")
    if not repository_value:
        pytest.skip("ML_LAB_FRANKENHOMIE_REPO is required for pinned integration evidence")
    repository = Path(repository_value).resolve()
    seed = _seed()
    target_commit = str(seed["target_frankenhomie_commit"])
    workspace = Workspace.create(tmp_path / "workspace")
    exporter = PhaseAFixtureExporter(workspace)

    cases = seed["cases"]
    assert isinstance(cases, list)
    for raw_case in cases:
        assert isinstance(raw_case, dict)
        receipt = exporter.export(
            repository=repository,
            ref=target_commit,
            fixture=_case_fixture(seed, raw_case),
        )

        assert receipt.commit_sha == target_commit
        assert receipt.status == raw_case["expected_status"]
        assert receipt.route == raw_case["expected_route"]
        assert (
            receipt.failed_deterministic_stage
            == raw_case["expected_failed_deterministic_stage"]
        )

        expected_permitted = raw_case["expected_permitted_decisions"]
        if expected_permitted is None:
            assert receipt.request is None
            assert receipt.request_sha256 is None
            continue

        assert receipt.request is not None
        assert receipt.request_sha256 is not None
        assert receipt.request["permitted_decisions"] == expected_permitted
        assessment = receipt.request["assessment"]
        assert isinstance(assessment, dict)
        assert assessment["declaration"] == raw_case["declaration"]

        context = receipt.request["context"]
        assert isinstance(context, dict)
        candidates = context["candidates"]
        assert isinstance(candidates, list)
        candidate_keys = {
            candidate["entity_key"]
            for candidate in candidates
            if isinstance(candidate, dict)
        }
        assert "npc:hidden-observer" not in candidate_keys
