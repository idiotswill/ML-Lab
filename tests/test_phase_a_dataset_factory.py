from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_lab.adapters.phase_a_dataset_factory as factory_module
from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_dataset_factory import run_phase_a_dataset_factory
from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExportReceipt
from ml_lab.storage.workspace import Workspace


def _request(declaration: str) -> dict[str, object]:
    return {
        "version": "semantic-residual-v2",
        "request_id": "semantic:test",
        "assessment": {
            "version": "input-assessment-v1",
            "declaration": declaration,
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "COMMITMENT:UNRESOLVED_DECLARATION",
            "mechanics_authorized": False,
            "unsupported_family": None,
            "candidates": [
                {
                    "entity_key": "object:hatch",
                    "name": "Brass Hatch",
                    "aliases": [],
                    "source_fact_keys": ["campaign.entity.object.hatch"],
                    "focus_rank": None,
                }
            ],
            "clauses": [
                {
                    "index": 0,
                    "raw": declaration,
                    "commitment": "UNRESOLVED_DECLARATION",
                    "family": "UNKNOWN",
                    "mention": None,
                    "entity_key": None,
                    "evidence": [],
                }
            ],
            "trace": [],
        },
        "context": {
            "version": "semantic-context-v1",
            "actor_id": "tester",
            "audience": "TABLE",
            "snapshot_revision": "test-revision",
            "scene_facts": [],
            "discourse_facts": [],
            "campaign_facts": [],
            "combat_facts": [],
            "inventory_facts": [],
            "rules_facts": [],
            "candidates": [
                {
                    "entity_key": "object:hatch",
                    "name": "Brass Hatch",
                    "aliases": [],
                    "source_fact_keys": ["campaign.entity.object.hatch"],
                    "focus_rank": None,
                }
            ],
            "context": None,
        },
        "failed_deterministic_stage": "COMMITMENT:UNRESOLVED_DECLARATION",
        "allowed_action_families": ["SEARCH_INSPECT"],
        "family_slots": [
            {
                "family": "SEARCH_INSPECT",
                "slots": [
                    {
                        "name": "SUBJECT",
                        "required": True,
                        "allowed_values": ["object:hatch"],
                        "prebound": None,
                    }
                ],
            }
        ],
        "clarification_answer": None,
        "previous_question": None,
        "previous_reason_code": None,
        "unresolved_slots": [],
        "question_history": [],
        "selected_primary_action": None,
        "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        "instruction": "INTERPRET_ENGLISH_ONLY_WITHIN_ENVELOPE",
    }


class _FakeExporter:
    fail = False
    calls = 0
    leak_hidden = False

    def __init__(self, _workspace: Workspace):
        pass

    def export(self, *, repository: Path, ref: str, fixture: dict[str, object]):
        del repository
        type(self).calls += 1
        request = _request(str(fixture["declaration"]))
        if self.leak_hidden:
            context = request["context"]
            assert isinstance(context, dict)
            context["campaign_facts"] = [
                {
                    "key": "campaign.entity.npc.hidden-observer",
                    "value": "Hidden Observer",
                    "source": "fixture:test",
                    "visibility": "GM_ONLY",
                }
            ]
        if self.fail:
            return PhaseAFixtureExportReceipt(
                status="TERMINATED_BEFORE_RESIDUAL",
                commit_sha=ref,
                contract_version="semantic-residual-v2",
                fixture_id=str(fixture["fixture_id"]),
                fixture_sha256="f" * 64,
                route="EXACT",
                failed_deterministic_stage=None,
                request_sha256=None,
                request=None,
                fresh_process=True,
                ephemeral_sqlite_only=True,
                network_access_allowed=False,
                resolver_dispatch_available=False,
                authority_mutation_allowed=False,
                error_code=None,
                error_message=None,
                receipt_artifact_digest="artifact",
            )
        return PhaseAFixtureExportReceipt(
            status="RESIDUAL_EXPORTED",
            commit_sha=ref,
            contract_version="semantic-residual-v2",
            fixture_id=str(fixture["fixture_id"]),
            fixture_sha256="f" * 64,
            route="SEMANTIC_REQUIRED",
            failed_deterministic_stage="COMMITMENT:UNRESOLVED_DECLARATION",
            request_sha256=hashlib.sha256(
                factory_module.canonical_json(request).encode("utf-8")
            ).hexdigest(),
            request=request,
            fresh_process=True,
            ephemeral_sqlite_only=True,
            network_access_allowed=False,
            resolver_dispatch_available=False,
            authority_mutation_allowed=False,
            error_code=None,
            error_message=None,
            receipt_artifact_digest="artifact",
        )


class _FakeReference:
    def __init__(self, _workspace: Workspace):
        pass

    def validate(self, **_kwargs):
        return SimpleNamespace(status="ACCEPTED", error_code=None)


def _write_seed(path: Path, train_text: str, dev_text: str) -> Path:
    payload = {
        "schema": "ml-lab-phase-a-synthetic-seed/1",
        "status": "SYNTHETIC_NON_CANON",
        "target_frankenhomie_commit": "1" * 40,
        "target_contract": "semantic-residual-v2",
        "production_data": False,
        "transcript_derived": False,
        "actor_id": "tester",
        "audience": "TABLE",
        "scenes": {
            "room": {
                "snapshot_revision": "seed-1",
                "facts": [
                    {
                        "key": "session.scene",
                        "value": "Synthetic room",
                        "source": "fixture:test",
                        "visibility": "TABLE",
                    },
                    {
                        "key": "campaign.entity.object.hatch",
                        "value": "Brass Hatch",
                        "source": "fixture:test",
                        "visibility": "TABLE",
                    },
                ],
            }
        },
        "cases": [
            {
                "case_id": "train-one",
                "split": "TRAIN",
                "scene_id": "room",
                "lineage_group": "train/one",
                "declaration": train_text,
                "expected": {
                    "decision": "RESOLVE",
                    "action_family": "SEARCH_INSPECT",
                    "slots": [{"name": "SUBJECT", "value": "object:hatch"}],
                },
                "tags": ["synthetic"],
            },
            {
                "case_id": "dev-one",
                "split": "DEV",
                "scene_id": "room",
                "lineage_group": "dev/one",
                "declaration": dev_text,
                "expected": {
                    "decision": "RESOLVE",
                    "action_family": "SEARCH_INSPECT",
                    "slots": [{"name": "SUBJECT", "value": "object:hatch"}],
                },
                "tags": ["synthetic"],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_protected(path: Path, text: str = "Protected held out wording.") -> Path:
    payload = {
        "schema": "ml-lab-phase-a-authoritative-fixture-seed/1",
        "status": "PROTECTED_EVALUATION_ONLY",
        "target_frankenhomie_commit": "1" * 40,
        "target_contract": "semantic-residual-v2",
        "cases": [
            {
                "case_id": "held-out",
                "split": "TEST",
                "lineage_group": "protected/held-out",
                "declaration": text,
                "training_allowed": False,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeExporter.fail = False
    _FakeExporter.calls = 0
    _FakeExporter.leak_hidden = False
    monkeypatch.setattr(factory_module, "PhaseAFixtureExporter", _FakeExporter)
    monkeypatch.setattr(factory_module, "PhaseAReferenceValidator", _FakeReference)


def test_dataset_factory_emits_only_valid_train_dev_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    output = tmp_path / "out"
    result = run_phase_a_dataset_factory(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=output,
        seed_path=_write_seed(
            tmp_path / "seed.json",
            "I carefully appraise Brass Hatch.",
            "I closely survey Brass Hatch.",
        ),
        protected_seed_path=_write_protected(tmp_path / "protected.json"),
    )

    assert result["ok"] is True
    assert result["case_count"] == 2
    assert result["emitted_count"] == 2
    assert result["split_counts"] == {"DEV": 1, "TRAIN": 1}
    assert result["candidate_training_data"] is True
    assert result["protected_splits_in_output"] is False
    assert result["production_data"] is False
    assert result["transcript_derived"] is False
    assert result["integration_gate"] == "NO_GO"

    dataset_path = output / "phase-a-train-dev-v1.jsonl"
    dataset_bytes = dataset_path.read_bytes()
    assert b"\r\n" not in dataset_bytes
    assert hashlib.sha256(dataset_bytes).hexdigest() == result["dataset_sha256"]
    rows = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
    ]
    assert {row["split"] for row in rows} == {"TRAIN", "DEV"}
    adapter = PhaseAResidualAdapter()
    for line_number, row in enumerate(rows, start=1):
        validated = adapter.validate_dataset_row(row, line_number, "factory.jsonl")
        assert validated.split.value in {"TRAIN", "DEV"}


def test_dataset_factory_fails_closed_when_case_does_not_reach_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    _FakeExporter.fail = True
    output = tmp_path / "out"

    result = run_phase_a_dataset_factory(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=output,
        seed_path=_write_seed(
            tmp_path / "seed.json",
            "Train wording.",
            "Dev wording.",
        ),
        protected_seed_path=_write_protected(tmp_path / "protected.json"),
    )

    assert result["ok"] is False
    assert result["emitted_count"] == 0
    assert result["candidate_training_data"] is False
    assert not (output / "phase-a-train-dev-v1.jsonl").exists()
    assert {
        error["code"] for error in result["errors"]
    } == {"NOT_RESIDUAL_EXPORTED"}


def test_dataset_factory_blocks_protected_language_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    leaked = "I appraise Brass Hatch."
    output = tmp_path / "out"

    result = run_phase_a_dataset_factory(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=output,
        seed_path=_write_seed(
            tmp_path / "seed.json",
            leaked,
            "Distinct development wording.",
        ),
        protected_seed_path=_write_protected(tmp_path / "protected.json", leaked),
    )

    assert result["ok"] is False
    assert result["protected_language_leakage"]["blocking_count"] >= 1
    assert any(
        error["code"] == "PROTECTED_LANGUAGE_LEAKAGE"
        for error in result["errors"]
    )
    assert not (output / "phase-a-train-dev-v1.jsonl").exists()


def test_default_synthetic_seed_contains_only_train_dev_and_no_private_sources() -> None:
    seed = json.loads(factory_module.DEFAULT_SYNTHETIC_SEED.read_text(encoding="utf-8"))
    assert seed["production_data"] is False
    assert seed["transcript_derived"] is False
    assert len(seed["cases"]) == 18
    assert {case["split"] for case in seed["cases"]} == {"TRAIN", "DEV"}
    assert sum(case["split"] == "TRAIN" for case in seed["cases"]) == 12
    assert sum(case["split"] == "DEV" for case in seed["cases"]) == 6
    assert all(
        fact["source"].startswith("fixture:")
        for scene in seed["scenes"].values()
        for fact in scene["facts"]
    )
    movement = next(
        case for case in seed["cases"]
        if case["case_id"] == "train-move-stride-east"
    )
    assert movement["declaration"] == "I stride eastward toward East Passage."
    assert movement["expected"] == {
        "decision": "RESOLVE",
        "action_family": "MOVE_TRAVEL",
        "slots": [{"name": "DESTINATION", "value": "location:east-passage"}],
    }



def test_dataset_factory_reuses_exact_case_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    seed = _write_seed(
        tmp_path / "seed.json",
        "I appraise Brass Hatch.",
        "I survey Brass Hatch.",
    )
    protected = _write_protected(tmp_path / "protected.json")
    cache = tmp_path / "case-cache"
    workspace = Workspace.create(tmp_path / "workspace")

    first = run_phase_a_dataset_factory(
        workspace=workspace,
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "first",
        seed_path=seed,
        protected_seed_path=protected,
        case_cache_dir=cache,
    )
    assert first["ok"] is True
    assert first["case_cache"] == {
        "enabled": True,
        "hits": 0,
        "misses": 2,
    }
    assert _FakeExporter.calls == 2

    second = run_phase_a_dataset_factory(
        workspace=workspace,
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "second",
        seed_path=seed,
        protected_seed_path=protected,
        case_cache_dir=cache,
    )
    assert second["ok"] is True
    assert second["case_cache"] == {
        "enabled": True,
        "hits": 2,
        "misses": 0,
    }
    assert _FakeExporter.calls == 2
    assert (
        tmp_path / "first" / "phase-a-train-dev-v1.jsonl"
    ).read_bytes() == (
        tmp_path / "second" / "phase-a-train-dev-v1.jsonl"
    ).read_bytes()


def test_dataset_factory_cache_invalidates_only_changed_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    protected = _write_protected(tmp_path / "protected.json")
    cache = tmp_path / "case-cache"
    workspace = Workspace.create(tmp_path / "workspace")
    original = _write_seed(
        tmp_path / "seed-original.json",
        "I appraise Brass Hatch.",
        "I survey Brass Hatch.",
    )
    run_phase_a_dataset_factory(
        workspace=workspace,
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "first",
        seed_path=original,
        protected_seed_path=protected,
        case_cache_dir=cache,
    )
    changed = _write_seed(
        tmp_path / "seed-changed.json",
        "I carefully appraise Brass Hatch.",
        "I survey Brass Hatch.",
    )
    result = run_phase_a_dataset_factory(
        workspace=workspace,
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "second",
        seed_path=changed,
        protected_seed_path=protected,
        case_cache_dir=cache,
    )

    assert result["case_cache"] == {
        "enabled": True,
        "hits": 1,
        "misses": 1,
    }
    assert _FakeExporter.calls == 3


def test_dataset_factory_blocks_gm_only_fixture_fact_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    _FakeExporter.leak_hidden = True
    seed_path = _write_seed(
        tmp_path / "seed.json",
        "I appraise Brass Hatch.",
        "I survey Brass Hatch.",
    )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    seed["scenes"]["room"]["facts"].append(
        {
            "key": "campaign.entity.npc.hidden-observer",
            "value": "Hidden Observer",
            "source": "fixture:test",
            "visibility": "GM_ONLY",
        }
    )
    seed_path.write_text(json.dumps(seed), encoding="utf-8")

    result = run_phase_a_dataset_factory(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "out",
        seed_path=seed_path,
        protected_seed_path=_write_protected(tmp_path / "protected.json"),
    )

    assert result["ok"] is False
    assert result["candidate_training_data"] is False
    assert any(
        error["code"] == "HIDDEN_FIXTURE_FACT_LEAK"
        for error in result["errors"]
    )
    assert not (tmp_path / "out" / "phase-a-train-dev-v1.jsonl").exists()


def test_dataset_factory_uses_seed_source_id_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    seed_path = _write_seed(
        tmp_path / "seed.json",
        "I appraise Brass Hatch.",
        "I survey Brass Hatch.",
    )
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    seed["source_id_prefix"] = "synthetic-corpus-v1"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")

    result = run_phase_a_dataset_factory(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "out",
        seed_path=seed_path,
        protected_seed_path=_write_protected(tmp_path / "protected.json"),
    )

    assert result["ok"] is True
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "out" / "phase-a-train-dev-v1.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        row["source_id"].startswith("synthetic-corpus-v1:")
        for row in rows
    )
