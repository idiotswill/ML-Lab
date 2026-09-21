from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ml_lab.adapters.phase_a_fixture_seed as fixture_seed_module
from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExportReceipt
from ml_lab.adapters.phase_a_fixture_seed import run_phase_a_fixture_seed_evidence
from ml_lab.datasets.leakage import canonical_json


class _FakeExporter:
    def __init__(self, _workspace: object) -> None:
        pass

    def export(
        self,
        *,
        repository: Path,
        ref: str,
        fixture: dict[str, object],
        timeout_seconds: float = 30.0,
    ) -> PhaseAFixtureExportReceipt:
        del repository, timeout_seconds
        case_id = str(fixture["fixture_id"])
        if case_id == "zero-model":
            return PhaseAFixtureExportReceipt(
                status="TERMINATED_BEFORE_RESIDUAL",
                commit_sha=ref,
                contract_version="semantic-residual-v2",
                fixture_id=case_id,
                fixture_sha256="a" * 64,
                route="NO_ACTION",
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
                receipt_artifact_digest="artifact-zero",
            )
        request = {
            "version": "semantic-residual-v2",
            "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        }
        request_sha = hashlib.sha256(
            canonical_json(request).encode("utf-8")
        ).hexdigest()
        return PhaseAFixtureExportReceipt(
            status="RESIDUAL_EXPORTED",
            commit_sha=ref,
            contract_version="semantic-residual-v2",
            fixture_id=case_id,
            fixture_sha256="b" * 64,
            route="SEMANTIC_REQUIRED",
            failed_deterministic_stage="COMMITMENT:UNRESOLVED_DECLARATION",
            request_sha256=request_sha,
            request=request,
            fresh_process=True,
            ephemeral_sqlite_only=True,
            network_access_allowed=False,
            resolver_dispatch_available=False,
            authority_mutation_allowed=False,
            error_code=None,
            error_message=None,
            receipt_artifact_digest="artifact-residual",
        )


def _seed(path: Path, *, wrong_route: bool = False) -> Path:
    payload = {
        "schema": "ml-lab-phase-a-authoritative-fixture-seed/1",
        "status": "PROTECTED_EVALUATION_ONLY",
        "target_frankenhomie_commit": "1" * 40,
        "target_contract": "semantic-residual-v2",
        "fixture_kind": "NON_CANON_FIXTURE",
        "production_data": False,
        "actor_id": "tester",
        "audience": "TABLE",
        "snapshot_revision": "seed-1",
        "facts": [
            {
                "key": "session.scene",
                "value": "Synthetic room",
                "source": "fixture:test",
                "visibility": "TABLE",
            }
        ],
        "cases": [
            {
                "case_id": "zero-model",
                "split": "REDTEAM",
                "lineage_group": "test/zero",
                "declaration": "Never mind.",
                "expected_status": "TERMINATED_BEFORE_RESIDUAL",
                "expected_route": "EXACT" if wrong_route else "NO_ACTION",
                "expected_failed_deterministic_stage": None,
                "expected_permitted_decisions": None,
                "training_allowed": False,
                "tags": ["zero-model"],
            },
            {
                "case_id": "residual",
                "split": "TEST",
                "lineage_group": "test/residual",
                "declaration": "I scrutinize the thing.",
                "expected_status": "RESIDUAL_EXPORTED",
                "expected_route": "SEMANTIC_REQUIRED",
                "expected_failed_deterministic_stage": (
                    "COMMITMENT:UNRESOLVED_DECLARATION"
                ),
                "expected_permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
                "training_allowed": False,
                "tags": ["residual"],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_fixture_seed_runner_freezes_requests_and_hashes_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(fixture_seed_module, "PhaseAFixtureExporter", _FakeExporter)
    output = tmp_path / "evidence.json"
    seed = _seed(tmp_path / "seed.json")

    result = run_phase_a_fixture_seed_evidence(
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_path=output,
        seed_path=seed,
    )

    assert result["ok"] is True
    assert result["case_count"] == 2
    assert result["request_count"] == 1
    assert result["mismatch_count"] == 0
    assert result["training_allowed"] is False
    assert result["production_data"] is False
    assert result["integration_gate"] == "NO_GO"

    persisted = json.loads(output.read_text(encoding="utf-8"))
    payload_sha = persisted.pop("payload_sha256")
    assert payload_sha == hashlib.sha256(
        canonical_json(persisted).encode("utf-8")
    ).hexdigest()
    residual = next(row for row in persisted["cases"] if row["case_id"] == "residual")
    assert residual["actual"]["request"]["version"] == "semantic-residual-v2"
    assert residual["actual"]["request_sha256"]


def test_fixture_seed_runner_fails_closed_on_boundary_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(fixture_seed_module, "PhaseAFixtureExporter", _FakeExporter)
    output = tmp_path / "evidence.json"

    result = run_phase_a_fixture_seed_evidence(
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_path=output,
        seed_path=_seed(tmp_path / "seed.json", wrong_route=True),
    )

    assert result["ok"] is False
    assert result["mismatch_count"] == 1
    mismatch = result["mismatches"][0]
    assert mismatch["case_id"] == "zero-model"
    assert mismatch["mismatch"]["route"] == {
        "expected": "EXACT",
        "actual": "NO_ACTION",
    }
