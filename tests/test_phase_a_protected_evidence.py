from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_lab.adapters.phase_a_expanded_completion as completion_module
import ml_lab.adapters.phase_a_protected_evidence as protected_module
from ml_lab import app
from ml_lab.adapters.phase_a_corpus import generate_phase_a_corpus
from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExportReceipt
from ml_lab.adapters.phase_a_reference import ReferenceValidationReceipt
from ml_lab.storage.workspace import Workspace


class _FakeAdapter:
    def validate_proposal(self, _request, _proposal):
        return SimpleNamespace(accepted=True, error_code=None)


def _export_receipt(
    fixture: dict[str, object],
    *,
    zero_fail_case: str | None = None,
) -> PhaseAFixtureExportReceipt:
    fixture_id = str(fixture["fixture_id"])
    is_zero = fixture_id.startswith("zero:")
    case_id = fixture_id.split(":", 1)[1]
    fail = zero_fail_case == case_id
    request = None if is_zero else {
        "assessment": {"declaration": fixture["declaration"]},
        "context": {},
        "family_slots": [],
    }
    return PhaseAFixtureExportReceipt(
        status=(
            "RESIDUAL_EXPORTED"
            if not is_zero
            else ("RESIDUAL_EXPORTED" if fail else "TERMINATED_BEFORE_RESIDUAL")
        ),
        commit_sha="1ff3e2155a3c2d2b316023e9933edf3a6d53697f",
        contract_version="semantic-residual-v2",
        fixture_id=fixture_id,
        fixture_sha256="a" * 64,
        route=("SEMANTIC_REQUIRED" if not is_zero or fail else "NO_ACTION"),
        failed_deterministic_stage=None,
        request_sha256=("b" * 64 if request is not None else None),
        request=request,
        fresh_process=True,
        ephemeral_sqlite_only=True,
        network_access_allowed=False,
        resolver_dispatch_available=False,
        authority_mutation_allowed=False,
        error_code=None,
        error_message=None,
        receipt_artifact_digest=f"artifact:{case_id}",
    )


def _reference_receipt(request, proposal) -> ReferenceValidationReceipt:
    del request, proposal
    return ReferenceValidationReceipt(
        status="ACCEPTED",
        commit_sha="1ff3e2155a3c2d2b316023e9933edf3a6d53697f",
        contract_version="semantic-residual-v2",
        request_sha256="b" * 64,
        proposal_sha256="c" * 64,
        validator="fake",
        fresh_process=True,
        authority_mutation_allowed=False,
        error_code=None,
        error_message=None,
        receipt_artifact_digest="reference",
    )


def _install_protected_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    zero_fail_case: str | None = None,
) -> None:
    monkeypatch.setattr(protected_module, "PhaseAResidualAdapter", _FakeAdapter)
    monkeypatch.setattr(
        protected_module,
        "_hidden_fixture_leaks",
        lambda **_kwargs: [],
    )

    def fake_export(**kwargs):
        return _export_receipt(
            dict(kwargs["fixture"]),
            zero_fail_case=zero_fail_case,
        ), False

    def fake_reference(**kwargs):
        return _reference_receipt(kwargs["request"], kwargs["proposal"]), False

    monkeypatch.setattr(protected_module, "_cached_export", fake_export)
    monkeypatch.setattr(
        protected_module,
        "_cached_reference_validation",
        fake_reference,
    )


def test_protected_builder_emits_96_residual_and_keeps_24_zero_model_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = tmp_path / "authored"
    generate_phase_a_corpus(output_dir=authored)
    _install_protected_fakes(monkeypatch)

    result = protected_module.build_phase_a_protected_evidence(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        authored_root=authored,
        output_dir=tmp_path / "protected",
        case_cache_dir=tmp_path / "cache",
    )

    assert result["ok"] is True
    assert result["residual_case_count"] == 96
    assert result["residual_emitted_count"] == 96
    assert result["residual_split_counts"] == {"TEST": 60, "REDTEAM": 36}
    assert result["zero_model_case_count"] == 24
    assert result["zero_model_passed_count"] == 24
    rows = [
        json.loads(line)
        for line in (
            tmp_path / "protected" / "phase-a-protected-residual-v1.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 96
    assert sum(row["split"] == "TEST" for row in rows) == 60
    assert sum(row["split"] == "REDTEAM" for row in rows) == 36
    assert all(not row["example_id"].startswith("redteam-zero-") for row in rows)
    zero = json.loads(
        (
            tmp_path / "protected" / "phase-a-zero-model-evidence-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert zero["case_count"] == 24
    assert zero["passed_count"] == 24


def test_zero_model_failure_withholds_protected_residual_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = tmp_path / "authored"
    generate_phase_a_corpus(output_dir=authored)
    _install_protected_fakes(
        monkeypatch,
        zero_fail_case="redteam-zero-negation-01",
    )

    result = protected_module.build_phase_a_protected_evidence(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        authored_root=authored,
        output_dir=tmp_path / "protected",
        case_cache_dir=tmp_path / "cache",
    )

    assert result["ok"] is False
    assert result["residual_emitted_count"] == 0
    assert any(
        row["code"] == "ZERO_MODEL_INVARIANT_FAILED"
        for row in result["errors"]
    )
    assert not (
        tmp_path / "protected" / "phase-a-protected-residual-v1.jsonl"
    ).exists()


def test_completion_freezes_only_after_protected_evidence_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "expanded"
    Workspace.create(root / "working-workspace")
    calls: list[str] = []

    def fake_protected(**_kwargs):
        calls.append("protected")
        return {
            "ok": True,
            "payload_sha256": "a" * 64,
            "residual_case_count": 96,
            "residual_emitted_count": 96,
            "zero_model_case_count": 24,
            "zero_model_passed_count": 24,
            "errors": [],
        }

    def fake_freeze(**_kwargs):
        calls.append("freeze")
        return {
            "ok": True,
            "payload_sha256": "b" * 64,
            "frozen": True,
            "dataset": {"example_count": 396},
        }

    monkeypatch.setattr(
        completion_module,
        "build_phase_a_protected_evidence",
        fake_protected,
    )
    monkeypatch.setattr(completion_module, "freeze_expanded_phase_a", fake_freeze)

    result = completion_module.complete_expanded_phase_a(
        expanded_root=root,
        frankenhomie_repository=tmp_path / "frankenhomie",
    )

    assert calls == ["protected", "freeze"]
    assert result["ok"] is True
    assert result["protected_residual_emitted_count"] == 96
    assert result["zero_model_passed_count"] == 24
    assert result["frozen_dataset_example_count"] == 396
    assert result["frozen"] is True
    assert result["model_training_started"] is False
    assert result["integration_gate"] == "NO_GO"


def test_completion_does_not_freeze_after_protected_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "expanded"
    Workspace.create(root / "working-workspace")

    monkeypatch.setattr(
        completion_module,
        "build_phase_a_protected_evidence",
        lambda **_kwargs: {
            "ok": False,
            "payload_sha256": "a" * 64,
            "residual_case_count": 96,
            "residual_emitted_count": 0,
            "zero_model_case_count": 24,
            "zero_model_passed_count": 23,
            "errors": [{"code": "ZERO_MODEL_INVARIANT_FAILED"}],
        },
    )
    monkeypatch.setattr(
        completion_module,
        "freeze_expanded_phase_a",
        lambda **_kwargs: pytest.fail("freeze must not run"),
    )

    result = completion_module.complete_expanded_phase_a(
        expanded_root=root,
        frankenhomie_repository=tmp_path / "frankenhomie",
    )

    assert result["ok"] is False
    assert result["frozen"] is False
    assert result["protected_errors"] == [
        {"code": "ZERO_MODEL_INVARIANT_FAILED"}
    ]


def test_complete_expanded_cli_requires_repo(
    tmp_path: Path,
    capsys,
) -> None:
    result = app.main(
        ["--phase-a-complete-expanded-dataset", str(tmp_path / "expanded")]
    )
    assert result == 28
    assert "--frankenhomie-repo is required" in capsys.readouterr().err
