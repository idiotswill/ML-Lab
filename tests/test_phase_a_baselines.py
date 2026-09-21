from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ml_lab.adapters.phase_a_baselines as baseline_module
from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_baselines import (
    V2BoundedLexical,
    run_phase_a_baselines,
)
from ml_lab.storage.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "benchmarks"
    / "phase_a"
    / "receipts"
    / "fixture-evidence-1ff3e215-v1.json"
)
LABELS = ROOT / "benchmarks" / "phase_a" / "fixture_labels_v1.json"

_REGISTRY = {
    "schema_version": "asterra-semantic-family-registry/1",
    "families": [
        {
            "family": "HARM_TARGET",
            "entity_slot": "TARGET_COMBATANT",
            "deterministic_aliases": ["attack"],
            "residual_hints": [],
        },
        {
            "family": "INTERACT_OBJECT",
            "entity_slot": "OBJECT",
            "deterministic_aliases": ["open", "close"],
            "residual_hints": [],
            "operation_aliases": {"OPEN": ["open"], "CLOSE": ["close"]},
        },
        {
            "family": "MOVE_TRAVEL",
            "entity_slot": "DESTINATION",
            "deterministic_aliases": ["go"],
            "residual_hints": [],
        },
        {
            "family": "SEARCH_INSPECT",
            "entity_slot": "SUBJECT",
            "deterministic_aliases": ["inspect", "search"],
            "residual_hints": [],
        },
        {
            "family": "SPEECH_ONLY",
            "entity_slot": "ADDRESSEE",
            "deterministic_aliases": ["say"],
            "residual_hints": [],
        },
    ],
}


class _FakeReference:
    source_root: Path | None = None

    def __init__(self, _workspace: Workspace):
        pass

    def materialize(self, _repository: Path, ref: str):
        assert ref == "1ff3e2155a3c2d2b316023e9933edf3a6d53697f"
        assert self.source_root is not None
        return SimpleNamespace(commit_sha=ref, source_root=self.source_root)

    def validate(
        self,
        *,
        repository: Path,
        ref: str,
        request: object,
        proposal: object,
        timeout_seconds: float = 30.0,
    ):
        del repository, ref, timeout_seconds
        assert isinstance(request, dict)
        assert isinstance(proposal, dict)
        validation = PhaseAResidualAdapter().validate_proposal(request, proposal)
        return SimpleNamespace(
            status="ACCEPTED" if validation.accepted else "REJECTED",
            error_code=validation.error_code,
        )


def _install_fake_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "materialized"
    registry_path = (
        source_root
        / "frankenhomie-asterra-v0.9.0"
        / "data"
        / "semantic_family_registry.json"
    )
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(_REGISTRY, sort_keys=True),
        encoding="utf-8",
    )
    _FakeReference.source_root = source_root
    monkeypatch.setattr(
        baseline_module,
        "PhaseAReferenceValidator",
        _FakeReference,
    )


def test_v2_baselines_establish_zero_coverage_safe_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reference(tmp_path, monkeypatch)
    workspace = Workspace.create(tmp_path / "workspace")
    output = tmp_path / "baseline-evidence.json"

    result = run_phase_a_baselines(
        workspace=workspace,
        frankenhomie_repository=tmp_path / "frankenhomie",
        receipt_path=RECEIPT,
        labels_path=LABELS,
        output_path=output,
    )

    assert result["schema"] == "ml-lab-phase-a-baseline-evidence/1"
    assert result["training_allowed"] is False
    assert result["production_data"] is False
    assert result["integration_gate"] == "NO_GO"
    assert result["source_receipt_payload_sha256"] == (
        "15af7aa04a9546fc4db6c8fae71706efa9de6f5bec364dfa84e903810d2d805c"
    )

    reports = {
        report["candidate"]: report
        for report in result["reports"]
    }
    assert set(reports) == {
        "deterministic-abstention-v2",
        "bounded-lexical-v2",
    }
    for report in reports.values():
        assert report["case_count"] == 3
        assert report["decision_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
        assert report["semantic_accuracy"] == pytest.approx(2 / 3, abs=0.0001)
        assert report["useful_resolution_coverage"] == 0.0
        assert report["resolution_precision"] is None
        assert report["ask_player_accuracy"] == 1.0
        assert report["false_commitments"] == 0
        assert report["contract_failures"] == 0
        assert all(row["reference_status"] == "ACCEPTED" for row in report["rows"])

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["payload_sha256"] == result["payload_sha256"]


def test_bounded_lexical_v2_can_resolve_registered_family_and_visible_slot() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    row = next(
        case
        for case in receipt["cases"]
        if case["case_id"] == "seed-unfamiliar-scrutinize"
    )
    request = row["actual"]["request"]
    request["assessment"]["declaration"] = "I inspect Brass Hatch."
    request["assessment"]["clauses"][0]["raw"] = "I inspect Brass Hatch."

    proposal = V2BoundedLexical(_REGISTRY).predict(request)

    assert proposal["decision"] == "RESOLVE"
    assert proposal["action_family"] == "SEARCH_INSPECT"
    assert proposal["slots"] == [{"name": "SUBJECT", "value": "object:hatch"}]
    assert PhaseAResidualAdapter().validate_proposal(request, proposal).accepted is True


def test_baseline_runner_refuses_labels_bound_to_other_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reference(tmp_path, monkeypatch)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    labels["receipt_payload_sha256"] = "0" * 64
    labels_path = tmp_path / "wrong-labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    with pytest.raises(ValueError, match="not bound"):
        run_phase_a_baselines(
            workspace=Workspace.create(tmp_path / "workspace"),
            frankenhomie_repository=tmp_path / "frankenhomie",
            receipt_path=RECEIPT,
            labels_path=labels_path,
            output_path=tmp_path / "baseline-evidence.json",
        )
