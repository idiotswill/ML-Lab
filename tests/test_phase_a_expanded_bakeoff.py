from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import ml_lab.adapters.phase_a_expanded_bakeoff as bakeoff_module
from ml_lab import app
from ml_lab.datasets.leakage import canonical_json
from ml_lab.trainers.service import PHASE_A_CANDIDATE_TRAINER_ID


def _freeze_receipt() -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-freeze-receipt/1",
        "ok": True,
        "project_id": "project",
        "adapter_id": "frankenhomie.phase-a-residual",
        "adapter_version": "1.0.0",
        "target_frankenhomie_commit": "1" * 40,
        "target_contract": "semantic-residual-v2",
        "contract_snapshot": {
            "id": "snapshot",
            "commit_sha": "1" * 40,
            "contract_version": "semantic-residual-v2",
        },
        "dataset": {
            "id": "dataset",
            "state": "FROZEN",
            "example_count": 396,
            "partitions": {
                "TRAIN": {"example_count": 240},
                "DEV": {"example_count": 60},
                "TEST": {"example_count": 60},
                "REDTEAM": {"example_count": 36},
            },
        },
        "zero_model_redteam": {
            "case_count": 24,
            "passed_count": 24,
            "artifact_sha256": "a" * 64,
            "part_of_residual_dataset": False,
        },
        "ordinary_dataset_service": True,
        "ordinary_contract_snapshot_service": True,
        "trainer_handles": {"TRAIN": "train-digest", "DEV": "dev-digest"},
        "trainer_visible_splits": ["DEV", "TRAIN"],
        "protected_splits_in_trainer_handles": False,
        "evaluation_handles": {
            "TRAIN": "train-digest",
            "DEV": "dev-digest",
            "TEST": "test-digest",
            "REDTEAM": "redteam-digest",
        },
        "frozen": True,
        "model_training_started": False,
        "production_data": False,
        "transcript_derived": False,
        "integration_gate": "NO_GO",
    }
    return {
        **base,
        "payload_sha256": hashlib.sha256(
            canonical_json(base).encode()
        ).hexdigest(),
    }


def test_expanded_bakeoff_accepts_expected_frozen_shape() -> None:
    bakeoff_module._validate_expanded_freeze_receipt(_freeze_receipt())


def test_expanded_bakeoff_rejects_zero_model_regression() -> None:
    receipt = _freeze_receipt()
    zero = receipt["zero_model_redteam"]
    assert isinstance(zero, dict)
    zero["passed_count"] = 23
    base = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = hashlib.sha256(
        canonical_json(base).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="24/24 zero-model"):
        bakeoff_module._validate_expanded_freeze_receipt(receipt)


def test_expanded_bakeoff_rejects_partition_drift() -> None:
    receipt = _freeze_receipt()
    dataset = receipt["dataset"]
    assert isinstance(dataset, dict)
    partitions = dataset["partitions"]
    assert isinstance(partitions, dict)
    dev = partitions["DEV"]
    assert isinstance(dev, dict)
    dev["example_count"] = 59
    base = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = hashlib.sha256(
        canonical_json(base).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="partition counts changed"):
        bakeoff_module._validate_expanded_freeze_receipt(receipt)


def test_candidate_dev_gate_requires_zero_veto_failures() -> None:
    clean = {
        "trainer_id": PHASE_A_CANDIDATE_TRAINER_ID,
        "veto_failure_count": 0,
        "dev": {"total": 60, "veto_failures": 0},
    }
    assert bakeoff_module._dev_gate_passed(clean) is True

    veto = {
        **clean,
        "veto_failure_count": 1,
        "dev": {"total": 60, "veto_failures": 1},
    }
    assert bakeoff_module._dev_gate_passed(veto) is False

    incomplete = {
        **clean,
        "dev": {"total": 59, "veto_failures": 0},
    }
    assert bakeoff_module._dev_gate_passed(incomplete) is False

    wrong_candidate = {
        **clean,
        "trainer_id": "baseline.phase_a_deterministic_abstention.v2",
    }
    assert bakeoff_module._dev_gate_passed(wrong_candidate) is False


def test_expanded_dev_bakeoff_cli_forwards_workspace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "frozen-workspace"
    calls: list[Path] = []

    def fake_run(*, workspace_path: Path) -> dict[str, object]:
        calls.append(workspace_path)
        return {
            "schema": "ml-lab-phase-a-expanded-dev-bakeoff/1",
            "ok": True,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(
        bakeoff_module,
        "run_expanded_phase_a_dev_bakeoff",
        fake_run,
    )

    result = app.main(["--phase-a-expanded-dev-bakeoff", str(workspace)])

    assert result == 0
    assert calls == [workspace]
    assert '"ok": true' in capsys.readouterr().out
