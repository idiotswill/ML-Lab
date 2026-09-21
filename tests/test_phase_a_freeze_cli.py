from __future__ import annotations

import json
from pathlib import Path

import ml_lab.adapters.phase_a_freeze as freeze_module
from ml_lab import app


def test_phase_a_freeze_cli_requires_repo_and_factory_output(
    tmp_path: Path,
    capsys,
) -> None:
    result = app.main(
        ["--phase-a-freeze-dataset", str(tmp_path / "freeze")]
    )
    assert result == 19
    assert "--frankenhomie-repo is required" in capsys.readouterr().err

    result = app.main(
        [
            "--phase-a-freeze-dataset",
            str(tmp_path / "freeze"),
            "--frankenhomie-repo",
            str(tmp_path / "frankenhomie"),
        ]
    )
    assert result == 20
    assert "--phase-a-factory-output is required" in capsys.readouterr().err


def test_phase_a_freeze_cli_forwards_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    freeze_workspace = tmp_path / "freeze"
    factory_root = tmp_path / "factory"
    repository = tmp_path / "frankenhomie"
    calls: list[dict[str, Path]] = []

    def fake_freeze(
        *,
        frankenhomie_repository: Path,
        factory_receipt_path: Path,
        dataset_path: Path,
        workspace_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "frankenhomie_repository": frankenhomie_repository,
                "factory_receipt_path": factory_receipt_path,
                "dataset_path": dataset_path,
                "workspace_path": workspace_path,
            }
        )
        return {
            "schema": "ml-lab-phase-a-freeze-receipt/1",
            "ok": True,
            "frozen": True,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(
        freeze_module,
        "freeze_phase_a_train_dev",
        fake_freeze,
    )

    result = app.main(
        [
            "--phase-a-freeze-dataset",
            str(freeze_workspace),
            "--phase-a-factory-output",
            str(factory_root),
            "--frankenhomie-repo",
            str(repository),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "frankenhomie_repository": repository,
            "factory_receipt_path": factory_root / "factory-receipt.json",
            "dataset_path": factory_root / "phase-a-train-dev-v1.jsonl",
            "workspace_path": freeze_workspace,
        }
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["frozen"] is True
    assert printed["integration_gate"] == "NO_GO"
