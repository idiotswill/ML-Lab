from __future__ import annotations

import json
from pathlib import Path

import ml_lab.adapters.phase_a_baselines as baseline_module
from ml_lab import app


def test_phase_a_baseline_cli_requires_frankenhomie_repo(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "baseline-evidence.json"

    result = app.main(["--phase-a-baseline-evidence", str(output)])

    assert result == 14
    assert "--frankenhomie-repo is required" in capsys.readouterr().err
    assert not output.exists()


def test_phase_a_baseline_cli_runs_frozen_inputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "baseline-evidence.json"
    repository = tmp_path / "frankenhomie"
    receipt = tmp_path / "receipt.json"
    labels = tmp_path / "labels.json"
    receipt.write_text("{}\n", encoding="utf-8")
    labels.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(
        *,
        workspace,
        frankenhomie_repository: Path,
        receipt_path: Path,
        labels_path: Path,
        output_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "workspace_root": workspace.root,
                "frankenhomie_repository": frankenhomie_repository,
                "receipt_path": receipt_path,
                "labels_path": labels_path,
                "output_path": output_path,
            }
        )
        output_path.write_text('{"ok":true}\n', encoding="utf-8")
        return {
            "schema": "ml-lab-phase-a-baseline-evidence/1",
            "integration_gate": "NO_GO",
            "reports": [],
        }

    monkeypatch.setattr(
        baseline_module,
        "run_phase_a_baselines",
        fake_run,
    )

    result = app.main(
        [
            "--phase-a-baseline-evidence",
            str(output),
            "--frankenhomie-repo",
            str(repository),
            "--phase-a-baseline-receipt",
            str(receipt),
            "--phase-a-baseline-labels",
            str(labels),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["frankenhomie_repository"] == repository
    assert calls[0]["receipt_path"] == receipt
    assert calls[0]["labels_path"] == labels
    assert calls[0]["output_path"] == output
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == "ml-lab-phase-a-baseline-evidence/1"
    assert printed["integration_gate"] == "NO_GO"
