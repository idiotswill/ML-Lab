from __future__ import annotations

import json
from pathlib import Path

import ml_lab.adapters.phase_a_dataset_factory as factory_module
from ml_lab import app


def test_phase_a_dataset_cli_requires_frankenhomie_repo(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "dataset"

    result = app.main(["--phase-a-build-dataset", str(output)])

    assert result == 16
    assert "--frankenhomie-repo is required" in capsys.readouterr().err
    assert not output.exists()


def test_phase_a_dataset_cli_runs_factory(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "dataset"
    repository = tmp_path / "frankenhomie"
    seed = tmp_path / "seed.json"
    protected = tmp_path / "protected.json"
    seed.write_text("{}\n", encoding="utf-8")
    protected.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run(
        *,
        workspace,
        frankenhomie_repository: Path,
        output_dir: Path,
        seed_path: Path,
        protected_seed_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "workspace_root": workspace.root,
                "frankenhomie_repository": frankenhomie_repository,
                "output_dir": output_dir,
                "seed_path": seed_path,
                "protected_seed_path": protected_seed_path,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "schema": "ml-lab-phase-a-dataset-factory-receipt/1",
            "ok": True,
            "candidate_training_data": True,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(
        factory_module,
        "run_phase_a_dataset_factory",
        fake_run,
    )

    result = app.main(
        [
            "--phase-a-build-dataset",
            str(output),
            "--frankenhomie-repo",
            str(repository),
            "--phase-a-dataset-seed",
            str(seed),
            "--phase-a-protected-seed",
            str(protected),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    assert calls[0]["frankenhomie_repository"] == repository
    assert calls[0]["output_dir"] == output
    assert calls[0]["seed_path"] == seed
    assert calls[0]["protected_seed_path"] == protected
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["candidate_training_data"] is True
    assert printed["integration_gate"] == "NO_GO"
