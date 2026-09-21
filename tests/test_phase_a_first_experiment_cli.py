from pathlib import Path

import ml_lab.adapters.phase_a_first_experiment as runner
from ml_lab import app


def test_first_experiment_cli_forwards_workspace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "phase-a-freeze-v1"
    calls: list[Path] = []

    def fake_run(*, workspace_path: Path) -> dict[str, object]:
        calls.append(workspace_path)
        return {
            "schema": "ml-lab-phase-a-first-sparse-experiment-receipt/1",
            "ok": True,
            "stage": "EXPERIMENT",
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(runner, "run_phase_a_first_sparse_experiment", fake_run)

    result = app.main(["--phase-a-first-experiment", str(workspace)])

    assert result == 0
    assert calls == [workspace]
    assert '"ok": true' in capsys.readouterr().out
