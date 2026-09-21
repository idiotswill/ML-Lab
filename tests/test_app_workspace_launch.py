from pathlib import Path

from ml_lab import app


def test_gui_workspace_argument_is_forwarded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "phase-a-freeze-v1"
    calls: list[Path | None] = []

    def fake_run_gui(workspace_path: Path | None = None) -> int:
        calls.append(workspace_path)
        return 0

    monkeypatch.setattr(app, "run_gui", fake_run_gui)

    result = app.main(["--workspace", str(workspace)])

    assert result == 0
    assert calls == [workspace]
