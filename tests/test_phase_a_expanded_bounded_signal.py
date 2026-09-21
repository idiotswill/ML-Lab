from pathlib import Path

import ml_lab.adapters.phase_a_expanded_bounded_signal as bounded_module
from ml_lab import app


def test_expanded_bounded_provider_signal_cli_forwards_configuration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "frozen-workspace"
    calls: list[dict[str, object]] = []

    def fake_run(
        *,
        workspace_path: Path,
        model: str,
        endpoint: str,
        timeout_seconds: float,
    ) -> dict[str, object]:
        calls.append(
            {
                "workspace": workspace_path,
                "model": model,
                "endpoint": endpoint,
                "timeout": timeout_seconds,
            }
        )
        return {
            "schema": "ml-lab-phase-a-expanded-bounded-provider-signal/1",
            "ok": True,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(
        bounded_module,
        "run_expanded_phase_a_bounded_provider_signal",
        fake_run,
    )

    result = app.main(
        [
            "--phase-a-expanded-bounded-provider-signal",
            str(workspace),
            "--phase-a-provider-model",
            "fixture-model",
            "--phase-a-provider-endpoint",
            "http://127.0.0.1:11434/v1/chat/completions",
            "--phase-a-provider-timeout",
            "45",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "workspace": workspace,
            "model": "fixture-model",
            "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "timeout": 45.0,
        }
    ]
    assert '"ok": true' in capsys.readouterr().out
