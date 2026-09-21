from __future__ import annotations

import json
from pathlib import Path

from ml_lab import app
import ml_lab.adapters.phase_a_fixture_seed as fixture_seed_module


def test_fixture_evidence_cli_requires_frankenhomie_repo(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "evidence.json"

    result = app.main(["--phase-a-fixture-seed-evidence", str(output)])

    assert result == 11
    assert "--frankenhomie-repo is required" in capsys.readouterr().err
    assert not output.exists()


def test_fixture_evidence_cli_runs_receipt_generator(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "evidence.json"
    repository = tmp_path / "frankenhomie"
    seed = tmp_path / "seed.json"
    seed.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, Path]] = []

    def fake_run(
        *,
        frankenhomie_repository: Path,
        output_path: Path,
        seed_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "frankenhomie_repository": frankenhomie_repository,
                "output_path": output_path,
                "seed_path": seed_path,
            }
        )
        output_path.write_text('{"ok":true}\n', encoding="utf-8")
        return {
            "ok": True,
            "payload_sha256": "a" * 64,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(
        fixture_seed_module,
        "run_phase_a_fixture_seed_evidence",
        fake_run,
    )

    result = app.main(
        [
            "--phase-a-fixture-seed-evidence",
            str(output),
            "--frankenhomie-repo",
            str(repository),
            "--phase-a-fixture-seed",
            str(seed),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "frankenhomie_repository": repository,
            "output_path": output,
            "seed_path": seed,
        }
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["integration_gate"] == "NO_GO"
