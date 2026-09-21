from pathlib import Path

import ml_lab.adapters.phase_a_corpus as corpus_module
from ml_lab import app


def test_phase_a_generate_corpus_cli_forwards_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "corpus"
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    calls: list[dict[str, Path]] = []

    def fake_generate(
        *,
        output_dir: Path,
        plan_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "output_dir": output_dir,
                "plan_path": plan_path,
            }
        )
        return {
            "schema": "ml-lab-phase-a-corpus-generation-receipt/1",
            "ok": True,
            "case_count": 420,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(corpus_module, "generate_phase_a_corpus", fake_generate)

    result = app.main(
        [
            "--phase-a-generate-corpus",
            str(output),
            "--phase-a-corpus-plan",
            str(plan),
        ]
    )

    assert result == 0
    assert calls == [{"output_dir": output, "plan_path": plan}]
    printed = capsys.readouterr().out
    assert '"case_count": 420' in printed
    assert '"integration_gate": "NO_GO"' in printed
