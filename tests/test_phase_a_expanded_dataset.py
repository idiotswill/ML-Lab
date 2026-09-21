from __future__ import annotations

import json
from pathlib import Path

import ml_lab.adapters.phase_a_expanded_dataset as expanded_module
from ml_lab import app
from ml_lab.storage.workspace import Workspace


def test_expanded_builder_orchestrates_authored_and_authoritative_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_generate(*, output_dir: Path, plan_path: Path) -> dict[str, object]:
        calls.append(("generate", output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase-a-expanded-train-dev-seed-v1.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (output_dir / "phase-a-expanded-protected-language-v1.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "ok": True,
            "case_count": 420,
            "payload_sha256": "a" * 64,
        }

    def fake_factory(
        *,
        workspace: Workspace,
        frankenhomie_repository: Path,
        output_dir: Path,
        seed_path: Path,
        protected_seed_path: Path,
        case_cache_dir: Path,
    ) -> dict[str, object]:
        del workspace, frankenhomie_repository
        calls.append(("factory", output_dir))
        assert seed_path.name == "phase-a-expanded-train-dev-seed-v1.json"
        assert (
            protected_seed_path.name
            == "phase-a-expanded-protected-language-v1.json"
        )
        assert case_cache_dir.name == "case-cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "ok": True,
            "target_frankenhomie_commit": "1" * 40,
            "target_contract": "semantic-residual-v2",
            "case_count": 300,
            "emitted_count": 300,
            "split_counts": {"TRAIN": 240, "DEV": 60},
            "payload_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "errors": [],
            "case_cache": {
                "enabled": True,
                "export_hits": 0,
                "export_misses": 300,
                "reference_hits": 0,
                "reference_misses": 300,
            },
        }

    monkeypatch.setattr(expanded_module, "generate_phase_a_corpus", fake_generate)
    monkeypatch.setattr(
        expanded_module,
        "run_phase_a_dataset_factory",
        fake_factory,
    )

    root = tmp_path / "expanded"
    result = expanded_module.build_expanded_phase_a_dataset(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=root,
        plan_path=tmp_path / "plan.json",
    )

    assert result["ok"] is True
    assert result["authored_case_count"] == 420
    assert result["authoritative_candidate_case_count"] == 300
    assert result["authoritative_emitted_count"] == 300
    assert result["authoritative_split_counts"] == {"TRAIN": 240, "DEV": 60}
    assert result["training_started"] is False
    assert result["frozen"] is False
    assert result["integration_gate"] == "NO_GO"
    assert calls == [
        ("generate", root / "authored"),
        ("factory", root / "authoritative"),
    ]
    frozen = json.loads(
        (root / "expanded-build-receipt.json").read_text(encoding="utf-8")
    )
    assert frozen == result


def test_expanded_builder_preserves_authoritative_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_generate(*, output_dir: Path, plan_path: Path) -> dict[str, object]:
        del plan_path
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "phase-a-expanded-train-dev-seed-v1.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (output_dir / "phase-a-expanded-protected-language-v1.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return {
            "ok": True,
            "case_count": 420,
            "payload_sha256": "a" * 64,
        }

    def fake_factory(**_kwargs) -> dict[str, object]:
        return {
            "ok": False,
            "target_frankenhomie_commit": "1" * 40,
            "target_contract": "semantic-residual-v2",
            "case_count": 300,
            "emitted_count": 0,
            "split_counts": {},
            "payload_sha256": "b" * 64,
            "dataset_sha256": None,
            "errors": [
                {
                    "case_id": "train-example",
                    "code": "NOT_RESIDUAL_EXPORTED",
                }
            ],
            "case_cache": {
                "enabled": True,
                "export_hits": 299,
                "export_misses": 1,
                "reference_hits": 299,
                "reference_misses": 0,
            },
        }

    monkeypatch.setattr(expanded_module, "generate_phase_a_corpus", fake_generate)
    monkeypatch.setattr(
        expanded_module,
        "run_phase_a_dataset_factory",
        fake_factory,
    )

    result = expanded_module.build_expanded_phase_a_dataset(
        workspace=Workspace.create(tmp_path / "workspace"),
        frankenhomie_repository=tmp_path / "frankenhomie",
        output_dir=tmp_path / "expanded",
        plan_path=tmp_path / "plan.json",
    )

    assert result["ok"] is False
    assert result["authoritative_emitted_count"] == 0
    assert result["dataset_sha256"] is None
    assert result["errors"] == [
        {
            "case_id": "train-example",
            "code": "NOT_RESIDUAL_EXPORTED",
        }
    ]
    assert result["integration_gate"] == "NO_GO"


def test_expanded_builder_cli_requires_frankenhomie_repo(
    tmp_path: Path,
    capsys,
) -> None:
    result = app.main(
        ["--phase-a-build-expanded-dataset", str(tmp_path / "expanded")]
    )

    assert result == 25
    assert "--frankenhomie-repo is required" in capsys.readouterr().err


def test_expanded_builder_cli_uses_persistent_working_workspace(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "expanded"
    repository = tmp_path / "frankenhomie"
    calls: list[dict[str, object]] = []

    def fake_build(
        *,
        workspace: Workspace,
        frankenhomie_repository: Path,
        output_dir: Path,
        plan_path: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "workspace": workspace.root,
                "repository": frankenhomie_repository,
                "output": output_dir,
                "plan": plan_path,
            }
        )
        return {
            "schema": "ml-lab-phase-a-expanded-authoritative-build/1",
            "ok": True,
            "integration_gate": "NO_GO",
        }

    monkeypatch.setattr(expanded_module, "build_expanded_phase_a_dataset", fake_build)

    result = app.main(
        [
            "--phase-a-build-expanded-dataset",
            str(output),
            "--frankenhomie-repo",
            str(repository),
        ]
    )

    assert result == 0
    assert calls[0]["workspace"] == (output / "working-workspace").resolve()
    assert calls[0]["repository"] == repository
    assert calls[0]["output"] == output.resolve()
    assert (output / "working-workspace" / ".ml-lab-workspace.json").is_file()
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["integration_gate"] == "NO_GO"
