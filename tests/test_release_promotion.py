from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ml_lab.release.promotion import promote_testing_ready
from ml_lab.release.windows import create_windows_portable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    dist = tmp_path / "main.dist"
    dist.mkdir()
    executable = dist / "MLLab.exe"
    executable.write_bytes(b"exact-compiled-candidate")
    (dist / "support.dll").write_bytes(b"support")

    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    recipe = tmp_path / "MLLab.iss"
    recipe.write_text("PrivilegesRequired=lowest\n", encoding="utf-8")
    installer = tmp_path / "MLLab-Setup-v0.1.0-testing.1-x64.exe"
    installer.write_bytes(b"exact-installer")

    candidate_dir = tmp_path / "candidate"
    portable, manifest = create_windows_portable(
        dist_dir=dist,
        output_dir=candidate_dir,
        source_commit="abcdef0123456789",
        lock_path=lock,
        version="0.1.0-testing.1",
        release_label="v0.1.0-testing.1",
        python_version="3.12.10",
        qt_version="6.11.2",
        nuitka_version="2.8.9",
        installer_recipe=recipe,
        installer_path=installer,
        testing_candidate=True,
    )
    candidate_installer = candidate_dir / installer.name
    candidate_installer.write_bytes(installer.read_bytes())

    receipt = {
        "format_version": 2,
        "kind": "ML_LAB_PHASE4_REPRESENTATIVE_PERFORMANCE",
        "ok": True,
        "smoke": False,
        "gate_evaluable": True,
        "representative_hardware_review_required": True,
        "application_executable": {
            "filename": "MLLab.exe",
            "size_bytes": executable.stat().st_size,
            "sha256": _sha256(executable),
        },
        "hardware": {"os": "Windows", "cpu": "Representative CPU"},
        "ui": {
            "primary_rows": 100_000,
            "materialized_page_examples": 100,
            "background_rows_requested": 20_000,
            "idle_navigation": {
                "rounds_requested": 3,
                "rounds_completed": 3,
                "rounds_with_navigation_over_100ms": 1,
                "repeatable_navigation_over_100ms": False,
            },
        },
        "hard_checks": {
            "startup_under_4s": True,
            "idle_working_set_under_300mb": True,
            "ordinary_navigation_no_repeatable_over_100ms_stall": True,
            "bounded_project_paging": True,
            "background_import_no_over_100ms_stall": True,
            "worker_load_no_over_100ms_stall": True,
            "cancellation_visible_within_1s": True,
        },
        "hard_checks_pass": True,
        "instrumentation_checks": {
            "startup_interactive_signal": True,
            "idle_memory_measured": True,
            "ordinary_navigation_sampled": True,
            "ordinary_navigation_repeatability_sampled": True,
            "bounded_paging_observed": True,
            "background_import_completed_and_sampled": True,
            "worker_load_sampled": True,
            "cancellation_observed": True,
            "compiled_executable_fingerprinted": True,
        },
        "instrumentation_checks_pass": True,
        "testing_ready_authorized": False,
        "integration_gate": "NO_GO",
    }
    receipt_path = tmp_path / "representative-performance.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return candidate_dir, portable, manifest, receipt_path, receipt


def test_testing_ready_promotion_reuses_exact_candidate_bytes(tmp_path: Path) -> None:
    candidate_dir, portable, manifest, receipt_path, _receipt = _candidate(tmp_path)
    installer = candidate_dir / "MLLab-Setup-v0.1.0-testing.1-x64.exe"
    output = tmp_path / "release"

    outputs = promote_testing_ready(
        candidate_manifest_path=manifest,
        performance_receipt_path=receipt_path,
        candidate_dir=candidate_dir,
        output_dir=output,
    )

    promoted_installer, promoted_portable, promoted_receipt, release_manifest = outputs
    assert promoted_installer.read_bytes() == installer.read_bytes()
    assert promoted_portable.read_bytes() == portable.read_bytes()
    assert promoted_receipt.read_bytes() == receipt_path.read_bytes()

    payload = json.loads(release_manifest.read_text(encoding="utf-8"))
    assert payload["testing_ready"] is True
    assert payload["promotion_scope"] == "PRODUCT_TESTING_ONLY"
    assert payload["integration_gate"] == "NO_GO"
    assert payload["model_integration_approved"] is False
    assert payload["candidate_build"]["executable_sha256"] == _sha256(
        candidate_dir.parent / "main.dist" / "MLLab.exe"
    )
    assert payload["artifacts"]["installer_exe"]["sha256"] == _sha256(installer)
    assert payload["artifacts"]["portable_zip"]["sha256"] == _sha256(portable)


def test_testing_ready_promotion_rejects_smoke_receipt(tmp_path: Path) -> None:
    candidate_dir, _portable, manifest, receipt_path, receipt = _candidate(tmp_path)
    receipt["smoke"] = True
    receipt["gate_evaluable"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="gate-evaluable"):
        promote_testing_ready(
            candidate_manifest_path=manifest,
            performance_receipt_path=receipt_path,
            candidate_dir=candidate_dir,
            output_dir=tmp_path / "release",
        )


def test_testing_ready_promotion_rejects_other_executable(tmp_path: Path) -> None:
    candidate_dir, _portable, manifest, receipt_path, receipt = _candidate(tmp_path)
    executable = receipt["application_executable"]
    assert isinstance(executable, dict)
    executable["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="executable hash"):
        promote_testing_ready(
            candidate_manifest_path=manifest,
            performance_receipt_path=receipt_path,
            candidate_dir=candidate_dir,
            output_dir=tmp_path / "release",
        )



def test_testing_ready_promotion_rejects_old_performance_schema(tmp_path: Path) -> None:
    candidate_dir, _portable, manifest, receipt_path, receipt = _candidate(tmp_path)
    receipt["format_version"] = 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        promote_testing_ready(
            candidate_manifest_path=manifest,
            performance_receipt_path=receipt_path,
            candidate_dir=candidate_dir,
            output_dir=tmp_path / "release",
        )


def test_testing_ready_promotion_rejects_repeatable_navigation_stall(
    tmp_path: Path,
) -> None:
    candidate_dir, _portable, manifest, receipt_path, receipt = _candidate(tmp_path)
    ui = receipt["ui"]
    assert isinstance(ui, dict)
    ordinary = ui["idle_navigation"]
    assert isinstance(ordinary, dict)
    ordinary["rounds_with_navigation_over_100ms"] = 2
    ordinary["repeatable_navigation_over_100ms"] = True
    hard = receipt["hard_checks"]
    assert isinstance(hard, dict)
    hard["ordinary_navigation_no_repeatable_over_100ms_stall"] = False
    receipt["hard_checks_pass"] = False
    receipt["ok"] = False
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="did not pass"):
        promote_testing_ready(
            candidate_manifest_path=manifest,
            performance_receipt_path=receipt_path,
            candidate_dir=candidate_dir,
            output_dir=tmp_path / "release",
        )
