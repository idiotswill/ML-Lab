from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

_FULL_ROWS = 100_000
_FULL_IMPORT_ROWS = 20_000
_PAGE_SIZE = 100
_ORDINARY_NAVIGATION_ROUNDS = 3
_REPEATABLE_STALL_ROUNDS = 2


def promote_testing_ready(
    *,
    candidate_manifest_path: Path,
    performance_receipt_path: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Promote exact tested candidate bytes to a testing-ready product handoff.

    This is product-release evidence only. It never changes model promotion state
    and always preserves Frankenhomie integration as NO_GO.
    """
    candidate = _read_object(candidate_manifest_path)
    receipt = _read_object(performance_receipt_path)
    _validate_candidate(candidate)
    _validate_performance_receipt(receipt)

    application = _require_dict(candidate, "application")
    release_label = _require_str(application, "release_label")
    source = _require_dict(candidate, "source")
    source_commit = _require_str(source, "commit")

    candidate_executable = _candidate_executable(candidate)
    receipt_executable = _require_dict(receipt, "application_executable")
    receipt_executable_sha = _require_sha256(receipt_executable, "sha256")
    receipt_executable_size = _require_int(receipt_executable, "size_bytes")
    if receipt_executable_sha != candidate_executable["sha256"]:
        raise ValueError(
            "Representative performance receipt executable hash does not match "
            "the candidate MLLab.exe hash."
        )
    if receipt_executable_size != candidate_executable["size"]:
        raise ValueError(
            "Representative performance receipt executable size does not match "
            "the candidate MLLab.exe size."
        )

    artifacts = _require_dict(candidate, "artifacts")
    portable = _require_dict(artifacts, "portable_zip")
    installer = _require_dict(artifacts, "installer_exe")
    portable_source = _verified_artifact(candidate_dir, portable, "portable_zip")
    installer_source = _verified_artifact(candidate_dir, installer, "installer_exe")

    output_dir.mkdir(parents=True, exist_ok=True)
    portable_target = output_dir / portable_source.name
    installer_target = output_dir / installer_source.name
    shutil.copyfile(portable_source, portable_target)
    shutil.copyfile(installer_source, installer_target)

    if _sha256_file(portable_target) != _require_sha256(portable, "sha256"):
        raise RuntimeError("Promoted portable artifact changed during copy.")
    if _sha256_file(installer_target) != _require_sha256(installer, "sha256"):
        raise RuntimeError("Promoted installer artifact changed during copy.")

    performance_target = (
        output_dir / f"MLLab-{release_label}-representative-performance.json"
    )
    shutil.copyfile(performance_receipt_path, performance_target)

    candidate_manifest_target = output_dir / candidate_manifest_path.name
    shutil.copyfile(candidate_manifest_path, candidate_manifest_target)

    hardware = _require_dict(receipt, "hardware")
    release_manifest = {
        "schema_version": 1,
        "release_kind": "testing_ready_windows_release",
        "testing_ready": True,
        "promotion_scope": "PRODUCT_TESTING_ONLY",
        "integration_gate": "NO_GO",
        "model_integration_approved": False,
        "application": application,
        "source": {"commit": source_commit},
        "candidate_build": {
            "manifest_filename": candidate_manifest_target.name,
            "manifest_sha256": _sha256_file(candidate_manifest_target),
            "executable_sha256": candidate_executable["sha256"],
            "executable_size": candidate_executable["size"],
        },
        "representative_performance": {
            "receipt_filename": performance_target.name,
            "receipt_sha256": _sha256_file(performance_target),
            "hardware": hardware,
            "hard_checks": _require_dict(receipt, "hard_checks"),
            "instrumentation_checks": _require_dict(
                receipt,
                "instrumentation_checks",
            ),
        },
        "artifacts": {
            "installer_exe": {
                "filename": installer_target.name,
                "size": installer_target.stat().st_size,
                "sha256": _sha256_file(installer_target),
            },
            "portable_zip": {
                "filename": portable_target.name,
                "size": portable_target.stat().st_size,
                "sha256": _sha256_file(portable_target),
            },
        },
    }
    release_manifest_path = (
        output_dir / f"MLLab-{release_label}-testing-ready-manifest.json"
    )
    _atomic_json(release_manifest_path, release_manifest)
    return (
        installer_target,
        portable_target,
        performance_target,
        release_manifest_path,
    )


def _validate_candidate(candidate: Mapping[str, object]) -> None:
    if _require_str(candidate, "build_kind") != "testing_candidate_windows_package":
        raise ValueError("Candidate build must be a testing_candidate_windows_package.")
    if _require_bool(candidate, "testing_ready"):
        raise ValueError("Candidate build must not already claim testing_ready.")
    if _require_str(candidate, "integration_gate") != "NO_GO":
        raise ValueError("Candidate build must retain integration_gate NO_GO.")
    application = _require_dict(candidate, "application")
    release_label = _require_str(application, "release_label")
    if not release_label.startswith("v0.1.0-testing."):
        raise ValueError("Candidate release label is not a Phase 4 testing label.")
    _require_str(application, "version")
    _require_str(_require_dict(candidate, "source"), "commit")
    _candidate_executable(candidate)


def _validate_performance_receipt(receipt: Mapping[str, object]) -> None:
    if _require_int(receipt, "format_version") != 2:
        raise ValueError("Representative performance receipt schema is not current.")
    if _require_str(receipt, "kind") != "ML_LAB_PHASE4_REPRESENTATIVE_PERFORMANCE":
        raise ValueError("Unexpected representative performance receipt kind.")
    if not _require_bool(receipt, "gate_evaluable"):
        raise ValueError("Performance receipt is not gate-evaluable.")
    if _require_bool(receipt, "smoke"):
        raise ValueError("Performance smoke evidence cannot authorize testing-ready.")
    if not _require_bool(receipt, "representative_hardware_review_required"):
        raise ValueError("Performance receipt is missing representative-hardware scope.")
    if not _require_bool(receipt, "ok"):
        raise ValueError("Representative performance receipt did not pass.")
    if not _require_bool(receipt, "hard_checks_pass"):
        raise ValueError("Representative performance hard checks did not pass.")
    if not _require_bool(receipt, "instrumentation_checks_pass"):
        raise ValueError("Representative performance instrumentation did not pass.")
    if _require_bool(receipt, "testing_ready_authorized"):
        raise ValueError("Performance receipt must not self-authorize testing-ready.")
    if _require_str(receipt, "integration_gate") != "NO_GO":
        raise ValueError("Performance receipt must retain integration_gate NO_GO.")

    hardware = _require_dict(receipt, "hardware")
    if _require_str(hardware, "os") != "Windows":
        raise ValueError("Representative performance evidence must come from Windows.")

    hard_checks = _require_dict(receipt, "hard_checks")
    if not hard_checks or not all(value is True for value in hard_checks.values()):
        raise ValueError("Every representative performance hard check must pass.")
    if hard_checks.get("ordinary_navigation_no_repeatable_over_100ms_stall") is not True:
        raise ValueError("Ordinary navigation repeatability gate did not pass.")
    instrumentation = _require_dict(receipt, "instrumentation_checks")
    if not instrumentation or not all(value is True for value in instrumentation.values()):
        raise ValueError("Every performance instrumentation check must pass.")

    ui = _require_dict(receipt, "ui")
    if _require_int(ui, "primary_rows") != _FULL_ROWS:
        raise ValueError("Representative performance receipt must exercise 100,000 rows.")
    if _require_int(ui, "materialized_page_examples") != _PAGE_SIZE:
        raise ValueError("Representative performance paging must remain bounded to 100.")
    if _require_int(ui, "background_rows_requested") != _FULL_IMPORT_ROWS:
        raise ValueError("Representative background import workload is incomplete.")

    ordinary = _require_dict(ui, "idle_navigation")
    if _require_int(ordinary, "rounds_requested") != _ORDINARY_NAVIGATION_ROUNDS:
        raise ValueError("Ordinary navigation repeatability round count is incomplete.")
    if _require_int(ordinary, "rounds_completed") != _ORDINARY_NAVIGATION_ROUNDS:
        raise ValueError("Ordinary navigation repeatability rounds did not complete.")
    if _require_bool(ordinary, "repeatable_navigation_over_100ms"):
        raise ValueError("Ordinary navigation contains a repeatable >100 ms GUI stall.")
    if _require_int(ordinary, "rounds_with_navigation_over_100ms") >= _REPEATABLE_STALL_ROUNDS:
        raise ValueError("Ordinary navigation repeatability evidence is inconsistent.")
    if instrumentation.get("ordinary_navigation_repeatability_sampled") is not True:
        raise ValueError("Ordinary navigation repeatability instrumentation is missing.")

    executable = _require_dict(receipt, "application_executable")
    if _require_str(executable, "filename").casefold() != "mllab.exe":
        raise ValueError("Performance receipt is not bound to MLLab.exe.")
    _require_sha256(executable, "sha256")
    if _require_int(executable, "size_bytes") <= 0:
        raise ValueError("Performance receipt executable size is invalid.")


def _candidate_executable(candidate: Mapping[str, object]) -> dict[str, object]:
    tree = _require_dict(candidate, "standalone_tree")
    files = _require_list(tree, "files")
    for raw in files:
        if not isinstance(raw, dict):
            continue
        item = {str(key): value for key, value in raw.items()}
        if item.get("path") != "MLLab.exe":
            continue
        size = _require_int(item, "size")
        if size <= 0:
            raise ValueError("Candidate MLLab.exe size is invalid.")
        return {
            "sha256": _require_sha256(item, "sha256"),
            "size": size,
        }
    raise ValueError("Candidate manifest does not contain MLLab.exe.")


def _verified_artifact(
    candidate_dir: Path,
    evidence: Mapping[str, object],
    kind: str,
) -> Path:
    filename = _require_str(evidence, "filename")
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe {kind} filename in candidate manifest.")
    source = candidate_dir / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size != _require_int(evidence, "size"):
        raise ValueError(f"Candidate {kind} size does not match its manifest.")
    if _sha256_file(source) != _require_sha256(evidence, "sha256"):
        raise ValueError(f"Candidate {kind} hash does not match its manifest.")
    return source


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _require_dict(value: Mapping[str, object], key: str) -> dict[str, object]:
    raw = value.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object.")
    return {str(child_key): child for child_key, child in raw.items()}


def _require_list(value: Mapping[str, object], key: str) -> list[object]:
    raw = value.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be an array.")
    return list(raw)


def _require_str(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return raw


def _require_bool(value: Mapping[str, object], key: str) -> bool:
    raw = value.get(key)
    if not isinstance(raw, bool):
        raise ValueError(f"{key} must be a boolean.")
    return raw


def _require_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer.")
    return raw


def _require_sha256(value: Mapping[str, object], key: str) -> str:
    raw = _require_str(value, key)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest.")
    return raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote exact tested Phase 4 candidate bytes to testing-ready."
    )
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--performance-receipt", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = promote_testing_ready(
        candidate_manifest_path=args.candidate_manifest,
        performance_receipt_path=args.performance_receipt,
        candidate_dir=args.candidate_dir,
        output_dir=args.output,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
