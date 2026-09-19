from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

_REQUIRED_MEMBERS = frozenset(
    {
        "bundle_manifest.json",
        "compatibility.json",
        "contract.json",
        "datasets.json",
        "environment.json",
        "hashes.sha256",
        "known_failures.json",
        "metrics.json",
        "model/model.bin",
        "model_manifest.json",
        "reproduction.json",
        "break_it_guide.md",
    }
)


def verify_bundle_file(bundle_path: Path) -> dict[str, object]:
    """Verify a bundle without access to Lab workspace/trainer state."""
    try:
        bundle_sha256 = _file_sha256(bundle_path)
        with zipfile.ZipFile(bundle_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos if not info.is_dir()]
            if len(names) != len(set(names)):
                raise ValueError("Bundle contains duplicate member names.")
            for name in names:
                _validate_member_name(name)
            name_set = set(names)
            missing = sorted(_REQUIRED_MEMBERS - name_set)
            if missing:
                raise ValueError(f"Bundle is missing required members: {missing}")
            hashes = _parse_hash_manifest(archive.read("hashes.sha256"))
            expected_names = name_set - {"hashes.sha256"}
            if set(hashes) != expected_names:
                missing_hashes = sorted(expected_names - set(hashes))
                extra_hashes = sorted(set(hashes) - expected_names)
                raise ValueError(
                    "SHA manifest membership mismatch: "
                    f"missing={missing_hashes}, extra={extra_hashes}"
                )
            for name in sorted(expected_names):
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                if actual != hashes[name]:
                    raise ValueError(f"SHA-256 mismatch for {name}.")

            manifest = _json_object(archive.read("bundle_manifest.json"), "bundle manifest")
            compatibility = _json_object(
                archive.read("compatibility.json"),
                "compatibility manifest",
            )
            contract = _json_object(archive.read("contract.json"), "contract manifest")
            datasets = _json_object(archive.read("datasets.json"), "datasets manifest")
            _json_object(archive.read("environment.json"), "environment manifest")
            known_failures = _json_array(
                archive.read("known_failures.json"),
                "known failures manifest",
            )
            metrics = _json_object(archive.read("metrics.json"), "metrics manifest")
            model_manifest = _json_object(
                archive.read("model_manifest.json"),
                "model manifest",
            )
            reproduction = _json_object(
                archive.read("reproduction.json"),
                "reproduction spec",
            )
            _validate_manifest_consistency(
                manifest=manifest,
                compatibility=compatibility,
                contract=contract,
                datasets=datasets,
                known_failures=known_failures,
                metrics=metrics,
                model_manifest=model_manifest,
                reproduction=reproduction,
                hashes=hashes,
            )

        return {
            "status": "PASS",
            "bundle_sha256": bundle_sha256,
            "file_count": len(names),
            "model_id": manifest.get("model_id"),
            "experiment_id": manifest.get("experiment_id"),
            "dataset_id": manifest.get("dataset_id"),
            "integration_gate": manifest.get("integration_gate"),
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "bundle_sha256": _safe_file_sha256(bundle_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def write_verification_receipt(bundle_path: Path, receipt_path: Path) -> int:
    receipt = verify_bundle_file(bundle_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0 if receipt.get("status") == "PASS" else 8


def _validate_manifest_consistency(
    *,
    manifest: dict[str, object],
    compatibility: dict[str, object],
    contract: dict[str, object],
    datasets: dict[str, object],
    known_failures: list[object],
    metrics: dict[str, object],
    model_manifest: dict[str, object],
    reproduction: dict[str, object],
    hashes: dict[str, str],
) -> None:
    if manifest.get("integration_gate") != "NO_GO":
        raise ValueError("Bundle integration_gate must remain NO_GO.")

    model_digest = hashes["model/model.bin"]
    _require_equal(
        manifest.get("model_artifact_sha256"),
        model_digest,
        "Bundle manifest model artifact hash",
    )
    _require_equal(
        model_manifest.get("model_artifact_sha256"),
        model_digest,
        "Model manifest model artifact hash",
    )
    _require_equal(
        manifest.get("model_manifest_sha256"),
        hashes["model_manifest.json"],
        "Bundle manifest model manifest hash",
    )
    _require_equal(
        manifest.get("metrics_artifact_sha256"),
        hashes["metrics.json"],
        "Bundle manifest metrics hash",
    )

    experiment_id = manifest.get("experiment_id")
    dataset_id = manifest.get("dataset_id")
    _require_equal(
        model_manifest.get("experiment_id"),
        experiment_id,
        "Model manifest experiment id",
    )
    _require_equal(
        metrics.get("experiment_id"),
        experiment_id,
        "Metrics manifest experiment id",
    )
    _require_equal(
        reproduction.get("experiment_id"),
        experiment_id,
        "Reproduction spec experiment id",
    )
    _require_equal(
        datasets.get("dataset_id"),
        dataset_id,
        "Datasets manifest dataset id",
    )

    if manifest.get("compatibility") != compatibility:
        raise ValueError("compatibility.json does not match bundle manifest compatibility.")
    if model_manifest.get("compatibility") != compatibility:
        raise ValueError("model_manifest.json does not match compatibility.json.")

    dataset_manifest_digest = manifest.get("dataset_manifest_sha256")
    _require_equal(
        datasets.get("dataset_manifest_sha256"),
        dataset_manifest_digest,
        "Datasets manifest frozen dataset hash",
    )
    _require_equal(
        reproduction.get("dataset_manifest_sha256"),
        dataset_manifest_digest,
        "Reproduction spec frozen dataset hash",
    )

    experiment_manifest_digest = manifest.get("experiment_manifest_sha256")
    _require_equal(
        model_manifest.get("experiment_manifest_sha256"),
        experiment_manifest_digest,
        "Model manifest experiment manifest hash",
    )

    contract_snapshot_id = manifest.get("contract_snapshot_id")
    if contract_snapshot_id is None:
        if contract.get("contract_snapshot") is not None:
            raise ValueError("Contract manifest does not match an unpinned bundle.")
    else:
        _require_equal(
            contract.get("snapshot_id"),
            contract_snapshot_id,
            "Contract manifest snapshot id",
        )

    if not all(isinstance(item, dict) for item in known_failures):
        raise ValueError("known_failures.json entries must be JSON objects.")

    _validate_reproducibility(reproduction)

    input_partitions = reproduction.get("input_partitions")
    if not isinstance(input_partitions, dict):
        raise ValueError("Reproduction spec input_partitions must be an object.")
    input_partitions = {str(key): value for key, value in input_partitions.items()}
    forbidden = {"TEST", "REDTEAM"} & set(input_partitions)
    if forbidden:
        raise ValueError(
            "Reproduction spec illegally exposes protected partition(s): "
            f"{sorted(forbidden)}"
        )
    if "TRAIN" not in input_partitions:
        raise ValueError("Reproduction spec is missing the TRAIN partition.")
    if set(input_partitions) - {"TRAIN", "DEV"}:
        raise ValueError("Reproduction spec contains unsupported trainer partition names.")

    dataset_partitions = datasets.get("partitions")
    if not isinstance(dataset_partitions, dict):
        raise ValueError("Datasets manifest partitions must be an object.")
    for split, digest in input_partitions.items():
        if not isinstance(digest, str):
            raise ValueError(f"Reproduction {split} partition hash must be a string.")
        partition = dataset_partitions.get(split)
        if not isinstance(partition, dict):
            raise ValueError(f"Datasets manifest is missing the {split} partition.")
        _require_equal(
            partition.get("sha256"),
            digest,
            f"Reproduction {split} partition hash",
        )


def _validate_reproducibility(reproduction: dict[str, object]) -> None:
    raw = reproduction.get("reproducibility")
    if not isinstance(raw, dict):
        raise ValueError("Reproduction spec is missing reproducibility declaration.")
    mode = raw.get("mode")
    if mode not in {"DETERMINISTIC", "NONDETERMINISTIC"}:
        raise ValueError("Reproduction spec has an invalid reproducibility mode.")
    tolerances = raw.get("metric_tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("Reproduction metric_tolerances must be an object.")
    for metric, tolerance in tolerances.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("Reproduction tolerance metric id must be non-empty.")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or float(tolerance) < 0
        ):
            raise ValueError(
                f"Reproduction tolerance for {metric!r} must be a non-negative number."
            )
    if mode == "NONDETERMINISTIC" and not tolerances:
        raise ValueError(
            "Nondeterministic reproduction requires explicit metric tolerances."
        )


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match the bundle evidence.")


def _parse_hash_manifest(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8", errors="strict")
    hashes: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or not name:
            raise ValueError(f"Malformed hashes.sha256 line {line_number}.")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"Malformed SHA-256 digest on hashes.sha256 line {line_number}."
            ) from exc
        _validate_member_name(name)
        if name == "hashes.sha256":
            raise ValueError("hashes.sha256 must not hash itself.")
        if name in hashes:
            raise ValueError(f"Duplicate hash entry for {name}.")
        hashes[name] = digest.lower()
    return hashes


def _validate_member_name(name: str) -> None:
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ValueError(f"Unsafe bundle member path {name!r}.")
    if "\\" in name:
        raise ValueError(f"Bundle member must use POSIX separators: {name!r}.")


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return {str(key): item for key, item in value.items()}


def _json_array(payload: bytes, label: str) -> list[object]:
    value = json.loads(payload.decode("utf-8", errors="strict"))
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array.")
    return list(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file_sha256(path: Path) -> str | None:
    try:
        return _file_sha256(path)
    except OSError:
        return None
