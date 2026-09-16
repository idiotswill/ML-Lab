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
            reproduction = _json_object(
                archive.read("reproduction.json"),
                "reproduction spec",
            )
            input_partitions = reproduction.get("input_partitions")
            if not isinstance(input_partitions, dict):
                raise ValueError("Reproduction spec input_partitions must be an object.")
            forbidden = {"TEST", "REDTEAM"} & {str(key) for key in input_partitions}
            if forbidden:
                raise ValueError(
                    "Reproduction spec illegally exposes protected partition(s): "
                    f"{sorted(forbidden)}"
                )
            if "TRAIN" not in input_partitions:
                raise ValueError("Reproduction spec is missing the TRAIN partition.")

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
