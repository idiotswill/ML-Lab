from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import ModelStage, utc_now_iso
from ml_lab.core.process import application_command
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class BundleRecord:
    id: str
    model_id: str
    bundle_artifact_digest: str
    manifest_artifact_digest: str
    created_at: str


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    id: str
    bundle_id: str
    status: str
    receipt_artifact_digest: str
    created_at: str


class BundleService:
    """Build deterministic release-candidate bundles and verify them fresh."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts
        self.models = ModelRegistryService(workspace)
        self.experiments = ExperimentService(workspace)
        self.datasets = DatasetService(workspace)
        self.failures = FailureService(workspace)
        self.contracts = ContractSnapshotService(workspace)

    def build_release_candidate(
        self,
        model_id: str,
        *,
        break_it_guide: str | None = None,
    ) -> BundleRecord:
        model = self.models.get(model_id)
        if model.stage is not ModelStage.RELEASE_CANDIDATE:
            raise RuntimeError("Only RELEASE_CANDIDATE models can be bundled.")
        experiment = self.experiments.get(model.experiment_id)
        dataset = self.datasets.get(experiment.dataset_id)
        model_path = self.artifacts.resolve(model.model_artifact_digest)
        if experiment.metrics_artifact_digest is None:
            raise RuntimeError("Release candidate experiment is missing metrics.")
        if experiment.manifest_artifact_digest is None:
            raise RuntimeError("Release candidate experiment is missing its manifest.")
        if dataset.manifest_artifact_digest is None:
            raise RuntimeError("Release candidate dataset is missing its frozen manifest.")
        if model.manifest_artifact_digest is None:
            raise RuntimeError("Registered model is missing its manifest.")

        partitions = self.datasets.partitions(dataset.id)
        datasets_payload = {
            "format_version": 1,
            "dataset_id": dataset.id,
            "dataset_manifest_sha256": dataset.manifest_artifact_digest,
            "leakage_report_sha256": dataset.leakage_report_artifact_digest,
            "partitions": {
                partition.split.value: {
                    "sha256": partition.partition_sha256,
                    "example_count": partition.example_count,
                }
                for partition in partitions
            },
        }
        reproduction = self.experiments.trainer_job_spec(experiment.id)
        input_partitions = reproduction.get("input_partitions")
        if not isinstance(input_partitions, dict):
            raise RuntimeError("Trainer reproduction spec has invalid input partitions.")
        if {"TEST", "REDTEAM"} & set(input_partitions):
            raise RuntimeError("Protected partitions escaped into reproduction spec.")

        failures = self._experiment_failures(experiment.id)
        contract_payload = self._contract_payload(experiment.contract_snapshot_id)
        compatibility = json.loads(model.compatibility_json)
        bundle_manifest = {
            "format_version": 1,
            "model_id": model.id,
            "model_stage": model.stage.value,
            "experiment_id": experiment.id,
            "dataset_id": dataset.id,
            "contract_snapshot_id": experiment.contract_snapshot_id,
            "model_artifact_sha256": model.model_artifact_digest,
            "model_manifest_sha256": model.manifest_artifact_digest,
            "experiment_manifest_sha256": experiment.manifest_artifact_digest,
            "dataset_manifest_sha256": dataset.manifest_artifact_digest,
            "metrics_artifact_sha256": experiment.metrics_artifact_digest,
            "integration_gate": "NO_GO",
            "compatibility": compatibility,
        }
        guide = break_it_guide or _default_break_it_guide()
        text_members: dict[str, bytes] = {
            "bundle_manifest.json": _json_bytes(bundle_manifest),
            "compatibility.json": _json_bytes(compatibility),
            "contract.json": _json_bytes(contract_payload),
            "datasets.json": _json_bytes(datasets_payload),
            "environment.json": _json_bytes(json.loads(experiment.environment_json)),
            "known_failures.json": _json_bytes(failures),
            "metrics.json": self.artifacts.resolve(
                experiment.metrics_artifact_digest
            ).read_bytes(),
            "model_manifest.json": self.artifacts.resolve(
                model.manifest_artifact_digest
            ).read_bytes(),
            "reproduction.json": _json_bytes(reproduction),
            "break_it_guide.md": (guide.rstrip() + "\n").encode("utf-8"),
        }
        hashes = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in text_members.items()
        }
        hashes["model/model.bin"] = model.model_artifact_digest
        hash_manifest = "".join(
            f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
        ).encode("utf-8")
        text_members["hashes.sha256"] = hash_manifest

        with TemporaryDirectory(dir=self.workspace.root / "cache") as temp_dir:
            bundle_path = Path(temp_dir) / "release-candidate.zip"
            self._write_zip(bundle_path, text_members, model_path)
            bundle_ref = self.artifacts.commit_file(
                bundle_path,
                media_type="application/vnd.ml-lab.release-candidate+zip",
                metadata={"model_id": model.id, "experiment_id": experiment.id},
            )
        manifest_ref = self.artifacts.commit_bytes(
            text_members["bundle_manifest.json"],
            media_type="application/vnd.ml-lab.bundle-manifest+json",
            metadata={"model_id": model.id, "experiment_id": experiment.id},
        )
        record = BundleRecord(
            id=str(uuid.uuid4()),
            model_id=model.id,
            bundle_artifact_digest=bundle_ref.digest,
            manifest_artifact_digest=manifest_ref.digest,
            created_at=utc_now_iso(),
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO bundles"
                "(id,model_id,bundle_artifact_digest,manifest_artifact_digest,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    record.id,
                    record.model_id,
                    record.bundle_artifact_digest,
                    record.manifest_artifact_digest,
                    record.created_at,
                ),
            )
        return record

    def get(self, bundle_id: str) -> BundleRecord:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM bundles WHERE id=?", (bundle_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown bundle {bundle_id}")
        return _bundle_from_row(row)

    def verify_fresh(self, bundle_id: str) -> VerificationReceipt:
        bundle = self.get(bundle_id)
        bundle_path = self.artifacts.resolve(bundle.bundle_artifact_digest)
        with TemporaryDirectory(dir=self.workspace.root / "cache") as temp_dir:
            receipt_path = Path(temp_dir) / "verification-receipt.json"
            completed = subprocess.run(
                application_command(
                    "--verify-bundle-worker",
                    str(bundle_path),
                    "--verification-receipt",
                    str(receipt_path),
                ),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if receipt_path.exists():
                receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            else:
                receipt_payload = {
                    "status": "FAIL",
                    "error_type": "VerifierProcessError",
                    "error": "Fresh verifier did not produce a receipt.",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-2000:],
                }
        if not isinstance(receipt_payload, dict):
            receipt_payload = {
                "status": "FAIL",
                "error_type": "ReceiptFormatError",
                "error": "Fresh verifier receipt was not a JSON object.",
            }
        status = "PASS" if receipt_payload.get("status") == "PASS" else "FAIL"
        if completed.returncode != 0:
            status = "FAIL"
            receipt_payload["verifier_returncode"] = completed.returncode
        receipt_payload["bundle_id"] = bundle.id
        receipt_payload["fresh_process"] = True
        receipt_payload["verified_at"] = utc_now_iso()
        receipt_ref = self.artifacts.commit_bytes(
            _json_bytes(receipt_payload),
            media_type="application/vnd.ml-lab.verification-receipt+json",
            metadata={"bundle_id": bundle.id, "status": status},
        )
        receipt = VerificationReceipt(
            id=str(uuid.uuid4()),
            bundle_id=bundle.id,
            status=status,
            receipt_artifact_digest=receipt_ref.digest,
            created_at=str(receipt_payload["verified_at"]),
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO verification_receipts"
                "(id,bundle_id,status,receipt_artifact_digest,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    receipt.id,
                    receipt.bundle_id,
                    receipt.status,
                    receipt.receipt_artifact_digest,
                    receipt.created_at,
                ),
            )
        return receipt

    def verification_receipts(self, bundle_id: str) -> list[VerificationReceipt]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM verification_receipts "
                "WHERE bundle_id=? ORDER BY created_at,id",
                (bundle_id,),
            ).fetchall()
        return [_receipt_from_row(row) for row in rows]

    def _experiment_failures(self, experiment_id: str) -> list[dict[str, object]]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM failures WHERE experiment_id=? ORDER BY created_at,id",
                (experiment_id,),
            ).fetchall()
        return [self.failures.immutable_payload(str(row["id"])) for row in rows]

    def _contract_payload(self, snapshot_id: str | None) -> dict[str, object]:
        if snapshot_id is None:
            return {"contract_snapshot": None}
        snapshot = self.contracts.get(snapshot_id)
        source = self.contracts.manifest(snapshot_id)
        source.pop("repository_path", None)
        return {
            "snapshot_id": snapshot.id,
            "commit_sha": snapshot.commit_sha,
            "contract_version": snapshot.contract_version,
            "compatibility_signature": snapshot.compatibility_signature,
            "manifest_sha256": snapshot.manifest_artifact_digest,
            "manifest": source,
        }

    @staticmethod
    def _write_zip(
        bundle_path: Path,
        text_members: dict[str, bytes],
        model_path: Path,
    ) -> None:
        with zipfile.ZipFile(
            bundle_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            strict_timestamps=True,
        ) as archive:
            names = sorted([*text_members, "model/model.bin"])
            for name in names:
                if name == "model/model.bin":
                    _write_file_member(archive, name, model_path)
                else:
                    _write_bytes_member(archive, name, text_members[name])


def _write_bytes_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = _zip_info(name)
    with archive.open(info, "w", force_zip64=True) as output:
        output.write(payload)


def _write_file_member(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    info = _zip_info(name)
    with source.open("rb") as input_file, archive.open(
        info,
        "w",
        force_zip64=True,
    ) as output:
        shutil.copyfileobj(input_file, output, length=1024 * 1024)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _bundle_from_row(row: sqlite3.Row) -> BundleRecord:
    return BundleRecord(
        id=row["id"],
        model_id=row["model_id"],
        bundle_artifact_digest=row["bundle_artifact_digest"],
        manifest_artifact_digest=row["manifest_artifact_digest"],
        created_at=row["created_at"],
    )


def _receipt_from_row(row: sqlite3.Row) -> VerificationReceipt:
    return VerificationReceipt(
        id=row["id"],
        bundle_id=row["bundle_id"],
        status=row["status"],
        receipt_artifact_digest=row["receipt_artifact_digest"],
        created_at=row["created_at"],
    )


def _default_break_it_guide() -> str:
    return """# Break-it guide

This bundle is a release-candidate experiment artifact, not runtime authority.

Attack the model and adapter with ambiguous, adversarial, malformed, and boundary inputs.
Prioritize veto failures, hidden/out-of-envelope selections, contract violations, false
commitments, unsupported mechanics authority, and protected-split leakage. Preserve every
failure with its exact bundle/model/dataset/contract hashes and reproduction seed.

Do not integrate automatically. A passing bundle remains integration NO-GO until an
independent process outside ordinary training supplies explicit approval evidence.
"""
