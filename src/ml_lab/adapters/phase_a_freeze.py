from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace

FREEZE_RECEIPT_NAME = "phase-a-freeze-receipt.json"


def freeze_phase_a_train_dev(
    *,
    frankenhomie_repository: Path,
    factory_receipt_path: Path,
    dataset_path: Path,
    workspace_path: Path,
) -> dict[str, object]:
    factory_bytes = factory_receipt_path.read_bytes()
    dataset_bytes = dataset_path.read_bytes()
    factory = json.loads(factory_bytes)
    if not isinstance(factory, dict):
        raise ValueError("Phase A factory receipt must be a JSON object")
    _validate_factory_receipt(factory, dataset_bytes)

    workspace_root = workspace_path.expanduser().resolve()
    if workspace_root.exists() and any(workspace_root.iterdir()):
        raise ValueError(
            f"Freeze workspace must be new or empty: {workspace_root}"
        )
    workspace = Workspace.create(workspace_root)
    adapter = PhaseAResidualAdapter()
    project = workspace.create_project(
        "Phase A Semantic ML",
        adapter.adapter_id,
        "Pinned semantic-residual-v2 TRAIN/DEV experiment workspace.",
    )

    target_commit = _required_text(factory, "target_frankenhomie_commit")
    target_contract = _required_text(factory, "target_contract")
    snapshot_service = ContractSnapshotService(workspace)
    snapshot = snapshot_service.capture(
        project_id=project.id,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.version,
        repository=frankenhomie_repository,
        ref=target_commit,
        contract_version=target_contract,
        files=adapter.contract_files(),
    )
    contract_manifest = snapshot_service.manifest(snapshot.id)

    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "Phase A authoritative synthetic TRAIN DEV v1",
        contract_snapshot_id=snapshot.id,
    )
    imported = datasets.import_jsonl(
        dataset.id,
        dataset_path,
        validator=adapter.validate_dataset_row,
    )
    if imported.rejected:
        raise ValueError(
            f"Ordinary DatasetService rejected {imported.rejected} row(s)"
        )
    expected_count = _required_int(factory, "emitted_count")
    if imported.imported != expected_count:
        raise ValueError(
            f"Imported {imported.imported} rows; expected {expected_count}"
        )

    pre_freeze_report = datasets.scan_leakage(dataset.id)
    if pre_freeze_report.has_blockers:
        raise ValueError(
            "Ordinary DatasetService leakage scan found blockers before freeze"
        )

    frozen = datasets.freeze(dataset.id)
    if frozen.state is not DatasetState.FROZEN:
        raise RuntimeError("DatasetService did not produce a FROZEN dataset")
    if frozen.example_count != expected_count:
        raise RuntimeError("Frozen dataset example count changed")

    partitions = datasets.partitions(dataset.id)
    partition_rows = {
        item.split.value: {
            "sha256": item.partition_sha256,
            "artifact_digest": item.artifact_digest,
            "example_count": item.example_count,
        }
        for item in partitions
    }
    expected_splits = {
        DatasetSplit.TRAIN.value,
        DatasetSplit.DEV.value,
        DatasetSplit.TEST.value,
        DatasetSplit.REDTEAM.value,
    }
    if set(partition_rows) != expected_splits:
        raise RuntimeError("Frozen dataset does not contain all four partitions")

    expected_counts = _split_counts(factory)
    expected_counts.setdefault(DatasetSplit.TEST.value, 0)
    expected_counts.setdefault(DatasetSplit.REDTEAM.value, 0)
    actual_counts = {
        split: int(row["example_count"])
        for split, row in partition_rows.items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Frozen partition counts changed: {actual_counts} != {expected_counts}"
        )

    trainer_handles = datasets.trainer_partition_handles(dataset.id)
    if set(trainer_handles) != {
        DatasetSplit.TRAIN.value,
        DatasetSplit.DEV.value,
    }:
        raise RuntimeError("Trainer handles must expose only TRAIN and DEV")
    evaluation_handles = datasets.evaluation_partition_handles(dataset.id)
    if set(evaluation_handles) != expected_splits:
        raise RuntimeError("Evaluation handles must expose all frozen partitions")

    if frozen.manifest_artifact_digest is None:
        raise RuntimeError("Frozen dataset manifest digest is missing")
    if frozen.leakage_report_artifact_digest is None:
        raise RuntimeError("Frozen dataset leakage report digest is missing")

    dataset_manifest_path = workspace.artifacts.resolve(
        frozen.manifest_artifact_digest
    )
    leakage_report_path = workspace.artifacts.resolve(
        frozen.leakage_report_artifact_digest
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    leakage_report = json.loads(leakage_report_path.read_text(encoding="utf-8"))
    if not isinstance(dataset_manifest, dict) or not isinstance(leakage_report, dict):
        raise ValueError("Frozen dataset artifacts must be JSON objects")
    if int(leakage_report.get("blocking_count", -1)) != 0:
        raise RuntimeError("Frozen DatasetService leakage artifact contains blockers")

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-freeze-receipt/1",
        "ok": True,
        "project_id": project.id,
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.version,
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
        "factory_receipt_file_sha256": hashlib.sha256(factory_bytes).hexdigest(),
        "factory_receipt_payload_sha256": factory["payload_sha256"],
        "source_dataset_file_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "source_dataset_case_count": expected_count,
        "contract_snapshot": {
            "id": snapshot.id,
            "compatibility_signature": snapshot.compatibility_signature,
            "manifest_artifact_digest": snapshot.manifest_artifact_digest,
            "working_tree_dirty_at_capture": bool(
                contract_manifest.get("working_tree_dirty_at_capture")
            ),
            "commit_sha": snapshot.commit_sha,
            "contract_version": snapshot.contract_version,
        },
        "dataset": {
            "id": frozen.id,
            "state": frozen.state.value,
            "example_count": frozen.example_count,
            "manifest_artifact_digest": frozen.manifest_artifact_digest,
            "leakage_report_artifact_digest": frozen.leakage_report_artifact_digest,
            "partitions": partition_rows,
        },
        "ordinary_dataset_service": True,
        "ordinary_contract_snapshot_service": True,
        "trainer_handles": trainer_handles,
        "trainer_visible_splits": sorted(trainer_handles),
        "protected_splits_in_trainer_handles": False,
        "evaluation_handles": evaluation_handles,
        "frozen": True,
        "candidate_training_data": True,
        "model_training_started": False,
        "production_data": False,
        "transcript_derived": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    receipt = {**base_receipt, "payload_sha256": payload_sha}
    (workspace_root / FREEZE_RECEIPT_NAME).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt


def _validate_factory_receipt(
    factory: Mapping[str, object],
    dataset_bytes: bytes,
) -> None:
    if factory.get("schema") != "ml-lab-phase-a-dataset-factory-receipt/1":
        raise ValueError("Unsupported Phase A factory receipt schema")
    if factory.get("ok") is not True:
        raise ValueError("Phase A factory receipt is not clean")
    if factory.get("candidate_training_data") is not True:
        raise ValueError("Factory did not mark the dataset as candidate training data")
    if factory.get("frozen") is not False:
        raise ValueError("Factory output must still be unfrozen")
    if factory.get("production_data") is not False:
        raise ValueError("Production data cannot enter Phase A training")
    if factory.get("transcript_derived") is not False:
        raise ValueError("Transcript-derived data cannot enter Phase A training")
    if factory.get("protected_splits_in_output") is not False:
        raise ValueError("Protected splits cannot enter the TRAIN/DEV candidate")
    if factory.get("trainer_visible_splits") != ["TRAIN", "DEV"]:
        raise ValueError("Factory trainer-visible splits changed")
    errors = factory.get("errors")
    if errors != []:
        raise ValueError("Factory receipt contains errors")
    if factory.get("target_contract") != "semantic-residual-v2":
        raise ValueError("Freeze requires semantic-residual-v2")
    if _required_int(factory, "case_count") != _required_int(factory, "emitted_count"):
        raise ValueError("Factory must emit every authored case before freeze")
    for field in ("full_payload_leakage", "protected_language_leakage"):
        report = factory.get(field)
        if not isinstance(report, Mapping):
            raise ValueError(f"{field} must be an object")
        if report.get("blocking_count") != 0:
            raise ValueError(f"{field} contains blocking leakage")

    claimed_payload = factory.get("payload_sha256")
    if not isinstance(claimed_payload, str) or len(claimed_payload) != 64:
        raise ValueError("Factory payload SHA-256 is missing")
    base_factory = {
        key: value
        for key, value in factory.items()
        if key != "payload_sha256"
    }
    observed_payload = hashlib.sha256(
        canonical_json(base_factory).encode("utf-8")
    ).hexdigest()
    if observed_payload != claimed_payload:
        raise ValueError("Factory receipt payload SHA-256 mismatch")

    claimed_dataset = factory.get("dataset_sha256")
    observed_dataset = hashlib.sha256(dataset_bytes).hexdigest()
    if claimed_dataset != observed_dataset:
        raise ValueError(
            "Physical TRAIN/DEV JSONL SHA-256 does not match factory receipt"
        )
    if dataset_bytes.count(b"\r\n"):
        raise ValueError("Phase A candidate JSONL must use canonical LF bytes")
    line_count = len(
        [line for line in dataset_bytes.split(b"\n") if line.strip()]
    )
    if line_count != _required_int(factory, "emitted_count"):
        raise ValueError("Physical TRAIN/DEV JSONL line count mismatch")


def _split_counts(factory: Mapping[str, object]) -> dict[str, int]:
    raw = factory.get("split_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("Factory split_counts must be an object")
    result: dict[str, int] = {}
    for split, value in raw.items():
        if split not in {"TRAIN", "DEV"}:
            raise ValueError("Factory output may contain only TRAIN and DEV")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Factory split counts must be non-negative integers")
        result[str(split)] = value
    if set(result) != {"TRAIN", "DEV"}:
        raise ValueError("Factory split_counts must contain TRAIN and DEV")
    return result


def _required_text(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _required_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw
