from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_freeze import _validate_factory_receipt
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, DatasetState
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace

EXPANDED_FREEZE_RECEIPT = "phase-a-expanded-freeze-receipt.json"


def freeze_expanded_phase_a(
    *,
    frankenhomie_repository: Path,
    expanded_root: Path,
    protected_root: Path,
    workspace_path: Path,
) -> dict[str, object]:
    expanded_root = expanded_root.expanduser().resolve()
    protected_root = protected_root.expanduser().resolve()

    train_dev_receipt_path = expanded_root / "authoritative" / "factory-receipt.json"
    train_dev_dataset_path = expanded_root / "authoritative" / "phase-a-train-dev-v1.jsonl"
    protected_receipt_path = protected_root / "protected-authority-receipt.json"
    protected_dataset_path = protected_root / "phase-a-protected-residual-v1.jsonl"
    zero_evidence_path = protected_root / "phase-a-zero-model-evidence-v1.json"

    train_dev_receipt = _read_object(train_dev_receipt_path)
    train_dev_bytes = train_dev_dataset_path.read_bytes()
    _validate_factory_receipt(train_dev_receipt, train_dev_bytes)

    protected_receipt_bytes = protected_receipt_path.read_bytes()
    protected_receipt = json.loads(protected_receipt_bytes)
    if not isinstance(protected_receipt, dict):
        raise ValueError("Protected authority receipt must be an object")
    protected_bytes = protected_dataset_path.read_bytes()
    zero_bytes = zero_evidence_path.read_bytes()
    _validate_protected_receipt(
        protected_receipt,
        protected_bytes=protected_bytes,
        zero_bytes=zero_bytes,
    )

    target_commit = _required_text(train_dev_receipt, "target_frankenhomie_commit")
    target_contract = _required_text(train_dev_receipt, "target_contract")
    if protected_receipt.get("target_frankenhomie_commit") != target_commit:
        raise ValueError("Protected evidence targets a different Frankenhomie commit")
    if protected_receipt.get("target_contract") != target_contract:
        raise ValueError("Protected evidence targets a different contract")

    workspace_root = workspace_path.expanduser().resolve()
    if workspace_root.exists() and any(workspace_root.iterdir()):
        raise ValueError(f"Expanded freeze workspace must be new or empty: {workspace_root}")
    workspace = Workspace.create(workspace_root)
    adapter = PhaseAResidualAdapter()
    project = workspace.create_project(
        "Phase A Semantic ML Expanded",
        adapter.adapter_id,
        "Frozen expanded semantic-residual-v2 corpus with sealed protected evaluation.",
    )

    snapshots = ContractSnapshotService(workspace)
    snapshot = snapshots.capture(
        project_id=project.id,
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.version,
        repository=frankenhomie_repository,
        ref=target_commit,
        contract_version=target_contract,
        files=adapter.contract_files(),
    )
    contract_manifest = snapshots.manifest(snapshot.id)

    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "Phase A expanded authoritative residual corpus v1",
        contract_snapshot_id=snapshot.id,
    )
    imported_train_dev = datasets.import_jsonl(
        dataset.id,
        train_dev_dataset_path,
        validator=adapter.validate_dataset_row,
    )
    if imported_train_dev.rejected:
        raise ValueError(
            f"DatasetService rejected {imported_train_dev.rejected} TRAIN/DEV row(s)"
        )
    imported_protected = datasets.import_jsonl(
        dataset.id,
        protected_dataset_path,
        validator=adapter.validate_dataset_row,
    )
    if imported_protected.rejected:
        raise ValueError(
            f"DatasetService rejected {imported_protected.rejected} protected row(s)"
        )

    expected_train_dev = _required_int(train_dev_receipt, "emitted_count")
    expected_protected = _required_int(protected_receipt, "residual_emitted_count")
    if imported_train_dev.imported != expected_train_dev:
        raise ValueError("Expanded TRAIN/DEV import count changed")
    if imported_protected.imported != expected_protected:
        raise ValueError("Expanded protected import count changed")

    leakage = datasets.scan_leakage(dataset.id)
    if leakage.has_blockers:
        raise ValueError("Ordinary DatasetService found expanded cross-split leakage blockers")

    frozen = datasets.freeze(dataset.id)
    if frozen.state is not DatasetState.FROZEN:
        raise RuntimeError("Expanded Phase A dataset did not freeze")
    if frozen.example_count != expected_train_dev + expected_protected:
        raise RuntimeError("Expanded frozen example count changed")

    partitions = datasets.partitions(dataset.id)
    partition_rows = {
        item.split.value: {
            "sha256": item.partition_sha256,
            "artifact_digest": item.artifact_digest,
            "example_count": item.example_count,
        }
        for item in partitions
    }
    expected_counts = {
        DatasetSplit.TRAIN.value: 240,
        DatasetSplit.DEV.value: 60,
        DatasetSplit.TEST.value: 60,
        DatasetSplit.REDTEAM.value: 36,
    }
    actual_counts = {
        item.split.value: item.example_count
        for item in partitions
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Expanded frozen partition counts changed: {actual_counts} != {expected_counts}"
        )

    trainer_handles = datasets.trainer_partition_handles(dataset.id)
    if set(trainer_handles) != {"TRAIN", "DEV"}:
        raise RuntimeError("Expanded trainer handles exposed protected splits")
    evaluation_handles = datasets.evaluation_partition_handles(dataset.id)
    if set(evaluation_handles) != {"TRAIN", "DEV", "TEST", "REDTEAM"}:
        raise RuntimeError("Expanded evaluation handles are incomplete")

    if frozen.manifest_artifact_digest is None:
        raise RuntimeError("Expanded dataset manifest digest missing")
    if frozen.leakage_report_artifact_digest is None:
        raise RuntimeError("Expanded leakage report digest missing")

    zero_evidence = json.loads(zero_bytes)
    if not isinstance(zero_evidence, dict):
        raise ValueError("Zero-model evidence must be an object")
    if zero_evidence.get("case_count") != 24 or zero_evidence.get("passed_count") != 24:
        raise ValueError("All 24 zero-model invariants must pass before freeze")

    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-freeze-receipt/1",
        "ok": True,
        "project_id": project.id,
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.version,
        "target_frankenhomie_commit": target_commit,
        "target_contract": target_contract,
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
        "source_artifacts": {
            "train_dev_sha256": hashlib.sha256(train_dev_bytes).hexdigest(),
            "protected_residual_sha256": hashlib.sha256(protected_bytes).hexdigest(),
            "protected_receipt_sha256": hashlib.sha256(
                protected_receipt_bytes
            ).hexdigest(),
            "zero_model_evidence_sha256": hashlib.sha256(zero_bytes).hexdigest(),
        },
        "zero_model_redteam": {
            "case_count": 24,
            "passed_count": 24,
            "artifact_sha256": hashlib.sha256(zero_bytes).hexdigest(),
            "part_of_residual_dataset": False,
        },
        "ordinary_dataset_service": True,
        "ordinary_contract_snapshot_service": True,
        "trainer_handles": trainer_handles,
        "trainer_visible_splits": sorted(trainer_handles),
        "protected_splits_in_trainer_handles": False,
        "evaluation_handles": evaluation_handles,
        "frozen": True,
        "model_training_started": False,
        "production_data": False,
        "transcript_derived": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    (workspace_root / EXPANDED_FREEZE_RECEIPT).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt


def _validate_protected_receipt(
    receipt: Mapping[str, object],
    *,
    protected_bytes: bytes,
    zero_bytes: bytes,
) -> None:
    if receipt.get("schema") != "ml-lab-phase-a-protected-authority-receipt/1":
        raise ValueError("Unsupported protected authority receipt")
    if receipt.get("ok") is not True:
        raise ValueError("Protected authority evidence did not pass")
    if receipt.get("production_data") is not False:
        raise ValueError("Protected evidence cannot use production data")
    if receipt.get("transcript_derived") is not False:
        raise ValueError("Protected evidence cannot be transcript-derived")
    if receipt.get("training_allowed") is not False:
        raise ValueError("Protected evidence became training-allowed")
    if receipt.get("errors") != []:
        raise ValueError("Protected authority receipt contains errors")
    if _required_int(receipt, "residual_case_count") != 96:
        raise ValueError("Protected residual authored count changed")
    if _required_int(receipt, "residual_emitted_count") != 96:
        raise ValueError("All protected residual cases must be emitted")
    counts = receipt.get("residual_split_counts")
    if counts != {"TEST": 60, "REDTEAM": 36}:
        raise ValueError("Protected residual split counts changed")
    if receipt.get("zero_model_case_count") != 24:
        raise ValueError("Zero-model case count changed")
    if receipt.get("zero_model_passed_count") != 24:
        raise ValueError("All zero-model invariants must pass")
    if receipt.get("residual_dataset_sha256") != hashlib.sha256(protected_bytes).hexdigest():
        raise ValueError("Protected residual physical hash mismatch")
    if receipt.get("zero_model_evidence_sha256") != hashlib.sha256(zero_bytes).hexdigest():
        raise ValueError("Zero-model physical hash mismatch")

    claimed_payload = receipt.get("payload_sha256")
    if not isinstance(claimed_payload, str):
        raise ValueError("Protected receipt payload hash missing")
    base = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    observed = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    if observed != claimed_payload:
        raise ValueError("Protected receipt payload hash mismatch")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return {str(key): item for key, item in value.items()}


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
