from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import (
    TERMINAL_JOB_STATUSES,
    DatasetSplit,
    ExperimentStatus,
    JobStatus,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.evaluation.phase_a_metrics import summarize_phase_a_experiment
from ml_lab.evaluation.runners import evaluate_phase_a_sparse_experiment
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    PHASE_A_RUNTIME_PACK_ID,
    PHASE_A_TRAINER_ID,
    TrainingService,
)

EXPERIMENT_RECEIPT_NAME = "phase-a-first-sparse-experiment-receipt.json"


def run_phase_a_first_sparse_experiment(
    *,
    workspace_path: Path,
    feature_dim: int = 32768,
    alpha: float = 0.5,
    seed: int = 0,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    workspace = Workspace.open(workspace_path)
    freeze_receipt_path = workspace.root / "phase-a-freeze-receipt.json"
    if not freeze_receipt_path.is_file():
        raise FileNotFoundError(
            "Phase A freeze receipt is missing from the experiment workspace"
        )
    freeze_bytes = freeze_receipt_path.read_bytes()
    freeze_receipt = json.loads(freeze_bytes)
    if not isinstance(freeze_receipt, dict):
        raise ValueError("Phase A freeze receipt must be a JSON object")
    _validate_freeze_receipt(freeze_receipt)

    project_id = _required_text(freeze_receipt, "project_id")
    dataset_info = _mapping(freeze_receipt, "dataset")
    dataset_id = _required_text(dataset_info, "id")
    contract_info = _mapping(freeze_receipt, "contract_snapshot")
    contract_snapshot_id = _required_text(contract_info, "id")

    datasets = DatasetService(workspace)
    frozen_dataset = datasets.get(dataset_id)
    if frozen_dataset.project_id != project_id:
        raise ValueError("Freeze receipt project does not own the frozen dataset")
    if frozen_dataset.state.value != "FROZEN":
        raise ValueError("Workspace dataset is no longer FROZEN")
    if frozen_dataset.contract_snapshot_id != contract_snapshot_id:
        raise ValueError("Frozen dataset contract snapshot does not match freeze receipt")
    actual_handles = datasets.trainer_partition_handles(dataset_id)
    receipt_handles = {
        str(key): str(value)
        for key, value in _mapping(freeze_receipt, "trainer_handles").items()
    }
    if actual_handles != receipt_handles:
        raise ValueError("Frozen trainer handles do not match freeze receipt")

    snapshot = ContractSnapshotService(workspace).get(contract_snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Freeze receipt project does not own the contract snapshot")
    if snapshot.adapter_id != "frankenhomie.phase-a-residual":
        raise ValueError("Frozen contract snapshot is not the Phase A residual adapter")
    if snapshot.commit_sha != freeze_receipt.get("target_frankenhomie_commit"):
        raise ValueError("Frozen contract commit does not match freeze receipt")
    if snapshot.contract_version != freeze_receipt.get("target_contract"):
        raise ValueError("Frozen contract version does not match freeze receipt")

    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id=PHASE_A_TRAINER_ID,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={
            "feature_dim": feature_dim,
            "alpha": alpha,
            "experiment_kind": "phase-a-first-sparse-proof",
        },
        seed=seed,
        contract_snapshot_id=contract_snapshot_id,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
    )

    training = TrainingService(workspace)
    try:
        state = training.launch(experiment.id)
        deadline = time.monotonic() + timeout_seconds
        while state.job.status not in TERMINAL_JOB_STATUSES:
            if time.monotonic() >= deadline:
                training.cancel_job(experiment.id, state.job.id)
                raise TimeoutError(
                    f"Phase A training did not finish within {timeout_seconds:g} seconds"
                )
            time.sleep(0.05)
            state = training.refresh_job(experiment.id, state.job.id)

        if state.job.status is not JobStatus.COMPLETED:
            raise RuntimeError(
                f"Phase A training job ended as {state.job.status.value}: "
                f"{state.job.error or 'no worker error'}"
            )
        state = training.refresh_job(experiment.id, state.job.id)
        completed = state.experiment
    finally:
        training.shutdown()

    if completed.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError(
            f"Phase A experiment ended as {completed.status.value}"
        )
    if completed.model_artifact_digest is None:
        raise RuntimeError("Completed Phase A experiment has no model artifact")

    summaries = evaluate_phase_a_sparse_experiment(
        workspace,
        completed.id,
        splits=(DatasetSplit.DEV,),
    )
    if len(summaries) != 1 or summaries[0].split is not DatasetSplit.DEV:
        raise RuntimeError("Phase A first experiment did not produce exactly one DEV summary")
    dev_summary = summaries[0]
    metrics = summarize_phase_a_experiment(
        workspace,
        completed.id,
        DatasetSplit.DEV,
    )
    metric_rows = [
        {
            "metric_id": metric.metric_id,
            "value": metric.value,
            "direction": metric.direction.value,
            "veto": metric.veto,
            "unit": metric.unit,
            "note": metric.note,
        }
        for metric in metrics
    ]

    veto_failures = [
        metric
        for metric in metric_rows
        if metric["veto"] is True
        and isinstance(metric["value"], (int, float))
        and float(metric["value"]) != 0.0
    ]
    model_path = workspace.artifacts.resolve(completed.model_artifact_digest)
    model_bytes = model_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != completed.model_artifact_digest:
        raise RuntimeError("Model artifact digest does not match physical model bytes")

    failures = _failure_rows(workspace, completed.id, DatasetSplit.DEV)
    training_job = _training_job_summary(workspace, completed.id)

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-first-sparse-experiment-receipt/1",
        "ok": True,
        "stage": "EXPERIMENT",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "contract_snapshot_id": contract_snapshot_id,
        "target_frankenhomie_commit": freeze_receipt["target_frankenhomie_commit"],
        "target_contract": freeze_receipt["target_contract"],
        "freeze_receipt_file_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "freeze_receipt_payload_sha256": freeze_receipt["payload_sha256"],
        "trainer_id": PHASE_A_TRAINER_ID,
        "runtime_pack_id": PHASE_A_RUNTIME_PACK_ID,
        "training_config": {
            "feature_dim": feature_dim,
            "alpha": alpha,
            "seed": seed,
        },
        "training_partition_handles": {
            "TRAIN": _mapping(freeze_receipt, "trainer_handles")["TRAIN"],
        },
        "dev_was_used_for_training": False,
        "test_was_used_for_training": False,
        "redteam_was_used_for_training": False,
        "training_job": training_job,
        "experiment": {
            "id": completed.id,
            "status": completed.status.value,
            "model_artifact_digest": completed.model_artifact_digest,
            "metrics_artifact_digest": completed.metrics_artifact_digest,
            "manifest_artifact_digest": completed.manifest_artifact_digest,
        },
        "model": {
            "sha256": completed.model_artifact_digest,
            "size_bytes": len(model_bytes),
            "kind": "phase-a-bounded-sparse-v1",
        },
        "dev_evaluation": {
            "total": dev_summary.total,
            "correct": dev_summary.correct,
            "failures": dev_summary.failures,
            "veto_failures": dev_summary.veto_failures,
            "mean_latency_ms": dev_summary.mean_latency_ms,
            "p95_latency_ms": dev_summary.p95_latency_ms,
            "metrics": metric_rows,
            "failure_rows": failures,
        },
        "veto_failure_count": len(veto_failures),
        "veto_metrics_nonzero": veto_failures,
        "promotion_stage": "EXPERIMENT",
        "promotion_allowed": False,
        "training_only_proof": True,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    receipt = {**base_receipt, "payload_sha256": payload_sha}
    (workspace.root / EXPERIMENT_RECEIPT_NAME).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt


def _validate_freeze_receipt(receipt: dict[str, object]) -> None:
    if receipt.get("schema") != "ml-lab-phase-a-freeze-receipt/1":
        raise ValueError("Unsupported Phase A freeze receipt schema")
    claimed_payload = receipt.get("payload_sha256")
    if not isinstance(claimed_payload, str) or len(claimed_payload) != 64:
        raise ValueError("Phase A freeze receipt payload SHA-256 is missing")
    base_receipt = {
        key: value
        for key, value in receipt.items()
        if key != "payload_sha256"
    }
    observed_payload = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    if observed_payload != claimed_payload:
        raise ValueError("Phase A freeze receipt payload SHA-256 mismatch")
    if receipt.get("ok") is not True or receipt.get("frozen") is not True:
        raise ValueError("Phase A first experiment requires a clean frozen dataset")
    if receipt.get("ordinary_dataset_service") is not True:
        raise ValueError("Phase A first experiment requires ordinary DatasetService freeze")
    if receipt.get("ordinary_contract_snapshot_service") is not True:
        raise ValueError("Phase A first experiment requires ordinary contract snapshot")
    if receipt.get("model_training_started") is not False:
        raise ValueError("Freeze receipt already claims model training")
    if receipt.get("production_data") is not False:
        raise ValueError("Production data cannot enter Phase A experiment")
    if receipt.get("transcript_derived") is not False:
        raise ValueError("Transcript-derived data cannot enter Phase A experiment")
    if receipt.get("protected_splits_in_trainer_handles") is not False:
        raise ValueError("Protected split escaped into trainer handles")
    handles = _mapping(receipt, "trainer_handles")
    if set(handles) != {"TRAIN", "DEV"}:
        raise ValueError("Frozen trainer handles must contain exactly TRAIN and DEV")
    dataset = _mapping(receipt, "dataset")
    if dataset.get("state") != "FROZEN":
        raise ValueError("Dataset state must be FROZEN")
    partitions = _mapping(dataset, "partitions")
    if _partition_count(partitions, "TRAIN") <= 0:
        raise ValueError("Frozen dataset has no TRAIN examples")
    if _partition_count(partitions, "DEV") <= 0:
        raise ValueError("First experiment requires non-empty DEV")
    if _partition_count(partitions, "TEST") != 0:
        raise ValueError("First proof workspace must not contain TEST examples")
    if _partition_count(partitions, "REDTEAM") != 0:
        raise ValueError("First proof workspace must not contain REDTEAM examples")


def _training_job_summary(
    workspace: Workspace,
    experiment_id: str,
) -> dict[str, object]:
    training = TrainingService(workspace)
    try:
        job = training.job_for_experiment(experiment_id)
    finally:
        training.shutdown()
    return {
        "id": job.id,
        "status": job.status.value,
        "result_artifact_digest": job.result_artifact_digest,
        "event_artifact_digest": job.event_artifact_digest,
        "error": job.error,
    }


def _failure_rows(
    workspace: Workspace,
    experiment_id: str,
    split: DatasetSplit,
) -> list[dict[str, object]]:
    with workspace.database.connection() as conn:
        rows = conn.execute(
            "SELECT f.kind,f.severity,f.example_id,f.evidence_artifact_digest "
            "FROM failures f WHERE f.experiment_id=? AND f.split=? "
            "ORDER BY f.example_id",
            (experiment_id, split.value),
        ).fetchall()
    return [
        {
            "example_id": str(row["example_id"]),
            "kind": str(row["kind"]),
            "severity": str(row["severity"]),
            "evidence_artifact_digest": row["evidence_artifact_digest"],
        }
        for row in rows
    ]


def _mapping(value: MappingLike, key: str) -> dict[str, object]:
    raw = value.get(key)
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object")
    return {str(item_key): item_value for item_key, item_value in raw.items()}


def _required_text(value: MappingLike, key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _partition_count(partitions: dict[str, object], split: str) -> int:
    raw = partitions.get(split)
    if not isinstance(raw, dict):
        raise ValueError(f"Missing {split} partition")
    count = raw.get("example_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{split} partition example_count is invalid")
    return count


MappingLike = dict[str, object]
