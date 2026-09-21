from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from ml_lab.adapters.phase_a_first_experiment import (
    _failure_rows,
    _mapping,
    _required_text,
    _training_job_summary,
    _validate_freeze_receipt,
)
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
from ml_lab.evaluation.runners import evaluate_phase_a_candidate_sparse_experiment
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    PHASE_A_CANDIDATE_TRAINER_ID,
    PHASE_A_RUNTIME_PACK_ID,
    TrainingService,
)

CANDIDATE_EXPERIMENT_RECEIPT_NAME = (
    "phase-a-candidate-sparse-experiment-receipt.json"
)


def run_phase_a_candidate_sparse_experiment(
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
        trainer_id=PHASE_A_CANDIDATE_TRAINER_ID,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={
            "feature_dim": feature_dim,
            "alpha": alpha,
            "experiment_kind": "phase-a-candidate-relative-sparse-proof",
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
                    "Phase A candidate-relative training did not finish within "
                    f"{timeout_seconds:g} seconds"
                )
            time.sleep(0.05)
            state = training.refresh_job(experiment.id, state.job.id)

        if state.job.status is not JobStatus.COMPLETED:
            raise RuntimeError(
                "Phase A candidate-relative training job ended as "
                f"{state.job.status.value}: "
                f"{state.job.error or 'no worker error'}"
            )
        state = training.refresh_job(experiment.id, state.job.id)
        completed = state.experiment
    finally:
        training.shutdown()

    if completed.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError(
            "Phase A candidate-relative experiment ended as "
            f"{completed.status.value}"
        )
    if completed.model_artifact_digest is None:
        raise RuntimeError(
            "Completed Phase A candidate-relative experiment has no model artifact"
        )

    summaries = evaluate_phase_a_candidate_sparse_experiment(
        workspace,
        completed.id,
        splits=(DatasetSplit.DEV,),
    )
    if len(summaries) != 1 or summaries[0].split is not DatasetSplit.DEV:
        raise RuntimeError(
            "Phase A candidate-relative experiment did not produce exactly one "
            "DEV summary"
        )
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
        raise RuntimeError(
            "Candidate-relative model artifact digest does not match physical bytes"
        )

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-candidate-sparse-experiment-receipt/1",
        "ok": True,
        "stage": "EXPERIMENT",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "contract_snapshot_id": contract_snapshot_id,
        "target_frankenhomie_commit": freeze_receipt["target_frankenhomie_commit"],
        "target_contract": freeze_receipt["target_contract"],
        "freeze_receipt_file_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "freeze_receipt_payload_sha256": freeze_receipt["payload_sha256"],
        "trainer_id": PHASE_A_CANDIDATE_TRAINER_ID,
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
        "candidate_identity_learned_as_class": False,
        "candidate_binding_mode": "CURRENT_VISIBLE_ENVELOPE_EXPLICIT_MENTION_ONLY",
        "training_job": _training_job_summary(workspace, completed.id),
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
            "kind": "phase-a-candidate-relative-sparse-v1",
        },
        "dev_evaluation": {
            "total": dev_summary.total,
            "correct": dev_summary.correct,
            "failures": dev_summary.failures,
            "veto_failures": dev_summary.veto_failures,
            "mean_latency_ms": dev_summary.mean_latency_ms,
            "p95_latency_ms": dev_summary.p95_latency_ms,
            "metrics": metric_rows,
            "failure_rows": _failure_rows(
                workspace,
                completed.id,
                DatasetSplit.DEV,
            ),
        },
        "veto_failure_count": len(veto_failures),
        "veto_metrics_nonzero": veto_failures,
        "promotion_stage": "EXPERIMENT",
        "promotion_allowed": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    receipt = {**base_receipt, "payload_sha256": payload_sha}
    (workspace.root / CANDIDATE_EXPERIMENT_RECEIPT_NAME).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt
