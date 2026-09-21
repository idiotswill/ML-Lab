from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_CONTRACT_VERSION,
)
from ml_lab.adapters.phase_a_baselines import (
    V2BoundedLexical,
    V2DeterministicAbstention,
)
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import (
    TERMINAL_JOB_STATUSES,
    DatasetSplit,
    DatasetState,
    ExperimentStatus,
    JobStatus,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.evaluation.phase_a import make_phase_a_case_evaluator
from ml_lab.evaluation.phase_a_metrics import summarize_phase_a_experiment
from ml_lab.evaluation.runners import evaluate_packaged_experiment
from ml_lab.evaluation.service import EvaluationService, EvaluationSummary
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    PHASE_A_CANDIDATE_TRAINER_ID,
    PHASE_A_RUNTIME_PACK_ID,
    PHASE_A_TRAINER_ID,
    TrainingService,
)

_APP_ROOT = "frankenhomie-asterra-v0.9.0"
BAKEOFF_RECEIPT_NAME = "phase-a-expanded-dev-bakeoff-receipt.json"
DETERMINISTIC_BASELINE_ID = "baseline.phase_a_deterministic_abstention.v2"
LEXICAL_BASELINE_ID = "baseline.phase_a_bounded_lexical.v2"


def run_expanded_phase_a_dev_bakeoff(
    *,
    workspace_path: Path,
    feature_dim: int = 32768,
    alpha: float = 0.5,
    seed: int = 0,
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    workspace = Workspace.open(workspace_path)
    freeze_path = workspace.root / "phase-a-expanded-freeze-receipt.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    if not isinstance(freeze, dict):
        raise ValueError("Expanded Phase A freeze receipt must be an object")
    _validate_expanded_freeze_receipt(freeze)

    project_id = _required_text(freeze, "project_id")
    dataset_info = _mapping(freeze, "dataset")
    dataset_id = _required_text(dataset_info, "id")
    snapshot_info = _mapping(freeze, "contract_snapshot")
    snapshot_id = _required_text(snapshot_info, "id")

    datasets = DatasetService(workspace)
    dataset = datasets.get(dataset_id)
    if dataset.project_id != project_id:
        raise ValueError("Expanded dataset belongs to a different project")
    if dataset.state is not DatasetState.FROZEN:
        raise ValueError("Expanded dataset is no longer FROZEN")
    if dataset.contract_snapshot_id != snapshot_id:
        raise ValueError("Expanded dataset contract snapshot changed")
    if datasets.trainer_partition_handles(dataset_id) != {
        str(key): str(value)
        for key, value in _mapping(freeze, "trainer_handles").items()
    }:
        raise ValueError("Expanded trainer handles changed after freeze")

    snapshot = ContractSnapshotService(workspace).get(snapshot_id)
    if snapshot.project_id != project_id:
        raise ValueError("Expanded contract snapshot belongs to another project")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("Expanded snapshot is not the Phase A residual adapter")
    if snapshot.contract_version != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Expanded snapshot contract version changed")
    if snapshot.commit_sha != freeze["target_frankenhomie_commit"]:
        raise ValueError("Expanded snapshot commit changed")

    repository = Path(snapshot.repo_path)
    reference = PhaseAReferenceValidator(workspace)
    materialized = reference.materialize(repository, snapshot.commit_sha)
    registry_path = (
        materialized.source_root
        / _APP_ROOT
        / "data"
        / "semantic_family_registry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("Pinned Phase A registry must be an object")

    reports: list[dict[str, object]] = []
    reports.append(
        _run_baseline(
            workspace=workspace,
            project_id=project_id,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            trainer_id=DETERMINISTIC_BASELINE_ID,
            display_name="Deterministic abstention v2",
            predictor=V2DeterministicAbstention().predict,
            reference=reference,
            repository=repository,
            commit_sha=snapshot.commit_sha,
            seed=seed,
        )
    )
    reports.append(
        _run_baseline(
            workspace=workspace,
            project_id=project_id,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            trainer_id=LEXICAL_BASELINE_ID,
            display_name="Bounded lexical v2",
            predictor=V2BoundedLexical(registry).predict,
            reference=reference,
            repository=repository,
            commit_sha=snapshot.commit_sha,
            seed=seed,
        )
    )
    reports.append(
        _run_trained(
            workspace=workspace,
            project_id=project_id,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            trainer_id=PHASE_A_TRAINER_ID,
            display_name="Bounded sparse v1",
            feature_dim=feature_dim,
            alpha=alpha,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
    )
    reports.append(
        _run_trained(
            workspace=workspace,
            project_id=project_id,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
            trainer_id=PHASE_A_CANDIDATE_TRAINER_ID,
            display_name="Candidate-relative sparse v2",
            feature_dim=feature_dim,
            alpha=alpha,
            seed=seed,
            timeout_seconds=timeout_seconds,
        )
    )

    evaluation = EvaluationService(workspace)
    for report in reports:
        experiment_id = _required_text(report, "experiment_id")
        if evaluation.case_count(experiment_id, DatasetSplit.TEST) != 0:
            raise RuntimeError("TEST evidence was created during DEV bake-off")
        if evaluation.case_count(experiment_id, DatasetSplit.REDTEAM) != 0:
            raise RuntimeError("REDTEAM evidence was created during DEV bake-off")

    candidate = next(
        report
        for report in reports
        if report["trainer_id"] == PHASE_A_CANDIDATE_TRAINER_ID
    )
    candidate_dev_gate = _dev_gate_passed(candidate)

    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-dev-bakeoff/1",
        "ok": True,
        "stage": "EXPERIMENT",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "contract_snapshot_id": snapshot_id,
        "target_frankenhomie_commit": snapshot.commit_sha,
        "target_contract": snapshot.contract_version,
        "freeze_receipt_file_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "freeze_receipt_payload_sha256": freeze["payload_sha256"],
        "dataset_example_count": dataset.example_count,
        "partition_counts": {
            "TRAIN": 240,
            "DEV": 60,
            "TEST": 60,
            "REDTEAM": 36,
        },
        "training_partition": "TRAIN",
        "dev_used_for_training": False,
        "test_used_for_training": False,
        "redteam_used_for_training": False,
        "dev_evaluated": True,
        "test_evaluated": False,
        "redteam_evaluated": False,
        "zero_model_redteam_evaluated_by_model": False,
        "reports": reports,
        "candidate_relative_dev_gate_passed": candidate_dev_gate,
        "protected_evaluation_allowed": candidate_dev_gate,
        "promotion_allowed": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    (workspace.root / BAKEOFF_RECEIPT_NAME).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt


def _run_baseline(
    *,
    workspace: Workspace,
    project_id: str,
    dataset_id: str,
    snapshot_id: str,
    trainer_id: str,
    display_name: str,
    predictor: Callable[[Mapping[str, object]], Mapping[str, object]],
    reference: PhaseAReferenceValidator,
    repository: Path,
    commit_sha: str,
    seed: int,
) -> dict[str, object]:
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id=trainer_id,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={"experiment_kind": "phase-a-expanded-dev-baseline"},
        seed=seed,
        contract_snapshot_id=snapshot_id,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "training_performed": False,
        },
    )
    experiments.start(experiment.id)
    completed = experiments.complete(experiment.id, metrics=[])
    evaluator = make_phase_a_case_evaluator(
        predictor=predictor,
        reference_check=lambda request, proposal: reference.validate(
            repository=repository,
            ref=commit_sha,
            request=request,
            proposal=proposal,
        ),
        reference_preflight=lambda request: reference.preflight(
            repository=repository,
            ref=commit_sha,
            request=request,
        ),
    )
    summary = EvaluationService(workspace).evaluate_partition(
        completed.id,
        DatasetSplit.DEV,
        evaluator,
    )
    return _report(
        workspace=workspace,
        experiment_id=completed.id,
        trainer_id=trainer_id,
        display_name=display_name,
        trained=False,
        summary=summary,
        training_job=None,
    )


def _run_trained(
    *,
    workspace: Workspace,
    project_id: str,
    dataset_id: str,
    snapshot_id: str,
    trainer_id: str,
    display_name: str,
    feature_dim: int,
    alpha: float,
    seed: int,
    timeout_seconds: float,
) -> dict[str, object]:
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id=trainer_id,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={
            "feature_dim": feature_dim,
            "alpha": alpha,
            "experiment_kind": "phase-a-expanded-dev-bakeoff",
        },
        seed=seed,
        contract_snapshot_id=snapshot_id,
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
                    f"{display_name} training exceeded {timeout_seconds:g} seconds"
                )
            time.sleep(0.05)
            state = training.refresh_job(experiment.id, state.job.id)

        if state.job.status is not JobStatus.COMPLETED:
            raise RuntimeError(
                f"{display_name} training ended {state.job.status.value}: "
                f"{state.job.error or 'no worker error'}"
            )
        state = training.refresh_job(experiment.id, state.job.id)
        completed = state.experiment
        job = state.job
    finally:
        training.shutdown()

    if completed.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError(f"{display_name} experiment did not complete")
    if completed.model_artifact_digest is None:
        raise RuntimeError(f"{display_name} has no model artifact")

    summaries = evaluate_packaged_experiment(
        workspace,
        completed.id,
        splits=(DatasetSplit.DEV,),
    )
    if len(summaries) != 1 or summaries[0].split is not DatasetSplit.DEV:
        raise RuntimeError(f"{display_name} did not produce exactly one DEV summary")
    return _report(
        workspace=workspace,
        experiment_id=completed.id,
        trainer_id=trainer_id,
        display_name=display_name,
        trained=True,
        summary=summaries[0],
        training_job={
            "id": job.id,
            "status": job.status.value,
            "result_artifact_digest": job.result_artifact_digest,
            "event_artifact_digest": job.event_artifact_digest,
            "error": job.error,
        },
    )


def _report(
    *,
    workspace: Workspace,
    experiment_id: str,
    trainer_id: str,
    display_name: str,
    trained: bool,
    summary: EvaluationSummary,
    training_job: Mapping[str, object] | None,
) -> dict[str, object]:
    experiment = ExperimentService(workspace).get(experiment_id)
    metrics = summarize_phase_a_experiment(
        workspace,
        experiment_id,
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
    veto_nonzero = [
        row
        for row in metric_rows
        if row["veto"] is True
        and isinstance(row["value"], (int, float))
        and float(row["value"]) != 0.0
    ]
    model_size = None
    if experiment.model_artifact_digest is not None:
        model_size = workspace.artifacts.resolve(
            experiment.model_artifact_digest
        ).stat().st_size
    failures = _failure_rows(workspace, experiment_id)
    return {
        "experiment_id": experiment_id,
        "trainer_id": trainer_id,
        "display_name": display_name,
        "trained": trained,
        "training_job": dict(training_job) if training_job is not None else None,
        "model_artifact_digest": experiment.model_artifact_digest,
        "model_size_bytes": model_size,
        "dev": {
            "total": summary.total,
            "correct": summary.correct,
            "failures": summary.failures,
            "veto_failures": summary.veto_failures,
            "mean_latency_ms": summary.mean_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "metrics": metric_rows,
            "failure_rows": failures,
        },
        "veto_failure_count": len(veto_nonzero),
        "veto_metrics_nonzero": veto_nonzero,
    }


def _dev_gate_passed(report: Mapping[str, object]) -> bool:
    dev = _mapping(report, "dev")
    return (
        report.get("trainer_id") == PHASE_A_CANDIDATE_TRAINER_ID
        and report.get("veto_failure_count") == 0
        and dev.get("total") == 60
        and dev.get("veto_failures") == 0
    )


def _failure_rows(
    workspace: Workspace,
    experiment_id: str,
) -> list[dict[str, object]]:
    with workspace.database.connection() as conn:
        rows = conn.execute(
            "SELECT kind,severity,example_id,evidence_artifact_digest "
            "FROM failures WHERE experiment_id=? AND split=? "
            "ORDER BY example_id",
            (experiment_id, DatasetSplit.DEV.value),
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


def _validate_expanded_freeze_receipt(receipt: Mapping[str, object]) -> None:
    if receipt.get("schema") != "ml-lab-phase-a-expanded-freeze-receipt/1":
        raise ValueError("Unsupported expanded Phase A freeze receipt")
    claimed = receipt.get("payload_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("Expanded freeze payload hash missing")
    base = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    observed = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    if observed != claimed:
        raise ValueError("Expanded freeze payload hash mismatch")
    if receipt.get("ok") is not True or receipt.get("frozen") is not True:
        raise ValueError("Expanded DEV bake-off requires a clean frozen corpus")
    if receipt.get("model_training_started") is not False:
        raise ValueError("Expanded freeze receipt already claims model training")
    if receipt.get("production_data") is not False:
        raise ValueError("Production data cannot enter expanded bake-off")
    if receipt.get("transcript_derived") is not False:
        raise ValueError("Transcript-derived data cannot enter expanded bake-off")
    if receipt.get("protected_splits_in_trainer_handles") is not False:
        raise ValueError("Protected splits escaped into trainer handles")
    if set(_mapping(receipt, "trainer_handles")) != {"TRAIN", "DEV"}:
        raise ValueError("Expanded trainer handles must remain TRAIN/DEV only")
    zero = _mapping(receipt, "zero_model_redteam")
    if zero.get("case_count") != 24 or zero.get("passed_count") != 24:
        raise ValueError("Expanded bake-off requires 24/24 zero-model invariants")
    if zero.get("part_of_residual_dataset") is not False:
        raise ValueError("Zero-model invariants entered residual dataset")
    dataset = _mapping(receipt, "dataset")
    if dataset.get("state") != "FROZEN" or dataset.get("example_count") != 396:
        raise ValueError("Expanded residual dataset must be frozen at 396 examples")
    partitions = _mapping(dataset, "partitions")
    expected = {"TRAIN": 240, "DEV": 60, "TEST": 60, "REDTEAM": 36}
    actual = {
        split: _partition_count(partitions, split)
        for split in expected
    }
    if actual != expected:
        raise ValueError(f"Expanded partition counts changed: {actual}")


def _mapping(value: Mapping[str, object], key: str) -> dict[str, object]:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be an object")
    return {str(item_key): item_value for item_key, item_value in raw.items()}


def _required_text(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _partition_count(partitions: Mapping[str, object], split: str) -> int:
    row = partitions.get(split)
    if not isinstance(row, Mapping):
        raise ValueError(f"Missing {split} partition")
    count = row.get("example_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{split} partition count is invalid")
    return count
