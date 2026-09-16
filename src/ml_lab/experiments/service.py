from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping, Sequence

from ml_lab.core.models import (
    DatasetState,
    ExperimentRecord,
    ExperimentStatus,
    MetricDirection,
    MetricValue,
    utc_now_iso,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace

_TERMINAL_EXPERIMENTS = frozenset(
    {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.CANCELLED,
        ExperimentStatus.INTERRUPTED,
    }
)


class ExperimentService:
    """Create immutable experiment records around frozen dataset artifacts."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts
        self.datasets = DatasetService(workspace)

    def create(
        self,
        *,
        project_id: str,
        dataset_id: str,
        trainer_id: str,
        runtime_pack_id: str,
        config: Mapping[str, object] | None = None,
        seed: int = 0,
        contract_snapshot_id: str | None = None,
        environment: Mapping[str, object] | None = None,
    ) -> ExperimentRecord:
        dataset = self.datasets.get(dataset_id)
        if dataset.project_id != project_id:
            raise ValueError("Dataset does not belong to the experiment project.")
        if dataset.state is not DatasetState.FROZEN:
            raise RuntimeError("Experiments require a FROZEN dataset.")
        if dataset.contract_snapshot_id is not None:
            if contract_snapshot_id is None:
                contract_snapshot_id = dataset.contract_snapshot_id
            elif contract_snapshot_id != dataset.contract_snapshot_id:
                raise ValueError("Experiment contract does not match the frozen dataset contract.")
        clean_trainer = trainer_id.strip()
        clean_runtime = runtime_pack_id.strip()
        if not clean_trainer or not clean_runtime:
            raise ValueError("trainer_id and runtime_pack_id are required.")

        record = ExperimentRecord(
            id=str(uuid.uuid4()),
            project_id=project_id,
            dataset_id=dataset_id,
            contract_snapshot_id=contract_snapshot_id,
            trainer_id=clean_trainer,
            runtime_pack_id=clean_runtime,
            status=ExperimentStatus.QUEUED,
            config_json=canonical_json(dict(config or {})),
            seed=seed,
            environment_json=canonical_json(dict(environment or {})),
            created_at=utc_now_iso(),
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO experiments"
                "(id,project_id,dataset_id,contract_snapshot_id,trainer_id,runtime_pack_id,status,"
                "config_json,seed,environment_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.project_id,
                    record.dataset_id,
                    record.contract_snapshot_id,
                    record.trainer_id,
                    record.runtime_pack_id,
                    record.status.value,
                    record.config_json,
                    record.seed,
                    record.environment_json,
                    record.created_at,
                ),
            )
        return record

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self.database.connection() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment {experiment_id}")
        return _record_from_row(row)

    def list_for_project(self, project_id: str, *, limit: int = 200) -> list[ExperimentRecord]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def trainer_job_spec(self, experiment_id: str, *, include_dev: bool = True) -> dict[str, object]:
        record = self.get(experiment_id)
        handles = self.datasets.trainer_partition_handles(record.dataset_id, include_dev=include_dev)
        if "TEST" in handles or "REDTEAM" in handles:
            raise AssertionError("Protected evaluation partition escaped into trainer spec.")
        dataset = self.datasets.get(record.dataset_id)
        return {
            "protocol_version": 1,
            "kind": "TRAIN",
            "experiment_id": record.id,
            "project_id": record.project_id,
            "trainer_id": record.trainer_id,
            "runtime_pack_id": record.runtime_pack_id,
            "seed": record.seed,
            "config": json.loads(record.config_json),
            "contract_snapshot_id": record.contract_snapshot_id,
            "dataset_manifest_sha256": dataset.manifest_artifact_digest,
            "input_partitions": handles,
        }

    def evaluation_job_spec(self, experiment_id: str) -> dict[str, object]:
        record = self.get(experiment_id)
        dataset = self.datasets.get(record.dataset_id)
        return {
            "protocol_version": 1,
            "kind": "EVALUATE",
            "experiment_id": record.id,
            "project_id": record.project_id,
            "seed": record.seed,
            "contract_snapshot_id": record.contract_snapshot_id,
            "dataset_manifest_sha256": dataset.manifest_artifact_digest,
            "input_partitions": self.datasets.evaluation_partition_handles(record.dataset_id),
            "model_artifact_sha256": record.model_artifact_digest,
        }

    def start(self, experiment_id: str) -> ExperimentRecord:
        now = utc_now_iso()
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE experiments SET status=?,started_at=? WHERE id=? AND status=?",
                (
                    ExperimentStatus.RUNNING.value,
                    now,
                    experiment_id,
                    ExperimentStatus.QUEUED.value,
                ),
            )
            if cursor.rowcount != 1:
                existing = self.get(experiment_id)
                raise RuntimeError(f"Cannot start experiment from {existing.status.value} state.")
        return self.get(experiment_id)

    def complete(
        self,
        experiment_id: str,
        *,
        metrics: Sequence[MetricValue],
        model_artifact_digest: str | None = None,
    ) -> ExperimentRecord:
        record = self.get(experiment_id)
        if record.status is not ExperimentStatus.RUNNING:
            raise RuntimeError(f"Cannot complete experiment from {record.status.value} state.")
        _validate_metrics(metrics)
        if model_artifact_digest is not None:
            self.artifacts.resolve(model_artifact_digest)

        metric_payload = {
            "format_version": 1,
            "experiment_id": record.id,
            "metrics": [
                {
                    "metric_id": metric.metric_id,
                    "value": metric.value,
                    "direction": metric.direction.value,
                    "veto": metric.veto,
                }
                for metric in metrics
            ],
        }
        metrics_ref = self.artifacts.commit_bytes(
            (canonical_json(metric_payload) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.metrics+json",
            metadata={"experiment_id": record.id},
        )
        completed_at = utc_now_iso()
        dataset = self.datasets.get(record.dataset_id)
        manifest_payload = {
            "format_version": 1,
            "experiment_id": record.id,
            "project_id": record.project_id,
            "dataset_id": record.dataset_id,
            "dataset_manifest_sha256": dataset.manifest_artifact_digest,
            "contract_snapshot_id": record.contract_snapshot_id,
            "trainer_id": record.trainer_id,
            "runtime_pack_id": record.runtime_pack_id,
            "seed": record.seed,
            "config": json.loads(record.config_json),
            "environment": json.loads(record.environment_json),
            "model_artifact_sha256": model_artifact_digest,
            "metrics_artifact_sha256": metrics_ref.digest,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": completed_at,
        }
        manifest_ref = self.artifacts.commit_bytes(
            (canonical_json(manifest_payload) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.experiment-manifest+json",
            metadata={"experiment_id": record.id},
        )

        with self.database.transaction() as conn:
            for metric in metrics:
                conn.execute(
                    "INSERT INTO experiment_metrics(experiment_id,metric_id,value,direction,veto) "
                    "VALUES(?,?,?,?,?)",
                    (
                        record.id,
                        metric.metric_id,
                        metric.value,
                        metric.direction.value,
                        int(metric.veto),
                    ),
                )
            cursor = conn.execute(
                "UPDATE experiments SET status=?,model_artifact_digest=?,metrics_artifact_digest=?,"
                "manifest_artifact_digest=?,completed_at=? WHERE id=? AND status=?",
                (
                    ExperimentStatus.COMPLETED.value,
                    model_artifact_digest,
                    metrics_ref.digest,
                    manifest_ref.digest,
                    completed_at,
                    record.id,
                    ExperimentStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Experiment state changed while completing.")
        return self.get(record.id)

    def finish_without_success(
        self,
        experiment_id: str,
        status: ExperimentStatus,
    ) -> ExperimentRecord:
        if status not in {
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.INTERRUPTED,
        }:
            raise ValueError("status must be FAILED, CANCELLED, or INTERRUPTED")
        record = self.get(experiment_id)
        if record.status in _TERMINAL_EXPERIMENTS:
            raise RuntimeError("Terminal experiments are immutable.")
        completed_at = utc_now_iso()
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE experiments SET status=?,completed_at=? WHERE id=? AND status IN (?,?)",
                (
                    status.value,
                    completed_at,
                    experiment_id,
                    ExperimentStatus.QUEUED.value,
                    ExperimentStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Experiment state changed while finalizing.")
        return self.get(experiment_id)

    def metrics(self, experiment_id: str) -> list[MetricValue]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT metric_id,value,direction,veto FROM experiment_metrics "
                "WHERE experiment_id=? ORDER BY metric_id",
                (experiment_id,),
            ).fetchall()
        return [
            MetricValue(
                metric_id=row["metric_id"],
                value=float(row["value"]),
                direction=MetricDirection(row["direction"]),
                veto=bool(row["veto"]),
            )
            for row in rows
        ]


def _validate_metrics(metrics: Sequence[MetricValue]) -> None:
    seen: set[str] = set()
    for metric in metrics:
        clean = metric.metric_id.strip()
        if not clean:
            raise ValueError("metric_id cannot be empty")
        if clean in seen:
            raise ValueError(f"Duplicate metric_id {clean!r}")
        seen.add(clean)


def _record_from_row(row: sqlite3.Row) -> ExperimentRecord:
    return ExperimentRecord(
        id=row["id"],
        project_id=row["project_id"],
        dataset_id=row["dataset_id"],
        contract_snapshot_id=row["contract_snapshot_id"],
        trainer_id=row["trainer_id"],
        runtime_pack_id=row["runtime_pack_id"],
        status=ExperimentStatus(row["status"]),
        config_json=row["config_json"],
        seed=int(row["seed"]),
        environment_json=row["environment_json"],
        model_artifact_digest=row["model_artifact_digest"],
        metrics_artifact_digest=row["metrics_artifact_digest"],
        manifest_artifact_digest=row["manifest_artifact_digest"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
