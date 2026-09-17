from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass

from ml_lab.core.models import (
    TERMINAL_JOB_STATUSES,
    ExperimentRecord,
    ExperimentStatus,
    JobRecord,
    JobStatus,
    MetricDirection,
    MetricValue,
)
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.manager import JobManager
from ml_lab.jobs.protocol import JobSpec
from ml_lab.storage.workspace import Workspace

SPARSE_TRAINER_ID = "builtin.sparse_nb.v1"
SPARSE_RUNTIME_PACK_ID = "builtin-python"


@dataclass(frozen=True, slots=True)
class TrainingOption:
    trainer_id: str
    display_name: str
    runtime_pack_id: str
    runtime_display_name: str
    adapter_ids: tuple[str, ...]
    default_feature_dim: int = 32768
    default_alpha: float = 0.5
    default_text_key: str = "text"
    default_label_key: str = "class"


@dataclass(frozen=True, slots=True)
class TrainingState:
    experiment: ExperimentRecord
    job: JobRecord


_BUILTIN_TRAINING_OPTIONS = (
    TrainingOption(
        trainer_id=SPARSE_TRAINER_ID,
        display_name="Sparse Naive Bayes (CPU)",
        runtime_pack_id=SPARSE_RUNTIME_PACK_ID,
        runtime_display_name="Built-in Sparse CPU Runtime",
        adapter_ids=("generic",),
    ),
)


def training_options(adapter_id: str) -> tuple[TrainingOption, ...]:
    """Return trainers that are both packaged and compatible with this adapter.

    Model implementations that are not wired through the isolated worker boundary are
    intentionally absent. The UI must never imply that an experimental Python class is
    a runnable trainer merely because it exists in the source tree.
    """
    return tuple(
        option
        for option in _BUILTIN_TRAINING_OPTIONS
        if adapter_id in option.adapter_ids
    )


class TrainingService:
    """UI-facing training orchestration over split-safe immutable artifacts."""

    def __init__(self, workspace: Workspace, jobs: JobManager | None = None):
        self.workspace = workspace
        self.experiments = ExperimentService(workspace)
        self.jobs = jobs or JobManager(
            workspace.root,
            workspace.database,
            workspace.artifacts,
        )
        self._owns_jobs = jobs is None

    def launch(self, experiment_id: str) -> TrainingState:
        experiment = self.experiments.get(experiment_id)
        if experiment.trainer_id == SPARSE_TRAINER_ID:
            return self.launch_sparse(experiment_id)
        raise ValueError(f"No packaged launcher for trainer {experiment.trainer_id!r}.")

    def launch_sparse(self, experiment_id: str) -> TrainingState:
        experiment = self.experiments.get(experiment_id)
        if experiment.status is not ExperimentStatus.QUEUED:
            raise RuntimeError(
                f"Cannot launch experiment from {experiment.status.value} state."
            )
        if experiment.trainer_id != SPARSE_TRAINER_ID:
            raise ValueError(
                f"Sparse launcher requires trainer_id={SPARSE_TRAINER_ID!r}."
            )
        if experiment.runtime_pack_id != SPARSE_RUNTIME_PACK_ID:
            raise ValueError(
                f"Sparse launcher requires runtime_pack_id={SPARSE_RUNTIME_PACK_ID!r}."
            )
        spec = self.experiments.trainer_job_spec(experiment_id, include_dev=True)
        raw_handles = spec.get("input_partitions")
        if not isinstance(raw_handles, dict):
            raise RuntimeError("Trainer spec input_partitions must be an object.")
        handles = {str(key): str(value) for key, value in raw_handles.items()}
        if "TEST" in handles or "REDTEAM" in handles:
            raise AssertionError("Protected split escaped into training orchestration.")
        try:
            train_digest = handles["TRAIN"]
        except KeyError as exc:
            raise RuntimeError("Training experiment has no TRAIN partition.") from exc

        staged = {"train.jsonl": train_digest}
        if "DEV" in handles:
            staged["dev.jsonl"] = handles["DEV"]
        config = json.loads(experiment.config_json)
        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a JSON object.")
        payload = {
            "experiment_id": experiment.id,
            "text_key": str(config.get("text_key", "text")),
            "label_key": str(config.get("label_key", "class")),
            "feature_dim": _positive_int(
                config.get("feature_dim", 32768),
                "feature_dim",
            ),
            "alpha": _positive_float(config.get("alpha", 0.5), "alpha"),
        }

        self.experiments.start(experiment.id)
        try:
            job = self.jobs.start(
                "trainer.sparse_nb.v1",
                payload,
                staged_artifacts=staged,
            )
        except Exception:
            self.experiments.finish_without_success(
                experiment.id,
                ExperimentStatus.FAILED,
            )
            raise
        return TrainingState(self.experiments.get(experiment.id), job)

    def refresh(self, experiment_id: str) -> TrainingState:
        job = self.job_for_experiment(experiment_id)
        return self.refresh_job(experiment_id, job.id)

    def refresh_job(self, experiment_id: str, job_id: str) -> TrainingState:
        experiment = self.experiments.get(experiment_id)
        job = self.jobs.get(job_id)
        if job.status not in TERMINAL_JOB_STATUSES:
            return TrainingState(experiment, job)
        if experiment.status in {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.INTERRUPTED,
        }:
            return TrainingState(experiment, job)

        if job.status is JobStatus.COMPLETED:
            try:
                completed = self._complete_from_job(experiment, job)
            except Exception:
                current = self.experiments.get(experiment.id)
                if current.status is ExperimentStatus.RUNNING:
                    with suppress(RuntimeError):
                        self.experiments.finish_without_success(
                            experiment.id,
                            ExperimentStatus.FAILED,
                        )
                raise
            return TrainingState(completed, job)
        status = {
            JobStatus.FAILED: ExperimentStatus.FAILED,
            JobStatus.CANCELLED: ExperimentStatus.CANCELLED,
            JobStatus.INTERRUPTED: ExperimentStatus.INTERRUPTED,
        }.get(job.status)
        if status is None:
            raise RuntimeError(f"Unexpected terminal job status {job.status.value}.")
        failed = self.experiments.finish_without_success(experiment.id, status)
        return TrainingState(failed, job)

    def cancel(self, experiment_id: str) -> TrainingState:
        job = self.job_for_experiment(experiment_id)
        return self.cancel_job(experiment_id, job.id)

    def cancel_job(self, experiment_id: str, job_id: str) -> TrainingState:
        job = self.jobs.get(job_id)
        self.jobs.cancel(job.id)
        return TrainingState(self.experiments.get(experiment_id), self.jobs.get(job.id))

    def job_for_experiment(self, experiment_id: str) -> JobRecord:
        matches: list[JobRecord] = []
        for job in self.jobs.list_recent(limit=500):
            spec_path = job.staging_dir / "job_spec.json"
            if not spec_path.is_file():
                continue
            try:
                spec = JobSpec.read(spec_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if spec.payload.get("experiment_id") == experiment_id:
                matches.append(job)
        if not matches:
            raise KeyError(f"No training job found for experiment {experiment_id}")
        matches.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return matches[0]

    def shutdown(self) -> None:
        if self._owns_jobs:
            self.jobs.shutdown()

    def _complete_from_job(
        self,
        experiment: ExperimentRecord,
        job: JobRecord,
    ) -> ExperimentRecord:
        if job.result_artifact_digest is None:
            raise RuntimeError("Completed training job has no result artifact.")
        result_path = self.workspace.artifacts.resolve(job.result_artifact_digest)
        decoded = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Training result must be a JSON object.")
        if decoded.get("experiment_id") != experiment.id:
            raise ValueError("Training result experiment identity mismatch.")
        raw_outputs = decoded.get("output_artifacts")
        if not isinstance(raw_outputs, dict):
            raise ValueError("Training result is missing output_artifacts.")
        model_output = raw_outputs.get("model")
        if not isinstance(model_output, dict):
            raise ValueError("Training result is missing model output.")
        model_digest = model_output.get("sha256")
        if not isinstance(model_digest, str) or not model_digest:
            raise ValueError("Training model output is missing sha256.")

        metrics: list[MetricValue] = []
        dev_accuracy = decoded.get("dev_accuracy")
        if isinstance(dev_accuracy, (int, float)) and not isinstance(dev_accuracy, bool):
            metrics.append(
                MetricValue(
                    metric_id="dev_accuracy",
                    value=float(dev_accuracy),
                    direction=MetricDirection.HIGHER,
                    veto=False,
                )
            )
        return self.experiments.complete(
            experiment.id,
            metrics=metrics,
            model_artifact_digest=model_digest,
        )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be > 0")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be > 0") from exc
    if result <= 0:
        raise ValueError(f"{field} must be > 0")
    return result
