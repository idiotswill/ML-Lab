from __future__ import annotations

import json

from PySide6.QtCore import Property, QObject, Signal, Slot

from ml_lab.core.models import (
    DatasetState,
    ExperimentRecord,
    ExperimentStatus,
    JobRecord,
    JobStatus,
)
from ml_lab.datasets.service import DatasetService
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import TrainingOption, TrainingService, training_options


class ExperimentsController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._jobs: JobManager | None = None
        self._training: TrainingService | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._job_ids: dict[str, str] = {}

    def bind_project(
        self,
        workspace: Workspace,
        jobs: JobManager,
        project_id: str,
        adapter_id: str,
    ) -> None:
        self._workspace = workspace
        self._jobs = jobs
        self._training = TrainingService(workspace, jobs)
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_experiment_id = ""
        self._job_ids.clear()
        self._reconcile_project()
        self.changed.emit()

    def clear_project(self) -> None:
        self._workspace = None
        self._jobs = None
        self._training = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._job_ids.clear()
        self.changed.emit()

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(bool, notify=changed)
    def hasRunnableTrainer(self) -> bool:
        if self._workspace is None or not self._project_id:
            return False
        return bool(training_options(self._adapter_id))

    @Property(list, notify=changed)
    def frozenDatasets(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        return [
            {
                "id": item.id,
                "name": item.name,
                "exampleCount": item.example_count,
                "frozenAt": item.frozen_at or "",
                "manifestDigest": item.manifest_artifact_digest or "",
            }
            for item in DatasetService(self._workspace).list_for_project(self._project_id)
            if item.state is DatasetState.FROZEN
        ]

    @Property(list, notify=changed)
    def trainerOptions(self) -> list[dict[str, object]]:
        if self._workspace is None or not self._project_id:
            return []
        return [_training_option_dict(item) for item in training_options(self._adapter_id)]

    @Property(list, notify=changed)
    def runtimeOptions(self) -> list[dict[str, str]]:
        if self._workspace is None or not self._project_id:
            return []
        seen: set[str] = set()
        result: list[dict[str, str]] = []
        for option in training_options(self._adapter_id):
            if option.runtime_pack_id in seen:
                continue
            seen.add(option.runtime_pack_id)
            result.append(
                {
                    "id": option.runtime_pack_id,
                    "name": option.runtime_display_name,
                }
            )
        return result

    @Property(list, notify=changed)
    def experiments(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        service = ExperimentService(self._workspace)
        dataset_names = {
            item.id: item.name
            for item in DatasetService(self._workspace).list_for_project(self._project_id)
        }
        return [
            _experiment_dict(item, dataset_names.get(item.dataset_id, "Unknown dataset"))
            for item in service.list_for_project(self._project_id)
        ]

    @Property(dict, notify=changed)
    def selectedExperiment(self) -> dict[str, object]:
        item = self._selected_experiment()
        if item is None:
            return {}
        return _experiment_dict(item, self._dataset_name(item.dataset_id))

    @Property(dict, notify=changed)
    def selectedJob(self) -> dict[str, object]:
        job = self._selected_job()
        return _job_dict(job) if job is not None else {}

    @Property(list, notify=changed)
    def selectedMetrics(self) -> list[dict[str, object]]:
        if not self._workspace or not self._selected_experiment_id:
            return []
        return [
            {
                "metricId": item.metric_id,
                "value": item.value,
                "direction": item.direction.value,
                "veto": item.veto,
            }
            for item in ExperimentService(self._workspace).metrics(
                self._selected_experiment_id
            )
        ]

    @Property(bool, notify=changed)
    def canCancelSelected(self) -> bool:
        item = self._selected_experiment()
        if item is None or item.status is not ExperimentStatus.RUNNING:
            return False
        job = self._selected_job()
        if job is None:
            return False
        return job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING}

    @Slot(str, str, str, str, str, str, str, str)
    def createAndLaunch(
        self,
        dataset_id: str,
        trainer_id: str,
        runtime_pack_id: str,
        seed: str,
        feature_dim: str,
        alpha: str,
        text_key: str,
        label_key: str,
    ) -> None:
        if not self._workspace or not self._training or not self._project_id:
            self.operationFailed.emit("Experiment error", "Open a project first.")
            return
        try:
            option = self._require_option(trainer_id, runtime_pack_id)
            clean_text_key = text_key.strip()
            clean_label_key = label_key.strip()
            if not clean_text_key or not clean_label_key:
                raise ValueError("Text key and label key are required.")
            seed_value = int(seed.strip())
            feature_dim_value = int(feature_dim.strip())
            alpha_value = float(alpha.strip())
            if feature_dim_value < 256:
                raise ValueError("Feature dimension must be at least 256.")
            if alpha_value <= 0:
                raise ValueError("Alpha must be greater than zero.")

            dataset = DatasetService(self._workspace).get(dataset_id)
            if dataset.project_id != self._project_id:
                raise ValueError("Dataset does not belong to this project.")
            if dataset.state is not DatasetState.FROZEN:
                raise RuntimeError("Training requires a frozen dataset.")

            experiment_service = ExperimentService(self._workspace)
            experiment = experiment_service.create(
                project_id=self._project_id,
                dataset_id=dataset.id,
                trainer_id=option.trainer_id,
                runtime_pack_id=option.runtime_pack_id,
                config={
                    "feature_dim": feature_dim_value,
                    "alpha": alpha_value,
                    "text_key": clean_text_key,
                    "label_key": clean_label_key,
                },
                seed=seed_value,
                contract_snapshot_id=dataset.contract_snapshot_id,
            )
            self._selected_experiment_id = experiment.id
            try:
                state = self._training.launch(experiment.id)
            except Exception:
                current = experiment_service.get(experiment.id)
                if current.status is ExperimentStatus.QUEUED:
                    experiment_service.finish_without_success(
                        experiment.id,
                        ExperimentStatus.FAILED,
                    )
                raise
            self._job_ids[experiment.id] = state.job.id
            self.changed.emit()
            self.operationCompleted.emit(f"Launched experiment {experiment.id[:8]}")
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            self.changed.emit()
            self.operationFailed.emit("Experiment error", str(exc))

    @Slot(str)
    def selectExperiment(self, experiment_id: str) -> None:
        if not self._workspace or not self._project_id:
            return
        try:
            item = ExperimentService(self._workspace).get(experiment_id)
        except KeyError as exc:
            self.operationFailed.emit("Experiment error", str(exc))
            return
        if item.project_id != self._project_id:
            self.operationFailed.emit(
                "Experiment error",
                "Experiment does not belong to this project.",
            )
            return
        self._selected_experiment_id = item.id
        self._ensure_job_link(item)
        self.changed.emit()

    @Slot()
    def cancelSelected(self) -> None:
        if not self._training or not self._selected_experiment_id:
            return
        item = self._selected_experiment()
        if item is None or item.status is not ExperimentStatus.RUNNING:
            return
        try:
            job_id = self._job_ids.get(item.id)
            if not job_id:
                job_id = self._training.job_for_experiment(item.id).id
                self._job_ids[item.id] = job_id
            self._training.cancel_job(item.id, job_id)
            self.changed.emit()
            self.operationCompleted.emit("Cancellation requested")
        except (KeyError, RuntimeError, OSError) as exc:
            self.operationFailed.emit("Cancellation error", str(exc))

    def refresh(self) -> None:
        if not self._workspace or not self._training or not self._project_id:
            return
        changed = False
        for item in ExperimentService(self._workspace).list_for_project(self._project_id):
            if item.status is not ExperimentStatus.RUNNING:
                continue
            job_id = self._job_ids.get(item.id)
            if not job_id:
                try:
                    job_id = self._training.job_for_experiment(item.id).id
                except KeyError:
                    continue
                self._job_ids[item.id] = job_id
            try:
                before = self._job_signature(job_id, item.status)
                state = self._training.refresh_job(item.id, job_id)
                after = (
                    state.job.status.value,
                    state.job.progress,
                    state.job.message,
                    state.experiment.status.value,
                )
            except (KeyError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
                self.operationFailed.emit("Training refresh error", str(exc))
                continue
            if before != after:
                changed = True
        if changed:
            self.changed.emit()

    def _reconcile_project(self) -> None:
        if not self._workspace or not self._training or not self._project_id:
            return
        for item in ExperimentService(self._workspace).list_for_project(self._project_id):
            if item.status is not ExperimentStatus.RUNNING:
                continue
            try:
                job = self._training.job_for_experiment(item.id)
                self._job_ids[item.id] = job.id
                self._training.refresh_job(item.id, job.id)
            except (KeyError, RuntimeError, ValueError, OSError, json.JSONDecodeError):
                continue

    def _ensure_job_link(self, item: ExperimentRecord) -> None:
        if not self._training or item.id in self._job_ids:
            return
        if item.status is ExperimentStatus.QUEUED:
            return
        try:
            self._job_ids[item.id] = self._training.job_for_experiment(item.id).id
        except KeyError:
            pass

    def _selected_experiment(self) -> ExperimentRecord | None:
        if not self._workspace or not self._selected_experiment_id:
            return None
        try:
            item = ExperimentService(self._workspace).get(self._selected_experiment_id)
        except KeyError:
            self._selected_experiment_id = ""
            return None
        if item.project_id != self._project_id:
            self._selected_experiment_id = ""
            return None
        return item

    def _selected_job(self) -> JobRecord | None:
        if not self._jobs or not self._selected_experiment_id:
            return None
        job_id = self._job_ids.get(self._selected_experiment_id)
        if not job_id:
            return None
        try:
            return self._jobs.get(job_id)
        except KeyError:
            return None

    def _dataset_name(self, dataset_id: str) -> str:
        if not self._workspace:
            return "Unknown dataset"
        try:
            return DatasetService(self._workspace).get(dataset_id).name
        except KeyError:
            return "Unknown dataset"

    def _require_option(self, trainer_id: str, runtime_pack_id: str) -> TrainingOption:
        for option in training_options(self._adapter_id):
            if option.trainer_id == trainer_id:
                if option.runtime_pack_id != runtime_pack_id:
                    raise ValueError(
                        "Selected runtime is not compatible with the trainer."
                    )
                return option
        raise ValueError("Selected trainer is not runnable for this project adapter.")

    def _job_signature(
        self,
        job_id: str,
        experiment_status: ExperimentStatus,
    ) -> tuple[str, float, str, str]:
        if not self._jobs:
            return ("", 0.0, "", experiment_status.value)
        job = self._jobs.get(job_id)
        return (
            job.status.value,
            job.progress,
            job.message,
            experiment_status.value,
        )


def _training_option_dict(item: TrainingOption) -> dict[str, object]:
    return {
        "trainerId": item.trainer_id,
        "name": item.display_name,
        "runtimeId": item.runtime_pack_id,
        "runtimeName": item.runtime_display_name,
        "defaultFeatureDim": item.default_feature_dim,
        "defaultAlpha": item.default_alpha,
        "defaultTextKey": item.default_text_key,
        "defaultLabelKey": item.default_label_key,
    }


def _experiment_dict(item: ExperimentRecord, dataset_name: str) -> dict[str, object]:
    config = json.loads(item.config_json)
    return {
        "id": item.id,
        "datasetId": item.dataset_id,
        "datasetName": dataset_name,
        "trainerId": item.trainer_id,
        "runtimeId": item.runtime_pack_id,
        "status": item.status.value,
        "seed": item.seed,
        "createdAt": item.created_at,
        "startedAt": item.started_at or "",
        "completedAt": item.completed_at or "",
        "modelDigest": item.model_artifact_digest or "",
        "metricsDigest": item.metrics_artifact_digest or "",
        "manifestDigest": item.manifest_artifact_digest or "",
        "contractSnapshotId": item.contract_snapshot_id or "",
        "config": config if isinstance(config, dict) else {},
    }


def _job_dict(item: JobRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "status": item.status.value,
        "progress": item.progress,
        "message": item.message,
        "error": item.error or "",
        "createdAt": item.created_at,
        "updatedAt": item.updated_at,
        "resultDigest": item.result_artifact_digest or "",
    }
