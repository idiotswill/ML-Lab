from __future__ import annotations

import json

from PySide6.QtCore import Property, QObject, Signal, Slot

from ml_lab.core.models import ExperimentRecord, ExperimentStatus, ModelStage, RegisteredModel
from ml_lab.experiments.service import ExperimentService
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace


class ModelsRegistryController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_model_id = ""

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_model_id = ""
        self.changed.emit()

    def clear_project(self) -> None:
        self._workspace = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_model_id = ""
        self.changed.emit()

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(list, notify=changed)
    def modelRows(self) -> list[dict[str, object]]:
        return [_model_row(item) for item in self._models()]

    @Property(list, notify=changed)
    def registerableExperiments(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        registry = ModelRegistryService(self._workspace)
        rows: list[dict[str, object]] = []
        for experiment in ExperimentService(self._workspace).list_for_project(self._project_id):
            if experiment.status is not ExperimentStatus.COMPLETED:
                continue
            if experiment.model_artifact_digest is None:
                continue
            if registry.find_by_experiment(experiment.id) is not None:
                continue
            rows.append(_experiment_row(experiment))
        return rows

    @Property(dict, notify=changed)
    def selectedModel(self) -> dict[str, object]:
        model = self._selected_model()
        return _model_detail(model) if model is not None else {}

    @Property(list, notify=changed)
    def stageHistory(self) -> list[dict[str, object]]:
        if not self._workspace or not self._selected_model_id:
            return []
        registry = ModelRegistryService(self._workspace)
        return [
            {
                "fromStage": str(item["from_stage"] or "REGISTERED"),
                "toStage": str(item["to_stage"]),
                "createdAt": str(item["created_at"]),
                "evidence": _pretty_object(item["evidence"]),
            }
            for item in registry.stage_history(self._selected_model_id)
        ]

    @Property(str, notify=changed)
    def nextStage(self) -> str:
        if not self._workspace or not self._selected_model_id:
            return ""
        next_stage = ModelRegistryService(self._workspace).next_stage(
            self._selected_model_id
        )
        return next_stage.value if next_stage is not None else ""

    @Property(bool, notify=changed)
    def canPromote(self) -> bool:
        return self._can_promote()

    @Property(str, notify=changed)
    def promotionMessage(self) -> str:
        if not self._workspace or not self._selected_model_id:
            return "Select a registered model to inspect its promotion path."
        registry = ModelRegistryService(self._workspace)
        model = registry.get(self._selected_model_id)
        next_stage = registry.next_stage(model.id)
        if next_stage is None:
            if model.stage is ModelStage.RELEASE_CANDIDATE:
                return (
                    "Release candidate reached. INTEGRATION_APPROVED is intentionally "
                    "outside the ordinary ML Lab promotion API."
                )
            return "No further ordinary Lab promotion stage is available."
        blockers = registry.promotion_blockers(model.id)
        if blockers:
            return f"{next_stage.value} blocked: " + " ".join(blockers)
        return f"Eligible for the next sequential stage: {next_stage.value}."

    @Slot(str)
    def selectModel(self, model_id: str) -> None:
        if any(item.id == model_id for item in self._models()):
            self._selected_model_id = model_id
            self.changed.emit()

    @Slot(str)
    def registerExperiment(self, experiment_id: str) -> None:
        if not self._workspace or not self._project_id:
            return
        try:
            experiments = ExperimentService(self._workspace)
            experiment = experiments.get(experiment_id)
            if experiment.project_id != self._project_id:
                raise ValueError("Experiment does not belong to the selected project.")
            model = ModelRegistryService(self._workspace).register_from_experiment(
                experiment.id,
                compatibility={
                    "adapter_id": self._adapter_id,
                    "trainer_id": experiment.trainer_id,
                    "runtime_pack_id": experiment.runtime_pack_id,
                    "dataset_id": experiment.dataset_id,
                    "contract_snapshot_id": experiment.contract_snapshot_id,
                },
            )
        except Exception as exc:
            self.operationFailed.emit("Model registration", str(exc))
            return
        self._selected_model_id = model.id
        self.changed.emit()
        self.operationCompleted.emit("Model registered at EXPERIMENT stage")

    @Slot(str)
    def promoteSelected(self, note: str) -> None:
        if not self._workspace or not self._selected_model_id:
            return
        registry = ModelRegistryService(self._workspace)
        next_stage = registry.next_stage(self._selected_model_id)
        if next_stage is None:
            self.operationFailed.emit(
                "Model promotion",
                "No further ordinary Lab promotion stage is available.",
            )
            return
        try:
            registry.promote(
                self._selected_model_id,
                next_stage,
                evidence={
                    "source": "models-registry-ui",
                    "note": note.strip(),
                },
            )
        except Exception as exc:
            self.operationFailed.emit("Model promotion", str(exc))
            self.changed.emit()
            return
        self.changed.emit()
        self.operationCompleted.emit(f"Model promoted to {next_stage.value}")

    def _models(self) -> list[RegisteredModel]:
        if not self._workspace or not self._project_id:
            return []
        return ModelRegistryService(self._workspace).list_for_project(
            self._project_id,
            limit=200,
        )

    def _selected_model(self) -> RegisteredModel | None:
        if not self._selected_model_id:
            return None
        return next(
            (item for item in self._models() if item.id == self._selected_model_id),
            None,
        )

    def _can_promote(self) -> bool:
        if not self._workspace or not self._selected_model_id:
            return False
        registry = ModelRegistryService(self._workspace)
        next_stage = registry.next_stage(self._selected_model_id)
        if next_stage is None:
            return False
        return not registry.promotion_blockers(self._selected_model_id)


def _experiment_row(item: ExperimentRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "trainer": item.trainer_id,
        "runtime": item.runtime_pack_id,
        "modelSha256": item.model_artifact_digest or "",
        "completedAt": item.completed_at or "",
    }


def _model_row(item: RegisteredModel) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "experimentId": item.experiment_id,
        "experimentShortId": item.experiment_id[:8],
        "stage": item.stage.value,
        "modelSha256": item.model_artifact_digest,
        "updatedAt": item.updated_at,
    }


def _model_detail(item: RegisteredModel) -> dict[str, object]:
    try:
        compatibility = json.loads(item.compatibility_json)
    except json.JSONDecodeError:
        compatibility = item.compatibility_json
    return {
        **_model_row(item),
        "manifestSha256": item.manifest_artifact_digest or "",
        "compatibility": _pretty_object(compatibility),
        "createdAt": item.created_at,
    }


def _pretty_object(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
