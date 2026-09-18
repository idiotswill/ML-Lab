from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from ml_lab.core.models import ExperimentStatus, ModelStage
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.models_registry import ModelsRegistryController

pytestmark = pytest.mark.scale

_STAGES = (
    ModelStage.EXPERIMENT,
    ModelStage.SHADOW,
    ModelStage.ADVISORY,
    ModelStage.RELEASE_CANDIDATE,
)


def _workspace_with_registered_models(
    tmp_path: Path,
    *,
    count: int = 100,
) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Registry scale", "generic")
    dataset = DatasetService(workspace).create(project.id, "registry-scale-v1")
    model_artifact = workspace.artifacts.commit_bytes(b"shared-model-artifact")
    manifest_artifact = workspace.artifacts.commit_bytes(b"{}\n")

    experiment_rows: list[tuple[object, ...]] = []
    model_rows: list[tuple[object, ...]] = []
    history_rows: list[tuple[object, ...]] = []
    for index in range(count):
        experiment_id = f"exp-{index:04d}"
        model_id = f"model-{index:04d}"
        timestamp = f"2026-01-01T00:00:00.{index:06d}+00:00"
        stage = _STAGES[index % len(_STAGES)]
        experiment_rows.append(
            (
                experiment_id,
                project.id,
                dataset.id,
                None,
                "scale-fixture",
                "builtin-scale",
                ExperimentStatus.COMPLETED.value,
                "{}",
                index,
                "{}",
                model_artifact.digest,
                None,
                None,
                timestamp,
                timestamp,
                timestamp,
            )
        )
        model_rows.append(
            (
                model_id,
                project.id,
                experiment_id,
                model_artifact.digest,
                stage.value,
                '{"fixture":"registry-scale"}',
                manifest_artifact.digest,
                timestamp,
                timestamp,
            )
        )

        history_rows.append(
            (
                model_id,
                None,
                ModelStage.EXPERIMENT.value,
                "{}",
                timestamp,
            )
        )
        if stage in {
            ModelStage.SHADOW,
            ModelStage.ADVISORY,
            ModelStage.RELEASE_CANDIDATE,
        }:
            history_rows.append(
                (
                    model_id,
                    ModelStage.EXPERIMENT.value,
                    ModelStage.SHADOW.value,
                    "{}",
                    timestamp,
                )
            )
        if stage in {ModelStage.ADVISORY, ModelStage.RELEASE_CANDIDATE}:
            history_rows.append(
                (
                    model_id,
                    ModelStage.SHADOW.value,
                    ModelStage.ADVISORY.value,
                    "{}",
                    timestamp,
                )
            )
        if stage is ModelStage.RELEASE_CANDIDATE:
            history_rows.append(
                (
                    model_id,
                    ModelStage.ADVISORY.value,
                    ModelStage.RELEASE_CANDIDATE.value,
                    "{}",
                    timestamp,
                )
            )

    with workspace.database.transaction() as conn:
        conn.executemany(
            "INSERT INTO experiments("
            "id,project_id,dataset_id,contract_snapshot_id,trainer_id,runtime_pack_id,"
            "status,config_json,seed,environment_json,model_artifact_digest,"
            "metrics_artifact_digest,manifest_artifact_digest,created_at,started_at,"
            "completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            experiment_rows,
        )
        conn.executemany(
            "INSERT INTO models("
            "id,project_id,experiment_id,model_artifact_digest,stage,"
            "compatibility_json,manifest_artifact_digest,created_at,updated_at"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            model_rows,
        )
        conn.executemany(
            "INSERT INTO model_stage_history("
            "model_id,from_stage,to_stage,evidence_json,created_at"
            ") VALUES(?,?,?,?,?)",
            history_rows,
        )
    return workspace, project.id


def test_registry_pages_and_filters_one_hundred_models_without_eager_loading(
    tmp_path: Path,
) -> None:
    workspace, project_id = _workspace_with_registered_models(tmp_path)
    app = QCoreApplication.instance() or QCoreApplication([])
    assert app is not None

    controller = ModelsRegistryController()
    controller.bind_project(workspace, project_id, "generic")
    try:
        assert controller.modelTotal == 100
        assert controller.modelPageNumber == 1
        assert controller.canPreviousModelPage is False
        assert controller.canNextModelPage is True

        first_page = controller.modelRows
        assert len(first_page) == 50
        assert first_page[0]["id"] == "model-0099"
        assert first_page[-1]["id"] == "model-0050"

        controller.selectModel("model-0099")
        assert controller.selectedModelId == "model-0099"
        assert controller.selectedModel["id"] == "model-0099"
        assert len(controller.stageHistory) == 4

        controller.nextModelPage()
        assert controller.modelPageNumber == 2
        assert controller.selectedModelId == ""
        assert controller.selectedModel == {}
        assert controller.canNextModelPage is False
        second_page = controller.modelRows
        assert len(second_page) == 50
        assert second_page[0]["id"] == "model-0049"
        assert second_page[-1]["id"] == "model-0000"

        controller.setModelStageFilter(ModelStage.ADVISORY.value)
        assert controller.modelStageFilter == ModelStage.ADVISORY.value
        assert controller.modelPageNumber == 1
        assert controller.modelTotal == 25
        assert controller.canPreviousModelPage is False
        assert controller.canNextModelPage is False
        filtered = controller.modelRows
        assert len(filtered) == 25
        assert filtered[0]["id"] == "model-0098"
        assert filtered[-1]["id"] == "model-0002"
        assert {row["stage"] for row in filtered} == {ModelStage.ADVISORY.value}

        controller.setModelStageFilter("ALL")
        assert controller.modelTotal == 100
        assert controller.modelPageNumber == 1
        assert len(controller.modelRows) == 50
        assert controller.registerableExperiments == []
    finally:
        controller.shutdown()
