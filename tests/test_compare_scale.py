from pathlib import Path

from ml_lab.core.models import ExperimentStatus
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.compare import CompareController


def _workspace_with_completed_experiments(
    tmp_path: Path,
    *,
    count: int = 1000,
) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Compare scale", "generic")
    dataset = DatasetService(workspace).create(project.id, "scale-v1")
    values = []
    for index in range(count):
        created_at = f"2026-01-01T00:00:00.{index:06d}+00:00"
        values.append(
            (
                f"exp-{index:04d}",
                project.id,
                dataset.id,
                None,
                "baseline:scale-fixture",
                "baseline-host-v1",
                ExperimentStatus.COMPLETED.value,
                "{}",
                0,
                "{}",
                None,
                None,
                None,
                created_at,
                created_at,
                created_at,
            )
        )
    with workspace.database.transaction() as conn:
        conn.executemany(
            "INSERT INTO experiments("
            "id,project_id,dataset_id,contract_snapshot_id,trainer_id,runtime_pack_id,"
            "status,config_json,seed,environment_json,model_artifact_digest,"
            "metrics_artifact_digest,manifest_artifact_digest,created_at,started_at,"
            "completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            values,
        )
    return workspace, project.id


def test_compare_pages_one_thousand_completed_experiments_without_eager_loading(
    tmp_path: Path,
) -> None:
    workspace, project_id = _workspace_with_completed_experiments(tmp_path)
    controller = CompareController()
    controller.bind_project(workspace, project_id, "generic")
    try:
        assert controller.experimentTotal == 1000
        assert controller.experimentPageNumber == 1
        assert controller.canPreviousExperimentPage is False
        assert controller.canNextExperimentPage is True

        first_page = controller.comparisonRows
        assert len(first_page) == 100
        assert first_page[0]["id"] == "exp-0999"
        assert first_page[-1]["id"] == "exp-0900"

        controller.selectExperiment("exp-0999")
        assert controller.selectedExperiment["id"] == "exp-0999"

        controller.nextExperimentPage()
        assert controller.experimentPageNumber == 2
        assert controller.selectedExperiment == {}
        second_page = controller.comparisonRows
        assert len(second_page) == 100
        assert second_page[0]["id"] == "exp-0899"
        assert second_page[-1]["id"] == "exp-0800"

        controller.selectExperiment("exp-0500")
        assert controller.experimentPageNumber == 5
        assert controller.selectedExperiment["id"] == "exp-0500"

        for _ in range(5):
            controller.nextExperimentPage()
        assert controller.experimentPageNumber == 10
        assert controller.canNextExperimentPage is False
        last_page = controller.comparisonRows
        assert len(last_page) == 100
        assert last_page[0]["id"] == "exp-0099"
        assert last_page[-1]["id"] == "exp-0000"

        controller.previousExperimentPage()
        assert controller.experimentPageNumber == 9
        assert controller.canPreviousExperimentPage is True
        assert len(controller.comparisonRows) == 100
    finally:
        controller.shutdown()
