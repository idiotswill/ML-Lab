from pathlib import Path

import pytest

from ml_lab.baselines.service import BaselineService
from ml_lab.core.models import DatasetSplit, ExperimentStatus
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.storage.workspace import Workspace


def test_baseline_cancellation_terminalizes_as_cancelled(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Baseline cancellation", "generic")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "protected")
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="test-1",
            split=DatasetSplit.TEST,
            source_id="synthetic:test-1",
            lineage_group="lineage:test-1",
            payload={"text": "distinct protected example"},
            label={"class": "A"},
            tags=(),
        ),
    )
    datasets.freeze(dataset.id)

    service = BaselineService(workspace)
    experiment = service.create(
        project_id=project.id,
        dataset_id=dataset.id,
        baseline_id="cancel-test",
    )
    with pytest.raises(InterruptedError, match="cancellation requested"):
        service.run(
            experiment.id,
            evaluator=lambda _row: pytest.fail("cancelled baseline must not evaluate a case"),
            splits=(DatasetSplit.TEST,),
            cancelled=lambda: True,
        )
    assert service.experiments.get(experiment.id).status is ExperimentStatus.CANCELLED
