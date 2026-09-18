from pathlib import Path

import pytest

from ml_lab.core.models import DatasetSplit, ExperimentStatus, MetricDirection, MetricValue
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


def _frozen_dataset(tmp_path: Path) -> tuple[Workspace, str, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Experiment tests")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "frozen")
    texts = {
        DatasetSplit.TRAIN: "red comet above eastern forest",
        DatasetSplit.DEV: "silver key beneath quiet bridge",
        DatasetSplit.TEST: "amber candle inside old chapel",
        DatasetSplit.REDTEAM: "cancel the blue bargain immediately",
    }
    for index, (split, text) in enumerate(texts.items()):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=f"e-{index}",
                split=split,
                source_id=f"s-{index}",
                lineage_group=f"g-{index}",
                payload={"text": text},
                label={"class": split.value},
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)
    return workspace, project.id, dataset.id


def test_trainer_spec_cannot_expose_protected_partitions(tmp_path: Path) -> None:
    workspace, project_id, dataset_id = _frozen_dataset(tmp_path)
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id="sparse-v1",
        runtime_pack_id="builtin-sparse",
        config={"alpha": 0.1},
        seed=42,
    )

    train_spec = experiments.trainer_job_spec(experiment.id)
    partitions = train_spec["input_partitions"]
    assert isinstance(partitions, dict)
    assert set(partitions) == {"TRAIN", "DEV"}

    eval_spec = experiments.evaluation_job_spec(experiment.id)
    eval_partitions = eval_spec["input_partitions"]
    assert isinstance(eval_partitions, dict)
    assert set(eval_partitions) == {"TRAIN", "DEV", "TEST", "REDTEAM"}


def test_completed_experiment_is_immutable_and_persists_metrics(tmp_path: Path) -> None:
    workspace, project_id, dataset_id = _frozen_dataset(tmp_path)
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id="sparse-v1",
        runtime_pack_id="builtin-sparse",
        seed=7,
        environment={"cpu": "test"},
    )
    running = experiments.start(experiment.id)
    assert running.status is ExperimentStatus.RUNNING
    model = workspace.artifacts.commit_bytes(b"model-bytes", media_type="application/octet-stream")
    completed = experiments.complete(
        experiment.id,
        model_artifact_digest=model.digest,
        metrics=(
            MetricValue("accuracy", 0.9, MetricDirection.HIGHER, False),
            MetricValue("false_commitments", 0.0, MetricDirection.ZERO, True),
        ),
    )
    assert completed.status is ExperimentStatus.COMPLETED
    assert completed.model_artifact_digest == model.digest
    assert completed.metrics_artifact_digest is not None
    assert completed.manifest_artifact_digest is not None
    metrics = experiments.metrics(experiment.id)
    assert [metric.metric_id for metric in metrics] == ["accuracy", "false_commitments"]

    with pytest.raises(RuntimeError, match="Cannot complete"):
        experiments.complete(experiment.id, metrics=())
    with pytest.raises(RuntimeError, match="Terminal experiments"):
        experiments.finish_without_success(experiment.id, ExperimentStatus.FAILED)


def test_experiment_requires_frozen_dataset(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Experiment tests")
    dataset = DatasetService(workspace).create(project.id, "draft")
    experiments = ExperimentService(workspace)
    with pytest.raises(RuntimeError, match="FROZEN"):
        experiments.create(
            project_id=project.id,
            dataset_id=dataset.id,
            trainer_id="trainer",
            runtime_pack_id="runtime",
        )
