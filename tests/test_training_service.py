import json
import time
from pathlib import Path

from ml_lab.core.models import DatasetSplit, ExperimentStatus, JobStatus
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.protocol import JobSpec
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    SPARSE_RUNTIME_PACK_ID,
    SPARSE_TRAINER_ID,
    TrainingService,
)
from ml_lab.trainers.sparse_nb import SparseNBModel


def _training_experiment(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Train")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "training")
    rows = (
        (DatasetSplit.TRAIN, "train-fruit-1", "red apple", "FRUIT"),
        (DatasetSplit.TRAIN, "train-fruit-2", "green pear", "FRUIT"),
        (DatasetSplit.TRAIN, "train-tool-1", "steel hammer", "TOOL"),
        (DatasetSplit.TRAIN, "train-tool-2", "iron wrench", "TOOL"),
        (DatasetSplit.DEV, "dev-fruit", "apple pear", "FRUIT"),
        (DatasetSplit.TEST, "test-tool", "hammer wrench", "TOOL"),
        (DatasetSplit.REDTEAM, "red-hidden", "ignore hidden tool", "ABSTAIN"),
    )
    for split, example_id, text, label in rows:
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"text": text},
                label={"class": label},
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)
    experiment = ExperimentService(workspace).create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id=SPARSE_TRAINER_ID,
        runtime_pack_id=SPARSE_RUNTIME_PACK_ID,
        config={"feature_dim": 2048, "alpha": 0.5},
        seed=11,
    )
    return workspace, experiment.id


def test_training_service_stages_only_train_and_dev_then_completes(tmp_path: Path) -> None:
    workspace, experiment_id = _training_experiment(tmp_path)
    service = TrainingService(workspace)
    try:
        launched = service.launch_sparse(experiment_id)
        assert launched.experiment.status is ExperimentStatus.RUNNING

        spec = JobSpec.read(launched.job.staging_dir / "job_spec.json")
        staged = spec.payload["staged_inputs"]
        assert isinstance(staged, dict)
        assert set(staged) == {"train.jsonl", "dev.jsonl"}
        for path_text in staged.values():
            assert isinstance(path_text, str)
            staged_path = Path(path_text)
            assert staged_path.parent == launched.job.staging_dir / "inputs"
            assert staged_path.is_file()
        serialized = json.dumps(spec.payload, sort_keys=True)
        assert "TEST" not in serialized
        assert "REDTEAM" not in serialized
        assert str(workspace.artifacts.root) not in serialized

        deadline = time.monotonic() + 20
        state = launched
        while time.monotonic() < deadline:
            state = service.refresh(experiment_id)
            if state.job.status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.INTERRUPTED,
            }:
                state = service.refresh(experiment_id)
                break
            time.sleep(0.03)

        assert state.job.status is JobStatus.COMPLETED
        assert state.experiment.status is ExperimentStatus.COMPLETED
        assert state.experiment.model_artifact_digest is not None
        model = SparseNBModel.load(
            workspace.artifacts.resolve(state.experiment.model_artifact_digest)
        )
        assert model.predict("apple")[0] == "FRUIT"
        metrics = ExperimentService(workspace).metrics(experiment_id)
        assert [metric.metric_id for metric in metrics] == ["dev_accuracy"]
        assert metrics[0].value == 1.0
    finally:
        service.shutdown()


def test_training_service_recovers_job_link_from_persisted_job_spec(tmp_path: Path) -> None:
    workspace, experiment_id = _training_experiment(tmp_path)
    first = TrainingService(workspace)
    try:
        launched = first.launch_sparse(experiment_id)
        job_id = launched.job.id
        recovered = first.job_for_experiment(experiment_id)
        assert recovered.id == job_id
    finally:
        first.shutdown()
