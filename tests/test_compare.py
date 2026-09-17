from pathlib import Path

from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.runners import evaluate_generic_sparse_experiment
from ml_lab.evaluation.service import EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID
from ml_lab.trainers.sparse_nb import SparseNBBuilder


def _completed_sparse_experiment(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Compare", "generic")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "compare-v1")
    rows = (
        (DatasetSplit.TRAIN, "train-a", "amber cedar orbit alpha", "A"),
        (DatasetSplit.TRAIN, "train-b", "violet marble signal beta", "B"),
        (DatasetSplit.DEV, "dev-a", "amber lantern alpha", "A"),
        (DatasetSplit.TEST, "test-a", "amber cedar beacon", "A"),
        (DatasetSplit.TEST, "test-b", "violet marble beacon", "B"),
        (DatasetSplit.REDTEAM, "red-a", "violet adversarial prism", "B"),
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

    builder = SparseNBBuilder(feature_dim=1024, alpha=0.5)
    builder.add("amber cedar orbit alpha", "A")
    builder.add("violet marble signal beta", "B")
    model = builder.finish()
    model_path = tmp_path / "model.json"
    model.save(model_path)
    model_ref = workspace.artifacts.commit_file(
        model_path,
        media_type="application/vnd.ml-lab.sparse-nb+json",
    )

    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id=SPARSE_TRAINER_ID,
        runtime_pack_id=SPARSE_RUNTIME_PACK_ID,
        config={
            "feature_dim": 1024,
            "alpha": 0.5,
            "text_key": "text",
            "label_key": "class",
        },
    )
    experiments.start(experiment.id)
    experiments.complete(
        experiment.id,
        metrics=(),
        model_artifact_digest=model_ref.digest,
    )
    return workspace, experiment.id


def test_generic_compare_runner_persists_protected_evidence_once(tmp_path: Path) -> None:
    workspace, experiment_id = _completed_sparse_experiment(tmp_path)

    first = evaluate_generic_sparse_experiment(
        workspace,
        experiment_id,
        splits=(DatasetSplit.TEST,),
    )
    assert len(first) == 1
    assert first[0].total == 2

    service = EvaluationService(workspace)
    progress = service.progress(experiment_id, DatasetSplit.TEST)
    assert progress.expected == 2
    assert progress.evaluated == 2
    assert progress.complete
    first_case_ids = [
        item.id for item in service.page_cases(experiment_id, DatasetSplit.TEST)
    ]

    repeated = evaluate_generic_sparse_experiment(
        workspace,
        experiment_id,
        splits=(DatasetSplit.TEST,),
    )
    assert repeated[0] == first[0]
    assert [
        item.id for item in service.page_cases(experiment_id, DatasetSplit.TEST)
    ] == first_case_ids
