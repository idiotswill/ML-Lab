from pathlib import Path

from ml_lab.core.models import DatasetSplit, FailureSeverity
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.redteam.generic import GENERIC_REDTEAM_SUITE_ID, run_generic_sparse_redteam
from ml_lab.redteam.service import RedTeamService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID
from ml_lab.trainers.sparse_nb import SparseNBBuilder


def _completed_sparse_experiment(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Red Team", "generic")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "redteam-v1")
    rows = (
        (DatasetSplit.TRAIN, "train-a", "amber cedar orbit alpha", "A"),
        (DatasetSplit.TRAIN, "train-b", "violet marble signal beta", "B"),
        (DatasetSplit.DEV, "dev-a", "amber lantern alpha", "A"),
        (DatasetSplit.TEST, "test-a", "amber cedar beacon", "A"),
        (DatasetSplit.REDTEAM, "red-a", "violet marble adversarial", "A"),
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


def test_generic_redteam_records_runs_failures_and_regression_membership(
    tmp_path: Path,
) -> None:
    workspace, experiment_id = _completed_sparse_experiment(tmp_path)
    experiment = ExperimentService(workspace).get(experiment_id)

    summary = run_generic_sparse_redteam(
        workspace,
        experiment_id,
        seed=42,
        max_base_cases=10,
    )
    assert summary.base_cases == 1
    assert summary.generated_cases == 3
    assert summary.failures >= 1

    runs = RedTeamService(workspace)
    assert runs.count_for_project(experiment.project_id) == 1
    page = runs.page_for_project(experiment.project_id, limit=10)
    assert [item.id for item in page] == [summary.run_id]
    assert page[0].mutator_version == GENERIC_REDTEAM_SUITE_ID
    generated = runs.generated_cases(summary.run_id)
    assert len(generated) == 3
    assert {item["mutator_id"] for item in generated} == {
        "case-perturbation",
        "spacing-noise",
        "punctuation-noise",
    }

    failures = FailureService(workspace)
    failure_page = failures.page_for_project(experiment.project_id, limit=100)
    assert len(failure_page) == summary.failures
    assert all(item.redteam_run_id == summary.run_id for item in failure_page)
    assert failures.count_for_project(experiment.project_id) == summary.failures
    assert failures.count_for_project(
        experiment.project_id,
        severity=FailureSeverity.NON_VETO,
    ) == summary.failures

    failure_id = failure_page[0].id
    immutable_before = failures.immutable_payload(failure_id)
    failures.promote_to_regression(failure_id)
    assert failures.is_regression_case(failure_id)
    assert failures.regression_count(experiment.project_id) == 1
    assert failures.immutable_payload(failure_id) == immutable_before
