import json
from pathlib import Path

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID
from ml_lab.core.models import (
    DatasetSplit,
    ExperimentRecord,
    ExperimentStatus,
    FailureSeverity,
    utc_now_iso,
)
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.runners import evaluate_generic_sparse_experiment
from ml_lab.evaluation.service import CaseOutcome, EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    PHASE_A_CANDIDATE_TRAINER_ID,
    PHASE_A_RUNTIME_PACK_ID,
    SPARSE_RUNTIME_PACK_ID,
    SPARSE_TRAINER_ID,
)
from ml_lab.trainers.sparse_nb import SparseNBBuilder
from ml_lab.ui.compare import CompareController


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



def test_compare_drills_from_case_to_immutable_failure_record(tmp_path: Path) -> None:
    workspace, experiment_id = _completed_sparse_experiment(tmp_path)
    experiment = ExperimentService(workspace).get(experiment_id)
    evaluation = EvaluationService(workspace)

    def evaluator(row: dict[str, object]) -> CaseOutcome:
        if row["example_id"] == "test-a":
            return CaseOutcome(
                observed={"prediction": "WRONG"},
                correct=False,
                latency_ms=0.1,
                failure_kind="CLASSIFICATION_MISMATCH",
                failure_severity=FailureSeverity.NON_VETO,
                evidence={"source": "compare-drilldown-test"},
            )
        return CaseOutcome(
            observed=row["label"],
            correct=True,
            latency_ms=0.1,
        )

    evaluation.evaluate_partition(
        experiment_id,
        DatasetSplit.TEST,
        evaluator,
    )

    controller = CompareController()
    try:
        controller.bind_project(
            workspace,
            experiment.project_id,
            "generic",
        )
        controller.selectExperiment(experiment_id)
        controller.setOnlyIncorrect(True)
        cases = controller.cases
        assert len(cases) == 1
        controller.selectCase(int(cases[0]["id"]))

        selected_case = controller.selectedCase
        failure_id = str(selected_case["failureId"])
        assert failure_id

        immutable = FailureService(workspace).immutable_payload(failure_id)
        selected_failure = controller.selectedFailure
        assert selected_failure["id"] == failure_id
        assert selected_failure["kind"] == "CLASSIFICATION_MISMATCH"
        assert selected_failure["severity"] == FailureSeverity.NON_VETO.value
        assert json.loads(str(selected_failure["payload"])) == immutable
        assert (
            selected_failure["evidenceDigest"]
            == immutable["evidence_artifact_sha256"]
        )
    finally:
        controller.shutdown()


def test_compare_allows_dev_only_for_phase_a_and_recognizes_candidate_evaluator(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "phase-a-workspace")
    project = workspace.create_project("Phase A", PHASE_A_ADAPTER_ID)
    controller = CompareController()
    try:
        controller.bind_project(workspace, project.id, PHASE_A_ADAPTER_ID)
        controller.setSplit("DEV")
        assert controller.split == DatasetSplit.DEV.value
        assert controller.developmentSplit is True

        candidate = ExperimentRecord(
            id="candidate-experiment",
            project_id=project.id,
            dataset_id="dataset",
            trainer_id=PHASE_A_CANDIDATE_TRAINER_ID,
            runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
            status=ExperimentStatus.COMPLETED,
            config_json="{}",
            seed=0,
            environment_json="{}",
            created_at=utc_now_iso(),
            contract_snapshot_id="snapshot",
            model_artifact_digest="a" * 64,
        )
        assert controller._has_packaged_evaluator(candidate) is True
    finally:
        controller.shutdown()

    generic_workspace = Workspace.create(tmp_path / "generic-workspace")
    generic_project = generic_workspace.create_project("Generic", "generic")
    generic = CompareController()
    try:
        generic.bind_project(generic_workspace, generic_project.id, "generic")
        generic.setSplit("DEV")
        assert generic.split == DatasetSplit.TEST.value
        assert generic.developmentSplit is False
    finally:
        generic.shutdown()
