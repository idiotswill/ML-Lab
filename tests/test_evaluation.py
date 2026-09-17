import json
from pathlib import Path

import pytest

from ml_lab.core.models import DatasetSplit, FailureSeverity
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.service import CaseOutcome, EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


def _experiment(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Evaluation")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "frozen")
    rows = (
        (DatasetSplit.TRAIN, "train-a", "alpha training sample", "ALPHA"),
        (DatasetSplit.DEV, "dev-a", "beta development sample", "BETA"),
        (DatasetSplit.TEST, "test-a", "alpha evaluation sample", "ALPHA"),
        (DatasetSplit.TEST, "test-b", "beta evaluation sample", "BETA"),
        (DatasetSplit.REDTEAM, "red-a", "do not classify hidden value", "ABSTAIN"),
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
        trainer_id="test",
        runtime_pack_id="test",
    )
    return workspace, experiment.id


def test_evaluation_streams_protected_partition_and_records_failures(tmp_path: Path) -> None:
    workspace, experiment_id = _experiment(tmp_path)
    service = EvaluationService(workspace)

    def evaluator(row: dict[str, object]) -> CaseOutcome:
        label = row["label"]
        assert isinstance(label, dict)
        expected = str(label["class"])
        observed = "ALPHA"
        correct = observed == expected
        return CaseOutcome(
            observed={"class": observed},
            correct=correct,
            latency_ms=1.25,
            failure_kind=None if correct else "CLASS_MISMATCH",
            failure_severity=None if correct else FailureSeverity.NON_VETO,
            evidence=None if correct else {"source": "unit-test"},
        )

    summary = service.evaluate_partition(experiment_id, DatasetSplit.TEST, evaluator)
    assert summary.total == 2
    assert summary.correct == 1
    assert summary.failures == 1
    assert summary.veto_failures == 0
    assert summary.accuracy == 0.5
    assert summary.mean_latency_ms == pytest.approx(1.25)
    assert summary.p95_latency_ms == pytest.approx(1.25)

    cases = service.page_cases(experiment_id, DatasetSplit.TEST)
    assert [case.example_id for case in cases] == ["test-a", "test-b"]
    assert cases[0].failure_id is None
    assert cases[1].failure_id is not None
    assert json.loads(cases[1].expected_json) == {"class": "BETA"}
    assert json.loads(cases[1].observed_json) == {"class": "ALPHA"}

    failures = service.failures.page_for_project(
        service.experiments.get(experiment_id).project_id
    )
    assert len(failures) == 1
    assert failures[0].kind == "CLASS_MISMATCH"
    assert failures[0].example_id == "test-b"
    assert failures[0].split is DatasetSplit.TEST


def test_evaluation_resumes_partial_partition_without_rewriting_cases(tmp_path: Path) -> None:
    workspace, experiment_id = _experiment(tmp_path)
    service = EvaluationService(workspace)
    first_pass_calls: list[str] = []

    def interrupted(row: dict[str, object]) -> CaseOutcome:
        example_id = str(row["example_id"])
        first_pass_calls.append(example_id)
        if example_id == "test-b":
            raise RuntimeError("simulated interruption")
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.1)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        service.evaluate_partition(experiment_id, DatasetSplit.TEST, interrupted)

    progress = service.progress(experiment_id, DatasetSplit.TEST)
    assert progress.expected == 2
    assert progress.evaluated == 1
    assert not progress.complete
    assert first_pass_calls == ["test-a", "test-b"]

    resumed_calls: list[str] = []

    def resumed(row: dict[str, object]) -> CaseOutcome:
        resumed_calls.append(str(row["example_id"]))
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.2)

    summary = service.evaluate_partition(experiment_id, DatasetSplit.TEST, resumed)
    assert resumed_calls == ["test-b"]
    assert summary.total == 2
    assert summary.correct == 2
    assert service.progress(experiment_id, DatasetSplit.TEST).complete


def test_evaluation_cancel_keeps_partial_evidence_resumable(tmp_path: Path) -> None:
    workspace, experiment_id = _experiment(tmp_path)
    service = EvaluationService(workspace)
    cancel_requested = False

    def evaluator(row: dict[str, object]) -> CaseOutcome:
        nonlocal cancel_requested
        cancel_requested = True
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.1)

    with pytest.raises(InterruptedError, match="cancellation"):
        service.evaluate_partition(
            experiment_id,
            DatasetSplit.TEST,
            evaluator,
            cancelled=lambda: cancel_requested,
        )

    progress = service.progress(experiment_id, DatasetSplit.TEST)
    assert progress.evaluated == 1
    assert progress.expected == 2
    assert not progress.complete


def test_evaluation_is_immutable_after_partition_completes(tmp_path: Path) -> None:
    workspace, experiment_id = _experiment(tmp_path)
    service = EvaluationService(workspace)

    def evaluator(row: dict[str, object]) -> CaseOutcome:
        return CaseOutcome(observed=row["label"], correct=True, latency_ms=0.1)

    service.evaluate_partition(experiment_id, DatasetSplit.REDTEAM, evaluator)
    with pytest.raises(RuntimeError, match="Evaluation is immutable"):
        service.evaluate_partition(experiment_id, DatasetSplit.REDTEAM, evaluator)


def test_trainer_handles_remain_split_safe_while_evaluator_sees_all(tmp_path: Path) -> None:
    workspace, experiment_id = _experiment(tmp_path)
    experiment = ExperimentService(workspace).get(experiment_id)
    datasets = DatasetService(workspace)

    trainer = datasets.trainer_partition_handles(experiment.dataset_id)
    evaluator = datasets.evaluation_partition_handles(experiment.dataset_id)

    assert set(trainer) == {"TRAIN", "DEV"}
    assert set(evaluator) == {"TRAIN", "DEV", "TEST", "REDTEAM"}
