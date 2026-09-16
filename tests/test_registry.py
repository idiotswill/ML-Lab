from pathlib import Path

import pytest

from ml_lab.core.models import (
    DatasetSplit,
    FailureSeverity,
    MetricDirection,
    MetricValue,
    ModelStage,
)
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace


def _completed_model_experiment(
    tmp_path: Path,
    *,
    veto_value: float = 0.0,
) -> tuple[Workspace, str, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Registry tests")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "frozen")
    texts = {
        DatasetSplit.TRAIN: "north wind over red canyon",
        DatasetSplit.DEV: "glass compass under cedar table",
        DatasetSplit.TEST: "gold bell beside quiet river",
        DatasetSplit.REDTEAM: "do not open violet gate",
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
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id="sparse-v1",
        runtime_pack_id="builtin-sparse",
        seed=11,
    )
    experiments.start(experiment.id)
    model_artifact = workspace.artifacts.commit_bytes(b"model")
    experiments.complete(
        experiment.id,
        model_artifact_digest=model_artifact.digest,
        metrics=(
            MetricValue("precision", 0.99, MetricDirection.HIGHER, False),
            MetricValue("false_commitments", veto_value, MetricDirection.ZERO, True),
        ),
    )
    return workspace, project.id, experiment.id


def test_model_promotion_is_sequential_and_stops_before_integration(tmp_path: Path) -> None:
    workspace, project_id, experiment_id = _completed_model_experiment(tmp_path)
    registry = ModelRegistryService(workspace)
    model = registry.register_from_experiment(
        experiment_id,
        compatibility={"adapter": "test", "contract": "v1"},
    )
    assert model.stage is ModelStage.EXPERIMENT
    model = registry.promote(model.id, ModelStage.SHADOW)
    model = registry.promote(model.id, ModelStage.ADVISORY)
    model = registry.promote(model.id, ModelStage.RELEASE_CANDIDATE)
    assert model.stage is ModelStage.RELEASE_CANDIDATE
    assert len(registry.stage_history(model.id)) == 4
    assert len(registry.list_for_project(project_id)) == 1

    with pytest.raises(PermissionError, match="cannot be granted"):
        registry.promote(model.id, ModelStage.INTEGRATION_APPROVED)


def test_nonzero_veto_metric_blocks_release_candidate(tmp_path: Path) -> None:
    workspace, _, experiment_id = _completed_model_experiment(tmp_path, veto_value=1.0)
    registry = ModelRegistryService(workspace)
    model = registry.register_from_experiment(experiment_id)
    model = registry.promote(model.id, ModelStage.SHADOW)
    model = registry.promote(model.id, ModelStage.ADVISORY)
    with pytest.raises(RuntimeError, match="veto metric"):
        registry.promote(model.id, ModelStage.RELEASE_CANDIDATE)


def test_failure_regression_membership_preserves_historical_payload(tmp_path: Path) -> None:
    workspace, project_id, experiment_id = _completed_model_experiment(tmp_path)
    failures = FailureService(workspace)
    failure = failures.record(
        project_id=project_id,
        experiment_id=experiment_id,
        kind="FALSE_COMMITMENT",
        severity=FailureSeverity.VETO,
        expected={"decision": "NO_ACTION"},
        observed={"decision": "RESOLVE"},
        split=DatasetSplit.REDTEAM,
        evidence={"seed": 42},
    )
    before = failures.immutable_payload(failure.id)
    failures.promote_to_regression(failure.id, suite_name="phase-a-safety")
    after = failures.immutable_payload(failure.id)
    assert after == before
    cases = failures.regression_cases(suite_name="phase-a-safety")
    assert [case.id for case in cases] == [failure.id]

    registry = ModelRegistryService(workspace)
    model = registry.register_from_experiment(experiment_id)
    model = registry.promote(model.id, ModelStage.SHADOW)
    model = registry.promote(model.id, ModelStage.ADVISORY)
    with pytest.raises(RuntimeError, match="veto failure"):
        registry.promote(model.id, ModelStage.RELEASE_CANDIDATE)
