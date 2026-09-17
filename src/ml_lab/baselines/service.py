from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ml_lab.core.models import (
    DatasetSplit,
    ExperimentRecord,
    ExperimentStatus,
    MetricDirection,
    MetricValue,
)
from ml_lab.evaluation.service import CaseEvaluator, EvaluationService, EvaluationSummary
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class BaselineRunResult:
    experiment: ExperimentRecord
    summaries: tuple[EvaluationSummary, ...]


class BaselineService:
    """Run deterministic/reference baselines as ordinary immutable experiments.

    The baseline implementation is adapter-owned through ``CaseEvaluator``. The host
    owns state transitions, immutable case evidence, failure records, metrics, and
    comparison. Baselines receive evaluation partitions only; they never gain trainer
    access or Frankenhomie mutation authority.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.experiments = ExperimentService(workspace)
        self.evaluation = EvaluationService(workspace)

    def create(
        self,
        *,
        project_id: str,
        dataset_id: str,
        baseline_id: str,
        contract_snapshot_id: str | None = None,
        config: Mapping[str, object] | None = None,
        environment: Mapping[str, object] | None = None,
        seed: int = 0,
    ) -> ExperimentRecord:
        clean_id = baseline_id.strip()
        if not clean_id:
            raise ValueError("baseline_id is required")
        return self.experiments.create(
            project_id=project_id,
            dataset_id=dataset_id,
            trainer_id=f"baseline:{clean_id}",
            runtime_pack_id="baseline-host-v1",
            config={"baseline_id": clean_id, **dict(config or {})},
            environment=environment,
            seed=seed,
            contract_snapshot_id=contract_snapshot_id,
        )

    def run(
        self,
        experiment_id: str,
        *,
        evaluator: CaseEvaluator,
        splits: Sequence[DatasetSplit] = (DatasetSplit.TEST, DatasetSplit.REDTEAM),
    ) -> BaselineRunResult:
        experiment = self.experiments.get(experiment_id)
        if not experiment.trainer_id.startswith("baseline:"):
            raise ValueError("BaselineService requires a baseline experiment.")
        if experiment.status is not ExperimentStatus.QUEUED:
            raise RuntimeError(
                f"Cannot run baseline from {experiment.status.value} state."
            )
        ordered_splits = _unique_splits(splits)
        if not ordered_splits:
            raise ValueError("At least one evaluation split is required.")

        self.experiments.start(experiment_id)
        summaries: list[EvaluationSummary] = []
        try:
            for split in ordered_splits:
                handles = self.evaluation.datasets.evaluation_partition_handles(
                    experiment.dataset_id
                )
                if split.value not in handles:
                    continue
                partition = next(
                    (
                        item
                        for item in self.evaluation.datasets.partitions(
                            experiment.dataset_id
                        )
                        if item.split is split
                    ),
                    None,
                )
                if partition is None or partition.example_count == 0:
                    continue
                summaries.append(
                    self.evaluation.evaluate_partition(
                        experiment_id,
                        split,
                        evaluator,
                    )
                )
            metrics = _summary_metrics(summaries)
            completed = self.experiments.complete(
                experiment_id,
                metrics=metrics,
                model_artifact_digest=None,
            )
        except Exception:
            current = self.experiments.get(experiment_id)
            if current.status is ExperimentStatus.RUNNING:
                self.experiments.finish_without_success(
                    experiment_id,
                    ExperimentStatus.FAILED,
                )
            raise
        return BaselineRunResult(completed, tuple(summaries))


def _unique_splits(splits: Sequence[DatasetSplit]) -> tuple[DatasetSplit, ...]:
    seen: set[DatasetSplit] = set()
    ordered: list[DatasetSplit] = []
    for split in splits:
        if split in seen:
            continue
        seen.add(split)
        ordered.append(split)
    return tuple(ordered)


def _summary_metrics(summaries: Sequence[EvaluationSummary]) -> list[MetricValue]:
    metrics: list[MetricValue] = []
    total_cases = 0
    total_correct = 0
    total_vetoes = 0
    for summary in summaries:
        prefix = summary.split.value.casefold()
        total_cases += summary.total
        total_correct += summary.correct
        total_vetoes += summary.veto_failures
        metrics.extend(
            (
                MetricValue(
                    metric_id=f"{prefix}.accuracy",
                    value=summary.accuracy,
                    direction=MetricDirection.HIGHER,
                    veto=False,
                ),
                MetricValue(
                    metric_id=f"{prefix}.veto_failures",
                    value=float(summary.veto_failures),
                    direction=MetricDirection.ZERO,
                    veto=True,
                ),
                MetricValue(
                    metric_id=f"{prefix}.mean_latency_ms",
                    value=summary.mean_latency_ms,
                    direction=MetricDirection.LOWER,
                    veto=False,
                ),
                MetricValue(
                    metric_id=f"{prefix}.p95_latency_ms",
                    value=summary.p95_latency_ms,
                    direction=MetricDirection.LOWER,
                    veto=False,
                ),
            )
        )
    if total_cases:
        metrics.extend(
            (
                MetricValue(
                    metric_id="overall.accuracy",
                    value=total_correct / total_cases,
                    direction=MetricDirection.HIGHER,
                    veto=False,
                ),
                MetricValue(
                    metric_id="overall.veto_failures",
                    value=float(total_vetoes),
                    direction=MetricDirection.ZERO,
                    veto=True,
                ),
            )
        )
    return metrics
