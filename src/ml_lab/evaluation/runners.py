from __future__ import annotations

from collections.abc import Sequence

from ml_lab.core.models import DatasetSplit, ExperimentStatus
from ml_lab.evaluation.generic import make_sparse_classification_evaluator
from ml_lab.evaluation.service import EvaluationService, EvaluationSummary
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID
from ml_lab.trainers.sparse_nb import SparseNBModel


def evaluate_generic_sparse_experiment(
    workspace: Workspace,
    experiment_id: str,
    *,
    splits: Sequence[DatasetSplit] = (DatasetSplit.TEST, DatasetSplit.REDTEAM),
) -> tuple[EvaluationSummary, ...]:
    """Evaluate a completed generic sparse model on protected frozen partitions.

    Training artifacts remain immutable. Evaluation appends one immutable case ledger
    per protected split and refuses to overwrite any existing partition evidence.
    """

    experiments = ExperimentService(workspace)
    experiment = experiments.get(experiment_id)
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError("Protected evaluation requires a completed experiment.")
    if experiment.trainer_id != SPARSE_TRAINER_ID:
        raise ValueError("No packaged generic evaluator for this trainer.")
    if experiment.runtime_pack_id != SPARSE_RUNTIME_PACK_ID:
        raise ValueError("Sparse experiment runtime is incompatible with this evaluator.")
    if experiment.model_artifact_digest is None:
        raise RuntimeError("Completed sparse experiment has no model artifact.")

    model = SparseNBModel.load(workspace.artifacts.resolve(experiment.model_artifact_digest))
    evaluator = make_sparse_classification_evaluator(model)
    service = EvaluationService(workspace)
    partitions = {
        item.split: item.example_count
        for item in service.datasets.partitions(experiment.dataset_id)
    }

    summaries: list[EvaluationSummary] = []
    seen: set[DatasetSplit] = set()
    for split in splits:
        if split in seen:
            continue
        seen.add(split)
        if partitions.get(split, 0) <= 0:
            continue
        existing = service.summary(experiment.id, split)
        if existing.total:
            summaries.append(existing)
            continue
        summaries.append(service.evaluate_partition(experiment.id, split, evaluator))
    if not summaries:
        raise RuntimeError("Dataset has no protected TEST/REDTEAM examples to evaluate.")
    return tuple(summaries)
