from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_CONTRACT_VERSION,
)
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, ExperimentStatus
from ml_lab.evaluation.generic import make_sparse_classification_evaluator
from ml_lab.evaluation.phase_a import make_phase_a_sparse_evaluator
from ml_lab.evaluation.service import CaseEvaluator, EvaluationService, EvaluationSummary
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.phase_a_sparse import PhaseASparseModel
from ml_lab.trainers.service import (
    PHASE_A_RUNTIME_PACK_ID,
    PHASE_A_TRAINER_ID,
    SPARSE_RUNTIME_PACK_ID,
    SPARSE_TRAINER_ID,
)
from ml_lab.trainers.sparse_nb import SparseNBModel


def evaluate_packaged_experiment(
    workspace: Workspace,
    experiment_id: str,
    *,
    splits: Sequence[DatasetSplit] = (DatasetSplit.TEST, DatasetSplit.REDTEAM),
    cancelled: Callable[[], bool] | None = None,
) -> tuple[EvaluationSummary, ...]:
    experiment = ExperimentService(workspace).get(experiment_id)
    if experiment.trainer_id == SPARSE_TRAINER_ID:
        return evaluate_generic_sparse_experiment(
            workspace,
            experiment_id,
            splits=splits,
            cancelled=cancelled,
        )
    if experiment.trainer_id == PHASE_A_TRAINER_ID:
        return evaluate_phase_a_sparse_experiment(
            workspace,
            experiment_id,
            splits=splits,
            cancelled=cancelled,
        )
    raise ValueError(f"No packaged evaluator for trainer {experiment.trainer_id!r}.")


def evaluate_generic_sparse_experiment(
    workspace: Workspace,
    experiment_id: str,
    *,
    splits: Sequence[DatasetSplit] = (DatasetSplit.TEST, DatasetSplit.REDTEAM),
    cancelled: Callable[[], bool] | None = None,
) -> tuple[EvaluationSummary, ...]:
    """Evaluate a completed generic sparse model on protected frozen partitions."""

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
    return _evaluate_protected_partitions(
        workspace,
        experiment_id,
        experiment.dataset_id,
        evaluator,
        splits=splits,
        cancelled=cancelled,
    )


def evaluate_phase_a_sparse_experiment(
    workspace: Workspace,
    experiment_id: str,
    *,
    splits: Sequence[DatasetSplit] = (DatasetSplit.TEST, DatasetSplit.REDTEAM),
    cancelled: Callable[[], bool] | None = None,
) -> tuple[EvaluationSummary, ...]:
    """Evaluate a bounded Phase A scorer against the pinned real Frankenhomie validator."""

    experiments = ExperimentService(workspace)
    experiment = experiments.get(experiment_id)
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError("Protected evaluation requires a completed experiment.")
    if experiment.trainer_id != PHASE_A_TRAINER_ID:
        raise ValueError("No packaged Phase A evaluator for this trainer.")
    if experiment.runtime_pack_id != PHASE_A_RUNTIME_PACK_ID:
        raise ValueError("Phase A runtime is incompatible with this evaluator.")
    if experiment.model_artifact_digest is None:
        raise RuntimeError("Completed Phase A experiment has no model artifact.")
    if experiment.contract_snapshot_id is None:
        raise RuntimeError("Phase A evaluation requires a pinned contract snapshot.")

    snapshot = ContractSnapshotService(workspace).get(experiment.contract_snapshot_id)
    if snapshot.project_id != experiment.project_id:
        raise ValueError("Pinned contract snapshot belongs to a different project.")
    if snapshot.adapter_id != PHASE_A_ADAPTER_ID:
        raise ValueError("Pinned contract snapshot is not a Phase A residual contract.")
    if snapshot.contract_version != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Pinned Phase A contract version is incompatible with this evaluator.")

    repository = Path(snapshot.repo_path)
    model = PhaseASparseModel.load(
        workspace.artifacts.resolve(experiment.model_artifact_digest)
    )
    reference = PhaseAReferenceValidator(workspace)
    evaluator = make_phase_a_sparse_evaluator(
        model,
        reference_check=lambda request, proposal: reference.validate(
            repository=repository,
            ref=snapshot.commit_sha,
            request=request,
            proposal=proposal,
        ),
        reference_preflight=lambda request: reference.preflight(
            repository=repository,
            ref=snapshot.commit_sha,
            request=request,
        ),
    )
    return _evaluate_protected_partitions(
        workspace,
        experiment_id,
        experiment.dataset_id,
        evaluator,
        splits=splits,
        cancelled=cancelled,
    )


def _evaluate_protected_partitions(
    workspace: Workspace,
    experiment_id: str,
    dataset_id: str,
    evaluator: CaseEvaluator,
    *,
    splits: Sequence[DatasetSplit],
    cancelled: Callable[[], bool] | None,
) -> tuple[EvaluationSummary, ...]:
    service = EvaluationService(workspace)
    partitions = {
        item.split: item.example_count for item in service.datasets.partitions(dataset_id)
    }
    summaries: list[EvaluationSummary] = []
    seen: set[DatasetSplit] = set()
    for split in splits:
        if split in seen:
            continue
        seen.add(split)
        if partitions.get(split, 0) <= 0:
            continue
        if cancelled is not None and cancelled():
            raise InterruptedError("Evaluation cancellation requested")
        progress = service.progress(experiment_id, split)
        if progress.complete:
            summaries.append(service.summary(experiment_id, split))
            continue
        summaries.append(
            service.evaluate_partition(
                experiment_id,
                split,
                evaluator,
                cancelled=cancelled,
            )
        )
    if not summaries:
        raise RuntimeError("Dataset has no protected TEST/REDTEAM examples to evaluate.")
    return tuple(summaries)
