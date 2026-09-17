from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ml_lab.core.models import DatasetSplit, ExperimentStatus, FailureSeverity
from ml_lab.datasets.service import DatasetService
from ml_lab.evaluation.generic import make_sparse_classification_evaluator
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.redteam.service import RedTeamCase, RedTeamMutator, RedTeamService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID
from ml_lab.trainers.sparse_nb import SparseNBModel


GENERIC_REDTEAM_SUITE_ID = "generic.text-perturbation.v1"


@dataclass(frozen=True, slots=True)
class GenericRedTeamSummary:
    run_id: str
    base_cases: int
    generated_cases: int
    failures: int


def run_generic_sparse_redteam(
    workspace: Workspace,
    experiment_id: str,
    *,
    seed: int,
    max_base_cases: int = 1000,
    cancelled: Callable[[], bool] | None = None,
) -> GenericRedTeamSummary:
    """Run the packaged generic text perturbation suite against a completed model.

    The runner consumes only the immutable REDTEAM partition and the completed model
    artifact. It cannot mutate a project, dataset, or Frankenhomie state. Generated
    misses are preserved as ordinary immutable failure records linked to the red-team run.
    The run itself remains RUNNING until every generated case has been scored.
    """
    if not 1 <= max_base_cases <= 5000:
        raise ValueError("max_base_cases must be between 1 and 5000")

    experiment = ExperimentService(workspace).get(experiment_id)
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError("Red-team runs require a completed experiment.")
    if experiment.trainer_id != SPARSE_TRAINER_ID:
        raise ValueError("No packaged generic red-team evaluator for this trainer.")
    if experiment.runtime_pack_id != SPARSE_RUNTIME_PACK_ID:
        raise ValueError("Sparse experiment runtime is incompatible with this red-team suite.")
    if experiment.model_artifact_digest is None:
        raise RuntimeError("Completed sparse experiment has no model artifact.")

    model = SparseNBModel.load(workspace.artifacts.resolve(experiment.model_artifact_digest))
    evaluator = make_sparse_classification_evaluator(model)
    datasets = DatasetService(workspace)
    handles = datasets.evaluation_partition_handles(experiment.dataset_id)
    try:
        redteam_path = workspace.artifacts.resolve(handles[DatasetSplit.REDTEAM.value])
    except KeyError as exc:
        raise RuntimeError("Frozen dataset is missing a REDTEAM partition.") from exc

    base_cases: list[RedTeamCase] = []
    with redteam_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if cancelled is not None and cancelled():
                raise InterruptedError("Red-team cancellation requested")
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError("REDTEAM partition row must be an object.")
            example_id = row.get("example_id")
            payload = row.get("payload")
            label = row.get("label")
            tags = row.get("tags", [])
            if not isinstance(example_id, str) or not example_id:
                raise ValueError("REDTEAM row example_id must be a non-empty string.")
            if not isinstance(payload, Mapping) or not isinstance(label, Mapping):
                raise ValueError("REDTEAM rows require payload and label objects.")
            clean_tags = tuple(str(item) for item in tags) if isinstance(tags, list) else ()
            base_cases.append(
                RedTeamCase(
                    case_id=example_id,
                    payload=dict(payload),
                    expected=dict(label),
                    tags=clean_tags,
                )
            )
            if len(base_cases) >= max_base_cases:
                break

    if not base_cases:
        raise RuntimeError("REDTEAM partition contains no runnable examples.")

    mutators = _text_mutators(model.text_key)
    runs = RedTeamService(workspace)
    run = runs.start_run(
        project_id=experiment.project_id,
        dataset_id=experiment.dataset_id,
        experiment_id=experiment.id,
        seed=seed,
        mutator_version=GENERIC_REDTEAM_SUITE_ID,
    )
    generated = runs.generate_cases(
        seed=seed,
        base_cases=base_cases,
        mutators=mutators,
    )

    failures = FailureService(workspace)
    failure_count = 0
    try:
        for case in generated:
            if cancelled is not None and cancelled():
                raise InterruptedError("Red-team cancellation requested during scoring")
            payload = case.get("payload")
            expected = case.get("expected")
            if not isinstance(payload, Mapping) or not isinstance(expected, Mapping):
                raise ValueError("Generated red-team case has invalid payload/expected shape.")
            outcome = evaluator({"payload": payload, "label": expected})
            if outcome.correct:
                continue
            failure_count += 1
            evidence: dict[str, object] = {
                "suite_id": GENERIC_REDTEAM_SUITE_ID,
                "base_case_id": str(case.get("base_case_id", "")),
                "mutator_id": str(case.get("mutator_id", "")),
                "derived_seed": int(case.get("derived_seed", 0)),
                "authority_mutation_attempted": False,
            }
            if outcome.evidence is not None:
                evidence.update(dict(outcome.evidence))
            failures.record(
                project_id=experiment.project_id,
                experiment_id=experiment.id,
                dataset_id=experiment.dataset_id,
                redteam_run_id=run.id,
                example_id=str(case.get("case_id", "")),
                split=DatasetSplit.REDTEAM,
                kind=outcome.failure_kind or "REDTEAM_MISMATCH",
                severity=outcome.failure_severity or FailureSeverity.NON_VETO,
                expected=expected,
                observed=outcome.observed,
                evidence=evidence,
            )
    except InterruptedError:
        runs.interrupt_run(run.id)
        raise
    except Exception:
        runs.fail_run(run.id)
        raise

    runs.complete_run(
        run.id,
        generated_cases=generated,
        mutator_ids=[mutator.mutator_id for mutator in mutators],
        metadata={
            "suite_id": GENERIC_REDTEAM_SUITE_ID,
            "base_case_limit": max_base_cases,
            "protected_split": DatasetSplit.REDTEAM.value,
            "scored_cases": len(generated),
            "failure_count": failure_count,
        },
    )
    return GenericRedTeamSummary(
        run_id=run.id,
        base_cases=len(base_cases),
        generated_cases=len(generated),
        failures=failure_count,
    )


def _text_mutators(text_key: str) -> tuple[RedTeamMutator, ...]:
    def mutate_case(payload: object, _rng: random.Random) -> object:
        return _mutate_text(payload, text_key, lambda text: text.swapcase())

    def mutate_spacing(payload: object, rng: random.Random) -> object:
        gap = " " * rng.randint(2, 5)
        return _mutate_text(payload, text_key, lambda text: gap.join(text.split()))

    def mutate_punctuation(payload: object, rng: random.Random) -> object:
        suffix = rng.choice(("?", "!", "...", "?!", "!!"))
        return _mutate_text(payload, text_key, lambda text: f"{text}{suffix}")

    return (
        RedTeamMutator("case-perturbation", mutate_case),
        RedTeamMutator("spacing-noise", mutate_spacing),
        RedTeamMutator("punctuation-noise", mutate_punctuation),
    )


def _mutate_text(
    payload: object,
    text_key: str,
    transform: Callable[[str], str],
) -> object:
    if not isinstance(payload, Mapping):
        raise ValueError("Generic red-team payload must be an object.")
    text = payload.get(text_key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Generic red-team payload.{text_key} must be a non-empty string.")
    return {**dict(payload), text_key: transform(text)}
