from __future__ import annotations

from collections.abc import Mapping

from ml_lab.core.models import FailureSeverity
from ml_lab.evaluation.service import CaseEvaluator, CaseOutcome
from ml_lab.trainers.sparse_nb import SparseNBModel


def make_sparse_classification_evaluator(model: SparseNBModel) -> CaseEvaluator:
    """Build the generic sparse-model evaluator used by the Compare surface.

    The evaluator only classifies immutable exported examples. It has no workspace,
    rules, state-mutation, or authority handle.
    """

    def evaluate(row: Mapping[str, object]) -> CaseOutcome:
        payload = row.get("payload")
        label = row.get("label")
        if not isinstance(payload, Mapping) or not isinstance(label, Mapping):
            raise ValueError("Generic evaluation rows require payload and label objects")

        text = payload.get(model.text_key)
        expected = label.get(model.label_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Evaluation row payload.{model.text_key} must be a non-empty string"
            )
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(
                f"Evaluation row label.{model.label_key} must be a non-empty string"
            )

        predicted, scores = model.predict(text)
        observed = {
            "prediction": predicted,
            "expected": expected,
            "scores": scores,
            "text_key": model.text_key,
            "label_key": model.label_key,
        }
        if predicted == expected:
            return CaseOutcome(observed=observed, correct=True, latency_ms=-1.0)
        return CaseOutcome(
            observed=observed,
            correct=False,
            latency_ms=-1.0,
            failure_kind="CLASSIFICATION_MISMATCH",
            failure_severity=FailureSeverity.NON_VETO,
            evidence={
                "predicted": predicted,
                "expected": expected,
                "authority_mutation_attempted": False,
            },
        )

    return evaluate
