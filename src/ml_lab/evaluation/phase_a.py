from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.adapters.phase_a_reference import (
    ReferencePreflightReceipt,
    ReferenceValidationReceipt,
)
from ml_lab.core.models import FailureSeverity
from ml_lab.evaluation.service import CaseEvaluator, CaseOutcome
from ml_lab.trainers.phase_a_candidate_sparse import (
    PhaseACandidateSparseModel,
    predict_phase_a_candidate_sparse,
)
from ml_lab.trainers.phase_a_sparse import (
    BoundedPredictionError,
    PhaseASparseModel,
    predict_phase_a_sparse,
)

PhaseAPredictor = Callable[[Mapping[str, object]], Mapping[str, object]]
PhaseAReferenceCheck = Callable[
    [Mapping[str, object], Mapping[str, object]],
    ReferenceValidationReceipt,
]
PhaseAReferencePreflight = Callable[
    [Mapping[str, object]],
    ReferencePreflightReceipt,
]

_ENVELOPE_ERRORS = frozenset(
    {
        "ENTITY_OUT_OF_ENVELOPE",
        "FACT_OUT_OF_ENVELOPE",
        "FAMILY_OUT_OF_ENVELOPE",
        "SLOT_OUT_OF_ENVELOPE",
        "MISSING_SLOT_OUT_OF_ENVELOPE",
        "PREBOUND_SLOT_CHANGED",
        "LEGACY_ENTITY_KEYS_FORBIDDEN",
    }
)


def make_phase_a_case_evaluator(
    *,
    predictor: PhaseAPredictor,
    reference_check: PhaseAReferenceCheck,
    reference_preflight: PhaseAReferencePreflight | None = None,
) -> CaseEvaluator:
    """Build an evaluator that keeps Frankenhomie reference validation authoritative."""

    adapter = PhaseAResidualAdapter()

    def evaluate(row: Mapping[str, object]) -> CaseOutcome:
        payload = _required_mapping(row.get("payload"), "payload")
        request = _required_mapping(payload.get("request"), "payload.request")
        expected = _required_mapping(row.get("label"), "label")

        # Fail before any model/provider call if a row escaped the residual-only seam.
        adapter.validate_exported_request(request)
        preflight_summary: dict[str, object] | None = None
        if reference_preflight is not None:
            preflight = reference_preflight(request)
            if preflight.status == "ERROR":
                raise RuntimeError(
                    "Phase A reference preflight infrastructure failed: "
                    f"{preflight.error_code or 'UNKNOWN'}"
                )
            if preflight.status != "MODEL_ALLOWED" or not preflight.model_call_allowed:
                raise RuntimeError(
                    "Pinned Frankenhomie preflight blocked the model call: "
                    f"{preflight.error_code or preflight.status}"
                )
            preflight_summary = _preflight_summary(preflight)

        try:
            proposal = dict(predictor(request))
        except BoundedPredictionError as exc:
            return CaseOutcome(
                observed={
                    "status": "PREDICTION_FAILED_CLOSED",
                    "error": str(exc),
                },
                correct=False,
                latency_ms=-1.0,
                failure_kind="PREDICTION_FAILED_CLOSED",
                failure_severity=FailureSeverity.NON_VETO,
                evidence={"authority_mutation_attempted": False},
            )

        precheck = adapter.validate_proposal(request, proposal)
        if not precheck.accepted:
            error_code = precheck.error_code or "ADAPTER_PRECHECK_REJECTED"
            return _rejected_outcome(
                proposal=proposal,
                error_code=error_code,
                validator="ml-lab.phase-a-adapter-precheck",
                receipt_digest=None,
                preflight=preflight_summary,
            )

        receipt = reference_check(request, proposal)
        if receipt.status == "ERROR":
            raise RuntimeError(
                "Phase A reference validator infrastructure failed: "
                f"{receipt.error_code or 'UNKNOWN'}"
            )
        if receipt.status == "REJECTED":
            return _rejected_outcome(
                proposal=proposal,
                error_code=receipt.error_code or "REFERENCE_REJECTED",
                validator=receipt.validator,
                receipt_digest=receipt.receipt_artifact_digest,
                reference=receipt,
                preflight=preflight_summary,
            )
        if receipt.status != "ACCEPTED":
            raise RuntimeError(f"Unknown Phase A reference status {receipt.status!r}")

        observed = {
            "proposal": proposal,
            "reference": _reference_summary(receipt),
        }
        if preflight_summary is not None:
            observed["preflight"] = preflight_summary
        correct = _semantically_equivalent(expected, proposal)
        if correct:
            return CaseOutcome(
                observed=observed,
                correct=True,
                latency_ms=-1.0,
            )

        false_commitment = proposal.get("decision") == "RESOLVE"
        evidence = _comparison_evidence(expected, proposal)
        evidence["reference"] = _reference_summary(receipt)
        return CaseOutcome(
            observed=observed,
            correct=False,
            latency_ms=-1.0,
            failure_kind=("FALSE_COMMITMENT" if false_commitment else "SEMANTIC_MISMATCH"),
            failure_severity=(
                FailureSeverity.VETO if false_commitment else FailureSeverity.NON_VETO
            ),
            evidence=evidence,
        )

    return evaluate


def make_phase_a_candidate_sparse_evaluator(
    model: PhaseACandidateSparseModel,
    *,
    reference_check: PhaseAReferenceCheck,
    reference_preflight: PhaseAReferencePreflight | None = None,
) -> CaseEvaluator:
    return make_phase_a_case_evaluator(
        predictor=lambda request: predict_phase_a_candidate_sparse(model, request),
        reference_check=reference_check,
        reference_preflight=reference_preflight,
    )


def make_phase_a_sparse_evaluator(
    model: PhaseASparseModel,
    *,
    reference_check: PhaseAReferenceCheck,
    reference_preflight: PhaseAReferencePreflight | None = None,
) -> CaseEvaluator:
    return make_phase_a_case_evaluator(
        predictor=lambda request: predict_phase_a_sparse(model, request),
        reference_check=reference_check,
        reference_preflight=reference_preflight,
    )


def _rejected_outcome(
    *,
    proposal: Mapping[str, object],
    error_code: str,
    validator: str,
    receipt_digest: str | None,
    reference: ReferenceValidationReceipt | None = None,
    preflight: Mapping[str, object] | None = None,
) -> CaseOutcome:
    failure_kind = (
        "HIDDEN_OR_OUT_OF_ENVELOPE"
        if error_code in _ENVELOPE_ERRORS
        else "UNSUPPORTED_MECHANICS_AUTHORITY"
        if error_code == "UNSUPPORTED_AUTHORITY_FIELD"
        else "FALSE_COMMITMENT"
        if error_code == "DETERMINISTIC_VETO"
        else "CONTRACT_FAILURE"
    )
    reference_summary: dict[str, object] = {
        "status": "REJECTED",
        "validator": validator,
        "error_code": error_code,
        "receipt_artifact_digest": receipt_digest,
    }
    if reference is not None:
        reference_summary = _reference_summary(reference)
    observed: dict[str, object] = {
        "proposal": dict(proposal),
        "reference": reference_summary,
    }
    if preflight is not None:
        observed["preflight"] = dict(preflight)
    return CaseOutcome(
        observed=observed,
        correct=False,
        latency_ms=-1.0,
        failure_kind=failure_kind,
        failure_severity=FailureSeverity.VETO,
        evidence={
            "validator": validator,
            "error_code": error_code,
            "receipt_artifact_digest": receipt_digest,
        },
    )


def _semantically_equivalent(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> bool:
    expected_decision = expected.get("decision")
    observed_decision = observed.get("decision")
    if expected_decision != observed_decision:
        return False
    if expected_decision == "RESOLVE":
        return (
            expected.get("action_family") == observed.get("action_family")
            and _slot_map(expected.get("slots")) == _slot_map(observed.get("slots"))
        )
    if expected_decision == "ASK_PLAYER":
        return (
            expected.get("action_family") == observed.get("action_family")
            and _slot_map(expected.get("slots")) == _slot_map(observed.get("slots"))
            and _string_set(expected.get("missing_slots"))
            == _string_set(observed.get("missing_slots"))
            and _string_set(expected.get("candidate_keys"))
            == _string_set(observed.get("candidate_keys"))
        )
    return False


def _comparison_evidence(
    expected: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    expected_decision = expected.get("decision")
    observed_decision = observed.get("decision")
    return {
        "decision_correct": expected_decision == observed_decision,
        "family_slot_correct": (
            expected.get("action_family") == observed.get("action_family")
            and _slot_map(expected.get("slots")) == _slot_map(observed.get("slots"))
        ),
        "ask_player_correct": (
            expected_decision == "ASK_PLAYER" and observed_decision == "ASK_PLAYER"
        ),
        "false_commitment": observed_decision == "RESOLVE",
    }


def _preflight_summary(receipt: ReferencePreflightReceipt) -> dict[str, object]:
    return {
        "status": receipt.status,
        "commit_sha": receipt.commit_sha,
        "contract_version": receipt.contract_version,
        "validator": receipt.validator,
        "fresh_process": receipt.fresh_process,
        "model_call_allowed": receipt.model_call_allowed,
        "authority_mutation_allowed": receipt.authority_mutation_allowed,
        "error_code": receipt.error_code,
        "receipt_artifact_digest": receipt.receipt_artifact_digest,
    }


def _reference_summary(receipt: ReferenceValidationReceipt) -> dict[str, object]:
    return {
        "status": receipt.status,
        "commit_sha": receipt.commit_sha,
        "contract_version": receipt.contract_version,
        "validator": receipt.validator,
        "fresh_process": receipt.fresh_process,
        "authority_mutation_allowed": receipt.authority_mutation_allowed,
        "error_code": receipt.error_code,
        "receipt_artifact_digest": receipt.receipt_artifact_digest,
    }


def _required_mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _slot_map(value: object) -> dict[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        slot_value = item.get("value")
        if isinstance(name, str) and isinstance(slot_value, str):
            result[name] = slot_value
    return result


def _string_set(value: object) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))
