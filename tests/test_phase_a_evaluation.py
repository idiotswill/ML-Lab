import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION
from ml_lab.adapters.phase_a_reference import ReferenceValidationReceipt
from ml_lab.core.models import DatasetSplit, FailureSeverity
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.phase_a import make_phase_a_case_evaluator
from ml_lab.evaluation.service import EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


def _request(
    declaration: str,
    family: str,
    slot_name: str,
    allowed_values: list[str],
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "failed_deterministic_stage": "IDENTITY:AMBIGUOUS",
        "assessment": {
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "IDENTITY:AMBIGUOUS",
            "declaration": declaration,
        },
        "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        "allowed_action_families": [family],
        "family_slots": [
            {
                "family": family,
                "slots": [
                    {
                        "name": slot_name,
                        "required": True,
                        "allowed_values": allowed_values,
                        "prebound": None,
                    }
                ],
            }
        ],
    }


def _resolve(family: str, slot_name: str, value: str) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": [{"name": slot_name, "value": value}],
        "reason_code": "TEST",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _receipt(
    *,
    status: str = "ACCEPTED",
    error_code: str | None = None,
) -> ReferenceValidationReceipt:
    return ReferenceValidationReceipt(
        status=status,
        commit_sha="a" * 40,
        contract_version=PHASE_A_CONTRACT_VERSION,
        request_sha256="b" * 64,
        proposal_sha256="c" * 64,
        validator="frankenhomie.run_residual_semantics",
        fresh_process=True,
        authority_mutation_allowed=False,
        error_code=error_code,
        error_message=None,
        receipt_artifact_digest="d" * 64,
    )


def _row(
    request: dict[str, object],
    expected: dict[str, object],
) -> dict[str, object]:
    return {"payload": {"request": request}, "label": expected}


def test_wrong_bounded_resolve_is_veto_level_false_commitment() -> None:
    request = _request(
        "I open the bronze hatch",
        "INTERACT_OBJECT",
        "OPERATION",
        ["OPEN", "CLOSE"],
    )
    expected = _resolve("INTERACT_OBJECT", "OPERATION", "OPEN")

    evaluator = make_phase_a_case_evaluator(
        predictor=lambda _request: _resolve("INTERACT_OBJECT", "OPERATION", "CLOSE"),
        reference_check=lambda _request, _proposal: _receipt(),
    )
    outcome = evaluator(_row(request, expected))

    assert outcome.correct is False
    assert outcome.failure_kind == "FALSE_COMMITMENT"
    assert outcome.failure_severity is FailureSeverity.VETO
    assert isinstance(outcome.evidence, Mapping)
    assert outcome.evidence["false_commitment"] is True


def test_adapter_rejection_never_reaches_reference_validator() -> None:
    request = _request(
        "I strike Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    expected = _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara")
    called = False

    def reference_check(
        _request: Mapping[str, object],
        _proposal: Mapping[str, object],
    ) -> ReferenceValidationReceipt:
        nonlocal called
        called = True
        return _receipt()

    evaluator = make_phase_a_case_evaluator(
        predictor=lambda _request: _resolve(
            "HARM_TARGET",
            "TARGET_COMBATANT",
            "combatant:hidden",
        ),
        reference_check=reference_check,
    )
    outcome = evaluator(_row(request, expected))

    assert called is False
    assert outcome.correct is False
    assert outcome.failure_kind == "HIDDEN_OR_OUT_OF_ENVELOPE"
    assert outcome.failure_severity is FailureSeverity.VETO


def test_reference_rejection_is_veto_but_reference_error_aborts_evaluation() -> None:
    request = _request(
        "I strike Mara",
        "HARM_TARGET",
        "TARGET_COMBATANT",
        ["combatant:mara"],
    )
    expected = _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara")
    proposal = _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara")

    rejected = make_phase_a_case_evaluator(
        predictor=lambda _request: proposal,
        reference_check=lambda _request, _proposal: _receipt(
            status="REJECTED",
            error_code="DETERMINISTIC_VETO",
        ),
    )(_row(request, expected))
    assert rejected.failure_kind == "FALSE_COMMITMENT"
    assert rejected.failure_severity is FailureSeverity.VETO

    broken = make_phase_a_case_evaluator(
        predictor=lambda _request: proposal,
        reference_check=lambda _request, _proposal: _receipt(
            status="ERROR",
            error_code="REFERENCE_IMPORT_DEPENDENCY_MISSING",
        ),
    )
    with pytest.raises(RuntimeError, match="reference validator infrastructure failed"):
        broken(_row(request, expected))


def test_phase_a_evidence_persists_through_generic_evaluation_service(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A comparison")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "phase-a-frozen")

    train_request = _request(
        "I travel toward the eastern camp",
        "MOVE_TRAVEL",
        "DESTINATION",
        ["location:east-camp"],
    )
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="train-move",
            split=DatasetSplit.TRAIN,
            source_id="synthetic:train-move",
            lineage_group="lineage:train-move",
            payload={"request": train_request},
            label=_resolve("MOVE_TRAVEL", "DESTINATION", "location:east-camp"),
            tags=("synthetic",),
        ),
    )

    test_request = _request(
        "I open the bronze hatch beside the altar",
        "INTERACT_OBJECT",
        "OPERATION",
        ["OPEN", "CLOSE"],
    )
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="test-hatch",
            split=DatasetSplit.TEST,
            source_id="synthetic:test-hatch",
            lineage_group="lineage:test-hatch",
            payload={"request": test_request},
            label=_resolve("INTERACT_OBJECT", "OPERATION", "OPEN"),
            tags=("synthetic",),
        ),
    )
    datasets.freeze(dataset.id)

    experiment = ExperimentService(workspace).create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id="phase-a-comparison-test",
        runtime_pack_id="builtin-core",
    )
    evaluator = make_phase_a_case_evaluator(
        predictor=lambda _request: _resolve("INTERACT_OBJECT", "OPERATION", "CLOSE"),
        reference_check=lambda _request, _proposal: _receipt(),
    )

    service = EvaluationService(workspace)
    summary = service.evaluate_partition(experiment.id, DatasetSplit.TEST, evaluator)
    assert summary.total == 1
    assert summary.correct == 0
    assert summary.veto_failures == 1

    case = service.page_cases(experiment.id, DatasetSplit.TEST)[0]
    observed = json.loads(case.observed_json)
    assert observed["proposal"]["slots"] == [{"name": "OPERATION", "value": "CLOSE"}]
    assert observed["reference"]["validator"] == "frankenhomie.run_residual_semantics"
    assert observed["reference"]["receipt_artifact_digest"] == "d" * 64

    failure = service.failures.page_for_project(project.id)[0]
    assert failure.kind == "FALSE_COMMITMENT"
    assert failure.severity is FailureSeverity.VETO
