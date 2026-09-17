from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ml_lab.adapters.phase_a import PhaseAResidualAdapter
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetSplit, MetricDirection
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class PhaseAMetricValue:
    metric_id: str
    label: str
    value: float | None
    direction: MetricDirection
    veto: bool
    unit: str = ""
    note: str = ""


@dataclass(frozen=True, slots=True)
class PhaseAProviderReference:
    model: str
    timestamp: str
    source_head: str
    source_dirty: bool
    independent_qa: bool
    corpus_sha256: str
    total_cases: int
    model_calls: int
    metrics: tuple[PhaseAMetricValue, ...]


_LABELS = {
    "resolution_precision": "Resolution precision",
    "useful_resolution_coverage": "Useful resolution coverage",
    "ask_player_accuracy": "ASK_PLAYER accuracy",
    "decision_accuracy": "Decision accuracy",
    "family_slot_accuracy": "Family + slot accuracy",
    "false_commitments": "False commitments",
    "contract_failures": "Contract failures",
    "zero_model_route_violations": "Zero-model route violations",
    "hidden_or_out_of_envelope": "Hidden / envelope escapes",
    "adversarial_failures": "Adversarial failures",
    "latency_ms": "Latency",
    "ram_mb": "RAM",
    "model_size_mb": "Model size",
}

_PERCENT_METRICS = frozenset(
    {
        "resolution_precision",
        "useful_resolution_coverage",
        "ask_player_accuracy",
        "decision_accuracy",
        "family_slot_accuracy",
    }
)


def summarize_phase_a_experiment(
    workspace: Workspace,
    experiment_id: str,
    split: DatasetSplit,
) -> tuple[PhaseAMetricValue, ...]:
    experiment = ExperimentService(workspace).get(experiment_id)
    with workspace.database.connection() as conn:
        rows = conn.execute(
            "SELECT ec.expected_json,ec.observed_json,ec.correct,ec.latency_ms,"
            "f.kind AS failure_kind FROM evaluation_cases ec "
            "LEFT JOIN failures f ON f.id=ec.failure_id "
            "WHERE ec.experiment_id=? AND ec.split=? ORDER BY ec.example_id",
            (experiment_id, split.value),
        ).fetchall()

    total = len(rows)
    has_evidence = total > 0
    emitted_resolve = 0
    expected_resolve = 0
    expected_ask = 0
    correct_resolve = 0
    correct_ask = 0
    decision_matches = 0
    family_slot_matches = 0
    false_commitments = 0
    contract_failures = 0
    envelope_failures = 0
    incorrect = 0
    latency_total = 0.0

    for row in rows:
        expected = _object_json(str(row["expected_json"]))
        observed = _object_json(str(row["observed_json"]))
        proposal = observed.get("proposal")
        proposal_object = proposal if isinstance(proposal, dict) else {}
        expected_decision = expected.get("decision")
        observed_decision = proposal_object.get("decision")
        is_correct = bool(row["correct"])

        expected_resolve += int(expected_decision == "RESOLVE")
        expected_ask += int(expected_decision == "ASK_PLAYER")
        emitted_resolve += int(observed_decision == "RESOLVE")
        decision_matches += int(expected_decision == observed_decision)
        correct_resolve += int(
            expected_decision == "RESOLVE"
            and observed_decision == "RESOLVE"
            and is_correct
        )
        correct_ask += int(
            expected_decision == "ASK_PLAYER"
            and observed_decision == "ASK_PLAYER"
            and is_correct
        )
        family_slot_matches += int(
            expected_decision == "RESOLVE"
            and observed_decision == "RESOLVE"
            and _family_slots_equal(expected, proposal_object)
        )
        failure_kind = row["failure_kind"]
        false_commitments += int(failure_kind == "FALSE_COMMITMENT")
        contract_failures += int(failure_kind == "CONTRACT_FAILURE")
        envelope_failures += int(failure_kind == "HIDDEN_OR_OUT_OF_ENVELOPE")
        incorrect += int(not is_correct)
        latency_total += float(row["latency_ms"])

    model_size_mb: float | None = None
    if experiment.model_artifact_digest:
        model_path = workspace.artifacts.resolve(experiment.model_artifact_digest)
        model_size_mb = model_path.stat().st_size / (1024.0 * 1024.0)

    no_evidence_note = "No protected evaluation evidence for this split"
    values: dict[str, tuple[float | None, str]] = {
        "resolution_precision": (
            correct_resolve / emitted_resolve if emitted_resolve else None,
            (
                f"{correct_resolve}/{emitted_resolve} emitted RESOLVE decisions were correct"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "useful_resolution_coverage": (
            correct_resolve / total if has_evidence else None,
            (
                f"{correct_resolve}/{total} protected cases resolved correctly"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "ask_player_accuracy": (
            correct_ask / expected_ask if expected_ask else None,
            (
                f"{correct_ask}/{expected_ask} expected ASK_PLAYER cases were exact"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "decision_accuracy": (
            decision_matches / total if has_evidence else None,
            (
                f"{decision_matches}/{total} decisions matched"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "family_slot_accuracy": (
            family_slot_matches / expected_resolve if expected_resolve else None,
            (
                f"{family_slot_matches}/{expected_resolve} expected RESOLVE cases "
                "matched family + slots"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "false_commitments": (
            float(false_commitments) if has_evidence else None,
            (
                "Veto: any unsafe or semantically wrong committed RESOLVE"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "contract_failures": (
            float(contract_failures) if has_evidence else None,
            (
                "Veto: proposal rejected by adapter/reference contract"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "zero_model_route_violations": (
            None,
            "Not measured inside the residual-only Lab dataset",
        ),
        "hidden_or_out_of_envelope": (
            float(envelope_failures) if has_evidence else None,
            (
                "Veto: hidden/fact/family/slot envelope escape"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "adversarial_failures": (
            float(incorrect) if split is DatasetSplit.REDTEAM and has_evidence else None,
            (
                "Incorrect REDTEAM cases"
                if split is DatasetSplit.REDTEAM and has_evidence
                else no_evidence_note
                if split is DatasetSplit.REDTEAM
                else "Measured on REDTEAM only"
            ),
        ),
        "latency_ms": (
            latency_total / total if has_evidence else None,
            (
                "Mean persisted case latency, including reference validation"
                if has_evidence
                else no_evidence_note
            ),
        ),
        "ram_mb": (None, "Runtime RAM measurement is not recorded per evaluation"),
        "model_size_mb": (
            model_size_mb,
            (
                "Immutable model artifact size"
                if model_size_mb is not None
                else "No model artifact"
            ),
        ),
    }
    return _ordered_metrics(values)


def load_pinned_provider_reference(
    workspace: Workspace,
    snapshot_id: str,
) -> PhaseAProviderReference | None:
    manifest = ContractSnapshotService(workspace).manifest(snapshot_id)
    digest = _role_digest(manifest, "provider-baseline")
    if digest is None:
        return None
    decoded = json.loads(workspace.artifacts.resolve(digest).read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Pinned provider baseline must be a JSON object.")
    if decoded.get("schema") != "frankenhomie-a5-intake-results/1":
        raise ValueError("Unsupported pinned provider baseline schema.")
    provenance = decoded.get("provenance")
    raw_metrics = decoded.get("metrics")
    raw_rows = decoded.get("rows")
    if not isinstance(provenance, dict) or not isinstance(raw_metrics, dict):
        raise ValueError("Pinned provider baseline is missing provenance or metrics.")
    if not isinstance(raw_rows, list):
        raise ValueError("Pinned provider baseline rows must be a list.")

    provider_rows = [
        row for row in raw_rows if isinstance(row, dict) and row.get("model_called") is True
    ]
    total = len(provider_rows)
    emitted_resolve = sum(row.get("actual") == "RESOLVE" for row in provider_rows)
    expected_resolve = sum(row.get("expected") == "RESOLVE" for row in provider_rows)
    expected_ask = sum(row.get("expected") == "ASK_PLAYER" for row in provider_rows)
    correct_resolve = sum(
        row.get("actual") == "RESOLVE" and row.get("correct") is True
        for row in provider_rows
    )
    correct_ask = sum(
        row.get("expected") == "ASK_PLAYER"
        and row.get("actual") == "ASK_PLAYER"
        and row.get("correct") is True
        for row in provider_rows
    )
    decision_matches = sum(
        row.get("expected") == row.get("actual") for row in provider_rows
    )
    family_slot_matches = sum(
        row.get("expected") == "RESOLVE"
        and row.get("actual") == "RESOLVE"
        and row.get("correct") is True
        for row in provider_rows
    )
    latency = raw_metrics.get("latency_seconds")
    latency_median_ms: float | None = None
    if isinstance(latency, dict) and isinstance(latency.get("median"), (int, float)):
        latency_median_ms = float(latency["median"]) * 1000.0

    false_commitments = _number(raw_metrics.get("false_commitment_count"))
    contract_failures = _number(raw_metrics.get("provider_transport_failures"))
    candidate_escapes = _number(raw_metrics.get("candidate_envelope_violations"))
    fact_escapes = _number(raw_metrics.get("fact_envelope_violations"))
    zero_model = _number(raw_metrics.get("zero_model_violations"))

    values: dict[str, tuple[float | None, str]] = {
        "resolution_precision": (
            correct_resolve / emitted_resolve if emitted_resolve else None,
            f"Historical provider-only rows: {correct_resolve}/{emitted_resolve}",
        ),
        "useful_resolution_coverage": (
            correct_resolve / total if total else None,
            f"Historical provider-only rows: {correct_resolve}/{total}",
        ),
        "ask_player_accuracy": (
            correct_ask / expected_ask if expected_ask else None,
            f"Historical provider-only rows: {correct_ask}/{expected_ask}",
        ),
        "decision_accuracy": (
            decision_matches / total if total else None,
            f"Historical provider-only rows: {decision_matches}/{total}",
        ),
        "family_slot_accuracy": (
            family_slot_matches / expected_resolve if expected_resolve else None,
            f"Historical provider-only RESOLVE rows: {family_slot_matches}/{expected_resolve}",
        ),
        "false_commitments": (
            false_commitments,
            "Historical A5 metric; veto",
        ),
        "contract_failures": (
            contract_failures,
            "Historical provider transport/contract failures; veto",
        ),
        "zero_model_route_violations": (
            zero_model,
            "Historical full-corpus zero-model route violations; veto",
        ),
        "hidden_or_out_of_envelope": (
            _sum_optional(candidate_escapes, fact_escapes),
            "Historical candidate + fact envelope violations; veto",
        ),
        "adversarial_failures": (
            None,
            "Historical A5 result does not expose the Lab REDTEAM partition metric",
        ),
        "latency_ms": (
            latency_median_ms,
            "Historical provider-call median latency",
        ),
        "ram_mb": (None, "Not recorded in the historical A5 result"),
        "model_size_mb": (None, "Not recorded in the historical A5 result"),
    }
    return PhaseAProviderReference(
        model=str(provenance.get("model") or "unknown"),
        timestamp=str(provenance.get("timestamp") or ""),
        source_head=str(provenance.get("head") or ""),
        source_dirty=bool(provenance.get("dirty")),
        independent_qa=bool(provenance.get("independent_qa")),
        corpus_sha256=str(provenance.get("corpus_sha256") or ""),
        total_cases=int(_number(raw_metrics.get("total_cases")) or 0),
        model_calls=int(_number(raw_metrics.get("model_call_count")) or total),
        metrics=_ordered_metrics(values),
    )


def metric_rows_for_ui(metrics: tuple[PhaseAMetricValue, ...]) -> list[dict[str, object]]:
    ordered = sorted(metrics, key=lambda item: (not item.veto, item.metric_id))
    return [
        {
            "metricId": item.metric_id,
            "label": item.label,
            "value": item.value if item.value is not None else 0.0,
            "measured": item.value is not None,
            "direction": item.direction.value,
            "veto": item.veto,
            "unit": item.unit,
            "note": item.note,
            "percent": item.metric_id in _PERCENT_METRICS,
        }
        for item in ordered
    ]


def _ordered_metrics(
    values: dict[str, tuple[float | None, str]],
) -> tuple[PhaseAMetricValue, ...]:
    definitions = PhaseAResidualAdapter().metric_definitions()
    result: list[PhaseAMetricValue] = []
    for definition in definitions:
        value, note = values.get(definition.metric_id, (None, "Not measured"))
        unit = "ms" if definition.metric_id == "latency_ms" else ""
        if definition.metric_id in {"ram_mb", "model_size_mb"}:
            unit = "MB"
        result.append(
            PhaseAMetricValue(
                metric_id=definition.metric_id,
                label=_LABELS.get(definition.metric_id, definition.metric_id),
                value=value,
                direction=definition.direction,
                veto=definition.veto,
                unit=unit,
                note=note,
            )
        )
    return tuple(result)


def _family_slots_equal(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    return (
        expected.get("action_family") == observed.get("action_family")
        and _slot_map(expected.get("slots")) == _slot_map(observed.get("slots"))
    )


def _slot_map(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    result: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        slot_value = item.get("value")
        if isinstance(name, str) and isinstance(slot_value, str):
            result[name] = slot_value
    return result


def _object_json(raw: str) -> dict[str, Any]:
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("Phase A evaluation JSON must contain an object.")
    return {str(key): value for key, value in decoded.items()}


def _role_digest(manifest: dict[str, object], role: str) -> str | None:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        return None
    for raw_file in raw_files:
        if not isinstance(raw_file, dict) or raw_file.get("role") != role:
            continue
        digest = raw_file.get("sha256")
        if isinstance(digest, str) and digest:
            return digest
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sum_optional(*values: float | None) -> float | None:
    measured = [value for value in values if value is not None]
    return sum(measured) if measured else None
