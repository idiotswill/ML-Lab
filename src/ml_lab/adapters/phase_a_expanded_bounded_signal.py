from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ml_lab.adapters.phase_a_expanded_bakeoff import (
    _failure_rows,
    _mapping,
    _required_text,
    _validate_expanded_freeze_receipt,
)
from ml_lab.baselines.phase_a_bounded_provider_signal import (
    run_phase_a_bounded_provider_signal_dev,
)
from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.leakage import canonical_json
from ml_lab.evaluation.phase_a_metrics import summarize_phase_a_experiment
from ml_lab.evaluation.service import EvaluationService
from ml_lab.storage.workspace import Workspace

BOUNDED_SIGNAL_RECEIPT_NAME = "phase-a-expanded-bounded-provider-signal-receipt.json"


def run_expanded_phase_a_bounded_provider_signal(
    *,
    workspace_path: Path,
    model: str = "qwen3.5:4b-q4_K_M",
    endpoint: str = "http://127.0.0.1:11434/v1/chat/completions",
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    workspace = Workspace.open(workspace_path)
    freeze_path = workspace.root / "phase-a-expanded-freeze-receipt.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    if not isinstance(freeze, dict):
        raise ValueError("Expanded Phase A freeze receipt must be an object")
    _validate_expanded_freeze_receipt(freeze)

    project_id = _required_text(freeze, "project_id")
    dataset_id = _required_text(_mapping(freeze, "dataset"), "id")
    snapshot_id = _required_text(_mapping(freeze, "contract_snapshot"), "id")

    result = run_phase_a_bounded_provider_signal_dev(
        workspace,
        project_id=project_id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot_id,
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )
    if len(result.summaries) != 1 or result.summaries[0].split is not DatasetSplit.DEV:
        raise RuntimeError("Bounded provider signal did not produce exactly one DEV summary")
    summary = result.summaries[0]

    evaluation = EvaluationService(workspace)
    if evaluation.case_count(result.experiment.id, DatasetSplit.TEST) != 0:
        raise RuntimeError("Bounded provider signal created TEST evidence")
    if evaluation.case_count(result.experiment.id, DatasetSplit.REDTEAM) != 0:
        raise RuntimeError("Bounded provider signal created REDTEAM evidence")

    metrics = summarize_phase_a_experiment(
        workspace,
        result.experiment.id,
        DatasetSplit.DEV,
    )
    metric_rows = [
        {
            "metric_id": metric.metric_id,
            "value": metric.value,
            "direction": metric.direction.value,
            "veto": metric.veto,
            "unit": metric.unit,
            "note": metric.note,
        }
        for metric in metrics
    ]
    veto_nonzero = [
        row
        for row in metric_rows
        if row["veto"] is True
        and isinstance(row["value"], (int, float))
        and float(row["value"]) != 0.0
    ]

    base: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-bounded-provider-signal/1",
        "ok": True,
        "stage": "EXPERIMENT",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "contract_snapshot_id": snapshot_id,
        "target_frankenhomie_commit": freeze["target_frankenhomie_commit"],
        "target_contract": freeze["target_contract"],
        "provider": {
            "model": model.strip(),
            "endpoint": endpoint.strip(),
            "timeout_seconds": float(timeout_seconds),
            "role": "SEMANTIC_SIGNAL_ONLY",
            "network_policy": "LOOPBACK_HTTP_ONLY",
            "database_access_allowed": False,
            "authority_mutation_allowed": False,
        },
        "assembly": {
            "deterministic_abstention_first": True,
            "entity_identity_from_provider": False,
            "facts_from_provider": False,
            "candidate_keys_from_provider": False,
            "entity_binding": "CURRENT_VISIBLE_ENVELOPE_EXPLICIT_MENTION_ONLY",
            "non_entity_values_must_be_allowlisted": True,
            "adapter_validation_required": True,
        },
        "experiment_id": result.experiment.id,
        "trainer_id": result.experiment.trainer_id,
        "dev": {
            "total": summary.total,
            "correct": summary.correct,
            "failures": summary.failures,
            "veto_failures": summary.veto_failures,
            "mean_latency_ms": summary.mean_latency_ms,
            "p95_latency_ms": summary.p95_latency_ms,
            "metrics": metric_rows,
            "failure_rows": _failure_rows(workspace, result.experiment.id),
        },
        "veto_failure_count": len(veto_nonzero),
        "veto_metrics_nonzero": veto_nonzero,
        "dev_evaluated": True,
        "test_evaluated": False,
        "redteam_evaluated": False,
        "zero_model_redteam_evaluated_by_model": False,
        "training_performed": False,
        "promotion_allowed": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(canonical_json(base).encode()).hexdigest()
    receipt = {**base, "payload_sha256": payload_sha}
    (workspace.root / BOUNDED_SIGNAL_RECEIPT_NAME).write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt
