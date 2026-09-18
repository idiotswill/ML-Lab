import json
import subprocess
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_ADAPTER_VERSION,
    PHASE_A_CONTRACT_VERSION,
)
from ml_lab.baselines.phase_a import (
    PHASE_A_ABSTENTION_BASELINE_ID,
    PHASE_A_LEXICAL_BASELINE_ID,
    completed_phase_a_baseline,
    run_phase_a_baseline,
)
from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.core.models import DatasetSplit, ExperimentStatus
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.phase_a_metrics import (
    load_pinned_provider_reference,
    metric_rows_for_ui,
    summarize_phase_a_experiment,
)
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import PHASE_A_RUNTIME_PACK_ID, PHASE_A_TRAINER_ID

_FAKE_RESIDUAL = '''from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ResidualSemanticError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ResidualSemanticRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str


class ResidualSemanticDecisionV2(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str
    decision: str


def run_residual_semantics(provider, request):
    return provider.resolve(request)
'''

_REGISTRY = {
    "schema_version": "asterra-semantic-family-registry/1",
    "families": [
        {
            "family": "HARM_TARGET",
            "entity_slot": "TARGET_COMBATANT",
            "deterministic_aliases": ["attack", "clobber"],
            "residual_hints": [],
        },
        {
            "family": "MOVE_TRAVEL",
            "entity_slot": "DESTINATION",
            "deterministic_aliases": ["move", "retreat"],
            "residual_hints": [],
        },
        {
            "family": "SEARCH_INSPECT",
            "entity_slot": "SUBJECT",
            "deterministic_aliases": ["search", "inspect"],
            "residual_hints": [],
        },
    ],
}

_PROVIDER_RESULT = {
    "schema": "frankenhomie-a5-intake-results/1",
    "provenance": {
        "model": "qwen3.5:4b-q4_K_M",
        "timestamp": "2026-09-05T00:00:00Z",
        "head": "historical123",
        "dirty": True,
        "independent_qa": False,
        "corpus_sha256": "corpus-hash",
    },
    "metrics": {
        "total_cases": 3,
        "model_call_count": 2,
        "false_commitment_count": 0,
        "provider_transport_failures": 0,
        "zero_model_violations": 0,
        "candidate_envelope_violations": 0,
        "fact_envelope_violations": 1,
        "latency_seconds": {"median": 1.25},
    },
    "rows": [
        {
            "id": "provider-resolve",
            "model_called": True,
            "expected": "RESOLVE",
            "actual": "RESOLVE",
            "correct": True,
        },
        {
            "id": "provider-ask",
            "model_called": True,
            "expected": "ASK_PLAYER",
            "actual": "ASK_PLAYER",
            "correct": True,
        },
        {
            "id": "zero-model",
            "model_called": False,
            "expected": "RESOLVE",
            "actual": "RESOLVE",
            "correct": True,
        },
    ],
}


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _repo_and_snapshot(tmp_path: Path) -> tuple[Workspace, str, str]:
    repo = tmp_path / "frankenhomie"
    package = repo / "frankenhomie-asterra-v0.9.0" / "asterra"
    data = repo / "frankenhomie-asterra-v0.9.0" / "data"
    benchmarks = repo / "frankenhomie-asterra-v0.9.0" / "benchmarks"
    package.mkdir(parents=True)
    data.mkdir(parents=True)
    benchmarks.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "semantic_residual.py").write_text(_FAKE_RESIDUAL, encoding="utf-8")
    (data / "semantic_family_registry.json").write_text(
        json.dumps(_REGISTRY, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (benchmarks / "PHASE_A5_SEMANTIC_INTAKE_RESULTS_V7.json").write_text(
        json.dumps(_PROVIDER_RESULT, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "scientific comparison fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    root = "frankenhomie-asterra-v0.9.0"
    snapshot = ContractSnapshotService(workspace).capture(
        project_id=project.id,
        adapter_id=PHASE_A_ADAPTER_ID,
        adapter_version=PHASE_A_ADAPTER_VERSION,
        repository=repo,
        ref=commit,
        contract_version=PHASE_A_CONTRACT_VERSION,
        files=(
            ContractFileSpec(
                f"{root}/asterra/semantic_residual.py",
                "residual-contract",
            ),
            ContractFileSpec(
                f"{root}/data/semantic_family_registry.json",
                "semantic-registry",
            ),
            ContractFileSpec(
                f"{root}/benchmarks/PHASE_A5_SEMANTIC_INTAKE_RESULTS_V7.json",
                "provider-baseline",
            ),
        ),
    )
    return workspace, project.id, snapshot.id


def _request(
    declaration: str,
    family: str,
    slot_name: str,
    value: str,
    visible_name: str,
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
                        "allowed_values": [value],
                        "prebound": None,
                    }
                ],
            }
        ],
        "candidates": [{"key": value, "name": visible_name}],
    }


def _resolve(family: str, slot_name: str, value: str) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": [{"name": slot_name, "value": value}],
        "reason_code": "EXPECTED",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _ask() -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "ASK_PLAYER",
        "action_family": None,
        "entity_keys": [],
        "slots": [],
        "reason_code": "EXPECTED_ASK",
        "facts_used": [],
        "question": "What action do you want to take?",
        "missing_slots": ["ACTION_FAMILY"],
        "candidate_keys": [],
    }


def _dataset(workspace: Workspace, project_id: str, snapshot_id: str) -> str:
    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project_id,
        "science",
        contract_snapshot_id=snapshot_id,
    )
    rows = (
        (
            DatasetSplit.TEST,
            "test-harm",
            _request(
                "I clobber Mara",
                "HARM_TARGET",
                "TARGET_COMBATANT",
                "combatant:mara",
                "Mara",
            ),
            _resolve("HARM_TARGET", "TARGET_COMBATANT", "combatant:mara"),
        ),
        (
            DatasetSplit.TEST,
            "test-unclear",
            _request(
                "I hesitate and wait for a clearer idea",
                "SEARCH_INSPECT",
                "SUBJECT",
                "object:chest",
                "Ancient Chest",
            ),
            _ask(),
        ),
        (
            DatasetSplit.REDTEAM,
            "red-travel",
            _request(
                "I retreat toward Camp through the distant northern road",
                "MOVE_TRAVEL",
                "DESTINATION",
                "location:camp",
                "Camp",
            ),
            _ask(),
        ),
    )
    for split, example_id, request, label in rows:
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"request": request},
                label=label,
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)
    return dataset.id


def _metric_map(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["metricId"]): row for row in rows}


def test_phase_a_baselines_share_frozen_dataset_and_expose_veto_metrics(
    tmp_path: Path,
) -> None:
    workspace, project_id, snapshot_id = _repo_and_snapshot(tmp_path)
    dataset_id = _dataset(workspace, project_id, snapshot_id)

    abstention = run_phase_a_baseline(
        workspace,
        project_id=project_id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot_id,
        baseline_id=PHASE_A_ABSTENTION_BASELINE_ID,
    )
    assert abstention.experiment.status is ExperimentStatus.COMPLETED
    assert abstention.experiment.dataset_id == dataset_id
    assert abstention.experiment.contract_snapshot_id == snapshot_id
    assert abstention.experiment.model_artifact_digest is None
    assert completed_phase_a_baseline(
        workspace,
        project_id=project_id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot_id,
        baseline_id=PHASE_A_ABSTENTION_BASELINE_ID,
    ) == abstention.experiment.id

    abstention_metrics = _metric_map(
        metric_rows_for_ui(
            summarize_phase_a_experiment(
                workspace,
                abstention.experiment.id,
                DatasetSplit.TEST,
            )
        )
    )
    assert abstention_metrics["false_commitments"]["measured"] is True
    assert abstention_metrics["false_commitments"]["value"] == 0.0
    assert abstention_metrics["resolution_precision"]["measured"] is False

    lexical = run_phase_a_baseline(
        workspace,
        project_id=project_id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot_id,
        baseline_id=PHASE_A_LEXICAL_BASELINE_ID,
    )
    assert lexical.experiment.status is ExperimentStatus.COMPLETED
    test_metrics = _metric_map(
        metric_rows_for_ui(
            summarize_phase_a_experiment(
                workspace,
                lexical.experiment.id,
                DatasetSplit.TEST,
            )
        )
    )
    assert test_metrics["resolution_precision"]["value"] == 1.0
    assert test_metrics["decision_accuracy"]["value"] == 1.0
    assert test_metrics["false_commitments"]["value"] == 0.0

    red_metrics = _metric_map(
        metric_rows_for_ui(
            summarize_phase_a_experiment(
                workspace,
                lexical.experiment.id,
                DatasetSplit.REDTEAM,
            )
        )
    )
    assert red_metrics["false_commitments"]["value"] == 1.0
    assert red_metrics["false_commitments"]["veto"] is True
    assert red_metrics["adversarial_failures"]["value"] == 1.0


def test_historical_provider_evidence_stays_distinct_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    workspace, _project_id, snapshot_id = _repo_and_snapshot(tmp_path)
    provider = load_pinned_provider_reference(workspace, snapshot_id)
    assert provider is not None
    assert provider.model == "qwen3.5:4b-q4_K_M"
    assert provider.source_dirty is True
    assert provider.independent_qa is False
    assert provider.model_calls == 2
    assert provider.total_cases == 3
    rows = _metric_map(metric_rows_for_ui(provider.metrics))
    assert rows["hidden_or_out_of_envelope"]["value"] == 1.0
    assert rows["hidden_or_out_of_envelope"]["veto"] is True
    assert rows["adversarial_failures"]["measured"] is False
    assert rows["latency_ms"]["value"] == 1250.0


def test_unevaluated_phase_a_veto_metrics_are_not_reported_as_zero(
    tmp_path: Path,
) -> None:
    workspace, project_id, snapshot_id = _repo_and_snapshot(tmp_path)
    dataset_id = _dataset(workspace, project_id, snapshot_id)
    model = workspace.artifacts.commit_bytes(b"{}\n", media_type="application/json")
    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project_id,
        dataset_id=dataset_id,
        trainer_id=PHASE_A_TRAINER_ID,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        contract_snapshot_id=snapshot_id,
    )
    experiments.start(experiment.id)
    experiment = experiments.complete(
        experiment.id,
        metrics=(),
        model_artifact_digest=model.digest,
    )

    rows = _metric_map(
        metric_rows_for_ui(
            summarize_phase_a_experiment(
                workspace,
                experiment.id,
                DatasetSplit.TEST,
            )
        )
    )
    assert rows["false_commitments"]["measured"] is False
    assert rows["contract_failures"]["measured"] is False
    assert rows["hidden_or_out_of_envelope"]["measured"] is False
    assert rows["model_size_mb"]["measured"] is True
