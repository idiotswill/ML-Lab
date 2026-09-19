import json
import subprocess
import time
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_ADAPTER_VERSION,
    PHASE_A_CONTRACT_VERSION,
)
from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.phase_a_metrics import (
    metric_rows_for_ui,
    summarize_phase_a_experiment,
)
from ml_lab.evaluation.runners import evaluate_phase_a_sparse_experiment
from ml_lab.evaluation.service import EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.phase_a_sparse import train_phase_a_sparse
from ml_lab.trainers.service import PHASE_A_RUNTIME_PACK_ID, PHASE_A_TRAINER_ID
from ml_lab.ui.experiments import ExperimentsController

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


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _fake_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "frankenhomie"
    package = repo / "frankenhomie-asterra-v0.9.0" / "asterra"
    data = repo / "frankenhomie-asterra-v0.9.0" / "data"
    package.mkdir(parents=True)
    data.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "semantic_residual.py").write_text(_FAKE_RESIDUAL, encoding="utf-8")
    (data / "semantic_family_registry.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fake residual contract")
    return repo, _git(repo, "rev-parse", "HEAD")


def _request(declaration: str) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "failed_deterministic_stage": "IDENTITY:AMBIGUOUS",
        "assessment": {
            "route": "SEMANTIC_REQUIRED",
            "model_required_because": "IDENTITY:AMBIGUOUS",
            "declaration": declaration,
        },
        "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        "allowed_action_families": ["HARM_TARGET"],
        "family_slots": [
            {
                "family": "HARM_TARGET",
                "slots": [
                    {
                        "name": "TARGET_COMBATANT",
                        "required": True,
                        "allowed_values": ["combatant:mara"],
                        "prebound": None,
                    }
                ],
            }
        ],
    }


def _resolve() -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": "HARM_TARGET",
        "entity_keys": [],
        "slots": [{"name": "TARGET_COMBATANT", "value": "combatant:mara"}],
        "reason_code": "EXPECTED",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def test_phase_a_protected_evaluation_uses_pinned_reference_commit(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    residual_path = "frankenhomie-asterra-v0.9.0/asterra/semantic_residual.py"
    snapshot = ContractSnapshotService(workspace).capture(
        project_id=project.id,
        adapter_id=PHASE_A_ADAPTER_ID,
        adapter_version=PHASE_A_ADAPTER_VERSION,
        repository=repo,
        ref=commit,
        contract_version=PHASE_A_CONTRACT_VERSION,
        files=(ContractFileSpec(residual_path, "residual-contract"),),
    )

    train_request = _request("I clobber Mara")
    test_request = _request("Bring the iron hammer down upon Mara before she can flee")
    expected = _resolve()
    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "phase-a-e2e",
        contract_snapshot_id=snapshot.id,
    )
    for split, example_id, request in (
        (DatasetSplit.TRAIN, "train-mara", train_request),
        (DatasetSplit.TEST, "test-mara", test_request),
    ):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"request": request},
                label=expected,
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)

    model = train_phase_a_sparse(
        [{"payload": {"request": train_request}, "label": expected}],
        feature_dim=1024,
    )
    model_path = tmp_path / "phase-a-model.json"
    model.save(model_path)
    model_ref = workspace.artifacts.commit_file(
        model_path,
        media_type="application/vnd.ml-lab.phase-a-bounded-sparse+json",
    )

    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id=PHASE_A_TRAINER_ID,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={"feature_dim": 1024, "alpha": 0.5},
        seed=7,
        contract_snapshot_id=snapshot.id,
    )
    experiments.start(experiment.id)
    experiment = experiments.complete(
        experiment.id,
        metrics=(),
        model_artifact_digest=model_ref.digest,
    )

    (repo / residual_path).write_text("raise RuntimeError('dirty bytes')\n", encoding="utf-8")
    summaries = evaluate_phase_a_sparse_experiment(
        workspace,
        experiment.id,
        splits=(DatasetSplit.TEST,),
    )
    assert len(summaries) == 1
    assert summaries[0].total == 1
    assert summaries[0].correct == 1
    assert summaries[0].veto_failures == 0
    assert summaries[0].mean_latency_ms >= 0

    cases = EvaluationService(workspace).page_cases(
        experiment.id,
        DatasetSplit.TEST,
    )
    assert len(cases) == 1
    observed = json.loads(cases[0].observed_json)
    preflight = observed["preflight"]
    assert preflight["status"] == "MODEL_ALLOWED"
    assert preflight["model_call_allowed"] is True
    assert preflight["fresh_process"] is True
    assert preflight["authority_mutation_allowed"] is False
    preflight_receipt = json.loads(
        workspace.artifacts.resolve(
            preflight["receipt_artifact_digest"]
        ).read_text(encoding="utf-8")
    )
    assert preflight_receipt["database_access_allowed"] is False
    assert preflight_receipt["network_access_allowed"] is False
    assert preflight_receipt["resolver_dispatch_available"] is False

    reference = observed["reference"]
    assert reference["status"] == "ACCEPTED"
    assert reference["commit_sha"] == commit
    assert reference["validator"] == "frankenhomie.run_residual_semantics"
    assert reference["fresh_process"] is True
    assert reference["authority_mutation_allowed"] is False
    receipt_digest = reference["receipt_artifact_digest"]
    assert isinstance(receipt_digest, str) and receipt_digest
    persisted_receipt = json.loads(
        workspace.artifacts.resolve(receipt_digest).read_text(encoding="utf-8")
    )
    assert persisted_receipt["database_access_allowed"] is False
    assert persisted_receipt["network_access_allowed"] is False
    assert persisted_receipt["resolver_dispatch_available"] is False

    metrics = {
        row["metricId"]: row
        for row in metric_rows_for_ui(
            summarize_phase_a_experiment(
                workspace,
                experiment.id,
                DatasetSplit.TEST,
            )
        )
    }
    assert metrics["zero_model_route_violations"]["measured"] is True
    assert metrics["zero_model_route_violations"]["value"] == 0.0
    assert metrics["unsupported_mechanics_authority"]["measured"] is True
    assert metrics["unsupported_mechanics_authority"]["value"] == 0.0



def test_phase_a_experiment_can_launch_through_gui_controller(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "gui-workspace")
    project = workspace.create_project("Phase A GUI", adapter_id=PHASE_A_ADAPTER_ID)
    residual_path = "frankenhomie-asterra-v0.9.0/asterra/semantic_residual.py"
    snapshot = ContractSnapshotService(workspace).capture(
        project_id=project.id,
        adapter_id=PHASE_A_ADAPTER_ID,
        adapter_version=PHASE_A_ADAPTER_VERSION,
        repository=repo,
        ref=commit,
        contract_version=PHASE_A_CONTRACT_VERSION,
        files=(ContractFileSpec(residual_path, "residual-contract"),),
    )

    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "gui-phase-a",
        contract_snapshot_id=snapshot.id,
    )
    datasets.add_example(
        dataset.id,
        ValidatedExampleInput(
            example_id="train-gui-mara",
            split=DatasetSplit.TRAIN,
            source_id="synthetic:train-gui-mara",
            lineage_group="lineage:train-gui-mara",
            payload={"request": _request("I clobber Mara")},
            label=_resolve(),
            tags=(),
        ),
    )
    datasets.freeze(dataset.id)

    jobs = JobManager(workspace.root, workspace.database, workspace.artifacts)
    controller = ExperimentsController()
    failures: list[tuple[str, str]] = []
    controller.operationFailed.connect(
        lambda title, message: failures.append((str(title), str(message)))
    )
    try:
        controller.bind_project(
            workspace,
            jobs,
            project.id,
            PHASE_A_ADAPTER_ID,
        )
        assert controller.hasRunnableTrainer
        assert [item["trainerId"] for item in controller.trainerOptions] == [
            PHASE_A_TRAINER_ID
        ]
        assert [item["id"] for item in controller.frozenDatasets] == [dataset.id]

        controller.createAndLaunch(
            dataset.id,
            PHASE_A_TRAINER_ID,
            PHASE_A_RUNTIME_PACK_ID,
            "17",
            "1024",
            "0.5",
            "",
            "",
        )
        assert failures == []
        selected = controller.selectedExperiment
        assert selected["contractSnapshotId"] == snapshot.id
        experiment_id = str(selected["id"])

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            controller.refresh()
            selected = controller.selectedExperiment
            if selected.get("status") in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "INTERRUPTED",
            }:
                break
            time.sleep(0.03)

        assert failures == []
        assert selected["id"] == experiment_id
        assert selected["status"] == "COMPLETED"
        assert selected["modelDigest"]
        assert selected["contractSnapshotId"] == snapshot.id
    finally:
        jobs.shutdown()
