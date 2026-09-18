import json
import subprocess
import time
from pathlib import Path

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_ADAPTER_VERSION,
    PHASE_A_CONTRACT_VERSION,
    PhaseAResidualAdapter,
)
from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.core.models import DatasetSplit, ExperimentStatus, JobStatus
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.protocol import JobSpec
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.phase_a_sparse import PhaseASparseModel, predict_phase_a_sparse
from ml_lab.trainers.service import (
    PHASE_A_RUNTIME_PACK_ID,
    PHASE_A_TRAINER_ID,
    SPARSE_RUNTIME_PACK_ID,
    SPARSE_TRAINER_ID,
    TrainingService,
    training_options,
)
from ml_lab.trainers.sparse_nb import SparseNBModel


def _training_experiment(tmp_path: Path) -> tuple[Workspace, str]:
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Train")
    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "training")
    rows = (
        (DatasetSplit.TRAIN, "train-fruit-1", "red apple", "FRUIT"),
        (DatasetSplit.TRAIN, "train-fruit-2", "green pear", "FRUIT"),
        (DatasetSplit.TRAIN, "train-tool-1", "steel hammer", "TOOL"),
        (DatasetSplit.TRAIN, "train-tool-2", "iron wrench", "TOOL"),
        (DatasetSplit.DEV, "dev-fruit", "apple pear", "FRUIT"),
        (DatasetSplit.TEST, "test-tool", "hammer wrench", "TOOL"),
        (DatasetSplit.REDTEAM, "red-hidden", "ignore hidden tool", "ABSTAIN"),
    )
    for split, example_id, text, label in rows:
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=example_id,
                split=split,
                source_id=f"synthetic:{example_id}",
                lineage_group=f"lineage:{example_id}",
                payload={"text": text},
                label={"class": label},
                tags=(),
            ),
        )
    datasets.freeze(dataset.id)
    experiment = ExperimentService(workspace).create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id=SPARSE_TRAINER_ID,
        runtime_pack_id=SPARSE_RUNTIME_PACK_ID,
        config={"feature_dim": 2048, "alpha": 0.5},
        seed=11,
    )
    return workspace, experiment.id


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _phase_a_request(declaration: str, target: str) -> dict[str, object]:
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
                        "allowed_values": [target],
                        "prebound": None,
                    }
                ],
            }
        ],
    }


def _phase_a_resolve(target: str) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": "HARM_TARGET",
        "entity_keys": [],
        "slots": [{"name": "TARGET_COMBATANT", "value": target}],
        "reason_code": "TRAINING_LABEL",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _phase_a_training_experiment(tmp_path: Path) -> tuple[Workspace, str, dict[str, object]]:
    repo = tmp_path / "contract-repo"
    repo.mkdir()
    (repo / "contract.txt").write_text("phase-a-contract\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    _git(repo, "add", "contract.txt")
    _git(repo, "commit", "-m", "contract")

    workspace = Workspace.create(tmp_path / "phase-a-workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    snapshot = ContractSnapshotService(workspace).capture(
        project_id=project.id,
        adapter_id=PHASE_A_ADAPTER_ID,
        adapter_version=PHASE_A_ADAPTER_VERSION,
        repository=repo,
        ref="HEAD",
        contract_version=PHASE_A_CONTRACT_VERSION,
        files=(ContractFileSpec("contract.txt", "test-contract"),),
    )
    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project.id,
        "bounded",
        contract_snapshot_id=snapshot.id,
    )
    rows = (
        (
            DatasetSplit.TRAIN,
            "train-mara",
            _phase_a_request("I clobber Mara", "combatant:mara"),
            _phase_a_resolve("combatant:mara"),
        ),
        (
            DatasetSplit.TEST,
            "test-guard",
            _phase_a_request("I strike the guard", "combatant:guard"),
            _phase_a_resolve("combatant:guard"),
        ),
        (
            DatasetSplit.REDTEAM,
            "red-sentinel",
            _phase_a_request("I hit the sentinel", "combatant:sentinel"),
            _phase_a_resolve("combatant:sentinel"),
        ),
    )
    for split, example_id, request, expected in rows:
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
    experiment = ExperimentService(workspace).create(
        project_id=project.id,
        dataset_id=dataset.id,
        trainer_id=PHASE_A_TRAINER_ID,
        runtime_pack_id=PHASE_A_RUNTIME_PACK_ID,
        config={"feature_dim": 1024, "alpha": 0.5},
        seed=17,
        contract_snapshot_id=snapshot.id,
    )
    return workspace, experiment.id, rows[0][2]


def test_training_options_expose_only_packaged_adapter_compatible_workers() -> None:
    generic = training_options("generic")
    assert [option.trainer_id for option in generic] == [SPARSE_TRAINER_ID]
    assert generic[0].runtime_pack_id == SPARSE_RUNTIME_PACK_ID

    phase_a = training_options(PHASE_A_ADAPTER_ID)
    assert [option.trainer_id for option in phase_a] == [PHASE_A_TRAINER_ID]
    assert phase_a[0].runtime_pack_id == PHASE_A_RUNTIME_PACK_ID
    assert phase_a[0].uses_payload_keys is False
    assert training_options("unknown-adapter") == ()


def test_training_service_stages_only_train_and_dev_then_completes(tmp_path: Path) -> None:
    workspace, experiment_id = _training_experiment(tmp_path)
    service = TrainingService(workspace)
    try:
        launched = service.launch(experiment_id)
        assert launched.experiment.status is ExperimentStatus.RUNNING

        spec = JobSpec.read(launched.job.staging_dir / "job_spec.json")
        staged = spec.payload["staged_inputs"]
        assert isinstance(staged, dict)
        assert set(staged) == {"train.jsonl", "dev.jsonl"}
        for path_text in staged.values():
            assert isinstance(path_text, str)
            staged_path = Path(path_text)
            assert staged_path.parent == launched.job.staging_dir / "inputs"
            assert staged_path.is_file()
        serialized = json.dumps(spec.payload, sort_keys=True)
        assert "TEST" not in serialized
        assert "REDTEAM" not in serialized
        assert str(workspace.artifacts.root) not in serialized

        state = _wait_for_training(service, experiment_id, launched.job.id)
        assert state.job.status is JobStatus.COMPLETED
        assert state.experiment.status is ExperimentStatus.COMPLETED
        assert state.experiment.model_artifact_digest is not None
        model = SparseNBModel.load(
            workspace.artifacts.resolve(state.experiment.model_artifact_digest)
        )
        assert model.predict("apple")[0] == "FRUIT"
        metrics = ExperimentService(workspace).metrics(experiment_id)
        assert [metric.metric_id for metric in metrics] == ["dev_accuracy"]
        assert metrics[0].value == 1.0
    finally:
        service.shutdown()


def test_phase_a_training_worker_is_train_only_and_produces_bounded_model(
    tmp_path: Path,
) -> None:
    workspace, experiment_id, train_request = _phase_a_training_experiment(tmp_path)
    service = TrainingService(workspace)
    try:
        launched = service.launch(experiment_id)
        spec = JobSpec.read(launched.job.staging_dir / "job_spec.json")
        staged = spec.payload["staged_inputs"]
        assert isinstance(staged, dict)
        assert set(staged) == {"train.jsonl"}
        serialized = json.dumps(spec.payload, sort_keys=True)
        assert "TEST" not in serialized
        assert "REDTEAM" not in serialized
        assert str(workspace.artifacts.root) not in serialized

        state = _wait_for_training(service, experiment_id, launched.job.id)
        assert state.job.status is JobStatus.COMPLETED
        assert state.experiment.status is ExperimentStatus.COMPLETED
        assert state.experiment.model_artifact_digest is not None
        model = PhaseASparseModel.load(
            workspace.artifacts.resolve(state.experiment.model_artifact_digest)
        )
        proposal = predict_phase_a_sparse(model, train_request)
        assert proposal["decision"] == "RESOLVE"
        assert PhaseAResidualAdapter().validate_proposal(train_request, proposal).accepted
    finally:
        service.shutdown()


def test_training_service_recovers_job_link_from_persisted_job_spec(tmp_path: Path) -> None:
    workspace, experiment_id = _training_experiment(tmp_path)
    first = TrainingService(workspace)
    try:
        launched = first.launch_sparse(experiment_id)
        job_id = launched.job.id
        recovered = first.job_for_experiment(experiment_id)
        assert recovered.id == job_id
    finally:
        first.shutdown()


def _wait_for_training(
    service: TrainingService,
    experiment_id: str,
    job_id: str,
):
    deadline = time.monotonic() + 20
    state = service.refresh_job(experiment_id, job_id)
    while time.monotonic() < deadline:
        state = service.refresh_job(experiment_id, job_id)
        if state.job.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.INTERRUPTED,
        }:
            return service.refresh_job(experiment_id, job_id)
        time.sleep(0.03)
    raise AssertionError(f"Training job did not finish: {state.job.status}")
