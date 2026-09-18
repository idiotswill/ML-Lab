from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a import (
    PHASE_A_ADAPTER_ID,
    PHASE_A_ADAPTER_VERSION,
    PHASE_A_CONTRACT_VERSION,
)
from ml_lab.adapters.phase_a_provider import (
    PhaseALocalProviderError,
    PhaseALocalProviderRunner,
    validate_loopback_endpoint,
)
from ml_lab.baselines.phase_a_provider import (
    local_provider_baseline_id,
    run_phase_a_local_provider_baseline,
)
from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.core.models import (
    ContractSnapshot,
    DatasetSplit,
    ExperimentStatus,
    FailureSeverity,
)
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.evaluation.service import EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.storage.workspace import Workspace

_FAKE_RESIDUAL = '''from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ResidualSemanticError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ResidualSemanticRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str
    allowed_action_families: list[str] = Field(default_factory=list)
    family_slots: list[dict[str, Any]] = Field(default_factory=list)


class ResidualSemanticDecisionV2(BaseModel):
    model_config = ConfigDict(extra="allow")
    version: str
    decision: str
    action_family: str | None = None
    entity_keys: list[str] = Field(default_factory=list)
    slots: list[dict[str, str]] = Field(default_factory=list)
    reason_code: str = "FIXTURE"
    facts_used: list[str] = Field(default_factory=list)
    question: str | None = None
    missing_slots: list[str] = Field(default_factory=list)
    candidate_keys: list[str] = Field(default_factory=list)


class LocalResidualSemanticProvider:
    def __init__(self, transport):
        self.transport = transport
        self.provider_name = "fixture-local-residual"

    def resolve(self, request):
        return self.transport.resolve(request, ResidualSemanticDecisionV2)


def run_residual_semantics(provider, request):
    return provider.resolve(request)
'''

_FAKE_TRANSPORT = '''from __future__ import annotations

import socket
import sqlite3


class LocalChatCompletionsTurnProvider:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        timeout_seconds: float,
        max_tokens: int,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def resolve(self, request, contract):
        if self.model == "touch-db":
            sqlite3.connect(":memory:")
        if self.model == "touch-network":
            socket.create_connection(("example.com", 80), timeout=0.01)
        family = request.allowed_action_families[0]
        family_spec = next(
            item for item in request.family_slots if item["family"] == family
        )
        slots = []
        for slot in family_spec["slots"]:
            if slot.get("required", True) is False:
                continue
            allowed = slot.get("allowed_values", [])
            if allowed:
                slots.append({"name": slot["name"], "value": allowed[0]})
        return contract.model_validate(
            {
                "version": "semantic-residual-v2",
                "decision": "RESOLVE",
                "action_family": family,
                "entity_keys": [],
                "slots": slots,
                "reason_code": "FIXTURE_PROVIDER",
                "facts_used": [],
                "question": None,
                "missing_slots": [],
                "candidate_keys": [],
            }
        )
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
    app = repo / "frankenhomie-asterra-v0.9.0"
    package = app / "asterra"
    data = app / "data"
    package.mkdir(parents=True)
    data.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "semantic_residual.py").write_text(_FAKE_RESIDUAL, encoding="utf-8")
    (package / "turn_shadow.py").write_text(_FAKE_TRANSPORT, encoding="utf-8")
    (data / "fixture.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fake local provider contract")
    return repo, _git(repo, "rev-parse", "HEAD")


def _request(
    *,
    declaration: str,
    family: str = "HARM_TARGET",
    slot_name: str = "TARGET_COMBATANT",
    slot_value: str = "combatant:mara",
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "request_id": "fixture:" + family.casefold(),
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
                        "allowed_values": [slot_value],
                        "prebound": None,
                    }
                ],
            }
        ],
        "context": {
            "actor_id": "pc:test",
            "audience": "PC_PRIVATE",
            "candidates": [],
            "facts": [],
        },
    }


def _expected(
    *,
    family: str = "HARM_TARGET",
    slot_name: str = "TARGET_COMBATANT",
    slot_value: str = "combatant:mara",
) -> dict[str, object]:
    return {
        "version": PHASE_A_CONTRACT_VERSION,
        "decision": "RESOLVE",
        "action_family": family,
        "entity_keys": [],
        "slots": [{"name": slot_name, "value": slot_value}],
        "reason_code": "EXPECTED",
        "facts_used": [],
        "question": None,
        "missing_slots": [],
        "candidate_keys": [],
    }


def _snapshot(
    workspace: Workspace,
    project_id: str,
    repo: Path,
    commit: str,
) -> ContractSnapshot:
    root = "frankenhomie-asterra-v0.9.0/asterra"
    return ContractSnapshotService(workspace).capture(
        project_id=project_id,
        adapter_id=PHASE_A_ADAPTER_ID,
        adapter_version=PHASE_A_ADAPTER_VERSION,
        repository=repo,
        ref=commit,
        contract_version=PHASE_A_CONTRACT_VERSION,
        files=(
            ContractFileSpec(f"{root}/semantic_residual.py", "residual-contract"),
            ContractFileSpec(f"{root}/turn_shadow.py", "provider-transport"),
        ),
    )


def _frozen_dataset(
    workspace: Workspace,
    project_id: str,
    snapshot_id: str,
    *,
    include_redteam: bool = True,
) -> str:
    datasets = DatasetService(workspace)
    dataset = datasets.create(
        project_id,
        "local-provider-eval",
        contract_snapshot_id=snapshot_id,
    )
    rows = [
        (
            DatasetSplit.TEST,
            "test-harm",
            _request(
                declaration=(
                    "With a furious overhead smash beneath the eclipse I pulverize Mara, "
                    "the obsidian knight standing before me."
                )
            ),
            _expected(),
        )
    ]
    if include_redteam:
        rows.append(
            (
                DatasetSplit.REDTEAM,
                "redteam-travel",
                _request(
                    declaration=(
                        "At first light I journey through the salt marsh and reed beds toward "
                        "the distant eastern causeway beyond the flooded flats."
                    ),
                    family="MOVE_TRAVEL",
                    slot_name="DESTINATION",
                    slot_value="location:east",
                ),
                _expected(
                    family="MOVE_TRAVEL",
                    slot_name="DESTINATION",
                    slot_value="location:east",
                ),
            )
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
                tags=("local-provider",),
            ),
        )
    datasets.freeze(dataset.id)
    return dataset.id


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434/v1/chat/completions",
        "http://example.com:11434/v1/chat/completions",
        "http://user:pass@127.0.0.1:11434/v1/chat/completions",
        "http://127.0.0.1:11434/",
        "http://127.0.0.1:99999/v1/chat/completions",
        "http://127.0.0.1:11434/v1/chat/completions?remote=true",
    ],
)
def test_local_provider_endpoint_rejects_non_loopback_or_ambiguous_urls(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_loopback_endpoint(endpoint)


def test_local_provider_endpoint_accepts_loopback() -> None:
    assert (
        validate_loopback_endpoint("http://localhost:11434/v1/chat/completions")
        == "http://localhost:11434/v1/chat/completions"
    )
    assert (
        validate_loopback_endpoint("http://127.0.0.1:11434/v1/chat/completions")
        == "http://127.0.0.1:11434/v1/chat/completions"
    )


def test_provider_runner_uses_committed_bytes_and_persists_sandbox_receipt(
    tmp_path: Path,
) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    runner = PhaseALocalProviderRunner(
        workspace,
        repository=repo,
        ref=commit,
        model="fixture",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=5,
    )
    residual_path = (
        repo / "frankenhomie-asterra-v0.9.0" / "asterra" / "semantic_residual.py"
    )
    dirty_source = "raise RuntimeError('dirty bytes must not execute')\n"
    residual_path.write_text(dirty_source, encoding="utf-8")

    result = runner.propose(_request(declaration="I smash Mara with the hammer now."))
    assert result.proposal["decision"] == "RESOLVE"
    assert result.proposal["action_family"] == "HARM_TARGET"
    receipt = json.loads(
        workspace.artifacts.resolve(result.receipt_artifact_digest).read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "PROPOSED"
    assert receipt["commit_sha"] == commit
    assert receipt["model"] == "fixture"
    assert receipt["fresh_process"] is True
    assert receipt["database_access_allowed"] is False
    assert receipt["authority_mutation_allowed"] is False
    assert receipt["network_policy"] == "LOOPBACK_HTTP_ONLY"
    assert receipt["resolver_dispatch_invoked"] is False


@pytest.mark.parametrize(
    ("model", "error_code"),
    [
        ("touch-db", "LOCAL_PROVIDER_DB_ACCESS_FORBIDDEN"),
        ("touch-network", "LOCAL_PROVIDER_NON_LOOPBACK_NETWORK_FORBIDDEN"),
    ],
)
def test_provider_runner_fails_closed_on_forbidden_authority(
    tmp_path: Path,
    model: str,
    error_code: str,
) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / f"workspace-{model}")
    runner = PhaseALocalProviderRunner(
        workspace,
        repository=repo,
        ref=commit,
        model=model,
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=5,
    )
    with pytest.raises(PhaseALocalProviderError) as captured:
        runner.propose(_request(declaration="I smash Mara now."))
    assert captured.value.error_code == error_code
    digest = captured.value.receipt_artifact_digest
    assert digest is not None
    receipt = json.loads(
        workspace.artifacts.resolve(digest).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "ERROR"
    assert receipt["error_code"] == error_code
    assert receipt["database_access_allowed"] is False
    assert receipt["network_policy"] == "LOOPBACK_HTTP_ONLY"


def test_local_provider_baseline_is_same_dataset_and_persists_provider_receipts(
    tmp_path: Path,
) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    snapshot = _snapshot(workspace, project.id, repo, commit)
    dataset_id = _frozen_dataset(workspace, project.id, snapshot.id)

    result = run_phase_a_local_provider_baseline(
        workspace,
        project_id=project.id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot.id,
        model="fixture",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=5,
    )
    assert result.experiment.status is ExperimentStatus.COMPLETED
    assert result.experiment.dataset_id == dataset_id
    assert result.experiment.contract_snapshot_id == snapshot.id
    assert result.experiment.trainer_id.startswith("baseline:phase-a-local-provider-v2-")
    assert {item.split for item in result.summaries} == {
        DatasetSplit.TEST,
        DatasetSplit.REDTEAM,
    }
    assert all(item.correct == item.total for item in result.summaries)
    assert all(item.veto_failures == 0 for item in result.summaries)

    evaluation = EvaluationService(workspace)
    for split in (DatasetSplit.TEST, DatasetSplit.REDTEAM):
        cases = evaluation.page_cases(result.experiment.id, split)
        assert len(cases) == 1
        observed = json.loads(cases[0].observed_json)
        provider = observed["provider"]
        digest = provider["receipt_artifact_digest"]
        receipt = json.loads(
            workspace.artifacts.resolve(digest).read_text(encoding="utf-8")
        )
        assert receipt["status"] == "PROPOSED"
        assert receipt["commit_sha"] == commit
        assert receipt["network_policy"] == "LOOPBACK_HTTP_ONLY"
        assert observed["reference"]["commit_sha"] == commit
        assert observed["reference"]["authority_mutation_allowed"] is False


def test_provider_transport_failure_is_veto_case_not_hidden_fallback(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    snapshot = _snapshot(workspace, project.id, repo, commit)
    dataset_id = _frozen_dataset(
        workspace,
        project.id,
        snapshot.id,
        include_redteam=False,
    )

    result = run_phase_a_local_provider_baseline(
        workspace,
        project_id=project.id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot.id,
        model="touch-db",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=5,
    )
    assert result.experiment.status is ExperimentStatus.COMPLETED
    assert len(result.summaries) == 1
    assert result.summaries[0].total == 1
    assert result.summaries[0].veto_failures == 1

    failures = FailureService(workspace).page_for_project(
        project.id,
        severity=FailureSeverity.VETO,
    )
    assert len(failures) == 1
    assert failures[0].kind == "CONTRACT_FAILURE"
    observed = json.loads(failures[0].observed_json)
    assert observed["status"] == "LOCAL_PROVIDER_ERROR"
    assert observed["error_code"] == "LOCAL_PROVIDER_DB_ACCESS_FORBIDDEN"
    assert observed["provider_receipt_artifact_digest"]


def test_local_provider_baseline_identity_changes_with_configuration() -> None:
    base = local_provider_baseline_id(
        model="qwen3.5:4b-q4_K_M",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=120,
    )
    assert base != local_provider_baseline_id(
        model="other-model",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=120,
    )
    assert base != local_provider_baseline_id(
        model="qwen3.5:4b-q4_K_M",
        endpoint="http://localhost:11434/v1/chat/completions",
        timeout_seconds=120,
    )
    assert base != local_provider_baseline_id(
        model="qwen3.5:4b-q4_K_M",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=60,
    )


def test_same_provider_configuration_preserves_each_manual_rerun(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    snapshot = _snapshot(workspace, project.id, repo, commit)
    dataset_id = _frozen_dataset(
        workspace,
        project.id,
        snapshot.id,
        include_redteam=False,
    )
    common = {
        "project_id": project.id,
        "dataset_id": dataset_id,
        "contract_snapshot_id": snapshot.id,
        "model": "fixture",
        "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
        "timeout_seconds": 5.0,
    }
    first = run_phase_a_local_provider_baseline(workspace, **common)
    second = run_phase_a_local_provider_baseline(workspace, **common)
    assert first.experiment.id != second.experiment.id
    assert first.experiment.trainer_id == second.experiment.trainer_id
    assert EvaluationService(workspace).case_count(
        first.experiment.id,
        DatasetSplit.TEST,
    ) == 1
    assert EvaluationService(workspace).case_count(
        second.experiment.id,
        DatasetSplit.TEST,
    ) == 1


def test_provider_experiment_remains_an_ordinary_immutable_baseline(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)
    snapshot = _snapshot(workspace, project.id, repo, commit)
    dataset_id = _frozen_dataset(
        workspace,
        project.id,
        snapshot.id,
        include_redteam=False,
    )
    result = run_phase_a_local_provider_baseline(
        workspace,
        project_id=project.id,
        dataset_id=dataset_id,
        contract_snapshot_id=snapshot.id,
        model="fixture",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        timeout_seconds=5,
    )
    stored = ExperimentService(workspace).get(result.experiment.id)
    assert stored.status is ExperimentStatus.COMPLETED
    assert stored.model_artifact_digest is None
    assert stored.metrics_artifact_digest is not None
