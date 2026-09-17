import json
import subprocess
from pathlib import Path

from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.storage.workspace import Workspace

_FAKE_RESIDUAL = '''from __future__ import annotations

import sqlite3
from pydantic import BaseModel, ConfigDict


class ResidualSemanticError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ResidualSemanticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    veto: bool = False
    touch_db: bool = False


class ResidualSemanticDecisionV2(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    decision: str


def run_residual_semantics(provider, request):
    if request.touch_db:
        sqlite3.connect(":memory:")
    if request.veto:
        raise ResidualSemanticError("DETERMINISTIC_VETO", "blocked upstream")
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


def _proposal() -> dict[str, object]:
    return {"version": "semantic-residual-v2", "decision": "RESOLVE"}


def test_reference_validation_uses_committed_bytes_not_dirty_worktree(tmp_path: Path) -> None:
    repo, commit = _fake_repo(tmp_path)
    residual_path = repo / "frankenhomie-asterra-v0.9.0" / "asterra" / "semantic_residual.py"
    residual_path.write_text("raise RuntimeError('dirty worktree imported')\n", encoding="utf-8")

    workspace = Workspace.create(tmp_path / "workspace")
    validator = PhaseAReferenceValidator(workspace)
    receipt = validator.validate(
        repository=repo,
        ref="HEAD",
        request={"version": "semantic-residual-v2"},
        proposal=_proposal(),
    )

    assert receipt.status == "ACCEPTED"
    assert receipt.commit_sha == commit
    assert receipt.fresh_process is True
    assert receipt.authority_mutation_allowed is False
    persisted = json.loads(
        workspace.artifacts.resolve(receipt.receipt_artifact_digest).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["database_access_allowed"] is False
    assert persisted["resolver_dispatch_available"] is False
    assert "dirty worktree imported" in residual_path.read_text(encoding="utf-8")


def test_reference_validation_preserves_deterministic_veto(tmp_path: Path) -> None:
    repo, _ = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    receipt = PhaseAReferenceValidator(workspace).validate(
        repository=repo,
        ref="HEAD",
        request={"version": "semantic-residual-v2", "veto": True},
        proposal=_proposal(),
    )
    assert receipt.status == "REJECTED"
    assert receipt.error_code == "DETERMINISTIC_VETO"


def test_reference_validation_forbids_sqlite_access(tmp_path: Path) -> None:
    repo, _ = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    receipt = PhaseAReferenceValidator(workspace).validate(
        repository=repo,
        ref="HEAD",
        request={"version": "semantic-residual-v2", "touch_db": True},
        proposal=_proposal(),
    )
    assert receipt.status == "ERROR"
    assert receipt.error_code == "REFERENCE_DB_ACCESS_FORBIDDEN"
