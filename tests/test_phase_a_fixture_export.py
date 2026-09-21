from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a_fixture_export import PhaseAFixtureExporter
from ml_lab.storage.workspace import Workspace

_FAKE_DB = """
SCHEMA = "CREATE TABLE IF NOT EXISTS fixture_marker (id INTEGER);"

def _execute_sql_script(conn, script):
    conn.executescript(script)

def _legacy_migrate(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS fixture_legacy_migration (id INTEGER)")

def _run_migrations(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS fixture_numbered_migration (id INTEGER)")
"""

_FAKE_TURN_ORCHESTRATOR = """
class TurnFact:
    @classmethod
    def model_validate(cls, raw):
        return dict(raw)
"""

_FAKE_SEMANTIC_DISPATCH = """
def registered_semantic_families():
    return ("SEARCH_INSPECT",)
"""

_FAKE_ROUTING = """
class FakeRequest:
    def __init__(self, declaration):
        self.declaration = declaration

    def model_dump(self, mode="json"):
        return {
            "version": "semantic-residual-v2",
            "request_id": "semantic:fixture",
            "assessment": {
                "route": "SEMANTIC_REQUIRED",
                "model_required_because": "FAMILY:UNKNOWN_COMMITTED_FAMILY",
            },
            "context": {"actor_id": "tester", "candidates": []},
            "failed_deterministic_stage": "FAMILY:UNKNOWN_COMMITTED_FAMILY",
            "allowed_action_families": ["SEARCH_INSPECT"],
            "family_slots": [
                {
                    "family": "SEARCH_INSPECT",
                    "slots": [
                        {
                            "name": "SUBJECT",
                            "required": True,
                            "allowed_values": ["scene:current"],
                            "prebound": None,
                        }
                    ],
                }
            ],
            "permitted_decisions": ["RESOLVE", "ASK_PLAYER"],
        }


class FakeAssessment:
    def __init__(self, reason=None):
        self.model_required_because = reason


class FakeResult:
    def __init__(self, route, reason=None):
        self.route = route
        self.assessment = FakeAssessment(reason)


def route_player_semantics(conn, *, declaration, provider, **kwargs):
    assert kwargs["allowed_action_families"] == ("SEARCH_INSPECT",)
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fixture_legacy_migration'"
    ).fetchone()
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fixture_numbered_migration'"
    ).fetchone()
    if declaration.startswith("residual"):
        try:
            provider.resolve(FakeRequest(declaration))
        except RuntimeError as exc:
            if str(exc) != "ML_LAB_CAPTURE_ONLY":
                raise
        return FakeResult("PROVIDER_ERROR", "FAMILY:UNKNOWN_COMMITTED_FAMILY")
    return FakeResult("NO_ACTION")
"""


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
    (package / "db.py").write_text(_FAKE_DB, encoding="utf-8")
    (package / "semantic_routing.py").write_text(_FAKE_ROUTING, encoding="utf-8")
    (package / "semantic_residual.py").write_text("# materialization marker\n", encoding="utf-8")
    (package / "semantic_dispatch.py").write_text(_FAKE_SEMANTIC_DISPATCH, encoding="utf-8")
    (package / "turn_orchestrator.py").write_text(_FAKE_TURN_ORCHESTRATOR, encoding="utf-8")
    (data / "semantic_family_registry.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fake routing contract")
    return repo, _git(repo, "rev-parse", "HEAD")


def _fixture(declaration: str) -> dict[str, object]:
    return {
        "fixture_id": "fixture-1",
        "fixture_kind": "NON_CANON_FIXTURE",
        "production_data": False,
        "declaration": declaration,
        "actor_id": "tester",
        "audience": "TABLE",
        "snapshot_revision": "fixture-revision-1",
        "facts": [
            {
                "key": "session.scene",
                "value": "A non-canon empty test room.",
                "source": "fixture:test",
            }
        ],
    }


def test_fixture_export_captures_exact_residual_request_in_fresh_process(
    tmp_path: Path,
) -> None:
    repo, commit = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    receipt = PhaseAFixtureExporter(workspace).export(
        repository=repo,
        ref="HEAD",
        fixture=_fixture("residual unfamiliar wording"),
    )

    assert receipt.status == "RESIDUAL_EXPORTED"
    assert receipt.commit_sha == commit
    assert receipt.route == "SEMANTIC_REQUIRED"
    assert receipt.failed_deterministic_stage == "FAMILY:UNKNOWN_COMMITTED_FAMILY"
    assert receipt.request is not None
    assert receipt.request["version"] == "semantic-residual-v2"
    assert receipt.request_sha256
    assert receipt.fresh_process is True
    assert receipt.ephemeral_sqlite_only is True
    assert receipt.network_access_allowed is False
    assert receipt.resolver_dispatch_available is False
    assert receipt.authority_mutation_allowed is False

    persisted = json.loads(
        workspace.artifacts.resolve(receipt.receipt_artifact_digest).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["request"] == receipt.request
    assert persisted["request_sha256"] == receipt.request_sha256


def test_fixture_export_preserves_zero_model_terminal_route(tmp_path: Path) -> None:
    repo, _ = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    receipt = PhaseAFixtureExporter(workspace).export(
        repository=repo,
        ref="HEAD",
        fixture=_fixture("deterministic cancellation"),
    )

    assert receipt.status == "TERMINATED_BEFORE_RESIDUAL"
    assert receipt.route == "NO_ACTION"
    assert receipt.request is None
    assert receipt.request_sha256 is None


@pytest.mark.parametrize(
    "mutation",
    (
        {"production_data": True},
        {"allowed_action_families": ["HARM_TARGET"]},
        {
            "facts": [
                {
                    "key": "campaign.secret",
                    "value": "must not enter fixture export",
                    "source": "production",
                }
            ]
        },
    ),
)
def test_fixture_export_refuses_production_or_non_fixture_facts(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    repo, _ = _fake_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    fixture = _fixture("residual wording")
    fixture.update(mutation)

    with pytest.raises(ValueError):
        PhaseAFixtureExporter(workspace).export(
            repository=repo,
            ref="HEAD",
            fixture=fixture,
        )
