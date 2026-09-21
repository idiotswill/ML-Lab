from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION, PhaseAResidualAdapter
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.core.process import application_command
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace

_APP_ROOT = "frankenhomie-asterra-v0.9.0"
_ALLOWED_AUDIENCES = {"TABLE", "PC_PRIVATE", "GM_ONLY"}
_ALLOWED_STATUSES = {"RESIDUAL_EXPORTED", "TERMINATED_BEFORE_RESIDUAL", "ERROR"}


@dataclass(frozen=True, slots=True)
class PhaseAFixtureExportReceipt:
    status: str
    commit_sha: str
    contract_version: str
    fixture_id: str
    fixture_sha256: str
    route: str | None
    failed_deterministic_stage: str | None
    request_sha256: str | None
    request: dict[str, object] | None
    fresh_process: bool
    ephemeral_sqlite_only: bool
    network_access_allowed: bool
    resolver_dispatch_available: bool
    authority_mutation_allowed: bool
    error_code: str | None
    error_message: str | None
    receipt_artifact_digest: str


class PhaseAFixtureExporter:
    """Export the exact pinned routing boundary from a non-canon fixture."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.reference = PhaseAReferenceValidator(workspace)
        self.jobs_root = workspace.root / "cache" / "fixture-export" / "phase-a"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        *,
        repository: Path,
        ref: str,
        fixture: Mapping[str, object],
        timeout_seconds: float = 30.0,
    ) -> PhaseAFixtureExportReceipt:
        normalized = _validate_fixture(fixture)
        materialized = self.reference.materialize(repository, ref)
        fixture_sha = _sha256_json(normalized)

        with tempfile.TemporaryDirectory(
            prefix=f"fixture-{materialized.commit_sha[:12]}-",
            dir=self.jobs_root,
        ) as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.json"
            receipt_path = root / "receipt.json"
            spec_path = root / "spec.json"
            input_path.write_text(canonical_json(normalized) + "\n", encoding="utf-8")
            spec_path.write_text(
                canonical_json(
                    {
                        "format_version": 1,
                        "commit_sha": materialized.commit_sha,
                        "source_root": str(materialized.source_root),
                        "input_path": str(input_path),
                        "receipt_path": str(receipt_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            isolated_home = root / "asterra-home"
            isolated_home.mkdir()
            env = os.environ.copy()
            for key in tuple(env):
                if key.startswith("ASTERRA_"):
                    env.pop(key, None)
            env.update(
                {
                    "ASTERRA_HOME": str(isolated_home),
                    "ASTERRA_CONFIG_FILE": str(root / "no-config.json"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                completed = subprocess.run(
                    application_command("--phase-a-fixture-export-child", str(spec_path)),
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    cwd=root,
                    env=env,
                    creationflags=flags,
                )
            except subprocess.TimeoutExpired as exc:
                decoded = _error_receipt(
                    materialized.commit_sha,
                    normalized,
                    "FIXTURE_EXPORT_TIMEOUT",
                    str(exc),
                )
            else:
                if receipt_path.is_file():
                    decoded = _read_object(receipt_path)
                else:
                    message = completed.stderr[-2000:] or completed.stdout[-2000:]
                    decoded = _error_receipt(
                        materialized.commit_sha,
                        normalized,
                        "FIXTURE_EXPORT_NO_RECEIPT",
                        message or f"child exited with {completed.returncode}",
                    )

        _validate_receipt_identity(
            decoded,
            commit_sha=materialized.commit_sha,
            fixture_id=str(normalized["fixture_id"]),
            fixture_sha=fixture_sha,
        )
        request = decoded.get("request")
        if request is not None:
            if not isinstance(request, dict):
                raise ValueError("Fixture export request must be an object")
            PhaseAResidualAdapter().validate_exported_request(request)
            request_sha = _sha256_json(request)
            if decoded.get("request_sha256") != request_sha:
                raise ValueError("Fixture export request hash mismatch")
        elif decoded.get("request_sha256") is not None:
            raise ValueError("Fixture export cannot hash a missing request")

        artifact = self.workspace.artifacts.commit_bytes(
            (canonical_json(decoded) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.phase-a-fixture-export+json",
            metadata={
                "commit_sha": materialized.commit_sha,
                "contract_version": PHASE_A_CONTRACT_VERSION,
                "fixture_id": str(normalized["fixture_id"]),
                "status": str(decoded.get("status")),
            },
        )
        return PhaseAFixtureExportReceipt(
            status=str(decoded["status"]),
            commit_sha=str(decoded["commit_sha"]),
            contract_version=str(decoded["contract_version"]),
            fixture_id=str(decoded["fixture_id"]),
            fixture_sha256=str(decoded["fixture_sha256"]),
            route=(str(decoded["route"]) if decoded.get("route") is not None else None),
            failed_deterministic_stage=(
                str(decoded["failed_deterministic_stage"])
                if decoded.get("failed_deterministic_stage") is not None
                else None
            ),
            request_sha256=(
                str(decoded["request_sha256"])
                if decoded.get("request_sha256") is not None
                else None
            ),
            request=request if isinstance(request, dict) else None,
            fresh_process=bool(decoded["fresh_process"]),
            ephemeral_sqlite_only=bool(decoded["ephemeral_sqlite_only"]),
            network_access_allowed=bool(decoded["network_access_allowed"]),
            resolver_dispatch_available=bool(decoded["resolver_dispatch_available"]),
            authority_mutation_allowed=bool(decoded["authority_mutation_allowed"]),
            error_code=(
                str(decoded["error_code"]) if decoded.get("error_code") is not None else None
            ),
            error_message=(
                str(decoded["error_message"])
                if decoded.get("error_message") is not None
                else None
            ),
            receipt_artifact_digest=artifact.digest,
        )


def run_phase_a_fixture_export_child(spec_path: Path) -> int:
    """Fresh-process export using pinned routing, capture-only ML, and RAM SQLite."""
    try:
        spec = _read_object(spec_path)
        commit_sha = _required_text(spec, "commit_sha")
        source_root = Path(_required_text(spec, "source_root")).resolve()
        input_path = Path(_required_text(spec, "input_path")).resolve()
        receipt_path = Path(_required_text(spec, "receipt_path")).resolve()
        fixture = _validate_fixture(_read_object(input_path))
        fixture_sha = _sha256_json(fixture)
    except Exception as exc:
        return _write_bootstrap_error(spec_path, exc)

    try:
        _install_network_veto()
        app_root = source_root / _APP_ROOT
        if not app_root.is_dir():
            raise FileNotFoundError(app_root)
        sys.path.insert(0, str(app_root))
        db_module = __import__("asterra.db", fromlist=["SCHEMA"])
        routing = __import__("asterra.semantic_routing", fromlist=["route_player_semantics"])
        orchestrator = __import__("asterra.turn_orchestrator", fromlist=["TurnFact"])

        conn = sqlite3.connect(":memory:")
        try:
            conn.row_factory = sqlite3.Row
            conn.executescript(str(getattr(db_module, "SCHEMA")))
            database_rows = conn.execute("PRAGMA database_list").fetchall()
            if any(str(row[2] or "") for row in database_rows):
                raise RuntimeError("Fixture export opened a file-backed SQLite database")

            facts = tuple(
                orchestrator.TurnFact.model_validate(raw)
                for raw in fixture["facts"]
            )

            class CaptureProvider:
                provider_name = "ml-lab-fixture-capture"

                def __init__(self) -> None:
                    self.request: object | None = None

                def resolve(self, request: object) -> object:
                    self.request = request
                    raise RuntimeError("ML_LAB_CAPTURE_ONLY")

            provider = CaptureProvider()
            result = routing.route_player_semantics(
                conn,
                session_id=1,
                actor_id=str(fixture["actor_id"]),
                device_id=f"fixture:{fixture['fixture_id']}",
                event_id=f"fixture:{fixture['fixture_id']}",
                declaration=str(fixture["declaration"]),
                audience=str(fixture["audience"]),
                facts=facts,
                snapshot_revision=str(fixture["snapshot_revision"]),
                allowed_action_families=tuple(fixture["allowed_action_families"]),
                provider=provider,
                combat_revision=0,
            )

            if provider.request is not None:
                request = provider.request.model_dump(mode="json")
                receipt = {
                    "status": "RESIDUAL_EXPORTED",
                    "commit_sha": commit_sha,
                    "contract_version": PHASE_A_CONTRACT_VERSION,
                    "fixture_id": fixture["fixture_id"],
                    "fixture_sha256": fixture_sha,
                    "route": "SEMANTIC_REQUIRED",
                    "failed_deterministic_stage": request["failed_deterministic_stage"],
                    "request_sha256": _sha256_json(request),
                    "request": request,
                    "fresh_process": True,
                    "ephemeral_sqlite_only": True,
                    "network_access_allowed": False,
                    "resolver_dispatch_available": False,
                    "authority_mutation_allowed": False,
                    "error_code": None,
                    "error_message": None,
                }
            else:
                assessment = result.assessment
                receipt = {
                    "status": "TERMINATED_BEFORE_RESIDUAL",
                    "commit_sha": commit_sha,
                    "contract_version": PHASE_A_CONTRACT_VERSION,
                    "fixture_id": fixture["fixture_id"],
                    "fixture_sha256": fixture_sha,
                    "route": result.route,
                    "failed_deterministic_stage": assessment.model_required_because,
                    "request_sha256": None,
                    "request": None,
                    "fresh_process": True,
                    "ephemeral_sqlite_only": True,
                    "network_access_allowed": False,
                    "resolver_dispatch_available": False,
                    "authority_mutation_allowed": False,
                    "error_code": None,
                    "error_message": None,
                }
        finally:
            conn.close()
    except Exception as exc:
        receipt = _error_receipt(
            commit_sha,
            fixture,
            type(exc).__name__,
            f"{type(exc).__name__}: {exc}"[:2000],
        )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return 0 if receipt["status"] in {
        "RESIDUAL_EXPORTED",
        "TERMINATED_BEFORE_RESIDUAL",
    } else 3


def _validate_fixture(fixture: Mapping[str, object]) -> dict[str, object]:
    fixture_id = _required_text(fixture, "fixture_id")
    declaration = _required_text(fixture, "declaration")
    actor_id = _required_text(fixture, "actor_id")
    audience = _required_text(fixture, "audience")
    if audience not in _ALLOWED_AUDIENCES:
        raise ValueError("fixture audience must be TABLE, PC_PRIVATE, or GM_ONLY")
    if fixture.get("fixture_kind") != "NON_CANON_FIXTURE":
        raise ValueError("Phase A fixture export requires NON_CANON_FIXTURE")
    if fixture.get("production_data") is not False:
        raise ValueError("Phase A fixture export refuses Production/campaign data")
    snapshot_revision = _required_text(fixture, "snapshot_revision")

    raw_families = fixture.get("allowed_action_families")
    if not isinstance(raw_families, (list, tuple)) or not raw_families:
        raise ValueError("allowed_action_families must be a non-empty list")
    families = tuple(str(value) for value in raw_families)
    if any(not value for value in families) or len(set(families)) != len(families):
        raise ValueError("allowed_action_families must contain unique non-empty strings")

    raw_facts = fixture.get("facts")
    if not isinstance(raw_facts, (list, tuple)):
        raise ValueError("facts must be a list")
    facts: list[dict[str, object]] = []
    for raw in raw_facts:
        if not isinstance(raw, Mapping):
            raise ValueError("fixture facts must be objects")
        source = raw.get("source")
        if not isinstance(source, str) or not source.startswith("fixture"):
            raise ValueError("fixture facts must use a fixture source")
        facts.append(dict(raw))

    return {
        "fixture_id": fixture_id,
        "fixture_kind": "NON_CANON_FIXTURE",
        "production_data": False,
        "declaration": declaration,
        "actor_id": actor_id,
        "audience": audience,
        "snapshot_revision": snapshot_revision,
        "allowed_action_families": list(families),
        "facts": facts,
    }


def _install_network_veto() -> None:
    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("REFERENCE_NETWORK_ACCESS_FORBIDDEN")

    socket.create_connection = forbidden  # type: ignore[assignment]
    socket.socket.connect = forbidden  # type: ignore[assignment]


def _error_receipt(
    commit_sha: str,
    fixture: Mapping[str, object],
    code: str,
    message: str,
) -> dict[str, object]:
    return {
        "status": "ERROR",
        "commit_sha": commit_sha,
        "contract_version": PHASE_A_CONTRACT_VERSION,
        "fixture_id": str(fixture.get("fixture_id") or "unknown"),
        "fixture_sha256": _sha256_json(fixture),
        "route": None,
        "failed_deterministic_stage": None,
        "request_sha256": None,
        "request": None,
        "fresh_process": True,
        "ephemeral_sqlite_only": True,
        "network_access_allowed": False,
        "resolver_dispatch_available": False,
        "authority_mutation_allowed": False,
        "error_code": code,
        "error_message": message[:2000],
    }


def _write_bootstrap_error(spec_path: Path, exc: Exception) -> int:
    try:
        spec = _read_object(spec_path)
        receipt_path = Path(_required_text(spec, "receipt_path")).resolve()
        commit_sha = str(spec.get("commit_sha") or "unknown")
        input_path = Path(str(spec.get("input_path") or ""))
        fixture = _read_object(input_path) if input_path.is_file() else {}
        receipt = _error_receipt(
            commit_sha,
            fixture,
            "FIXTURE_EXPORT_BOOTSTRAP_ERROR",
            f"{type(exc).__name__}: {exc}",
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    except Exception:
        pass
    return 3


def _validate_receipt_identity(
    receipt: Mapping[str, object],
    *,
    commit_sha: str,
    fixture_id: str,
    fixture_sha: str,
) -> None:
    if receipt.get("status") not in _ALLOWED_STATUSES:
        raise ValueError("Unknown fixture export status")
    if receipt.get("commit_sha") != commit_sha:
        raise ValueError("Fixture export commit mismatch")
    if receipt.get("contract_version") != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Fixture export contract mismatch")
    if receipt.get("fixture_id") != fixture_id:
        raise ValueError("Fixture export id mismatch")
    if receipt.get("fixture_sha256") != fixture_sha:
        raise ValueError("Fixture export fixture hash mismatch")
    if receipt.get("fresh_process") is not True:
        raise ValueError("Fixture export must run in a fresh process")
    if receipt.get("ephemeral_sqlite_only") is not True:
        raise ValueError("Fixture export must use ephemeral SQLite only")
    if receipt.get("network_access_allowed") is not False:
        raise ValueError("Fixture export cannot allow network access")
    if receipt.get("resolver_dispatch_available") is not False:
        raise ValueError("Fixture export cannot expose resolver dispatch")
    if receipt.get("authority_mutation_allowed") is not False:
        raise ValueError("Fixture export cannot allow authority mutation")


def _read_object(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return decoded


def _required_text(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _sha256_json(value: object) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
