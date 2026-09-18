from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast
from urllib.parse import urlparse

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION
from ml_lab.adapters.phase_a_reference import PhaseAReferenceValidator
from ml_lab.core.process import application_command
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace

_APP_ROOT = "frankenhomie-asterra-v0.9.0"


class PhaseALocalProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        receipt_artifact_digest: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.receipt_artifact_digest = receipt_artifact_digest


@dataclass(frozen=True, slots=True)
class LocalProviderProposal:
    proposal: dict[str, object]
    receipt_artifact_digest: str
    latency_ms: float


class PhaseALocalProviderRunner:
    """Execute the pinned Frankenhomie residual provider against loopback only.

    The child receives committed source bytes, one already-exported residual request,
    and an explicit local model endpoint. SQLite is process-wide forbidden. The child
    calls only ``LocalResidualSemanticProvider.resolve``; resolver dispatch is never
    invoked. Authoritative proposal acceptance remains a separate reference-validator
    step in the Lab evaluator.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        repository: Path,
        ref: str,
        model: str,
        endpoint: str,
        timeout_seconds: float,
    ) -> None:
        clean_model = model.strip()
        if not clean_model:
            raise ValueError("A local model name is required.")
        if len(clean_model) > 240:
            raise ValueError("Local model name must be at most 240 characters.")
        self.endpoint = validate_loopback_endpoint(endpoint)
        self.model = clean_model
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 900.0))
        self.workspace = workspace
        self.repository = repository.expanduser().resolve()
        self.ref = ref
        self.reference = PhaseAReferenceValidator(workspace)
        self.jobs_root = workspace.root / "cache" / "provider-jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def resolve(self, request: object) -> dict[str, object]:
        return self.propose(request).proposal

    def propose(self, request: object) -> LocalProviderProposal:
        materialized = self.reference.materialize(self.repository, self.ref)
        request_sha = _sha256_json(request)
        with tempfile.TemporaryDirectory(
            prefix=f"provider-{materialized.commit_sha[:12]}-",
            dir=self.jobs_root,
        ) as temp_dir:
            root = Path(temp_dir)
            input_path = root / "request.json"
            receipt_path = root / "receipt.json"
            spec_path = root / "spec.json"
            input_path.write_text(canonical_json(request) + "\n", encoding="utf-8")
            spec_path.write_text(
                canonical_json(
                    {
                        "format_version": 1,
                        "commit_sha": materialized.commit_sha,
                        "source_root": str(materialized.source_root),
                        "request_path": str(input_path),
                        "receipt_path": str(receipt_path),
                        "model": self.model,
                        "endpoint": self.endpoint,
                        "timeout_seconds": self.timeout_seconds,
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
            child_timeout = self.timeout_seconds * 2.0 + 30.0
            try:
                completed = subprocess.run(
                    application_command("--phase-a-provider-child", str(spec_path)),
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=child_timeout,
                    cwd=root,
                    env=env,
                    creationflags=flags,
                )
            except subprocess.TimeoutExpired as exc:
                decoded = _provider_error_receipt(
                    commit_sha=materialized.commit_sha,
                    model=self.model,
                    endpoint=self.endpoint,
                    request_sha=request_sha,
                    error_code="LOCAL_PROVIDER_CHILD_TIMEOUT",
                    error_message=str(exc),
                )
            else:
                if receipt_path.is_file():
                    decoded = _read_object(receipt_path)
                else:
                    message = completed.stderr[-2000:] or completed.stdout[-2000:]
                    decoded = _provider_error_receipt(
                        commit_sha=materialized.commit_sha,
                        model=self.model,
                        endpoint=self.endpoint,
                        request_sha=request_sha,
                        error_code="LOCAL_PROVIDER_NO_RECEIPT",
                        error_message=message or f"child exited with {completed.returncode}",
                    )

        _validate_provider_receipt(
            decoded,
            commit_sha=materialized.commit_sha,
            model=self.model,
            endpoint=self.endpoint,
            request_sha=request_sha,
        )
        artifact = self.workspace.artifacts.commit_bytes(
            (canonical_json(decoded) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.phase-a-local-provider-receipt+json",
            metadata={
                "commit_sha": materialized.commit_sha,
                "contract_version": PHASE_A_CONTRACT_VERSION,
                "model": self.model,
                "endpoint": self.endpoint,
                "status": str(decoded.get("status")),
            },
        )
        if decoded.get("status") != "PROPOSED":
            raise PhaseALocalProviderError(
                str(decoded.get("error_message") or "Local provider failed."),
                error_code=str(decoded.get("error_code") or "LOCAL_PROVIDER_ERROR"),
                receipt_artifact_digest=artifact.digest,
            )
        raw_proposal = decoded.get("proposal")
        if not isinstance(raw_proposal, dict):
            raise PhaseALocalProviderError(
                "Local provider receipt did not contain a proposal object.",
                error_code="LOCAL_PROVIDER_INVALID_RECEIPT",
                receipt_artifact_digest=artifact.digest,
            )
        latency = decoded.get("latency_ms")
        latency_ms = float(latency) if isinstance(latency, (int, float)) else 0.0
        return LocalProviderProposal(
            proposal={str(key): value for key, value in raw_proposal.items()},
            receipt_artifact_digest=artifact.digest,
            latency_ms=latency_ms,
        )


def run_local_provider_child(spec_path: Path) -> int:
    """Fresh-process provider entry point with loopback network and no SQLite."""
    try:
        spec = _read_object(spec_path)
        commit_sha = _required_text(spec, "commit_sha")
        source_root = Path(_required_text(spec, "source_root")).resolve()
        request_path = Path(_required_text(spec, "request_path")).resolve()
        receipt_path = Path(_required_text(spec, "receipt_path")).resolve()
        model = _required_text(spec, "model")
        endpoint = validate_loopback_endpoint(_required_text(spec, "endpoint"))
        raw_timeout = spec.get("timeout_seconds", 120.0)
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            raise ValueError("timeout_seconds must be numeric")
        timeout_seconds = max(1.0, min(float(raw_timeout), 900.0))
        request = _read_json_value(request_path)
        request_sha = _sha256_json(request)
    except Exception as exc:
        return _write_child_bootstrap_error(spec_path, exc)

    _install_provider_sandbox()
    started = time.perf_counter()
    try:
        app_root = source_root / _APP_ROOT
        if not app_root.is_dir():
            raise FileNotFoundError(app_root)
        sys.path.insert(0, str(app_root))
        residual_module = __import__("asterra.semantic_residual", fromlist=["*"])
        transport_module = __import__("asterra.turn_shadow", fromlist=["*"])
        request_type = cast(Any, vars(residual_module)["ResidualSemanticRequest"])
        provider_type = cast(Any, vars(residual_module)["LocalResidualSemanticProvider"])
        transport_type = cast(Any, vars(transport_module)["LocalChatCompletionsTurnProvider"])
        request_model = request_type.model_validate(request)
        transport = transport_type(
            model=model,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            max_tokens=700,
        )
        provider = provider_type(transport)
        proposal_model = provider.resolve(request_model)
        proposal = proposal_model.model_dump(mode="json")
        receipt = _provider_receipt(
            status="PROPOSED",
            commit_sha=commit_sha,
            model=model,
            endpoint=endpoint,
            request_sha=request_sha,
            proposal=proposal,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code=None,
            error_message=None,
        )
    except Exception as exc:
        receipt = _provider_receipt(
            status="ERROR",
            commit_sha=commit_sha,
            model=model,
            endpoint=endpoint,
            request_sha=request_sha,
            proposal=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code=_provider_exception_code(exc),
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return 0 if receipt["status"] == "PROPOSED" else 3


def validate_loopback_endpoint(endpoint: str) -> str:
    clean = endpoint.strip()
    if not clean:
        raise ValueError("A local provider endpoint is required.")
    parsed = urlparse(clean)
    if parsed.scheme != "http":
        raise ValueError("Local provider endpoint must use http://.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Local provider endpoint must not contain credentials.")
    hostname = parsed.hostname
    if hostname is None or not _is_loopback_host(hostname):
        raise ValueError("Local provider endpoint must target localhost or a loopback IP.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Local provider endpoint has an invalid port.") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Local provider endpoint has an invalid port.")
    if not parsed.path or parsed.path == "/":
        raise ValueError("Local provider endpoint must include the chat-completions path.")
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError("Local provider endpoint must not contain params, query, or fragment.")
    return clean


def _install_provider_sandbox() -> None:
    original_connection = cast(Any, socket.create_connection)

    def forbidden_sqlite(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("ML_LAB_PROVIDER_DB_ACCESS_FORBIDDEN")

    def loopback_connection(address: object, *args: object, **kwargs: object) -> socket.socket:
        if not isinstance(address, tuple) or len(address) < 2:
            raise RuntimeError("ML_LAB_PROVIDER_NON_LOOPBACK_NETWORK_FORBIDDEN")
        host = str(address[0])
        if not _is_loopback_host(host):
            raise RuntimeError("ML_LAB_PROVIDER_NON_LOOPBACK_NETWORK_FORBIDDEN")
        return cast(socket.socket, original_connection(address, *args, **kwargs))

    sqlite3.connect = cast(Any, forbidden_sqlite)
    socket.create_connection = cast(Any, loopback_connection)


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _provider_receipt(
    *,
    status: str,
    commit_sha: str,
    model: str,
    endpoint: str,
    request_sha: str,
    proposal: object | None,
    latency_ms: float,
    error_code: str | None,
    error_message: str | None,
) -> dict[str, object]:
    proposal_sha = _sha256_json(proposal) if proposal is not None else None
    return {
        "format_version": 1,
        "status": status,
        "commit_sha": commit_sha,
        "contract_version": PHASE_A_CONTRACT_VERSION,
        "model": model,
        "endpoint": endpoint,
        "request_sha256": request_sha,
        "proposal_sha256": proposal_sha,
        "proposal": proposal,
        "latency_ms": round(latency_ms, 3),
        "fresh_process": True,
        "authority_mutation_allowed": False,
        "database_access_allowed": False,
        "network_policy": "LOOPBACK_HTTP_ONLY",
        "resolver_dispatch_invoked": False,
        "error_code": error_code,
        "error_message": error_message,
    }


def _provider_error_receipt(
    *,
    commit_sha: str,
    model: str,
    endpoint: str,
    request_sha: str,
    error_code: str,
    error_message: str,
) -> dict[str, object]:
    return _provider_receipt(
        status="ERROR",
        commit_sha=commit_sha,
        model=model,
        endpoint=endpoint,
        request_sha=request_sha,
        proposal=None,
        latency_ms=0.0,
        error_code=error_code,
        error_message=error_message[:2000],
    )


def _provider_exception_code(exc: Exception) -> str:
    text = str(exc)
    if "ML_LAB_PROVIDER_DB_ACCESS_FORBIDDEN" in text:
        return "LOCAL_PROVIDER_DB_ACCESS_FORBIDDEN"
    if "ML_LAB_PROVIDER_NON_LOOPBACK_NETWORK_FORBIDDEN" in text:
        return "LOCAL_PROVIDER_NON_LOOPBACK_NETWORK_FORBIDDEN"
    if isinstance(exc, ModuleNotFoundError):
        return "LOCAL_PROVIDER_IMPORT_DEPENDENCY_MISSING"
    return "LOCAL_PROVIDER_ERROR"


def _validate_provider_receipt(
    receipt: dict[str, object],
    *,
    commit_sha: str,
    model: str,
    endpoint: str,
    request_sha: str,
) -> None:
    expected = {
        "commit_sha": commit_sha,
        "contract_version": PHASE_A_CONTRACT_VERSION,
        "model": model,
        "endpoint": endpoint,
        "request_sha256": request_sha,
        "fresh_process": True,
        "authority_mutation_allowed": False,
        "database_access_allowed": False,
        "network_policy": "LOOPBACK_HTTP_ONLY",
        "resolver_dispatch_invoked": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"Local provider receipt identity mismatch for {key}.")
    if receipt.get("status") not in {"PROPOSED", "ERROR"}:
        raise RuntimeError("Local provider receipt has an unknown status.")


def _write_child_bootstrap_error(spec_path: Path, exc: Exception) -> int:
    try:
        spec = _read_object(spec_path)
        receipt_raw = spec.get("receipt_path")
        if not isinstance(receipt_raw, str) or not receipt_raw.strip():
            return 4
        receipt_path = Path(receipt_raw)
        commit_sha = str(spec.get("commit_sha", "UNKNOWN"))
        model = str(spec.get("model", "UNKNOWN"))
        endpoint = str(spec.get("endpoint", "UNKNOWN"))
        request_raw = spec.get("request_path")
        request_path = Path(request_raw) if isinstance(request_raw, str) else Path()
        request = _read_json_value(request_path) if request_path.is_file() else None
        request_sha = _sha256_json(request) if request is not None else "UNKNOWN"
        receipt_path.write_text(
            canonical_json(
                _provider_error_receipt(
                    commit_sha=commit_sha,
                    model=model,
                    endpoint=endpoint,
                    request_sha=request_sha,
                    error_code="LOCAL_PROVIDER_CHILD_BOOTSTRAP_ERROR",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return 4


def _read_object(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return {str(key): value for key, value in decoded.items()}


def _read_json_value(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_text(value: dict[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
