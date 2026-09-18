from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Never, cast

from pydantic import ValidationError

from ml_lab.adapters.phase_a import PHASE_A_CONTRACT_VERSION
from ml_lab.core.process import application_command
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace

_APP_ROOT = "frankenhomie-asterra-v0.9.0"
_ARCHIVE_PATHS = (f"{_APP_ROOT}/asterra", f"{_APP_ROOT}/data")


@dataclass(frozen=True, slots=True)
class MaterializedPhaseAContract:
    commit_sha: str
    source_root: Path


@dataclass(frozen=True, slots=True)
class ReferenceValidationReceipt:
    status: str
    commit_sha: str
    contract_version: str
    request_sha256: str
    proposal_sha256: str
    validator: str
    fresh_process: bool
    authority_mutation_allowed: bool
    error_code: str | None
    error_message: str | None
    receipt_artifact_digest: str


class PhaseAReferenceValidator:
    """Run Frankenhomie's pinned residual validator without campaign authority.

    Source is materialized from committed Git bytes only. The child gets an isolated
    ASTERRA_HOME, no local config, and a process-wide SQLite connection veto. It calls
    only ``run_residual_semantics`` with a recorded proposal provider; no dispatcher or
    resolver is exposed by this harness.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.cache_root = workspace.root / "cache" / "reference" / "phase-a"
        self.jobs_root = workspace.root / "cache" / "reference-jobs"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        repository: Path,
        ref: str,
    ) -> MaterializedPhaseAContract:
        repo = repository.expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(repo)
        commit_sha = _git_text(repo, "rev-parse", f"{ref}^{{commit}}").strip()
        if len(commit_sha) != 40:
            raise RuntimeError(f"Could not resolve a full Git commit for {ref!r}.")
        destination = self.cache_root / commit_sha
        marker = destination / ".ml-lab-reference.json"
        if _materialization_matches(marker, commit_sha):
            return MaterializedPhaseAContract(commit_sha, destination)

        with tempfile.TemporaryDirectory(
            prefix=f"materialize-{commit_sha[:12]}-",
            dir=self.cache_root,
        ) as temp_dir:
            temp = Path(temp_dir)
            archive_path = temp / "source.tar"
            extract_root = temp / "source"
            extract_root.mkdir()
            command = [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                commit_sha,
                *_ARCHIVE_PATHS,
            ]
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            _extract_regular_files(archive_path, extract_root)
            app_root = extract_root / _APP_ROOT
            if not (app_root / "asterra" / "semantic_residual.py").is_file():
                raise FileNotFoundError("Pinned contract is missing semantic_residual.py")
            if not (app_root / "data").is_dir():
                raise FileNotFoundError("Pinned contract is missing its data directory")
            (extract_root / ".ml-lab-reference.json").write_text(
                canonical_json(
                    {
                        "format_version": 1,
                        "commit_sha": commit_sha,
                        "contract_version": PHASE_A_CONTRACT_VERSION,
                        "archive_paths": list(_ARCHIVE_PATHS),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(extract_root), str(destination))
        return MaterializedPhaseAContract(commit_sha, destination)

    def validate(
        self,
        *,
        repository: Path,
        ref: str,
        request: object,
        proposal: object,
        timeout_seconds: float = 30.0,
    ) -> ReferenceValidationReceipt:
        materialized = self.materialize(repository, ref)
        input_payload = {"request": request, "proposal": proposal}
        request_sha = _sha256_json(request)
        proposal_sha = _sha256_json(proposal)
        with tempfile.TemporaryDirectory(
            prefix=f"validate-{materialized.commit_sha[:12]}-",
            dir=self.jobs_root,
        ) as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.json"
            receipt_path = root / "receipt.json"
            spec_path = root / "spec.json"
            input_path.write_text(
                canonical_json(input_payload) + "\n",
                encoding="utf-8",
            )
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
                    application_command("--phase-a-validator-child", str(spec_path)),
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
                    request_sha,
                    proposal_sha,
                    "REFERENCE_VALIDATOR_TIMEOUT",
                    str(exc),
                )
            else:
                if receipt_path.is_file():
                    decoded = _read_object(receipt_path)
                else:
                    message = completed.stderr[-2000:] or completed.stdout[-2000:]
                    decoded = _error_receipt(
                        materialized.commit_sha,
                        request_sha,
                        proposal_sha,
                        "REFERENCE_VALIDATOR_NO_RECEIPT",
                        message or f"child exited with {completed.returncode}",
                    )

        _validate_receipt_identity(
            decoded,
            commit_sha=materialized.commit_sha,
            request_sha=request_sha,
            proposal_sha=proposal_sha,
        )
        artifact = self.workspace.artifacts.commit_bytes(
            (canonical_json(decoded) + "\n").encode("utf-8"),
            media_type="application/vnd.ml-lab.phase-a-reference-receipt+json",
            metadata={
                "commit_sha": materialized.commit_sha,
                "contract_version": PHASE_A_CONTRACT_VERSION,
                "status": str(decoded.get("status")),
            },
        )
        return ReferenceValidationReceipt(
            status=str(decoded["status"]),
            commit_sha=str(decoded["commit_sha"]),
            contract_version=str(decoded["contract_version"]),
            request_sha256=str(decoded["request_sha256"]),
            proposal_sha256=str(decoded["proposal_sha256"]),
            validator=str(decoded["validator"]),
            fresh_process=bool(decoded["fresh_process"]),
            authority_mutation_allowed=bool(decoded["authority_mutation_allowed"]),
            error_code=(
                str(decoded["error_code"])
                if decoded.get("error_code") is not None
                else None
            ),
            error_message=(
                str(decoded["error_message"])
                if decoded.get("error_message") is not None
                else None
            ),
            receipt_artifact_digest=artifact.digest,
        )


def run_reference_validator_child(spec_path: Path) -> int:
    """Fresh-process entry point. Never receives a workspace or campaign DB path."""
    try:
        spec = _read_object(spec_path)
        commit_sha = _required_text(spec, "commit_sha")
        source_root = Path(_required_text(spec, "source_root")).resolve()
        input_path = Path(_required_text(spec, "input_path")).resolve()
        receipt_path = Path(_required_text(spec, "receipt_path")).resolve()
        payload = _read_object(input_path)
        request = payload.get("request")
        proposal = payload.get("proposal")
        request_sha = _sha256_json(request)
        proposal_sha = _sha256_json(proposal)
    except Exception as exc:
        return _write_child_bootstrap_error(spec_path, exc)

    receipt: dict[str, object]
    _install_reference_sandbox()
    try:
        app_root = source_root / _APP_ROOT
        if not app_root.is_dir():
            raise FileNotFoundError(app_root)
        sys.path.insert(0, str(app_root))
        module = importlib.import_module("asterra.semantic_residual")
        namespace = vars(module)
        request_type = cast(Any, namespace["ResidualSemanticRequest"])
        decision_type = cast(Any, namespace["ResidualSemanticDecisionV2"])
        residual_error_type = cast(type[Exception], namespace["ResidualSemanticError"])
        run_residual = cast(Any, namespace["run_residual_semantics"])
        request_model = request_type.model_validate(request)
        decision_model = decision_type.model_validate(proposal)

        class RecordedProposalProvider:
            provider_name = "ml-lab-recorded-proposal"

            def resolve(self, _request: object) -> object:
                return decision_model

        try:
            accepted = run_residual(RecordedProposalProvider(), request_model)
        except residual_error_type as exc:
            receipt = _child_receipt(
                status="REJECTED",
                commit_sha=commit_sha,
                request_sha=request_sha,
                proposal_sha=proposal_sha,
                error_code=str(vars(exc).get("code", type(exc).__name__)),
                error_message=str(exc),
            )
        else:
            receipt = _child_receipt(
                status="ACCEPTED",
                commit_sha=commit_sha,
                request_sha=request_sha,
                proposal_sha=proposal_sha,
                error_code=None,
                error_message=None,
                accepted_decision=accepted.model_dump(mode="json"),
            )
    except ValidationError as exc:
        receipt = _child_receipt(
            status="REJECTED",
            commit_sha=commit_sha,
            request_sha=request_sha,
            proposal_sha=proposal_sha,
            error_code="CONTRACT_VALIDATION_ERROR",
            error_message=str(exc)[:2000],
        )
    except Exception as exc:
        receipt = _child_receipt(
            status="ERROR",
            commit_sha=commit_sha,
            request_sha=request_sha,
            proposal_sha=proposal_sha,
            error_code=_exception_code(exc),
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return 0 if receipt["status"] in {"ACCEPTED", "REJECTED"} else 3


def _install_reference_sandbox() -> None:
    def forbidden_sqlite(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("ML_LAB_REFERENCE_DB_ACCESS_FORBIDDEN")

    def forbidden_network(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("ML_LAB_REFERENCE_NETWORK_ACCESS_FORBIDDEN")

    sqlite3.connect = cast(Any, forbidden_sqlite)
    socket.create_connection = cast(Any, forbidden_network)


def _child_receipt(
    *,
    status: str,
    commit_sha: str,
    request_sha: str,
    proposal_sha: str,
    error_code: str | None,
    error_message: str | None,
    accepted_decision: object | None = None,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "status": status,
        "commit_sha": commit_sha,
        "contract_version": PHASE_A_CONTRACT_VERSION,
        "request_sha256": request_sha,
        "proposal_sha256": proposal_sha,
        "validator": "frankenhomie.run_residual_semantics",
        "fresh_process": True,
        "authority_mutation_allowed": False,
        "database_access_allowed": False,
        "network_access_allowed": False,
        "resolver_dispatch_available": False,
        "accepted_decision": accepted_decision,
        "error_code": error_code,
        "error_message": error_message,
    }


def _error_receipt(
    commit_sha: str,
    request_sha: str,
    proposal_sha: str,
    error_code: str,
    error_message: str,
) -> dict[str, object]:
    return _child_receipt(
        status="ERROR",
        commit_sha=commit_sha,
        request_sha=request_sha,
        proposal_sha=proposal_sha,
        error_code=error_code,
        error_message=error_message[:2000],
    )


def _exception_code(exc: Exception) -> str:
    text = str(exc)
    if "ML_LAB_REFERENCE_DB_ACCESS_FORBIDDEN" in text:
        return "REFERENCE_DB_ACCESS_FORBIDDEN"
    if "ML_LAB_REFERENCE_NETWORK_ACCESS_FORBIDDEN" in text:
        return "REFERENCE_NETWORK_ACCESS_FORBIDDEN"
    if isinstance(exc, ModuleNotFoundError):
        return "REFERENCE_IMPORT_DEPENDENCY_MISSING"
    return "REFERENCE_VALIDATOR_ERROR"


def _write_child_bootstrap_error(spec_path: Path, exc: Exception) -> int:
    try:
        raw = _read_object(spec_path)
        receipt_path = Path(str(raw.get("receipt_path", "")))
        commit_sha = str(raw.get("commit_sha", "UNKNOWN"))
        if receipt_path:
            receipt_path.write_text(
                canonical_json(
                    _error_receipt(
                        commit_sha,
                        "UNKNOWN",
                        "UNKNOWN",
                        "REFERENCE_CHILD_BOOTSTRAP_ERROR",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
    except Exception:
        pass
    return 4


def _extract_regular_files(archive_path: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ValueError(f"Unsafe Git archive member {member.name!r}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported Git archive member type {member.name!r}")
            target = destination.joinpath(*pure.parts).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError(f"Git archive member escaped destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read Git archive member {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _materialization_matches(marker: Path, commit_sha: str) -> bool:
    if not marker.is_file():
        return False
    try:
        decoded = _read_object(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        decoded.get("commit_sha") == commit_sha
        and decoded.get("contract_version") == PHASE_A_CONTRACT_VERSION
        and (marker.parent / _APP_ROOT / "asterra" / "semantic_residual.py").is_file()
    )


def _validate_receipt_identity(
    receipt: dict[str, object],
    *,
    commit_sha: str,
    request_sha: str,
    proposal_sha: str,
) -> None:
    if receipt.get("commit_sha") != commit_sha:
        raise ValueError("Reference receipt commit identity mismatch.")
    if receipt.get("contract_version") != PHASE_A_CONTRACT_VERSION:
        raise ValueError("Reference receipt contract version mismatch.")
    if receipt.get("request_sha256") != request_sha:
        raise ValueError("Reference receipt request digest mismatch.")
    if receipt.get("proposal_sha256") != proposal_sha:
        raise ValueError("Reference receipt proposal digest mismatch.")
    if receipt.get("fresh_process") is not True:
        raise ValueError("Reference validation must come from a fresh process.")
    if receipt.get("authority_mutation_allowed") is not False:
        raise ValueError("Reference validation receipt granted mutation authority.")


def _read_object(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): value for key, value in decoded.items()}


def _required_text(value: dict[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{key} is required")
    return raw


def _sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _git_text(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout
