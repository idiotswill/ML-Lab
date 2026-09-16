from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ml_lab.core.models import ContractSnapshot, utc_now_iso
from ml_lab.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class ContractFileSpec:
    path: str
    role: str
    required: bool = True

    def normalized_path(self) -> str:
        candidate = PurePosixPath(self.path.replace("\\", "/"))
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError(f"Unsafe contract path {self.path!r}")
        return candidate.as_posix()


class ContractSnapshotService:
    """Freeze declared files from an explicit Git commit without checking it out.

    The service only uses read-only Git commands. Uncommitted working-tree content is
    deliberately excluded because file bytes come from ``git show <commit>:<path>``.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts

    def capture(
        self,
        *,
        project_id: str,
        adapter_id: str,
        adapter_version: str,
        repository: Path,
        ref: str,
        contract_version: str,
        files: tuple[ContractFileSpec, ...],
    ) -> ContractSnapshot:
        if not files:
            raise ValueError("At least one contract file is required.")
        root = repository.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        self._require_project_adapter(project_id, adapter_id)

        commit_sha = self._git_text(root, "rev-parse", f"{ref}^{{commit}}").strip()
        if len(commit_sha) != 40:
            raise RuntimeError(f"Could not resolve a full Git commit for {ref!r}.")
        repo_identity = self._repo_identity(root)
        working_tree_dirty = bool(self._git_text(root, "status", "--porcelain").strip())

        frozen_files: list[dict[str, object]] = []
        signature_files: list[dict[str, str]] = []
        for spec in files:
            path = spec.normalized_path()
            try:
                data = self._git_bytes(root, "show", f"{commit_sha}:{path}")
            except subprocess.CalledProcessError:
                if spec.required:
                    raise FileNotFoundError(
                        f"Required contract file {path!r} does not exist at {commit_sha}."
                    ) from None
                continue
            digest = hashlib.sha256(data).hexdigest()
            artifact = self.artifacts.commit_bytes(
                data,
                media_type="application/octet-stream",
                metadata={
                    "kind": "contract-file",
                    "project_id": project_id,
                    "adapter_id": adapter_id,
                    "commit_sha": commit_sha,
                    "path": path,
                    "role": spec.role,
                },
            )
            if artifact.digest != digest:
                raise OSError("Contract artifact digest mismatch.")
            frozen_files.append(
                {
                    "path": path,
                    "role": spec.role,
                    "required": spec.required,
                    "sha256": artifact.digest,
                    "size_bytes": artifact.size_bytes,
                }
            )
            signature_files.append({"path": path, "sha256": artifact.digest, "role": spec.role})

        signature_payload = {
            "format_version": 1,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "contract_version": contract_version,
            "files": sorted(signature_files, key=lambda item: item["path"]),
        }
        signature = hashlib.sha256(_canonical_bytes(signature_payload)).hexdigest()
        existing = self._by_signature(project_id, signature)
        if existing is not None:
            return existing

        created_at = utc_now_iso()
        manifest = {
            "format_version": 1,
            "project_id": project_id,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "contract_version": contract_version,
            "repository_identity": repo_identity,
            "repository_path": str(root),
            "commit_sha": commit_sha,
            "working_tree_dirty_at_capture": working_tree_dirty,
            "compatibility_signature": signature,
            "files": frozen_files,
            "created_at": created_at,
        }
        manifest_artifact = self.artifacts.commit_bytes(
            _canonical_bytes(manifest) + b"\n",
            media_type="application/vnd.ml-lab.contract-manifest+json",
            metadata={
                "kind": "contract-manifest",
                "project_id": project_id,
                "adapter_id": adapter_id,
                "commit_sha": commit_sha,
            },
        )
        snapshot = ContractSnapshot(
            id=str(uuid.uuid4()),
            project_id=project_id,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            repo_path=str(root),
            repo_identity=repo_identity,
            commit_sha=commit_sha,
            contract_version=contract_version,
            compatibility_signature=signature,
            manifest_artifact_digest=manifest_artifact.digest,
            created_at=created_at,
        )
        try:
            with self.database.transaction() as conn:
                conn.execute(
                    "INSERT INTO contract_snapshots"
                    "(id,project_id,adapter_id,adapter_version,repo_path,repo_identity,commit_sha,"
                    "contract_version,compatibility_signature,manifest_artifact_digest,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot.id,
                        snapshot.project_id,
                        snapshot.adapter_id,
                        snapshot.adapter_version,
                        snapshot.repo_path,
                        snapshot.repo_identity,
                        snapshot.commit_sha,
                        snapshot.contract_version,
                        snapshot.compatibility_signature,
                        snapshot.manifest_artifact_digest,
                        snapshot.created_at,
                    ),
                )
        except sqlite3.IntegrityError:
            raced = self._by_signature(project_id, signature)
            if raced is None:
                raise
            return raced
        return snapshot

    def get(self, snapshot_id: str) -> ContractSnapshot:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM contract_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown contract snapshot {snapshot_id}")
        return _snapshot_from_row(row)

    def list_for_project(self, project_id: str) -> list[ContractSnapshot]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM contract_snapshots WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def manifest(self, snapshot_id: str) -> dict[str, object]:
        snapshot = self.get(snapshot_id)
        path = self.artifacts.resolve(snapshot.manifest_artifact_digest)
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Contract manifest artifact is not a JSON object.")
        return {str(key): value for key, value in decoded.items()}

    def _require_project_adapter(self, project_id: str, adapter_id: str) -> None:
        with self.database.connection() as conn:
            row = conn.execute("SELECT adapter_id FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown project {project_id}")
        configured = str(row["adapter_id"])
        if configured != adapter_id:
            raise ValueError(
                f"Project adapter is {configured!r}; cannot capture contract for {adapter_id!r}."
            )

    def _by_signature(self, project_id: str, signature: str) -> ContractSnapshot | None:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM contract_snapshots WHERE project_id=? AND compatibility_signature=?",
                (project_id, signature),
            ).fetchone()
        return _snapshot_from_row(row) if row is not None else None

    @staticmethod
    def _repo_identity(root: Path) -> str:
        try:
            remote = ContractSnapshotService._git_text(
                root, "config", "--get", "remote.origin.url"
            ).strip()
        except subprocess.CalledProcessError:
            remote = ""
        return remote or root.as_posix()

    @staticmethod
    def _git_text(root: Path, *args: str) -> str:
        return ContractSnapshotService._git_bytes(root, *args).decode("utf-8", errors="strict")

    @staticmethod
    def _git_bytes(root: Path, *args: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        return completed.stdout


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _snapshot_from_row(row: sqlite3.Row) -> ContractSnapshot:
    return ContractSnapshot(
        id=row["id"],
        project_id=row["project_id"],
        adapter_id=row["adapter_id"],
        adapter_version=row["adapter_version"],
        repo_path=row["repo_path"],
        repo_identity=row["repo_identity"],
        commit_sha=row["commit_sha"],
        contract_version=row["contract_version"],
        compatibility_signature=row["compatibility_signature"],
        manifest_artifact_digest=row["manifest_artifact_digest"],
        created_at=row["created_at"],
    )
