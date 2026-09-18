from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ml_lab.core.models import utc_now_iso
from ml_lab.storage.database import Database


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    digest: str
    size_bytes: int
    media_type: str
    path: Path


class ArtifactStore:
    def __init__(self, root: Path, database: Database):
        self.root = root
        self.database = database
        self.sha_root = root / "sha256"
        self.sha_root.mkdir(parents=True, exist_ok=True)

    def _target(self, digest: str) -> Path:
        return self.sha_root / digest[:2] / digest[2:4] / digest

    def commit_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        target = self._target(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temp_path: Path | None = None
            try:
                with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(data)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                if hashlib.sha256(temp_path.read_bytes()).hexdigest() != digest:
                    raise OSError("Artifact hash verification failed before commit.")
                os.replace(temp_path, target)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return self._register(target, digest, len(data), media_type, metadata or {})

    def commit_file(
        self,
        source: Path,
        *,
        media_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        digest, size = _hash_file(source)
        target = self._target(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temp_path: Path | None = None
            try:
                with NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                    temp_path = Path(tmp.name)
                with source.open("rb") as src, temp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                copied_digest, copied_size = _hash_file(temp_path)
                if copied_digest != digest or copied_size != size:
                    raise OSError("Artifact changed or failed verification during commit.")
                os.replace(temp_path, target)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return self._register(target, digest, size, media_type, metadata or {})

    def resolve(self, digest: str) -> Path:
        path = self._target(digest)
        if not path.exists():
            raise FileNotFoundError(f"Artifact {digest} is not present.")
        actual, _ = _hash_file(path)
        if actual != digest:
            raise OSError(f"Artifact {digest} is corrupt.")
        return path

    def _register(
        self,
        path: Path,
        digest: str,
        size: int,
        media_type: str,
        metadata: dict[str, Any],
    ) -> ArtifactRef:
        relative = path.relative_to(self.root).as_posix()
        metadata_json = json.dumps(metadata, sort_keys=True)
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO artifacts"
                "(digest,size_bytes,media_type,relative_path,created_at,metadata_json) "
                "VALUES(?,?,?,?,?,?)",
                (digest, size, media_type, relative, utc_now_iso(), metadata_json),
            )
        return ArtifactRef(digest, size, media_type, path)


def _hash_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size
