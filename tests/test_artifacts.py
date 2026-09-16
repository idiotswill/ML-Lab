import hashlib
from pathlib import Path

import pytest

from ml_lab.storage.workspace import Workspace


def test_artifact_commit_is_content_addressed_and_verified(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    payload = b"same bytes forever"
    expected = hashlib.sha256(payload).hexdigest()
    first = workspace.artifacts.commit_bytes(payload, media_type="text/plain")
    second = workspace.artifacts.commit_bytes(payload, media_type="text/plain")
    assert first.digest == expected == second.digest
    assert first.path == second.path
    assert workspace.artifacts.resolve(expected).read_bytes() == payload


def test_corrupt_artifact_is_rejected(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    artifact = workspace.artifacts.commit_bytes(b"original")
    artifact.path.write_bytes(b"tampered")
    with pytest.raises(IOError):
        workspace.artifacts.resolve(artifact.digest)
