import subprocess
from pathlib import Path

import pytest

from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.storage.workspace import Workspace


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    contract = repo / "contract.txt"
    contract.write_text("committed-v1\n", encoding="utf-8")
    _git(repo, "add", "contract.txt")
    _git(repo, "commit", "-m", "contract v1")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_snapshot_reads_commit_not_dirty_worktree(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    (repo / "contract.txt").write_text("dirty-v2\n", encoding="utf-8")

    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Contract", adapter_id="test-adapter")
    service = ContractSnapshotService(workspace)
    snapshot = service.capture(
        project_id=project.id,
        adapter_id="test-adapter",
        adapter_version="1.0.0",
        repository=repo,
        ref="HEAD",
        contract_version="test-v1",
        files=(ContractFileSpec("contract.txt", "schema"),),
    )

    assert snapshot.commit_sha == commit
    manifest = service.manifest(snapshot.id)
    assert manifest["working_tree_dirty_at_capture"] is True
    files = manifest["files"]
    assert isinstance(files, list)
    assert len(files) == 1
    file_entry = files[0]
    assert isinstance(file_entry, dict)
    digest = file_entry["sha256"]
    assert isinstance(digest, str)
    frozen = workspace.artifacts.resolve(digest).read_text(encoding="utf-8")
    assert frozen == "committed-v1\n"
    assert (repo / "contract.txt").read_text(encoding="utf-8") == "dirty-v2\n"
    assert _git(repo, "status", "--porcelain")


def test_identical_contract_capture_is_content_identity_stable(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Contract", adapter_id="test-adapter")
    service = ContractSnapshotService(workspace)
    kwargs = {
        "project_id": project.id,
        "adapter_id": "test-adapter",
        "adapter_version": "1.0.0",
        "repository": repo,
        "ref": "HEAD",
        "contract_version": "test-v1",
        "files": (ContractFileSpec("contract.txt", "schema"),),
    }

    first = service.capture(**kwargs)
    second = service.capture(**kwargs)
    assert second.id == first.id
    assert second.compatibility_signature == first.compatibility_signature
    assert len(service.list_for_project(project.id)) == 1


def test_snapshot_rejects_adapter_mismatch(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Contract", adapter_id="expected")
    service = ContractSnapshotService(workspace)
    with pytest.raises(ValueError, match="Project adapter"):
        service.capture(
            project_id=project.id,
            adapter_id="wrong",
            adapter_version="1",
            repository=repo,
            ref="HEAD",
            contract_version="v1",
            files=(ContractFileSpec("contract.txt", "schema"),),
        )
