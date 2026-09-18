import subprocess
from pathlib import Path

import pytest

from ml_lab.adapters.builtin import contract_capture_descriptor_for
from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID
from ml_lab.contracts.snapshot import ContractFileSpec, ContractSnapshotService
from ml_lab.storage.workspace import Workspace
from ml_lab.ui.contracts import _perform_contract_capture


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
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


def _phase_a_repo(tmp_path: Path) -> tuple[Path, str]:
    descriptor = contract_capture_descriptor_for(PHASE_A_ADAPTER_ID)
    assert descriptor is not None
    repo = tmp_path / "phase-a-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "ML Lab Test")
    _git(repo, "config", "user.email", "ml-lab@example.invalid")
    for index, spec in enumerate(descriptor.files):
        target = repo / spec.normalized_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"contract-file-{index}:{spec.role}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "phase a contract")
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


def test_builtin_contract_descriptor_is_adapter_owned() -> None:
    assert contract_capture_descriptor_for("generic") is None
    descriptor = contract_capture_descriptor_for(PHASE_A_ADAPTER_ID)
    assert descriptor is not None
    assert descriptor.adapter_id == PHASE_A_ADAPTER_ID
    assert descriptor.contract_version == "semantic-residual-v2"
    paths = {item.normalized_path() for item in descriptor.files}
    assert any(path.endswith("asterra/semantic_residual.py") for path in paths)
    assert any(path.endswith("docs/CORE_SUCCESSOR_ROADMAP.md") for path in paths)


def test_phase_a_app_capture_uses_full_declared_descriptor(tmp_path: Path) -> None:
    repo, commit = _phase_a_repo(tmp_path)
    workspace = Workspace.create(tmp_path / "workspace")
    project = workspace.create_project("Phase A", adapter_id=PHASE_A_ADAPTER_ID)

    snapshot = _perform_contract_capture(
        workspace_root=workspace.root,
        project_id=project.id,
        adapter_id=PHASE_A_ADAPTER_ID,
        repository=repo,
        ref="HEAD",
    )

    descriptor = contract_capture_descriptor_for(PHASE_A_ADAPTER_ID)
    assert descriptor is not None
    assert snapshot.commit_sha == commit
    assert snapshot.adapter_version == descriptor.adapter_version
    assert snapshot.contract_version == descriptor.contract_version
    manifest = ContractSnapshotService(workspace).manifest(snapshot.id)
    files = manifest["files"]
    assert isinstance(files, list)
    assert len(files) == len(descriptor.files)
    assert {str(item["role"]) for item in files if isinstance(item, dict)} == {
        item.role for item in descriptor.files
    }
