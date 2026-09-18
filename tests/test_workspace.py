from pathlib import Path

import pytest

from ml_lab.core.models import ProjectState
from ml_lab.storage.database import SCHEMA_VERSION
from ml_lab.storage.workspace import WORKSPACE_MARKER, Workspace


def test_workspace_create_project_archive(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    assert (workspace.root / WORKSPACE_MARKER).exists()
    assert workspace.database.schema_version() == SCHEMA_VERSION

    project = workspace.create_project(" Semantic Lab ", "phase-a", " bounded semantics ")
    assert project.name == "Semantic Lab"
    assert project.state is ProjectState.ACTIVE
    assert workspace.list_projects()[0].id == project.id

    workspace.archive_project(project.id)
    assert workspace.list_projects() == []
    archived = workspace.list_projects(include_archived=True)
    assert archived[0].state is ProjectState.ARCHIVED


def test_open_requires_marker(tmp_path: Path) -> None:
    root = tmp_path / "not-workspace"
    root.mkdir()
    with pytest.raises(ValueError):
        Workspace.open(root)
