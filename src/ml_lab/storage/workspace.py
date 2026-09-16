from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path

from ml_lab.core.models import Project, ProjectState, utc_now_iso
from ml_lab.storage.artifacts import ArtifactStore
from ml_lab.storage.database import Database

WORKSPACE_MARKER = ".ml-lab-workspace.json"
WORKSPACE_DIRECTORIES = (
    "artifacts",
    "jobs",
    "cache",
    "exports",
    "logs",
    "extensions/adapters",
    "extensions/runtimes",
)


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.database = Database(self.root / "lab.db")
        self.artifacts = ArtifactStore(self.root / "artifacts", self.database)

    @classmethod
    def create(cls, root: Path) -> Workspace:
        root = root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        marker = root / WORKSPACE_MARKER
        if not marker.exists():
            marker.write_text(
                json.dumps({"format": 1, "created_at": utc_now_iso()}, indent=2),
                encoding="utf-8",
            )
        for child in WORKSPACE_DIRECTORIES:
            (root / child).mkdir(parents=True, exist_ok=True)
        workspace = cls(root)
        workspace.database.migrate()
        return workspace

    @classmethod
    def open(cls, root: Path) -> Workspace:
        root = root.expanduser().resolve()
        if not (root / WORKSPACE_MARKER).exists():
            raise ValueError(f"{root} is not an ML Lab workspace.")
        workspace = cls(root)
        workspace.database.migrate()
        return workspace

    def create_project(
        self,
        name: str,
        adapter_id: str = "generic",
        description: str = "",
    ) -> Project:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Project name is required.")
        now = utc_now_iso()
        project = Project(
            id=str(uuid.uuid4()),
            name=clean_name,
            adapter_id=adapter_id.strip() or "generic",
            description=description.strip(),
            state=ProjectState.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO projects(id,name,adapter_id,description,state,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    project.id,
                    project.name,
                    project.adapter_id,
                    project.description,
                    project.state.value,
                    project.created_at,
                    project.updated_at,
                ),
            )
        return project

    def list_projects(self, *, include_archived: bool = False) -> list[Project]:
        query = "SELECT * FROM projects"
        params: tuple[object, ...] = ()
        if not include_archived:
            query += " WHERE state=?"
            params = (ProjectState.ACTIVE.value,)
        query += " ORDER BY updated_at DESC, name COLLATE NOCASE"
        with self.database.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            Project(
                id=row["id"],
                name=row["name"],
                adapter_id=row["adapter_id"],
                description=row["description"],
                state=ProjectState(row["state"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def archive_project(self, project_id: str) -> None:
        now = utc_now_iso()
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE projects SET state=?, updated_at=? WHERE id=?",
                (ProjectState.ARCHIVED.value, now, project_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown project {project_id}")

    def project_dicts(self) -> list[dict[str, object]]:
        return [
            {**asdict(project), "state": project.state.value}
            for project in self.list_projects()
        ]
