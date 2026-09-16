from __future__ import annotations

from pathlib import Path

from ml_lab.diagnostics.hardware import HardwareInfo, collect_hardware_info
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace


class LabServices:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.jobs = JobManager(workspace.root, workspace.database, workspace.artifacts)
        self.jobs.reconcile_startup()

    def diagnostics(self) -> HardwareInfo:
        return collect_hardware_info(self.workspace.root)

    def close(self) -> None:
        self.jobs.shutdown()


def open_or_create_workspace(path: Path, *, create: bool) -> LabServices:
    workspace = Workspace.create(path) if create else Workspace.open(path)
    return LabServices(workspace)
