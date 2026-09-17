from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, QUrl, Signal, Slot

from ml_lab.adapters.builtin import contract_capture_descriptor_for
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import ContractSnapshot
from ml_lab.storage.workspace import Workspace


class _ContractSignals(QObject):
    completed = Signal(int, str)
    failed = Signal(int, str)


class _ContractCaptureOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        project_id: str,
        adapter_id: str,
        repository: Path,
        ref: str,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.project_id = project_id
        self.adapter_id = adapter_id
        self.repository = repository
        self.ref = ref
        self.signals = _ContractSignals()

    @Slot()
    def run(self) -> None:
        try:
            descriptor = contract_capture_descriptor_for(self.adapter_id)
            if descriptor is None:
                raise RuntimeError(
                    f"Adapter {self.adapter_id!r} does not declare a capture contract."
                )
            workspace = Workspace.open(self.workspace_root)
            snapshot = ContractSnapshotService(workspace).capture(
                project_id=self.project_id,
                adapter_id=descriptor.adapter_id,
                adapter_version=descriptor.adapter_version,
                repository=self.repository,
                ref=self.ref,
                contract_version=descriptor.contract_version,
                files=descriptor.files,
            )
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.signals.completed.emit(self.context_token, snapshot.id)


class ContractSnapshotsController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_snapshot_id = ""
        self._busy = False
        self._busy_message = ""
        self._context_token = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._context_token += 1
        self._pool.clear()
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_snapshot_id = ""
        self._busy = False
        self._busy_message = ""
        self.changed.emit()

    def clear_project(self) -> None:
        self._context_token += 1
        self._pool.clear()
        self._workspace = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_snapshot_id = ""
        self._busy = False
        self._busy_message = ""
        self.changed.emit()

    def shutdown(self) -> None:
        self._context_token += 1
        self._pool.clear()
        self._pool.waitForDone(2500)

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(bool, notify=changed)
    def busy(self) -> bool:
        return self._busy

    @Property(str, notify=changed)
    def busyMessage(self) -> str:
        return self._busy_message

    @Property(bool, notify=changed)
    def captureSupported(self) -> bool:
        try:
            return contract_capture_descriptor_for(self._adapter_id) is not None
        except KeyError:
            return False

    @Property(str, notify=changed)
    def captureMessage(self) -> str:
        if not self.hasProject:
            return "Select a project first."
        try:
            descriptor = contract_capture_descriptor_for(self._adapter_id)
        except KeyError:
            descriptor = None
        if descriptor is None:
            return (
                "This adapter has no declared contract capture descriptor. "
                "The generic host will not invent one."
            )
        return (
            f"{descriptor.adapter_id} v{descriptor.adapter_version} · "
            f"contract {descriptor.contract_version} · {len(descriptor.files)} declared files"
        )

    @Property(str, notify=changed)
    def suggestedRepository(self) -> str:
        snapshots = self._snapshots()
        return snapshots[0].repo_path if snapshots else ""

    @Property(list, notify=changed)
    def snapshotRows(self) -> list[dict[str, object]]:
        return [_snapshot_row(item) for item in self._snapshots()]

    @Property(dict, notify=changed)
    def selectedSnapshot(self) -> dict[str, object]:
        snapshot = self._selected_snapshot()
        if snapshot is None or self._workspace is None:
            return {}
        manifest = ContractSnapshotService(self._workspace).manifest(snapshot.id)
        files = manifest.get("files", [])
        return {
            **_snapshot_row(snapshot),
            "repositoryPath": snapshot.repo_path,
            "repositoryIdentity": snapshot.repo_identity,
            "workingTreeDirty": bool(manifest.get("working_tree_dirty_at_capture", False)),
            "fileCount": len(files) if isinstance(files, list) else 0,
        }

    @Property(list, notify=changed)
    def selectedFiles(self) -> list[dict[str, object]]:
        snapshot = self._selected_snapshot()
        if snapshot is None or self._workspace is None:
            return []
        manifest = ContractSnapshotService(self._workspace).manifest(snapshot.id)
        raw_files = manifest.get("files", [])
        if not isinstance(raw_files, list):
            return []
        rows: list[dict[str, object]] = []
        for raw in raw_files:
            if not isinstance(raw, dict):
                continue
            rows.append(
                {
                    "path": str(raw.get("path", "")),
                    "role": str(raw.get("role", "")),
                    "required": bool(raw.get("required", False)),
                    "sha256": str(raw.get("sha256", "")),
                    "sizeBytes": int(raw.get("size_bytes", 0)),
                }
            )
        return rows

    @Slot(str)
    def selectSnapshot(self, snapshot_id: str) -> None:
        if any(item.id == snapshot_id for item in self._snapshots()):
            self._selected_snapshot_id = snapshot_id
            self.changed.emit()

    @Slot(str, str)
    def capture(self, repository: str, ref: str) -> None:
        if not self._workspace or not self._project_id:
            self.operationFailed.emit("Contract snapshot", "Open a project first.")
            return
        try:
            descriptor = contract_capture_descriptor_for(self._adapter_id)
        except KeyError as exc:
            self.operationFailed.emit("Contract snapshot", str(exc))
            return
        if descriptor is None:
            self.operationFailed.emit(
                "Contract snapshot",
                "This adapter does not declare a contract capture descriptor.",
            )
            return
        if self._busy:
            self.operationFailed.emit(
                "Contract snapshot",
                "A contract capture is already running.",
            )
            return
        repo_path = _url_or_path(repository)
        clean_ref = ref.strip()
        if not clean_ref:
            self.operationFailed.emit("Contract snapshot", "A Git ref is required.")
            return

        self._busy = True
        self._busy_message = f"Freezing committed contract bytes from {clean_ref}…"
        self.changed.emit()
        worker = _ContractCaptureOperation(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            project_id=self._project_id,
            adapter_id=self._adapter_id,
            repository=repo_path,
            ref=clean_ref,
        )
        worker.signals.completed.connect(self._capture_completed)
        worker.signals.failed.connect(self._capture_failed)
        self._pool.start(worker)

    @Slot(int, str)
    def _capture_completed(self, context_token: int, snapshot_id: str) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        self._selected_snapshot_id = snapshot_id
        self.changed.emit()
        self.operationCompleted.emit("Contract snapshot frozen from committed Git bytes")

    @Slot(int, str)
    def _capture_failed(self, context_token: int, error: str) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        self.changed.emit()
        self.operationFailed.emit("Contract snapshot", error)

    def _snapshots(self) -> list[ContractSnapshot]:
        if not self._workspace or not self._project_id:
            return []
        return ContractSnapshotService(self._workspace).list_for_project(self._project_id)

    def _selected_snapshot(self) -> ContractSnapshot | None:
        if not self._selected_snapshot_id:
            return None
        return next(
            (item for item in self._snapshots() if item.id == self._selected_snapshot_id),
            None,
        )


def _snapshot_row(item: ContractSnapshot) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "adapterId": item.adapter_id,
        "adapterVersion": item.adapter_version,
        "contractVersion": item.contract_version,
        "commitSha": item.commit_sha,
        "commitShort": item.commit_sha[:12],
        "compatibilitySignature": item.compatibility_signature,
        "manifestSha256": item.manifest_artifact_digest,
        "createdAt": item.created_at,
    }


def _url_or_path(value: str) -> Path:
    if value.startswith("file:"):
        return Path(QUrl(value).toLocalFile())
    return Path(value)
