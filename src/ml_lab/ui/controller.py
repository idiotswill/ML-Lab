from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    Property,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QGuiApplication

from ml_lab.core.config import AppConfig, user_config_dir
from ml_lab.diagnostics.hardware import HardwareInfo, collect_hardware_info
from ml_lab.extensions.catalog import ExtensionCatalog
from ml_lab.services import LabServices, open_or_create_workspace
from ml_lab.ui.data_studio import DataStudioController
from ml_lab.ui.experiments import ExperimentsController

LOGGER = logging.getLogger(__name__)


class _HardwareProbeSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class _HardwareProbe(QRunnable):
    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.signals = _HardwareProbeSignals()

    @Slot()
    def run(self) -> None:
        workspace_key = str(self.workspace_root)
        try:
            hardware = collect_hardware_info(self.workspace_root)
        except Exception as exc:
            self.signals.failed.emit(workspace_key, f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(workspace_key, hardware)


class AppController(QObject):
    workspaceChanged = Signal()
    projectsChanged = Signal()
    selectionChanged = Signal()
    jobsChanged = Signal()
    diagnosticsChanged = Signal()
    extensionsChanged = Signal()
    themeChanged = Signal()
    errorRaised = Signal(str, str)
    noticeRaised = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._config_path = user_config_dir() / "config.json"
        self._config = AppConfig.load(self._config_path)
        self._services: LabServices | None = None
        self._catalog: ExtensionCatalog | None = None
        self._diagnostics: dict[str, object] = {}
        self._selected_project_id = ""
        self._data_studio = DataStudioController(self)
        self._data_studio.operationCompleted.connect(self.noticeRaised.emit)
        self._data_studio.operationFailed.connect(self.errorRaised.emit)
        self._experiments = ExperimentsController(self)
        self._experiments.operationCompleted.connect(self.noticeRaised.emit)
        self._experiments.operationFailed.connect(self.errorRaised.emit)
        self._diagnostic_pool = QThreadPool(self)
        self._diagnostic_pool.setMaxThreadCount(1)
        self._diagnostics_loading = False
        self._poller = QTimer(self)
        self._poller.setInterval(400)
        self._poller.timeout.connect(self._poll_jobs)
        color_scheme_changed = QGuiApplication.styleHints().colorSchemeChanged
        color_scheme_changed.connect(lambda _scheme: self.themeChanged.emit())
        if self._config.recent_workspace:
            candidate = Path(self._config.recent_workspace)
            try:
                self._activate(open_or_create_workspace(candidate, create=False))
            except Exception as exc:
                LOGGER.warning("Unable to reopen workspace %s: %s", candidate, exc)

    @Property(bool, notify=workspaceChanged)
    def hasWorkspace(self) -> bool:
        return self._services is not None

    @Property(str, notify=workspaceChanged)
    def workspacePath(self) -> str:
        return str(self._services.workspace.root) if self._services else ""

    @Property(QObject, constant=True)
    def dataStudio(self) -> QObject:
        return self._data_studio

    @Property(QObject, constant=True)
    def experiments(self) -> QObject:
        return self._experiments

    @Property(list, notify=projectsChanged)
    def projects(self) -> list[dict[str, object]]:
        return self._services.workspace.project_dicts() if self._services else []

    @Property(dict, notify=selectionChanged)
    def selectedProject(self) -> dict[str, object]:
        if not self._services or not self._selected_project_id:
            return {}
        for project in self._services.workspace.project_dicts():
            if project["id"] == self._selected_project_id:
                return project
        return {}

    @Property(list, notify=jobsChanged)
    def jobs(self) -> list[dict[str, object]]:
        if not self._services:
            return []
        return [
            {
                "id": item.id,
                "taskType": item.task_type,
                "status": item.status.value,
                "progress": item.progress,
                "message": item.message,
                "correlationId": item.correlation_id or "",
                "error": item.error or "",
                "resultDigest": item.result_artifact_digest or "",
            }
            for item in self._services.jobs.list_recent(25)
        ]

    @Property(dict, notify=diagnosticsChanged)
    def diagnostics(self) -> dict[str, object]:
        return self._diagnostics

    @Property(list, notify=extensionsChanged)
    def adapters(self) -> list[dict[str, str]]:
        return self._catalog.adapter_options() if self._catalog else []

    @Property(str, notify=themeChanged)
    def themeMode(self) -> str:
        return self._config.theme

    @Property(bool, notify=themeChanged)
    def darkTheme(self) -> bool:
        if self._config.theme == "dark":
            return True
        if self._config.theme == "light":
            return False
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark

    @Slot(str, bool)
    def openWorkspace(self, path: str, create: bool) -> None:
        try:
            normalized = _url_or_path(path)
            services = open_or_create_workspace(normalized, create=create)
            if self._services:
                self._services.close()
            self._activate(services)
            self._config.recent_workspace = str(normalized)
            self._config.save(self._config_path)
            self.workspaceChanged.emit()
            self.projectsChanged.emit()
            self.jobsChanged.emit()
            self.noticeRaised.emit("Workspace ready")
        except Exception as exc:
            LOGGER.exception("Workspace open/create failed")
            self.errorRaised.emit("Workspace error", str(exc))

    @Slot(str, str, str)
    def createProject(self, name: str, adapter_id: str, description: str) -> None:
        if not self._services:
            self.errorRaised.emit("No workspace", "Create or open a workspace first.")
            return
        registered_adapters = (
            {item["id"] for item in self._catalog.adapter_options()}
            if self._catalog
            else set()
        )
        if adapter_id not in registered_adapters:
            self.errorRaised.emit("Adapter error", f"Adapter '{adapter_id}' is not registered.")
            return
        try:
            project = self._services.workspace.create_project(name, adapter_id, description)
            self._selected_project_id = project.id
            self._bind_project_tools()
            self.projectsChanged.emit()
            self.selectionChanged.emit()
            self.noticeRaised.emit(f"Created project {name.strip()}")
        except Exception as exc:
            self.errorRaised.emit("Project error", str(exc))

    @Slot(str)
    def openProject(self, project_id: str) -> None:
        if not self._services:
            return
        project_ids = {
            str(item["id"]) for item in self._services.workspace.project_dicts()
        }
        if project_id not in project_ids:
            self.errorRaised.emit("Project error", "The selected project no longer exists.")
            return
        self._selected_project_id = project_id
        self._bind_project_tools()
        self.selectionChanged.emit()

    @Slot(str)
    def archiveProject(self, project_id: str) -> None:
        if not self._services:
            return
        try:
            self._services.workspace.archive_project(project_id)
            if self._selected_project_id == project_id:
                self._selected_project_id = ""
                self._data_studio.clear_project()
                self._experiments.clear_project()
                self.selectionChanged.emit()
            self.projectsChanged.emit()
            self.noticeRaised.emit("Project archived")
        except Exception as exc:
            self.errorRaised.emit("Project error", str(exc))

    @Slot()
    def runSelfTest(self) -> None:
        if not self._services:
            return
        try:
            self._services.jobs.start("core.self_test", {"steps": 12, "delay": 0.05})
            self.jobsChanged.emit()
        except Exception as exc:
            LOGGER.exception("Self-test job failed to start")
            self.errorRaised.emit("Job error", str(exc))

    @Slot(str)
    def cancelJob(self, job_id: str) -> None:
        if not self._services:
            return
        try:
            self._services.jobs.cancel(job_id)
            self.jobsChanged.emit()
        except Exception as exc:
            self.errorRaised.emit("Cancellation error", str(exc))

    @Slot(str)
    def setThemeMode(self, mode: str) -> None:
        if mode not in {"system", "dark", "light"}:
            self.errorRaised.emit("Theme error", f"Unsupported theme mode: {mode}")
            return
        self._config.theme = mode
        self._config.save(self._config_path)
        self.themeChanged.emit()

    @Slot()
    def refreshDiagnostics(self) -> None:
        self._refresh_diagnostics()

    @Slot()
    def openLogsFolder(self) -> None:
        log_dir = user_config_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))

    @Slot(str)
    def exportDiagnostics(self, path: str) -> None:
        if not self._services:
            return
        try:
            target = _url_or_path(path)
            if target.suffix.casefold() != ".zip":
                target = target.with_suffix(".zip")
            target.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "workspace": str(self._services.workspace.root),
                "diagnostics": self._diagnostics,
                "extensions": self._catalog.summary() if self._catalog else {},
                "jobs": self.jobs,
            }
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "diagnostics.json",
                    json.dumps(summary, indent=2, sort_keys=True),
                )
                app_log_dir = user_config_dir() / "logs"
                if app_log_dir.exists():
                    for log in sorted(app_log_dir.glob("*.log*")):
                        archive.write(log, f"app-logs/{log.name}")
                for job in self._services.jobs.list_recent(10):
                    for name in ("events.jsonl", "stderr.log", "job_spec.json"):
                        source = job.staging_dir / name
                        if source.exists():
                            archive.write(source, f"jobs/{job.id}/{name}")
            self.noticeRaised.emit(f"Diagnostics exported to {target.name}")
        except Exception as exc:
            LOGGER.exception("Diagnostics export failed")
            self.errorRaised.emit("Export error", str(exc))

    def shutdown(self) -> None:
        self._data_studio.shutdown()
        self._diagnostic_pool.clear()
        self._diagnostic_pool.waitForDone(2500)
        if self._services:
            self._services.close()

    def _activate(self, services: LabServices) -> None:
        self._services = services
        self._selected_project_id = ""
        self._data_studio.clear_project()
        self._experiments.clear_project()
        self._catalog = ExtensionCatalog(services.workspace.root)
        self._catalog.load()
        self._diagnostics_loading = False
        self._refresh_diagnostics()
        self._poller.start()
        self.extensionsChanged.emit()
        self.selectionChanged.emit()

    def _bind_project_tools(self) -> None:
        if not self._services or not self._selected_project_id:
            self._data_studio.clear_project()
            self._experiments.clear_project()
            return
        project = next(
            (
                item
                for item in self._services.workspace.project_dicts()
                if item["id"] == self._selected_project_id
            ),
            None,
        )
        if project is None:
            self._data_studio.clear_project()
            self._experiments.clear_project()
            return
        adapter_id = str(project["adapter_id"])
        self._data_studio.bind_project(
            self._services.workspace,
            self._selected_project_id,
            adapter_id,
        )
        self._experiments.bind_project(
            self._services.workspace,
            self._services.jobs,
            self._selected_project_id,
            adapter_id,
        )

    def _refresh_diagnostics(self) -> None:
        if not self._services:
            self._diagnostics = {}
            self._diagnostics_loading = False
            self.diagnosticsChanged.emit()
            return
        if self._diagnostics_loading:
            return

        self._diagnostics_loading = True
        self._diagnostics = {**self._diagnostics, "loading": True, "error": ""}
        self.diagnosticsChanged.emit()
        probe = _HardwareProbe(self._services.workspace.root)
        probe.signals.finished.connect(self._apply_hardware)
        probe.signals.failed.connect(self._diagnostics_failed)
        self._diagnostic_pool.start(probe)

    @Slot(str, object)
    def _apply_hardware(self, workspace_key: str, result: object) -> None:
        if not self._services or str(self._services.workspace.root) != workspace_key:
            return
        self._diagnostics_loading = False
        if not isinstance(result, HardwareInfo):
            self._diagnostics = {"loading": False, "error": "Invalid diagnostics result"}
            self.diagnosticsChanged.emit()
            return

        catalog: dict[str, Any] = (
            self._catalog.summary()
            if self._catalog
            else {"adapters": 0, "runtimes": 0, "errors": []}
        )
        self._diagnostics = {
            "loading": False,
            "error": "",
            "os": f"{result.os} {result.architecture}",
            "cpu": result.cpu,
            "logicalCpus": result.logical_cpus,
            "ram": _human_bytes(result.ram_bytes),
            "diskFree": _human_bytes(result.disk_free_bytes),
            "gpu": result.nvidia_gpu or "No NVIDIA GPU detected",
            "vram": (
                f"{result.nvidia_vram_mb / 1024:.1f} GB"
                if result.nvidia_vram_mb is not None
                else "—"
            ),
            "adapterCount": catalog["adapters"],
            "runtimeCount": catalog["runtimes"],
            "extensionErrors": len(catalog["errors"]),
        }
        self.diagnosticsChanged.emit()

    @Slot(str, str)
    def _diagnostics_failed(self, workspace_key: str, error: str) -> None:
        if not self._services or str(self._services.workspace.root) != workspace_key:
            return
        self._diagnostics_loading = False
        LOGGER.warning("Hardware diagnostics failed: %s", error)
        self._diagnostics = {**self._diagnostics, "loading": False, "error": error}
        self.diagnosticsChanged.emit()

    def _poll_jobs(self) -> None:
        if self._services:
            self._experiments.refresh()
            self.jobsChanged.emit()


def _url_or_path(value: str) -> Path:
    if value.startswith("file:"):
        return Path(QUrl(value).toLocalFile())
    return Path(value)


def _human_bytes(value: int | float | None) -> str:
    if value is None:
        return "Unknown"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TB"
