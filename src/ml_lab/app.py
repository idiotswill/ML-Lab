from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path

from ml_lab.core.config import user_config_dir
from ml_lab.core.models import TERMINAL_JOB_STATUSES, JobStatus
from ml_lab.diagnostics.logging_setup import configure_logging
from ml_lab.jobs.manager import JobManager
from ml_lab.jobs.worker import run_worker
from ml_lab.storage.workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ML Lab")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--prepare-interrupted-job",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--startup-probe-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a non-GUI installation smoke test.",
    )
    parser.add_argument("--job-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--restart-recovery-smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--performance-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--qml-smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    if args.prepare_interrupted_job:
        return prepare_interrupted_job(args.prepare_interrupted_job)
    if args.startup_probe_child:
        return startup_probe_child()
    if args.smoke_test:
        return smoke_test()
    if args.job_smoke_test:
        return job_smoke_test()
    if args.restart_recovery_smoke_test:
        return restart_recovery_smoke_test()
    if args.performance_probe:
        return performance_probe()
    if args.qml_smoke_test:
        return qml_smoke_test()
    return run_gui()


def smoke_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ml-lab-smoke-") as temp:
        workspace = Workspace.create(Path(temp) / "workspace")
        project = workspace.create_project("Smoke project", "generic")
        artifact = workspace.artifacts.commit_bytes(b"ml-lab-smoke")
        assert workspace.database.schema_version() == 1
        assert workspace.artifacts.resolve(artifact.digest).read_bytes() == b"ml-lab-smoke"
        assert workspace.list_projects()[0].id == project.id
        print(json.dumps({"ok": True, "project_id": project.id, "artifact": artifact.digest}))
    return 0


def job_smoke_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ml-lab-job-smoke-") as temp:
        workspace = Workspace.create(Path(temp) / "workspace")
        manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
        record = manager.start("core.self_test", {"steps": 2, "delay": 0.01})
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            record = manager.get(record.id)
            if record.status in TERMINAL_JOB_STATUSES:
                break
            time.sleep(0.03)
        manager.shutdown()
        if record.status.value != "COMPLETED" or not record.result_artifact_digest:
            print(json.dumps({"ok": False, "status": record.status.value, "error": record.error}))
            return 3
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": record.status.value,
                    "result_artifact": record.result_artifact_digest,
                }
            )
        )
    return 0


def prepare_interrupted_job(workspace_path: Path) -> int:
    """Start a real worker, publish its PID, then simulate an abrupt app death."""
    workspace = Workspace.create(workspace_path)
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    record = manager.start("core.self_test", {"steps": 600, "delay": 0.1})
    print(
        json.dumps(
            {
                "job_id": record.id,
                "pid": record.pid,
                "status": record.status.value,
            }
        ),
        flush=True,
    )
    # Deliberately bypass normal manager/application shutdown. This fixture exists
    # to prove startup reconciliation after a hard host-process interruption.
    os._exit(0)


def restart_recovery_smoke_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ml-lab-recovery-smoke-") as temp:
        root = Path(temp) / "workspace"
        fixture = subprocess.run(
            _application_command("--prepare-interrupted-job", str(root)),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if fixture.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "stage": "fixture",
                        "returncode": fixture.returncode,
                        "stderr": fixture.stderr[-1000:],
                    }
                )
            )
            return 4

        lines = [line for line in fixture.stdout.splitlines() if line.strip()]
        if not lines:
            print(json.dumps({"ok": False, "stage": "fixture", "error": "no fixture output"}))
            return 4
        details = json.loads(lines[-1])
        job_id = str(details["job_id"])
        worker_pid = int(details["pid"])

        workspace = Workspace.open(root)
        manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
        before = manager.get(job_id)
        _terminate_process(worker_pid)
        recovered = manager.reconcile_startup()
        after = manager.get(job_id)
        manager.shutdown()

        ok = (
            before.status is JobStatus.RUNNING
            and recovered == 1
            and after.status is JobStatus.INTERRUPTED
            and after.result_artifact_digest is None
        )
        print(
            json.dumps(
                {
                    "ok": ok,
                    "before": before.status.value,
                    "after": after.status.value,
                    "reconciled": recovered,
                }
            )
        )
        return 0 if ok else 5


def startup_probe_child() -> int:
    """Load the real QML shell and report the process working set at UI readiness."""
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    from ml_lab.ui.controller import AppController

    started = time.perf_counter()
    app = QGuiApplication(["ml-lab-startup-probe"])
    app.setOrganizationName("Frankenhomie")
    app.setOrganizationDomain("local.frankenhomie")
    app.setApplicationName("ML Lab")
    controller = AppController()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parent / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    app.processEvents()
    ok = bool(engine.rootObjects())
    payload = {
        "ok": ok,
        "qml_load_ms": round((time.perf_counter() - started) * 1000, 2),
        "working_set_bytes": _working_set_bytes(),
    }
    print(json.dumps(payload), flush=True)
    controller.shutdown()
    del engine
    del app
    return 0 if ok else 2


def performance_probe() -> int:
    """Measure process launch through QML readiness using this same build."""
    started = time.perf_counter()
    child = subprocess.run(
        _application_command("--startup-probe-child"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    lines = [line for line in child.stdout.splitlines() if line.strip()]
    if child.returncode != 0 or not lines:
        print(
            json.dumps(
                {
                    "ok": False,
                    "returncode": child.returncode,
                    "stderr": child.stderr[-1000:],
                }
            )
        )
        return 6
    details = json.loads(lines[-1])
    working_set = details.get("working_set_bytes")
    working_set_mb = (
        round(float(working_set) / (1024 * 1024), 2)
        if isinstance(working_set, (int, float))
        else None
    )
    within_startup = elapsed_ms <= 4000.0
    within_memory = working_set_mb is not None and working_set_mb <= 300.0
    print(
        json.dumps(
            {
                "ok": bool(details.get("ok")),
                "process_start_to_qml_ready_ms": elapsed_ms,
                "child_qml_load_ms": details.get("qml_load_ms"),
                "working_set_mb": working_set_mb,
                "hard_budget_startup_ms": 4000,
                "hard_budget_working_set_mb": 300,
                "within_hard_budgets": within_startup and within_memory,
            }
        )
    )
    # CI runners are evidence collectors, not the representative hardware named
    # by the product gate, so budget overages are reported rather than hidden or
    # converted into environment-dependent build failures.
    return 0 if details.get("ok") else 6


def qml_smoke_test() -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    from ml_lab.ui.controller import AppController

    started = time.perf_counter()
    app = QGuiApplication.instance() or QGuiApplication(["ml-lab-qml-smoke"])
    controller = AppController()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parent / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    ok = bool(engine.rootObjects())
    controller.shutdown()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    print(json.dumps({"ok": ok, "qml_load_ms": elapsed_ms, "qml": str(qml_path)}))
    del engine
    del app
    return 0 if ok else 2


def run_gui() -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    from ml_lab.ui.controller import AppController

    app = QGuiApplication(sys.argv)
    app.setOrganizationName("Frankenhomie")
    app.setOrganizationDomain("local.frankenhomie")
    app.setApplicationName("ML Lab")
    app.setApplicationDisplayName("Frankenhomie ML Lab")

    configure_logging(user_config_dir() / "logs")
    controller = AppController()
    app.aboutToQuit.connect(controller.shutdown)

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parent / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        return 2
    return app.exec()


def _application_command(*args: str) -> list[str]:
    executable = Path(sys.executable).name.casefold()
    python_names = {"python", "python3", "python.exe", "pythonw.exe", "pypy", "pypy3"}
    if executable not in python_names:
        return [sys.executable, *args]
    return [sys.executable, "-m", "ml_lab", *args]


def _terminate_process(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
        return
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def _working_set_bytes() -> int | None:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = windll.kernel32.GetCurrentProcess()
        if not windll.psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        ):
            return None
        return int(counters.WorkingSetSize)

    statm = Path("/proc/self/statm")
    sysconf = getattr(os, "sysconf", None)
    if statm.exists() and callable(sysconf):
        fields = statm.read_text(encoding="utf-8").split()
        if len(fields) >= 2:
            return int(fields[1]) * int(sysconf("SC_PAGE_SIZE"))
    return None
