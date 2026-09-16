from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from ml_lab.core.config import user_config_dir
from ml_lab.core.models import TERMINAL_JOB_STATUSES
from ml_lab.diagnostics.logging_setup import configure_logging
from ml_lab.jobs.manager import JobManager
from ml_lab.jobs.worker import run_worker
from ml_lab.storage.workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ML Lab")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a non-GUI installation smoke test.",
    )
    parser.add_argument("--job-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--qml-smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    if args.smoke_test:
        return smoke_test()
    if args.job_smoke_test:
        return job_smoke_test()
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
