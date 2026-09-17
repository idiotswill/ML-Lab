from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")

    with tempfile.TemporaryDirectory(prefix="ml-lab-qml-workspace-") as temp:
        temp_root = Path(temp)
        if os.name == "nt":
            os.environ["LOCALAPPDATA"] = str(temp_root / "config")
        else:
            os.environ["XDG_CONFIG_HOME"] = str(temp_root / "config")

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine

        from ml_lab.ui.controller import AppController

        app = QGuiApplication(["ml-lab-qml-workspace-smoke"])
        controller = AppController()
        controller.openWorkspace(str(temp_root / "workspace"), True)
        controller.createProject("QML Workspace Smoke", "generic", "")

        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("appController", controller)
        qml_path = Path(__file__).parent / "qml" / "Main.qml"
        engine.load(QUrl.fromLocalFile(str(qml_path)))
        app.processEvents()

        roots = engine.rootObjects()
        ok = bool(roots)
        if roots:
            roots[0].setProperty("currentPage", 2)
            app.processEvents()
            roots[0].setProperty("currentPage", 3)
            app.processEvents()

        print(
            json.dumps(
                {
                    "ok": ok,
                    "experiments_page": 2,
                    "compare_page": 3,
                    "qml": str(qml_path),
                },
                sort_keys=True,
            )
        )
        controller.shutdown()
        del engine
        del app
        return 0 if ok else 7


if __name__ == "__main__":
    raise SystemExit(main())
