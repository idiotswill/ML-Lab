from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ml_lab.core.config import AppConfig, user_config_dir

_PAGE_NAMES = (
    "projects",
    "data-studio",
    "experiments",
    "compare",
    "red-team-failures",
    "models-registry",
    "package-verify",
    "jobs",
    "diagnostics",
    "settings",
)


def run_release_visual_smoke(output_dir: Path, theme: str) -> dict[str, object]:
    if theme not in {"dark", "light"}:
        raise ValueError("Visual smoke theme must be dark or light.")

    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")

    with tempfile.TemporaryDirectory(prefix="ml-lab-release-visual-") as temp:
        temp_root = Path(temp)
        if os.name == "nt":
            os.environ["LOCALAPPDATA"] = str(temp_root / "config")
        else:
            os.environ["XDG_CONFIG_HOME"] = str(temp_root / "config")

        AppConfig(theme=theme).save(user_config_dir() / "config.json")

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtQml import QQmlApplicationEngine
        from PySide6.QtQuick import QQuickWindow

        from ml_lab.ui.controller import AppController

        app = QGuiApplication(["ml-lab-release-visual-smoke"])
        controller = AppController()
        controller.openWorkspace(str(temp_root / "workspace"), True)
        controller.createProject("Release Visual Sample", "generic", "UX visual evidence")

        engine = QQmlApplicationEngine()
        engine.rootContext().setContextProperty("appController", controller)
        qml_path = Path(__file__).parent / "qml" / "Main.qml"
        engine.load(QUrl.fromLocalFile(str(qml_path)))
        app.processEvents()

        roots = engine.rootObjects()
        if not roots:
            controller.shutdown()
            return {"ok": False, "error": "QML root did not load"}

        raw_window = roots[0]
        if not isinstance(raw_window, QQuickWindow):
            controller.shutdown()
            return {"ok": False, "error": "QML root is not a QQuickWindow"}
        window = raw_window
        window.setWidth(1366)
        window.setHeight(768)
        window.show()
        app.processEvents()

        scale = os.environ.get("QT_SCALE_FACTOR", "1")
        scale_label = scale.replace(".", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        captures: list[str] = []
        for index, page_name in enumerate(_PAGE_NAMES):
            window.setProperty("currentPage", index)
            app.processEvents()
            image = window.grabWindow()
            target = output_dir / (
                f"{theme}-scale-{scale_label}-{index:02d}-{page_name}.png"
            )
            if image.isNull() or not image.save(str(target)):
                controller.shutdown()
                raise RuntimeError(f"Could not capture release visual evidence: {target}")
            captures.append(target.name)

        payload = {
            "ok": True,
            "theme": theme,
            "scale_factor": scale,
            "logical_width": int(window.width()),
            "logical_height": int(window.height()),
            "captures": captures,
        }
        (output_dir / f"{theme}-scale-{scale_label}-manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        controller.shutdown()
        del engine
        del app
        return payload
