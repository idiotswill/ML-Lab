from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from ml_lab.core.process import application_command, compiled_application_executable
from ml_lab.datasets.service import DatasetService
from ml_lab.diagnostics.hardware import collect_hardware_info
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace

_FULL_ROWS = 100_000
_SMOKE_ROWS = 2_000
_FULL_IMPORT_ROWS = 20_000
_SMOKE_IMPORT_ROWS = 500
_PAGE_SIZE = 100
_STALL_MS = 100.0
_PAGE_COUNT = 10


def run_release_performance_evidence(
    output_path: Path,
    *,
    smoke: bool = False,
) -> dict[str, object]:
    """Collect Phase 4 performance evidence without authorizing release or integration."""
    row_count = _SMOKE_ROWS if smoke else _FULL_ROWS
    import_rows = _SMOKE_IMPORT_ROWS if smoke else _FULL_IMPORT_ROWS
    executable = compiled_application_executable()
    executable_evidence = _executable_evidence(executable)
    if not smoke and os.name != "nt":
        payload = {
            "ok": False,
            "error": "Representative performance evidence must run on Windows.",
            "application_executable": executable_evidence,
            "testing_ready_authorized": False,
            "integration_gate": "NO_GO",
        }
        _write_receipt(output_path, payload)
        return payload

    with tempfile.TemporaryDirectory(prefix="ml-lab-release-performance-") as temp:
        temp_root = Path(temp)
        config_root = temp_root / "config"
        workspace_root = temp_root / "workspace"
        source = temp_root / "dataset.jsonl"
        import_source = temp_root / "background-import.jsonl"
        env = os.environ.copy()
        config_key = "LOCALAPPDATA" if os.name == "nt" else "XDG_CONFIG_HOME"
        env[config_key] = str(config_root)
        previous_config = os.environ.get(config_key)
        os.environ[config_key] = str(config_root)

        try:
            startup = _collect_startup(env)
            idle_memory = _collect_idle_memory(env)

            workspace = Workspace.create(workspace_root)
            project = workspace.create_project(
                "Phase 4 Performance Evidence",
                "generic",
                "Synthetic release-gate performance evidence only",
            )
            datasets = DatasetService(workspace)
            primary = datasets.create(project.id, "100k bounded paging probe")
            background = datasets.create(project.id, "Background import responsiveness")

            _write_dataset_source(source, row_count, prefix="primary")
            imported = datasets.import_jsonl(primary.id, source)
            if imported.imported != row_count or imported.rejected:
                raise RuntimeError(
                    "Primary performance dataset import did not match requested row count."
                )
            _write_dataset_source(import_source, import_rows, prefix="background")

            hardware = collect_hardware_info(workspace_root).to_dict()
            ui = _run_ui_evidence(
                workspace_root=workspace_root,
                project_id=project.id,
                primary_dataset_id=primary.id,
                background_dataset_id=background.id,
                background_source=import_source,
                background_rows=import_rows,
                smoke=smoke,
            )
        finally:
            if previous_config is None:
                os.environ.pop(config_key, None)
            else:
                os.environ[config_key] = previous_config

    startup_hard = bool(startup.get("interactive_window_ready")) and _le(
        startup.get("process_start_to_qml_ready_ms"),
        4000.0,
    )
    memory_hard = _le(idle_memory.get("idle_working_set_mb"), 300.0)
    idle_stalls_ok = _eq_zero(_nested(ui, "idle_navigation", "stalls_over_100ms"))
    import_stalls_ok = _eq_zero(_nested(ui, "background_import", "stalls_over_100ms"))
    worker_stalls_ok = _eq_zero(_nested(ui, "worker_load", "stalls_over_100ms"))
    bounded_100k = (
        ui.get("primary_rows") == row_count
        and ui.get("materialized_page_examples") == _PAGE_SIZE
    )
    import_complete = bool(_nested(ui, "background_import", "completed"))
    worker_cancelled = bool(_nested(ui, "cancellation", "terminal_cancelled"))
    cancellation_prompt = _le(
        _nested(ui, "cancellation", "visible_ack_ms"),
        1000.0,
    )

    hard_checks = {
        "startup_under_4s": startup_hard,
        "idle_working_set_under_300mb": memory_hard,
        "ordinary_navigation_no_over_100ms_stall": idle_stalls_ok,
        "bounded_project_paging": bounded_100k,
        "background_import_no_over_100ms_stall": import_stalls_ok and import_complete,
        "worker_load_no_over_100ms_stall": worker_stalls_ok,
        "cancellation_visible_within_1s": cancellation_prompt and worker_cancelled,
    }
    hard_checks_pass = all(hard_checks.values())
    instrumentation_checks = {
        "startup_interactive_signal": bool(startup.get("interactive_window_ready")),
        "idle_memory_measured": _is_number(idle_memory.get("idle_working_set_mb")),
        "ordinary_navigation_sampled": _gt_zero(
            _nested(ui, "idle_navigation", "samples")
        ),
        "bounded_paging_observed": bounded_100k,
        "background_import_completed_and_sampled": import_complete
        and _gt_zero(_nested(ui, "background_import", "samples")),
        "worker_load_sampled": _gt_zero(_nested(ui, "worker_load", "samples")),
        "cancellation_observed": worker_cancelled
        and _is_number(_nested(ui, "cancellation", "visible_ack_ms")),
        "compiled_executable_fingerprinted": executable_evidence is not None,
    }
    instrumentation_checks_pass = all(instrumentation_checks.values())
    gate_evaluable = not smoke and os.name == "nt" and row_count == _FULL_ROWS
    overall_ok = (
        instrumentation_checks_pass and hard_checks_pass
        if gate_evaluable
        else instrumentation_checks_pass
    )
    payload = {
        "format_version": 1,
        "kind": "ML_LAB_PHASE4_REPRESENTATIVE_PERFORMANCE",
        "ok": overall_ok,
        "smoke": smoke,
        "gate_evaluable": gate_evaluable,
        "representative_hardware_review_required": True,
        "application_executable": executable_evidence,
        "hardware": hardware,
        "startup": startup,
        "idle_memory": idle_memory,
        "ui": ui,
        "hard_checks": hard_checks,
        "hard_checks_pass": hard_checks_pass,
        "instrumentation_checks": instrumentation_checks,
        "instrumentation_checks_pass": instrumentation_checks_pass,
        "targets": {
            "startup_target_ms": 2500,
            "startup_hard_ms": 4000,
            "idle_working_set_target_mb": 220,
            "idle_working_set_hard_mb": 300,
            "gui_stall_threshold_ms": 100,
            "cancellation_visible_threshold_ms": 1000,
        },
        "testing_ready_authorized": False,
        "integration_gate": "NO_GO",
    }
    _write_receipt(output_path, payload)
    return payload


def _collect_startup(env: Mapping[str, str]) -> dict[str, object]:
    started = time.perf_counter()
    child = subprocess.run(
        application_command("--startup-probe-child"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env=dict(env),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    details = _last_json(child.stdout)
    if child.returncode != 0 or not details.get("ok"):
        raise RuntimeError(
            f"Startup probe failed rc={child.returncode}: {child.stderr[-1000:]}"
        )
    working_set = details.get("working_set_bytes")
    return {
        "process_start_to_qml_ready_ms": elapsed_ms,
        "child_qml_load_ms": details.get("qml_load_ms"),
        "child_interactive_ready_ms": details.get("interactive_ready_ms"),
        "interactive_window_ready": bool(details.get("interactive_window_ready")),
        "working_set_at_ready_mb": _bytes_to_mb(working_set),
        "target_ms": 2500,
        "hard_ms": 4000,
    }


def _collect_idle_memory(env: Mapping[str, str]) -> dict[str, object]:
    child = subprocess.run(
        application_command("--idle-memory-probe-child"),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env=dict(env),
    )
    details = _last_json(child.stdout)
    if child.returncode != 0 or not details.get("ok"):
        raise RuntimeError(
            f"Idle memory probe failed rc={child.returncode}: {child.stderr[-1000:]}"
        )
    return {
        "idle_working_set_mb": _bytes_to_mb(details.get("idle_working_set_bytes")),
        "target_mb": 220,
        "hard_mb": 300,
    }


def _run_ui_evidence(
    *,
    workspace_root: Path,
    project_id: str,
    primary_dataset_id: str,
    background_dataset_id: str,
    background_source: Path,
    background_rows: int,
    smoke: bool,
) -> dict[str, object]:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuick import QQuickWindow

    from ml_lab.ui.controller import AppController

    app = QGuiApplication(["ml-lab-release-performance-evidence"])
    app.setQuitOnLastWindowClosed(False)
    controller = AppController()

    workspace_started = time.perf_counter()
    controller.openWorkspace(str(workspace_root), False)
    app.processEvents()
    workspace_open_ms = round((time.perf_counter() - workspace_started) * 1000.0, 2)
    if not controller.hasWorkspace:
        controller.shutdown()
        raise RuntimeError("Performance evidence could not open the prepared workspace.")

    project_started = time.perf_counter()
    controller.openProject(project_id)
    app.processEvents()
    project_open_ms = round((time.perf_counter() - project_started) * 1000.0, 2)
    selected_project = cast(Any, controller).selectedProject
    if str(selected_project.get("id", "")) != project_id:
        controller.shutdown()
        raise RuntimeError("Performance evidence could not open the prepared project.")

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parents[1] / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    app.processEvents()
    roots = engine.rootObjects()
    if not roots or not isinstance(roots[0], QQuickWindow):
        controller.shutdown()
        del engine
        del app
        raise RuntimeError("Performance evidence QML root did not load as QQuickWindow.")
    window = roots[0]

    data = cast(Any, controller).dataStudio
    page_started = time.perf_counter()
    data.selectDataset(primary_dataset_id)
    examples = data.examples
    app.processEvents()
    page_materialize_ms = round((time.perf_counter() - page_started) * 1000.0, 2)
    primary_rows = int(data.splitCounts["ALL"])
    materialized_page_examples = len(examples)

    idle_navigation = _measure_navigation(
        app,
        window,
        duration_seconds=0.35 if smoke else 1.5,
    )

    data.selectDataset(background_dataset_id)
    data.importJsonl(str(background_source))
    background_import = _measure_navigation(
        app,
        window,
        duration_seconds=0.35 if smoke else 0.75,
        done=lambda: not data.busy,
        timeout_seconds=15.0 if smoke else 60.0,
    )
    background_import["completed"] = not data.busy
    background_import["imported_rows"] = int(data.splitCounts["ALL"])

    workspace = Workspace.open(workspace_root)
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    job = manager.start(
        "core.performance_load",
        {"duration_seconds": 3.0 if smoke else 20.0},
    )
    worker_load = _measure_navigation(
        app,
        window,
        duration_seconds=0.45 if smoke else 1.5,
    )
    worker_load["job_id"] = job.id
    worker_load["task_type"] = job.task_type

    cancel_started = time.perf_counter()
    controller.cancelJob(job.id)
    visible_status = ""
    visible_ack_ms: float | None = None
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        app.processEvents()
        for row in cast(Any, controller).jobs:
            if str(row.get("id", "")) == job.id:
                visible_status = str(row.get("status", ""))
                if visible_status in {"CANCELLING", "CANCELLED"}:
                    visible_ack_ms = round(
                        (time.perf_counter() - cancel_started) * 1000.0,
                        2,
                    )
                    break
        if visible_ack_ms is not None:
            break
        time.sleep(0.01)

    terminal = manager.get(job.id)
    deadline = time.monotonic() + 8.0
    while terminal.status.value not in {"COMPLETED", "FAILED", "CANCELLED", "INTERRUPTED"}:
        if time.monotonic() >= deadline:
            break
        app.processEvents()
        time.sleep(0.02)
        terminal = manager.get(job.id)

    cancellation = {
        "visible_ack_ms": visible_ack_ms,
        "visible_status": visible_status,
        "terminal_status": terminal.status.value,
        "terminal_cancelled": terminal.status.value == "CANCELLED",
    }

    manager.shutdown()
    controller.shutdown()
    del engine
    del app
    return {
        "workspace_open_ms": workspace_open_ms,
        "project_open_ms": project_open_ms,
        "primary_rows": primary_rows,
        "materialized_page_examples": materialized_page_examples,
        "page_materialize_ms": page_materialize_ms,
        "idle_navigation": idle_navigation,
        "background_import": background_import,
        "worker_load": worker_load,
        "cancellation": cancellation,
        "background_rows_requested": background_rows,
    }


def _measure_navigation(
    app: object,
    window: object,
    *,
    duration_seconds: float,
    done: Callable[[], bool] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    process_events = cast(Any, app).processEvents
    set_property = cast(Any, window).setProperty
    minimum_end = time.monotonic() + duration_seconds
    hard_end = time.monotonic() + (timeout_seconds or duration_seconds)
    expected_sleep = 0.01
    stalls: list[float] = []
    navigation_samples: list[float] = []
    page = 0
    completed = False
    last_end = time.perf_counter()

    while time.monotonic() < hard_end:
        time.sleep(expected_sleep)
        woke = time.perf_counter()
        scheduler_delay_ms = max(
            0.0,
            ((woke - last_end) - expected_sleep) * 1000.0,
        )
        op_started = time.perf_counter()
        page = (page + 1) % _PAGE_COUNT
        set_property("currentPage", page)
        process_events()
        ended = time.perf_counter()
        navigation_ms = (ended - op_started) * 1000.0
        navigation_samples.append(navigation_ms)
        stalls.append(max(scheduler_delay_ms, navigation_ms))
        last_end = ended
        if done is not None and done() and time.monotonic() >= minimum_end:
            completed = True
            break
        if done is None and time.monotonic() >= minimum_end:
            completed = True
            break

    over = [value for value in stalls if value > _STALL_MS]
    return {
        "completed": completed,
        "samples": len(stalls),
        "max_stall_ms": round(max(stalls, default=0.0), 2),
        "stalls_over_100ms": len(over),
        "max_navigation_process_ms": round(max(navigation_samples, default=0.0), 2),
    }


def _write_dataset_source(path: Path, count: int, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            split = _split_for(index)
            example_id = f"{prefix}-{index:06d}"
            row = {
                "example_id": example_id,
                "split": split,
                "source_id": f"synthetic:{example_id}",
                "lineage_group": f"lineage:{example_id}",
                "payload": {"text": f"performance sample {prefix} {index}"},
                "label": {"class": "A" if index % 2 == 0 else "B"},
                "tags": ["phase4-performance"],
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _split_for(index: int) -> str:
    bucket = index % 10
    if bucket < 7:
        return "TRAIN"
    if bucket == 7:
        return "DEV"
    if bucket == 8:
        return "TEST"
    return "REDTEAM"


def _last_json(stdout: str) -> dict[str, object]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Probe produced no JSON output.")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise RuntimeError("Probe JSON output must be an object.")
    return {str(key): item for key, item in value.items()}


def _executable_evidence(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.is_file():
        return None
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_to_mb(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value) / (1024.0 * 1024.0), 2)


def _nested(value: Mapping[str, object], first: str, second: str) -> object:
    child = value.get(first)
    if not isinstance(child, dict):
        return None
    return child.get(second)


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _gt_zero(value: object) -> bool:
    return _is_number(value) and float(cast(int | float, value)) > 0.0


def _le(value: object, limit: float) -> bool:
    return _is_number(value) and float(cast(int | float, value)) <= limit


def _eq_zero(value: object) -> bool:
    return _is_number(value) and float(cast(int | float, value)) == 0.0


def _write_receipt(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
