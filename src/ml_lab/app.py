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

from ml_lab.bundles.verify import write_verification_receipt
from ml_lab.core.config import user_config_dir
from ml_lab.core.models import TERMINAL_JOB_STATUSES, JobStatus
from ml_lab.core.process import application_command
from ml_lab.diagnostics.crash import install_local_crash_handler
from ml_lab.diagnostics.logging_setup import configure_logging
from ml_lab.jobs.manager import JobManager
from ml_lab.jobs.worker import run_worker
from ml_lab.storage.database import SCHEMA_VERSION
from ml_lab.storage.workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ML Lab")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset-import-stage-child",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-validator-child",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-preflight-child",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-fixture-export-child",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-provider-child",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-fixture-seed-evidence",
        type=Path,
        help="Write protected Phase A fixture evidence to this JSON path.",
    )
    parser.add_argument(
        "--frankenhomie-repo",
        type=Path,
        help="Local Frankenhomie Git checkout used for pinned Phase A evidence.",
    )
    parser.add_argument(
        "--phase-a-fixture-seed",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-baseline-evidence",
        type=Path,
        help="Write Phase A v2 baseline evidence to this JSON path.",
    )
    parser.add_argument(
        "--phase-a-baseline-receipt",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-baseline-labels",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-build-dataset",
        type=Path,
        help="Build protected Phase A TRAIN/DEV candidate data into this directory.",
    )
    parser.add_argument(
        "--phase-a-dataset-seed",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-protected-seed",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-a-freeze-dataset",
        type=Path,
        help="Freeze Phase A TRAIN/DEV candidate data into this ML Lab workspace.",
    )
    parser.add_argument(
        "--phase-a-factory-output",
        type=Path,
        help="Phase A factory output directory containing receipt and JSONL.",
    )
    parser.add_argument(
        "--phase-a-first-experiment",
        type=Path,
        help="Run the first TRAIN-only Phase A sparse experiment in this frozen workspace.",
    )
    parser.add_argument(
        "--phase-a-candidate-experiment",
        type=Path,
        help=(
            "Run the candidate-relative TRAIN-only Phase A sparse experiment "
            "in this frozen workspace."
        ),
    )
    parser.add_argument(
        "--phase-a-generate-corpus",
        type=Path,
        help="Generate the expanded Phase A synthetic corpus into this directory.",
    )
    parser.add_argument(
        "--phase-a-corpus-plan",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-bundle-worker",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verification-receipt",
        type=Path,
        help=argparse.SUPPRESS,
    )
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
        "--idle-memory-probe-child",
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
        "--release-smoke-prepare",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--release-smoke-reopen",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--restart-recovery-smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--release-visual-smoke",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--visual-theme",
        choices=("dark", "light"),
        default="dark",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--performance-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--release-performance-evidence",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--performance-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--qml-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Open this existing ML Lab workspace when the GUI starts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args.worker)
    if args.dataset_import_stage_child:
        from ml_lab.datasets.import_stage import run_import_stage_child

        return run_import_stage_child(args.dataset_import_stage_child)
    if args.phase_a_validator_child:
        from ml_lab.adapters.phase_a_reference import run_reference_validator_child

        return run_reference_validator_child(args.phase_a_validator_child)
    if args.phase_a_preflight_child:
        from ml_lab.adapters.phase_a_reference import run_reference_preflight_child

        return run_reference_preflight_child(args.phase_a_preflight_child)
    if args.phase_a_fixture_export_child:
        from ml_lab.adapters.phase_a_fixture_export import run_phase_a_fixture_export_child

        return run_phase_a_fixture_export_child(args.phase_a_fixture_export_child)
    if args.phase_a_provider_child:
        from ml_lab.adapters.phase_a_provider import run_local_provider_child

        return run_local_provider_child(args.phase_a_provider_child)
    if args.phase_a_fixture_seed_evidence:
        if args.frankenhomie_repo is None:
            print("--frankenhomie-repo is required", file=sys.stderr)
            return 11
        from ml_lab.adapters.phase_a_fixture_seed import (
            DEFAULT_PHASE_A_FIXTURE_SEED,
            run_phase_a_fixture_seed_evidence,
        )

        seed_path = args.phase_a_fixture_seed or DEFAULT_PHASE_A_FIXTURE_SEED
        try:
            result = run_phase_a_fixture_seed_evidence(
                frankenhomie_repository=args.frankenhomie_repo,
                output_path=args.phase_a_fixture_seed_evidence,
                seed_path=seed_path,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 12
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") is True else 13
    if args.phase_a_baseline_evidence:
        if args.frankenhomie_repo is None:
            print("--frankenhomie-repo is required", file=sys.stderr)
            return 14
        from ml_lab.adapters.phase_a_baselines import (
            DEFAULT_PHASE_A_FIXTURE_LABELS,
            DEFAULT_PHASE_A_FIXTURE_RECEIPT,
            run_phase_a_baselines,
        )

        receipt_path = args.phase_a_baseline_receipt or DEFAULT_PHASE_A_FIXTURE_RECEIPT
        labels_path = args.phase_a_baseline_labels or DEFAULT_PHASE_A_FIXTURE_LABELS
        try:
            with tempfile.TemporaryDirectory(prefix="ml-lab-phase-a-baselines-") as temp:
                workspace = Workspace.create(Path(temp) / "workspace")
                result = run_phase_a_baselines(
                    workspace=workspace,
                    frankenhomie_repository=args.frankenhomie_repo,
                    receipt_path=receipt_path,
                    labels_path=labels_path,
                    output_path=args.phase_a_baseline_evidence,
                )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 15
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase_a_build_dataset:
        if args.frankenhomie_repo is None:
            print("--frankenhomie-repo is required", file=sys.stderr)
            return 16
        from ml_lab.adapters.phase_a_dataset_factory import (
            DEFAULT_PROTECTED_SEED,
            DEFAULT_SYNTHETIC_SEED,
            run_phase_a_dataset_factory,
        )

        seed_path = args.phase_a_dataset_seed or DEFAULT_SYNTHETIC_SEED
        protected_seed_path = (
            args.phase_a_protected_seed or DEFAULT_PROTECTED_SEED
        )
        try:
            with tempfile.TemporaryDirectory(prefix="ml-lab-phase-a-dataset-") as temp:
                workspace = Workspace.create(Path(temp) / "workspace")
                result = run_phase_a_dataset_factory(
                    workspace=workspace,
                    frankenhomie_repository=args.frankenhomie_repo,
                    output_dir=args.phase_a_build_dataset,
                    seed_path=seed_path,
                    protected_seed_path=protected_seed_path,
                )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 17
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") is True else 18
    if args.phase_a_freeze_dataset:
        if args.frankenhomie_repo is None:
            print("--frankenhomie-repo is required", file=sys.stderr)
            return 19
        if args.phase_a_factory_output is None:
            print("--phase-a-factory-output is required", file=sys.stderr)
            return 20
        from ml_lab.adapters.phase_a_freeze import freeze_phase_a_train_dev

        factory_root = args.phase_a_factory_output.expanduser().resolve()
        try:
            result = freeze_phase_a_train_dev(
                frankenhomie_repository=args.frankenhomie_repo,
                factory_receipt_path=factory_root / "factory-receipt.json",
                dataset_path=factory_root / "phase-a-train-dev-v1.jsonl",
                workspace_path=args.phase_a_freeze_dataset,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 21
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase_a_first_experiment:
        from ml_lab.adapters.phase_a_first_experiment import (
            run_phase_a_first_sparse_experiment,
        )

        try:
            result = run_phase_a_first_sparse_experiment(
                workspace_path=args.phase_a_first_experiment,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 22
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase_a_candidate_experiment:
        from ml_lab.adapters.phase_a_candidate_experiment import (
            run_phase_a_candidate_sparse_experiment,
        )

        try:
            result = run_phase_a_candidate_sparse_experiment(
                workspace_path=args.phase_a_candidate_experiment,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 23
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.phase_a_generate_corpus:
        from ml_lab.adapters.phase_a_corpus import (
            DEFAULT_CORPUS_PLAN,
            generate_phase_a_corpus,
        )

        try:
            result = generate_phase_a_corpus(
                output_dir=args.phase_a_generate_corpus,
                plan_path=args.phase_a_corpus_plan or DEFAULT_CORPUS_PLAN,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "integration_gate": "NO_GO",
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 24
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.verify_bundle_worker:
        if args.verification_receipt is None:
            print("--verification-receipt is required", file=sys.stderr)
            return 9
        return write_verification_receipt(
            args.verify_bundle_worker,
            args.verification_receipt,
        )
    if args.prepare_interrupted_job:
        return prepare_interrupted_job(args.prepare_interrupted_job)
    if args.startup_probe_child:
        return startup_probe_child()
    if args.idle_memory_probe_child:
        return idle_memory_probe_child()
    if args.smoke_test:
        return smoke_test()
    if args.job_smoke_test:
        return job_smoke_test()
    if args.release_smoke_prepare:
        from ml_lab.release.smoke import prepare_clean_machine_smoke

        print(json.dumps(prepare_clean_machine_smoke(args.release_smoke_prepare)))
        return 0
    if args.release_smoke_reopen:
        from ml_lab.release.smoke import reopen_clean_machine_smoke

        print(json.dumps(reopen_clean_machine_smoke(args.release_smoke_reopen)))
        return 0
    if args.restart_recovery_smoke_test:
        return restart_recovery_smoke_test()
    if args.release_visual_smoke:
        from ml_lab.ui.release_visual_smoke import run_release_visual_smoke

        result = run_release_visual_smoke(
            args.release_visual_smoke,
            args.visual_theme,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") else 8
    if args.performance_probe:
        return performance_probe()
    if args.release_performance_evidence:
        from ml_lab.release.performance import run_release_performance_evidence

        result = run_release_performance_evidence(
            args.release_performance_evidence,
            smoke=args.performance_smoke,
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") else 10
    if args.qml_smoke_test:
        return qml_smoke_test()
    return run_gui(args.workspace)


def smoke_test() -> int:
    with tempfile.TemporaryDirectory(prefix="ml-lab-smoke-") as temp:
        workspace = Workspace.create(Path(temp) / "workspace")
        project = workspace.create_project("Smoke project", "generic")
        artifact = workspace.artifacts.commit_bytes(b"ml-lab-smoke")
        assert workspace.database.schema_version() == SCHEMA_VERSION
        assert workspace.artifacts.resolve(artifact.digest).read_bytes() == b"ml-lab-smoke"
        assert workspace.list_projects()[0].id == project.id
        print(
            json.dumps(
                {
                    "ok": True,
                    "project_id": project.id,
                    "artifact": artifact.digest,
                }
            )
        )
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
            print(
                json.dumps(
                    {
                        "ok": False,
                        "status": record.status.value,
                        "error": record.error,
                    }
                )
            )
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
            application_command("--prepare-interrupted-job", str(root)),
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
            print(
                json.dumps(
                    {
                        "ok": False,
                        "stage": "fixture",
                        "error": "no fixture output",
                    }
                )
            )
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
    roots = engine.rootObjects()
    ok = bool(roots)
    qml_load_ms = round((time.perf_counter() - started) * 1000, 2)
    interactive_window_ready = False
    interactive_ready_ms: float | None = None
    if roots:
        root = roots[0]
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            app.processEvents()
            if bool(root.property("visible")):
                interactive_window_ready = True
                interactive_ready_ms = round(
                    (time.perf_counter() - started) * 1000,
                    2,
                )
                break
            time.sleep(0.005)
    payload = {
        "ok": ok,
        "interactive_window_ready": interactive_window_ready,
        "interactive_ready_ms": interactive_ready_ms,
        "qml_load_ms": qml_load_ms,
        "working_set_bytes": _working_set_bytes(),
    }
    print(json.dumps(payload), flush=True)
    controller.shutdown()
    del engine
    del app
    return 0 if ok else 2



def idle_memory_probe_child() -> int:
    """Load the base QML shell, settle briefly, then report idle working set."""
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine

    from ml_lab.ui.controller import AppController

    app = QGuiApplication(["ml-lab-idle-memory-probe"])
    app.setOrganizationName("Frankenhomie")
    app.setOrganizationDomain("local.frankenhomie")
    app.setApplicationName("ML Lab")
    controller = AppController()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parent / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    app.processEvents()
    roots = engine.rootObjects()
    if not roots:
        controller.shutdown()
        del engine
        del app
        print(json.dumps({"ok": False, "error": "QML root did not load"}))
        return 7

    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    working_set = _working_set_bytes()
    payload = {
        "ok": working_set is not None,
        "idle_working_set_bytes": working_set,
    }
    print(json.dumps(payload), flush=True)
    controller.shutdown()
    del engine
    del app
    return 0 if working_set is not None else 7

def performance_probe() -> int:
    """Measure process launch through QML readiness using this same build."""
    started = time.perf_counter()
    child = subprocess.run(
        application_command("--startup-probe-child"),
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
    return 0 if details.get("ok") and working_set_mb is not None else 6


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
    print(
        json.dumps(
            {
                "ok": ok,
                "qml_load_ms": elapsed_ms,
                "qml": str(qml_path),
            }
        )
    )
    del engine
    del app
    return 0 if ok else 2


def run_gui(workspace_path: Path | None = None) -> int:
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

    log_dir = user_config_dir() / "logs"
    configure_logging(log_dir)
    install_local_crash_handler(log_dir)
    controller = AppController()
    app.aboutToQuit.connect(controller.shutdown)

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml_path = Path(__file__).parent / "ui" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if not engine.rootObjects():
        return 2
    if workspace_path is not None:
        controller.openWorkspace(str(workspace_path.expanduser().resolve()), False)
    return app.exec()


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

        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            return None
        kernel32 = loader("kernel32", use_last_error=True)
        psapi = loader("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = wintypes.HANDLE
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_process_memory_info.restype = wintypes.BOOL

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = get_current_process()
        if not get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)

    statm = Path("/proc/self/statm")
    sysconf = getattr(os, "sysconf", None)
    if statm.exists() and callable(sysconf):
        fields = statm.read_text(encoding="utf-8").split()
        if len(fields) >= 2:
            return int(fields[1]) * int(sysconf("SC_PAGE_SIZE"))
    return None
