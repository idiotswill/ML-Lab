import time
from pathlib import Path

from ml_lab.core.models import JobStatus
from ml_lab.core.process import worker_command
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.workspace import Workspace

TERMINAL = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.INTERRUPTED,
}


def wait_terminal(manager: JobManager, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record.status in TERMINAL:
            return record
        time.sleep(0.03)
    raise AssertionError("job did not finish")


def test_worker_completes_and_writes_manifest(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    manager = JobManager(workspace.root, workspace.database)
    record = manager.start("core.self_test", {"steps": 2, "delay": 0.01})
    finished = wait_terminal(manager, record.id)
    assert finished.status is JobStatus.COMPLETED
    assert (finished.staging_dir / "result_manifest.json").exists()


def test_worker_failure_is_persisted_without_result_artifact(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    record = manager.start("core.unknown_task")
    finished = wait_terminal(manager, record.id)
    assert finished.status is JobStatus.FAILED
    assert finished.exit_code == 2
    assert finished.error == "Unknown task type: core.unknown_task"
    assert finished.result_artifact_digest is None
    assert finished.event_artifact_digest


def test_cancel_worker(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    manager = JobManager(workspace.root, workspace.database)
    record = manager.start("core.self_test", {"steps": 80, "delay": 0.02})
    manager.cancel(record.id)
    finished = wait_terminal(manager, record.id)
    assert finished.status is JobStatus.CANCELLED


def test_reconcile_marks_inflight_jobs_interrupted(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    manager = JobManager(workspace.root, workspace.database)
    now = "2026-01-01T00:00:00+00:00"
    with workspace.database.transaction() as conn:
        conn.execute(
            "INSERT INTO jobs("
            "id,task_type,status,progress,message,staging_dir,correlation_id,created_at,updated_at"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "dead",
                "core.self_test",
                "RUNNING",
                0.3,
                "Running",
                str(tmp_path / "dead"),
                "corr",
                now,
                now,
            ),
        )
    assert manager.reconcile_startup() == 1
    assert manager.get("dead").status is JobStatus.INTERRUPTED


def test_worker_command_uses_python_module_in_dev(monkeypatch, tmp_path: Path) -> None:
    import ml_lab.core.process as process_module

    monkeypatch.delitem(process_module.__dict__, "__compiled__", raising=False)
    monkeypatch.setattr(process_module.sys, "executable", "python.exe")
    command = worker_command(tmp_path / "spec.json")
    assert command[:3] == ["python.exe", "-m", "ml_lab.jobs.worker"]


def test_worker_command_uses_compiled_entrypoint(monkeypatch, tmp_path: Path) -> None:
    import ml_lab.core.process as process_module

    executable = tmp_path / "MLLab.exe"
    executable.write_bytes(b"test executable marker")
    monkeypatch.setitem(process_module.__dict__, "__compiled__", object())
    monkeypatch.setattr(process_module.sys, "argv", [str(executable)])
    command = worker_command(tmp_path / "spec.json")
    assert command[0] == str(executable.resolve())
    assert command[1] == "--worker"


def test_worker_result_is_registered_as_immutable_artifact(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "lab")
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    record = manager.start("core.self_test", {"steps": 1, "delay": 0.0})
    finished = wait_terminal(manager, record.id)
    assert finished.status is JobStatus.COMPLETED
    assert finished.result_artifact_digest
    assert workspace.artifacts.resolve(finished.result_artifact_digest).exists()
    assert finished.event_artifact_digest
