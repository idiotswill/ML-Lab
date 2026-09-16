from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from ml_lab.core.models import JobRecord, JobStatus, utc_now_iso
from ml_lab.jobs.protocol import JobSpec
from ml_lab.storage.artifacts import ArtifactStore
from ml_lab.storage.database import Database


class JobManager:
    def __init__(self, workspace_root: Path, database: Database, artifacts: ArtifactStore | None = None):
        self.workspace_root = workspace_root
        self.database = database
        self.artifacts = artifacts
        self.jobs_root = workspace_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def reconcile_startup(self) -> int:
        now = utc_now_iso()
        with self.database.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status=?, updated_at=?, message=? "
                "WHERE status IN (?, ?, ?)",
                (
                    JobStatus.INTERRUPTED.value,
                    now,
                    "Application restarted before job completion.",
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                    JobStatus.CANCELLING.value,
                ),
            )
            return int(cursor.rowcount)

    def start(self, task_type: str, payload: dict[str, Any] | None = None) -> JobRecord:
        job_id = str(uuid.uuid4())
        correlation_id = uuid.uuid4().hex[:12]
        staging = self.jobs_root / job_id
        staging.mkdir(parents=True, exist_ok=False)
        spec = JobSpec(job_id=job_id, task_type=task_type, staging_dir=str(staging), payload=payload or {})
        spec_path = staging / "job_spec.json"
        spec.write(spec_path)
        now = utc_now_iso()
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs(id,task_type,status,progress,message,staging_dir,correlation_id,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    task_type,
                    JobStatus.QUEUED.value,
                    0.0,
                    "Queued",
                    str(staging),
                    correlation_id,
                    now,
                    now,
                ),
            )

        command = _worker_command(spec_path)
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        with self._lock:
            self._processes[job_id] = process
        self._update(job_id, status=JobStatus.RUNNING, pid=process.pid, message="Running")
        thread = threading.Thread(
            target=self._monitor, args=(job_id, process, staging), daemon=True, name=f"job-{job_id[:8]}"
        )
        thread.start()
        return self.get(job_id)

    def cancel(self, job_id: str) -> None:
        record = self.get(job_id)
        if record.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            return
        (record.staging_dir / "cancel.request").write_text(utc_now_iso(), encoding="utf-8")
        self._update(job_id, status=JobStatus.CANCELLING, message="Cancellation requested")
        timer = threading.Timer(3.0, self._terminate_if_running, args=(job_id,))
        timer.daemon = True
        timer.start()

    def list_recent(self, limit: int = 50) -> list[JobRecord]:
        with self.database.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, job_id: str) -> JobRecord:
        with self.database.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._row_to_record(row)

    def shutdown(self) -> None:
        with self._lock:
            processes = list(self._processes.items())
        for job_id, process in processes:
            if process.poll() is None:
                try:
                    record = self.get(job_id)
                    (record.staging_dir / "cancel.request").write_text(utc_now_iso(), encoding="utf-8")
                    process.terminate()
                except (OSError, KeyError):
                    pass

    def _terminate_if_running(self, job_id: str) -> None:
        with self._lock:
            process = self._processes.get(job_id)
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            self._update(job_id, message="Worker did not stop cooperatively; termination requested")
        except OSError:
            return

    def _monitor(self, job_id: str, process: subprocess.Popen[str], staging: Path) -> None:
        event_log = staging / "events.jsonl"
        stderr_log = staging / "stderr.log"
        assert process.stdout is not None
        with event_log.open("a", encoding="utf-8") as events:
            for line in process.stdout:
                events.write(line)
                events.flush()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = payload.get("kind")
                if kind == "progress":
                    self._update(
                        job_id,
                        progress=float(payload.get("progress", 0.0)),
                        message=str(payload.get("message", "Running")),
                    )
                elif kind == "completed":
                    self._update(job_id, progress=1.0, message=str(payload.get("message", "Completed")))
                elif kind == "failed":
                    self._update(job_id, error=str(payload.get("error", "Worker failed")))
                elif kind == "cancelled":
                    self._update(job_id, message=str(payload.get("message", "Cancelled")))
        stderr = process.stderr.read() if process.stderr is not None else ""
        if stderr:
            stderr_log.write_text(stderr, encoding="utf-8")
        code = process.wait()
        current = self.get(job_id)
        result_digest = None
        event_digest = None
        manifest_path = staging / "result_manifest.json"
        if self.artifacts is not None and event_log.exists():
            event_digest = self.artifacts.commit_file(event_log, media_type="application/x-ndjson", metadata={"job_id": job_id, "kind": "events"}).digest
        if code == 0 and manifest_path.exists():
            if self.artifacts is not None:
                result_digest = self.artifacts.commit_file(manifest_path, media_type="application/json", metadata={"job_id": job_id, "kind": "result"}).digest
            final = JobStatus.COMPLETED
            error = current.error
        elif code == 130 or current.status == JobStatus.CANCELLING:
            final = JobStatus.CANCELLED
            error = current.error
        else:
            final = JobStatus.FAILED
            error = current.error or (stderr.strip() if stderr else f"Worker exited with code {code}")
        self._update(
            job_id,
            status=final,
            exit_code=code,
            error=error,
            result_artifact_digest=result_digest,
            event_artifact_digest=event_digest,
        )
        with self._lock:
            self._processes.pop(job_id, None)

    def _update(self, job_id: str, **changes: object) -> None:
        allowed = {"status", "progress", "message", "pid", "exit_code", "error", "result_artifact_digest", "event_artifact_digest"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unknown job fields: {unknown}")
        if "status" in changes and isinstance(changes["status"], JobStatus):
            changes["status"] = changes["status"].value
        changes["updated_at"] = utc_now_iso()
        fields = ", ".join(f"{key}=?" for key in changes)
        values = list(changes.values()) + [job_id]
        with self.database.transaction() as conn:
            conn.execute(f"UPDATE jobs SET {fields} WHERE id=?", values)

    @staticmethod
    def _row_to_record(row: Any) -> JobRecord:
        return JobRecord(
            id=row["id"],
            task_type=row["task_type"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            progress=float(row["progress"]),
            message=row["message"],
            staging_dir=Path(row["staging_dir"]),
            pid=row["pid"],
            correlation_id=row["correlation_id"],
            exit_code=row["exit_code"],
            error=row["error"],
            result_artifact_digest=row["result_artifact_digest"],
            event_artifact_digest=row["event_artifact_digest"],
        )


def _worker_command(spec_path: Path) -> list[str]:
    """Return a worker command that works in Python dev and compiled standalone builds."""
    executable = Path(sys.executable).name.casefold()
    python_names = {"python", "python3", "python.exe", "pythonw.exe", "pypy", "pypy3"}
    if executable not in python_names:
        return [sys.executable, "--worker", str(spec_path)]
    return [sys.executable, "-m", "ml_lab.jobs.worker", str(spec_path)]
