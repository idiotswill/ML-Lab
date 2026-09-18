from __future__ import annotations

import hashlib
import json
import logging
import socket
import time
from pathlib import Path

import pytest

import ml_lab.storage.artifacts as artifacts_module
from ml_lab.diagnostics.crash import (
    AUTOMATIC_CRASH_UPLOADS,
    write_local_crash_report,
)
from ml_lab.jobs.manager import JobManager
from ml_lab.storage.database import MIGRATIONS, SCHEMA_VERSION, Database
from ml_lab.storage.workspace import Workspace


def test_artifact_disk_write_failure_is_atomic_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    payload = b"partial immutable artifacts must never register"
    digest = hashlib.sha256(payload).hexdigest()
    target = (
        workspace.artifacts.sha_root
        / digest[:2]
        / digest[2:4]
        / digest
    )

    def fail_fsync(_fd: int) -> None:
        raise OSError("simulated disk full during artifact fsync")

    monkeypatch.setattr(artifacts_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated disk full"):
        workspace.artifacts.commit_bytes(payload)

    assert not target.exists()
    assert target.parent.is_dir()
    assert list(target.parent.iterdir()) == []
    with workspace.database.connection() as conn:
        row = conn.execute(
            "SELECT digest FROM artifacts WHERE digest=?",
            (digest,),
        ).fetchone()
    assert row is None


def test_corrupt_schema_metadata_is_actionable_and_not_silently_repaired(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    with workspace.database.transaction() as conn:
        conn.execute(
            "UPDATE schema_meta SET value='not-a-version' "
            "WHERE key='schema_version'"
        )

    with pytest.raises(RuntimeError, match="invalid schema_version"):
        Workspace.open(workspace.root)

    with workspace.database.connection() as conn:
        value = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
    assert value == "not-a-version"


def test_pre_migration_backup_can_restore_then_remigrate(tmp_path: Path) -> None:
    path = tmp_path / "lab.db"
    database = Database(path)
    with database.transaction() as conn:
        conn.execute(
            "CREATE TABLE schema_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        for version in range(1, 4):
            for statement in MIGRATIONS[version]:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(version),),
            )
        conn.execute(
            "INSERT INTO projects("
            "id,name,adapter_id,description,state,created_at,updated_at"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                "before-migration",
                "Before migration",
                "generic",
                "",
                "ACTIVE",
                "2026-09-18T00:00:00+00:00",
                "2026-09-18T00:00:00+00:00",
            ),
        )

    assert database.schema_version() == 3
    database.migrate()
    assert database.schema_version() == SCHEMA_VERSION
    assert database.pre_migration_backup_path.is_file()

    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO projects("
            "id,name,adapter_id,description,state,created_at,updated_at"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                "after-migration",
                "After migration",
                "generic",
                "",
                "ACTIVE",
                "2026-09-18T00:00:01+00:00",
                "2026-09-18T00:00:01+00:00",
            ),
        )

    restored_version = database.restore_pre_migration_backup()
    assert restored_version == 3
    with database.connection() as conn:
        project_ids = {
            row[0] for row in conn.execute("SELECT id FROM projects").fetchall()
        }
        eval_table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='evaluation_cases'"
        ).fetchone()
    assert project_ids == {"before-migration"}
    assert eval_table is None

    database.migrate()
    assert database.schema_version() == SCHEMA_VERSION


def test_user_visible_job_failure_log_includes_correlation_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    caplog.set_level(logging.ERROR, logger="ml_lab.jobs.manager")
    try:
        started = manager.start("core.unknown_task")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            finished = manager.get(started.id)
            if finished.status.value == "FAILED":
                break
            time.sleep(0.03)
        else:
            raise AssertionError("failing worker did not reach FAILED")
    finally:
        manager.shutdown()

    assert finished.correlation_id
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"correlation_id={finished.correlation_id}" in message
        and f"job_id={finished.id}" in message
        for message in messages
    )


def test_crash_report_is_local_only_and_never_auto_uploads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_attempts: list[object] = []

    def reject_network(*args: object, **kwargs: object) -> object:
        network_attempts.append((args, kwargs))
        raise AssertionError("crash handler attempted network access")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    try:
        raise RuntimeError("simulated local crash")
    except RuntimeError as exc:
        report = write_local_crash_report(
            tmp_path / "logs",
            type(exc),
            exc,
            exc.__traceback__,
        )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert AUTOMATIC_CRASH_UPLOADS is False
    assert payload["automatic_upload"] is False
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["message"] == "simulated local crash"
    assert network_attempts == []
