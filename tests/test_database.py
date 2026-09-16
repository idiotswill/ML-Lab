import sqlite3
from pathlib import Path

import pytest

import ml_lab.storage.database as database_module
from ml_lab.storage.database import Database


def test_migration_failure_rolls_back_all_statements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lab.db"
    db = Database(path)
    monkeypatch.setattr(database_module, "SCHEMA_VERSION", 1)
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        {1: ("CREATE TABLE should_rollback(id INTEGER)", "THIS IS INVALID SQL")},
    )
    with pytest.raises(sqlite3.OperationalError):
        db.migrate()
    with db.connect() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='should_rollback'"
        ).fetchone()
    assert row is None


def test_sequential_upgrade_from_existing_schema_creates_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "lab.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES('schema_version', '1')")
    conn.execute("CREATE TABLE legacy(id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(database_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        {
            1: ("CREATE TABLE legacy(id INTEGER PRIMARY KEY)",),
            2: ("ALTER TABLE legacy ADD COLUMN note TEXT NOT NULL DEFAULT ''",),
        },
    )

    db = Database(path)
    db.migrate()
    assert db.schema_version() == 2
    with db.connection() as migrated:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(legacy)")}
    assert columns == {"id", "note"}

    backup = path.with_suffix(path.suffix + ".pre-migrate.bak")
    assert backup.exists()
    backup_db = Database(backup)
    assert backup_db.schema_version() == 1


def test_newer_workspace_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lab.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES('schema_version', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer than supported"):
        Database(path).migrate()
