from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            adapter_id TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            state TEXT NOT NULL CHECK (state IN ('ACTIVE', 'ARCHIVED')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_projects_state_updated ON projects(state, updated_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            digest TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
            media_type TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            status TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
            message TEXT NOT NULL DEFAULT '',
            staging_dir TEXT NOT NULL,
            pid INTEGER,
            correlation_id TEXT NOT NULL,
            exit_code INTEGER,
            error TEXT,
            result_artifact_digest TEXT REFERENCES artifacts(digest),
            event_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_jobs_status_updated ON jobs(status, updated_at DESC)",
        """
        CREATE TABLE IF NOT EXISTS extensions (
            kind TEXT NOT NULL,
            extension_id TEXT NOT NULL,
            version TEXT NOT NULL,
            protocol_version INTEGER NOT NULL,
            manifest_json TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(kind, extension_id, version)
        )
        """,
    )
}


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        current_version = self._current_version()
        if current_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Workspace schema {current_version} is newer than supported {SCHEMA_VERSION}."
            )
        if current_version == SCHEMA_VERSION:
            return

        if self.path.exists() and self.path.stat().st_size > 0:
            self._backup_before_migration()

        with self.transaction() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                try:
                    statements = MIGRATIONS[version]
                except KeyError as exc:
                    raise RuntimeError(f"Missing migration {version}") from exc
                for statement in statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(version),),
                )

    def schema_version(self) -> int:
        return self._current_version()

    def _current_version(self) -> int:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT value FROM schema_meta WHERE key='schema_version'"
                ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row else 0

    def _backup_before_migration(self) -> Path:
        backup_path = self.path.with_suffix(self.path.suffix + ".pre-migrate.bak")
        source = sqlite3.connect(self.path)
        target = sqlite3.connect(backup_path)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return backup_path
