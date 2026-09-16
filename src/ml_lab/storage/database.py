from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 4

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
        """
        CREATE INDEX IF NOT EXISTS idx_projects_state_updated
        ON projects(state, updated_at DESC)
        """,
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
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_status_updated
        ON jobs(status, updated_at DESC)
        """,
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
    ),
    2: (
        """
        CREATE TABLE contract_snapshots (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            adapter_id TEXT NOT NULL,
            adapter_version TEXT NOT NULL,
            repo_path TEXT NOT NULL,
            repo_identity TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            contract_version TEXT NOT NULL,
            compatibility_signature TEXT NOT NULL,
            manifest_artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_contract_project_created
        ON contract_snapshots(project_id, created_at DESC)
        """,
        """
        CREATE UNIQUE INDEX idx_contract_signature
        ON contract_snapshots(project_id, compatibility_signature)
        """,
        """
        CREATE TABLE dataset_versions (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            name TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('DRAFT', 'FROZEN', 'INVALID')),
            contract_snapshot_id TEXT REFERENCES contract_snapshots(id),
            example_count INTEGER NOT NULL DEFAULT 0 CHECK(example_count >= 0),
            manifest_artifact_digest TEXT REFERENCES artifacts(digest),
            leakage_report_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL,
            frozen_at TEXT
        )
        """,
        """
        CREATE INDEX idx_dataset_project_created
        ON dataset_versions(project_id, created_at DESC)
        """,
        """
        CREATE TABLE dataset_examples (
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
            example_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('TRAIN', 'DEV', 'TEST', 'REDTEAM')),
            source_id TEXT NOT NULL,
            lineage_group TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            normalized_fingerprint TEXT NOT NULL,
            near_signature TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            label_json TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            PRIMARY KEY(dataset_id, example_id)
        )
        """,
        """
        CREATE INDEX idx_examples_dataset_split
        ON dataset_examples(dataset_id, split, example_id)
        """,
        """
        CREATE INDEX idx_examples_dataset_lineage
        ON dataset_examples(dataset_id, lineage_group)
        """,
        """
        CREATE INDEX idx_examples_dataset_fingerprint
        ON dataset_examples(dataset_id, fingerprint)
        """,
        """
        CREATE INDEX idx_examples_dataset_normfingerprint
        ON dataset_examples(dataset_id, normalized_fingerprint)
        """,
        """
        CREATE TABLE dataset_partitions (
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE RESTRICT,
            split TEXT NOT NULL CHECK(split IN ('TRAIN', 'DEV', 'TEST', 'REDTEAM')),
            artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            example_count INTEGER NOT NULL CHECK(example_count >= 0),
            partition_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(dataset_id, split)
        )
        """,
        """
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id),
            contract_snapshot_id TEXT REFERENCES contract_snapshots(id),
            trainer_id TEXT NOT NULL,
            runtime_pack_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN (
                    'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'
                )
            ),
            config_json TEXT NOT NULL,
            seed INTEGER NOT NULL,
            environment_json TEXT NOT NULL DEFAULT '{}',
            model_artifact_digest TEXT REFERENCES artifacts(digest),
            metrics_artifact_digest TEXT REFERENCES artifacts(digest),
            manifest_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        )
        """,
        """
        CREATE INDEX idx_experiment_project_created
        ON experiments(project_id, created_at DESC)
        """,
        """
        CREATE INDEX idx_experiment_dataset
        ON experiments(dataset_id, created_at DESC)
        """,
        """
        CREATE TABLE experiment_metrics (
            experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            metric_id TEXT NOT NULL,
            value REAL NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('HIGHER', 'LOWER', 'ZERO')),
            veto INTEGER NOT NULL CHECK(veto IN (0, 1)),
            PRIMARY KEY(experiment_id, metric_id)
        )
        """,
        """
        CREATE TABLE redteam_runs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            dataset_id TEXT REFERENCES dataset_versions(id),
            experiment_id TEXT REFERENCES experiments(id),
            seed INTEGER NOT NULL,
            mutator_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN (
                    'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'INTERRUPTED'
                )
            ),
            manifest_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL,
            completed_at TEXT
        )
        """,
        """
        CREATE TABLE failures (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            experiment_id TEXT REFERENCES experiments(id),
            dataset_id TEXT REFERENCES dataset_versions(id),
            redteam_run_id TEXT REFERENCES redteam_runs(id),
            example_id TEXT,
            split TEXT CHECK(
                split IS NULL OR split IN ('TRAIN', 'DEV', 'TEST', 'REDTEAM')
            ),
            kind TEXT NOT NULL,
            severity TEXT NOT NULL CHECK(severity IN ('VETO', 'NON_VETO')),
            status TEXT NOT NULL CHECK(
                status IN ('OPEN', 'REGRESSION', 'FIXED', 'ACCEPTED')
            ),
            expected_json TEXT NOT NULL,
            observed_json TEXT NOT NULL,
            evidence_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_failures_project_status
        ON failures(project_id, status, created_at DESC)
        """,
        """
        CREATE INDEX idx_failures_experiment
        ON failures(experiment_id, created_at DESC)
        """,
        """
        CREATE TABLE models (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id),
            experiment_id TEXT NOT NULL REFERENCES experiments(id),
            model_artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            stage TEXT NOT NULL CHECK(
                stage IN (
                    'EXPERIMENT', 'SHADOW', 'ADVISORY',
                    'RELEASE_CANDIDATE', 'INTEGRATION_APPROVED'
                )
            ),
            compatibility_json TEXT NOT NULL DEFAULT '{}',
            manifest_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_models_project_stage
        ON models(project_id, stage, updated_at DESC)
        """,
        """
        CREATE TABLE model_stage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL REFERENCES models(id),
            from_stage TEXT,
            to_stage TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE bundles (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL REFERENCES models(id),
            bundle_artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            manifest_artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE verification_receipts (
            id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL REFERENCES bundles(id),
            status TEXT NOT NULL CHECK(status IN ('PASS', 'FAIL')),
            receipt_artifact_digest TEXT NOT NULL REFERENCES artifacts(digest),
            created_at TEXT NOT NULL
        )
        """,
    ),
    3: (
        """
        CREATE TABLE dataset_imports (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
            source_name TEXT NOT NULL,
            imported INTEGER NOT NULL CHECK(imported >= 0),
            rejected INTEGER NOT NULL CHECK(rejected >= 0),
            error_report_artifact_digest TEXT REFERENCES artifacts(digest),
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_dataset_imports_dataset
        ON dataset_imports(dataset_id, created_at DESC)
        """,
        """
        CREATE TABLE regression_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            failure_id TEXT NOT NULL UNIQUE REFERENCES failures(id),
            suite_name TEXT NOT NULL,
            promoted_at TEXT NOT NULL
        )
        """,
        """
        CREATE INDEX idx_regression_suite
        ON regression_cases(suite_name, promoted_at DESC)
        """,
    ),
    4: (
        """
        CREATE TABLE evaluation_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            example_id TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('TRAIN', 'DEV', 'TEST', 'REDTEAM')),
            expected_json TEXT NOT NULL,
            observed_json TEXT NOT NULL,
            correct INTEGER NOT NULL CHECK(correct IN (0, 1)),
            latency_ms REAL NOT NULL DEFAULT 0 CHECK(latency_ms >= 0),
            failure_id TEXT REFERENCES failures(id),
            created_at TEXT NOT NULL,
            UNIQUE(experiment_id, split, example_id)
        )
        """,
        """
        CREATE INDEX idx_eval_cases_experiment_split
        ON evaluation_cases(experiment_id, split, correct, example_id)
        """,
        """
        CREATE INDEX idx_eval_cases_failure
        ON evaluation_cases(failure_id)
        """,
    ),
}


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        """Open a configured connection.

        Callers that need a short-lived connection should prefer ``connection()`` so the
        handle is always closed on Windows as well as POSIX.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

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
                "CREATE TABLE IF NOT EXISTS schema_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
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
            with self.connection() as conn:
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
