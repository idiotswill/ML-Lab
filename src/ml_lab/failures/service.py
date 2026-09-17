from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Mapping

from ml_lab.core.models import (
    DatasetSplit,
    FailureRecord,
    FailureSeverity,
    FailureStatus,
    utc_now_iso,
)
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace


class FailureService:
    """Store immutable failure evidence and separate regression membership."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts

    def record(
        self,
        *,
        project_id: str,
        kind: str,
        severity: FailureSeverity,
        expected: object,
        observed: object,
        experiment_id: str | None = None,
        dataset_id: str | None = None,
        redteam_run_id: str | None = None,
        example_id: str | None = None,
        split: DatasetSplit | None = None,
        evidence: Mapping[str, object] | None = None,
    ) -> FailureRecord:
        clean_kind = kind.strip()
        if not clean_kind:
            raise ValueError("Failure kind is required.")
        evidence_digest: str | None = None
        if evidence is not None:
            ref = self.artifacts.commit_bytes(
                (canonical_json(dict(evidence)) + "\n").encode("utf-8"),
                media_type="application/vnd.ml-lab.failure-evidence+json",
                metadata={"project_id": project_id, "kind": clean_kind},
            )
            evidence_digest = ref.digest
        record = FailureRecord(
            id=str(uuid.uuid4()),
            project_id=project_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            redteam_run_id=redteam_run_id,
            example_id=example_id,
            split=split,
            kind=clean_kind,
            severity=severity,
            status=FailureStatus.OPEN,
            expected_json=canonical_json(expected),
            observed_json=canonical_json(observed),
            evidence_artifact_digest=evidence_digest,
            created_at=utc_now_iso(),
        )
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO failures"
                "(id,project_id,experiment_id,dataset_id,redteam_run_id,example_id,"
                "split,kind,severity,status,expected_json,observed_json,"
                "evidence_artifact_digest,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.project_id,
                    record.experiment_id,
                    record.dataset_id,
                    record.redteam_run_id,
                    record.example_id,
                    record.split.value if record.split else None,
                    record.kind,
                    record.severity.value,
                    record.status.value,
                    record.expected_json,
                    record.observed_json,
                    record.evidence_artifact_digest,
                    record.created_at,
                ),
            )
        return record

    def get(self, failure_id: str) -> FailureRecord:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT * FROM failures WHERE id=?", (failure_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown failure {failure_id}")
        return _failure_from_row(row)

    def page_for_project(
        self,
        project_id: str,
        *,
        severity: FailureSeverity | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[FailureRecord]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        where = "project_id=?"
        params: list[object] = [project_id]
        if severity is not None:
            where += " AND severity=?"
            params.append(severity.value)
        params.extend((limit, offset))
        with self.database.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM failures WHERE {where} "
                "ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
                tuple(params),
            ).fetchall()
        return [_failure_from_row(row) for row in rows]

    def count_for_project(
        self,
        project_id: str,
        *,
        severity: FailureSeverity | None = None,
    ) -> int:
        where = "project_id=?"
        params: list[object] = [project_id]
        if severity is not None:
            where += " AND severity=?"
            params.append(severity.value)
        with self.database.connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM failures WHERE {where}",
                tuple(params),
            ).fetchone()
        return int(row[0]) if row else 0

    def promote_to_regression(
        self,
        failure_id: str,
        *,
        suite_name: str = "default",
    ) -> None:
        clean_suite = suite_name.strip()
        if not clean_suite:
            raise ValueError("Regression suite name is required.")
        self.get(failure_id)
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO regression_cases"
                "(failure_id,suite_name,promoted_at) VALUES(?,?,?)",
                (failure_id, clean_suite, utc_now_iso()),
            )

    def is_regression_case(
        self,
        failure_id: str,
        *,
        suite_name: str = "default",
    ) -> bool:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM regression_cases WHERE failure_id=? AND suite_name=?",
                (failure_id, suite_name),
            ).fetchone()
        return row is not None

    def regression_count(
        self,
        project_id: str,
        *,
        suite_name: str = "default",
    ) -> int:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM regression_cases r "
                "JOIN failures f ON f.id=r.failure_id "
                "WHERE f.project_id=? AND r.suite_name=?",
                (project_id, suite_name),
            ).fetchone()
        return int(row[0]) if row else 0

    def regression_cases(
        self,
        *,
        suite_name: str = "default",
        limit: int = 1000,
    ) -> list[FailureRecord]:
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT f.* FROM failures f "
                "JOIN regression_cases r ON r.failure_id=f.id "
                "WHERE r.suite_name=? ORDER BY r.promoted_at,f.id LIMIT ?",
                (suite_name, limit),
            ).fetchall()
        return [_failure_from_row(row) for row in rows]

    def open_veto_count(self, *, experiment_id: str) -> int:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM failures "
                "WHERE experiment_id=? AND severity=? AND status IN (?,?)",
                (
                    experiment_id,
                    FailureSeverity.VETO.value,
                    FailureStatus.OPEN.value,
                    FailureStatus.REGRESSION.value,
                ),
            ).fetchone()
        return int(row[0]) if row else 0

    def immutable_payload(self, failure_id: str) -> dict[str, object]:
        record = self.get(failure_id)
        return {
            "failure_id": record.id,
            "project_id": record.project_id,
            "experiment_id": record.experiment_id,
            "dataset_id": record.dataset_id,
            "redteam_run_id": record.redteam_run_id,
            "example_id": record.example_id,
            "split": record.split.value if record.split else None,
            "kind": record.kind,
            "severity": record.severity.value,
            "expected": json.loads(record.expected_json),
            "observed": json.loads(record.observed_json),
            "evidence_artifact_sha256": record.evidence_artifact_digest,
            "created_at": record.created_at,
        }


def _failure_from_row(row: sqlite3.Row) -> FailureRecord:
    split = DatasetSplit(row["split"]) if row["split"] is not None else None
    return FailureRecord(
        id=row["id"],
        project_id=row["project_id"],
        experiment_id=row["experiment_id"],
        dataset_id=row["dataset_id"],
        redteam_run_id=row["redteam_run_id"],
        example_id=row["example_id"],
        split=split,
        kind=row["kind"],
        severity=FailureSeverity(row["severity"]),
        status=FailureStatus(row["status"]),
        expected_json=row["expected_json"],
        observed_json=row["observed_json"],
        evidence_artifact_digest=row["evidence_artifact_digest"],
        created_at=row["created_at"],
    )
