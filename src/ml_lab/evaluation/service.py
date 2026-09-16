from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ml_lab.core.models import DatasetSplit, FailureSeverity, utc_now_iso
from ml_lab.datasets.service import DatasetService
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.storage.workspace import Workspace


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    observed: object
    correct: bool
    latency_ms: float
    failure_kind: str | None = None
    failure_severity: FailureSeverity | None = None
    evidence: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationCaseRecord:
    id: int
    experiment_id: str
    example_id: str
    split: DatasetSplit
    expected_json: str
    observed_json: str
    correct: bool
    latency_ms: float
    failure_id: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    experiment_id: str
    split: DatasetSplit
    total: int
    correct: int
    failures: int
    veto_failures: int
    mean_latency_ms: float
    p95_latency_ms: float

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


CaseEvaluator = Callable[[Mapping[str, object]], CaseOutcome]


class EvaluationService:
    """Run adapter-owned evaluation over immutable frozen partition artifacts.

    The host owns iteration, evidence persistence, paging, and comparison. Domain
    correctness remains adapter-owned through ``CaseEvaluator``.
    """

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.database = workspace.database
        self.artifacts = workspace.artifacts
        self.datasets = DatasetService(workspace)
        self.experiments = ExperimentService(workspace)
        self.failures = FailureService(workspace)

    def evaluate_partition(
        self,
        experiment_id: str,
        split: DatasetSplit,
        evaluator: CaseEvaluator,
    ) -> EvaluationSummary:
        experiment = self.experiments.get(experiment_id)
        existing = self._case_count(experiment_id, split)
        if existing:
            raise RuntimeError(
                f"Evaluation is immutable: {existing} {split.value} case(s) already exist."
            )
        handles = self.datasets.evaluation_partition_handles(experiment.dataset_id)
        try:
            digest = handles[split.value]
        except KeyError as exc:
            raise RuntimeError(f"Frozen dataset is missing {split.value} partition.") from exc
        path = self.artifacts.resolve(digest)

        inserted = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                row = _decode_case(raw_line, path.name, line_number)
                row_split = row.get("split")
                if row_split != split.value:
                    message = (
                        f"{path.name}:{line_number}: expected split {split.value}, "
                        f"got {row_split!r}"
                    )
                    raise ValueError(message)
                example_id = row.get("example_id")
                expected = row.get("label")
                if not isinstance(example_id, str) or not example_id:
                    raise ValueError(
                        f"{path.name}:{line_number}: example_id must be a non-empty string"
                    )

                started = time.perf_counter()
                outcome = evaluator(row)
                measured_ms = (time.perf_counter() - started) * 1000.0
                latency_ms = outcome.latency_ms if outcome.latency_ms >= 0 else measured_ms
                if not math.isfinite(latency_ms) or latency_ms < 0:
                    raise ValueError("Evaluator latency_ms must be finite and >= 0")

                failure_id = self._record_failure(
                    experiment_id=experiment.id,
                    project_id=experiment.project_id,
                    dataset_id=experiment.dataset_id,
                    example_id=example_id,
                    split=split,
                    expected=expected,
                    outcome=outcome,
                )
                self._insert_case(
                    experiment_id=experiment.id,
                    example_id=example_id,
                    split=split,
                    expected=expected,
                    observed=outcome.observed,
                    correct=outcome.correct,
                    latency_ms=latency_ms,
                    failure_id=failure_id,
                )
                inserted += 1

        if inserted == 0:
            raise ValueError(f"{split.value} partition contains no evaluation examples.")
        return self.summary(experiment_id, split)

    def summary(self, experiment_id: str, split: DatasetSplit) -> EvaluationSummary:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT ec.correct,ec.latency_ms,f.severity "
                "FROM evaluation_cases ec "
                "LEFT JOIN failures f ON f.id=ec.failure_id "
                "WHERE ec.experiment_id=? AND ec.split=? "
                "ORDER BY ec.example_id",
                (experiment_id, split.value),
            ).fetchall()
        if not rows:
            return EvaluationSummary(
                experiment_id=experiment_id,
                split=split,
                total=0,
                correct=0,
                failures=0,
                veto_failures=0,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
            )
        latencies = sorted(float(row["latency_ms"]) for row in rows)
        failure_rows = [row for row in rows if row["severity"] is not None]
        return EvaluationSummary(
            experiment_id=experiment_id,
            split=split,
            total=len(rows),
            correct=sum(bool(row["correct"]) for row in rows),
            failures=len(failure_rows),
            veto_failures=sum(
                row["severity"] == FailureSeverity.VETO.value for row in failure_rows
            ),
            mean_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=_percentile_nearest_rank(latencies, 0.95),
        )

    def compare(
        self,
        experiment_ids: list[str],
        split: DatasetSplit,
    ) -> list[EvaluationSummary]:
        summaries = [self.summary(experiment_id, split) for experiment_id in experiment_ids]
        return sorted(summaries, key=lambda item: item.experiment_id)

    def page_cases(
        self,
        experiment_id: str,
        split: DatasetSplit,
        *,
        only_incorrect: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> list[EvaluationCaseRecord]:
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        incorrect_clause = " AND correct=0" if only_incorrect else ""
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_cases "
                "WHERE experiment_id=? AND split=?"
                + incorrect_clause
                + " ORDER BY example_id LIMIT ? OFFSET ?",
                (experiment_id, split.value, limit, offset),
            ).fetchall()
        return [_case_from_row(row) for row in rows]

    def _case_count(self, experiment_id: str, split: DatasetSplit) -> int:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM evaluation_cases WHERE experiment_id=? AND split=?",
                (experiment_id, split.value),
            ).fetchone()
        return int(row[0]) if row else 0

    def _record_failure(
        self,
        *,
        experiment_id: str,
        project_id: str,
        dataset_id: str,
        example_id: str,
        split: DatasetSplit,
        expected: object,
        outcome: CaseOutcome,
    ) -> str | None:
        kind = outcome.failure_kind.strip() if outcome.failure_kind else ""
        if not kind:
            if outcome.failure_severity is not None:
                raise ValueError("failure_severity requires failure_kind")
            return None
        severity = outcome.failure_severity or FailureSeverity.NON_VETO
        record = self.failures.record(
            project_id=project_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            example_id=example_id,
            split=split,
            kind=kind,
            severity=severity,
            expected=expected,
            observed=outcome.observed,
            evidence=outcome.evidence,
        )
        return record.id

    def _insert_case(
        self,
        *,
        experiment_id: str,
        example_id: str,
        split: DatasetSplit,
        expected: object,
        observed: object,
        correct: bool,
        latency_ms: float,
        failure_id: str | None,
    ) -> None:
        with self.database.transaction() as conn:
            conn.execute(
                "INSERT INTO evaluation_cases"
                "(experiment_id,example_id,split,expected_json,observed_json,correct,"
                "latency_ms,failure_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    example_id,
                    split.value,
                    _canonical_json(expected),
                    _canonical_json(observed),
                    int(correct),
                    latency_ms,
                    failure_id,
                    utc_now_iso(),
                ),
            )


def _decode_case(raw_line: str, filename: str, line_number: int) -> dict[str, object]:
    try:
        decoded = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{filename}:{line_number}: invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{filename}:{line_number}: row must be a JSON object")
    return {str(key): value for key, value in decoded.items()}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _percentile_nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    rank = max(1, math.ceil(fraction * len(values)))
    return values[min(rank - 1, len(values) - 1)]


def _case_from_row(row: sqlite3.Row) -> EvaluationCaseRecord:
    return EvaluationCaseRecord(
        id=int(row["id"]),
        experiment_id=row["experiment_id"],
        example_id=row["example_id"],
        split=DatasetSplit(row["split"]),
        expected_json=row["expected_json"],
        observed_json=row["observed_json"],
        correct=bool(row["correct"]),
        latency_ms=float(row["latency_ms"]),
        failure_id=row["failure_id"],
        created_at=row["created_at"],
    )
