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


@dataclass(frozen=True, slots=True)
class EvaluationProgress:
    experiment_id: str
    split: DatasetSplit
    expected: int
    evaluated: int

    @property
    def complete(self) -> bool:
        return self.expected > 0 and self.evaluated == self.expected


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
        progress = self.progress(experiment_id, split)
        if progress.expected <= 0:
            raise ValueError(f"{split.value} partition contains no evaluation examples.")
        if progress.complete:
            raise RuntimeError(
                f"Evaluation is immutable: all {progress.expected} {split.value} case(s) "
                "already exist."
            )

        handles = self.datasets.evaluation_partition_handles(experiment.dataset_id)
        try:
            digest = handles[split.value]
        except KeyError as exc:
            raise RuntimeError(f"Frozen dataset is missing {split.value} partition.") from exc
        path = self.artifacts.resolve(digest)
        existing_ids = self._case_ids(experiment_id, split)
        source_ids: set[str] = set()

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
                if example_id in source_ids:
                    raise ValueError(
                        f"{path.name}:{line_number}: duplicate example_id {example_id!r}"
                    )
                source_ids.add(example_id)
                if example_id in existing_ids:
                    continue

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

        finished = self.progress(experiment_id, split)
        if not finished.complete:
            raise RuntimeError(
                f"Evaluation evidence is incomplete for {split.value}: "
                f"{finished.evaluated}/{finished.expected} case(s)."
            )
        return self.summary(experiment_id, split)

    def progress(self, experiment_id: str, split: DatasetSplit) -> EvaluationProgress:
        experiment = self.experiments.get(experiment_id)
        partition = next(
            (
                item
                for item in self.datasets.partitions(experiment.dataset_id)
                if item.split is split
            ),
            None,
        )
        expected = partition.example_count if partition is not None else 0
        evaluated = self.case_count(experiment_id, split)
        if evaluated > expected:
            raise RuntimeError(
                f"Evaluation evidence exceeds frozen {split.value} partition: "
                f"{evaluated}>{expected}."
            )
        return EvaluationProgress(
            experiment_id=experiment_id,
            split=split,
            expected=expected,
            evaluated=evaluated,
        )

    def summary(self, experiment_id: str, split: DatasetSplit) -> EvaluationSummary:
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total,COALESCE(SUM(ec.correct),0) AS correct,"
                "COUNT(f.id) AS failures,"
                "COALESCE(SUM(CASE WHEN f.severity=? THEN 1 ELSE 0 END),0) AS vetoes,"
                "COALESCE(AVG(ec.latency_ms),0) AS mean_latency "
                "FROM evaluation_cases ec "
                "LEFT JOIN failures f ON f.id=ec.failure_id "
                "WHERE ec.experiment_id=? AND ec.split=?",
                (FailureSeverity.VETO.value, experiment_id, split.value),
            ).fetchone()
            total = int(row["total"]) if row is not None else 0
            p95_latency_ms = 0.0
            if total:
                rank = max(1, math.ceil(0.95 * total))
                latency_row = conn.execute(
                    "SELECT latency_ms FROM evaluation_cases "
                    "WHERE experiment_id=? AND split=? "
                    "ORDER BY latency_ms LIMIT 1 OFFSET ?",
                    (experiment_id, split.value, rank - 1),
                ).fetchone()
                if latency_row is not None:
                    p95_latency_ms = float(latency_row["latency_ms"])
        if row is None or total == 0:
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
        return EvaluationSummary(
            experiment_id=experiment_id,
            split=split,
            total=total,
            correct=int(row["correct"]),
            failures=int(row["failures"]),
            veto_failures=int(row["vetoes"]),
            mean_latency_ms=float(row["mean_latency"]),
            p95_latency_ms=p95_latency_ms,
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

    def case_count(
        self,
        experiment_id: str,
        split: DatasetSplit,
        *,
        only_incorrect: bool = False,
    ) -> int:
        incorrect_clause = " AND correct=0" if only_incorrect else ""
        with self.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM evaluation_cases "
                "WHERE experiment_id=? AND split=?" + incorrect_clause,
                (experiment_id, split.value),
            ).fetchone()
        return int(row[0]) if row else 0

    def _case_ids(self, experiment_id: str, split: DatasetSplit) -> set[str]:
        with self.database.connection() as conn:
            rows = conn.execute(
                "SELECT example_id FROM evaluation_cases "
                "WHERE experiment_id=? AND split=?",
                (experiment_id, split.value),
            ).fetchall()
        return {str(row["example_id"]) for row in rows}

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
