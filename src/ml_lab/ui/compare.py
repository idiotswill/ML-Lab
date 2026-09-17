from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, Signal, Slot

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID
from ml_lab.core.models import DatasetSplit, ExperimentRecord, ExperimentStatus
from ml_lab.evaluation.runners import evaluate_packaged_experiment
from ml_lab.evaluation.service import EvaluationCaseRecord, EvaluationService
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import (
    PHASE_A_RUNTIME_PACK_ID,
    PHASE_A_TRAINER_ID,
    SPARSE_RUNTIME_PACK_ID,
    SPARSE_TRAINER_ID,
)
from ml_lab.ui.phase_a_science import PhaseAScienceController


class _EvaluationSignals(QObject):
    completed = Signal(int, str, str)
    cancelled = Signal(int, str, str)
    failed = Signal(int, str, str, str)


class _EvaluationOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        experiment_id: str,
        split: DatasetSplit,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.experiment_id = experiment_id
        self.split = split
        self.cancel_event = cancel_event
        self.signals = _EvaluationSignals()

    @Slot()
    def run(self) -> None:
        try:
            workspace = Workspace.open(self.workspace_root)
            evaluate_packaged_experiment(
                workspace,
                self.experiment_id,
                splits=(self.split,),
                cancelled=self.cancel_event.is_set,
            )
        except InterruptedError:
            self.signals.cancelled.emit(
                self.context_token,
                self.experiment_id,
                self.split.value,
            )
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                self.experiment_id,
                self.split.value,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            self.signals.completed.emit(
                self.context_token,
                self.experiment_id,
                self.split.value,
            )


class CompareController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._experiment_offset = 0
        self._experiment_page_size = 100
        self._split = DatasetSplit.TEST
        self._only_incorrect = False
        self._offset = 0
        self._page_size = 100
        self._selected_case_id = 0
        self._busy = False
        self._context_token = 0
        self._cancel_event: threading.Event | None = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._science = PhaseAScienceController(self)
        self._science.operationCompleted.connect(self.operationCompleted.emit)
        self._science.operationFailed.connect(self.operationFailed.emit)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_experiment_id = ""
        self._experiment_offset = 0
        self._split = DatasetSplit.TEST
        self._only_incorrect = False
        self._offset = 0
        self._selected_case_id = 0
        self._busy = False
        self._science.bind_project(workspace, project_id, adapter_id)
        self.changed.emit()

    def clear_project(self) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._experiment_offset = 0
        self._split = DatasetSplit.TEST
        self._only_incorrect = False
        self._offset = 0
        self._selected_case_id = 0
        self._busy = False
        self._science.clear_project()
        self.changed.emit()

    def shutdown(self) -> None:
        self._cancel_active()
        self._science.shutdown()
        self._pool.clear()
        self._pool.waitForDone(2500)

    @Property(QObject, constant=True)
    def science(self) -> QObject:
        return self._science

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(bool, notify=changed)
    def busy(self) -> bool:
        return self._busy

    @Property(str, notify=changed)
    def split(self) -> str:
        return self._split.value

    @Property(bool, notify=changed)
    def onlyIncorrect(self) -> bool:
        return self._only_incorrect

    @Property(int, notify=changed)
    def experimentPageNumber(self) -> int:
        return (self._experiment_offset // self._experiment_page_size) + 1

    @Property(int, notify=changed)
    def experimentTotal(self) -> int:
        return self._comparison_total()

    @Property(bool, notify=changed)
    def canPreviousExperimentPage(self) -> bool:
        return self._experiment_offset > 0

    @Property(bool, notify=changed)
    def canNextExperimentPage(self) -> bool:
        return (
            self._experiment_offset + self._experiment_page_size
            < self._comparison_total()
        )

    @Property(int, notify=changed)
    def pageNumber(self) -> int:
        return (self._offset // self._page_size) + 1

    @Property(int, notify=changed)
    def caseTotal(self) -> int:
        return self._case_total()

    @Property(bool, notify=changed)
    def canPreviousPage(self) -> bool:
        return self._offset > 0

    @Property(bool, notify=changed)
    def canNextPage(self) -> bool:
        return self._offset + self._page_size < self._case_total()

    @Property(list, notify=changed)
    def comparisonRows(self) -> list[dict[str, object]]:
        return self._load_comparison_rows()

    @Property(str, notify=changed)
    def selectedExperimentId(self) -> str:
        return self._selected_experiment_id

    @Property(dict, notify=changed)
    def selectedExperiment(self) -> dict[str, object]:
        if not self._selected_experiment_id:
            return {}
        for row in self._load_comparison_rows():
            if row["id"] == self._selected_experiment_id:
                return row
        return {}

    @Property(dict, notify=changed)
    def selectedSummary(self) -> dict[str, object]:
        if not self._workspace or not self._selected_experiment_id:
            return {}
        service = EvaluationService(self._workspace)
        progress = service.progress(self._selected_experiment_id, self._split)
        summary = service.summary(self._selected_experiment_id, self._split)
        return {
            "expected": progress.expected,
            "evaluated": progress.evaluated,
            "complete": progress.complete,
            "accuracy": summary.accuracy,
            "correct": summary.correct,
            "failures": summary.failures,
            "vetoFailures": summary.veto_failures,
            "meanLatencyMs": summary.mean_latency_ms,
            "p95LatencyMs": summary.p95_latency_ms,
        }

    @Property(list, notify=changed)
    def cases(self) -> list[dict[str, object]]:
        return [_case_row(item) for item in self._current_cases()]

    @Property(dict, notify=changed)
    def selectedCase(self) -> dict[str, object]:
        if not self._selected_case_id:
            return {}
        for item in self._current_cases():
            if item.id == self._selected_case_id:
                return _case_detail(item)
        return {}

    @Property(bool, notify=changed)
    def evaluatorAvailable(self) -> bool:
        return self._has_packaged_evaluator(self._selected_record())

    @Property(bool, notify=changed)
    def canRunEvaluation(self) -> bool:
        return self._can_run_evaluation()

    @Property(str, notify=changed)
    def evaluatorMessage(self) -> str:
        record = self._selected_record()
        if record is None:
            return "Select a completed experiment to inspect protected evidence."
        if self._workspace:
            progress = EvaluationService(self._workspace).progress(record.id, self._split)
            if progress.expected <= 0:
                return f"This dataset has no {self._split.value} examples."
            if progress.complete:
                return f"{self._split.value} evidence is complete and immutable."
            if progress.evaluated:
                return (
                    f"Resume {self._split.value} evaluation from "
                    f"{progress.evaluated}/{progress.expected} committed cases."
                )
        if self._has_packaged_evaluator(record):
            if self._adapter_id == PHASE_A_ADAPTER_ID:
                return (
                    f"Run protected {self._split.value} evaluation against the pinned "
                    "Frankenhomie residual validator."
                )
            return f"Run protected {self._split.value} evaluation."
        if self._adapter_id == PHASE_A_ADAPTER_ID:
            if record.trainer_id.startswith("baseline:"):
                return "Baseline evidence is created as an immutable protected evaluation run."
            if record.trainer_id == PHASE_A_TRAINER_ID and not record.contract_snapshot_id:
                return "Phase A evaluation requires the experiment's pinned contract snapshot."
            return "No packaged pinned Phase A evaluator is compatible with this experiment."
        return "No packaged evaluator is compatible with this experiment."

    @Slot(str)
    def selectExperiment(self, experiment_id: str) -> None:
        if not self._workspace or not self._project_id:
            return
        try:
            record = ExperimentService(self._workspace).get(experiment_id)
        except KeyError as exc:
            self.operationFailed.emit("Compare error", str(exc))
            return
        if record.project_id != self._project_id:
            self.operationFailed.emit(
                "Compare error",
                "Experiment does not belong to this project.",
            )
            return
        if record.status is not ExperimentStatus.COMPLETED:
            self.operationFailed.emit(
                "Compare error",
                "Only completed experiments can be compared.",
            )
            return
        self._experiment_offset = self._experiment_page_offset(record.id)
        self._selected_experiment_id = record.id
        self._offset = 0
        self._selected_case_id = 0
        self.changed.emit()

    @Slot()
    def previousExperimentPage(self) -> None:
        if self._experiment_offset <= 0:
            return
        self._experiment_offset = max(
            0,
            self._experiment_offset - self._experiment_page_size,
        )
        self._clear_selected_experiment()
        self.changed.emit()

    @Slot()
    def nextExperimentPage(self) -> None:
        if (
            self._experiment_offset + self._experiment_page_size
            >= self._comparison_total()
        ):
            return
        self._experiment_offset += self._experiment_page_size
        self._clear_selected_experiment()
        self.changed.emit()

    @Slot(str)
    def setSplit(self, split: str) -> None:
        normalized = split.strip().upper()
        if normalized not in {DatasetSplit.TEST.value, DatasetSplit.REDTEAM.value}:
            self.operationFailed.emit(
                "Compare error",
                f"Unsupported protected split: {split}",
            )
            return
        self._split = DatasetSplit(normalized)
        self._offset = 0
        self._selected_case_id = 0
        self.changed.emit()

    @Slot(bool)
    def setOnlyIncorrect(self, enabled: bool) -> None:
        self._only_incorrect = enabled
        self._offset = 0
        self._selected_case_id = 0
        self.changed.emit()

    @Slot()
    def previousPage(self) -> None:
        self._offset = max(0, self._offset - self._page_size)
        self._selected_case_id = 0
        self.changed.emit()

    @Slot()
    def nextPage(self) -> None:
        if self._offset + self._page_size < self._case_total():
            self._offset += self._page_size
            self._selected_case_id = 0
            self.changed.emit()

    @Slot(int)
    def selectCase(self, case_id: int) -> None:
        if any(item.id == case_id for item in self._current_cases()):
            self._selected_case_id = case_id
            self.changed.emit()

    @Slot()
    def runSelectedEvaluation(self) -> None:
        if not self._workspace or not self._can_run_evaluation():
            return
        record = self._selected_record()
        if record is None:
            return
        self._busy = True
        self._cancel_event = threading.Event()
        operation = _EvaluationOperation(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            experiment_id=record.id,
            split=self._split,
            cancel_event=self._cancel_event,
        )
        operation.signals.completed.connect(self._evaluation_completed)
        operation.signals.cancelled.connect(self._evaluation_cancelled)
        operation.signals.failed.connect(self._evaluation_failed)
        self._pool.start(operation)
        self.changed.emit()

    @Slot()
    def cancelEvaluation(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    @Slot(int, str, str)
    def _evaluation_completed(self, token: int, experiment_id: str, split: str) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self._science.refresh()
        self.changed.emit()
        self.operationCompleted.emit(
            f"Protected {split} evaluation complete for {experiment_id[:8]}"
        )

    @Slot(int, str, str)
    def _evaluation_cancelled(self, token: int, experiment_id: str, split: str) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self._science.refresh()
        self.changed.emit()
        self.operationCompleted.emit(
            f"{split} evaluation cancelled; committed evidence for "
            f"{experiment_id[:8]} is resumable"
        )

    @Slot(int, str, str, str)
    def _evaluation_failed(
        self,
        token: int,
        experiment_id: str,
        split: str,
        error: str,
    ) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self._science.refresh()
        self.changed.emit()
        self.operationFailed.emit(
            "Evaluation error",
            f"{experiment_id[:8]} {split}: {error}",
        )

    def _cancel_active(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None

    def _clear_selected_experiment(self) -> None:
        self._selected_experiment_id = ""
        self._offset = 0
        self._selected_case_id = 0

    def _selected_record(self) -> ExperimentRecord | None:
        if not self._workspace or not self._selected_experiment_id:
            return None
        try:
            record = ExperimentService(self._workspace).get(self._selected_experiment_id)
        except KeyError:
            return None
        if record.project_id != self._project_id:
            return None
        return record

    def _has_packaged_evaluator(self, record: ExperimentRecord | None) -> bool:
        if (
            record is None
            or record.status is not ExperimentStatus.COMPLETED
            or not record.model_artifact_digest
        ):
            return False
        if self._adapter_id == "generic":
            return (
                record.trainer_id == SPARSE_TRAINER_ID
                and record.runtime_pack_id == SPARSE_RUNTIME_PACK_ID
            )
        if self._adapter_id == PHASE_A_ADAPTER_ID:
            return bool(
                record.trainer_id == PHASE_A_TRAINER_ID
                and record.runtime_pack_id == PHASE_A_RUNTIME_PACK_ID
                and record.contract_snapshot_id
            )
        return False

    def _can_run_evaluation(self) -> bool:
        if self._busy or not self._workspace:
            return False
        record = self._selected_record()
        if not self._has_packaged_evaluator(record):
            return False
        assert record is not None
        progress = EvaluationService(self._workspace).progress(record.id, self._split)
        return progress.expected > 0 and not progress.complete

    def _comparison_total(self) -> int:
        if not self._workspace or not self._project_id:
            return 0
        with self._workspace.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE project_id=? AND status=?",
                (self._project_id, ExperimentStatus.COMPLETED.value),
            ).fetchone()
        return int(row[0]) if row else 0

    def _experiment_page_offset(self, experiment_id: str) -> int:
        if not self._workspace or not self._project_id:
            return 0
        with self._workspace.database.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM experiments e "
                "JOIN experiments target ON target.id=? "
                "WHERE e.project_id=? AND e.status=? AND "
                "(e.created_at>target.created_at OR "
                "(e.created_at=target.created_at AND e.id>target.id))",
                (
                    experiment_id,
                    self._project_id,
                    ExperimentStatus.COMPLETED.value,
                ),
            ).fetchone()
        position = int(row[0]) if row else 0
        return (position // self._experiment_page_size) * self._experiment_page_size

    def _load_comparison_rows(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        with self._workspace.database.connection() as conn:
            rows = conn.execute(
                "WITH page AS ("
                "SELECT e.id,e.dataset_id,e.trainer_id,e.runtime_pack_id,e.created_at,"
                "e.model_artifact_digest,d.name AS dataset_name,"
                "COALESCE(dp.example_count,0) AS expected "
                "FROM experiments e "
                "JOIN dataset_versions d ON d.id=e.dataset_id "
                "LEFT JOIN dataset_partitions dp "
                "ON dp.dataset_id=e.dataset_id AND dp.split=? "
                "WHERE e.project_id=? AND e.status=? "
                "ORDER BY e.created_at DESC,e.id DESC LIMIT ? OFFSET ?"
                ") "
                "SELECT p.id,p.dataset_id,p.trainer_id,p.runtime_pack_id,p.created_at,"
                "p.model_artifact_digest,p.dataset_name,p.expected,COUNT(ec.id) AS evaluated,"
                "COALESCE(SUM(ec.correct),0) AS correct,COUNT(f.id) AS failures,"
                "COALESCE(SUM(CASE WHEN f.severity='VETO' THEN 1 ELSE 0 END),0) AS vetoes,"
                "COALESCE(AVG(ec.latency_ms),0) AS mean_latency "
                "FROM page p "
                "LEFT JOIN evaluation_cases ec "
                "ON ec.experiment_id=p.id AND ec.split=? "
                "LEFT JOIN failures f ON f.id=ec.failure_id "
                "GROUP BY p.id,p.dataset_id,p.trainer_id,p.runtime_pack_id,p.created_at,"
                "p.model_artifact_digest,p.dataset_name,p.expected "
                "ORDER BY p.created_at DESC,p.id DESC",
                (
                    self._split.value,
                    self._project_id,
                    ExperimentStatus.COMPLETED.value,
                    self._experiment_page_size,
                    self._experiment_offset,
                    self._split.value,
                ),
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            expected = int(row["expected"])
            evaluated = int(row["evaluated"])
            correct = int(row["correct"])
            result.append(
                {
                    "id": str(row["id"]),
                    "shortId": str(row["id"])[:8],
                    "datasetId": str(row["dataset_id"]),
                    "datasetName": str(row["dataset_name"]),
                    "trainerId": str(row["trainer_id"]),
                    "runtimePackId": str(row["runtime_pack_id"]),
                    "createdAt": str(row["created_at"]),
                    "modelDigest": str(row["model_artifact_digest"] or ""),
                    "expected": expected,
                    "evaluated": evaluated,
                    "complete": expected > 0 and evaluated == expected,
                    "accuracy": correct / evaluated if evaluated else 0.0,
                    "failures": int(row["failures"]),
                    "vetoFailures": int(row["vetoes"]),
                    "meanLatencyMs": float(row["mean_latency"]),
                }
            )
        return result

    def _case_total(self) -> int:
        if not self._workspace or not self._selected_experiment_id:
            return 0
        return EvaluationService(self._workspace).case_count(
            self._selected_experiment_id,
            self._split,
            only_incorrect=self._only_incorrect,
        )

    def _current_cases(self) -> list[EvaluationCaseRecord]:
        if not self._workspace or not self._selected_experiment_id:
            return []
        return EvaluationService(self._workspace).page_cases(
            self._selected_experiment_id,
            self._split,
            only_incorrect=self._only_incorrect,
            offset=self._offset,
            limit=self._page_size,
        )


def _case_row(item: EvaluationCaseRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "exampleId": item.example_id,
        "correct": item.correct,
        "latencyMs": item.latency_ms,
        "failureId": item.failure_id or "",
        "expectedPreview": _json_preview(item.expected_json),
        "observedPreview": _json_preview(item.observed_json),
    }


def _case_detail(item: EvaluationCaseRecord) -> dict[str, object]:
    return {
        **_case_row(item),
        "split": item.split.value,
        "expected": _pretty_json(item.expected_json),
        "observed": _pretty_json(item.observed_json),
        "createdAt": item.created_at,
    }


def _pretty_json(raw: str) -> str:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(decoded, ensure_ascii=False, indent=2, sort_keys=True)


def _json_preview(raw: str, limit: int = 140) -> str:
    text = " ".join(_pretty_json(raw).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
