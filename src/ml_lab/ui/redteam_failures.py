from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, Signal, Slot

from ml_lab.core.models import ExperimentRecord, ExperimentStatus, FailureRecord, FailureSeverity
from ml_lab.experiments.service import ExperimentService
from ml_lab.failures.service import FailureService
from ml_lab.redteam.generic import run_generic_sparse_redteam
from ml_lab.redteam.service import RedTeamRunRecord, RedTeamService
from ml_lab.storage.workspace import Workspace
from ml_lab.trainers.service import SPARSE_RUNTIME_PACK_ID, SPARSE_TRAINER_ID


class _RedTeamSignals(QObject):
    completed = Signal(int, str, int, int)
    cancelled = Signal(int, str)
    failed = Signal(int, str)


class _RedTeamOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        experiment_id: str,
        seed: int,
        max_base_cases: int,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.experiment_id = experiment_id
        self.seed = seed
        self.max_base_cases = max_base_cases
        self.cancel_event = cancel_event
        self.signals = _RedTeamSignals()

    @Slot()
    def run(self) -> None:
        try:
            workspace = Workspace.open(self.workspace_root)
            summary = run_generic_sparse_redteam(
                workspace,
                self.experiment_id,
                seed=self.seed,
                max_base_cases=self.max_base_cases,
                cancelled=self.cancel_event.is_set,
            )
        except InterruptedError:
            self.signals.cancelled.emit(self.context_token, self.experiment_id)
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            self.signals.completed.emit(
                self.context_token,
                summary.run_id,
                summary.generated_cases,
                summary.failures,
            )


class RedTeamFailuresController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._selected_run_id = ""
        self._selected_failure_id = ""
        self._failure_severity = "ALL"
        self._failure_offset = 0
        self._run_offset = 0
        self._page_size = 100
        self._busy = False
        self._context_token = 0
        self._cancel_event: threading.Event | None = None
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_experiment_id = ""
        self._selected_run_id = ""
        self._selected_failure_id = ""
        self._failure_severity = "ALL"
        self._failure_offset = 0
        self._run_offset = 0
        self._busy = False
        self.changed.emit()

    def clear_project(self) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._selected_run_id = ""
        self._selected_failure_id = ""
        self._failure_severity = "ALL"
        self._failure_offset = 0
        self._run_offset = 0
        self._busy = False
        self.changed.emit()

    def shutdown(self) -> None:
        self._cancel_active()
        self._pool.clear()
        self._pool.waitForDone(2500)

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(bool, notify=changed)
    def busy(self) -> bool:
        return self._busy

    @Property(list, notify=changed)
    def experimentRows(self) -> list[dict[str, object]]:
        return [_experiment_row(item) for item in self._completed_experiments()]

    @Property(str, notify=changed)
    def selectedExperimentId(self) -> str:
        return self._selected_experiment_id

    @Property(bool, notify=changed)
    def canRunRedTeam(self) -> bool:
        return self._can_run_redteam()

    @Property(str, notify=changed)
    def runnerMessage(self) -> str:
        record = self._selected_experiment()
        if record is None:
            return "Select a completed experiment to run or inspect adversarial evidence."
        if self._can_run_record(record):
            return "Run the packaged deterministic generic text-perturbation suite."
        if self._adapter_id != "generic":
            return (
                "No adapter-supplied runnable red-team suite is installed for this project. "
                "Existing failure and run evidence remains browsable."
            )
        return "No packaged red-team runner is compatible with this experiment."

    @Property(list, notify=changed)
    def runRows(self) -> list[dict[str, object]]:
        return [_run_row(item) for item in self._current_runs()]

    @Property(int, notify=changed)
    def runTotal(self) -> int:
        if not self._workspace or not self._project_id:
            return 0
        return RedTeamService(self._workspace).count_for_project(self._project_id)

    @Property(int, notify=changed)
    def runPageNumber(self) -> int:
        return (self._run_offset // self._page_size) + 1

    @Property(bool, notify=changed)
    def canPreviousRunPage(self) -> bool:
        return self._run_offset > 0

    @Property(bool, notify=changed)
    def canNextRunPage(self) -> bool:
        return self._run_offset + self._page_size < self.runTotal

    @Property(dict, notify=changed)
    def selectedRun(self) -> dict[str, object]:
        if not self._workspace or not self._selected_run_id:
            return {}
        try:
            run = RedTeamService(self._workspace).get(self._selected_run_id)
        except KeyError:
            return {}
        if run.project_id != self._project_id:
            return {}
        return _run_detail(run)

    @Property(str, notify=changed)
    def failureSeverity(self) -> str:
        return self._failure_severity

    @Property(list, notify=changed)
    def failureRows(self) -> list[dict[str, object]]:
        return [_failure_row(item) for item in self._current_failures()]

    @Property(int, notify=changed)
    def failureTotal(self) -> int:
        return self._failure_count()

    @Property(int, notify=changed)
    def vetoFailureTotal(self) -> int:
        if not self._workspace or not self._project_id:
            return 0
        return FailureService(self._workspace).count_for_project(
            self._project_id,
            severity=FailureSeverity.VETO,
        )

    @Property(int, notify=changed)
    def regressionTotal(self) -> int:
        if not self._workspace or not self._project_id:
            return 0
        return FailureService(self._workspace).regression_count(self._project_id)

    @Property(int, notify=changed)
    def failurePageNumber(self) -> int:
        return (self._failure_offset // self._page_size) + 1

    @Property(bool, notify=changed)
    def canPreviousFailurePage(self) -> bool:
        return self._failure_offset > 0

    @Property(bool, notify=changed)
    def canNextFailurePage(self) -> bool:
        return self._failure_offset + self._page_size < self._failure_count()

    @Property(dict, notify=changed)
    def selectedFailure(self) -> dict[str, object]:
        if not self._workspace or not self._selected_failure_id:
            return {}
        service = FailureService(self._workspace)
        try:
            record = service.get(self._selected_failure_id)
        except KeyError:
            return {}
        if record.project_id != self._project_id:
            return {}
        return _failure_detail(
            record,
            is_regression=service.is_regression_case(record.id),
        )

    @Slot(str)
    def selectExperiment(self, experiment_id: str) -> None:
        if any(item.id == experiment_id for item in self._completed_experiments()):
            self._selected_experiment_id = experiment_id
            self.changed.emit()

    @Slot(int, int)
    def runRedTeam(self, seed: int, max_base_cases: int) -> None:
        if not self._workspace or not self._can_run_redteam():
            return
        record = self._selected_experiment()
        if record is None:
            return
        self._busy = True
        self._cancel_event = threading.Event()
        operation = _RedTeamOperation(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            experiment_id=record.id,
            seed=seed,
            max_base_cases=max_base_cases,
            cancel_event=self._cancel_event,
        )
        operation.signals.completed.connect(self._run_completed)
        operation.signals.cancelled.connect(self._run_cancelled)
        operation.signals.failed.connect(self._run_failed)
        self._pool.start(operation)
        self.changed.emit()

    @Slot()
    def cancelRedTeam(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    @Slot(str)
    def selectRun(self, run_id: str) -> None:
        if any(item.id == run_id for item in self._current_runs()):
            self._selected_run_id = run_id
            self.changed.emit()

    @Slot()
    def previousRunPage(self) -> None:
        self._run_offset = max(0, self._run_offset - self._page_size)
        self._selected_run_id = ""
        self.changed.emit()

    @Slot()
    def nextRunPage(self) -> None:
        if self._run_offset + self._page_size < self.runTotal:
            self._run_offset += self._page_size
            self._selected_run_id = ""
            self.changed.emit()

    @Slot(str)
    def setFailureSeverity(self, severity: str) -> None:
        normalized = severity.strip().upper()
        if normalized not in {"ALL", FailureSeverity.VETO.value, FailureSeverity.NON_VETO.value}:
            self.operationFailed.emit("Failure filter", f"Unsupported severity: {severity}")
            return
        self._failure_severity = normalized
        self._failure_offset = 0
        self._selected_failure_id = ""
        self.changed.emit()

    @Slot(str)
    def selectFailure(self, failure_id: str) -> None:
        if any(item.id == failure_id for item in self._current_failures()):
            self._selected_failure_id = failure_id
            self.changed.emit()

    @Slot()
    def previousFailurePage(self) -> None:
        self._failure_offset = max(0, self._failure_offset - self._page_size)
        self._selected_failure_id = ""
        self.changed.emit()

    @Slot()
    def nextFailurePage(self) -> None:
        if self._failure_offset + self._page_size < self._failure_count():
            self._failure_offset += self._page_size
            self._selected_failure_id = ""
            self.changed.emit()

    @Slot(str)
    def promoteSelectedFailure(self, suite_name: str) -> None:
        if not self._workspace or not self._selected_failure_id:
            return
        try:
            service = FailureService(self._workspace)
            record = service.get(self._selected_failure_id)
            if record.project_id != self._project_id:
                raise ValueError("Failure does not belong to the selected project.")
            service.promote_to_regression(record.id, suite_name=suite_name)
        except Exception as exc:
            self.operationFailed.emit("Regression promotion", str(exc))
            return
        self.changed.emit()
        self.operationCompleted.emit("Failure promoted to immutable regression membership")

    @Slot(int, str, int, int)
    def _run_completed(self, token: int, run_id: str, generated: int, failures: int) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self._run_offset = 0
        self._failure_offset = 0
        self._selected_run_id = run_id
        self.changed.emit()
        self.operationCompleted.emit(
            f"Red-team run complete: {generated} generated case(s), {failures} failure(s)"
        )

    @Slot(int, str)
    def _run_cancelled(self, token: int, experiment_id: str) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self.changed.emit()
        self.operationCompleted.emit(
            f"Red-team scoring cancelled for experiment {experiment_id[:8]}; preserved evidence remains visible"
        )

    @Slot(int, str)
    def _run_failed(self, token: int, error: str) -> None:
        if token != self._context_token:
            return
        self._busy = False
        self._cancel_event = None
        self.changed.emit()
        self.operationFailed.emit("Red-team run failed", error)

    def _cancel_active(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None

    def _completed_experiments(self) -> list[ExperimentRecord]:
        if not self._workspace or not self._project_id:
            return []
        return [
            item
            for item in ExperimentService(self._workspace).list_for_project(self._project_id)
            if item.status is ExperimentStatus.COMPLETED
        ]

    def _selected_experiment(self) -> ExperimentRecord | None:
        if not self._selected_experiment_id:
            return None
        return next(
            (
                item
                for item in self._completed_experiments()
                if item.id == self._selected_experiment_id
            ),
            None,
        )

    def _can_run_record(self, record: ExperimentRecord) -> bool:
        return (
            self._adapter_id == "generic"
            and record.trainer_id == SPARSE_TRAINER_ID
            and record.runtime_pack_id == SPARSE_RUNTIME_PACK_ID
            and bool(record.model_artifact_digest)
        )

    def _can_run_redteam(self) -> bool:
        return not self._busy and (record := self._selected_experiment()) is not None and self._can_run_record(record)

    def _current_runs(self) -> list[RedTeamRunRecord]:
        if not self._workspace or not self._project_id:
            return []
        return RedTeamService(self._workspace).page_for_project(
            self._project_id,
            offset=self._run_offset,
            limit=self._page_size,
        )

    def _failure_severity_value(self) -> FailureSeverity | None:
        if self._failure_severity == "ALL":
            return None
        return FailureSeverity(self._failure_severity)

    def _current_failures(self) -> list[FailureRecord]:
        if not self._workspace or not self._project_id:
            return []
        return FailureService(self._workspace).page_for_project(
            self._project_id,
            severity=self._failure_severity_value(),
            offset=self._failure_offset,
            limit=self._page_size,
        )

    def _failure_count(self) -> int:
        if not self._workspace or not self._project_id:
            return 0
        return FailureService(self._workspace).count_for_project(
            self._project_id,
            severity=self._failure_severity_value(),
        )


def _experiment_row(item: ExperimentRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "name": item.name,
        "trainer": item.trainer_id,
        "runtime": item.runtime_pack_id,
        "createdAt": item.created_at,
    }


def _run_row(item: RedTeamRunRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "status": item.status.value,
        "seed": item.seed,
        "suite": item.mutator_version,
        "experimentId": item.experiment_id or "",
        "createdAt": item.created_at,
    }


def _run_detail(item: RedTeamRunRecord) -> dict[str, object]:
    return {
        **_run_row(item),
        "datasetId": item.dataset_id or "",
        "manifestSha256": item.manifest_artifact_digest or "",
        "completedAt": item.completed_at or "",
    }


def _failure_row(item: FailureRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "kind": item.kind,
        "severity": item.severity.value,
        "status": item.status.value,
        "exampleId": item.example_id or "",
        "split": item.split.value if item.split else "",
        "redteamRunId": item.redteam_run_id or "",
        "createdAt": item.created_at,
    }


def _failure_detail(item: FailureRecord, *, is_regression: bool) -> dict[str, object]:
    return {
        **_failure_row(item),
        "experimentId": item.experiment_id or "",
        "datasetId": item.dataset_id or "",
        "expected": _pretty_json(item.expected_json),
        "observed": _pretty_json(item.observed_json),
        "evidenceSha256": item.evidence_artifact_digest or "",
        "isRegression": is_regression,
    }


def _pretty_json(raw: str) -> str:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return json.dumps(decoded, ensure_ascii=False, indent=2, sort_keys=True)
