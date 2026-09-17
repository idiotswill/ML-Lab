from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, Signal, Slot

from ml_lab.adapters.phase_a import PHASE_A_ADAPTER_ID
from ml_lab.baselines.phase_a import (
    phase_a_baseline_options,
    run_phase_a_baseline,
)
from ml_lab.core.models import DatasetSplit, ExperimentRecord, ExperimentStatus
from ml_lab.evaluation.phase_a_metrics import (
    load_pinned_provider_reference,
    metric_rows_for_ui,
    summarize_phase_a_experiment,
)
from ml_lab.experiments.service import ExperimentService
from ml_lab.storage.workspace import Workspace


class _ScienceSignals(QObject):
    completed = Signal(int, str, str, object, object)
    failed = Signal(int, str, str, str)


class _ScienceAnalysis(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        experiment_id: str,
        split: DatasetSplit,
        snapshot_id: str | None,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.experiment_id = experiment_id
        self.split = split
        self.snapshot_id = snapshot_id
        self.signals = _ScienceSignals()

    @Slot()
    def run(self) -> None:
        try:
            workspace = Workspace.open(self.workspace_root)
            metrics = metric_rows_for_ui(
                summarize_phase_a_experiment(
                    workspace,
                    self.experiment_id,
                    self.split,
                )
            )
            provider = (
                load_pinned_provider_reference(workspace, self.snapshot_id)
                if self.snapshot_id
                else None
            )
            provider_payload = _provider_payload(provider)
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                self.experiment_id,
                self.split.value,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.signals.completed.emit(
            self.context_token,
            self.experiment_id,
            self.split.value,
            metrics,
            provider_payload,
        )


class _BaselineSignals(QObject):
    completed = Signal(int, str, str)
    cancelled = Signal(int, str)
    failed = Signal(int, str, str)


class _BaselineOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        project_id: str,
        dataset_id: str,
        snapshot_id: str,
        baseline_id: str,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.snapshot_id = snapshot_id
        self.baseline_id = baseline_id
        self.cancel_event = cancel_event
        self.signals = _BaselineSignals()

    @Slot()
    def run(self) -> None:
        try:
            workspace = Workspace.open(self.workspace_root)
            result = run_phase_a_baseline(
                workspace,
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                contract_snapshot_id=self.snapshot_id,
                baseline_id=self.baseline_id,
                cancelled=self.cancel_event.is_set,
            )
        except InterruptedError:
            self.signals.cancelled.emit(self.context_token, self.baseline_id)
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                self.baseline_id,
                f"{type(exc).__name__}: {exc}",
            )
        else:
            self.signals.completed.emit(
                self.context_token,
                self.baseline_id,
                result.experiment.id,
            )


class PhaseAScienceController(QObject):
    changed = Signal()
    baselineReady = Signal(str)
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._split = DatasetSplit.TEST
        self._context_token = 0
        self._analysis_loading = False
        self._baseline_busy = False
        self._busy_message = ""
        self._metrics: list[dict[str, object]] = []
        self._provider: dict[str, object] = {}
        self._cancel_event: threading.Event | None = None
        self._analysis_pool = QThreadPool(self)
        self._analysis_pool.setMaxThreadCount(1)
        self._baseline_pool = QThreadPool(self)
        self._baseline_pool.setMaxThreadCount(1)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_experiment_id = ""
        self._split = DatasetSplit.TEST
        self._analysis_loading = False
        self._baseline_busy = False
        self._busy_message = ""
        self._metrics = []
        self._provider = {}
        self.changed.emit()

    def clear_project(self) -> None:
        self._cancel_active()
        self._context_token += 1
        self._workspace = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_experiment_id = ""
        self._split = DatasetSplit.TEST
        self._analysis_loading = False
        self._baseline_busy = False
        self._busy_message = ""
        self._metrics = []
        self._provider = {}
        self.changed.emit()

    def shutdown(self) -> None:
        self._cancel_active()
        self._analysis_pool.clear()
        self._baseline_pool.clear()
        self._analysis_pool.waitForDone(2500)
        self._baseline_pool.waitForDone(2500)

    @Property(bool, notify=changed)
    def enabled(self) -> bool:
        return bool(
            self._workspace is not None
            and self._project_id
            and self._adapter_id == PHASE_A_ADAPTER_ID
        )

    @Property(bool, notify=changed)
    def busy(self) -> bool:
        return self._baseline_busy

    @Property(bool, notify=changed)
    def loadingMetrics(self) -> bool:
        return self._analysis_loading

    @Property(str, notify=changed)
    def busyMessage(self) -> str:
        return self._busy_message

    @Property(list, notify=changed)
    def scientificMetrics(self) -> list[dict[str, object]]:
        return list(self._metrics)

    @Property(dict, notify=changed)
    def providerReference(self) -> dict[str, object]:
        return dict(self._provider)

    @Property(list, notify=changed)
    def baselineOptions(self) -> list[dict[str, object]]:
        if not self._workspace or self._adapter_id != PHASE_A_ADAPTER_ID:
            return []
        record = self._selected_record()
        if record is None or record.contract_snapshot_id is None:
            return []
        rows: list[dict[str, object]] = []
        for option in phase_a_baseline_options(self._adapter_id):
            existing_id = self._completed_baseline_id(record, option.baseline_id)
            rows.append(
                {
                    "baselineId": option.baseline_id,
                    "name": option.display_name,
                    "description": option.description,
                    "completed": bool(existing_id),
                    "experimentId": existing_id or "",
                }
            )
        return rows

    @Slot(str, str)
    def setSelection(self, experiment_id: str, split: str) -> None:
        if not self._workspace or self._adapter_id != PHASE_A_ADAPTER_ID:
            return
        normalized = split.strip().upper()
        if normalized not in {DatasetSplit.TEST.value, DatasetSplit.REDTEAM.value}:
            return
        clean_id = experiment_id.strip()
        if not clean_id:
            self._selected_experiment_id = ""
            self._metrics = []
            self._provider = {}
            self._analysis_loading = False
            self.changed.emit()
            return
        try:
            record = ExperimentService(self._workspace).get(clean_id)
        except KeyError:
            return
        if record.project_id != self._project_id or record.status is not ExperimentStatus.COMPLETED:
            return
        selected_split = DatasetSplit(normalized)
        if (
            clean_id == self._selected_experiment_id
            and selected_split is self._split
        ):
            return
        self._selected_experiment_id = clean_id
        self._split = selected_split
        self._start_analysis(record)

    @Slot()
    def refresh(self) -> None:
        record = self._selected_record()
        if record is not None:
            self._start_analysis(record)

    @Slot(str)
    def runBaseline(self, baseline_id: str) -> None:
        if not self._workspace or self._baseline_busy:
            return
        record = self._selected_record()
        if record is None or record.contract_snapshot_id is None:
            self.operationFailed.emit(
                "Baseline error",
                "Select a completed Phase A experiment pinned to a contract snapshot.",
            )
            return
        valid_ids = {
            item.baseline_id for item in phase_a_baseline_options(self._adapter_id)
        }
        if baseline_id not in valid_ids:
            self.operationFailed.emit("Baseline error", "Unknown Phase A baseline.")
            return
        existing_id = self._completed_baseline_id(record, baseline_id)
        if existing_id:
            self.baselineReady.emit(existing_id)
            self.operationCompleted.emit("Existing immutable baseline evidence selected")
            return

        self._baseline_busy = True
        self._busy_message = "Running same-dataset protected baseline…"
        self._cancel_event = threading.Event()
        operation = _BaselineOperation(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            project_id=self._project_id,
            dataset_id=record.dataset_id,
            snapshot_id=record.contract_snapshot_id,
            baseline_id=baseline_id,
            cancel_event=self._cancel_event,
        )
        operation.signals.completed.connect(self._baseline_completed)
        operation.signals.cancelled.connect(self._baseline_cancelled)
        operation.signals.failed.connect(self._baseline_failed)
        self._baseline_pool.start(operation)
        self.changed.emit()

    @Slot()
    def cancelBaseline(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()

    @Slot(int, str, str, object, object)
    def _analysis_completed(
        self,
        token: int,
        experiment_id: str,
        split: str,
        metrics: object,
        provider: object,
    ) -> None:
        if not self._active_analysis(token, experiment_id, split):
            return
        self._analysis_loading = False
        self._metrics = list(metrics) if isinstance(metrics, list) else []
        self._provider = dict(provider) if isinstance(provider, dict) else {}
        self.changed.emit()

    @Slot(int, str, str, str)
    def _analysis_failed(
        self,
        token: int,
        experiment_id: str,
        split: str,
        error: str,
    ) -> None:
        if not self._active_analysis(token, experiment_id, split):
            return
        self._analysis_loading = False
        self._metrics = []
        self._provider = {}
        self.changed.emit()
        self.operationFailed.emit("Scientific comparison error", error)

    @Slot(int, str, str)
    def _baseline_completed(self, token: int, baseline_id: str, experiment_id: str) -> None:
        if token != self._context_token:
            return
        self._baseline_busy = False
        self._busy_message = ""
        self._cancel_event = None
        self.changed.emit()
        self.baselineReady.emit(experiment_id)
        self.operationCompleted.emit(
            f"Completed {baseline_id}; immutable TEST/REDTEAM evidence committed"
        )

    @Slot(int, str)
    def _baseline_cancelled(self, token: int, baseline_id: str) -> None:
        if token != self._context_token:
            return
        self._baseline_busy = False
        self._busy_message = ""
        self._cancel_event = None
        self.changed.emit()
        self.operationCompleted.emit(
            f"Cancelled {baseline_id}; any committed case evidence remains immutable"
        )

    @Slot(int, str, str)
    def _baseline_failed(self, token: int, baseline_id: str, error: str) -> None:
        if token != self._context_token:
            return
        self._baseline_busy = False
        self._busy_message = ""
        self._cancel_event = None
        self.changed.emit()
        self.operationFailed.emit("Baseline error", f"{baseline_id}: {error}")

    def _start_analysis(self, record: ExperimentRecord) -> None:
        if not self._workspace:
            return
        self._analysis_loading = True
        self._metrics = []
        self._provider = {}
        operation = _ScienceAnalysis(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            experiment_id=record.id,
            split=self._split,
            snapshot_id=record.contract_snapshot_id,
        )
        operation.signals.completed.connect(self._analysis_completed)
        operation.signals.failed.connect(self._analysis_failed)
        self._analysis_pool.clear()
        self._analysis_pool.start(operation)
        self.changed.emit()

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

    def _completed_baseline_id(self, record: ExperimentRecord, baseline_id: str) -> str | None:
        if not self._workspace or record.contract_snapshot_id is None:
            return None
        trainer_id = f"baseline:{baseline_id}"
        with self._workspace.database.connection() as conn:
            row = conn.execute(
                "SELECT id FROM experiments WHERE project_id=? AND dataset_id=? "
                "AND contract_snapshot_id=? AND trainer_id=? AND status=? "
                "ORDER BY created_at DESC LIMIT 1",
                (
                    self._project_id,
                    record.dataset_id,
                    record.contract_snapshot_id,
                    trainer_id,
                    ExperimentStatus.COMPLETED.value,
                ),
            ).fetchone()
        return str(row["id"]) if row is not None else None

    def _active_analysis(self, token: int, experiment_id: str, split: str) -> bool:
        return (
            token == self._context_token
            and experiment_id == self._selected_experiment_id
            and split == self._split.value
        )

    def _cancel_active(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cancel_event = None


def _provider_payload(provider: object) -> dict[str, object]:
    if provider is None:
        return {"available": False}
    from ml_lab.evaluation.phase_a_metrics import PhaseAProviderReference

    if not isinstance(provider, PhaseAProviderReference):
        raise TypeError("Invalid Phase A provider reference")
    return {
        "available": True,
        "model": provider.model,
        "timestamp": provider.timestamp,
        "sourceHead": provider.source_head,
        "sourceDirty": provider.source_dirty,
        "independentQa": provider.independent_qa,
        "corpusSha256": provider.corpus_sha256,
        "totalCases": provider.total_cases,
        "modelCalls": provider.model_calls,
        "differentCorpus": True,
        "metrics": metric_rows_for_ui(provider.metrics),
    }
