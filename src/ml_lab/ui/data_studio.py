from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, QUrl, Signal, Slot

from ml_lab.adapters.builtin import dataset_validator_for
from ml_lab.core.models import DatasetExample, DatasetSplit, DatasetState, DatasetVersion
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace


class _DatasetOperationSignals(QObject):
    completed = Signal(str, str, object)
    failed = Signal(str, str, str)


class _DatasetOperation(QRunnable):
    def __init__(
        self,
        *,
        workspace_root: Path,
        operation: str,
        dataset_id: str,
        adapter_id: str,
        source: Path | None = None,
    ) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.operation = operation
        self.dataset_id = dataset_id
        self.adapter_id = adapter_id
        self.source = source
        self.signals = _DatasetOperationSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = _perform_dataset_operation(
                workspace_root=self.workspace_root,
                operation=self.operation,
                dataset_id=self.dataset_id,
                adapter_id=self.adapter_id,
                source=self.source,
            )
        except Exception as exc:
            self.signals.failed.emit(
                self.operation,
                self.dataset_id,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.signals.completed.emit(self.operation, self.dataset_id, result)


class DataStudioController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_dataset_id = ""
        self._split_filter = "ALL"
        self._offset = 0
        self._page_size = 100
        self._busy = False
        self._busy_message = ""
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_dataset_id = ""
        self._split_filter = "ALL"
        self._offset = 0
        self.changed.emit()

    def clear_project(self) -> None:
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_dataset_id = ""
        self._split_filter = "ALL"
        self._offset = 0
        self.changed.emit()

    def shutdown(self) -> None:
        self._pool.clear()
        self._pool.waitForDone(2500)

    @Property(bool, notify=changed)
    def hasProject(self) -> bool:
        return self._workspace is not None and bool(self._project_id)

    @Property(bool, notify=changed)
    def busy(self) -> bool:
        return self._busy

    @Property(str, notify=changed)
    def busyMessage(self) -> str:
        return self._busy_message

    @Property(str, notify=changed)
    def splitFilter(self) -> str:
        return self._split_filter

    @Property(int, notify=changed)
    def pageNumber(self) -> int:
        return (self._offset // self._page_size) + 1

    @Property(bool, notify=changed)
    def canPreviousPage(self) -> bool:
        return self._offset > 0

    @Property(bool, notify=changed)
    def canNextPage(self) -> bool:
        total = self._filtered_total()
        return self._offset + self._page_size < total

    @Property(list, notify=changed)
    def datasets(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        service = DatasetService(self._workspace)
        return [_dataset_dict(item) for item in service.list_for_project(self._project_id)]

    @Property(dict, notify=changed)
    def selectedDataset(self) -> dict[str, object]:
        item = self._selected_dataset()
        return _dataset_dict(item) if item else {}

    @Property(dict, notify=changed)
    def splitCounts(self) -> dict[str, int]:
        if not self._workspace or not self._selected_dataset_id:
            return {split.value: 0 for split in DatasetSplit} | {"ALL": 0}
        with self._workspace.database.connection() as conn:
            rows = conn.execute(
                "SELECT split,COUNT(*) AS count FROM dataset_examples "
                "WHERE dataset_id=? GROUP BY split",
                (self._selected_dataset_id,),
            ).fetchall()
        counts = {split.value: 0 for split in DatasetSplit}
        for row in rows:
            counts[str(row["split"])] = int(row["count"])
        counts["ALL"] = sum(counts.values())
        return counts

    @Property(list, notify=changed)
    def examples(self) -> list[dict[str, object]]:
        if not self._workspace or not self._selected_dataset_id:
            return []
        split = None if self._split_filter == "ALL" else DatasetSplit(self._split_filter)
        service = DatasetService(self._workspace)
        return [
            _example_dict(item)
            for item in service.page_examples(
                self._selected_dataset_id,
                split=split,
                offset=self._offset,
                limit=self._page_size,
            )
        ]

    @Slot(str)
    def createDataset(self, name: str) -> None:
        if not self._workspace or not self._project_id:
            self.operationFailed.emit("Dataset error", "Open a project first.")
            return
        try:
            item = DatasetService(self._workspace).create(self._project_id, name)
        except Exception as exc:
            self.operationFailed.emit("Dataset error", str(exc))
            return
        self._selected_dataset_id = item.id
        self._split_filter = "ALL"
        self._offset = 0
        self.changed.emit()
        self.operationCompleted.emit(f"Created dataset {item.name}")

    @Slot(str)
    def selectDataset(self, dataset_id: str) -> None:
        if not self._workspace:
            return
        try:
            item = DatasetService(self._workspace).get(dataset_id)
        except Exception as exc:
            self.operationFailed.emit("Dataset error", str(exc))
            return
        if item.project_id != self._project_id:
            self.operationFailed.emit("Dataset error", "Dataset does not belong to this project.")
            return
        self._selected_dataset_id = item.id
        self._split_filter = "ALL"
        self._offset = 0
        self.changed.emit()

    @Slot(str)
    def setSplitFilter(self, split: str) -> None:
        normalized = split.upper()
        if normalized != "ALL" and normalized not in {item.value for item in DatasetSplit}:
            self.operationFailed.emit("Filter error", f"Unknown dataset split: {split}")
            return
        self._split_filter = normalized
        self._offset = 0
        self.changed.emit()

    @Slot()
    def previousPage(self) -> None:
        self._offset = max(0, self._offset - self._page_size)
        self.changed.emit()

    @Slot()
    def nextPage(self) -> None:
        if self.canNextPage:
            self._offset += self._page_size
            self.changed.emit()

    @Slot(str)
    def importJsonl(self, source: str) -> None:
        item = self._selected_dataset()
        if item is None:
            self.operationFailed.emit("Import error", "Select a dataset first.")
            return
        if item.state is not DatasetState.DRAFT:
            self.operationFailed.emit("Import error", "Only DRAFT datasets accept imports.")
            return
        path = _url_or_path(source)
        self._start_operation("import", item.id, source=path)

    @Slot()
    def freezeSelectedDataset(self) -> None:
        item = self._selected_dataset()
        if item is None:
            self.operationFailed.emit("Freeze error", "Select a dataset first.")
            return
        if item.state is not DatasetState.DRAFT:
            self.operationFailed.emit("Freeze error", "Only DRAFT datasets can be frozen.")
            return
        self._start_operation("freeze", item.id)

    def _selected_dataset(self) -> DatasetVersion | None:
        if not self._workspace or not self._selected_dataset_id:
            return None
        try:
            item = DatasetService(self._workspace).get(self._selected_dataset_id)
        except KeyError:
            self._selected_dataset_id = ""
            return None
        if item.project_id != self._project_id:
            self._selected_dataset_id = ""
            return None
        return item

    def _filtered_total(self) -> int:
        counts = self.splitCounts
        return int(counts.get(self._split_filter, 0))

    def _start_operation(
        self,
        operation: str,
        dataset_id: str,
        *,
        source: Path | None = None,
    ) -> None:
        if not self._workspace:
            return
        if self._busy:
            self.operationFailed.emit(
                "Data Studio busy",
                "Finish the current data operation first.",
            )
            return
        self._busy = True
        self._busy_message = "Importing dataset…" if operation == "import" else "Freezing dataset…"
        self.changed.emit()
        task = _DatasetOperation(
            workspace_root=self._workspace.root,
            operation=operation,
            dataset_id=dataset_id,
            adapter_id=self._adapter_id,
            source=source,
        )
        task.signals.completed.connect(self._operation_completed)
        task.signals.failed.connect(self._operation_failed)
        self._pool.start(task)

    @Slot(str, str, object)
    def _operation_completed(self, operation: str, dataset_id: str, result: object) -> None:
        if dataset_id != self._selected_dataset_id:
            return
        self._busy = False
        self._busy_message = ""
        self._offset = 0
        self.changed.emit()
        details = result if isinstance(result, dict) else {}
        if operation == "import":
            imported = int(details.get("imported", 0))
            rejected = int(details.get("rejected", 0))
            self.operationCompleted.emit(
                f"Import complete: {imported} imported, {rejected} rejected"
            )
        else:
            self.operationCompleted.emit("Dataset frozen with immutable split artifacts")

    @Slot(str, str, str)
    def _operation_failed(self, operation: str, dataset_id: str, error: str) -> None:
        if dataset_id != self._selected_dataset_id:
            return
        self._busy = False
        self._busy_message = ""
        self.changed.emit()
        title = "Import error" if operation == "import" else "Freeze error"
        self.operationFailed.emit(title, error)


def _perform_dataset_operation(
    *,
    workspace_root: Path,
    operation: str,
    dataset_id: str,
    adapter_id: str,
    source: Path | None,
) -> dict[str, object]:
    workspace = Workspace.open(workspace_root)
    service = DatasetService(workspace)
    if operation == "import":
        if source is None:
            raise ValueError("Import source is required.")
        validator = dataset_validator_for(adapter_id)
        return service.import_jsonl(dataset_id, source, validator=validator).to_dict()
    if operation == "freeze":
        frozen = service.freeze(dataset_id)
        return {
            "dataset_id": frozen.id,
            "state": frozen.state.value,
            "manifest_sha256": frozen.manifest_artifact_digest or "",
            "leakage_report_sha256": frozen.leakage_report_artifact_digest or "",
        }
    raise ValueError(f"Unsupported dataset operation: {operation}")


def _dataset_dict(item: DatasetVersion) -> dict[str, object]:
    value = asdict(item)
    value["state"] = item.state.value
    return value


def _example_dict(item: DatasetExample) -> dict[str, object]:
    return {
        "exampleId": item.example_id,
        "split": item.split.value,
        "sourceId": item.source_id,
        "lineageGroup": item.lineage_group,
        "payload": _compact_json(item.payload_json),
        "label": _compact_json(item.label_json),
        "tags": _compact_json(item.tags_json),
    }


def _compact_json(value: str, *, limit: int = 180) -> str:
    try:
        rendered = json.dumps(json.loads(value), ensure_ascii=False, sort_keys=True)
    except json.JSONDecodeError:
        rendered = value
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def _url_or_path(value: str) -> Path:
    if value.startswith("file:"):
        return Path(QUrl(value).toLocalFile())
    return Path(value)
