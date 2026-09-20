from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Property, QObject, QRunnable, QThreadPool, QUrl, Signal, Slot

from ml_lab.adapters.builtin import contract_capture_descriptor_for, dataset_validator_for
from ml_lab.contracts.snapshot import ContractSnapshotService
from ml_lab.core.models import DatasetExample, DatasetSplit, DatasetState, DatasetVersion
from ml_lab.datasets.service import DatasetService
from ml_lab.storage.workspace import Workspace


class _DatasetOperationSignals(QObject):
    completed = Signal(int, str, str, object)
    failed = Signal(int, str, str, str)


class _DatasetOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        operation: str,
        dataset_id: str,
        adapter_id: str,
        source: Path | None = None,
        split_filter: str = "ALL",
        page_size: int = 100,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.operation = operation
        self.dataset_id = dataset_id
        self.adapter_id = adapter_id
        self.source = source
        self.split_filter = split_filter
        self.page_size = page_size
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
                split_filter=self.split_filter,
                page_size=self.page_size,
            )
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                self.operation,
                self.dataset_id,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.signals.completed.emit(
            self.context_token,
            self.operation,
            self.dataset_id,
            result,
        )


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
        self._context_token = 0
        self._datasets_cache: list[dict[str, object]] = []
        self._selected_dataset_cache: dict[str, object] = {}
        self._split_counts_cache = _empty_split_counts()
        self._examples_cache: list[dict[str, object]] = []
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        self._context_token += 1
        self._workspace = workspace
        self._project_id = project_id
        self._adapter_id = adapter_id
        self._selected_dataset_id = ""
        self._split_filter = "ALL"
        self._offset = 0
        self._busy = False
        self._busy_message = ""
        self._refresh_view_cache()
        self.changed.emit()

    def clear_project(self) -> None:
        self._context_token += 1
        self._project_id = ""
        self._adapter_id = "generic"
        self._selected_dataset_id = ""
        self._split_filter = "ALL"
        self._offset = 0
        self._busy = False
        self._busy_message = ""
        self._reset_view_cache()
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

    @Property(bool, notify=changed)
    def requiresContractSnapshot(self) -> bool:
        if not self._project_id:
            return False
        try:
            return contract_capture_descriptor_for(self._adapter_id) is not None
        except KeyError:
            return False

    @Property(list, notify=changed)
    def contractSnapshots(self) -> list[dict[str, str]]:
        if not self._workspace or not self._project_id:
            return []
        return [
            {
                "id": item.id,
                "name": f"{item.commit_sha[:12]} · {item.contract_version}",
                "commitSha": item.commit_sha,
                "contractVersion": item.contract_version,
            }
            for item in ContractSnapshotService(self._workspace).list_for_project(
                self._project_id
            )
            if item.adapter_id == self._adapter_id
        ]

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
        return self._datasets_cache

    @Property(dict, notify=changed)
    def selectedDataset(self) -> dict[str, object]:
        return self._selected_dataset_cache

    @Property(dict, notify=changed)
    def splitCounts(self) -> dict[str, int]:
        return self._split_counts_cache

    @Property(list, notify=changed)
    def examples(self) -> list[dict[str, object]]:
        return self._examples_cache

    @Slot(str, str)
    def createDataset(self, name: str, contract_snapshot_id: str) -> None:
        if not self._workspace or not self._project_id:
            self.operationFailed.emit("Dataset error", "Open a project first.")
            return
        try:
            snapshot_id = contract_snapshot_id.strip() or None
            descriptor = contract_capture_descriptor_for(self._adapter_id)
            if descriptor is not None and snapshot_id is None:
                raise ValueError(
                    "This adapter requires a frozen contract snapshot before creating a dataset."
                )
            if snapshot_id is not None:
                snapshot = ContractSnapshotService(self._workspace).get(snapshot_id)
                if snapshot.project_id != self._project_id:
                    raise ValueError("Contract snapshot belongs to a different project.")
                if snapshot.adapter_id != self._adapter_id:
                    raise ValueError("Contract snapshot adapter does not match this project.")
            item = DatasetService(self._workspace).create(
                self._project_id,
                name,
                contract_snapshot_id=snapshot_id,
            )
        except Exception as exc:
            self.operationFailed.emit("Dataset error", str(exc))
            return
        self._selected_dataset_id = item.id
        self._split_filter = "ALL"
        self._offset = 0
        self._refresh_view_cache()
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
        self._refresh_view_cache()
        self.changed.emit()

    @Slot(str)
    def setSplitFilter(self, split: str) -> None:
        normalized = split.upper()
        if normalized != "ALL" and normalized not in {item.value for item in DatasetSplit}:
            self.operationFailed.emit("Filter error", f"Unknown dataset split: {split}")
            return
        self._split_filter = normalized
        self._offset = 0
        self._refresh_view_cache()
        self.changed.emit()

    @Slot()
    def previousPage(self) -> None:
        self._offset = max(0, self._offset - self._page_size)
        self._refresh_view_cache()
        self.changed.emit()

    @Slot()
    def nextPage(self) -> None:
        if self._offset + self._page_size < self._filtered_total():
            self._offset += self._page_size
            self._refresh_view_cache()
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
            self._reset_view_cache()
            return None
        if item.project_id != self._project_id:
            self._selected_dataset_id = ""
            self._reset_view_cache()
            return None
        return item

    def _reset_view_cache(self) -> None:
        self._datasets_cache = []
        self._selected_dataset_cache = {}
        self._split_counts_cache = _empty_split_counts()
        self._examples_cache = []

    def _refresh_view_cache(self) -> None:
        if not self._workspace or not self._project_id:
            self._reset_view_cache()
            return
        snapshot = _build_view_snapshot(
            workspace=self._workspace,
            project_id=self._project_id,
            selected_dataset_id=self._selected_dataset_id,
            split_filter=self._split_filter,
            offset=self._offset,
            page_size=self._page_size,
        )
        self._apply_view_snapshot(snapshot)

    def _apply_view_snapshot(self, snapshot: dict[str, object]) -> None:
        datasets = snapshot.get("datasets")
        selected = snapshot.get("selected_dataset")
        counts = snapshot.get("split_counts")
        examples = snapshot.get("examples")
        if not isinstance(datasets, list):
            raise ValueError("Data Studio snapshot datasets must be a list.")
        if not isinstance(selected, dict):
            raise ValueError("Data Studio snapshot selected_dataset must be an object.")
        if not isinstance(counts, dict):
            raise ValueError("Data Studio snapshot split_counts must be an object.")
        if not isinstance(examples, list):
            raise ValueError("Data Studio snapshot examples must be a list.")
        self._datasets_cache = [
            {str(key): value for key, value in item.items()}
            for item in datasets
            if isinstance(item, dict)
        ]
        self._selected_dataset_cache = {
            str(key): value for key, value in selected.items()
        }
        clean_counts = _empty_split_counts()
        for key, value in counts.items():
            if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool):
                clean_counts[key] = value
        self._split_counts_cache = clean_counts
        self._examples_cache = [
            {str(key): value for key, value in item.items()}
            for item in examples
            if isinstance(item, dict)
        ]

    def _filtered_total(self) -> int:
        return int(self._split_counts_cache.get(self._split_filter, 0))

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
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            operation=operation,
            dataset_id=dataset_id,
            adapter_id=self._adapter_id,
            source=source,
            split_filter=self._split_filter,
            page_size=self._page_size,
        )
        task.signals.completed.connect(self._operation_completed)
        task.signals.failed.connect(self._operation_failed)
        self._pool.start(task)

    @Slot(int, str, str, object)
    def _operation_completed(
        self,
        context_token: int,
        operation: str,
        dataset_id: str,
        result: object,
    ) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        details = result if isinstance(result, dict) else {}
        if dataset_id == self._selected_dataset_id:
            self._offset = 0
            snapshot = details.get("view_snapshot")
            if isinstance(snapshot, dict):
                self._apply_view_snapshot(snapshot)
            else:
                self._refresh_view_cache()
        self.changed.emit()
        if operation == "import":
            imported = int(details.get("imported", 0))
            rejected = int(details.get("rejected", 0))
            self.operationCompleted.emit(
                f"Import complete: {imported} imported, {rejected} rejected"
            )
        else:
            self.operationCompleted.emit("Dataset frozen with immutable split artifacts")

    @Slot(int, str, str, str)
    def _operation_failed(
        self,
        context_token: int,
        operation: str,
        dataset_id: str,
        error: str,
    ) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        if dataset_id == self._selected_dataset_id:
            self._offset = 0
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
    split_filter: str = "ALL",
    page_size: int = 100,
) -> dict[str, object]:
    workspace = Workspace.open(workspace_root)
    service = DatasetService(workspace)
    dataset = service.get(dataset_id)
    if operation == "import":
        if source is None:
            raise ValueError("Import source is required.")
        validator = dataset_validator_for(adapter_id)
        result = service.import_jsonl(dataset_id, source, validator=validator).to_dict()
        result["view_snapshot"] = _build_view_snapshot(
            workspace=workspace,
            project_id=dataset.project_id,
            selected_dataset_id=dataset_id,
            split_filter=split_filter,
            offset=0,
            page_size=page_size,
        )
        return result
    if operation == "freeze":
        frozen = service.freeze(dataset_id)
        return {
            "dataset_id": frozen.id,
            "state": frozen.state.value,
            "manifest_sha256": frozen.manifest_artifact_digest or "",
            "leakage_report_sha256": frozen.leakage_report_artifact_digest or "",
            "view_snapshot": _build_view_snapshot(
                workspace=workspace,
                project_id=frozen.project_id,
                selected_dataset_id=frozen.id,
                split_filter=split_filter,
                offset=0,
                page_size=page_size,
            ),
        }
    raise ValueError(f"Unsupported dataset operation: {operation}")


def _empty_split_counts() -> dict[str, int]:
    return {split.value: 0 for split in DatasetSplit} | {"ALL": 0}


def _build_view_snapshot(
    *,
    workspace: Workspace,
    project_id: str,
    selected_dataset_id: str,
    split_filter: str,
    offset: int,
    page_size: int,
) -> dict[str, object]:
    service = DatasetService(workspace)
    datasets = [_dataset_dict(item) for item in service.list_for_project(project_id)]
    if not selected_dataset_id:
        return {
            "datasets": datasets,
            "selected_dataset": {},
            "split_counts": _empty_split_counts(),
            "examples": [],
        }

    selected = service.get(selected_dataset_id)
    if selected.project_id != project_id:
        raise ValueError("Selected dataset does not belong to this project.")

    with workspace.database.connection() as conn:
        rows = conn.execute(
            "SELECT split,COUNT(*) AS count FROM dataset_examples "
            "WHERE dataset_id=? GROUP BY split",
            (selected_dataset_id,),
        ).fetchall()
    counts = _empty_split_counts()
    for row in rows:
        counts[str(row["split"])] = int(row["count"])
    counts["ALL"] = sum(counts[split.value] for split in DatasetSplit)

    split = None if split_filter == "ALL" else DatasetSplit(split_filter)
    examples = [
        _example_dict(item)
        for item in service.page_examples(
            selected_dataset_id,
            split=split,
            offset=offset,
            limit=page_size,
        )
    ]
    return {
        "datasets": datasets,
        "selected_dataset": _dataset_dict(selected),
        "split_counts": counts,
        "examples": examples,
    }


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
