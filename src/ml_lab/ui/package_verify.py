from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QCoreApplication,
    QObject,
    QRunnable,
    QThreadPool,
    QUrl,
    Signal,
    Slot,
)

from ml_lab.bundles.service import BundleRecord, BundleService, VerificationReceipt
from ml_lab.core.models import ModelStage, RegisteredModel
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace


class _PackageSignals(QObject):
    completed = Signal(int, str, str, str)
    failed = Signal(int, str, str)


class _PackageOperation(QRunnable):
    def __init__(
        self,
        *,
        context_token: int,
        workspace_root: Path,
        operation: str,
        item_id: str,
        target: Path | None = None,
    ) -> None:
        super().__init__()
        self.context_token = context_token
        self.workspace_root = workspace_root
        self.operation = operation
        self.item_id = item_id
        self.target = target
        self.signals = _PackageSignals()

    @Slot()
    def run(self) -> None:
        try:
            workspace = Workspace.open(self.workspace_root)
            service = BundleService(workspace)
            if self.operation == "build":
                bundle = service.build_release_candidate(self.item_id)
                primary_id = bundle.id
                detail = bundle.bundle_artifact_digest
            elif self.operation == "verify":
                receipt = service.verify_fresh(self.item_id)
                primary_id = receipt.id
                detail = receipt.status
            elif self.operation == "export":
                if self.target is None:
                    raise ValueError("Export target is required.")
                exported = service.export_bundle(self.item_id, self.target)
                primary_id = self.item_id
                detail = str(exported)
            else:
                raise ValueError(f"Unknown package operation: {self.operation}")
        except Exception as exc:
            self.signals.failed.emit(
                self.context_token,
                self.operation,
                f"{type(exc).__name__}: {exc}",
            )
            return
        self.signals.completed.emit(
            self.context_token,
            self.operation,
            primary_id,
            detail,
        )


class PackageVerifyController(QObject):
    changed = Signal()
    operationCompleted = Signal(str)
    operationFailed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workspace: Workspace | None = None
        self._project_id = ""
        self._selected_bundle_id = ""
        self._selected_receipt_id = ""
        self._busy = False
        self._busy_message = ""
        self._context_token = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.shutdown)

    def bind_project(self, workspace: Workspace, project_id: str, adapter_id: str) -> None:
        del adapter_id
        self._pool.clear()
        self._context_token += 1
        self._workspace = workspace
        self._project_id = project_id
        self._selected_bundle_id = ""
        self._selected_receipt_id = ""
        self._busy = False
        self._busy_message = ""
        self.changed.emit()

    def clear_project(self) -> None:
        self._pool.clear()
        self._context_token += 1
        self._workspace = None
        self._project_id = ""
        self._selected_bundle_id = ""
        self._selected_receipt_id = ""
        self._busy = False
        self._busy_message = ""
        self.changed.emit()

    @Slot()
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

    @Property(list, notify=changed)
    def releaseCandidateRows(self) -> list[dict[str, object]]:
        if not self._workspace or not self._project_id:
            return []
        models = ModelRegistryService(self._workspace).list_for_project(
            self._project_id,
            limit=1000,
        )
        return [
            _model_row(model)
            for model in models
            if model.stage is ModelStage.RELEASE_CANDIDATE
        ]

    @Property(list, notify=changed)
    def bundleRows(self) -> list[dict[str, object]]:
        return [_bundle_row(item) for item in self._bundles()]

    @Property(dict, notify=changed)
    def selectedBundle(self) -> dict[str, object]:
        item = self._selected_bundle()
        return _bundle_row(item) if item is not None else {}

    @Property(list, notify=changed)
    def receiptRows(self) -> list[dict[str, object]]:
        if not self._workspace or not self._selected_bundle_id:
            return []
        service = BundleService(self._workspace)
        return [
            _receipt_row(item)
            for item in service.verification_receipts(self._selected_bundle_id)
        ]

    @Property(str, notify=changed)
    def selectedReceiptPayload(self) -> str:
        if not self._workspace or not self._selected_receipt_id:
            return ""
        payload = BundleService(self._workspace).receipt_payload(
            self._selected_receipt_id
        )
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    @Slot(str)
    def selectBundle(self, bundle_id: str) -> None:
        if any(item.id == bundle_id for item in self._bundles()):
            self._selected_bundle_id = bundle_id
            self._selected_receipt_id = ""
            self.changed.emit()

    @Slot(str)
    def selectReceipt(self, receipt_id: str) -> None:
        if not self._workspace or not self._selected_bundle_id:
            return
        receipts = BundleService(self._workspace).verification_receipts(
            self._selected_bundle_id
        )
        if any(item.id == receipt_id for item in receipts):
            self._selected_receipt_id = receipt_id
            self.changed.emit()

    @Slot(str)
    def buildBundle(self, model_id: str) -> None:
        if not self._workspace or not self._project_id:
            return
        try:
            model = ModelRegistryService(self._workspace).get(model_id)
            if model.project_id != self._project_id:
                raise ValueError("Model does not belong to the selected project.")
            if model.stage is not ModelStage.RELEASE_CANDIDATE:
                raise RuntimeError("Only RELEASE_CANDIDATE models can be packaged.")
        except Exception as exc:
            self.operationFailed.emit("Package error", str(exc))
            return
        self._start_operation(
            "build",
            model.id,
            f"Building release-candidate bundle for {model.id[:8]}…",
        )

    @Slot()
    def verifySelected(self) -> None:
        bundle = self._selected_bundle()
        if bundle is None:
            self.operationFailed.emit("Verification error", "Select a bundle first.")
            return
        self._start_operation(
            "verify",
            bundle.id,
            f"Verifying bundle {bundle.id[:8]} in a fresh process…",
        )

    @Slot(str)
    def exportSelected(self, target: str) -> None:
        bundle = self._selected_bundle()
        if bundle is None:
            self.operationFailed.emit("Export error", "Select a bundle first.")
            return
        path = _url_or_path(target)
        self._start_operation(
            "export",
            bundle.id,
            f"Exporting bundle {bundle.id[:8]}…",
            target=path,
        )

    def _start_operation(
        self,
        operation: str,
        item_id: str,
        message: str,
        *,
        target: Path | None = None,
    ) -> None:
        if not self._workspace or not self._project_id:
            self.operationFailed.emit("Package error", "Open a project first.")
            return
        if self._busy:
            self.operationFailed.emit(
                "Package busy",
                "Another package or verification operation is already running.",
            )
            return
        self._busy = True
        self._busy_message = message
        self.changed.emit()
        worker = _PackageOperation(
            context_token=self._context_token,
            workspace_root=self._workspace.root,
            operation=operation,
            item_id=item_id,
            target=target,
        )
        worker.signals.completed.connect(self._operation_completed)
        worker.signals.failed.connect(self._operation_failed)
        self._pool.start(worker)

    @Slot(int, str, str, str)
    def _operation_completed(
        self,
        context_token: int,
        operation: str,
        primary_id: str,
        detail: str,
    ) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        if operation == "build":
            self._selected_bundle_id = primary_id
            self._selected_receipt_id = ""
            message = f"Release-candidate bundle built: {detail[:12]}…"
        elif operation == "verify":
            self._selected_receipt_id = primary_id
            message = f"Fresh verification completed: {detail}"
        else:
            message = f"Bundle exported to {detail}"
        self.changed.emit()
        self.operationCompleted.emit(message)

    @Slot(int, str, str)
    def _operation_failed(
        self,
        context_token: int,
        operation: str,
        error: str,
    ) -> None:
        if context_token != self._context_token:
            return
        self._busy = False
        self._busy_message = ""
        self.changed.emit()
        title = {
            "build": "Package error",
            "verify": "Verification error",
            "export": "Export error",
        }.get(operation, "Package error")
        self.operationFailed.emit(title, error)

    def _bundles(self) -> list[BundleRecord]:
        if not self._workspace or not self._project_id:
            return []
        return BundleService(self._workspace).list_for_project(
            self._project_id,
            limit=500,
        )

    def _selected_bundle(self) -> BundleRecord | None:
        if not self._selected_bundle_id:
            return None
        return next(
            (item for item in self._bundles() if item.id == self._selected_bundle_id),
            None,
        )


def _model_row(item: RegisteredModel) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "experimentShortId": item.experiment_id[:8],
        "modelSha256": item.model_artifact_digest,
        "stage": item.stage.value,
        "updatedAt": item.updated_at,
    }


def _bundle_row(item: BundleRecord) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "modelId": item.model_id,
        "modelShortId": item.model_id[:8],
        "bundleSha256": item.bundle_artifact_digest,
        "manifestSha256": item.manifest_artifact_digest,
        "createdAt": item.created_at,
    }


def _receipt_row(item: VerificationReceipt) -> dict[str, object]:
    return {
        "id": item.id,
        "shortId": item.id[:8],
        "status": item.status,
        "receiptSha256": item.receipt_artifact_digest,
        "createdAt": item.created_at,
    }


def _url_or_path(value: str) -> Path:
    if value.startswith("file:"):
        return Path(QUrl(value).toLocalFile())
    return Path(value)
