from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ml_lab.bundles.service import BundleService
from ml_lab.bundles.verify import verify_bundle_file
from ml_lab.core.models import (
    TERMINAL_JOB_STATUSES,
    DatasetSplit,
    ExperimentStatus,
    MetricDirection,
    MetricValue,
    ModelStage,
)
from ml_lab.datasets.service import DatasetService, ValidatedExampleInput
from ml_lab.experiments.service import ExperimentService
from ml_lab.jobs.manager import JobManager
from ml_lab.models.registry import ModelRegistryService
from ml_lab.storage.workspace import Workspace

_MARKER_NAME = ".phase4-clean-machine-smoke.json"


def prepare_clean_machine_smoke(workspace_path: Path) -> dict[str, object]:
    workspace = Workspace.create(workspace_path)
    project = workspace.create_project(
        "Phase 4 clean-machine sample",
        "generic",
        "Installed-build persistence and bundle verification fixture.",
    )

    datasets = DatasetService(workspace)
    dataset = datasets.create(project.id, "clean-machine-frozen")
    texts = {
        DatasetSplit.TRAIN: "carry bronze compass toward western ridge",
        DatasetSplit.DEV: "inspect blue lantern under stone arch",
        DatasetSplit.TEST: "whisper quiet phrase beside amber fountain",
        DatasetSplit.REDTEAM: "do not strike the silver statue",
    }
    for index, (split, text) in enumerate(texts.items()):
        datasets.add_example(
            dataset.id,
            ValidatedExampleInput(
                example_id=f"clean-machine-{index}",
                split=split,
                source_id=f"synthetic:clean-machine:{index}",
                lineage_group=f"clean-machine-lineage-{index}",
                payload={"text": text},
                label={"class": split.value},
                tags=("phase4-smoke",),
            ),
        )
    frozen = datasets.freeze(dataset.id)

    experiments = ExperimentService(workspace)
    experiment = experiments.create(
        project_id=project.id,
        dataset_id=frozen.id,
        trainer_id="phase4-smoke-trainer",
        runtime_pack_id="phase4-smoke-runtime",
        config={"purpose": "clean-machine-smoke"},
        seed=404,
        environment={"kind": "installed-build-smoke"},
    )
    experiments.start(experiment.id)
    model_artifact = workspace.artifacts.commit_bytes(
        b"phase4-clean-machine-model",
        media_type="application/octet-stream",
    )
    completed = experiments.complete(
        experiment.id,
        model_artifact_digest=model_artifact.digest,
        metrics=(
            MetricValue("smoke_precision", 1.0, MetricDirection.HIGHER, False),
            MetricValue("false_commitments", 0.0, MetricDirection.ZERO, True),
        ),
    )
    if completed.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError("Sample experiment did not complete.")

    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    try:
        job = manager.start("core.self_test", {"steps": 2, "delay": 0.01})
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            job = manager.get(job.id)
            if job.status in TERMINAL_JOB_STATUSES:
                break
            time.sleep(0.03)
        if job.status.value != "COMPLETED" or not job.result_artifact_digest:
            raise RuntimeError(
                f"Clean-machine worker did not complete: {job.status.value}: {job.error}"
            )
    finally:
        manager.shutdown()

    registry = ModelRegistryService(workspace)
    model = registry.register_from_experiment(
        experiment.id,
        compatibility={"adapter": "generic", "contract": "phase4-smoke-v1"},
    )
    for stage in (ModelStage.SHADOW, ModelStage.ADVISORY, ModelStage.RELEASE_CANDIDATE):
        model = registry.promote(
            model.id,
            stage,
            evidence={"source": "phase4-clean-machine-smoke"},
        )

    bundles = BundleService(workspace)
    bundle = bundles.build_release_candidate(
        model.id,
        break_it_guide="Phase 4 clean-machine smoke fixture. Do not integrate.",
    )
    receipt = bundles.verify_fresh(bundle.id)
    if receipt.status != "PASS":
        raise RuntimeError("Fresh bundle verification did not pass.")

    exported = bundles.export_bundle(
        bundle.id,
        workspace.root / "exports" / "phase4-clean-machine-sample.zip",
    )
    exported_sha256 = _sha256_file(exported)
    if exported_sha256 != bundle.bundle_artifact_digest:
        raise RuntimeError("Exported bundle digest changed after export.")

    marker: dict[str, object] = {
        "format_version": 1,
        "project_id": project.id,
        "dataset_id": frozen.id,
        "experiment_id": experiment.id,
        "model_id": model.id,
        "bundle_id": bundle.id,
        "bundle_sha256": bundle.bundle_artifact_digest,
        "verification_receipt_id": receipt.id,
        "worker_job_id": job.id,
        "exported_bundle": exported.relative_to(workspace.root).as_posix(),
    }
    marker_path = workspace.root / _MARKER_NAME
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "stage": "prepare",
        **marker,
    }


def reopen_clean_machine_smoke(workspace_path: Path) -> dict[str, object]:
    workspace = Workspace.open(workspace_path)
    marker_path = workspace.root / _MARKER_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError("Clean-machine smoke marker must be an object.")

    project_id = _required_string(marker, "project_id")
    dataset_id = _required_string(marker, "dataset_id")
    experiment_id = _required_string(marker, "experiment_id")
    model_id = _required_string(marker, "model_id")
    bundle_id = _required_string(marker, "bundle_id")
    bundle_sha256 = _required_string(marker, "bundle_sha256")
    receipt_id = _required_string(marker, "verification_receipt_id")
    worker_job_id = _required_string(marker, "worker_job_id")
    exported_relative = _required_string(marker, "exported_bundle")

    projects = {project.id: project for project in workspace.list_projects()}
    if project_id not in projects:
        raise RuntimeError("Persisted sample project is missing after reopen.")

    dataset = DatasetService(workspace).get(dataset_id)
    if dataset.project_id != project_id or dataset.manifest_artifact_digest is None:
        raise RuntimeError("Persisted frozen dataset is invalid after reopen.")

    experiment = ExperimentService(workspace).get(experiment_id)
    if experiment.status is not ExperimentStatus.COMPLETED:
        raise RuntimeError("Persisted experiment is not completed after reopen.")

    model = ModelRegistryService(workspace).get(model_id)
    if model.stage is not ModelStage.RELEASE_CANDIDATE:
        raise RuntimeError("Persisted model stage changed after reopen.")

    bundles = BundleService(workspace)
    bundle = bundles.get(bundle_id)
    if bundle.bundle_artifact_digest != bundle_sha256:
        raise RuntimeError("Persisted bundle digest changed after reopen.")
    receipt = bundles.receipt_payload(receipt_id)
    if receipt.get("status") != "PASS" or receipt.get("fresh_process") is not True:
        raise RuntimeError("Persisted fresh-verification receipt is not PASS.")

    manager = JobManager(workspace.root, workspace.database, workspace.artifacts)
    try:
        job = manager.get(worker_job_id)
        if job.status.value != "COMPLETED" or not job.result_artifact_digest:
            raise RuntimeError("Persisted worker completion is missing after reopen.")
    finally:
        manager.shutdown()

    exported = (workspace.root / exported_relative).resolve()
    if not exported.is_relative_to(workspace.root):
        raise ValueError("Exported bundle marker escaped the workspace.")
    if _sha256_file(exported) != bundle_sha256:
        raise RuntimeError("Exported bundle hash changed after reopen.")
    direct = verify_bundle_file(exported)
    if direct.get("status") != "PASS" or direct.get("integration_gate") != "NO_GO":
        raise RuntimeError("Reopened exported bundle failed verification.")

    return {
        "ok": True,
        "stage": "reopen",
        "project_id": project_id,
        "bundle_sha256": bundle_sha256,
        "fresh_verification": "PASS",
        "integration_gate": "NO_GO",
    }


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Clean-machine smoke marker is missing {key}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
