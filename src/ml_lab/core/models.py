from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class ProjectState(StrEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


TERMINAL_JOB_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED}
)


class DatasetSplit(StrEnum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    TEST = "TEST"
    REDTEAM = "REDTEAM"


PROTECTED_SPLITS = frozenset({DatasetSplit.TEST, DatasetSplit.REDTEAM})
TRAINER_VISIBLE_SPLITS = frozenset({DatasetSplit.TRAIN, DatasetSplit.DEV})


class DatasetState(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
    INVALID = "INVALID"


class ExperimentStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class MetricDirection(StrEnum):
    HIGHER = "HIGHER"
    LOWER = "LOWER"
    ZERO = "ZERO"


class FailureSeverity(StrEnum):
    VETO = "VETO"
    NON_VETO = "NON_VETO"


class FailureStatus(StrEnum):
    OPEN = "OPEN"
    REGRESSION = "REGRESSION"
    FIXED = "FIXED"
    ACCEPTED = "ACCEPTED"


class ModelStage(StrEnum):
    EXPERIMENT = "EXPERIMENT"
    SHADOW = "SHADOW"
    ADVISORY = "ADVISORY"
    RELEASE_CANDIDATE = "RELEASE_CANDIDATE"
    INTEGRATION_APPROVED = "INTEGRATION_APPROVED"


@dataclass(frozen=True, slots=True)
class Project:
    id: str
    name: str
    adapter_id: str
    created_at: str
    updated_at: str
    state: ProjectState
    description: str = ""


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    task_type: str
    status: JobStatus
    created_at: str
    updated_at: str
    progress: float
    message: str
    staging_dir: Path
    pid: int | None = None
    correlation_id: str | None = None
    exit_code: int | None = None
    error: str | None = None
    result_artifact_digest: str | None = None
    event_artifact_digest: str | None = None


@dataclass(frozen=True, slots=True)
class ContractSnapshot:
    id: str
    project_id: str
    adapter_id: str
    adapter_version: str
    repo_path: str
    repo_identity: str
    commit_sha: str
    contract_version: str
    compatibility_signature: str
    manifest_artifact_digest: str
    created_at: str


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    id: str
    project_id: str
    name: str
    state: DatasetState
    created_at: str
    example_count: int = 0
    contract_snapshot_id: str | None = None
    manifest_artifact_digest: str | None = None
    leakage_report_artifact_digest: str | None = None
    frozen_at: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetExample:
    dataset_id: str
    example_id: str
    split: DatasetSplit
    source_id: str
    lineage_group: str
    fingerprint: str
    normalized_fingerprint: str
    near_signature: str
    payload_json: str
    label_json: str
    tags_json: str
    created_at: str


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    dataset_id: str
    split: DatasetSplit
    artifact_digest: str
    example_count: int
    partition_sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class MetricValue:
    metric_id: str
    value: float
    direction: MetricDirection
    veto: bool


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    id: str
    project_id: str
    dataset_id: str
    trainer_id: str
    runtime_pack_id: str
    status: ExperimentStatus
    config_json: str
    seed: int
    environment_json: str
    created_at: str
    contract_snapshot_id: str | None = None
    model_artifact_digest: str | None = None
    metrics_artifact_digest: str | None = None
    manifest_artifact_digest: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class FailureRecord:
    id: str
    project_id: str
    kind: str
    severity: FailureSeverity
    status: FailureStatus
    expected_json: str
    observed_json: str
    created_at: str
    experiment_id: str | None = None
    dataset_id: str | None = None
    redteam_run_id: str | None = None
    example_id: str | None = None
    split: DatasetSplit | None = None
    evidence_artifact_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    id: str
    project_id: str
    experiment_id: str
    model_artifact_digest: str
    stage: ModelStage
    compatibility_json: str
    created_at: str
    updated_at: str
    manifest_artifact_digest: str | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
