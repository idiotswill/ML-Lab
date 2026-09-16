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


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
