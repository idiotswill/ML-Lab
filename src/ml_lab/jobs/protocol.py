from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JOB_PROTOCOL_VERSION = 1


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: str
    task_type: str
    staging_dir: str
    payload: dict[str, Any] = field(default_factory=dict)
    protocol_version: int = JOB_PROTOCOL_VERSION

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> JobSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") != JOB_PROTOCOL_VERSION:
            raise ValueError("Incompatible job protocol version.")
        return cls(**payload)


def event(kind: str, **fields: Any) -> str:
    payload = {"protocol_version": JOB_PROTOCOL_VERSION, "kind": kind, **fields}
    return json.dumps(payload, sort_keys=True)
