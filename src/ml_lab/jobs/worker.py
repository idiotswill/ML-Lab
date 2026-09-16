from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

from ml_lab.jobs.protocol import JobSpec, event

Task = Callable[[JobSpec, Path], dict[str, object]]


def _cancelled(staging: Path) -> bool:
    return (staging / "cancel.request").exists()


def _emit(kind: str, **fields: object) -> None:
    print(event(kind, **fields), flush=True)


def task_self_test(spec: JobSpec, staging: Path) -> dict[str, object]:
    steps = max(1, int(spec.payload.get("steps", 5)))
    delay = max(0.0, min(float(spec.payload.get("delay", 0.03)), 2.0))
    for index in range(steps):
        if _cancelled(staging):
            raise InterruptedError("Cancellation requested")
        time.sleep(delay)
        _emit("progress", progress=(index + 1) / steps, message=f"Self-test step {index + 1}/{steps}")
    return {"ok": True, "steps": steps}


def task_hash_file(spec: JobSpec, staging: Path) -> dict[str, object]:
    source = Path(str(spec.payload["path"]))
    total = source.stat().st_size
    done = 0
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if _cancelled(staging):
                raise InterruptedError("Cancellation requested")
            hasher.update(chunk)
            done += len(chunk)
            _emit(
                "progress",
                progress=(done / total if total else 1.0),
                message=f"Hashed {done:,} / {total:,} bytes",
            )
    return {"sha256": hasher.hexdigest(), "size_bytes": total}


TASKS: dict[str, Task] = {
    "core.self_test": task_self_test,
    "core.hash_file": task_hash_file,
}


def run_worker(spec_path: Path) -> int:
    spec = JobSpec.read(spec_path)
    staging = Path(spec.staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    _emit("started", job_id=spec.job_id)
    try:
        task = TASKS[spec.task_type]
    except KeyError:
        _emit("failed", error=f"Unknown task type: {spec.task_type}")
        return 2
    try:
        result = task(spec, staging)
        manifest = staging / "result_manifest.json"
        manifest.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        _emit("completed", progress=1.0, message="Completed", result_manifest=str(manifest))
        return 0
    except InterruptedError as exc:
        _emit("cancelled", message=str(exc))
        return 130
    except Exception as exc:
        _emit("failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ml_lab.jobs.worker <job_spec.json>")
    raise SystemExit(run_worker(Path(sys.argv[1])))
