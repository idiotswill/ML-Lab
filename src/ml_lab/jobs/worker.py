from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from ml_lab.jobs.protocol import JobSpec, event
from ml_lab.trainers.phase_a_sparse import train_phase_a_sparse
from ml_lab.trainers.sparse_nb import (
    evaluate_sparse_nb,
    read_jsonl_records,
    train_sparse_nb,
)

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
        _emit(
            "progress",
            progress=(index + 1) / steps,
            message=f"Self-test step {index + 1}/{steps}",
        )
    return {"ok": True, "steps": steps}


def task_performance_load(spec: JobSpec, staging: Path) -> dict[str, object]:
    duration = max(0.25, min(float(spec.payload.get("duration_seconds", 20.0)), 120.0))
    block = b"ml-lab-performance-load" * 65536
    started = time.perf_counter()
    last_emit = started
    iterations = 0
    digest = b""
    while True:
        if _cancelled(staging):
            raise InterruptedError("Cancellation requested")
        now = time.perf_counter()
        if now - started >= duration:
            break
        digest = hashlib.sha256(block).digest()
        iterations += 1
        if now - last_emit >= 0.25:
            elapsed = now - started
            _emit(
                "progress",
                progress=min(0.99, elapsed / duration),
                message=f"Performance load {elapsed:.1f}s / {duration:.1f}s",
            )
            last_emit = now
    return {
        "ok": True,
        "duration_seconds": duration,
        "iterations": iterations,
        "digest_prefix": digest.hex()[:16],
    }


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


def task_sparse_nb_train(spec: JobSpec, staging: Path) -> dict[str, object]:
    train_paths, dev_paths = _training_paths(spec.payload)
    text_key = str(spec.payload.get("text_key", "text"))
    label_key = str(spec.payload.get("label_key", "class"))
    feature_dim = int(spec.payload.get("feature_dim", 32768))
    alpha = float(spec.payload.get("alpha", 0.5))

    last_reported = 0

    def on_example(count: int) -> None:
        nonlocal last_reported
        if count == 1 or count - last_reported >= 250:
            last_reported = count
            _emit(
                "progress",
                progress=min(0.85, 0.05 + count / 100_000),
                message=f"Trained on {count:,} examples",
            )

    model, train_examples = train_sparse_nb(
        read_jsonl_records(train_paths),
        text_key=text_key,
        label_key=label_key,
        feature_dim=feature_dim,
        alpha=alpha,
        cancelled=lambda: _cancelled(staging),
        on_example=on_example,
    )
    if _cancelled(staging):
        raise InterruptedError("Cancellation requested")
    model_path = staging / "model.json"
    model.save(model_path)

    dev_correct = 0
    dev_total = 0
    if dev_paths:
        _emit("progress", progress=0.9, message="Evaluating DEV partition")
        dev_correct, dev_total = evaluate_sparse_nb(
            model,
            read_jsonl_records(dev_paths),
            cancelled=lambda: _cancelled(staging),
        )
    dev_accuracy = dev_correct / dev_total if dev_total else None
    return {
        "ok": True,
        "trainer_id": "builtin.sparse_nb.v1",
        "experiment_id": spec.payload.get("experiment_id"),
        "train_examples": train_examples,
        "dev_examples": dev_total,
        "dev_accuracy": dev_accuracy,
        "classes": list(model.classes),
        "feature_dim": model.feature_dim,
        "alpha": model.alpha,
        "outputs": [
            {
                "name": "model",
                "path": "model.json",
                "media_type": "application/vnd.ml-lab.sparse-nb+json",
            }
        ],
    }


def task_phase_a_sparse_train(spec: JobSpec, staging: Path) -> dict[str, object]:
    train_paths, _dev_paths = _training_paths(spec.payload)
    feature_dim = int(spec.payload.get("feature_dim", 32768))
    alpha = float(spec.payload.get("alpha", 0.5))
    train_examples = 0
    last_reported = 0

    def records() -> Iterator[Mapping[str, object]]:
        nonlocal train_examples, last_reported
        for record in read_jsonl_records(train_paths):
            if _cancelled(staging):
                raise InterruptedError("Cancellation requested")
            train_examples += 1
            if train_examples == 1 or train_examples - last_reported >= 100:
                last_reported = train_examples
                _emit(
                    "progress",
                    progress=min(0.9, 0.05 + train_examples / 50_000),
                    message=f"Trained bounded Phase A scorer on {train_examples:,} examples",
                )
            yield record

    model = train_phase_a_sparse(
        records(),
        feature_dim=feature_dim,
        alpha=alpha,
    )
    if _cancelled(staging):
        raise InterruptedError("Cancellation requested")
    model_path = staging / "model.json"
    model.save(model_path)
    return {
        "ok": True,
        "trainer_id": "builtin.phase_a_sparse.v1",
        "experiment_id": spec.payload.get("experiment_id"),
        "train_examples": train_examples,
        "feature_dim": feature_dim,
        "alpha": alpha,
        "outputs": [
            {
                "name": "model",
                "path": "model.json",
                "media_type": "application/vnd.ml-lab.phase-a-bounded-sparse+json",
            }
        ],
    }


def _training_paths(payload: dict[str, object]) -> tuple[list[Path], list[Path]]:
    staged = payload.get("staged_inputs")
    if staged is not None:
        if not isinstance(staged, dict):
            raise ValueError("staged_inputs must be an object")
        train_value = staged.get("train.jsonl")
        dev_value = staged.get("dev.jsonl")
        if not isinstance(train_value, str):
            raise ValueError("staged_inputs must contain train.jsonl")
        train_paths = _path_list([train_value], field="staged train")
        dev_paths = (
            _path_list([dev_value], field="staged dev")
            if isinstance(dev_value, str)
            else []
        )
        return train_paths, dev_paths

    train_paths = _path_list(payload.get("train_paths"), field="train_paths")
    dev_paths = _path_list(
        payload.get("dev_paths", []),
        field="dev_paths",
        allow_empty=True,
    )
    return train_paths, dev_paths


def _path_list(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> list[Path]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of paths")
    paths = [Path(item) for item in value]
    if not paths and not allow_empty:
        raise ValueError(f"{field} cannot be empty")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


TASKS: dict[str, Task] = {
    "core.self_test": task_self_test,
    "core.performance_load": task_performance_load,
    "core.hash_file": task_hash_file,
    "trainer.sparse_nb.v1": task_sparse_nb_train,
    "trainer.phase_a_sparse.v1": task_phase_a_sparse_train,
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
        manifest.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _emit(
            "completed",
            progress=1.0,
            message="Completed",
            result_manifest=str(manifest),
        )
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
