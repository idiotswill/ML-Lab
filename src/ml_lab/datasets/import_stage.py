from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from ml_lab.adapters.builtin import dataset_validator_for
from ml_lab.datasets.service import (
    ImportErrorRecord,
    prepared_example_record,
    validate_example_row,
)


def prepare_import_stage(
    *,
    source: Path,
    staged_database: Path,
    adapter_id: str,
    max_reported_errors: int = 1000,
) -> dict[str, object]:
    """Parse and validate JSONL into a disposable staging database.

    This function deliberately never opens the ML Lab workspace database. The staged
    database is non-authoritative scratch output consumed by the parent process.
    """
    validator = dataset_validator_for(adapter_id)
    errors: list[ImportErrorRecord] = []
    accepted = 0
    rejected = 0
    source_name = source.name

    staged_database.parent.mkdir(parents=True, exist_ok=True)
    staged_database.unlink(missing_ok=True)
    conn = sqlite3.connect(staged_database)
    try:
        conn.execute(
            "CREATE TABLE prepared_examples("
            "line_number INTEGER NOT NULL,"
            "example_id TEXT PRIMARY KEY,"
            "split TEXT NOT NULL,"
            "source_id TEXT NOT NULL,"
            "lineage_group TEXT NOT NULL,"
            "fingerprint TEXT NOT NULL,"
            "normalized_fingerprint TEXT NOT NULL,"
            "near_signature TEXT NOT NULL,"
            "payload_json TEXT NOT NULL,"
            "label_json TEXT NOT NULL,"
            "tags_json TEXT NOT NULL,"
            "created_at TEXT NOT NULL"
            ")"
        )
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    decoded = json.loads(raw_line)
                    if not isinstance(decoded, dict):
                        raise ValueError("JSONL row must be an object.")
                    row = {str(key): value for key, value in decoded.items()}
                    item = validate_example_row(
                        row,
                        line_number,
                        source_name,
                        validator=validator,
                    )
                    record = prepared_example_record(item)
                    conn.execute(
                        "INSERT INTO prepared_examples("
                        "line_number,example_id,split,source_id,lineage_group,"
                        "fingerprint,normalized_fingerprint,near_signature,"
                        "payload_json,label_json,tags_json,created_at"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            line_number,
                            record["example_id"],
                            record["split"],
                            record["source_id"],
                            record["lineage_group"],
                            record["fingerprint"],
                            record["normalized_fingerprint"],
                            record["near_signature"],
                            record["payload_json"],
                            record["label_json"],
                            record["tags_json"],
                            record["created_at"],
                        ),
                    )
                    accepted += 1
                except (
                    json.JSONDecodeError,
                    ValueError,
                    TypeError,
                    sqlite3.IntegrityError,
                ) as exc:
                    rejected += 1
                    if len(errors) < max_reported_errors:
                        errors.append(
                            ImportErrorRecord(
                                line_number=line_number,
                                code=_stage_error_code(exc),
                                message=str(exc),
                            )
                        )
        conn.commit()
    finally:
        conn.close()

    return {
        "format_version": 1,
        "source_name": source_name,
        "accepted": accepted,
        "rejected": rejected,
        "errors_truncated": rejected > len(errors),
        "errors": [error.to_dict() for error in errors],
    }


def run_import_stage_child(spec_path: Path) -> int:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or spec.get("format_version") != 1:
            raise ValueError("Invalid dataset import stage spec.")
        source = Path(_required_str(spec, "source"))
        staged_database = Path(_required_str(spec, "staged_database"))
        summary_path = Path(_required_str(spec, "summary_path"))
        adapter_id = _required_str(spec, "adapter_id")
        summary = prepare_import_stage(
            source=source,
            staged_database=staged_database,
            adapter_id=adapter_id,
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 12


def _required_str(value: dict[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Dataset import stage spec is missing {key}.")
    return raw


def _stage_error_code(exc: Exception) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(exc, sqlite3.IntegrityError):
        return "DUPLICATE_OR_CONSTRAINT"
    return "INVALID_EXAMPLE"
