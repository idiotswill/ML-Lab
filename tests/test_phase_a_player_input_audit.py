from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "phase_a" / "player_input_audit_v1.jsonl.gz"

_ALLOWED_CLASSES = {
    "CLEAN_CURRENT_CONTRACT",
    "DETERMINISTIC_PREPROCESSING_REQUIRED",
    "ASK_PLAYER",
    "RULES_META_OUTSIDE_RESIDUAL",
    "MISSING_OR_UNCLEAR_CONTRACT",
}
_FORBIDDEN_RAW_TOKENS = (
    "Milo", "Mylo", "Torrek", "Torek", "Cassian", "Casian", "Sylvara", "Silvara",
    "Silara", "Savara", "Eirenhold", "Pilgrim", "Brannick", "Brenic", "Caldris",
)


def _rows():
    with gzip.open(CORPUS, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_player_input_audit_is_protected_and_provenanced():
    rows = _rows()
    assert rows
    assert len(rows) == 55
    assert len({row["audit_id"] for row in rows}) == len(rows)
    assert all(row["split"] in {"TEST", "REDTEAM"} for row in rows)
    assert all(row["training_allowed"] is False for row in rows)
    assert all(row["raw_source_in_repo"] is False for row in rows)
    assert all(row["audit_class"] in _ALLOWED_CLASSES for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    assert all(len(row["source_text_sha256"]) == 64 for row in rows)
    assert all(row["lineage_group"] for row in rows)
    assert all(row["source_line"] >= 1 for row in rows)


def test_player_input_audit_contains_no_known_raw_campaign_identity_tokens():
    for row in _rows():
        text = row["text"].casefold()
        for token in _FORBIDDEN_RAW_TOKENS:
            assert token.casefold() not in text, (row["audit_id"], token)


def test_exact_duplicate_source_occurrences_are_not_independent_rows():
    rows = _rows()
    by_text_hash = defaultdict(list)
    for row in rows:
        by_text_hash[row["source_text_sha256"]].append(row)
    assert all(len(group) == 1 for group in by_text_hash.values())
    assert any(row["duplicate_source_lines"] for row in rows)


def test_multiple_action_cases_are_never_train_and_expect_player_choice():
    multi = [row for row in _rows() if row["audit_class"] == "ASK_PLAYER"]
    assert multi
    assert all(row["expected_boundary"] == "PRIMARY_ACTION_REQUIRED" for row in multi)
    assert all(row["split"] == "REDTEAM" for row in multi)
