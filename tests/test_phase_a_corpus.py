from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab.adapters.phase_a_corpus import (
    DEFAULT_CORPUS_PLAN,
    generate_phase_a_corpus,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_phase_a_corpus_generation_hits_420_case_coverage_target(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    receipt = generate_phase_a_corpus(output_dir=output)

    assert receipt["schema"] == "ml-lab-phase-a-corpus-generation-receipt/1"
    assert receipt["ok"] is True
    assert receipt["case_count"] == 420
    assert receipt["split_counts"] == {
        "TRAIN": 240,
        "DEV": 60,
        "TEST": 60,
        "REDTEAM": 60,
    }
    assert receipt["production_data"] is False
    assert receipt["transcript_derived"] is False
    assert receipt["protected_splits_trainer_visible"] is False
    assert receipt["training_started"] is False
    assert receipt["integration_gate"] == "NO_GO"
    assert receipt["language_leakage"]["blocking_count"] == 0

    expected_family_counts = {
        "TRAIN": 24,
        "DEV": 6,
        "TEST": 6,
        "REDTEAM": 3,
    }
    for split, expected in expected_family_counts.items():
        counts = receipt["family_resolve_counts"][split]
        assert set(counts) == {
            "HARM_TARGET",
            "MOVE_TRAVEL",
            "SEARCH_INSPECT",
            "INTERACT_OBJECT",
            "SPEECH_ONLY",
            "CAST_SPELL",
            "USE_ITEM",
        }
        assert set(counts.values()) == {expected}


def test_phase_a_corpus_outputs_sealed_train_dev_and_protected_seeds(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    generate_phase_a_corpus(output_dir=output)

    train_dev = _load(output / "phase-a-expanded-train-dev-seed-v1.json")
    protected = _load(
        output / "phase-a-expanded-protected-residual-seed-v1.json"
    )
    zero_model = _load(output / "phase-a-expanded-zero-model-seed-v1.json")
    protected_language = _load(
        output / "phase-a-expanded-protected-language-v1.json"
    )

    assert len(train_dev["cases"]) == 300
    assert sum(row["split"] == "TRAIN" for row in train_dev["cases"]) == 240
    assert sum(row["split"] == "DEV" for row in train_dev["cases"]) == 60
    assert sum(row["training_allowed"] is True for row in train_dev["cases"]) == 240
    assert sum(row["training_allowed"] is False for row in train_dev["cases"]) == 60

    assert len(protected["cases"]) == 96
    assert {row["split"] for row in protected["cases"]} == {"TEST", "REDTEAM"}
    assert all(row["training_allowed"] is False for row in protected["cases"])

    assert len(zero_model["cases"]) == 24
    assert {row["split"] for row in zero_model["cases"]} == {"REDTEAM"}
    assert all(row["training_allowed"] is False for row in zero_model["cases"])
    assert {
        row["case_kind"].removeprefix("ZERO_MODEL_")
        for row in zero_model["cases"]
    } == {
        "NEGATION",
        "QUESTION",
        "CANCELLATION",
        "HYPOTHETICAL",
        "REPORTED",
        "QUOTED",
        "CORRECTION",
        "FRAMING",
    }

    # 60 TEST + 36 residual REDTEAM + 24 zero-model REDTEAM + 7 bootstrap held-out.
    assert len(protected_language["cases"]) == 127
    assert {row["split"] for row in protected_language["cases"]} == {
        "TEST",
        "REDTEAM",
    }
    assert all(
        row["training_allowed"] is False
        for row in protected_language["cases"]
    )


def test_phase_a_corpus_covers_interaction_operations_and_spell_shapes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    generate_phase_a_corpus(output_dir=output)
    train_dev = _load(output / "phase-a-expanded-train-dev-seed-v1.json")

    train_resolves = [
        row
        for row in train_dev["cases"]
        if row["split"] == "TRAIN" and row["expected"]["decision"] == "RESOLVE"
    ]
    interactions = [
        row for row in train_resolves
        if row["expected"]["action_family"] == "INTERACT_OBJECT"
    ]
    operations = {
        slot["value"]
        for row in interactions
        for slot in row["expected"]["slots"]
        if slot["name"] == "OPERATION"
    }
    assert operations == {"OPEN", "CLOSE", "EQUIP", "STOW", "TRANSFER"}
    assert any(
        any(slot["name"] == "RECIPIENT" for slot in row["expected"]["slots"])
        for row in interactions
    )

    spells = [
        row for row in train_resolves
        if row["expected"]["action_family"] == "CAST_SPELL"
    ]
    assert any(
        any(slot["name"] == "TARGET_COMBATANT" for slot in row["expected"]["slots"])
        for row in spells
    )
    assert any(
        not any(
            slot["name"] == "TARGET_COMBATANT"
            for slot in row["expected"]["slots"]
        )
        for row in spells
    )

    searches = [
        row for row in train_resolves
        if row["expected"]["action_family"] == "SEARCH_INSPECT"
    ]
    assert any(
        row["expected"]["slots"] == [
            {"name": "SUBJECT", "value": "scene:current"}
        ]
        for row in searches
    )


def test_phase_a_corpus_scenes_include_privacy_and_admission_facts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "corpus"
    generate_phase_a_corpus(output_dir=output)
    train_dev = _load(output / "phase-a-expanded-train-dev-seed-v1.json")

    assert train_dev["audience"] == "PC_PRIVATE"
    for scene in train_dev["scenes"].values():
        facts = scene["facts"]
        assert any(
            fact["visibility"] == "GM_ONLY"
            and "hidden-observer" in fact["key"]
            for fact in facts
        )
        assert any(
            fact["key"].startswith("spell.")
            and fact["key"].endswith(".fixture")
            and fact["visibility"] == "PC_PRIVATE"
            for fact in facts
        )
        assert any(
            fact["key"].startswith("inventory.item.")
            and fact["key"].endswith(".catalog_item_id")
            and fact["value"] == "potion_healing"
            for fact in facts
        )
        assert any(
            fact["key"].startswith("campaign.entity.pc.")
            for fact in facts
        )


def test_phase_a_corpus_generation_is_byte_stable(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_receipt = generate_phase_a_corpus(output_dir=first)
    second_receipt = generate_phase_a_corpus(output_dir=second)

    assert first_receipt == second_receipt
    for name in (
        "phase-a-expanded-train-dev-seed-v1.json",
        "phase-a-expanded-protected-residual-seed-v1.json",
        "phase-a-expanded-zero-model-seed-v1.json",
        "phase-a-expanded-protected-language-v1.json",
        "phase-a-expanded-language-leakage-v1.json",
        "corpus-receipt.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_phase_a_corpus_rejects_deterministic_term_in_residual_lexicon(
    tmp_path: Path,
) -> None:
    plan = json.loads(DEFAULT_CORPUS_PLAN.read_text(encoding="utf-8"))
    plan["resolve_lexicons"]["TRAIN"]["SEARCH_INSPECT"][0] = "inspect"
    plan_path = tmp_path / "bad-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="contains deterministic term"):
        generate_phase_a_corpus(
            output_dir=tmp_path / "out",
            plan_path=plan_path,
        )
