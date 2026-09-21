from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from ml_lab.core.models import DatasetSplit
from ml_lab.datasets.leakage import (
    LeakageExample,
    LeakageReport,
    canonical_json,
    content_fingerprint,
    near_signature,
    normalized_fingerprint,
    scan_leakage,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_PLAN = _REPO_ROOT / "benchmarks" / "phase_a" / "corpus_plan_v1.json"
DEFAULT_BOOTSTRAP_PROTECTED_SEED = (
    _REPO_ROOT / "benchmarks" / "phase_a" / "authoritative_fixture_seed_v1.json"
)

_FAMILIES = (
    "HARM_TARGET",
    "MOVE_TRAVEL",
    "SEARCH_INSPECT",
    "INTERACT_OBJECT",
    "SPEECH_ONLY",
    "CAST_SPELL",
    "USE_ITEM",
)

_DETERMINISTIC_TERMS = {
    "attack", "hit", "stab", "slash", "shoot", "strike", "kill", "clobber",
    "smack", "wallop", "punch", "skewer", "move", "walk", "run", "go",
    "enter", "leave", "head", "journey", "proceed", "approach", "retreat",
    "travel", "search", "inspect", "examine", "investigate", "look for",
    "rummage", "scan", "study", "check", "look over", "pick up", "open",
    "close", "take", "give", "drop", "equip", "stow", "push", "pull",
    "break", "unlatch", "grab", "hand", "lift", "turn", "say", "tell",
    "ask", "request", "bargain", "threaten", "chat", "speak", "negotiate",
    "warn", "greet", "cast", "invoke", "drink", "use", "consume", "activate",
    "quaff", "swallow", "apply", "sip",
}

_AMBIGUOUS_PREFIXES = (
    "I {action}",
    "I carefully {action}",
    "I quietly {action}",
    "I slowly {action}",
    "I deliberately {action}",
    "We {action}",
    "We carefully {action}",
    "I will {action}",
    "We will {action}",
    "I try to {action}",
)


def generate_phase_a_corpus(
    *,
    output_dir: Path,
    plan_path: Path = DEFAULT_CORPUS_PLAN,
) -> dict[str, object]:
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict):
        raise ValueError("Phase A corpus plan must be a JSON object")
    _validate_plan(plan)
    bootstrap_bytes = DEFAULT_BOOTSTRAP_PROTECTED_SEED.read_bytes()
    bootstrap = json.loads(bootstrap_bytes)
    if not isinstance(bootstrap, dict):
        raise ValueError("Bootstrap protected seed must be a JSON object")
    bootstrap_cases = bootstrap.get("cases")
    if not isinstance(bootstrap_cases, list) or not all(
        isinstance(row, dict) for row in bootstrap_cases
    ):
        raise ValueError("Bootstrap protected seed cases must be objects")

    scenes: dict[str, dict[str, object]] = {}
    residual_cases: dict[str, list[dict[str, object]]] = {
        split.value: [] for split in DatasetSplit
    }
    zero_model_cases: list[dict[str, object]] = []

    for split in DatasetSplit:
        split_name = split.value
        themes = _string_list(_mapping(plan, "scene_themes").get(split_name))
        for index, theme in enumerate(themes):
            scene_id = f"{split_name.lower()}-{index + 1:02d}"
            scenes[scene_id] = _scene(
                split=split,
                scene_id=scene_id,
                theme=theme,
                ordinal=index,
            )

        target = _mapping(_mapping(plan, "target_counts"), split_name)
        resolve_per_family = _int_field(target, "resolve_per_family")
        ambiguous_count = _int_field(target, "ambiguous")
        compound_count = _int_field(target, "compound")
        zero_model_count = _int_field(target, "zero_model")

        for family in _FAMILIES:
            residual_cases[split_name].extend(
                _resolve_cases(
                    plan=plan,
                    split=split,
                    family=family,
                    count=resolve_per_family,
                    scenes=scenes,
                )
            )
        residual_cases[split_name].extend(
            _ambiguous_cases(
                plan=plan,
                split=split,
                count=ambiguous_count,
                scenes=scenes,
            )
        )
        residual_cases[split_name].extend(
            _compound_cases(
                plan=plan,
                split=split,
                count=compound_count,
                scenes=scenes,
            )
        )
        if zero_model_count:
            zero_model_cases.extend(
                _zero_model_cases(
                    split=split,
                    count=zero_model_count,
                    scenes=scenes,
                    categories=_string_list(plan.get("zero_model_categories")),
                )
            )

    _assert_unique_case_ids(residual_cases, zero_model_cases)
    _assert_target_counts(plan, residual_cases, zero_model_cases)
    _assert_family_coverage(plan, residual_cases)
    _assert_resolve_lexicons_are_residual_only(plan)

    bootstrap_language_rows = [
        {
            "case_id": f"bootstrap:{row['case_id']}",
            "split": row["split"],
            "lineage_group": f"bootstrap:{row['lineage_group']}",
            "declaration": row["declaration"],
            "tags": row.get("tags", []),
            "training_allowed": False,
        }
        for row in bootstrap_cases
    ]
    all_language_rows = [
        *(
            row
            for split_rows in residual_cases.values()
            for row in split_rows
        ),
        *zero_model_cases,
        *bootstrap_language_rows,
    ]
    language_report = _language_leakage(all_language_rows)
    if language_report.has_blockers:
        raise ValueError(
            "Generated Phase A corpus has cross-split language leakage blockers"
        )

    train_dev_scenes = _scenes_for_cases(
        scenes,
        [
            *residual_cases[DatasetSplit.TRAIN.value],
            *residual_cases[DatasetSplit.DEV.value],
        ],
    )
    train_dev_seed = {
        "schema": "ml-lab-phase-a-synthetic-seed/1",
        "status": "SYNTHETIC_NON_CANON",
        "target_frankenhomie_commit": plan["target_frankenhomie_commit"],
        "target_contract": plan["target_contract"],
        "production_data": False,
        "transcript_derived": False,
        "actor_id": plan["actor_id"],
        "audience": plan["audience"],
        "source_id_prefix": "synthetic-corpus-v1",
        "scenes": train_dev_scenes,
        "cases": [
            *residual_cases[DatasetSplit.TRAIN.value],
            *residual_cases[DatasetSplit.DEV.value],
        ],
    }
    protected_residual_seed = {
        "schema": "ml-lab-phase-a-protected-residual-seed/1",
        "status": "PROTECTED_EVALUATION_ONLY",
        "target_frankenhomie_commit": plan["target_frankenhomie_commit"],
        "target_contract": plan["target_contract"],
        "production_data": False,
        "transcript_derived": False,
        "actor_id": plan["actor_id"],
        "audience": plan["audience"],
        "scenes": _scenes_for_cases(
            scenes,
            [
                *residual_cases[DatasetSplit.TEST.value],
                *residual_cases[DatasetSplit.REDTEAM.value],
            ],
        ),
        "cases": [
            *residual_cases[DatasetSplit.TEST.value],
            *residual_cases[DatasetSplit.REDTEAM.value],
        ],
    }
    zero_model_seed = {
        "schema": "ml-lab-phase-a-zero-model-seed/1",
        "status": "PROTECTED_EVALUATION_ONLY",
        "target_frankenhomie_commit": plan["target_frankenhomie_commit"],
        "target_contract": plan["target_contract"],
        "production_data": False,
        "transcript_derived": False,
        "actor_id": plan["actor_id"],
        "audience": plan["audience"],
        "scenes": _scenes_for_cases(scenes, zero_model_cases),
        "cases": zero_model_cases,
    }
    protected_language = {
        "schema": "ml-lab-phase-a-authoritative-fixture-seed/1",
        "status": "PROTECTED_EVALUATION_ONLY",
        "target_frankenhomie_commit": plan["target_frankenhomie_commit"],
        "target_contract": plan["target_contract"],
        "fixture_kind": "NON_CANON_FIXTURE",
        "production_data": False,
        "actor_id": plan["actor_id"],
        "audience": plan["audience"],
        "snapshot_revision": "phase-a-corpus-language-only-v1",
        "facts": [
            {
                "key": "session.scene",
                "value": "Synthetic language-leakage placeholder only.",
                "source": "fixture:phase-a-corpus-language-only",
                "visibility": "TABLE",
            }
        ],
        "purpose": "LANGUAGE_LEAKAGE_ONLY",
        "cases": [
            _protected_language_case(row)
            for row in [
                *residual_cases[DatasetSplit.TEST.value],
                *residual_cases[DatasetSplit.REDTEAM.value],
                *zero_model_cases,
                *bootstrap_language_rows,
            ]
        ],
    }

    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    files = {
        "train_dev_seed": output_root / "phase-a-expanded-train-dev-seed-v1.json",
        "protected_residual_seed": (
            output_root / "phase-a-expanded-protected-residual-seed-v1.json"
        ),
        "zero_model_seed": output_root / "phase-a-expanded-zero-model-seed-v1.json",
        "protected_language_seed": (
            output_root / "phase-a-expanded-protected-language-v1.json"
        ),
        "language_leakage": output_root / "phase-a-expanded-language-leakage-v1.json",
    }
    payloads = {
        "train_dev_seed": train_dev_seed,
        "protected_residual_seed": protected_residual_seed,
        "zero_model_seed": zero_model_seed,
        "protected_language_seed": protected_language,
        "language_leakage": language_report.to_dict(),
    }
    artifact_rows: dict[str, dict[str, object]] = {}
    for name, path in files.items():
        payload = payloads[name]
        raw = (canonical_json(payload) + "\n").encode("utf-8")
        path.write_bytes(raw)
        artifact_rows[name] = {
            "file": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    split_counts = {
        split.value: len(residual_cases[split.value])
        + sum(
            1
            for row in zero_model_cases
            if row["split"] == split.value
        )
        for split in DatasetSplit
    }
    family_counts = {
        split.value: dict(
            sorted(
                Counter(
                    str(_mapping(row, "expected")["action_family"])
                    for row in residual_cases[split.value]
                    if _mapping(row, "expected")["decision"] == "RESOLVE"
                ).items()
            )
        )
        for split in DatasetSplit
    }
    case_kind_counts = {
        split.value: dict(
            sorted(
                Counter(
                    str(row["case_kind"])
                    for row in [
                        *residual_cases[split.value],
                        *(
                            row
                            for row in zero_model_cases
                            if row["split"] == split.value
                        ),
                    ]
                ).items()
            )
        )
        for split in DatasetSplit
    }

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-corpus-generation-receipt/1",
        "ok": True,
        "target_frankenhomie_commit": plan["target_frankenhomie_commit"],
        "target_contract": plan["target_contract"],
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "bootstrap_protected_seed_sha256": hashlib.sha256(
            bootstrap_bytes
        ).hexdigest(),
        "case_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "family_resolve_counts": family_counts,
        "case_kind_counts": case_kind_counts,
        "language_leakage": language_report.to_dict(),
        "artifacts": artifact_rows,
        "production_data": False,
        "transcript_derived": False,
        "protected_splits_trainer_visible": False,
        "training_started": False,
        "integration_gate": "NO_GO",
    }
    receipt_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    receipt = {**base_receipt, "payload_sha256": receipt_sha}
    receipt_path = output_root / "corpus-receipt.json"
    receipt_bytes = (canonical_json(receipt) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    return receipt


def _resolve_cases(
    *,
    plan: Mapping[str, object],
    split: DatasetSplit,
    family: str,
    count: int,
    scenes: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    split_name = split.value
    lexicons = _mapping(_mapping(plan, "resolve_lexicons"), split_name)
    raw_lexicon = lexicons.get(family)
    if not isinstance(raw_lexicon, list) or not raw_lexicon:
        raise ValueError(f"Missing {split_name} lexicon for {family}")
    styles = _string_list(_mapping(plan, "surface_styles").get(split_name))
    scene_ids = _scene_ids(split, scenes)
    cases: list[dict[str, object]] = []

    for index in range(count):
        cycle = index // len(raw_lexicon)
        scene_id = scene_ids[(index + cycle) % len(scene_ids)]
        scene = scenes[scene_id]
        details = _mapping(scene, "details")
        style = styles[cycle % len(styles)]
        candidate_index = index % 2
        phrase: str
        slots: list[dict[str, str]]

        if family == "HARM_TARGET":
            verb = _string_item(raw_lexicon, index)
            key = _detail_key(details, "combatants", candidate_index)
            phrase = f"{verb} {_detail_name(details, 'combatants', candidate_index)}"
            slots = [{"name": "TARGET_COMBATANT", "value": key}]
        elif family == "MOVE_TRAVEL":
            verb = _string_item(raw_lexicon, index)
            key = _detail_key(details, "locations", candidate_index)
            location = _detail_name(details, "locations", candidate_index)
            phrase = (
                f"{verb} {location}"
                if verb.endswith(("toward", "for"))
                else f"{verb} toward {location}"
            )
            slots = [{"name": "DESTINATION", "value": key}]
        elif family == "SEARCH_INSPECT":
            verb = _string_item(raw_lexicon, index)
            if index % 5 == 4:
                phrase = f"{verb} the surroundings"
                slots = [{"name": "SUBJECT", "value": "scene:current"}]
            else:
                key = _detail_key(details, "objects", candidate_index)
                phrase = f"{verb} {_detail_name(details, 'objects', candidate_index)}"
                slots = [{"name": "SUBJECT", "value": key}]
        elif family == "SPEECH_ONLY":
            verb = _string_item(raw_lexicon, index)
            key = _detail_key(details, "npcs", candidate_index)
            phrase = f"{verb} {_detail_name(details, 'npcs', candidate_index)}"
            slots = [{"name": "ADDRESSEE", "value": key}]
        elif family == "CAST_SPELL":
            verb = _string_item(raw_lexicon, index)
            if index % 2 == 0:
                spell_key = _detail_key(details, "target_spells", 0)
                target_key = _detail_key(details, "combatants", candidate_index)
                phrase = (
                    f"{verb} {_detail_name(details, 'target_spells', 0)} at "
                    f"{_detail_name(details, 'combatants', candidate_index)}"
                )
                slots = [
                    {"name": "SPELL", "value": spell_key},
                    {"name": "TARGET_COMBATANT", "value": target_key},
                ]
            else:
                spell_key = _detail_key(details, "self_spells", 0)
                phrase = f"{verb} {_detail_name(details, 'self_spells', 0)}"
                slots = [{"name": "SPELL", "value": spell_key}]
        elif family == "USE_ITEM":
            verb = _string_item(raw_lexicon, index)
            key = _detail_key(details, "items", candidate_index)
            phrase = f"{verb} {_detail_name(details, 'items', candidate_index)}"
            slots = [
                {"name": "ITEM", "value": key},
                {"name": "OPERATION", "value": "SELF_CONSUME"},
            ]
        elif family == "INTERACT_OBJECT":
            entry = raw_lexicon[index % len(raw_lexicon)]
            if not isinstance(entry, Mapping):
                raise ValueError("INTERACT_OBJECT lexicon entries must be objects")
            operation = _required_text(entry, "operation")
            kind = _required_text(entry, "kind")
            template = _required_text(entry, "phrase")
            values = {
                "object": _detail_name(details, "objects", candidate_index),
                "item": _detail_name(details, "items", candidate_index),
                "recipient": _detail_name(details, "recipients", 0),
            }
            phrase = template.format(**values)
            if kind == "object":
                object_key = _detail_key(details, "objects", candidate_index)
            elif kind in {"item", "transfer"}:
                object_key = _detail_key(details, "items", candidate_index)
            else:
                raise ValueError(f"Unknown interaction kind {kind!r}")
            slots = [
                {"name": "OBJECT", "value": object_key},
                {"name": "OPERATION", "value": operation},
            ]
            if kind == "transfer":
                slots.append(
                    {
                        "name": "RECIPIENT",
                        "value": _detail_key(details, "recipients", 0),
                    }
                )
        else:
            raise ValueError(f"Unsupported Phase A family {family}")

        declaration = style.format(phrase=phrase)
        cases.append(
            {
                "case_id": (
                    f"{split_name.lower()}-resolve-{family.lower()}-{index + 1:03d}"
                ),
                "split": split_name,
                "scene_id": scene_id,
                "lineage_group": (
                    f"{split_name.lower()}/resolve/{family.lower()}/"
                    f"{index % len(raw_lexicon):02d}"
                ),
                "declaration": declaration,
                "expected": {
                    "decision": "RESOLVE",
                    "action_family": family,
                    "slots": slots,
                },
                "case_kind": "RESIDUAL_RESOLVE",
                "training_allowed": split is DatasetSplit.TRAIN,
                "tags": [
                    "synthetic",
                    "residual",
                    "resolve",
                    family.lower(),
                    "corpus-v1",
                ],
            }
        )
    return cases


def _ambiguous_cases(
    *,
    plan: Mapping[str, object],
    split: DatasetSplit,
    count: int,
    scenes: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    templates = _mapping(plan, "ambiguous_templates").get(split.value)
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"Missing ambiguous templates for {split.value}")
    scene_ids = _scene_ids(split, scenes)
    rows: list[dict[str, object]] = []
    for index in range(count):
        raw = templates[index % len(templates)]
        if not isinstance(raw, Mapping):
            raise ValueError("Ambiguous templates must be objects")
        scene_id = scene_ids[index % len(scene_ids)]
        details = _mapping(scenes[scene_id], "details")
        family = _required_text(raw, "family")
        missing_slot = _required_text(raw, "missing_slot")
        action = _required_text(raw, "declaration").removeprefix("I ").rstrip(".")
        prefix = _AMBIGUOUS_PREFIXES[(index // len(templates)) % len(_AMBIGUOUS_PREFIXES)]
        declaration = prefix.format(action=action) + "."
        candidate_keys = _candidate_keys_for_missing(details, family, missing_slot)
        rows.append(
            {
                "case_id": (
                    f"{split.value.lower()}-ask-ambiguous-{index + 1:03d}"
                ),
                "split": split.value,
                "scene_id": scene_id,
                "lineage_group": (
                    f"{split.value.lower()}/ask/ambiguous/"
                    f"{index % len(templates):02d}"
                ),
                "declaration": declaration,
                "expected": {
                    "decision": "ASK_PLAYER",
                    "action_family": family,
                    "slots": [],
                    "missing_slots": [missing_slot],
                    "question": _question_for_missing(missing_slot),
                    "candidate_keys": candidate_keys,
                },
                "case_kind": "RESIDUAL_AMBIGUOUS_REFERENCE",
                "training_allowed": split is DatasetSplit.TRAIN,
                "tags": [
                    "synthetic",
                    "residual",
                    "ask-player",
                    "ambiguous-reference",
                    family.lower(),
                    "corpus-v1",
                ],
            }
        )
    return rows


def _compound_cases(
    *,
    plan: Mapping[str, object],
    split: DatasetSplit,
    count: int,
    scenes: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    pairs = _mapping(plan, "compound_pairs").get(split.value)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"Missing compound pairs for {split.value}")
    scene_ids = _scene_ids(split, scenes)
    rows: list[dict[str, object]] = []
    for index in range(count):
        raw_pair = pairs[index % len(pairs)]
        if (
            not isinstance(raw_pair, list)
            or len(raw_pair) != 2
            or not all(isinstance(item, str) for item in raw_pair)
        ):
            raise ValueError("Compound pairs must contain two strings")
        cycle = index // len(pairs)
        scene_id = scene_ids[(index + cycle) % len(scene_ids)]
        details = _mapping(scenes[scene_id], "details")
        values = {
            "object": _detail_name(details, "objects", index % 2),
            "location": _detail_name(details, "locations", index % 2),
            "npc": _detail_name(details, "npcs", index % 2),
            "combatant": _detail_name(details, "combatants", index % 2),
            "item": _detail_name(details, "items", index % 2),
            "recipient": _detail_name(details, "recipients", 0),
        }
        left = raw_pair[0].format(**values)
        right = raw_pair[1].format(**values)
        declaration = f"I {left} and {right}."
        rows.append(
            {
                "case_id": f"{split.value.lower()}-ask-compound-{index + 1:03d}",
                "split": split.value,
                "scene_id": scene_id,
                "lineage_group": (
                    f"{split.value.lower()}/ask/compound/"
                    f"{index % len(pairs):02d}"
                ),
                "declaration": declaration,
                "expected": {
                    "decision": "ASK_PLAYER",
                    "action_family": None,
                    "slots": [],
                    "missing_slots": ["PRIMARY_ACTION"],
                    "question": "Which action do you want to take first?",
                    "candidate_keys": [],
                },
                "case_kind": "RESIDUAL_COMPOUND_ASK_ONLY",
                "training_allowed": split is DatasetSplit.TRAIN,
                "tags": [
                    "synthetic",
                    "residual",
                    "ask-player",
                    "compound",
                    "corpus-v1",
                ],
            }
        )
    return rows


def _zero_model_cases(
    *,
    split: DatasetSplit,
    count: int,
    scenes: Mapping[str, Mapping[str, object]],
    categories: Sequence[str],
) -> list[dict[str, object]]:
    if split is not DatasetSplit.REDTEAM:
        raise ValueError("Zero-model invariants belong in REDTEAM only")
    if count % len(categories) != 0:
        raise ValueError("Zero-model count must divide evenly across categories")
    scene_ids = _scene_ids(split, scenes)
    rows: list[dict[str, object]] = []
    per_category = count // len(categories)
    for category in categories:
        for variant in range(per_category):
            scene_id = scene_ids[variant % len(scene_ids)]
            details = _mapping(scenes[scene_id], "details")
            declaration = _zero_model_declaration(
                category,
                details,
                variant,
            )
            rows.append(
                {
                    "case_id": f"redteam-zero-{category.lower()}-{variant + 1:02d}",
                    "split": split.value,
                    "scene_id": scene_id,
                    "lineage_group": (
                        f"redteam/zero-model/{category.lower()}"
                    ),
                    "declaration": declaration,
                    "expected_status": "TERMINATED_BEFORE_RESIDUAL",
                    "expected_route": "NO_ACTION",
                    "expected_failed_deterministic_stage": None,
                    "expected_permitted_decisions": None,
                    "case_kind": f"ZERO_MODEL_{category}",
                    "training_allowed": False,
                    "tags": [
                        "synthetic",
                        "zero-model",
                        "commitment-veto",
                        category.lower(),
                        "corpus-v1",
                    ],
                }
            )
    return rows


def _zero_model_declaration(
    category: str,
    details: Mapping[str, object],
    variant: int,
) -> str:
    combatant = _detail_name(details, "combatants", variant % 2)
    obj = _detail_name(details, "objects", variant % 2)
    npc = _detail_name(details, "npcs", variant % 2)
    ally = _detail_name(details, "recipients", 0)
    if category == "NEGATION":
        return f"I do not attack {combatant}."
    if category == "QUESTION":
        return f"Should I inspect {obj}?"
    if category == "CANCELLATION":
        return f"Never mind, I open {obj}."
    if category == "HYPOTHETICAL":
        return f"If I attack {combatant}, I would retreat."
    if category == "REPORTED":
        return f"{npc} said I should inspect {obj}."
    if category == "QUOTED":
        return f'I quote "attack {combatant}".'
    if category == "CORRECTION":
        return f"Actually, I inspect {obj}."
    if category == "FRAMING":
        return f"I plan to attack {combatant} while {ally} watches."
    raise ValueError(f"Unknown zero-model category {category!r}")


def _scene(
    *,
    split: DatasetSplit,
    scene_id: str,
    theme: str,
    ordinal: int,
) -> dict[str, object]:
    slug = _slug(theme)
    item_base = {
        DatasetSplit.TRAIN: 1,
        DatasetSplit.DEV: 101,
        DatasetSplit.TEST: 201,
        DatasetSplit.REDTEAM: 301,
    }[split] + ordinal * 2
    objects = [
        _entity(f"object:{slug}-hatch", f"{theme} Hatch"),
        _entity(f"object:{slug}-coffer", f"{theme} Coffer"),
    ]
    locations = [
        _entity(f"location:{slug}-passage", f"{theme} Passage"),
        _entity(f"location:{slug}-landing", f"{theme} Landing"),
    ]
    npcs = [
        _entity(f"npc:{slug}-steward", f"{theme} Steward"),
        _entity(f"npc:{slug}-scribe", f"{theme} Scribe"),
    ]
    combatants = [
        _entity(f"combatant:{slug}-raider", f"{theme} Raider"),
        _entity(f"combatant:{slug}-scout", f"{theme} Scout"),
    ]
    target_spells = [
        _entity(f"spell:{slug}-lance", f"{theme} Lance"),
    ]
    self_spells = [
        _entity(f"spell:{slug}-nova", f"{theme} Nova"),
    ]
    items = [
        _entity(f"inventory:item-{item_base}", f"{theme} Tonic"),
        _entity(f"inventory:item-{item_base + 1}", f"{theme} Draught"),
    ]
    recipients = [
        _entity(f"pc:{slug}-ally", f"{theme} Ally"),
    ]
    source = "fixture:phase-a-corpus-v1"
    facts: list[dict[str, object]] = [
        _fact(
            "session.scene",
            (
                f"A non-canon {theme.lower()} chamber with two objects, two exits, "
                "two NPCs, two combatants, admitted spells, and carried items."
            ),
            source,
            "TABLE",
        ),
    ]
    for entity in objects:
        identity = str(entity["key"]).split(":", 1)[1]
        facts.append(
            _fact(
                f"campaign.entity.object.{identity}",
                entity["name"],
                source,
                "TABLE",
            )
        )
    for entity in locations:
        identity = str(entity["key"]).split(":", 1)[1]
        facts.append(
            _fact(
                f"campaign.entity.location.{identity}",
                entity["name"],
                source,
                "TABLE",
            )
        )
    for entity in npcs:
        identity = str(entity["key"]).split(":", 1)[1]
        facts.append(
            _fact(
                f"campaign.entity.npc.{identity}",
                entity["name"],
                source,
                "TABLE",
            )
        )
    for entity in combatants:
        identity = str(entity["key"]).split(":", 1)[1]
        facts.extend(
            (
                _fact(
                    f"combatant.{identity}.name",
                    entity["name"],
                    source,
                    "TABLE",
                ),
                _fact(
                    f"combatant.{identity}.hidden",
                    False,
                    source,
                    "TABLE",
                ),
            )
        )
    target_identity = str(target_spells[0]["key"]).split(":", 1)[1]
    self_identity = str(self_spells[0]["key"]).split(":", 1)[1]
    facts.extend(
        (
            _fact(
                f"spell.{target_identity}.name",
                target_spells[0]["name"],
                source,
                "PC_PRIVATE",
            ),
            _fact(
                f"spell.{target_identity}.fixture",
                "SIMPLE_ATTACK",
                source,
                "PC_PRIVATE",
            ),
            _fact(
                f"spell.{self_identity}.name",
                self_spells[0]["name"],
                source,
                "PC_PRIVATE",
            ),
            _fact(
                f"spell.{self_identity}.fixture",
                "SELF_RADIUS_SAVE_DAMAGE",
                source,
                "PC_PRIVATE",
            ),
        )
    )
    for offset, entity in enumerate(items):
        item_id = item_base + offset
        facts.extend(
            (
                _fact(
                    f"inventory.item.{item_id}.name",
                    entity["name"],
                    source,
                    "PC_PRIVATE",
                ),
                _fact(
                    f"inventory.item.{item_id}.catalog_item_id",
                    "potion_healing",
                    source,
                    "PC_PRIVATE",
                ),
                _fact(
                    f"inventory.item.{item_id}.carried_by_player_id",
                    "tester",
                    source,
                    "PC_PRIVATE",
                ),
            )
        )
    ally_identity = str(recipients[0]["key"]).split(":", 1)[1]
    facts.extend(
        (
            _fact(
                f"campaign.entity.pc.{ally_identity}",
                recipients[0]["name"],
                source,
                "TABLE",
            ),
            _fact(
                f"character.{ally_identity}.name",
                recipients[0]["name"],
                source,
                "TABLE",
            ),
            _fact(
                f"campaign.entity.npc.{slug}-hidden-observer",
                f"{theme} Hidden Observer",
                source,
                "GM_ONLY",
            ),
        )
    )
    return {
        "snapshot_revision": f"phase-a-corpus-v1-{scene_id}",
        "facts": facts,
        "details": {
            "objects": objects,
            "locations": locations,
            "npcs": npcs,
            "combatants": combatants,
            "target_spells": target_spells,
            "self_spells": self_spells,
            "items": items,
            "recipients": recipients,
        },
    }


def _candidate_keys_for_missing(
    details: Mapping[str, object],
    family: str,
    missing_slot: str,
) -> list[str]:
    groups: tuple[str, ...]
    if family == "HARM_TARGET" and missing_slot == "TARGET_COMBATANT":
        groups = ("combatants",)
    elif family == "MOVE_TRAVEL" and missing_slot == "DESTINATION":
        groups = ("locations",)
    elif family == "SEARCH_INSPECT" and missing_slot == "SUBJECT":
        groups = (
            "objects",
            "locations",
            "npcs",
            "combatants",
            "target_spells",
            "self_spells",
            "items",
            "recipients",
        )
    elif family == "INTERACT_OBJECT" and missing_slot == "OBJECT":
        groups = ("objects", "items")
    elif family == "SPEECH_ONLY" and missing_slot == "ADDRESSEE":
        groups = ("npcs",)
    else:
        raise ValueError(
            f"Unsupported ambiguous slot {family}.{missing_slot}"
        )
    return sorted(
        str(entity["key"])
        for group in groups
        for entity in _entity_list(details, group)
    )


def _question_for_missing(slot: str) -> str:
    return {
        "TARGET_COMBATANT": "Which visible combatant do you mean?",
        "DESTINATION": "Which visible destination do you mean?",
        "SUBJECT": "Which visible subject do you mean?",
        "OBJECT": "Which visible object do you mean?",
        "ADDRESSEE": "Who are you speaking to?",
    }[slot]


def _protected_language_case(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "case_id": row["case_id"],
        "split": row["split"],
        "lineage_group": row["lineage_group"],
        "declaration": row["declaration"],
        "training_allowed": False,
        "tags": row["tags"],
    }


def _language_leakage(
    rows: Sequence[Mapping[str, object]],
) -> LeakageReport:
    examples = []
    for row in rows:
        split = DatasetSplit(str(row["split"]))
        payload = {"declaration": str(row["declaration"])}
        examples.append(
            LeakageExample(
                example_id=str(row["case_id"]),
                split=split,
                lineage_group=str(row["lineage_group"]),
                fingerprint=content_fingerprint(payload),
                normalized_fingerprint=normalized_fingerprint(payload),
                near_signature=near_signature(payload),
            )
        )
    return scan_leakage(examples)


def _assert_target_counts(
    plan: Mapping[str, object],
    residual_cases: Mapping[str, Sequence[Mapping[str, object]]],
    zero_model_cases: Sequence[Mapping[str, object]],
) -> None:
    targets = _mapping(plan, "target_counts")
    for split in DatasetSplit:
        expected = _int_field(_mapping(targets, split.value), "total")
        actual = len(residual_cases[split.value]) + sum(
            1 for row in zero_model_cases if row["split"] == split.value
        )
        if actual != expected:
            raise ValueError(
                f"{split.value} corpus count is {actual}; expected {expected}"
            )


def _assert_family_coverage(
    plan: Mapping[str, object],
    residual_cases: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    required = set(_string_list(plan.get("required_family_coverage")))
    for split in DatasetSplit:
        observed = {
            str(_mapping(row, "expected")["action_family"])
            for row in residual_cases[split.value]
            if _mapping(row, "expected")["decision"] == "RESOLVE"
        }
        if observed != required:
            raise ValueError(
                f"{split.value} family coverage {sorted(observed)} "
                f"does not match {sorted(required)}"
            )


def _assert_resolve_lexicons_are_residual_only(
    plan: Mapping[str, object],
) -> None:
    lexicons = _mapping(plan, "resolve_lexicons")
    for split in DatasetSplit:
        split_rows = _mapping(lexicons, split.value)
        for family in _FAMILIES:
            raw = split_rows.get(family)
            if not isinstance(raw, list):
                raise ValueError(f"Missing {split.value}.{family} lexicon")
            phrases = []
            for item in raw:
                if isinstance(item, str):
                    phrases.append(item)
                elif isinstance(item, Mapping):
                    phrases.append(_required_text(item, "phrase"))
                else:
                    raise ValueError("Corpus lexicon entries must be strings or objects")
            for phrase in phrases:
                normalized = phrase.casefold()
                for term in _DETERMINISTIC_TERMS:
                    if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized):
                        raise ValueError(
                            f"{split.value}.{family} residual lexicon phrase "
                            f"{phrase!r} contains deterministic term {term!r}"
                        )


def _assert_unique_case_ids(
    residual_cases: Mapping[str, Sequence[Mapping[str, object]]],
    zero_model_cases: Sequence[Mapping[str, object]],
) -> None:
    ids = [
        str(row["case_id"])
        for rows in residual_cases.values()
        for row in rows
    ] + [str(row["case_id"]) for row in zero_model_cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated Phase A case IDs are not unique")


def _scenes_for_cases(
    scenes: Mapping[str, Mapping[str, object]],
    cases: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    used = sorted({str(row["scene_id"]) for row in cases})
    return {
        scene_id: {
            "snapshot_revision": scenes[scene_id]["snapshot_revision"],
            "facts": scenes[scene_id]["facts"],
        }
        for scene_id in used
    }


def _scene_ids(
    split: DatasetSplit,
    scenes: Mapping[str, Mapping[str, object]],
) -> list[str]:
    prefix = split.value.lower() + "-"
    values = sorted(
        scene_id
        for scene_id in scenes
        if scene_id.startswith(prefix)
    )
    if not values:
        raise ValueError(f"No scenes generated for {split.value}")
    return values


def _entity(key: str, name: str) -> dict[str, str]:
    return {"key": key, "name": name}


def _fact(
    key: str,
    value: object,
    source: str,
    visibility: str,
) -> dict[str, object]:
    return {
        "key": key,
        "value": value,
        "source": source,
        "visibility": visibility,
    }


def _entity_list(
    details: Mapping[str, object],
    group: str,
) -> list[Mapping[str, object]]:
    raw = details.get(group)
    if not isinstance(raw, list) or not raw or not all(
        isinstance(item, Mapping) for item in raw
    ):
        raise ValueError(f"Scene details missing {group}")
    return [item for item in raw if isinstance(item, Mapping)]


def _detail_name(
    details: Mapping[str, object],
    group: str,
    index: int,
) -> str:
    rows = _entity_list(details, group)
    return _required_text(rows[index % len(rows)], "name")


def _detail_key(
    details: Mapping[str, object],
    group: str,
    index: int,
) -> str:
    rows = _entity_list(details, group)
    return _required_text(rows[index % len(rows)], "key")


def _string_item(raw: Sequence[object], index: int) -> str:
    value = raw[index % len(raw)]
    if not isinstance(value, str) or not value:
        raise ValueError("Expected string lexicon item")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError("Expected non-empty string list")
    return [str(item) for item in value]


def _mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    raw = value.get(key)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be an object")
    return raw


def _required_text(value: Mapping[str, object], key: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw.strip()


def _int_field(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _validate_plan(plan: Mapping[str, object]) -> None:
    if plan.get("schema") != "ml-lab-phase-a-corpus-plan/1":
        raise ValueError("Unsupported Phase A corpus plan schema")
    if plan.get("status") != "SYNTHETIC_NON_CANON":
        raise ValueError("Phase A corpus plan must remain synthetic non-canon")
    if plan.get("production_data") is not False:
        raise ValueError("Production data cannot enter Phase A corpus generation")
    if plan.get("transcript_derived") is not False:
        raise ValueError("Phase A corpus plan cannot contain transcript-derived text")
    if plan.get("target_contract") != "semantic-residual-v2":
        raise ValueError("Phase A corpus plan targets the wrong contract")
    targets = _mapping(plan, "target_counts")
    if sum(_int_field(_mapping(targets, split.value), "total") for split in DatasetSplit) < 400:
        raise ValueError("Phase A corpus expansion must contain at least 400 authored cases")
