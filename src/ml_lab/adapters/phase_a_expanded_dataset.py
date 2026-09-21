from __future__ import annotations

import hashlib
from pathlib import Path

from ml_lab.adapters.phase_a_corpus import (
    DEFAULT_CORPUS_PLAN,
    generate_phase_a_corpus,
)
from ml_lab.adapters.phase_a_dataset_factory import run_phase_a_dataset_factory
from ml_lab.datasets.leakage import canonical_json
from ml_lab.storage.workspace import Workspace


def build_expanded_phase_a_dataset(
    *,
    workspace: Workspace,
    frankenhomie_repository: Path,
    output_dir: Path,
    plan_path: Path = DEFAULT_CORPUS_PLAN,
) -> dict[str, object]:
    root = output_dir.expanduser().resolve()
    authored_root = root / "authored"
    authoritative_root = root / "authoritative"
    cache_root = root / "case-cache"
    root.mkdir(parents=True, exist_ok=True)

    corpus_receipt = generate_phase_a_corpus(
        output_dir=authored_root,
        plan_path=plan_path,
    )
    if corpus_receipt.get("ok") is not True:
        raise ValueError("Expanded Phase A corpus generation did not pass")

    train_dev_seed = authored_root / "phase-a-expanded-train-dev-seed-v1.json"
    protected_language = authored_root / "phase-a-expanded-protected-language-v1.json"
    factory_receipt = run_phase_a_dataset_factory(
        workspace=workspace,
        frankenhomie_repository=frankenhomie_repository,
        output_dir=authoritative_root,
        seed_path=train_dev_seed,
        protected_seed_path=protected_language,
        case_cache_dir=cache_root,
    )

    base_receipt: dict[str, object] = {
        "schema": "ml-lab-phase-a-expanded-authoritative-build/1",
        "ok": factory_receipt.get("ok") is True,
        "target_frankenhomie_commit": factory_receipt.get(
            "target_frankenhomie_commit"
        ),
        "target_contract": factory_receipt.get("target_contract"),
        "authored_case_count": corpus_receipt.get("case_count"),
        "authoritative_candidate_case_count": factory_receipt.get("case_count"),
        "authoritative_emitted_count": factory_receipt.get("emitted_count"),
        "authoritative_split_counts": factory_receipt.get("split_counts"),
        "corpus_payload_sha256": corpus_receipt.get("payload_sha256"),
        "factory_payload_sha256": factory_receipt.get("payload_sha256"),
        "dataset_sha256": factory_receipt.get("dataset_sha256"),
        "errors": factory_receipt.get("errors"),
        "case_cache": factory_receipt.get("case_cache"),
        "production_data": False,
        "transcript_derived": False,
        "training_started": False,
        "frozen": False,
        "integration_gate": "NO_GO",
    }
    payload_sha = hashlib.sha256(
        canonical_json(base_receipt).encode("utf-8")
    ).hexdigest()
    receipt = {**base_receipt, "payload_sha256": payload_sha}
    (root / "expanded-build-receipt.json").write_bytes(
        (canonical_json(receipt) + "\n").encode("utf-8")
    )
    return receipt
