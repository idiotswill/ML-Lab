# Phase 3 Exit Audit

Status: **PASS — PHASE 3 EXIT GATES VERIFIED**

Audit date: 2026-09-18

Verified implementation baseline: `a8fcdf3655f5dd76d4c5ee95ad6f1dbd55f144c8`

Frankenhomie audit anchor: `1ff3e2155a3c2d2b316023e9933edf3a6d53697f`

Frankenhomie roadmap status at audit: `IN_PROGRESS_PHASE_B2`. Phase A independent acceptance remains deferred by the user; the binding authority rule remains unchanged.

This audit separates **implementation evidence** from **execution evidence**. Phase 3 implementation and its inherited Phase 2 regression gates were executed successfully on GitHub Actions run `35373237778` for implementation head `a8fcdf3655f5dd76d4c5ee95ad6f1dbd55f144c8`.

## Closure decision

Phase 3 exit gates are **verified**.

Run `35373237778` passed Ubuntu core, Windows core, the dedicated Phase 3 scale suite, and the dependent Windows standalone build/smoke. The core suite reported 102 passed / 10 deselected on Ubuntu; the scale suite reported 10 passed / 102 deselected. Ruff and strict mypy passed, as did storage, worker, restart-recovery, QML load, full-workspace QML, and performance smokes on both core platforms. The standalone executable built successfully and passed compiled storage/worker/restart/QML/performance smokes before its artifact was retained.

No remaining Phase 3 authority/workflow implementation blocker was found. Phase 4 has not started. Integration remains `NO_GO`.

## Authority boundary

ML Lab remains a standalone development workbench. It freezes committed Frankenhomie contract bytes, trains/evaluates bounded candidates, and packages models plus evidence. It does not deploy into Frankenhomie, mutate game state, call the ordinary resolver, grant itself integration approval, or become a second campaign authority.

The Phase A reference harness executes only pinned `run_residual_semantics` with a recorded proposal provider in an isolated child. SQLite and network access are denied in that validator, resolver dispatch is unavailable, and receipts record `authority_mutation_allowed = false`. The localhost-provider runner permits loopback HTTP only, denies SQLite/non-loopback access, does not invoke resolver dispatch, and still sends proposals through the separate network-disabled pinned reference validator.

## Phase 3 functional gates

| Gate | Repository status | Evidence |
| --- | --- | --- |
| Streaming dataset import + per-row errors | PASS | `tests/test_datasets.py::test_streaming_jsonl_import_reports_row_errors` |
| Physical TRAIN/DEV/TEST/REDTEAM partitions + hashes | PASS | `test_freeze_creates_four_partitions_and_hides_protected_handles`; dataset freeze manifests |
| Exact/normalized/lineage/near-duplicate detection | PASS | `test_cross_split_duplicate_blocks_freeze`, `test_lineage_group_crossing_partitions_is_blocking`, `test_streaming_leakage_scan_preserves_normalized_and_near_duplicate_checks` |
| Protected leakage blocks eligibility | PASS | dataset freeze/eligibility checks and blocking leakage fixtures |
| Trainer sees only TRAIN + permitted DEV | PASS | `tests/test_experiments.py::test_trainer_spec_cannot_expose_protected_partitions`; `tests/test_training_service.py` |
| Completed experiments immutable | PASS | `test_completed_experiment_is_immutable_and_persists_metrics` |
| Deterministic abstention baseline | PASS | `tests/test_phase_a_science.py::test_phase_a_baselines_share_frozen_dataset_and_expose_veto_metrics` |
| Bounded lexical/reference baseline | PASS | same Phase A science fixture; vocabulary is pinned-contract derived |
| Localhost-provider baseline | PASS | `tests/test_phase_a_local_provider.py::test_local_provider_baseline_is_same_dataset_and_persists_provider_receipts` |
| Trainable local baseline | PASS | isolated Phase A bounded sparse trainer + `test_phase_a_training_worker_is_train_only_and_produces_bounded_model` |
| Baselines/candidate comparable in one project | PASS | completed experiments share project/dataset/contract identity and Compare pages project-scoped completed experiments |
| Metric direction + veto classification | PASS | `MetricValue` records direction/veto; Phase A science tests expose veto metrics and distinguish NOT MEASURED from zero |
| Aggregate -> cases -> immutable failure drilldown | PASS | `tests/test_compare.py::test_compare_drills_from_case_to_immutable_failure_record` |
| Seeded red-team reproducibility | PASS | `tests/test_redteam.py::test_redteam_suite_is_seeded_and_order_independent` |
| Failure -> regression without history mutation | PASS | `tests/test_registry.py::test_failure_regression_membership_preserves_historical_payload`; red-team/failure fixture |
| Multiple models + explicit promotion history | PASS | `tests/test_registry.py::test_model_promotion_is_sequential_and_stops_before_integration` |
| Release bundle required hashes/manifests/failures | PASS | `tests/test_bundles.py::test_bundle_is_deterministic_split_safe_and_fresh_verifiable` |
| Separate-process fresh verification receipt | PASS | bundle fresh verifier and immutable receipt tests |
| Phase A pinned real contract/read-only validator | PASS | `tests/test_phase_a_reference.py`; `tests/test_phase_a_workflow.py::test_phase_a_protected_evaluation_uses_pinned_reference_commit` |
| Phase A experiment launchable from GUI controller | PASS | `tests/test_phase_a_workflow.py::test_phase_a_experiment_can_launch_through_gui_controller` |
| No Lab game-state mutation path | PASS | pinned reference/provider sandboxes deny DB/authority access; resolver dispatch is not exposed; ordinary promotion cannot reach `INTEGRATION_APPROVED` |

## Phase 3 scale/stability gates

| Gate | Repository status | Current-head execution |
| --- | --- | --- |
| 100k dataset examples | PASS | PASS — run `35373237778`, scale suite |
| 10k failures/regressions | PASS | PASS — run `35373237778`, scale suite |
| 1k experiment metadata | PASS | PASS — run `35373237778`, scale suite |
| 100 registry models | PASS | PASS — run `35373237778`, scale suite |
| Hash/import/evaluation interruption + restart | PASS | PASS — run `35373237778`, scale suite |
| Corrupt artifacts/manifests | PASS | PASS — run `35373237778`, scale suite |
| Paged/filterable large collections | PASS | PASS — scale fixtures executed in run `35373237778` |
| No eager 100k payload materialization | PASS | PASS — 100k Data Studio fixture executed in run `35373237778` |
| UI remains structurally asynchronous during long work | PASS | PASS — core QML/full-workspace smokes and scale suite passed in run `35373237778` |

## Inherited Phase 2 gates

Run `35373237778` re-executed the inherited Phase 2 regression gates on the Phase 3 implementation head. Ubuntu and Windows core both passed Ruff, strict mypy, 102 core tests, storage/worker/restart smokes, QML load/full-workspace smokes, and performance instrumentation. The dependent Windows standalone job then built the self-contained executable and passed compiled storage/worker/restart/QML/performance smokes.

## Exact-head implementation verification

GitHub Actions run `35373237778` for `a8fcdf3655f5dd76d4c5ee95ad6f1dbd55f144c8` completed successfully:

- Ubuntu core: success;
- Windows core: success;
- Phase 3 scale: success;
- Windows standalone: success.

The retained developer standalone artifact has digest `sha256:95028be87754fce9ba98ce86398cf881d94146825fadf0edbea0007fdda6087c`. It is a developer verification artifact, **not** a Phase 4 `TESTING_READY` release.

## What is required after Phase 3

1. Keep PR #5 integration status `NO_GO`.
2. Complete normal review/merge of the verified Phase 3 PR.
3. Only after Phase 3 is merged, create `phase/4-testing-ready`.
4. Phase 4 remains the first phase eligible to produce a normal user-testing build.
