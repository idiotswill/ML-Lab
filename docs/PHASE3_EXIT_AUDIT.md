# Phase 3 Exit Audit

Status: **BLOCKED ON CURRENT-HEAD VERIFICATION**

Audit date: 2026-09-18

Audited implementation baseline: `21675d697e84ee5352534fad59167b5c093fd822`

Frankenhomie audit anchor: `1ff3e2155a3c2d2b316023e9933edf3a6d53697f`

Frankenhomie roadmap status at audit: `IN_PROGRESS_PHASE_B2`. Phase A independent acceptance remains deferred by the user; the binding authority rule remains unchanged.

This audit separates **implementation evidence** from **execution evidence**. A gate marked IMPLEMENTED has a concrete service/UI path and regression fixture in the repository. It is not considered fully passed on the current head until the applicable exact-head CI jobs execute successfully.

## Closure decision

Phase 3 is **not complete**.

No remaining Phase 3 authority/workflow implementation blocker was found in this audit. The closure blocker is verification:

- exact-head GitHub Actions is failing before checkout/step 1, so current core, QML, Windows, type/lint, and standalone regressions have not executed;
- the Phase 3 scale job and the later scale/stability fixtures have no successful exact-head execution evidence;
- therefore the inherited Phase 2 gates cannot yet be asserted to still pass on the Phase 3 head.

Do not merge PR #5 or begin Phase 4 from this audit alone. Integration remains `NO_GO`.

## Authority boundary

ML Lab remains a standalone development workbench. It freezes committed Frankenhomie contract bytes, trains/evaluates bounded candidates, and packages models plus evidence. It does not deploy into Frankenhomie, mutate game state, call the ordinary resolver, grant itself integration approval, or become a second campaign authority.

The Phase A reference harness executes only pinned `run_residual_semantics` with a recorded proposal provider in an isolated child. SQLite and network access are denied in that validator, resolver dispatch is unavailable, and receipts record `authority_mutation_allowed = false`. The localhost-provider runner permits loopback HTTP only, denies SQLite/non-loopback access, does not invoke resolver dispatch, and still sends proposals through the separate network-disabled pinned reference validator.

## Phase 3 functional gates

| Gate | Repository status | Evidence |
| --- | --- | --- |
| Streaming dataset import + per-row errors | IMPLEMENTED | `tests/test_datasets.py::test_streaming_jsonl_import_reports_row_errors` |
| Physical TRAIN/DEV/TEST/REDTEAM partitions + hashes | IMPLEMENTED | `test_freeze_creates_four_partitions_and_hides_protected_handles`; dataset freeze manifests |
| Exact/normalized/lineage/near-duplicate detection | IMPLEMENTED | `test_cross_split_duplicate_blocks_freeze`, `test_lineage_group_crossing_partitions_is_blocking`, `test_streaming_leakage_scan_preserves_normalized_and_near_duplicate_checks` |
| Protected leakage blocks eligibility | IMPLEMENTED | dataset freeze/eligibility checks and blocking leakage fixtures |
| Trainer sees only TRAIN + permitted DEV | IMPLEMENTED | `tests/test_experiments.py::test_trainer_spec_cannot_expose_protected_partitions`; `tests/test_training_service.py` |
| Completed experiments immutable | IMPLEMENTED | `test_completed_experiment_is_immutable_and_persists_metrics` |
| Deterministic abstention baseline | IMPLEMENTED | `tests/test_phase_a_science.py::test_phase_a_baselines_share_frozen_dataset_and_expose_veto_metrics` |
| Bounded lexical/reference baseline | IMPLEMENTED | same Phase A science fixture; vocabulary is pinned-contract derived |
| Localhost-provider baseline | IMPLEMENTED | `tests/test_phase_a_local_provider.py::test_local_provider_baseline_is_same_dataset_and_persists_provider_receipts` |
| Trainable local baseline | IMPLEMENTED | isolated Phase A bounded sparse trainer + `test_phase_a_training_worker_is_train_only_and_produces_bounded_model` |
| Baselines/candidate comparable in one project | IMPLEMENTED | completed experiments share project/dataset/contract identity and Compare pages project-scoped completed experiments |
| Metric direction + veto classification | IMPLEMENTED | `MetricValue` records direction/veto; Phase A science tests expose veto metrics and distinguish NOT MEASURED from zero |
| Aggregate -> cases -> immutable failure drilldown | IMPLEMENTED | `tests/test_compare.py::test_compare_drills_from_case_to_immutable_failure_record` |
| Seeded red-team reproducibility | IMPLEMENTED | `tests/test_redteam.py::test_redteam_suite_is_seeded_and_order_independent` |
| Failure -> regression without history mutation | IMPLEMENTED | `tests/test_registry.py::test_failure_regression_membership_preserves_historical_payload`; red-team/failure fixture |
| Multiple models + explicit promotion history | IMPLEMENTED | `tests/test_registry.py::test_model_promotion_is_sequential_and_stops_before_integration` |
| Release bundle required hashes/manifests/failures | IMPLEMENTED | `tests/test_bundles.py::test_bundle_is_deterministic_split_safe_and_fresh_verifiable` |
| Separate-process fresh verification receipt | IMPLEMENTED | bundle fresh verifier and immutable receipt tests |
| Phase A pinned real contract/read-only validator | IMPLEMENTED | `tests/test_phase_a_reference.py`; `tests/test_phase_a_workflow.py::test_phase_a_protected_evaluation_uses_pinned_reference_commit` |
| Phase A experiment launchable from GUI controller | IMPLEMENTED | `tests/test_phase_a_workflow.py::test_phase_a_experiment_can_launch_through_gui_controller` |
| No Lab game-state mutation path | IMPLEMENTED | pinned reference/provider sandboxes deny DB/authority access; resolver dispatch is not exposed; ordinary promotion cannot reach `INTEGRATION_APPROVED` |

## Phase 3 scale/stability gates

| Gate | Repository status | Current-head execution |
| --- | --- | --- |
| 100k dataset examples | IMPLEMENTED | UNVERIFIED — `tests/test_data_studio_scale.py` is marked scale |
| 10k failures/regressions | IMPLEMENTED | UNVERIFIED — `tests/test_failures_scale.py` |
| 1k experiment metadata | IMPLEMENTED | UNVERIFIED — `tests/test_compare_scale.py` |
| 100 registry models | IMPLEMENTED | UNVERIFIED — `tests/test_registry_scale.py` |
| Hash/import/evaluation interruption + restart | IMPLEMENTED | UNVERIFIED — `tests/test_restart_stability.py` |
| Corrupt artifacts/manifests | IMPLEMENTED | UNVERIFIED — corruption fixtures in `tests/test_bundles.py` |
| Paged/filterable large collections | IMPLEMENTED | UNVERIFIED — SQL paging + bounded page fixtures exist |
| No eager 100k payload materialization | IMPLEMENTED | UNVERIFIED — Data Studio scale fixture exercises 100-row pages |
| UI remains structurally asynchronous during long work | IMPLEMENTED | UNVERIFIED — QThreadPool/worker boundaries and virtualized QML lists exist; current-head QML/scale jobs have not executed |

## Inherited Phase 2 gates

The latest fully verified earlier head is `1e1dd038b4ed712493ffd996a02a7ed217a7b0c4`, with successful GitHub Actions run `35241361842`. That run executed and passed Ubuntu core, Windows core, Ruff, mypy, pytest, storage/worker/restart smokes, QML load/full-workspace smokes, performance instrumentation, Windows standalone build, and compiled application/worker/QML/performance smokes.

That successful run predates later Phase 3 increments and the dedicated scale job. It is useful regression history, but it cannot prove that all Phase 2 gates still pass on the current Phase 3 head.

## Current CI blocker

Exact-head run `35325399630` for `21675d697e84ee5352534fad59167b5c093fd822` completed as failure before repository steps ran:

- Ubuntu core: failure, `steps = null`;
- scale: failure, `steps = null`;
- Windows core: cancelled, `steps = null`;
- Windows standalone: skipped.

No Ruff, mypy, pytest, scale, QML, or standalone result is claimed from that run.

## What is required to close Phase 3

1. Obtain a successful exact-head CI execution in which core jobs actually run on Ubuntu and Windows.
2. Obtain a successful exact-head `pytest -m scale` job covering every marked Phase 3 scale/stability fixture.
3. Obtain a successful dependent Windows standalone build/smoke on that same Phase 3 head.
4. Re-run this audit against that verified head and confirm no gate regressed.
5. Only then mark Phase 3 complete, make PR #5 non-draft/merge as appropriate, and create `phase/4-testing-ready`.

Phase 4 remains the first phase eligible to produce a normal user-testing build.
