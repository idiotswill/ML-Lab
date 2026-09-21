# Phase 4 Exit Audit

Status: **PASS — v0.1.0-testing.5 PRODUCT TESTING READY**

Phase 4 is complete. All automated, packaging, clean-machine, reliability, UX, scientific/reproducibility, Frankenhomie-safety, representative-performance, and fail-closed promotion gates have current evidence.

`v0.1.0-testing.5` is authorized for **product testing only**. This does not authorize model integration into Frankenhomie.

## Authority boundary

ML Lab remains standalone development tooling. It does not install or activate a model in Frankenhomie, mutate game state, become a second state/memory system, legalize actions, establish canon, or bypass Frankenhomie safety/revalidation.

`INTEGRATION_GATE = NO_GO` remains binding.

A product build becoming `TESTING_READY` is not model integration approval. The testing-ready promotion manifest explicitly records:

- `promotion_scope: PRODUCT_TESTING_ONLY`;
- `integration_gate: NO_GO`;
- `model_integration_approved: false`.

## Frankenhomie contract anchor

Phase 4 remains anchored to Frankenhomie commit:

`1ff3e2155a3c2d2b316023e9933edf3a6d53697f`

Roadmap status at the latest Phase 4 contract check:

`IN_PROGRESS_PHASE_B2`

No Phase A residual-semantic contract drift was found during the current productization work.

## Gate status

| Phase 4 gate | Status | Evidence |
| --- | --- | --- |
| Windows installer + portable packaging | PASS | clean CI builds versioned x64 installer and deterministic portable archive with source/toolchain/hash provenance |
| No separately installed Python required | PASS | installed clean-machine workflow runs with Python removed from PATH |
| Per-user install / Unicode + space paths / uninstall | PASS | compiled installer smoke and preserved external workspace |
| Clean-machine end-to-end workflow | PASS | install → workspace/project → worker → frozen dataset/experiment/model → bundle → fresh verification → second-process persistence → uninstall |
| Reliability | PASS | interruption reconciliation, atomic artifact failure behavior, corruption handling, migration backup/restore, correlation IDs, local-only crash reports |
| UX | PASS | all ten pages reviewed at 100/125/150/200% in dark/light, exact 1366×768 evidence, live job controls |
| Scientific/reproducibility | PASS | deterministic retraining proof, explicit nondeterministic tolerance policy, all split hashes, TRAIN/DEV-only reproduction inputs |
| Frankenhomie safety | PASS | pinned preflight before model calls, no DB/network/resolver access, zero-model-route measurement, unsupported-authority veto, permanent integration NO_GO |
| Performance instrumentation | PASS | compiled candidate measures interactive startup, idle RAM, UI stalls, bounded paging, background import, CPU-heavy worker navigation, cancellation visibility |
| Representative physical-Windows performance | **PASS** | testing.5 full receipt: 100k rows, 20k import, zero >100 ms import stalls, three clean navigation rounds, 1.93 s startup, 125.93 MB idle RAM, 37.43 ms cancellation acknowledgement |

## Preserved testing.1 physical failure

The first physical-machine candidate, `v0.1.0-testing.1` at source head `3f5dc329c6615cf66d1e7bd6a9abdda33d5e303a`, was **not** promoted.

Its full representative receipt was gate-evaluable and bound to candidate executable SHA-256:

`97b6eef28ca20260a2a2ce2508527010ed9d5d579b4a7b59463de01b4aee61c6`

Receipt SHA-256:

`b13c7813e21b7ae0a4532721387d09a84f338d46bad575942a80e151207faff8`

The receipt passed startup, idle RAM, ordinary navigation, 100k bounded paging, CPU-worker navigation, cancellation responsiveness, and all instrumentation checks. It failed exactly one hard check: the 20,000-row background import produced one 132.31 ms GUI-thread navigation/process-events stall, above the 100 ms hard threshold.

No threshold was weakened and no testing-ready manifest was produced. The failure led to a Data Studio fix that prepares the post-import view snapshot on the worker thread and applies cached view state on the GUI thread rather than re-querying dataset state during completion notification.

The corrected binary was issued as `v0.1.0-testing.2` so the failed `testing.1` bytes remain unambiguous evidence.

## Preserved testing.2 physical non-promotion

The second physical-machine candidate, `v0.1.0-testing.2` at source head `67f3e18ae16657bf2b41bed08b86cc165106e53c`, was also **not** promoted.

Its full representative receipt is gate-evaluable, bound to executable SHA-256:

`f9202e7f588342f167496cd0c5289fccfe801123043dc04b3118ff01bba7c8e4`

and has receipt SHA-256:

`369a8c5589cdcfe5a68cf3132b4116aac5fae04e346475a1b242c1deae050b30`

The testing.1 regression was fixed: the 20,000-row background import completed with maximum navigation/event-processing time 83.11 ms and zero >100 ms import stalls. Startup, RAM, 100k bounded paging, worker-load navigation, cancellation, and all instrumentation checks also passed.

Testing.2 recorded one 116.83 ms ordinary-navigation event-processing sample across 77 samples. The harness treated any single >100 ms observation as failure, but the binding `QUALITY_GATES.md` wording is "ordinary navigation/input shows no **repeatable** >100 ms GUI-thread stall." Because testing.2 did not capture independent repeatability rounds, it cannot be reinterpreted or promoted after the fact.

Testing.3 corrected the evidence contract rather than lowering the threshold:
- the GUI-thread threshold remained 100 ms;
- ordinary navigation was measured in three independent rounds;
- an over-threshold GUI event-processing stall was "repeatable" only when it recurred in at least two rounds;
- raw isolated spikes remained recorded;
- OS scheduler delay was recorded separately and was not mislabeled as GUI-thread execution;
- background import and worker-load navigation retained strict zero >100 ms GUI-thread-stall checks;
- the receipt schema was bumped to format version 2, and promotion rejects old-schema receipts.

## Preserved testing.3 physical failure

The third physical-machine candidate, `v0.1.0-testing.3` at source head `6ade3a032bba5d0a3854f07f2f1a5550d8912d1d`, was **not** promoted.

Its full representative receipt is format version 2, gate-evaluable, and bound to executable SHA-256:

`d5b59f5f3a342221a8d31264adfe46d97ad9727206181f628daea7d54850d97c`

Receipt SHA-256:

`df872e77e97ad01b11e97e0af32db7e54a690d766784d3a280839f39cc526ea8`

Testing.3 proved the repeatability harness behaves correctly: all three ordinary-navigation rounds passed, with 340 samples total, zero >100 ms GUI event-processing stalls, zero >100 ms scheduler delays, and a maximum ordinary-navigation sample of 59.10 ms. Startup (~1422.58 ms), idle RAM (~126.13 MB), 100k bounded paging, CPU-worker navigation, cancellation (~49.81 ms), and all instrumentation checks also passed.

The remaining failure was genuine background-import responsiveness. The full 20,000-row import completed, but the GUI process recorded three event-processing stalls above 100 ms, peaking at 130.43 ms. Scheduler delay remained below 5 ms, so these were not external scheduler artifacts.

Root cause: Data Studio import was running in a `QThreadPool` thread but still inside the Qt/Python process. JSON parsing, adapter validation, fingerprinting, and per-row Python work therefore competed for the interpreter/GIL and could starve Qt despite not running on the GUI thread itself.

Testing.4 preserves the authority boundary while removing that contention:
- a dedicated hidden child mode parses, validates, fingerprints, and writes only a disposable staged SQLite database;
- the child never opens or mutates the authoritative ML Lab workspace database;
- the parent process remains the sole metadata authority;
- the parent commits staged rows using set-based SQLite rather than a Python per-row loop;
- duplicate/error semantics are preserved, including duplicates against already-authoritative rows;
- the existing strict zero >100 ms background-import GUI-stall gate remains unchanged.

The first automated `testing.4` build at source head `41ae35454d3a0ee3654bf26c2a99c855c8465a98` was not retained as a candidate. CI run `35526657222` passed Ruff, strict mypy, 135 Ubuntu core tests, Windows core, scale, standalone compilation, and the compiled application/performance smoke. The installed clean-machine workflow also reached and passed its normal prepare stage, but the repeated installed UX evidence harness failed at light/125% during temporary-workspace teardown with Windows `WinError 32` on `lab.db`. The QML/controller object graph was still alive when `TemporaryDirectory` attempted deletion. `testing.5` explicitly stops controller timers/services and destroys the QML/controller references before temporary workspace cleanup. This is a harness lifecycle correction only; no performance gate or authority boundary was weakened.

## Testing.5 physical PASS and product promotion

The final physical-machine candidate, `v0.1.0-testing.5`, is the exact compiled candidate from source head:

`6c923c528117348639c2d6823bc6fd3ff5fa863e`

Automated candidate CI:

`35527453293` — **SUCCESS**

Retained candidate artifact:

`MLLab-phase4-testing-candidate-6c923c528117348639c2d6823bc6fd3ff5fa863e`

Artifact digest:

`sha256:6fa00d5df307dad7757ef3e2c8add413fd358842667548b61a3370be49da023f`

The full physical Windows receipt is format version 2, non-smoke, gate-evaluable, and reports `ok: true`, `hard_checks_pass: true`, and `instrumentation_checks_pass: true`.

Exact tested executable:

- filename: `MLLab.exe`;
- SHA-256: `4af03345b9450a70429d68ceffce5f484bfba582043cc1d6afefe5fee2f89a9e`;
- size: 12,286,464 bytes.

Representative physical results:

- process start → interactive QML ready: **1932.10 ms** (target ≤2500 ms, hard ≤4000 ms);
- idle working set: **125.93 MB** (target ≤220 MB, hard ≤300 MB);
- 100,000 primary examples exercised with exactly **100** page examples materialized;
- three ordinary-navigation rounds completed with **353 samples**, zero >100 ms GUI stalls, zero >100 ms scheduler delays, maximum GUI event-processing sample **60.71 ms**;
- 20,000-row background import completed with **759 samples**, zero >100 ms GUI stalls, maximum GUI event-processing sample **92.85 ms**;
- CPU-heavy worker navigation completed with zero >100 ms stalls and maximum GUI sample **51.87 ms**;
- cancellation acknowledgement visible in **37.43 ms**, with terminal `CANCELLED`.

Physical receipt SHA-256:

`63fb4aeef5b9191aaa5e6bcf84191d9b394aab6cf1fdf3124f89058786cd0f72`

The fail-closed product promoter then verified the candidate manifest, full receipt schema, every hard/instrumentation check, exact executable hash/size, and exact installer/portable hashes. It copied the tested bytes unchanged and emitted:

`MLLab-v0.1.0-testing.5-testing-ready-manifest.json`

Promotion manifest SHA-256:

`02192d63c258daca77e98c79d1e0c96a789e0ccbab16fb7012b3a141b4f211d9`

Promoted byte identities:

- installer SHA-256: `d13bb84f477b439f16ae48aa1432796326e684cecfec0af75fda95233d8ca9ab`;
- portable ZIP SHA-256: `e095fe2b9e4802178cf3c39222d0bb1568ccfcf38b8431273a8d6a080d0a9bc0`;
- candidate build manifest SHA-256: `b571ae22e4796cf6fd496817b3413da201e99b036dc8e97a706ce457fcbff4f2`.

The promotion manifest records:

- `testing_ready: true`;
- `promotion_scope: PRODUCT_TESTING_ONLY`;
- `integration_gate: NO_GO`;
- `model_integration_approved: false`.

The promoted installer and portable archive are byte-for-byte the same artifacts that were tested. No release rebuild occurred.

## Performance evidence design

The complete candidate includes:

`Run-Representative-Performance.cmd`

It invokes the installed candidate's full release-performance evidence mode. The full mode:

- is Windows-only;
- exercises 100,000 indexed examples;
- verifies only a 100-row page is materialized;
- exercises a 20,000-row background import;
- samples three independent ordinary-navigation rounds for repeatable >100 ms GUI-thread stalls while preserving raw isolated spikes;
- samples navigation while a real child worker performs CPU-heavy work;
- observes cancellation acknowledgement and terminal cancellation;
- measures process-start → visible interactive window;
- measures idle working set;
- fingerprints the exact running `MLLab.exe` with SHA-256.

The receipt itself does **not** authorize testing-ready status. It records `testing_ready_authorized: false` and `integration_gate: NO_GO`.

## Fail-closed testing-ready promotion

`ml_lab.release.promotion` can promote a candidate only when all of the following are true:

- candidate manifest says `testing_candidate_windows_package`;
- candidate itself still says `testing_ready: false`;
- candidate integration gate is `NO_GO`;
- performance receipt is full, not smoke;
- performance receipt is gate-evaluable;
- every hard performance check passes;
- every instrumentation check passes;
- evidence comes from Windows;
- the 100k dataset and full background-import workloads are present;
- the receipt SHA-256/size for `MLLab.exe` exactly match the candidate build manifest;
- installer and portable archive hashes still match the candidate manifest.

Promotion copies the exact tested installer and portable bytes; it does not rebuild them.

The resulting release manifest may set `testing_ready: true` only for product testing and still records:

`INTEGRATION_GATE = NO_GO`

## Closure record

Phase 4 closure is complete:

1. testing.5 candidate CI is fully green;
2. the complete candidate was exercised on representative physical Windows hardware;
3. the full receipt was preserved under `docs/evidence/phase4/`;
4. the fail-closed promoter validated the exact candidate and receipt;
5. exact installer/portable bytes were promoted without rebuild;
6. the testing-ready manifest is preserved under `docs/evidence/phase4/`;
7. this audit is PASS.

Product testing readiness does **not** grant Frankenhomie model integration approval. `INTEGRATION_GATE = NO_GO` remains binding.
