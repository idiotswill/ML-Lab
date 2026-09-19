# Phase 4 Exit Audit

Status: **PENDING — REPRESENTATIVE WINDOWS HARDWARE PERFORMANCE EVIDENCE**

Phase 4 is the first phase eligible to produce a normal user-testing build. All automated, packaging, clean-machine, reliability, UX, scientific/reproducibility, and Frankenhomie-safety implementation gates have current evidence. The remaining release gate is the full representative-machine performance capture defined in `docs/QUALITY_GATES.md`.

This document is intentionally not a PASS declaration yet.

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
| Representative physical-Windows performance | **PENDING** | full 100k gate must be run on the complete candidate on representative modern Windows hardware |

## Performance evidence design

The complete candidate includes:

`Run-Representative-Performance.cmd`

It invokes the installed candidate's full release-performance evidence mode. The full mode:

- is Windows-only;
- exercises 100,000 indexed examples;
- verifies only a 100-row page is materialized;
- exercises a 20,000-row background import;
- samples ordinary navigation for >100 ms stalls;
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

## Remaining closure sequence

1. Produce a fully green testing-candidate artifact from the current Phase 4 branch.
2. Run the bundled representative-performance capture once on the intended physical Windows machine.
3. Preserve the generated JSON receipt.
4. Run the fail-closed promotion against the exact candidate manifest/artifacts and that receipt.
5. Verify the final testing-ready release manifest and byte hashes.
6. Record the exact source head, CI run, candidate artifact digest, physical receipt hash, and promoted release hashes here and in PR #7 / issue #6.
7. Only then change this audit status to PASS and mark the build `TESTING_READY`.

Nothing in this sequence grants Frankenhomie integration approval.
