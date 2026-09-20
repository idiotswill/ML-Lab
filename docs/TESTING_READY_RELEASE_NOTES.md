# v0.1.0-testing.4 Release Notes

Status: **candidate notes — testing-ready promotion pending representative physical-Windows performance evidence**

`v0.1.0-testing.1` was not promoted after a real 132.31 ms background-import stall. `testing.2` removed that specific completion-refresh hitch, while `testing.3` corrected the ordinary-navigation evidence contract to measure repeatability without lowering the 100 ms threshold. The full `testing.3` physical receipt then found three genuine >100 ms GUI event-processing stalls during the 20,000-row import, peaking at 130.43 ms, even though all three ordinary-navigation rounds were clean. `testing.4` therefore moves JSONL parse/validation/fingerprinting into a separate hidden application child process. The child produces only a disposable staged SQLite database; the parent process remains the sole owner of authoritative Lab metadata mutation and commits staged rows with set-based SQLite.

## Scope

This is the first complete Windows testing candidate for Frankenhomie ML Lab.

It provides the standalone development workflow for:

- projects and frozen contract snapshots;
- dataset import and immutable TRAIN/DEV/TEST/REDTEAM partitions;
- leakage detection;
- local training jobs;
- protected evaluation;
- deterministic/lexical/local-provider comparison;
- red-team and regression evidence;
- model registry and promotion history;
- release-candidate bundle export;
- separate-process fresh verification;
- diagnostics, logs and local crash evidence.

The first real adapter targets Frankenhomie's completed bounded Phase A residual-semantic seam.

## Authority / safety

The Lab is not Frankenhomie runtime.

Frankenhomie remains authoritative for game state, identity, visibility, legality, mechanics, dice, scheduling, persistence, generated-world establishment, reconciliation and canon.

The Lab does not install or activate models in Frankenhomie.

`INTEGRATION_GATE = NO_GO` remains binding even for a product build that becomes `TESTING_READY`.

## Phase A safety posture

Phase A protected evaluation preserves the pinned deterministic preflight before any model/provider call, validates proposals against the real pinned residual semantic contract, and records veto metrics including:

- false commitments;
- contract failures;
- hidden/out-of-envelope failures;
- zero-model-route violations;
- unsupported mechanics authority.

## Known limitations

- Phase A is the only real Frankenhomie ML adapter included in the first testing release.
- Future Phase C/D/E/F/G labs are intentionally deferred until their Frankenhomie contracts stabilize.
- Heavyweight CUDA/Torch runtime packs and model-hub management are intentionally deferred.
- Remote/cloud/distributed training is not part of this local-first release.
- The local provider baseline is explicitly nondeterministic and is governed by exact veto checks plus declared reporting policy for non-veto variation.
- Product `TESTING_READY` status is independent from model integration approval.
- The Phase H coordinator remains outside ML authority.

## Performance gate

The candidate contains the full representative-machine evidence harness, but the release must not be called `TESTING_READY` until that full gate passes on representative modern physical Windows hardware and the exact tested executable hash matches the candidate build manifest.
