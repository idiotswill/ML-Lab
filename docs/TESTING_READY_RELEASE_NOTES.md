# v0.1.0-testing.5 Release Notes

Status: **TESTING_READY — PRODUCT TESTING ONLY; INTEGRATION_GATE = NO_GO**

`v0.1.0-testing.1` was not promoted after a real 132.31 ms background-import stall. `testing.2` removed that specific completion-refresh hitch, while `testing.3` corrected the ordinary-navigation evidence contract to measure repeatability without lowering the 100 ms threshold. The full `testing.3` physical receipt then found three genuine >100 ms GUI event-processing stalls during the 20,000-row import, peaking at 130.43 ms, even though all three ordinary-navigation rounds were clean. `testing.4` moved JSONL parse/validation/fingerprinting into a separate hidden application child process. The child produces only a disposable staged SQLite database; the parent process remains the sole owner of authoritative Lab metadata mutation and commits staged rows with set-based SQLite. Its source, Windows, scale, compiled-standalone, and compiled-performance smokes passed, but the CI candidate was not retained because the installed visual-evidence harness hit a Windows temporary-workspace cleanup race: QML/controller references were still alive when `TemporaryDirectory` tried to remove `lab.db`. `testing.5` explicitly tears down the controller timer/services and QML object graph before temporary workspace cleanup. No performance threshold or authority rule changes.

## Scope

This is the first Windows product testing-ready release for Frankenhomie ML Lab.

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

The full representative physical-Windows gate passed on the exact testing.5 executable.

- startup: 1932.10 ms;
- idle RAM: 125.93 MB;
- ordinary navigation: three clean rounds, zero >100 ms GUI stalls;
- 20,000-row background import: zero >100 ms GUI stalls, max 92.85 ms;
- CPU-worker navigation: zero >100 ms GUI stalls;
- cancellation acknowledgement: 37.43 ms;
- receipt SHA-256: `63fb4aeef5b9191aaa5e6bcf84191d9b394aab6cf1fdf3124f89058786cd0f72`.

The fail-closed promoter verified the candidate and copied the exact tested installer/portable bytes without rebuilding them. Promotion manifest SHA-256:

`02192d63c258daca77e98c79d1e0c96a789e0ccbab16fb7012b3a141b4f211d9`

This status is product testing readiness only. It does not install or activate a model in Frankenhomie and does not change `INTEGRATION_GATE = NO_GO`.
