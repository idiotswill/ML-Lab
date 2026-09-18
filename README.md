# Frankenhomie ML Lab

Windows-first, local-first development tooling for training, benchmarking, red-teaming, comparing, and packaging bounded ML components intended to improve specific Frankenhomie decision seams.

> **Prime directive:** train against real, completed Frankenhomie contracts. Never create a parallel game authority.

## Status

**PHASE 3 — COMPLETE LAB WORKFLOW / EXIT AUDIT IN PROGRESS**

Phase 1 architecture and the Phase 2 product foundation are implemented. Phase 3's complete Lab workflow is under exit-gate audit; Phase 4 has not started, and no build is `TESTING_READY` yet.\n\nNo runtime integration is approved. `INTEGRATION_GATE = NO_GO` is the default and remains independent from model experiment/promotion status. ML Lab produces candidate models, bundles, compatibility data, metrics, failures, and verification evidence for a separate later Frankenhomie integration decision; it does not install or activate models in Frankenhomie.

Current Frankenhomie audit anchor used for the Phase 3 contract review:

- repository: `idiotswill/frankenhomie-dev`
- commit: `1ff3e2155a3c2d2b316023e9933edf3a6d53697f`
- roadmap: `frankenhomie-asterra-v0.9.0/docs/CORE_SUCCESSOR_ROADMAP.md`

Each ML Lab project must freeze its own exact Frankenhomie contract/commit snapshot. The commit above is an architecture audit anchor, not a global permanent target.

## Product shape

ML Lab is a standalone desktop workbench. The Lab owns development artifacts only: projects, datasets, experiments, failure regressions, model bundles, hashes, manifests, and local UI/application state.

Frankenhomie remains sole authority for game state, sessions, identity, visibility, entity keys, mechanics, legality, dice, world time, persistence/replay, generated-world establishment, reconciliation, and canon.

The host application is deliberately generic. Frankenhomie-specific ML problems are supplied through versioned **project adapters**. Phase A residual semantics is the first real adapter; later adapters may target bounded combat ranking, social ranking, generation/template ranking, NPC intention ranking, and situation salience once their Frankenhomie contracts are stable.

## Delivery phases

- **Phase 1 — Architecture freeze (complete):** product boundary, desktop stack, storage model, worker protocol, adapter contract, quality/performance gates, release plan.
- **Phase 2 — Product foundation (complete):** polished Windows shell, workspace/project system, immutable artifact store, SQLite metadata/migrations, subprocess job engine, diagnostics, CI, self-test, adapter/runtime-pack registries.
- **Phase 3 — Complete lab workflow (exit audit in progress):** dataset studio and frozen splits, leakage checks, experiment runner, comparison, red-team/failure library, model registry, bundle manifests, Phase A contract adapter, reference bounded-ranking trainer/runtime pack.
- **Phase 4 — Testing-ready Windows release (not started):** UX/performance pass, crash/recovery paths, fresh-load verification, Windows packaging, installer smoke tests, stress/e2e tests, downloadable `MLLab-Setup-<version>-x64.exe` and portable build artifact.

See `docs/ARCHITECTURE.md`, `docs/DELIVERY_PLAN.md`, `docs/PROJECT_ADAPTER_CONTRACT.md`, and `docs/QUALITY_GATES.md` for the implemented architecture, delivery sequence, adapter boundary, and current phase exit gates. Phase 3 is not complete until its exit gates, including exact-head test/CI evidence, are satisfied.

## Non-goals

ML Lab is not Frankenhomie runtime, not a second campaign database, not a legal-action engine, not an autonomous coordinator, not a hidden-memory system, and not an automatic model deployment mechanism.

A model can finish an experiment as a `RELEASE_CANDIDATE` while the integration gate remains `NO_GO`.
