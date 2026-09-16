# Phase 2 — Product Foundation Implementation

Status: **COMPLETE — AUTOMATED FOUNDATION GATES PASS; USER-SIDE PHYSICAL CHECKS DEFERRED TO PHASE 4**

This branch implements the product foundation from `docs/DELIVERY_PLAN.md`. It is deliberately not the first user-testing release, and it does not grant ML Lab any Frankenhomie runtime authority.

The user explicitly requested that they only test the full testing-ready release. Therefore the representative-PC visual/DPI/startup/working-set checks that require their real Windows machine are deferred to Phase 4, where they are still mandatory before `TESTING_READY`. They are not waived.

## Implemented

- src-layout Python application and Qt Quick/QML desktop bootstrap;
- first-run workspace create/open flow;
- compact project/job/diagnostics/settings shell;
- system/dark/light theme modes;
- project create/open/archive lifecycle;
- manifest-only adapter/runtime catalog with protocol rejection;
- Lab-only SQLite metadata store with WAL, foreign keys and sequential transactional migrations;
- pre-migration backup path plus sequential upgrade/rollback coverage;
- immutable SHA-256 artifact store with atomic write and post-copy verification;
- child-process job protocol with structured progress, cancellation, timed termination fallback and restart reconciliation;
- worker event/result finalization into the immutable artifact store;
- explicit FAILED/CANCELLED/INTERRUPTED terminal states that cannot become successful result artifacts;
- compiled-application self-spawn/worker path distinct from Python development execution;
- hard-host-interruption recovery smoke that starts a real worker, exits the host without cleanup, reopens the workspace and requires `RUNNING -> INTERRUPTED`;
- CPU/RAM/disk/NVIDIA diagnostics without a heavyweight ML import;
- hardware probing off the Qt GUI thread with loading/error state and workspace-generation protection;
- rotating application logs and explicit diagnostics bundle export;
- non-GUI storage, worker, restart-recovery and performance smoke commands;
- headless QML-load smoke command with load timing;
- explicit keyboard focus/navigation treatment for the primary Phase 2 workflow;
- Windows/Linux test CI plus Windows standalone deployment and compiled smoke tests;
- standalone Nuitka build that explicitly includes the Qt QML runtime/plugins and fails if `MLLab.exe` is not produced;
- unit/integration coverage for workspace, migrations, artifact corruption, extension compatibility and worker lifecycle.

## Final automated evidence

Final Phase 2 branch head `4d14411e8ac3a1637d137dfaf06a1e1c5098fcf8` passed GitHub Actions run `35140608780` on 2026-09-16.

### Core matrix

Windows and Ubuntu both passed:

- Ruff with no rule relaxation;
- strict mypy: 25 source files, no issues;
- pytest: 19 tests passed;
- storage/artifact smoke;
- real child-worker completion smoke;
- hard restart recovery smoke: `RUNNING -> INTERRUPTED`, one reconciliation, no success artifact;
- headless QML load smoke;
- startup/working-set instrumentation smoke.

### Windows standalone package

The clean CI build produced a standalone `MLLab.exe` and explicitly bundled the Qt QML runtime/plugins rather than depending on the runner's installed Qt environment.

The compiled distribution passed:

- workspace/storage/artifact smoke;
- compiled self-spawn worker completion with immutable result artifact;
- compiled hard-host-interruption recovery with `RUNNING -> INTERRUPTED`;
- compiled QML load from the deployed QML path;
- compiled startup/working-set instrumentation.

Final hosted Windows packaged-build observations:

- compiled QML load: 265.27 ms;
- process launch -> QML ready: 384.31 ms;
- working set at QML readiness: 77.8 MB;
- CI observation is within the engineering hard budgets of 4.0 s startup and 300 MB working set.

The exact smoke-tested distribution was retained as Actions artifact `10465043272`, named `MLLab-phase2-windows-standalone-93507da1b8d0d74c738f4d0fd64f65a183c96050`, ZIP SHA-256 `1bfbf0c6ad0f56728985142c9f5a8168ef6bef47c2489f5added903052674435`.

## Deliberate boundaries

- no Phase-A-specific concepts in host code;
- no game-state database connector;
- no arbitrary worker command execution;
- no Torch/Transformers dependency during normal application startup;
- no trainer/protected split interfaces yet (Phase 3);
- no installer/release claim yet (Phase 4);
- no automatic Frankenhomie integration or promotion path.

## Deferred physical evidence

At the user's request, no intermediate build is being handed off for manual testing. The following evidence moves intact to Phase 4 and must be recorded against the full testing-ready build before release:

1. visual/runtime check on supported Windows x64 at 100%, 125%, 150%, and 200% scaling, including 1366x768 usability, keyboard focus, dark/light/system themes and designed loading/empty/error states;
2. packaged performance probe on the representative Windows laptop, including process-start -> first-interactive-window and idle working set against the engineering budgets;
3. ordinary interaction check confirming no repeatable >100 ms GUI-thread stall.

The 100k-example paging and virtualized-table scale requirements belong to Phase 3, because Data Studio introduces those collection surfaces there. They remain mandatory Phase 3 exit gates and are not waived.

Phase 3 may proceed on the automated Phase 2 foundation. Integration remains `NO_GO` throughout ordinary lab development.