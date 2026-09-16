# Phase 2 — Product Foundation Implementation

Status: **AUTOMATED GATES PASS — PHYSICAL WINDOWS SIGN-OFF PENDING**

This branch implements the product foundation from `docs/DELIVERY_PLAN.md`. It is deliberately not the first user-testing release, and it does not grant ML Lab any Frankenhomie runtime authority.

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

## Automated evidence

The exact Phase 2 branch head `2638b704c4e684232101378a17f3a788849edbe9` passed GitHub Actions run `35139855895` on 2026-09-16.

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

Hosted-runner observations from that run:

- Windows source path: 469.69 ms process launch -> QML ready, 65.25 MB working set;
- Ubuntu source path: 301.79 ms process launch -> QML ready, 87.03 MB working set.

These are useful regression observations, not a substitute for the representative Windows laptop budget measurement required by the product gate.

### Windows standalone package

The same source head produced a standalone `MLLab.exe` (7,006,208 bytes for the executable itself, excluding the surrounding Qt payload directory). The clean CI build explicitly bundled the Qt QML runtime/plugins rather than depending on the runner's installed Qt environment.

The compiled distribution passed:

- workspace/storage/artifact smoke;
- compiled self-spawn worker completion with immutable result artifact;
- compiled hard-host-interruption recovery with `RUNNING -> INTERRUPTED`;
- compiled QML load from the deployed QML path;
- compiled startup/working-set instrumentation.

Hosted Windows packaged-build observations:

- compiled QML load: 248.31 ms;
- process launch -> QML ready: 379.88 ms;
- working set at QML readiness: 78.04 MB;
- CI observation is within the Phase 2 hard budgets of 4.0 s startup and 300 MB working set.

The hosted runner is not the representative laptop named by `QUALITY_GATES.md`, so this is supporting evidence rather than the physical performance sign-off.

## Deliberate boundaries

- no Phase-A-specific concepts in host code;
- no game-state database connector;
- no arbitrary worker command execution;
- no Torch/Transformers dependency during normal application startup;
- no trainer/protected split interfaces yet (Phase 3);
- no installer/release claim yet (Phase 4);
- no automatic Frankenhomie integration or promotion path.

## Remaining Phase 2 closure evidence

Automated CI/package gates are closed. Phase 2 remains open only for evidence that requires an actual supported Windows desktop rather than a headless hosted runner:

1. visually/runtime-check the real application on Windows x64 at normal and high DPI, including 1366x768 usability, keyboard focus, dark/light/system themes and loading/empty/error states;
2. run the packaged performance probe on the representative modern Windows laptop and record process-start -> first-interactive-window plus idle working set against the 2.5 s / 220 MB targets and 4.0 s / 300 MB hard gates;
3. exercise ordinary project/navigation/diagnostics interactions on that machine and confirm no user-input path exhibits a >100 ms GUI-thread stall.

The 100k-example paging and virtualized-table scale requirements belong to Phase 3, because `docs/DELIVERY_PLAN.md` introduces Data Studio and those collection surfaces there. They remain mandatory Phase 3 exit gates and are not waived.

Until the three physical checks above are recorded, this PR remains draft and Phase 3 does not start. Integration remains `NO_GO`.