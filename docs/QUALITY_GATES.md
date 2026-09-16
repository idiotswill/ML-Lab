# Quality Gates

Status: living gate specification

These gates define when ML Lab may advance between build phases. A successful compile is not sufficient evidence.

## Global rules

1. No phase may weaken Frankenhomie's authority boundary.
2. Existing experiment/dataset/model artifacts are migrated or remain readable; they are not silently discarded.
3. A veto metric cannot be hidden by an aggregate average.
4. Failed/cancelled worker jobs never become completed experiments.
5. TEST/REDTEAM partitions are unavailable to trainer jobs by construction.
6. Application startup must not import heavyweight optional ML frameworks.
7. UI operations that can take meaningful time run asynchronously/off the GUI thread.
8. Every released build records application version, source commit, dependency lock hash, and build manifest.

## Phase 2 exit — Product foundation

Phase 2 is complete only if all are true:

- application launches into a QML shell on supported Windows x64 in clean CI;
- first-run workspace creation works without CLI use;
- project create/open/archive flows are functional;
- SQLite schema migrations are transactional and covered by upgrade tests;
- immutable content-addressed artifact writes verify SHA-256 before registration;
- worker jobs can start, stream progress, cancel, fail, and recover after restart;
- a deliberately killed worker is shown as interrupted rather than success;
- runtime-pack/adapter manifests reject incompatible protocol versions;
- hardware diagnostics show CPU/RAM/disk and gracefully handle no NVIDIA GPU;
- logs can be opened/exported from the UI;
- keyboard navigation, DPI-aware layout primitives, dark/light mode, and designed empty/error/loading states exist;
- core Python unit/integration test suite is green;
- Windows CI creates a self-contained development standalone build and launches storage/worker/recovery/QML smokes;
- hosted-Windows startup and working-set probes remain within the engineering hard budgets.

### Phase 2 engineering budgets

Automated Phase 2 regression budgets:

- process start -> QML ready: hard gate <= 4.0 s on hosted Windows CI;
- base app working set at QML readiness: hard gate <= 300 MB on hosted Windows CI;
- potentially slow UI operations are structurally moved off the GUI thread;
- Phase 2 project/list surfaces must not eagerly materialize unbounded collections;
- background jobs must not own the GUI process.

Representative-laptop startup/RAM measurements, real Windows scaling checks, and interactive stall checks are **release gates in Phase 4**. The user requested no intermediate manual testing. Deferring those measurements preserves them for the full testing-ready build; it does not waive them.

The 100k-example paging and virtualized-table budgets begin in Phase 3 when Data Studio introduces those collection surfaces.

## Phase 3 exit — Complete lab workflow

Phase 3 is complete only if all Phase 2 gates still pass and:

- dataset import supports streaming validation with per-row error reporting;
- TRAIN/DEV/TEST/REDTEAM are physically partitioned and content hashed;
- freeze detects exact duplicates, lineage leakage, normalized duplicates, and near-duplicate candidates;
- protected-split leakage blocks experiment eligibility;
- trainer job specs contain only TRAIN and permitted DEV handles;
- experiment records are immutable after completion;
- at least deterministic abstention, bounded lexical/reference baseline, localhost-provider baseline import/run, and one trainable local baseline can be compared in one project workflow where the adapter supports them;
- metric definitions include direction and veto/non-veto classification;
- comparison UI can drill from aggregate metric -> cases -> immutable failure record;
- deterministic seeded red-team runs are reproducible;
- newly found failures can be promoted into a regression suite without altering the historical failure;
- model registry preserves multiple versions and explicit promotion history;
- bundle export includes all required hashes/manifests/known failures;
- fresh-load verification runs in a separate process and produces a separate receipt;
- the Phase A adapter targets a frozen real Frankenhomie contract and exercises the real contract validator/read-only harness;
- an end-to-end Phase A experiment can be created from the GUI without requiring code edits;
- no ML Lab path can execute a Frankenhomie game-state mutation.

### Phase 3 scale/stability gates

Synthetic scale fixtures must cover at minimum:

- 100k dataset examples;
- 10k failure/regression records;
- 1k experiment metadata records;
- 100 model registry entries;
- interruption/restart during hashing/import/evaluation;
- malformed/corrupt artifact and manifest handling.

The UI must page/filter these collections without loading them wholesale. Opening a project with 100k indexed examples must not eagerly materialize all example payloads, and scrolling virtualized tables must remain interactive while background jobs run.

## Phase 4 exit — Testing-ready Windows release

Only Phase 4 may be called **TESTING_READY**.

Required evidence:

### Packaging

- versioned x64 Windows installer EXE produced from a clean CI checkout;
- versioned portable/standalone archive produced from the same source commit;
- build manifest includes source commit, lock hash, Python/Qt/Nuitka versions, installer recipe hash, artifact SHA-256;
- installed app runs without a separately installed Python;
- installer supports per-user installation without administrative rights where feasible;
- uninstall removes application files while preserving user workspaces unless explicitly chosen otherwise;
- paths containing spaces and non-ASCII characters are exercised.

### Clean-machine smoke

A Windows CI/VM smoke pass must:

1. install the generated installer;
2. launch the installed executable;
3. create a temporary workspace;
4. create/open a sample project;
5. run built-in diagnostics/self-test;
6. execute a small worker job;
7. export and fresh-verify a sample bundle;
8. close/relaunch and confirm persisted metadata;
9. uninstall successfully.

### Reliability

- forced worker termination and app restart recover to a truthful state;
- disk-full/write-failure simulation does not register partial immutable artifacts;
- corrupt metadata/artifacts produce actionable errors rather than silent repair;
- migration backup/restore path is tested;
- logs include correlation ids for user-visible job failures;
- crash handler never uploads data automatically.

### UX

- no placeholder screens on the required first-run -> project -> data -> experiment -> compare -> failure -> package path;
- all destructive actions require clear scope and cannot silently destroy immutable history;
- long jobs expose progress, elapsed time, cancellation, and logs;
- empty/error/loading states are designed, not raw tracebacks;
- common operations are keyboard accessible;
- 100%, 125%, 150%, and 200% Windows scale factors are visually checked on the full release build;
- 1366x768 remains usable; larger displays gain density rather than giant controls;
- dark and light themes are visually checked on the full release build;
- primary tables preserve selection/filter/sort state sensibly.

### Performance

- on the representative modern Windows laptop: process start -> first interactive window target <= 2.5 s, hard gate <= 4.0 s;
- on that machine: idle base app working set target <= 220 MB, hard gate <= 300 MB;
- ordinary navigation/input shows no repeatable >100 ms GUI-thread stall;
- a 100k-example project opens without loading all example payloads;
- import/hash/benchmark work cannot freeze the UI;
- application remains navigable during a CPU-saturating worker job;
- cancellation response is visible promptly even if worker termination takes longer.

### Scientific/reproducibility

- same frozen inputs + same deterministic trainer/config/seed reproduce required deterministic metrics/hashes where algorithmically expected;
- nondeterministic backends declare nondeterminism and tolerances explicitly;
- TRAIN/DEV/TEST/REDTEAM hashes are visible in the resulting bundle;
- fresh-load verification verifies package hashes before evaluation;
- package verification cannot access trainer-only data paths implicitly.

### Frankenhomie safety

- Phase A outer deterministic commitment/no-model cases are explicitly exercised;
- accepted hidden/out-of-envelope selections = 0;
- accepted contract violations = 0;
- false commitments = 0;
- zero-model-route violations = 0;
- unsupported mechanics authority = 0;
- integration gate remains `NO_GO` unless an independent process outside ordinary training explicitly supplies integration approval evidence.

## Release naming

Pre-testing builds are developer artifacts only.

The first build offered to the user for normal PC testing will be tagged as a pre-release such as:

`v0.1.0-testing.1`

and must satisfy every Phase 4 gate above that is applicable to the included Phase A workflow.
