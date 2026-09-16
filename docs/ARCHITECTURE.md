# ML Lab Architecture

Status: **Phase 1 architecture freeze candidate**

Date: 2026-09-16

## 1. Product boundary

ML Lab is a standalone Windows-first ML development application. It hosts multiple independent ML projects that attack narrowly defined Frankenhomie problems where deterministic handling is insufficient, while leaving all game authority in Frankenhomie.

The host must be useful across projects. It knows generic ML-lab concepts; adapters know Frankenhomie-specific contracts.

```text
Frankenhomie contract snapshot
          |
          v
+-----------------------------+
| Project adapter             |
| - schemas                   |
| - input/label validation    |
| - metrics/vetoes            |
| - red-team families         |
| - compatibility checks      |
+-----------------------------+
          |
          v
+----------------------------------------------------------+
| ML Lab host                                              |
| projects | data | jobs | experiments | compare | failures|
| registry | package | verify | diagnostics                |
+----------------------------------------------------------+
          |
          v
+----------------------+     +-----------------------------+
| Runtime packs        |     | External bounded baselines  |
| sparse / neural /... |     | e.g. localhost Qwen         |
+----------------------+     +-----------------------------+
```

The host never owns mechanics, legality, action commitment, hidden facts, world state, canon, or persistence for the game.

## 2. Desktop stack decision

### Chosen

- Python 3.12 development baseline.
- PySide6 / Qt 6.
- Qt Quick + QML for the application UI.
- Qt Quick Controls with an ML-Lab design system and Windows system theme integration.
- Standard Windows title/frame behavior initially; no fragile custom chrome dependency.
- `pyproject.toml` + `uv.lock` for reproducible development dependencies.
- `pytest` for Python tests and Qt Quick/QML smoke/unit tests where appropriate.
- `pyside6-deploy` / Nuitka in **standalone** mode for the installed application.
- Inno Setup x64 for the user-facing installer.

### Why this stack

PySide6 is the official Qt Python binding and keeps the ML/application language unified. QML gives us a GPU-rendered declarative UI, virtualized list/table primitives, native window integration, and a mature desktop toolkit without embedding Chromium.

The application will deliberately avoid importing heavyweight ML frameworks in the UI process. Training and benchmarking happen in child processes/runtime packs.

### Rejected for v0.1

- **Electron/React:** excellent UI ecosystem, but Chromium/Node cost works directly against startup/RAM goals and still requires a Python ML sidecar.
- **Tauri/React/Rust + Python:** smaller than Electron but creates Rust + web + Python build/runtime surfaces and IPC that do not buy enough for this product.
- **WinUI/.NET + Python workers:** strong native Windows UX, but duplicates models/protocols across C# and Python and makes plugin/runtime work more expensive.
- **Tkinter/customtkinter:** insufficient component/UI depth for the intended long-lived workbench.
- **Browser-hosted local server:** wrong lifecycle and weaker local-desktop ergonomics for a file/model-heavy lab.

## 3. UI architecture

The QML shell is presentation only. Domain rules live in Python services and immutable records.

```text
QML views
  |
  v
Qt QObject controllers / QAbstractItemModels
  |
  v
Application services
  |
  +--> project service
  +--> dataset service
  +--> experiment service
  +--> artifact service
  +--> job service
  +--> registry service
  +--> verification service
  |
  v
SQLite metadata + immutable artifact store
```

Large collections are exposed through paged/lazy `QAbstractItemModel` implementations. QML never receives tens of thousands of examples as one JavaScript object.

### Visual direction

- Segoe UI / system font; no bundled font dependency.
- Dark/light follow Windows by default.
- Dense desktop layout, not oversized touch-first controls.
- Left navigation rail + project context header + command area.
- 8px spacing grid, restrained radii, subtle elevation, high contrast.
- Status chips reserved for meaningful state: frozen, running, veto, release candidate, no-go.
- Motion limited to short state transitions; no decorative animation that competes with data.
- Tables and inspectors are first-class; this is a technical workbench, not a dashboard mockup.

Primary surfaces:

1. Home / Projects
2. Project Overview
3. Contract Snapshot
4. Data Studio
5. Training
6. Experiments
7. Compare
8. Red Team
9. Failures
10. Models / Registry
11. Package / Verify
12. Settings / Diagnostics

## 4. Process model

The UI process must remain responsive regardless of dataset size or trainer behavior.

### UI process

Owns:

- window/QML engine;
- user commands;
- metadata reads/writes;
- lightweight validation;
- job orchestration;
- paged data models;
- notifications and diagnostics.

It does **not** train models and does not synchronously run large imports, hashing passes, fuzz suites, or provider benchmarks on the GUI thread.

### Worker jobs

Long-running work executes in child processes using a versioned file/stdio protocol.

A job receives an immutable `job_spec.json`, writes append-only structured events, and finishes with `result_manifest.json` plus artifacts.

```text
MLLab.exe
   |
   +-- worker: dataset import/freeze
   +-- worker: leakage scan
   +-- worker: benchmark
   +-- worker: red-team/fuzz
   +-- runtime pack: sparse trainer
   +-- runtime pack: future neural trainer
```

A child process cannot mutate the authoritative project record directly. It writes to its job staging directory; the parent validates and commits completed artifacts to the store.

Cancellation is cooperative through a cancellation token/file with forced termination only after a grace interval. Interrupted jobs remain diagnosable and cannot masquerade as completed experiments.

## 5. Runtime packs

Heavy ML dependencies must not bloat or destabilize the desktop shell.

A **runtime pack** is a versioned executable environment implementing a generic job protocol.

Examples:

- `builtin-core` — hashing, import, freeze, comparison, packaging.
- `sparse-v1` — scikit-learn/numpy-based bounded classifiers/rankers.
- future `encoder-onnx-v1` — compact encoder training/inference tooling.
- future `torch-cuda-v1` — optional large neural dependencies.

The base app discovers only explicitly installed/trusted packs. It does not auto-import arbitrary Python files from a project directory.

A runtime-pack manifest records:

- pack id/version;
- protocol version;
- executable hash;
- supported trainer/evaluator ids;
- task capabilities;
- dependency/runtime description;
- hardware requirements;
- configuration schema.

This keeps the base application quick while allowing later ML chats to add specialized trainers without restructuring the host.

## 6. Storage model

### Application metadata

A Lab-owned SQLite database stores only ML Lab state:

- workspaces;
- projects;
- adapter registrations;
- contract snapshots;
- dataset versions;
- experiments;
- jobs;
- failure records;
- model registry entries;
- bundle/verifier records;
- UI preferences.

SQLite settings include foreign keys, WAL mode, bounded busy timeout, explicit schema migrations, and backup-before-migration.

This database is **not** a campaign/game-state store.

### Immutable artifact store

Large/reproducible content is stored outside the metadata database using SHA-256 content addressing.

```text
workspace/
  lab.db
  artifacts/
    sha256/ab/cd/<full-hash>/...
  jobs/
    <job-id>/...
  cache/
  exports/
  logs/
```

Examples of immutable artifacts:

- frozen split partitions;
- contract snapshots;
- imported benchmark result evidence;
- trained model files;
- metrics;
- failure payloads;
- bundle manifests.

No artifact is silently overwritten. A changed payload produces a new hash/version.

## 7. Dataset architecture

A logical dataset version contains physically distinct partitions:

- TRAIN
- DEV
- TEST
- REDTEAM

Trainer job specs receive only TRAIN and permitted DEV handles. TEST/REDTEAM paths are not passed to trainer processes.

Each example records generic provenance fields plus adapter-owned payload/label data:

- stable example id;
- source/provenance id;
- lineage/template group;
- creation/import metadata;
- target contract snapshot id;
- payload;
- expected label/result;
- safety/red-team tags;
- split;
- content fingerprint.

Freeze checks include:

- adapter schema validation;
- exact duplicate detection;
- normalized duplicate detection;
- lineage-group split violations;
- near-duplicate candidate detection using deterministic fingerprints/LSH;
- partition hashes;
- manifest hash;
- immutable freeze receipt.

Any protected-split leakage blocks a frozen dataset from becoming experiment-eligible until explicitly resolved.

## 8. Project adapter boundary

The host core must not contain Phase-A names or mechanics.

A project adapter declares:

- project-kind id and adapter version;
- supported Frankenhomie contract identity rules;
- dataset payload/label schemas;
- adapter validation;
- supported task types;
- metric definitions;
- veto metrics;
- red-team mutation families;
- display/inspector metadata;
- baseline definitions;
- package compatibility data;
- optional external-reference-validator hook.

See `PROJECT_ADAPTER_CONTRACT.md`.

## 9. Frankenhomie contract snapshots

A project points at a local Frankenhomie repository and a specific commit/ref.

The Lab reads contract material using read-only Git operations such as `git show <commit>:<path>` rather than modifying/checking out the user's Frankenhomie working tree.

A snapshot records:

- repository identity;
- commit/tree identity;
- adapter version;
- selected source/schema/registry file hashes;
- contract version(s);
- compatibility signature;
- capture timestamp;
- dirty-working-tree information for diagnostics only.

Changing Frankenhomie does not mutate old experiments. A new contract snapshot is created and compatibility is assessed explicitly.

v0.1 exposes no Production database connector and no command capable of dispatching a learned action into Frankenhomie.

## 10. Experiment model

An experiment is immutable after completion.

It binds:

- project;
- contract snapshot;
- dataset version/hashes;
- trainer/runtime-pack version and hash;
- configuration;
- seeds;
- environment/hardware facts;
- produced model hash;
- evaluation metrics;
- failure ids;
- timestamps/durations;
- logs;
- reproducibility command/spec.

Re-running with a changed parameter creates a new experiment.

## 11. Model registry and promotion

Model stages:

`EXPERIMENT -> SHADOW -> ADVISORY -> RELEASE_CANDIDATE -> INTEGRATION_APPROVED`

The **model stage** and **Frankenhomie integration gate** are separate fields.

ML Lab v0.1 can produce through `RELEASE_CANDIDATE`; `INTEGRATION_APPROVED` is not automatically granted by training or internal benchmarks.

Any adapter-defined veto failure blocks release-candidate promotion regardless of aggregate score.

## 12. Red-team/failure architecture

A failed case is a first-class immutable regression object, not merely a red metric.

It captures:

- input and expected behavior;
- observed behavior;
- model/experiment/dataset/contract hashes;
- attack family;
- seed/mutator version;
- veto classification;
- raw proposal where safe;
- validator result;
- status and later regression results.

Future experiments can rerun the accumulated regression set without altering the historical failure record.

## 13. Bundle/export architecture

A release-candidate bundle contains no Frankenhomie mutation/deployment mechanism.

Minimum content:

- model artifact(s);
- model hash;
- target Frankenhomie commit/contract signature;
- adapter/version;
- feature/schema versions;
- TRAIN/DEV/TEST/REDTEAM hashes;
- trainer/runtime-pack/configuration hashes;
- complete metrics and veto results;
- known failures;
- environment manifest;
- reproduction job spec;
- break-it guide;
- compatibility manifest;
- bundle SHA-256 manifest.

A fresh-load verifier runs in a new process, validates all hashes, reloads the bundle, reruns protected evaluation, and emits a separate verification receipt.

## 14. Hardware and provider policy

The base app discovers CPU, RAM, storage, and available GPU information without importing a heavyweight ML framework.

NVIDIA information is obtained through supported OS/vendor command interfaces when present (for example `nvidia-smi`); absence is normal.

Trainer packs report their own compute capabilities.

Network use is explicit per runner. Localhost provider baselines may be enabled by an adapter. There is no silent cloud fallback or telemetry.

## 15. Performance rules

- Never train in the UI process.
- Never import Torch/Transformers or comparable heavy stacks at normal app startup.
- Page/virtualize large tables.
- Stream job progress; do not poll large files at high frequency.
- Hash/import in worker processes.
- Cache derived previews by content hash; cache results are disposable.
- Prefer installed standalone deployment over self-extracting one-file runtime.
- Keep plugin discovery manifest-based rather than importing every package at startup.
- Treat UI-thread stalls over 100 ms as defects unless unavoidable and documented.

Target budgets are defined in `QUALITY_GATES.md`.

## 16. Failure philosophy

The Lab fails closed on incompatible contracts, corrupt hashes, missing protected partitions, ambiguous promotion evidence, and adapter/runtime protocol mismatch.

A broken experiment may fail. The application itself must remain restartable, preserve logs/staging artifacts, and never reinterpret a failed job as success.
