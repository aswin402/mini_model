# LITTLE — Technical Specification & Implementation Reference

**Version:** 0.3  
**Status:** Active Technical Specification

---

## 1. Runtime & Toolchain Stack

- **Language & Runtime:** Python `>= 3.12` + Rust `edition = "2021"` (`rust/little_core` optional C-ABI `cdylib`).
- **Package & Build Manager:** `uv` (`uv_build` wrapped by custom PEP 517 backend `little_build_backend.py` to compile `liblittle_core.so` automatically during wheel/editable builds).
- **Persistence:** SQLite3 (`MemoryStore`) + Obsidian-compatible Open Knowledge Format Markdown files (`.okf.md` in `LivingWiki`).
- **Quality Gates:**
  - Linting & formatting: `uv run ruff check .` / `uv run ruff format .`
  - Static type checking: `uv run pyrefly check`
  - Unit & evaluation tests: `uv run pytest` (isolated via `tests/conftest.py` so live `data/little.db` and `data/knowledge_vault` are never mutated).

---

## 2. Package & Directory Layout

```text
mivi_model/
├── README.md                   # Quickstart, architecture overview, and test commands
├── CHANGELOG.md                # Release history and phase milestones (Phases 1-21 + v0.3)
├── AGENTS.md                   # AI agent instructions and build/test commands
├── onpkg.json                  # Project manifest
├── pyproject.toml              # Python package metadata and dependency declarations
├── little_build_backend.py     # PEP 517 build hook compiling rust/little_core
├── onpkg_docs/                 # Canonical engineering documentation
│   ├── INDEX.md                # Documentation index
│   ├── prd.md                  # Product requirements & core philosophy
│   ├── design.md               # Cognitive architecture & subsystem design
│   ├── implementation.md       # Technical specification & data contracts
│   ├── content.md              # CLI surface, cognitive directives & schema inventory
│   └── todo.md                 # Active engineering roadmap & verification tracker
├── research/                   # Deep-dive external architecture & paper studies
├── src/little/                 # Core Python package
│   ├── actions/                # ActionRouter, PreDispatchVerifier, ToolExecutor, Grounding
│   ├── active/                 # PlasticityController, LatentKnotInducer, SelfStudy, Cobweb
│   ├── core/                   # CognitiveKernel, ConceptKnot, CommandRegistry, RuntimePolicy
│   ├── dynamics/               # CfCCell, PhysicalSimulationEngine, CounterfactualEngine
│   ├── evaluation/             # Benchmark matrices and evaluation harnesses
│   ├── inference/              # InferenceEngine, DualSpeedInfillingEngine, InvariantGates
│   ├── knowledge/              # LivingWiki, OKFSerializer, KnowledgeImporter, SchemaRegistry
│   ├── language/               # LearningEngine, HybridPerception, LayaGatekeeper, Comparative
│   ├── memory/                 # MemoryStore, PagedConceptStore, Headroom, EvidenceLedger
│   ├── neural/                 # NeuralRouter, TurboQuantIndex, HashedLinearClassifier
│   ├── perception/             # SensoryGroundingEngine
│   ├── procedural/             # SymbolicCAS, MathStoryEngine, ASTCodeSynthesizer, Sandbox
│   └── rust/                   # RustCoreEngine ctypes wrapper
├── rust/little_core/           # Zero-dependency Rust cdylib (BFS, DAG, FWHT/TurboQuant)
├── data/schemas/               # Externalized JSON policy configurations
├── tests/                      # Unit, integration, and evaluation test suites
└── experiments/                # Reproducible benchmark experiments (001-011)
```

---

## 3. Core Data Contracts (`src/little/core/models.py` & Subsystems)

### 3.1 Belief & Update Semantics
- **`BeliefStatus`:** `SUPPORTED`, `LIKELY`, `UNCERTAIN`, `CONTRADICTED`, `REFUTED`, `AMBIGUOUS`, `UNKNOWN`.
- **`UpdateType`:** `NEW_CONCEPT`, `NEW_ENTITY`, `NEW_RELATION`, `PROPERTY_UPDATE`, `EVIDENCE_ADDITION`, `DUPLICATE`, `CONTRADICTION`, `REVISION`, `NO_OP`.
- **Non-Monotonic Truth Values:** Relations track independent positive and negative evidence weights ($w_+, w_-$) with NARS-style confidence $c = \frac{w_+ + w_-}{w_+ + w_- + k}$. Contradictions are never silently overwritten; they are recorded in the `EvidenceLedger`.

### 3.2 Memory & Paging Contracts (`src/little/memory/`)
- `MemoryStore.concept_mutation_epoch: int` increments monotonically only on mutations to `concepts` or `relations`.
- `MemoryStore.dirty_concepts_since(epoch: int) -> set[str] | None` returns exact mutated concept keys since `epoch` (or `None` if `epoch` has aged out of the bounded history window).
- `PagedStoreTelemetry` exposes `resident_count`, `lru_count`, `resident_hits`, `lru_hits`, `disk_fetches`, `evictions`, `slot_reuses`, `dirty_invalidations`, and `hit_rate`.

### 3.3 Continual Plasticity, Growth Brakes & Halting Contracts
- `PlasticityTelemetry` (`src/little/active/plasticity.py`): `scale`, `evaluations_count`, `slow_effect_size`, `fast_effect_size`, `regime_shifts`, `spikes_filtered`, `trunk_lr_mult`, `spike_detected`, `step`.
- `GrowthBrakeReport` (`src/little/active/latent_knot_inducer.py`): `room`, `used`, `earning`, `fits`, `honest`, `allowed`, `blocked_reasons`.
- `HeldOutCanaryResult` (`src/little/active/self_study.py`): `passed`, `checked_count`, `regressions`.
- `DualSpeedResult` (`src/little/inference/dual_speed.py`): `mode`, `supported`, `path`, `confidence`, `latency_ms`, `nodes_expanded`, `inspectable_trace`, `gate_verdicts`, `halt_probability`, `ponder_cost`.
- `CognitiveKernel.get_telemetry_snapshot()` (`src/little/core/kernel.py`): Aggregates `memory`, `paged_store`, `plasticity`, `growth_brakes`, and `canary` into a single JSON-serializable dictionary.
