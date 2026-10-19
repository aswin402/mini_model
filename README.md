# LITTLE — Local-First Neuro-Symbolic Cognitive Architecture

**LITTLE** (`src/little`) is a CPU-first, inspectable hybrid cognitive architecture that separates **factual knowledge** (explicit relational concept graphs in SQLite and Obsidian-compatible Markdown vaults), **deductive and non-monotonic reasoning** (dual-speed NARS-inspired inference and invariant gates), **procedural computation** (exact rational CAS, story-state tracking, and process-isolated Python skill execution), and **continual learning** (entropy-guided active questioning, Cobweb concept formation, and background sleep/self-study consolidation).

## Requirements & Tooling

- **Python:** `>= 3.12`
- **Package & Environment Manager:** [`uv`](https://docs.astral.sh/uv/)
- **Native Core (Optional):** Rust `cargo` (`rust/little_core` compiled as `cdylib` and loaded via `ctypes` with automatic pure-Python fallback)
- **Linting & Type Checking:** `ruff`, `pyrefly`

## Quickstart

```bash
# Synchronize environment and build the package
uv sync

# Run interactive cognitive REPL (using isolated temporary workspace or custom DB)
uv run little --db /tmp/little_demo.db interact

# Teach a fact or rule from the CLI
uv run little --db /tmp/little_demo.db learn "A robin is a bird."

# Query the cognitive engine with reasoning trace
uv run little --db /tmp/little_demo.db ask "Is a robin an animal?"

# Inspect live cognitive telemetry (paging, plasticity, 5-brake growth gate, canary)
uv run little --db /tmp/little_demo.db status

# Train, query, compare, simulate causal/counterfactual waves, and benchmark the 360° Spider-Web MLM
uv run little --db /tmp/little_demo.db mlm-train --bootstrap
uv run little --db /tmp/little_demo.db mlm-ask "What is an orca?"
uv run little --db /tmp/little_demo.db mlm-compare "orca" "dolphin"
uv run little --db /tmp/little_demo.db mlm-causal "drought"
uv run little --db /tmp/little_demo.db mlm-counterfactual "drought" "crop_failure"
uv run little --db /tmp/little_demo.db mlm-bridge "orca" "warm_blooded"
uv run little --db /tmp/little_demo.db mlm-clusters
uv run little --db /tmp/little_demo.db mlm-benchmark
```

## Architecture Overview

- **`little.core`** — [`CognitiveKernel`](src/little/core/kernel.py) orchestrator (including `get_telemetry_snapshot()`), [`ConceptKnot`](src/little/core/concept_knot.py) 6-spoke concept representation, [`CognitiveCommandRegistry`](src/little/core/cognitive_command.py), [`RuntimePaths`](src/little/core/runtime_paths.py), and [`RuntimePolicy`](src/little/core/runtime_policy.py).
- **`little.mlu`** — [`SpiderWebMLM`](src/little/mlu/spiderweb_net.py) and [`MiniLearningUnit`](src/little/mlu/unit.py): Rust-native 360° Fractal Spider-Web Mini Learning Model (`Core Hub -> 6 Sub-Core Hubs -> Leaf MLUs` with 5-Brake Fractal Mitosis, 64-D harmonic phase routing, Shannon character-entropy salience, JEPA latent surprise, LFM continuous-time state, DeepSeek-R1 `I_DAG`/`I_MUTEX` invariant gates, comparative wave intersection, counterfactual intervention, shortest relational bridge search, emergent cluster discovery, and non-monotonic belief revision).
- **`little.memory`** — [`MemoryStore`](src/little/memory/store.py) SQLite persistence (`concepts`, `relations`, `experiences`, `procedures`, `evidence_records`, `commit_decisions`, `proof_traces`, `mlu_spiderweb_nodes`, `mlu_arena_snapshots`) with monotonic `concept_mutation_epoch` dirty-concept tracking, [`EvidenceLedger`](src/little/memory/ledger.py), [`PagedConceptStore`](src/little/memory/paged_store.py) (3-tier Disk $\to$ RAM LRU $\to$ Resident working window with dirty-only invalidation, Resident slot-pinning, and 1/2/3-gram phrase voting), and [`HeadroomContextManager`](src/little/memory/headroom.py).
- **`little.language`** — [`LearningEngine`](src/little/language/parser.py) & [`HybridPerceptionEngine`](src/little/language/hybrid_perception.py) (rule-based parsing + lightweight hashed N-gram perception), [`ComparativeOrderReasoner`](src/little/language/comparative.py), and [`LayaGatekeeper`](src/little/language/laya_gatekeeper.py) epistemic calibration.
- **`little.inference`** — [`InferenceEngine`](src/little/inference/engine.py), [`DualSpeedInfillingEngine`](src/little/inference/dual_speed.py) (Fast 1-hop vs. System 2 bidirectional BFS with PonderNet cumulative halting probability $P(\text{halt} \le d)$, `ponder_cost`, and `halt_freeze`), [`DeepSeekInvariantVerifier`](src/little/inference/invariant_gates.py) (`I_DAG`, `I_MUTEX`, `I_SORT`, `I_TEMP`), and [`AnalogicalReasoningEngine`](src/little/inference/analogy.py).
- **`little.procedural`** — [`SymbolicCAS`](src/little/procedural/math_cas.py), [`MathStoryEngine`](src/little/procedural/math_story.py), [`ASTCodeSynthesizer`](src/little/procedural/program_synthesizer.py), [`VerifiedPythonSandbox`](src/little/procedural/code_sandbox.py), and [`restricted_execution`](src/little/procedural/restricted_execution.py).
- **`little.dynamics`** — [`CfCCell`](src/little/dynamics/cfc.py), [`PhysicalSimulationEngine`](src/little/dynamics/cfc_ode.py), and [`CounterfactualEngine`](src/little/dynamics/counterfactual.py).
- **`little.active`** — [`ActiveInquisitor`](src/little/active/inquisitor.py), [`PlasticityController`](src/little/active/plasticity.py) (dual EWLS $\lambda=0.97/\lambda_{\text{fast}}=0.85$ trend estimator, spike holdback filter, regime-shift reset, and Trunk `0.1x` vs. Leaf `1.0x` continual plasticity split), [`LatentKnotInducer`](src/little/active/latent_knot_inducer.py) (5-brake speculative growth gate `room & used & earning & fits & honest` and survival-window pruning), [`AutonomousSelfStudyEngine`](src/little/active/self_study.py) (load-balanced curiosity routing and held-out canary verification), [`CobwebClusterEngine`](src/little/active/cobweb_engine.py), and [`ContinuousStreamLearner`](src/little/active/stream_learner.py).
- **`little.knowledge`** — [`OKFSerializer`](src/little/knowledge/okf.py), [`LivingWiki`](src/little/knowledge/living_wiki.py), and [`SchemaRegistry`](src/little/knowledge/registry.py).
- **`little.actions`** — [`ActionRouter`](src/little/actions/action_router.py), [`PreDispatchVerifier`](src/little/actions/pre_dispatch_verifier.py), and [`ToolExecutor`](src/little/actions/tool_executor.py).
- **`little.rust`** — [`RustCoreEngine`](src/little/rust/engine.py) and [`RustSpiderWebArena`](src/little/rust/mlu_arena.py) (`ctypes` C-ABI bridges to `rust/little_core` for native 360° Spider-Web MLU arena waves, graph traversal, invariant verification, and FWHT/TurboQuant operations, with automatic pure-Python fallback).

## Running Tests

Tests run against isolated temporary SQLite databases and temporary workspace directories (via [`tests/conftest.py`](tests/conftest.py)) so live runtime data in `data/` is never mutated:

```bash
# Run targeted unit and repair verification suites
uv run pytest tests/unit/ tests/evaluation/ -q

# Run 360° Fractal Spider-Web MLM unit & benchmark suite (Phases 1–28)
uv run pytest tests/unit/test_spiderweb_mlm.py -q

# Run mini-AGI cognitive upgrade verification tests
uv run pytest tests/unit/test_mini_agi_upgrades.py -q

# Run end-to-end integration evaluation
uv run pytest tests/integration/test_real_world_comprehensive_evaluation.py -q
```

## Documentation

All project documentation is organized under [`onpkg_docs/`](onpkg_docs/INDEX.md) and root [`CHANGELOG.md`](CHANGELOG.md):

- **[Documentation Index (`onpkg_docs/INDEX.md`)](onpkg_docs/INDEX.md)** — Central entry point for project docs
- **[Product Requirements & Core Philosophy (`onpkg_docs/prd.md`)](onpkg_docs/prd.md)** — Vision, Open-World Assumption (`UNKNOWN`), and design principles
- **[Cognitive Architecture & System Design (`onpkg_docs/design.md`)](onpkg_docs/design.md)** — 6-spoke `ConceptKnot`, 3-tier paged memory, 4 invariant gates, continual plasticity, and Python/Rust boundary
- **[Technical Specification & Implementation (`onpkg_docs/implementation.md`)](onpkg_docs/implementation.md)** — Runtime stack, directory layout, and data/telemetry contracts
- **[CLI Surface, Directives & Policy Schemas (`onpkg_docs/content.md`)](onpkg_docs/content.md)** — Complete `little` CLI subcommands, interactive directives, and `data/schemas/*.json` inventory
- **[Engineering Task Tracker (`onpkg_docs/todo.md`)](onpkg_docs/todo.md)** — Completed milestones and active engineering backlog
- **[Changelog (`CHANGELOG.md`)](CHANGELOG.md)** — Chronological release history (v0.1.0 through v0.3.0)
- **[Research Compendium (`research/README.md`)](research/README.md)** — Deep-dive architectural studies

## Scope & Honest Capability Boundaries

- **Perception & Neural Layers:** `little.neural` implements lightweight, CPU-native hashed character/word N-gram linear classifiers and quantized vector indexing (`TurboQuantIndex`), not pretrained billion-parameter Transformer weights.
- **Procedural Synthesis:** `ASTCodeSynthesizer` and `VerifiedPythonSandbox` synthesize and verify bounded algorithmic skills in an AST-checked, resource-limited subprocess worker (`restricted_execution.py`); they do not provide OS-container or VM-level sandboxing for untrusted adversarial host code.
