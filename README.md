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
```

## Architecture Overview

- **`little.core`** — [`CognitiveKernel`](src/little/core/kernel.py) orchestrator, [`ConceptKnot`](src/little/core/concept_knot.py) 6-spoke concept representation, [`RuntimePaths`](src/little/core/runtime_paths.py), and [`RuntimePolicy`](src/little/core/runtime_policy.py).
- **`little.memory`** — [`MemoryStore`](src/little/memory/store.py) SQLite persistence (`concepts`, `relations`, `episodes`, `procedures`, `evidence_ledger`, `working_turns`), [`EvidenceLedger`](src/little/memory/ledger.py), [`PagedConceptStore`](src/little/memory/paged_store.py), and [`HeadroomContextManager`](src/little/memory/headroom.py).
- **`little.language`** — [`LearningEngine`](src/little/language/parser.py) & [`HybridPerceptionEngine`](src/little/language/hybrid_perception.py) (rule-based parsing + lightweight hashed N-gram perception), [`ComparativeOrderReasoner`](src/little/language/comparative.py), and [`LayaGatekeeper`](src/little/language/laya_gatekeeper.py) epistemic calibration.
- **`little.inference`** — [`InferenceEngine`](src/little/inference/engine.py), [`DualSpeedReasoner`](src/little/inference/dual_speed.py), [`DeepSeekInvariantVerifier`](src/little/inference/invariant_gates.py) (`I_DAG`, `I_MUTEX`, `I_SORT`), and [`AnalogicalEngine`](src/little/inference/analogy.py).
- **`little.procedural`** — [`SymbolicCAS`](src/little/procedural/math_cas.py), [`MathStorySolver`](src/little/procedural/math_story.py), [`ProgramSynthesizer`](src/little/procedural/program_synthesizer.py), [`CodeSandbox`](src/little/procedural/code_sandbox.py), and [`restricted_execution`](src/little/procedural/restricted_execution.py).
- **`little.dynamics`** — [`ContinuousDynamicsEngine`](src/little/dynamics/cfc.py), [`CfCContinuousODE`](src/little/dynamics/cfc_ode.py), and [`CounterfactualDynamicsEngine`](src/little/dynamics/counterfactual.py).
- **`little.active`** — [`ActiveInquisitor`](src/little/active/inquisitor.py), [`AutonomousSelfStudyEngine`](src/little/active/self_study.py), [`IncrementalCobwebEngine`](src/little/active/cobweb_engine.py), and [`ContinuousStreamLearner`](src/little/active/stream_learner.py).
- **`little.knowledge`** — [`ObsidianKnowledgeFormat`](src/little/knowledge/okf.py), [`LivingWiki`](src/little/knowledge/living_wiki.py), and [`SchemaRegistry`](src/little/knowledge/registry.py).
- **`little.actions`** — [`ActionRouter`](src/little/actions/action_router.py), [`PreDispatchVerifier`](src/little/actions/pre_dispatch_verifier.py), and [`ToolExecutor`](src/little/actions/tool_executor.py).

## Running Tests

Tests run against isolated temporary SQLite databases and temporary workspace directories (via [`tests/conftest.py`](tests/conftest.py)) so live runtime data in `data/` is never mutated:

```bash
# Run targeted unit and repair verification suites
uv run pytest tests/unit/ tests/evaluation/ -q

# Run end-to-end integration evaluation
uv run pytest tests/integration/test_real_world_comprehensive_evaluation.py -q
```

## Scope & Honest Capability Boundaries

- **Perception & Neural Layers:** `little.neural` implements lightweight, CPU-native hashed character/word N-gram linear classifiers and quantized vector indexing (`TurboQuantIndex`), not pretrained billion-parameter Transformer weights.
- **Procedural Synthesis:** `ProgramSynthesizer` and `CodeSandbox` synthesize and verify bounded algorithmic skills in an AST-checked, resource-limited subprocess worker (`restricted_execution.py`); they do not provide OS-container or VM-level sandboxing for untrusted adversarial host code.
