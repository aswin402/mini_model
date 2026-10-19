# Changelog

All notable architectural milestones, system repairs, and cognitive upgrades for **LITTLE (MIVI Cognitive Architecture)** are documented in this file.

---

## [0.3.0] — 2026-10-19 — Continual Plasticity, 5-Brake Growth, Dirty Paging & PonderNet Halting

### Added
- **Evidence-Governed Continual Plasticity (`src/little/active/plasticity.py`):**
  - Implemented `PlasticityController`, `EWLSFit`, and `PlasticityTelemetry` with dual exponentially weighted least-squares fits ($\lambda = 0.97$ slow, $\lambda_{\text{fast}} = 0.85$ fast) and scale-free effect size $e = -\text{slope} / \sigma$.
  - Added single-step spike holdback filter (`loss - mean > 2 * sigma`) and regime-shift reset (`|e_fast| > 0.3` and `|e_fast| > 2 * |e_slow|`).
  - Added Trunk (`0.1x` damping on high-degree/root concepts) vs. Leaf (`1.0x` decaying with `rate_half = 50.0`) plasticity split, wired into `ContinuousStreamLearner` and `CognitiveKernel`.
- **Dirty-Only Concept Invalidation, Resident Slot Pinning & Multi-Gram Phrase Voting (`src/little/memory/`):**
  - Added monotonic `concept_mutation_epoch` and per-epoch dirty concept tracking (`dirty_concepts_since`) to `MemoryStore` so episodic/ledger writes no longer flush the concept cache.
  - Updated `PagedConceptStore` to invalidate *only* dirtied concepts on refresh, pin active Resident slots (`slot_reuses`), and cast weighted activation votes for 1-grams (`1.0x`), 2-grams (`1.5x`), and 3-grams (`2.0x`).
- **5-Brake Speculative Growth Gate & Survival-Window Pruning (`src/little/active/latent_knot_inducer.py`):**
  - Added `GrowthBrakeReport` and `evaluate_growth_brakes()` (`room & used & earning & fits & honest`) gating Cobweb latent category creation.
  - Added `touch_latent_concept()` and `prune_unaddressed_latent_knots()` (*"Dead means unaddressed"*) to prune unreferenced `LATENT_*` concepts after `survival_window_cycles`.
- **Load-Balanced Curiosity Routing & Held-Out Canary Verification (`src/little/active/self_study.py`):**
  - Added running-share load-balancing (`effective_study_priority`, `_record_study_visit`) to prevent high-entropy stub concepts from starving the knowledge graph during dreaming.
  - Added `HeldOutCanaryResult` and `evaluate_heldout_canary()` to verify `I_DAG` and `I_MUTEX` invariants during `consolidate_sleep_cycle()`.
- **PonderNet Adaptive Halting (`src/little/inference/dual_speed.py`):**
  - Extended `DualSpeedInfillingEngine` and `DualSpeedResult` with cumulative halting probability $P(\text{halt} \le d) = 1 - \prod_{k=1}^{d} (1 - \lambda_k)$, `ponder_cost = ponder_beta * depth`, and `halt_freeze`.
- **Live Cognitive Telemetry Dashboard (`src/little/core/kernel.py`, `src/little/main.py`, `src/little/core/cognitive_command.py`):**
  - Added `CognitiveKernel.get_telemetry_snapshot()`, CLI `little status [--json]`, and interactive REPL command `"show cognitive status"`.
- **360° Fractal Spider-Web Mini Learning Model (`src/little/mlu/unit.py`, `src/little/mlu/spiderweb_net.py`):**
  - Implemented `MiniLearningUnit` (`MLU`) combining LFM-2.5 Gated Short-Convolutions + Liquid `CfC` state dynamics, JEPA latent anchor alignment, Laya epistemic entropy, and 6 radial 360° `SpokeHead` projections (`taxonomy`, `mereology`, `attributes`, `dynamics`, `procedural`, `episodic`).
  - Implemented `SpiderWebMLM` organizing `Central Core (CORE_ROOT) -> 360° Sub-Core Hubs -> 360° Spoke Leaf MLUs` with 5-Brake Fractal Mitosis (promoting heavily used multi-spoke Leaf MLUs into Sub-Core Hubs), DeepSeek-R1 style invariant-gated online training, and CLI/REPL directives (`little mlu train`, `little mlu ask`, `little mlu status`).
- **Persistent Zero-Copy Rust Graph Handle (`rust/little_core/src/lib.rs`, `src/little/rust/engine.py`, `src/little/inference/invariant_gates.py`):**
  - Implemented `GraphIndexRust` (`little_core_graph_*` C-ABI) and `RustGraphIndex` keyed by `MemoryStore.concept_mutation_epoch`, wired into `DeepSeekInvariantVerifier` to eliminate per-query N+1 SQLite/JSON serialization overhead during $I_{\text{DAG}}$ and $I_{\text{MUTEX}}$ verification.
- **Documentation Consolidation (`onpkg_docs/`, `README.md`, `CHANGELOG.md`):**
  - Consolidated root Markdown design/spec files into `onpkg_docs/` (`INDEX.md`, `prd.md`, `design.md`, `implementation.md`, `content.md`, `todo.md`) and removed duplicate root `.md` files.

---

## [0.2.1] — 2026-10-13 — Verified System Repair & Hardening

### Fixed
- **Subprocess Sandbox Isolation (`src/little/procedural/restricted_execution.py`, `code_sandbox.py`):**
  - Enforced AST whitelist inspection, subprocess timeouts, and RLIMIT memory bounds for synthesized Python skills.
- **SQLite Transaction & Workspace Hygiene (`src/little/memory/store.py`, `tests/conftest.py`):**
  - Isolated all test suites and benchmarks from live runtime state (`data/little.db`, `data/knowledge_vault`).
- **Memory & Headroom Contracts (`src/little/memory/headroom.py`, `paged_store.py`):**
  - Repaired reversible context compression (CCR) decompression fidelity and semantic revision tracking.

---

## [0.2.0] — Phases 15–21 — Typed Actions, Neural Routing, Code Synthesis, Rust Core & Living Wiki

### Added
- **Phase 21 — OKF Living Graph Wiki, TurboQuant & Headroom (`src/little/knowledge/`, `src/little/neural/turbo_quant.py`, `src/little/memory/headroom.py`):**
  - Bidirectional Obsidian Knowledge Format (`.okf.md`) synchronization (`OKFSerializer`, `LivingWiki`), Subsampled Randomized Hadamard Transform (`FWHT`) 4-bit/1-bit `TurboQuantIndex`, `HeadroomContextManager` reversible context compression, and `KnowledgeCrawler`.
- **Phase 20 — Empirical Evaluation Matrices (`src/little/evaluation/`, `experiments/010_real_world_sota_benchmark/`):**
  - Multi-modality evaluation harnesses across dialogue, exact math CAS, AST code synthesis, typed tool dispatch, Open-World Assumption (`UNKNOWN`), and continual learning.
- **Phase 19 — Rust Acceleration Core (`rust/little_core`, `src/little/rust/engine.py`):**
  - Zero-dependency C-ABI `cdylib` (`liblittle_core.so`) with `ctypes` wrapper (`RustCoreEngine`) and automatic pure-Python fallback for bidirectional BFS, Kahn's topological sort / cycle detection, and FWHT/TurboQuant.
- **Phase 18 — Programming & AST Code Synthesis (`src/little/procedural/program_synthesizer.py`, `code_sandbox.py`, `code_repair.py`):**
  - Deterministic Python AST synthesis, process-isolated verification, and traceback-guided AST auto-repair.
- **Phase 16–17 — Neural Latent Workspace, Router & Semantic Frame Extractor (`src/little/neural/`):**
  - 512-dim hashed N-gram latent workspace, calibrated 6-pathway `NeuralRouter`, and FrameNet-style `NeuralFrameExtractor`.
- **Phase 15 — Typed Action & Tool Core (`src/little/actions/`):**
  - `PreDispatchVerifier` ($I_{\text{tool\_exists}}, I_{\text{param\_required}}, I_{\text{sort\_type}}, I_{\text{security\_sandbox}}$), `ToolExecutor` with SHA-256 execution provenance, `ActionRouter`, and `ToolGroundingEngine`.

---

## [0.1.0] — Phases 1–14 — Core Neuro-Symbolic Foundation

### Added
- **Phases 12–14 — Conversational Surface, Discourse Anaphora & Cross-Domain Generalization (`src/little/language/`):**
  - Externalized JSON policy schemas (`data/schemas/`), `DialogueSurfaceEngine`, multi-turn pronoun coreference (`DialogueContext`), and proportional word-problem solving.
- **Phases 8–11 — Dual-Speed Reasoning, Comparative Posets, Symbolic CAS & Physical Dynamics (`src/little/inference/`, `src/little/procedural/`, `src/little/dynamics/`):**
  - `DualSpeedInfillingEngine`, `DeepSeekInvariantVerifier` ($I_{\text{DAG}}, I_{\text{MUTEX}}, I_{\text{SORT}}, I_{\text{TEMP}}$), `ComparativeOrderReasoner`, `SymbolicCAS`, `MathStoryEngine`, `CfCCell` / `PhysicalSimulationEngine`, and `AutonomousSelfStudyEngine`.
- **Phases 1–7 — Persistent Memory, Evidence Ledger & 360° Concept Knots (`src/little/memory/`, `src/little/core/`):**
  - SQLite `MemoryStore`, append-only `EvidenceLedger` with NARS truth-value confidence, `ConceptKnot` 6-spoke model, `ActiveInquisitor`, and `little` CLI entrypoint.
