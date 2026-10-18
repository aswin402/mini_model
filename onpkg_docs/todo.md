# LITTLE — Engineering Task Tracker & Roadmap

---

## Completed Milestones
> Full historical details for Phases 1–21, the 2026-10-13 verified system repairs, and the v0.3 `mini-AGI` cognitive upgrades are archived in [`../CHANGELOG.md`](file://../CHANGELOG.md).

- [x] **Core Neuro-Symbolic Foundation (Phases 1–14):** SQLite `MemoryStore`, `EvidenceLedger`, 4 Invariant Gates ($I_{\text{DAG}}, I_{\text{MUTEX}}, I_{\text{SORT}}, I_{\text{TEMP}}$), `DualSpeedInfillingEngine`, `SymbolicCAS`, `ComparativeOrderReasoner`, `CfCCell` physical dynamics, and `DialogueSurfaceEngine`.
- [x] **Typed Actions, Neural Routing & Code Synthesis (Phases 15–21):** `ActionRouter`, `PreDispatchVerifier`, `ToolExecutor`, `NeuralRouter`, `ASTCodeSynthesizer`, `VerifiedPythonSandbox`, `RustCoreEngine` (`ctypes` `cdylib`), `LivingWiki` (`.okf.md`), `TurboQuantIndex`, and `HeadroomContextManager`.
- [x] **Verified System Repair (2026-10-13):** Subprocess sandbox hardening, thread-safe SQLite transactions, honest benchmark separation, and workspace isolation.
- [x] **Continual Plasticity, 5-Brake Growth, Dirty Paging & PonderNet Halting (v0.3 — 2026-10-19):**
  - [x] Dirty-only concept invalidation (`concept_mutation_epoch`, `dirty_concepts_since`), Resident slot-pinning (`slot_reuses`), and 1/2/3-gram phrase voting in `PagedConceptStore`.
  - [x] Evidence-governed `PlasticityController` with dual EWLS ($\lambda=0.97, \lambda_{\text{fast}}=0.85$), spike holdback filter, regime-shift reset, and Trunk (`0.1x`) vs. Leaf (`1.0x`) plasticity split.
  - [x] 5-Brake Speculative Growth Gate (`room & used & earning & fits & honest`) and survival-window pruning (*"Dead means unaddressed"*) in `LatentKnotInducer`.
  - [x] Load-balanced curiosity routing (`effective_study_priority`) and held-out canary verification (`evaluate_heldout_canary`) in `AutonomousSelfStudyEngine`.
  - [x] PonderNet cumulative halting probability $P(\text{halt} \le d)$, `ponder_cost`, and `halt_freeze` in `DualSpeedInfillingEngine`.
  - [x] Live telemetry snapshot (`CognitiveKernel.get_telemetry_snapshot`) and CLI `little status [--json]`.

---

## Active & Upcoming Engineering Backlog

- [x] **Rust Core FFI Zero-Copy Graph Handle (`rust/little_core`, `src/little/rust/engine.py`, `src/little/inference/invariant_gates.py`):**
  - Maintain a persistent incremental graph index (`GraphIndexRust` / `RustGraphIndex`) in Rust across turns keyed by `concept_mutation_epoch` and wired into `DeepSeekInvariantVerifier` to eliminate per-query JSON edge-list serialization and N+1 SQLite traversals.
- [x] **Continual Plasticity & Paging Curriculum Verification (`tests/unit/test_mini_agi_upgrades.py`):**
  - Multi-phase streaming curriculum test verifying `PlasticityController` regime shifts, `PagedConceptStore` slot reuse counts, and `RustGraphIndex` invariant enforcement.
- [ ] **Modularization of `LearningEngine` (`src/little/language/parser.py`) & `MemoryStore` (`src/little/memory/store.py`):**
  - Incrementally extract specialized domain collaborators (query planner, ontology seeder, relation repository) while preserving backward-compatible facades.

