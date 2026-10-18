# LITTLE — Cognitive Architecture & System Design

**Version:** 0.3  
**Status:** Implemented Neuro-Symbolic & Continual Plasticity Architecture

---

## 1. Architectural Thesis

LITTLE is a **local-first hybrid neuro-symbolic cognitive architecture**, not a single neural network. It strictly separates:

1. **Perception & Frame Extraction** (`little.language`, `little.neural`, `little.perception`)
2. **Concept Representation** (`ConceptKnot` 6-spoke model in `little.core`)
3. **3-Tier Memory & Evidence Ledger** (`MemoryStore`, `PagedConceptStore`, `HeadroomContextManager`, `EvidenceLedger` in `little.memory`)
4. **Dual-Speed Inference & Invariant Verification** (`InferenceEngine`, `DualSpeedInfillingEngine`, `DeepSeekInvariantVerifier` in `little.inference`)
5. **Procedural & Physical World Modeling** (`little.procedural`, `little.dynamics`)
6. **Continual Plasticity, Concept Induction & Self-Study** (`PlasticityController`, `LatentKnotInducer`, `AutonomousSelfStudyEngine`, `CobwebClusterEngine` in `little.active`)
7. **Typed Action & Tool Execution** (`ActionRouter`, `PreDispatchVerifier`, `ToolExecutor` in `little.actions`)

---

## 2. High-Level Cognitive Pipeline

```text
                          ENVIRONMENT / USER INPUT
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   PERCEPTION & NEURAL ROUTER    │
                    │  - HybridPerceptionEngine       │
                    │  - NeuralRouter (S ∈ R^512)     │
                    │  - LayaGatekeeper Calibration   │
                    └────────────────┬────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │  LEARNING PATH  │      │ INFERENCE PATH  │      │  PROCEDURAL /   │
   │  4 Invariant    │      │ Fast 1-Hop vs.  │      │  ACTION PATH    │
   │  Gates:         │      │ System 2 Bi-BFS │      │  - SymbolicCAS  │
   │  I_DAG, I_MUTEX │      │ with PonderNet  │      │  - CfC Dynamics │
   │  I_SORT, I_TEMP │      │ Halting         │      │  - ToolExecutor │
   └────────┬────────┘      └────────┬────────┘      └────────┬────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │     3-TIER COGNITIVE MEMORY     │
                    │  - Tier 3: Resident Working Set │
                    │  - Tier 2: RAM LRU Cache        │
                    │  - Tier 1: SQLite + OKF Vault   │
                    │  - Headroom CCR Turn Compaction │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   HOMEOSTATIC ACTIVE LEARNING   │
                    │  - PlasticityController (EWLS)  │
                    │  - 5-Brake Growth Gate          │
                    │  - Load-Balanced Self-Study     │
                    │  - Held-Out Canary Verification │
                    └─────────────────────────────────┘
```

---

## 3. Core Subsystems & Boundaries

### 3.1 360° Concept Knot (`little.core.concept_knot`)
Every concept is hydrated as a 6-spoke `ConceptKnot`:
- **0° Episodic Grounding:** Concrete entity instances and sensory observations.
- **60° Taxonomy (`is_a`):** Hypernym/hyponym DAG hierarchy.
- **120° Mereology (`part_of`, `has_part`):** Structural part-whole decomposition.
- **180° Invariants & Attributes (`disjoint_with`, properties):** Strict ontological constraints and invariant mass.
- **240° Continuous Dynamics:** Closed-Form Continuous-Time (`CfC`) state vectors (temperature, oxidation, moisture, freshness).
- **300° Procedural Skills:** Bound executable operations and transformations.

### 3.2 3-Tier Paged Memory & Dirty-Only Invalidation (`little.memory`)
- **Tier 1 (Disk Vault):** Transactional SQLite (`MemoryStore`) + bidirectional `.okf.md` Markdown files (`LivingWiki`) indexed with 4-bit/1-bit `TurboQuantIndex`.
- **Tier 2 (RAM LRU):** Bounded `OrderedDict` of hydrated `ConceptKnot` objects.
- **Tier 3 (Resident Working Window):** Top-$K$ active concepts promoted via context-driven spreading activation (`"The text chooses"`).
- **Dirty-Only Epoch Tracking (`concept_mutation_epoch`):** `MemoryStore` records exact concept IDs/names modified per semantic epoch (`dirty_concepts_since`). `PagedConceptStore` invalidates *only* dirtied concepts on refresh, pins active Resident slots (`slot_reuses`), and casts weighted votes for 1-grams (`1.0x`), 2-grams (`1.5x`), and 3-grams (`2.0x`).

### 3.3 Deterministic Invariant Gates & Dual-Speed Inference (`little.inference`)
- **4 Verification Gates (`DeepSeekInvariantVerifier`):**
  - $I_{\text{DAG}}$: Rejects cycles in transitive taxonomic/partonomic hierarchies.
  - $I_{\text{MUTEX}}$: Rejects claims violating direct or inherited `disjoint_with` constraints.
  - $I_{\text{SORT}}$: Enforces domain/range sort compatibility across predicates.
  - $I_{\text{TEMP}}$: Enforces temporal and state consistency.
- **Dual-Speed Infilling with PonderNet Halting (`DualSpeedInfillingEngine`):**
  - **System 1 (Fast Mode):** Sub-millisecond direct edge lookup.
  - **System 2 (Thinking Mode):** Bidirectional frontier collision search with per-depth halting hazard $\lambda_d$, cumulative halting probability $P(\text{halt} \le d) = 1 - \prod_{k=1}^{d} (1 - \lambda_k)$, `ponder_cost = ponder_beta * depth`, and `halt_freeze` early stopping.

### 3.4 Continual Plasticity, 5-Brake Growth & Self-Study (`little.active`)
- **Evidence-Governed Continual Plasticity (`PlasticityController`):**
  - Dual EWLS fits ($\lambda = 0.97$ slow, $\lambda_{\text{fast}} = 0.85$ fast) compute scale-free effect size $e = -\text{slope} / \sigma$.
  - Single-step spike holdback filter (`loss - mean > 2 * sigma`) and regime-shift reset (`|e_fast| > 0.3` and `|e_fast| > 2 * |e_slow|`).
  - **Trunk vs. Leaf Split:** Upper-ontology/high-degree trunk concepts receive `0.1x` damping to prevent catastrophic forgetting; leaf concepts receive `1.0x` plasticity decaying with `rate_half = 50.0`.
- **5-Brake Speculative Growth Gate (`LatentKnotInducer.evaluate_growth_brakes`):**
  - Cobweb latent category creation requires `room & used & earning & fits & honest`.
- **Survival-Window Pruning (`LatentKnotInducer.prune_unaddressed_latent_knots`):**
  - Induced `LATENT_*` concepts that remain unaddressed (`address_hits == 0` and zero non-taxonomic outgoing relations) past `survival_window_cycles` are pruned (*"Dead means unaddressed"*).
- **Load-Balanced Curiosity & Held-Out Canary (`AutonomousSelfStudyEngine`):**
  - Penalizes concepts whose recent study share exceeds uniform share $1/N$ (`effective_study_priority`) and verifies `I_DAG` and `I_MUTEX` invariants across top-confidence relations during sleep consolidation (`evaluate_heldout_canary`).

---

## 4. Python / Rust Boundary

- **Python (`src/little/`)** owns cognitive orchestration, perception, dialogue, SQLite transactions, continual plasticity, and CLI/REPL interfaces.
- **Rust (`rust/little_core/`)** is compiled as a zero-dependency C-ABI `cdylib` (`liblittle_core.so`) via `little_build_backend.py` and loaded through `ctypes` in `src/little/rust/engine.py` (`RustCoreEngine`) with an automatic pure-Python fallback. It accelerates bidirectional BFS, Kahn's topological sort / cycle detection, and Fast Walsh-Hadamard Transform (`FWHT`) / `TurboQuant` similarity search.
