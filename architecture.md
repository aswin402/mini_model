# LITTLE — Architecture

**Version:** 0.2  
**Status:** Implemented Neuro-Symbolic & Continual Plasticity Architecture

---

## 1. Architectural thesis

LITTLE is a **cognitive architecture / learning system**, not a single neural network.

The architecture separates:

1. perception;
2. representation;
3. concept formation;
4. episodic memory;
5. semantic memory;
6. procedural memory;
7. world modeling;
8. inference;
9. learning;
10. feedback.

The key architectural principle is:

> **The computational core should provide mechanisms for learning and reasoning; acquired domain knowledge should live in persistent memory and world models whenever possible.**

---

## 2. High-level architecture

```text
                         ENVIRONMENT / USER
                                  |
                                  v
                        +-------------------+
                        |     INTERFACE     |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        |    PERCEPTION     |
                        | text/image/audio  |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        | REPRESENTATION    |
                        | entities/features |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        | CONCEPT ENGINE    |
                        +---------+---------+
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
          +-------------+  +-------------+  +-------------+
          |  EPISODIC   |  |  SEMANTIC   |  | PROCEDURAL  |
          |   MEMORY    |  |   MEMORY    |  |   MEMORY    |
          +------+------+  +------+------+  +------+------+
                 |                |                |
                 +----------------+----------------+
                                  |
                                  v
                        +-------------------+
                        |    WORLD MODEL    |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        |     INFERENCE     |
                        +---------+---------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             +-------------+             +-------------+
             |   ANSWER    |             |   ACTION    |
             +------+------+             +------+------+
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                        +-------------------+
                        |    FEEDBACK       |
                        +---------+---------+
                                  |
                                  v
                        +-------------------+
                        | LEARNING ENGINE   |
                        +---------+---------+
                                  |
                                  +------> MEMORY
```

---

## 3. Component responsibilities

### 3.1 Interface

Converts external interaction into a standard request format.

Example:

```python
Request(
    modality="text",
    content="A dog is an animal.",
    source="user",
)
```

It should not contain learning logic.

---

### 3.2 Perception

Converts raw input into representations.

Initial implementation:

```text
text -> normalized text representation
```

Later:

```text
image -> visual representation
audio -> acoustic representation
```

Perception is allowed to use neural networks.

It should not own long-term knowledge.

---

### 3.3 Representation layer

Produces intermediate objects such as:

```text
Entity("dog")
Concept("animal")
Relation("dog", "is_a", "animal")
```

This is the bridge between perception and the cognitive system.

---

### 3.4 Concept engine

Responsible for:

- concept creation;
- concept merging;
- concept comparison;
- attributes;
- category membership;
- abstraction;
- concept identity.

A concept should not simply be a label.

A conceptual record may contain:

```text
Concept
├── id
├── canonical_name
├── aliases
├── attributes
├── relationships
├── examples
├── parent concepts
├── child concepts
├── evidence
├── confidence
└── status
```

---

## 4. Memory architecture

LITTLE should use multiple memory types.

### 4.1 Episodic memory

Stores events.

Example:

```text
Experience #42

Input:
    "A green apple was shown."

Interpretation:
    APPLE
    color = GREEN

Source:
    user

Confidence:
    0.91
```

Episodic memory answers:

> What happened?

---

### 4.2 Semantic memory

Stores generalized knowledge.

```text
APPLE
  is_a -> FRUIT
  has_property -> EDIBLE
  color -> {RED, GREEN, YELLOW}
```

Semantic memory answers:

> What do I believe generally?

---

### 4.3 Procedural memory

Stores learned procedures.

```text
Procedure: ADD

Inputs:
    integer a
    integer b

Operation:
    addition

Output:
    integer
```

Procedural memory answers:

> How do I do this?

---

### 4.4 Working memory

Temporary state for a current reasoning episode.

It may contain:

- active entities;
- retrieved facts;
- current goal;
- intermediate conclusions;
- unresolved questions.

Working memory should be short-lived.

---

## 5. Knowledge representation

The initial representation should be graph-oriented.

A basic statement:

```text
(subject, relation, object)
```

Example:

```text
(DOG, is_a, ANIMAL)
```

Property:

```text
(APPLE, color, GREEN)
```

Possession:

```text
(ALICE, has, DOG)
```

More complex statements may eventually contain qualifiers:

```text
(subject, relation, object, context, confidence, evidence)
```

---

## 6. Belief model

A belief is not necessarily absolute truth.

Proposed structure:

```text
Belief
├── proposition
├── confidence
├── evidence[]
├── source
├── created_at
├── updated_at
├── status
└── contradictions[]
```

Possible statuses:

```text
SUPPORTED
UNCERTAIN
CONTRADICTED
RETRACTED
UNKNOWN
```

Important:

`UNKNOWN` means there is insufficient evidence.

It does not mean the proposition is false.

---

## 7. Inference engine

The first inference engine should be symbolic and deterministic.

Example:

```text
DOG is_a ANIMAL
ANIMAL is_a LIVING_THING

therefore:

DOG is_a LIVING_THING
```

The engine should retain the inference path:

```text
DOG
  -> ANIMAL
  -> LIVING_THING
```

This makes explanations possible.

---

## 8. Retrieval

Retrieval should happen before reasoning.

Pipeline:

```text
query
  ↓
identify entities/concepts
  ↓
retrieve candidate memories
  ↓
rank evidence
  ↓
construct reasoning context
  ↓
infer
```

Initial retrieval can be exact and graph-based.

Later we can add vector retrieval.

We should not introduce a vector database until the experiments demonstrate that exact/graph retrieval is insufficient.

---

## 9. Learning engine

Learning should operate on experiences.

```text
Experience
    ↓
interpretation
    ↓
candidate knowledge
    ↓
compare with existing knowledge
    ↓
new / duplicate / extension / contradiction
    ↓
update memory
```

Example:

Existing:

```text
APPLE color RED
```

New:

```text
APPLE color GREEN
```

The system should recognize that color may be a multi-valued/variable property rather than overwrite RED blindly.

---

## 10. Concept formation

Concept formation is one of the core research problems.

For v0.1:

```text
new observation
    ↓
similar known concept?
    ├── yes -> attach evidence
    └── no  -> create candidate concept
```

Later:

```text
candidate concepts
    ↓
cluster
    ↓
compare attributes
    ↓
discover abstraction
```

We must carefully distinguish:

- identity;
- similarity;
- category membership.

---

## 11. Unknown detection

Inference should return a structured result:

```text
InferenceResult(
    status="UNKNOWN",
    answer=None,
    confidence=0.18,
    evidence=[],
)
```

Not:

```text
answer="probably yes"
```

unless evidence actually supports it.

Possible result states:

```text
SUPPORTED
REFUTED
UNKNOWN
AMBIGUOUS
CONTRADICTED
```

---

## 12. Learning without full retraining

The architecture should distinguish:

### Stable computational mechanisms

Examples:

- parsers;
- retrieval;
- graph inference;
- memory;
- confidence management.

### Learned state

Examples:

- concepts;
- facts;
- relationships;
- procedures;
- examples;
- learned embeddings.

A new fact should normally modify learned state rather than retrain the entire system.

Neural model training will be introduced only where a neural representation is genuinely necessary.

---

## 13. Neural components

Potential future neural components:

```text
Text encoder
Vision encoder
Audio encoder
Concept representation model
Similarity model
Learned planner
```

These components should be treated as **subsystems**, not as the entire intelligence.

---

## 14. Future self-directed learning loop

Eventually:

```text
Observe
  ↓
Detect uncertainty
  ↓
Generate hypothesis
  ↓
Choose useful experiment
  ↓
Collect evidence
  ↓
Evaluate
  ↓
Update belief
  ↓
Store experience
```

Example:

```text
Hypothesis:
    apple color is not identity-defining

Experiment:
    compare red and green apples

Observation:
    both retain apple identity

Update:
    color marked as variable attribute
```

---

## 15. Rust/Python boundary

The project is **Python-first with optional native Rust acceleration**, not Rust-first.

Python owns:

- cognitive orchestration (`CognitiveKernel`);
- perception, frame extraction, and dialogue state;
- continual plasticity (`PlasticityController`), Cobweb clustering, and self-study;
- SQLite persistence and OKF Markdown vault synchronization;
- CLI (`little`) and interactive REPL;
- evaluation benchmarks and verification tests.

Rust (`rust/little_core`) owns CPU-intensive hot loops compiled as a zero-dependency C-ABI `cdylib` (`liblittle_core.so` / `.dylib` / `.dll`) via the custom PEP 517 build backend (`little_build_backend.py`) and loaded through `ctypes` in `src/little/rust/engine.py` (`RustCoreEngine`) with an automatic pure-Python fallback:

- bidirectional BFS and shortest-path graph traversal;
- DAG acyclicity (`I_DAG`) and mutex (`I_MUTEX`) invariant verification;
- Fast Walsh-Hadamard Transform (FWHT) and 2-bit/1-bit TurboQuant cosine-similarity search.

---

## 16. Implemented repository architecture

```text
mivi_model/
├── README.md
├── architecture.md
├── spec.md
├── prd.md
├── pyproject.toml
├── little_build_backend.py
├── uv.lock
│
├── src/
│   └── little/
│       ├── __init__.py
│       ├── main.py
│       ├── actions/        # ActionRouter, PreDispatchVerifier, ToolExecutor
│       ├── active/         # PlasticityController, LatentKnotInducer, AutonomousSelfStudyEngine,
│       │                   # CobwebClusterEngine, ActiveInquisitor, ContinuousStreamLearner
│       ├── core/           # CognitiveKernel, ConceptKnot, CognitiveCommandRegistry, RuntimePolicy
│       ├── dynamics/       # CfCCell, PhysicalSimulationEngine, CounterfactualEngine
│       ├── evaluation/     # Comprehensive, hybrid, real-world chat, and System 2 matrices
│       ├── inference/      # InferenceEngine, DualSpeedInfillingEngine, DeepSeekInvariantVerifier
│       ├── knowledge/      # LivingWiki, OKFSerializer, KnowledgeImporter, SchemaRegistry
│       ├── language/       # LearningEngine, HybridPerceptionEngine, LayaGatekeeper, ComparativeOrderReasoner
│       ├── memory/         # MemoryStore, PagedConceptStore, HeadroomContextManager, EvidenceLedger
│       ├── neural/         # NeuralRouter, TurboQuantIndex, HashedLinearClassifier
│       ├── perception/     # SensoryGroundingEngine
│       ├── procedural/     # SymbolicCAS, MathStoryEngine, ASTCodeSynthesizer, VerifiedPythonSandbox
│       └── rust/           # RustCoreEngine ctypes bridge + liblittle_core cdylib
│
├── rust/
│   └── little_core/
│       ├── Cargo.toml
│       └── src/            # lib.rs, turbo.rs
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── experiments/
│   ├── 001_basic_learning/ .. 010_real_world_sota_benchmark/
│
├── data/
│   └── schemas/            # Externalized JSON policy schemas
│
└── scripts/
```

---

## 17. Architectural rule

Do not add a subsystem because it is fashionable.

Add it only when an experiment shows that the current architecture cannot solve a defined problem.

That rule protects LITTLE from becoming an unnecessarily complicated collection of AI libraries.

---

## 18. Continual Plasticity, 3-Tier Paged Memory & Adaptive Halting

Inspired by continual adaptive systems (`volotat/mini-AGI`), LITTLE incorporates six tightly integrated homeostatic mechanisms across memory, active learning, and dual-speed inference:

1. **Dirty-Only Concept Invalidation, Resident Slot Pinning & Multi-Gram Phrase Voting (`MemoryStore` + `PagedConceptStore`)**:
   - `MemoryStore` tracks a monotonic `concept_mutation_epoch` and per-epoch dirty sets (`dirty_concepts_since`) that advance **only** when `concepts` or `relations` mutate—ignoring episodic experience logs, proof traces, or ledger metadata writes.
   - `PagedConceptStore` evicts only dirtied concepts on refresh, pins active Resident concepts (`slot_reuses`), and casts weighted activation votes for 1-grams (`1.0x`), 2-grams (`1.5x`), and 3-grams (`2.0x`) (`"The text chooses"`).
2. **Evidence-Governed Continual Plasticity & Trunk/Leaf Split (`PlasticityController`)**:
   - Maintains dual Exponentially Weighted Least-Squares (EWLS) fits ($\lambda = 0.97$ slow, $\lambda_{\text{fast}} = 0.85$ fast) to compute scale-free effect size $e = -\text{slope} / \sigma$.
   - Filters single-step loss spikes (`loss - mean > 2 * sigma`) unless corroborated on the following step, and resets running statistics upon regime shifts (`|e_fast| > 0.3` and `|e_fast| > 2 * |e_slow|`).
   - Applies a `0.1x` trunk damping multiplier to upper-ontology/high-degree concepts (`is_trunk_concept`) to prevent catastrophic forgetting while allowing newborn leaf concepts full plasticity (`1.0x`) that decays smoothly with experience (`rate_half = 50.0`).
3. **5-Brake Speculative Growth Gate (`LatentKnotInducer.evaluate_growth_brakes`)**:
   - Autonomous Cobweb latent concept induction is gated by five boolean brakes (`room & used & earning & fits & honest`):
     - `room`: total concept count is below `max_total_concepts` (`10,000`) and latent ratio is below `max_latent_ratio` (`0.25`).
     - `used`: existing latent concepts are actively addressed in queries or non-taxonomic edges.
     - `earning`: mean confidence across induced latent concepts meets `min_earning_confidence` (`0.55`).
     - `fits`: candidate cluster meets minimum constituent count and Category Utility threshold.
     - `honest`: no unresolved `I_DAG` or `I_MUTEX` invariant violations exist in memory.
4. **Survival-Window Pruning — "Dead Means Unaddressed" (`LatentKnotInducer.prune_unaddressed_latent_knots`)**:
   - Tracks induction cycle and address hits for every `LATENT_*` concept (`touch_latent_concept`). Latent concepts that survive `survival_window_cycles` with zero address hits and zero non-taxonomic outgoing relations are pruned automatically.
5. **Load-Balanced Curiosity Routing & Held-Out Canary Verification (`AutonomousSelfStudyEngine`)**:
   - Balances radial spoke entropy $H(c)$ with an exponential moving average of recent study share (`effective_study_priority`) so stub concepts cannot starve the rest of the graph during dreaming.
   - Runs `evaluate_heldout_canary()` at the end of `consolidate_sleep_cycle()` to verify that autonomous dreaming and invariant mining introduced zero `I_DAG` or `I_MUTEX` regressions.
6. **PonderNet Cumulative Halting & `halt_freeze` (`DualSpeedInfillingEngine`)**:
   - Computes per-depth conditional halting hazard $\lambda_d$, cumulative halting probability $P(\text{halt} \le d) = 1 - \prod_{k=1}^{d} (1 - \lambda_k)$, and regularization cost `ponder_cost = ponder_beta * depth`. When `halt_freeze=True` and $P(\text{halt} \le d) \ge \text{halt\_threshold}$, deep symbolic search freezes early.

