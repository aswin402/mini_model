# LITTLE — Architecture

**Version:** 0.1  
**Status:** Proposed research architecture

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

The project should be **Python-first with Rust acceleration**, not Rust-first.

Python owns:

- orchestration;
- experiments;
- model research;
- training;
- APIs;
- configuration;
- evaluation;
- notebooks;
- high-level reasoning.

Rust owns performance-sensitive components when profiling proves they matter:

- graph operations;
- memory indexing;
- serialization;
- high-throughput retrieval;
- CPU-heavy algorithms;
- custom data structures.

PyO3 provides Rust bindings for Python and supports native Python extension modules; Maturin provides a practical build/package workflow for Rust-based Python packages.

---

## 16. Proposed repository architecture

```text
little/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Cargo.toml
├── Cargo.lock
│
├── docs/
│   ├── prd.md
│   ├── architecture.md
│   ├── spec.md
│   ├── coreidea.md
│   ├── implementationplan.md
│   └── todo.md
│
├── src/
│   └── little/
│       ├── __init__.py
│       ├── api/
│       ├── core/
│       ├── concepts/
│       ├── memory/
│       ├── inference/
│       ├── learning/
│       ├── language/
│       ├── evaluation/
│       └── storage/
│
├── rust/
│   └── little_core/
│       ├── Cargo.toml
│       └── src/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── experiments/
│
├── experiments/
│   ├── 001_memory/
│   ├── 002_concepts/
│   ├── 003_unknown/
│   └── 004_continual_learning/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── little.db
│
└── scripts/
```

---

## 17. Architectural rule

Do not add a subsystem because it is fashionable.

Add it only when an experiment shows that the current architecture cannot solve a defined problem.

That rule should protect LITTLE from becoming an unnecessarily complicated collection of AI libraries.
