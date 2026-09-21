# LITTLE — Implementation Plan

**Version:** 0.1  
**Strategy:** Python-first, Rust-accelerated where justified

---

## 1. Development philosophy

Build the smallest experiment that can falsify the current design.

Do not implement the final architecture all at once.

Each milestone must produce:

1. working code;
2. tests;
3. benchmark;
4. observations;
5. architectural decision.

---

## 2. Recommended technology stack

### Core

- Python
- uv
- PyTorch
- NumPy
- SQLite

### Developer tooling

- Ruff
- Pyrefly
- pytest
- Git
- GitHub Actions later

### Rust integration

- Rust stable
- Cargo
- PyO3
- Maturin

### Optional later

- Polars
- NetworkX
- msgspec
- vector indexing
- specialized ANN libraries

These optional tools should only be introduced when an experiment demonstrates a need.

---

## 3. Why Python-first?

The early project contains many unknowns.

Python gives us:

- fast iteration;
- mature ML tooling;
- easy experimentation;
- easy notebooks;
- easier integration with datasets;
- easier visualization;
- access to PyTorch.

A research architecture needs iteration speed more than maximum runtime speed.

---

## 4. Why Rust too?

Rust becomes valuable when we discover hot paths.

Candidates:

```text
large graph traversal
memory indexing
serialization
similarity search
CPU-heavy transformations
parallel data processing
```

We should profile before moving code.

Rule:

```text
Python first
   ↓
benchmark
   ↓
profile
   ↓
identify bottleneck
   ↓
rewrite only bottleneck in Rust
```

---

## 5. Repository bootstrap

Initial commands conceptually:

```bash
uv init
uv python pin 3.12
uv add numpy torch
uv add --dev pytest ruff pyrefly
```

Rust workspace:

```bash
cargo new --lib rust/little_core
```

Then integrate PyO3/Maturin when the first Rust component is justified.

The exact dependency versions should be pinned by the generated lockfiles rather than manually copied into this document.

---

## 6. Milestone 0 — Repository

Create:

```text
little/
├── docs/
├── src/little/
├── tests/
├── experiments/
├── data/
├── rust/
└── scripts/
```

Success:

```text
uv run pytest
uv run ruff check .
uv run pyrefly check
cargo test
```

all execute successfully.

---

## 7. Milestone 1 — Persistent memory

Implement:

```text
MemoryStore
```

Operations:

```python
create_concept()
get_concept()
create_entity()
add_relation()
get_relations()
add_experience()
get_experience()
```

SQLite schema should initially be minimal.

Success test:

```text
create fact
close process
restart
retrieve fact
```

---

## 8. Milestone 2 — Knowledge graph

Implement:

```text
Concept
Entity
Relation
Evidence
```

Example:

```text
DOG -> is_a -> ANIMAL
```

Add graph traversal.

Success:

```text
DOG -> ANIMAL -> LIVING_THING
```

can be discovered.

---

## 9. Milestone 3 — Deterministic inference

Implement rules:

```text
is_a transitivity
property inheritance
basic relation lookup
```

Example:

```text
DOG is_a ANIMAL
ANIMAL is_a LIVING_THING

=> DOG is_a LIVING_THING
```

Return an evidence trace.

---

## 10. Milestone 4 — Unknown detection

Add:

```text
SUPPORTED
REFUTED
UNKNOWN
AMBIGUOUS
```

Create a benchmark containing both answerable and unanswerable questions.

Measure false confident answers.

---

## 11. Milestone 5 — English interface

Implement a deliberately constrained parser.

Initially support templates such as:

```text
A dog is an animal.
A dog is brown.
Alice has a dog.
```

Do not use an LLM initially.

The parser can be rule-based.

This gives us a clean experiment of the underlying architecture.

---

## 12. Milestone 6 — Natural language expansion

Once the symbolic pipeline works, add a small learned language component.

Possible options:

- a small encoder;
- a small local language model;
- a constrained parser model.

It should map language to structured representations.

The core memory and reasoning system should remain independent.

---

## 13. Milestone 7 — Continual learning benchmark

Create sequences:

```text
Task A -> animals
Task B -> objects
Task C -> people
Task D -> properties
Task E -> mixed
```

Measure:

- retention;
- learning speed;
- interference;
- memory growth;
- query accuracy.

Compare against:

1. static rule system;
2. simple classifier;
3. continual neural baseline where appropriate.

---

## 14. Milestone 8 — Generalization

Create held-out variations.

Example:

Training:

```text
red apple
green apple
```

Test:

```text
yellow apple
sliced apple
rotated apple
```

The evaluation must distinguish:

- memorization;
- similarity matching;
- genuine concept generalization.

---

## 15. Milestone 9 — Procedural memory

Implement procedures.

First:

```text
ADD
SUBTRACT
MULTIPLY
```

Then symbolic transformations.

The system should be able to retrieve and execute a learned procedure.

---

## 16. Milestone 10 — Mathematics

Progression:

```text
integer arithmetic
    ↓
fractions
    ↓
expressions
    ↓
equations
```

Use exact symbolic computation where appropriate.

Neural components should not be used for tasks where deterministic symbolic computation is clearly superior.

---

## 17. Milestone 11 — Programming

Start with a sandbox.

The model should:

```text
generate small program
execute
observe result
compare
repair
```

Safety requirement:

Never execute arbitrary generated code directly on the host without isolation.

Use a sandbox/container for later experiments.

---

## 18. Milestone 12 — Vision

Add a visual encoder.

Architecture:

```text
image
 ↓
vision encoder
 ↓
embedding/features
 ↓
concept matcher
 ↓
memory
 ↓
reasoning
```

Start with a small controlled dataset.

Do not begin with a massive image dataset.

---

## 19. Milestone 13 — Multimodal concepts

Connect:

```text
text "apple"
        ↕
APPLE concept
        ↕
image embedding
```

The concept should become modality-independent while preserving modality-specific evidence.

---

## 20. Milestone 14 — Active learning

Implement:

```text
detect uncertainty
      ↓
generate information request
      ↓
receive answer
      ↓
update knowledge
```

Example:

```text
System:
"I don't know whether this object is an apple.
Is it an apple?"
```

Later, allow tool/environment interaction.

---

## 21. Milestone 15 — Self-directed experiments

Introduce an experiment planner.

Input:

```text
knowledge gaps
```

Output:

```text
candidate experiments
```

Select experiments based on expected information gain.

This is a major research milestone and should not be rushed.

---

## 22. Evaluation methodology

Every milestone should have:

### Baseline

What happens with the simplest possible implementation?

### Proposed system

What does LITTLE do?

### Metric

How do we measure improvement?

### Ablation

What happens if we remove a component?

Example:

```text
LITTLE
vs
LITTLE without episodic memory
vs
LITTLE without semantic memory
vs
LITTLE without confidence
```

This will tell us which architectural components actually matter.

---

## 23. Hardware strategy

### Local PC

Use for:

- coding;
- SQLite;
- graph reasoning;
- small models;
- CPU experiments;
- evaluation.

### Google Colab

Use when GPU acceleration materially helps:

- vision experiments;
- neural representation training;
- larger embedding experiments.

Do not assume a specific free-tier GPU or runtime duration.

---

## 24. Resource budget

Initial target:

```text
RAM:
<= 16 GB

CPU:
works on Ryzen 7-class processor

GPU:
optional

Persistent storage:
local SSD

Model size:
small during early experiments
```

We should record peak RAM and runtime for every benchmark.

---

## 25. First 30-day development plan

### Week 1

- repository;
- tooling;
- memory schema;
- SQLite;
- basic data classes;
- tests.

### Week 2

- concepts;
- relations;
- evidence;
- graph traversal;
- deterministic inference.

### Week 3

- unknown detection;
- contradiction handling;
- English template parser;
- learning pipeline.

### Week 4

- benchmark;
- continual-learning experiment;
- documentation;
- architecture review.

At the end of 30 days, we should know whether the core idea is technically promising enough to continue.

---

## 26. Research notebook requirement

Every experiment gets a directory:

```text
experiments/001_memory/
```

containing:

```text
README.md
config.toml
results.json
notes.md
```

Record:

- hypothesis;
- setup;
- data;
- metrics;
- result;
- interpretation;
- next decision.

---

## 27. Versioning

Use semantic-style project versions:

```text
0.1.x  foundation
0.2.x  continual learning
0.3.x  procedural reasoning
0.4.x  vision
0.5.x  multimodal
1.0    mature research prototype
```

These are project milestones, not promises.

---

## 28. Definition of done for an architectural component

A component is done when:

- API is documented;
- implementation exists;
- tests exist;
- benchmark exists;
- failure cases are recorded;
- memory/performance characteristics are measured;
- integration behavior is understood.

---

## 29. Golden rule

Do not optimize the architecture before we know what the architecture needs.

First make it:

```text
correct
inspectable
measurable
```

Then make it:

```text
fast
small
scalable
```

## Official tooling references used for the stack decision

- uv — modern Python project/dependency/environment workflow.
- Pyrefly — Python type checker and language server.
- Ruff — Python linter and formatter.
- PyO3 — Rust bindings for Python and native Python extension modules.
- Maturin — build/package workflow for Rust-based Python packages.
- SQLite — embedded, serverless, transactional local database.
- Polars — optional later data-processing layer with a Rust core.

The architecture deliberately avoids pinning fast-moving tool versions in design documents. Exact versions belong in `pyproject.toml`, `Cargo.toml`, and lockfiles.
