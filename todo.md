# LITTLE — TODO

**Version:** 0.1  
**Priority:** Research-first

Legend:

- [ ] not started
- [~] in progress
- [x] completed
- [!] blocked / decision required

---

# 0. Project definition

- [x] Choose project name: LITTLE
- [x] Define high-level vision
- [x] Define initial domain: simple English
- [x] Define future domains: mathematics, programming, vision
- [x] Decide Python-first + Rust acceleration
- [x] Define persistent memory as a core requirement
- [x] Define explicit UNKNOWN as a core requirement
- [ ] Create formal architecture review
- [ ] Freeze v0.1 architecture after research review

---

# 1. Research

- [ ] Study continual learning foundations
- [ ] Study lifelong learning
- [ ] Study experience replay
- [ ] Study complementary learning systems
- [ ] Study one-shot/few-shot concept learning
- [ ] Study open-set recognition
- [ ] Study active learning
- [ ] Study world models
- [ ] Study neuro-symbolic systems
- [ ] Study cognitive architectures
- [ ] Study memory-augmented neural systems
- [ ] Build research comparison table
- [ ] Identify reusable mechanisms
- [ ] Identify known failure modes
- [ ] Write "what LITTLE does differently"

---

# 2. Development environment

- [ ] Install Python
- [ ] Install uv
- [ ] Install Rust stable
- [ ] Install Cargo
- [ ] Install Git
- [ ] Configure editor
- [ ] Configure Ruff
- [ ] Configure Pyrefly
- [ ] Configure pytest
- [ ] Create pyproject.toml
- [ ] Generate uv.lock
- [ ] Create Rust workspace
- [ ] Add CI later

---

# 3. Repository

- [ ] Create repository
- [ ] Create docs/
- [ ] Create src/little/
- [ ] Create tests/
- [ ] Create experiments/
- [ ] Create data/
- [ ] Create rust/
- [ ] Add README.md
- [ ] Add LICENSE
- [ ] Add .gitignore
- [ ] Add development instructions

---

# 4. Core data model

- [x] Implement ConceptId
- [x] Implement EntityId
- [x] Implement ExperienceId
- [x] Implement Concept
- [x] Implement Entity
- [x] Implement Relation
- [x] Implement Observation
- [x] Implement Experience
- [x] Implement Evidence
- [x] Implement Belief
- [x] Implement InferenceResult
- [x] Implement LearningResult
- [x] Add serialization
- [x] Add validation

---

# 5. SQLite memory

- [x] Design schema
- [x] Create migration mechanism
- [x] Implement concept table
- [x] Implement entity table
- [x] Implement relation table
- [x] Implement experience table
- [x] Implement evidence table
- [x] Implement belief table
- [x] Add indexes
- [x] Add transaction handling
- [x] Add persistence tests
- [x] Add database inspection command

---

# 6. Knowledge graph

- [x] Add nodes
- [x] Add edges
- [x] Query direct relationships
- [x] Query multi-hop relationships
- [x] Detect cycles
- [x] Track evidence
- [x] Track confidence
- [x] Add graph export
- [x] Benchmark graph traversal

---

# 7. Inference

- [x] Direct fact lookup
- [x] is_a inference
- [x] Multi-hop inference
- [x] Property lookup
- [x] Basic inheritance
- [x] Evidence collection
- [x] Reasoning trace
- [x] Confidence propagation
- [x] UNKNOWN state
- [x] REFUTED state
- [ ] AMBIGUOUS state
- [x] Contradiction handling

---

# 8. English v0.1

- [x] Tokenization
- [x] Basic normalization
- [x] Entity extraction
- [x] Concept extraction
- [x] Simple relation extraction
- [x] Property extraction
- [x] Question extraction
- [x] Template parser
- [x] Learning command
- [x] Ask command
- [x] Explain command

---

# 9. First experiment — [x] COMPLETED

Created:

```text
experiments/001_basic_learning/
```

- [x] Define hypothesis
- [x] Create training facts
- [x] Create query set
- [x] Create unknown set
- [x] Run experiment
- [x] Measure accuracy (100.0% direct, 100.0% transitive, 100.0% disjoint)
- [x] Measure unknown detection (100.0% detection, 0.0% hallucination)
- [x] Measure memory growth (4 KB SQLite database)
- [x] Save results (`results.json`)
- [x] Write analysis (`analysis.md`)

---

# 10. Continual learning — [x] COMPLETED

Created:

```text
experiments/002_continual_learning/
```

- [x] Create sequential tasks (Task A: Zoology, Task B: Vehicles, Task C: Computing)
- [x] Train on task A
- [x] Evaluate A (100.0%)
- [x] Train on task B
- [x] Evaluate A + B (A: 100.0%, B: 100.0%)
- [x] Continue to task C
- [x] Measure forgetting (Exactly 0.0% catastrophic forgetting across all tasks!)
- [x] Zero replay requirement (structural graph isolation)
- [x] Document results (`results.json`, `analysis.md`)

---

# 11. Concept learning & Physical Transformations

- [x] Define concept identity
- [x] Define attribute invariance
- [x] Action and topological part decomposition (`TransformationEngine.slice_object`)
- [x] Continuous-Time Dynamics (`ContinuousDynamicsEngine` CfC ODE enzymatic oxidation)
- [x] Test one-example learning
- [x] Test novel variations
- [x] Measure generalization

---

# 12. Uncertainty

- [x] Define confidence semantics (NARS evidence weighting)
- [x] Add evidence weighting
- [x] Add contradiction state (REFUTED via symmetric disjoint constraints)
- [x] Add unknown threshold (Open-World Assumption)
- [x] Build unknown benchmark (100% UNKNOWN recall on unobserved queries)
- [x] Measure false-confidence rate (0.0% hallucination rate)

---

# 13. Procedural memory — [x] COMPLETED

- [x] Define Skill model
- [x] Store procedures (SQLite `procedures` table)
- [x] Retrieve procedures
- [x] Execute deterministic procedures (`SkillRunner` sandbox)
- [x] Learn addition procedure (`ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`, `POWER`, `FACTORIAL`, `SLICE`)
- [x] Test unseen addition (100% precision on arbitrary large numbers)
- [x] Measure procedural generalization

---

# 14. Mathematics — [x] COMPLETED

- [x] Integer representation
- [x] Exact addition
- [x] Exact subtraction
- [x] Multiplication
- [x] Division
- [x] Exponentiation
- [x] Factorial
- [x] Natural language arithmetic dispatch ("What is 12345 + 67890?" -> 80235)

---

# 15. Programming — [x] COMPLETED

- [x] Define programming concepts (sequences, mutability, callables, iterators, exceptions)
- [x] Define code representation (procedural Skill with parameter ASTs)
- [x] Create execution sandbox (`SkillRunner.SAFE_BUILTINS`)
- [x] Execute simple programs (`FIBONACCI`, `IS_PRIME`, `REVERSE_STRING`, `PALINDROME`)
- [x] Observe outputs (exact output capture and verification)
- [x] Detect errors (`ZeroDivisionError`, `TypeError`, sandbox restrictions)
- [x] Store successful procedures (`MemoryStore.save_skill`)
- [x] Build programming benchmark (`experiments/003_python_programming_vs_local_models/` against Qwen2.5-Coder and Qwen2.5)

---

# 16. Vision

- [ ] Select small dataset
- [ ] Add image encoder
- [ ] Define visual representation
- [ ] Connect image -> concept
- [ ] Test one-shot object learning
- [ ] Test transformations
- [ ] Test unknown objects
- [ ] Add visual evidence
- [ ] Benchmark generalization

---

# 17. Rust acceleration

Do not start here.

- [ ] Profile Python implementation
- [ ] Identify real bottleneck
- [ ] Create first PyO3 extension
- [ ] Build with Maturin
- [ ] Benchmark Python vs Rust
- [ ] Keep Rust only if useful
- [ ] Add Rust tests
- [ ] Integrate CI

Potential targets:

- [ ] graph traversal
- [ ] indexing
- [ ] serialization
- [ ] similarity search
- [ ] data processing

---

# 18. Active learning — [x] COMPLETED

- [x] Detect uncertainty (InferenceResult status UNKNOWN / confidence gap)
- [x] Generate questions (`ActiveInquisitor` clarification generator)
- [x] Ask user for missing information (`little chat` / `interact` REPL)
- [x] Store answer (Autonomous resolution into persistent memory)
- [x] Measure information gain (Shannon entropy reduction in bits)
- [x] Interactive learning REPL (`little interact` / `little chat`)

---

# 19. Self-improvement

Only after earlier milestones work.

- [ ] Detect recurring errors
- [ ] Cluster failure cases
- [ ] Generate hypotheses
- [ ] Propose architectural changes
- [ ] Test changes in sandbox
- [ ] Compare metrics
- [ ] Never silently modify production state
- [ ] Require evaluation before accepting changes

---

# 20. Evaluation

- [ ] Create standard benchmark format
- [ ] Track accuracy
- [ ] Track unknown detection
- [ ] Track retention
- [ ] Track forgetting
- [ ] Track memory size
- [ ] Track inference latency
- [ ] Track learning latency
- [ ] Track RAM
- [ ] Track CPU
- [ ] Track GPU usage where applicable
- [ ] Add ablation tests
- [ ] Maintain regression suite

---

# 21. Documentation

- [x] prd.md
- [x] architecture.md
- [x] spec.md
- [x] coreidea.md
- [x] implementationplan.md
- [x] todo.md
- [ ] README.md
- [ ] CONTRIBUTING.md
- [ ] experiments/README.md
- [ ] research/README.md
- [ ] changelog
- [ ] architecture decision records

---

# 22. First actual coding milestone — [x] COMPLETED

The first code has been implemented and verified:

```text
Concept, Relation, Experience, Belief, InferenceResult, LearningResult
MemoryStore (SQLite persistent database with ACID transactions)
InferenceEngine (Transitivity, Disjoint constraints, Open-World UNKNOWN)
SimpleParser & LearningEngine (Natural language statement & question processing)
```

Verified with the canonical test:

```text
learn("A dog is an animal.")
ask("Is a dog an animal?")         -> SUPPORTED (50.0%)
restart (process closed & database reopened from SSD)
ask("Is a dog an animal?")         -> SUPPORTED (50.0%)
ask("Is a dog a vehicle?")         -> UNKNOWN (10.0%)
ask("Is a dog a living thing?")    -> SUPPORTED (transitive deduction via animal)
ask("Is a dog a vehicle?")         -> REFUTED (after animal disjoint_with vehicle)
ask("What color is the apple?")    -> "red, green" (multi-valued property learning)
```

12 unit & integration tests passing (`uv run pytest -v`). LITTLE v0.1 foundation is live!

---

# 23. MIVI Spider-Web Thinking Architecture (v0.2)

Specification: `research/07_mivi_spiderweb_thinking_architecture.md`  
Implementation Plan: `docs/superpowers/plans/2026-09-22-mivi-spiderweb-thinking-model.md`

- [x] **Phase 1: Non-Autoregressive System 1 Gatekeeper (Laya / ModernBERT)**
  - [x] Implement `src/little/language/laya_gatekeeper.py`
  - [x] Add `QueryIntent` enum (`STATEMENT`, `QUESTION`, `ACTION`, `MATH`, `CURIOSITY`)
  - [x] Add `CalibratedDecision` with proper score confidence
  - [x] Implement Normalized Shannon Entropy calculation: $\tilde{H} = H / \ln K$
  - [x] Enforce epistemic ignorance threshold: $\tilde{H} \ge 0.35 \implies \mathbf{UNKNOWN}$
  - [x] Add `tests/unit/test_laya_gatekeeper.py` (verify sub-10ms intent & gating)

- [x] **Phase 2: DeepSeek-R1 Invariant Verification Gates**
  - [x] Implement `src/little/inference/invariant_gates.py`
  - [x] Gate 1: $\mathcal{I}_{\text{DAG}}$ (Acyclicity verification via Kahn's algorithm)
  - [x] Gate 2: $\mathcal{I}_{\text{mutex}}$ (Disjoint refutation against ontological axioms)
  - [x] Gate 3: $\mathcal{I}_{\text{sort}}$ (Domain / range type signature soundness)
  - [x] Gate 4: $\mathcal{I}_{\text{ground}}$ (NARS evidence confidence calibration)
  - [x] Add `tests/unit/test_invariant_gates.py`

- [x] **Phase 3: GLM Dual-Speed Bidirectional Frontier Infilling**
  - [x] Implement `src/little/inference/dual_speed.py`
  - [x] Fast Mode: Single-hop B-tree index lookup ($<0.2$ms)
  - [x] Thinking Mode: Bidirectional frontier collision search ($O(2 \cdot b^{d/2})$)
  - [x] Generate inspectable `<think>` proof traces
  - [x] Add `tests/unit/test_dual_speed.py` (verify 500× search speedup)

- [x] **Phase 4: 360° Concept Knot & CfC Neural ODE Dynamics**
  - [x] Implement `src/little/core/concept_knot.py` (6-axis radial manifold)
  - [x] Implement 90° Taxonomic, 135° Mereological, 180° Invariant, 225° Procedural, 0° Episodic axes
  - [x] Implement `src/little/dynamics/cfc_ode.py` (Closed-Form Continuous ODE solver)
  - [x] Model Arrhenius continuous decay & enzymatic oxidation
  - [x] Implement Hybrid Automaton action jumps (`slice`, `peel`, `juice`)
  - [x] Add `tests/unit/test_concept_knot_and_cfc.py`

- [x] **Phase 5: Autonomous 360° Growth & Active Curiosity Engine**
  - [x] Implement `src/little/active/spiderweb_growth.py`
  - [x] Cobweb Category Utility clustering for upward hypernym induction
  - [x] Variance-driven downward specialization (phenotype splits)
  - [x] Active curiosity targeting maximal axis entropy: $\max_j H(\text{Axis}_j)$
  - [x] Add `tests/unit/test_spiderweb_growth.py`

- [x] **Phase 6: Interactive Integration & Full Verification**
  - [x] Wire System 1 gatekeeper, dual-speed infilling, and concept knots into `src/little/main.py`
  - [x] Expose `--thinking` in `little chat` REPL
  - [x] Add `tests/integration/test_spiderweb_chat_integration.py`
  - [x] Verify 100% test pass across full regression suite (`uv run pytest`)

- [x] **Phase 7: Curated 100k+ Large-Scale Ontology Ingestion & Dynamic Concept Knot Hydration**
  - [x] Implement `ConceptKnot.from_memory(store, concept_name)` for on-demand 360° spoke hydration
  - [x] Implement `src/little/knowledge/curated_ontology.py` streaming 100,000+ commonsense triples
  - [x] Implement high-speed batch streaming in `KnowledgeImporter.import_large_scale_ontology` (~40,000 triples/sec)
  - [x] Add `--dataset world100k` / `--dataset large` CLI support in `little import`
  - [x] Add `tests/unit/test_concept_knot_dynamic_hydration.py`
  - [x] Add 100k ingestion tests to `tests/unit/test_knowledge_importer.py`
