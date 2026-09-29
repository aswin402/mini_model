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

# 19. Self-improvement — [x] COMPLETED

- [x] Detect missing concept knot spokes via 360° radial entropy: $\max_j H(\text{Axis}_j)$
- [x] Generate analogical hypotheses via hypernym DAG traversal
- [x] 4 Deterministic Invariant Gates ($\mathcal{I}_{\text{DAG}}, \mathcal{I}_{\text{mutex}}, \mathcal{I}_{\text{sort}}, \mathcal{I}_{\text{ground}}$)
- [x] Test candidate hypotheses against SQLite invariant constraints
- [x] Live persistence of verified spokes with zero human intervention
- [x] Never persist refuted or invalid hypotheses
- [x] CLI commands (`little curious`, `little study`) & REPL commands (`curious`, `study`, `--auto-study`)

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

- [x] **Phase 8: Empirical Benchmark Evaluation vs. Statistical SLMs**
  - [x] Implement comprehensive 6-suite benchmark in `experiments/005_slm_comparison_benchmark/experiment.py`
  - [x] Run live head-to-head empirical evaluation against `qwen2.5:0.5b` via local Ollama API
  - [x] Verify 100% (34/34) accuracy for MIVI vs. 35.7% (10/28) for SLM
  - [x] Verify 0.0% hallucination under Open-World Assumption vs. 100.0% hallucination by SLM
  - [x] Verify 0.0% catastrophic forgetting across sequential multi-domain ingestion
  - [x] Verify 449x latency speedup (2.67 ms MIVI P50 on pure CPU vs. 1,201 ms SLM)
  - [x] Produce comprehensive report in `experiments/005_slm_comparison_benchmark/REPORT.md` and `results.json`
 
- [x] **Phase 9: English Dialogue Mastery, Exact Math CAS, and Autonomous Closed-Loop Self-Study**
  - [x] Conversational Dialogue Context & Centering Theory Pronoun/Anaphora Resolution (`src/little/language/dialogue.py`)
  - [x] Generalized Compound Sentences, Relative Clauses, and Polite/Indirect Inquiries (`src/little/language/parser.py`)
  - [x] Exact Rational Symbolic CAS (`fractions.Fraction`, linear systems, quadratics, unit conversion graph) (`src/little/procedural/math_cas.py`)
  - [x] Multi-Step Word Problem & Story Tracker with `<math_trace>` (`src/little/procedural/math_story.py`)
  - [x] Autonomous Closed-Loop Curiosity & Self-Study Engine (`src/little/active/self_study.py`)
  - [x] Zero-Hardcoding Externalized Policy Schemas (`parser_question_boundary_policy.json`, `self_study_policy.json`, `language_policy.json`)
  - [x] Full CLI integration (`little curious`, `little study`, `--auto-study`) and 462/462 passing tests

- [x] **Phase 10: Socratic Explanations, Deductive Syllogisms & Continuous Counterfactual Reasoning**
  - [x] Zero-Hardcoding Externalized Policy Schemas (`data/schemas/socratic_policy.json`, `data/schemas/counterfactual_policy.json`, `data/schemas/language_policy.json`, `data/schemas/parser_semantic_policy.json`)
  - [x] Deductive Socratic Reasoner (`src/little/language/socratic.py`, `src/little/language/socratic_policy.py`)
    - Multi-hop taxonomic syllogism generator ($N$-hop step-by-step deductive proofs)
    - Axiomatic disjoint refutation proofs ($\mathcal{I}_{\text{mutex}}$ violation proofs)
    - Hypernym capability and property inheritance proofs
    - Grammatically correct indefinite articles (`a` vs `an`)
  - [x] Continuous Counterfactual Dynamics Engine (`src/little/dynamics/counterfactual.py`, `src/little/dynamics/counterfactual_policy.py`)
    - Closed-Form Continuous (CfC) ODE simulation with Arrhenius kinetics and $Q_{10}$ temperature scaling
    - Contrastive parallel hypothetical world evaluation ("What if X is refrigerated vs in the sun for 24 hours?")
    - Single-scenario hypothetical projections ("What happens if X is left in the sun for 24 hours?")
  - [x] Seamless Natural Language & Cognitive Controller Routing:
    - Wired Socratic explanations into `SimpleParser.__why_is_a__`, `__why_not__`, `__why_can__`
    - Routed counterfactual simulations through `SimpleParser` and `ThinkingController`
  - [x] Full Verification & Test Coverage:
    - `tests/unit/test_socratic_explanations.py`
    - `tests/unit/test_counterfactual_reasoning.py`
    - `tests/integration/test_socratic_counterfactual_chat.py`

- [x] **Phase 11: Multi-Constraint Algebraic Word Problems & Comparative Order Reasoning (Math + English Synthesis)**
  - [x] Externalized Policy Schemas & Data Models:
    - `data/schemas/comparative_order_policy.json` (dimensions, comparative pairs, superlatives, statement & question patterns, proof templates)
    - `data/schemas/algebraic_word_problem_policy.json` (multipliers, positive/negative offsets, temporal shifts, sum markers, question targets, `<math_trace>` templates)
    - `data/schemas/relation_types.json` (registered comparative relations: `height_greater`, `age_greater`, `speed_greater`, `weight_greater`, `cost_greater`, `size_greater`, `strength_greater`, `depth_greater`)
  - [x] Comparative Order & Strict Partial Order Reasoner (`src/little/language/comparative.py`, `src/little/language/comparative_policy.py`):
    - Strict partial order poset DAG builder with dual-polarity normalization ($A > B \iff B < A$)
    - Transitive multi-hop proof generator ("Yes. A is taller than M, and M is taller than B. Therefore, A is taller than B.")
    - Direct and indirect refutation of inverted relations ("No. In fact, B is taller than A.")
    - Topological root/sink superlative extremum queries ("Who is the tallest?", "Who is the shortest?")
    - Cycle & contradiction detection preventing paradoxical assertions
    - ACID SQLite persistent storage of comparative relations
  - [x] Algebraic Word Problem Linear System Solver (`src/little/procedural/algebraic_word_problem.py`, `src/little/procedural/algebraic_word_problem_policy.py`):
    - Age word problems with proportional multipliers and signed temporal shifts ($t \pm k$ years)
    - Multi-entity coin, cost, and item linear combination problems
    - Exact 2x2 linear matrix formulation solved symbolically via Cramer's rule in `SymbolicCAS`
    - Transparent `<math_trace>` showing variable assignments, system constraints, determinants, and target answers
  - [x] Seamless Natural Language Integration & Cognitive Routing:
    - Integrated with `MathStorySolver.solve_story` and `LearningEngine.ask()`
    - Routed through `ThinkingController` fast mode
  - [x] Comprehensive Unit & Integration Tests:
    - `tests/unit/test_comparative_reasoner.py` (4/4 passed)
    - `tests/unit/test_algebraic_word_problems.py` (3/3 passed)
    - `tests/integration/test_math_english_synthesis_chat.py` (4/4 passed)

- [x] **Phase 12: Zero-Hardcoding Conversational Surface Engine & Social Protocol**
  - [x] Externalized Policy Schema (`data/schemas/dialogue_surface_policy.json`):
    - 6 core social intents (`greeting`, `how_are_you`, `thank_you`, `goodbye`, `identity`, `compliment`) with pattern matching and varied warm responses
    - `mixed_intent_splitter` with `leading_greeting_regex` extracting greetings from mixed inputs ("Hello! Is a robin a bird?")
    - `verbalizer_templates`: affirmative taxonomy, capability, feature, property, location, general affirmative/refuted, epistemic unknown invitations, and learning confirmations
  - [x] Typed Policy & Engine Implementation:
    - `src/little/language/dialogue_surface_policy.py`: Dataclass models for social intent, templates, and full schema loading
    - `src/little/language/dialogue_surface.py`: `DialogueSurfaceEngine` handling social turns, compound utterance decomposition, reply composition, learning confirmation verbalization, and inference answer verbalization
  - [x] Core Architecture Integration:
    - `src/little/core/runtime_policy.py`: Wired `dialogue_surface` policy into global `RuntimePolicy`
    - `src/little/core/kernel.py`: Added `chat(self, text: str) -> str` supporting mixed social + substantive turns
    - `src/little/language/parser.py`: Added `LearningEngine.chat(self, text: str) -> str`, wired comparative statement assertions into `commit_frame()`
    - `src/little/language/perception.py`: Added math story detection bypass to avoid splitting multi-sentence word problems
    - `src/little/main.py`: Interactive CLI `cmd_interact` updated with natural conversational replies
  - [x] Schema Robustness Fixes:
    - Updated `comparative_order_policy.json` to handle between/among clauses in extremum questions
    - Updated `algebraic_word_problem_policy.json` to support numeric digit multipliers
  - [x] Full Verification & Test Coverage:
    - `tests/unit/test_dialogue_surface.py` (4/4 passed)
    - `tests/integration/test_conversational_fluency_chat.py` (1/1 passed, 12 multi-turn interactions)
    - Total regression suite: 485/485 passed (100%)

- [x] **Phase 13: Multi-Turn Discourse Anaphora, Pronoun Coreference & Conversational Concept Summarization**
  - [x] Externalized Policy Schemas (`dialogue_surface_policy.json`, `parser_definition_policy.json`, `parser_question_policy.json`):
    - Added `creator` and `capabilities` social intents
    - Added definition verbalizer templates (`definition_taxonomy`, `definition_has`, `definition_part_of`, `definition_parts`, `definition_disjoint`, `definition_attributes`)
    - Extended definition routing with `what_do_you_know_about`, `summarize`, and `describe` with flexible punctuation stripping
    - Broadened `user_name` memory recall ("Do you remember my name?", "Do you know who I am?") and `identity_name`
  - [x] Dialogue Context & Anaphora Engine (`src/little/language/dialogue.py`):
    - Multi-turn pronoun co-reference resolution for `it` and generic plural `they` prioritizing salient subject antecedents
  - [x] Conversational Verbalizer & Surface Engine (`src/little/language/dialogue_surface.py`):
    - Implemented `verbalize_concept_definition` to convert semi-colon separated knowledge graph records into warm, flowing English paragraphs
    - Pronoun replacement in `verbalize_inference` substituting pronouns (`it`, `they`, `he`, `she`) with resolved concept antecedents
    - Natural textual answers for self/user identity without awkward prefixes
  - [x] Full Verification & Test Coverage:
    - `tests/integration/test_conversational_fluency_chat.py` (2/2 passed)
    - Full regression test suite: 487/487 passed (100% pass rate)

- [x] **Phase 14: 25-Turn Real-World Generalization, Strict Zero-Hallucination, and Cross-Domain Simulation**
  - [x] Proportional Quantity Word Problems:
    - Implemented `_solve_proportional_quantity_problem` in `src/little/procedural/algebraic_word_problem.py`
    - Added `quantity_entity_answer` to `data/schemas/algebraic_word_problem_policy.json`
  - [x] Proper Noun & Surface Grammar Hardening:
    - Added proper noun casing restoration and indefinite article stripping in `DialogueSurfaceEngine._article` (`src/little/language/dialogue_surface.py`)
    - Added multi-word entity greedy matching for comparative posets in `data/schemas/comparative_order_policy.json`
  - [x] Celestial & Physical Relational Invariants:
    - Registered `orbits` and `orbit` in `relation_types.json` and `semantic_predicate_policy.json` (`acyclic: false`)
    - Added `affirmative_relation` verbalizer template in `dialogue_surface_policy.json`
    - Wired base verb to 3sg transitive query mapping (`orbit` -> `orbits`) in `src/little/language/parser.py`
  - [x] Continuous Physical Simulation Integration:
    - Added `simulate` prefix to `counterfactual_policy.json` and extended regex in `counterfactual.py` for physical decay
  - [x] Real-World Multi-Turn Simulation Suite:
    - Added `tests/integration/test_real_world_simulation_dialogue.py` covering 25 live cross-domain turns (astronomy, biology, algebra, posets, physics, identity, memory recall, UNKNOWN boundary)
    - Regression suite: 491/491 tests passed (100% pass rate)
