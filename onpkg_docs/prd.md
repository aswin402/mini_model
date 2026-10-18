# LITTLE — Product Requirements & Core Philosophy

**Project:** LITTLE (MIVI Cognitive Architecture)  
**Version:** 0.3  
**Status:** Active Implementation

---

## 1. The Central Idea & Product Vision

LITTLE starts from one foundational question:

> *What if an AI did not need to contain all of its knowledge inside one enormous neural network?*

Most AI systems begin with a large pretrained language model containing broad statistical knowledge and then bolt on prompting, retrieval, or fine-tuning. LITTLE explores a different architecture:

- **Small computational core:** Mechanisms for perception, reasoning, verification, and learning live in code and lightweight neural adapters.
- **Externalized persistent memory:** Acquired domain knowledge lives outside the core in inspectable SQLite relational tables (`concepts`, `relations`, `experiences`, `procedures`, `evidence_records`, `commit_decisions`, `proof_traces`) and an Obsidian-compatible Open Knowledge Format (`.okf.md`) Markdown vault (`LivingWiki`).
- **Explicit concepts (`ConceptKnot`):** A concept is not a bare string token; it is a 360° 6-spoke structure (Taxonomy, Mereology, Attributes/Invariants, Dynamics, Procedural Skills, and Episodic Grounding) that evolves as evidence accumulates.
- **First-class uncertainty (`UNKNOWN`):** Under the Open-World Assumption, insufficient evidence produces an explicit `UNKNOWN` belief state—never a hallucinated guess (`0.0%` hallucination under OWA).
- **Continual learning without catastrophic forgetting:** Adding or revising a fact updates persistent semantic memory and adjusts per-concept continual plasticity (`PlasticityController`) without retraining a monolithic network.

---

## 2. Core Comparison: Monolithic LLM vs. LITTLE

```text
Monolithic LLM Pipeline:
  Huge Corpus -> Pretraining -> Billions of Frozen Weights -> Prompt -> Statistical Next-Token Guess

LITTLE Cognitive Loop:
  Experience -> Perception -> Concept Knot Formation -> 4 Invariant Gates (I_DAG, I_MUTEX, I_SORT, I_TEMP)
             -> Tiered Memory (Disk / LRU / Resident) -> Dual-Speed Reasoning (PonderNet Halting)
             -> Answer or Verified Action -> Feedback & Continual Plasticity Update
```

---

## 3. Product Principles

1. **Knowledge is externalizable:** Every learned concept, relation, and proof trace is inspectable and exportable (`little inspect`, `little export`, `little status`).
2. **Learning is incremental & homeostatic:**
   - Upper-ontology **trunk** concepts receive a `0.1x` learning-rate damping multiplier to prevent catastrophic forgetting.
   - Newborn **leaf** concepts receive full plasticity (`1.0x`) decaying smoothly over maturity (`rate_half = 50.0`).
   - Autonomous Cobweb latent concept induction is governed by a **5-Brake Speculative Growth Gate** (`room & used & earning & fits & honest`) and **Survival-Window Pruning** (*"Dead means unaddressed"*).
3. **Experience and abstraction are distinct:** Episodic memory records *what happened* (`experiences`); semantic memory records *generalized beliefs* (`concepts`, `relations`) backed by an append-only `EvidenceLedger`.
4. **Simplicity and honesty before scale:** Every capability is grounded in deterministic symbolic engines, exact rational arithmetic (`SymbolicCAS`), process-isolated Python synthesis, or lightweight CPU-native hashed N-gram classifiers—never hidden behind undocumented external LLM calls.

---

## 4. Target Capabilities & Scope

- **Natural Language Teaching & Inquiry:** Declarative fact learning, multi-hop transitive deduction (`is_a`, `part_of`, `has`), disjointness refutation (`disjoint_with`), comparative partial orders (`taller than`, `heavier than`), and multi-turn pronoun coreference.
- **Exact Procedural & Mathematical Reasoning:** Exact `Fraction`-based symbolic CAS, Cramer's rule $2 \times 2$ linear word-problem solver, and AST-verified Python skill synthesis (`ASTCodeSynthesizer` + `VerifiedPythonSandbox`).
- **Physical & Continuous-Time Dynamics:** Closed-Form Continuous-Time (`CfCCell` / `PhysicalSimulationEngine`) state evolution (Arrhenius thermal kinetics, oxidation, moisture decay) and discrete action jumps (`slice`, `peel`, `heat`, `chill`, `dehydrate`) with Quantitative Conservation of Mass.
- **Autonomous Self-Study & Dreaming:** Radial spoke entropy minimization (`AutonomousSelfStudyEngine`), running-share curiosity load-balancing, latent mutex mining, and held-out canary verification during sleep consolidation.
- **Typed Tool Execution:** Pre-dispatch invariant verification (`PreDispatchVerifier`) and SHA-256 provenance grounding (`ToolExecutor`, `ToolGroundingEngine`).
