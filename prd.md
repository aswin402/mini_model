# LITTLE — Product Requirements Document

**Project:** LITTLE  
**Working name:** LITTLE  
**Status:** Research / Architecture Definition  
**Document version:** 0.1  
**Date:** 2026-09-16

---

## 1. Product vision

LITTLE is an experimental continually-learning artificial intelligence system designed around **concepts, memory, reasoning, and experience** rather than around a single giant language model.

The long-term objective is to build a system that can:

- learn new knowledge incrementally;
- retain knowledge across sessions;
- learn concepts from relatively few examples;
- generalize a concept to new situations;
- represent relationships between concepts;
- distinguish known, uncertain, contradictory, and unknown information;
- acquire procedural skills;
- improve through feedback and experience;
- eventually learn multiple domains such as English, mathematics, programming, and vision;
- eventually perform self-directed learning and experimentation.

LITTLE is not intended to be a "small LLM." A language model may eventually be one component, but the architecture must not depend on a pretrained LLM as its hidden source of general knowledge.

---

## 2. Problem statement

Most current AI projects begin with a large pretrained model containing broad prior knowledge and then add prompting, retrieval, fine-tuning, or tools.

LITTLE explores a different question:

> Can a relatively small computational core acquire an expanding body of structured knowledge and skills through interaction, while keeping long-term memory outside the core model?

The project specifically targets the following problems:

1. Continuous knowledge acquisition.
2. Catastrophic forgetting.
3. One/few-example concept learning.
4. Explicit uncertainty and "I don't know."
5. Persistent memory.
6. Separation of general capabilities from acquired knowledge.
7. Transfer of knowledge between related concepts.
8. Learning procedures instead of memorizing input/output pairs.
9. Later, self-directed learning.

---

## 3. Product principles

### 3.1 Knowledge is externalizable

Knowledge should be represented in inspectable structures whenever practical.

### 3.2 Learning is incremental

Adding one concept should not require retraining the entire system.

### 3.3 Uncertainty is first-class

The system must be able to say:

- known;
- likely;
- uncertain;
- contradictory;
- unknown.

### 3.4 Memory is persistent

Knowledge learned in one process should survive restart.

### 3.5 Experience and abstraction are different

The system should distinguish:

- what happened;
- what it believes generally;
- how confident it is;
- what procedure it learned.

### 3.6 Simplicity before scale

We will prefer a simple mechanism that can be measured over a sophisticated mechanism that cannot be understood.

### 3.7 No hidden intelligence

For early experiments, a capability should not secretly come from a huge pretrained model if the experiment is intended to test LITTLE's own learning mechanism.

### 3.8 Scientific evaluation

Every major architectural claim needs an experiment and a baseline.

---

## 4. Target users

### Primary

The project developer/researcher building and evaluating LITTLE.

### Secondary

Future researchers or contributors interested in continual-learning architectures.

### Not a near-term target

A production consumer chatbot. The first versions are research prototypes.

---

## 5. Product scope

### MVP scope

LITTLE v0.1 will:

1. accept simple English statements;
2. extract basic entities, concepts, properties, and relationships;
3. store experiences;
4. create/update concept records;
5. retrieve relevant memories;
6. answer simple questions from learned knowledge;
7. expose uncertainty;
8. persist memory in a local database;
9. survive program restarts;
10. provide inspectable traces of why an answer was produced.

### Out of scope for v0.1

- full general-purpose conversation;
- internet-scale knowledge;
- autonomous web browsing;
- autonomous code execution;
- image understanding;
- audio;
- advanced mathematics;
- autonomous self-modification;
- training a large foundation model.

---

## 6. Domain progression

LITTLE will expand through controlled stages.

### Stage A — Simple English

Facts, entities, properties, relationships, questions.

Example:

> A dog is an animal.

Internal representation:

```text
DOG --is_a--> ANIMAL
```

### Stage B — General English concepts

Categories, attributes, negation, simple temporal statements, simple multi-hop reasoning.

### Stage C — Mathematics

Arithmetic, symbolic expressions, equations, and eventually more advanced mathematical procedures.

### Stage D — Programming

Variables, expressions, control flow, functions, data structures, execution feedback, debugging.

### Stage E — Vision

Objects, attributes, transformations, categories, and concept transfer.

### Stage F — Multimodal learning

Shared concepts connected to text, image, and eventually audio.

### Stage G — Active/lifelong learning

The system identifies uncertainty and seeks information or experiments.

---

## 7. Core user stories

### US-001 — Teach a fact

**Given** an empty memory  
**When** the user says "A cat is an animal."  
**Then** LITTLE stores the relationship.

### US-002 — Retrieve learned knowledge

**Given** the previous fact  
**When** the user asks "Is a cat an animal?"  
**Then** LITTLE answers from memory.

### US-003 — Persist knowledge

**Given** a learned fact  
**When** LITTLE restarts  
**Then** the fact remains available.

### US-004 — Learn a second concept

**Given** cat → animal  
**When** the user teaches dog → animal  
**Then** both concepts remain available.

### US-005 — Unknown detection

**Given** no evidence that a cat is a vehicle  
**When** asked "Is a cat a vehicle?"  
**Then** LITTLE must not fabricate evidence.

### US-006 — Explain an answer

**Given** a conclusion derived from stored relationships  
**When** asked why  
**Then** LITTLE should return the relevant evidence path.

### US-007 — Learn a procedure

**Given** examples of addition  
**When** asked to solve a novel addition problem  
**Then** LITTLE should use a learned procedure rather than only retrieve an identical example.

---

## 8. Functional requirements

### FR-001 Concept storage

The system shall represent concepts with stable identifiers.

### FR-002 Relationship storage

The system shall represent typed relationships such as:

- is_a;
- has;
- has_property;
- part_of;
- causes;
- used_for;
- similar_to;
- different_from.

The relationship vocabulary must remain extensible.

### FR-003 Experience storage

Every learning event should be optionally recorded as an experience containing:

- input;
- interpretation;
- source;
- timestamp;
- resulting updates;
- confidence;
- feedback.

### FR-004 Evidence

Beliefs should be linked to supporting evidence where practical.

### FR-005 Confidence

Claims should have confidence metadata rather than an implicit binary truth state.

### FR-006 Contradictions

The system shall be able to retain conflicting observations without immediately destroying one of them.

### FR-007 Unknown

The inference layer shall support an explicit UNKNOWN result.

### FR-008 Memory persistence

The MVP shall persist state to a local SQLite database.

### FR-009 Explainability

The system shall be able to expose a compact reasoning trace.

### FR-010 Versioned learning events

Knowledge updates should be traceable to learning events.

### FR-011 Reproducibility

Experiments shall record configuration and random seeds when applicable.

---

## 9. Non-functional requirements

### NFR-001 Hardware

The first implementation must run on a Ryzen 7-class CPU with 16 GB RAM.

### NFR-002 GPU independence

Core functionality must not require a GPU.

### NFR-003 Optional acceleration

Neural components may use a GPU when available.

### NFR-004 Inspectability

Core memory structures must be exportable to JSON or another human-readable representation.

### NFR-005 Deterministic core

Where practical, symbolic memory and inference should be deterministic.

### NFR-006 Testability

Every core subsystem must have unit tests.

### NFR-007 Modularity

Perception, memory, inference, learning, and interfaces must be replaceable independently.

---

## 10. Success criteria

The first major milestone is not "good conversation."

It is:

> LITTLE can learn a small set of concepts from interaction, persist them, reason over them, recognize insufficient evidence, and continue learning without retraining its entire core.

A successful v0.1 experiment should demonstrate:

- persistent learning;
- multi-hop retrieval;
- unknown detection;
- evidence tracking;
- basic generalization;
- no catastrophic deletion of old knowledge when new knowledge is added.

---

## 11. Failure criteria

We should consider an approach unsuccessful if:

- it requires a huge pretrained model to perform the target task;
- every new fact requires retraining the complete neural network;
- the system cannot preserve old knowledge;
- it always forces a known-category answer;
- its internal knowledge cannot be inspected;
- evaluation cannot distinguish memorization from generalization.

---

## 12. Long-term vision

The eventual LITTLE system may combine:

```text
Perception
    ↓
Representation
    ↓
Concept formation
    ↓
Memory
    ↓
World model
    ↓
Reasoning
    ↓
Action
    ↓
Feedback
    ↓
Learning
```

The system should gradually become more capable by accumulating structured experience rather than simply becoming a larger static pretrained model.
