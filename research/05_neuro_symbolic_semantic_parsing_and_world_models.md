# Research Report 05: Neuro-Symbolic Semantic Parsing, SLM Transducers & Qualitative World Models (2025–2026)

**Target System:** LITTLE Cognitive Architecture (`mivi_model`)  
**Context:** Overcoming the regex bottleneck, enabling open-domain text ingestion, and scaling physical dynamics.  
**Date:** September 2026  

---

## 1. The Core Dilemma: The Fragility of Regex vs. The Hallucination of LLMs

In our evaluation in **Experiment 004**, LITTLE achieved:
- **100.0%** on WordNet multi-hop deduction across 6 semantic hops.
- **100.0%** on exact Python arithmetic and algorithmic execution.
- **0.0%** catastrophic forgetting.
- **0.0%** hallucination (accurate open-world `UNKNOWN` detection).

However, when tested on uncurated Wikipedia sentences:
- Simple declarative sentences (`"A tiger is a mammal"`) scored **100%**.
- Complex real-world sentences (`"Apples originated in Central Asia, where their wild ancestor is still found today"`) scored **0.0%** semantic accuracy because our rule-based regex mangled the clauses into:
  `('apples originated in central asia, where their wild ancestor', 'is_a', 'still found today')`.

To make LITTLE truly intelligent without sacrificing its zero-hallucination, low-latency, and zero-cost advantages, we surveyed the latest 2024–2026 research literature across **four critical domains**.

---

## 2. Breakthrough 1: Small Language Models as Pure Semantic Transducers (Not Thinkers)

### The Paradigm Shift (2025–2026)
Historically, people tried to make LLMs do everything: store facts, do math, reason logically, and speak. This failed because transformer autoregression is fundamentally probabilistic, causing hallucinations and catastrophic forgetting.

Recent work (**KGGen (2025)**, **TextMine (2025)**, and **Fastino/GLiNER2 (2025/2026)**) demonstrates a powerful architectural separation:
> **The Neural Layer does Perception & Parsing. The Symbolic Layer does Memory & Reasoning.**

```
[Raw Uncurated Text] 
         │
         ▼
[Neural Transducer (Edge SLM / GLiNER-Relex)] ──> Emits Strictly Structured JSON Triples
         │
         ▼
[LITTLE MemoryStore & InferenceEngine] ───────> Validates Invariants, Persists to SQLite,
                                                 Executes Exact Multi-Hop Deductions
```

### Key Architectural Implementations:
1. **GLiNER-Relex (2026)**:
   - An open-source, non-generative, encoder-based model that performs joint Named Entity Recognition (NER) and Relation Extraction (RE) in a single forward pass.
   - **Zero-shot**: Accepts arbitrary schema relation labels at runtime (e.g. `is_a`, `part_of`, `originates_in`, `composed_of`).
   - Runs in **under 40 ms on CPU** without GPU acceleration.
2. **Constrained Schema Decoding with Local Edge SLMs (0.5B)**:
   - Using tiny local models (e.g., `Qwen2.5-0.5B` via Ollama/llama.cpp) with strict JSON grammar masks (Pydantic / BNF grammar).
   - The SLM is given a strict prompt:
     `"Extract subject-predicate-object triples from this text. Output JSON only."`
   - The SLM **never** does memory retrieval, **never** guesses answers, and **never** answers questions. It is purely a syntactic-to-semantic converter.

---

## 3. Breakthrough 2: Qualitative Process Theory (QPT) & Universal World Models

### Moving Beyond "Apple-Specific" Physics
Currently, LITTLE's `TransformationEngine.slice_object` hardcodes `exposed_flesh=True` and `interior_color='white'`.

In modern cognitive architectures (Forbus's **Qualitative Process Theory**, Kuipers's **QSIM**, and **OpenCog Hyperon's MeTTa**), physical processes are modeled as **domain-independent ontological schemas**:

### Generic Topological Slicing Schema:
When any composite entity $E$ undergoes the procedural action `SLICE(E, n)`:
1. **Quantity Conservation**: $\sum_{i=1}^n \text{mass}(e_i) = \text{mass}(E)$.
2. **Relational Generation**: $\forall i \in [1, n]: \text{part\_of}(e_i, E) \land \text{is\_a}(e_i, E_{\text{fragment}})$.
3. **Surface Exposure Transformation**:
   - $E$ has an `interior_material` property $M$.
   - The newly created fragments $e_i$ expose material $M$ to ambient conditions:
     $\text{exposed}(e_i, M, \text{ambient\_medium})$.

### Universal Decay & State Evolution:
Instead of hardcoding apple oxidation:
- Every material $M$ has an environmental response profile stored in semantic memory:
  - `organic_flesh`: reacts with $O_2$ (oxidation constant $\tau \approx 3600\,\text{s}$, discoloration $\to$ brown).
  - `iron`: reacts with $O_2 + H_2O$ (oxidation constant $\tau \approx 10^7\,\text{s}$, rust $\to$ reddish-orange).
  - `gold` / `plastic`: inert ($\tau = \infty$, discoloration $\to$ none).
- The continuous ODE engine ([`src/little/dynamics/cfc.py`](file:///home/aswin/programming/vscode/myProjects/ai_agent_tools/mivi_model/src/little/dynamics/cfc.py)) simply looks up $\tau$ dynamically from the material's concept node!

---

## 4. Breakthrough 3: OpenCog Hyperon & AtomSpace Inspiration

### The Lessons from OpenCog Hyperon (2025–2026):
OpenCog Hyperon uses **MeTTa** (Meta Type Talk) over an **AtomSpace metagraph**:
1. **Code and Data are the Same**:
   - In Hyperon, a procedural function (like `fibonacci` or `is_prime`) and a declarative fact (like `dog is_a animal`) are both represented as atoms/nodes in the same graph.
   - LITTLE already started doing this with the `procedures` table in SQLite, where procedural Python skills are stored and queried alongside semantic concepts.
2. **Probabilistic Logic Networks (PLN)**:
   - Instead of binary True/False, edges carry confidence intervals $(\text{strength}, \text{weight})$.
   - LITTLE's NARS evidence formulation ($c = \frac{w^+}{w^+ + w^- + 1}$) is conceptually aligned with PLN, providing a sound foundation for open-world uncertainty.

---

## 5. Proposed Roadmap for LITTLE v0.2

| Component | Current v0.1 State | Proposed v0.2 Evolution | Expected Impact |
|---|---|---|---|
| **Text Ingestion** | Heuristic Regex (`SimpleParser`) | **Dual-Engine Parser**: Fast Regex for clean statements + Local 0.5B SLM / GLiNER for complex uncurated sentences | Complex Wikipedia accuracy jumps from **0% to >85%** |
| **Material Physics** | Hardcoded apple oxidation | **Ontological Material Profiles**: Dynamic lookup of $\tau$, interior color, and decay kinetics | Slicing works universally for tomatoes, bread, iron, or wood |
| **Active Curiosity** | Rule-based question generator | **Information-Theoretic Question Ranking**: Targets highest-entropy nodes first | Optimizes learning rate in dialogue |
| **Procedural Memory** | Seeded Python whitelist | **AST Discovery & Procedural Synthesis**: Store and compose smaller skills into higher-level workflows | LITTLE composes math skills autonomously |
