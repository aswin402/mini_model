# LITTLE (MIVI) v2.0 — Comprehensive Implementation Plan: Hybrid Adaptive Neural Mesh (ANM)

**Version:** 2.0  
**Architecture:** Hybrid Neuro-Symbolic Cognitive Operating System  
**Budget Target:** < 500 MB RAM + ROM on pure CPU / edge hardware  
**Core Guarantee:** 0.0% Hallucination under Open-World Assumption (`UNKNOWN`), 0.0% Catastrophic Forgetting

---

## 1. Executive Summary & Vision

Current large language models (LLMs like GPT-4, Claude, and Llama) are monolithic statistical next-token guessing machines. While they excel at casual English chat, they suffer from three fatal flaws:
1. **Hallucination:** Statistical guessing without truth verification causes plausible falsehoods (15–40% error rate).
2. **Catastrophic Forgetting:** Fine-tuning on new data disrupts prior weights, destroying previously learned domains.
3. **Massive Compute Waste:** Activating billions of parameters for trivial questions like *"2 + 2"* or *"what is the time"*.

LITTLE (MIVI) v2.0 solves this by implementing an **Adaptive Neural Mesh (ANM)**:
- **Neural Perception & Learned Routing (System 1):** Continuous latent state representation ($S \in \mathbb{R}^{512}$) and a learned neural router that classifies intent and extracts semantic frames without brittle regex rules.
- **Mathematical & Symbolic Verification (System 2):** An Epistemic Directed Acyclic Graph (DAG) with **4 Deterministic Verification Gates** ($I_{\text{DAG}}, I_{\text{mutex}}, I_{\text{sort}}, I_{\text{ground}}$), exact rational Computer Algebra System (CAS), Closed-Form Continuous (CfC) Neural ODE dynamics, and transitive partial orders.
- **Layered Memory Tiers (L0–L3):** Long-term persistent memory inspired by Google Research's **Titans** and DeepSeek's low-rank relational storage.
- **Typed Action Boundary:** Needle-style typed tool and code execution with cryptographic SHA-256 provenance.

---

## 2. Cross-Model Research Synthesis: SOTA Inspirations

Our architecture synthesizes key breakthroughs from frontier models and research papers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SOTA RESEARCH INSPIRATIONS                             │
├─────────────────────┬───────────────────────┬──────────────────────────────────────────┤
│ Model / Research    │ Key Mechanism         │ Architectural Integration in LITTLE      │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Laya & Jev          │ Non-autoregressive    │ Fast System 1 gatekeeper; evaluates      │
│ (ModernBERT 421M)   │ typed decisions       │ intent in < 9ms P50 with calibrated      │
│                     │ (Choice, Score, Noul) │ probabilities and entropy (H >= 0.35).   │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ DeepSeek-R1         │ Rule-verified         │ 4 Verification Gates replace critic      │
│                     │ thinking & GRPO       │ models: deductive proofs verified by     │
│                     │                       │ DAG acyclicity and type constraints.     │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ DeepSeek-V3 / V4.1  │ Asymmetric compute,   │ Low-rank relational compression: stores  │
│                     │ MLA low-rank KV cache │ experiences as graph triples in SQLite   │
│                     │ compression           │ instead of massive token KV caches.      │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ GLM-5.3-Flash /     │ Dynamic dual-speed,   │ Fast Mode (< 0.2ms index lookup) vs.     │
│ GLM-4               │ bidirectional blank   │ Thinking Mode (bidirectional frontier    │
│                     │ infilling             │ collision search meeting-in-the-middle). │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Google Research     │ Test-time long-term   │ Dual-memory hierarchy: short-term latent │
│ Titans (2025)       │ neural memory         │ workspace (L0) + relational persistent   │
│                     │ (vs. attention)       │ episodic & semantic memory (L1-L2).      │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Liquid AI LFM-2.5   │ Closed-Form           │ Simulates continuous physical state      │
│ (Hasani & Lechner)  │ Continuous (CfC) ODEs │ vectors s(t) across time delta t in      │
│                     │ without stepping      │ closed form with 0% numerical drift.     │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Cactus Needle 3     │ Byte-level grammar    │ Pre-dispatch verification gate (I_sort); │
│                     │ constrained actions   │ type-safe, sandboxed tool dispatch.      │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Karpathy nanochat   │ Staged training &     │ Incremental scale testing (10M -> 30M -> │
│ & autoresearch      │ modular distillation  │ 75M) with automated evaluation.          │
└─────────────────────┴───────────────────────┴──────────────────────────────────────────┘
```

---

## 3. The 5-Layer Hybrid Adaptive Neural Mesh (ANM)

```
                                      USER INPUT
                 ("Yo, can you multiply 25 by 4 and simulate apple spoilage?")
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ LAYER 1: LANGUAGE PERCEPTION & EMBEDDING                                    │
   │ - Tokenizer + Compact Transformer Encoder (ModernBERT / MiniLM / Nano-Core)  │
   │ - Maps noisy, misspelled, casual English into continuous semantic space      │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ LAYER 2: SHARED LATENT WORKSPACE S ∈ R^512                                  │
   │ S = FFN([h_text, h_context, h_working_memory])                              │
   │ Shared semantic workspace for all cognitive pathways                         │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ LAYER 3: LEARNED NEURAL ROUTER (NO REGEXES!)                                │
   │ P(pathway | S) = Softmax(W_2 · GELU(W_1 S + b_1) + b_2)                     │
   │ Outputs: [chat: p1, math: p2, tool: p3, code: p4, poset: p5, physics: p6]   │
   │ If Entropy H >= 0.35: Trigger Socratic Disambiguation                       │
   └──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──┘
                  │              │              │              │              │
    ┌─────────────┘              │              │              │              └─────────────┐
    ▼                            ▼              ▼              ▼                            ▼
 ┌───────────────┐        ┌─────────────┐ ┌───────────┐ ┌─────────────┐              ┌─────────────┐
 │ PATHWAY: MATH │        │PATHWAY: TOOL│ │PATH: CODE │ │PATH: POSETS │              │PATH: DYNAM. │
 │ Symbolic CAS  │        │Needle Action│ │Python AST │ │Strict Order │              │CfC Continuous│
 │ Fractions/CAS │        │Sandboxed REPL││Synthesizer│ │Topological  │              │Neural ODEs  │
 └───────┬───────┘        └──────┬──────┘ └─────┬─────┘ └──────┬──────┘              └──────┬──────┘
         │                       │              │              │                            │
         └───────────────────────┼──────────────┴──────────────┴────────────────────────────┘
                                 ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ LAYER 4: 4 DETERMINISTIC VERIFICATION GATES (SYSTEM 2 BRAIN)                │
   │ - I_DAG: Acyclicity verification on taxonomies (O(V+E))                     │
   │ - I_mutex: Mutual exclusivity constraints (disjoint_with)                   │
   │ - I_sort: Type signature enforcement on tools, relations, and arguments     │
   │ - I_ground: Empirical evidence verification (Open-World Assumption: UNKNOWN)│
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ LAYER 5: 4-TIER PERSISTENT MEMORY ARCHITECTURE                              │
   │ - L0 (Working): Active problem trace, CAS fractions, ODE state vector s(t)  │
   │ - L1 (Episodic): SQLite transaction logs, experiences, tool execution hashes│
   │ - L2 (Semantic): 360° Concept Knot spider-web (Taxonomy, Mereology, Axioms) │
   │ - L3 (Procedural): Tool catalog, AST synthesis grammar, algorithm runners   │
   └──────────────────────────────────────┬──────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ CONVERSATIONAL DIALOGUE SURFACE & NEURAL VERBALIZER                         │
   │ Warm, articulate human dialogue grounded in proven System 2 facts           │
   └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technical Budget & Resource Allocation (< 500 MB)

To run efficiently on standard consumer CPUs, laptops, and edge devices without requiring $5,000 GPUs:

| Component | Architecture / Format | Memory Budget | Latency Target |
| :--- | :--- | :--- | :--- |
| **Neural Encoder & Latent Workspace** | 4-bit quantized ModernBERT / MiniLM | ~180 MB | < 4.0 ms |
| **Neural Router & Heads** | 2-layer MLP projection ($512 \to 256 \to 6$) | ~15 MB | < 0.2 ms |
| **System 2 Symbolic Engine & CAS** | Pure Python + Rust acceleration | ~25 MB | < 1.0 ms |
| **CfC Continuous ODE Simulator** | Closed-form vector math (NumPy / SIMD) | ~15 MB | < 0.5 ms |
| **Layered Memory (SQLite + Cache)** | SQLite WAL + In-Memory Index | ~50 MB | < 0.4 ms |
| **Tokenizer, Runtime & OS Buffers** | BPE / WordPiece runtime | ~65 MB | < 0.5 ms |
| **Total System Footprint** | **Complete Cognitive Ecosystem** | **~350 MB** | **< 7.0 ms P50** |

---

## 5. Detailed Phase-by-Phase Roadmap

### Phase 15: Typed Action & Tool Core (COMPLETED ✅)
- Externalized schemas: `tool_definitions_policy.json` and `action_policy.json`.
- Pre-dispatch verification gate: $I_{\text{sort}}$ and $I_{\text{security\_sandbox}}$.
- Built-in sandboxed tools: `calculator`, `python_repl`, `system_time`, `web_search`, `file_reader`, `file_writer`, `bash_executor`.
- SHA-256 verifiable execution hashing and SQLite empirical evidence grounding ($I_{\text{ground}}$).
- CLI subcommands: `little tool list` and `little tool exec <tool> [--args ...]`.
- Full regression test suite: 497 / 497 passed (100%).

---

### Phase 16: Neural Latent State & Router Engine (`src/little/neural/`)
**Objective:** Replace regex routing with a continuous learned latent workspace ($S \in \mathbb{R}^{512}$) and neural routing head.
1. **Continuous State Representation (`src/little/neural/state.py`):**
   - Encode token embeddings into continuous latent state $S \in \mathbb{R}^{512}$.
   - Fuses user text vector with conversational dialogue context and working memory.
2. **Learned Neural Router Head (`src/little/neural/router.py`):**
   - Lightweight MLP projection: $\mathbf{p} = \text{Softmax}(\mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 S + \mathbf{b}_1) + \mathbf{b}_2)$.
   - Outputs calibrated routing distribution across 6 pathways:
     `[chat, math, tool, code, poset, physics]`.
3. **Entropy & Epistemic Uncertainty Threshold:**
   - Compute Shannon entropy $H = -\sum p_i \ln p_i$.
   - If $H \ge 0.35$, trigger System 2 disambiguation/clarification question instead of guessing.
4. **Hybrid Reconciliation:**
   - Bridge neural router probabilities with existing `HybridPerceptionAdapter` and `CognitiveKernel`.

---

### Phase 17: Neural Semantic Frame & Slot Extractor (Eliminating Regexes)
**Objective:** Replace regex statement and question parsers with token-level semantic frame taggers.
1. **FrameNet-Style Semantic Role Labeling (`src/little/neural/frame_tagger.py`):**
   - Classify input sequence into semantic roles:
     $$\text{"Europa orbits Jupiter"} \longrightarrow \langle\text{Subject: Europa}, \text{Predicate: orbits}, \text{Object: Jupiter}\rangle$$
   - Handles typos, colloquial language, slang, and irregular grammar.
2. **Grammar Invariant Projection:**
   - Map extracted semantic roles directly into `CandidateClaim` and `CandidateFrame` contracts.
   - Run through the 4 Verification Gates ($I_{\text{DAG}}, I_{\text{mutex}}, I_{\text{sort}}, I_{\text{ground}}$).

---

### Phase 18: Programming & AST Code Synthesis Engine
**Objective:** Implement verifiable code generation and execution as an AST machine.
1. **Python AST Synthesizer (`src/little/procedural/code_synthesizer.py`):**
   - Synthesizes valid Python Abstract Syntax Trees for algorithms, math formulas, and data processing.
2. **Sandboxed Execution REPL (`src/little/procedural/code_sandbox.py`):**
   - Runs synthesized AST code in an isolated subprocess with memory and time caps.
3. **Socratic Auto-Repair Loop:**
   - If execution fails (`IndexError`, `TypeError`, `ZeroDivisionError`), the engine inspects the traceback, identifies the failing AST node, applies a semantic repair mutation, and re-executes until tests pass.

---

### Phase 19: Rust Acceleration Core (`rust/little_core` via PyO3)
**Objective:** Accelerate performance-critical graph traversal, transitive closure, and memory indexing in Rust.
1. **Scaffold PyO3 Workspace:** Configure `maturin` and `Cargo.toml`.
2. **Rust Graph Engine:** Implement DAG reachability, transitive closure, bidirectional collision search, and poset topological sort in Rust.
3. **Benchmark:** Achieve **sub-microsecond query resolution (< 5 µs P99)** and scale to > 500,000 concept nodes with zero garbage collection pauses.

---

### Phase 20: Real-World Empirical Benchmark Matrix (LITTLE vs. SLM vs. LLM)
**Objective:** Publish a comprehensive, verifiable benchmark comparing LITTLE against leading SLMs and LLMs.
1. **Benchmark Domains:**
   - **Multi-Turn Dialogue & Anaphora:** CoQA benchmark.
   - **Exact Mathematics:** GSM8K and SVAMP word problems.
   - **Code Synthesis & Execution:** HumanEval (Python).
   - **Typed Tool Calling:** Berkeley Function Calling Leaderboard (BFCL).
2. **Competitors Tested:**
   - Local SLMs: `Qwen2.5:0.5b`, `Llama-3.2:1b`, `Qwen2.5-Coder:1.5b`.
   - Frontier LLMs: `GPT-4o-mini`, `DeepSeek-R1-Distill-7B`.
3. **Metrics Evaluated:**
   - Exact Accuracy %
   - Hallucination Rate % (Target: 0.0% for LITTLE under Open-World Assumption)
   - Catastrophic Forgetting % across sequential domain ingestion
   - P50/P99 Latency (Pure CPU vs. GPU)
   - Total Memory & Power Footprint
4. **Deliverable:** Automated benchmark runner and report in `experiments/006_real_world_sota_benchmark/REPORT.md`.
