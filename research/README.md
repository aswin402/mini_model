# LITTLE / MIVI Model — Advanced Research Compendium

This directory contains deep technical investigations into external repositories, state-of-the-art models, and novel AI paradigms. These studies provide concrete mathematical, algorithmic, and architectural mechanisms to upgrade the **LITTLE** continually-learning cognitive architecture.

---

## Research Documents Index

### 1. [01. Non-Autoregressive & Edge Engines](01_non_autoregressive_and_edge_engines.md)
* **Investigated Systems:** 
  - [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) (Convai Innovations)
  - [`cactus-compute/needle`](https://github.com/cactus-compute/needle) & Needle 3 (Cactus Compute)
* **Core Topics:** 
  - Non-autoregressive single-forward-pass System 1 decision heads (33ms latency, zero hallucinations).
  - Strictly proper scoring rules (Brier score, RLCD policy gradients) and domain temperature calibration for honest Expected Calibration Error (ECE $\le 0.081$).
  - Simple Attention Networks (SAN) & Monarch Hadamard MLPs.
  - Needle 3 "Intelligence Laddering" (sliceable depth 2–20 layers, 8MB–29MB binary footprint).
  - Grammar-constrained structured extraction on resource-constrained edge devices (down to ESP32).
* **LITTLE Integration:** Deterministic extraction of Subject-Predicate-Object triples, calibrated confidence calculation for the `UNKNOWN` state, and low-latency concept classification.

---

### 2. [02. Ternary Quantization & Hardware Efficiency](02_ternary_quantization_and_hardware_efficiency.md)
* **Investigated Systems:** 
  - [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) (PrismML)
  - BitNet b1.58 paradigm & Hadamard Outlier Suppression
* **Core Topics:** 
  - Ternary weights $\{-1, 0, +1\}$ achieving extreme 1.75–2.25 bits-per-weight compression.
  - 27B parameter multimodal model compressed to 5.95GB–7.67GB, running comfortably within a 16GB RAM budget.
  - 98.2% reasoning retention across thinking benchmarks (AIME26: 95.83, LiveCodeBench: 90.07).
  - Multiplication-free linear algebra: replacing FP16 MACs (~3.7 pJ) with integer addition/subtraction (~0.05 pJ), achieving up to 74× dynamic ALU energy savings.
  - Elimination of the CPU memory-bandwidth bottleneck on consumer AMD Ryzen 7 systems (yielding ~10–14 tok/s).
* **LITTLE Integration:** Local execution of a 27B-class reasoning and perception backbone entirely within 16GB RAM, plus native ternary hyperdimensional concept centroids ($c_k \in \{-1,0,+1\}^d$) enabling sub-2ns bitwise POPCNT similarity searches.

---

### 3. [03. Liquid Foundation Models & Dynamical Systems](03_liquid_foundation_models_and_dynamical_systems.md)
* **Investigated Systems:** 
  - [`LiquidAI/LFM2.5-230M`](https://huggingface.co/LiquidAI/LFM2.5-230M) (Liquid AI)
  - Liquid Neural Networks (LNNs) & Continuous-Time Models (MIT CSAIL)
* **Core Topics:** 
  - 230M parameter edge-first hybrid model: interleaving 8 double-gated 1D convolutions ($k=4$, zero KV cache) with 6 GQA blocks across 14 layers.
  - Over 63%–75% reduction in KV cache footprint ($96\text{ MB}$ at 32K context FP16 vs $1.2\text{ GB}+$ for standard transformers).
  - Ultra-low latency: ~118 tok/s on Ryzen 7 CPU, footprint ~140MB (4-bit) / ~460MB (BF16).
  - Liquid Time-Constant (LTC) continuous ODEs and Closed-Form Continuous-Time (CfC) formulations: dynamic time-constant adaptation and continuous trajectory propagation without stiff numerical solvers.
  - Lyapunov stability and noise dissipation preventing catastrophic forgetting.
* **LITTLE Integration:** Ultra-lightweight sequence/sensory parser frontend (<500MB RAM), continuous physical entity state tracking (e.g. continuous ripening/decay kinetics of an apple across time $\Delta t$), and hybrid automaton discrete action state jumps (`SLICE`).

---

### 4. [04. DeepSeek Architectures & Memory Sparsity](04_deepseek_architectures_and_memory_sparsity.md)
* **Investigated Systems:** 
  - [`deepseek-ai` GitHub repositories](https://github.com/orgs/deepseek-ai/repositories?type=all) (DeepSeek-V3, DeepSeek-R1, Engram, DualPipe, DeepEP, 3FS)
* **Core Topics:** 
  - **Engram**: Conditional memory as a new axis of sparsity. $O(1)$ static knowledge lookup tables stored in host system RAM (DRAM) via multi-head hashing and contextual gating, bypassing GPU memory entirely.
  - **Multi-Head Latent Attention (MLA)**: Low-rank key-value joint compression into a 512-dim latent vector $\mathbf{c}_t^{KV}$, slashing KV cache footprint by 98.24% via weight absorption (full keys and values are never materialized in memory).
  - **DeepSeekMoE**: 256 fine-grained routed experts + isolated shared experts with dynamic routing bias modulation.
  - **DeepSeek-R1 & GRPO**: Group Relative Policy Optimization. Eliminates the memory-heavy Critic network ($V_\phi$), driving spontaneous reasoning chains, backtracking, and self-correction via pure rule-based verifiers.
* **LITTLE Integration:** Engram's host-RAM table serves as the exact neural counterpart to LITTLE's persistent concept memory; MLA provides compact episodic experience traces (100,000 episodes in 32MB RAM); GRPO driven by Semantic Graph Invariants (DAG acyclicity, mutual exclusivity) allows LITTLE to autonomously self-improve and prune invalid beliefs without human labeling.

---

## Architectural Synthesis: The Upgraded LITTLE Stack

```
                          USER / ENVIRONMENT INPUT
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │ PERCEPTION & SEQUENCE PARSER  │
                     │  - LFM2.5-230M / Needle 3 SAN │
                     │  - GBNF Grammar Constraints   │
                     │  - Footprint: < 200 MB RAM    │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                     ┌───────────────────────────────┐
                     │   FAST SYSTEM 1 DECISION      │
                     │  - Laya Non-Autoregressive    │
                     │  - Primitives: choice, noul   │
                     │  - Calibrated ECE <= 0.081    │
                     │  - Latency: ~30 ms            │
                     └───────┬───────────────┬───────┘
                             │               │
         High Confidence     │               │  Low Confidence / Unknown
         (Supported Fact)    │               │  (Trigger Investigation)
                             ▼               ▼
            ┌───────────────────┐    ┌──────────────────────────────────┐
            │ PERSISTENT MEMORY │    │ DEEP SYSTEM 2 REASONING          │
            │  - SQLite Engine  │    │  - Ternary Bonsai 2 (27B, 2-bit) │
            │  - Engram O(1)    │    │  - MLA Latent KV Compression     │
            │  - Rust petgraph  │    │  - GRPO Graph Invariant Verifier │
            │  - Dynamic CfC    │    │  - Footprint: ~6.0 GB RAM        │
            └───────────────────┘    └──────────────────────────────────┘
```
