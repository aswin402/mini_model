# LITTLE — CLI Surface, Cognitive Directives & Policy Schemas

---

## 1. CLI Command Surface (`src/little/main.py`)

Invoke via `uv run little [--db <path>] <subcommand>`:

| Subcommand | Example | Description |
| :--- | :--- | :--- |
| `init` | `uv run little init` | Initialize SQLite schema and seed base ontology |
| `learn` | `uv run little learn "A robin is a bird."` | Parse and commit a declarative fact through the 4 invariant gates |
| `ask` | `uv run little ask "Is a robin an animal?" -v --thinking` | Run dual-speed inference with optional reasoning and frontier trace |
| `status` | `uv run little status [--json]` | Display live telemetry dashboard (Memory, Paged Store, Plasticity, 5-Brake Growth Gate, Held-Out Canary) |
| `chat` / `interact` | `uv run little chat [--thinking] [--auto-study]` | Launch interactive continual-learning REPL |
| `inspect` | `uv run little inspect concept robin` | Inspect a concept's attributes, confidence, and incoming/outgoing relations |
| `memory list` | `uv run little memory list` | List stored concepts and semantic relations |
| `memory prune` | `uv run little memory prune [--dry-run]` | Prune polluted test/synthetic concepts from a database |
| `curious` | `uv run little curious [--top 10] [--concept <name>]` | Rank concepts by 360° radial spoke entropy $H(c)$ |
| `study` | `uv run little study [--steps 5] [--target <name>] -v` | Run closed-loop analogical self-study cycle |
| `skills list` | `uv run little skills list` | List registered procedural Python skills |
| `tool list` | `uv run little tool list` | List typed tools and sandbox configurations |
| `tool exec` | `uv run little tool exec calculator --args '{"expression": "2+2"}'` | Execute a typed tool with SHA-256 provenance hash |
| `import` | `uv run little import --dataset commonsense` | Ingest curated world-knowledge triples or external CSV/TSV/JSON files |
| `export` | `uv run little export -o state.json` | Export persistent cognitive memory state to JSON |

---

## 2. Interactive Cognitive Directives (`CognitiveCommandRegistry`)

Inside `little chat` or `CognitiveKernel.chat()`, natural-language meta-cognitive directives are dispatched by `src/little/core/cognitive_command.py`:

- **Telemetry & Status:** `"show cognitive status"`, `"telemetry status"`, `"plasticity status"`, `"growth brakes"`
- **Knowledge Ingestion:** `"study text: <paragraph>"`, `"crawl document: <paragraph>"`
- **Living Wiki Search & Navigation:** `"search wiki for <query>"`, `"what links to <concept>?"`, `"show wiki page for <concept>"`
- **Latent Concept Induction (Cobweb):** `"induce latent knots"`, `"show latent categories"`, `"explain concept cluster for <concept>"`
- **Autonomous Dreaming & Epistemic Gaps:** `"run a dream cycle"`, `"what are your knowledge gaps?"`, `"study concept <concept>"`
- **Continuous-Time Physical Dynamics (`CfC`):** `"what happens to an apple after 4 hours at 30°C?"`, `"slice an apple into 4 pieces"`, `"peel an apple"`, `"heat an apple"`
- **Sensory Grounding:** `"observe sensor: <reading>"`

---

## 3. Externalized JSON Policy Schemas (`data/schemas/`)

All linguistic patterns, thresholds, and ontological rules are externalized under `data/schemas/`:

- `action_policy.json` & `tool_definitions_policy.json` — Typed tool schemas, argument sorts, and sandbox boundaries.
- `algebraic_word_problem_policy.json` & `math_story_policy.json` — Symbolic math and word-problem extraction patterns.
- `comparative_order_policy.json` — Strict partial-order (`poset`) comparative adjectives and polarity mappings.
- `counterfactual_policy.json` — Counterfactual and physical transformation triggers.
- `dialogue_surface_policy.json` — Conversational templates, social intents, and pronoun verbalization rules.
- `headroom_policy.json` & `memory_policy.json` — Working-memory salience weights, CCR compression limits, and 3-tier paging capacities.
- `invariant_gate_policy.json` — Deterministic verification gate rules ($I_{\text{DAG}}, I_{\text{MUTEX}}, I_{\text{SORT}}, I_{\text{TEMP}}$).
- `laya_calibration_policy.json` & `neural_router_policy.json` — Epistemic entropy thresholds and neural pathway routing config.
- `living_wiki_policy.json` & `okf_policy.json` — Obsidian Knowledge Format (`.okf.md`) serialization and vault sync settings.
- `self_study_policy.json` — 6-spoke curiosity axes and analogical donor predicates.
- `turbo_quant_policy.json` — Subsampled Randomized Hadamard Transform (`FWHT`) and 4-bit/1-bit quantization parameters.
