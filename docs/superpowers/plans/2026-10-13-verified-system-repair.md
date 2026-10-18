# Verified System Repair Implementation Plan

> For agentic workers: execute independent file-owned batches with the dispatching-parallel-agents workflow; use test-driven-development and verification-before-completion for every behavior change. Integrate dependent batches sequentially.

**Goal:** Repair the verified audit defects and evaluate each change using reproducible, isolated scenarios.

**Architecture:** Preserve the Python symbolic core and optional Rust acceleration. Strengthen execution, evidence, persistence, semantic extraction, and configuration boundaries incrementally. Existing historical plans remain intact.

**Tech stack:** Python >=3.12, SQLite, SymPy, pytest, Ruff, optional Rust/ctypes. Use the existing environment; no automatic dependency installation.

## Global constraints

- Do not commit, push, publish, delete branches, or rewrite Git history.
- Do not modify live `data/little.db` or `data/knowledge_vault` during tests.
- No speculative neural rewrite; learned components require explicit training, persistence, and evaluation contracts.
- Run targeted tests first, without pytest cache or bytecode writes; no test parallelism or expensive repeated workspace checks.
- Preserve source compatibility except documented fixes to unsafe or false-success behavior.
- Every task: reproduce with a failing regression, implement the minimal correction, run related checks, inspect the diff, record practical validation in the tracker.

## Execution and evidence tracking

The authoritative checklist is `docs/superpowers/plans/2026-10-13-verified-system-repair-todo.md`. Status is pending / in progress / verified / blocked. Historical root `implementationplan.md` and `todo.md` point to this repair series rather than replace earlier research history.

### R01: Isolate tests and establish execution contracts

Files: `tests/conftest.py`, the repair specification, README and project instructions. Add temporary vault/database setup before broad tests; ensure default decision heads do not open live state implicitly. Validate using a before/after digest of tracked runtime artifacts. Check: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_package_metadata.py -q`.

### R02: Safe exact arithmetic

Files: `src/little/procedural/math_cas.py`, `tests/unit/test_math_cas.py`, new arithmetic regression tests. Replace arithmetic `eval` with AST traversal permitting finite numeric literals, unary +/- and bounded arithmetic operators, evaluated as Fractions. Reject attribute access, imports, calls, excessive exponents, and malformed expressions. Regressions include `(9007199254740993 / 1) + 0`, nested fractions, division by zero, and the benign import escape. Check math and tool calculator tests; practical check uses CLI calculation in a temporary workspace.

### R03: Tool and code execution boundaries

Files: `src/little/actions/{tool_executor,pre_dispatch_verifier,tool_policy,models}.py`, `src/little/procedural/{code_sandbox,code_policy,runner}.py`, execution regression tests. Enforce confirmation, normalize typed arguments, reject unknown fields, apply effective timeouts, and use constrained worker execution for Python. Validate ASTs in assertion execution, remove shell=True dispatch, and validate command arguments including Git subcommands, paths, and redirection. Implement timezone conversion and bounded byte reads. Do not claim OS isolation. Test benign Python, prohibited imports/file/network access, shell paths, strings representing booleans, output limits, and timeout cleanup.

### R04: Consistent reasoning and verification

Files: `src/little/inference/{engine,dual_speed,invariant_gates,thinking_controller}.py`, reasoning tests. Traverse supported edges only; retain explicit legacy disjoint semantics. Reconcile direct opposition with derived support, preserve evidence-dependent confidence, validate signatures, and make depth bounds consistent. Check the negative-edge cycle reproduction and explicit negative versus transitive positive conclusion. Practical case teaches conflicting zoology facts and inspects status and trace.

### R05: Atomic, idempotent evidence and source provenance

Files: `src/little/memory/{store,ledger,evidence,schema}.py`, `src/little/language/parser.py`, `src/little/active/{stream_learner,crawler,self_study}.py`. Add a transaction boundary that survives nested store calls, unique current promotion decisions, and replay-safe support contributions. Link actual experience IDs; preserve document source IDs. Ensure action failures do not retain accepted status. Preserve customized built-in procedures. Test injected SQLite failures, repeated evidence IDs, document provenance, property supersession, action failure, and file-backed reopening.

### R06: Faithful, safe OKF and wiki synchronization

Files: `src/little/knowledge/{okf,okf_policy,living_wiki,living_wiki_policy}.py`, OKF/wiki tests. Serialize full relation records; validate names/paths and required fields. Preserve polarity, weights, predicate direction, and legacy parsing. Rebuild or reconcile vector indexes and incremental backlinks. Test negative taxonomy, part_of direction, path traversal, deleted-page search, and changed backlinks. Practical scenario exports/imports into distinct temporary stores and compares answers and graph records.

### R07: Reversible bounded conversational memory

Files: `src/little/memory/{headroom,headroom_policy}.py`, `src/little/core/kernel.py`, Headroom tests. Persist raw turns, replace bulky text/token fields on compression, retain retrieval references after eviction, and enforce episode/token limits including oversized newest turns. Distinguish payload estimates from total process memory. Test compression/eviction/decompression, history recall, negative limits, and long multi-turn conversation.

### R08: Honest planning and error outcomes

Files: `src/little/actions/{agentic_planner,planning_policy,models}.py`, plan tests. Reject or expose unrecognized and truncated subgoals; validate step IDs/order/dependencies/max_steps at execution. Resolve variable references in one token-aware pass. Surface grounding failure, partial execution, retries, and diagnostics. Test empty and oversized goals, duplicate IDs, step_1 versus step_10, failed dependencies, and successful calculator/write/read plans in temporary workspaces.

### R09: Semantic extraction and bounded synthesis

Files: `src/little/neural/{frame_extractor,semantic_frame_policy}.py`, `data/schemas/semantic_frame_policy.json`, `src/little/procedural/code_synthesizer.py`, parser/extraction/synthesis tests. Correct contains/composed-of direction; preserve names such as Physics; use uncertainty rather than accept unsupported syntax. Reject unsupported additional algorithm constraints such as factorial modulo 7. Rename or document deterministic extractor boundaries without breaking imports. Evaluate unseen paraphrases and wrong-relation distractors.

### R10: Explicit neural learning and reliable decision contracts

Files: `src/little/neural/{encoder,router,router_policy,decision_head}.py`, neural tests. Make scoring validate labels/distributions; evaluate complete propositions via symbolic grounding rather than token overlap. Expose a router feedback/checkpoint interface and preserve learned weights on reload; remove unused hidden_dim claims. Keep the actual linear architecture and document that its encoder is hashed. Tests include predicate distractors, wrong labels, invalid dimensions, feedback effects, and checkpoint round trips.

### R11: Clustering and retrieval backend parity

Files: `src/little/active/cobweb_engine.py`, `src/little/neural/{turbo_quant,turbo_quant_policy}.py`, `rust/little_core/src/{lib,turbo}.rs`, native bindings and tests. Make COBWEB trial scoring non-mutating and define duplicate-ID behavior. Enforce meaningful thresholds. Support or reject vector dimensions consistently, synchronize add_quantized across backends, validate FFI inputs without panics, and provide explicit handle cleanup. Report payload bytes honestly. Run native checks with at most two Cargo jobs only if rebuilding is necessary; otherwise test Python fallback and existing native parity.

### R12: Hypotheses, bulk import, and observable partial failures

Files: `src/little/active/{self_study,latent_knot_inducer,stream_learner,crawler}.py`, `src/little/knowledge/importer.py`, ledger/store integration. Keep analogical conclusions as candidates until supported, persist donor/proof provenance, remove hardcoded invariant mining claims, and verify imported graph constraints. Reject malformed input records with diagnostics instead of silently inventing results. Replace swallowed failures with report fields/logging. Practical cases include an unsupported bird capability, cyclic imports, and wiki failure after successful learning.

### R13: Configuration, resources, and lifecycle

Files: `src/little/core/{runtime_paths,runtime_policy,kernel}.py`, policy loaders, `src/little/knowledge/living_wiki.py`, `src/little/rust/engine.py`, `src/little/main.py`, `chat.py`, `pyproject.toml`. Package immutable data under an explicit resource root, separate mutable state, scope vaults to databases/workspaces, and consistently inject memory/headroom/tool/wiki policies. Validate booleans and numeric limits; remove or reject unused control fields. Correct native library lookup, avoid implicit database opens, and add cleanup ownership. Validate two independent runtimes and installed-layout resource discovery.

### R14: Full snapshot restoration and truthful evaluation

Files: `src/little/memory/store.py`, `src/little/knowledge/importer.py`, `src/little/main.py`, `src/little/evaluation/comprehensive_matrix.py`, experiment runner and evaluation tests. Add a versioned state snapshot preserving evidence/proof/decision tables and explicit restore, while retaining dataset ingestion. Score each expected numeric/status result and reject constant-answer oracles. Deduplicate benchmark runners, remove invented model comparisons, separate timing/native availability from correctness. Practical check reopens/restores a temporary database and evaluates held-out dialogue, math, planning, unknowns, and persistence.

### R15: Documentation, integration, and final review

Files: `README.md`, `AGENTS.md`, relevant architecture/research claims, root tracker pointers, isolated practical evaluation tests. Describe Python/uv accurately, bounded neural and synthesis capabilities, trust mode, supported storage formats, and actual measured results. Run all affected suites once after targeted checks, inspect diffs for secrets/accidental runtime artifacts, and record exact commands/results. Do not claim completion while tracked defects remain.
