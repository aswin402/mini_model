# MIVI Final Broad Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final review findings around ledger-gated action effects, safe special-query inference, compound-question payloads, registry-authoritative properties, and one-pass question handling.

**Architecture:** Preserve direct `TransformationEngine.slice_object()` behavior while adding a staged execution mode that returns declarative relation effects. The action ledger will verify and commit each effect as its own provenance-bearing evidence record before reporting it. Candidate frames will carry all parsed question parts from perception through the controller into the resolver, while legacy direct callers retain their existing parse-on-demand behavior.

**Tech Stack:** Python, SQLite, pytest, Ruff, dataclass contracts, JSON schema registries.

## Global Constraints

- Do not commit or modify `.git`.
- Preserve unrelated dirty-worktree changes.
- Keep action effects and relation schemas data-driven; add no object-specific branches.
- Preserve direct public transformation behavior and legacy `LearningEngine.ask()` compatibility where possible.
- Kernel perception is the only semantic question parse in the kernel path; controller and REPL must consume its payload.

---

### Task 1: Stage transformation relation effects for ledger commits

**Files:**
- Modify: `src/little/core/models.py`
- Modify: `src/little/dynamics/transformations.py`
- Modify: `src/little/memory/ledger.py`
- Modify: `data/schemas/action_schemas.json`
- Test: `tests/unit/test_dynamics.py`
- Test: `tests/unit/test_learning_ledger_integration.py`

**Interfaces:**
- `TransformationEngine.slice_object(..., execution_mode="direct", effect_specs=None) -> SlicingResult` keeps direct mutation by default.
- `SlicingResult.relation_effects` contains declarative `RelationEffect` values.
- `LearningResult.relation_effects` carries staged action effects until the ledger commits them.
- `EvidenceLedger.commit_action()` persists one action decision, then proposes/verifies/commits every accepted relation effect through the ledger.

- [ ] Add `RelationEffect` and `LearningResult.relation_effects` with backward-compatible defaults.
- [ ] Add schema-declared relation effects for `slice` and `split`.
- [ ] Make transformation staged mode skip direct relation and action-experience writes while retaining concepts/entities and returning effects.
- [ ] Make the action executor invoke staged mode and let `commit_action()` create effect evidence, decisions, and one action experience.
- [ ] Add tests proving direct slicing still creates relations, ledger slicing creates effect evidence/decisions, and action execution does not duplicate the transformation experience.

### Task 2: Harden special-query inference

**Files:**
- Modify: `src/little/language/parser.py`
- Test: `tests/unit/test_cognitive_kernel.py`

**Interfaces:**
- `LearningEngine.ask()` returns `InferenceResult` UNKNOWN when a parsed query has no target and no specialized handler produced an answer.

- [ ] Add the explicit `CognitiveKernel(MemoryStore(...)).process("tell me a fact")` regression.
- [ ] Return a safe UNKNOWN result before the standard graph infer call when `target is None`.
- [ ] Verify existing seeded fact behavior remains supported.

### Task 3: Carry compound question payloads through kernel routing

**Files:**
- Modify: `src/little/core/contracts.py`
- Modify: `src/little/language/parser.py`
- Modify: `src/little/language/perception.py`
- Modify: `src/little/inference/thinking_controller.py`
- Modify: `src/little/core/kernel.py`
- Test: `tests/unit/test_cognitive_kernel.py`
- Test: `tests/unit/test_thinking_controller.py`

**Interfaces:**
- `CandidateFrame.parsed_queries` and `CandidateFrame.question_parts` carry compound question data.
- `ThinkingController.run(..., parsed_queries=..., question_parts=...)` forwards the payload without reparsing.
- `LearningEngine.ask(..., parsed_queries=..., question_parts=...)` consumes the payload; direct callers still parse on demand.

- [ ] Add a parser helper that splits and parses compound questions once during perception, resolving anaphora between parts.
- [ ] Populate the frame payload for compound questions and keep `parsed_query` as the first-part compatibility value.
- [ ] Extend controller plan/run payloads and resolver invocation.
- [ ] Use supplied parsed parts in `LearningEngine.ask()` without calling `SimpleParser.parse_question()` again.
- [ ] Add the eagle compound kernel regression and call-count coverage.

### Task 4: Make property schema metadata authoritative

**Files:**
- Modify: `src/little/language/parser.py`
- Test: `tests/unit/test_learning_ledger_integration.py`

**Interfaces:**
- `LearningEngine.commit_frame()` routes a claim to `commit_property()` whenever the registered relation schema has `attribute_key`, regardless of `CandidateClaim.is_property`.
- A candidate property with no attribute schema remains candidate/rejected per policy and never creates a graph edge.

- [ ] Use the ledger registry relation schema before candidate metadata.
- [ ] Synchronize verification with the active ledger registry for adversarial custom registries.
- [ ] Add false-metadata registered-property and true-metadata edge-relation tests.

### Task 5: Remove post-perception question reparsing

**Files:**
- Modify: `src/little/main.py`
- Modify: `src/little/core/kernel.py`
- Modify: `src/little/inference/thinking_controller.py`
- Test: `tests/integration/test_cognitive_kernel_cli.py`
- Test: `tests/unit/test_cognitive_kernel.py`

**Interfaces:**
- The REPL uses `outcome.frame.parsed_query` and never calls `SimpleParser.parse_question()` for normal clarification.
- Kernel semantic question routing passes the frame payload to the controller/resolver.

- [ ] Remove REPL pre-parse and fallback parse.
- [ ] Add monkeypatch/call-count tests covering kernel and REPL boundaries.
- [ ] Preserve direct controller and direct `LearningEngine.ask()` compatibility.

### Task 6: Run final verification and report

**Files:**
- Modify: `.superpowers/sdd/final-fix-report.md`

- [ ] Run focused regressions for all five findings.
- [ ] Run `.venv/bin/pytest -q`, touched Ruff, compileall, and `git diff --check`.
- [ ] Separate pre-existing Ruff/whitespace findings from new findings.
- [ ] Record changed files, verification commands, results, and any remaining issue in the final fix report.

