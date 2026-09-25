# Query Boundary Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the final review repairs for safe special-query inference, compound-query payload propagation, registry-authoritative property commits, and one-pass semantic question handling.

**Architecture:** Perception parses a question once and stores typed query parts on `CandidateFrame`. The kernel passes those fields through the controller to the resolver; direct legacy controller and `LearningEngine.ask()` calls retain parse-on-demand behavior when no payload is supplied. Commit routing consults the active ledger schema before candidate metadata.

**Tech Stack:** Python, dataclasses, SQLite-backed memory, pytest.

## Global Constraints

- Modify only the requested query-boundary files, focused tests, this plan, and `.superpowers/sdd/query-boundary-fix-report.md`.
- Do not modify action ledger or dynamics files.
- Preserve public APIs with optional parameters and preserve direct legacy callers.
- Keep query payload fields typed/data-driven; do not add domain-specific branches.
- Do not commit or modify `.git`.

---

### Task 1: Establish regressions for safe special queries and compound payloads

**Files:**
- Modify: `tests/unit/test_cognitive_kernel.py`
- Modify: `tests/unit/test_perception_adapter.py`
- Modify: `tests/unit/test_thinking_controller.py`

**Interfaces:**
- `CandidateFrame` will expose optional structured compound query fields while keeping `parsed_query` compatible.
- Kernel questions will pass the perceived payload to the controller and resolver.

- [ ] Add a kernel regression that processes `tell me a fact` on an empty in-memory store and asserts an UNKNOWN or supported `InferenceResult`, never an exception.
- [ ] Add a perception regression that asserts the eagle compound question contains both parsed query parts in the frame payload.
- [ ] Add a kernel regression with `eagle can fly` and `eagle has wing` proving the compound answer remains supported.
- [ ] Add monkeypatch/call-count assertions that the controller and resolver consume the perceived payload without reparsing.
- [ ] Run the focused tests and confirm they fail for the missing payload/special-query behavior before implementation.

### Task 2: Add typed compound-question payload fields and parse once in perception

**Files:**
- Modify: `src/little/core/contracts.py`
- Modify: `src/little/language/perception.py`
- Modify: `src/little/language/parser.py`

**Interfaces:**
- Add `CandidateFrame.parsed_queries: list[ParsedQuery]` and `CandidateFrame.question_parts: list[str]` with empty-list defaults.
- Add a parser helper that returns question text parts and parsed tuples from one semantic split, resolving anaphora using dialogue/current subject state.
- Keep `parsed_query` as the first parsed tuple for existing callers.

- [ ] Implement the smallest typed dataclass-field extension with backward-compatible defaults.
- [ ] Extract compound splitting/parsing into a reusable parser helper without changing direct `ask()` behavior.
- [ ] Make perception use that helper for conjoined questions and store all successfully parsed parts plus their source parts.
- [ ] Preserve the explicit `None` behavior for unparsed single questions.
- [ ] Run the perception/parser regressions and verify they pass.

### Task 3: Forward payloads through controller, kernel, and resolver

**Files:**
- Modify: `src/little/inference/thinking_controller.py`
- Modify: `src/little/core/kernel.py`
- Modify: `src/little/language/parser.py`

**Interfaces:**
- Extend `ThinkingController.plan/run` with optional `parsed_queries` and `question_parts` parameters.
- Extend the resolver callable boundary with optional payload parameters while accepting legacy one-argument resolvers.
- Extend `LearningEngine.ask` with optional `parsed_queries` and `question_parts` parameters.

- [ ] Let the controller plan from the first supplied parsed tuple and never call `_parse_question` when a payload is supplied.
- [ ] Pass frame payload fields from `CognitiveKernel.process` into controller execution.
- [ ] Make `LearningEngine.ask` consume supplied compound parts instead of splitting/reparsing raw text; retain existing parse-on-demand logic when payload is absent.
- [ ] Preserve legacy injected resolvers and direct callers through optional arguments/adaptation.
- [ ] Run kernel/controller/conversational compound tests and verify no extra parser calls occur.

### Task 4: Make schema metadata authoritative at `commit_frame`

**Files:**
- Modify: `src/little/language/parser.py`
- Modify: `tests/unit/test_learning_ledger_integration.py`

**Interfaces:**
- `LearningEngine.commit_frame` will consult `self.memory.ledger.registry.relation(predicate)` for property routing.

- [ ] Add a failing test where a registered `attribute_key` relation is marked `is_property=False` and assert attributes update without a graph edge.
- [ ] Add a failing test where an unregistered/non-attribute relation is marked `is_property=True` and assert candidate/rejected status with no graph edge or attribute write.
- [ ] Route registered attribute relations through `commit_property` regardless of candidate metadata.
- [ ] Route only schema-valid graph relations through `commit_relation`; keep unsupported property metadata candidate/rejected and avoid mutation.
- [ ] Use the active ledger registry for verification so custom registry tests are authoritative.
- [ ] Run focused ledger integration tests.

### Task 5: Remove post-perception semantic reparsing in the REPL

**Files:**
- Modify: `src/little/main.py`
- Modify: `tests/integration/test_cognitive_kernel_cli.py`

**Interfaces:**
- REPL clarification uses `outcome.frame.parsed_query` and the frame payload only; no pre-parse or fallback parse for normal questions.

- [ ] Remove the REPL's `SimpleParser.parse_question` pre-parse and fallback calls.
- [ ] Use the frame payload to call `ActiveInquisitor.inspect_uncertainty` when clarification is needed.
- [ ] Add a monkeypatch regression proving normal REPL question handling does not call `SimpleParser.parse_question` after perception.
- [ ] Run focused CLI tests.

### Task 6: Verify and report

**Files:**
- Create: `.superpowers/sdd/query-boundary-fix-report.md`

- [ ] Run focused regressions for all four boundary repairs.
- [ ] Run the full unit suite and build/compile checks available in the repository.
- [ ] Inspect the final diff and confirm action-ledger/dynamics files are untouched by this task.
- [ ] Record changed files, commands, pass/fail results, and any pre-existing issues in the requested report.
