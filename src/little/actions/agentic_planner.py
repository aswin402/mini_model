"""Multi-Step Agentic Planning Loop and ReAct Execution Engine (Needle Action Chains).

Transforms LITTLE from a single-turn tool caller into an autonomous multi-step problem solver
with goal decomposition, variable binding across chained steps, DeepSeek-R1 invariant gates
(I_DAG, I_sort, I_mutex, I_ground), and automatic backtracking on step failures.
"""

from __future__ import annotations

import datetime
import sqlite3
import sys
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from typing import Any

from little.actions.action_router import ActionRouter, _is_file_path_token
from little.actions.grounding import ToolGroundingEngine
from little.actions.models import (
    AgenticPlanExecutionResult,
    Plan,
    PlanStep,
    PlanStepStatus,
    ToolCall,
    ToolExecutionResult,
)
from little.actions.planning_policy import AgenticPlanningPolicy
from little.actions.pre_dispatch_verifier import PreDispatchVerifier
from little.actions.tool_executor import ToolExecutor
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy
from little.memory.store import MemoryStore


def _has_bounded_phrase(text: str, phrase: str) -> bool:
    low = text.lower()
    target = phrase.lower()
    start = 0
    while True:
        idx = low.find(target, start)
        if idx == -1:
            return False
        end = idx + len(target)
        left_ok = idx == 0 or not (low[idx - 1].isalnum() or low[idx - 1] == "_")
        right_ok = end == len(low) or not (low[end].isalnum() or low[end] == "_")
        if left_ok and right_ok:
            return True
        start = idx + 1


def _mask_quoted_spans(text: str) -> str:
    chars = list(text)
    n = len(chars)
    i = 0
    while i < n:
        if chars[i] in ('"', "'"):
            quote = chars[i]
            j = i + 1
            while j < n:
                if chars[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if chars[j] == quote:
                    j += 1
                    break
                j += 1
            for k in range(i, j):
                chars[k] = " "
            i = j
        else:
            i += 1
    return "".join(chars)


def _find_variable_tokens(text: str) -> list[tuple[int, int, int | None]]:
    """Scan text for $step_N[.output] or $(prev|previous|result|output)[.output] tokens."""
    spans: list[tuple[int, int, int | None]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "$":
            i += 1
            continue
        end = -1
        step_idx: int | None = None
        if text.startswith("$step_", i):
            j = i + 6
            while j < n and text[j].isdigit():
                j += 1
            if j > i + 6:
                step_idx = int(text[i + 6 : j])
                end = j
        else:
            for kw in ("$previous", "$prev", "$result", "$output"):
                if text.startswith(kw, i):
                    end = i + len(kw)
                    break
        if end == -1:
            i += 1
            continue
        if text.startswith(".output", end):
            end += 7
        if end < n and (text[end].isalnum() or text[end] in "_."):
            i += 1
            continue
        spans.append((i, end, step_idx))
        i = end
    return spans


class AgenticPlanner:
    """Decomposes compound user goals into ordered plans and executes them via ReAct loops."""

    def __init__(
        self,
        action_router: ActionRouter | None = None,
        tool_executor: ToolExecutor | None = None,
        tool_grounder: ToolGroundingEngine | None = None,
        policy: AgenticPlanningPolicy | None = None,
        tool_policy: ToolDefinitionsPolicy | None = None,
        action_policy: ActionPolicy | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.tool_policy = tool_policy or ToolDefinitionsPolicy.default()
        self.action_policy = action_policy or ActionPolicy.default()
        self.policy = policy or AgenticPlanningPolicy.default()
        self.action_router = action_router or ActionRouter(
            action_policy=self.action_policy, tool_policy=self.tool_policy
        )
        self.tool_executor = tool_executor or ToolExecutor(
            tool_policy=self.tool_policy, action_policy=self.action_policy
        )
        self.tool_grounder = tool_grounder or ToolGroundingEngine(memory_store=memory)
        self.verifier = PreDispatchVerifier(self.tool_policy, self.action_policy)

    def is_compound_goal(self, utterance: str) -> bool:
        """Determine if an utterance represents a multi-step agentic plan rather than a single tool call."""
        text = utterance.strip()
        if not text:
            return False

        # Must contain at least one conjunction delimiter or step marker
        has_conjunction = (
            any(
                _has_bounded_phrase(text, d)
                for d in self.policy.conjunction_delimiters
                if d != ";"
            )
            or ";" in text
        )

        words_low = [w.strip(",:;?.!") for w in text.lower().split()]
        has_step_marker = (
            any(w in ("first", "second") for w in words_low)
            or any(
                words_low[i] == "step" and words_low[i + 1].isdigit()
                for i in range(len(words_low) - 1)
            )
            or any(
                line.lstrip() and line.lstrip().split(" ", 1)[0].endswith(".") and line.lstrip().split(" ", 1)[0][:-1].isdigit()
                for line in text.splitlines()
            )
        )

        action_verbs = {"write", "save", "read", "run", "exec", "compute", "calculate", "search"}
        has_compound_action = any(
            words_low[i] in ("and", "then") and words_low[i + 1] in action_verbs
            for i in range(len(words_low) - 1)
        )

        if not (has_conjunction or has_step_marker or has_compound_action):
            return False

        # Preserve partially recognized compound goals so incomplete work is rejected
        # explicitly by execution instead of falling back to a single tool action.
        plan = self.decompose_goal(text)
        return bool(plan.steps) and len(self._split_compound_utterance(text)) >= 2

    def decompose_goal(self, goal: str) -> Plan:
        """Deconstruct a compound goal utterance into an ordered Directed Acyclic Graph (DAG) of PlanSteps."""
        raw_subgoals = self._split_compound_utterance(goal)
        steps: list[PlanStep] = []
        diagnostics: list[str] = []
        if not goal.strip():
            diagnostics.append("Goal is empty")

        for idx, sub_text in enumerate(raw_subgoals, start=1):
            cleaned = self._clean_subgoal_text(sub_text)
            if not cleaned:
                diagnostics.append(f"Subgoal {idx} is empty")
                continue

            tool_call = self._resolve_subgoal_tool_call(
                cleaned, prev_step_index=idx - 1
            )
            if tool_call is None:
                diagnostics.append(f"Unrecognized subgoal {idx}: {cleaned}")
                continue

            # Identify dependencies: step idx depends on prev step if variables reference previous output
            deps: list[int] = []
            args_template = dict(tool_call.arguments)

            for arg_key, arg_val in args_template.items():
                if isinstance(arg_val, str):
                    if any(
                        p in arg_val
                        for p in (
                            "$prev",
                            "$result",
                            "$output",
                            "the result",
                            "the output",
                            "the answer",
                        )
                    ):
                        if idx > 1:
                            deps.append(idx - 1)
                    for _, _, step_ref in _find_variable_tokens(arg_val):
                        if step_ref is not None and 1 <= step_ref < idx:
                            deps.append(step_ref)

            # If no explicit variable reference but sequential action, default dependency on previous step
            if not deps and idx > 1:
                deps.append(idx - 1)

            step = PlanStep(
                step_index=idx,
                description=cleaned,
                tool_name=tool_call.tool_name,
                arguments_template=args_template,
                dependencies=tuple(sorted(set(deps))),
                status=PlanStepStatus.PENDING,
                confirmed=tool_call.confirmed,
            )
            steps.append(step)

        # Enforce max step bound from policy
        if len(steps) > self.policy.max_steps:
            diagnostics.append(
                f"Plan exceeds max_steps={self.policy.max_steps}; no steps executed"
            )

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        created_at = datetime.datetime.now(datetime.UTC).isoformat()
        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=tuple(steps),
            created_at=created_at,
            diagnostics=tuple(diagnostics),
        )

    def plan_and_execute(self, goal: str) -> AgenticPlanExecutionResult:
        """Decompose goal into a DAG of steps and execute sequentially through ReAct loop."""
        plan = self.decompose_goal(goal)
        return self.execute_plan(plan)

    def _variable_dependencies(
        self, arguments: dict[str, Any], previous: int
    ) -> set[int]:
        references: set[int] = set()
        for value in arguments.values():
            if isinstance(value, str):
                for _, _, step_idx in _find_variable_tokens(value):
                    references.add(step_idx if step_idx is not None else previous)
        return references

    def execute_plan(self, plan: Plan) -> AgenticPlanExecutionResult:
        """Execute an ordered DAG, exposing incomplete, partial, and provenance failures."""
        start_time = time.perf_counter()
        invariants_checked: list[str] = ["I_dag_acyclicity"]
        diagnostics = list(plan.diagnostics)
        # Execution state belongs to this run, never to the reusable plan.
        steps = tuple(
            replace(
                step,
                arguments_template=deepcopy(step.arguments_template),
                status=PlanStepStatus.PENDING,
                retries=0,
                result=None,
                observation=None,
            )
            for step in plan.steps
        )
        provenance_available = (
            getattr(self.tool_grounder, "memory_store", None) is not None
        )

        def outcome(
            success: bool,
            error: str | None = None,
            output: Any = None,
            retries: int = 0,
            backtracks: int = 0,
        ) -> AgenticPlanExecutionResult:
            return AgenticPlanExecutionResult(
                plan_id=plan.plan_id,
                goal=plan.goal,
                steps=steps,
                success=success,
                final_output=output,
                error=error,
                total_execution_time_ms=(time.perf_counter() - start_time) * 1000,
                trace_hash=AgenticPlanExecutionResult.compute_trace_hash(
                    plan.plan_id, steps, success
                ),
                invariants_checked=tuple(sorted(set(invariants_checked))),
                retries_used=retries,
                backtracks_count=backtracks,
                diagnostics=tuple(diagnostics),
                provenance_available=provenance_available,
            )

        if not plan.goal.strip() or not steps:
            diagnostics.append(
                "Plan requires a nonempty goal and at least one actionable step"
            )
        if len(steps) > self.policy.max_steps:
            diagnostics.append(f"Plan exceeds max_steps={self.policy.max_steps}")
        if not plan.is_acyclic():
            diagnostics.append(
                "I_DAG violation: step IDs must be positive, unique, ordered, with existing earlier dependencies"
            )
        seen: set[int] = set()
        dependencies: dict[int, set[int]] = {}
        for step in steps:
            if step.tool_name not in self.policy.chainable_tools:
                diagnostics.append(
                    f"Step {step.step_index}: tool {step.tool_name} is not chainable"
                )
            refs = self._variable_dependencies(
                step.arguments_template, step.step_index - 1
            )
            missing = refs - seen
            if missing:
                diagnostics.append(
                    f"Step {step.step_index}: missing or forward output references {sorted(missing)}"
                )
            dependencies[step.step_index] = set(step.dependencies) | refs
            seen.add(step.step_index)
        if diagnostics:
            return outcome(False, "Plan rejected: " + "; ".join(diagnostics))

        invariants_checked.append("I_mutex_resource")
        mutex_error = self._check_mutex_invariants(plan)
        if mutex_error:
            diagnostics.append(mutex_error)
            return outcome(
                False, f"Plan rejected: resource mutex violation: {mutex_error}"
            )
        if not provenance_available:
            diagnostics.append(
                "Provenance unavailable: execution-only planner has no memory store"
            )

        outputs: dict[int, Any] = {}
        total_retries = total_backtracks = 0
        failures: list[str] = []
        for step in steps:
            if any(dep not in outputs for dep in dependencies[step.step_index]):
                step.status = PlanStepStatus.SKIPPED
                step.observation = "Skipped because a dependent step failed"
                diagnostics.append(f"Step {step.step_index}: {step.observation}")
                continue
            step.status = PlanStepStatus.RUNNING
            try:
                arguments = self._resolve_argument_variables(
                    step.tool_name,
                    step.arguments_template,
                    outputs,
                    step.step_index - 1,
                )
            except (ValueError, TypeError) as exc:
                step.status = PlanStepStatus.FAILED
                step.observation = str(exc)
                failures.append(f"Step {step.step_index}: {exc}")
                continue
            last_result = None
            for attempt in range(self.policy.max_retries_per_step + 1):
                call = ToolCall(
                    tool_name=step.tool_name,
                    arguments=arguments,
                    confirmed=step.confirmed,
                )
                invariants_checked.extend(("I_sort_type", "I_security_sandbox"))
                verification = self.verifier.verify(call)
                if not verification.valid:
                    last_result = ToolExecutionResult(
                        step.tool_name,
                        arguments,
                        False,
                        error=verification.error,
                        verifiable_hash=ToolExecutionResult.compute_hash(
                            step.tool_name, arguments, None, False
                        ),
                    )
                else:
                    try:
                        last_result = self.tool_executor.execute(
                            call, timeout_seconds=self.policy.step_timeout_seconds
                        )
                    except (ValueError, TypeError, OSError, RuntimeError) as exc:
                        last_result = ToolExecutionResult(
                            step.tool_name,
                            arguments,
                            False,
                            error=f"Dispatch raised {type(exc).__name__}: {exc}",
                        )
                if last_result.success:
                    break
                diagnostics.append(
                    f"Step {step.step_index} attempt {attempt + 1}: {last_result.error}"
                )
                if (
                    attempt >= self.policy.max_retries_per_step
                    or not self.policy.enable_backtracking
                ):
                    break
                corrected = self._attempt_self_correction(
                    step.tool_name, arguments, last_result.error or ""
                )
                if corrected is None or corrected == arguments:
                    break
                arguments = corrected
                step.retries += 1
                total_retries += 1
                total_backtracks += 1
            step.result = last_result
            if last_result is not None and last_result.success:
                if provenance_available:
                    try:
                        grounded = self.tool_grounder.ground(last_result)
                    except (
                        ValueError,
                        TypeError,
                        OSError,
                        RuntimeError,
                        sqlite3.Error,
                    ) as exc:
                        grounded = False
                        diagnostics.append(
                            f"Step {step.step_index}: grounding raised {type(exc).__name__}: {exc}"
                        )
                    if not grounded:
                        step.status = PlanStepStatus.FAILED
                        step.observation = "Tool completed but grounding failed; effects may already exist"
                        failures.append(f"Step {step.step_index}: grounding failed")
                        diagnostics.append(step.observation)
                        continue
                    invariants_checked.append("I_ground_provenance")
                step.status = PlanStepStatus.SUCCESS
                step.observation = str(last_result.output)
                outputs[step.step_index] = last_result.output
            else:
                step.status = PlanStepStatus.FAILED
                step.observation = f"Execution failed: {last_result.error if last_result else 'No execution result'}"
                failures.append(
                    f"Step {step.step_index} ({step.tool_name}) failed: {step.observation}"
                )
        success = all(step.status == PlanStepStatus.SUCCESS for step in steps)
        diagnostics.extend(failures)
        final_output = outputs.get(steps[-1].step_index) if success else None
        return outcome(
            success,
            "; ".join(failures) or None,
            final_output,
            total_retries,
            total_backtracks,
        )

    # -------------------------------------------------------------------------
    # Decomposition & Parsing Helpers
    # -------------------------------------------------------------------------

    def _split_compound_utterance(self, text: str) -> list[str]:
        """Split compound sentence into sub-goal clauses while respecting quoted expressions."""
        clean = text.strip()
        masked = _mask_quoted_spans(clean)
        low_masked = masked.lower()
        n = len(masked)

        # 1. Check for numbered "step <digits>[:]" markers
        step_spans: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if (i == 0 or low_masked[i - 1].isspace()) and low_masked.startswith("step", i):
                j = i + 4
                if j < n and low_masked[j].isspace():
                    while j < n and low_masked[j].isspace():
                        j += 1
                    k = j
                    while k < n and low_masked[k].isdigit():
                        k += 1
                    if k > j:
                        if k < n and low_masked[k] == ":":
                            k += 1
                        start_pos = i - 1 if (i > 0 and low_masked[i - 1].isspace()) else i
                        step_spans.append((start_pos, k))
                        i = k
                        continue
            i += 1

        if step_spans:
            parts: list[str] = []
            cursor = 0
            for sp_start, sp_end in step_spans:
                piece = clean[cursor:sp_start].strip()
                if not (cursor == 0 and not piece):
                    parts.append(piece)
                cursor = sp_end
            parts.append(clean[cursor:].strip())
            return parts

        # 2. Check for "<digits>. " at start of line
        num_dot_spans: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if (i == 0 or low_masked[i - 1] == "\n") and low_masked[i].isdigit():
                j = i
                while j < n and low_masked[j].isdigit():
                    j += 1
                if j < n and low_masked[j] == "." and j + 1 < n and low_masked[j + 1].isspace():
                    k = j + 1
                    while k < n and low_masked[k].isspace():
                        k += 1
                    start_pos = i - 1 if (i > 0 and low_masked[i - 1] == "\n") else i
                    num_dot_spans.append((start_pos, k))
                    i = k
                    continue
            i += 1

        if num_dot_spans:
            parts = []
            cursor = 0
            for sp_start, sp_end in num_dot_spans:
                piece = clean[cursor:sp_start].strip()
                if not (cursor == 0 and not piece):
                    parts.append(piece)
                cursor = sp_end
            parts.append(clean[cursor:].strip())
            return parts

        # 3. Conjunction delimiters and ",? and <action_verb>"
        delims = [
            d.lower()
            for d in sorted(self.policy.conjunction_delimiters, key=len, reverse=True)
            if d != ";"
        ]
        action_verbs = ("write", "save", "read", "run", "exec", "calculate", "compute", "search")

        split_spans: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if low_masked[i] == ";":
                split_spans.append((i, i + 1))
                i += 1
                continue

            matched_delim = False
            for d in delims:
                if low_masked.startswith(d, i):
                    end = i + len(d)
                    left_ok = i == 0 or not (low_masked[i - 1].isalnum() or low_masked[i - 1] == "_")
                    right_ok = end == n or not (low_masked[end].isalnum() or low_masked[end] == "_")
                    if left_ok and right_ok:
                        split_spans.append((i, end))
                        i = end
                        matched_delim = True
                        break
            if matched_delim:
                continue

            # Check optional comma + spaces + "and " + lookahead for action_verb
            sp_start = i
            j = i
            if j < n and low_masked[j] == ",":
                j += 1
            while j < n and low_masked[j].isspace():
                j += 1
            if (j == 0 or not (low_masked[j - 1].isalnum() or low_masked[j - 1] == "_")) and low_masked.startswith("and", j):
                k = j + 3
                if k < n and low_masked[k].isspace():
                    while k < n and low_masked[k].isspace():
                        k += 1
                    for verb in action_verbs:
                        if low_masked.startswith(verb, k):
                            v_end = k + len(verb)
                            if v_end == n or not (low_masked[v_end].isalnum() or low_masked[v_end] == "_"):
                                split_spans.append((sp_start, k))
                                i = k
                                matched_delim = True
                                break
            if not matched_delim:
                i += 1

        parts = []
        cursor = 0
        for sp_start, sp_end in split_spans:
            parts.append(clean[cursor:sp_start].strip())
            cursor = sp_end
        parts.append(clean[cursor:].strip())
        return parts

    def _clean_subgoal_text(self, text: str) -> str:
        """Strip introductory ordinal words and punctuation from sub-goal segment."""
        s = text.strip()
        intro_words = ("first", "second", "third", "fourth", "finally", "next", "please", "now")
        low_s = s.lower()
        for w in intro_words:
            if low_s.startswith(w) and (len(s) == len(w) or not s[len(w)].isalnum()):
                rest = s[len(w) :].lstrip()
                if rest.startswith(","):
                    rest = rest[1:].lstrip()
                s = rest
                low_s = s.lower()
                break
        if low_s.startswith("step"):
            rest = s[4:].lstrip()
            j = 0
            while j < len(rest) and rest[j].isdigit():
                j += 1
            if j > 0:
                after_num = rest[j:].lstrip()
                if after_num.startswith(":"):
                    s = after_num[1:].lstrip()
        j = 0
        while j < len(s) and s[j].isdigit():
            j += 1
        if j > 0 and j < len(s) and s[j] == ".":
            s = s[j + 1 :].lstrip()
        return s.strip().rstrip(";,")

    def _resolve_subgoal_tool_call(
        self, text: str, prev_step_index: int
    ) -> ToolCall | None:
        """Map individual sub-goal to a typed ToolCall, supporting variable references."""
        # 1. Direct ActionRouter match
        direct_call = self.action_router.route(text)
        if direct_call is not None:
            # Check if arguments contain natural phrases for variable references
            args = dict(direct_call.arguments)
            for k, v in args.items():
                if isinstance(v, str):
                    args[k] = self._normalize_variable_reference_string(
                        v, prev_step_index
                    )
            return ToolCall(tool_name=direct_call.tool_name, arguments=args)

        clean = text.strip()
        low = clean.lower()

        # 2. File Writer shortcuts: "write [the] result to [file] <path>", "save [the] result to <path>"
        for pref in ("write ", "save "):
            if low.startswith(pref):
                rest = clean[len(pref) :].strip()
                content_val: str | None = None
                if ":" in rest:
                    rest, content_val = rest.split(":", 1)
                tokens = rest.strip().split()
                low_toks = [t.lower() for t in tokens]
                idx = 0
                if idx < len(low_toks) and low_toks[idx] == "the":
                    idx += 1
                if idx < len(low_toks) and low_toks[idx] in ("result", "output", "answer", "that", "results", "it"):
                    idx += 1
                if idx < len(low_toks) and low_toks[idx] == "to":
                    idx += 1
                    if idx < len(low_toks) and low_toks[idx] == "file":
                        idx += 1
                if idx == len(tokens) - 1 and _is_file_path_token(tokens[idx]):
                    file_path = tokens[idx].strip()
                    if content_val is not None and content_val.strip():
                        content_template = self._normalize_variable_reference_string(
                            content_val.strip(), prev_step_index
                        )
                    else:
                        content_template = "$prev.output"
                    return ToolCall(
                        tool_name="file_writer",
                        arguments={"file_path": file_path, "content": content_template},
                    )

        # 3. File Reader shortcuts: "read [file] <path>"
        for pref in ("read ", "view ", "cat "):
            if low.startswith(pref):
                tokens = clean[len(pref) :].strip().split()
                if tokens and tokens[0].lower() == "file":
                    tokens = tokens[1:]
                if len(tokens) == 1 and _is_file_path_token(tokens[0]):
                    return ToolCall(
                        tool_name="file_reader",
                        arguments={"file_path": tokens[0].strip()},
                    )

        # 4. Calculator shortcuts: "calculate X", "what is 250 * 4", etc.
        for pref in ("calculate ", "calc ", "compute "):
            if low.startswith(pref):
                expr = clean[len(pref) :].strip()
                if expr:
                    return ToolCall(
                        tool_name="calculator",
                        arguments={"expression": expr},
                    )
        for pref in ("what is ", "how much is ", "evaluate "):
            if low.startswith(pref):
                math_cand = clean[len(pref) :].rstrip("?").strip()
                if any(c.isdigit() for c in math_cand) and any(
                    op in math_cand for op in ("+", "-", "*", "/", "%", "of")
                ):
                    return ToolCall(
                        tool_name="calculator",
                        arguments={"expression": math_cand},
                    )

        return None

    def _normalize_variable_reference_string(
        self, text: str, prev_step_index: int
    ) -> str:
        """Replace natural pronouns like 'the result', 'the answer' with standard variable references."""
        replacement = (
            f"$step_{prev_step_index}.output"
            if prev_step_index > 0
            else "$prev.output"
        )
        phrases = ("the result", "the output", "the answer", "that result", "it")
        s = text
        for ph in phrases:
            low = s.lower()
            idx = 0
            out: list[str] = []
            while True:
                pos = low.find(ph, idx)
                if pos == -1:
                    out.append(s[idx:])
                    break
                end = pos + len(ph)
                left_ok = pos == 0 or not (low[pos - 1].isalnum() or low[pos - 1] == "_")
                right_ok = end == len(low) or not (low[end].isalnum() or low[end] == "_")
                if left_ok and right_ok:
                    out.append(s[idx:pos])
                    out.append(replacement)
                    idx = end
                else:
                    out.append(s[idx : pos + 1])
                    idx = pos + 1
            s = "".join(out)
        return s

    # -------------------------------------------------------------------------
    # Invariant Verification & Variable Binding
    # -------------------------------------------------------------------------

    def _check_mutex_invariants(self, plan: Plan) -> str | None:
        """Verify resource mutual exclusivity invariants (e.g. read/write ordering)."""

        def path_key(step: PlanStep) -> str:
            raw = str(step.arguments_template.get("file_path", "")).strip()
            if not raw or "$" in raw:
                return raw
            return str((self.action_policy.workspace_root / raw).resolve())

        first_writes: dict[str, int] = {}
        for step in plan.steps:
            if step.tool_name == "file_writer" and path_key(step):
                first_writes.setdefault(path_key(step), step.step_index)
        for step in plan.steps:
            if step.tool_name == "file_reader":
                key = path_key(step)
                if key in first_writes and step.step_index < first_writes[key]:
                    return f"Step {step.step_index} attempts to read '{key}' before it is written in Step {first_writes[key]}"
        return None

    def _resolve_argument_variables(
        self,
        tool_name: str,
        arguments_template: dict[str, Any],
        step_outputs: dict[int, Any],
        prev_step_idx: int,
    ) -> dict[str, Any]:
        """Substitute variable references ($prev.output, $step_N.output, $result) with actual values."""
        resolved: dict[str, Any] = {}
        tool_def = self.tool_policy.tools.get(tool_name)

        for key, val in arguments_template.items():
            if not isinstance(val, str):
                resolved[key] = val
                continue

            def binding(step_idx: int | None) -> Any:
                index = step_idx if step_idx is not None else prev_step_idx
                if index not in step_outputs:
                    raise ValueError(f"Missing output binding for step {index}")
                return step_outputs[index]

            spans = _find_variable_tokens(val)
            if len(spans) == 1 and spans[0][0] == 0 and spans[0][1] == len(val):
                resolved[key] = binding(spans[0][2])
            elif spans:
                chunks: list[str] = []
                cursor = 0
                for start_i, end_i, step_i in spans:
                    chunks.append(val[cursor:start_i])
                    chunks.append(str(binding(step_i)))
                    cursor = end_i
                chunks.append(val[cursor:])
                resolved[key] = "".join(chunks)
            else:
                resolved[key] = val

            # Type conformity check against ToolDefinition schema
            if tool_def and key in tool_def.parameters:
                p_type = tool_def.parameters[key].param_type
                if p_type in ("string", "str") and not isinstance(resolved[key], str):
                    resolved[key] = str(resolved[key])
                elif (
                    p_type in ("int", "integer")
                    and isinstance(resolved[key], str)
                    and resolved[key].strip().isdigit()
                ):
                    resolved[key] = int(resolved[key].strip())
                elif p_type in ("float", "number") and isinstance(resolved[key], str):
                    try:
                        resolved[key] = float(resolved[key].strip())
                    except ValueError:
                        pass

        return resolved

    def _attempt_self_correction(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        error_msg: str,
    ) -> dict[str, Any] | None:
        """Heuristic backtracker: inspect error and synthesize corrected arguments."""
        # Case 1: Python REPL missing standard import
        if tool_name == "python_repl" and "code" in arguments:
            code = str(arguments["code"])
            missing_mod: str | None = None
            marker = "name '"
            suffix = "' is not defined"
            if marker in error_msg and suffix in error_msg:
                start_idx = error_msg.index(marker) + len(marker)
                end_idx = error_msg.find(suffix, start_idx)
                if end_idx > start_idx:
                    cand = error_msg[start_idx:end_idx]
                    if cand.isidentifier():
                        missing_mod = cand
            if missing_mod:
                stdlib_modules = getattr(
                    sys,
                    "stdlib_module_names",
                    {
                        "math",
                        "json",
                        "sys",
                        "os",
                        "re",
                        "datetime",
                        "random",
                        "statistics",
                    },
                )
                if missing_mod in stdlib_modules:
                    fixed_code = f"import {missing_mod}\n{code}"
                    corrected = dict(arguments)
                    corrected["code"] = fixed_code
                    return corrected

        # Case 2: Calculator trailing punctuation or question mark
        if tool_name == "calculator" and "expression" in arguments:
            expr = str(arguments["expression"]).strip().rstrip("?.!;")
            if expr != arguments["expression"]:
                corrected = dict(arguments)
                corrected["expression"] = expr
                return corrected

        # Case 3: File reader/writer leading/trailing quotes or whitespace in path
        if tool_name in ("file_reader", "file_writer") and "file_path" in arguments:
            raw_fp = str(arguments["file_path"]).strip().strip("'\"`")
            if raw_fp != arguments["file_path"]:
                corrected = dict(arguments)
                corrected["file_path"] = raw_fp
                return corrected

        return None
