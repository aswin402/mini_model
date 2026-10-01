"""Multi-Step Agentic Planning Loop and ReAct Execution Engine (Needle Action Chains).

Transforms LITTLE from a single-turn tool caller into an autonomous multi-step problem solver
with goal decomposition, variable binding across chained steps, DeepSeek-R1 invariant gates
(I_DAG, I_sort, I_mutex, I_ground), and automatic backtracking on step failures.
"""

from __future__ import annotations

import datetime
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from little.actions.action_router import ActionRouter
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
        has_conjunction = any(
            re.search(rf"\b{re.escape(d)}\b", text, re.IGNORECASE)
            for d in self.policy.conjunction_delimiters
            if d != ";"
        ) or ";" in text

        has_step_marker = bool(
            re.search(r"(?:^|\s)(?:step\s+\d+:?|\d+\.\s+|first\b|second\b)", text, re.IGNORECASE)
        )

        has_compound_action = bool(
            re.search(
                r"\b(?:and|then)\s+(?:write|save|read|run|exec|compute|calculate|search)\b",
                text,
                re.IGNORECASE,
            )
        )

        if not (has_conjunction or has_step_marker or has_compound_action):
            return False

        # Attempt trial decomposition: only compound if >= 2 valid steps are parsed
        plan = self.decompose_goal(text)
        return len(plan.steps) >= 2

    def decompose_goal(self, goal: str) -> Plan:
        """Deconstruct a compound goal utterance into an ordered Directed Acyclic Graph (DAG) of PlanSteps."""
        raw_subgoals = self._split_compound_utterance(goal)
        steps: list[PlanStep] = []

        for idx, sub_text in enumerate(raw_subgoals, start=1):
            cleaned = self._clean_subgoal_text(sub_text)
            if not cleaned:
                continue

            tool_call = self._resolve_subgoal_tool_call(cleaned, prev_step_index=idx - 1)
            if tool_call is None:
                continue

            # Identify dependencies: step idx depends on prev step if variables reference previous output
            deps: list[int] = []
            args_template = dict(tool_call.arguments)

            for arg_key, arg_val in args_template.items():
                if isinstance(arg_val, str):
                    if any(
                        p in arg_val
                        for p in ("$prev", "$result", "$output", "the result", "the output", "the answer")
                    ):
                        if idx > 1:
                            deps.append(idx - 1)
                    m_step = re.search(r"\$step_(\d+)", arg_val)
                    if m_step:
                        step_ref = int(m_step.group(1))
                        if 1 <= step_ref < idx:
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
            )
            steps.append(step)

        # Enforce max step bound from policy
        if len(steps) > self.policy.max_steps:
            steps = steps[: self.policy.max_steps]

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        created_at = datetime.datetime.now(datetime.UTC).isoformat()
        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=tuple(steps),
            created_at=created_at,
        )

    def plan_and_execute(self, goal: str) -> AgenticPlanExecutionResult:
        """Decompose goal into a DAG of steps and execute sequentially through ReAct loop."""
        plan = self.decompose_goal(goal)
        return self.execute_plan(plan)

    def execute_plan(self, plan: Plan) -> AgenticPlanExecutionResult:
        """Execute plan with Invariant Gates (I_DAG, I_sort, I_mutex, I_ground) and ReAct backtracking."""
        start_time = time.perf_counter()
        invariants_checked: list[str] = []

        # Gate 1: I_DAG Acyclicity check
        invariants_checked.append("I_dag_acyclicity")
        if not plan.is_acyclic():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AgenticPlanExecutionResult(
                plan_id=plan.plan_id,
                goal=plan.goal,
                steps=plan.steps,
                success=False,
                error="Plan rejected: cyclic or forward step dependencies detected (I_DAG violation)",
                total_execution_time_ms=elapsed_ms,
                trace_hash=AgenticPlanExecutionResult.compute_trace_hash(plan.plan_id, plan.steps, False),
                invariants_checked=tuple(invariants_checked),
            )

        # Gate 2: I_mutex Resource conflict check
        invariants_checked.append("I_mutex_resource")
        mutex_err = self._check_mutex_invariants(plan)
        if mutex_err:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return AgenticPlanExecutionResult(
                plan_id=plan.plan_id,
                goal=plan.goal,
                steps=plan.steps,
                success=False,
                error=f"Plan rejected: resource mutex violation: {mutex_err}",
                total_execution_time_ms=elapsed_ms,
                trace_hash=AgenticPlanExecutionResult.compute_trace_hash(plan.plan_id, plan.steps, False),
                invariants_checked=tuple(invariants_checked),
            )

        # Context environment for variable resolution across steps
        step_outputs: dict[int, Any] = {}
        total_retries = 0
        total_backtracks = 0
        overall_success = True
        plan_error: str | None = None
        executed_steps: list[PlanStep] = []

        for step in plan.steps:
            if not overall_success:
                step.status = PlanStepStatus.SKIPPED
                step.observation = "Skipped because plan execution failed at earlier step"
                executed_steps.append(step)
                continue

            # Check dependencies: if any dependency did not succeed, skip
            dep_failed = any(
                dep not in step_outputs for dep in step.dependencies
            )
            if dep_failed:
                step.status = PlanStepStatus.SKIPPED
                step.observation = "Skipped because previous dependent step failed"
                executed_steps.append(step)
                overall_success = False
                continue

            step.status = PlanStepStatus.RUNNING

            # Resolve variable bindings in arguments template
            resolved_args = self._resolve_argument_variables(
                step.tool_name,
                step.arguments_template,
                step_outputs=step_outputs,
                prev_step_idx=step.step_index - 1,
            )

            # ReAct Execution & Backtracking Loop
            success = False
            last_res: ToolExecutionResult | None = None
            current_args = resolved_args

            while step.retries <= self.policy.max_retries_per_step:
                tool_call = ToolCall(tool_name=step.tool_name, arguments=current_args)

                # Pre-dispatch verification gate
                invariants_checked.append("I_sort_type")
                invariants_checked.append("I_security_sandbox")
                ver_res = self.verifier.verify(tool_call)
                if not ver_res.valid:
                    last_res = ToolExecutionResult(
                        tool_name=step.tool_name,
                        arguments=current_args,
                        success=False,
                        error=ver_res.error,
                        verifiable_hash=ToolExecutionResult.compute_hash(
                            step.tool_name, current_args, None, False
                        ),
                    )
                else:
                    last_res = self.tool_executor.execute(tool_call)

                if last_res.success:
                    success = True
                    break

                # Step failed: try backtracking / self-correction if retries remain
                if step.retries < self.policy.max_retries_per_step and self.policy.enable_backtracking:
                    corrected_args = self._attempt_self_correction(
                        step.tool_name, current_args, last_res.error or ""
                    )
                    if corrected_args is not None and corrected_args != current_args:
                        step.retries += 1
                        total_retries += 1
                        total_backtracks += 1
                        current_args = corrected_args
                        continue

                step.retries += 1
                total_retries += 1
                break

            step.result = last_res

            if success and last_res is not None:
                step.status = PlanStepStatus.SUCCESS
                step.observation = str(last_res.output)
                step_outputs[step.step_index] = last_res.output

                # Gate 4: I_ground provenance - ground into memory
                invariants_checked.append("I_ground_provenance")
                self.tool_grounder.ground(last_res)
                executed_steps.append(step)
            else:
                step.status = PlanStepStatus.FAILED
                step.observation = f"Execution failed: {last_res.error if last_res else 'Unknown error'}"
                overall_success = False
                plan_error = f"Step {step.step_index} ({step.tool_name}) failed: {last_res.error if last_res else 'Unknown error'}"
                executed_steps.append(step)
                continue

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_output = (
            step_outputs[executed_steps[-1].step_index]
            if overall_success and executed_steps and executed_steps[-1].step_index in step_outputs
            else None
        )

        trace_hash = AgenticPlanExecutionResult.compute_trace_hash(
            plan.plan_id, tuple(executed_steps), overall_success
        )

        return AgenticPlanExecutionResult(
            plan_id=plan.plan_id,
            goal=plan.goal,
            steps=tuple(executed_steps),
            success=overall_success,
            final_output=final_output,
            total_execution_time_ms=elapsed_ms,
            trace_hash=trace_hash,
            invariants_checked=tuple(sorted(set(invariants_checked))),
            retries_used=total_retries,
            backtracks_count=total_backtracks,
            error=plan_error,
        )

    # -------------------------------------------------------------------------
    # Decomposition & Parsing Helpers
    # -------------------------------------------------------------------------

    def _split_compound_utterance(self, text: str) -> list[str]:
        """Split compound sentence into sub-goal clauses while respecting quoted expressions."""
        clean = text.strip()

        # Check for numbered/labeled steps (e.g. "step 1: ... step 2: ..." or "1. ... 2. ...")
        if re.search(r"(?:^|\s)step\s+\d+:?", clean, re.IGNORECASE):
            parts = re.split(r"(?:^|\s)step\s+\d+:?", clean, flags=re.IGNORECASE)
            return [p.strip() for p in parts if p.strip()]

        if re.search(r"(?:^|\n)\d+\.\s+", clean):
            parts = re.split(r"(?:^|\n)\d+\.\s+", clean)
            return [p.strip() for p in parts if p.strip()]

        # Conjunction delimiter split pattern:
        # Matches "and then", "then", "after that", "and finally", ";",
        # or "and write/read/run/calculate"
        split_pattern = (
            r"(?:;|\b(?:and\s+then|then|after\s+that|and\s+finally)\b"
            r"|,?\s*\band\s+(?=(?:write|save|read|run|exec|calculate|compute|search)\b))"
        )
        parts = re.split(split_pattern, clean, flags=re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]

    def _clean_subgoal_text(self, text: str) -> str:
        """Strip introductory ordinal words and punctuation from sub-goal segment."""
        s = text.strip()
        s = re.sub(r"^(?:first|second|third|fourth|finally|next|please|now)\s*,?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^step\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\d+\.\s*", "", s)
        return s.strip().rstrip(";,")

    def _resolve_subgoal_tool_call(self, text: str, prev_step_index: int) -> ToolCall | None:
        """Map individual sub-goal to a typed ToolCall, supporting variable references."""
        # 1. Direct ActionRouter match
        direct_call = self.action_router.route(text)
        if direct_call is not None:
            # Check if arguments contain natural phrases for variable references
            args = dict(direct_call.arguments)
            for k, v in args.items():
                if isinstance(v, str):
                    args[k] = self._normalize_variable_reference_string(v, prev_step_index)
            return ToolCall(tool_name=direct_call.tool_name, arguments=args)

        # 2. File Writer shortcuts: "write [the] result to [file] <path>", "save [the] result to <path>"
        m_write_res = re.search(
            r"^(?:write|save)\s+(?:the\s+)?(?:result|output|answer|that|results|it)?\s*(?:to\s+(?:file\s+)?)?(?P<file_path>[a-zA-Z0-9_\-\./]+)(?:\s*:\s*(?P<content>.+))?$",
            text,
            re.IGNORECASE,
        )
        if m_write_res:
            file_path = m_write_res.group("file_path").strip()
            content_val = m_write_res.group("content")
            if content_val:
                content_template = self._normalize_variable_reference_string(content_val.strip(), prev_step_index)
            else:
                content_template = "$prev.output"
            return ToolCall(
                tool_name="file_writer",
                arguments={"file_path": file_path, "content": content_template},
            )

        # 3. File Reader shortcuts: "read [file] <path>"
        m_read = re.search(r"^(?:read|view|cat)\s+(?:file\s+)?(?P<file_path>[a-zA-Z0-9_\-\./]+)$", text, re.IGNORECASE)
        if m_read:
            return ToolCall(
                tool_name="file_reader",
                arguments={"file_path": m_read.group("file_path").strip()},
            )

        # 4. Calculator shortcuts: "calculate X", "what is 250 * 4", etc.
        m_calc = re.search(r"^(?:calc(?:ulate)?|compute)\s+(.+)$", text, re.IGNORECASE)
        if not m_calc:
            m_what_math = re.search(r"^(?:what\s+is|how\s+much\s+is|evaluate)\s+(.+?)\??$", text, re.IGNORECASE)
            if m_what_math:
                math_cand = m_what_math.group(1).strip()
                if any(c.isdigit() for c in math_cand) and any(op in math_cand for op in ("+", "-", "*", "/", "%", "of")):
                    m_calc = m_what_math

        if m_calc:
            return ToolCall(
                tool_name="calculator",
                arguments={"expression": m_calc.group(1).strip()},
            )

        return None

    def _normalize_variable_reference_string(self, text: str, prev_step_index: int) -> str:
        """Replace natural pronouns like 'the result', 'the answer' with standard variable references."""
        s = text
        s = re.sub(
            r"\b(?:the\s+result|the\s+output|the\s+answer|that\s+result|it)\b",
            f"$step_{prev_step_index}.output" if prev_step_index > 0 else "$prev.output",
            s,
            flags=re.IGNORECASE,
        )
        return s

    # -------------------------------------------------------------------------
    # Invariant Verification & Variable Binding
    # -------------------------------------------------------------------------

    def _check_mutex_invariants(self, plan: Plan) -> str | None:
        """Verify resource mutual exclusivity invariants (e.g. read/write ordering)."""
        file_writes: dict[str, int] = {}
        for s in plan.steps:
            if s.tool_name == "file_writer":
                fp = str(s.arguments_template.get("file_path", "")).strip()
                if fp:
                    file_writes[fp] = s.step_index

        for s in plan.steps:
            if s.tool_name == "file_reader":
                fp = str(s.arguments_template.get("file_path", "")).strip()
                # If a file is written and read in the same plan, the read must occur after the write
                if fp in file_writes and s.step_index < file_writes[fp]:
                    return f"Step {s.step_index} attempts to read '{fp}' before it is written in Step {file_writes[fp]}"
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
        prev_output = step_outputs.get(prev_step_idx, "")
        tool_def = self.tool_policy.tools.get(tool_name)

        for key, val in arguments_template.items():
            if not isinstance(val, str):
                resolved[key] = val
                continue

            s_val = val

            # Exact match replacement (preserving numbers/objects when possible)
            if s_val in ("$prev.output", "$prev", "$result", "$output"):
                resolved[key] = prev_output
            elif m_exact := re.fullmatch(r"\$step_(\d+)(?:\.output)?", s_val):
                s_idx = int(m_exact.group(1))
                resolved[key] = step_outputs.get(s_idx, "")
            else:
                # Substring interpolation
                s_val = s_val.replace("$prev.output", str(prev_output))
                s_val = s_val.replace("$prev", str(prev_output))
                s_val = s_val.replace("$result", str(prev_output))
                s_val = s_val.replace("$output", str(prev_output))

                for s_idx, out in step_outputs.items():
                    s_val = s_val.replace(f"$step_{s_idx}.output", str(out))
                    s_val = s_val.replace(f"$step_{s_idx}", str(out))

                resolved[key] = s_val

            # Type conformity check against ToolDefinition schema
            if tool_def and key in tool_def.parameters:
                p_type = tool_def.parameters[key].param_type
                if p_type in ("string", "str") and not isinstance(resolved[key], str):
                    resolved[key] = str(resolved[key])
                elif p_type in ("int", "integer") and isinstance(resolved[key], str) and resolved[key].strip().isdigit():
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
        err_lower = error_msg.lower()

        # Case 1: Python REPL missing standard import
        if tool_name == "python_repl" and "code" in arguments:
            code = str(arguments["code"])
            m_name = re.search(r"name\s+'([a-zA-Z0-9_]+)'\s+is\s+not\s+defined", error_msg)
            if m_name:
                missing_mod = m_name.group(1)
                stdlib_modules = getattr(
                    sys,
                    "stdlib_module_names",
                    {"math", "json", "sys", "os", "re", "datetime", "random", "statistics"},
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
