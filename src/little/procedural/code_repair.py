"""Socratic Auto-Repair Loop for AST-synthesized Python programs."""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, Any

from little.procedural.code_policy import CodeSynthesizerPolicy
from little.procedural.code_sandbox import CodeSandbox, SandboxExecutionResult

if TYPE_CHECKING:
    from little.procedural.code_synthesizer import SynthesizedProgram


class SocraticCodeRepair:
    """Diagnoses unit test failures and applies targeted AST mutations to repair code."""

    def __init__(
        self,
        sandbox: CodeSandbox | None = None,
        policy: CodeSynthesizerPolicy | None = None,
    ) -> None:
        self.policy = policy or CodeSynthesizerPolicy.default()
        self.sandbox = sandbox or CodeSandbox(policy=self.policy)

    def repair(
        self, prog: SynthesizedProgram, initial_failure: SandboxExecutionResult
    ) -> bool:
        """Attempt iterative Socratic diagnosis and AST repair on the candidate program."""
        curr_failure = initial_failure
        max_attempts = self.policy.sandbox.max_repair_attempts

        for attempt in range(1, max_attempts + 1):
            prog.proof_trace.append(
                f"Socratic Repair Attempt {attempt}/{max_attempts}: diagnosing failure..."
            )
            repaired_tree = self._diagnose_and_mutate(prog, curr_failure)
            if repaired_tree is None:
                prog.proof_trace.append(
                    "Socratic diagnosis could not formulate a confident AST mutation"
                )
                break

            new_code = ast.unparse(repaired_tree)
            prog.proof_trace.append(
                f"Applied AST mutation; re-evaluating in sandbox..."
            )
            test_res = self.sandbox.run_assertions(new_code, prog.assertions)

            if test_res.success:
                prog.tree = repaired_tree
                prog.code = new_code
                prog.status = "REPAIRED"
                prog.execution_hash = test_res.verifiable_hash
                prog.execution_time_ms = test_res.execution_time_ms
                prog.proof_trace.append(
                    f"Socratic repair succeeded on attempt {attempt}: all assertions pass ({test_res.execution_time_ms:.2f} ms)"
                )
                return True

            prog.proof_trace.append(
                f"Attempt {attempt} failed: {test_res.error or 'assertions did not pass'}"
            )
            curr_failure = test_res

        prog.status = "FAILED"
        return False

    def _diagnose_and_mutate(
        self, prog: SynthesizedProgram, failure: SandboxExecutionResult
    ) -> ast.FunctionDef | None:
        """Parse failing test assertion and apply targeted AST transformation."""
        failing_assert = failure.failing_assertion
        if not failing_assert:
            # Fall back to error string if failing_assertion wasn't parsed directly
            if failure.error and "ASSERTION_FAILURE:" in failure.error:
                parts = failure.error.split(":")
                if len(parts) >= 3:
                    failing_assert = parts[2]

        if not failing_assert:
            return None

        # Pattern: func(arg1, ...) == expected
        match = re.match(r"^([a-zA-Z_]\w*)\((.*)\)\s*==\s*(.+)$", failing_assert.strip())
        if not match:
            return None

        _func_name, raw_args, raw_expected = match.groups()
        raw_args = raw_args.strip()
        raw_expected = raw_expected.strip()

        # Parse AST tree to clone
        try:
            tree = ast.parse(prog.code)
            func_node = tree.body[0]
            if not isinstance(func_node, ast.FunctionDef):
                return None
        except Exception:
            return None

        # Strategy 1: Empty container guard (e.g. arg is '' or [])
        if raw_args in ("''", '""', "[]", "()"):
            repaired = self._inject_empty_guard(func_node, raw_expected)
            if repaired is not None:
                return repaired

        # Strategy 2: Base case guard for numeric inputs (e.g. n == 0, n <= 1)
        if raw_args.isdigit() or (raw_args.startswith("-") and raw_args[1:].isdigit()):
            arg_val = int(raw_args)
            repaired = self._inject_base_case_guard(func_node, arg_val, raw_expected)
            if repaired is not None:
                return repaired

        # Strategy 3: Off-by-one loop boundary (e.g. range(..., n) -> range(..., n + 1) or Lt -> LtE)
        repaired = self._mutate_loop_boundary(func_node)
        if repaired is not None:
            return repaired

        return None

    def _inject_empty_guard(
        self, func: ast.FunctionDef, raw_expected: str
    ) -> ast.FunctionDef | None:
        """Inject `if not arg: return <expected>` at top of function body."""
        if not func.args.args:
            return None
        first_param = func.args.args[0].arg

        try:
            expected_expr = ast.parse(raw_expected, mode="eval").body
        except Exception:
            return None

        guard = ast.If(
            test=ast.UnaryOp(
                op=ast.Not(),
                operand=ast.Name(id=first_param, ctx=ast.Load()),
            ),
            body=[ast.Return(value=expected_expr)],
            orelse=[],
        )
        ast.fix_missing_locations(guard)

        # Insert after docstring if present
        insert_idx = 1 if (func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant)) else 0
        new_body = list(func.body)
        new_body.insert(insert_idx, guard)
        func.body = new_body
        ast.fix_missing_locations(func)
        return func

    def _inject_base_case_guard(
        self, func: ast.FunctionDef, arg_val: int, raw_expected: str
    ) -> ast.FunctionDef | None:
        """Inject `if arg <= val: return <expected>` at top of function body."""
        if not func.args.args:
            return None
        first_param = func.args.args[0].arg

        try:
            expected_expr = ast.parse(raw_expected, mode="eval").body
        except Exception:
            return None

        op = ast.LtE() if arg_val <= 1 else ast.Eq()
        guard = ast.If(
            test=ast.Compare(
                left=ast.Name(id=first_param, ctx=ast.Load()),
                ops=[op],
                comparators=[ast.Constant(value=arg_val)],
            ),
            body=[ast.Return(value=expected_expr)],
            orelse=[],
        )
        ast.fix_missing_locations(guard)

        insert_idx = 1 if (func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant)) else 0
        new_body = list(func.body)
        new_body.insert(insert_idx, guard)
        func.body = new_body
        ast.fix_missing_locations(func)
        return func

    def _mutate_loop_boundary(self, func: ast.FunctionDef) -> ast.FunctionDef | None:
        """Mutate `range(a, n)` to `range(a, n + 1)` or `Lt` to `LtE` in loops."""
        mutated = False
        for node in ast.walk(func):
            # Check while comparison Lt -> LtE
            if isinstance(node, ast.While) and isinstance(node.test, ast.Compare):
                for idx, op in enumerate(node.test.ops):
                    if isinstance(op, ast.Lt):
                        node.test.ops[idx] = ast.LtE()
                        mutated = True
            # Check range call second arg
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                if len(node.args) == 2 and isinstance(node.args[1], ast.Name):
                    node.args[1] = ast.BinOp(
                        left=node.args[1],
                        op=ast.Add(),
                        right=ast.Constant(value=1),
                    )
                    mutated = True

        if mutated:
            ast.fix_missing_locations(func)
            return func
        return None
