"""Restricted, resource-bounded subprocess execution for computational Python."""

from __future__ import annotations

import ast
import hashlib
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from little.procedural.code_policy import CodeSynthesizerPolicy, SandboxPolicy
from little.procedural.restricted_execution import run, validate


@dataclass(frozen=True)
class SandboxExecutionResult:
    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    timed_out: bool = False
    execution_time_ms: float = 0.0
    verifiable_hash: str = ""
    failing_assertion: str | None = None


class CodeSandbox:
    """Restricts computational Python; this is not operating-system isolation."""

    def __init__(self, policy: CodeSynthesizerPolicy | None = None) -> None:
        self.policy = policy or CodeSynthesizerPolicy.default()
        self.sandbox_policy: SandboxPolicy = self.policy.sandbox

    def verify_ast_safety(self, code: str) -> tuple[bool, str | None]:
        """Verify code contains only permitted AST nodes and safe built-in identifiers."""
        safe, error = validate(code)
        if not safe:
            return safe, error
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if type(node).__name__ in self.sandbox_policy.forbidden_ast_nodes:
                return False, f"Forbidden AST node '{type(node).__name__}'"
            if isinstance(node, ast.Name) and node.id in self.sandbox_policy.forbidden_names:
                return False, f"Forbidden identifier '{node.id}'"
        return True, None

    def execute_script(self, code: str) -> SandboxExecutionResult:
        """Execute Python code in an isolated subprocess with timeout and AST validation."""
        start_time = time.perf_counter()

        safe, err = self.verify_ast_safety(code)
        if not safe:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                success=False,
                error=err,
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(code, err, False),
            )

        timeout = self.sandbox_policy.timeout_seconds
        try:
            proc = run(code, timeout=timeout, memory_mb=self.sandbox_policy.max_memory_mb,
                       output_bytes=self.sandbox_policy.max_output_bytes,
                       builtins=list(self.sandbox_policy.allowed_builtin_names) + ['print', 'Exception', 'AssertionError', 'ValueError', 'type'])
            elapsed = (time.perf_counter() - start_time) * 1000.0
            success = proc.returncode == 0
            err_msg = proc.stderr.strip() if proc.returncode != 0 else None
            return SandboxExecutionResult(
                success=success,
                output=proc.stdout.strip(),
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=err_msg,
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(code, proc.stdout.strip(), success),
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                timed_out=True,
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(code, "TIMEOUT", False),
            )
        except OSError as ex:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                success=False,
                error=f"Execution error: {ex}",
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(code, str(ex), False),
            )

    def run_assertions(
        self,
        func_code: str,
        assertions: list[str] | tuple[str, ...],
    ) -> SandboxExecutionResult:
        """Run a set of assertions against the function in a sandboxed harness."""
        safe, error = self.verify_ast_safety(func_code)
        if not safe:
            return SandboxExecutionResult(success=False, error=error)
        for assertion in assertions:
            try:
                ast.parse(assertion, mode="eval")
            except (SyntaxError, ValueError) as exc:
                return SandboxExecutionResult(success=False, error=str(exc), failing_assertion=assertion)
            safe, error = self.verify_ast_safety(assertion)
            if not safe:
                return SandboxExecutionResult(success=False, error=error, failing_assertion=assertion)
        # Assertions use the same safety, timeout, memory and output gates as scripts.
        harness = func_code + "\n" + "\n".join(f"assert {assertion}" for assertion in assertions)
        result = self.execute_script(harness + "\nprint('ALL_ASSERTIONS_PASSED')")
        if result.success:
            return result
        # Identify the failing assertion from the child traceback line number.
        import re
        lines = [int(ln) for ln in re.findall(r'File "<restricted>", line (\d+)', result.stderr)]
        func_lines = len(func_code.splitlines())
        assert_lines = [ln - func_lines - 1 for ln in lines if 0 <= ln - func_lines - 1 < len(assertions)]
        index = assert_lines[0] if assert_lines else -1
        from dataclasses import replace
        return replace(result, failing_assertion=assertions[index] if 0 <= index < len(assertions) else None)

    @staticmethod
    def _compute_hash(code: str, output: Any, success: bool) -> str:
        payload = f"{code}|{output}|{success}".encode()
        return hashlib.sha256(payload).hexdigest()
