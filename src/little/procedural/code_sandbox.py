"""Secure, isolated execution sandbox for evaluating synthesized Python code."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

from little.procedural.code_policy import CodeSynthesizerPolicy, SandboxPolicy


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
    """Executes synthesized code and unit tests within process-isolated safety boundaries."""

    def __init__(self, policy: CodeSynthesizerPolicy | None = None) -> None:
        self.policy = policy or CodeSynthesizerPolicy.default()
        self.sandbox_policy: SandboxPolicy = self.policy.sandbox

    def verify_ast_safety(self, code: str) -> tuple[bool, str | None]:
        """Verify code contains only permitted AST nodes and safe built-in identifiers."""
        try:
            tree = ast.parse(code)
        except SyntaxError as ex:
            return False, f"SyntaxError: {ex.msg} (line {ex.lineno})"

        forbidden_nodes = set(self.sandbox_policy.forbidden_ast_nodes)
        forbidden_names = set(self.sandbox_policy.forbidden_names)

        for node in ast.walk(tree):
            node_name = type(node).__name__
            if node_name in forbidden_nodes:
                return False, f"Forbidden AST node '{node_name}' violates sandbox safety invariants"
            if isinstance(node, ast.Name):
                if node.id in forbidden_names:
                    return False, f"Forbidden identifier '{node.id}' violates sandbox safety invariants"
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Access to dunder attribute '{node.attr}' is forbidden in sandbox"

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
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
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
        # Wrap the function definition and assertions into a single harness script
        harness_lines = [
            func_code,
            "",
            "# Test Harness",
            "import sys",
        ]
        for idx, assertion in enumerate(assertions):
            assert_prefix = repr(f"ASSERTION_FAILURE:{idx}:{assertion}:")
            harness_lines.append("try:")
            harness_lines.append(f"    assert {assertion}")
            harness_lines.append("except Exception as e:")
            harness_lines.append(f"    print({assert_prefix} + type(e).__name__ + ':' + str(e), file=sys.stderr)")
            harness_lines.append("    sys.exit(1)")
        harness_lines.append("print('ALL_ASSERTIONS_PASSED')")

        harness_script = "\n".join(harness_lines)
        start_time = time.perf_counter()
        timeout = self.sandbox_policy.timeout_seconds

        try:
            proc = subprocess.run(
                [sys.executable, "-c", harness_script],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            if proc.returncode == 0 and "ALL_ASSERTIONS_PASSED" in proc.stdout:
                return SandboxExecutionResult(
                    success=True,
                    output="ALL_ASSERTIONS_PASSED",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    execution_time_ms=elapsed,
                    verifiable_hash=self._compute_hash(func_code, "ALL_PASSED", True),
                )
            
            failing_assertion = None
            err_line = None
            for line in proc.stderr.splitlines():
                if line.startswith("ASSERTION_FAILURE:"):
                    parts = line.split(":", 3)
                    if len(parts) >= 3:
                        failing_assertion = parts[2]
                        err_line = line
                    break

            error_msg = err_line or proc.stderr.strip() or f"Process failed with exit code {proc.returncode}"
            return SandboxExecutionResult(
                success=False,
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=error_msg,
                failing_assertion=failing_assertion,
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(func_code, error_msg, False),
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                success=False,
                error=f"Assertions timed out after {timeout} seconds",
                timed_out=True,
                execution_time_ms=elapsed,
                verifiable_hash=self._compute_hash(func_code, "TIMEOUT", False),
            )

    @staticmethod
    def _compute_hash(code: str, output: Any, success: bool) -> str:
        payload = f"{code}|{output}|{success}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
