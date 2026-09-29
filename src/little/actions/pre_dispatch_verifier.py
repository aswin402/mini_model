"""Pre-dispatch verification gate (I_sort and security invariants) for tool calls."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from little.actions.models import ToolCall, ToolDefinition, VerificationResult
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy


class PreDispatchVerifier:
    """Verifies that tool calls strictly adhere to typed schemas and security constraints."""

    def __init__(
        self,
        tool_policy: ToolDefinitionsPolicy,
        action_policy: ActionPolicy,
    ) -> None:
        self.tool_policy = tool_policy
        self.action_policy = action_policy

    def verify(self, call: ToolCall) -> VerificationResult:
        """Run all verification gates on a proposed tool call."""
        checked: list[str] = []

        # Gate 1: Tool existence check
        checked.append("I_tool_exists")
        if call.tool_name not in self.tool_policy.tools:
            return VerificationResult(
                valid=False,
                error=f"Unknown tool '{call.tool_name}'. Available tools: {list(self.tool_policy.tools.keys())}",
                checked_invariants=tuple(checked),
            )

        definition: ToolDefinition = self.tool_policy.tools[call.tool_name]

        # Gate 2: Required parameters check
        checked.append("I_param_required")
        for param_name, param_spec in definition.parameters.items():
            if param_spec.required and param_name not in call.arguments:
                return VerificationResult(
                    valid=False,
                    error=f"Missing required parameter '{param_name}' for tool '{call.tool_name}'",
                    checked_invariants=tuple(checked),
                )

        # Gate 3: Parameter type checking (I_sort)
        checked.append("I_sort_type")
        for arg_name, arg_val in call.arguments.items():
            if arg_name not in definition.parameters:
                continue
            param_spec = definition.parameters[arg_name]
            type_err = self._check_type(arg_name, arg_val, param_spec.param_type)
            if type_err:
                return VerificationResult(
                    valid=False,
                    error=f"Type signature mismatch for parameter '{arg_name}': {type_err}",
                    checked_invariants=tuple(checked),
                )

        # Gate 4: Security and domain invariants
        checked.append("I_security_sandbox")
        sec_err = self._check_security_invariants(call)
        if sec_err:
            return VerificationResult(
                valid=False,
                error=f"Security invariant violation: {sec_err}",
                checked_invariants=tuple(checked),
            )

        return VerificationResult(
            valid=True,
            error=None,
            checked_invariants=tuple(checked),
        )

    def _check_type(self, name: str, val: Any, expected_type: str) -> str | None:
        """Verify that val matches expected_type."""
        if expected_type in ("string", "str"):
            if not isinstance(val, str):
                return f"Expected string, got {type(val).__name__}"
        elif expected_type in ("int", "integer"):
            if isinstance(val, int) and not isinstance(val, bool):
                return None
            if isinstance(val, str) and val.strip().isdigit():
                return None
            return f"Expected integer, got {type(val).__name__}"
        elif expected_type in ("float", "number"):
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return None
            try:
                float(val)
                return None
            except (ValueError, TypeError):
                return f"Expected number, got {val!r}"
        elif expected_type in ("bool", "boolean"):
            if isinstance(val, bool):
                return None
            if isinstance(val, str) and val.lower() in ("true", "false"):
                return None
            return f"Expected boolean, got {type(val).__name__}"
        elif expected_type in ("list", "array"):
            if not isinstance(val, (list, tuple)):
                return f"Expected list, got {type(val).__name__}"
        elif expected_type in ("dict", "object") and not isinstance(val, dict):
            return f"Expected dict, got {type(val).__name__}"
        return None

    def _check_security_invariants(self, call: ToolCall) -> str | None:
        """Verify filesystem and subprocess security invariants."""
        # 1. Path traversal verification
        if "file_path" in call.arguments:
            raw_path = str(call.arguments["file_path"]).strip()
            if ".." in raw_path.split("/") or ".." in raw_path.split("\\"):
                return "Path traversal ('..') is strictly prohibited"
            path_obj = Path(raw_path)
            if path_obj.is_absolute():
                # Must reside inside workspace root
                try:
                    path_obj.resolve().relative_to(self.action_policy.workspace_root.resolve())
                except ValueError:
                    return f"Absolute path '{raw_path}' escapes workspace sandbox"

        # 2. Bash binary whitelist verification
        if call.tool_name == "bash_executor":
            raw_cmd = str(call.arguments.get("command", "")).strip()
            if not raw_cmd:
                return "Empty bash command"
            # Disallow dangerous shell chains unless validated
            for char in (";", "&", "|", "`", "$"):
                if char in raw_cmd:
                    return f"Dangerous shell operator '{char}' is restricted in bash_executor"
            try:
                tokens = shlex.split(raw_cmd)
            except ValueError as ex:
                return f"Malformed shell command: {ex}"
            if not tokens:
                return "Empty shell command tokens"
            binary = tokens[0]
            if binary not in self.action_policy.allowed_bash_binaries:
                return f"Binary '{binary}' is not in allowed whitelist: {list(self.action_policy.allowed_bash_binaries)}"

        return None
