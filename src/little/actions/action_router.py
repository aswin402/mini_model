"""Maps natural language user intents to structured, typed ToolCalls (Needle-style)."""

from __future__ import annotations

import re
from typing import Any

from little.actions.models import ToolCall, ToolDefinition
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy


class ActionRouter:
    """Dispatches user utterances to typed ToolCalls using versioned schema policies."""

    def __init__(
        self,
        action_policy: ActionPolicy | None = None,
        tool_policy: ToolDefinitionsPolicy | None = None,
    ) -> None:
        self.action_policy = action_policy or ActionPolicy.default()
        self.tool_policy = tool_policy or ToolDefinitionsPolicy.default()

    def route(self, utterance: str) -> ToolCall | None:
        """Parse utterance and return a ToolCall if a tool intent is identified."""
        text = utterance.strip()
        if not text:
            return None

        # Check each tool's patterns configured in action_policy
        for tool_name, patterns in self.action_policy.intent_patterns.items():
            tool_def = self.tool_policy.tools.get(tool_name)
            if not tool_def:
                continue

            for pattern in patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    raw_args = match.groupdict()
                    typed_args = self._cast_arguments(tool_def, raw_args)
                    return ToolCall(
                        tool_name=tool_name,
                        arguments=typed_args,
                    )

        return None

    def _cast_arguments(self, tool_def: ToolDefinition, raw_args: dict[str, str]) -> dict[str, Any]:
        """Cast string capture groups to schema-defined parameter types."""
        result: dict[str, Any] = {}

        # 1. Fill defaults
        for p_name, p_spec in tool_def.parameters.items():
            if p_spec.default is not None:
                result[p_name] = p_spec.default

        # 2. Populate captured values with casting
        for key, val_str in raw_args.items():
            if key not in tool_def.parameters:
                result[key] = val_str
                continue

            p_spec = tool_def.parameters[key]
            p_type = p_spec.param_type

            if p_type in ("int", "integer"):
                try:
                    result[key] = int(val_str.strip())
                except ValueError:
                    result[key] = val_str
            elif p_type in ("float", "number"):
                try:
                    result[key] = float(val_str.strip())
                except ValueError:
                    result[key] = val_str
            elif p_type in ("bool", "boolean"):
                result[key] = val_str.strip().lower() in ("true", "1", "yes")
            else:
                result[key] = val_str.strip()

        return result
