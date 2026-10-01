"""Maps natural language user intents to structured, typed ToolCalls (Needle-style)."""

from __future__ import annotations

import re
from typing import Any

from little.actions.models import ToolCall, ToolDefinition
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy
from little.neural.models import Pathway, RoutingDecision
from little.neural.router import NeuralRouter


class ActionRouter:
    """Dispatches user utterances to typed ToolCalls using schema policies and Neural Routing."""

    def __init__(
        self,
        action_policy: ActionPolicy | None = None,
        tool_policy: ToolDefinitionsPolicy | None = None,
        neural_router: NeuralRouter | None = None,
    ) -> None:
        self.action_policy = action_policy or ActionPolicy.default()
        self.tool_policy = tool_policy or ToolDefinitionsPolicy.default()
        self.neural_router = neural_router or NeuralRouter()

    def route(
        self,
        utterance: str,
        routing_decision: RoutingDecision | None = None,
    ) -> ToolCall | None:
        """Parse utterance and return a ToolCall if a tool intent is identified."""
        text = utterance.strip()
        if not text:
            return None

        # 1. Exact regex template dispatch configured in action_policy
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

        # 2. Natural arithmetic and neural pathway semantic dispatch
        math_expr = self._extract_natural_arithmetic(text)
        if math_expr:
            return ToolCall(
                tool_name="calculator",
                arguments={"expression": math_expr},
            )

        decision = routing_decision or self.neural_router.route(text)
        if decision.pathway == Pathway.TOOL:
            tool_call = self._extract_natural_tool(text)
            if tool_call:
                return tool_call

        return None

    def _extract_natural_arithmetic(self, text: str) -> str | None:
        """Extract arithmetic expression from conversational slang or word problems."""
        clean = text.lower().strip()

        # Multi-operator / compound arithmetic expressions first (avoid equations with '='):
        if "=" not in clean:
            m_compound = re.search(
                r"(\(?\s*\d+(?:\.\d+)?\s*(?:[+\-*/]|(?:\*\*))\s*[-+*/()0-9\s.^]+)", clean
            )
            if m_compound:
                candidate = m_compound.group(1).rstrip("?. ").strip()
                if any(op in candidate for op in ("+", "-", "*", "/", "**")):
                    return candidate

        # Percentage calculation: "15% of 800", "what is 15 percent of 800"
        m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)?\s*(\d+(?:\.\d+)?)", clean)
        if m_pct:
            return f"({m_pct.group(1)} / 100) * {m_pct.group(2)}"

        # Multiplication: "multiply 15 by 4", "15 times 4", "product of 15 and 4"
        m_mult = re.search(
            r"(?:multiply|times|product\s+of)\s+(\d+(?:\.\d+)?)\s+(?:by|and|\*)\s+(\d+(?:\.\d+)?)",
            clean,
        )
        if m_mult:
            return f"{m_mult.group(1)} * {m_mult.group(2)}"

        m_times = re.search(r"(\d+(?:\.\d+)?)\s*(?:\*|times)\s*(\d+(?:\.\d+)?)", clean)
        if m_times:
            return f"{m_times.group(1)} * {m_times.group(2)}"

        # Division: "divide 100 by 5"
        m_div = re.search(r"(?:divide)\s+(\d+(?:\.\d+)?)\s+(?:by|\/)\s+(\d+(?:\.\d+)?)", clean)
        if m_div:
            return f"{m_div.group(1)} / {m_div.group(2)}"

        # Addition: "add 25 and 75", "25 plus 75", "sum of 25 and 75"
        m_add = re.search(
            r"(?:add|sum\s+of)\s+(\d+(?:\.\d+)?)\s+(?:and|to|\+)\s+(\d+(?:\.\d+)?)",
            clean,
        )
        if m_add:
            return f"{m_add.group(1)} + {m_add.group(2)}"

        m_plus = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+|plus)\s*(\d+(?:\.\d+)?)", clean)
        if m_plus:
            return f"{m_plus.group(1)} + {m_plus.group(2)}"

        # Subtraction: "subtract 10 from 50", "50 minus 10"
        m_sub_from = re.search(r"(?:subtract)\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)", clean)
        if m_sub_from:
            return f"{m_sub_from.group(2)} - {m_sub_from.group(1)}"

        m_sub = re.search(r"(\d+(?:\.\d+)?)\s*(?:\-|minus)\s*(\d+(?:\.\d+)?)", clean)
        if m_sub:
            return f"{m_sub.group(1)} - {m_sub.group(2)}"

        return None

    _TIME_QUERY_PATTERN = re.compile(
        r"^(?:"
        r"what(?:'s|\s+is)?\s+(?:the\s+)?(?:current\s+)?(?:time(?:\s+and\s+date)?|date(?:\s+and\s+time)?|day|timestamp)(?:\s+today)?"
        r"|what\s+time\s+is\s+it(?:\s+now)?"
        r"|what\s+(?:day|date)\s+is\s+it(?:\s+today)?"
        r"|what(?:'s|\s+is)\s+today(?:'s)?(?:\s+date)?"
        r"|what\s+date\s+is\s+today"
        r"|tell\s+me\s+the\s+(?:current\s+)?(?:time(?:\s+and\s+date)?|date(?:\s+and\s+time)?)"
        r"|current\s+(?:time(?:\s+and\s+date)?|date(?:\s+and\s+time)?|timestamp)"
        r"|system\s+(?:time|date)"
        r"|(?:current\s+)?system\s+clock"
        r"|(?:the\s+)?(?:current\s+)?time\?*$"
        r")\??$",
        re.IGNORECASE,
    )

    def _extract_natural_tool(self, text: str) -> ToolCall | None:
        """Extract tool invocation from conversational tool requests."""
        clean = text.lower().strip()

        # System time & date query
        if self._TIME_QUERY_PATTERN.search(clean):
            return ToolCall(tool_name="system_time", arguments={"timezone": "UTC"})

        # Web search
        m_search = re.search(
            r"^(?:search(?:\s+(?:the\s+web\s+for|for))?|google|lookup)\s+(.+?)\??$",
            clean,
        )
        if m_search:
            return ToolCall(
                tool_name="web_search",
                arguments={"query": m_search.group(1).strip()},
            )

        return None

    def _cast_arguments(
        self, tool_def: ToolDefinition, raw_args: dict[str, str]
    ) -> dict[str, Any]:
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
