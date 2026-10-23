"""Maps natural language user intents to structured, typed ToolCalls (Needle-style)."""

from __future__ import annotations

from typing import Any

from little.actions.models import ToolCall, ToolDefinition
from little.actions.tool_policy import ActionPolicy, ToolDefinitionsPolicy
from little.neural.models import Pathway, RoutingDecision
from little.neural.router import NeuralRouter


def _is_num_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return bool(s) and s.count(".") <= 1 and s.replace(".", "", 1).isdigit()


def _is_file_path_token(tok: str) -> bool:
    s = tok.strip()
    return bool(s) and all(c.isalnum() or c in "_-./" for c in s)



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

        # 1. Policy-driven template dispatch configured in action_policy
        for tool_name, patterns in self.action_policy.intent_patterns.items():
            tool_def = self.tool_policy.tools.get(tool_name)
            if not tool_def or not patterns:
                continue

            raw_args = self._match_policy_intent(tool_name, patterns, text)
            if raw_args is not None:
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

    def _match_policy_intent(
        self, tool_name: str, patterns: tuple[str, ...], text: str
    ) -> dict[str, str] | None:
        """Match tool intent patterns defined in ActionPolicy using recursive-descent policy AST evaluation."""
        from little.language.parser import _match_pattern

        clean = text.strip()
        low = clean.lower()

        if tool_name == "web_search" and low.startswith(
            ("search wiki ", "search the wiki ")
        ):
            return None

        # Evaluate declarative patterns from ActionPolicy via our regex-free policy AST compiler
        for pat in patterns:
            m = _match_pattern(pat, clean)
            if m is not None:
                captured = {
                    k: str(v).strip()
                    for k, v in m.groupdict().items()
                    if v is not None and str(v).strip()
                }
                if tool_name == "calculator" and "expression" in captured:
                    expr = captured["expression"].rstrip("?").strip()
                    expr_low = expr.lower()
                    if low.startswith(("what is ", "what's ")):
                        allowed_chars = set("0123456789.+-*/^()% ")
                        is_pure_math = (
                            bool(expr)
                            and any(c.isdigit() for c in expr)
                            and all(c in allowed_chars for c in expr)
                            and any(op in expr for op in ("+", "-", "*", "/", "^", "%"))
                        )
                        is_pct_of = False
                        if " of " in expr_low and "%" in expr_low:
                            parts = expr_low.split(" of ", 1)
                            is_pct_of = _is_num_token(
                                parts[0].replace("%", "").strip()
                            ) and _is_num_token(parts[1].strip())
                        if not (is_pure_math or is_pct_of):
                            continue
                    captured["expression"] = expr
                if (
                    tool_name == "file_reader"
                    and "file_path" in captured
                    and not _is_file_path_token(captured["file_path"])
                ):
                    continue
                if tool_name == "file_writer" and not (
                    "file_path" in captured
                    and "content" in captured
                    and _is_file_path_token(captured["file_path"])
                ):
                    continue
                return captured

        return None

    def _extract_natural_arithmetic(self, text: str) -> str | None:
        """Extract arithmetic expression from conversational slang or word problems."""
        clean = text.lower().strip()

        # Multi-operator / compound arithmetic expressions first (avoid equations with '='):
        if "=" not in clean:
            cand = self._scan_compound_arithmetic(clean)
            if cand:
                return cand

        tokens = [t.strip("?.!,") for t in clean.split() if t.strip("?.!,")]

        # Percentage calculation: "15% of 800", "what is 15 percent of 800"
        for idx, tok in enumerate(tokens):
            if tok.endswith("%") and _is_num_token(tok[:-1]):
                n1 = tok[:-1]
                if idx + 2 < len(tokens) and tokens[idx + 1] == "of" and _is_num_token(tokens[idx + 2]):
                    return f"({n1} / 100) * {tokens[idx + 2]}"
                if idx + 1 < len(tokens) and _is_num_token(tokens[idx + 1]):
                    return f"({n1} / 100) * {tokens[idx + 1]}"
            if tok == "percent" and idx >= 1 and _is_num_token(tokens[idx - 1]):
                n1 = tokens[idx - 1]
                if idx + 2 < len(tokens) and tokens[idx + 1] == "of" and _is_num_token(tokens[idx + 2]):
                    return f"({n1} / 100) * {tokens[idx + 2]}"
                if idx + 1 < len(tokens) and _is_num_token(tokens[idx + 1]):
                    return f"({n1} / 100) * {tokens[idx + 1]}"

        # Word-based binary operations:
        for idx, tok in enumerate(tokens):
            # "multiply 15 by 4", "times 15 by 4"
            if (
                tok in ("multiply", "times")
                and idx + 3 < len(tokens)
                and _is_num_token(tokens[idx + 1])
                and tokens[idx + 2] in ("by", "and", "*")
                and _is_num_token(tokens[idx + 3])
            ):
                return f"{tokens[idx + 1]} * {tokens[idx + 3]}"
            # "product of 15 and 4"
            if (
                tok == "product"
                and idx + 4 < len(tokens)
                and tokens[idx + 1] == "of"
                and _is_num_token(tokens[idx + 2])
                and tokens[idx + 3] in ("by", "and", "*")
                and _is_num_token(tokens[idx + 4])
            ):
                return f"{tokens[idx + 2]} * {tokens[idx + 4]}"
            # "15 times 4" or "15 * 4"
            if (
                tok in ("times", "*")
                and 0 < idx < len(tokens) - 1
                and _is_num_token(tokens[idx - 1])
                and _is_num_token(tokens[idx + 1])
            ):
                return f"{tokens[idx - 1]} * {tokens[idx + 1]}"
            # "divide 100 by 5"
            if (
                tok == "divide"
                and idx + 3 < len(tokens)
                and _is_num_token(tokens[idx + 1])
                and tokens[idx + 2] in ("by", "/")
                and _is_num_token(tokens[idx + 3])
            ):
                return f"{tokens[idx + 1]} / {tokens[idx + 3]}"
            # "add 25 and 75"
            if (
                tok == "add"
                and idx + 3 < len(tokens)
                and _is_num_token(tokens[idx + 1])
                and tokens[idx + 2] in ("and", "to", "+")
                and _is_num_token(tokens[idx + 3])
            ):
                return f"{tokens[idx + 1]} + {tokens[idx + 3]}"
            # "sum of 25 and 75"
            if (
                tok == "sum"
                and idx + 4 < len(tokens)
                and tokens[idx + 1] == "of"
                and _is_num_token(tokens[idx + 2])
                and tokens[idx + 3] in ("and", "to", "+")
                and _is_num_token(tokens[idx + 4])
            ):
                return f"{tokens[idx + 2]} + {tokens[idx + 4]}"
            # "25 plus 75" or "25 + 75"
            if (
                tok in ("plus", "+")
                and 0 < idx < len(tokens) - 1
                and _is_num_token(tokens[idx - 1])
                and _is_num_token(tokens[idx + 1])
            ):
                return f"{tokens[idx - 1]} + {tokens[idx + 1]}"
            # "subtract 10 from 50"
            if (
                tok == "subtract"
                and idx + 3 < len(tokens)
                and _is_num_token(tokens[idx + 1])
                and tokens[idx + 2] == "from"
                and _is_num_token(tokens[idx + 3])
            ):
                return f"{tokens[idx + 3]} - {tokens[idx + 1]}"
            # "50 minus 10" or "50 - 10"
            if (
                tok in ("minus", "-")
                and 0 < idx < len(tokens) - 1
                and _is_num_token(tokens[idx - 1])
                and _is_num_token(tokens[idx + 1])
            ):
                return f"{tokens[idx - 1]} - {tokens[idx + 1]}"

        return None

    @staticmethod
    def _scan_compound_arithmetic(clean: str) -> str | None:
        """Scan for a contiguous arithmetic expression containing numbers and operators."""
        math_chars = set("0123456789.+-*/()^ ")
        n = len(clean)
        i = 0
        while i < n:
            # Start at '(' or a digit not preceded by an identifier/path character
            if (clean[i] == "(" or clean[i].isdigit()) and (
                i == 0 or not (clean[i - 1].isalnum() or clean[i - 1] in "_/.")
            ):
                j = i
                while j < n and clean[j] in math_chars:
                    j += 1
                # Must not end immediately into an identifier/unit slash like "km/h" or "m/s"
                while j > i and clean[j - 1] in " .+-*/^":
                    j -= 1
                if j < n and (clean[j].isalpha() or clean[j] == "_"):
                    # If the last char was adjacent to a letter (e.g. "60 km/h" where "/" was before "h"), trim back
                    pass
                candidate = clean[i:j].strip()
                has_digit = any(c.isdigit() for c in candidate)
                has_op = any(op in candidate for op in ("+", "-", "*", "/", "**", "^"))
                # Ensure at least two numeric operands exist around an operator
                num_chunks = [
                    tok
                    for tok in (
                        candidate.replace("(", " ")
                        .replace(")", " ")
                        .replace("+", " ")
                        .replace("-", " ")
                        .replace("*", " ")
                        .replace("/", " ")
                        .replace("^", " ")
                        .split()
                    )
                    if _is_num_token(tok)
                ]
                if has_digit and has_op and len(num_chunks) >= 2:
                    return candidate
                i = max(i + 1, j)
            else:
                i += 1
        return None

    def _extract_natural_tool(self, text: str) -> ToolCall | None:
        """Extract tool invocation from conversational tool requests using policy-driven token checks."""
        clean = text.lower().strip()

        # System time & date query via compositional ActionPolicy token check
        if self.action_policy.is_natural_time_query(clean):
            return ToolCall(tool_name="system_time", arguments={"timezone": "UTC"})

        # Web search via ActionPolicy intent patterns or natural prefix
        ws_patterns = self.action_policy.intent_patterns.get("web_search", ())
        ws_args = self._match_policy_intent("web_search", ws_patterns, text)
        if ws_args and "query" in ws_args:
            return ToolCall(tool_name="web_search", arguments={"query": ws_args["query"]})

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
