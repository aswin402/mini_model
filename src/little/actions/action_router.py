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
        """Match tool intent patterns defined in ActionPolicy without regex."""
        clean = text.strip()
        low = clean.lower()

        if tool_name == "calculator":
            for pref in ("calculate ", "calc ", "compute "):
                if low.startswith(pref):
                    expr = clean[len(pref) :].strip()
                    if expr:
                        return {"expression": expr}
            if low.startswith("what is ") or low.startswith("what's "):
                pref_len = 8 if low.startswith("what is ") else 7
                body = clean[pref_len:].rstrip("?").strip()
                body_low = body.lower()
                # Check if body is purely arithmetic or "<num>% of <num>"
                allowed_chars = set("0123456789.+-*/^()% ")
                if body and any(c.isdigit() for c in body):
                    if all(c in allowed_chars for c in body) and any(
                        op in body for op in ("+", "-", "*", "/", "^", "%")
                    ):
                        return {"expression": body}
                    if " of " in body_low and "%" in body_low:
                        parts = body_low.split(" of ", 1)
                        left_p = parts[0].replace("%", "").strip()
                        right_p = parts[1].strip()
                        if _is_num_token(left_p) and _is_num_token(right_p):
                            return {"expression": body}
            return None

        if tool_name == "python_repl":
            for pref in (
                "run python:",
                "run python ",
                "execute python:",
                "execute python ",
                "exec python:",
                "exec python ",
                "evaluate python:",
                "evaluate python ",
                "eval python:",
                "eval python ",
                "python:",
            ):
                if low.startswith(pref):
                    code = clean[len(pref) :].lstrip(":").strip()
                    if code:
                        return {"code": code}
            return None

        if tool_name == "system_time":
            norm = " ".join(low.rstrip("?").split())
            if norm in {
                "what's the time",
                "what is the time",
                "what's the current time",
                "what is the current time",
                "what time is it",
                "what's today's date",
                "what is today's date",
                "what's today date",
                "what is today date",
                "current date",
                "current date and time",
                "time",
            }:
                return {}
            return None

        if tool_name == "web_search":
            if low.startswith("search wiki ") or low.startswith("search the wiki "):
                return None
            for pref in (
                "search the web for ",
                "search for ",
                "search ",
                "google ",
                "lookup ",
            ):
                if low.startswith(pref):
                    q = clean[len(pref) :].rstrip("?").strip()
                    if q:
                        return {"query": q}
            return None

        if tool_name == "file_reader":
            for pref in ("read file ", "view file ", "cat "):
                if low.startswith(pref):
                    fp = clean[len(pref) :].strip()
                    if _is_file_path_token(fp):
                        return {"file_path": fp}
            return None

        if tool_name == "file_writer":
            if low.startswith("write to file ") and ":" in clean:
                rest = clean[len("write to file ") :]
                fp, content = rest.split(":", 1)
                if _is_file_path_token(fp.strip()) and content.strip():
                    return {"file_path": fp.strip(), "content": content.strip()}
            return None

        if tool_name == "bash_executor":
            for pref in (
                "run bash:",
                "run bash ",
                "execute bash:",
                "execute bash ",
                "exec bash:",
                "exec bash ",
                "bash:",
            ):
                if low.startswith(pref):
                    cmd = clean[len(pref) :].lstrip(":").strip()
                    if cmd:
                        return {"command": cmd}
            return None

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
            if tok in ("multiply", "times") and idx + 3 < len(tokens):
                if _is_num_token(tokens[idx + 1]) and tokens[idx + 2] in ("by", "and", "*") and _is_num_token(tokens[idx + 3]):
                    return f"{tokens[idx + 1]} * {tokens[idx + 3]}"
            # "product of 15 and 4"
            if tok == "product" and idx + 4 < len(tokens) and tokens[idx + 1] == "of":
                if _is_num_token(tokens[idx + 2]) and tokens[idx + 3] in ("by", "and", "*") and _is_num_token(tokens[idx + 4]):
                    return f"{tokens[idx + 2]} * {tokens[idx + 4]}"
            # "15 times 4" or "15 * 4"
            if tok in ("times", "*") and 0 < idx < len(tokens) - 1:
                if _is_num_token(tokens[idx - 1]) and _is_num_token(tokens[idx + 1]):
                    return f"{tokens[idx - 1]} * {tokens[idx + 1]}"
            # "divide 100 by 5"
            if tok == "divide" and idx + 3 < len(tokens):
                if _is_num_token(tokens[idx + 1]) and tokens[idx + 2] in ("by", "/") and _is_num_token(tokens[idx + 3]):
                    return f"{tokens[idx + 1]} / {tokens[idx + 3]}"
            # "add 25 and 75"
            if tok == "add" and idx + 3 < len(tokens):
                if _is_num_token(tokens[idx + 1]) and tokens[idx + 2] in ("and", "to", "+") and _is_num_token(tokens[idx + 3]):
                    return f"{tokens[idx + 1]} + {tokens[idx + 3]}"
            # "sum of 25 and 75"
            if tok == "sum" and idx + 4 < len(tokens) and tokens[idx + 1] == "of":
                if _is_num_token(tokens[idx + 2]) and tokens[idx + 3] in ("and", "to", "+") and _is_num_token(tokens[idx + 4]):
                    return f"{tokens[idx + 2]} + {tokens[idx + 4]}"
            # "25 plus 75" or "25 + 75"
            if tok in ("plus", "+") and 0 < idx < len(tokens) - 1:
                if _is_num_token(tokens[idx - 1]) and _is_num_token(tokens[idx + 1]):
                    return f"{tokens[idx - 1]} + {tokens[idx + 1]}"
            # "subtract 10 from 50"
            if tok == "subtract" and idx + 3 < len(tokens):
                if _is_num_token(tokens[idx + 1]) and tokens[idx + 2] == "from" and _is_num_token(tokens[idx + 3]):
                    return f"{tokens[idx + 3]} - {tokens[idx + 1]}"
            # "50 minus 10" or "50 - 10"
            if tok in ("minus", "-") and 0 < idx < len(tokens) - 1:
                if _is_num_token(tokens[idx - 1]) and _is_num_token(tokens[idx + 1]):
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

    _NATURAL_TIME_QUERIES = {
        "what is the time",
        "what's the time",
        "what is the current time",
        "what's the current time",
        "what is the time and date",
        "what's the time and date",
        "what is the current time and date",
        "what's the current time and date",
        "what is the date",
        "what's the date",
        "what is the current date",
        "what's the current date",
        "what is the date and time",
        "what's the date and time",
        "what is the current date and time",
        "what's the current date and time",
        "what is the day",
        "what's the day",
        "what is the day today",
        "what's the day today",
        "what is the timestamp",
        "what's the timestamp",
        "what is the current timestamp",
        "what's the current timestamp",
        "what time is it",
        "what time is it now",
        "what day is it",
        "what day is it today",
        "what date is it",
        "what date is it today",
        "what is today's date",
        "what's today's date",
        "what is today date",
        "what's today date",
        "what is today",
        "what's today",
        "what date is today",
        "tell me the time",
        "tell me the current time",
        "tell me the date",
        "tell me the current date",
        "tell me the time and date",
        "tell me the current time and date",
        "tell me the date and time",
        "tell me the current date and time",
        "current time",
        "current date",
        "current timestamp",
        "current time and date",
        "current date and time",
        "system time",
        "system date",
        "system clock",
        "current system clock",
        "the time",
        "the current time",
        "time",
    }

    def _extract_natural_tool(self, text: str) -> ToolCall | None:
        """Extract tool invocation from conversational tool requests."""
        clean = text.lower().strip()
        norm_q = " ".join(clean.rstrip("?").split())

        # System time & date query
        if norm_q in self._NATURAL_TIME_QUERIES:
            return ToolCall(tool_name="system_time", arguments={"timezone": "UTC"})

        # Web search
        if not (clean.startswith("search wiki ") or clean.startswith("search the wiki ")):
            for pref in (
                "search the web for ",
                "search for ",
                "search ",
                "google ",
                "lookup ",
            ):
                if clean.startswith(pref):
                    query = text.strip()[len(pref) :].rstrip("?").strip()
                    if query:
                        return ToolCall(
                            tool_name="web_search",
                            arguments={"query": query},
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
