"""Safe deterministic execution sandbox for Procedural Skills in LITTLE."""

from __future__ import annotations

import ast
import builtins as python_builtins
import math
import operator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

from little.core.models import Skill
from little.procedural.skill_policy import ProceduralSkillPolicy


@dataclass
class ExecutionResult:
    """Result of executing a procedural skill."""

    skill_name: str
    success: bool
    result: Any = None
    error: str | None = None
    trace: list[str] | None = None


_EXECUTION_MODULES = MappingProxyType(
    {
        "math": math,
        "ast": ast,
        "operator": operator,
        "sympy": sympy,
    }
)
_EXECUTION_HELPERS = MappingProxyType(
    {
        "parse_expr": parse_expr,
        "sympy_transformations": SYMPY_TRANSFORMATIONS,
    }
)


def _safe_builtins(policy: ProceduralSkillPolicy) -> dict[str, Any]:
    """Return an isolated namespace from the configured execution policy."""
    namespace: dict[str, Any] = {}
    for name in policy.execution_builtins:
        try:
            namespace[name] = getattr(python_builtins, name)
        except AttributeError as ex:
            raise ValueError(f"Unsupported procedural builtin {name!r}") from ex
    for name in policy.execution_modules:
        try:
            namespace[name] = _EXECUTION_MODULES[name]
        except KeyError as ex:
            raise ValueError(f"Unsupported procedural module {name!r}") from ex
    for name in policy.execution_helpers:
        try:
            namespace[name] = _EXECUTION_HELPERS[name]
        except KeyError as ex:
            raise ValueError(f"Unsupported procedural helper {name!r}") from ex
    return namespace


class SkillRunner:
    """Safely executes procedural algorithms with parameter verification and sandboxing."""

    @classmethod
    def execute(
        cls,
        skill: Skill,
        *,
        policy: ProceduralSkillPolicy | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a skill with keyword arguments."""
        active_policy = policy or ProceduralSkillPolicy.default()
        trace = [f"Preparing execution of skill '{skill.name}' with inputs: {kwargs}"]

        # 1. Parameter validation
        for param in skill.parameters:
            if param not in kwargs:
                err_msg = (
                    f"Missing required parameter '{param}' for skill '{skill.name}'"
                )
                trace.append(f"Error: {err_msg}")
                return ExecutionResult(
                    skill_name=skill.name,
                    success=False,
                    error=err_msg,
                    trace=trace,
                )

        # 2. Execution environment
        local_scope = dict(kwargs)
        global_scope = {"__builtins__": _safe_builtins(active_policy)}

        # 3. Execution
        try:
            # Check if code is a simple expression or block
            code_str = skill.code_body.strip()
            trace.append(f"Evaluating skill code: {code_str}")

            # If it's a single expression (e.g. "a + b"), evaluate it directly
            if (
                "\n" not in code_str
                and not code_str.startswith("def ")
                and not code_str.startswith("return ")
            ):
                val = eval(code_str, global_scope, local_scope)
            else:
                # Multi-line or function block: wrap in a function if needed
                if not code_str.startswith("def "):
                    params_str = ", ".join(skill.parameters)
                    indented_body = "\n".join(
                        "    " + line for line in code_str.splitlines()
                    )
                    wrapper = f"def _exec({params_str}):\n{indented_body}"
                    exec(wrapper, global_scope, local_scope)  # noqa: S102
                    fn = local_scope["_exec"]
                    val = fn(**{k: kwargs[k] for k in skill.parameters})
                else:
                    exec(code_str, global_scope, local_scope)  # noqa: S102
                    # Find function name
                    fn_name = code_str.split("(")[0].replace("def ", "").strip()
                    val = local_scope[fn_name](
                        **{k: kwargs[k] for k in skill.parameters}
                    )

            trace.append(f"Skill execution succeeded. Output: {val}")
            return ExecutionResult(
                skill_name=skill.name,
                success=True,
                result=val,
                trace=trace,
            )

        except ZeroDivisionError:
            err_msg = "Division by zero is undefined."
            trace.append(f"Runtime Exception: {err_msg}")
            return ExecutionResult(
                skill_name=skill.name,
                success=False,
                error=err_msg,
                trace=trace,
            )
        except (
            ArithmeticError,
            NameError,
            TypeError,
            ValueError,
            SyntaxError,
            KeyError,
            IndexError,
        ) as ex:
            err_msg = f"{type(ex).__name__}: {ex}"
            trace.append(f"Execution failed: {err_msg}")
            return ExecutionResult(
                skill_name=skill.name,
                success=False,
                error=err_msg,
                trace=trace,
            )
