"""Safe deterministic execution sandbox for Procedural Skills in LITTLE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from little.core.models import Skill


@dataclass
class ExecutionResult:
    """Result of executing a procedural skill."""

    skill_name: str
    success: bool
    result: Any = None
    error: str | None = None
    trace: list[str] | None = None


class SkillRunner:
    """Safely executes procedural algorithms with parameter verification and sandboxing."""

    # Whitelist of safe math and utility builtins
    SAFE_BUILTINS: ClassVar[dict[str, Any]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "len": len,
        "sum": sum,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "math": math,
    }

    @classmethod
    def execute(cls, skill: Skill, **kwargs: Any) -> ExecutionResult:
        """Execute a skill with keyword arguments."""
        trace = [f"Preparing execution of skill '{skill.name}' with inputs: {kwargs}"]

        # 1. Parameter validation
        for param in skill.parameters:
            if param not in kwargs:
                err_msg = f"Missing required parameter '{param}' for skill '{skill.name}'"
                trace.append(f"Error: {err_msg}")
                return ExecutionResult(
                    skill_name=skill.name,
                    success=False,
                    error=err_msg,
                    trace=trace,
                )

        # 2. Execution environment
        local_scope = dict(kwargs)
        global_scope = {"__builtins__": cls.SAFE_BUILTINS}

        # 3. Execution
        try:
            # Check if code is a simple expression or block
            code_str = skill.code_body.strip()
            trace.append(f"Evaluating skill code: {code_str}")

            # If it's a single expression (e.g. "a + b"), evaluate it directly
            if "\n" not in code_str and not code_str.startswith("def ") and not code_str.startswith("return "):
                val = eval(code_str, global_scope, local_scope)
            else:
                # Multi-line or function block: wrap in a function if needed
                if not code_str.startswith("def "):
                    params_str = ", ".join(skill.parameters)
                    indented_body = "\n".join("    " + line for line in code_str.splitlines())
                    wrapper = f"def _exec({params_str}):\n{indented_body}"
                    exec(wrapper, global_scope, local_scope)  # noqa: S102
                    fn = local_scope["_exec"]
                    val = fn(**{k: kwargs[k] for k in skill.parameters})
                else:
                    exec(code_str, global_scope, local_scope)  # noqa: S102
                    # Find function name
                    fn_name = code_str.split("(")[0].replace("def ", "").strip()
                    val = local_scope[fn_name](**{k: kwargs[k] for k in skill.parameters})

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
        except (ArithmeticError, NameError, TypeError, ValueError, SyntaxError, KeyError, IndexError) as ex:
            err_msg = f"{type(ex).__name__}: {ex}"
            trace.append(f"Execution failed: {err_msg}")
            return ExecutionResult(
                skill_name=skill.name,
                success=False,
                error=err_msg,
                trace=trace,
            )
