"""Safe deterministic execution sandbox for Procedural Skills in LITTLE."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

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

        # Only JSON data crosses the process boundary. Rich objects can expose capabilities.
        import json
        import subprocess

        from little.procedural.restricted_execution import run
        try:
            encoded = json.dumps(kwargs, allow_nan=False)
            inputs = json.loads(encoded)
            if any(not name.isidentifier() or name.startswith('_') for name in skill.parameters):
                raise ValueError("Invalid skill parameter identifier")
            code = skill.code_body.strip()
            call_inputs = {k: inputs[k] for k in skill.parameters if k in inputs}
            assignments = "\n".join(f"{key} = {value!r}" for key, value in call_inputs.items()
                                      if key.isidentifier() and not key.startswith('_'))
            if code.startswith('def '):
                tree = ast.parse(code)
                if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                    raise ValueError("Skill definition must begin with a function definition")
                name = tree.body[0].name
                script = assignments + '\n' + code + f"\nprint(repr({name}(**{call_inputs!r})))"
            elif '\n' in code or code.startswith('return '):
                body = '\n'.join('    ' + line for line in code.splitlines())
                script = assignments + f"\ndef skill_entry({', '.join(skill.parameters)}):\n{body}\nprint(repr(skill_entry(**{call_inputs!r})))"
            else:
                script = assignments + f"\nprint(repr({code}))"
            # repr is used for safe data serialization; ast.literal_eval decodes the result.
            result = run(script, memory_mb=256, builtins=list(active_policy.execution_builtins) + ['print', 'repr'],
                         modules=list(active_policy.execution_modules))
            if result.returncode:
                error = result.stderr.strip()
                if 'ZeroDivisionError' in error: error = 'Division by zero is undefined.'
                return ExecutionResult(skill_name=skill.name, success=False, error=error, trace=trace)
            value = ast.literal_eval(result.stdout.strip())
            trace.append(f"Skill execution succeeded. Output: {value}")
            return ExecutionResult(skill_name=skill.name, success=True, result=value, trace=trace)
        except (ValueError, TypeError, SyntaxError, OSError, subprocess.TimeoutExpired) as exc:
            return ExecutionResult(skill_name=skill.name, success=False, error=f"{type(exc).__name__}: {exc}", trace=trace)
