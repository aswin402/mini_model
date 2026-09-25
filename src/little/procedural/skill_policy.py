"""Versioned contracts for procedural skill outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class ProceduralSkillPolicy:
    enabled_skills: tuple[str, ...]
    slice_name_suffix: str
    slice_part_key: str
    slice_exposed_key: str
    slice_skin_key: str
    slice_exposed_value: bool
    slice_skin_value: bool
    execution_builtins: tuple[str, ...]
    execution_modules: tuple[str, ...]
    execution_helpers: tuple[str, ...]

    @classmethod
    def load(cls, directory: Path) -> ProceduralSkillPolicy:
        path = Path(directory) / "procedural_skill_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("procedural_skill_policy")
        if not isinstance(data, dict):
            raise TypeError(
                f"{path} must contain 'procedural_skill_policy' object"
            )
        slice_output = data.get("slice_output")
        if not isinstance(slice_output, dict):
            raise TypeError(f"{path} slice_output must be an object")
        enabled = data.get("enabled_skills")
        if not isinstance(enabled, list) or not all(
            isinstance(value, str) and value.strip() for value in enabled
        ):
            raise TypeError(f"{path} enabled_skills must be a list of strings")
        execution = data.get("execution")
        if not isinstance(execution, dict):
            raise TypeError(f"{path} execution must be an object")

        def names(container: dict[str, object], name: str) -> tuple[str, ...]:
            values = container.get(name)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise TypeError(f"{path} execution field {name!r} must be a list of strings")
            return tuple(value.strip() for value in values)

        def text(name: str) -> str:
            value = slice_output.get(name)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{path} slice_output field {name!r} must be a string")
            return value.strip()

        def boolean(name: str) -> bool:
            value = slice_output.get(name)
            if not isinstance(value, bool):
                raise TypeError(
                    f"{path} slice_output field {name!r} must be a boolean"
                )
            return value

        return cls(
            enabled_skills=tuple(value.strip().upper() for value in enabled),
            slice_name_suffix=text("name_suffix"),
            slice_part_key=text("part_key"),
            slice_exposed_key=text("exposed_key"),
            slice_skin_key=text("skin_key"),
            slice_exposed_value=boolean("exposed_value"),
            slice_skin_value=boolean("skin_value"),
            execution_builtins=names(execution, "builtins"),
            execution_modules=names(execution, "modules"),
            execution_helpers=names(execution, "helpers"),
        )

    @classmethod
    def default(cls) -> ProceduralSkillPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
