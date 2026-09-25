"""Versioned data catalog for deterministic procedural skill definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SkillDefinition:
    """Declarative definition for one procedural skill."""

    name: str
    parameters: tuple[str, ...]
    description: str
    code: str | None = None
    implementation: str | None = None


@dataclass(frozen=True)
class ProceduralSkillCatalog:
    """Validated, immutable catalog loaded from project data."""

    definitions: tuple[SkillDefinition, ...]

    @classmethod
    def load(cls, directory: str | Path) -> ProceduralSkillCatalog:
        directory = Path(directory)
        path = directory / "procedural_skill_catalog.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.procedural_skill_catalog.v1":
            raise ValueError(f"{path} has an unsupported catalog format")

        entries = payload.get("skills")
        if not isinstance(entries, list):
            raise TypeError(f"{path} skills must be a list")

        definitions: list[SkillDefinition] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"{path} skills[{index}] must be an object")

            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise TypeError(f"{path} skills[{index}] name must be a string")

            parameters = entry.get("parameters")
            if not isinstance(parameters, list) or not all(
                isinstance(value, str) and value.strip() for value in parameters
            ):
                raise TypeError(
                    f"{path} skills[{index}] parameters must be a list of strings"
                )

            description = entry.get("description", "")
            if not isinstance(description, str):
                raise TypeError(f"{path} skills[{index}] description must be a string")

            code = entry.get("code")
            code_lines = entry.get("code_lines")
            if code is not None and code_lines is not None:
                raise ValueError(
                    f"{path} skills[{index}] cannot define both code and code_lines"
                )
            if code_lines is not None:
                if not isinstance(code_lines, list) or not all(
                    isinstance(value, str) for value in code_lines
                ):
                    raise TypeError(
                        f"{path} skills[{index}] code_lines must be a list of strings"
                    )
                code = "\n".join(code_lines)
            if code is not None and (
                not isinstance(code, str) or not code.strip()
            ):
                raise TypeError(
                    f"{path} skills[{index}] code must be a non-empty string"
                )

            implementation = entry.get("implementation")
            if implementation is not None and (
                not isinstance(implementation, str) or not implementation.strip()
            ):
                raise TypeError(
                    f"{path} skills[{index}] implementation must be a string"
                )
            if (code is None) == (implementation is None):
                raise ValueError(
                    f"{path} skills[{index}] must define exactly one of code or implementation"
                )

            definitions.append(
                SkillDefinition(
                    name=name.strip().upper(),
                    parameters=tuple(value.strip() for value in parameters),
                    description=description.strip(),
                    code=code.strip() if isinstance(code, str) else None,
                    implementation=(
                        implementation.strip()
                        if isinstance(implementation, str)
                        else None
                    ),
                )
            )

        return cls(definitions=tuple(definitions))

    @classmethod
    def default(cls) -> ProceduralSkillCatalog:
        return cls.load(RuntimePaths.default().schema_directory)
