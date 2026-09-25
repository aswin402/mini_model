"""Versioned catalog for deterministic arithmetic parser patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class MathPattern:
    """Declarative binary arithmetic pattern and capture mapping."""

    name: str
    pattern: str
    argument_groups: tuple[int, int]
    skill: str | None = None
    skill_by_operator: Mapping[str, str] = MappingProxyType({})
    operator_group: int | None = None
    reverse_arguments: bool = False


@dataclass(frozen=True)
class MathPatternPolicy:
    """Immutable arithmetic pattern catalog."""

    patterns: tuple[MathPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> MathPatternPolicy:
        directory = Path(directory)
        path = directory / "parser_math_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_math_policy.v1":
            raise ValueError(f"{path} has an unsupported math-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[MathPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")

            def text(name: str) -> str:
                value = raw.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise TypeError(
                        f"{path} patterns[{index}] field {name!r} must be a string"
                    )
                return value.strip()

            raw_groups = raw.get("argument_groups")
            if (
                not isinstance(raw_groups, list)
                or len(raw_groups) != 2
                or not all(isinstance(value, int) and value > 0 for value in raw_groups)
            ):
                raise TypeError(
                    f"{path} patterns[{index}] argument_groups must contain two positive integers"
                )

            raw_skill = raw.get("skill")
            skill = None
            if raw_skill is not None:
                if not isinstance(raw_skill, str) or not raw_skill.strip():
                    raise TypeError(
                        f"{path} patterns[{index}] skill must be a non-empty string"
                    )
                skill = raw_skill.strip().upper()

            raw_operator_map = raw.get("skill_by_operator")
            operator_map: dict[str, str] = {}
            if raw_operator_map is not None:
                if not isinstance(raw_operator_map, dict) or not all(
                    isinstance(key, str)
                    and key
                    and isinstance(value, str)
                    and value.strip()
                    for key, value in raw_operator_map.items()
                ):
                    raise TypeError(
                        f"{path} patterns[{index}] skill_by_operator must map strings to strings"
                    )
                operator_map = {
                    key.lower(): value.strip().upper()
                    for key, value in raw_operator_map.items()
                }

            if (skill is None) == (not operator_map):
                raise ValueError(
                    f"{path} patterns[{index}] must define exactly one skill source"
                )

            operator_group = raw.get("operator_group")
            if operator_map and (
                not isinstance(operator_group, int) or operator_group <= 0
            ):
                raise TypeError(
                    f"{path} patterns[{index}] operator_group must be a positive integer"
                )
            reverse_arguments = raw.get("reverse_arguments", False)
            if not isinstance(reverse_arguments, bool):
                raise TypeError(
                    f"{path} patterns[{index}] reverse_arguments must be boolean"
                )

            patterns.append(
                MathPattern(
                    name=text("name"),
                    pattern=text("pattern"),
                    argument_groups=(raw_groups[0], raw_groups[1]),
                    skill=skill,
                    skill_by_operator=MappingProxyType(operator_map),
                    operator_group=operator_group,
                    reverse_arguments=reverse_arguments,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> MathPatternPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
