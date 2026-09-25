"""Versioned catalog for deterministic arithmetic parser patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class MathCapture:
    """Declarative conversion for one regular-expression capture."""

    group: int
    name: str
    value_type: str = "number"
    normalizers: tuple[str, ...] = ()
    sign_group: int | None = None
    include: bool = True


@dataclass(frozen=True)
class MathPattern:
    """Declarative arithmetic pattern, capture conversion, and argument mapping."""

    name: str
    pattern: str
    argument_groups: tuple[int, ...]
    skill: str | None = None
    skill_by_operator: Mapping[str, str] = MappingProxyType({})
    operator_group: int | None = None
    reverse_arguments: bool = False
    argument_names: tuple[str, ...] = ("a", "b")
    value_type: str = "number"
    fixed_arguments: Mapping[str, int | float] = MappingProxyType({})
    captures: tuple[MathCapture, ...] = ()


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

            raw_captures = raw.get("captures")
            captures: tuple[MathCapture, ...] = ()
            if raw_captures is not None:
                if not isinstance(raw_captures, list) or not raw_captures:
                    raise TypeError(
                        f"{path} patterns[{index}] captures must be a non-empty list"
                    )

                parsed_captures: list[MathCapture] = []
                capture_names: set[str] = set()
                for capture_index, raw_capture in enumerate(raw_captures):
                    if not isinstance(raw_capture, dict):
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] must be an object"
                        )
                    group = raw_capture.get("group")
                    name = raw_capture.get("name")
                    value_type = raw_capture.get("value_type", "number")
                    normalizers = raw_capture.get("normalizers", [])
                    sign_group = raw_capture.get("sign_group")
                    include = raw_capture.get("include", True)
                    if not isinstance(group, int) or group <= 0:
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] group must be a positive integer"
                        )
                    if not isinstance(name, str) or not name.strip():
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] name must be a non-empty string"
                        )
                    name = name.strip()
                    if name in capture_names:
                        raise ValueError(
                            f"{path} patterns[{index}] capture names must be unique"
                        )
                    capture_names.add(name)
                    if value_type not in {"number", "text", "numbers"}:
                        raise ValueError(
                            f"{path} patterns[{index}] captures[{capture_index}] value_type must be number, text, or numbers"
                        )
                    if not isinstance(normalizers, list) or not all(
                        isinstance(value, str) and value.strip()
                        for value in normalizers
                    ):
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] normalizers must be strings"
                        )
                    if sign_group is not None and (
                        not isinstance(sign_group, int) or sign_group <= 0
                    ):
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] sign_group must be a positive integer"
                        )
                    if not isinstance(include, bool):
                        raise TypeError(
                            f"{path} patterns[{index}] captures[{capture_index}] include must be boolean"
                        )
                    parsed_captures.append(
                        MathCapture(
                            group=group,
                            name=name,
                            value_type=value_type,
                            normalizers=tuple(
                                value.strip().lower() for value in normalizers
                            ),
                            sign_group=sign_group,
                            include=include,
                        )
                    )

                captures = tuple(parsed_captures)
                raw_groups = [capture.group for capture in captures]
                argument_names = tuple(capture.name for capture in captures)
            else:
                raw_groups = raw.get("argument_groups")
                if (
                    not isinstance(raw_groups, list)
                    or not raw_groups
                    or not all(
                        isinstance(value, int) and value > 0 for value in raw_groups
                    )
                ):
                    raise TypeError(
                        f"{path} patterns[{index}] argument_groups must contain positive integers"
                    )

                raw_names = raw.get("argument_names")
                if raw_names is None:
                    if len(raw_groups) != 2:
                        raise TypeError(
                            f"{path} patterns[{index}] argument_names is required for non-binary patterns"
                        )
                    argument_names = ("a", "b")
                elif (
                    not isinstance(raw_names, list)
                    or len(raw_names) != len(raw_groups)
                    or not all(
                        isinstance(value, str) and value.strip()
                        for value in raw_names
                    )
                    or len({value.strip() for value in raw_names}) != len(raw_names)
                ):
                    raise TypeError(
                        f"{path} patterns[{index}] argument_names must contain unique non-empty strings matching argument_groups"
                    )
                else:
                    argument_names = tuple(value.strip() for value in raw_names)

            value_type = raw.get("value_type", "number")
            if value_type not in {"number", "text", "numbers"}:
                raise ValueError(
                    f"{path} patterns[{index}] value_type must be number, text, or numbers"
                )

            raw_fixed_arguments = raw.get("fixed_arguments", {})
            if not isinstance(raw_fixed_arguments, dict) or not all(
                isinstance(key, str)
                and key.strip()
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                for key, value in raw_fixed_arguments.items()
            ):
                raise TypeError(
                    f"{path} patterns[{index}] fixed_arguments must map names to numbers"
                )
            fixed_arguments = {
                key.strip(): value for key, value in raw_fixed_arguments.items()
            }

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
                    argument_groups=tuple(raw_groups),
                    skill=skill,
                    skill_by_operator=MappingProxyType(operator_map),
                    operator_group=operator_group,
                    reverse_arguments=reverse_arguments,
                    argument_names=argument_names,
                    value_type=value_type,
                    fixed_arguments=MappingProxyType(fixed_arguments),
                    captures=captures,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> MathPatternPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
