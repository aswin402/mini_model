"""Versioned catalog for numeric argument tokenization patterns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _required_text(
    raw: dict[str, Any], field: str, path: Path, index: int
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"{path} patterns[{index}] field {field!r} must be a non-empty string"
        )
    return value.strip()


@dataclass(frozen=True)
class NumberTokenPattern:
    """One declarative separator for numeric argument lists."""

    name: str
    pattern: str
    case_insensitive: bool


@dataclass(frozen=True)
class NumberTokenPolicy:
    """Immutable numeric token boundary catalog."""

    patterns: tuple[NumberTokenPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> NumberTokenPolicy:
        directory = Path(directory)
        path = directory / "parser_number_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_number_policy.v1":
            raise ValueError(f"{path} has an unsupported number-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[NumberTokenPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            case_insensitive = raw.get("case_insensitive", False)
            if not isinstance(case_insensitive, bool):
                raise TypeError(
                    f"{path} patterns[{index}] case_insensitive must be boolean"
                )
            patterns.append(
                NumberTokenPattern(
                    name=_required_text(raw, "name", path, index),
                    pattern=_required_text(raw, "pattern", path, index),
                    case_insensitive=case_insensitive,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> NumberTokenPolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def split(self, text: str) -> list[str]:
        parts = [text]
        for pattern in self.patterns:
            flags = re.IGNORECASE if pattern.case_insensitive else 0
            parts = [
                part
                for current in parts
                for part in re.split(pattern.pattern, current, flags=flags)
            ]
        return parts
