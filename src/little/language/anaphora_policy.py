"""Versioned catalog for dialogue anaphora guards and replacements."""

from __future__ import annotations

import json
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
class AnaphoraPattern:
    """One ordered regular-expression replacement rule."""

    name: str
    pattern: str
    replacement: str
    case_insensitive: bool


@dataclass(frozen=True)
class AnaphoraPolicy:
    """Immutable anaphora guard and replacement catalog."""

    guard_patterns: tuple[str, ...]
    patterns: tuple[AnaphoraPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> AnaphoraPolicy:
        directory = Path(directory)
        path = directory / "parser_anaphora_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_anaphora_policy.v1":
            raise ValueError(f"{path} has an unsupported anaphora-policy format")

        raw_guards = payload.get("guard_patterns")
        if not isinstance(raw_guards, list) or not all(
            isinstance(value, str) and value.strip() for value in raw_guards
        ):
            raise TypeError(f"{path} guard_patterns must be a list of strings")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[AnaphoraPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            case_insensitive = raw.get("case_insensitive", True)
            if not isinstance(case_insensitive, bool):
                raise TypeError(
                    f"{path} patterns[{index}] case_insensitive must be boolean"
                )
            patterns.append(
                AnaphoraPattern(
                    name=_required_text(raw, "name", path, index),
                    pattern=_required_text(raw, "pattern", path, index),
                    replacement=_required_text(raw, "replacement", path, index),
                    case_insensitive=case_insensitive,
                )
            )

        return cls(
            guard_patterns=tuple(value.strip() for value in raw_guards),
            patterns=tuple(patterns),
        )

    @classmethod
    def default(cls) -> AnaphoraPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
