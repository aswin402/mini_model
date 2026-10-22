"""Versioned catalog for numeric argument tokenization patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _split_number_pattern(
    text: str, pattern: str, case_insensitive: bool
) -> list[str]:
    """Split numeric argument strings on declarative separator patterns without regex."""
    if not text:
        return [text]

    if pattern == r"\s*(?:,|\band\b)\s*|\s+":
        norm = text.replace(",", " ")
        tokens = norm.split()
        out: list[str] = []
        for tok in tokens:
            is_and = (tok.lower() == "and") if case_insensitive else (tok == "and")
            if not is_and:
                out.append(tok)
        return out if out else [""]

    cleaned = (
        pattern.replace(r"\s+", " ")
        .replace(r"\s*", "")
        .replace(r"\b", "")
        .strip()
    )
    if not cleaned:
        return text.split()

    out = []
    n = len(text)
    last = 0
    i = 0
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            end = j + len(cleaned)
            if end <= n:
                seg = text[j:end]
                matched = (
                    (seg.lower() == cleaned.lower())
                    if case_insensitive
                    else (seg == cleaned)
                )
                if matched and end < n and text[end].isspace():
                    k = end
                    while k < n and text[k].isspace():
                        k += 1
                    out.append(text[last:i])
                    last = k
                    i = k
                    continue
        i += 1
    out.append(text[last:])
    return out


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
            parts = [
                part
                for current in parts
                for part in _split_number_pattern(
                    current, pattern.pattern, pattern.case_insensitive
                )
            ]
        return parts
