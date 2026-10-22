"""Versioned catalog for question-clause boundary patterns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _split_question_boundary_pattern(
    clean: str, strategy: str, pattern: str, case_insensitive: bool
) -> list[str]:
    """Split question clauses on question-mark or coordinated lookahead boundaries without regex."""
    if strategy == "question_mark" or pattern.startswith(r"\?"):
        if "?" not in clean:
            return []
        return [part.strip() for part in clean.split("?") if part.strip()]

    # Parse coordinated pattern e.g. `,?\s+and\s+(?=(?:is|are|...)\b)` or `\s+plus\s+(?=what\b)`
    allow_comma = pattern.startswith(",?")
    if "(?=" in pattern:
        prefix_part, lookahead_part = pattern.split("(?=", 1)
    else:
        prefix_part, lookahead_part = pattern, ""

    conj_word = (
        prefix_part.replace(",?", "")
        .replace(r"\s+", " ")
        .replace(r"\s*", "")
        .strip()
    )
    lookahead_clean = (
        lookahead_part.rstrip(")")
        .replace(r"\b", "")
        .replace("(?:", "")
        .replace(")", "")
        .strip()
    )
    starters = {
        w.strip().lower() if case_insensitive else w.strip()
        for w in lookahead_clean.split("|")
        if w.strip()
    }
    if not conj_word:
        return []

    parts: list[str] = []
    n = len(clean)
    last = 0
    i = 0
    while i < n:
        sep_start = i
        if allow_comma and clean[i] == ",":
            j = i + 1
            if j >= n or not clean[j].isspace():
                i += 1
                continue
            while j < n and clean[j].isspace():
                j += 1
        elif clean[i].isspace():
            j = i
            while j < n and clean[j].isspace():
                j += 1
        else:
            i += 1
            continue

        end_conj = j + len(conj_word)
        if end_conj <= n:
            seg = clean[j:end_conj]
            matched_conj = (
                (seg.lower() == conj_word.lower())
                if case_insensitive
                else (seg == conj_word)
            )
            if matched_conj and end_conj < n and clean[end_conj].isspace():
                k = end_conj
                while k < n and clean[k].isspace():
                    k += 1
                # Check lookahead word at position k
                word_end = k
                while word_end < n and (clean[word_end].isalnum() or clean[word_end] == "_"):
                    word_end += 1
                next_word = clean[k:word_end]
                check_word = next_word.lower() if case_insensitive else next_word
                if not starters or check_word in starters:
                    chunk = clean[last:sep_start].strip()
                    if chunk:
                        parts.append(chunk)
                    last = k
                    i = k
                    continue
        i += 1

    if parts:
        tail = clean[last:].strip()
        if tail:
            parts.append(tail)
        return parts
    return []


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
class QuestionBoundaryPattern:
    """One declarative question-clause boundary."""

    name: str
    strategy: str
    pattern: str
    case_insensitive: bool


@dataclass(frozen=True)
class QuestionBoundaryPolicy:
    """Immutable question-clause boundary catalog."""

    patterns: tuple[QuestionBoundaryPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> QuestionBoundaryPolicy:
        directory = Path(directory)
        path = directory / "parser_question_boundary_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_question_boundary_policy.v1":
            raise ValueError(
                f"{path} has an unsupported question-boundary-policy format"
            )

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[QuestionBoundaryPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")
            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"question_mark", "coordinated"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported question-boundary strategy"
                )
            case_insensitive = raw.get("case_insensitive", False)
            if not isinstance(case_insensitive, bool):
                raise TypeError(
                    f"{path} patterns[{index}] case_insensitive must be boolean"
                )
            patterns.append(
                QuestionBoundaryPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    pattern=_required_text(raw, "pattern", path, index),
                    case_insensitive=case_insensitive,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> QuestionBoundaryPolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def split(self, text: str) -> list[str]:
        clean = text.strip()
        for pattern in self.patterns:
            parts = _split_question_boundary_pattern(
                clean,
                pattern.strategy,
                pattern.pattern,
                pattern.case_insensitive,
            )
            if len(parts) > 1:
                return parts
        return [clean] if clean else []
