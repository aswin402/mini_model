"""Versioned policy for algebraic word problems and multi-entity linear systems."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class AlgebraicWordProblemPolicy:
    """Policy governing linguistic markers and templates for algebraic word problems."""

    multipliers: dict[str, float]
    positive_offsets: tuple[str, ...]
    negative_offsets: tuple[str, ...]
    temporal_shifts: dict[str, str]
    sum_keywords: tuple[str, ...]
    question_patterns: dict[str, str]
    templates: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> AlgebraicWordProblemPolicy:
        path = Path(directory) / "algebraic_word_problem_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.algebraic_word_problem_policy.v1":
            raise ValueError(f"{path} has an unsupported algebraic-word-problem-policy format")

        raw_mults = payload.get("multipliers", {})
        mults = {k.lower(): float(v) for k, v in raw_mults.items()}

        pos_offsets = tuple(str(x).lower() for x in payload.get("positive_offsets", []))
        neg_offsets = tuple(str(x).lower() for x in payload.get("negative_offsets", []))
        temp_shifts = dict(payload.get("temporal_shifts", {}))
        sum_keys = tuple(str(x).lower() for x in payload.get("sum_keywords", []))
        q_patterns = dict(payload.get("question_patterns", {}))
        templates = dict(payload.get("templates", {}))

        return cls(
            multipliers=mults,
            positive_offsets=pos_offsets,
            negative_offsets=neg_offsets,
            temporal_shifts=temp_shifts,
            sum_keywords=sum_keys,
            question_patterns=q_patterns,
            templates=templates,
        )

    @classmethod
    def default(cls) -> AlgebraicWordProblemPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
