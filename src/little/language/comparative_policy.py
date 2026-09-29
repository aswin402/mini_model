"""Versioned policy for comparative ordering and strict partial orders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class ComparativeDimension:
    """A single dimension of comparative comparison (e.g., height, age)."""

    dimension: str
    positive_comparatives: tuple[str, ...]
    negative_comparatives: tuple[str, ...]
    positive_superlatives: tuple[str, ...]
    negative_superlatives: tuple[str, ...]
    noun: str


@dataclass(frozen=True)
class ComparativeOrderPolicy:
    """Policy governing parsing and proof templates for comparative orders."""

    dimensions: tuple[ComparativeDimension, ...]
    question_patterns: dict[str, str]
    statement_patterns: tuple[str, ...]
    templates: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> ComparativeOrderPolicy:
        path = Path(directory) / "comparative_order_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.comparative_order_policy.v1":
            raise ValueError(f"{path} has an unsupported comparative-order-policy format")

        raw_dims = payload.get("dimensions", [])
        dims = []
        for d in raw_dims:
            dims.append(
                ComparativeDimension(
                    dimension=d["dimension"],
                    positive_comparatives=tuple(d.get("positive_comparatives", [])),
                    negative_comparatives=tuple(d.get("negative_comparatives", [])),
                    positive_superlatives=tuple(d.get("positive_superlatives", [])),
                    negative_superlatives=tuple(d.get("negative_superlatives", [])),
                    noun=d.get("noun", d["dimension"]),
                )
            )

        return cls(
            dimensions=tuple(dims),
            question_patterns=dict(payload.get("question_patterns", {})),
            statement_patterns=tuple(payload.get("statement_patterns", [])),
            templates=dict(payload.get("templates", {})),
        )

    @classmethod
    def default(cls) -> ComparativeOrderPolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def find_dimension_by_comparative(
        self, comp_word: str
    ) -> tuple[ComparativeDimension, int] | None:
        """Look up dimension and polarity (+1 for positive, -1 for negative)."""
        w = comp_word.strip().lower()
        for dim in self.dimensions:
            if w in dim.positive_comparatives:
                return dim, 1
            if w in dim.negative_comparatives:
                return dim, -1
        return None

    def find_dimension_by_superlative(
        self, sup_word: str
    ) -> tuple[ComparativeDimension, int] | None:
        """Look up dimension and polarity (+1 for max, -1 for min)."""
        w = sup_word.strip().lower()
        for dim in self.dimensions:
            if w in dim.positive_superlatives:
                return dim, 1
            if w in dim.negative_superlatives:
                return dim, -1
        return None
