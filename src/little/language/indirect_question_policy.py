"""Versioned catalog for embedded polar-question routes."""

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


def _required_group(
    raw: dict[str, Any], field: str, path: Path, index: int
) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or value <= 0:
        raise TypeError(
            f"{path} patterns[{index}] field {field!r} must be a positive integer"
        )
    return value


@dataclass(frozen=True)
class IndirectQuestionPattern:
    """One embedded-question matching and inversion strategy."""

    name: str
    strategy: str
    pattern: str
    body_group: int
    exclusion_pattern: str
    inverse_pattern: str
    inverse_subject_group: int
    inverse_auxiliary_group: int
    inverse_object_group: int


@dataclass(frozen=True)
class IndirectQuestionPolicy:
    """Immutable embedded polar-question catalog."""

    patterns: tuple[IndirectQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> IndirectQuestionPolicy:
        directory = Path(directory)
        path = directory / "parser_indirect_question_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_indirect_question_policy.v1":
            raise ValueError(
                f"{path} has an unsupported indirect-question-policy format"
            )

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[IndirectQuestionPattern] = []
        for index, raw in enumerate(raw_patterns):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} patterns[{index}] must be an object")

            strategy = _required_text(raw, "strategy", path, index)
            if strategy not in {"embedded_polar"}:
                raise ValueError(
                    f"{path} patterns[{index}] has an unsupported indirect-question strategy"
                )
            patterns.append(
                IndirectQuestionPattern(
                    name=_required_text(raw, "name", path, index),
                    strategy=strategy,
                    pattern=_required_text(raw, "pattern", path, index),
                    body_group=_required_group(raw, "body_group", path, index),
                    exclusion_pattern=_required_text(
                        raw, "exclusion_pattern", path, index
                    ),
                    inverse_pattern=_required_text(
                        raw, "inverse_pattern", path, index
                    ),
                    inverse_subject_group=_required_group(
                        raw, "inverse_subject_group", path, index
                    ),
                    inverse_auxiliary_group=_required_group(
                        raw, "inverse_auxiliary_group", path, index
                    ),
                    inverse_object_group=_required_group(
                        raw, "inverse_object_group", path, index
                    ),
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> IndirectQuestionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
