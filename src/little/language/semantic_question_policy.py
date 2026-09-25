"""Versioned catalog for direct semantic question routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SemanticQuestionPattern:
    """One semantic relation query with configurable capture sources."""

    name: str
    pattern: str
    predicate: str | None
    predicate_role: str | None
    subject_groups: tuple[int, ...]
    subject_literal: str | None
    target_groups: tuple[int, ...]
    target_literal: str | None
    validate_subject: bool
    validate_target: bool
    split_strategy: str | None = None
    body_group: int | None = None
    phase: str = "direct"


@dataclass(frozen=True)
class SemanticQuestionPolicy:
    """Immutable direct semantic-question catalog."""

    patterns: tuple[SemanticQuestionPattern, ...]

    @classmethod
    def load(cls, directory: str | Path) -> SemanticQuestionPolicy:
        directory = Path(directory)
        path = directory / "parser_semantic_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.parser_semantic_policy.v1":
            raise ValueError(f"{path} has an unsupported semantic-policy format")

        raw_patterns = payload.get("patterns")
        if not isinstance(raw_patterns, list):
            raise TypeError(f"{path} patterns must be a list")

        patterns: list[SemanticQuestionPattern] = []
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

            def groups(name: str) -> tuple[int, ...]:
                value = raw.get(name, [])
                if not isinstance(value, list) or not all(
                    isinstance(group, int) and group > 0 for group in value
                ):
                    raise TypeError(
                        f"{path} patterns[{index}] field {name!r} must contain positive integers"
                    )
                return tuple(value)

            raw_predicate = raw.get("predicate")
            raw_role = raw.get("predicate_role")
            has_predicate = isinstance(raw_predicate, str) and bool(raw_predicate.strip())
            has_role = isinstance(raw_role, str) and bool(raw_role.strip())
            if has_predicate == has_role:
                raise ValueError(
                    f"{path} patterns[{index}] must define exactly one of predicate or predicate_role"
                )

            subject_groups = groups("subject_groups")
            target_groups = groups("target_groups")
            subject_literal = raw.get("subject_literal")
            target_literal = raw.get("target_literal")
            if subject_literal is not None and (
                not isinstance(subject_literal, str) or not subject_literal.strip()
            ):
                raise TypeError(
                    f"{path} patterns[{index}] subject_literal must be a string or null"
                )
            if target_literal is not None and (
                not isinstance(target_literal, str) or not target_literal.strip()
            ):
                raise TypeError(
                    f"{path} patterns[{index}] target_literal must be a string or null"
                )
            split_strategy = raw.get("split_strategy")
            body_group = raw.get("body_group")
            phase = raw.get("phase", "direct")
            if not isinstance(phase, str) or not phase.strip():
                raise TypeError(
                    f"{path} patterns[{index}] phase must be a non-empty string"
                )
            phase = phase.strip().lower()
            if split_strategy is not None:
                if split_strategy != "known_concepts_or_last_token":
                    if split_strategy not in {
                        "known_subject_or_tail",
                        "action_verb_tail",
                        "known_concepts_or_valid_last_token",
                    }:
                        raise ValueError(
                            f"{path} patterns[{index}] has an unsupported split_strategy"
                        )
                if not isinstance(body_group, int) or body_group <= 0:
                    raise TypeError(
                        f"{path} patterns[{index}] body_group must be a positive integer"
                    )
                if (
                    subject_groups
                    or target_groups
                    or subject_literal is not None
                    or target_literal is not None
                ):
                    raise ValueError(
                        f"{path} patterns[{index}] split strategies cannot define subject or target sources"
                    )
            elif (bool(subject_groups) == (subject_literal is not None)) or (
                bool(target_groups) == (target_literal is not None)
            ):
                raise ValueError(
                    f"{path} patterns[{index}] must define exactly one source for subject and target"
                )

            validate_subject = raw.get(
                "validate_subject",
                False if split_strategy is not None else subject_literal != "?",
            )
            validate_target = raw.get(
                "validate_target",
                False if split_strategy is not None else target_literal != "?",
            )
            if not isinstance(validate_subject, bool) or not isinstance(
                validate_target, bool
            ):
                raise TypeError(
                    f"{path} patterns[{index}] validation flags must be boolean"
                )

            patterns.append(
                SemanticQuestionPattern(
                    name=text("name"),
                    pattern=text("pattern"),
                    predicate=raw_predicate.strip() if has_predicate else None,
                    predicate_role=raw_role.strip().lower() if has_role else None,
                    subject_groups=subject_groups,
                    subject_literal=(
                        subject_literal.strip()
                        if isinstance(subject_literal, str)
                        else None
                    ),
                    target_groups=target_groups,
                    target_literal=(
                        target_literal.strip()
                        if isinstance(target_literal, str)
                        else None
                    ),
                    validate_subject=validate_subject,
                    validate_target=validate_target,
                    split_strategy=split_strategy,
                    body_group=body_group,
                    phase=phase,
                )
            )

        return cls(patterns=tuple(patterns))

    @classmethod
    def default(cls) -> SemanticQuestionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
