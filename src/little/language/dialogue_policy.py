"""Versioned vocabulary and boundary policy for dialogue references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _words(payload: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    values = payload.get(key)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise TypeError(f"{path} field {key!r} must be a list of strings")
    return tuple(value.strip().lower() for value in values)


@dataclass(frozen=True)
class DialogueReferencePolicy:
    former_references: tuple[str, ...]
    latter_references: tuple[str, ...]
    plural_pronouns: tuple[str, ...]
    person_pronouns: tuple[str, ...]
    singular_nonhuman_pronouns: tuple[str, ...]
    text_pronouns: tuple[str, ...]
    relative_pronouns: tuple[str, ...]
    relative_prefix_exclusions: tuple[str, ...]
    relative_suffix_exclusions: tuple[str, ...]
    person_categories: tuple[str, ...]
    conditional_guard_pattern: str

    @classmethod
    def load(cls, directory: Path) -> DialogueReferencePolicy:
        path = Path(directory) / "dialogue_reference_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "little.dialogue_reference_policy.v1":
            raise ValueError(f"{path} has unsupported dialogue policy format")
        conditional_guard_pattern = payload.get("conditional_guard_pattern")
        if not isinstance(conditional_guard_pattern, str):
            raise TypeError(
                f"{path} field 'conditional_guard_pattern' must be a string"
            )
        return cls(
            former_references=_words(payload, "former_references", path),
            latter_references=_words(payload, "latter_references", path),
            plural_pronouns=_words(payload, "plural_pronouns", path),
            person_pronouns=_words(payload, "person_pronouns", path),
            singular_nonhuman_pronouns=_words(
                payload, "singular_nonhuman_pronouns", path
            ),
            text_pronouns=_words(payload, "text_pronouns", path),
            relative_pronouns=_words(payload, "relative_pronouns", path),
            relative_prefix_exclusions=_words(
                payload, "relative_prefix_exclusions", path
            ),
            relative_suffix_exclusions=_words(
                payload, "relative_suffix_exclusions", path
            ),
            person_categories=_words(payload, "person_categories", path),
            conditional_guard_pattern=conditional_guard_pattern,
        )

    @classmethod
    def default(cls) -> DialogueReferencePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
