"""Versioned language and grounding policy loaded from data files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class LanguagePolicy:
    math_markers: tuple[str, ...]
    curiosity_markers: tuple[str, ...]
    action_prefixes: tuple[str, ...]
    question_prefixes: tuple[str, ...]
    concept_associations: dict[str, frozenset[str]]
    greetings: tuple[str, ...] = ()
    help_commands: tuple[str, ...] = ()
    identity_commands: tuple[str, ...] = ()
    exit_commands: tuple[str, ...] = ()
    clear_commands: tuple[str, ...] = ()
    memory_commands: tuple[str, ...] = ()
    skills_commands: tuple[str, ...] = ()
    cancel_words: tuple[str, ...] = ()
    positive_affirmations: tuple[str, ...] = ()
    negative_denials: tuple[str, ...] = ()
    discourse_prefixes: tuple[str, ...] = ()

    @classmethod
    def load(cls, directory: Path) -> LanguagePolicy:
        path = Path(directory) / "language_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("language_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain an object under 'language_policy'")

        def read_words(key: str) -> tuple[str, ...]:
            values = data.get(key)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise TypeError(f"{path} field {key!r} must be a list of strings")
            if key == "question_prefixes":
                return tuple(value.lower() for value in values if value.strip())
            return tuple(value.strip().lower() for value in values if value.strip())

        raw_associations = data.get("concept_associations")
        if not isinstance(raw_associations, dict):
            raise TypeError(
                f"{path} field 'concept_associations' must be an object"
            )
        associations: dict[str, frozenset[str]] = {}
        for concept, values in raw_associations.items():
            if not isinstance(concept, str) or not isinstance(values, list):
                raise TypeError(
                    f"{path} concept associations must map strings to string lists"
                )
            if not all(isinstance(value, str) for value in values):
                raise TypeError(
                    f"{path} concept association values must be strings"
                )
            associations[concept.strip().lower()] = frozenset(
                value.strip().lower() for value in values if value.strip()
            )

        return cls(
            math_markers=read_words("math_markers"),
            curiosity_markers=read_words("curiosity_markers"),
            action_prefixes=read_words("action_prefixes"),
            question_prefixes=read_words("question_prefixes"),
            concept_associations=associations,
            greetings=read_words("greetings"),
            help_commands=read_words("help_commands"),
            identity_commands=read_words("identity_commands"),
            exit_commands=read_words("exit_commands"),
            clear_commands=read_words("clear_commands"),
            memory_commands=read_words("memory_commands"),
            skills_commands=read_words("skills_commands"),
            cancel_words=read_words("cancel_words"),
            positive_affirmations=read_words("positive_affirmations"),
            negative_denials=read_words("negative_denials"),
            discourse_prefixes=read_words("discourse_prefixes"),
        )

    @classmethod
    def default(cls) -> LanguagePolicy:
        return cls.load(RuntimePaths.default().schema_directory)
