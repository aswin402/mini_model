"""Versioned configuration for external knowledge normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class KnowledgeImportPolicy:
    relation_aliases: dict[str, str]
    discard_concepts: frozenset[str]
    max_concept_words: int
    max_concept_characters: int

    @classmethod
    def load(cls, directory: Path) -> KnowledgeImportPolicy:
        path = Path(directory) / "knowledge_import_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("knowledge_import_policy")
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain 'knowledge_import_policy' object")

        aliases = data.get("relation_aliases")
        discard = data.get("discard_concepts")
        limits = data.get("concept_limits")
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in aliases.items()
        ):
            raise TypeError(f"{path} relation_aliases must map strings to strings")
        if not isinstance(discard, list) or not all(
            isinstance(value, str) and value.strip() for value in discard
        ):
            raise TypeError(f"{path} discard_concepts must be a list of strings")
        if not isinstance(limits, dict):
            raise TypeError(f"{path} concept_limits must be an object")

        def positive_int(name: str) -> int:
            value = limits.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TypeError(f"{path} concept limit {name!r} must be a positive integer")
            return value

        return cls(
            relation_aliases={
                key.strip().lower(): value.strip().lower()
                for key, value in aliases.items()
            },
            discard_concepts=frozenset(value.strip().lower() for value in discard),
            max_concept_words=positive_int("max_words"),
            max_concept_characters=positive_int("max_characters"),
        )

    @classmethod
    def default(cls) -> KnowledgeImportPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
