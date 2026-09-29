"""Versioned policy governing conversational surface responses, social protocols, and natural verbalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SocialIntent:
    name: str
    patterns: tuple[str, ...]
    responses: tuple[str, ...]


@dataclass(frozen=True)
class DialogueSurfacePolicy:
    """Policy for natural conversational surface dialogue."""

    social_intents: dict[str, SocialIntent]
    leading_greeting_regex: str
    verbalizer_templates: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> DialogueSurfacePolicy:
        path = Path(directory) / "dialogue_surface_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.dialogue_surface_policy.v1":
            raise ValueError(f"{path} has an unsupported dialogue-surface-policy format")

        social_dict: dict[str, SocialIntent] = {}
        raw_intents = payload.get("social_intents", {})
        for name, data in raw_intents.items():
            social_dict[name] = SocialIntent(
                name=name,
                patterns=tuple(data.get("patterns", [])),
                responses=tuple(data.get("responses", [])),
            )

        splitter = payload.get("mixed_intent_splitter", {})
        leading_rgx = splitter.get("leading_greeting_regex", "")
        templates = dict(payload.get("verbalizer_templates", {}))

        return cls(
            social_intents=social_dict,
            leading_greeting_regex=leading_rgx,
            verbalizer_templates=templates,
        )

    @classmethod
    def default(cls) -> DialogueSurfacePolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def match_social_intent(self, text: str) -> tuple[str, str] | None:
        """Match purely social inputs (greetings, pleasantries, identity) against policy."""
        s = text.strip()
        for name, intent in self.social_intents.items():
            for pat in intent.patterns:
                if re.match(pat, s, re.IGNORECASE):
                    resp = intent.responses[0] if intent.responses else ""
                    return name, resp
        return None

    def decompose_leading_greeting(self, text: str) -> tuple[str, str] | None:
        """Decompose mixed utterance like 'Hello! Can an eagle fly?' into ('Hello', 'Can an eagle fly?')."""
        if not self.leading_greeting_regex:
            return None
        m = re.match(self.leading_greeting_regex, text.strip(), re.IGNORECASE)
        if m:
            greeting_word = m.group(1).strip()
            remainder = m.group(2).strip()
            return greeting_word, remainder
        return None

    def get_template(self, key: str, default: str = "") -> str:
        """Get verbalizer template string from loaded policy."""
        return self.verbalizer_templates.get(key, default)
