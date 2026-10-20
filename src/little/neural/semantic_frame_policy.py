"""Policy loader for Neural Semantic Frame extraction and slot mapping."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class SemanticFramePolicy:
    """Configures relational indicators, filler prefixes, and extraction thresholds."""

    min_extraction_confidence: float
    filler_prefixes: tuple[str, ...]
    relational_indicators: dict[str, tuple[str, ...]]
    passive_patterns: dict[str, str]
    negation_tokens: tuple[str, ...]
    stop_determiners: tuple[str, ...]
    operator_canonical_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = self.min_extraction_confidence
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "Minimum extraction confidence must be a finite probability"
            )

    @classmethod
    def load(cls, directory: str | Path) -> SemanticFramePolicy:
        path = Path(directory) / "semantic_frame_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.semantic_frame_policy.v1":
            raise ValueError(f"{path} has an unsupported semantic-frame format")

        min_conf = payload.get("min_extraction_confidence", 0.5)
        fillers = tuple(
            str(p).lower().strip() for p in payload.get("filler_prefixes", [])
        )

        rel_inds: dict[str, tuple[str, ...]] = {}
        for rel, phrase_list in payload.get("relational_indicators", {}).items():
            # Sort phrases by descending length so longer multi-word phrases match first
            sorted_phrases = sorted(
                (str(p).lower().strip() for p in phrase_list), key=len, reverse=True
            )
            rel_inds[rel] = tuple(sorted_phrases)

        passives = {
            str(k).lower(): str(v).lower()
            for k, v in payload.get("passive_patterns", {}).items()
        }
        negations = tuple(
            str(n).lower().strip() for n in payload.get("negation_tokens", [])
        )
        determiners = tuple(
            str(d).lower().strip() for d in payload.get("stop_determiners", [])
        )
        op_canon = {
            str(k).lower().strip(): str(v).lower().strip()
            for k, v in payload.get("operator_canonical_map", {}).items()
        }

        return cls(
            min_extraction_confidence=min_conf,
            filler_prefixes=fillers,
            relational_indicators=rel_inds,
            passive_patterns=passives,
            negation_tokens=negations,
            stop_determiners=determiners,
            operator_canonical_map=op_canon,
        )

    @classmethod
    def default(cls) -> SemanticFramePolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def clean_filler(self, text: str) -> str:
        """Strip conversational filler phrases from the beginning of an utterance without regular expressions."""
        cleaned = text.strip()
        lower = cleaned.lower()

        # Check prefixes ordered by length
        sorted_fillers = sorted(self.filler_prefixes, key=len, reverse=True)
        for prefix in sorted_fillers:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip(",:- \t\n\r")
                lower = cleaned.lower()

        # Strip introductory prepositional phrases like "In Mexico," if followed by comma
        if lower.startswith("in ") and "," in cleaned:
            comma_idx = cleaned.find(",")
            intro = cleaned[3:comma_idx].strip()
            if intro and all(c.isalnum() or c in "_ " for c in intro):
                cleaned = cleaned[comma_idx + 1 :].lstrip()

        return cleaned.strip()

    def strip_determiner(self, token_phrase: str) -> str:
        """Strip leading grammatical determiners ('a', 'an', 'the', 'some')."""
        phrase = token_phrase.strip()
        words = phrase.split()
        if words and words[0].lower() in self.stop_determiners:
            return " ".join(words[1:])
        return phrase
