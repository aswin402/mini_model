"""Versioned configuration for active uncertainty resolution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths
from typing import Any


@dataclass(frozen=True)
class ActiveLearningPolicy:
    """Data-owned guards, binary calibration, and clarification templates."""

    version: str
    meta_words: tuple[str, ...]
    default_predicate: str
    entropy_probability_floor: float
    unknown_confidence_threshold: float
    binary_prior: float
    confidence_scale: float
    round_digits: int
    prompt_templates: dict[str, str]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("active-learning policy version must be non-empty")
        if not self.meta_words or any(not word.strip() for word in self.meta_words):
            raise ValueError("active-learning meta_words must not be empty")
        if not self.default_predicate.strip():
            raise ValueError("active-learning default_predicate must be non-empty")
        for field_name in (
            "entropy_probability_floor",
            "unknown_confidence_threshold",
            "binary_prior",
            "confidence_scale",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 < self.entropy_probability_floor < 0.5:
            raise ValueError("entropy_probability_floor must be in (0, 0.5)")
        if not 0.0 <= self.unknown_confidence_threshold <= 1.0:
            raise ValueError("unknown_confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.binary_prior <= 1.0:
            raise ValueError("binary_prior must be in [0, 1]")
        if not 0.0 <= self.confidence_scale <= 1.0:
            raise ValueError("confidence_scale must be in [0, 1]")
        if (
            isinstance(self.round_digits, bool)
            or not isinstance(self.round_digits, int)
            or self.round_digits < 0
        ):
            raise ValueError("round_digits must be a non-negative integer")
        required_templates = {
            "definition_unknown",
            "subject_unknown",
            "target_unknown",
            "known_relation",
        }
        if not required_templates.issubset(self.prompt_templates):
            raise ValueError("prompt_templates is missing a required clarification template")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.prompt_templates.items()
        ):
            raise ValueError("prompt_templates must map non-empty names to strings")

    @classmethod
    def load(cls, directory: Path) -> ActiveLearningPolicy:
        path = Path(directory) / "active_learning_policy.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            root = payload["active_learning_policy"]
            if not isinstance(root, dict):
                raise TypeError("active_learning_policy must be an object")
            meta_words = root["meta_words"]
            templates = root["prompt_templates"]
            if not isinstance(meta_words, list) or not all(
                isinstance(word, str) for word in meta_words
            ):
                raise TypeError("meta_words must be a list of strings")
            if not isinstance(templates, dict):
                raise TypeError("prompt_templates must be an object")
            return cls(
                version=root["version"],
                meta_words=tuple(word.strip().lower() for word in meta_words),
                default_predicate=root["default_predicate"],
                entropy_probability_floor=root["entropy_probability_floor"],
                unknown_confidence_threshold=root["unknown_confidence_threshold"],
                binary_prior=root["binary_prior"],
                confidence_scale=root["confidence_scale"],
                round_digits=root["round_digits"],
                prompt_templates=dict(templates),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid active-learning policy at {path}: {exc}") from exc

    @classmethod
    def default(cls) -> ActiveLearningPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
