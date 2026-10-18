"""Policy loader for Headroom reversible context compression and memory budgeting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class HeadroomPolicy:
    """Configures active context bounds, salience weights, and memory budget."""

    max_active_tokens: int
    max_active_concepts: int
    max_working_episodes: int
    recency_decay_lambda: float
    salience_weights: dict[str, float]
    eviction_salience_threshold: float
    memory_headroom_budget_mb: float

    def __post_init__(self) -> None:
        for name in ("max_active_tokens", "max_active_concepts", "max_working_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("recency_decay_lambda", "eviction_salience_threshold", "memory_headroom_budget_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.memory_headroom_budget_mb <= 0 or self.eviction_salience_threshold > 1:
            raise ValueError("headroom budget must be positive and salience threshold at most one")
        if not self.salience_weights or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 for value in self.salience_weights.values()
        ) or sum(self.salience_weights.values()) <= 0:
            raise ValueError("salience weights must be finite, non-negative, and not all zero")

    @classmethod
    def load(cls, directory: str | Path) -> HeadroomPolicy:
        path = Path(directory) / "headroom_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.headroom.v1":
            raise ValueError(f"{path} has an unsupported headroom format")

        tokens = payload.get("max_active_tokens", 2048)
        concepts = payload.get("max_active_concepts", 64)
        episodes = payload.get("max_working_episodes", 32)
        decay = payload.get("recency_decay_lambda", 0.01)
        weights = dict(payload.get("salience_weights", {"recency": 0.4, "frequency": 0.3, "activation": 0.3}))
        thresh = payload.get("eviction_salience_threshold", 0.25)
        budget = payload.get("memory_headroom_budget_mb", 50.0)

        return cls(
            max_active_tokens=tokens,
            max_active_concepts=concepts,
            max_working_episodes=episodes,
            recency_decay_lambda=decay,
            salience_weights=weights,
            eviction_salience_threshold=thresh,
            memory_headroom_budget_mb=budget,
        )

    @classmethod
    def default(cls) -> HeadroomPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
