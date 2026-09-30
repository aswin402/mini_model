"""Policy loader for Headroom reversible context compression and memory budgeting."""

from __future__ import annotations

import json
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

    @classmethod
    def load(cls, directory: str | Path) -> HeadroomPolicy:
        path = Path(directory) / "headroom_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.headroom.v1":
            raise ValueError(f"{path} has an unsupported headroom format")

        tokens = int(payload.get("max_active_tokens", 2048))
        concepts = int(payload.get("max_active_concepts", 64))
        episodes = int(payload.get("max_working_episodes", 32))
        decay = float(payload.get("recency_decay_lambda", 0.01))
        weights = dict(payload.get("salience_weights", {"recency": 0.4, "frequency": 0.3, "activation": 0.3}))
        thresh = float(payload.get("eviction_salience_threshold", 0.25))
        budget = float(payload.get("memory_headroom_budget_mb", 50.0))

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
