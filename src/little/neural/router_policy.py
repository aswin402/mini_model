"""Policy loader for the Neural Router configuration and semantic anchors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths
from little.neural.models import Pathway


@dataclass(frozen=True)
class NeuralRouterPolicy:
    """Policy configuring dimensions, pathway targets, and entropy limits."""

    input_dim: int
    hidden_dim: int
    pathways: tuple[Pathway, ...]
    entropy_threshold: float
    uncertainty_margin: float
    semantic_clusters: dict[Pathway, tuple[str, ...]]

    @classmethod
    def load(cls, directory: str | Path) -> NeuralRouterPolicy:
        path = Path(directory) / "neural_router_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.neural_router_policy.v1":
            raise ValueError(f"{path} has an unsupported neural-router format")

        in_dim = int(payload.get("input_dim", 512))
        hid_dim = int(payload.get("hidden_dim", 256))
        raw_pathways = payload.get("pathways", [])
        pathways = tuple(Pathway(p.lower()) for p in raw_pathways)

        entropy_thresh = float(payload.get("entropy_threshold", 0.35))
        uncert_margin = float(payload.get("uncertainty_margin", 0.15))

        raw_clusters = payload.get("semantic_clusters", {})
        clusters: dict[Pathway, tuple[str, ...]] = {}
        for k, words in raw_clusters.items():
            pw = Pathway(k.lower())
            clusters[pw] = tuple(str(w).lower() for w in words)

        return cls(
            input_dim=in_dim,
            hidden_dim=hid_dim,
            pathways=pathways,
            entropy_threshold=entropy_thresh,
            uncertainty_margin=uncert_margin,
            semantic_clusters=clusters,
        )

    @classmethod
    def default(cls) -> NeuralRouterPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
