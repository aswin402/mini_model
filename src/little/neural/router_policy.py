"""Policy loader for the Neural Router configuration and semantic anchors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths
from little.neural.models import Pathway


@dataclass(frozen=True)
class NeuralRouterPolicy:
    """Policy configuring dimensions, pathway targets, and entropy limits."""

    input_dim: int
    hidden_dim: int  # Legacy metadata only; the router is a linear classifier.
    pathways: tuple[Pathway, ...]
    entropy_threshold: float
    uncertainty_margin: float
    semantic_clusters: dict[Pathway, tuple[str, ...]]

    def __post_init__(self) -> None:
        if type(self.input_dim) is not int or self.input_dim != 512:
            raise ValueError("The hashed encoder and NeuralState require input_dim=512")
        if not self.pathways or len(set(self.pathways)) != len(self.pathways):
            raise ValueError("Pathways must be nonempty and unique")
        if any(not isinstance(pathway, Pathway) for pathway in self.pathways):
            raise TypeError("Pathways must be Pathway values")
        if not isinstance(self.semantic_clusters, dict) or any(
            not isinstance(key, Pathway)
            or not isinstance(words, tuple)
            or any(not isinstance(word, str) or not word.strip() for word in words)
            for key, words in self.semantic_clusters.items()
        ):
            raise ValueError(
                "Semantic clusters must map pathways to tuples of nonempty strings"
            )
        if (
            isinstance(self.entropy_threshold, bool)
            or not isinstance(self.entropy_threshold, (float, int))
            or not math.isfinite(self.entropy_threshold)
            or not 0 <= self.entropy_threshold <= math.log(len(self.pathways))
        ):
            raise ValueError(
                "Entropy threshold must be in the categorical entropy range"
            )
        if (
            isinstance(self.uncertainty_margin, bool)
            or not isinstance(self.uncertainty_margin, (float, int))
            or not math.isfinite(self.uncertainty_margin)
            or not 0 <= self.uncertainty_margin <= 1
        ):
            raise ValueError("Uncertainty margin must be between zero and one")

    @classmethod
    def load(cls, directory: str | Path) -> NeuralRouterPolicy:
        path = Path(directory) / "neural_router_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.neural_router_policy.v1":
            raise ValueError(f"{path} has an unsupported neural-router format")

        in_dim = payload.get("input_dim", 512)
        hid_dim = int(payload.get("hidden_dim", 256))
        raw_pathways = payload.get("pathways", [])
        pathways = tuple(Pathway(p.lower()) for p in raw_pathways)

        entropy_thresh = payload.get("entropy_threshold", 0.35)
        uncert_margin = payload.get("uncertainty_margin", 0.15)

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
