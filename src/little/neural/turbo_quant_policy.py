"""Policy loader for TurboQuant vector quantization and indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class TurboQuantPolicy:
    """Configures deterministic orthogonal rotation, bit-depth, and coarse-to-fine search."""

    dimension: int
    rotation_seed: int
    quantization_modes: tuple[str, ...]
    bins_4bit: int
    top_k_default: int
    coarse_filter_multiplier: int
    similarity_threshold: float

    @classmethod
    def load(cls, directory: str | Path) -> TurboQuantPolicy:
        path = Path(directory) / "turbo_quant_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.turbo_quant.v1":
            raise ValueError(f"{path} has an unsupported turbo-quant format")

        dim = int(payload.get("dimension", 512))
        seed = int(payload.get("rotation_seed", 42))
        modes = tuple(str(m).lower() for m in payload.get("quantization_modes", ["4bit", "1bit"]))
        bins = int(payload.get("bins_4bit", 16))
        top_k = int(payload.get("top_k_default", 5))
        mult = int(payload.get("coarse_filter_multiplier", 4))
        thresh = float(payload.get("similarity_threshold", 0.55))

        return cls(
            dimension=dim,
            rotation_seed=seed,
            quantization_modes=modes,
            bins_4bit=bins,
            top_k_default=top_k,
            coarse_filter_multiplier=mult,
            similarity_threshold=thresh,
        )

    @classmethod
    def default(cls) -> TurboQuantPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
