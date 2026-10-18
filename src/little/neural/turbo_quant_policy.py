"""Policy loader for TurboQuant vector quantization and indexing."""

from __future__ import annotations

import json
import math
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

    def __post_init__(self) -> None:
        # The shared Python/native encoding is fixed at eight 64-bit words.
        if type(self.dimension) is not int or self.dimension != 512:
            raise ValueError("TurboQuant supports dimension 512 only")
        if self.bins_4bit != 16 or len(self.quantization_modes) != 2 or set(self.quantization_modes) != {'1bit', '4bit'}:
            raise ValueError("TurboQuant requires 1bit and 4bit modes with 16 bins")
        if type(self.top_k_default) is not int or type(self.coarse_filter_multiplier) is not int or self.top_k_default < 1 or self.coarse_filter_multiplier < 1:
            raise ValueError("TurboQuant search limits must be positive")
        if not math.isfinite(self.similarity_threshold) or not -1 <= self.similarity_threshold <= 1:
            raise ValueError("TurboQuant similarity threshold must be finite and between -1 and 1")

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
