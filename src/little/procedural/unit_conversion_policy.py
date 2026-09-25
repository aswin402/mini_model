"""Versioned unit aliases and conversion edges shared by procedural math."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class UnitConversionEdge:
    source: str
    target: str
    scale: float
    offset: float = 0.0


@dataclass(frozen=True)
class UnitConversionPolicy:
    aliases: dict[str, str]
    edges: tuple[UnitConversionEdge, ...]

    @classmethod
    def load(cls, directory: Path) -> UnitConversionPolicy:
        path = Path(directory) / "unit_conversion_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        data = payload.get("unit_conversion_policy")
        if not isinstance(data, dict):
            raise TypeError(
                f"{path} must contain 'unit_conversion_policy' object"
            )
        aliases = data.get("aliases")
        raw_edges = data.get("edges")
        if not isinstance(aliases, dict) or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in aliases.items()
        ):
            raise TypeError(f"{path} aliases must map strings to strings")
        if not isinstance(raw_edges, list):
            raise TypeError(f"{path} edges must be a list")

        edges: list[UnitConversionEdge] = []
        for index, raw in enumerate(raw_edges):
            if not isinstance(raw, dict):
                raise TypeError(f"{path} edge {index} must be an object")
            source = raw.get("source")
            target = raw.get("target")
            scale = raw.get("scale")
            offset = raw.get("offset", 0.0)
            if not isinstance(source, str) or not source.strip():
                raise TypeError(f"{path} edge {index} source must be a string")
            if not isinstance(target, str) or not target.strip():
                raise TypeError(f"{path} edge {index} target must be a string")
            if (
                not isinstance(scale, (int, float))
                or isinstance(scale, bool)
                or not math.isfinite(float(scale))
                or float(scale) == 0.0
            ):
                raise TypeError(f"{path} edge {index} scale must be a non-zero number")
            if (
                not isinstance(offset, (int, float))
                or isinstance(offset, bool)
                or not math.isfinite(float(offset))
            ):
                raise TypeError(f"{path} edge {index} offset must be a number")
            edges.append(
                UnitConversionEdge(
                    source=source.strip().lower(),
                    target=target.strip().lower(),
                    scale=float(scale),
                    offset=float(offset),
                )
            )
        return cls(
            aliases={
                key.strip().lower(): value.strip().lower()
                for key, value in aliases.items()
            },
            edges=tuple(edges),
        )

    @classmethod
    def default(cls) -> UnitConversionPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
