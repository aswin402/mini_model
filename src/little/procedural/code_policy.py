"""Declarative policy configuration for Python AST code synthesis and sandbox execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.knowledge.registry import SchemaRegistry


@dataclass(frozen=True)
class SandboxPolicy:
    timeout_seconds: float = 3.0
    max_memory_mb: int = 64
    max_repair_attempts: int = 3
    allowed_builtin_names: tuple[str, ...] = (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip"
    )
    forbidden_ast_nodes: tuple[str, ...] = (
        "Import", "ImportFrom", "Delete", "Global", "Nonlocal"
    )
    forbidden_names: tuple[str, ...] = (
        "eval", "exec", "open", "input", "compile", "__import__", "globals", "locals"
    )


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    params: tuple[str, ...]
    docstring: str
    assertions: tuple[str, ...]


@dataclass(frozen=True)
class CodeSynthesizerPolicy:
    version: str = "1.0.0"
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)
    primitives: dict[str, PrimitiveSpec] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeSynthesizerPolicy:
        sb_data = data.get("sandbox", {})
        sandbox = SandboxPolicy(
            timeout_seconds=float(sb_data.get("timeout_seconds", 3.0)),
            max_memory_mb=int(sb_data.get("max_memory_mb", 64)),
            max_repair_attempts=int(sb_data.get("max_repair_attempts", 3)),
            allowed_builtin_names=tuple(sb_data.get("allowed_builtin_names", ())),
            forbidden_ast_nodes=tuple(sb_data.get("forbidden_ast_nodes", ())),
            forbidden_names=tuple(sb_data.get("forbidden_names", ())),
        )
        prims: dict[str, PrimitiveSpec] = {}
        for k, v in data.get("algorithmic_primitives", {}).items():
            prims[k] = PrimitiveSpec(
                name=v["name"],
                params=tuple(v.get("params", ())),
                docstring=v.get("docstring", ""),
                assertions=tuple(v.get("assertions", ())),
            )
        return cls(
            version=str(data.get("version", "1.0.0")),
            sandbox=sandbox,
            primitives=prims,
        )

    @classmethod
    def default(cls) -> CodeSynthesizerPolicy:
        try:
            registry = SchemaRegistry.default()
            payload = registry.load_schema_dict("code_synthesizer_policy")
            return cls.from_dict(payload)
        except Exception:
            path = Path(__file__).resolve().parents[3] / "data" / "schemas" / "code_synthesizer_policy.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            return cls()
