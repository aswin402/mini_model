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
    max_output_bytes: int = 65536
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


_DEFAULT_ALLOWED_FRAMING_WORDS: frozenset[str] = frozenset({
    "write", "create", "generate", "synthesize", "implement", "build", "make", "give", "me",
    "please", "can", "could", "you", "a", "an", "the", "python", "code", "function", "program",
    "algorithm", "to", "for", "of", "in", "that", "which", "check", "determine", "test", "if",
    "whether", "is", "computes", "compute", "calculates", "calculate", "returns", "return",
    "number", "numbers", "integer", "integers", "non", "negative", "string", "strings", "list",
    "lists", "array", "arrays", "nested", "two", "dimensional", "2d", "element", "elements",
    "n", "s", "arr", "target",
})


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    params: tuple[str, ...]
    docstring: str
    assertions: tuple[str, ...]
    trigger_tokens: tuple[str, ...] = ()
    trigger_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodeSynthesizerPolicy:
    version: str = "1.0.0"
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)
    primitives: dict[str, PrimitiveSpec] = field(default_factory=dict)
    allowed_framing_words: frozenset[str] = _DEFAULT_ALLOWED_FRAMING_WORDS

    @property
    def allowed_request_vocabulary(self) -> frozenset[str]:
        """Dynamically compute allowed request words from framing words and registered primitive specs."""
        vocab = set(self.allowed_framing_words)
        for key, spec in self.primitives.items():
            vocab.add(key.lower())
            vocab.add(spec.name.lower())
            for tok in spec.trigger_tokens:
                vocab.add(tok.lower())
            for phrase in spec.trigger_phrases:
                for w in phrase.lower().split():
                    if w:
                        vocab.add(w)
        return frozenset(vocab)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeSynthesizerPolicy:
        sb_data = data.get("sandbox", {})
        sandbox = SandboxPolicy(
            timeout_seconds=float(sb_data.get("timeout_seconds", 3.0)),
            max_memory_mb=int(sb_data.get("max_memory_mb", 64)),
            max_output_bytes=int(sb_data.get("max_output_bytes", 65536)),
            max_repair_attempts=int(sb_data.get("max_repair_attempts", 3)),
            allowed_builtin_names=tuple(sb_data.get("allowed_builtin_names", ())),
            forbidden_ast_nodes=tuple(sb_data.get("forbidden_ast_nodes", ())),
            forbidden_names=tuple(sb_data.get("forbidden_names", ())),
        )
        prims: dict[str, PrimitiveSpec] = {}
        for k, v in data.get("algorithmic_primitives", {}).items():
            raw_triggers = v.get("trigger_tokens", [v["name"]])
            raw_phrases = v.get("trigger_phrases", ())
            prims[k] = PrimitiveSpec(
                name=v["name"],
                params=tuple(v.get("params", ())),
                docstring=v.get("docstring", ""),
                assertions=tuple(v.get("assertions", ())),
                trigger_tokens=tuple(str(t).lower().strip() for t in raw_triggers if str(t).strip()),
                trigger_phrases=tuple(str(p).lower().strip() for p in raw_phrases if str(p).strip()),
            )
        framing = frozenset(
            str(w).lower().strip()
            for w in data.get("allowed_framing_words", list(_DEFAULT_ALLOWED_FRAMING_WORDS))
            if str(w).strip()
        )
        return cls(
            version=str(data.get("version", "1.0.0")),
            sandbox=sandbox,
            primitives=prims,
            allowed_framing_words=framing,
        )

    @classmethod
    def default(cls) -> CodeSynthesizerPolicy:
        try:
            registry = SchemaRegistry.default()
            payload = registry.load_schema_dict("code_synthesizer_policy")
            return cls.from_dict(payload)
        except (AttributeError, KeyError, ValueError, OSError):
            path = Path(__file__).resolve().parents[3] / "data" / "schemas" / "code_synthesizer_policy.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            return cls()
