"""Policy loader for the Neural Router configuration and semantic anchors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths
from little.neural.models import Pathway

_DEFAULT_GENERIC_WORDS: frozenset[str] = frozenset({
    "who", "what", "is", "are", "you", "your", "today", "can",
    "the", "a", "an", "or", "by", "in", "on", "of", "for", "to",
    "with", "and", "me", "my", "please",
})

_DEFAULT_CODE_INTENT_PHRASES: tuple[str, ...] = (
    "write code", "generate code", "python code", "write a function",
    "synthesize code", "implement", "code for", "code to", "python function",
    "program to", "script to",
)

_DEFAULT_CODE_INTENT_TOKENS: frozenset[str] = frozenset({
    "code", "function", "write", "generate", "synthesize", "implement", "script", "program",
})


@dataclass(frozen=True)
class NeuralRouterPolicy:
    """Policy configuring dimensions, pathway targets, and entropy limits."""

    input_dim: int
    hidden_dim: int  # Legacy metadata only; the router is a linear classifier.
    pathways: tuple[Pathway, ...]
    entropy_threshold: float
    uncertainty_margin: float
    semantic_clusters: dict[Pathway, tuple[str, ...]]
    generic_words: frozenset[str] = _DEFAULT_GENERIC_WORDS
    code_intent_phrases: tuple[str, ...] = _DEFAULT_CODE_INTENT_PHRASES
    code_intent_tokens: frozenset[str] = _DEFAULT_CODE_INTENT_TOKENS

    def is_code_intent(self, text: str, routed_pathway: str = "") -> bool:
        """Evaluate whether an input requests AST code synthesis via policy phrases or neural CODE routing."""
        low = text.lower()
        if any(phrase in low for phrase in self.code_intent_phrases):
            return True
        return routed_pathway == "code" and any(tok in low for tok in self.code_intent_tokens)

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

        raw_generic = payload.get("generic_words", list(_DEFAULT_GENERIC_WORDS))
        generic_words = frozenset(str(w).lower().strip() for w in raw_generic if str(w).strip())
        raw_code_phrases = payload.get("code_intent_phrases", list(_DEFAULT_CODE_INTENT_PHRASES))
        code_phrases = tuple(str(p).lower().strip() for p in raw_code_phrases if str(p).strip())
        raw_code_tokens = payload.get("code_intent_tokens", list(_DEFAULT_CODE_INTENT_TOKENS))
        code_tokens = frozenset(str(t).lower().strip() for t in raw_code_tokens if str(t).strip())

        return cls(
            input_dim=in_dim,
            hidden_dim=hid_dim,
            pathways=pathways,
            entropy_threshold=entropy_thresh,
            uncertainty_margin=uncert_margin,
            semantic_clusters=clusters,
            generic_words=generic_words,
            code_intent_phrases=code_phrases,
            code_intent_tokens=code_tokens,
        )

    @classmethod
    def default(cls) -> NeuralRouterPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
