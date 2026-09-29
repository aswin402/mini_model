"""Neural state encoder transforming raw text and context into continuous S in R^512."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from little.neural.models import NeuralState
from little.neural.router_policy import NeuralRouterPolicy

# Stop words and broad interrogatives with lower domain specificity
GENERIC_WORDS: frozenset[str] = frozenset({
    "who", "what", "is", "are", "you", "your", "today", "can",
    "the", "a", "an", "or", "by", "in", "on", "of", "for", "to",
    "with", "and", "me", "my", "please"
})


class NeuralStateEncoder:
    """Transforms arbitrary text and working memory into a continuous normalized latent vector."""

    def __init__(self, policy: NeuralRouterPolicy | None = None) -> None:
        self.policy = policy or NeuralRouterPolicy.default()
        self.dim = self.policy.input_dim

    def encode(
        self,
        text: str,
        *,
        context_text: str | None = None,
        working_memory: dict[str, Any] | None = None,
    ) -> NeuralState:
        """Project text, dialogue context, and working memory into S in R^512."""
        tokens = self._tokenize(text)
        vec = [0.0] * self.dim

        # 1. Project tokens and character n-grams into continuous vector space
        for pos, tok in enumerate(tokens):
            w = 1.5 if tok not in GENERIC_WORDS else 0.5
            decayed_weight = w / (1.0 + 0.05 * pos)
            self._project_string(tok, decayed_weight, vec)

            # Character subword n-grams (lengths 3, 4) for typo tolerance on substantive tokens
            if len(tok) >= 4 and tok not in GENERIC_WORDS:
                for n in (3, 4):
                    for i in range(len(tok) - n + 1):
                        ngram = tok[i : i + n]
                        self._project_string(ngram, 0.15, vec)

            # Number and math operator detection
            if tok.isdigit() or any(c in "+-*/^=<>" for c in tok):
                self._project_string("type:numeric", 2.0, vec)

        # 2. Project semantic cluster alignments with domain specificity
        tok_set = set(tokens)
        for pathway, cluster_words in self.policy.semantic_clusters.items():
            overlap = tok_set.intersection(cluster_words)
            if overlap:
                score = sum(0.5 if w in GENERIC_WORDS else 2.5 for w in overlap)
                self._project_string(f"pathway:{pathway.value}", 3.0 * score, vec)
                for word in overlap:
                    w_w = 0.5 if word in GENERIC_WORDS else 2.0
                    self._project_string(f"cluster_word:{pathway.value}", w_w, vec)

        # 3. Fuse conversational dialogue context if present
        if context_text:
            ctx_tokens = self._tokenize(context_text)
            for pos, tok in enumerate(ctx_tokens):
                self._project_string(tok, 0.25 / (1.0 + 0.05 * pos), vec)

        # 4. Fuse working memory if present
        if working_memory:
            active_concepts = working_memory.get("active_concepts", [])
            for c in active_concepts:
                self._project_string(f"concept:{c}", 1.0, vec)

        # 5. Normalize to unit sphere (L2 norm)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            normalized = tuple(x / norm for x in vec)
        else:
            normalized = tuple(1.0 / math.sqrt(self.dim) for _ in range(self.dim))

        return NeuralState(
            embedding=normalized,
            text=text,
            tokens=tuple(tokens),
        )

    def _tokenize(self, text: str) -> list[str]:
        """Extract alphanumeric words and math symbols as tokens."""
        clean = text.lower().strip()
        return [t for t in re.findall(r"[a-z0-9_]+|[\+\-\*\/\^=><]+", clean) if t]

    def _project_string(self, s: str, weight: float, vec: list[float]) -> None:
        """Project string into pseudo-orthogonal dimensions using deterministic double hashing."""
        h1 = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
        h2 = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

        idx1 = h1 % self.dim
        idx2 = (h1 + h2) % self.dim
        idx3 = (h1 ^ h2) % self.dim

        # Distributed sign pattern
        sign1 = 1.0 if (h1 & 1) == 0 else -1.0
        sign2 = 1.0 if (h2 & 1) == 0 else -1.0
        sign3 = 1.0 if ((h1 + h2) & 1) == 0 else -1.0

        vec[idx1] += weight * sign1
        vec[idx2] += (weight * 0.7) * sign2
        vec[idx3] += (weight * 0.5) * sign3
