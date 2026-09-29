"""Learned Neural Router evaluating calibrated routing probabilities across cognitive pathways."""

from __future__ import annotations

import math
from typing import Any

from little.neural.encoder import GENERIC_WORDS, NeuralStateEncoder
from little.neural.models import NeuralState, Pathway, RoutingDecision
from little.neural.router_policy import NeuralRouterPolicy


class NeuralRouter:
    """Evaluates continuous latent states S in R^512 and dispatches to specialized cognitive pathways."""

    def __init__(
        self,
        policy: NeuralRouterPolicy | None = None,
        encoder: NeuralStateEncoder | None = None,
        temperature: float = 8.0,
    ) -> None:
        self.policy = policy or NeuralRouterPolicy.default()
        self.encoder = encoder or NeuralStateEncoder(policy=self.policy)
        self.pathways = self.policy.pathways
        self.temperature = temperature

        # Build prototype anchor weights for each pathway from semantic clusters
        self._w, self._b = self._initialize_weights()

    def route(
        self,
        text_or_state: str | NeuralState,
        *,
        context_text: str | None = None,
        working_memory: dict[str, Any] | None = None,
    ) -> RoutingDecision:
        """Route input into a calibrated decision across cognitive deliberation pathways."""
        if isinstance(text_or_state, str):
            state = self.encoder.encode(
                text_or_state,
                context_text=context_text,
                working_memory=working_memory,
            )
        else:
            state = text_or_state

        S = state.embedding
        in_dim = len(S)

        # Prototypical Linear Projection: z_k = sum_i (W_{k,i} * S_i) + b_k
        logits = [0.0] * len(self.pathways)
        for k in range(len(self.pathways)):
            dot = sum(S[i] * self._w[k][i] for i in range(in_dim))
            logits[k] = dot + self._b[k]

        # Numerically stable Softmax: p_k = exp(z_k - max_z) / sum_j exp(z_j - max_z)
        max_logit = max(logits)
        exps = [math.exp(z - max_logit) for z in logits]
        sum_exp = sum(exps)
        probs = [e / sum_exp for e in exps]

        # Map to pathway dict
        prob_dict: dict[Pathway, float] = {
            pw: round(probs[idx], 4) for idx, pw in enumerate(self.pathways)
        }

        # Shannon entropy: H = -sum(p * ln(p))
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)

        # Ranked pathways
        ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_pw, top_conf = ranked[0]
        second_conf = ranked[1][1] if len(ranked) > 1 else 0.0

        # Epistemic uncertainty check: high entropy or narrow margin between top-2
        margin = top_conf - second_conf
        is_uncertain = (entropy >= self.policy.entropy_threshold) or (
            margin < self.policy.uncertainty_margin
        )

        return RoutingDecision(
            pathway=top_pw,
            confidence=top_conf,
            probabilities=prob_dict,
            entropy=round(entropy, 4),
            is_uncertain=is_uncertain,
            latent_state=state,
        )

    def learn(
        self,
        state: NeuralState,
        target_pathway: Pathway,
        learning_rate: float = 0.05,
    ) -> None:
        """Online parameter adaptation for lifelong learning without catastrophic forgetting."""
        if target_pathway not in self.pathways:
            return

        target_idx = self.pathways.index(target_pathway)
        decision = self.route(state)
        current_probs = [decision.probabilities.get(pw, 0.0) for pw in self.pathways]

        # Gradient step on softmax cross-entropy
        for k, pw in enumerate(self.pathways):
            y_k = 1.0 if k == target_idx else 0.0
            error = y_k - current_probs[k]
            grad_scale = learning_rate * error
            for i in range(len(state.embedding)):
                self._w[k][i] += grad_scale * state.embedding[i]
            self._b[k] += grad_scale

    def _initialize_weights(self) -> tuple[list[list[float]], list[float]]:
        """Initialize continuous projection weights aligned with semantic cluster centroids."""
        in_dim = self.policy.input_dim
        num_classes = len(self.pathways)

        w: list[list[float]] = []
        b: list[float] = [0.0] * num_classes

        for pw in self.pathways:
            words = list(self.policy.semantic_clusters.get(pw, ()))
            c_vec = [0.0] * in_dim

            for w_tok in words:
                w_w = 0.5 if w_tok in GENERIC_WORDS else 2.0
                self.encoder._project_string(w_tok, w_w, c_vec)

            # Strong pathway identity attractor
            self.encoder._project_string(f"pathway:{pw.value}", 8.0, c_vec)
            self.encoder._project_string(f"cluster_word:{pw.value}", 4.0, c_vec)

            # Special inductive biases
            if pw == Pathway.MATH:
                self.encoder._project_string("type:numeric", 2.5, c_vec)

            # L2 normalize centroid
            norm = math.sqrt(sum(x * x for x in c_vec))
            if norm > 1e-12:
                normalized_c = [x / norm for x in c_vec]
            else:
                normalized_c = [0.0] * in_dim

            # Scale row by temperature
            row = [self.temperature * val for val in normalized_c]
            w.append(row)

        return w, b
