"""Learned Neural Router evaluating calibrated routing probabilities across cognitive pathways."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from little.neural.encoder import GENERIC_WORDS, NeuralStateEncoder
from little.neural.models import NeuralState, Pathway, RoutingDecision
from little.neural.router_policy import NeuralRouterPolicy


class NeuralRouter:
    """Linear softmax classifier over deterministic hashed encoder features; feedback is explicit."""

    def __init__(
        self,
        policy: NeuralRouterPolicy | None = None,
        encoder: NeuralStateEncoder | None = None,
        temperature: float = 8.0,
    ) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (float, int))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Temperature must be a finite positive scale")
        self.policy = policy or NeuralRouterPolicy.default()
        self.encoder = encoder or NeuralStateEncoder(policy=self.policy)
        if self.encoder.dim != self.policy.input_dim:
            raise ValueError("Encoder dimension must match router policy")
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

        if len(state.embedding) != self.policy.input_dim or any(
            not math.isfinite(value) for value in state.embedding
        ):
            raise ValueError(
                "State embedding must match dimensions and contain finite values"
            )
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
        """Explicit supervised feedback using a softmax cross-entropy gradient."""
        if target_pathway not in self.pathways:
            raise ValueError("Unknown target pathway")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (float, int))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Learning rate must be finite and positive")

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

    def checkpoint(self) -> dict[str, Any]:
        """Return JSON serializable parameters and encoder policy, without writing state."""
        return {
            "format": "little.neural_router_checkpoint.v1",
            "temperature": self.temperature,
            "policy": {
                "input_dim": self.policy.input_dim,
                "pathways": [pathway.value for pathway in self.pathways],
                "entropy_threshold": self.policy.entropy_threshold,
                "uncertainty_margin": self.policy.uncertainty_margin,
                "semantic_clusters": {
                    pathway.value: list(words)
                    for pathway, words in self.policy.semantic_clusters.items()
                },
            },
            "encoder_semantic_clusters": {
                pathway.value: list(words)
                for pathway, words in self.encoder.policy.semantic_clusters.items()
            },
            "weights": [row[:] for row in self._w],
            "biases": self._b[:],
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> NeuralRouter:
        """Restore validated linear parameters; never learn or open runtime storage."""
        try:
            if (
                not isinstance(payload, dict)
                or payload.get("format") != "little.neural_router_checkpoint.v1"
            ):
                raise ValueError("Unsupported router checkpoint format")
            data = payload["policy"]
            policy = NeuralRouterPolicy(
                input_dim=data["input_dim"],
                hidden_dim=0,
                pathways=tuple(Pathway(value) for value in data["pathways"]),
                entropy_threshold=data["entropy_threshold"],
                uncertainty_margin=data["uncertainty_margin"],
                semantic_clusters={
                    Pathway(key): tuple(words)
                    for key, words in data["semantic_clusters"].items()
                },
            )
            weights, biases = payload["weights"], payload["biases"]
            if (
                not isinstance(weights, list)
                or len(weights) != len(policy.pathways)
                or not isinstance(biases, list)
                or len(biases) != len(policy.pathways)
            ):
                raise ValueError(
                    "Checkpoint parameter dimensions do not match pathways"
                )
            if any(
                not isinstance(row, list) or len(row) != policy.input_dim
                for row in weights
            ):
                raise ValueError("Checkpoint weight dimensions do not match encoder")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                for row in weights + [biases]
                for value in row
            ):
                raise ValueError("Checkpoint parameters must be finite numbers")
            encoder_clusters = payload.get(
                "encoder_semantic_clusters", data["semantic_clusters"]
            )
            encoder_policy = replace(
                policy,
                semantic_clusters={
                    Pathway(key): tuple(words)
                    for key, words in encoder_clusters.items()
                },
            )
            router = cls(
                policy=policy,
                encoder=NeuralStateEncoder(policy=encoder_policy),
                temperature=payload["temperature"],
            )
            router._w, router._b = [row[:] for row in weights], biases[:]
            return router
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed router checkpoint") from exc

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
