"""Mini Learning Unit (MLU) — 360-Degree Liquid-Gated Neural-Symbolic Cell.

Combines inspirations from:
- Liquid AI LFM-2.5 (Gated Short-Convolution + Continuous-Time Liquid CfC state update)
- JEPA / Joint Epistemic Workspace (predictive alignment in latent concept space)
- Laya Epistemic Gatekeeper (normalized Shannon entropy + NARS truth-value calibration)
- mini-AGI (Trunk 0.1x vs. Leaf 1.0x online plasticity scaling)
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

SPOKE_NAMES: tuple[str, ...] = (
    "taxonomy",
    "mereology",
    "attributes",
    "dynamics",
    "procedural",
    "episodic",
)

def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(x, 30.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -30.0))
    return z / (1.0 + z)


def _silu(x: float) -> float:
    return x * _sigmoid(x)


def _normalize_vec(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return list(vec)
    return [v / norm for v in vec]


def deterministic_seed_vector(name: str, dim: int = 64) -> list[float]:
    """Generate a deterministic unit-norm seed vector from a concept or token name."""
    clean = name.strip().upper().replace(" ", "_")
    vec = [0.0] * dim
    digest = hashlib.sha256(clean.encode("utf-8")).digest()
    for i in range(dim):
        byte_val = digest[i % len(digest)]
        angle = ((byte_val / 255.0) * 2.0 - 1.0) * math.pi
        vec[i] = math.sin(angle * (1 + (i // len(digest))))
    return _normalize_vec(vec)


def harmonic_spoke_basis(spoke_idx: int, dim: int = 64) -> list[float]:
    """Generate the k-th orthogonal 360° angular phase basis vector in R^dim."""
    num_spokes = len(SPOKE_NAMES)
    theta = (2.0 * math.pi * float(spoke_idx % num_spokes)) / float(num_spokes)
    vec = [0.0] * dim
    for d in range(dim):
        freq = float(d + 1)
        vec[d] = math.cos(freq * theta + 0.5 * math.sin(freq * theta))
    return _normalize_vec(vec)


def route_phase_spoke_idx(
    rel_vec: list[float],
    target_vec: list[float] | None = None,
) -> int:
    """Route a relation/transition vector to a 360° spoke index [0..5] via harmonic cosine resonance."""
    dim = len(rel_vec)
    if target_vec is not None and len(target_vec) == dim:
        combined = _normalize_vec(
            [
                0.75 * r + 0.25 * t + 0.15 * (r * t)
                for r, t in zip(rel_vec, target_vec, strict=False)
            ]
        )
    else:
        combined = _normalize_vec(rel_vec)

    best_idx = 0
    best_sim = -1e9
    for idx in range(len(SPOKE_NAMES)):
        basis = harmonic_spoke_basis(idx, dim)
        sim = sum(c * b for c, b in zip(combined, basis, strict=False))
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    return best_idx


def map_predicate_to_spoke(
    predicate: str,
    target_vec: list[float] | None = None,
    dim: int = 64,
) -> str:
    """Project a relation descriptor into 64-D phase space and select its resonant 360° spoke."""
    rel_vec = deterministic_seed_vector(predicate, dim)
    idx = route_phase_spoke_idx(rel_vec, target_vec)
    return SPOKE_NAMES[idx]


@dataclass(frozen=True)
class MLUSpokePrediction:
    """A 360° spoke prediction emitted by a MiniLearningUnit."""

    source_concept: str
    spoke: str
    predicate: str
    target_concept: str
    neural_score: float
    nars_confidence: float
    epistemic_entropy: float
    combined_confidence: float
    verified: bool


@dataclass
class SpokeHead:
    """Single radial spoke head inside a 360° MiniLearningUnit."""

    spoke_name: str
    dim: int = 64
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    # Maps target_concept -> [positive_evidence, negative_evidence, learned_affinity, default_predicate]
    target_affinities: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = deterministic_seed_vector(f"SPOKE::{self.spoke_name}", self.dim)

    def project(self, state_vec: list[float]) -> list[float]:
        """Project a neural state into this radial spoke's latent target query space."""
        projected = [
            math.tanh(s * w + self.bias * 0.1)
            for s, w in zip(state_vec, self.weights, strict=False)
        ]
        return _normalize_vec(projected)

    def score_target(self, state_vec: list[float], target_name: str, target_vec: list[float]) -> float:
        """Compute bilinear spoke compatibility score in [0, 1] between state_vec and target_vec."""
        proj = self.project(state_vec)
        cosine = sum(p * t for p, t in zip(proj, target_vec, strict=False))
        rec = self.target_affinities.get(target_name)
        affinity_bonus = float(rec[2]) if rec is not None else 0.0
        raw = 1.8 * cosine + affinity_bonus + self.bias
        return _sigmoid(raw)


class MiniLearningUnit:
    """A 360° Liquid-Gated Neural-Symbolic Mini Learning Unit (MLU).

    Each MLU encapsulates:
    1. A trainable anchor vector in the Joint Epistemic Workspace (`dim=64`).
    2. An LFM-2.5 style Gated Short-Convolution + Continuous-Time Liquid (`CfC`) state cell.
    3. Six 360° radial `SpokeHead` projections (`taxonomy`, `mereology`, `attributes`, `dynamics`,
       `procedural`, `episodic`).
    4. Online predictive-coding / contrastive weight updates governed by trunk vs. leaf plasticity.
    5. Surface-token bigram/trigram transition memory for fluent grounded generation.
    """

    def __init__(
        self,
        concept_id: str,
        role: str = "leaf",
        parent_hub: str | None = None,
        dim: int = 64,
    ) -> None:
        self.concept_id = concept_id.strip().upper().replace(" ", "_")
        self.role = role  # "core" | "subcore" | "leaf"
        self.parent_hub = parent_hub
        self.dim = dim

        # 1. Trainable JEPA concept anchor & liquid hidden state
        self.anchor_vec: list[float] = deterministic_seed_vector(self.concept_id, dim)
        self.hidden_state: list[float] = list(self.anchor_vec)

        # 2. LFM-2.5 Gated Short-Conv (k=3) + Liquid CfC time-constant parameters
        self.conv_k1: list[float] = [0.60] * dim
        self.conv_k2: list[float] = [0.25] * dim
        self.conv_k3: list[float] = [0.15] * dim
        self.gate_weights: list[float] = deterministic_seed_vector(f"GATE::{self.concept_id}", dim)
        self.tau_weights: list[float] = [0.15] * dim

        # 3. 360° 6-Spoke radial heads
        self.spokes: dict[str, SpokeHead] = {
            spoke: SpokeHead(spoke_name=spoke, dim=dim) for spoke in SPOKE_NAMES
        }

        # 4. Grounded surface-token transition table (Multi-Token Prediction / MTP memory)
        # Maps context_token -> {next_token: count}
        self.token_transitions: dict[str, dict[str, float]] = {}
        self.context_Phrases: list[str] = []

        # 5. Epistemic & lifecycle telemetry (for 5-Brake Growth & Laya Confidence)
        self.activation_count: int = 0
        self.verified_hits: float = 0.0
        self.refuted_hits: float = 0.0
        self.cumulative_surprise: float = 0.0
        self.last_surprise: float = 0.0
        self.created_cycle: int = 0
        self.last_active_cycle: int = 0

    @property
    def is_trunk(self) -> bool:
        """Core and Sub-Core hubs act as trunk nodes (0.1x damped plasticity)."""
        return self.role in ("core", "subcore")

    @property
    def nars_confidence(self) -> float:
        """Compute NARS truth-value confidence c = (w+ + w-) / (w+ + w- + k) with horizon k=1."""
        total = self.verified_hits + self.refuted_hits
        if total <= 0.0:
            return 0.0
        freq = self.verified_hits / total
        conf = total / (total + 1.0)
        return round(freq * conf, 4)

    def forward_liquid_step(
        self,
        context_vectors: list[list[float]],
        dt: float = 1.0,
    ) -> list[float]:
        """Run one LFM-2.5 Gated Short-Conv + Liquid CfC state update step."""
        self.activation_count += 1
        if not context_vectors:
            return list(self.hidden_state)

        v1 = context_vectors[-1]
        v2 = context_vectors[-2] if len(context_vectors) >= 2 else v1
        v3 = context_vectors[-3] if len(context_vectors) >= 3 else v2

        new_hidden: list[float] = [0.0] * self.dim
        for i in range(self.dim):
            # Gated short convolution over local 3-step window
            conv_val = (
                self.conv_k1[i] * v1[i]
                + self.conv_k2[i] * v2[i]
                + self.conv_k3[i] * v3[i]
            )
            gate = _sigmoid(self.gate_weights[i] * v1[i] + self.anchor_vec[i])
            gated_conv = _silu(conv_val) * gate

            # Closed-Form Continuous-Time (CfC) liquid relaxation toward anchor + context
            tau_gate = _sigmoid((self.tau_weights[i] + abs(gated_conv)) * dt)
            target_state = math.tanh(0.65 * self.anchor_vec[i] + 0.35 * gated_conv)
            new_hidden[i] = (1.0 - tau_gate) * self.hidden_state[i] + tau_gate * target_state

        self.hidden_state = _normalize_vec(new_hidden)
        return list(self.hidden_state)

    def learn_spoke_relation(
        self,
        predicate: str,
        target_concept: str,
        target_vec: list[float],
        learning_rate: float = 0.15,
        verified: bool = True,
    ) -> float:
        """Online predictive-coding & contrastive update for a 360° spoke connection.

        Returns the prediction surprise (loss) in [0.0, 1.0] prior to the weight update.
        """
        spoke_name = map_predicate_to_spoke(predicate, target_vec, self.dim)
        spoke = self.spokes[spoke_name]
        clean_target = target_concept.strip().upper().replace(" ", "_")

        pred_prob = spoke.score_target(self.hidden_state, clean_target, target_vec)
        target_label = 1.0 if verified else 0.0
        surprise = abs(target_label - pred_prob)
        self.last_surprise = surprise
        self.cumulative_surprise += surprise

        eff_lr = max(0.005, min(0.5, learning_rate * (0.1 if self.is_trunk else 1.0)))
        direction = 1.0 if verified else -1.0

        # Update spoke weights & anchor vector via JEPA latent alignment / repulsion
        for i in range(self.dim):
            grad = direction * surprise * self.hidden_state[i] * target_vec[i]
            spoke.weights[i] += eff_lr * grad
            self.anchor_vec[i] += 0.35 * eff_lr * direction * surprise * target_vec[i]

        spoke.weights = _normalize_vec(spoke.weights)
        self.anchor_vec = _normalize_vec(self.anchor_vec)
        self.hidden_state = _normalize_vec(
            [0.8 * h + 0.2 * a for h, a in zip(self.hidden_state, self.anchor_vec, strict=False)]
        )

        # Update per-target evidence and affinity
        rec = spoke.target_affinities.get(clean_target)
        if rec is None:
            rec = [0.0, 0.0, 0.0, predicate.strip().lower()]
            spoke.target_affinities[clean_target] = rec
        if verified:
            rec[0] = float(rec[0]) + 1.0
            rec[2] = min(3.5, float(rec[2]) + eff_lr * (0.5 + surprise))
            rec[3] = predicate.strip().lower()
            self.verified_hits += 1.0
        else:
            rec[1] = float(rec[1]) + 1.0
            rec[2] = max(-3.5, float(rec[2]) - eff_lr * (0.8 + surprise))
            self.refuted_hits += 1.0

        return round(surprise, 4)

    def learn_token_sequence(self, tokens: list[str], learning_rate: float = 0.2) -> float:
        """Online Multi-Token Prediction (MTP) update from a raw token stream."""
        clean_tokens = [t.strip().lower() for t in tokens if t.strip()]
        if len(clean_tokens) < 2:
            return 0.0

        total_surprise = 0.0
        pairs = 0
        for idx in range(len(clean_tokens) - 1):
            curr_tok = clean_tokens[idx]
            next_tok = clean_tokens[idx + 1]
            row = self.token_transitions.setdefault(curr_tok, {})
            row_sum = sum(row.values())
            prob = (row.get(next_tok, 0.0) / row_sum) if row_sum > 0 else 0.0
            surprise = 1.0 - prob
            total_surprise += surprise
            pairs += 1
            row[next_tok] = row.get(next_tok, 0.0) + max(0.05, learning_rate)

            # Also store 2-gram prefix -> next_tok for sharper phrase fluency
            if idx >= 1:
                bigram_key = f"{clean_tokens[idx - 1]} {curr_tok}"
                b_row = self.token_transitions.setdefault(bigram_key, {})
                b_row[next_tok] = b_row.get(next_tok, 0.0) + max(0.05, learning_rate * 1.25)

        avg_surprise = total_surprise / max(1, pairs)
        self.last_surprise = avg_surprise
        self.cumulative_surprise += avg_surprise
        return round(avg_surprise, 4)

    def predict_spoke_targets(
        self,
        spoke_name: str,
        candidate_vectors: dict[str, list[float]],
        top_k: int = 5,
    ) -> list[MLUSpokePrediction]:
        """Predict top-k 360° spoke targets with Laya normalized entropy & NARS confidence."""
        req_idx = SPOKE_NAMES.index(spoke_name) if spoke_name in SPOKE_NAMES else 0
        primary_spoke = self.spokes.get(spoke_name)
        if primary_spoke is not None and primary_spoke.target_affinities:
            active_spokes = [(req_idx, spoke_name, primary_spoke)]
        else:
            active_spokes = [
                (idx, s_name, self.spokes[s_name])
                for idx, s_name in enumerate(SPOKE_NAMES)
                if self.spokes[s_name].target_affinities
            ]

        if not active_spokes:
            return []

        total_known_targets = sum(len(s.target_affinities) for _, _, s in active_spokes)
        filter_by_candidates = 0 < len(candidate_vectors) < total_known_targets

        raw_scores: list[tuple[str, float, float, str, bool, str]] = []
        num_spokes = float(len(SPOKE_NAMES))
        for s_idx, s_name, spoke in active_spokes:
            angle_diff = (2.0 * math.pi * float(s_idx - req_idx)) / num_spokes
            phase_coherence = 0.75 + 0.25 * math.cos(angle_diff)
            for target_name, rec in spoke.target_affinities.items():
                if filter_by_candidates and target_name not in candidate_vectors:
                    continue
                pos_ev = float(rec[0])
                neg_ev = float(rec[1])
                pred_label = str(rec[3])
                if neg_ev > pos_ev:
                    continue
                t_vec = candidate_vectors.get(
                    target_name, deterministic_seed_vector(target_name, self.dim)
                )
                neural_score = (
                    spoke.score_target(self.hidden_state, target_name, t_vec) * phase_coherence
                )
                total_ev = pos_ev + neg_ev
                nars_conf = (pos_ev / max(1.0, total_ev)) * (total_ev / (total_ev + 0.5))
                raw_scores.append(
                    (target_name, neural_score, nars_conf, pred_label, pos_ev > 0, s_name)
                )

        if not raw_scores:
            return []

        # Compute Laya normalized Shannon entropy H(P) / ln(K) over the candidate distribution
        score_sum = sum(max(1e-6, item[1]) for item in raw_scores)
        probs = [max(1e-6, item[1]) / score_sum for item in raw_scores]
        if len(probs) <= 1:
            norm_entropy = 0.0
        else:
            shannon = -sum(p * math.log(p) for p in probs)
            norm_entropy = min(1.0, max(0.0, shannon / math.log(len(probs))))

        results: list[MLUSpokePrediction] = []
        for target_name, neural_score, nars_conf, pred_label, is_ver, s_name in raw_scores:
            combined = round(0.65 * nars_conf + 0.35 * neural_score, 4)
            results.append(
                MLUSpokePrediction(
                    source_concept=self.concept_id,
                    spoke=s_name,
                    predicate=pred_label,
                    target_concept=target_name,
                    neural_score=round(neural_score, 4),
                    nars_confidence=round(nars_conf, 4),
                    epistemic_entropy=round(norm_entropy, 4),
                    combined_confidence=combined,
                    verified=is_ver,
                )
            )
        results.sort(key=lambda r: r.combined_confidence, reverse=True)
        return results[:top_k]

    def generate_token_continuation(self, seed_tokens: list[str], max_tokens: int = 12) -> list[str]:
        """Generate a continuation sequence from this MLU's learned Multi-Token transitions."""
        if not seed_tokens or not self.token_transitions:
            return []
        generated = [t.strip().lower() for t in seed_tokens if t.strip()]
        for _ in range(max_tokens):
            next_candidates: dict[str, float] | None = None
            if len(generated) >= 2:
                bigram_key = f"{generated[-2]} {generated[-1]}"
                next_candidates = self.token_transitions.get(bigram_key)
            if not next_candidates:
                next_candidates = self.token_transitions.get(generated[-1])
            if not next_candidates:
                break
            best_tok = max(next_candidates.items(), key=lambda kv: kv[1])[0]
            if len(generated) >= 2 and best_tok == generated[-1]:
                break
            generated.append(best_tok)
            if best_tok.endswith("."):
                break
        return generated[len(seed_tokens) :]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this MiniLearningUnit for SQLite/JSON persistence."""
        return {
            "concept_id": self.concept_id,
            "role": self.role,
            "parent_hub": self.parent_hub,
            "dim": self.dim,
            "anchor_vec": [round(x, 6) for x in self.anchor_vec],
            "hidden_state": [round(x, 6) for x in self.hidden_state],
            "spokes": {
                name: {
                    "weights": [round(w, 6) for w in head.weights],
                    "bias": round(head.bias, 6),
                    "target_affinities": head.target_affinities,
                }
                for name, head in self.spokes.items()
            },
            "token_transitions": self.token_transitions,
            "activation_count": self.activation_count,
            "verified_hits": self.verified_hits,
            "refuted_hits": self.refuted_hits,
            "cumulative_surprise": round(self.cumulative_surprise, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MiniLearningUnit:
        """Restore a MiniLearningUnit from serialized state."""
        unit = cls(
            concept_id=str(data["concept_id"]),
            role=str(data.get("role", "leaf")),
            parent_hub=data.get("parent_hub"),
            dim=int(data.get("dim", 64)),
        )
        if "anchor_vec" in data and len(data["anchor_vec"]) == unit.dim:
            unit.anchor_vec = [float(x) for x in data["anchor_vec"]]
        if "hidden_state" in data and len(data["hidden_state"]) == unit.dim:
            unit.hidden_state = [float(x) for x in data["hidden_state"]]
        for name, s_data in data.get("spokes", {}).items():
            if name in unit.spokes:
                head = unit.spokes[name]
                if "weights" in s_data and len(s_data["weights"]) == unit.dim:
                    head.weights = [float(w) for w in s_data["weights"]]
                head.bias = float(s_data.get("bias", 0.0))
                head.target_affinities = {
                    str(k): list(v) for k, v in s_data.get("target_affinities", {}).items()
                }
        unit.token_transitions = {
            str(k): {str(nk): float(nv) for nk, nv in v.items()}
            for k, v in data.get("token_transitions", {}).items()
        }
        unit.activation_count = int(data.get("activation_count", 0))
        unit.verified_hits = float(data.get("verified_hits", 0.0))
        unit.refuted_hits = float(data.get("refuted_hits", 0.0))
        unit.cumulative_surprise = float(data.get("cumulative_surprise", 0.0))
        return unit
