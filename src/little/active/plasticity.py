"""Continual-Learning Plasticity & Trunk/Leaf Rate Controller.

Inspired by volotat/mini-AGI (`minagi/plasticity.py` and `minagi/paged.py`):
1. Dual Exponentially Weighted Least-Squares (EWLS) fits (slow lambda=0.97, fast lambda=0.85)
   evaluating scale-free effect size e = -slope / sigma rather than a horizon schedule.
2. Spike holdback filter (SPIKE = 5.0 sigmas) and Regime-shift detector (JUMP_SE = 4.0,
   HOLD_SE = 2.0) that steps plasticity back up when entering a novel domain.
3. Dual-speed Trunk vs. Leaf plasticity (`trunk_lr_mult = 0.1`): foundational upper-ontology
   concepts (high centrality / low depth) move at 0.1x the leaf plasticity rate to prevent
   catastrophic interference during batch-1 stream ingestion, while newborn leaf concepts
   start at full plasticity and relax with maturity (`rate_half`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


@dataclass
class EWLSFit:
    """Running exponentially weighted least-squares fit without a finite window edge."""

    lam: float
    w: float = 0.0
    w2: float = 0.0
    x: float = 0.0
    y: float = 0.0
    xx: float = 0.0
    xy: float = 0.0
    yy: float = 0.0

    def update(self, idx_or_val: float, val: float | None = None) -> None:
        if val is None:
            self._auto_idx = getattr(self, "_auto_idx", 0) + 1
            idx = float(self._auto_idx)
            actual_val = float(idx_or_val)
        else:
            idx = float(idx_or_val)
            actual_val = float(val)
        self.w = self.lam * self.w + 1.0
        self.w2 = (self.lam * self.lam) * self.w2 + 1.0
        self.x = self.lam * self.x + idx
        self.y = self.lam * self.y + actual_val
        self.xx = self.lam * self.xx + idx * idx
        self.xy = self.lam * self.xy + idx * actual_val
        self.yy = self.lam * self.yy + actual_val * actual_val

    def shift_level(self, delta: float) -> None:
        """Shift running y sums after a configuration restart re-leveling."""
        if self.w <= 0.0:
            return
        self.yy += 2.0 * delta * self.y + delta * delta * self.w
        self.xy += delta * self.x
        self.y += delta * self.w

    @property
    def n_eff(self) -> float:
        return (self.w * self.w) / self.w2 if self.w2 > 0.0 else 0.0

    def predict(self, idx: float) -> float:
        if self.w <= 0.0:
            return 0.0
        mean_x = self.x / self.w
        mean_y = self.y / self.w
        return mean_y + self.slope * (idx - mean_x)

    @property
    def slope(self) -> float:
        if self.w <= 0.0:
            return 0.0
        cov_xy = self.xy - (self.x * self.y) / self.w
        var_x = self.xx - (self.x * self.x) / self.w
        if var_x <= 1e-12:
            return 0.0
        return cov_xy / var_x

    def stats(self) -> tuple[float, float, float]:
        """Return (slope, residual_sigma, t_stat)."""
        if self.w <= 2.0:
            return 0.0, 0.0, 0.0
        var_x = self.xx - (self.x * self.x) / self.w
        var_y = self.yy - (self.y * self.y) / self.w
        cov_xy = self.xy - (self.x * self.y) / self.w
        if var_x <= 1e-12:
            return 0.0, 0.0, 0.0
        slp = cov_xy / var_x
        rss = max(0.0, var_y - slp * cov_xy)
        dof = max(1.0, self.n_eff - 2.0)
        sigma = max(1e-4, math.sqrt(rss / (self.w / self.n_eff * dof))) if self.n_eff > 2.0 else 0.0
        se_slope = sigma / math.sqrt(var_x * (self.n_eff / self.w)) if sigma > 0.0 else 0.0
        t_stat = -slp / se_slope if se_slope > 1e-12 else 0.0
        return slp, sigma, t_stat

    @property
    def effect_size(self) -> float:
        """Scale-free improvement per evaluation: e = -slope / sigma."""
        slp, sigma, _ = self.stats()
        if sigma <= 1e-12:
            return 0.0
        return -slp / sigma


@dataclass
class PlasticityTelemetry:
    """Inspectable state of the continual-learning plasticity controller."""

    scale: float
    evaluations_count: int
    slow_effect_size: float
    fast_effect_size: float
    regime_shifts: int
    spikes_filtered: int
    trunk_lr_mult: float
    spike_detected: bool = False

    @property
    def step(self) -> int:
        return self.evaluations_count

    @property
    def plasticity_multiplier(self) -> float:
        return self.scale


@dataclass
class PlasticityController:
    """Evidence-governed plasticity and trunk/leaf learning rate controller."""

    FLOOR: float = 0.01
    CEIL: float = 1.0
    LAM: float = 0.97
    LAM_FAST: float = 0.85
    EFFECT: float = 45.0
    MIN_EFF: float = 6.0
    NUDGE: float = 0.0008
    NUDGE_DOWN: float = 0.004
    T_W_DOWN: float = 6.0
    T_MID: float = 2.2
    T_W: float = 0.75
    MAX_MOVE: float = 0.02
    JUMP_SE: float = 4.0
    HOLD_SE: float = 2.0
    FLOOR_JUMP: float = 0.25
    UP: float = 2.0
    SPIKE: float = 5.0

    base_lr: float = 1.0
    spike_sigma: float = 5.0
    hold_steps: int = 1
    trunk_lr_mult: float = 0.1
    rate_half: float = 15.0
    scale: float = 1.0

    _slow: EWLSFit = field(default_factory=lambda: EWLSFit(0.97))
    _fast: EWLSFit = field(default_factory=lambda: EWLSFit(0.85))
    _eval_index: int = 0
    _held_spike: tuple[int, float] | None = None
    _last_spike_detected: bool = False
    _pending_regime_jump: bool = False
    _regime_shifts: int = 0
    _spikes_filtered: int = 0
    _concept_steps: dict[str, int] = field(default_factory=dict)

    @property
    def plasticity_multiplier(self) -> float:
        return self.scale

    def observe_loss(self, heldout_loss: float) -> float:
        """Update EWLS fits with a new held-out loss / contradiction rate reading and nudge scale."""
        if not math.isfinite(heldout_loss) or heldout_loss < 0.0:
            return self.scale

        self._eval_index += 1
        idx = float(self._eval_index)
        self._last_spike_detected = False
        spike_thresh = min(self.SPIKE, self.spike_sigma)

        # Check spike filter & regime detector if slow fit is warmed up
        if self._slow.n_eff >= self.MIN_EFF:
            pred = self._slow.predict(idx)
            _, sigma, _ = self._slow.stats()
            residual = heldout_loss - pred

            if self._held_spike is not None:
                prev_idx, prev_val = self._held_spike
                self._held_spike = None
                # If current reading landed nearer the held reading than the trend line, both count
                if abs(heldout_loss - prev_val) < abs(heldout_loss - pred):
                    self._slow.update(float(prev_idx), prev_val)
                    self._fast.update(float(prev_idx), prev_val)
                    if (
                        sigma > 1e-9
                        and (prev_val - pred) >= max(self.FLOOR_JUMP, self.JUMP_SE * sigma)
                        and (heldout_loss - pred) >= self.HOLD_SE * sigma
                    ):
                        self.scale = min(self.CEIL, self.scale * self.UP)
                        self._regime_shifts += 1
                else:
                    self._spikes_filtered += 1
            elif sigma > 1e-9 and abs(residual) > spike_thresh * sigma:
                # Hold back single anomalous spike and temporarily damp scale during holdback
                self._held_spike = (self._eval_index, heldout_loss)
                self._last_spike_detected = True
                self._spikes_filtered += 1
                self.scale = max(self.FLOOR, self.scale * 0.5)
                return self.scale

        self._slow.update(idx, heldout_loss)
        self._fast.update(idx, heldout_loss)

        if self._slow.n_eff < self.MIN_EFF:
            return self.scale

        _, _, t_slow = self._slow.stats()
        e_slow = self._slow.effect_size
        verdict_slow = min(t_slow, self.EFFECT * e_slow)

        if self._fast.n_eff >= min(self.MIN_EFF, 4.0):
            _, _, t_fast = self._fast.stats()
            e_fast = self._fast.effect_size
            verdict_fast = min(t_fast, self.EFFECT * e_fast)
            verdict = min(verdict_slow, verdict_fast)
        else:
            verdict = verdict_slow

        delta = verdict - self.T_MID
        if delta >= 0.0:
            log_move = self.NUDGE * math.tanh(delta / self.T_W)
        else:
            log_move = self.NUDGE_DOWN * math.tanh(delta / self.T_W_DOWN)

        factor = math.exp(max(-self.MAX_MOVE, min(self.MAX_MOVE, log_move)))
        self.scale = max(self.FLOOR, min(self.CEIL, self.scale * factor))
        return self.scale

    def observe_error(self, error: float) -> PlasticityTelemetry:
        """Observe an online prediction/epistemic error and return updated telemetry."""
        self.observe_loss(error)
        return self.get_telemetry()

    def concept_learning_rate(
        self, *, is_trunk: bool = False, evidence_count: int = 0
    ) -> float:
        """Compute effective learning rate for trunk (`0.1x`) vs. leaf (`rate_half` decay) concepts."""
        if is_trunk:
            return round(self.base_lr * self.scale * self.trunk_lr_mult, 4)
        decay = 0.5 ** (max(0, evidence_count) / max(1.0, self.rate_half))
        leaf_mult = self.scale * (0.25 + 0.75 * decay)
        return round(self.base_lr * max(self.FLOOR, min(self.CEIL, leaf_mult)), 4)

    def is_trunk_concept(self, memory: MemoryStore, concept_name: str) -> bool:
        """Identify upper-ontology trunk concepts (high incoming taxonomic degree or root category)."""
        c = memory.get_concept(concept_name)
        if not c:
            return False
        trunk_roots = {
            "entity",
            "animal",
            "plant",
            "living_thing",
            "physical_object",
            "substance",
            "matter",
            "event",
            "location",
            "place",
            "abstract_concept",
        }
        if c.name.lower() in trunk_roots:
            return True
        incoming = memory.get_relations(object_id=c.id)
        taxonomic_children = sum(
            1
            for r in incoming
            if r.predicate in ("is_a", "subclass_of", "instance_of")
            and r.weight_positive > r.weight_negative
        )
        return taxonomic_children >= 3

    def concept_plasticity_multiplier(
        self, memory: MemoryStore, concept_name: str, *, step_trained: bool = False
    ) -> float:
        """Compute per-concept learning rate multiplier (`0.1x` for trunk, decaying newborn rate for leaves)."""
        key = concept_name.strip().lower()
        if step_trained:
            self._concept_steps[key] = self._concept_steps.get(key, 0) + 1

        if self.is_trunk_concept(memory, key):
            return round(self.scale * self.trunk_lr_mult, 4)

        steps = self._concept_steps.get(key, 0)
        # Newborn leaf concepts start at 1.0 and relax toward self.scale with half-life `rate_half`
        decay = 0.5 ** (steps / max(1.0, self.rate_half))
        leaf_mult = self.scale + (1.0 - self.scale) * decay
        return round(max(self.FLOOR, min(self.CEIL, leaf_mult)), 4)

    def get_telemetry(self) -> PlasticityTelemetry:
        return PlasticityTelemetry(
            scale=round(self.scale, 4),
            evaluations_count=self._eval_index,
            slow_effect_size=round(self._slow.effect_size, 4),
            fast_effect_size=round(self._fast.effect_size, 4),
            regime_shifts=self._regime_shifts,
            spikes_filtered=self._spikes_filtered,
            trunk_lr_mult=self.trunk_lr_mult,
            spike_detected=self._last_spike_detected,
        )
