"""Continuous-Time Dynamical Systems Engine (CfC / Liquid AI inspired) for LITTLE.

Tracks physical state trajectories across arbitrary continuous time deltas (Delta t)
using closed-form ODE solutions rather than static token predictions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class PhysicalState:
    """Continuous physical state of an object instance."""

    freshness: float = 1.0  # 1.0 (fresh) to 0.0 (decayed)
    oxidation: float = 0.0  # 0.0 (unoxidized) to 1.0 (fully oxidized/brown)
    temperature: float = 20.0  # Celsius
    exposed_to_air: bool = False
    tau_seconds: float = 86400.0 * 14  # Default 14 days for intact whole object
    timestamp: float = 0.0  # Epoch seconds when state was measured

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "freshness": round(self.freshness, 4),
            "oxidation": round(self.oxidation, 4),
            "temperature": self.temperature,
            "exposed_to_air": self.exposed_to_air,
            "tau_seconds": self.tau_seconds,
            "timestamp": self.timestamp,
        }


class ContinuousDynamicsEngine:
    """Evaluates continuous physical decay and chemical reactions over arbitrary time deltas."""

    # Baseline time constants in seconds
    TAU_INTACT_WHOLE = 86400.0 * 14.0  # 14 days
    TAU_SLICED_EXPOSED = 1800.0  # 30 minutes (rapid enzymatic browning)
    TAU_REFRIGERATED_WHOLE = 86400.0 * 60.0  # 60 days at 4 deg C

    @classmethod
    def compute_effective_tau(
        cls, base_tau: float, temperature_celsius: float
    ) -> float:
        """Apply Arrhenius thermal scaling: reaction rates double roughly every 10 deg C."""
        # Q10 temperature coefficient = 2.0
        thermal_rate_factor = 2.0 ** ((temperature_celsius - 20.0) / 10.0)
        thermal_rate_factor = max(0.1, min(thermal_rate_factor, 10.0))
        return base_tau / thermal_rate_factor

    @classmethod
    def evolve_state(cls, state: PhysicalState, delta_seconds: float) -> PhysicalState:
        """Evolve the physical state across delta_seconds using Closed-Form Continuous (CfC) ODE."""
        if delta_seconds <= 0:
            return state

        eff_tau = cls.compute_effective_tau(state.tau_seconds, state.temperature)

        # 1. Freshness decay: dx/dt = - (1 / tau) * x -> x(t) = x0 * exp(-t / tau)
        new_freshness = state.freshness * math.exp(-delta_seconds / eff_tau)

        # 2. Enzymatic Oxidation (browning): dx/dt = (1 / tau) * (1 - x)
        # -> x(t) = 1.0 - (1.0 - x0) * exp(-t / tau)
        if state.exposed_to_air:
            new_oxidation = 1.0 - (1.0 - state.oxidation) * math.exp(
                -delta_seconds / eff_tau
            )
        else:
            # Shielded by skin/rind: oxidation occurs at whole-body rate
            intact_tau = cls.compute_effective_tau(
                cls.TAU_INTACT_WHOLE, state.temperature
            )
            new_oxidation = 1.0 - (1.0 - state.oxidation) * math.exp(
                -delta_seconds / intact_tau
            )

        new_freshness = max(0.0, min(1.0, new_freshness))
        new_oxidation = max(0.0, min(1.0, new_oxidation))

        return PhysicalState(
            freshness=new_freshness,
            oxidation=new_oxidation,
            temperature=state.temperature,
            exposed_to_air=state.exposed_to_air,
            tau_seconds=state.tau_seconds,
            timestamp=state.timestamp + delta_seconds,
        )

    @classmethod
    def get_perceived_color(
        cls, state: PhysicalState, base_interior_color: str = "white"
    ) -> str:
        """Map continuous oxidation coordinate to symbolic sensory color descriptor."""
        if not state.exposed_to_air and state.oxidation < 0.15:
            return base_interior_color
        if state.oxidation < 0.25:
            return base_interior_color
        elif state.oxidation < 0.60:
            return "light brown"
        else:
            return "brown"

    @classmethod
    def get_perceived_condition(cls, state: PhysicalState) -> str:
        """Map continuous freshness coordinate to human condition descriptor."""
        if state.freshness >= 0.75:
            return "fresh"
        elif state.freshness >= 0.40:
            return "stale"
        elif state.freshness >= 0.15:
            return "spoiled"
        else:
            return "rotten"

    @classmethod
    def create_initial_state(
        cls, exposed_to_air: bool = False, temperature: float = 20.0
    ) -> PhysicalState:
        """Create a new pristine physical state."""
        base_tau = cls.TAU_SLICED_EXPOSED if exposed_to_air else cls.TAU_INTACT_WHOLE
        return PhysicalState(
            freshness=1.0,
            oxidation=0.0,
            temperature=temperature,
            exposed_to_air=exposed_to_air,
            tau_seconds=base_tau,
            timestamp=time.time(),
        )
