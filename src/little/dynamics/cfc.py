"""Continuous-Time Dynamical Systems Engine (CfC / Liquid AI inspired) for LITTLE.

Tracks physical state trajectories across arbitrary continuous time deltas (Delta t)
using closed-form ODE solutions rather than static token predictions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from little.dynamics.profiles import DynamicsProfileRegistry


def _default_continuous_value(name: str) -> float:
    return float(getattr(DynamicsProfileRegistry.default().continuous, name))


@dataclass
class PhysicalState:
    """Continuous physical state of an object instance."""

    freshness: float = field(
        default_factory=lambda: _default_continuous_value("initial_freshness")
    )
    oxidation: float = field(
        default_factory=lambda: _default_continuous_value("initial_oxidation")
    )
    temperature: float = field(
        default_factory=lambda: _default_continuous_value("default_temperature_c")
    )
    exposed_to_air: bool = False
    tau_seconds: float = field(
        default_factory=lambda: DynamicsProfileRegistry.default()
        .default_for(False)
        .tau_whole_seconds
    )
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

    @classmethod
    def compute_effective_tau(
        cls,
        base_tau: float,
        temperature_celsius: float,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> float:
        """Apply Arrhenius thermal scaling: reaction rates double roughly every 10 deg C."""
        registry = profile_registry or DynamicsProfileRegistry.default()
        parameters = registry.continuous
        thermal_rate_factor = parameters.q10 ** (
            (
                temperature_celsius - registry.ode.reference_temperature_c
            )
            / registry.ode.temperature_scale_c
        )
        thermal_rate_factor = max(
            parameters.min_rate_factor,
            min(thermal_rate_factor, parameters.max_rate_factor),
        )
        return base_tau / thermal_rate_factor

    @classmethod
    def evolve_state(
        cls,
        state: PhysicalState,
        delta_seconds: float,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> PhysicalState:
        """Evolve the physical state across delta_seconds using Closed-Form Continuous (CfC) ODE."""
        if delta_seconds <= 0:
            return state

        eff_tau = cls.compute_effective_tau(
            state.tau_seconds, state.temperature, profile_registry=profile_registry
        )

        # 1. Freshness decay: dx/dt = - (1 / tau) * x -> x(t) = x0 * exp(-t / tau)
        new_freshness = state.freshness * math.exp(-delta_seconds / eff_tau)

        # 2. Enzymatic Oxidation (browning): dx/dt = (1 / tau) * (1 - x)
        # -> x(t) = 1.0 - (1.0 - x0) * exp(-t / tau)
        if state.exposed_to_air:
            new_oxidation = 1.0 - (1.0 - state.oxidation) * math.exp(
                -delta_seconds / eff_tau
            )
        else:
            # Shielded by the external boundary: use the state's whole-body rate.
            intact_tau = cls.compute_effective_tau(
                state.tau_seconds,
                state.temperature,
                profile_registry=profile_registry,
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
        cls,
        state: PhysicalState,
        base_interior_color: str | None = None,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> str:
        """Map continuous oxidation coordinate to symbolic sensory color descriptor."""
        registry = profile_registry or DynamicsProfileRegistry.default()
        profile = registry
        base_interior_color = base_interior_color or profile.default_for(
            state.exposed_to_air
        ).default_interior_color
        low, high = profile.perception.color
        _, intermediate_label, advanced_label = profile.perception.color_labels
        if (
            not state.exposed_to_air
            and state.oxidation
            < profile.continuous.protected_color_oxidation_threshold
        ):
            return base_interior_color
        if state.oxidation < low:
            return base_interior_color
        elif state.oxidation < high:
            return intermediate_label
        else:
            return advanced_label

    @classmethod
    def get_perceived_condition(cls, state: PhysicalState) -> str:
        """Map continuous freshness coordinate to human condition descriptor."""
        registry = DynamicsProfileRegistry.default()
        fresh, stale, spoiled = registry.perception.condition
        fresh_label, stale_label, spoiled_label, rotten_label = (
            registry.perception.condition_labels
        )
        if state.freshness >= fresh:
            return fresh_label
        elif state.freshness >= stale:
            return stale_label
        elif state.freshness >= spoiled:
            return spoiled_label
        else:
            return rotten_label

    @classmethod
    def create_initial_state(
        cls,
        exposed_to_air: bool = False,
        temperature: float | None = None,
        profile: str | None = None,
        tau_seconds: float | None = None,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> PhysicalState:
        """Create a new pristine physical state."""
        registry = profile_registry or DynamicsProfileRegistry.default()
        selected = registry.profile(profile) if profile else registry.default_for(exposed_to_air)
        base_tau = tau_seconds if tau_seconds is not None else (
            selected.tau_exposed_seconds
            if exposed_to_air
            else selected.tau_whole_seconds
        )
        return PhysicalState(
            freshness=registry.continuous.initial_freshness,
            oxidation=registry.continuous.initial_oxidation,
            temperature=(
                registry.continuous.default_temperature_c
                if temperature is None
                else temperature
            ),
            exposed_to_air=exposed_to_air,
            tau_seconds=base_tau,
            timestamp=time.time(),
        )
