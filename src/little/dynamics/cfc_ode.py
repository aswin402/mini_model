"""Closed-Form Continuous (CfC) Neural ODE & Hybrid Automaton Action Jumps."""

from __future__ import annotations

import math

from little.core.concept_knot import ConceptKnot, ContinuousPhysicalState
from little.dynamics.profiles import DynamicsProfileRegistry
from little.knowledge.registry import SchemaRegistry


class CfCContinuousODE:
    """Closed-Form Continuous ODE solver for physical state evolution over delta t."""

    def __init__(
        self,
        enzymatic_rate: float | None = None,
        basal_decay_rate: float | None = None,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> None:
        self.profile_registry = profile_registry or DynamicsProfileRegistry.default()
        parameters = self.profile_registry.ode
        self.reference_temperature_c = parameters.reference_temperature_c
        self.temperature_scale_c = parameters.temperature_scale_c
        self.protected_oxidation_rate = (
            parameters.protected_oxidation_rate_per_hour
        )
        self.exposed_decay_rate = parameters.exposed_decay_rate_per_hour
        self.protected_moisture_loss = (
            parameters.protected_moisture_loss_per_hour
        )
        self.exposed_moisture_loss = parameters.exposed_moisture_loss_per_hour
        self.minimum_moisture = parameters.minimum_moisture
        self.enzymatic_rate = (
            parameters.exposed_enzymatic_rate_per_hour
            if enzymatic_rate is None
            else enzymatic_rate
        )
        self.basal_decay_rate = (
            parameters.protected_decay_rate_per_hour
            if basal_decay_rate is None
            else basal_decay_rate
        )

    def evolve(
        self,
        state: ContinuousPhysicalState,
        delta_t_hours: float,
        skin_intact: bool = True,
    ) -> ContinuousPhysicalState:
        """Evolves the continuous physical state across delta_t_hours."""
        # Arrhenius temperature factor: decay accelerates with heat
        temp_factor = math.exp(
            (state.temperature - self.reference_temperature_c)
            / self.temperature_scale_c
        )

        if skin_intact:
            # Low basal decay when protected by external skin boundary
            decay_rate = self.basal_decay_rate * temp_factor
            d_ox = (
                self.protected_oxidation_rate
                * delta_t_hours
                * temp_factor
            )
        else:
            # Rapid enzymatic oxidation when tissue is exposed
            decay_rate = self.exposed_decay_rate * temp_factor
            d_ox = (
                (1.0 - state.oxidation)
                * (1.0 - math.exp(-self.enzymatic_rate * delta_t_hours))
                * temp_factor
            )

        new_freshness = max(
            0.0, state.freshness * math.exp(-decay_rate * delta_t_hours)
        )
        new_oxidation = min(1.0, state.oxidation + d_ox)
        new_moisture = max(
            self.minimum_moisture,
            state.moisture
            - (
                self.exposed_moisture_loss
                if not skin_intact
                else self.protected_moisture_loss
            )
            * delta_t_hours,
        )

        return ContinuousPhysicalState(
            freshness=round(new_freshness, 4),
            oxidation=round(new_oxidation, 4),
            moisture=round(new_moisture, 4),
            temperature=state.temperature,
        )


def apply_action_jump(
    knot: ConceptKnot, action: str, **kwargs
) -> list[ConceptKnot]:
    """Applies a discrete Hybrid Automaton jump to a Concept Knot."""
    schema = SchemaRegistry.default().action(action)
    if schema is None:
        raise ValueError(f"Unknown procedural action: {action}")

    n = int(kwargs.get("num_pieces", schema.defaults.get("count", 1)))
    if n < 1:
        raise ValueError("num_pieces must be positive")

    piece_mass = knot.invariant_mass / n
    transformed_parts = {
        part_name: (
            "partial_boundary"
            if topology == "external_boundary"
            else "exposed"
        )
        for part_name, topology in knot.mereology_parts.items()
    }
    pieces = []
    for i in range(n):
        piece = ConceptKnot(
            concept_id=f"{knot.concept_id}_PIECE_{i + 1}",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=transformed_parts,
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=knot.dynamics_state.oxidation,
                moisture=knot.dynamics_state.moisture,
                temperature=knot.dynamics_state.temperature,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=round(piece_mass, 2),
            procedural_skills=list(knot.procedural_skills),
            episodic_instances=[
                {"action": schema.name, "parent": knot.concept_id}
            ],
        )
        pieces.append(piece)
    return pieces
