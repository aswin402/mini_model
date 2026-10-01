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
    """Applies a discrete Hybrid Automaton jump to a Concept Knot with QPT invariants."""
    schema = SchemaRegistry.default().action(action)
    if schema is None:
        raise ValueError(f"Unknown procedural action: {action}")

    action_name = schema.name.lower()

    if action_name == "peel":
        # Separate external boundary skin from interior pulp with strict mass conservation
        peel_ratio = float(kwargs.get("peel_fraction", 0.15))
        skin_mass = round(knot.invariant_mass * peel_ratio, 2)
        flesh_mass = round(knot.invariant_mass - skin_mass, 2)

        skin_parts = {
            k: v for k, v in knot.mereology_parts.items()
            if v == "external_boundary" or "skin" in k or "peel" in k or "shell" in k
        }
        if not skin_parts:
            skin_parts = {"skin": "external_boundary"}

        flesh_parts = {
            k: ("removed" if k in skin_parts else "exposed")
            for k in knot.mereology_parts
        }
        if "pulp" not in flesh_parts and "core" not in flesh_parts:
            flesh_parts["pulp"] = "exposed"

        skin_piece = ConceptKnot(
            concept_id=f"{knot.concept_id}_PEEL",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=skin_parts,
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=knot.dynamics_state.oxidation,
                moisture=knot.dynamics_state.moisture,
                temperature=knot.dynamics_state.temperature,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=skin_mass,
            procedural_skills=["dry", "dehydrate"],
            episodic_instances=[{"action": "peel", "parent": knot.concept_id, "part": "skin"}],
        )

        flesh_piece = ConceptKnot(
            concept_id=f"{knot.concept_id}_FLESH",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=flesh_parts,
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=knot.dynamics_state.oxidation,
                moisture=knot.dynamics_state.moisture,
                temperature=knot.dynamics_state.temperature,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=flesh_mass,
            procedural_skills=["slice", "juice"],
            episodic_instances=[{"action": "peel", "parent": knot.concept_id, "part": "flesh"}],
        )
        return [skin_piece, flesh_piece]

    elif action_name == "heat":
        delta_c = float(kwargs.get("delta_celsius", schema.defaults.get("delta_celsius", 20.0)))
        new_temp = round(knot.dynamics_state.temperature + delta_c, 2)
        heated = ConceptKnot(
            concept_id=f"{knot.concept_id}_HEATED",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=dict(knot.mereology_parts),
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=knot.dynamics_state.oxidation,
                moisture=max(0.05, knot.dynamics_state.moisture - 0.02),
                temperature=new_temp,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=knot.invariant_mass,
            procedural_skills=list(knot.procedural_skills),
            episodic_instances=[{"action": "heat", "parent": knot.concept_id, "delta_celsius": delta_c}],
        )
        return [heated]

    elif action_name in ("chill", "cool"):
        delta_c = float(kwargs.get("delta_celsius", schema.defaults.get("delta_celsius", 15.0)))
        new_temp = round(max(0.0, knot.dynamics_state.temperature - delta_c), 2)
        chilled = ConceptKnot(
            concept_id=f"{knot.concept_id}_CHILLED",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=dict(knot.mereology_parts),
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=knot.dynamics_state.oxidation,
                moisture=knot.dynamics_state.moisture,
                temperature=new_temp,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=knot.invariant_mass,
            procedural_skills=list(knot.procedural_skills),
            episodic_instances=[{"action": "chill", "parent": knot.concept_id, "delta_celsius": delta_c}],
        )
        return [chilled]

    elif action_name in ("dehydrate", "dry"):
        min_moisture = 0.05
        current_moisture = knot.dynamics_state.moisture
        dry_mass = knot.invariant_mass * (1.0 - current_moisture)
        new_mass = round(dry_mass / (1.0 - min_moisture), 2)
        dried = ConceptKnot(
            concept_id=f"{knot.concept_id}_DRIED",
            taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
            taxonomy_hyponyms=[],
            mereology_parts=dict(knot.mereology_parts),
            dynamics_state=ContinuousPhysicalState(
                freshness=knot.dynamics_state.freshness,
                oxidation=min(1.0, knot.dynamics_state.oxidation + 0.05),
                moisture=min_moisture,
                temperature=knot.dynamics_state.temperature,
            ),
            invariant_disjoints=list(knot.invariant_disjoints),
            invariant_mass=new_mass,
            procedural_skills=list(knot.procedural_skills),
            episodic_instances=[{"action": "dehydrate", "parent": knot.concept_id}],
        )
        return [dried]

    # Default to slicing / partitioning
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
