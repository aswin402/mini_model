"""Closed-Form Continuous (CfC) Neural ODE & Hybrid Automaton Action Jumps."""

from __future__ import annotations

import math
from typing import List
from little.core.concept_knot import ConceptKnot, ContinuousPhysicalState


class CfCContinuousODE:
    """Closed-Form Continuous ODE solver for physical state evolution over delta t."""

    def __init__(
        self, enzymatic_rate: float = 0.15, basal_decay_rate: float = 0.002
    ) -> None:
        self.enzymatic_rate = enzymatic_rate
        self.basal_decay_rate = basal_decay_rate

    def evolve(
        self,
        state: ContinuousPhysicalState,
        delta_t_hours: float,
        skin_intact: bool = True,
    ) -> ContinuousPhysicalState:
        """Evolves the continuous physical state across delta_t_hours."""
        # Arrhenius temperature factor: decay accelerates with heat
        temp_factor = math.exp((state.temperature - 20.0) / 10.0)

        if skin_intact:
            # Low basal decay when protected by external skin boundary
            decay_rate = self.basal_decay_rate * temp_factor
            d_ox = 0.0005 * delta_t_hours * temp_factor
        else:
            # Rapid enzymatic oxidation when tissue is exposed
            decay_rate = 0.05 * temp_factor
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
            0.05,
            state.moisture
            - (0.01 if not skin_intact else 0.0005) * delta_t_hours,
        )

        return ContinuousPhysicalState(
            freshness=round(new_freshness, 4),
            oxidation=round(new_oxidation, 4),
            moisture=round(new_moisture, 4),
            temperature=state.temperature,
        )


def apply_action_jump(
    knot: ConceptKnot, action: str, **kwargs
) -> List[ConceptKnot]:
    """Applies a discrete Hybrid Automaton jump to a Concept Knot."""
    if action == "slice":
        n = kwargs.get("num_pieces", 4)
        piece_mass = knot.invariant_mass / max(1, n)
        pieces = []
        for i in range(n):
            piece = ConceptKnot(
                concept_id=f"{knot.concept_id}_PIECE_{i+1}",
                taxonomy_hypernyms=list(knot.taxonomy_hypernyms),
                taxonomy_hyponyms=[],
                mereology_parts={
                    "skin": "partial_boundary",
                    "pulp": "exposed",
                    "core": "fragmented",
                },
                dynamics_state=ContinuousPhysicalState(
                    freshness=knot.dynamics_state.freshness,
                    oxidation=knot.dynamics_state.oxidation,
                    moisture=knot.dynamics_state.moisture,
                    temperature=knot.dynamics_state.temperature,
                ),
                invariant_disjoints=list(knot.invariant_disjoints),
                invariant_mass=round(piece_mass, 2),
                procedural_skills=["eat", "dehydrate", "compost"],
                episodic_instances=[
                    {"action": "slice", "parent": knot.concept_id}
                ],
            )
            pieces.append(piece)
        return pieces

    raise ValueError(f"Unknown procedural action: {action}")
