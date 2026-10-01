"""Continuous-Time Physical Simulation Engine for predicting entity trajectories over time."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from little.core.concept_knot import ConceptKnot, ContinuousPhysicalState
from little.dynamics.cfc_ode import CfCContinuousODE


@dataclass
class PhysicalMilestone:
    """A key qualitative physical event along the temporal trajectory."""

    time_hours: float
    state: ContinuousPhysicalState
    description: str


@dataclass
class SimulationResult:
    """Comprehensive summary of a continuous-time physical simulation."""

    concept_name: str
    initial_state: ContinuousPhysicalState
    final_state: ContinuousPhysicalState
    delta_t_hours: float
    ambient_temperature_c: float
    skin_intact: bool
    milestones: list[PhysicalMilestone] = field(default_factory=list)
    summary: str = ""


class PhysicalSimulationEngine:
    """Simulates physical state trajectories across arbitrary time deltas using CfC ODEs."""

    def __init__(self, ode: CfCContinuousODE | None = None) -> None:
        self.ode = ode or CfCContinuousODE()

    def simulate(
        self,
        target: ConceptKnot | ContinuousPhysicalState | str,
        delta_t_hours: float,
        *,
        ambient_temperature_c: float | None = None,
        skin_intact: bool = True,
        sample_points: int = 10,
    ) -> SimulationResult:
        """Run continuous-time simulation forward delta_t_hours."""
        if delta_t_hours < 0:
            raise ValueError("delta_t_hours must be non-negative")

        concept_name = "PhysicalEntity"
        if isinstance(target, ConceptKnot):
            concept_name = target.concept_id
            init_s = target.dynamics_state
        elif isinstance(target, ContinuousPhysicalState):
            init_s = target
        else:
            concept_name = str(target)
            init_s = ContinuousPhysicalState(
                freshness=1.0, oxidation=0.0, moisture=0.86, temperature=20.0
            )

        temp = ambient_temperature_c if ambient_temperature_c is not None else init_s.temperature
        cur_state = ContinuousPhysicalState(
            freshness=init_s.freshness,
            oxidation=init_s.oxidation,
            moisture=init_s.moisture,
            temperature=temp,
        )

        final_state = self.ode.evolve(cur_state, delta_t_hours=delta_t_hours, skin_intact=skin_intact)

        # Generate trajectory milestones
        milestones: list[PhysicalMilestone] = []
        steps = max(2, sample_points)
        dt_step = delta_t_hours / float(steps)

        flag_freshness_dropped = False
        flag_spoilage = False
        flag_browning = False
        flag_severe_oxidation = False
        flag_moisture_loss = False

        for i in range(1, steps + 1):
            t_curr = round(dt_step * i, 2)
            step_state = self.ode.evolve(cur_state, delta_t_hours=t_curr, skin_intact=skin_intact)

            if not flag_freshness_dropped and step_state.freshness < 0.75:
                milestones.append(
                    PhysicalMilestone(
                        time_hours=t_curr,
                        state=step_state,
                        description=f"Freshness drops to {step_state.freshness:.2f} (peak freshness lost).",
                    )
                )
                flag_freshness_dropped = True

            if not flag_browning and step_state.oxidation >= 0.20:
                milestones.append(
                    PhysicalMilestone(
                        time_hours=t_curr,
                        state=step_state,
                        description=f"Enzymatic oxidation reaches {step_state.oxidation:.2f} (visible browning).",
                    )
                )
                flag_browning = True

            if not flag_severe_oxidation and step_state.oxidation >= 0.60:
                milestones.append(
                    PhysicalMilestone(
                        time_hours=t_curr,
                        state=step_state,
                        description=f"Severe oxidation ({step_state.oxidation:.2f}); discoloration and softening.",
                    )
                )
                flag_severe_oxidation = True

            if not flag_spoilage and step_state.freshness < 0.30:
                milestones.append(
                    PhysicalMilestone(
                        time_hours=t_curr,
                        state=step_state,
                        description=f"Freshness falls to {step_state.freshness:.2f} (advanced spoilage / decay).",
                    )
                )
                flag_spoilage = True

            if not flag_moisture_loss and (init_s.moisture - step_state.moisture) >= 0.20:
                milestones.append(
                    PhysicalMilestone(
                        time_hours=t_curr,
                        state=step_state,
                        description=f"Significant moisture loss ({step_state.moisture:.2f}); tissue shriveling.",
                    )
                )
                flag_moisture_loss = True

        # Build natural language summary
        condition_str = (
            "decayed / spoiled" if final_state.freshness < 0.30
            else "past peak freshness" if final_state.freshness < 0.70
            else "relatively fresh"
        )
        browning_str = (
            "severely browned" if final_state.oxidation > 0.60
            else "partially oxidized" if final_state.oxidation > 0.20
            else "minimal oxidation"
        )

        summary = (
            f"Simulation of {concept_name} over {delta_t_hours:.1f} hours at {temp:.1f}°C "
            f"({'intact skin' if skin_intact else 'exposed flesh'}):\n"
            f"- Freshness: {init_s.freshness:.2f} -> {final_state.freshness:.2f} ({condition_str})\n"
            f"- Oxidation: {init_s.oxidation:.2f} -> {final_state.oxidation:.2f} ({browning_str})\n"
            f"- Moisture: {init_s.moisture:.2f} -> {final_state.moisture:.2f}\n"
            f"- Temperature: {temp:.1f}°C"
        )

        return SimulationResult(
            concept_name=concept_name,
            initial_state=init_s,
            final_state=final_state,
            delta_t_hours=delta_t_hours,
            ambient_temperature_c=temp,
            skin_intact=skin_intact,
            milestones=milestones,
            summary=summary,
        )
