"""Continuous Counterfactual Reasoning Engine — Simulating hypothetical worlds via CfC ODEs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from little.core.models import BeliefStatus, InferenceResult
from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.counterfactual_policy import CounterfactualPolicy

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


@dataclass
class CounterfactualScenario:
    """Quantitative outcome of a hypothetical physical trajectory."""

    temperature_c: float
    duration_hours: float
    freshness: float
    oxidation: float
    condition: str
    perceived_color: str
    rate_factor: float


@dataclass
class ContrastiveComparisonResult:
    """Comparative analysis of two hypothetical worlds."""

    subject: str
    scenario_1: CounterfactualScenario
    scenario_2: CounterfactualScenario
    thermal_ratio: float
    verbalized: str
    trace: list[str] = field(default_factory=list)


class CounterfactualDynamicsEngine:
    """Evaluates hypothetical and contrastive physical dynamics over continuous time."""

    def __init__(
        self,
        memory: MemoryStore,
        policy: CounterfactualPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or CounterfactualPolicy.default()

    def resolve_temperature(self, text: str, default: float = 20.0) -> float:
        """Resolve numeric temperature in Celsius or text alias from data-driven policy."""
        clean = text.lower().strip()
        # Direct regex for numbers: e.g. 4C, 4°C, 4 degrees, 30 deg
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*c|deg\s*c|c\b|degrees?)?", clean)
        if m:
            val = float(m.group(1))
            # If explicitly Fahrenheit (e.g. 68F)
            if "f" in clean and "c" not in clean:
                return (val - 32.0) * 5.0 / 9.0
            return val

        for alias, temp in self.policy.temperature_aliases.items():
            if alias in clean:
                return temp
        return default

    def resolve_duration_hours(self, text: str, default: float = 2.0) -> float:
        """Extract duration in hours from natural language query."""
        clean = text.lower().strip()
        m_day = re.search(r"(\d+(?:\.\d+)?)\s*(?:day|days|d\b)", clean)
        if m_day:
            return float(m_day.group(1)) * 24.0

        m_hr = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hr|hrs|h\b)", clean)
        if m_hr:
            return float(m_hr.group(1))

        m_min = re.search(r"(\d+(?:\.\d+)?)\s*(?:minute|minutes|min|mins|m\b)", clean)
        if m_min:
            return float(m_min.group(1)) / 60.0

        return default

    def determine_condition(self, freshness: float) -> str:
        """Categorize freshness value based on policy thresholds."""
        for label, threshold in sorted(
            self.policy.freshness_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if freshness >= threshold:
                return label
        return "spoiled"

    def simulate_scenario(
        self,
        subject: str,
        temperature_c: float,
        duration_hours: float,
        is_exposed: bool | None = None,
    ) -> CounterfactualScenario:
        """Simulate a single hypothetical world trajectory via Closed-Form Continuous ODE."""
        clean_subj = subject.strip().lower()
        concept = self.memory.get_concept(clean_subj)

        if is_exposed is None:
            is_exposed = bool(
                (concept and concept.attributes.get("exposed_flesh", False))
                or "slice" in clean_subj
                or "peel" in clean_subj
                or "cut" in clean_subj
            )

        state = ContinuousDynamicsEngine.create_initial_state(
            exposed_to_air=is_exposed
        )
        state.temperature = temperature_c

        dt_seconds = duration_hours * 3600.0
        evolved = ContinuousDynamicsEngine.evolve_state(state, delta_seconds=dt_seconds)

        base_color = (
            concept.attributes.get("interior_color", "white")
            if concept
            else "white"
        )
        color = ContinuousDynamicsEngine.get_perceived_color(evolved, base_color)
        condition = self.determine_condition(evolved.freshness)

        eff_tau = ContinuousDynamicsEngine.compute_effective_tau(
            state.tau_seconds, temperature_c
        )
        rate_factor = state.tau_seconds / eff_tau if eff_tau > 0 else 1.0

        return CounterfactualScenario(
            temperature_c=round(temperature_c, 1),
            duration_hours=round(duration_hours, 2),
            freshness=round(evolved.freshness, 3),
            oxidation=round(evolved.oxidation, 3),
            condition=condition,
            perceived_color=color,
            rate_factor=round(rate_factor, 2),
        )

    def compare_scenarios(
        self,
        subject: str,
        temp1: float,
        dur1: float,
        temp2: float,
        dur2: float,
        label1: str = "Condition 1",
        label2: str = "Condition 2",
    ) -> ContrastiveComparisonResult:
        """Compare two parallel counterfactual worlds and evaluate Arrhenius divergence."""
        sc1 = self.simulate_scenario(subject, temp1, dur1)
        sc2 = self.simulate_scenario(subject, temp2, dur2)

        ratio = sc2.rate_factor / sc1.rate_factor if sc1.rate_factor > 0 else 1.0

        if ratio > 1.0:
            summary = (
                f"{label1} preserves freshness significantly longer by slowing decay {ratio:.1f}x."
            )
        elif ratio < 1.0:
            summary = (
                f"{label2} slows enzymatic decay {1.0 / ratio:.1f}x relative to {label1}."
            )
        else:
            summary = "Both conditions evolve at approximately equal rates."

        rendered = self.policy.contrastive_template.format(
            subject=subject,
            label1=label1,
            temp1=sc1.temperature_c,
            dur1=sc1.duration_hours,
            f1=sc1.freshness,
            ox1=sc1.oxidation,
            cond1=sc1.condition,
            color1=sc1.perceived_color,
            label2=label2,
            temp2=sc2.temperature_c,
            dur2=sc2.duration_hours,
            f2=sc2.freshness,
            ox2=sc2.oxidation,
            cond2=sc2.condition,
            color2=sc2.perceived_color,
            thermal_ratio=ratio,
            summary=summary,
        )

        trace = [
            f"Counterfactual simulation for '{subject}'",
            f"World 1: T={sc1.temperature_c}°C, dt={sc1.duration_hours}h -> Freshness={sc1.freshness:.3f}, Oxidation={sc1.oxidation:.3f}",
            f"World 2: T={sc2.temperature_c}°C, dt={sc2.duration_hours}h -> Freshness={sc2.freshness:.3f}, Oxidation={sc2.oxidation:.3f}",
            f"Arrhenius kinetic scaling ratio: {ratio:.2f}x",
        ]

        return ContrastiveComparisonResult(
            subject=subject,
            scenario_1=sc1,
            scenario_2=sc2,
            thermal_ratio=round(ratio, 2),
            verbalized=rendered,
            trace=trace,
        )

    def evaluate_query(self, query: str) -> InferenceResult | None:
        """Parse natural language counterfactual query and return structured InferenceResult."""
        clean = query.lower().strip()
        matched_prefix = next(
            (p for p in self.policy.counterfactual_prefixes if clean.startswith(p)),
            None,
        )
        if not matched_prefix and " vs " not in clean and " instead of " not in clean:
            return None

        body = clean[len(matched_prefix):].strip() if matched_prefix else clean

        # Contrastive pattern 1: "compare X at T1 vs T2 after H hours" or "X at T1 vs at T2 for H hours"
        if " vs " in body or " instead of " in body:
            sep = " vs " if " vs " in body else " instead of "
            part_a, part_b = body.split(sep, 1)

            # Extract duration (may be at the very end of part_b or query)
            dur = self.resolve_duration_hours(body, default=self.policy.default_duration_hours)

            # Determine subject: extract first noun before 'at' or 'in'
            m_subj = re.search(r"^(?:an?\s+)?([a-z0-9_ ]+?)(?:\s+at|\s+in|\s+kept|\s+left|\s+stored|\s+refrigerated)", part_a)
            subject = m_subj.group(1).strip() if m_subj else "apple slice"

            temp1 = self.resolve_temperature(part_a, default=self.policy.default_counterfactual_temperature_c)
            temp2 = self.resolve_temperature(part_b, default=self.policy.default_baseline_temperature_c)

            label1 = "Refrigerated" if temp1 < 10 else f"{temp1:.0f}°C"
            label2 = "Ambient/Hot" if temp2 > 15 else f"{temp2:.0f}°C"

            comp = self.compare_scenarios(
                subject=subject,
                temp1=temp1,
                dur1=dur,
                temp2=temp2,
                dur2=dur,
                label1=label1,
                label2=label2,
            )
            return InferenceResult(
                query=query,
                status=BeliefStatus.SUPPORTED,
                answer=comp.verbalized,
                confidence=1.0,
                evidence=comp.trace,
                trace=comp.trace,
            )

        # Single hypothetical: "What if an apple slice is kept at 4C for 10 hours?"
        dur = self.resolve_duration_hours(body, default=self.policy.default_duration_hours)
        temp = self.resolve_temperature(body, default=self.policy.default_counterfactual_temperature_c)

        m_subj = re.search(
            r"^(?:an?\s+)?([a-z0-9_ ]+?)(?:\s+is|\s+was|\s+were|\s+at|\s+in|\s+kept|\s+left|\s+decay|\s+oxidation|\s+over|\s+for)",
            body,
        )
        subject = m_subj.group(1).strip() if m_subj else "apple slice"

        sc = self.simulate_scenario(subject, temp, dur)
        rendered = self.policy.single_hypothetical_template.format(
            subject=subject,
            temp=sc.temperature_c,
            dur=sc.duration_hours,
            freshness=sc.freshness,
            oxidation=sc.oxidation,
            condition=sc.condition,
            color=sc.perceived_color,
            rate_factor=sc.rate_factor,
        )

        trace = [
            f"Simulated hypothetical scenario for '{subject}'",
            f"Temperature: {sc.temperature_c}°C, Duration: {sc.duration_hours}h",
            f"CfC outcome: Freshness={sc.freshness:.3f}, Oxidation={sc.oxidation:.3f} -> {sc.condition}, {sc.perceived_color}",
        ]

        return InferenceResult(
            query=query,
            status=BeliefStatus.SUPPORTED,
            answer=rendered,
            confidence=1.0,
            evidence=trace,
            trace=trace,
        )
