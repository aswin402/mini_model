"""Continuous Counterfactual Reasoning Engine — Simulating hypothetical worlds via CfC ODEs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from little.core.models import BeliefStatus, InferenceResult
from little.dynamics.cfc import ContinuousDynamicsEngine
from little.dynamics.counterfactual_policy import CounterfactualPolicy

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


def _parse_leading_float(s: str) -> tuple[float, str] | None:
    """Parse a leading signed float and return (value, remaining_suffix)."""
    s = s.strip()
    if not s:
        return None
    i = 0
    if s[0] == "-":
        i = 1
    start_digits = i
    has_dot = False
    while i < len(s) and (s[i].isdigit() or (s[i] == "." and not has_dot)):
        if s[i] == ".":
            has_dot = True
        i += 1
    if i == start_digits or s[start_digits:i] == ".":
        return None
    return float(s[:i]), s[i:].strip()


def _extract_temperature_value(clean: str) -> tuple[float, str] | None:
    """Scan clean lowercase text for '<num> (°, deg, degrees, celsius, fahrenheit, c, f)' without regex."""
    n = len(clean)
    i = 0
    while i < n:
        if clean[i].isdigit() or (clean[i] == "-" and i + 1 < n and clean[i + 1].isdigit()):
            parsed = _parse_leading_float(clean[i:])
            if parsed is not None:
                val, rest = parsed
                if rest.startswith("°"):
                    rest_u = rest[1:].lstrip()
                    if rest_u.startswith("c"):
                        return val, "c"
                    if rest_u.startswith("f"):
                        return val, "f"
                for prefix, unit_code in (
                    ("degrees c", "c"),
                    ("degrees f", "f"),
                    ("degree c", "c"),
                    ("degree f", "f"),
                    ("deg c", "c"),
                    ("deg f", "f"),
                    ("degrees", "c"),
                    ("degree", "c"),
                    ("celsius", "c"),
                    ("fahrenheit", "f"),
                    ("deg", "c"),
                ):
                    if rest.startswith(prefix):
                        after = rest[len(prefix) :]
                        if not after or not after[0].isalnum():
                            return val, unit_code
                if rest.startswith("c") and (len(rest) == 1 or not rest[1].isalnum()):
                    return val, "c"
                if rest.startswith("f") and (len(rest) == 1 or not rest[1].isalnum()):
                    return val, "f"
        i += 1
    return None


def _extract_unit_number(clean: str, units: tuple[str, ...]) -> float | None:
    """Extract float immediately preceding one of the given unit tokens without regex."""
    n = len(clean)
    i = 0
    while i < n:
        if clean[i].isdigit():
            parsed = _parse_leading_float(clean[i:])
            if parsed is not None:
                val, rest = parsed
                for u in units:
                    if rest.startswith(u):
                        after = rest[len(u) :]
                        if not after or not after[0].isalnum():
                            return val
        i += 1
    return None


def _extract_subject_before_markers(text: str, markers: tuple[str, ...]) -> str | None:
    """Extract subject noun phrase before any marker word without regex."""
    s = text.strip()
    for art in ("an ", "a "):
        if s.startswith(art):
            s = s[len(art) :].strip()
            break
    words = s.split()
    subj_words: list[str] = []
    for w in words:
        if w in markers:
            if subj_words:
                return " ".join(subj_words)
            return None
        subj_words.append(w)
    return None


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
        extracted = _extract_temperature_value(clean)
        if extracted is not None:
            val, unit = extracted
            if unit in ("f", "fahrenheit"):
                return (val - 32.0) * 5.0 / 9.0
            return val

        for alias, temp in self.policy.temperature_aliases.items():
            if alias in clean:
                return temp

        bare = _parse_leading_float(clean)
        if bare is not None and bare[1] == "":
            return bare[0]
        return default

    def resolve_duration_hours(self, text: str, default: float = 2.0) -> float:
        """Extract duration in hours from natural language query."""
        clean = text.lower().strip()
        val_day = _extract_unit_number(clean, ("days", "day", "d"))
        if val_day is not None:
            return val_day * 24.0

        val_hr = _extract_unit_number(clean, ("hours", "hour", "hrs", "hr", "h"))
        if val_hr is not None:
            return val_hr

        val_min = _extract_unit_number(clean, ("minutes", "minute", "mins", "min", "m"))
        if val_min is not None:
            return val_min / 60.0

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
            extracted_subj = _extract_subject_before_markers(
                part_a, ("at", "in", "kept", "left", "stored", "refrigerated")
            )
            subject = extracted_subj if extracted_subj else "apple slice"

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
        default_temp = (
            self.policy.default_baseline_temperature_c
            if matched_prefix and matched_prefix.startswith("simulate")
            else self.policy.default_counterfactual_temperature_c
        )
        temp = self.resolve_temperature(body, default=default_temp)

        extracted_subj = _extract_subject_before_markers(
            body,
            (
                "is",
                "was",
                "were",
                "at",
                "in",
                "kept",
                "left",
                "decay",
                "oxidation",
                "over",
                "for",
            ),
        )
        subject = extracted_subj if extracted_subj else "apple slice"

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
