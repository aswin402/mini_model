"""Autonomous 360-Degree Spider-Web Growth & Active Curiosity Engine.

Implements Cobweb Category Utility clustering for upward concept induction,
variance-driven downward specialization, and entropy-targeted curiosity.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from little.core.concept_knot import ConceptKnot


class AutonomousSpiderWebEngine:
    """Engine driving autonomous 360-degree self-expansion of the Concept Knot manifold."""

    def compute_category_utility(self, knots: List[ConceptKnot]) -> float:
        """Computes Cobweb Category Utility across mereological and procedural attributes."""
        if not knots:
            return 0.0
        n = len(knots)
        all_skills: Dict[str, int] = {}
        for k in knots:
            for s in k.procedural_skills:
                all_skills[s] = all_skills.get(s, 0) + 1

        cu = sum((count / n) ** 2 for count in all_skills.values())
        return cu

    def induce_hypernym_cobweb(self, knots: List[ConceptKnot]) -> Optional[str]:
        """Clusters knots with overlapping traits to synthesize higher-order hypernyms."""
        if len(knots) < 2:
            return None
        cu = self.compute_category_utility(knots)
        if cu >= 1.0:
            shared_hyper = set(knots[0].taxonomy_hypernyms)
            for k in knots[1:]:
                shared_hyper &= set(k.taxonomy_hypernyms)
            prefix = list(shared_hyper)[0] if shared_hyper else "ENTITY"
            return f"POME_{prefix.upper()}_CLUSTER"
        return None

    def identify_epistemic_gaps(self, knot: ConceptKnot) -> List[str]:
        """Scans the 6 radial axes of a Concept Knot to identify missing spokes."""
        inquiries = []

        if not knot.procedural_skills:
            inquiries.append(
                f"Procedural Axis Gap: What actions or skills can be performed on {knot.concept_id}?"
            )
        if len(knot.mereology_parts) < 2:
            inquiries.append(
                f"Mereological Axis Gap: What are the internal and boundary parts of {knot.concept_id}?"
            )
        if not knot.invariant_disjoints:
            inquiries.append(
                f"Invariant Axis Gap: What categories are mutually exclusive with {knot.concept_id}?"
            )
        if (
            knot.dynamics_state.freshness == 1.0
            and knot.dynamics_state.oxidation == 0.0
            and not knot.episodic_instances
        ):
            inquiries.append(
                f"Dynamical Axis Gap: How does {knot.concept_id} transform or decay over time?"
            )

        return inquiries
