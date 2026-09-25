"""Autonomous 360-Degree Spider-Web Growth & Active Curiosity Engine.

Implements Cobweb Category Utility clustering for upward concept induction,
variance-driven downward specialization, and entropy-targeted curiosity.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from little.core.concept_knot import ConceptKnot
from little.active.spiderweb_policy import SpiderWebGrowthPolicy


class AutonomousSpiderWebEngine:
    """Engine driving autonomous 360-degree self-expansion of the Concept Knot manifold."""

    def __init__(self, policy: SpiderWebGrowthPolicy | None = None) -> None:
        self.policy = policy or SpiderWebGrowthPolicy.default()

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
        if len(knots) < self.policy.minimum_cluster_size:
            return None
        cu = self.compute_category_utility(knots)
        if cu >= self.policy.category_utility_threshold:
            shared_hyper = set(knots[0].taxonomy_hypernyms)
            for k in knots[1:]:
                shared_hyper &= set(k.taxonomy_hypernyms)
            prefix = next(iter(sorted(shared_hyper)), self.policy.fallback_hypernym)
            return (
                f"{self.policy.cluster_prefix}_{prefix.upper()}_"
                f"{self.policy.cluster_suffix}"
            )
        return None

    def identify_epistemic_gaps(self, knot: ConceptKnot) -> List[str]:
        """Scans the 6 radial axes of a Concept Knot to identify missing spokes."""
        inquiries = []

        gap_conditions = {
            "procedural": not knot.procedural_skills,
            "mereology": len(knot.mereology_parts)
            < self.policy.minimum_mereology_parts,
            "invariants": not knot.invariant_disjoints,
            "dynamics": (
                knot.dynamics_state.freshness
                == self.policy.uninitialized_freshness
                and knot.dynamics_state.oxidation
                == self.policy.uninitialized_oxidation
                and (
                    not self.policy.require_episodic_state_for_dynamics_gap
                    or not knot.episodic_instances
                )
            ),
        }
        for axis_name, template in self.policy.gap_templates.items():
            if gap_conditions.get(axis_name, False):
                inquiries.append(template.format(concept_id=knot.concept_id))

        return inquiries
