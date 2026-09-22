"""360-Degree Radial Concept Knot Data Structures.

Formalizes K = <V_tax, M_mereo, D_dyn, I_ax, P_proc, E_epis> across 6 orthogonal axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContinuousPhysicalState:
    freshness: float = 1.0  # [0.0 = rotten, 1.0 = freshly picked]
    oxidation: float = 0.0  # [0.0 = fresh, 1.0 = fully oxidized / browned]
    moisture: float = 0.85  # water fraction in [0.0, 1.0]
    temperature: float = 22.0  # degrees Celsius


@dataclass
class ConceptKnot:
    """Radial 360-degree Concept Knot holding 6 orthogonal dimensions of reality."""

    concept_id: str

    # 90° Taxonomic Axis (is_a hierarchy)
    taxonomy_hypernyms: List[str] = field(default_factory=list)
    taxonomy_hyponyms: List[str] = field(default_factory=list)

    # 135° Mereological Axis (RCC-8 part-whole topology)
    mereology_parts: Dict[str, str] = field(default_factory=dict)

    # 45° Continuous Dynamical Axis (Neural ODE state)
    dynamics_state: ContinuousPhysicalState = field(
        default_factory=ContinuousPhysicalState
    )

    # 180° Invariant Axioms & Mutex (QPT mass conservation & disjoint constraints)
    invariant_disjoints: List[str] = field(default_factory=list)
    invariant_mass: float = 180.0  # grams

    # 225° Procedural Skills (available discrete action jumps)
    procedural_skills: List[str] = field(default_factory=list)

    # 0° Grounded Episodic Instances (NARS evidence traces)
    episodic_instances: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_apple_exemplar(cls) -> ConceptKnot:
        """Instantiates the canonical Apple Concept Knot from coreidea.md."""
        return cls(
            concept_id="APPLE",
            taxonomy_hypernyms=[
                "PomeFruit",
                "Fruit",
                "PlantEntity",
                "PhysicalObject",
            ],
            taxonomy_hyponyms=["GalaApple", "GrannySmith", "Honeycrisp"],
            mereology_parts={
                "skin": "external_boundary",
                "pulp": "non_tangential_proper_part",
                "core": "tangential_proper_part",
                "seeds": "interior_proper_part",
            },
            dynamics_state=ContinuousPhysicalState(
                freshness=1.0, oxidation=0.0, moisture=0.86, temperature=20.0
            ),
            invariant_disjoints=["Animal", "Vehicle", "Mineral"],
            invariant_mass=180.0,
            procedural_skills=["slice", "peel", "juice", "dehydrate"],
            episodic_instances=[
                {"obs_id": "OBS_001", "color": "red", "confidence": 0.95}
            ],
        )
