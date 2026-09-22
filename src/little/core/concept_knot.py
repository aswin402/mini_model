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

    @classmethod
    def from_memory(
        cls, store: Any, concept_id_or_name: str
    ) -> Optional[ConceptKnot]:
        """Dynamically hydrates a 360° Concept Knot directly from the persistent knowledge graph."""
        c = store.get_concept(concept_id_or_name)
        variants = {
            concept_id_or_name,
            concept_id_or_name.lower(),
            concept_id_or_name.upper(),
        }
        if c:
            variants.add(c.id)
            variants.add(c.name)
            canonical_name = c.name
            attrs = c.attributes or {}
        else:
            canonical_name = concept_id_or_name.strip()
            attrs = {}

        def _resolve_name(raw_id: str) -> str:
            target = store.get_concept(raw_id)
            if target:
                return target.name
            return raw_id

        # 1. 90° Taxonomic Axis (hypernyms and hyponyms)
        hypernyms: List[str] = []
        hyponyms: List[str] = []
        for v in variants:
            for r in store.get_relations(subject_id=v, predicate="is_a"):
                name = _resolve_name(r.object_id)
                if name not in hypernyms:
                    hypernyms.append(name)
            for r in store.get_relations(object_id=v, predicate="is_a"):
                name = _resolve_name(r.subject_id)
                if name not in hyponyms:
                    hyponyms.append(name)

        # 2. 135° Mereological Axis (part_of, has_part, made_of)
        mereology: Dict[str, str] = {}
        for v in variants:
            for r in store.get_relations(subject_id=v):
                p = r.predicate.lower()
                if p in ("has_part", "has"):
                    mereology[_resolve_name(r.object_id)] = "proper_part"
                elif p == "made_of":
                    mereology[_resolve_name(r.object_id)] = "material_substance"
            for r in store.get_relations(object_id=v):
                p = r.predicate.lower()
                if p == "part_of":
                    mereology[_resolve_name(r.subject_id)] = "proper_part"

        # 3. 45° Continuous Dynamical Axis
        freshness = float(attrs.get("freshness", 1.0))
        oxidation = float(attrs.get("oxidation", 0.0))
        moisture = float(attrs.get("moisture", 0.85))
        temp = float(attrs.get("temperature", 22.0))
        dyn_state = ContinuousPhysicalState(
            freshness=freshness,
            oxidation=oxidation,
            moisture=moisture,
            temperature=temp,
        )

        # 4. 180° Invariant Axioms & Mutex
        disjoints: List[str] = []
        for v in variants:
            for r in store.get_relations(
                subject_id=v, predicate="disjoint_with"
            ):
                name = _resolve_name(r.object_id)
                if name not in disjoints:
                    disjoints.append(name)
            for r in store.get_relations(
                object_id=v, predicate="disjoint_with"
            ):
                name = _resolve_name(r.subject_id)
                if name not in disjoints:
                    disjoints.append(name)

        # 5. 225° Procedural Skills (can, capable_of, used_for)
        skills: List[str] = []
        for v in variants:
            for r in store.get_relations(subject_id=v):
                p = r.predicate.lower()
                if p in ("can", "capable_of", "used_for"):
                    name = _resolve_name(r.object_id)
                    if name not in skills:
                        skills.append(name)

        # If no relations and no concept exists, return None
        if (
            not hypernyms
            and not hyponyms
            and not mereology
            and not disjoints
            and not skills
            and not c
        ):
            return None

        return cls(
            concept_id=canonical_name.upper(),
            taxonomy_hypernyms=hypernyms,
            taxonomy_hyponyms=hyponyms,
            mereology_parts=mereology,
            dynamics_state=dyn_state,
            invariant_disjoints=disjoints,
            invariant_mass=float(attrs.get("mass", 150.0)),
            procedural_skills=skills,
            episodic_instances=[],
        )
