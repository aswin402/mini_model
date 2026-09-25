"""360-Degree Radial Concept Knot Data Structures.

Formalizes K = <V_tax, M_mereo, D_dyn, I_ax, P_proc, E_epis> across 6 orthogonal axes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from little.core.concept_knot_policy import ConceptKnotPolicy
from little.core.runtime_paths import RuntimePaths
from little.knowledge.registry import SchemaRegistry


def _knot_default(name: str) -> float:
    return ConceptKnotPolicy.default().defaults[name]


@dataclass
class ContinuousPhysicalState:
    freshness: float = field(default_factory=lambda: _knot_default("freshness"))
    oxidation: float = field(default_factory=lambda: _knot_default("oxidation"))
    moisture: float = field(default_factory=lambda: _knot_default("moisture"))
    temperature: float = field(default_factory=lambda: _knot_default("temperature"))


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
    invariant_mass: float = field(
        default_factory=lambda: _knot_default("constructor_mass")
    )

    # 225° Procedural Skills (available discrete action jumps)
    procedural_skills: List[str] = field(default_factory=list)

    # 0° Grounded Episodic Instances (NARS evidence traces)
    episodic_instances: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create_apple_exemplar(cls) -> ConceptKnot:
        """Load the compatibility Apple fixture from the data directory."""
        return cls.load_fixture("apple", RuntimePaths.default().concept_fixture_directory)

    @classmethod
    def from_record(
        cls,
        record: Dict[str, Any],
        policy: ConceptKnotPolicy | None = None,
    ) -> ConceptKnot:
        """Construct a knot from a schema-compatible data record."""
        policy = policy or ConceptKnotPolicy.default()
        state = record.get("dynamics_state", {})
        return cls(
            concept_id=str(record["concept_id"]),
            taxonomy_hypernyms=list(record.get("taxonomy_hypernyms", [])),
            taxonomy_hyponyms=list(record.get("taxonomy_hyponyms", [])),
            mereology_parts=dict(record.get("mereology_parts", {})),
            dynamics_state=ContinuousPhysicalState(
                freshness=float(state.get("freshness", policy.defaults["freshness"])),
                oxidation=float(state.get("oxidation", policy.defaults["oxidation"])),
                moisture=float(state.get("moisture", policy.defaults["moisture"])),
                temperature=float(state.get("temperature", policy.defaults["temperature"])),
            ),
            invariant_disjoints=list(record.get("invariant_disjoints", [])),
            invariant_mass=float(record.get("invariant_mass", policy.defaults["record_mass"])),
            procedural_skills=list(record.get("procedural_skills", [])),
            episodic_instances=list(record.get("episodic_instances", [])),
        )

    @classmethod
    def load_fixture(cls, name: str, directory: Path) -> ConceptKnot:
        """Load any concept knot fixture without adding a new Python branch."""
        path = Path(directory) / f"{name.strip().lower()}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get("concept_knot", payload)
        if not isinstance(record, dict):
            raise TypeError(f"{path} must contain an object under 'concept_knot'")
        return cls.from_record(record)

    @classmethod
    def from_memory(
        cls,
        store: Any,
        concept_id_or_name: str,
        registry: SchemaRegistry | None = None,
        policy: ConceptKnotPolicy | None = None,
    ) -> Optional[ConceptKnot]:
        """Dynamically hydrates a 360° Concept Knot directly from the persistent knowledge graph."""
        store_registry = getattr(getattr(store, "ledger", None), "registry", None)
        registry = registry or store_registry or SchemaRegistry.default()
        policy = policy or ConceptKnotPolicy.default()
        c = store.get_concept(concept_id_or_name)
        variants = {
            concept_id_or_name.strip(),
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
        taxonomy_predicates = registry.predicates(axis=policy.taxonomy_axis)
        for v in sorted(variants):
            for r in store.get_relations(subject_id=v):
                if (
                    r.predicate not in taxonomy_predicates
                    or r.weight_positive <= r.weight_negative
                ):
                    continue
                name = _resolve_name(r.object_id)
                if name not in hypernyms:
                    hypernyms.append(name)
            for r in store.get_relations(object_id=v):
                if (
                    r.predicate not in taxonomy_predicates
                    or r.weight_positive <= r.weight_negative
                ):
                    continue
                name = _resolve_name(r.subject_id)
                if name not in hyponyms:
                    hyponyms.append(name)

        # 2. 135° Mereological Axis
        mereology: Dict[str, str] = {}
        mereology_roles = policy.mereology_roles
        for v in sorted(variants):
            for r in store.get_relations(subject_id=v):
                p = r.predicate.lower()
                if p in mereology_roles and r.weight_positive > r.weight_negative:
                    mereology[_resolve_name(r.object_id)] = mereology_roles[p]
            for r in store.get_relations(object_id=v):
                p = r.predicate.lower()
                if p in mereology_roles and r.weight_positive > r.weight_negative:
                    mereology[_resolve_name(r.subject_id)] = mereology_roles[p]

        # 3. 45° Continuous Dynamical Axis
        freshness = float(attrs.get("freshness", policy.defaults["freshness"]))
        oxidation = float(attrs.get("oxidation", policy.defaults["oxidation"]))
        moisture = float(attrs.get("moisture", policy.defaults["moisture"]))
        temp = float(attrs.get("temperature", policy.defaults["temperature"]))
        dyn_state = ContinuousPhysicalState(
            freshness=freshness,
            oxidation=oxidation,
            moisture=moisture,
            temperature=temp,
        )

        # 4. 180° Invariant Axioms & Mutex
        disjoints: List[str] = []
        disjoint_predicates = registry.predicates(disjoint=True)
        for v in sorted(variants):
            for r in store.get_relations(subject_id=v):
                if (
                    r.predicate not in disjoint_predicates
                    or r.weight_positive <= r.weight_negative
                ):
                    continue
                name = _resolve_name(r.object_id)
                if name not in disjoints:
                    disjoints.append(name)
            for r in store.get_relations(object_id=v):
                if (
                    r.predicate not in disjoint_predicates
                    or r.weight_positive <= r.weight_negative
                ):
                    continue
                name = _resolve_name(r.subject_id)
                if name not in disjoints:
                    disjoints.append(name)

        # 5. 225° Procedural Skills
        skills: List[str] = []
        procedural_predicates = registry.predicates(axis=policy.procedural_axis)
        for v in sorted(variants):
            for r in store.get_relations(subject_id=v):
                if (
                    r.predicate in procedural_predicates
                    and r.weight_positive > r.weight_negative
                ):
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
            invariant_mass=float(attrs.get("mass", policy.defaults["mass"])),
            procedural_skills=skills,
            episodic_instances=[],
        )
