"""Physical entity transformations and topological part decomposition for LITTLE.

Implements the user's core vision: When an object (e.g. apple) is cut into pieces,
the cognitive system automatically decomposes the entity, establishes part_of relations,
inherits interior properties, and initializes exposed physical dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from little.core.models import Entity
from little.dynamics.cfc import ContinuousDynamicsEngine, PhysicalState
from little.memory.store import MemoryStore


@dataclass
class SlicingResult:
    """Result of slicing an entity or concept into parts."""

    parent_concept: str
    slice_concept: str
    num_pieces: int
    entities_created: list[Entity]
    relations_created: list[str]
    initial_state: PhysicalState
    message: str


class TransformationEngine:
    """Performs topological modifications and actions on concepts and physical entities."""

    @classmethod
    def slice_object(
        cls,
        memory: MemoryStore,
        object_name: str,
        num_pieces: int = 4,
    ) -> SlicingResult:
        """Perform topological slicing: decompose an object into parts with mass conservation.

        Follows Qualitative Process Theory (QPT):
        1. Queries object material and interior attributes from the knowledge graph in memory.
        2. Inverts topological boundary (interior becomes exposed surface).
        3. Conserves total mass (fractional mass 1/n per piece).
        4. Initializes continuous dynamics ODEs calibrated to the object's specific material properties.
        """
        clean_name = object_name.strip().lower()
        parent_concept = memory.get_concept(clean_name)
        if not parent_concept:
            # If the concept was not yet known, register it first
            parent_concept = memory.create_concept(name=clean_name)

        slice_concept_name = f"{clean_name} slice"
        slice_concept = memory.get_concept(slice_concept_name)
        if not slice_concept:
            # 1. Retrieve interior color dynamically from attributes or relations in memory
            interior_color = parent_concept.attributes.get("interior_color")
            if not interior_color:
                rels = memory.get_relations(subject_id=parent_concept.id)
                for r in rels:
                    if r.predicate in ("interior_color", "has_interior", "color"):
                        target_c = memory.get_concept(r.object_id)
                        if target_c:
                            interior_color = target_c.name
                            break
            # Fallback default if unspecified
            if not interior_color:
                interior_color = parent_concept.attributes.get("color", "white")

            # 2. Material & Perishability inspection
            material = parent_concept.attributes.get(
                "material"
            ) or parent_concept.attributes.get("interior_material")
            category = parent_concept.category or "object"
            is_organic = (
                category
                in ("fruit", "vegetable", "food", "plant", "organism", "produce")
                or parent_concept.attributes.get("is_organic", False)
                or parent_concept.attributes.get("perishable", False)
                or "fruit" in category
                or "food" in category
            )

            slice_attrs: dict[str, Any] = {
                "interior_color": interior_color,
                "exposed_interior": True,
                # Retain exposed_flesh alias for organic entities for backward compatibility
                "exposed_flesh": is_organic or category in ("fruit", "food", "object"),
                "part_of": clean_name,
                "mass_fraction": round(1.0 / max(1, num_pieces), 4),
            }
            if material:
                slice_attrs["interior_material"] = material

            slice_concept = memory.create_concept(
                name=slice_concept_name,
                category=parent_concept.category or "partitioned_object",
                attributes=slice_attrs,
            )

        # Establish explicit part_of relation in semantic knowledge graph
        memory.add_relation(
            subject_id=slice_concept.id,
            predicate="part_of",
            object_id=parent_concept.id,
            positive=True,
        )
        relations_created = [f"({slice_concept.name} part_of {parent_concept.name})"]

        # Also add taxonomic relation: slice is_a parent concept
        memory.add_relation(
            subject_id=slice_concept.id,
            predicate="is_a",
            object_id=parent_concept.id,
            positive=True,
        )
        relations_created.append(f"({slice_concept.name} is_a {parent_concept.name})")

        # Determine decay tau based on material and perishability from concept attributes
        custom_tau = parent_concept.attributes.get("decay_tau")
        if custom_tau is not None:
            initial_state = PhysicalState(
                exposed_to_air=True, tau_seconds=float(custom_tau)
            )
        else:
            category = parent_concept.category or ""
            is_perishable = (
                category
                in ("fruit", "vegetable", "food", "plant", "organism", "produce")
                or parent_concept.attributes.get("is_organic", False)
                or parent_concept.attributes.get("perishable", False)
                or "fruit" in category
                or "food" in category
                or parent_concept.category is None
            )
            if is_perishable:
                initial_state = ContinuousDynamicsEngine.create_initial_state(
                    exposed_to_air=True
                )
            else:
                # Inorganic/inert materials (e.g. metal, stone, wood) do not rapidly decompose
                initial_state = PhysicalState(
                    exposed_to_air=True, tau_seconds=86400.0 * 3650.0
                )

        # Instantiate physical slice entities with mass conservation
        entities_created: list[Entity] = []
        for i in range(1, num_pieces + 1):
            entity_name = f"{clean_name}_slice_{i}"
            entity = memory.create_entity(
                name=entity_name,
                concept_id=slice_concept.id,
                properties={
                    "piece_number": i,
                    "of_total": num_pieces,
                    "mass_fraction": round(1.0 / max(1, num_pieces), 4),
                    "physical_state": initial_state.to_dict(),
                },
            )
            entities_created.append(entity)

        # Log episodic experience of the slicing event
        memory.add_experience(
            input_text=f"Sliced {clean_name} into {num_pieces} pieces.",
            extracted_triples=[
                {
                    "subject": slice_concept.name,
                    "predicate": "part_of",
                    "object": parent_concept.name,
                },
                {
                    "subject": slice_concept.name,
                    "predicate": "is_a",
                    "object": parent_concept.name,
                },
            ],
            source="action_execution",
        )

        return SlicingResult(
            parent_concept=parent_concept.name,
            slice_concept=slice_concept.name,
            num_pieces=num_pieces,
            entities_created=entities_created,
            relations_created=relations_created,
            initial_state=initial_state,
            message=(
                f"Successfully sliced {clean_name} into {num_pieces} pieces. "
                f"Established 'part_of' and 'is_a' relations with parent '{parent_concept.name}'. "
                f"Initial state: fresh, interior exposed to air (tau = {initial_state.tau_seconds:.0f}s)."
            ),
        )
