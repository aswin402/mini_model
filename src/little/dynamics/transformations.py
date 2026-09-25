"""Physical entity transformations and topological part decomposition for LITTLE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from little.core.models import Entity
from little.dynamics.cfc import ContinuousDynamicsEngine, PhysicalState
from little.dynamics.profiles import DynamicsProfileRegistry
from little.dynamics.transformation_policy import TransformationPolicy
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
        num_pieces: int | None = None,
        *,
        commit_relations: bool = True,
        record_experience: bool = True,
        policy: TransformationPolicy | None = None,
        profile_registry: DynamicsProfileRegistry | None = None,
    ) -> SlicingResult:
        """Perform topological slicing: decompose an object into parts with mass conservation.

        Follows Qualitative Process Theory (QPT):
        1. Queries object material and interior attributes from the knowledge graph in memory.
        2. Inverts topological boundary (interior becomes exposed surface).
        3. Conserves total mass (fractional mass 1/n per piece).
        4. Initializes continuous dynamics ODEs calibrated to the object's specific material properties.
        """
        active_policy = policy or TransformationPolicy.default()
        predicate_policy = active_policy.predicates
        attribute_policy = active_policy.attributes
        defaults = active_policy.defaults
        piece_count = num_pieces or active_policy.default_piece_count
        clean_name = object_name.strip().lower()
        parent_concept = memory.get_concept(clean_name)
        if not parent_concept:
            # If the concept was not yet known, register it first
            parent_concept = memory.create_concept(name=clean_name)

        active_profile_registry = profile_registry or DynamicsProfileRegistry.default()
        profile = active_profile_registry.for_concept(
            parent_concept.category, parent_concept.attributes
        )

        slice_concept_name = (
            f"{clean_name} {defaults['slice_suffix']}"
        )
        slice_concept = memory.get_concept(slice_concept_name)
        if not slice_concept:
            # 1. Retrieve interior color dynamically from attributes or relations in memory
            interior_color = parent_concept.attributes.get(
                attribute_policy["interior_color"]
            )
            if not interior_color:
                rels = memory.get_relations(subject_id=parent_concept.id)
                for r in rels:
                    if r.predicate in predicate_policy["interior_color_relations"]:
                        target_c = memory.get_concept(r.object_id)
                        if target_c:
                            interior_color = target_c.name
                            break
            # Fall back to the configured material profile when unspecified.
            if not interior_color:
                interior_color = parent_concept.attributes.get(
                    attribute_policy["color"], profile.default_interior_color
                )

            # 2. Material profile inspection
            material = parent_concept.attributes.get(
                attribute_policy["material"]
            ) or parent_concept.attributes.get(attribute_policy["interior_material"])

            slice_attrs: dict[str, Any] = {
                attribute_policy["interior_color"]: interior_color,
                attribute_policy["exposed_interior"]: True,
                attribute_policy["exposed_flesh"]: profile.perishable,
                attribute_policy["mass_fraction"]: round(
                    1.0 / max(1, piece_count), 4
                ),
                attribute_policy["dynamics_profile"]: profile.name,
            }
            if material:
                slice_attrs[attribute_policy["interior_material"]] = material

            slice_concept = memory.create_concept(
                name=slice_concept_name,
                category=parent_concept.category or defaults["category"],
                attributes=slice_attrs,
            )

        relations_created: list[str] = []
        if commit_relations:
            # Direct callers retain the historical convenience behavior. Ledger
            # action execution disables this and commits schema effects itself.
            memory.add_relation(
                subject_id=slice_concept.id,
                predicate=predicate_policy["part"],
                object_id=parent_concept.id,
                positive=True,
            )
            relations_created.append(
                f"({slice_concept.name} {predicate_policy['part']} {parent_concept.name})"
            )

            memory.add_relation(
                subject_id=slice_concept.id,
                predicate=predicate_policy["taxonomy"],
                object_id=parent_concept.id,
                positive=True,
            )
            relations_created.append(
                f"({slice_concept.name} {predicate_policy['taxonomy']} {parent_concept.name})"
            )

        # Determine decay tau from the configured material profile.
        custom_tau = parent_concept.attributes.get("decay_tau")
        if custom_tau is not None:
            initial_state = PhysicalState(
                exposed_to_air=True, tau_seconds=float(custom_tau)
            )
        else:
            initial_state = ContinuousDynamicsEngine.create_initial_state(
                exposed_to_air=True,
                profile=profile.name,
                profile_registry=active_profile_registry,
            )

        # Instantiate physical slice entities with mass conservation
        entities_created: list[Entity] = []
        for i in range(1, piece_count + 1):
            entity_name = f"{clean_name}_slice_{i}"
            entity = memory.create_entity(
                name=entity_name,
                concept_id=slice_concept.id,
                properties={
                    attribute_policy["piece_number"]: i,
                    attribute_policy["of_total"]: piece_count,
                    attribute_policy["mass_fraction"]: round(
                        1.0 / max(1, piece_count), 4
                    ),
                    attribute_policy["physical_state"]: initial_state.to_dict(),
                },
            )
            entities_created.append(entity)

        if record_experience:
            # Direct callers retain the historical episodic record. Ledger
            # action execution records one provenance-linked experience after
            # its effect evidence has been decided.
            memory.add_experience(
                input_text=f"Sliced {clean_name} into {piece_count} pieces.",
                extracted_triples=[
                    {
                        "subject": slice_concept.name,
                        "predicate": predicate_policy["part"],
                        "object": parent_concept.name,
                    },
                    {
                        "subject": slice_concept.name,
                        "predicate": predicate_policy["taxonomy"],
                        "object": parent_concept.name,
                    },
                ],
                source=defaults["experience_source"],
            )

        return SlicingResult(
            parent_concept=parent_concept.name,
            slice_concept=slice_concept.name,
            num_pieces=piece_count,
            entities_created=entities_created,
            relations_created=relations_created,
            initial_state=initial_state,
            message=(
                f"Successfully sliced {clean_name} into {piece_count} pieces. "
                f"Established '{predicate_policy['part']}' and '{predicate_policy['taxonomy']}' relations with parent '{parent_concept.name}'. "
                f"Initial state: fresh, interior exposed to air (tau = {initial_state.tau_seconds:.0f}s)."
            ),
        )
