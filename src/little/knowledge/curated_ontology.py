"""Data-driven large-scale commonsense ontology generator."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths
from little.knowledge.ontology_generation_policy import OntologyGenerationPolicy

Triple = tuple[str, str, str, float, bool]


def _parse_triple(record: Any, path: Path, index: int) -> Triple:
    if not isinstance(record, list) or len(record) != 5:
        raise ValueError(f"{path} seed triple {index} must be a five-item array")
    subject, predicate, object_, weight, positive = record
    if not all(isinstance(value, str) and value.strip() for value in record[:3]):
        raise ValueError(f"{path} seed triple {index} has invalid text fields")
    if not isinstance(positive, bool):
        raise TypeError(f"{path} seed triple {index} has a non-boolean flag")
    return (
        subject.strip().lower(),
        predicate.strip().lower(),
        object_.strip().lower(),
        float(weight),
        positive,
    )


def _load_configuration(
    path: str | Path,
) -> tuple[list[Triple], dict[str, Any], dict[str, Any]]:
    ontology_path = Path(path)
    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{ontology_path} must contain a JSON object")
    if payload.get("format") != "little.knowledge.extended.v1":
        raise ValueError(f"{ontology_path} has an unsupported ontology format")

    raw_seed = payload.get("seed_triples")
    if not isinstance(raw_seed, list):
        raise TypeError(f"{ontology_path} field 'seed_triples' must be a list")
    expansion = payload.get("expansion")
    if not isinstance(expansion, dict):
        raise TypeError(f"{ontology_path} field 'expansion' must be an object")

    required_lists = ("base_entities", "materials", "locations")
    for key in required_lists:
        values = expansion.get(key)
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise TypeError(f"{ontology_path} expansion field {key!r} is invalid")
    compatible = expansion.get("compatible_materials", {})
    if not isinstance(compatible, dict) or any(
        not isinstance(values, list) or not all(isinstance(value, str) for value in values)
        for values in compatible.values()
    ):
        raise TypeError(f"{ontology_path} expansion field 'compatible_materials' is invalid")

    generation = payload.get("generation", {})
    if not isinstance(generation, dict):
        raise TypeError(f"{ontology_path} field 'generation' must be an object")

    return (
        [_parse_triple(record, ontology_path, index) for index, record in enumerate(raw_seed)],
        expansion,
        generation,
    )


def generate_extended_commonsense_triples(
    target_count: int = 100_000,
    pack_path: str | Path | None = None,
) -> Iterator[Triple]:
    """Stream verified ontology triples from an external seed/config pack."""
    seed_triples, expansion, generation_overrides = _load_configuration(
        pack_path or RuntimePaths.default().ontology_pack
    )
    generation = OntologyGenerationPolicy.from_mapping(
        generation_overrides,
        base=OntologyGenerationPolicy.default(),
    )
    yielded = 0

    def emit(
        subject: str,
        predicate: str,
        object_: str,
        weight: float = 2.0,
        positive: bool = True,
    ) -> Iterator[Triple]:
        nonlocal yielded
        if yielded < target_count:
            yielded += 1
            yield (
                subject.strip().lower(),
                predicate.strip().lower(),
                object_.strip().lower(),
                weight,
                positive,
            )

    for triple in seed_triples:
        yield from emit(*triple)

    base_entities = expansion["base_entities"]
    materials = expansion["materials"]
    locations = expansion["locations"]
    compatible_materials = expansion["compatible_materials"]
    component = str(expansion.get("component", "component"))
    item_prefix = str(expansion.get("item_prefix", "item_"))

    for entity in base_entities:
        yield from emit(
            entity,
            generation.taxonomy_predicate,
            generation.root_category,
        )
        for material in compatible_materials.get(entity, []):
            instance = f"{material} {entity}"
            yield from emit(instance, generation.taxonomy_predicate, entity)
            yield from emit(instance, generation.composition_predicate, material)

    counter = 1
    while yielded < target_count:
        item_id = f"{item_prefix}{counter}"
        category = base_entities[counter % len(base_entities)]
        material = materials[counter % len(materials)]
        location = locations[counter % len(locations)]
        yield from emit(item_id, generation.taxonomy_predicate, category)
        yield from emit(item_id, generation.composition_predicate, material)
        yield from emit(item_id, generation.location_predicate, location)
        yield from emit(item_id, generation.component_predicate, component)
        counter += 1
