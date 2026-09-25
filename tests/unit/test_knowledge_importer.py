"""Unit tests for ConceptNet and WordNet Knowledge Ingestion Pipeline in LITTLE."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from little.core.models import BeliefStatus
from little.knowledge.bundle import get_commonsense_triples
from little.knowledge.curated_ontology import generate_extended_commonsense_triples
from little.knowledge.importer import ConceptNetNormalizer, KnowledgeImporter
from little.knowledge.import_policy import KnowledgeImportPolicy
from little.knowledge.ontology_generation_policy import OntologyGenerationPolicy
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_conceptnet_normalizer_has_no_process_global_import_policy():
    assert "POLICY" not in ConceptNetNormalizer.__dict__


def test_commonsense_bundle_reads_an_injected_data_pack(tmp_path: Path):
    pack_path = tmp_path / "commonsense.json"
    pack_path.write_text(
        json.dumps(
            {
                "format": "little.knowledge.triples.v1",
                "triples": [
                    {
                        "subject": "quartz",
                        "predicate": "is_a",
                        "object": "mineral",
                        "weight": 3.0,
                        "positive": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert get_commonsense_triples(pack_path) == [
        ("quartz", "is_a", "mineral", 3.0, True)
    ]


def test_extended_ontology_reads_seed_and_expansion_config(tmp_path: Path):
    pack_path = tmp_path / "extended.json"
    pack_path.write_text(
        json.dumps(
            {
                "format": "little.knowledge.extended.v1",
                "seed_triples": [["quartz", "is_a", "mineral", 3.0, True]],
                "expansion": {
                    "base_entities": ["sample"],
                    "materials": ["silica"],
                    "locations": ["lab"],
                    "compatible_materials": {},
                    "component": "part",
                    "item_prefix": "item_",
                },
            }
        ),
        encoding="utf-8",
    )

    triples = list(
        generate_extended_commonsense_triples(
            target_count=6, pack_path=pack_path
        )
    )

    assert triples[:2] == [
        ("quartz", "is_a", "mineral", 3.0, True),
        ("sample", "is_a", "artifact", 2.0, True),
    ]
    assert triples[-1][0] == "item_1"


def test_extended_ontology_reads_generation_relation_config(tmp_path: Path):
    pack_path = tmp_path / "extended.json"
    pack_path.write_text(
        json.dumps(
            {
                "format": "little.knowledge.extended.v1",
                "seed_triples": [],
                "generation": {
                    "root_category": "generated_object",
                    "taxonomy_predicate": "subclass_of",
                    "composition_predicate": "composed_of",
                    "location_predicate": "situated_in",
                    "component_predicate": "contains_part",
                },
                "expansion": {
                    "base_entities": ["sample"],
                    "materials": ["silica"],
                    "locations": ["lab"],
                    "compatible_materials": {},
                    "component": "part",
                    "item_prefix": "item_",
                },
            }
        ),
        encoding="utf-8",
    )

    triples = list(generate_extended_commonsense_triples(target_count=6, pack_path=pack_path))

    assert triples[0:2] == [
        ("sample", "subclass_of", "generated_object", 2.0, True),
        ("item_1", "subclass_of", "sample", 2.0, True),
    ]


def test_conceptnet_normalizer_uris():
    """Verify robust sanitization of ConceptNet 5 URIs."""
    # Standard English nouns
    assert ConceptNetNormalizer.clean_concept_uri("/c/en/dog") == "dog"
    assert ConceptNetNormalizer.clean_concept_uri("/c/en/dog/n") == "dog"
    assert (
        ConceptNetNormalizer.clean_concept_uri("/c/en/golden_retriever")
        == "golden retriever"
    )
    assert (
        ConceptNetNormalizer.clean_concept_uri("/c/en/apple_pie/n/wn/food")
        == "apple pie"
    )

    # Discard non-English
    assert ConceptNetNormalizer.clean_concept_uri("/c/fr/chien") is None
    assert ConceptNetNormalizer.clean_concept_uri("/c/es/perro") is None
    assert ConceptNetNormalizer.clean_concept_uri("/c/de/hund") is None

    # Discard stopwords and vague concepts
    assert ConceptNetNormalizer.clean_concept_uri("/c/en/something") is None
    assert ConceptNetNormalizer.clean_concept_uri("/c/en/it") is None
    assert ConceptNetNormalizer.clean_concept_uri("/c/en/anything") is None

    # Discard overlong phrases
    assert (
        ConceptNetNormalizer.clean_concept_uri(
            "/c/en/person_walking_down_the_street_on_a_rainy_day"
        )
        is None
    )


def test_conceptnet_normalizer_relations():
    """Verify relation mapping to LITTLE canonical semantics."""
    assert ConceptNetNormalizer.map_relation("/r/IsA") == "is_a"
    assert ConceptNetNormalizer.map_relation("/r/PartOf") == "part_of"
    assert ConceptNetNormalizer.map_relation("/r/HasA") == "has"
    assert ConceptNetNormalizer.map_relation("/r/CapableOf") == "can"
    assert ConceptNetNormalizer.map_relation("/r/AtLocation") == "located_in"
    assert ConceptNetNormalizer.map_relation("/r/MadeOf") == "made_of"
    assert ConceptNetNormalizer.map_relation("/r/UsedFor") == "used_for"
    assert ConceptNetNormalizer.map_relation("/r/Causes") == "causes"
    assert ConceptNetNormalizer.map_relation("/r/HasProperty") == "has_property"
    assert ConceptNetNormalizer.map_relation("/r/DistinctFrom") == "disjoint_with"
    assert ConceptNetNormalizer.map_relation("/r/Antonym") == "disjoint_with"


def test_knowledge_import_policy_can_override_aliases_and_filters(
    tmp_path: Path,
):
    payload = json.loads(
        Path("data/schemas/knowledge_import_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["knowledge_import_policy"]["relation_aliases"]["custom_relation"] = "is_a"
    payload["knowledge_import_policy"]["discard_concepts"].append("blocked")
    policy_path = tmp_path / "knowledge_import_policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = KnowledgeImportPolicy.load(tmp_path)

    assert ConceptNetNormalizer.map_relation("custom_relation", policy=policy) == "is_a"
    assert ConceptNetNormalizer.clean_concept_uri("blocked", policy=policy) is None

    store = MemoryStore(":memory:", seed_ontology=False)
    importer = KnowledgeImporter(store, import_policy=policy)
    stats = importer.import_triples(
        [
            ("blocked", "custom_relation", "animal", 1.0, True),
            ("wolf", "custom_relation", "animal", 1.0, True),
        ]
    )

    assert stats.relations_processed == 1


def test_conceptnet_tsv_import():
    """Verify importing ConceptNet 5 tab-separated assertion dump format."""
    tsv_content = (
        '/a/[/r/IsA/,/c/en/poodle/n/,/c/en/dog/n/]\t/r/IsA\t/c/en/poodle/n\t/c/en/dog/n\t{"weight": 2.5}\n'
        '/a/[/r/IsA/,/c/en/dog/n/,/c/en/canine/n/]\t/r/IsA\t/c/en/dog/n\t/c/en/canine/n\t{"weight": 3.0}\n'
        '/a/[/r/IsA/,/c/en/canine/n/,/c/en/mammal/n/]\t/r/IsA\t/c/en/canine/n\t/c/en/mammal/n\t{"weight": 2.0}\n'
        '/a/[/r/UsedFor/,/c/en/pencil/n/,/c/en/write/v/]\t/r/UsedFor\t/c/en/pencil/n\t/c/en/write/v\t{"weight": 2.0}\n'
        '/a/[/r/AtLocation/,/c/en/eiffel_tower/n/,/c/en/paris/n/]\t/r/AtLocation\t/c/en/eiffel_tower/n\t/c/en/paris/n\t{"weight": 2.0}\n'
        '/a/[/r/Causes/,/c/en/virus/n/,/c/en/fever/n/]\t/r/Causes\t/c/en/virus/n\t/c/en/fever/n\t{"weight": 1.5}\n'
        # Low weight assertion - should be skipped if min_weight=1.0
        '/a/[/r/IsA/,/c/en/dog/n/,/c/en/cat/n/]\t/r/IsA\t/c/en/dog/n\t/c/en/cat/n\t{"weight": 0.2}\n'
        # Non-English - should be skipped
        '/a/[/r/IsA/,/c/fr/chien/,/c/fr/animal/]\t/r/IsA\t/c/fr/chien\t/c/fr/animal\t{"weight": 2.0}\n'
    )

    with TemporaryDirectory() as tmpdir:
        dump_path = Path(tmpdir) / "conceptnet_sample.csv"
        dump_path.write_text(tsv_content, encoding="utf-8")

        store = MemoryStore(":memory:", seed_ontology=False)
        importer = KnowledgeImporter(store)
        stats = importer.import_conceptnet_file(dump_path, min_weight=1.0)

        assert stats.triples_parsed == 6
        assert stats.relations_processed == 6

        engine = LearningEngine(store)

        # Multi-hop transitive deduction over imported assertions: poodle -> dog -> canine -> mammal
        res = engine.ask("Is a poodle a mammal?")
        assert res.status == BeliefStatus.SUPPORTED
        assert res.answer is True
        assert "poodle is a mammal" in res.verbalize().lower()

        # Purpose query
        res_tool = engine.ask("What is a pencil used for?")
        assert res_tool.status == BeliefStatus.SUPPORTED
        assert "write" in str(res_tool.answer)

        # Causality query
        res_cause = engine.ask("What does a virus cause?")
        assert res_cause.status == BeliefStatus.SUPPORTED
        assert "fever" in str(res_cause.answer)


def test_json_and_tsv_import():
    """Verify importing JSON lines and simple TSV files."""
    jsonl_content = "\n".join(
        [
            json.dumps(
                {
                    "subject": "microscope",
                    "predicate": "used_for",
                    "object": "magnify small objects",
                    "weight": 2.0,
                }
            ),
            json.dumps(
                {
                    "subject": "telescope",
                    "predicate": "used_for",
                    "object": "observe stars",
                    "weight": 2.0,
                }
            ),
            json.dumps(
                {
                    "subject": "microscope",
                    "predicate": "is_a",
                    "object": "scientific instrument",
                    "weight": 2.0,
                }
            ),
        ]
    )

    with TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "instruments.jsonl"
        jsonl_path.write_text(jsonl_content, encoding="utf-8")

        store = MemoryStore(":memory:", seed_ontology=False)
        importer = KnowledgeImporter(store)
        stats = importer.import_json_file(jsonl_path)

        assert stats.relations_processed == 3

        engine = LearningEngine(store)
        res = engine.ask("What is a microscope used for?")
        assert res.status == BeliefStatus.SUPPORTED
        assert "magnify small objects" in str(res.answer)


def test_commonsense_world_bundle_reasoning():
    """Verify loading the full curated commonsense world bundle and reasoning across dimensions."""
    store = MemoryStore(":memory:", seed_ontology=True)
    importer = KnowledgeImporter(store)
    stats = importer.import_commonsense_bundle()

    assert stats.relations_processed > 500
    assert stats.elapsed_seconds < 1.0  # Must ingest sub-second

    engine = LearningEngine(store)

    # 1. Multi-hop Zoology Taxonomy: cheetah -> feline -> mammal -> vertebrate -> animal
    res_tax = engine.ask("Is a cheetah an animal?")
    assert res_tax.status == BeliefStatus.SUPPORTED
    assert res_tax.answer is True

    # 2. Tool usage: hammer used for hitting nails
    res_tool = engine.ask("What is a hammer used for?")
    assert res_tool.status == BeliefStatus.SUPPORTED
    assert "hit nail" in str(res_tool.answer)

    # 3. Flightless Bird Exception: penguin cannot fly
    res_peng = engine.ask("Can a penguin fly?")
    assert res_peng.status == BeliefStatus.REFUTED
    assert res_peng.answer is False

    # 4. Standard Bird Flight: eagle can fly
    res_eagle = engine.ask("Can an eagle fly?")
    assert res_eagle.status == BeliefStatus.SUPPORTED
    assert res_eagle.answer is True

    # 5. Causality: fire causes smoke
    res_cause = engine.ask("What does fire cause?")
    assert res_cause.status == BeliefStatus.SUPPORTED
    assert "smoke" in str(res_cause.answer)

    # 6. Reverse Causality: What causes smoke?
    res_rev_cause = engine.ask("What causes smoke?")
    assert res_rev_cause.status == BeliefStatus.SUPPORTED
    assert "fire" in str(res_rev_cause.answer).lower()

    # 7. Material Property: glass is transparent
    res_prop = engine.ask("Is glass transparent?")
    assert res_prop.status == BeliefStatus.SUPPORTED
    assert res_prop.answer is True

    # 8. Geography Transitivity: Tokyo located in Japan -> Asia
    res_geo = engine.ask("Is Tokyo located in Asia?")
    assert res_geo.status == BeliefStatus.SUPPORTED
    assert res_geo.answer is True

    # 9. Disjoint Refutation: dog is a reptile
    res_disj = engine.ask("Is a dog a reptile?")
    assert res_disj.status == BeliefStatus.REFUTED
    assert res_disj.answer is False

    # 10. Open-World Epistemic Honesty: unknown fact
    res_unk = engine.ask("Is an elephant capable of quantum teleportation?")
    assert res_unk.status == BeliefStatus.UNKNOWN
    assert res_unk.answer is None


def test_large_scale_ontology_generator_and_ingestion():
    memory = MemoryStore(":memory:")
    importer = KnowledgeImporter(memory, batch_size=2000)

    # Ingest 5,000 extended ontology triples
    stats = importer.import_large_scale_ontology(target_count=5000)
    assert stats.relations_processed >= 3000
    assert stats.new_concepts > 500

    engine = LearningEngine(memory)

    # Verify zoology deduction
    res_fel = engine.ask("Is a lion a feline?")
    assert res_fel.status == BeliefStatus.SUPPORTED

    res_mamm = engine.ask("Is a lion a mammal?")
    assert res_mamm.status == BeliefStatus.SUPPORTED

    res_anim = engine.ask("Is a lion an animal?")
    assert res_anim.status == BeliefStatus.SUPPORTED

    # Verify vehicle deduction
    res_veh = engine.ask("Is a car a vehicle?")
    assert res_veh.status == BeliefStatus.SUPPORTED

    # Verify chemistry
    res_chem = engine.ask("Is gold a metal?")
    assert res_chem.status == BeliefStatus.SUPPORTED

    # Verify disjoint refutation
    res_disj = engine.ask("Is a lion a plant?")
    assert res_disj.status == BeliefStatus.REFUTED
