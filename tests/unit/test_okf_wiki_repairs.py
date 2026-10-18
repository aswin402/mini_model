import pytest

from little.knowledge.living_wiki import LivingWiki
from little.knowledge.okf import (
    OKFDocument,
    OKFEngine,
    parse_yaml_frontmatter,
    serialize_yaml_value,
)
from little.memory.store import MemoryStore


def test_weighted_negative_direction_roundtrip(tmp_path):
    source = MemoryStore(tmp_path / "source.db")
    target = MemoryStore(tmp_path / "target.db")
    a = source.create_concept("wheel")
    b = source.create_concept("vehicle")
    for _ in range(3):
        source.add_relation(a.id, "is_a", b.id, positive=False)
    source.add_relation(a.id, "part_of", b.id)
    path = OKFEngine(source).export_concept("wheel", tmp_path / "vault")
    engine = OKFEngine(target)
    assert engine.import_okf_file(path).passed_verification
    rels = {r.predicate: r for r in target.get_relations()}
    assert set(rels) == {"is_a", "part_of"}
    assert (rels["is_a"].weight_positive, rels["is_a"].weight_negative) == (0, 3)
    assert (
        rels["is_a"].confidence == source.get_relations(predicate="is_a")[0].confidence
    )
    engine.import_okf_file(path)
    assert target.get_relations(predicate="is_a")[0].weight_negative == 3


def test_safe_names_required_fields_and_quoting(tmp_path):
    store = MemoryStore(tmp_path / "db")
    store.create_concept("../escape")
    with pytest.raises(ValueError):
        OKFEngine(store).export_concept("../escape", tmp_path / "vault")
    path = tmp_path / "bad.okf.md"
    path.write_text("---\nname: only\n---\n")
    assert not OKFEngine(store).import_okf_file(path).passed_verification
    for value in ['a "quoted" name', "line\nnext", "1", "none", "a: b"]:
        fm, _ = parse_yaml_frontmatter(
            "---\n" + serialize_yaml_value({"value": value}) + "\n---\n"
        )
        assert fm["value"] == value


def test_wiki_updates_backlinks_and_removes_deleted_vectors(tmp_path):
    wiki = LivingWiki(MemoryStore(tmp_path / "db"), tmp_path / "vault")
    path = wiki.save_or_update_page(OKFDocument({"name": "wheel"}, "[[vehicle]]"))
    assert wiki.get_backlinks("vehicle") == ["wheel"]
    wiki.save_or_update_page(OKFDocument({"name": "wheel"}, "[[machine]]"))
    assert wiki.get_backlinks("vehicle") == []
    assert wiki.get_backlinks("machine") == ["wheel"]
    assert "vehicle" not in wiki.get_unexplored_frontier()
    path.unlink()
    wiki.build_graph()
    assert wiki.stats().vector_index_size == 0
    assert wiki.search_semantic("wheel", threshold=-1) == []


def test_explicit_evidence_survives_roundtrip(tmp_path):
    from little.core.contracts import (
        CandidateClaim,
        EvidenceRecord,
        EvidenceStatus,
        SourceType,
    )

    source = MemoryStore(tmp_path / "source.db")
    target = MemoryStore(tmp_path / "target.db")
    a = source.create_concept("wheel")
    b = source.create_concept("vehicle")
    source.add_relation(a.id, "part_of", b.id)
    evidence = EvidenceRecord.from_claim(
        CandidateClaim("wheel", "part_of", "vehicle", 0.91),
        SourceType.DOCUMENT,
        "manual:17",
        "The wheel belongs to the vehicle.",
    )
    evidence.status = EvidenceStatus.ACCEPTED
    source.evidence.append(evidence)
    path = OKFEngine(source).export_concept("wheel", tmp_path / "vault")
    OKFEngine(target).import_okf_file(path)
    records = target.evidence.list_for_claim("wheel", "part_of", "vehicle")
    assert any(
        r.source_reference == "manual:17" and r.source_text == evidence.source_text
        for r in records
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("auto_sync_on_mutation", "false"),
        ("batch_crawl_limit", -1),
        ("min_extraction_confidence", 1.1),
    ],
)
def test_wiki_policy_rejects_invalid_controls(tmp_path, field, value):
    import json

    from little.knowledge.living_wiki_policy import LivingWikiPolicy

    payload = {"format": "little.living_wiki.v1", field: value}
    (tmp_path / "living_wiki_policy.json").write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        LivingWikiPolicy.load(tmp_path)


def test_invalid_versioned_record_has_no_store_side_effects(tmp_path):
    store = MemoryStore(tmp_path / "db")
    path = tmp_path / "bad.okf.md"
    payload = {
        "id": "X",
        "name": "wheel",
        "taxonomy": {},
        "format": "little.okf.v2",
        "relations": [
            {
                "predicate": "part_of",
                "object": "vehicle",
                "weight_positive": 1,
                "weight_negative": 0,
                "confidence": 0.5,
                "evidence_json": "invalid",
            }
        ],
    }
    path.write_text("---\n" + serialize_yaml_value(payload) + "\n---\n")
    result = OKFEngine(store).import_okf_file(path)
    assert not result.passed_verification
    assert store.list_concepts() == []


@pytest.mark.parametrize(
    "change", [{"name": 123}, {"id": None}, {"taxonomy": []}, {"confidence": 2}]
)
def test_invalid_document_fields_are_rejected_before_mutation(tmp_path, change):
    store = MemoryStore(tmp_path / "db")
    fields = {"id": "X", "name": "wheel", "taxonomy": {}}
    fields.update(change)
    path = tmp_path / "invalid.okf.md"
    path.write_text("---\n" + serialize_yaml_value(fields) + "\n---\n")
    assert not OKFEngine(store).import_okf_file(path).passed_verification
    assert store.list_concepts() == []
