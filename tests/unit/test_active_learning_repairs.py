from dataclasses import replace
from unittest.mock import Mock

import pytest

from little.active.crawler import KnowledgeCrawler
from little.active.latent_knot_inducer import LatentKnotInducer
from little.active.self_study import AutonomousSelfStudyEngine
from little.core.contracts import CandidateClaim, SourceType
from little.core.models import BeliefStatus
from little.inference.engine import InferenceEngine
from little.knowledge.living_wiki import LivingWiki
from little.memory.store import MemoryStore


@pytest.fixture
def state(tmp_path):
    memory = MemoryStore(':memory:', seed_ontology=False)
    wiki = LivingWiki(memory, vault_dir=tmp_path / 'wiki')
    yield memory, wiki
    memory.close()


def teach_birds(memory):
    ids = {name: memory.create_concept(name).id for name in ('groundbird', 'gliderbird', 'bird', 'fly')}
    for subject, predicate, obj in [('groundbird', 'is_a', 'bird'), ('gliderbird', 'is_a', 'bird'), ('gliderbird', 'can', 'fly')]:
        memory.add_relation(ids[subject], predicate, ids[obj])


def test_analogy_is_candidate_with_donor_proof_not_fact(state):
    memory, wiki = state
    teach_birds(memory)
    result = AutonomousSelfStudyEngine(memory, wiki=wiki).study_step('groundbird', preferred_axis='procedural')
    assert result.status == 'HYPOTHESIS'
    assert InferenceEngine(memory).infer('groundbird', 'can', 'fly').status is BeliefStatus.UNKNOWN
    evidence = memory.evidence.list_by_source(SourceType.HYPOTHESIS)
    assert evidence and evidence[0].status.value == 'candidate'
    assert 'gliderbird' in evidence[0].source_text
    assert evidence[0].derivation_proof_id
    row = memory._conn.execute('SELECT premises_json,status FROM proof_traces WHERE proof_id=?', (evidence[0].derivation_proof_id,)).fetchone()
    assert row['status'] == 'unknown'
    assert 'rel_' in row['premises_json']
    assert result.gate_verdicts['I_GROUND'] is False


def test_no_hardcoded_mutex_mining(state):
    memory, wiki = state
    memory.create_concept('plant')
    memory.create_concept('animal')
    before = memory.count_relations()
    assert AutonomousSelfStudyEngine(memory, wiki=wiki)._mine_latent_invariants() == []
    assert memory.count_relations() == before


def test_real_corpus_source_can_support_target_independently(state, tmp_path):
    memory, wiki = state
    teach_birds(memory)
    corpus = tmp_path / 'birds.md'
    corpus.write_text('A groundbird can fly.')
    engine = AutonomousSelfStudyEngine(memory, wiki=wiki, corpus_paths=(corpus,))
    result = engine.study_step('groundbird', preferred_axis='procedural')
    assert result.status == 'PERSISTED'
    assert InferenceEngine(memory).infer('groundbird', 'can', 'fly').status is BeliefStatus.SUPPORTED
    documents = memory.evidence.list_by_source(SourceType.DOCUMENT)
    assert any(e.source_reference == str(corpus.resolve()) and e.source_text == 'A groundbird can fly.' for e in documents)


def test_crawler_rejected_claim_creates_no_concepts(state):
    memory, wiki = state
    extractor = Mock()
    extractor.extract_candidate_claims.return_value = [CandidateClaim('ghost', 'invented_predicate', 'phantom', 0.99)]
    report = KnowledgeCrawler(memory, wiki, frame_extractor=extractor).crawl_text('A ghost invents a phantom.')
    assert report.rejected_claims == 1
    assert not memory.get_concept('ghost')
    assert not memory.get_concept('phantom')


def test_crawler_deduplicates_replay_and_link_frame_same_claim(state):
    memory, wiki = state
    extractor = Mock()
    extractor.extract_candidate_claims.return_value = [CandidateClaim('apple', 'is_a', 'fruit', 0.95)]
    crawler = KnowledgeCrawler(memory, wiki, frame_extractor=extractor)
    first = crawler.crawl_text('An [[apple]] is a [[fruit]].', source_reference='doc:fruit')
    second = crawler.crawl_text('An [[apple]] is a [[fruit]].', source_reference='doc:fruit')
    assert first.verified_claims == 1
    assert second.verified_claims == 1
    relation = memory.find_relation_by_names('apple', 'is_a', 'fruit')
    assert relation.weight_positive == 1
    assert len(memory.evidence.list_by_source(SourceType.DOCUMENT)) == 1
    assert relation.source_experience_id


def test_crawler_enforces_extraction_threshold_and_directory_budget(state, tmp_path):
    memory, wiki = state
    wiki.policy = replace(wiki.policy, min_extraction_confidence=0.9, batch_crawl_limit=1)
    extractor = Mock()
    extractor.extract_candidate_claims.return_value = [CandidateClaim('apple', 'is_a', 'fruit', 0.8)]
    crawler = KnowledgeCrawler(memory, wiki, frame_extractor=extractor)
    result = crawler.crawl_text('An apple is a fruit.')
    assert result.verified_claims == 0
    assert result.rejected_claims == 1
    assert not memory.get_concept('apple')
    docs = tmp_path / 'docs'
    docs.mkdir()
    for name in ('one.md', 'two.md'):
        (docs / name).write_text('An apple is a fruit.')
    assert len(crawler.crawl_directory(docs)) == 1


def test_crawler_reports_sync_failure_after_successful_learning(state, monkeypatch):
    memory, wiki = state
    monkeypatch.setattr(wiki.okf_engine, 'export_concept', Mock(side_effect=OSError('disk full')))
    result = KnowledgeCrawler(memory, wiki).crawl_text('An apple is a fruit.')
    assert result.verified_claims == 1
    assert result.sync_errors
    assert any('disk full' in e for e in result.sync_errors)
    assert memory.find_relation_by_names('apple', 'is_a', 'fruit')


def test_latent_membership_has_definition_provenance(state):
    memory, wiki = state
    teach_birds(memory)
    # Both members have actual shared capability, beyond their common category.
    memory.add_relation(memory.get_concept('groundbird').id, 'can', memory.get_concept('fly').id)
    for name in ('fish', 'swim', 'tuna', 'shark'):
        memory.create_concept(name)
    for name in ('tuna', 'shark'):
        memory.add_relation(memory.get_concept(name).id, 'is_a', memory.get_concept('fish').id)
        memory.add_relation(memory.get_concept(name).id, 'can', memory.get_concept('swim').id)
    from little.active.spiderweb_policy import SpiderWebGrowthPolicy
    policy = replace(SpiderWebGrowthPolicy.default(), category_utility_threshold=0.0)
    result = LatentKnotInducer(memory, wiki, policy=policy).induce_latent_knots()
    assert result.latent_knots_induced > 0
    derived = memory.evidence.list_by_source(SourceType.INFERENCE)
    assert derived
    assert all(e.derivation_proof_id for e in derived)
    assert any('defined cluster' in e.source_text for e in derived)
