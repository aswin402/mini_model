from dataclasses import replace

import pytest

from little.active.cobweb_engine import CobwebInstance, CobwebTree
from little.neural.turbo_quant import TurboQuantCompressor, TurboQuantIndex
from little.neural.turbo_quant_policy import TurboQuantPolicy


def test_policy_rejects_unsupported_encoding():
    policy = TurboQuantPolicy.default()
    for kwargs in [{'dimension': 32}, {'dimension': 0}, {'bins_4bit': 8}, {'quantization_modes': ('1bit',)}]:
        with pytest.raises(ValueError):
            TurboQuantCompressor(replace(policy, **kwargs))


def test_trial_evaluation_preserves_existing_id():
    tree = CobwebTree()
    first = CobwebInstance('first', {'kind': {'bird'}})
    tree.insert(first)
    tree.insert(CobwebInstance('second', {'kind': {'fish'}}))
    before = [(set(c.concept_ids), c.concept_count, {k: dict(v) for k, v in c.attribute_counts.items()}) for c in tree.root.children]
    tree._evaluate_children(tree.root, first)
    after = [(set(c.concept_ids), c.concept_count, {k: dict(v) for k, v in c.attribute_counts.items()}) for c in tree.root.children]
    assert before == after


def test_duplicates_rejected_without_mutation():
    tree = CobwebTree()
    first = CobwebInstance('first', {'kind': {'bird'}})
    tree.insert(first)
    with pytest.raises(ValueError): tree.insert(first)
    assert tree.root.concept_count == 1
    with pytest.raises(ValueError): tree.root.add_instance(first)
    assert tree.root.concept_count == 1


def test_prequantized_insert_falls_back_instead_of_silent_miss():
    index = TurboQuantIndex()
    vector = [1.0] + [0.0] * 511
    quantized = index.compressor.compress('existing', vector)
    index.add_quantized(quantized)
    assert index.search(vector, threshold=0.5)[0].id == 'existing'
    index.close()
    index.close()
    with pytest.raises(RuntimeError): index.add('closed', vector)


def test_threshold_controls_cluster_acceptance():
    instances = [CobwebInstance(name, {'kind': {kind}}) for name, kind in [('a', 'bird'), ('b', 'bird'), ('c', 'fish'), ('d', 'fish')]]
    permissive, strict = CobwebTree(0), CobwebTree(100)
    for instance in instances:
        permissive.insert(instance)
        strict.insert(instance)
    assert permissive.get_all_clusters()
    assert strict.get_all_clusters() == []
    assert permissive.root.concept_count == len(permissive.root.concept_ids) == 4


def test_fallback_search_and_context_cleanup(monkeypatch):
    from little.rust.engine import RustAccelerationCore
    core = RustAccelerationCore.get()
    monkeypatch.setattr(core, '_lib', None)
    with TurboQuantIndex() as index:
        vector = [1.0] + [0.0] * 511
        index.add_quantized(index.compressor.compress('fallback', vector))
        assert index.search(vector, threshold=0.5)[0].id == 'fallback'
        assert index.memory_bytes() == 336
        assert index.search(vector, top_k=0) == []
        with pytest.raises(ValueError): index.search([0.0] * 32)
        with pytest.raises(ValueError): index.search(vector, top_k=-1)
    assert index._closed and index.compressor._closed


def test_payload_validation_and_upsert():
    with TurboQuantIndex() as index:
        vector = [1.0] + [0.0] * 511
        quantized = index.add('same', vector, {'old': True})
        index.add('same', vector, {'new': True})
        assert len(index) == 1
        assert index.search(vector, threshold=0.5)[0].metadata == {'new': True}
        with pytest.raises(ValueError): index.add_quantized(replace(quantized, vector_4bit=b''))
        with pytest.raises(ValueError): index.add_quantized(replace(quantized, scale=float('nan')))
        with pytest.raises(ValueError): index.add('nul\x00id', vector)
