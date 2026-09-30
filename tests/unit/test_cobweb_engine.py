"""Unit tests for Cobweb Conceptual Clustering Engine."""

from __future__ import annotations

import pytest

from little.active.cobweb_engine import CobwebInstance, CobwebNode, CobwebTree


def test_cobweb_node_attribute_tracking() -> None:
    node = CobwebNode(node_id="test_node")
    inst1 = CobwebInstance("eagle", {"is_a": {"bird"}, "capable_of": {"fly", "hunt"}})
    inst2 = CobwebInstance("falcon", {"is_a": {"bird"}, "capable_of": {"fly"}})

    node.add_instance(inst1)
    assert node.concept_count == 1
    assert node.probability_attribute_value("is_a", "bird") == 1.0
    assert node.probability_attribute_value("capable_of", "fly") == 1.0
    assert node.probability_attribute_value("capable_of", "hunt") == 1.0

    node.add_instance(inst2)
    assert node.concept_count == 2
    assert node.probability_attribute_value("is_a", "bird") == 1.0
    assert node.probability_attribute_value("capable_of", "fly") == 1.0
    assert node.probability_attribute_value("capable_of", "hunt") == 0.5

    shared = node.get_shared_features(min_ratio=0.6)
    assert "bird" in shared["is_a"]
    assert "fly" in shared["capable_of"]
    assert "hunt" not in shared.get("capable_of", [])

    node.remove_instance(inst2)
    assert node.concept_count == 1
    assert node.probability_attribute_value("capable_of", "hunt") == 1.0


def test_cobweb_tree_clustering_and_category_utility() -> None:
    tree = CobwebTree(category_utility_threshold=0.15)

    instances = [
        CobwebInstance("eagle", {"is_a": {"bird"}, "capable_of": {"fly", "hunt"}}),
        CobwebInstance("falcon", {"is_a": {"bird"}, "capable_of": {"fly", "hunt"}}),
        CobwebInstance("hawk", {"is_a": {"bird"}, "capable_of": {"fly", "hunt"}}),
        CobwebInstance("shark", {"is_a": {"fish"}, "capable_of": {"swim", "hunt"}}),
        CobwebInstance("tuna", {"is_a": {"fish"}, "capable_of": {"swim"}}),
    ]

    for inst in instances:
        leaf = tree.insert(inst)
        assert leaf is not None

    clusters = tree.get_all_clusters(min_size=2)
    assert len(clusters) >= 2

    # Verify that eagle, falcon, hawk are clustered together
    bird_clusters = [c for c in clusters if "eagle" in c.concept_ids]
    assert len(bird_clusters) > 0
    best_bird_cluster = max(bird_clusters, key=lambda c: len(c.concept_ids))
    assert "falcon" in best_bird_cluster.concept_ids or "hawk" in best_bird_cluster.concept_ids
    assert "shark" not in best_bird_cluster.concept_ids
