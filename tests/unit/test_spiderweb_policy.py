from dataclasses import replace
from pathlib import Path

from little.active.spiderweb_growth import AutonomousSpiderWebEngine
from little.active.spiderweb_policy import SpiderWebGrowthPolicy
from little.core.concept_knot import ConceptKnot


def _matching_knots() -> list[ConceptKnot]:
    return [
        ConceptKnot(
            concept_id="APPLE",
            taxonomy_hypernyms=["Fruit"],
            procedural_skills=["slice", "juice"],
        ),
        ConceptKnot(
            concept_id="PEAR",
            taxonomy_hypernyms=["Fruit"],
            procedural_skills=["slice", "juice"],
        ),
    ]


def test_spiderweb_policy_loads_growth_decisions_from_schema():
    policy = SpiderWebGrowthPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.minimum_cluster_size == 2
    assert policy.category_utility_threshold == 1.0
    assert policy.cluster_prefix == "POME"
    assert policy.cluster_suffix == "CLUSTER"
    assert policy.minimum_mereology_parts == 2
    assert "procedural" in policy.gap_templates


def test_spiderweb_engine_uses_configured_cluster_threshold():
    policy = SpiderWebGrowthPolicy.load(Path("data/schemas"))
    strict_policy = replace(policy, category_utility_threshold=3.0)

    assert AutonomousSpiderWebEngine(strict_policy).induce_hypernym_cobweb(
        _matching_knots()
    ) is None
