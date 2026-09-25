from pathlib import Path

from little.core.concept_knot_policy import ConceptKnotPolicy


def test_concept_knot_policy_loads_axes_roles_and_defaults():
    policy = ConceptKnotPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.taxonomy_axis == "taxonomy"
    assert policy.mereology_roles["made_of"] == "material_substance"
    assert policy.defaults["mass"] == 150.0
    assert policy.defaults["constructor_mass"] == 180.0
