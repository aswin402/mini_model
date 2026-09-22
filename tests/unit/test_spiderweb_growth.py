import pytest
from little.core.concept_knot import ConceptKnot
from little.active.spiderweb_growth import AutonomousSpiderWebEngine


def test_autonomous_upward_hypernym_induction():
    engine = AutonomousSpiderWebEngine()

    apple = ConceptKnot(
        concept_id="APPLE",
        taxonomy_hypernyms=["Fruit"],
        mereology_parts={
            "skin": "boundary",
            "pulp": "interior",
            "seeds": "core",
        },
        procedural_skills=["slice", "juice"],
    )
    pear = ConceptKnot(
        concept_id="PEAR",
        taxonomy_hypernyms=["Fruit"],
        mereology_parts={
            "skin": "boundary",
            "pulp": "interior",
            "seeds": "core",
        },
        procedural_skills=["slice", "juice"],
    )

    # Induces common category utility cluster
    super_concept = engine.induce_hypernym_cobweb([apple, pear])
    assert super_concept is not None
    assert "POME" in super_concept or "CLUSTER" in super_concept


def test_active_curiosity_gap_identification():
    engine = AutonomousSpiderWebEngine()

    # Concept with missing dynamical & procedural spokes
    incomplete_knot = ConceptKnot(
        concept_id="QUINCE",
        taxonomy_hypernyms=["PomeFruit"],
        mereology_parts={"skin": "boundary"},
        procedural_skills=[],  # Missing skills!
    )

    gaps = engine.identify_epistemic_gaps(incomplete_knot)
    assert len(gaps) > 0
    assert any("procedural" in g.lower() or "skills" in g.lower() for g in gaps)
