from dataclasses import replace
from pathlib import Path

from little.core.kernel import CognitiveKernel
from little.core.models import Construction
from little.core.runtime_policy import RuntimePolicy
from little.language.parser import LearningEngine, SimpleParser, configured_parser
from little.language.parser_confidence_policy import ParserConfidencePolicy
from little.language.parser_policy import ParserPolicy
from little.memory.store import MemoryStore
from little.procedural.unit_conversion_policy import UnitConversionPolicy


def test_runtime_policy_loads_the_default_policy_bundle():
    policy = RuntimePolicy.load(Path("data/schemas"))

    assert policy.semantic.taxonomy == "is_a"
    assert policy.reasoning.max_depth == 8
    assert policy.units.aliases["kilometers"] == "km"
    assert policy.dynamics.continuous.q10 == 2.0
    assert any(
        construction.name == "cxn_is_a" for construction in policy.constructions
    )


def test_kernel_propagates_injected_runtime_policies():
    default = RuntimePolicy.default()
    custom_units = UnitConversionPolicy(
        aliases={"custom": "custom", "base": "base"},
        edges=(),
    )
    custom = replace(default, units=custom_units)

    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory, runtime_policy=custom)

    assert kernel.runtime_policy is custom
    assert kernel.learner.runtime_policy is custom
    assert kernel.learner.unit_conversions.policy is custom_units
    assert kernel.learner.transformation_policy is custom.transformation
    assert kernel.learner.memory.ledger.transformation_policy is custom.transformation
    assert kernel.learner.memory.ledger.dynamics_registry is custom.dynamics
    assert kernel.controller.runtime_policy is custom
    assert kernel.controller.inference.policy is custom.reasoning


def test_configured_parser_uses_injected_lexical_policy_without_global_mutation():
    default = ParserPolicy.default()
    custom = replace(
        default,
        normalization_replacements=(
            *default.normalization_replacements,
            ("falcon", "eagle"),
        ),
    )

    parser = configured_parser(custom)

    parsed = parser.parse_statement("A falcon is a bird.")

    assert parsed[0].subject == "eagle"
    assert SimpleParser.parse_statement("A falcon is a bird.")[0].subject == "falcon"


def test_configured_parser_preserves_legacy_policy_override_when_unspecified(
    monkeypatch,
):
    custom = replace(
        SimpleParser.POLICY,
        normalization_replacements=(
            *SimpleParser.POLICY.normalization_replacements,
            ("falcon", "eagle"),
        ),
    )
    monkeypatch.setattr(SimpleParser, "POLICY", custom)

    parser = configured_parser()

    assert parser.POLICY is custom
    assert parser.parse_statement("A falcon is a bird.")[0].subject == "eagle"


def test_runtime_parser_uses_injected_construction_pack():
    default = RuntimePolicy.default()
    custom_construction = Construction.create(
        name="custom_gliding_relation",
        pattern_tokens=["{X}", "glides", "over", "{Y}"],
        slot_roles={"X": "subject", "Y": "object"},
        predicate_template="moves_over",
    )
    runtime = replace(default, constructions=(custom_construction,))
    memory = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(memory, runtime_policy=runtime)

    parsed = engine.parser.parse_statement("A bird glides over clouds.")

    assert [(triple.subject, triple.predicate, triple.object_) for triple in parsed] == [
        ("bird", "moves_over", "cloud")
    ]


def test_kernel_runtime_construction_pack_precedes_persisted_default_catalog():
    default = RuntimePolicy.default()
    custom_construction = Construction.create(
        name="custom_taxonomy_relation",
        pattern_tokens=["{X}", "is", "a", "{Y}"],
        slot_roles={"X": "subject", "Y": "object"},
        predicate_template="classified_as",
    )
    runtime = replace(default, constructions=(custom_construction,))
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory, runtime_policy=runtime).process(
        "A bird is a animal."
    )

    assert outcome.frame.claims[0].predicate == "classified_as"


def test_learning_engine_uses_runtime_parser_confidence_policy():
    default = RuntimePolicy.default()
    confidence = ParserConfidencePolicy(
        version="test-confidence",
        certain=0.61,
        derived=0.52,
        unknown=0.13,
        low=0.21,
    )
    parser = replace(default.parser, confidence=confidence)
    runtime = replace(default, parser=parser)
    memory = MemoryStore(":memory:", seed_ontology=False)

    result = LearningEngine(memory, runtime_policy=runtime).ask(
        "What is your name?"
    )

    assert result.confidence == 0.61


def test_learning_engine_perception_uses_runtime_parser_policy():
    default = RuntimePolicy.default()
    parser = replace(
        default.parser,
        normalization_replacements=(
            *default.parser.normalization_replacements,
            ("falcon", "eagle"),
        ),
    )
    runtime = replace(default, parser=parser)
    memory = MemoryStore(":memory:", seed_ontology=False)

    LearningEngine(memory, runtime_policy=runtime).learn("A falcon is a bird.")

    assert memory.get_concept("eagle") is not None
    assert memory.get_concept("falcon") is None


def test_kernel_perception_uses_runtime_parser_policy():
    default = RuntimePolicy.default()
    parser = replace(
        default.parser,
        normalization_replacements=(
            *default.parser.normalization_replacements,
            ("falcon", "eagle"),
        ),
    )
    runtime = replace(default, parser=parser)
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory, runtime_policy=runtime).process(
        "A falcon is a bird."
    )

    assert outcome.frame.claims[0].subject == "eagle"
