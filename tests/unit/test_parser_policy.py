import json
from pathlib import Path

from little.core.grammar_registry import GrammarRegistry
from little.core.models import Construction
from little.language.construction import ConstructionEngine
from little.language.parser import SimpleParser
from little.language.parser_policy import ParserPolicy


def test_simple_parser_has_no_process_global_parser_policy():
    assert "POLICY" not in SimpleParser.__dict__


def test_simple_parser_has_no_process_global_runtime_resources():
    assert "CONSTRUCTIONS" not in SimpleParser.__dict__
    assert "UNIT_CONVERSIONS" not in SimpleParser.__dict__
    assert "CONSTRUCTION_ENGINE" not in SimpleParser.__dict__


def test_parser_policy_loads_shared_lexical_tables():
    policy = ParserPolicy.load(Path("data/schemas"))

    assert policy.action_verbs["flies"] == "fly"
    assert policy.irregular_plurals["mice"] == "mouse"
    assert "transparent" in policy.known_properties
    assert "hello" in policy.isolated_greetings
    assert "into" in policy.pivot_prepositions
    assert "used for" in policy.invalid_definition_markers
    assert "than" in policy.invalid_category_markers
    assert "ADD" in policy.numeric_skills
    assert "used_for" in policy.open_query_predicates
    assert "cannot" in policy.noun_auxiliaries
    assert "oes" in policy.plural_es_suffixes
    assert "wives" in policy.plural_f_to_fe_words


def test_parser_uses_injected_action_vocabulary(tmp_path: Path, monkeypatch):
    payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    payload["parser_policy"]["action_verbs"]["glides"] = "glide"
    custom_path = tmp_path / "parser_policy.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    parsed = SimpleParser.parse_statement("A bird glides")

    assert parsed[0].predicate == "can"
    assert parsed[0].object_ == "glide"


def test_parser_uses_injected_special_question_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_question_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_question_policy.v1",
                "patterns": [
                    {
                        "name": "secret_question",
                        "pattern": "^what is the secret$",
                        "subject": "vault",
                        "predicate": "__secret__",
                        "target": "?",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("What is the secret?") == (
        "vault",
        "__secret__",
        "?",
    )


def test_parser_uses_injected_math_pattern_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_math_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_math_policy.v1",
                "patterns": [
                    {
                        "name": "custom_addition",
                        "pattern": "^(\\d+)\\s*\\+\\s*(\\d+)$",
                        "skill": "SUBTRACT",
                        "argument_groups": [1, 2],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("7 + 5?") == (
        "SUBTRACT",
        "__math__",
        {"a": 7, "b": 5},
    )


def test_parser_uses_injected_unary_math_pattern_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_math_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_math_policy.v1",
                "patterns": [
                    {
                        "name": "custom_unary",
                        "pattern": r"^double\s+(\d+)$",
                        "skill": "CUSTOM",
                        "argument_groups": [1],
                        "argument_names": ["value"],
                        "value_type": "number",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("double 7?") == (
        "CUSTOM",
        "__math__",
        {"value": 7},
    )


def test_parser_uses_injected_math_capture_normalizer(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_math_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_math_policy.v1",
                "patterns": [
                    {
                        "name": "custom_expression",
                        "pattern": r"^calculate\s+(.+)$",
                        "skill": "EVAL_EXPR",
                        "captures": [
                            {
                                "group": 1,
                                "name": "expression",
                                "value_type": "text",
                                "normalizers": ["caret_to_power"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("calculate 2^3?") == (
        "EVAL_EXPR",
        "__math__",
        {"expression": "2**3"},
    )


def test_linear_math_capture_uses_sign_as_metadata_only():
    assert SimpleParser.parse_question("solve 2x - 3 = 7") == (
        "SOLVE_LINEAR",
        "__math__",
        {"a": 2.0, "b": -3.0, "c": 7.0},
    )


def test_parser_uses_injected_temporal_question_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_temporal_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_temporal_policy.v1",
                "patterns": [
                    {
                        "name": "custom_shade_after_duration",
                        "pattern": r"^what shade is\s+(.*?)\s+after\s+(\d+)\s+(hours?)$",
                        "predicate": "__custom_temporal__",
                        "subject_group": 1,
                        "value_group": 2,
                        "unit_group": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("What shade is an apple slice after 2 hours?") == (
        "apple slice",
        "__custom_temporal__",
        (2.0, "hours"),
    )


def test_parser_uses_injected_semantic_question_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_semantic_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_semantic_policy.v1",
                "patterns": [
                    {
                        "name": "custom_shade_question",
                        "pattern": r"^what shade is\s+(.+)$",
                        "predicate": "__custom_color__",
                        "subject_groups": [1],
                        "target_literal": "?",
                        "validate_subject": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("What shade is moon?") == (
        "moon",
        "__custom_color__",
        "?",
    )


def test_parser_does_not_use_hardcoded_comparative_and_why_routes(
    tmp_path: Path, monkeypatch
):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_semantic_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_semantic_policy.v1",
                "patterns": [],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [])

    comparison = SimpleParser.parse_question("Is an eagle larger than a bird?")
    why_not = SimpleParser.parse_question("Why is water not a solid?")

    assert comparison is None or comparison[1] != "larger_than"
    assert why_not is None or why_not[1] != "__why_not__"


def test_parser_uses_injected_definition_question_catalog(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_definition_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_definition_policy.v1",
                "patterns": [
                    {
                        "name": "custom_definition",
                        "pattern": r"^define\s+([a-z0-9_\s-]+)$",
                        "predicate": "__custom_definition__",
                        "subject_group": 1,
                        "exclude_non_concept": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)

    assert SimpleParser.parse_question("Define moon?") == (
        "moon",
        "__custom_definition__",
        None,
    )


def test_parser_does_not_use_hardcoded_taxonomy_fallback(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_semantic_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_semantic_policy.v1",
                "patterns": [],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [])

    parsed = SimpleParser.parse_question("Is a dog an animal?")

    assert parsed is None or parsed[1] != policy.semantic.taxonomy


def test_parser_does_not_use_hardcoded_capability_fallback(tmp_path: Path, monkeypatch):
    parser_payload = json.loads(
        Path("data/schemas/parser_policy.json").read_text(encoding="utf-8")
    )
    (tmp_path / "parser_policy.json").write_text(
        json.dumps(parser_payload), encoding="utf-8"
    )
    (tmp_path / "parser_semantic_policy.json").write_text(
        json.dumps(
            {
                "format": "little.parser_semantic_policy.v1",
                "patterns": [],
            }
        ),
        encoding="utf-8",
    )
    policy = ParserPolicy.load(tmp_path)
    monkeypatch.setattr(SimpleParser, "POLICY", policy)
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [])

    can_question = SimpleParser.parse_question("Can a bird fly?")
    does_question = SimpleParser.parse_question("Does a bird fly?")

    assert can_question is None or can_question[1] != policy.semantic.capability
    assert does_question is None or does_question[1] != policy.semantic.capability


def test_statement_parser_uses_an_injected_construction_catalog(monkeypatch):
    custom = Construction.create(
        name="custom_gliding_relation",
        pattern_tokens=["{X}", "glides", "over", "{Y}"],
        slot_roles={"X": "subject", "Y": "object"},
        predicate_template="moves_over",
    )
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [custom])

    parsed = SimpleParser.parse_statement("A bird glides over clouds.")

    assert [(triple.subject, triple.predicate, triple.object_) for triple in parsed] == [
        ("bird", "moves_over", "cloud")
    ]


def test_question_parser_uses_an_injected_construction_catalog(monkeypatch):
    custom = Construction.create(
        name="custom_residence_question",
        pattern_tokens=["where", "does", "{X}", "reside"],
        slot_roles={"X": "subject"},
        predicate_template="located_in",
        construction_type="question",
    )
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [custom])

    parsed = SimpleParser.parse_question("Where does Paris reside?")

    assert parsed == ("paris", "located_in", "?")


def test_question_parser_uses_injected_procedural_catalog(monkeypatch):
    custom = Construction.create(
        name="custom_addition",
        pattern_tokens=["compute", "{a}", "plus", "{b}"],
        slot_roles={"a": "a", "b": "b"},
        predicate_template="add",
        construction_type="procedural",
    )
    monkeypatch.setattr(SimpleParser, "CONSTRUCTIONS", [custom])

    parsed = SimpleParser.parse_question("compute 7 plus 5?")

    assert parsed == ("ADD", "__math__", {"a": 7, "b": 5})


def test_default_question_catalog_covers_relation_question_forms():
    constructions = GrammarRegistry.default()
    cases = [
        ("Where is Paris?", "paris", "located_in", "?"),
        ("Why is Paris in Europe?", "paris", "__why_located_in__", "europe"),
        ("Is an animal not a vehicle?", "animal", "disjoint_with", "vehicle"),
        ("Is a wheel part of a car?", "wheel", "part_of", "car"),
        ("Does a salmon live in water?", "salmon", "lives_in", "water"),
        ("Is ice made of water?", "ice", "made_of", "water"),
        ("Does a tiger eat meat?", "tiger", "eats", "meat"),
        ("Does Alice have a dog?", "alice", "has", "dog"),
        ("Is a hammer used for hitting nails?", "hammer", "used_for", "hitting nail"),
        ("Does fire cause smoke?", "fire", "causes", "smoke"),
    ]
    for text, subject, predicate, object_ in cases:
        parsed = ConstructionEngine.parse_question_from_catalog(text, constructions)
        assert parsed == (subject, predicate, object_)


def test_default_catalog_covers_procedural_question_forms():
    constructions = GrammarRegistry.default()
    cases = [
        ("2 + 3", "ADD", {"a": 2, "b": 3}),
        ("what is 2 plus 3", "ADD", {"a": 2, "b": 3}),
        ("what is the square root of 9", "SQRT", {"n": 9}),
        ("is 7 prime", "IS_PRIME", {"n": 7}),
        ("prime 11", "IS_PRIME", {"n": 11}),
        ("what is 5 factorial", "FACTORIAL", {"n": 5}),
        ("what is the gcd of 12 and 8", "GCD", {"a": 12, "b": 8}),
        ("what is the lcm of 4 and 6", "LCM", {"a": 4, "b": 6}),
    ]
    for text, skill, args in cases:
        assert ConstructionEngine.parse_procedural_from_catalog(
            text, constructions
        ) == (skill, args)


def test_default_catalog_covers_relation_forms_without_fallback_regexes():
    constructions = GrammarRegistry.default()

    cases = [
        ("A hammer is used for hitting nails.", "hammer", "used_for", "hitting nail"),
        ("Fire causes smoke and heat.", "fire", "causes", "smoke"),
        ("Penguins cannot fly.", "penguin", "can", "fly"),
        ("Cities are in countries.", "city", "located_in", "country"),
        ("Tools are made of metal.", "tool", "made_of", "metal"),
        ("A truck has wheels, an engine, and headlights.", "truck", "has", "wheel"),
        ("All felines are carnivores.", "feline", "is_a", "carnivore"),
        ("Every tiger is a cat.", "tiger", "is_a", "cat"),
        ("An animal is not a vehicle.", "animal", "disjoint_with", "vehicle"),
        (
            "If an animal is a canine, then it is a vertebrate.",
            "canine",
            "is_a",
            "vertebrate",
        ),
        (
            "If mammals are animals, then they are living things.",
            "animal",
            "is_a",
            "living thing",
        ),
        ("I am Aswin.", "user", "has_name", "aswin"),
        ("My name's Aswin.", "user", "has_name", "aswin"),
    ]
    for text, subject, predicate, object_ in cases:
        parsed = ConstructionEngine.parse_from_catalog(text, constructions)
        assert (parsed[0].subject, parsed[0].predicate, parsed[0].object_) == (
            subject,
            predicate,
            object_,
        )
