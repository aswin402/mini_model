from types import SimpleNamespace

from little.language.dialogue import DialogueContext


def _reference_policy(**overrides):
    values = {
        "former_references": ("the former", "former"),
        "latter_references": ("the latter", "latter"),
        "plural_pronouns": ("they", "them", "their", "theirs", "these", "those"),
        "person_pronouns": ("he", "him", "his", "she", "her", "hers"),
        "singular_nonhuman_pronouns": ("it", "its", "that", "this"),
        "text_pronouns": ("it", "they", "them", "that", "these", "those"),
        "relative_pronouns": ("that", "these", "those"),
        "relative_prefix_exclusions": (
            "is", "are", "was", "were", "do", "does", "did",
            "if", "and", "or", "but", "so", "than", "as", "like",
        ),
        "relative_suffix_exclusions": (
            "is", "are", "was", "were", "has", "have", "can",
            "will", "would", "do", "does", "did", "in", "on", "at", "?", ".",
        ),
        "person_categories": ("person", "human", "character"),
        "conditional_guard_pattern": r"^if\s+",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dialogue_pronoun_resolution_uses_injected_reference_catalog():
    ctx = DialogueContext(_reference_policy(singular_nonhuman_pronouns=()))
    ctx.register_entity("falcon", category="animal", is_animate=True)

    assert ctx.resolve_pronoun("it") is None


def test_dialogue_pronoun_resolution_uses_custom_reference():
    ctx = DialogueContext(
        _reference_policy(singular_nonhuman_pronouns=("yon",))
    )
    ctx.register_entity("falcon", category="animal", is_animate=True)

    assert ctx.resolve_pronoun("yon") == "falcon"


def test_dialogue_entity_registration_and_pronoun_resolution():
    ctx = DialogueContext()

    # Turn 1: Register "falcon"
    ctx.record_turn("user", "The falcon is a fierce raptor.")
    ctx.register_entity("falcon", category="animal", is_animate=True, is_plural=False)

    # Resolve "it" should find "falcon"
    assert ctx.resolve_pronoun("it") == "falcon"
    assert ctx.resolve_pronoun("that") == "falcon"

    # Turn 2: Register plural "wolves"
    ctx.record_turn("user", "Wolves hunt in packs.")
    ctx.register_entity("wolves", category="animal", is_animate=True, is_plural=True)

    # Plural pronoun "they" should resolve to "wolves"
    assert ctx.resolve_pronoun("they") == "wolves"
    assert ctx.resolve_pronoun("them") == "wolves"

    # Singular pronoun "it" should still resolve to "falcon" (most recent singular)
    assert ctx.resolve_pronoun("it") == "falcon"


def test_resolve_anaphora_in_text():
    ctx = DialogueContext()
    ctx.register_entity("cheetah", category="animal", is_animate=True, is_plural=False)

    resolved_text = ctx.resolve_anaphora_in_text("It is fast and it runs in Africa.")
    assert "cheetah" in resolved_text.lower()
    assert "it" not in resolved_text.lower()


def test_the_former_and_the_latter_resolution():
    ctx = DialogueContext()
    ctx.record_turn("user", "Compare the sun and the moon.")
    ctx.register_entity("sun", category="celestial_body", is_plural=False)
    ctx.register_entity("moon", category="celestial_body", is_plural=False)

    assert ctx.resolve_pronoun("the former") == "sun"
    assert ctx.resolve_pronoun("the latter") == "moon"
