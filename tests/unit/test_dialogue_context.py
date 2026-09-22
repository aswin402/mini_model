import pytest
from little.language.dialogue import DialogueContext, SalientEntity


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
