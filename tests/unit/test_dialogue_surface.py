"""Unit tests for DialogueSurfaceEngine, DialogueSurfacePolicy, and Conversational Chat."""

from pathlib import Path

from little.core.kernel import CognitiveKernel
from little.language.dialogue_surface import DialogueSurfaceEngine
from little.language.dialogue_surface_policy import DialogueSurfacePolicy
from little.memory.store import MemoryStore


def test_dialogue_surface_policy_loading():
    """Verify that dialogue surface policy loads dynamically from schema with zero hardcoding."""
    policy = DialogueSurfacePolicy.load(Path("data/schemas"))
    assert "greeting" in policy.social_intents
    assert "how_are_you" in policy.social_intents
    assert "thank_you" in policy.social_intents
    assert "goodbye" in policy.social_intents
    assert "identity" in policy.social_intents
    assert policy.leading_greeting_regex != ""
    assert "affirmative_taxonomy" in policy.verbalizer_templates
    assert "learning_confirmation" in policy.verbalizer_templates


def test_pure_social_intents():
    """Verify that pure greetings and pleasantries return schema-driven warm replies."""
    engine = DialogueSurfaceEngine()

    # Greetings
    for text in ["Hello", "Hi!", "hey there", "Good morning", "Greetings"]:
        reply = engine.handle_social(text)
        assert reply is not None
        assert any(
            greet_word in reply.lower()
            for greet_word in ["hello", "hi", "greetings", "help", "explore"]
        )

    # How are you
    reply_how = engine.handle_social("How are you doing?")
    assert reply_how is not None
    assert "operating" in reply_how.lower() or "well" in reply_how.lower()

    # Thank you
    reply_thanks = engine.handle_social("Thank you very much!")
    assert reply_thanks is not None
    assert "welcome" in reply_thanks.lower() or "glad" in reply_thanks.lower()

    # Goodbye
    reply_bye = engine.handle_social("Goodbye!")
    assert reply_bye is not None
    assert "goodbye" in reply_bye.lower() or "care" in reply_bye.lower() or "time" in reply_bye.lower()

    # Identity
    reply_who = engine.handle_social("Who are you?")
    assert reply_who is not None
    assert "LITTLE" in reply_who


def test_mixed_utterance_decomposition():
    """Verify that mixed greetings like 'Hello! Can an eagle fly?' separate correctly."""
    engine = DialogueSurfaceEngine()

    greeting, remainder = engine.decompose_utterance("Hello! Can an eagle fly?")
    assert greeting == "Hello"
    assert remainder == "Can an eagle fly?"

    greeting, remainder = engine.decompose_utterance("Hi, a flamingo is a bird.")
    assert greeting == "Hi"
    assert remainder == "a flamingo is a bird."

    greeting, remainder = engine.decompose_utterance("Can an eagle fly?")
    assert greeting is None
    assert remainder == "Can an eagle fly?"


def test_composed_chat_learning_and_inference():
    """Verify end-to-end chat flow through CognitiveKernel and LearningEngine."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    # 1. Social greeting
    res_greet = kernel.chat("Hello there!")
    assert "help" in res_greet.lower() or "what" in res_greet.lower() or "ready" in res_greet.lower()

    # 2. Mixed learning utterance
    res_learn = kernel.chat("Hi, an axolotl is an amphibian.")
    assert "Hi!" in res_learn
    assert "learned" in res_learn.lower()
    assert "axolotl is an amphibian" in res_learn.lower()

    # 3. Mixed question
    res_ask = kernel.chat("Hello! Can an eagle fly?")
    assert "Hello!" in res_ask
    assert "Yes" in res_ask
    assert "can certainly fly" in res_ask

    # 4. Pure question
    res_pure = kernel.chat("Is a dog an animal?")
    assert "Yes" in res_pure
    assert "is indeed an animal" in res_pure

    # 5. Refuted question
    res_refuted = kernel.chat("Is a dog a vehicle?")
    assert "No" in res_refuted
    assert "is not a vehicle" in res_refuted

    # 6. Epistemic unknown
    res_unknown = kernel.chat("Is an alien an animal?")
    assert "I don't have that information in my memory yet" in res_unknown
    assert "alien" in res_unknown

    # 7. Gratitude and farewell
    res_thanks = kernel.chat("Thanks a lot!")
    assert "welcome" in res_thanks.lower() or "glad" in res_thanks.lower() or "anytime" in res_thanks.lower()

    res_bye = kernel.chat("Goodbye!")
    assert "goodbye" in res_bye.lower() or "care" in res_bye.lower() or "time" in res_bye.lower()


def test_multi_clause_social_utterances():
    """Verify compound social statements like greeting + pleasantry, and gratitude + goodbye."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    res1 = kernel.chat("Hello there! How are you doing today?")
    assert "hello" in res1.lower()
    assert "operating" in res1.lower() or "well" in res1.lower()

    res2 = kernel.chat("Thank you so much for your help! Goodbye!")
    assert "welcome" in res2.lower()
    assert "goodbye" in res2.lower() or "preserved" in res2.lower()

