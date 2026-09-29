"""Integration tests for Conversational Fluency & Multi-Turn English Chat."""

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_multi_turn_conversational_english_chat():
    """Verify a complete, human-like multi-turn conversational dialogue with 0% hallucination."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    # Turn 1: Greeting & Pleasantry
    turn1 = kernel.chat("How are you doing?")
    assert "operating smoothly" in turn1.lower() or "doing well" in turn1.lower()

    # Turn 2: Identity
    turn2 = kernel.chat("Who are you?")
    assert "LITTLE" in turn2
    assert "neuro-symbolic" in turn2.lower() or "cognitive architecture" in turn2.lower()

    # Turn 3: Mixed greeting + learning statement
    turn3 = kernel.chat("Hi, an axolotl is an amphibian.")
    assert "Hi!" in turn3
    assert "Got it! I've learned that an axolotl is an amphibian." in turn3

    # Turn 4: Transitive question
    turn4 = kernel.chat("Is an axolotl a living thing?")
    assert "Yes" in turn4
    assert "is indeed a living thing" in turn4

    # Turn 5: Refuted question
    turn5 = kernel.chat("Is an axolotl a vehicle?")
    assert "No" in turn5
    assert "is not a vehicle" in turn5

    # Turn 6: Capability inquiry
    turn6 = kernel.chat("Can an eagle fly?")
    assert "Yes" in turn6
    assert "can certainly fly" in turn6

    # Turn 7: Feature inquiry
    turn7 = kernel.chat("Does an eagle have wings?")
    assert "Yes" in turn7
    assert "has wings" in turn7

    # Turn 8: Comparative order reasoning
    kernel.chat("Cheetah is faster than horse.")
    kernel.chat("Horse is faster than snail.")
    turn8 = kernel.chat("Who is the fastest between cheetah, horse, and snail?")
    assert "Cheetah is the fastest" in turn8
    assert "cheetah >" in turn8

    # Turn 9: Multi-constraint algebraic word problem
    story = (
        "Elena is twice as old as Marco. "
        "In 7 years, the sum of their ages will be 50. "
        "How old is Elena?"
    )
    turn9 = kernel.chat(story)
    assert "Elena is 24 years old" in turn9

    # Turn 10: Epistemic honesty on novel entity
    turn10 = kernel.chat("Is a quokka an amphibian?")
    assert "I don't have that information in my memory yet" in turn10
    assert "quokka" in turn10

    # Turn 11: Gratitude
    turn11 = kernel.chat("Thank you very much!")
    assert "welcome" in turn11.lower() or "glad" in turn11.lower()

    # Turn 12: Farewell
    turn12 = kernel.chat("Goodbye!")
    assert "goodbye" in turn12.lower() or "safely preserved" in turn12.lower()


def test_multi_turn_anaphora_and_concept_summarization():
    """Verify discourse anaphora resolution, concept summaries, creator identity, and memory recall."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    # 1. Learning with pronouns
    kernel.chat("A cheetah is a mammal. Cheetahs can run fast.")
    turn_pronoun1 = kernel.chat("Can it run fast?")
    assert "cheetah can certainly run fast" in turn_pronoun1.lower()

    turn_pronoun2 = kernel.chat("Is it an animal?")
    assert "cheetah is indeed an animal" in turn_pronoun2.lower()

    turn_pronoun3 = kernel.chat("Can it fly?")
    assert "cheetah cannot fly" in turn_pronoun3.lower()

    # 2. Concept Summarization & Definition Verbalization
    turn_summary = kernel.chat("Summarize cheetah.")
    assert "cheetah is a mammal" in turn_summary.lower()

    turn_bird_summary = kernel.chat("What do you know about bird?")
    assert "bird is a vertebrate" in turn_bird_summary.lower()
    assert "feather" in turn_bird_summary.lower()
    assert "mammal" in turn_bird_summary.lower()

    # 3. Creator & Capabilities
    turn_creator = kernel.chat("Who created you?")
    assert "LITTLE (MIVI)" in turn_creator
    assert "neuro-symbolic" in turn_creator.lower()

    turn_caps = kernel.chat("What can you do?")
    assert "deductive" in turn_caps.lower() or "learn" in turn_caps.lower()

    # 4. User profile name introduction and recall
    turn_intro = kernel.chat("My name is Aswin.")
    assert "Aswin" in turn_intro
    assert "recorded" in turn_intro.lower() or "nice to meet you" in turn_intro.lower()

    turn_recall = kernel.chat("Do you remember my name?")
    assert "Aswin" in turn_recall

