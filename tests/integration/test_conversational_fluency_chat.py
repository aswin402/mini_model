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


def test_conversational_teaching_and_explanatory_chat():
    """Verify teaching, explanatory definitions, multi-hop deduction, math, and parting in chat dialogue."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    # 1. Courtesy greeting
    r1 = kernel.chat("Hello! Nice to meet you.")
    assert "hello" in r1.lower()
    assert "nice to meet you" in r1.lower() or "pleasure" in r1.lower()

    # 2. Compound identity & learning inquiry
    r2 = kernel.chat("Can you tell me about yourself and how do you learn?")
    assert "LITTLE" in r2
    assert "dialogue" in r2.lower() or "neuro-symbolic" in r2.lower()

    # 3. Epistemic unknown before teaching
    r3 = kernel.chat("Tell me what you know about ocelot.")
    assert "don't have that information in my memory yet" in r3.lower()

    # 4. Teaching taxonomy
    r4 = kernel.chat("An ocelot is a feline.")
    assert "learned that An ocelot is a feline" in r4

    # 5. Teaching features
    r5 = kernel.chat("An ocelot has sharp claws and nocturnal vision.")
    assert "learned that An ocelot has sharp claws and nocturnal vision" in r5

    # 6. Teaching disjoint invariant
    r6 = kernel.chat("An ocelot is not a canine.")
    assert "learned that An ocelot is not a canine" in r6

    # 7. Conversational explanatory definition
    r7 = kernel.chat("Describe what you know about ocelot.")
    assert "ocelot is a feline" in r7.lower()
    assert "sharp claw" in r7.lower()
    assert "nocturnal vision" in r7.lower()
    assert "canine" in r7.lower()
    assert "latent_" not in r7.lower()

    # 8. Alternative explanatory query
    r8 = kernel.chat("Explain what an ocelot is.")
    assert "ocelot is a feline" in r8.lower()
    assert "latent_" not in r8.lower()

    # 9. Multi-hop transitive deduction (ocelot -> feline -> mammal -> animal)
    r9 = kernel.chat("Is an ocelot an animal?")
    assert "Yes" in r9
    assert "is indeed an animal" in r9.lower()

    # 10. Capability refutation
    r10 = kernel.chat("Can an ocelot fly?")
    assert "No" in r10
    assert "cannot fly" in r10.lower()

    # 11. Natural arithmetic
    r11 = kernel.chat("What is 15% of 800?")
    assert "120" in r11

    # 12. Strict poset reasoning
    r12 = kernel.chat("The ocelot is faster than the badger, and the cheetah is faster than the ocelot. Who is the fastest?")
    assert "Cheetah is the fastest" in r12

    # 13. Gratitude & compound farewell
    r13 = kernel.chat("Thank you so much for chatting with me! Goodbye for now!")
    assert "welcome" in r13.lower()
    assert "goodbye" in r13.lower() or "preserved" in r13.lower()


def test_metacognitive_introspection_and_history_recall():
    """Verify informal greeting normalization, identity, math introspection, concept queries, working memory recall, and assurance."""
    store = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(store)

    # 1. Informal lengthened greeting
    t1 = kernel.chat("hii")
    assert "hello" in t1.lower() or "hi" in t1.lower()
    assert "based on my knowledge" not in t1.lower()

    # 2. Discourse marker + slang identity
    t2 = kernel.chat("so who are u")
    assert "LITTLE (MIVI)" in t2
    assert "1. 🧠 Multi-Hop Deductive" not in t2

    # 3. Arithmetic query
    t3 = kernel.chat("so whats 4+4")
    assert "8" in t3

    # 4. Introspective reasoning explanation of prior calculation
    t4 = kernel.chat("how u get that answer")
    assert "4+4" in t4
    assert "deterministic" in t4.lower() or "arithmetic" in t4.lower() or "8" in t4

    # 5. Concept inquiry 1
    t5 = kernel.chat("so what is a bird")
    assert "bird is a vertebrate" in t5.lower()
    assert "feather" in t5.lower()

    # 6. Concept inquiry 2
    t6 = kernel.chat("so what is animal")
    assert "animal is an organism" in t6.lower()

    # 7. Episodic history recall from working memory
    t7 = kernel.chat("so what i asked before")
    assert "so what is animal" in t7
    assert "so what is a bird" in t7

    # 8. Reassurance and invariant sanity
    t8 = kernel.chat("is we done anything wrong")
    assert "not at all" in t8.lower()
    assert "consistent" in t8.lower() or "sound" in t8.lower()

    # 9. Multi-hop proof trace introspection
    kernel.chat("An ocelot is a feline.")
    t9_ans = kernel.chat("Is an ocelot an animal?")
    assert "Yes" in t9_ans
    t9_why = kernel.chat("how did you know that?")
    assert "OCELOT" in t9_why
    assert "ANIMAL" in t9_why


