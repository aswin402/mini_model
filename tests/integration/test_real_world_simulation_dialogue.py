"""Integration test verifying full 25-turn real-world simulation of LITTLE (MIVI).

Covers multi-domain conversation:
- Social greetings, identity & capabilities
- Persistent episodic user memory & recall
- Multi-word & proper noun astronomy learning & query
- Honest open-world admissions (UNKNOWN)
- Biology learning, multi-word refutation & discourse anaphora
- Capability queries (affirmative & negative)
- Knowledge graph concept definition summarization
- Comparative poset ordering & transitive deductions
- Algebraic linear word problems (proportional quantity & cost systems)
- Exact CAS procedural mathematics
- Continuous physical CfC ODE decay simulation & counterfactual analysis
- Social compliments & farewell
"""

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_25_turn_real_world_simulation():
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Social Greeting
    r1 = kernel.chat("Hello there! How are you doing today?")
    assert "operating smoothly" in r1

    # 2. Identity & Capabilities
    r2 = kernel.chat("Who are you and what can you do?")
    assert "I am LITTLE" in r2
    assert "Multi-Hop Deductive Reasoning" in r2
    assert not r2.startswith("[")  # Not list formatted

    # 3. User Name Learning
    r3 = kernel.chat("My name is Aswin.")
    assert "Nice to meet you, Aswin!" in r3

    # 4. Memory Recall
    r4 = kernel.chat("Do you remember who I am?")
    assert "Aswin" in r4

    # 5. Astronomy Learning (Proper Noun + Multi-word)
    r5 = kernel.chat("Europa is an icy moon.")
    assert "learned that Europa is an icy moon" in r5

    # 6. Astronomy Relation
    r6 = kernel.chat("Europa orbits Jupiter.")
    assert "learned that Europa orbits Jupiter" in r6

    # 7. Taxonomy Query (Proper Noun Article Test)
    r7 = kernel.chat("Is Europa an icy moon?")
    assert r7 == "Yes, Europa is indeed an icy moon."

    # 8. Affirmative Relation Query
    r8 = kernel.chat("Does Europa orbit Jupiter?")
    assert r8 == "Yes, Europa orbits Jupiter."

    # 9. Epistemic Unknown Query (Zero-Hallucination)
    r9 = kernel.chat("Does Europa have an atmosphere?")
    assert "don't have that information in my memory yet" in r9

    # 10. Multi-Word Biology Learning
    r10 = kernel.chat("A snow leopard is a mammal.")
    assert "learned that A snow leopard is a mammal" in r10

    # 11. Biology Feature Learning
    r11 = kernel.chat("A snow leopard has thick fur.")
    assert "learned that A snow leopard has thick fur" in r11

    # 12. Refutation on Multi-Word Entity
    r12 = kernel.chat("Is a snow leopard a reptile?")
    assert r12 == "No, a snow leopard is not a reptile."

    # 13. Discourse Anaphora Resolution ("it" -> snow leopard)
    r13 = kernel.chat("Does it have thick fur?")
    assert r13 == "Yes, a snow leopard has thick fur."

    # 14. Negative Capability Query
    r14 = kernel.chat("Can a penguin fly?")
    assert "penguin cannot fly" in r14

    # 15. Affirmative Capability Query
    r15 = kernel.chat("Can a sparrow fly?")
    assert "sparrow can certainly fly" in r15

    # 16. Concept Definition Summarization
    r16 = kernel.chat("Describe what you know about sparrow.")
    assert "A sparrow is a bird" in r16

    # 17. Comparative Poset Extremum Query
    r17 = kernel.chat(
        "The cheetah is faster than the lion, and the lion is faster than the elephant. Who is the fastest?"
    )
    assert "Cheetah is the fastest" in r17

    # 18. Transitive Poset Comparison
    r18 = kernel.chat("Is the cheetah faster than the elephant?")
    assert "Yes. Cheetah is faster than Lion" in r18
    assert "Therefore, Cheetah is faster than Elephant" in r18

    # 19. Algebraic Proportional Word Problem
    r19 = kernel.chat(
        "Lucas has 3 times as many books as Maya. Together they have 48 books. How many books does Lucas have?"
    )
    assert r19 == "Lucas has 36 books."

    # 20. Cost Linear System
    r20 = kernel.chat(
        "A pen costs 3 dollars more than a notebook. 2 pens and 3 notebooks cost 21 dollars. What is the cost of a pen?"
    )
    assert r20 == "The cost of a pen is $6."

    # 21. Symbolic CAS Factorial
    r21 = kernel.chat("What is 7 factorial?")
    assert r21 == "5040"

    # 22. Continuous Physical ODE Simulation
    r22 = kernel.chat("Simulate Apple decay over 30 days.")
    assert "Hypothetical Scenario for 'apple'" in r22
    assert "spoiled" in r22

    # 23. Counterfactual ODE Simulation
    r23 = kernel.chat("What if temperature is 40C vs 10C for Apple?")
    assert "Counterfactual Analysis for 'apple slice':" in r23
    assert "Arrhenius thermal kinetics" in r23

    # 24. Social Compliment
    r24 = kernel.chat("You did a great job!")
    assert "Thank you!" in r24

    # 25. Social Goodbye
    r25 = kernel.chat("Goodbye for now!")
    assert "Goodbye!" in r25
