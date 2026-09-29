"""Unit tests for the Neural State Encoder, Learned Router, and Calibrated Uncertainty."""

import math

from little.neural import NeuralRouter, NeuralRouterPolicy, NeuralStateEncoder, Pathway


def test_neural_state_encoder_dimensions_and_norm():
    """Verify encoder produces a continuous unit vector S in R^512."""
    encoder = NeuralStateEncoder()
    state = encoder.encode("Calculate 15 * 4 and check memory")

    assert len(state.embedding) == 512
    norm = math.sqrt(sum(x * x for x in state.embedding))
    assert abs(norm - 1.0) < 1e-5
    assert len(state.tokens) > 0


def test_cosine_similarity_on_semantic_variants():
    """Verify semantic paraphrases have strong cosine similarity."""
    encoder = NeuralStateEncoder()

    s1 = encoder.encode("multiply 15 by 4")
    s2 = encoder.encode("what is 15 times 4")
    s3 = encoder.encode("who is your creator")

    # Cosine similarity between s1 and s2
    dot_math = sum(a * b for a, b in zip(s1.embedding, s2.embedding, strict=True))
    dot_unrelated = sum(a * b for a, b in zip(s1.embedding, s3.embedding, strict=True))

    assert dot_math > dot_unrelated
    assert dot_math > 0.40


def test_neural_router_classifies_casual_and_slang_queries():
    """Verify neural router classifies queries across the 6 cognitive pathways without regex."""
    router = NeuralRouter()

    # 1. Math pathway (slang, word problem, operation)
    d_math1 = router.route("yo multiply 15 by 4 real quick")
    assert d_math1.pathway == Pathway.MATH
    assert d_math1.confidence > 0.35

    d_math2 = router.route("can you solve equation 2x + 10 = 30")
    assert d_math2.pathway == Pathway.MATH

    # 2. Tool pathway (search, time, shell)
    d_tool1 = router.route("search the web for europa water ocean")
    assert d_tool1.pathway == Pathway.TOOL

    d_tool2 = router.route("what is the current time and date today")
    assert d_tool2.pathway == Pathway.TOOL

    # 3. Code pathway
    d_code = router.route("write a python def to reverse a string algorithm")
    assert d_code.pathway == Pathway.CODE

    # 4. Poset pathway (comparative)
    d_poset = router.route("who is faster cheetah or falcon")
    assert d_poset.pathway == Pathway.POSET

    # 5. Physics pathway (continuous dynamics)
    d_physics = router.route("simulate apple decay over 5 days with temperature")
    assert d_physics.pathway == Pathway.PHYSICS

    # 6. Chat pathway
    d_chat = router.route("hello my friend how are you doing today")
    assert d_chat.pathway == Pathway.CHAT


def test_entropy_and_uncertainty_quantification():
    """Verify Shannon entropy flags high uncertainty when multiple pathways compete."""
    router = NeuralRouter()

    # Very clear math query -> low entropy
    d_clear = router.route("multiply 100 by 500 calculate sum")
    assert d_clear.pathway == Pathway.MATH
    assert d_clear.confidence > 0.50

    # Ambiguous mixed query spanning math, code, tool, and chat
    d_ambiguous = router.route("def calculate time search")
    assert d_ambiguous.entropy > 0.30


def test_online_lifelong_adaptation():
    """Verify router can adapt its weights continuously via online learning."""
    router = NeuralRouter()
    encoder = router.encoder

    # Initial query that might be slightly ambiguous or novel
    query = "crunch numbers on dataset matrix"
    state = encoder.encode(query)
    initial_decision = router.route(state)

    # Perform online gradient steps targeting MATH pathway
    for _ in range(5):
        router.learn(state, Pathway.MATH, learning_rate=0.2)

    adapted_decision = router.route(state)
    assert adapted_decision.pathway == Pathway.MATH
    assert adapted_decision.probabilities[Pathway.MATH] > initial_decision.probabilities.get(Pathway.MATH, 0.0)

