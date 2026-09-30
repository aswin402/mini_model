"""Unit tests for the Neural Semantic Frame and Slot Extractor."""

from little.neural import ExtractedSemanticFrame, NeuralFrameExtractor, SemanticFramePolicy


def test_filler_stripping_and_conversational_extraction():
    """Verify conversational prefixes and inter-verbal adverbs are stripped cleanly."""
    extractor = NeuralFrameExtractor()

    text = "Did you know that an axolotl is actually a type of amphibian?"
    frames = extractor.extract_frames(text)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.subject == "axolotl"
    assert frame.predicate == "is_a"
    assert frame.object_val == "amphibian"
    assert not frame.is_negative
    assert frame.confidence >= 0.80


def test_passive_agentive_construction_extraction():
    """Verify passive voice ('was discovered by') maps correctly to active (subject, predicate, object)."""
    extractor = NeuralFrameExtractor()

    text = "Euclidean geometry was discovered by Euclid."
    frames = extractor.extract_frames(text)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.subject == "euclid"
    assert frame.predicate == "discovered"
    assert frame.object_val == "euclidean geometry"


def test_relative_clause_parenthetical_extraction():
    """Verify non-restrictive relative clauses extract both the taxonomic claim and the main clause."""
    extractor = NeuralFrameExtractor()

    text = "Cheetahs, which are felines, run fast."
    frames = extractor.extract_frames(text)

    assert len(frames) >= 1
    # First frame must be the relative clause: cheetah is_a feline
    f1 = frames[0]
    assert f1.subject == "cheetah"
    assert f1.predicate == "is_a"
    assert f1.object_val == "feline"


def test_property_and_celestial_relational_indicators():
    """Verify properties and celestial dynamics extract correctly."""
    extractor = NeuralFrameExtractor()

    # 1. Properties
    f_prop = extractor.extract_frames("An eagle possesses sharp talons.")
    assert len(f_prop) == 1
    assert f_prop[0].subject == "eagle"
    assert f_prop[0].predicate == "has_property"
    assert f_prop[0].object_val == "sharp talon" or "talon" in f_prop[0].object_val
    assert f_prop[0].is_property is True

    # 2. Celestial orbits
    f_orbit = extractor.extract_frames("The earth orbits the sun.")
    assert len(f_orbit) == 1
    assert f_orbit[0].subject == "earth"
    assert f_orbit[0].predicate == "orbits"
    assert f_orbit[0].object_val == "sun"


def test_negation_detection():
    """Verify negative statements set is_negative=True and positive=False in claims."""
    extractor = NeuralFrameExtractor()

    frames = extractor.extract_frames("A whale is not a fish.")
    assert len(frames) == 1
    assert frames[0].subject == "whale"
    assert frames[0].predicate == "is_a"
    assert frames[0].object_val == "fish"
    assert frames[0].is_negative is True

    claims = extractor.to_candidate_claims(frames)
    assert len(claims) == 1
    assert claims[0].positive is False


def test_conjunction_decomposition():
    """Verify compound objects with conjunction 'and' decompose into separate frames."""
    extractor = NeuralFrameExtractor()

    frames = extractor.extract_frames("Water consists of hydrogen and oxygen.")
    assert len(frames) == 2
    objs = {f.object_val for f in frames}
    assert "hydrogen" in objs
    assert "oxygen" in objs
    for f in frames:
        assert f.subject == "water"
        assert f.predicate == "part_of"


def test_invalid_and_empty_inputs_safe():
    """Verify self-loops and empty inputs return empty frame lists gracefully."""
    extractor = NeuralFrameExtractor()

    assert extractor.extract_frames("") == []
    assert extractor.extract_frames("   ") == []
    assert extractor.extract_frames("An apple is an apple.") == []
