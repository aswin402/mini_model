"""Integration tests verifying Neural Semantic Frame extraction through CognitiveKernel and chat."""

from little.core.kernel import CognitiveKernel
from little.core.models import UpdateType
from little.memory.store import MemoryStore


def test_conversational_filler_learning_and_inference():
    """Verify conversational statement with filler prefixes is learned and answered correctly."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Learn with conversational filler prefixes and adverbs
    reply_learn = kernel.chat("Did you know that an axolotl is actually a type of amphibian?")
    assert "learned" in reply_learn.lower() or "understood" in reply_learn.lower() or "amphibian" in reply_learn.lower()

    # 2. Ask question about the newly acquired knowledge
    reply_ask = kernel.chat("Is an axolotl an amphibian?")
    assert "yes" in reply_ask.lower()
    assert "amphibian" in reply_ask.lower()


def test_parenthetical_relative_clause_learning():
    """Verify non-restrictive relative clause extracts taxonomic fact and answers subsequent query."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Statement with parenthetical relative clause
    reply_learn = kernel.chat("Cheetahs, which are felines, run fast.")
    assert "learned" in reply_learn.lower() or "understood" in reply_learn.lower() or "feline" in reply_learn.lower()

    # 2. Query the taxonomic relationship
    reply_ask = kernel.chat("Is a cheetah a feline?")
    assert "yes" in reply_ask.lower()
    assert "feline" in reply_ask.lower()


def test_passive_voice_knowledge_learning():
    """Verify passive construction maps to active relation in knowledge graph."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    outcome = kernel.process("Euclidean geometry was discovered by Euclid.")
    assert outcome.learning is not None
    assert outcome.learning.update_type in (
        UpdateType.NEW_RELATION,
        UpdateType.NEW_CONCEPT,
        UpdateType.EVIDENCE_ADDITION,
    )
    assert len(outcome.learning.relations_created) >= 1
    assert "discovered" in outcome.learning.relations_created[0]
    assert "euclid" in outcome.learning.relations_created[0]


def test_celestial_relative_clause_and_grounded_recall():
    """Verify celestial dynamic fact with appositive is learned and recalled."""
    memory = MemoryStore(":memory:", seed_ontology=True)
    kernel = CognitiveKernel(memory=memory)

    # 1. Ingest complex sentence
    reply_learn = kernel.chat("Actually, Europa, which is a moon, orbits Jupiter.")
    assert "learned" in reply_learn.lower() or "understood" in reply_learn.lower() or "moon" in reply_learn.lower() or "jupiter" in reply_learn.lower()

    # 2. Verify moon taxonomy
    reply_moon = kernel.chat("Is Europa a moon?")
    assert "yes" in reply_moon.lower()


def test_verification_gates_prevent_hallucination_on_noisy_inputs():
    """Verify 4 invariant gates reject contradictory claims even when extracted by neural frame extractor."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    # 1. Establish base taxonomic and disjoint facts
    kernel.process("A plant is disjoint with an animal.")
    kernel.process("An apple is a plant.")

    # 2. Try asserting disjoint category: apple is an animal
    outcome = kernel.process("Did you know that an apple is actually a type of animal?")

    # The claim is extracted, but verification gate I_MUTEX rejects the contradictory assignment
    assert outcome.learning is not None
    assert outcome.learning.update_type == UpdateType.NO_OP
    assert len(outcome.learning.relations_created) == 0
