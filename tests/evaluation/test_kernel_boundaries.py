from little.core.kernel import CognitiveKernel
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


def test_perception_is_side_effect_free_for_new_domain_words():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = DeterministicPerceptionAdapter()
    before = (len(memory.list_concepts()), memory.count_relations())

    frame = adapter.perceive("A zephyr is a vehicle.", memory=memory)

    assert frame.claims[0].subject == "zephyr"
    assert (len(memory.list_concepts()), memory.count_relations()) == before


def test_unknown_and_unregistered_inputs_create_no_edges():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)

    unknown = kernel.process("qwerty zorp")
    assert unknown.inference is not None
    assert unknown.inference.status.value == "UNKNOWN"
    assert memory.count_relations() == 0

    before_unsupported = (len(memory.list_concepts()), memory.count_relations())
    unsupported = kernel.process("A zephyr invents a relation.")
    assert unsupported.learning is not None
    assert (len(memory.list_concepts()), memory.count_relations()) == before_unsupported
    assert memory.evidence.list_by_source("user")[-1].status.value == "candidate"


def test_same_parser_path_handles_multiple_unlisted_concepts():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)

    first = kernel.process("A zephyr is a vehicle.")
    second = kernel.process("A lantern is an object.")

    assert first.learning is not None
    assert second.learning is not None
    assert memory.find_relation_by_names("zephyr", "is_a", "vehicle") is not None
    assert memory.find_relation_by_names("lantern", "is_a", "object") is not None


def test_deterministic_fallback_identity_is_not_laya():
    memory = MemoryStore(":memory:", seed_ontology=False)

    frame = DeterministicPerceptionAdapter().perceive(
        "A zephyr is a vehicle.", memory=memory
    )

    assert frame.model_id == "deterministic-parser"
    assert frame.model_id != "laya"
