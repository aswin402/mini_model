"""Integration tests verifying Neural Router integration into CognitiveKernel and conversational chat."""

from little.core.kernel import CognitiveKernel
from little.inference.thinking_controller import CognitiveMode
from little.memory.store import MemoryStore
from little.neural.models import Pathway


def test_casual_slang_math_chat_execution():
    """Verify casual slang math queries route through neural router and execute calculator tool."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    # 1. Slang multiplication
    reply1 = kernel.chat("yo multiply 15 by 4 real quick")
    assert "60" in reply1

    # 2. Natural language multiplication
    reply2 = kernel.chat("what is 15 times 4")
    assert "60" in reply2

    # 3. Addition
    reply3 = kernel.chat("add 25 and 75")
    assert "100" in reply3

    # 4. Division
    reply4 = kernel.chat("divide 100 by 5")
    assert "20" in reply4


def test_conversational_tool_chat_execution():
    """Verify casual tool queries route through neural router and execute sandboxed tools."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    # 1. Current time & date
    reply_time = kernel.chat("what is the current time and date today")
    assert "System time:" in reply_time or "UTC" in reply_time or "timestamp" in reply_time.lower()

    # 2. Web search
    reply_search = kernel.chat("search the web for europa water ocean")
    assert "europa water ocean" in reply_search.lower()


def test_neural_routing_process_outcome_metadata():
    """Verify KernelOutcome includes routing decision, pathway probabilities, and uncertainty."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    outcome = kernel.process("multiply 100 by 500 calculate sum")
    assert outcome.routing_decision is not None
    assert outcome.routing_decision.pathway == Pathway.MATH
    assert outcome.routing_decision.confidence > 0.50
    assert outcome.tool_result is not None
    assert outcome.tool_result.output == 50000


def test_epistemic_uncertainty_in_thinking_mode():
    """Verify ambiguous multi-intent query triggers thinking mode due to high entropy."""
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory=memory)

    # Learn some base facts first
    kernel.process("A cheetah is a feline.")
    kernel.process("A falcon is a bird.")

    outcome = kernel.process("def calculate time search")
    assert outcome.routing_decision is not None
    assert outcome.routing_decision.is_uncertain is True
