"""Integration tests for Typed Tool Actions within CognitiveKernel conversational chat."""

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_tool_actions_conversational_chat():
    """Verify tool calls execute cleanly and verbalize naturally in chat."""
    store = MemoryStore(":memory:")
    kernel = CognitiveKernel(memory=store)

    # 1. Direct calculator tool chat
    ans1 = kernel.chat("calculate 15 * 12 + 10")
    assert "190" in ans1

    # 2. Python REPL tool chat
    ans2 = kernel.chat("run python: print(sum([1, 2, 3, 4, 5]))")
    assert "15" in ans2
    assert "Python execution output" in ans2

    # 3. System time tool chat
    ans3 = kernel.chat("what time is it?")
    assert "The current time is" in ans3

    # 4. Web search tool chat
    ans4 = kernel.chat("search for Europa water vapor")
    assert "Europa water vapor" in ans4
    assert "search results" in ans4

    # 5. Mixed social greeting + tool call
    ans5 = kernel.chat("Hello! What is 25 * 4")
    assert "Hello!" in ans5
    assert "100" in ans5

    # 6. Verify tool execution grounding in SQLite memory
    exp_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM experiences WHERE source = 'tool_execution';"
    ).fetchone()["c"]
    assert exp_count >= 5

    # Concepts generated for tools
    tool_concept = store.get_concept("tool_calculator")
    assert tool_concept is not None
