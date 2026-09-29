"""Comprehensive unit tests for the Typed Action & Tool Core (Needle-style execution)."""

from pathlib import Path

from little.actions import (
    ActionPolicy,
    ActionRouter,
    PreDispatchVerifier,
    ToolCall,
    ToolDefinitionsPolicy,
    ToolExecutor,
    ToolGroundingEngine,
)
from little.memory.store import MemoryStore


def test_tool_definitions_and_action_policy_load():
    """Verify policies load properly from versioned data/schemas."""
    tool_policy = ToolDefinitionsPolicy.default()
    action_policy = ActionPolicy.default()

    assert "calculator" in tool_policy.tools
    assert "python_repl" in tool_policy.tools
    assert "system_time" in tool_policy.tools
    assert "web_search" in tool_policy.tools
    assert "file_reader" in tool_policy.tools
    assert "file_writer" in tool_policy.tools
    assert "bash_executor" in tool_policy.tools

    assert "calculator" in action_policy.intent_patterns
    assert "echo" in action_policy.allowed_bash_binaries


def test_pre_dispatch_verifier_gates():
    """Verify I_sort, required params, and security invariants."""
    tool_policy = ToolDefinitionsPolicy.default()
    action_policy = ActionPolicy.default()
    verifier = PreDispatchVerifier(tool_policy, action_policy)

    # 1. Unknown tool
    res_unknown = verifier.verify(ToolCall(tool_name="nonexistent_tool", arguments={}))
    assert not res_unknown.valid
    assert "Unknown tool" in (res_unknown.error or "")

    # 2. Missing required parameter
    res_missing = verifier.verify(ToolCall(tool_name="calculator", arguments={}))
    assert not res_missing.valid
    assert "Missing required parameter" in (res_missing.error or "")

    # 3. Type mismatch (I_sort)
    res_type = verifier.verify(ToolCall(tool_name="calculator", arguments={"expression": 12345}))
    assert not res_type.valid
    assert "Type signature mismatch" in (res_type.error or "")

    # 4. Path traversal security gate
    res_traversal = verifier.verify(
        ToolCall(tool_name="file_reader", arguments={"file_path": "../../etc/passwd"})
    )
    assert not res_traversal.valid
    assert "Path traversal" in (res_traversal.error or "")

    # 5. Shell command blacklist gate
    res_unsafe_sh = verifier.verify(
        ToolCall(tool_name="bash_executor", arguments={"command": "rm -rf /"})
    )
    assert not res_unsafe_sh.valid
    assert "not in allowed whitelist" in (res_unsafe_sh.error or "")

    # 6. Valid call
    res_valid = verifier.verify(
        ToolCall(tool_name="calculator", arguments={"expression": "10 + 20"})
    )
    assert res_valid.valid
    assert res_valid.error is None


def test_tool_executor_builtin_tools(tmp_path: Path):
    """Verify safe sandboxed execution of all built-in tools."""
    executor = ToolExecutor(workspace_root=tmp_path)

    # 1. Calculator
    calc_res = executor.execute(ToolCall(tool_name="calculator", arguments={"expression": "14 * 28 + 5"}))
    assert calc_res.success
    assert calc_res.output == 397
    assert len(calc_res.verifiable_hash) == 16

    # 2. Python REPL
    py_res = executor.execute(
        ToolCall(tool_name="python_repl", arguments={"code": "print(2**10)"})
    )
    assert py_res.success
    assert py_res.output == "1024"

    # 3. System time
    time_res = executor.execute(ToolCall(tool_name="system_time", arguments={}))
    assert time_res.success
    assert "iso" in time_res.output
    assert "date" in time_res.output

    # 4. Web search
    search_res = executor.execute(
        ToolCall(tool_name="web_search", arguments={"query": "Europa icy moon", "max_results": 2})
    )
    assert search_res.success
    assert isinstance(search_res.output, list)
    assert len(search_res.output) == 2

    # 5. File writer & file reader
    write_res = executor.execute(
        ToolCall(
            tool_name="file_writer",
            arguments={"file_path": "test_output.txt", "content": "MIVI Cognitive OS"},
        )
    )
    assert write_res.success

    read_res = executor.execute(
        ToolCall(tool_name="file_reader", arguments={"file_path": "test_output.txt"})
    )
    assert read_res.success
    assert read_res.output == "MIVI Cognitive OS"

    # 6. Bash executor
    bash_res = executor.execute(
        ToolCall(tool_name="bash_executor", arguments={"command": "echo 'verified'"})
    )
    assert bash_res.success
    assert bash_res.output["stdout"] == "verified"


def test_action_router_intent_dispatch():
    """Verify natural language utterances are parsed into typed ToolCalls."""
    router = ActionRouter()

    # Calculator
    call1 = router.route("calculate 45 * 2")
    assert call1 is not None
    assert call1.tool_name == "calculator"
    assert call1.arguments["expression"] == "45 * 2"

    call2 = router.route("what is 100 / 4")
    assert call2 is not None
    assert call2.tool_name == "calculator"

    # Python REPL
    call3 = router.route("run python: print('test')")
    assert call3 is not None
    assert call3.tool_name == "python_repl"
    assert "print('test')" in call3.arguments["code"]

    # System time
    call4 = router.route("what time is it?")
    assert call4 is not None
    assert call4.tool_name == "system_time"

    # Web search
    call5 = router.route("search for Europa water ocean")
    assert call5 is not None
    assert call5.tool_name == "web_search"
    assert call5.arguments["query"] == "Europa water ocean"

    # File reader
    call6 = router.route("read file docs/README.md")
    assert call6 is not None
    assert call6.tool_name == "file_reader"
    assert call6.arguments["file_path"] == "docs/README.md"

    # Non-tool utterance
    assert router.route("A falcon is a bird.") is None


def test_tool_grounding_into_memory():
    """Verify execution results are persisted into SQLite memory as I_ground evidence."""
    store = MemoryStore(":memory:")
    grounder = ToolGroundingEngine(memory_store=store)

    call = ToolCall(tool_name="calculator", arguments={"expression": "12 * 12"})
    executor = ToolExecutor()
    result = executor.execute(call)

    grounded = grounder.ground(result)
    assert grounded is True

    # Verify experience was recorded in SQLite
    row = store._conn.execute("SELECT COUNT(*) AS c FROM experiences WHERE source = 'tool_execution';").fetchone()
    assert int(row["c"]) == 1

    # Verify semantic concept was recorded
    concept = store.get_concept("tool_calculator")
    assert concept is not None
