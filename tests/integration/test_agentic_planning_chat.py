"""Integration tests for Multi-Step Agentic Planning Loop within CognitiveKernel conversational chat."""

from pathlib import Path

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_agentic_planning_calculator_and_file_writer_chat(tmp_path: Path):
    """Verify that compound goals decompose into multi-step plans and verbalize with full provenance."""
    store = MemoryStore(":memory:")
    kernel = CognitiveKernel(memory=store, workspace_root=tmp_path)

    # 1. 2-Step action chain: calculate -> write to file
    reply1 = kernel.chat("calculate 25 * 4 and then write the result to calc.txt")
    assert "autonomous 2-step plan" in reply1 or "Step 1" in reply1
    assert "Step 1: calculator" in reply1
    assert "Step 2: file_writer" in reply1
    assert "[OK]" in reply1
    assert "[Trace:" in reply1

    # Verify file was written
    calc_file = tmp_path / "calc.txt"
    assert calc_file.exists()
    assert calc_file.read_text(encoding="utf-8") == "100"


def test_agentic_planning_3_step_chain_chat(tmp_path: Path):
    """Verify 3-step action chain: calculate -> write file -> read file."""
    store = MemoryStore(":memory:")
    kernel = CognitiveKernel(memory=store, workspace_root=tmp_path)

    reply = kernel.chat("calculate 15 * 4, then write the answer to math.txt, and then read file math.txt")
    assert "3-step plan" in reply or "Step 1" in reply
    assert "Step 1: calculator" in reply
    assert "Step 2: file_writer" in reply
    assert "Step 3: file_reader" in reply
    assert "Final Result: 60" in reply

    # Verify memory grounding of experiences
    exp_count = store._conn.execute(
        "SELECT COUNT(*) AS c FROM experiences WHERE source = 'tool_execution';"
    ).fetchone()["c"]
    assert exp_count >= 3


def test_agentic_planning_python_and_save_chat(tmp_path: Path):
    """Verify Python code execution followed by saving result."""
    store = MemoryStore(":memory:")
    kernel = CognitiveKernel(memory=store, workspace_root=tmp_path)

    reply = kernel.chat("run python: print(12 * 12) and then write the result to square.txt")
    assert "Step 1: python_repl" in reply
    assert "Step 2: file_writer" in reply
    assert "144" in reply

    sq_file = tmp_path / "square.txt"
    assert sq_file.exists()
    assert "144" in sq_file.read_text(encoding="utf-8")


def test_agentic_planning_with_social_greeting(tmp_path: Path):
    """Verify social greeting + compound agentic planning goal."""
    store = MemoryStore(":memory:")
    kernel = CognitiveKernel(memory=store, workspace_root=tmp_path)

    reply = kernel.chat("Hello! Please calculate 50 * 2 and then write the result to greeting_calc.txt")
    assert "Hello!" in reply
    assert "calculator" in reply
    assert "file_writer" in reply

    res_file = tmp_path / "greeting_calc.txt"
    assert res_file.exists()
    assert res_file.read_text(encoding="utf-8") == "100"
