"""Unit tests for agentic tool agency and multi-step plan execution."""

from pathlib import Path
from little.actions.models import AgenticPlanExecutionResult, PlanStepStatus
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_agentic_tool_planning_and_grounding(tmp_path: Path):
    db_path = tmp_path / "test_agency.db"
    store = MemoryStore(str(db_path), seed_ontology=True)
    kernel = CognitiveKernel(store, workspace_root=tmp_path)

    # 1. Multi-step goal: compute math and save to file
    goal = "calculate 25 * 4 and write result to answer.txt"
    plan_res = kernel.plan(goal)

    assert isinstance(plan_res, AgenticPlanExecutionResult)
    assert plan_res.success is True
    assert len(plan_res.steps) == 2

    # Step 1: calculator
    step1 = plan_res.steps[0]
    assert step1.tool_name == "calculator"
    assert step1.status == PlanStepStatus.SUCCESS
    assert step1.result is not None
    assert step1.result.output == 100

    # Step 2: file_writer
    step2 = plan_res.steps[1]
    assert step2.tool_name == "file_writer"
    assert step2.status == PlanStepStatus.SUCCESS

    # Verify the physical file was written in workspace root
    ans_file = tmp_path / "answer.txt"
    assert ans_file.exists()
    assert "100" in ans_file.read_text(encoding="utf-8")

    # 2. Verify episodic experience was grounded in memory store
    exps = store.list_experiences()
    assert any("tool" in e.source or "Tool execution" in e.input_text for e in exps)
