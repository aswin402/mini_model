"""Grounding engine for persisting tool execution outcomes as empirical evidence (I_ground)."""

from __future__ import annotations

import json
import sqlite3

from little.actions.models import ToolExecutionResult
from little.memory.store import MemoryStore


class ToolGroundingEngine:
    """Grounds tool outputs into the persistent episodic and semantic memory tiers."""

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store

    def ground(self, result: ToolExecutionResult) -> bool:
        """Persist a tool execution result as empirical evidence in SQLite memory."""
        if not self.memory_store or not result.success:
            return False

        try:
            # 1. Record an episodic experience entry
            exp_text = f"Tool execution: {result.tool_name} with {json.dumps(result.arguments, sort_keys=True)}"
            exp = self.memory_store.add_experience(
                input_text=exp_text,
                extracted_triples=[
                    {
                        "subject": f"tool_{result.tool_name}",
                        "predicate": "produced_result",
                        "object": str(result.output)[:200],
                    }
                ],
                source="tool_execution",
            )

            # 2. Persist semantic relation if output is clean string or number
            if isinstance(result.output, (str, int, float, bool)):
                tool_concept = self.memory_store.get_or_create_concept(
                    f"tool_{result.tool_name}", category="tool"
                )
                output_str = str(result.output).strip()[:100]
                if output_str:
                    out_concept = self.memory_store.get_or_create_concept(
                        output_str, category="observation"
                    )
                    self.memory_store.add_relation(
                        subject_id=tool_concept.id,
                        predicate="produced_result",
                        object_id=out_concept.id,
                        source_experience_id=exp.id,
                        positive=True,
                    )
            return True
        except (sqlite3.Error, ValueError, KeyError, TypeError):
            return False
