"""Unit tests for CognitiveCommand and CognitiveCommandRegistry."""

from __future__ import annotations

import re
import pytest

from little.core.cognitive_command import CognitiveCommand, CognitiveCommandRegistry
from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_default_commands_registered():
    """Verify built-in cognitive commands are registered with patterns and descriptions."""
    registry = CognitiveCommandRegistry()
    assert len(registry._commands) >= 7

    command_ids = {cmd.command_id for cmd in registry._commands}
    assert "knowledge_ingestion" in command_ids
    assert "wiki_semantic_search" in command_ids
    assert "wiki_backlinks" in command_ids
    assert "wiki_page_summary" in command_ids
    assert "concept_clustering" in command_ids
    assert "list_latent_knots" in command_ids
    assert "explain_concept_cluster" in command_ids


def test_custom_command_registration():
    """Verify extensible registration of user-defined cognitive commands."""
    registry = CognitiveCommandRegistry(commands=[])
    assert len(registry._commands) == 0

    custom_cmd = CognitiveCommand(
        command_id="custom_ping",
        description="A custom cognitive health check command",
        patterns=(re.compile(r"^ping\s+kernel\??$", re.IGNORECASE),),
        handler=lambda k, t, m: "Kernel is cognitively healthy and active.",
    )
    registry.register(custom_cmd)

    kernel = CognitiveKernel(MemoryStore(":memory:"))

    # Positive match
    res = registry.dispatch("ping kernel?", kernel)
    assert res == "Kernel is cognitively healthy and active."

    # Negative match
    res_none = registry.dispatch("what is an apple?", kernel)
    assert res_none is None


def test_command_dispatch_fallthrough_on_non_command():
    """Verify that ordinary conversation or questions return None."""
    registry = CognitiveCommandRegistry()
    kernel = CognitiveKernel(MemoryStore(":memory:"))

    assert registry.dispatch("A falcon is a bird.", kernel) is None
    assert registry.dispatch("Can an eagle fly?", kernel) is None
    assert registry.dispatch("Hello there!", kernel) is None
