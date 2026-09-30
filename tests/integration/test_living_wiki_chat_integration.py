"""Integration test verifying end-to-end Living Wiki, Knowledge Crawler, and Headroom CCR via CognitiveKernel chat."""

import tempfile
import pytest

from little.core.kernel import CognitiveKernel
from little.memory.store import MemoryStore


def test_living_wiki_and_crawler_chat_flow():
    """Verify chat-driven knowledge ingestion, multi-hop reasoning, wiki search, and headroom bounds."""
    with tempfile.TemporaryDirectory() as tmp_vault:
        memory = MemoryStore(":memory:", seed_ontology=True)
        kernel = CognitiveKernel(memory=memory)
        # Point wiki vault to temporary directory for isolation
        kernel.wiki.vault_dir = kernel.wiki.vault_dir.__class__(tmp_vault)
        kernel.crawler.wiki.vault_dir = kernel.wiki.vault_dir

        # 1. Ingest text via natural language chat
        r_ingest = kernel.chat(
            "study text: A dolphin is a mammal. Dolphins are intelligent cetaceans. A dolphin can swim."
        )
        assert "Crawled and ingested" in r_ingest
        assert "Verified claims committed to memory" in r_ingest

        # 2. Query deductive reasoning on the ingested knowledge
        r_query = kernel.chat("Is a dolphin a mammal?")
        assert "dolphin is indeed a mammal." in r_query

        # 3. Microsecond TurboQuant Semantic Search via chat
        r_search = kernel.chat("search wiki for swimming dolphin")
        assert "dolphin" in r_search.lower()

        # 4. Backlinks Query via chat
        r_backlinks = kernel.chat("what links to mammal?")
        assert "dolphin" in r_backlinks.lower()

        # 5. Wiki Page Summary via chat
        r_page = kernel.chat("show wiki page for dolphin")
        assert "dolphin" in r_page.lower()
        assert "mammal" in r_page.lower()

        # 6. Strict Zero-Hallucination Open-World Assumption (OWA)
        r_unknown = kernel.chat("Is a zorblaxian a mammal?")
        assert "don't have" in r_unknown.lower() or "not in my memory" in r_unknown.lower() or "unknown" in r_unknown.lower()

        # 7. Headroom Working Memory Health
        stats = kernel.headroom.stats()
        assert stats.active_turns >= 6
        assert stats.estimated_ram_kb < 100.0  # < 100 KB RAM
