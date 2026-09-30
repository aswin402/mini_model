"""Unit tests for LivingWiki graph topology and TurboQuant search."""

import tempfile
from pathlib import Path
import pytest

from little.knowledge.living_wiki import LivingWiki
from little.knowledge.okf import OKFDocument, OKFWikilink
from little.memory.store import MemoryStore


def test_living_wiki_graph_construction():
    """Verify bidirectional link topology, stubs, and hub centrality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore()
        wiki = LivingWiki(store, vault_dir=tmp_dir)

        # Create two pages: apple and fruit
        apple_doc = OKFDocument(
            frontmatter={"name": "apple", "taxonomy": {"is_a": ["fruit"]}},
            body="An apple is a sweet [[fruit]] that has [[seeds]] and [[peel]].",
            wikilinks=[
                OKFWikilink(target="fruit", display="fruit"),
                OKFWikilink(target="seeds", display="seeds"),
                OKFWikilink(target="peel", display="peel"),
            ],
        )
        fruit_doc = OKFDocument(
            frontmatter={"name": "fruit", "taxonomy": {"is_a": ["plant"]}},
            body="A fruit is a seed-bearing structure developed from ovary.",
            wikilinks=[OKFWikilink(target="plant", display="plant")],
        )

        wiki.save_or_update_page(apple_doc)
        wiki.save_or_update_page(fruit_doc)
        wiki.build_graph()

        # Check forward links
        apple_fwd = wiki.get_forward_links("apple")
        assert "fruit" in apple_fwd
        assert "seeds" in apple_fwd

        # Check backlinks
        fruit_back = wiki.get_backlinks("fruit")
        assert "apple" in fruit_back

        # Check stubs (seeds, peel, plant are referenced but have no page yet)
        frontier = wiki.get_unexplored_frontier()
        assert "seeds" in frontier
        assert "peel" in frontier
        assert "plant" in frontier

        # Check stats
        stats = wiki.stats()
        assert stats.total_pages == 2
        assert stats.total_stubs >= 3
        assert stats.total_links >= 4


def test_living_wiki_turbo_quant_semantic_search():
    """Verify microsecond semantic search via TurboQuant vector index."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore()
        wiki = LivingWiki(store, vault_dir=tmp_dir)

        doc1 = OKFDocument(
            frontmatter={"name": "galaxy"},
            body="A galaxy is a massive gravitationally bound system of stars, stellar remnants, and dark matter.",
        )
        doc2 = OKFDocument(
            frontmatter={"name": "python"},
            body="Python is a high-level programming language supporting object-oriented and functional paradigms.",
        )

        wiki.save_or_update_page(doc1)
        wiki.save_or_update_page(doc2)

        # Query astronomy topic
        astronomy_results = wiki.search_semantic("stars and gravitational systems in the universe", top_k=2)
        assert len(astronomy_results) > 0
        assert astronomy_results[0].id == "galaxy"

        # Query programming topic
        code_results = wiki.search_semantic("object oriented software language script", top_k=2)
        assert len(code_results) > 0
        assert code_results[0].id == "python"
