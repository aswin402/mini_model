"""Unit tests for KnowledgeCrawler autonomous ingestion and verification."""

import tempfile
from pathlib import Path
import pytest

from little.active.crawler import KnowledgeCrawler, CrawlReport
from little.knowledge.living_wiki import LivingWiki
from little.memory.store import MemoryStore


def test_knowledge_crawler_text_ingestion():
    """Verify autonomous crawling of unstructured text, verification, and wiki page creation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore()
        wiki = LivingWiki(store, vault_dir=tmp_dir)
        crawler = KnowledgeCrawler(store, wiki=wiki)

        # Ingest text declaring taxonomy and relations
        doc_text = """
        A cheetah is a feline.
        Felines are mammals.
        A cheetah can run fast.
        """
        report = crawler.crawl_text(doc_text, source_reference="biology_intro.txt")

        assert isinstance(report, CrawlReport)
        assert report.extracted_claims >= 2
        assert report.verified_claims >= 2
        assert report.rejected_claims == 0
        assert report.new_pages_written >= 2

        # Verify concepts were persisted into SQLite
        cheetah = store.get_concept("cheetah")
        assert cheetah is not None
        feline = store.get_concept("feline")
        assert feline is not None

        # Verify relation exists in SQLite
        rels = store.get_relations(subject_id=cheetah.id, predicate="is_a")
        assert len(rels) >= 1
        assert store.get_concept(rels[0].object_id).name == "feline"

        # Verify OKF wiki page was written to vault
        page = wiki.get_page("cheetah")
        assert page is not None
        assert page.frontmatter["name"] == "cheetah"
        assert "feline" in page.frontmatter["taxonomy"]["is_a"]


def test_knowledge_crawler_rejects_cycles():
    """Verify that crawler invariant gates prevent cycle contamination."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = MemoryStore()
        c_animal = store.create_concept("animal")
        c_mammal = store.create_concept("mammal")
        store.add_relation(c_mammal.id, "is_a", c_animal.id)

        wiki = LivingWiki(store, vault_dir=tmp_dir)
        crawler = KnowledgeCrawler(store, wiki=wiki)

        # Ingest contradictory sentence claiming animal is a mammal (cycle!)
        bad_text = "An animal is a mammal."
        report = crawler.crawl_text(bad_text, source_reference="contradictory_doc.txt")

        # Must be rejected by I_DAG invariant gate!
        assert report.rejected_claims > 0
        assert any("cycle" in reason.lower() or "i_dag" in reason.lower() for reason in report.rejection_reasons)


def test_knowledge_crawler_file_and_directory():
    """Verify crawler batch ingesting files from a directory."""
    with tempfile.TemporaryDirectory() as tmp_vault, tempfile.TemporaryDirectory() as tmp_docs:
        docs_path = Path(tmp_docs)
        (docs_path / "birds.md").write_text("An eagle is a bird. Birds can fly.", encoding="utf-8")
        (docs_path / "fish.md").write_text("A salmon is a fish. Salmon lives in water.", encoding="utf-8")

        store = MemoryStore()
        wiki = LivingWiki(store, vault_dir=tmp_vault)
        crawler = KnowledgeCrawler(store, wiki=wiki)

        reports = crawler.crawl_directory(docs_path, pattern="*.md")
        assert len(reports) == 2
        assert all(r.verified_claims > 0 for r in reports)

        # Check concepts exist
        assert store.get_concept("eagle") is not None
        assert store.get_concept("salmon") is not None
