"""Living Graph Wiki Engine for LITTLE.

Implements an interconnected, living, bidirectional graph wiki of Concept Knots
(inspired by Dox, Obsidian, LLM OS, and GraphRAG).
Maintains forward links, backlinks, graph degree centrality, frontier stubs,
and microsecond semantic search via TurboQuant vector indexing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.knowledge.living_wiki_policy import LivingWikiPolicy
from little.knowledge.okf import (
    OKFDocument,
    OKFEngine,
    concept_file_path,
    validate_concept_name,
)
from little.memory.store import MemoryStore
from little.neural.encoder import NeuralStateEncoder
from little.neural.turbo_quant import (
    TurboQuantCompressor,
    TurboQuantIndex,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


@dataclass
class WikiNode:
    """A concept node in the living wiki graph."""

    concept_name: str
    file_path: Path | None = None
    document: OKFDocument | None = None
    out_links: set[str] = field(default_factory=set)
    in_links: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)
    is_stub: bool = False

    @property
    def degree(self) -> int:
        return len(self.out_links) + len(self.in_links)


@dataclass(frozen=True)
class WikiGraphStats:
    """Topology and volume statistics of the living wiki graph."""

    total_pages: int
    total_stubs: int
    total_links: int
    orphan_concepts: list[str]
    top_hubs: list[tuple[str, int]]
    vector_index_size: int


class LivingWiki:
    """Interactive graph wiki managing interconnected OKF documents and TurboQuant search."""

    def __init__(
        self,
        memory: MemoryStore,
        vault_dir: Path | str | None = None,
        policy: LivingWikiPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or LivingWikiPolicy.default()
        self.vault_dir = Path(vault_dir or self.policy.vault_directory)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

        self.okf_engine = OKFEngine(memory)
        self.encoder = NeuralStateEncoder()
        self.compressor = TurboQuantCompressor()
        self.vector_index = TurboQuantIndex(self.compressor)

        self.nodes: dict[str, WikiNode] = {}
        self.build_graph()

    @staticmethod
    def _extract_out_links(doc: OKFDocument) -> set[str]:
        name = doc.concept_name
        out_targets: set[str] = set()
        tax = doc.frontmatter.get("taxonomy", {})
        if isinstance(tax, dict):
            for parent in tax.get("is_a", []):
                if parent and isinstance(parent, str):
                    out_targets.add(parent.strip().lower())
        for relation in doc.frontmatter.get("relations", []):
            if isinstance(relation, dict) and isinstance(
                relation.get("object"), str
            ):
                out_targets.add(validate_concept_name(relation["object"]))
        mereology = doc.frontmatter.get("mereology", {})
        if isinstance(mereology, dict):
            out_targets.update(
                validate_concept_name(part)
                for part in mereology.get("parts", [])
            )
        for link in doc.wikilinks:
            if link.target and link.target != name:
                out_targets.add(link.target)
        return out_targets

    def build_graph(self) -> None:
        """Scan knowledge vault, construct link topology, and build TurboQuant index."""
        if getattr(self, "vector_index", None) is not None:
            self.vector_index.close()
        self.nodes.clear()
        self.vector_index = TurboQuantIndex(self.compressor)

        # 1. First pass: Load all existing OKF documents
        for file_path in self.vault_dir.glob(
            f"*{self.okf_engine.policy.file_extension}"
        ):
            try:
                doc = self.okf_engine.parse_file(file_path)
                name = doc.concept_name
                validate_concept_name(name)
                out_targets = self._extract_out_links(doc)

                node = WikiNode(
                    concept_name=name,
                    file_path=file_path,
                    document=doc,
                    out_links=out_targets,
                    attributes=dict(doc.frontmatter.get("attributes", {})),
                    is_stub=False,
                )
                self.nodes[name] = node

                # Index continuous embedding into TurboQuant
                full_text = f"{name} {doc.body}"
                state = self.encoder.encode(full_text)
                self.vector_index.add(
                    id=name,
                    vector=state.embedding,
                    metadata={"name": name, "file": str(file_path)},
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                logger.warning("Unable to index wiki page %s: %s", file_path, exc)
                continue

        # 2. Second pass: Construct backward links and stub nodes for missing concepts
        all_targets: set[str] = set()
        for name, node in list(self.nodes.items()):
            for target in node.out_links:
                all_targets.add(target)
                if target in self.nodes:
                    self.nodes[target].in_links.add(name)
                else:
                    # Target is referenced but has no page yet: create stub
                    stub = WikiNode(
                        concept_name=target,
                        in_links={name},
                        is_stub=True,
                    )
                    self.nodes[target] = stub

    def get_page(self, concept_name: str) -> OKFDocument | None:
        """Retrieve OKF document for a concept."""
        clean = concept_name.strip().lower()
        node = self.nodes.get(clean)
        return node.document if (node and not node.is_stub) else None

    def get_backlinks(self, concept_name: str) -> list[str]:
        """Return list of concepts that link TO this concept."""
        clean = concept_name.strip().lower()
        node = self.nodes.get(clean)
        return sorted(node.in_links) if node else []

    def get_forward_links(self, concept_name: str) -> list[str]:
        """Return list of concepts linked FROM this concept."""
        clean = concept_name.strip().lower()
        node = self.nodes.get(clean)
        return sorted(node.out_links) if node else []

    def get_unexplored_frontier(self) -> list[str]:
        """Return stub concepts that are referenced by wikilinks but lack full pages."""
        return sorted(name for name, node in self.nodes.items() if node.is_stub)

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.08,
    ) -> list[VectorSearchResult]:
        """Sub-millisecond coarse-to-fine TurboQuant semantic search over living wiki."""
        state = self.encoder.encode(query)
        return self.vector_index.search(
            state.embedding, top_k=top_k, threshold=threshold
        )

    def save_or_update_page(self, doc: OKFDocument) -> Path:
        """Persist or update an OKF page in the vault and refresh graph & vector index."""
        name = validate_concept_name(doc.concept_name)
        file_path = concept_file_path(
            self.vault_dir, name, self.okf_engine.policy.file_extension
        )

        from little.knowledge.okf import serialize_yaml_value

        content = f"---\n{serialize_yaml_value(doc.frontmatter)}\n---\n\n{doc.body}\n"
        file_path.write_text(content, encoding="utf-8")

        parsed_doc = self.okf_engine.parse_file(file_path)
        out_targets = self._extract_out_links(parsed_doc)

        prev_node = self.nodes.get(name)
        existing_in_links = set(prev_node.in_links) if prev_node else set()
        if prev_node:
            for old_target in prev_node.out_links - out_targets:
                t_node = self.nodes.get(old_target)
                if t_node:
                    t_node.in_links.discard(name)
                    if t_node.is_stub and not t_node.in_links:
                        self.nodes.pop(old_target, None)

        self.nodes[name] = WikiNode(
            concept_name=name,
            file_path=file_path,
            document=parsed_doc,
            out_links=out_targets,
            in_links=existing_in_links,
            attributes=dict(parsed_doc.frontmatter.get("attributes", {})),
            is_stub=False,
        )

        for target in out_targets:
            if target in self.nodes:
                self.nodes[target].in_links.add(name)
            else:
                self.nodes[target] = WikiNode(
                    concept_name=target,
                    in_links={name},
                    is_stub=True,
                )

        full_text = f"{name} {parsed_doc.body}"
        state = self.encoder.encode(full_text)
        self.vector_index.add(
            id=name,
            vector=state.embedding,
            metadata={"name": name, "file": str(file_path)},
        )

        return file_path

    def sync_concept(self, concept_name: str) -> Path | None:
        """Export concept from memory into the vault as OKF document and refresh wiki node and index."""
        clean_name = concept_name.strip().lower()
        concept = self.memory.get_concept(clean_name)
        if not concept:
            return None
        try:
            path = self.okf_engine.export_concept(clean_name, output_dir=self.vault_dir)
            doc = self.okf_engine.parse_file(path)
            self.save_or_update_page(doc)
            return path
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Unable to synchronize wiki concept %s: %s", clean_name, exc)
            return None

    def stats(self) -> WikiGraphStats:
        """Compute graph centrality, orphans, hubs, and index metrics."""
        total_pages = sum(1 for n in self.nodes.values() if not n.is_stub)
        total_stubs = sum(1 for n in self.nodes.values() if n.is_stub)
        total_links = sum(len(n.out_links) for n in self.nodes.values())

        orphans = sorted(
            n.concept_name
            for n in self.nodes.values()
            if not n.is_stub and len(n.out_links) == 0 and len(n.in_links) == 0
        )

        hubs = sorted(
            [(n.concept_name, n.degree) for n in self.nodes.values() if not n.is_stub],
            key=lambda x: x[1],
            reverse=True,
        )

        return WikiGraphStats(
            total_pages=total_pages,
            total_stubs=total_stubs,
            total_links=total_links,
            orphan_concepts=orphans,
            top_hubs=hubs[:10],
            vector_index_size=len(self.vector_index),
        )
