"""High-performance Knowledge Ingestion Pipeline for LITTLE.

Ingests commonsense world knowledge from ConceptNet 5 assertions, WordNet hierarchies,
and curated structured datasets into the persistent transactional MemoryStore with
zero hardcoding, zero catastrophic forgetting, and sub-millisecond execution.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from little.knowledge.import_policy import KnowledgeImportPolicy

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


@dataclass
class ImportStats:
    """Statistics recorded during a knowledge ingestion batch."""

    total_lines: int = 0
    triples_parsed: int = 0
    new_concepts: int = 0
    relations_processed: int = 0
    elapsed_seconds: float = 0.0

    @property
    def triples_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return round(self.relations_processed / self.elapsed_seconds, 1)

    def summary(self) -> str:
        return (
            f"Ingested {self.relations_processed:,} relations ({self.new_concepts:,} new concepts) "
            f"from {self.triples_parsed:,} parsed triples in {self.elapsed_seconds:.2f}s "
            f"({self.triples_per_second:,} triples/sec)."
        )


class ConceptNetNormalizer:
    """Normalizes and sanitizes ConceptNet 5 URIs and relations into LITTLE semantics."""

    @classmethod
    def _policy(cls) -> KnowledgeImportPolicy:
        """Load the versioned default without retaining mutable global state."""
        configured = getattr(cls, "_POLICY", None)
        return configured if configured is not None else KnowledgeImportPolicy.default()

    @classmethod
    def clean_concept_uri(
        cls,
        uri_or_text: str,
        *,
        policy: KnowledgeImportPolicy | None = None,
    ) -> str | None:
        """Extract clean English concept name from a ConceptNet URI or string.

        Examples:
            '/c/en/dog/n' -> 'dog'
            '/c/en/golden_retriever' -> 'golden retriever'
            '/c/en/apple_juice/n/wn/food' -> 'apple juice'
            '/c/fr/chien' -> None (non-English discarded)
        """
        active_policy = policy or cls._policy()
        raw = uri_or_text.strip()
        if not raw:
            return None

        # Check if ConceptNet URI
        if raw.startswith("/c/"):
            if not raw.startswith("/c/en/"):
                # Discard non-English concepts to maintain high English reasoning quality
                return None
            parts = raw.split("/")
            # parts[0] = '', parts[1] = 'c', parts[2] = 'en', parts[3] = term
            if len(parts) < 4 or not parts[3]:
                return None
            term = parts[3]
        else:
            term = raw

        # Replace underscores and hyphens
        term = term.replace("_", " ").strip().lower()

        # Remove trailing POS markers if accidentally retained: "dog.n.01" -> "dog"
        term = re.sub(r"\.[nvarsp]\.\d+$", "", term)

        # Remove special characters, keep letters, numbers, and single spaces
        term = re.sub(r"[^a-z0-9\s-]", "", term).strip()
        term = re.sub(r"\s+", " ", term)

        # Filters
        if not term or term in active_policy.discard_concepts:
            return None
        # Discard multi-word sentences (> 4 words or > 35 chars) to prevent graph noise
        words = term.split()
        if (
            len(words) > active_policy.max_concept_words
            or len(term) > active_policy.max_concept_characters
        ):
            return None

        return term

    @classmethod
    def map_relation(
        cls,
        rel_str: str,
        *,
        policy: KnowledgeImportPolicy | None = None,
    ) -> str | None:
        active_policy = policy or cls._policy()
        return active_policy.relation_aliases.get(rel_str.strip().lower())


class KnowledgeImporter:
    """Streamlined engine for loading verified world knowledge into persistent MemoryStore."""

    def __init__(
        self,
        memory: MemoryStore,
        batch_size: int | None = None,
        import_policy: KnowledgeImportPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.batch_size = batch_size or memory.memory_policy.bulk_import_batch_size
        self.import_policy = import_policy or KnowledgeImportPolicy.default()

    def import_triples(
        self,
        triples: Iterable[tuple[str, str, str, float | int, bool]],
        progress_cb: Callable[[int, int], None] | None = None,
        max_triples: int | None = None,
    ) -> ImportStats:
        """Stream and batch-insert pre-parsed (subject, predicate, object, weight, positive) triples."""
        start_time = time.perf_counter()
        stats = ImportStats()

        batch: list[tuple[str, str, str, float | int, bool]] = []
        count = 0

        for subj, pred, obj, weight, pos in triples:
            s = ConceptNetNormalizer.clean_concept_uri(
                subj, policy=self.import_policy
            )
            o = ConceptNetNormalizer.clean_concept_uri(obj, policy=self.import_policy)
            p = ConceptNetNormalizer.map_relation(pred, policy=self.import_policy)

            if not s or not o or not p or s == o:
                continue

            batch.append((s, p, o, weight, pos))
            stats.triples_parsed += 1
            count += 1

            if len(batch) >= self.batch_size:
                new_c, rels_p = self.memory.bulk_import_triples(
                    batch, batch_size=self.batch_size
                )
                stats.new_concepts += new_c
                stats.relations_processed += rels_p
                batch = []
                if progress_cb:
                    progress_cb(stats.triples_parsed, stats.relations_processed)

            if max_triples and count >= max_triples:
                break

        if batch:
            new_c, rels_p = self.memory.bulk_import_triples(
                batch, batch_size=self.batch_size
            )
            stats.new_concepts += new_c
            stats.relations_processed += rels_p
            if progress_cb:
                progress_cb(stats.triples_parsed, stats.relations_processed)

        stats.elapsed_seconds = time.perf_counter() - start_time
        return stats

    def import_conceptnet_file(
        self,
        file_path: str | Path,
        min_weight: float = 1.0,
        max_triples: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ImportStats:
        """Import from official ConceptNet 5 CSV/TSV assertion dumps.

        Format:
        URI \t Relation \t StartConcept \t EndConcept \t JSONMetadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ConceptNet file not found: {path}")

        def generate_triples():
            with path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) < 4:
                        continue
                    # ConceptNet 5 columns: URI, relation, start, end, metadata
                    rel = row[1]
                    subj = row[2]
                    obj = row[3]
                    weight = 1.0
                    if len(row) >= 5 and row[4]:
                        try:
                            meta = json.loads(row[4])
                            weight = float(meta.get("weight", 1.0))
                        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                            weight = 1.0

                    if weight < min_weight:
                        continue

                    yield (subj, rel, obj, weight, True)

        return self.import_triples(
            generate_triples(), progress_cb=progress_cb, max_triples=max_triples
        )

    def import_tsv_file(
        self,
        file_path: str | Path,
        min_weight: float = 1.0,
        max_triples: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ImportStats:
        """Import simple TSV or CSV triples: subject, predicate, object, [weight], [positive]."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        delim = "\t" if path.suffix in (".tsv", ".txt") else ","

        def generate_triples():
            with path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delim)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    subj, pred, obj = row[0].strip(), row[1].strip(), row[2].strip()
                    weight = 1.0
                    pos = True
                    if len(row) >= 4 and row[3].strip():
                        try:
                            weight = float(row[3].strip())
                        except ValueError:
                            weight = 1.0
                    if len(row) >= 5 and row[4].strip():
                        pos = row[4].strip().lower() not in ("false", "0", "no", "not")

                    if weight < min_weight:
                        continue

                    yield (subj, pred, obj, weight, pos)

        return self.import_triples(
            generate_triples(), progress_cb=progress_cb, max_triples=max_triples
        )

    def import_json_file(
        self,
        file_path: str | Path,
        min_weight: float = 1.0,
        max_triples: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ImportStats:
        """Import triples from JSON Lines (.jsonl) or JSON Array (.json)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        def generate_triples():
            with path.open("r", encoding="utf-8", errors="replace") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":
                    data = json.load(f)
                    for item in data:
                        s = item.get("subject") or item.get("s")
                        p = item.get("predicate") or item.get("p") or item.get("rel")
                        o = item.get("object") or item.get("o")
                        w = float(item.get("weight", 1.0))
                        pos = bool(item.get("positive", True))
                        if s and p and o and w >= min_weight:
                            yield (s, p, o, w, pos)
                else:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            s = item.get("subject") or item.get("s")
                            p = (
                                item.get("predicate")
                                or item.get("p")
                                or item.get("rel")
                            )
                            o = item.get("object") or item.get("o")
                            w = float(item.get("weight", 1.0))
                            pos = bool(item.get("positive", True))
                            if s and p and o and w >= min_weight:
                                yield (s, p, o, w, pos)
                        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                            continue

        return self.import_triples(
            generate_triples(), progress_cb=progress_cb, max_triples=max_triples
        )

    def import_commonsense_bundle(
        self,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ImportStats:
        """Import the verified, curated commonsense world-knowledge bundle."""
        from little.knowledge.bundle import get_commonsense_triples

        triples = get_commonsense_triples()
        return self.import_triples(triples, progress_cb=progress_cb)

    def import_large_scale_ontology(
        self,
        target_count: int = 100_000,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> ImportStats:
        """Stream and batch-ingest the 100,000+ extended commonsense ontology into persistent memory."""
        from little.knowledge.curated_ontology import (
            generate_extended_commonsense_triples,
        )

        triples = generate_extended_commonsense_triples(target_count=target_count)
        return self.import_triples(
            triples, progress_cb=progress_cb, max_triples=target_count
        )
