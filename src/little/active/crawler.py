"""Autonomous Knowledge Ingestion & Continuous Web Self-Study Crawler.

Ingests raw unstructured text, markdown files, and web dumps.
Extracts semantic frames (via NeuralFrameExtractor), verifies candidate claims
against DeepSeek-R1 Invariant Gates (I_DAG, I_MUTEX, I_SORT, I_GROUND),
commits verified facts to SQLite MemoryStore, synthesizes OKF living wiki pages,
and indexes concepts into TurboQuant for microsecond retrieval.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.active.self_study import AutonomousSelfStudyEngine, CuriosityStepResult
from little.core.contracts import CandidateClaim, EvidenceRecord, EvidenceStatus, SourceType
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.knowledge.living_wiki import LivingWiki
from little.knowledge.okf import OKFEngine
from little.memory.store import MemoryStore
from little.neural.frame_extractor import NeuralFrameExtractor


@dataclass
class CrawlReport:
    """Detailed trace and metrics from a knowledge crawling and ingestion session."""

    source_reference: str
    text_length: int = 0
    extracted_claims: int = 0
    verified_claims: int = 0
    rejected_claims: int = 0
    new_concepts_created: int = 0
    new_pages_written: int = 0
    duration_ms: float = 0.0
    rejection_reasons: list[str] = field(default_factory=list)
    verified_triples: list[dict[str, str]] = field(default_factory=list)


class KnowledgeCrawler:
    """Autonomous crawler that continuously reads documents, verifies facts, and expands the wiki."""

    def __init__(
        self,
        memory: MemoryStore,
        wiki: LivingWiki | None = None,
        frame_extractor: NeuralFrameExtractor | None = None,
        self_study: AutonomousSelfStudyEngine | None = None,
    ) -> None:
        self.memory = memory
        self.wiki = wiki or LivingWiki(memory)
        self.frame_extractor = frame_extractor or NeuralFrameExtractor()
        self.verifier = DeepSeekInvariantVerifier(memory)
        self.self_study = self_study or AutonomousSelfStudyEngine(memory)
        self.wikilink_re = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, stripping empty lines."""
        raw_chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = []
        for c in raw_chunks:
            cleaned = c.strip()
            if len(cleaned) >= 5:
                sentences.append(cleaned)
        return sentences

    def crawl_text(
        self,
        text: str,
        source_reference: str = "crawl_text",
    ) -> CrawlReport:
        """Process unstructured text, extract candidate frames, verify invariants, and update wiki."""
        t0 = time.perf_counter()
        sentences = self._split_into_sentences(text)

        report = CrawlReport(
            source_reference=source_reference,
            text_length=len(text),
        )

        all_candidate_claims: list[tuple[CandidateClaim, str]] = []

        # 1. Extraction: FrameNet frames + wikilinks
        for sent in sentences:
            claims = self.frame_extractor.extract_candidate_claims(sent)
            for claim in claims:
                all_candidate_claims.append((claim, sent))

            # Also check for explicit wikilinks in sentence: "An [[apple]] is a [[fruit]]"
            matches = self.wikilink_re.findall(sent)
            if len(matches) >= 2:
                s_name = matches[0][0].strip().lower()
                o_name = matches[1][0].strip().lower()
                if "is a" in sent.lower() or "is an" in sent.lower():
                    all_candidate_claims.append(
                        (CandidateClaim(subject=s_name, predicate="is_a", object=o_name, probability=0.95), sent)
                    )

        report.extracted_claims = len(all_candidate_claims)

        # 2. Verification Gate & Knowledge Persistence
        touched_concepts: set[str] = set()

        for claim, source_sentence in all_candidate_claims:
            subj_name = claim.subject.strip().lower()
            obj_name = claim.object.strip().lower()

            # Ensure concepts exist in memory
            subj_concept = self.memory.get_concept(subj_name)
            if not subj_concept:
                subj_concept = self.memory.create_concept(subj_name)
                report.new_concepts_created += 1

            obj_concept = self.memory.get_concept(obj_name)
            if not obj_concept:
                obj_concept = self.memory.create_concept(obj_name)
                report.new_concepts_created += 1

            # Verify through 4 DeepSeek-R1 Invariant Gates (I_DAG, I_MUTEX, I_SORT, I_GROUND)
            gate_result = self.verifier.verify_relation(
                subject=subj_name,
                predicate=claim.predicate,
                obj=obj_name,
            )

            if not gate_result.passed:
                report.rejected_claims += 1
                msg = f"Rejected ({subj_name}, {claim.predicate}, {obj_name}): {gate_result.error_message}"
                report.rejection_reasons.append(msg)
                continue

            # Invariant passed: commit to SQLite MemoryStore
            self.memory.add_relation(
                subject_id=subj_concept.id,
                predicate=claim.predicate,
                object_id=obj_concept.id,
                positive=claim.positive,
            )

            # Record provenance evidence
            ev_rec = EvidenceRecord.from_claim(
                claim=claim,
                source_type=SourceType.DOCUMENT,
                source_reference=source_reference,
                source_text=source_sentence,
            )
            ev_rec.status = EvidenceStatus.ACCEPTED
            self.memory.evidence.append(ev_rec)

            report.verified_claims += 1
            report.verified_triples.append({
                "subject": subj_name,
                "predicate": claim.predicate,
                "object": obj_name,
            })
            touched_concepts.add(subj_name)
            touched_concepts.add(obj_name)

        # 3. Synchronize Living OKF Knowledge Vault
        for c_name in touched_concepts:
            try:
                self.wiki.okf_engine.export_concept(c_name, output_dir=self.wiki.vault_dir)
                report.new_pages_written += 1
            except Exception:
                continue

        # Refresh wiki graph topology and vector index
        if touched_concepts:
            self.wiki.build_graph()

        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return report

    def crawl_file(self, file_path: str | Path) -> CrawlReport:
        """Crawl a local markdown or text document."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return self.crawl_text(content, source_reference=path.name)

    def crawl_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*.md",
    ) -> list[CrawlReport]:
        """Batch crawl a directory of text/markdown documents."""
        folder = Path(dir_path)
        reports = []
        if not folder.exists():
            return []
        for p in sorted(folder.glob(pattern)):
            if p.is_file():
                rep = self.crawl_file(p)
                reports.append(rep)
        return reports

    def autonomous_self_study_step(self) -> CuriosityStepResult:
        """Execute one autonomous curiosity cycle, updating SQLite and the living wiki."""
        step_result = self.self_study.step()
        if step_result.status == "PERSISTED":
            # Sync new knowledge to living wiki page
            try:
                self.wiki.okf_engine.export_concept(
                    step_result.target_concept,
                    output_dir=self.wiki.vault_dir,
                )
                self.wiki.build_graph()
            except Exception:
                pass
        return step_result
