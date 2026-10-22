"""Autonomous Knowledge Ingestion & Continuous Web Self-Study Crawler.

Ingests raw unstructured text, markdown files, and web dumps.
Extracts semantic frames (via NeuralFrameExtractor), verifies candidate claims
against DeepSeek-R1 Invariant Gates (I_DAG, I_MUTEX, I_SORT, I_GROUND),
commits verified facts to SQLite MemoryStore, synthesizes OKF living wiki pages,
and indexes concepts into TurboQuant for microsecond retrieval.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


def _unwrap_wikilinks(sentence: str) -> str:
    """Replace [[target]] or [[target|display]] with target without regex."""
    out: list[str] = []
    i = 0
    n = len(sentence)
    last = 0
    while i < n - 3:
        start = sentence.find("[[", i)
        if start == -1:
            break
        end = sentence.find("]]", start + 2)
        if end == -1:
            break
        inner = sentence[start + 2 : end]
        if "[" not in inner and "]" not in inner:
            out.append(sentence[last:start])
            target = inner.split("|", 1)[0]
            out.append(target)
            last = end + 2
            i = end + 2
        else:
            i = start + 2
    out.append(sentence[last:])
    return "".join(out)

from little.active.self_study import AutonomousSelfStudyEngine, CuriosityStepResult
from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.knowledge.living_wiki import LivingWiki
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
    sync_errors: list[str] = field(default_factory=list)
    processing_errors: list[str] = field(default_factory=list)


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

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences, stripping empty lines."""
        sentences: list[str] = []
        for line in text.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            n = len(line_s)
            last = 0
            i = 0
            while i < n:
                if line_s[i] in ".!?" and i + 1 < n and line_s[i + 1].isspace():
                    j = i + 1
                    while j < n and line_s[j].isspace():
                        j += 1
                    cleaned = line_s[last : i + 1].strip()
                    if len(cleaned) >= 5:
                        sentences.append(cleaned)
                    last = j
                    i = j
                    continue
                i += 1
            tail = line_s[last:].strip()
            if len(tail) >= 5:
                sentences.append(tail)
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

        candidates: dict[tuple[str, str, str, bool, str], CandidateClaim] = {}
        for sentence in sentences:
            # Parse wikilinks as text once rather than invent a second assertion.
            normalized = _unwrap_wikilinks(sentence)
            try:
                claims = self.frame_extractor.extract_candidate_claims(normalized)
            except Exception as exc:
                report.processing_errors.append(f"Extraction failed: {type(exc).__name__}: {exc}")
                continue
            for claim in claims:
                key = (claim.subject.strip().lower(), claim.predicate.strip().lower(),
                       claim.object.strip().lower(), claim.positive, sentence)
                existing = candidates.get(key)
                if existing is None or claim.probability > existing.probability:
                    candidates[key] = claim
        report.extracted_claims = len(candidates)
        touched_concepts: set[str] = set()
        initial_concepts = {c.id for c in self.memory.list_concepts()}
        for key, claim in candidates.items():
            subject, predicate, obj, positive, sentence = key
            gate = self.verifier.verify_relation(subject, predicate, obj)
            threshold_met = claim.probability >= self.wiki.policy.min_extraction_confidence
            verification = gate.to_verification_result()
            if not threshold_met:
                verification = VerificationResult(VerificationStatus.FAILED,
                    checks={"extraction_threshold": False},
                    reasons=[f"Extraction confidence {claim.probability} below configured minimum"])
            # Document extraction establishes what this source asserts, not external truth.
            identity = json.dumps([source_reference, sentence, subject, predicate, obj, positive], ensure_ascii=False)
            evidence_id = "evidence_crawl_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            try:
                with self.memory.transaction():
                    frame = CandidateFrame.from_text(sentence, claims=[claim])
                    evidence = self.memory.ledger.propose_relation(frame, claim,
                        source_type=SourceType.DOCUMENT, source_reference=source_reference,
                        evidence_id=evidence_id)
                    decision = self.memory.ledger.commit_relation(evidence, verification)
            except Exception as exc:
                report.processing_errors.append(f"Persistence failed for {subject} {predicate} {obj}: {type(exc).__name__}: {exc}")
                continue
            if decision.status.value != "accepted":
                report.rejected_claims += 1
                report.rejection_reasons.append(f"Rejected ({subject}, {predicate}, {obj}): {'; '.join(verification.reasons)}")
                continue
            report.verified_claims += 1
            report.verified_triples.append({"subject": subject, "predicate": predicate, "object": obj})
            touched_concepts.update((subject, obj))
        report.new_concepts_created = len({c.id for c in self.memory.list_concepts()} - initial_concepts)
        if self.wiki.policy.auto_sync_on_mutation:
            for name in sorted(touched_concepts):
                try:
                    self.wiki.okf_engine.export_concept(name, output_dir=self.wiki.vault_dir)
                    report.new_pages_written += 1
                except Exception as exc:
                    report.sync_errors.append(f"Wiki export failed for {name}: {type(exc).__name__}: {exc}")
            if touched_concepts:
                try:
                    self.wiki.build_graph()
                except Exception as exc:
                    report.sync_errors.append(f"Wiki graph refresh failed: {type(exc).__name__}: {exc}")

        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return report

    def crawl_file(self, file_path: str | Path) -> CrawlReport:
        """Crawl a local markdown or text document."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")
        return self.crawl_text(content, source_reference=str(path.resolve()))

    def crawl_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*.md",
    ) -> list[CrawlReport]:
        """Batch crawl a directory of text/markdown documents."""
        folder = Path(dir_path)
        reports = []
        if not folder.exists() or self.wiki.policy.batch_crawl_limit == 0:
            return []
        for p in sorted(folder.glob(pattern)):
            if p.is_file():
                rep = self.crawl_file(p)
                reports.append(rep)
                if len(reports) >= self.wiki.policy.batch_crawl_limit:
                    break
        return reports

    def autonomous_self_study_step(self) -> CuriosityStepResult:
        """Execute one autonomous curiosity cycle, updating SQLite and the living wiki."""
        step_result = self.self_study.study_step()
        if step_result is None:
            return None
        if step_result.status == "PERSISTED":
            # Sync new knowledge to living wiki page
            try:
                self.wiki.okf_engine.export_concept(
                    step_result.target_concept,
                    output_dir=self.wiki.vault_dir,
                )
                self.wiki.build_graph()
            except Exception as exc:
                step_result.sync_errors.append(f"Wiki sync failed: {type(exc).__name__}: {exc}")
        return step_result
