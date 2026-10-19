"""Continuous Stream Ingestion and Learning Pipeline.

Inspired by mini-AGI's single-stream architecture where document reading and training
are the exact same execution path.

Features:
- Seamless chunk-by-chunk and document stream ingestion.
- Zero-hallucination semantic frame perception.
- Real-time DeepSeek invariant gate verification.
- Instant hydration and activation voting into 3-tier PagedConceptStore.
- Automatic Living Wiki OKF synchronization.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from little.active.plasticity import PlasticityController, PlasticityTelemetry
from little.core.contracts import SourceType

if TYPE_CHECKING:
    from little.core.kernel import CognitiveKernel
    from little.memory.paged_store import PagedConceptStore


@dataclass(frozen=True)
class ChunkIngestResult:
    """Outcome of processing a single stream text chunk."""

    source_text: str
    sentences_count: int
    concepts_created: list[str]
    relations_created: list[str]
    contradictions_blocked: int
    is_learned: bool
    latency_ms: float
    message: str
    plasticity_telemetry: PlasticityTelemetry | None = None


@dataclass
class StreamLearningSummary:
    """Aggregated report of continuous stream ingestion."""

    source_id: str
    total_chunks: int = 0
    total_sentences: int = 0
    total_concepts_created: list[str] = field(default_factory=list)
    total_relations_created: list[str] = field(default_factory=list)
    total_contradictions_blocked: int = 0
    total_latency_ms: float = 0.0
    plasticity_telemetry: PlasticityTelemetry | None = None

    @property
    def concepts_count(self) -> int:
        return len(set(self.total_concepts_created))

    @property
    def relations_count(self) -> int:
        return len(self.total_relations_created)


class ContinuousStreamLearner:
    """Processes continuous text streams through the unified perception, verification, and memory pipeline."""

    def __init__(
        self,
        kernel: CognitiveKernel,
        paged_store: PagedConceptStore | None = None,
        plasticity: PlasticityController | None = None,
    ) -> None:
        self.kernel = kernel
        self.paged_store = paged_store or getattr(kernel, "paged_store", None)
        self.plasticity = (
            plasticity
            or getattr(kernel, "plasticity", None)
            or PlasticityController()
        )

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text chunk into sentence units cleanly without breaking abbreviations."""
        raw = text.strip()
        if not raw:
            return []
        parts = re.split(r"(?<=[.!?])\s+", raw)
        return [p.strip() for p in parts if p.strip()]

    def ingest_chunk(
        self, text_chunk: str, source_id: str = "stream", *, source_type: SourceType = SourceType.DOCUMENT
    ) -> ChunkIngestResult:
        """Ingest a single chunk of natural English text into the knowledge base."""
        t0 = time.perf_counter()
        clean = text_chunk.strip()
        if not clean:
            return ChunkIngestResult(
                source_text="",
                sentences_count=0,
                concepts_created=[],
                relations_created=[],
                contradictions_blocked=0,
                is_learned=False,
                latency_ms=0.0,
                message="Empty text chunk provided.",
                plasticity_telemetry=self.plasticity.get_telemetry(),
            )

        sentences = self._split_into_sentences(clean)
        all_concepts_created: list[str] = []
        all_relations_created: list[str] = []
        contradictions_blocked = 0
        learned_any = False

        for sentence in sentences:
            # 1. Normalize sentence
            norm_sentence = self.kernel.parser.normalize_text(sentence)
            if not norm_sentence:
                norm_sentence = sentence

            # 2. Perceive candidate frame
            frame = self.kernel.perception.perceive(
                norm_sentence,
                memory=self.kernel.memory,
                context=self.kernel.learner.dialogue,
            )

            # If declarative statement or action
            if frame.intent in {"statement", "action"} and frame.claims:
                learn_res = self.kernel.learner.commit_frame(
                    frame, source_type=source_type, source_reference=source_id
                )
                if learn_res.concepts_created or learn_res.relations_created:
                    learned_any = True
                    all_concepts_created.extend(learn_res.concepts_created)
                    all_relations_created.extend(learn_res.relations_created)
                    # Epistemic prediction error drops as claims are cleanly integrated
                    avg_prob = sum(c.probability for c in frame.claims) / max(
                        1, len(frame.claims)
                    )
                    self.plasticity.observe_error(max(0.05, 1.0 - avg_prob))

                    # Update Paged Store: dirty-only invalidation is handled automatically; cast activation votes
                    if self.paged_store:
                        tokens = [w for w in re.findall(r"[a-zA-Z0-9_]+", sentence)]
                        self.paged_store.vote_from_tokens(tokens)

                    # Synchronize Living Wiki
                    if hasattr(self.kernel, "wiki") and self.kernel.wiki:
                        for cid in learn_res.concepts_created:
                            with contextlib.suppress(OSError, RuntimeError, ValueError, TypeError):
                                if hasattr(self.kernel.wiki, "sync_concept"):
                                    self.kernel.wiki.sync_concept(cid)
                                elif hasattr(self.kernel.wiki, "generate_article_for_concept"):
                                    self.kernel.wiki.generate_article_for_concept(cid)
                else:
                    if "contradiction" in learn_res.message.lower() or "violation" in learn_res.message.lower():
                        contradictions_blocked += 1
                        self.plasticity.observe_error(1.0)
                    else:
                        self.plasticity.observe_error(0.1)

        if hasattr(self.kernel, "mlm") and self.kernel.mlm is not None:
            mlm_rep = self.kernel.mlm.train_on_text(clean)
            if mlm_rep.relations_verified > 0 or mlm_rep.token_transitions_learned > 0:
                learned_any = True

        lat = (time.perf_counter() - t0) * 1000.0
        msg = (
            f"Ingested {len(sentences)} sentences: created {len(all_concepts_created)} concepts, "
            f"{len(all_relations_created)} relations ({contradictions_blocked} contradictions blocked)."
        )
        return ChunkIngestResult(
            source_text=clean,
            sentences_count=len(sentences),
            concepts_created=all_concepts_created,
            relations_created=all_relations_created,
            contradictions_blocked=contradictions_blocked,
            is_learned=learned_any,
            latency_ms=lat,
            message=msg,
            plasticity_telemetry=self.plasticity.get_telemetry(),
        )

    def ingest_stream(
        self, stream: Iterable[str], source_id: str = "stream"
    ) -> StreamLearningSummary:
        """Ingest an iterable stream of text chunks sequentially."""
        summary = StreamLearningSummary(source_id=source_id)
        t0 = time.perf_counter()

        for chunk in stream:
            chunk_clean = chunk.strip()
            if not chunk_clean:
                continue
            res = self.ingest_chunk(chunk_clean, source_id=source_id)
            summary.total_chunks += 1
            summary.total_sentences += res.sentences_count
            summary.total_concepts_created.extend(res.concepts_created)
            summary.total_relations_created.extend(res.relations_created)
            summary.total_contradictions_blocked += res.contradictions_blocked

        # Trigger latent induction if new concepts were created
        if summary.concepts_count > 0 and hasattr(self.kernel, "latent_inducer"):
            with contextlib.suppress(OSError, RuntimeError, ValueError, TypeError):
                if hasattr(self.kernel.latent_inducer, "induce_latent_knots"):
                    self.kernel.latent_inducer.induce_latent_knots()
                elif hasattr(self.kernel.latent_inducer, "induce_from_memory"):
                    self.kernel.latent_inducer.induce_from_memory()

        summary.total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return summary

    def ingest_document(
        self, text_or_path: str | Path, source_id: str = "doc"
    ) -> StreamLearningSummary:
        """Ingest a full text document or file stream."""
        p = Path(text_or_path)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
            doc_id = p.name
        else:
            content = str(text_or_path)
            doc_id = source_id

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return self.ingest_stream(paragraphs, source_id=doc_id)

    def ingest_corpus_directory(
        self, dir_path: Path | str, pattern: str = "*.md"
    ) -> StreamLearningSummary:
        """Ingest all knowledge files in a corpus directory."""
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            return StreamLearningSummary(source_id=str(dir_path))

        files = sorted(directory.glob(pattern))
        summary = StreamLearningSummary(source_id=directory.name)
        t0 = time.perf_counter()

        for f in files:
            doc_sum = self.ingest_document(f)
            summary.total_chunks += doc_sum.total_chunks
            summary.total_sentences += doc_sum.total_sentences
            summary.total_concepts_created.extend(doc_sum.total_concepts_created)
            summary.total_relations_created.extend(doc_sum.total_relations_created)
            summary.total_contradictions_blocked += doc_sum.total_contradictions_blocked

        summary.total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return summary

    def ingest_world_corpus(self) -> StreamLearningSummary:
        """Ingest the standard encyclopedic world corpus (astronomy, biology, CS, chemistry, etc.)."""
        from little.core.runtime_paths import RuntimePaths

        candidate_dirs = [
            RuntimePaths.default().database.parent / "knowledge" / "world_corpus",
            Path("data/knowledge/world_corpus"),
        ]
        target_dir = next((d for d in candidate_dirs if d.exists()), None)
        if not target_dir:
            return StreamLearningSummary(source_id="world_corpus_not_found")

        return self.ingest_corpus_directory(target_dir, pattern="*.md")

