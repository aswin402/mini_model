"""Extensible Command Registry for Cognitive and Epistemic Directives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from little.core.kernel import CognitiveKernel


@dataclass(frozen=True)
class CognitiveCommand:
    """A registered cognitive directive with compiled match patterns and execution handler."""

    command_id: str
    patterns: tuple[re.Pattern[str], ...]
    handler: Callable[[CognitiveKernel, str, re.Match[str]], str]
    description: str = ""


class CognitiveCommandRegistry:
    """Dispatches meta-cognitive instructions (crawling, wiki search, latent knot induction)."""

    def __init__(self, commands: list[CognitiveCommand] | None = None) -> None:
        self._commands: list[CognitiveCommand] = (
            list(commands) if commands is not None else self._default_commands()
        )

    def register(self, command: CognitiveCommand) -> None:
        """Register a new cognitive command handler."""
        self._commands.append(command)

    def dispatch(self, utterance: str, kernel: CognitiveKernel) -> str | None:
        """Attempt to match utterance against registered cognitive commands and execute."""
        clean = utterance.strip()
        for cmd in self._commands:
            for pat in cmd.patterns:
                m = pat.search(clean)
                if m:
                    return cmd.handler(kernel, clean, m)
        return None

    @classmethod
    def _default_commands(cls) -> list[CognitiveCommand]:
        """Construct the default built-in cognitive commands."""
        return [
            CognitiveCommand(
                command_id="knowledge_ingestion",
                description="Autonomous knowledge crawling and document ingestion",
                patterns=(
                    re.compile(
                        r"^(?:study|crawl|ingest)\s+(?:text|document)\s*:\s*(?P<doc_text>.+)$",
                        re.IGNORECASE | re.DOTALL,
                    ),
                ),
                handler=cls._handle_knowledge_ingestion,
            ),
            CognitiveCommand(
                command_id="wiki_semantic_search",
                description="TurboQuant semantic search over living knowledge wiki",
                patterns=(
                    re.compile(
                        r"^search\s+(?:the\s+)?wiki\s+for\s+(?P<query>.+?)\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_wiki_semantic_search,
            ),
            CognitiveCommand(
                command_id="wiki_backlinks",
                description="Query incoming graph backlinks for a concept",
                patterns=(
                    re.compile(
                        r"^(?:backlinks\s+to|what\s+links\s+to|what\s+are\s+the\s+backlinks\s+to)\s+(?P<target>[^?.!]+)\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_wiki_backlinks,
            ),
            CognitiveCommand(
                command_id="wiki_page_summary",
                description="Retrieve OKF wiki page summary and structured relations",
                patterns=(
                    re.compile(
                        r"^(?:show\s+)?wiki\s+page\s+for\s+(?P<target>[^?.!]+)\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_wiki_page_summary,
            ),
            CognitiveCommand(
                command_id="concept_clustering",
                description="Autonomous Cobweb concept clustering and latent knot induction",
                patterns=(
                    re.compile(
                        r"^(?:run\s+)?(?:concept\s+clustering|concept\s+induction|cluster\s+concepts|discover\s+latent\s+(?:categories|knots)|induce\s+latent\s+knots)\b",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_concept_clustering,
            ),
            CognitiveCommand(
                command_id="list_latent_knots",
                description="List active latent categories and concepts in memory",
                patterns=(
                    re.compile(
                        r"^(?:what\s+latent\s+categories\s+(?:have\s+you\s+discovered|do\s+you\s+have)|(?:show|list)\s+latent\s+(?:knots|categories))\b",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_list_latent_knots,
            ),
            CognitiveCommand(
                command_id="explain_concept_cluster",
                description="Explain latent concept cluster membership and shared features",
                patterns=(
                    re.compile(
                        r"^explain\s+(?:concept\s+)?cluster\s+for\s+(?P<target>[^?.!]+)\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_explain_concept_cluster,
            ),
        ]

    @staticmethod
    def _handle_knowledge_ingestion(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        doc_text = match.group("doc_text").strip()
        report = kernel.crawler.crawl_text(doc_text, source_reference="chat_ingest")
        return kernel.dialogue_surface.verbalize_crawl_report(report)

    @staticmethod
    def _handle_wiki_semantic_search(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        query_str = match.group("query").strip()
        search_results = kernel.wiki.search_semantic(query_str, top_k=5)
        return kernel.dialogue_surface.verbalize_wiki_search(query_str, search_results)

    @staticmethod
    def _handle_wiki_backlinks(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        target = match.group("target").strip().lower()
        backlinks = kernel.wiki.get_backlinks(target)
        return kernel.dialogue_surface.verbalize_backlinks(target, backlinks)

    @staticmethod
    def _handle_wiki_page_summary(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        target = match.group("target").strip().lower()
        page = kernel.wiki.get_page(target)
        if page is not None:
            return kernel.dialogue_surface.verbalize_wiki_page(page)
        return f"I do not have a wiki page for '{target}' in my memory yet."

    @staticmethod
    def _handle_concept_clustering(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        report = kernel.latent_inducer.induce_latent_knots(min_cluster_size=2)
        return kernel.dialogue_surface.verbalize_latent_induction(report)

    @staticmethod
    def _handle_list_latent_knots(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        all_concepts = kernel.memory.list_concepts()
        latent_concepts = [
            c
            for c in all_concepts
            if c.name.startswith("latent_") or c.attributes.get("is_latent")
        ]
        return kernel.dialogue_surface.verbalize_latent_knots(latent_concepts)

    @staticmethod
    def _handle_explain_concept_cluster(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        target = match.group("target").strip().rstrip("?.!")
        info = kernel.latent_inducer.explain_concept_cluster(target)
        return kernel.dialogue_surface.verbalize_cluster_explanation(
            info, concept_name=target
        )
