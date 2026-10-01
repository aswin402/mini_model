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
            CognitiveCommand(
                command_id="dream_cycle",
                description="Autonomous cognitive dream cycle and curiosity-driven self-study",
                patterns=(
                    re.compile(
                        r"^(?:dream|run\s+(?:a\s+)?dream\s+cycle|run\s+(?:autonomous\s+)?self\s+study|curiosity\s+cycle|autonomous\s+study)\b",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_dream_cycle,
            ),
            CognitiveCommand(
                command_id="knowledge_gaps",
                description="Identify concepts with highest missing radial spoke entropy and wiki stubs",
                patterns=(
                    re.compile(
                        r"^(?:what\s+are\s+your\s+knowledge\s+gaps|show\s+knowledge\s+gaps|explore\s+knowledge\s+gaps|what\s+are\s+your\s+epistemic\s+gaps)\b",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_knowledge_gaps,
            ),
            CognitiveCommand(
                command_id="study_concept",
                description="Targeted autonomous curiosity inquiry on a specific concept",
                patterns=(
                    re.compile(
                        r"^(?:study|explore)\s+concept\s+(?P<target>[a-zA-Z0-9_\-]+)(?:\s+(?:on|for|axis)\s+(?P<axis>[a-zA-Z0-9_\-]+))?\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_study_concept,
            ),
            CognitiveCommand(
                command_id="simulate_dynamics",
                description="Predict continuous-time physical state evolution over arbitrary time deltas",
                patterns=(
                    re.compile(
                        r"^(?:simulate\s+(?:physical\s+)?dynamics\s+(?:of|for)|predict|what\s+happens\s+to|what\s+would\s+happen\s+if\s+I\s+leave)\s+(?:an?\s+)?(?P<target>[a-zA-Z0-9_\-]+)(?:\s+(?:for|over|after)\s+(?P<duration>\d+(?:\.\d+)?)\s*(?P<unit>hours?|days?|minutes?|hrs?|mins?))?(?:\s+at\s+(?P<temp>\d+(?:\.\d+)?)\s*(?:°?C|degrees?(?:\s+celsius)?|celsius|centigrade)?)?(?:.*)?\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_simulate_dynamics,
            ),
            CognitiveCommand(
                command_id="physical_action",
                description="Perform physical action jump (slice, peel, heat, chill, dehydrate) with QPT mass conservation",
                patterns=(
                    re.compile(
                        r"^(?:apply\s+action|perform\s+action|what\s+happens\s+if\s+I|if\s+I)\s+(?P<action>slice|peel|heat|chill|cool|dehydrate|dry)\s+(?:an?\s+)?(?P<target>[a-zA-Z0-9_\-]+)(?:\s+(?:into|in)\s+(?P<count>\d+)\s*(?:pieces|slices|wedges)?)?(?:.*)?\??$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_physical_action,
            ),
            CognitiveCommand(
                command_id="sensory_observation",
                description="Ground multimodal sensory observations to 0° episodic instance axis",
                patterns=(
                    re.compile(
                        r"^(?:sensory\s+observation|observe\s+sensor|perceive\s+sensor):\s*(?P<reading>.+)$",
                        re.IGNORECASE,
                    ),
                ),
                handler=cls._handle_sensory_observation,
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

    @staticmethod
    def _handle_dream_cycle(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        steps = 1
        m_steps = re.search(r"\b(\d+)\s+steps?\b", text, re.IGNORECASE)
        if m_steps:
            steps = min(5, int(m_steps.group(1)))
        report = kernel.self_study.dream(max_steps=steps)
        return kernel.dialogue_surface.verbalize_dream_report(report)

    @staticmethod
    def _handle_knowledge_gaps(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        gaps = kernel.self_study.get_epistemic_gaps(top_n=5)
        return kernel.dialogue_surface.verbalize_epistemic_gaps(gaps)

    @staticmethod
    def _handle_study_concept(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        target = match.group("target").strip().lower()
        axis_match = match.groupdict().get("axis")
        preferred_axis = axis_match.strip().lower() if axis_match else None
        res = kernel.self_study.study_step(target_concept=target, preferred_axis=preferred_axis)
        if res is not None:
            return kernel.dialogue_surface.verbalize_curiosity_step(res)
        return f"No active knowledge gaps found for concept '{target}'."

    @staticmethod
    def _handle_simulate_dynamics(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        target = match.group("target").strip().lower()
        duration_val = match.groupdict().get("duration")
        unit_val = match.groupdict().get("unit")
        temp_val = match.groupdict().get("temp")

        hours = 24.0
        if duration_val:
            d = float(duration_val)
            if unit_val and "day" in unit_val.lower():
                hours = d * 24.0
            elif unit_val and ("min" in unit_val.lower()):
                hours = d / 60.0
            else:
                hours = d

        temp = float(temp_val) if temp_val else 22.0

        from little.core.concept_knot import ConceptKnot
        knot = None
        if target == "apple":
            knot = ConceptKnot.create_apple_exemplar()
        sim_res = kernel.simulation_engine.simulate(
            knot if knot else target,
            delta_t_hours=hours,
            ambient_temperature_c=temp,
            skin_intact=True,
        )
        return kernel.dialogue_surface.verbalize_physical_simulation(sim_res)

    @staticmethod
    def _handle_physical_action(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        from little.core.concept_knot import ConceptKnot
        from little.dynamics.cfc_ode import apply_action_jump
        from little.dynamics.transformations import TransformationEngine

        action = match.group("action").strip().lower()
        target = match.group("target").strip().lower()
        count_str = match.groupdict().get("count")
        count = int(count_str) if count_str else 4

        knot = ConceptKnot.create_apple_exemplar() if target == "apple" else ConceptKnot(
            concept_id=target.upper(),
            taxonomy_hypernyms=["PhysicalObject"],
            mereology_parts={"skin": "external_boundary", "pulp": "interior"},
            invariant_mass=180.0,
            procedural_skills=[action],
        )

        if action in ("slice", "split"):
            pieces = apply_action_jump(knot, "slice", num_pieces=count)
            TransformationEngine.slice_object(kernel.memory, target, num_pieces=count)
            return kernel.dialogue_surface.verbalize_action_jump(
                action, target, pieces,
                details=f"Decomposed into {count} pieces. Total invariant mass conserved under QPT."
            )
        elif action == "peel":
            pieces = apply_action_jump(knot, "peel")
            TransformationEngine.peel_object(kernel.memory, target)
            return kernel.dialogue_surface.verbalize_action_jump(
                action, target, pieces,
                details="Separated external protective skin from interior flesh. Total invariant mass conserved under QPT."
            )
        elif action == "heat":
            pieces = apply_action_jump(knot, "heat", delta_celsius=20.0)
            TransformationEngine.heat_object(kernel.memory, target, delta_celsius=20.0)
            return kernel.dialogue_surface.verbalize_action_jump(
                action, target, pieces,
                details="Temperature increased by +20.0°C. Arrhenius reaction rates accelerated."
            )
        elif action in ("chill", "cool"):
            pieces = apply_action_jump(knot, "chill", delta_celsius=15.0)
            TransformationEngine.chill_object(kernel.memory, target, delta_celsius=15.0)
            return kernel.dialogue_surface.verbalize_action_jump(
                action, target, pieces,
                details="Temperature reduced by -15.0°C. Freshness preserved."
            )
        elif action in ("dehydrate", "dry"):
            pieces = apply_action_jump(knot, "dehydrate")
            return kernel.dialogue_surface.verbalize_action_jump(
                action, target, pieces,
                details="Moisture reduced to minimal bound. Dry solid mass strictly conserved."
            )
        else:
            pieces = apply_action_jump(knot, action)
            return kernel.dialogue_surface.verbalize_action_jump(action, target, pieces)

    @staticmethod
    def _handle_sensory_observation(
        kernel: CognitiveKernel, text: str, match: re.Match[str]
    ) -> str:
        obs = kernel.sensory_engine.parse_sensory_directive(text)
        if not obs:
            return "Could not parse structured sensory directive format."
        res = kernel.sensory_engine.ground_observation(obs)
        return kernel.dialogue_surface.verbalize_sensory_grounding(res)

