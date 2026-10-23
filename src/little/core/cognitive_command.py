"""Extensible Command Registry for Cognitive and Epistemic Directives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from little.core.kernel import CognitiveKernel


@dataclass(frozen=True)
class _CommandMatch:
    """Lightweight match result supporting .group() and .groupdict() without regex."""

    groups_map: dict[str, str | None]

    def group(self, key: str | int = 0) -> str:
        if isinstance(key, int):
            vals = list(self.groups_map.values())
            return (vals[key - 1] or "") if 1 <= key <= len(vals) else ""
        return self.groups_map.get(key) or ""

    def groupdict(self) -> dict[str, str | None]:
        return dict(self.groups_map)


@dataclass(frozen=True)
class _DirectiveMatcher:
    """Callable-backed directive matcher providing a .search(text) interface."""

    matcher_fn: Callable[[str], dict[str, str | None] | None]

    def search(self, text: str) -> _CommandMatch | None:
        res = self.matcher_fn(text)
        return _CommandMatch(res) if res is not None else None


def _is_ident_slug(token: str) -> bool:
    return bool(token) and all(c.isalnum() or c in "_-" for c in token)


def _is_num(token: str) -> bool:
    s = token.strip()
    if not s:
        return False
    if s.count(".") <= 1 and s.replace(".", "", 1).isdigit():
        return True
    return False


def _colon_directive_matcher(
    prefixes: tuple[str, ...], group_name: str
) -> _DirectiveMatcher:
    def _match(text: str) -> dict[str, str | None] | None:
        if ":" not in text:
            return None
        head, tail = text.split(":", 1)
        norm_head = " ".join(head.lower().split())
        if norm_head in prefixes and tail.strip():
            return {group_name: tail.strip()}
        return None

    return _DirectiveMatcher(_match)


def _prefix_target_matcher(
    prefixes: tuple[str, ...], group_name: str, allow_punct: bool = False
) -> _DirectiveMatcher:
    def _match(text: str) -> dict[str, str | None] | None:
        stripped = text.strip()
        if stripped.endswith("?"):
            stripped = stripped[:-1].strip()
        tokens = stripped.split()
        low_tokens = [t.lower() for t in tokens]
        for pref in prefixes:
            pref_tokens = pref.split()
            n = len(pref_tokens)
            if len(tokens) > n and low_tokens[:n] == pref_tokens:
                target = " ".join(tokens[n:]).strip()
                if not target:
                    continue
                if not allow_punct and any(c in target for c in "?.!"):
                    continue
                return {group_name: target}
        return None

    return _DirectiveMatcher(_match)


def _trigger_prefix_matcher(
    prefixes: tuple[str, ...], exact_only: bool = False
) -> _DirectiveMatcher:
    def _match(text: str) -> dict[str, str | None] | None:
        stripped = text.strip()
        if stripped.endswith("?"):
            stripped = stripped[:-1].strip()
        norm = " ".join(stripped.lower().split())
        if exact_only:
            return {} if norm in prefixes else None
        for pref in prefixes:
            if norm == pref or norm.startswith(pref + " "):
                return {}
        return None

    return _DirectiveMatcher(_match)


def _match_study_concept(text: str) -> dict[str, str | None] | None:
    stripped = text.strip()
    if stripped.endswith("?"):
        stripped = stripped[:-1].strip()
    tokens = stripped.split()
    if len(tokens) not in (3, 5):
        return None
    if tokens[0].lower() not in ("study", "explore") or tokens[1].lower() != "concept":
        return None
    target = tokens[2]
    if not _is_ident_slug(target):
        return None
    axis: str | None = None
    if len(tokens) == 5:
        if tokens[3].lower() not in ("on", "for", "axis") or not _is_ident_slug(tokens[4]):
            return None
        axis = tokens[4]
    return {"target": target, "axis": axis}


def _match_simulate_dynamics(text: str) -> dict[str, str | None] | None:
    stripped = text.strip()
    if stripped.endswith("?"):
        stripped = stripped[:-1].strip()
    tokens = stripped.split()
    low_tokens = [t.lower() for t in tokens]
    prefixes = (
        ("simulate", "physical", "dynamics", "of"),
        ("simulate", "physical", "dynamics", "for"),
        ("simulate", "dynamics", "of"),
        ("simulate", "dynamics", "for"),
        ("what", "would", "happen", "if", "i", "leave"),
        ("what", "happens", "to"),
        ("predict",),
    )
    idx = -1
    for p in prefixes:
        if len(low_tokens) > len(p) and tuple(low_tokens[: len(p)]) == p:
            idx = len(p)
            break
    if idx == -1:
        return None
    if idx < len(low_tokens) and low_tokens[idx] in ("a", "an"):
        idx += 1
    if idx >= len(tokens) or not _is_ident_slug(tokens[idx]):
        return None
    target = tokens[idx]
    idx += 1

    duration: str | None = None
    unit: str | None = None
    temp: str | None = None
    valid_units = {"hour", "hours", "day", "days", "minute", "minutes", "hr", "hrs", "min", "mins"}

    while idx < len(tokens):
        lt = low_tokens[idx]
        if lt in ("for", "over", "after") and idx + 1 < len(tokens):
            cand = tokens[idx + 1]
            # Handle attached unit like "10days" or separated "10 days"
            num_part = ""
            unit_part = ""
            for ch in cand:
                if (ch.isdigit() or ch == ".") and not unit_part:
                    num_part += ch
                else:
                    unit_part += ch
            if _is_num(num_part):
                if unit_part.lower() in valid_units:
                    duration = num_part
                    unit = unit_part
                    idx += 2
                    continue
                elif not unit_part and idx + 2 < len(tokens) and low_tokens[idx + 2].rstrip(".,") in valid_units:
                    duration = num_part
                    unit = tokens[idx + 2].rstrip(".,")
                    idx += 3
                    continue
        if lt == "at" and idx + 1 < len(tokens):
            cand_t = tokens[idx + 1].rstrip("°cC.,")
            if _is_num(cand_t):
                temp = cand_t
                idx += 2
                continue
        idx += 1

    return {"target": target, "duration": duration, "unit": unit, "temp": temp}


def _match_physical_action(text: str) -> dict[str, str | None] | None:
    stripped = text.strip()
    if stripped.endswith("?"):
        stripped = stripped[:-1].strip()
    tokens = stripped.split()
    low_tokens = [t.lower() for t in tokens]
    prefixes = (
        ("apply", "action"),
        ("perform", "action"),
        ("what", "happens", "if", "i"),
        ("if", "i"),
    )
    idx = -1
    for p in prefixes:
        if len(low_tokens) > len(p) and tuple(low_tokens[: len(p)]) == p:
            idx = len(p)
            break
    if idx == -1 or idx >= len(low_tokens):
        return None
    action = low_tokens[idx]
    if action not in ("slice", "peel", "heat", "chill", "cool", "dehydrate", "dry"):
        return None
    idx += 1
    if idx < len(low_tokens) and low_tokens[idx] in ("a", "an"):
        idx += 1
    if idx >= len(tokens):
        return None
    target = tokens[idx].rstrip(".,?!")
    if not _is_ident_slug(target):
        return None
    idx += 1

    count: str | None = None
    while idx < len(tokens):
        if low_tokens[idx] in ("into", "in") and idx + 1 < len(tokens):
            cand = tokens[idx + 1]
            digits = "".join(ch for ch in cand if ch.isdigit())
            if digits:
                count = digits
                break
        idx += 1

    return {"action": action, "target": target, "count": count}


@dataclass(frozen=True)
class CognitiveCommand:
    """A registered cognitive directive with matchers and execution handler."""

    command_id: str
    patterns: tuple[Any, ...]
    handler: Callable[[CognitiveKernel, str, Any], str]
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
                m = pat.search(clean) if hasattr(pat, "search") else pat(clean)
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
                    _colon_directive_matcher(
                        (
                            "study text",
                            "crawl text",
                            "ingest text",
                            "study document",
                            "crawl document",
                            "ingest document",
                        ),
                        "doc_text",
                    ),
                ),
                handler=cls._handle_knowledge_ingestion,
            ),
            CognitiveCommand(
                command_id="wiki_semantic_search",
                description="TurboQuant semantic search over living knowledge wiki",
                patterns=(
                    _prefix_target_matcher(
                        ("search the wiki for", "search wiki for"),
                        "query",
                        allow_punct=True,
                    ),
                ),
                handler=cls._handle_wiki_semantic_search,
            ),
            CognitiveCommand(
                command_id="wiki_backlinks",
                description="Query incoming graph backlinks for a concept",
                patterns=(
                    _prefix_target_matcher(
                        (
                            "what are the backlinks to",
                            "what links to",
                            "backlinks to",
                        ),
                        "target",
                        allow_punct=False,
                    ),
                ),
                handler=cls._handle_wiki_backlinks,
            ),
            CognitiveCommand(
                command_id="wiki_page_summary",
                description="Retrieve OKF wiki page summary and structured relations",
                patterns=(
                    _prefix_target_matcher(
                        ("show wiki page for", "wiki page for"),
                        "target",
                        allow_punct=False,
                    ),
                ),
                handler=cls._handle_wiki_page_summary,
            ),
            CognitiveCommand(
                command_id="concept_clustering",
                description="Autonomous Cobweb concept clustering and latent knot induction",
                patterns=(
                    _trigger_prefix_matcher(
                        (
                            "run concept clustering",
                            "concept clustering",
                            "run concept induction",
                            "concept induction",
                            "run cluster concepts",
                            "cluster concepts",
                            "run discover latent categories",
                            "discover latent categories",
                            "run discover latent knots",
                            "discover latent knots",
                            "run induce latent knots",
                            "induce latent knots",
                        )
                    ),
                ),
                handler=cls._handle_concept_clustering,
            ),
            CognitiveCommand(
                command_id="list_latent_knots",
                description="List active latent categories and concepts in memory",
                patterns=(
                    _trigger_prefix_matcher(
                        (
                            "what latent categories have you discovered",
                            "what latent categories do you have",
                            "show latent knots",
                            "show latent categories",
                            "list latent knots",
                            "list latent categories",
                        )
                    ),
                ),
                handler=cls._handle_list_latent_knots,
            ),
            CognitiveCommand(
                command_id="explain_concept_cluster",
                description="Explain latent concept cluster membership and shared features",
                patterns=(
                    _prefix_target_matcher(
                        ("explain concept cluster for", "explain cluster for"),
                        "target",
                        allow_punct=False,
                    ),
                ),
                handler=cls._handle_explain_concept_cluster,
            ),
            CognitiveCommand(
                command_id="dream_cycle",
                description="Autonomous cognitive dream cycle and curiosity-driven self-study",
                patterns=(
                    _trigger_prefix_matcher(
                        (
                            "run a dream cycle",
                            "run dream cycle",
                            "run autonomous self study",
                            "run self study",
                            "curiosity cycle",
                            "autonomous study",
                            "dream",
                        )
                    ),
                ),
                handler=cls._handle_dream_cycle,
            ),
            CognitiveCommand(
                command_id="knowledge_gaps",
                description="Identify concepts with highest missing radial spoke entropy and wiki stubs",
                patterns=(
                    _trigger_prefix_matcher(
                        (
                            "what are your knowledge gaps",
                            "show knowledge gaps",
                            "explore knowledge gaps",
                            "what are your epistemic gaps",
                        )
                    ),
                ),
                handler=cls._handle_knowledge_gaps,
            ),
            CognitiveCommand(
                command_id="study_concept",
                description="Targeted autonomous curiosity inquiry on a specific concept",
                patterns=(_DirectiveMatcher(_match_study_concept),),
                handler=cls._handle_study_concept,
            ),
            CognitiveCommand(
                command_id="simulate_dynamics",
                description="Predict continuous-time physical state evolution over arbitrary time deltas",
                patterns=(_DirectiveMatcher(_match_simulate_dynamics),),
                handler=cls._handle_simulate_dynamics,
            ),
            CognitiveCommand(
                command_id="physical_action",
                description="Perform physical action jump (slice, peel, heat, chill, dehydrate) with QPT mass conservation",
                patterns=(_DirectiveMatcher(_match_physical_action),),
                handler=cls._handle_physical_action,
            ),
            CognitiveCommand(
                command_id="sensory_observation",
                description="Ground multimodal sensory observations to 0° episodic instance axis",
                patterns=(
                    _colon_directive_matcher(
                        ("sensory observation", "observe sensor", "perceive sensor"),
                        "reading",
                    ),
                ),
                handler=cls._handle_sensory_observation,
            ),
            CognitiveCommand(
                command_id="cognitive_status",
                description="Inspect live cognitive telemetry (paging, plasticity, growth brakes, canary, 360° MLM)",
                patterns=(
                    _trigger_prefix_matcher(
                        (
                            "show cognitive status",
                            "cognitive status",
                            "show telemetry status",
                            "telemetry status",
                            "show plasticity status",
                            "plasticity status",
                            "show growth brakes",
                            "growth brakes",
                            "show mlu status",
                            "mlu status",
                            "show mlm status",
                            "mlm status",
                        ),
                        exact_only=True,
                    ),
                ),
                handler=cls._handle_cognitive_status,
            ),
            CognitiveCommand(
                command_id="mlm_train",
                description="Train the 360° Spider-Web Mini Learning Model (MLM/MLU) on raw text",
                patterns=(
                    _colon_directive_matcher(
                        ("mlm train", "train mlm", "mlu train", "train mlu"),
                        "train_text",
                    ),
                ),
                handler=cls._handle_mlm_train,
            ),
            CognitiveCommand(
                command_id="mlm_ask",
                description="Query the 360° Spider-Web Mini Learning Model (MLM/MLU)",
                patterns=(
                    _colon_directive_matcher(
                        ("mlm ask", "ask mlm", "mlu ask", "ask mlu"),
                        "query_text",
                    ),
                ),
                handler=cls._handle_mlm_ask,
            ),
        ]

    @staticmethod
    def _handle_knowledge_ingestion(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        doc_text = match.group("doc_text").strip()
        report = kernel.crawler.crawl_text(doc_text, source_reference="chat_ingest")
        return kernel.dialogue_surface.verbalize_crawl_report(report)

    @staticmethod
    def _handle_wiki_semantic_search(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        query_str = match.group("query").strip()
        search_results = kernel.wiki.search_semantic(query_str, top_k=5)
        return kernel.dialogue_surface.verbalize_wiki_search(query_str, search_results)

    @staticmethod
    def _handle_wiki_backlinks(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        target = match.group("target").strip().lower()
        backlinks = kernel.wiki.get_backlinks(target)
        return kernel.dialogue_surface.verbalize_backlinks(target, backlinks)

    @staticmethod
    def _handle_wiki_page_summary(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        target = match.group("target").strip().lower()
        page = kernel.wiki.get_page(target)
        if page is not None:
            return kernel.dialogue_surface.verbalize_wiki_page(page)
        return f"I do not have a wiki page for '{target}' in my memory yet."

    @staticmethod
    def _handle_concept_clustering(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        report = kernel.latent_inducer.induce_latent_knots(min_cluster_size=2)
        return kernel.dialogue_surface.verbalize_latent_induction(report)

    @staticmethod
    def _handle_list_latent_knots(
        kernel: CognitiveKernel, text: str, match: Any
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
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        target = match.group("target").strip().rstrip("?.!")
        info = kernel.latent_inducer.explain_concept_cluster(target)
        return kernel.dialogue_surface.verbalize_cluster_explanation(
            info, concept_name=target
        )

    @staticmethod
    def _handle_dream_cycle(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        steps = 1
        words = text.lower().split()
        for idx, w in enumerate(words[:-1]):
            if w.isdigit() and words[idx + 1].rstrip("?.!,") in ("step", "steps"):
                steps = min(5, int(w))
                break
        report = kernel.self_study.dream(max_steps=steps)
        return kernel.dialogue_surface.verbalize_dream_report(report)

    @staticmethod
    def _handle_knowledge_gaps(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        gaps = kernel.self_study.get_epistemic_gaps(top_n=5)
        return kernel.dialogue_surface.verbalize_epistemic_gaps(gaps)

    @staticmethod
    def _handle_study_concept(
        kernel: CognitiveKernel, text: str, match: Any
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
        kernel: CognitiveKernel, text: str, match: Any
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
        kernel: CognitiveKernel, text: str, match: Any
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
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        obs = kernel.sensory_engine.parse_sensory_directive(text)
        if not obs:
            return "Could not parse structured sensory directive format."
        res = kernel.sensory_engine.ground_observation(obs)
        return kernel.dialogue_surface.verbalize_sensory_grounding(res)

    @staticmethod
    def _handle_cognitive_status(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        snap = kernel.get_telemetry_snapshot()
        mem = snap["memory"]
        paged = snap["paged_store"]
        plast = snap["plasticity"]
        brakes = snap["growth_brakes"]
        canary = snap["canary"]
        mlm = snap.get("mlm_spiderweb", {})
        return (
            f"Cognitive Telemetry: {mem['total_concepts']} concepts, {mem['total_relations']} relations "
            f"(epoch {mem['concept_mutation_epoch']}) | "
            f"Paged Store: {paged['active_count']} resident, {paged['slot_reuses']} slot reuses, "
            f"{paged['dirty_invalidations']} dirty invalidations | "
            f"Plasticity: lr={plast['effective_lr']:.4f} (e_slow={plast['slow_effect_size']:+.2f}, "
            f"shifts={plast['regime_shifts']}, spikes_held={plast['spikes_held_back']}) | "
            f"360° Spider-Web MLM: {mlm.get('total_mlu_nodes', 0)} MLUs "
            f"({mlm.get('subcore_hubs', 0)} Sub-Cores, {mlm.get('mitosis_promotions', 0)} mitosis) | "
            f"Growth Brakes: {'OPEN' if brakes['allowed'] else 'ENGAGED'} | "
            f"Canary: {'PASS' if canary['passed'] else 'FAIL'} ({canary['checks_run']} checks)."
        )

    @staticmethod
    def _handle_mlm_train(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        raw_text = match.group("train_text").strip()
        rep = kernel.mlm.train_on_text(raw_text)
        mitosis_str = (
            f" | Fractal Mitosis -> Sub-Cores: {', '.join(rep.mitosis_promotions)}"
            if rep.mitosis_promotions
            else ""
        )
        return (
            f"[360° Spider-Web MLM Trained] {rep.sentences_processed} sentences | "
            f"{rep.relations_verified} verified 360° spoke links, {rep.relations_refuted} refuted | "
            f"{rep.token_transitions_learned} token transitions | "
            f"surprise={rep.average_surprise:.3f}, plasticity={rep.plasticity_multiplier:.3f} | "
            f"Topology: {rep.total_mlu_nodes} MLUs ({rep.active_subcores} Sub-Cores){mitosis_str}."
        )

    @staticmethod
    def _handle_mlm_ask(
        kernel: CognitiveKernel, text: str, match: Any
    ) -> str:
        q_text = match.group("query_text").strip()
        res = kernel.mlm.query_360(q_text)
        return f"{res.answer}\n  {res.inspectable_trace}"



