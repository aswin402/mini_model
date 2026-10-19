"""360-Degree Fractal Spider-Web Mini Learning Model (SpiderWebMLM).

Architecture:
- Central Core (`CORE_ROOT`): Always-resident LFM + JEPA global coordinator.
- 360° Sub-Core Ring (`SUBCORE_*`): Domain hubs arranged radially around the Central Core.
- 360° Spoke Leaf MLUs: Individual concept MiniLearningUnits attached to Sub-Core hubs,
  each possessing 6 radial spoke heads (`taxonomy`, `mereology`, `attributes`, `dynamics`,
  `procedural`, `episodic`).
- Fractal Mitosis & 5-Brake Growth: Leaf MLUs that accumulate high traffic and verified
  multi-spoke connections dynamically promote into new Sub-Core hubs.
- DeepSeek-R1 Verification-Guided Online Plasticity: Every relational update is checked by
  `DeepSeekInvariantVerifier` (I_DAG, I_MUTEX, I_SORT, I_TEMP) and scaled by `PlasticityController`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from little.active.plasticity import PlasticityController
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.memory.store import MemoryStore
from little.mlu.unit import (
    SPOKE_NAMES,
    MiniLearningUnit,
    MLUSpokePrediction,
    deterministic_seed_vector,
    map_predicate_to_spoke,
)

DEFAULT_SUBCORES: tuple[str, ...] = (
    "SUBCORE_TAXONOMY",
    "SUBCORE_MEREOLOGY",
    "SUBCORE_ATTRIBUTES",
    "SUBCORE_DYNAMICS",
    "SUBCORE_PROCEDURAL",
    "SUBCORE_EPISODIC",
)

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "where",
        "when",
        "does",
        "do",
        "did",
        "tell",
        "me",
        "about",
    }
)

# Natural-language relational patterns for open-stream extraction
STREAM_RELATION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:is\s+a|is\s+an|are\s+a|are)\s+(?:kind\s+of\s+|type\s+of\s+)?([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "is_a",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:is\s+not\s+a|is\s+not\s+an|cannot\s+be\s+a|is\s+disjoint\s+with)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "disjoint_with",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:is\s+part\s+of|belongs\s+to)\s+(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "part_of",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:has\s+a|has\s+an|has|contains|consists\s+of)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "has_part",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:can|is\s+able\s+to|is\s+capable\s+of)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "can",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:is\s+used\s+for|is\s+used\s+to|helps\s+to)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "used_for",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:causes|leads\s+to|triggers|produces|creates)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "causes",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:requires|needs|depends\s+on)\s+([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "requires",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:lives\s+in|is\s+found\s+in|is\s+located\s+in|inhabits)\s+(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+)$",
            re.IGNORECASE,
        ),
        "located_in",
    ),
    (
        re.compile(
            r"^(?:(?:the|an|a)\b\s+)?([a-z0-9_\s]+?)\s+(?:is|are)\s+([a-z0-9_]+)$",
            re.IGNORECASE,
        ),
        "has_property",
    ),
)


@dataclass(frozen=True)
class MLMTrainingStepReport:
    """Telemetry emitted after training the 360° Spider-Web MLM on a text stream."""

    sentences_processed: int
    relations_verified: int
    relations_refuted: int
    token_transitions_learned: int
    average_surprise: float
    plasticity_multiplier: float
    active_subcores: int
    total_mlu_nodes: int
    mitosis_promotions: list[str] = field(default_factory=list)
    pruned_nodes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MLMQueryResult:
    """Grounded 360° neural-symbolic response from `SpiderWebMLM`."""

    query: str
    focal_concept: str
    status: str  # "VERIFIED" | "NEURAL_GROUNDED" | "REFUTED" | "UNKNOWN"
    answer: str
    confidence: float
    epistemic_entropy: float
    active_subcore: str
    activated_mlus: list[str]
    spoke_predictions: list[MLUSpokePrediction]
    inspectable_trace: str


class SpiderWebMLM:
    """360-Degree Fractal Spider-Web Mini Learning Model (MLM / MLU Network).

    Combines a dynamically growing hierarchy of `MiniLearningUnit` cells
    (`Central Core -> Sub-Core Hubs -> 360° Spoke MLUs`) with LITTLE's
    `MemoryStore`, `DeepSeekInvariantVerifier`, and `PlasticityController`.
    """

    def __init__(
        self,
        memory: MemoryStore,
        verifier: DeepSeekInvariantVerifier | None = None,
        plasticity: PlasticityController | None = None,
        dim: int = 64,
        max_mlu_nodes: int = 2048,
        mitosis_min_activations: int = 4,
        mitosis_min_verified_spokes: int = 2,
        survival_window_cycles: int = 12,
    ) -> None:
        self.memory = memory
        self.verifier = verifier or DeepSeekInvariantVerifier(memory)
        self.plasticity = plasticity or PlasticityController(base_lr=0.20)
        self.dim = dim
        self.max_mlu_nodes = max_mlu_nodes
        self.mitosis_min_activations = mitosis_min_activations
        self.mitosis_min_verified_spokes = mitosis_min_verified_spokes
        self.survival_window_cycles = survival_window_cycles

        self.cycle_step: int = 0
        self.nodes: dict[str, MiniLearningUnit] = {}
        self.subcore_children: dict[str, set[str]] = {}
        self.total_mitosis_events: int = 0
        self.total_pruned_nodes: int = 0
        from little.rust.engine import RustMluArena

        self.rust_arena = RustMluArena(
            mitosis_min_activations=self.mitosis_min_activations,
            mitosis_min_verified_spokes=self.mitosis_min_verified_spokes,
        )

        self._ensure_persistence_table()
        self._init_core_topology()
        self._load_persisted_nodes()
        self.sync_from_symbolic_memory()

    # ------------------------------------------------------------------
    # Initialization & Persistence
    # ------------------------------------------------------------------

    def _ensure_persistence_table(self) -> None:
        with self.memory.transaction():
            self.memory.transaction().execute(
                """
                CREATE TABLE IF NOT EXISTS mlu_spiderweb_nodes (
                    concept_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    parent_hub TEXT,
                    payload_json TEXT NOT NULL,
                    updated_step INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _init_core_topology(self) -> None:
        """Create the always-resident Central Core and initial 360° ring of 6 Sub-Core Hubs."""
        if "CORE_ROOT" not in self.nodes:
            self.nodes["CORE_ROOT"] = MiniLearningUnit(
                concept_id="CORE_ROOT", role="core", parent_hub=None, dim=self.dim
            )
        for subcore_id in DEFAULT_SUBCORES:
            if subcore_id not in self.nodes:
                self.nodes[subcore_id] = MiniLearningUnit(
                    concept_id=subcore_id,
                    role="subcore",
                    parent_hub="CORE_ROOT",
                    dim=self.dim,
                )
            self.subcore_children.setdefault(subcore_id, set())

    def _load_persisted_nodes(self) -> None:
        cursor = self.memory.transaction().cursor()
        cursor.execute("SELECT payload_json FROM mlu_spiderweb_nodes")
        for (payload_json,) in cursor.fetchall():
            try:
                data = json.loads(payload_json)
                unit = MiniLearningUnit.from_dict(data)
                self.nodes[unit.concept_id] = unit
                if unit.role == "subcore":
                    self.subcore_children.setdefault(unit.concept_id, set())
                if unit.parent_hub and unit.parent_hub != "CORE_ROOT":
                    self.subcore_children.setdefault(unit.parent_hub, set()).add(unit.concept_id)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    def save_nodes(self, concept_ids: list[str] | None = None) -> None:
        """Persist modified MLU nodes to SQLite."""
        ids_to_save = concept_ids if concept_ids is not None else list(self.nodes.keys())
        rows: list[tuple[str, str, str | None, str, int]] = []
        for cid in ids_to_save:
            unit = self.nodes.get(cid)
            if unit is None:
                continue
            rows.append(
                (
                    unit.concept_id,
                    unit.role,
                    unit.parent_hub,
                    json.dumps(unit.to_dict()),
                    self.cycle_step,
                )
            )
        if rows:
            with self.memory.transaction():
                self.memory.transaction().executemany(
                    """
                    INSERT INTO mlu_spiderweb_nodes (concept_id, role, parent_hub, payload_json, updated_step)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(concept_id) DO UPDATE SET
                        role = excluded.role,
                        parent_hub = excluded.parent_hub,
                        payload_json = excluded.payload_json,
                        updated_step = excluded.updated_step
                    """,
                    rows,
                )

    def sync_from_symbolic_memory(self) -> int:
        """Ensure existing symbolic relations in `MemoryStore` are mirrored as 360° MLU spokes."""
        concepts = {c.id: c.name for c in self.memory.list_concepts()}
        relations = self.memory.get_relations()
        synced = 0
        for rel in relations:
            sub_name = concepts.get(rel.subject_id)
            obj_name = concepts.get(rel.object_id)
            if not sub_name or not obj_name:
                continue
            sub_cid = self._clean_id(sub_name)
            obj_cid = self._clean_id(obj_name)
            if sub_cid.startswith("LATENT_") or obj_cid.startswith("LATENT_"):
                continue
            spoke = map_predicate_to_spoke(rel.predicate)
            sub_unit = self.get_or_create_mlu(sub_cid, preferred_spoke=spoke)
            obj_unit = self.get_or_create_mlu(obj_cid, preferred_spoke=spoke)
            if obj_cid not in sub_unit.spokes[spoke].target_affinities:
                sub_unit.learn_spoke_relation(
                    predicate=rel.predicate,
                    target_concept=obj_cid,
                    target_vec=obj_unit.anchor_vec,
                    learning_rate=0.15,
                    verified=True,
                )
                synced += 1
        return synced

    # ------------------------------------------------------------------
    # 360° Spider-Web Routing & Node Lifecycle (5-Brake Growth + Mitosis)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_id(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_\s]", "", text).strip().upper()
        cleaned = re.sub(r"\s+", "_", cleaned)
        return cleaned

    def _route_to_subcore(self, preferred_spoke: str, concept_vec: list[float]) -> str:
        """Select the best Sub-Core Hub using spoke alignment + cosine similarity."""
        default_hub = f"SUBCORE_{preferred_spoke.upper()}"
        best_hub = default_hub if default_hub in self.nodes else "SUBCORE_TAXONOMY"
        best_score = -1e9
        for hub_id, unit in self.nodes.items():
            if unit.role != "subcore":
                continue
            sim = sum(a * b for a, b in zip(concept_vec, unit.anchor_vec, strict=False))
            bonus = 0.35 if hub_id == default_hub else 0.0
            if sim + bonus > best_score:
                best_score = sim + bonus
                best_hub = hub_id
        return best_hub

    def get_or_create_mlu(self, concept_name: str, preferred_spoke: str = "taxonomy") -> MiniLearningUnit:
        """Retrieve an existing MLU or spawn a new Leaf MLU attached to its nearest Sub-Core Hub."""
        cid = self._clean_id(concept_name)
        if not cid:
            return self.nodes["CORE_ROOT"]
        existing = self.nodes.get(cid)
        if existing is not None:
            existing.last_active_cycle = self.cycle_step
            return existing

        seed_vec = deterministic_seed_vector(cid, self.dim)
        hub_id = self._route_to_subcore(preferred_spoke, seed_vec)
        unit = MiniLearningUnit(
            concept_id=cid,
            role="leaf",
            parent_hub=hub_id,
            dim=self.dim,
        )
        unit.created_cycle = self.cycle_step
        unit.last_active_cycle = self.cycle_step
        self.nodes[cid] = unit
        self.subcore_children.setdefault(hub_id, set()).add(cid)
        return unit

    def evaluate_5_brakes_for_mitosis(self, unit: MiniLearningUnit) -> dict[str, bool]:
        """Evaluate the 5-Brake Structural Growth Gate (`room & used & earning & fits & honest`)."""
        populated_spokes = sum(
            1 for head in unit.spokes.values() if len(head.target_affinities) > 0
        )
        room = len(self.nodes) < self.max_mlu_nodes
        used = unit.activation_count >= self.mitosis_min_activations
        earning = (
            populated_spokes >= self.mitosis_min_verified_spokes
            and unit.verified_hits >= float(self.mitosis_min_verified_spokes)
        )
        fits = unit.role == "leaf" and not unit.concept_id.startswith("SUBCORE_")
        honest = unit.verified_hits > unit.refuted_hits
        return {
            "room": room,
            "used": used,
            "earning": earning,
            "fits": fits,
            "honest": honest,
            "all_passed": bool(room and used and earning and fits and honest),
        }

    def run_structural_plasticity_cycle(self) -> tuple[list[str], list[str]]:
        """Promote high-utility Leaf MLUs into 360° Sub-Core Hubs (Mitosis) and prune dead leaves."""
        promoted: list[str] = []
        pruned: list[str] = []

        for cid, unit in list(self.nodes.items()):
            if unit.role in ("core", "subcore"):
                continue

            # 1. Check 5-Brake Fractal Mitosis (Leaf MLU -> 360° Sub-Core Hub)
            brakes = self.evaluate_5_brakes_for_mitosis(unit)
            if brakes["all_passed"]:
                unit.role = "subcore"
                self.subcore_children.setdefault(cid, set())
                # Attach its connected target leaf MLUs into its 360° sub-core neighborhood
                for head in unit.spokes.values():
                    for target_cid in head.target_affinities:
                        target_unit = self.nodes.get(target_cid)
                        if target_unit is not None and target_unit.role == "leaf":
                            self.subcore_children[cid].add(target_cid)
                promoted.append(cid)
                self.total_mitosis_events += 1
                continue

            # 2. Check Survival-Window Pruning ("Dead means unaddressed")
            age = self.cycle_step - unit.last_active_cycle
            if age > self.survival_window_cycles and unit.verified_hits <= 0.0:
                if unit.parent_hub and unit.parent_hub in self.subcore_children:
                    self.subcore_children[unit.parent_hub].discard(cid)
                del self.nodes[cid]
                with self.memory.transaction():
                    self.memory.transaction().execute(
                        "DELETE FROM mlu_spiderweb_nodes WHERE concept_id = ?", (cid,)
                    )
                pruned.append(cid)
                self.total_pruned_nodes += 1

        return promoted, pruned

    # ------------------------------------------------------------------
    # Online Stream Training (JEPA + LFM + DeepSeek-R1 Verification Gate)
    # ------------------------------------------------------------------

    def _extract_candidate_triples(self, sentence: str) -> list[tuple[str, str, str]]:
        """Extract structured relational triples AND open-domain concept associations from a sentence."""
        clean = sentence.strip().rstrip(".!?").strip()
        if not clean:
            return []

        triples: list[tuple[str, str, str]] = []
        for pattern, pred in STREAM_RELATION_PATTERNS:
            m = pattern.match(clean)
            if m:
                subj = re.sub(r"^(?:the|an|a)\b\s+", "", m.group(1).strip().lower()).strip()
                obj = re.sub(r"^(?:the|an|a)\b\s+", "", m.group(2).strip().lower()).strip()
                if subj and obj and subj not in STOPWORDS and obj not in STOPWORDS:
                    triples.append((subj, pred, obj))
                break

        # Even if a sentence is free-form prose, extract salient content concepts for episodic co-occurrence
        if not triples:
            words = [
                w.lower()
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", clean)
                if w.lower() not in STOPWORDS
            ]
            if len(words) >= 2:
                anchor = words[0]
                for other in words[1:4]:
                    if other != anchor:
                        triples.append((anchor, "co_occurs_with", other))
        return triples

    def train_on_text(self, text: str, persist: bool = True) -> MLMTrainingStepReport:
        """Train the 360° Spider-Web MLM step-by-step on raw text or multi-sentence documents."""
        raw_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n+", text)
            if s and s.strip()
        ]
        if not raw_sentences and text.strip():
            raw_sentences = [text.strip()]

        sentences_processed = 0
        relations_verified = 0
        relations_refuted = 0
        token_transitions_learned = 0
        surprises: list[float] = []
        touched_nodes: set[str] = {"CORE_ROOT"}

        core_unit = self.nodes["CORE_ROOT"]
        if self.rust_arena.is_native and text.strip():
            self.rust_arena.train_stream(
                text, learning_rate=0.15 * self.plasticity.plasticity_multiplier
            )

        for sent in raw_sentences:
            self.cycle_step += 1
            sentences_processed += 1

            # 1. Tokenize and train Central Core Multi-Token Prediction (MTP)
            tokens = re.findall(r"[a-zA-Z0-9_]+", sent.lower())
            if len(tokens) >= 2:
                tok_surprise = core_unit.learn_token_sequence(
                    tokens, learning_rate=0.15 * self.plasticity.plasticity_multiplier
                )
                surprises.append(tok_surprise)
                token_transitions_learned += len(tokens) - 1

            # 2. Extract candidate relational & co-occurrence triples
            triples = self._extract_candidate_triples(sent)
            context_vecs: list[list[float]] = []

            for subj, pred, obj in triples:
                spoke = map_predicate_to_spoke(pred)
                subj_unit = self.get_or_create_mlu(subj, preferred_spoke=spoke)
                obj_unit = self.get_or_create_mlu(obj, preferred_spoke=spoke)
                touched_nodes.add(subj_unit.concept_id)
                touched_nodes.add(obj_unit.concept_id)
                if subj_unit.parent_hub:
                    touched_nodes.add(subj_unit.parent_hub)

                context_vecs.append(subj_unit.anchor_vec)
                context_vecs.append(obj_unit.anchor_vec)

                # Step LFM-2.5 liquid state on both units
                subj_unit.forward_liquid_step([obj_unit.anchor_vec])
                obj_unit.forward_liquid_step([subj_unit.anchor_vec])
                if subj_unit.parent_hub and subj_unit.parent_hub in self.nodes:
                    self.nodes[subj_unit.parent_hub].forward_liquid_step(
                        [subj_unit.hidden_state, obj_unit.hidden_state]
                    )

                # 3. DeepSeek-R1 style Formal Verification Gate Check
                gate_res = self.verifier.verify_relation(subj, pred, obj)
                lr = self.plasticity.concept_learning_rate(
                    is_trunk=subj_unit.is_trunk,
                    evidence_count=int(subj_unit.verified_hits + subj_unit.refuted_hits),
                )

                if gate_res.passed:
                    relations_verified += 1
                    if self.rust_arena.is_native:
                        self.rust_arena.learn_spoke(
                            subj_unit.concept_id,
                            pred,
                            obj_unit.concept_id,
                            learning_rate=lr,
                            verified=True,
                        )
                    surprise = subj_unit.learn_spoke_relation(
                        predicate=pred,
                        target_concept=obj_unit.concept_id,
                        target_vec=obj_unit.anchor_vec,
                        learning_rate=lr,
                        verified=True,
                    )
                    # Also train focal concept's surface token memory
                    subj_unit.learn_token_sequence(tokens, learning_rate=lr)
                    surprises.append(surprise)
                    self.plasticity.observe_error(surprise)

                    # Commit verified symbolic fact into MemoryStore so symbolic & neural stay 100% unified
                    if pred != "co_occurs_with":
                        c_sub = self.memory.create_concept(subj.upper().replace(" ", "_"))
                        c_obj = self.memory.create_concept(obj.upper().replace(" ", "_"))
                        self.memory.add_relation(c_sub.id, pred, c_obj.id)
                else:
                    relations_refuted += 1
                    if self.rust_arena.is_native:
                        self.rust_arena.learn_spoke(
                            subj_unit.concept_id,
                            pred,
                            obj_unit.concept_id,
                            learning_rate=lr,
                            verified=False,
                        )
                    surprise = subj_unit.learn_spoke_relation(
                        predicate=pred,
                        target_concept=obj_unit.concept_id,
                        target_vec=obj_unit.anchor_vec,
                        learning_rate=lr,
                        verified=False,
                    )
                    surprises.append(surprise)
                    self.plasticity.observe_error(surprise)

            if context_vecs:
                core_unit.forward_liquid_step(context_vecs)

        promoted, pruned = self.run_structural_plasticity_cycle()
        for cid in promoted:
            touched_nodes.add(cid)

        if persist:
            self.save_nodes(list(touched_nodes))

        avg_surprise = round(sum(surprises) / max(1, len(surprises)), 4)
        subcore_count = sum(1 for u in self.nodes.values() if u.role == "subcore")
        return MLMTrainingStepReport(
            sentences_processed=sentences_processed,
            relations_verified=relations_verified,
            relations_refuted=relations_refuted,
            token_transitions_learned=token_transitions_learned,
            average_surprise=avg_surprise,
            plasticity_multiplier=round(self.plasticity.plasticity_multiplier, 4),
            active_subcores=subcore_count,
            total_mlu_nodes=len(self.nodes),
            mitosis_promotions=promoted,
            pruned_nodes=pruned,
        )

    # ------------------------------------------------------------------
    # 360° Spider-Web Inference & Grounded Neural Generation
    # ------------------------------------------------------------------

    def _collect_360_predictions(
        self, focal_unit: MiniLearningUnit, max_hops: int = 3
    ) -> tuple[list[MLUSpokePrediction], list[str]]:
        """Traverse the 360° Spider-Web around `focal_unit` (including transitive taxonomic inheritance)."""
        candidate_vecs = {cid: u.anchor_vec for cid, u in self.nodes.items()}
        all_preds: list[MLUSpokePrediction] = []
        activated: list[str] = ["CORE_ROOT"]
        if focal_unit.parent_hub:
            activated.append(focal_unit.parent_hub)
        activated.append(focal_unit.concept_id)

        # Direct 360° 6-spoke predictions on the focal MLU
        for spoke_name in SPOKE_NAMES:
            all_preds.extend(focal_unit.predict_spoke_targets(spoke_name, candidate_vecs, top_k=4))

        # Multi-hop transitive taxonomic & property inheritance across the Spider-Web
        frontier = [
            p.target_concept
            for p in all_preds
            if p.spoke == "taxonomy" and p.predicate == "is_a"
        ]
        visited = {focal_unit.concept_id}
        hop = 1
        decay = 0.88
        while frontier and hop < max_hops:
            next_frontier: list[str] = []
            for parent_cid in frontier:
                if parent_cid in visited:
                    continue
                visited.add(parent_cid)
                parent_unit = self.nodes.get(parent_cid)
                if parent_unit is None:
                    continue
                if parent_cid not in activated:
                    activated.append(parent_cid)
                for spoke_name in ("taxonomy", "mereology", "attributes", "procedural", "episodic"):
                    parent_preds = parent_unit.predict_spoke_targets(
                        spoke_name, candidate_vecs, top_k=3
                    )
                    for pp in parent_preds:
                        if pp.target_concept == focal_unit.concept_id or pp.predicate == "disjoint_with":
                            continue
                        inherited = MLUSpokePrediction(
                            source_concept=focal_unit.concept_id,
                            spoke=pp.spoke,
                            predicate=pp.predicate,
                            target_concept=pp.target_concept,
                            neural_score=round(pp.neural_score * (decay**hop), 4),
                            nars_confidence=round(pp.nars_confidence * (decay**hop), 4),
                            epistemic_entropy=pp.epistemic_entropy,
                            combined_confidence=round(
                                pp.combined_confidence * (decay**hop), 4
                            ),
                            verified=pp.verified,
                        )
                        all_preds.append(inherited)
                        if pp.spoke == "taxonomy" and pp.predicate == "is_a":
                            next_frontier.append(pp.target_concept)
            frontier = next_frontier
            hop += 1

        # Deduplicate keeping highest confidence per (spoke, predicate, target)
        dedup: dict[tuple[str, str, str], MLUSpokePrediction] = {}
        for pred in all_preds:
            key = (pred.spoke, pred.predicate, pred.target_concept)
            if key not in dedup or pred.combined_confidence > dedup[key].combined_confidence:
                dedup[key] = pred

        sorted_preds = sorted(dedup.values(), key=lambda r: r.combined_confidence, reverse=True)
        return sorted_preds, activated

    def query_360(self, query: str) -> MLMQueryResult:
        """Answer a natural-language question or concept inquiry using the 360° Spider-Web MLM."""
        clean_q = query.strip()
        tokens = [
            w.lower()
            for w in re.findall(r"[a-zA-Z0-9_]+", clean_q)
            if w.lower() not in STOPWORDS
        ]

        # Identify focal concept (and optional target concept for yes/no relational queries)
        focal_cid: str | None = None
        target_cid: str | None = None

        # Check multi-word and single-word matches in self.nodes
        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                bigram_cid = f"{tokens[i].upper()}_{tokens[i + 1].upper()}"
                if bigram_cid in self.nodes and bigram_cid not in DEFAULT_SUBCORES:
                    if focal_cid is None:
                        focal_cid = bigram_cid
                    elif bigram_cid != focal_cid:
                        target_cid = bigram_cid
            single_cid = tokens[i].upper()
            if single_cid in self.nodes and single_cid not in DEFAULT_SUBCORES:
                if focal_cid is None:
                    focal_cid = single_cid
                elif single_cid != focal_cid and target_cid is None:
                    target_cid = single_cid

        if focal_cid is None and tokens:
            focal_cid = tokens[0].upper()

        if not focal_cid or focal_cid not in self.nodes:
            return MLMQueryResult(
                query=clean_q,
                focal_concept=focal_cid or "NONE",
                status="UNKNOWN",
                answer=f"UNKNOWN: No 360° MLU node has grounded evidence for '{focal_cid or clean_q}'.",
                confidence=0.0,
                epistemic_entropy=1.0,
                active_subcore="CORE_ROOT",
                activated_mlus=["CORE_ROOT"],
                spoke_predictions=[],
                inspectable_trace="[MLU-360] Focal concept not present in Spider-Web -> Abstaining (UNKNOWN).",
            )

        focal_unit = self.nodes[focal_cid]
        focal_unit.forward_liquid_step([self.nodes["CORE_ROOT"].hidden_state])
        preds, activated = self._collect_360_predictions(focal_unit)
        active_subcore = focal_unit.parent_hub or "CORE_ROOT"

        if not preds:
            return MLMQueryResult(
                query=clean_q,
                focal_concept=focal_cid,
                status="UNKNOWN",
                answer=f"UNKNOWN: MLU '{focal_cid}' exists in {active_subcore}, but has insufficient verified 360° spoke evidence.",
                confidence=0.0,
                epistemic_entropy=1.0,
                active_subcore=active_subcore,
                activated_mlus=activated,
                spoke_predictions=[],
                inspectable_trace=f"[MLU-360] Activated {activated} -> 0 verified spoke links -> UNKNOWN.",
            )

        # Case A: Targeted relational verification question (e.g., "Is an orca a mammal?" or "Can a penguin fly?")
        if target_cid is not None:
            matching = [p for p in preds if p.target_concept == target_cid]
            if matching:
                best = matching[0]
                verb = best.predicate.replace("_", " ")
                ans = (
                    f"Yes — {focal_cid.lower().replace('_', ' ')} {verb} "
                    f"{target_cid.lower().replace('_', ' ')} "
                    f"(spoke={best.spoke}, confidence={best.combined_confidence:.2f}, "
                    f"subcore={active_subcore})."
                )
                trace = (
                    f"[MLU-360] Core -> {active_subcore} -> {' -> '.join(activated[2:])} | "
                    f"spoke={best.spoke}({best.predicate}) -> {target_cid} "
                    f"[neural={best.neural_score:.2f}, nars={best.nars_confidence:.2f}, "
                    f"H={best.epistemic_entropy:.2f}]"
                )
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED" if best.verified else "NEURAL_GROUNDED",
                    answer=ans,
                    confidence=best.combined_confidence,
                    epistemic_entropy=best.epistemic_entropy,
                    active_subcore=active_subcore,
                    activated_mlus=activated,
                    spoke_predictions=preds[:6],
                    inspectable_trace=trace,
                )

            # Check if DeepSeekInvariantVerifier refutes the link via I_MUTEX / disjointness
            gate_check = self.verifier.verify_relation(focal_cid, "is_a", target_cid)
            if not gate_check.passed:
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="REFUTED",
                    answer=(
                        f"No — {focal_cid.lower().replace('_', ' ')} cannot be "
                        f"{target_cid.lower().replace('_', ' ')} "
                        f"(Refuted by {gate_check.violated_gate}: {gate_check.error_message})."
                    ),
                    confidence=0.95,
                    epistemic_entropy=0.0,
                    active_subcore=active_subcore,
                    activated_mlus=activated,
                    spoke_predictions=preds[:6],
                    inspectable_trace=f"[MLU-360] Refuted by {gate_check.violated_gate}: {gate_check.error_message}",
                )

        # Case B: Open 360° synthesis across all 6 radial spokes + learned neural sequence continuation
        spoke_clauses: list[str] = []
        for p in preds[:5]:
            rel_text = p.predicate.replace("_", " ")
            tgt_text = p.target_concept.lower().replace("_", " ")
            spoke_clauses.append(f"{rel_text} {tgt_text} [{p.spoke}:{p.combined_confidence:.2f}]")

        rust_cont = (
            self.rust_arena.generate(focal_cid.lower().replace("_", " "), max_tokens=8)
            if self.rust_arena.is_native
            else ""
        )
        if rust_cont:
            continuation_text = rust_cont
        else:
            continuation = focal_unit.generate_token_continuation(
                [focal_cid.lower().replace("_", " ")], max_tokens=8
            )
            continuation_text = " ".join(continuation)
        fluency_suffix = (
            f" | Neural stream: \"{focal_cid.lower().replace('_', ' ')} {continuation_text}\""
            if continuation_text
            else ""
        )

        top_conf = preds[0].combined_confidence
        avg_entropy = round(sum(p.epistemic_entropy for p in preds[:5]) / min(5, len(preds)), 4)
        summary = (
            f"{focal_cid.replace('_', ' ').title()} ({active_subcore}, role={focal_unit.role}): "
            + "; ".join(spoke_clauses)
            + fluency_suffix
        )
        trace = (
            f"[MLU-360] Activated {len(activated)} units ({' -> '.join(activated)}) | "
            f"top_conf={top_conf:.2f} | entropy={avg_entropy:.2f}"
        )
        return MLMQueryResult(
            query=clean_q,
            focal_concept=focal_cid,
            status="VERIFIED" if preds[0].verified else "NEURAL_GROUNDED",
            answer=summary,
            confidence=top_conf,
            epistemic_entropy=avg_entropy,
            active_subcore=active_subcore,
            activated_mlus=activated,
            spoke_predictions=preds[:6],
            inspectable_trace=trace,
        )

    def get_telemetry(self) -> dict[str, Any]:
        """Return live telemetry of the 360° Spider-Web MLM topology."""
        subcores = [cid for cid, u in self.nodes.items() if u.role == "subcore"]
        leaves = [cid for cid, u in self.nodes.items() if u.role == "leaf"]
        total_spoke_edges = sum(
            len(head.target_affinities)
            for u in self.nodes.values()
            for head in u.spokes.values()
        )
        return {
            "total_mlu_nodes": len(self.nodes),
            "central_core": "CORE_ROOT",
            "subcore_hubs": len(subcores),
            "subcore_names": sorted(subcores),
            "leaf_mlus": len(leaves),
            "total_360_spoke_edges": total_spoke_edges,
            "mitosis_promotions": self.total_mitosis_events,
            "pruned_nodes": self.total_pruned_nodes,
            "cycle_step": self.cycle_step,
            "rust_arena_native": self.rust_arena.is_native,
            "rust_arena_nodes": self.rust_arena.node_count,
        }
