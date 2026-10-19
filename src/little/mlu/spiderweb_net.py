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
import math
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
from little.neural.semantic_frame_policy import SemanticFramePolicy

DEFAULT_SUBCORES: tuple[str, ...] = (
    "SUBCORE_TAXONOMY",
    "SUBCORE_MEREOLOGY",
    "SUBCORE_ATTRIBUTES",
    "SUBCORE_DYNAMICS",
    "SUBCORE_PROCEDURAL",
    "SUBCORE_EPISODIC",
)


def _tokenize_words(text: str) -> list[str]:
    """Pure character-state alphanumeric tokenizer without regular expressions."""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            buf.append(ch.lower())
        elif buf:
            tokens.append("".join(buf))
            buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def _segment_sentences(text: str) -> list[str]:
    """Segment raw stream into utterances via character boundary state transitions."""
    sentences: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in ".!?\n":
            candidate = "".join(buf).strip()
            if candidate:
                sentences.append(candidate)
            buf.clear()
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _shannon_char_entropy(token: str) -> float:
    """Compute Shannon character-level entropy H(X) = -sum p(c) log2 p(c) for a token."""
    n = len(token)
    if n <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for ch in token:
        counts[ch] = counts.get(ch, 0) + 1
    inv_n = 1.0 / float(n)
    entropy = 0.0
    for c in counts.values():
        p = float(c) * inv_n
        entropy -= p * math.log2(p)
    return entropy


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
    fluent_english: str = ""


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
        self.frame_policy = SemanticFramePolicy.default()
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
        self.conversation_liquid_state: list[float] = [0.0] * dim
        self.recent_focal_stack: list[tuple[str, float]] = []
        self._operator_codebook: list[tuple[tuple[str, ...], str, list[float]]] = []
        self._operator_token_mass: dict[str, float] = {}
        self._build_operator_codebook()

        from little.rust.engine import RustMluArena

        self.rust_arena = RustMluArena(
            mitosis_min_activations=self.mitosis_min_activations,
            mitosis_min_verified_spokes=self.mitosis_min_verified_spokes,
        )

        self._ensure_persistence_table()
        self._init_core_topology()
        self.conversation_liquid_state = list(self.nodes["CORE_ROOT"].hidden_state)
        self._load_persisted_nodes()
        self.sync_from_symbolic_memory()

    def close(self) -> None:
        """Release native Rust arena and graph handles owned by this MLM."""
        if hasattr(self, "rust_arena") and self.rust_arena is not None:
            self.rust_arena.close()
        if hasattr(self.verifier, "_rust_graph") and self.verifier._rust_graph is not None:
            self.verifier._rust_graph.close()

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
            self.memory.transaction().execute(
                """
                CREATE TABLE IF NOT EXISTS mlu_arena_snapshots (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    snapshot_json TEXT NOT NULL,
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

        if self.rust_arena.is_native:
            cursor.execute("SELECT snapshot_json, updated_step FROM mlu_arena_snapshots WHERE id = 1")
            row = cursor.fetchone()
            if row and row[0]:
                self.rust_arena.restore_json(str(row[0]))
                self.cycle_step = max(self.cycle_step, int(row[1] or 0))

    def save_nodes(self, concept_ids: list[str] | None = None) -> None:
        """Persist modified MLU nodes and the native Rust 360° Arena snapshot to SQLite."""
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
        with self.memory.transaction():
            if rows:
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
            if self.rust_arena.is_native:
                snap_json = self.rust_arena.dump_json()
                self.memory.transaction().execute(
                    """
                    INSERT INTO mlu_arena_snapshots (id, snapshot_json, updated_step)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        snapshot_json = excluded.snapshot_json,
                        updated_step = excluded.updated_step
                    """,
                    (snap_json, self.cycle_step),
                )

    def sync_from_symbolic_memory(self) -> int:
        """Ensure existing symbolic relations in `MemoryStore` are mirrored as 360° MLU spokes."""
        concepts = {c.id: c.name for c in self.memory.list_concepts()}
        relations = self.memory.get_relations()
        synced = 0
        for rel in relations:
            if rel.weight_positive <= rel.weight_negative:
                continue
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
                if self.rust_arena.is_native:
                    self.rust_arena.learn_spoke(
                        sub_cid, rel.predicate, obj_cid, learning_rate=0.15, verified=True
                    )
                synced += 1
        return synced

    # ------------------------------------------------------------------
    # 360° Spider-Web Routing & Node Lifecycle (5-Brake Growth + Mitosis)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_id(text: str) -> str:
        return "_".join(t.upper() for t in _tokenize_words(text))

    def _build_operator_codebook(self) -> None:
        """Build 64-D relational operator prototypes from SchemaRegistry and SemanticFramePolicy."""
        neg_tuples = [
            tuple(_tokenize_words(phrase))
            for phrase in self.frame_policy.negation_tokens
            if _tokenize_words(phrase)
        ]
        for det in self.frame_policy.stop_determiners:
            for tok in _tokenize_words(det):
                self._operator_token_mass[tok] = self._operator_token_mass.get(tok, 0.0) + 3.0

        disjoint_pred = (
            "disjoint_with"
            if "disjoint_with" in self.verifier.disjoint_predicates
            else next(iter(self.verifier.disjoint_predicates), "disjoint_with")
        )
        seen_tuples: set[tuple[str, ...]] = set()

        def _contains_negation(seq: tuple[str, ...]) -> bool:
            for nt in neg_tuples:
                k = len(nt)
                if k > 0 and any(seq[idx : idx + k] == nt for idx in range(len(seq) - k + 1)):
                    return True
            return False

        for base_rel, phrases in self.frame_policy.relational_indicators.items():
            for phrase in phrases:
                ptoks = tuple(_tokenize_words(phrase))
                if not ptoks:
                    continue
                for tok in ptoks:
                    self._operator_token_mass[tok] = self._operator_token_mass.get(tok, 0.0) + 2.0

                has_neg = _contains_negation(ptoks)
                if has_neg:
                    canonical = disjoint_pred
                elif base_rel == "has_property" and ptoks[0] in ("has", "have", "had"):
                    canonical = "has_part"
                elif base_rel == "capable_of" and ptoks == ("can",):
                    canonical = "can"
                else:
                    canonical = base_rel

                if self.verifier.registry.relation(canonical) is None:
                    canonical = self.verifier.taxonomy_predicate or "is_a"

                for candidate_toks in (
                    ptoks,
                    ptoks[:-1]
                    if len(ptoks) > 1 and ptoks[-1] in self.frame_policy.stop_determiners
                    else ptoks,
                ):
                    if candidate_toks and candidate_toks not in seen_tuples:
                        seen_tuples.add(candidate_toks)
                        vec = deterministic_seed_vector("_".join(candidate_toks), self.dim)
                        self._operator_codebook.append((candidate_toks, canonical, vec))

        for reg_pred in self.verifier.registry.predicates():
            ptoks = tuple(_tokenize_words(reg_pred))
            if ptoks and ptoks not in seen_tuples:
                seen_tuples.add(ptoks)
                for tok in ptoks:
                    self._operator_token_mass[tok] = self._operator_token_mass.get(tok, 0.0) + 1.5
                vec = deterministic_seed_vector(reg_pred, self.dim)
                self._operator_codebook.append((ptoks, reg_pred, vec))

    def _token_salience(self, token: str) -> float:
        """Compute Shannon character-entropy & epistemic anchor salience for a token."""
        if not token:
            return 0.0
        char_h = _shannon_char_entropy(token)
        morph_info = char_h * math.log(1.0 + float(len(token)))
        if "_" in token:
            morph_info += 2.0
        cid = token.upper()
        anchor_bonus = (
            2.0
            if (cid in self.nodes and cid != "CORE_ROOT" and cid not in DEFAULT_SUBCORES)
            else 0.0
        )
        op_mass = self._operator_token_mass.get(token, 0.0)
        return (morph_info + anchor_bonus) / (1.0 + op_mass)

    def _extract_salient_concept(self, span_tokens: list[str]) -> tuple[str, float]:
        """Select the highest-information concept token(s) from a token span."""
        if not span_tokens:
            return "", 0.0
        scored = [(t, self._token_salience(t)) for t in span_tokens]
        max_sal = max(s for _, s in scored)
        if max_sal <= 0.15:
            return "", 0.0
        kept = [
            t
            for t, s in scored
            if s >= 0.60 * max_sal and self._operator_token_mass.get(t, 0.0) < 2.0
        ]
        if not kept:
            best_tok = max(scored, key=lambda kv: kv[1])[0]
            kept = [best_tok]
        return "_".join(kept), max_sal

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
        """Extract relational triples via joint Shannon salience & 64-D operator subspace optimization."""
        tokens = _tokenize_words(sentence)
        if len(tokens) < 2:
            return []

        best_score = -1e9
        best_triple: tuple[str, str, str] | None = None
        n = len(tokens)

        for i in range(1, n):
            for j in range(i + 1, min(n, i + 6)):
                bridge = tuple(tokens[i:j])
                # Strip trailing determiners inside the bridge if needed
                core_bridge = tuple(
                    t for t in bridge if t not in self.frame_policy.stop_determiners
                )
                matched_pred: str | None = None
                op_resonance = -1.0

                for proto_toks, canonical_pred, _vec in self._operator_codebook:
                    if bridge == proto_toks or (core_bridge and core_bridge == proto_toks):
                        res = 1.0 + 0.25 * float(len(proto_toks))
                        if res > op_resonance:
                            op_resonance = res
                            matched_pred = canonical_pred

                if matched_pred is None:
                    continue

                subj_concept, subj_sal = self._extract_salient_concept(tokens[:i])
                obj_concept, obj_sal = self._extract_salient_concept(tokens[j:])
                if not subj_concept or not obj_concept or subj_concept == obj_concept:
                    continue

                total_score = subj_sal + obj_sal + 4.0 * op_resonance
                if total_score > best_score:
                    best_score = total_score
                    best_triple = (subj_concept, matched_pred, obj_concept)

        if best_triple is not None:
            return [best_triple]

        # Open-vocabulary SVO projection: rank tokens by Shannon salience and project interior bridge
        salient = [
            (idx, tok, self._token_salience(tok))
            for idx, tok in enumerate(tokens)
            if self._token_salience(tok) > 0.5
        ]
        if len(salient) >= 3:
            subj_tok = salient[0][1]
            bridge_tok = salient[1][1]
            obj_tok = salient[-1][1]
            if subj_tok != obj_tok:
                bridge_vec = deterministic_seed_vector(bridge_tok, self.dim)
                best_pred = "causes"
                best_sim = -1e9
                for _ptoks, canonical_pred, proto_vec in self._operator_codebook:
                    sim = sum(a * b for a, b in zip(bridge_vec, proto_vec, strict=False))
                    if sim > best_sim:
                        best_sim = sim
                        best_pred = canonical_pred
                return [(subj_tok, best_pred, obj_tok)]

        if len(salient) >= 2:
            anchor = salient[0][1]
            return [
                (anchor, "related_to", other_tok)
                for _idx, other_tok, _sal in salient[1:4]
                if other_tok != anchor
            ]
        return []

    def train_on_text(self, text: str, persist: bool = True) -> MLMTrainingStepReport:
        """Train the 360° Spider-Web MLM step-by-step on raw text or multi-sentence documents."""
        raw_sentences = _segment_sentences(text)

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
            tokens = _tokenize_words(sent)
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
                        self._update_conversation_context(subj_unit.concept_id)
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

    def _update_conversation_context(self, focal_cid: str) -> None:
        """Update the continuous multi-turn liquid state and active focal stack."""
        unit = self.nodes.get(focal_cid)
        if unit is None or focal_cid == "CORE_ROOT" or focal_cid in DEFAULT_SUBCORES:
            return
        raw = [
            0.65 * c + 0.35 * h
            for c, h in zip(self.conversation_liquid_state, unit.hidden_state, strict=False)
        ]
        norm = math.sqrt(sum(v * v for v in raw))
        if norm > 1e-12:
            self.conversation_liquid_state = [v / norm for v in raw]

        decayed: list[tuple[str, float]] = [(focal_cid, 1.0)]
        for cid, energy in self.recent_focal_stack:
            if cid != focal_cid and cid in self.nodes and energy * 0.82 >= 0.15:
                decayed.append((cid, round(energy * 0.82, 4)))
        self.recent_focal_stack = decayed[:6]

    @staticmethod
    def _synthesize_fluent_english(
        focal_cid: str,
        preds: list[MLUSpokePrediction],
        continuation_text: str,
    ) -> str:
        """Synthesize natural, articulate English from verified 360° spoke edges and liquid stream."""
        subj = focal_cid.lower().replace("_", " ")
        clauses: list[str] = []
        seen_clauses: set[str] = set()
        for p in preds[:4]:
            verb = p.predicate.replace("_", " ")
            obj = p.target_concept.lower().replace("_", " ")
            clause = f"{verb} {obj}"
            if clause not in seen_clauses:
                seen_clauses.add(clause)
                clauses.append(clause)
        if not clauses:
            return f"No verified 360° spoke facts are currently recorded for {subj}."
        if len(clauses) == 1:
            base = f"{subj.capitalize()} {clauses[0]}."
        else:
            base = f"{subj.capitalize()} {', '.join(clauses[:-1])}, and {clauses[-1]}."
        if continuation_text:
            return f'{base} (Liquid stream: "{subj} {continuation_text}".)'
        return base

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

        # Phase 1 & 7: Fuse native Rust 360° Multi-Hop Beam Traversal predictions
        self._last_rust_beam: dict[str, Any] = {}
        if self.rust_arena.is_native:
            beam_dto = self.rust_arena.query_360_beam(
                focal_unit.concept_id, max_hops=max_hops, top_k=12
            )
            self._last_rust_beam = beam_dto
            for cell_name in beam_dto.get("activated_cells", []):
                if cell_name in self.nodes and cell_name not in activated:
                    activated.append(str(cell_name))
            for item in beam_dto.get("predictions", []):
                pred_name = str(item["predicate"])
                tgt_name = str(item["target_concept"])
                if (
                    self.verifier.registry.relation(pred_name) is None
                    or tgt_name not in self.nodes
                ):
                    continue
                all_preds.append(
                    MLUSpokePrediction(
                        source_concept=str(item["source_concept"]),
                        spoke=str(item["spoke"]),
                        predicate=pred_name,
                        target_concept=tgt_name,
                        neural_score=round(float(item["neural_score"]), 4),
                        nars_confidence=round(float(item["nars_confidence"]), 4),
                        epistemic_entropy=round(float(item["epistemic_entropy"]), 4),
                        combined_confidence=round(float(item["combined_confidence"]), 4),
                        verified=bool(item["verified"]),
                    )
                )

        # Multi-hop transitive inheritance across verified acyclic links in the Spider-Web
        frontier = [
            p.target_concept
            for p in all_preds
            if p.predicate in self.verifier.acyclic_predicates
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
                for spoke_name in SPOKE_NAMES:
                    parent_preds = parent_unit.predict_spoke_targets(
                        spoke_name, candidate_vecs, top_k=3
                    )
                    for pp in parent_preds:
                        if (
                            pp.target_concept == focal_unit.concept_id
                            or pp.predicate in self.verifier.disjoint_predicates
                        ):
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
                        if pp.predicate in self.verifier.acyclic_predicates:
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
        tokens = _tokenize_words(clean_q)

        # Identify focal concept (and optional target concept for yes/no relational queries)
        focal_cid: str | None = None
        focal_pos: int = -1
        target_cid: str | None = None

        # Check multi-word and single-word matches in self.nodes
        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                bigram_cid = f"{tokens[i].upper()}_{tokens[i + 1].upper()}"
                if (
                    bigram_cid in self.nodes
                    and bigram_cid != "CORE_ROOT"
                    and bigram_cid not in DEFAULT_SUBCORES
                ):
                    if focal_cid is None:
                        focal_cid = bigram_cid
                        focal_pos = i
                    elif bigram_cid != focal_cid:
                        target_cid = bigram_cid
            single_cid = tokens[i].upper()
            if (
                single_cid in self.nodes
                and single_cid != "CORE_ROOT"
                and single_cid not in DEFAULT_SUBCORES
            ):
                if focal_cid is None:
                    focal_cid = single_cid
                    focal_pos = i
                elif single_cid != focal_cid and target_cid is None:
                    target_cid = single_cid

        # Multi-Turn Liquid Context Resonance for elliptic / follow-up inquiries
        if self.recent_focal_stack:
            top_ctx_cid = self.recent_focal_stack[0][0]
            op_tuples = {ptoks for ptoks, _, _ in self._operator_codebook}
            has_leading_operator = focal_pos > 0 and any(
                tuple(tokens[idx : idx + k]) in op_tuples
                for idx in range(focal_pos)
                for k in (1, 2, 3)
                if idx + k <= focal_pos
            )
            if (
                focal_cid is not None
                and target_cid is None
                and focal_cid != top_ctx_cid
                and has_leading_operator
                and top_ctx_cid in self.nodes
            ):
                ctx_preds, _ = self._collect_360_predictions(self.nodes[top_ctx_cid])
                if any(p.target_concept == focal_cid for p in ctx_preds) or (
                    not self.verifier.verify_relation(top_ctx_cid, "is_a", focal_cid).passed
                ):
                    target_cid = focal_cid
                    focal_cid = top_ctx_cid
            elif focal_cid is None and tokens:
                max_sal = max(self._token_salience(t) for t in tokens)
                if max_sal < 2.8 and top_ctx_cid in self.nodes:
                    focal_cid = top_ctx_cid

        if focal_cid is None and tokens:
            # Select the token with the highest Shannon character-entropy & morphological salience
            best_tok = max(tokens, key=self._token_salience)
            focal_cid = best_tok.upper()

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
                fluent_english=f"I do not yet have verified 360° MLU knowledge about '{focal_cid or clean_q}'.",
            )

        focal_unit = self.nodes[focal_cid]
        focal_unit.forward_liquid_step([self.nodes["CORE_ROOT"].hidden_state])
        self._update_conversation_context(focal_cid)
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
                fluent_english=f"I recognize '{focal_cid.lower().replace('_', ' ')}', but I do not yet have verified 360° spoke facts for it.",
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
                fluent = (
                    f"Yes, {focal_cid.lower().replace('_', ' ')} {verb} "
                    f"{target_cid.lower().replace('_', ' ')}."
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
                    fluent_english=fluent,
                )

            # Check if DeepSeekInvariantVerifier refutes the link via I_MUTEX / disjointness
            gate_check = self.verifier.verify_relation(focal_cid, "is_a", target_cid)
            if not gate_check.passed:
                ans_ref = (
                    f"No — {focal_cid.lower().replace('_', ' ')} cannot be "
                    f"{target_cid.lower().replace('_', ' ')} "
                    f"(Refuted by {gate_check.violated_gate}: {gate_check.error_message})."
                )
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="REFUTED",
                    answer=ans_ref,
                    confidence=0.95,
                    epistemic_entropy=0.0,
                    active_subcore=active_subcore,
                    activated_mlus=activated,
                    spoke_predictions=preds[:6],
                    inspectable_trace=f"[MLU-360] Refuted by {gate_check.violated_gate}: {gate_check.error_message}",
                    fluent_english=ans_ref,
                )

        # Case B: Open 360° synthesis across all 6 radial spokes + learned neural sequence continuation
        spoke_clauses: list[str] = []
        for p in preds[:5]:
            rel_text = p.predicate.replace("_", " ")
            tgt_text = p.target_concept.lower().replace("_", " ")
            spoke_clauses.append(f"{rel_text} {tgt_text} [{p.spoke}:{p.combined_confidence:.2f}]")

        beam_cache = getattr(self, "_last_rust_beam", {}) or {}
        rust_cont = str(beam_cache.get("generated_continuation", "")).strip()
        if not rust_cont and self.rust_arena.is_native:
            rust_cont = self.rust_arena.generate(
                focal_cid.lower().replace("_", " "), max_tokens=8
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
        rust_fluent = str(beam_cache.get("fluent_synthesis", "")).strip()
        fluent_eng = rust_fluent or self._synthesize_fluent_english(
            focal_cid, preds, continuation_text
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
            fluent_english=fluent_eng,
        )

    def consolidate(
        self, min_transition_weight: float = 0.05, replay_lr: float = 0.15
    ) -> dict[str, Any]:
        """Phase 2: Run Sleep-Cycle Consolidation (spoke replay + transition pruning + mitosis)."""
        rust_report = (
            self.rust_arena.consolidate(
                min_transition_weight=min_transition_weight, replay_lr=replay_lr
            )
            if self.rust_arena.is_native
            else {
                "replayed_edges": 0,
                "pruned_transitions": 0,
                "average_replay_surprise": 0.0,
                "mitosis_promotions": 0,
                "total_cells": len(self.nodes),
            }
        )
        promoted, pruned = self.run_structural_plasticity_cycle()
        self.save_nodes(promoted if promoted else None)
        return {
            "rust_consolidation": rust_report,
            "python_mitosis_promotions": promoted,
            "python_pruned_nodes": pruned,
            "total_mlu_nodes": len(self.nodes),
            "rust_arena_nodes": self.rust_arena.node_count,
        }

    def ingest_corpus_path(
        self, target_path: str, epochs: int = 2
    ) -> dict[str, Any]:
        """Phase 3: Stream-train the 360° Spider-Web MLM on a file or directory of text/markdown files."""
        from pathlib import Path

        root = Path(target_path)
        files: list[Path] = []
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            for ext in ("*.md", "*.txt"):
                files.extend(sorted(root.rglob(ext)))

        total_sentences = 0
        total_verified = 0
        total_refuted = 0
        total_tokens = 0
        epoch_surprises: list[float] = []
        all_promotions: list[str] = []

        for _ep in range(max(1, epochs)):
            ep_surprises: list[float] = []
            for fpath in files:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not content.strip():
                    continue
                rep = self.train_on_text(content, persist=False)
                total_sentences += rep.sentences_processed
                total_verified += rep.relations_verified
                total_refuted += rep.relations_refuted
                total_tokens += rep.token_transitions_learned
                ep_surprises.append(rep.average_surprise)
                for p in rep.mitosis_promotions:
                    if p not in all_promotions:
                        all_promotions.append(p)
            if ep_surprises:
                epoch_surprises.append(round(sum(ep_surprises) / len(ep_surprises), 4))

        cons = self.consolidate()
        return {
            "files_processed": len(files),
            "epochs": max(1, epochs),
            "sentences_processed": total_sentences,
            "relations_verified": total_verified,
            "relations_refuted": total_refuted,
            "token_transitions_learned": total_tokens,
            "epoch_surprises": epoch_surprises,
            "mitosis_promotions": all_promotions,
            "consolidation": cons,
            "telemetry": self.get_telemetry(),
        }

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
            "rust_arena_mitosis": self.rust_arena.mitosis_count,
        }
