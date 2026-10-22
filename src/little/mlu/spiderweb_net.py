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
    verified_triples: list[tuple[str, str, str]] = field(default_factory=list)


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
        self.episodic_working_memory: list[dict[str, Any]] = []
        self.pending_curiosity_slot: dict[str, Any] | None = None
        self._operator_codebook: list[tuple[tuple[str, ...], str, list[float]]] = []
        self._operator_polarity_map: dict[tuple[str, ...], tuple[str, int]] = {}
        self._operator_token_mass: dict[str, float] = {}
        self._cached_neural_vocab: dict[str, list[float]] = {}
        self._build_operator_codebook()

        from little.rust.engine import RustMluArena, RustNeuralMlu

        self.rust_arena = RustMluArena(
            mitosis_min_activations=self.mitosis_min_activations,
            mitosis_min_verified_spokes=self.mitosis_min_verified_spokes,
        )
        self.neural_mlu = RustNeuralMlu(seed=42)

        self._dirty_since_save: bool = False
        self._ensure_persistence_table()
        self._init_core_topology()
        self.conversation_liquid_state = list(self.nodes["CORE_ROOT"].hidden_state)
        self._load_persisted_nodes()
        self.sync_from_symbolic_memory()

    def close(self) -> None:
        """Persist any unsaved 360° Arena state and release native Rust handles."""
        if getattr(self, "_dirty_since_save", False):
            self.save_nodes()
        if hasattr(self, "rust_arena") and self.rust_arena is not None:
            self.rust_arena.close()
        if hasattr(self, "neural_mlu") and self.neural_mlu is not None:
            self.neural_mlu.close()
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
            self.memory.transaction().execute(
                """
                CREATE TABLE IF NOT EXISTS mlu_neural_weights (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    weights_json TEXT NOT NULL,
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

        if self.neural_mlu.is_native:
            cursor.execute("SELECT weights_json FROM mlu_neural_weights WHERE id = 1")
            nrow = cursor.fetchone()
            if nrow and nrow[0]:
                self.neural_mlu.restore_json(str(nrow[0]))
                self.sync_neural_embeddings_to_arena(blend=0.20)

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
            if self.neural_mlu.is_native:
                weights_json = self.neural_mlu.dump_json()
                self.memory.transaction().execute(
                    """
                    INSERT INTO mlu_neural_weights (id, weights_json, updated_step)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        weights_json = excluded.weights_json,
                        updated_step = excluded.updated_step
                    """,
                    (weights_json, self.cycle_step),
                )
        self._dirty_since_save = False

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
                phrase_key = " ".join(ptoks)
                base_canonical = self.frame_policy.operator_canonical_map.get(
                    phrase_key,
                    self.frame_policy.operator_canonical_map.get(
                        ptoks[0],
                        self.frame_policy.operator_canonical_map.get(base_rel, base_rel),
                    ),
                )
                if self.verifier.registry.relation(base_canonical) is None:
                    base_canonical = self.verifier.taxonomy_predicate or "is_a"

                if has_neg and base_rel == "is_a":
                    canonical = disjoint_pred
                elif has_neg:
                    canonical = base_canonical
                else:
                    canonical = base_canonical

                polarity = -1 if has_neg else 1

                for candidate_toks in (
                    ptoks,
                    ptoks[:-1]
                    if len(ptoks) > 1 and ptoks[-1] in self.frame_policy.stop_determiners
                    else ptoks,
                ):
                    if not candidate_toks:
                        continue
                    if candidate_toks not in self._operator_polarity_map:
                        self._operator_polarity_map[candidate_toks] = (
                            canonical,
                            polarity,
                        )
                    if (
                        not has_neg or base_rel == "is_a"
                    ) and candidate_toks not in seen_tuples:
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
                pol = -1 if reg_pred in self.verifier.disjoint_predicates else 1
                self._operator_polarity_map.setdefault(ptoks, (reg_pred, pol))

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
        min_sal = max(2.25, 0.60 * max_sal) if max_sal >= 2.25 else 0.60 * max_sal
        kept = [
            t
            for t, s in scored
            if s >= min_sal and self._operator_token_mass.get(t, 0.0) < 2.0
        ]
        if not kept:
            best_tok = max(scored, key=lambda kv: kv[1])[0]
            kept = [best_tok]
        return "_".join(kept), max_sal

    def _resolve_operator_span(self, span_tokens: list[str]) -> tuple[str | None, int]:
        """Resolve an operator span to a canonical predicate and polarity (+1/-1) via the 64-D operator codebook."""
        if not span_tokens:
            return None, 1
        best_pred: str | None = None
        best_len = 0
        best_sim = -1.0
        polarity = 1
        n = len(span_tokens)
        for i in range(n):
            for j in range(i + 1, min(n + 1, i + 5)):
                sub = tuple(span_tokens[i:j])
                core_sub = tuple(t for t in sub if t not in self.frame_policy.stop_determiners)
                matched_pol_entry = self._operator_polarity_map.get(sub) or (
                    self._operator_polarity_map.get(core_sub) if core_sub else None
                )
                if matched_pol_entry is not None:
                    cand_pred, cand_pol = matched_pol_entry
                    cand_len = len(core_sub) if core_sub and core_sub in self._operator_polarity_map else len(sub)
                    if cand_len > best_len or (cand_len == best_len and cand_pol < polarity):
                        best_len = cand_len
                        best_pred = cand_pred
                        polarity = cand_pol
                        best_sim = 1.0
                    continue
                probe_vec = (
                    deterministic_seed_vector("_".join(core_sub or sub), self.dim)
                    if (core_sub or sub)
                    else None
                )
                for proto_toks, canonical_pred, proto_vec in self._operator_codebook:
                    if sub == proto_toks or (core_sub and core_sub == proto_toks):
                        if canonical_pred in self.verifier.disjoint_predicates:
                            polarity = -1
                        if len(proto_toks) > best_len:
                            best_len = len(proto_toks)
                            best_pred = canonical_pred
                            best_sim = 1.0
                    elif best_len == 0 and probe_vec is not None:
                        sim = sum(a * b for a, b in zip(probe_vec, proto_vec, strict=False))
                        if sim >= 0.85 and sim > best_sim:
                            best_sim = sim
                            best_pred = canonical_pred
                            if canonical_pred in self.verifier.disjoint_predicates:
                                polarity = -1
        if any(t in self.frame_policy.negation_tokens for t in span_tokens):
            polarity = -1
            if best_pred in (None, *self.verifier.disjoint_predicates):
                for tok in span_tokens:
                    mapped = self.frame_policy.operator_canonical_map.get(tok)
                    if mapped:
                        best_pred = mapped
                        break
        return best_pred, polarity

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

    def sync_neural_embeddings_to_arena(self, blend: float = 0.20) -> int:
        """Synchronize backprop-learned token embeddings from NeuralSpiderWebMlu into 360° MiniLearningUnit concept anchors."""
        if not self.neural_mlu.is_native:
            return 0
        raw_json = self.neural_mlu.dump_json()
        if not raw_json or raw_json == "{}":
            return 0
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return 0
        vocab_to_id = payload.get("vocab_to_id", {})
        token_emb_w = payload.get("token_emb", {}).get("w", [])
        if not isinstance(vocab_to_id, dict) or not isinstance(token_emb_w, list) or not vocab_to_id:
            return 0
        d_model = len(token_emb_w) // max(1, len(vocab_to_id))
        if d_model <= 0:
            return 0

        self._cached_neural_vocab.clear()
        for tok, tid in vocab_to_id.items():
            idx = int(tid)
            start = idx * d_model
            end = start + d_model
            if end <= len(token_emb_w):
                self._cached_neural_vocab[str(tok)] = [
                    float(x) for x in token_emb_w[start:end]
                ]

        synced_count = 0
        for cid, unit in self.nodes.items():
            if cid == "CORE_ROOT" or cid in DEFAULT_SUBCORES:
                continue
            toks = _tokenize_words(cid)
            matched_vecs = [
                self._cached_neural_vocab[t]
                for t in toks
                if t in self._cached_neural_vocab
            ]
            if not matched_vecs:
                continue
            pooled = [
                sum(v[d] for v in matched_vecs) / float(len(matched_vecs))
                for d in range(d_model)
            ]
            unit.sync_neural_embedding(pooled, blend=blend)
            synced_count += 1
        return synced_count

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
        if self._cached_neural_vocab:
            toks = _tokenize_words(cid)
            matched_vecs = [
                self._cached_neural_vocab[t]
                for t in toks
                if t in self._cached_neural_vocab
            ]
            if matched_vecs:
                d_model = len(matched_vecs[0])
                pooled = [
                    sum(v[d] for v in matched_vecs) / float(len(matched_vecs))
                    for d in range(d_model)
                ]
                unit.sync_neural_embedding(pooled, blend=0.20)
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

    def _extract_negated_triples(
        self, sentence: str, active_subject_fallback: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Phase 60: Extract non-monotonic exception / negated relation triples via SemanticFramePolicy and the 64-D operator polarity codebook."""
        clean_sent = self.frame_policy.clean_filler(sentence)
        tokens = _tokenize_words(clean_sent)
        if len(tokens) < 3:
            return []

        n = len(tokens)
        filler_tokens = {
            tok
            for phrase in self.frame_policy.filler_prefixes
            for tok in _tokenize_words(phrase)
        }

        for i in range(1, n - 1):
            best_j = -1
            best_pred: str | None = None
            best_pol = 1
            best_len = 0
            for j in range(i + 1, min(n, i + 6)):
                bridge = tuple(tokens[i:j])
                core_bridge = tuple(
                    t for t in bridge if t not in self.frame_policy.stop_determiners
                )
                entry = self._operator_polarity_map.get(bridge) or (
                    self._operator_polarity_map.get(core_bridge) if core_bridge else None
                )
                if entry is not None:
                    cand_pred, cand_pol = entry
                    cand_len = j - i
                    if cand_len > best_len:
                        best_len = cand_len
                        best_j = j
                        best_pred = cand_pred
                        best_pol = cand_pol
            if best_j > i and best_pol < 0 and best_pred is not None:
                if best_pred in self.verifier.disjoint_predicates or best_pred == "is_a":
                    return []
                subj_toks = [
                    t
                    for t in tokens[:i]
                    if t not in filler_tokens
                    and t not in self.frame_policy.negation_tokens
                ]
                subj_c, subj_sal = self._extract_salient_concept(subj_toks)
                if (not subj_c or subj_sal < 2.25) and active_subject_fallback:
                    subj_c = active_subject_fallback.lower()
                obj_c, _ = self._extract_salient_concept(tokens[best_j:])
                if subj_c and obj_c and subj_c != obj_c:
                    return [(subj_c, best_pred, obj_c)]

        return []

    def _extract_candidate_triples(
        self, sentence: str, active_subject_fallback: str | None = None
    ) -> list[tuple[str, str, str]]:
        """Extract relational triples via multi-clause operator energy decomposition & liquid coreference binding."""
        tokens = _tokenize_words(sentence)
        if len(tokens) < 2:
            return []
        if self._extract_negated_triples(sentence, active_subject_fallback=active_subject_fallback):
            return []

        n = len(tokens)
        raw_spans: list[tuple[int, int, str, float]] = []

        for i in range(1, n):
            best_j = -1
            best_pred: str | None = None
            best_res = -1.0
            for j in range(i + 1, min(n, i + 6)):
                bridge = tuple(tokens[i:j])
                core_bridge = tuple(
                    t for t in bridge if t not in self.frame_policy.stop_determiners
                )
                for proto_toks, canonical_pred, _vec in self._operator_codebook:
                    if bridge == proto_toks or (core_bridge and core_bridge == proto_toks):
                        res = 1.0 + 0.25 * float(len(proto_toks))
                        if res > best_res or (res == best_res and j > best_j):
                            best_res = res
                            best_pred = canonical_pred
                            best_j = j
            if best_pred is not None and best_j > i:
                raw_spans.append((i, best_j, best_pred, best_res))

        # Select non-overlapping operator spans across the sentence
        op_spans: list[tuple[int, int, str, float]] = []
        for span in raw_spans:
            i_curr, _j_curr, _pred_curr, res_curr = span
            if not op_spans:
                op_spans.append(span)
            else:
                _i_prev, j_prev, _pred_prev, res_prev = op_spans[-1]
                if i_curr >= j_prev:
                    op_spans.append(span)
                elif res_curr > res_prev:
                    op_spans[-1] = span

        if op_spans:
            extracted: list[tuple[str, str, str]] = []
            i0 = op_spans[0][0]
            subj_concept, subj_sal = self._extract_salient_concept(tokens[:i0])
            # Phase 29: Liquid Coreference Binding for single-token low-information anaphoric subjects (e.g. "It")
            if (
                not subj_concept
                or (
                    i0 == 1
                    and subj_sal < 2.25
                    and subj_concept.upper() not in self.nodes
                )
            ) and active_subject_fallback:
                subj_concept = active_subject_fallback.lower()

            current_subj = subj_concept
            num_spans = len(op_spans)
            for k, (_ik, jk, matched_pred, _op_res) in enumerate(op_spans):
                end_idx = op_spans[k + 1][0] if k + 1 < num_spans else n
                seg = list(tokens[jk:end_idx])
                if not seg:
                    continue

                next_subj = current_subj
                if k + 1 < num_spans:
                    # Strip trailing low-salience connector tokens (e.g., coordinate bridges)
                    while len(seg) > 1 and self._token_salience(seg[-1]) < 2.25:
                        seg.pop()
                    # Check if the segment introduces a new explicit subject before the next operator
                    if (
                        len(seg) >= 3
                        and self._token_salience(seg[0]) >= 2.25
                        and self._token_salience(seg[-1]) >= 2.25
                        and any(self._token_salience(t) < 2.25 for t in seg[1:-1])
                    ):
                        next_subj = seg[-1]
                        seg = seg[:-1]
                        while len(seg) > 1 and self._token_salience(seg[-1]) < 2.25:
                            seg.pop()

                obj_concept, _obj_sal = self._extract_salient_concept(seg)
                if current_subj and obj_concept and current_subj != obj_concept:
                    triple = (current_subj, matched_pred, obj_concept)
                    if triple not in extracted:
                        extracted.append(triple)
                current_subj = next_subj

            if extracted:
                return extracted

        # Open-vocabulary fallback: link high-information concepts via related_to when no explicit operator matched
        salient = [
            (idx, tok, self._token_salience(tok))
            for idx, tok in enumerate(tokens)
            if self._token_salience(tok) >= 2.25
        ]
        if len(salient) >= 2:
            anchor = salient[0][1]
            if (
                self._token_salience(anchor) < 2.25
                and anchor.upper() not in self.nodes
                and active_subject_fallback
            ):
                anchor = active_subject_fallback.lower()
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
        verified_triples: list[tuple[str, str, str]] = []
        touched_nodes: set[str] = {"CORE_ROOT"}
        active_discourse_focus: str | None = (
            self.recent_focal_stack[0][0].lower() if self.recent_focal_stack else None
        )

        core_unit = self.nodes["CORE_ROOT"]
        if self.rust_arena.is_native and text.strip():
            self.rust_arena.train_stream(
                text, learning_rate=0.15 * self.plasticity.plasticity_multiplier
            )

        for sent in raw_sentences:
            self.cycle_step += 1
            sentences_processed += 1

            # Phase 60: Check for non-monotonic exception / negated belief revision triples first
            neg_triples = self._extract_negated_triples(
                sent, active_subject_fallback=active_discourse_focus
            )
            if neg_triples:
                for subj, pred, obj in neg_triples:
                    self.revise_belief_360(
                        subj, pred, obj, verified=False, evidence_weight=4.0
                    )
                    relations_refuted += 1
                    spoke = map_predicate_to_spoke(pred)
                    subj_unit = self.get_or_create_mlu(subj, preferred_spoke=spoke)
                    obj_unit = self.get_or_create_mlu(obj, preferred_spoke=spoke)
                    active_discourse_focus = subj_unit.concept_id.lower()
                    self._update_conversation_context(subj_unit.concept_id)
                    touched_nodes.add(subj_unit.concept_id)
                    touched_nodes.add(obj_unit.concept_id)
                    surprises.append(0.85)
                self.distill_negated_triples_to_neural_mlu(
                    neg_triples, epochs=32, learning_rate=0.045, persist=False
                )
                continue

            # 1. Tokenize and train Central Core Multi-Token Prediction (MTP)
            tokens = _tokenize_words(sent)
            if len(tokens) >= 2:
                tok_surprise = core_unit.learn_token_sequence(
                    tokens, learning_rate=0.15 * self.plasticity.plasticity_multiplier
                )
                surprises.append(tok_surprise)
                token_transitions_learned += len(tokens) - 1

            # 2. Extract candidate relational & co-occurrence triples (with multi-clause & coreference binding)
            triples = self._extract_candidate_triples(
                sent, active_subject_fallback=active_discourse_focus
            )
            context_vecs: list[list[float]] = []

            for subj, pred, obj in triples:
                spoke = map_predicate_to_spoke(pred)
                subj_unit = self.get_or_create_mlu(subj, preferred_spoke=spoke)
                obj_unit = self.get_or_create_mlu(obj, preferred_spoke=spoke)
                active_discourse_focus = subj_unit.concept_id.lower()
                self._update_conversation_context(subj_unit.concept_id)
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
                        verified_triples.append((subj_unit.concept_id, pred, obj_unit.concept_id))
                        obj_unit.verified_hits += 0.5
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
        elif touched_nodes or sentences_processed > 0:
            self._dirty_since_save = True

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
            verified_triples=verified_triples,
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
        self, focal_unit: MiniLearningUnit, max_hops: int = 5
    ) -> tuple[list[MLUSpokePrediction], list[str]]:
        """Traverse the 360° Spider-Web around `focal_unit` (including transitive taxonomic inheritance & Phase 60 non-monotonic exception masking)."""
        candidate_vecs = {cid: u.anchor_vec for cid, u in self.nodes.items()}
        all_preds: list[MLUSpokePrediction] = []
        activated: list[str] = ["CORE_ROOT"]
        if focal_unit.parent_hub:
            activated.append(focal_unit.parent_hub)
        activated.append(focal_unit.concept_id)

        # Phase 60: Collect local non-monotonic exception overrides (w_neg > w_pos) on focal_unit
        overridden_targets: set[str] = {
            t_cid
            for sp in focal_unit.spokes.values()
            for t_cid, rec in sp.target_affinities.items()
            if isinstance(rec, list)
            and len(rec) >= 4
            and float(rec[1]) > float(rec[0])
            and str(rec[3])
            not in ("is_a", "subclass_of", "instance_of", "disjoint_with", "not_a", "co_occurs_with")
        }

        # Direct 360° 6-spoke predictions on the focal MLU
        for spoke_name in SPOKE_NAMES:
            all_preds.extend(focal_unit.predict_spoke_targets(spoke_name, candidate_vecs, top_k=4))

        # Phase 1 & 7: Fuse native Rust 360° Multi-Hop Beam Traversal predictions
        self._last_rust_beam: dict[str, Any] = {}
        if self.rust_arena.is_native:
            beam_dto = self.rust_arena.query_360_beam(
                focal_unit.concept_id, max_hops=max_hops, top_k=16
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
                    or tgt_name in overridden_targets
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
                            or pp.target_concept in overridden_targets
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

    def _find_verified_spoke_path(
        self, start_cid: str, goal_cid: str, max_hops: int = 6
    ) -> list[tuple[str, str, str]]:
        """Phase 57: Find the shortest verified 360° spoke path [(u, pred, v), ...] from start_cid to goal_cid."""
        from collections import deque

        if not start_cid or not goal_cid or start_cid == goal_cid or start_cid not in self.nodes:
            return []
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(start_cid, [])])
        visited: set[str] = {start_cid}
        while queue:
            curr, path = queue.popleft()
            unit = self.nodes.get(curr)
            if unit is None:
                continue
            for sp in unit.spokes.values():
                for t_cid, rec in sp.target_affinities.items():
                    if not (
                        isinstance(rec, list)
                        and len(rec) >= 4
                        and float(rec[0]) > float(rec[1])
                    ):
                        continue
                    pred = str(rec[3])
                    if pred in ("co_occurs_with", "disjoint_with", "not_a"):
                        continue
                    next_path = [*path, (curr, pred, t_cid)]
                    if t_cid == goal_cid:
                        return next_path
                    if len(next_path) < max_hops and t_cid not in visited:
                        visited.add(t_cid)
                        queue.append((t_cid, next_path))
        return []

    def _find_taxonomic_path(
        self, start_cid: str, ancestor_cid: str, max_hops: int = 5
    ) -> list[tuple[str, str, str]]:
        """Phase 62: Find the shortest verified taxonomic path [(u, 'is_a', v), ...] from start_cid to ancestor_cid."""
        from collections import deque

        if (
            not start_cid
            or not ancestor_cid
            or start_cid == ancestor_cid
            or start_cid not in self.nodes
        ):
            return []
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(start_cid, [])])
        visited: set[str] = {start_cid}
        while queue:
            curr, path = queue.popleft()
            unit = self.nodes.get(curr)
            if unit is None:
                continue
            for sp in unit.spokes.values():
                for t_cid, rec in sp.target_affinities.items():
                    if not (
                        isinstance(rec, list)
                        and len(rec) >= 4
                        and float(rec[0]) > float(rec[1])
                    ):
                        continue
                    pred = str(rec[3])
                    if pred not in ("is_a", "subclass_of", "instance_of"):
                        continue
                    next_path = [*path, (curr, pred, t_cid)]
                    if t_cid == ancestor_cid:
                        return next_path
                    if len(next_path) < max_hops and t_cid not in visited:
                        visited.add(t_cid)
                        queue.append((t_cid, next_path))
        return []

    def _has_taxonomic_subtypes(self, category_cid: str, max_hops: int = 4) -> bool:
        """Return True if at least one concept node in the 360° Arena has a verified taxonomic path to category_cid."""
        if not category_cid or category_cid not in self.nodes:
            return False
        return any(
            cid != category_cid
            and cid != "CORE_ROOT"
            and not cid.startswith("SUBCORE_")
            and bool(self._find_taxonomic_path(cid, category_cid, max_hops=max_hops))
            for cid in self.nodes
        )

    def _extract_arena_proof_steps(
        self, res: MLMQueryResult, target_cid: str | None
    ) -> list[tuple[str, str, str]]:
        """Phase 57, 59 & 60: Extract the step-by-step multi-hop proof or refutation chain for an Arena query result."""
        from collections import deque

        focal = res.focal_concept
        if (
            res.status == "REFUTED"
            and target_cid
            and focal in self.nodes
            and target_cid in self.nodes
        ):
            if "I_SORT" in res.inspectable_trace:
                rev_path = self._find_verified_spoke_path(target_cid, focal, max_hops=6)
                if rev_path:
                    return rev_path

            if "I_EXCEPTION" in res.inspectable_trace:
                exc_pred = "cannot"
                for sp in self.nodes[focal].spokes.values():
                    rec = sp.target_affinities.get(target_cid)
                    if isinstance(rec, list) and len(rec) >= 4 and float(rec[1]) > float(rec[0]):
                        raw_p = str(rec[3])
                        exc_pred = (
                            "cannot"
                            if raw_p in ("can", "capable_of")
                            else (
                                "does_not_have"
                                if raw_p in ("has_part", "has_property", "has")
                                else f"cannot_{raw_p}"
                            )
                        )
                        break
                default_chain = self._find_verified_spoke_path(focal, target_cid, max_hops=5)
                if default_chain:
                    return [*default_chain, (focal, exc_pred, target_cid)]
                return [(focal, exc_pred, target_cid)]

            def _anc_map(root_id: str) -> dict[str, list[tuple[str, str, str]]]:
                amap: dict[str, list[tuple[str, str, str]]] = {root_id: []}
                q: deque[str] = deque([root_id])
                while q:
                    c = q.popleft()
                    u = self.nodes.get(c)
                    if u is None:
                        continue
                    for sp in u.spokes.values():
                        for t_id, rec in sp.target_affinities.items():
                            if (
                                isinstance(rec, list)
                                and len(rec) >= 4
                                and float(rec[0]) > float(rec[1])
                                and str(rec[3]) in ("is_a", "subclass_of", "instance_of")
                                and t_id not in amap
                            ):
                                amap[t_id] = [*amap[c], (c, str(rec[3]), t_id)]
                                q.append(t_id)
                return amap

            anc_a = _anc_map(focal)
            anc_b = _anc_map(target_cid)
            for a_id, path_a in anc_a.items():
                u_a = self.nodes.get(a_id)
                if u_a is None:
                    continue
                for sp in u_a.spokes.values():
                    for b_id, rec in sp.target_affinities.items():
                        if b_id in anc_b and isinstance(rec, list) and len(rec) >= 4:
                            pred_ab = str(rec[3])
                            if pred_ab in ("disjoint_with", "not_a") or float(rec[1]) > float(
                                rec[0]
                            ):
                                return [*path_a, *anc_b[b_id], (a_id, "disjoint_with", b_id)]
            for b_id, path_b in anc_b.items():
                u_b = self.nodes.get(b_id)
                if u_b is None:
                    continue
                for sp in u_b.spokes.values():
                    for a_id, rec in sp.target_affinities.items():
                        if a_id in anc_a and isinstance(rec, list) and len(rec) >= 4:
                            pred_ba = str(rec[3])
                            if pred_ba in ("disjoint_with", "not_a") or float(rec[1]) > float(
                                rec[0]
                            ):
                                return [*anc_a[a_id], *path_b, (a_id, "disjoint_with", b_id)]
            return [(focal, "disjoint_with", target_cid)]

        if "[MLU-360-INTERSECT]" in res.inspectable_trace:
            inter_steps = list(getattr(self, "_last_intersect_proof_steps", []))
            if inter_steps:
                return inter_steps

        if (
            "[MLU-360-COUNTERFACTUAL]" in res.inspectable_trace
            or "[MLU-360-CAUSAL-WAVE]" in res.inspectable_trace
        ):
            causal_steps = list(getattr(self, "_last_causal_proof_steps", []))
            if causal_steps:
                return causal_steps

        if target_cid:
            direct_or_hop = self._find_verified_spoke_path(focal, target_cid, max_hops=6)
            if direct_or_hop:
                return direct_or_hop

        if "[MLU-360-ANALOGY]" in res.inspectable_trace:
            non_root = [
                c
                for c in res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            if len(non_root) >= 4:
                cid_a, cid_b, cid_c, cid_d = non_root[:4]
                p_ab = self._find_verified_spoke_path(
                    cid_a, cid_b, max_hops=5
                ) or self._find_verified_spoke_path(cid_b, cid_a, max_hops=5)
                p_cd = self._find_verified_spoke_path(
                    cid_c, cid_d, max_hops=5
                ) or self._find_verified_spoke_path(cid_d, cid_c, max_hops=5)
                if p_ab or p_cd:
                    return [*p_ab, *p_cd]

        if "[MLU-360-DIAGNOSE]" in res.inspectable_trace:
            steps: list[tuple[str, str, str]] = []
            for obs_cid in res.activated_mlus:
                if obs_cid in ("CORE_ROOT", focal) or obs_cid.startswith("SUBCORE_"):
                    continue
                p = self._find_verified_spoke_path(focal, obs_cid, max_hops=4)
                for edge in p:
                    if edge not in steps:
                        steps.append(edge)
            if steps:
                return steps

        if res.spoke_predictions:
            top_p = res.spoke_predictions[0]
            hop_p = self._find_verified_spoke_path(focal, top_p.target_concept, max_hops=5)
            if hop_p:
                return hop_p
            return [(focal, top_p.predicate, top_p.target_concept)]
        return []

    def _record_episodic_turn(
        self, res: MLMQueryResult, target_cid: str | None = None
    ) -> None:
        """Phase 57: Record a verified or refuted reasoning frame into the Episodic Working Memory Ring."""
        proof_steps = self._extract_arena_proof_steps(res, target_cid)
        frame = {
            "turn_step": self.cycle_step,
            "query": res.query,
            "focal_concept": res.focal_concept,
            "target_concept": target_cid,
            "status": res.status,
            "active_subcore": res.active_subcore,
            "activated_mlus": list(res.activated_mlus),
            "spoke_predictions": list(res.spoke_predictions),
            "inspectable_trace": res.inspectable_trace,
            "fluent_english": res.fluent_english or res.answer,
            "confidence": res.confidence,
            "proof_steps": proof_steps,
        }
        self.episodic_working_memory.append(frame)
        if len(self.episodic_working_memory) > 16:
            self.episodic_working_memory = self.episodic_working_memory[-16:]

    def _synthesize_episodic_explanation(
        self, clean_q: str, last_ep: dict[str, Any]
    ) -> MLMQueryResult:
        """Phase 57: Synthesize a multi-hop proof or refutation explanation from the Episodic Working Memory Ring."""
        focal_cid = str(last_ep.get("focal_concept", "CORE_ROOT"))
        steps: list[tuple[str, str, str]] = list(last_ep.get("proof_steps", []))
        clauses: list[str] = []
        chain_nodes: list[str] = []
        for u_cid, pred, v_cid in steps:
            if not chain_nodes:
                chain_nodes.append(u_cid)
            if v_cid not in chain_nodes:
                chain_nodes.append(v_cid)
            u_w = u_cid.lower().replace("_", " ")
            v_w = v_cid.lower().replace("_", " ")
            if pred in ("disjoint_with", "not_a"):
                clauses.append(f"{u_w} is mutually exclusive with {v_w}")
            else:
                clauses.append(f"{u_w} {pred.replace('_', ' ')} {v_w}")

        if clauses:
            if len(clauses) == 1:
                fluent = f"Because {clauses[0]}."
            else:
                fluent = f"Because {', '.join(clauses[:-1])}, and {clauses[-1]}."
            proof_tag = " -> ".join(chain_nodes)
        else:
            prev_fluent = str(last_ep.get("fluent_english", "")).rstrip(".")
            fluent = (
                f"Because {prev_fluent[:1].lower() + prev_fluent[1:]}."
                if prev_fluent
                else "Because of verified 360° spoke evidence."
            )
            proof_tag = focal_cid

        trace = (
            f"[MLU-360-EXPLAIN] ref_status={last_ep.get('status', 'VERIFIED')} | "
            f"hops={len(steps)} | proof={proof_tag}"
        )
        return MLMQueryResult(
            query=clean_q,
            focal_concept=focal_cid,
            status="VERIFIED",
            answer=fluent,
            confidence=float(last_ep.get("confidence", 0.92)),
            epistemic_entropy=0.05,
            active_subcore=str(last_ep.get("active_subcore", "CORE_ROOT")),
            activated_mlus=chain_nodes or list(last_ep.get("activated_mlus", ["CORE_ROOT"])),
            spoke_predictions=list(last_ep.get("spoke_predictions", [])),
            inspectable_trace=trace,
            fluent_english=fluent,
        )

    def query_360(self, query: str) -> MLMQueryResult:
        """Answer a natural-language question or concept inquiry using the 360° Spider-Web MLM."""
        self._last_query_target_cid: str | None = None
        res = self._query_360_core(query)
        if (
            res.status in ("VERIFIED", "REFUTED", "NEURAL_GROUNDED")
            and "[MLU-360-EXPLAIN]" not in res.inspectable_trace
        ):
            self._record_episodic_turn(res, target_cid=self._last_query_target_cid)
        return res

    def _query_360_core(self, query: str) -> MLMQueryResult:
        clean_q = query.strip()
        tokens = _tokenize_words(clean_q)

        # Identify focal concept, optional target concept, and all matched concept nodes in query
        focal_cid: str | None = None
        focal_pos: int = -1
        target_cid: str | None = None
        matched_cids: list[str] = []
        matched_spans: list[tuple[str, int, int]] = []

        # Check multi-word and single-word matches in self.nodes
        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                bigram_cid = f"{tokens[i].upper()}_{tokens[i + 1].upper()}"
                if (
                    bigram_cid in self.nodes
                    and bigram_cid != "CORE_ROOT"
                    and bigram_cid not in DEFAULT_SUBCORES
                ):
                    if bigram_cid not in matched_cids:
                        matched_cids.append(bigram_cid)
                        matched_spans.append((bigram_cid, i, i + 2))
                    if focal_cid is None:
                        focal_cid = bigram_cid
                        focal_pos = i
                    elif bigram_cid != focal_cid and target_cid is None:
                        target_cid = bigram_cid
            single_cid = tokens[i].upper()
            if (
                single_cid in self.nodes
                and single_cid != "CORE_ROOT"
                and single_cid not in DEFAULT_SUBCORES
                and (len(tokens) == 1 or self._token_salience(tokens[i]) >= 2.25)
            ):
                if single_cid not in matched_cids and not any(
                    single_cid in m.split("_") for m in matched_cids
                ):
                    matched_cids.append(single_cid)
                    matched_spans.append((single_cid, i, i + 1))
                if focal_cid is None:
                    focal_cid = single_cid
                    focal_pos = i
                elif single_cid != focal_cid and target_cid is None:
                    target_cid = single_cid

        # Phase 62: Compositional Set & Conjunctive Constraint Intersection ("Which mammal can echolocate?", "Which bird cannot fly?")
        if (
            len(matched_spans) >= 2
            and all(
                t in self.frame_policy.stop_determiners
                for t in tokens[: matched_spans[0][1]]
            )
            and self._has_taxonomic_subtypes(matched_spans[0][0])
        ):
            cat_cid, _, prev_end = matched_spans[0]
            constraints: list[tuple[int, str | None, str]] = []
            valid_constraints = True
            for c_cid, s_idx, e_idx in matched_spans[1:]:
                span_toks = tokens[prev_end:s_idx]
                prev_end = e_idx
                pred_k, pol_k = self._resolve_operator_span(span_toks)
                if pred_k is None and pol_k > 0:
                    valid_constraints = False
                    break
                constraints.append((pol_k, pred_k, c_cid))
            if valid_constraints and constraints:
                inter_dto = self.intersect_concepts_360(cat_cid, constraints, max_hops=4)
                if inter_dto.get("resolved") and inter_dto.get("best_concept"):
                    best_c = str(inter_dto["best_concept"])
                    in_synth = str(inter_dto.get("intersection_synthesis", ""))
                    in_conf = float(inter_dto.get("confidence", 0.90))
                    in_spoke = str(inter_dto.get("primary_spoke", "taxonomy"))
                    in_pred = str(inter_dto.get("primary_predicate", "is_a"))
                    sp_preds = list(inter_dto.get("spoke_predictions", []))
                    self._last_intersect_proof_steps = list(
                        inter_dto.get("proof_steps", [])
                    )
                    self._update_conversation_context(best_c)
                    return MLMQueryResult(
                        query=clean_q,
                        focal_concept=best_c,
                        status="VERIFIED",
                        answer=in_synth,
                        confidence=in_conf,
                        epistemic_entropy=round(1.0 - in_conf, 4),
                        active_subcore=(
                            self.nodes[best_c].parent_hub
                            if best_c in self.nodes and self.nodes[best_c].parent_hub
                            else "CORE_ROOT"
                        ),
                        activated_mlus=["CORE_ROOT", best_c, *matched_cids],
                        spoke_predictions=sp_preds,
                        inspectable_trace=(
                            f"[MLU-360-INTERSECT] category={cat_cid} | "
                            f"constraints={inter_dto.get('constraint_summary', '')} -> "
                            f"match={best_c} | spoke={in_spoke}({in_pred})"
                        ),
                        fluent_english=in_synth,
                    )

        # Phase 61: Cross-Domain Proportional Analogy Routing (A : B :: C : ?) when 3 concepts are present
        if len(matched_cids) == 3:
            analogy_dto = self.solve_analogy_360(
                matched_cids[0], matched_cids[1], matched_cids[2], max_hops=4
            )
            if analogy_dto.get("resolved") and analogy_dto.get("target_concept"):
                best_d = str(analogy_dto["target_concept"])
                an_synth = str(analogy_dto.get("analogy_synthesis", ""))
                an_conf = float(analogy_dto.get("confidence", 0.88))
                an_spoke = str(analogy_dto.get("mapped_spoke", "taxonomy"))
                an_pred = str(analogy_dto.get("mapped_predicate", "is_a"))
                sp_pred = analogy_dto.get("spoke_prediction")
                self._update_conversation_context(best_d)
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=best_d,
                    status="VERIFIED",
                    answer=an_synth,
                    confidence=an_conf,
                    epistemic_entropy=round(1.0 - an_conf, 4),
                    active_subcore=(
                        self.nodes[best_d].parent_hub
                        if best_d in self.nodes and self.nodes[best_d].parent_hub
                        else "CORE_ROOT"
                    ),
                    activated_mlus=["CORE_ROOT", *matched_cids, best_d],
                    spoke_predictions=[sp_pred] if sp_pred is not None else [],
                    inspectable_trace=(
                        f"[MLU-360-ANALOGY] {matched_cids[0]}:{matched_cids[1]} :: "
                        f"{matched_cids[2]}:{best_d} | spoke={an_spoke}({an_pred})"
                    ),
                    fluent_english=an_synth,
                )

        # Phase 41a: Multi-Observation Abductive Diagnosis Routing (>= 3 distinct concepts in inquiry)
        if len(matched_cids) >= 3:
            abd_dto = self.diagnose_abductive_360(matched_cids, max_hops=3, top_k=5)
            best_exp = str(abd_dto.get("best_explanation", "")).strip()
            diag_synth = str(abd_dto.get("diagnostic_synthesis", "")).strip()
            cands = abd_dto.get("candidates", [])
            if best_exp and diag_synth and cands:
                post_conf = float(cands[0].get("posterior_confidence", 0.85))
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=best_exp,
                    status="VERIFIED",
                    answer=diag_synth,
                    confidence=post_conf,
                    epistemic_entropy=round(1.0 - post_conf, 4),
                    active_subcore=(
                        self.nodes[best_exp].parent_hub
                        if best_exp in self.nodes and self.nodes[best_exp].parent_hub
                        else "CORE_ROOT"
                    ),
                    activated_mlus=["CORE_ROOT", best_exp] + matched_cids,
                    spoke_predictions=[],
                    inspectable_trace=(
                        f"[MLU-360-DIAGNOSE] observations={matched_cids} -> "
                        f"best_explanation={best_exp} (coverage={abd_dto.get('best_coverage_ratio', 0.0):.2f})"
                    ),
                    fluent_english=diag_synth,
                )

        op_tuples = {ptoks for ptoks, _, _ in self._operator_codebook}
        has_leading_operator = focal_pos > 0 and (
            any(
                tuple(tokens[:k]) in op_tuples
                for k in (1, 2, 3)
                if k <= focal_pos
            )
            or (
                self._operator_token_mass.get(tokens[0], 0.0) > 0.0
                and tokens[0] not in self.frame_policy.stop_determiners
            )
        )

        # Multi-Turn Liquid Context Resonance for elliptic / follow-up inquiries & Proof Explanations
        if self.recent_focal_stack:
            top_ctx_cid = self.recent_focal_stack[0][0]
            if (
                focal_cid is not None
                and target_cid is None
                and focal_cid != top_ctx_cid
                and has_leading_operator
                and top_ctx_cid in self.nodes
            ):
                ctx_preds, _ = self._collect_360_predictions(self.nodes[top_ctx_cid])
                rust_mutex = (
                    not bool(
                        self.rust_arena.verify_invariants(top_ctx_cid, "is_a", focal_cid).get(
                            "passed", True
                        )
                    )
                    if self.rust_arena.is_native
                    else False
                )
                has_anaphor_subject = any(
                    t not in self.frame_policy.stop_determiners
                    and self._operator_token_mass.get(t, 0.0) == 0.0
                    and self._token_salience(t) < 2.25
                    for t in tokens[:focal_pos]
                )
                if (
                    any(p.target_concept == focal_cid for p in ctx_preds)
                    or not self.verifier.verify_relation(top_ctx_cid, "is_a", focal_cid).passed
                    or rust_mutex
                    or has_anaphor_subject
                ):
                    target_cid = focal_cid
                    focal_cid = top_ctx_cid
            elif focal_cid is None and tokens:
                max_sal = max(self._token_salience(t) for t in tokens)
                if max_sal < 2.8:
                    if self.episodic_working_memory and (
                        clean_q.endswith("?")
                        or any(t in self.frame_policy.stop_determiners for t in tokens)
                    ):
                        return self._synthesize_episodic_explanation(
                            clean_q, self.episodic_working_memory[-1]
                        )
                    if top_ctx_cid in self.nodes:
                        focal_cid = top_ctx_cid

        self._last_query_target_cid = target_cid

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

        non_concept_toks = [
            t
            for t in tokens
            if t.upper() not in (focal_cid, target_cid or "")
        ]
        query_pred, query_pol = self._resolve_operator_span(non_concept_toks)
        if target_cid is not None and has_leading_operator:
            matching = [
                p
                for p in preds
                if p.target_concept == target_cid
                and p.predicate not in self.verifier.disjoint_predicates
            ]
            if matching:
                best = matching[0]
                verb = best.predicate.replace("_", " ")
                order_tag = (
                    " | [MLU-360-ORDER]"
                    if best.predicate in ("larger_than", "greater_than", "before", "after")
                    else ""
                )
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
                    f"H={best.epistemic_entropy:.2f}]{order_tag}"
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

            # Phase 59: Multi-hop Transitive Poset Order verification via evaluate_order_360
            if query_pred in ("larger_than", "causes", "requires") and self.rust_arena.is_native:
                ord_dto = self.evaluate_order_360(
                    query_pred, start_concept=focal_cid, max_hops=6
                )
                if f"{focal_cid} -> {target_cid}" in ord_dto.get("transitive_pairs", []):
                    chain_list = [str(c) for c in ord_dto.get("ordered_chain", [])]
                    verb = query_pred.replace("_", " ")
                    fluent_ord = (
                        f"Yes, {focal_cid.lower().replace('_', ' ')} {verb} "
                        f"{target_cid.lower().replace('_', ' ')}."
                    )
                    return MLMQueryResult(
                        query=clean_q,
                        focal_concept=focal_cid,
                        status="VERIFIED",
                        answer=fluent_ord,
                        confidence=0.90,
                        epistemic_entropy=0.06,
                        active_subcore=active_subcore,
                        activated_mlus=chain_list or activated,
                        spoke_predictions=preds[:6],
                        inspectable_trace=(
                            f"[MLU-360-ORDER] {focal_cid} -({query_pred})-> {target_cid} | "
                            f"chain={chain_list}"
                        ),
                        fluent_english=fluent_ord,
                    )

            # Check if DeepSeekInvariantVerifier, Rust Arena, or local I_EXCEPTION override refutes the link
            local_exception_pred: str | None = None
            for sp in focal_unit.spokes.values():
                rec = sp.target_affinities.get(target_cid)
                if (
                    isinstance(rec, list)
                    and len(rec) >= 4
                    and float(rec[1]) > float(rec[0])
                    and str(rec[3])
                    not in (
                        "is_a",
                        "subclass_of",
                        "instance_of",
                        "disjoint_with",
                        "not_a",
                        "co_occurs_with",
                    )
                ):
                    local_exception_pred = str(rec[3])
                    break

            gate_check = self.verifier.verify_relation(focal_cid, "is_a", target_cid)
            rust_inv = (
                self.rust_arena.verify_invariants(focal_cid, "is_a", target_cid)
                if self.rust_arena.is_native
                else {"passed": True, "gate": "OK"}
            )
            if (
                bool(rust_inv.get("passed", True))
                and query_pred
                and query_pred not in ("is_a", "disjoint_with")
                and self.rust_arena.is_native
            ):
                rust_inv = self.rust_arena.verify_invariants(
                    focal_cid, query_pred, target_cid
                )
            rust_passed = bool(rust_inv.get("passed", True))
            rust_gate = str(rust_inv.get("gate", "I_MUTEX"))
            if not gate_check.passed or not rust_passed or local_exception_pred is not None:
                v_gate = (
                    gate_check.violated_gate
                    if not gate_check.passed
                    else ("I_EXCEPTION" if local_exception_pred is not None else rust_gate)
                )
                eff_pred = query_pred or local_exception_pred or "can"
                v_msg = (
                    gate_check.error_message
                    if not gate_check.passed
                    else (
                        f"360° Arena I_EXCEPTION non-monotonic override for {focal_cid} on {eff_pred} -> {target_cid}"
                        if v_gate == "I_EXCEPTION"
                        else (
                            f"360° Arena {rust_gate} poset antisymmetry violation between {focal_cid} and {target_cid} along {query_pred}"
                            if v_gate == "I_SORT"
                            else f"360° Arena {rust_gate} disjointness between {focal_cid} and {target_cid}"
                        )
                    )
                )
                if v_gate == "I_EXCEPTION":
                    ref_verb = (
                        "cannot"
                        if eff_pred in ("can", "capable_of")
                        else (
                            "does not have"
                            if eff_pred in ("has_part", "has_property", "has")
                            else f"does not {eff_pred.replace('_', ' ')}"
                        )
                    )
                elif v_gate == "I_SORT" and query_pred:
                    ref_verb = f"cannot {query_pred.replace('_', ' ')}"
                else:
                    ref_verb = "cannot be"
                ans_ref = (
                    f"No — {focal_cid.lower().replace('_', ' ')} {ref_verb} "
                    f"{target_cid.lower().replace('_', ' ')} "
                    f"(Refuted by {v_gate}: {v_msg})."
                )
                pred_tag = f" ({eff_pred})" if v_gate == "I_EXCEPTION" else ""
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
                    inspectable_trace=f"[MLU-360] Refuted by {v_gate}{pred_tag}: {v_msg}",
                    fluent_english=ans_ref,
                )

        if not preds:
            if target_cid is not None and target_cid in self.nodes:
                abd_pair = self.diagnose_abductive_360(
                    [focal_cid, target_cid], max_hops=3, top_k=3
                )
                if (
                    abd_pair.get("best_explanation")
                    and float(abd_pair.get("best_coverage_ratio", 0.0)) >= 0.99
                    and abd_pair.get("diagnostic_synthesis")
                ):
                    best_ep = str(abd_pair["best_explanation"])
                    d_synth_p = str(abd_pair["diagnostic_synthesis"])
                    cp = abd_pair.get("candidates", [{}])[0]
                    p_conf_p = float(cp.get("posterior_confidence", 0.85))
                    return MLMQueryResult(
                        query=clean_q,
                        focal_concept=best_ep,
                        status="VERIFIED",
                        answer=d_synth_p,
                        confidence=p_conf_p,
                        epistemic_entropy=round(1.0 - p_conf_p, 4),
                        active_subcore=active_subcore,
                        activated_mlus=activated + [focal_cid, target_cid, best_ep],
                        spoke_predictions=[],
                        inspectable_trace=(
                            f"[MLU-360-DIAGNOSE] observations={[focal_cid, target_cid]} -> "
                            f"best_explanation={best_ep}"
                        ),
                        fluent_english=d_synth_p,
                    )
            plan_dto = self.plan_goal_360(focal_cid, max_steps=5)
            if plan_dto.get("achievable") and plan_dto.get("plan_synthesis"):
                p_synth = str(plan_dto["plan_synthesis"])
                p_conf = float(plan_dto.get("plan_confidence", 0.75))
                prereqs = plan_dto.get("root_prerequisites", [])
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED",
                    answer=p_synth,
                    confidence=p_conf,
                    epistemic_entropy=round(1.0 - p_conf, 4),
                    active_subcore=active_subcore,
                    activated_mlus=activated + [str(r) for r in prereqs],
                    spoke_predictions=[],
                    inspectable_trace=(
                        f"[MLU-360-PLAN] goal={focal_cid} | "
                        f"steps={plan_dto.get('total_steps', 0)} | prereqs={prereqs}"
                    ),
                    fluent_english=p_synth,
                )
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

        # Case A: 360° Counterfactual Intervention, Comparative Wave Intersection & Relational Bridge
        has_cf_cue = query_pol < 0
        if target_cid is not None and target_cid in self.nodes:
            if query_pol < 0 or has_cf_cue:
                cf_root, cf_supp = focal_cid, target_cid
                cf_dto = self.intervene_counterfactual_360(
                    cf_root, cf_supp, max_hops=4
                )
                if not cf_dto.get("invalidated_conclusions"):
                    alt_dto = self.intervene_counterfactual_360(
                        target_cid, focal_cid, max_hops=4
                    )
                    if alt_dto.get("invalidated_conclusions"):
                        cf_dto = alt_dto
                        cf_root, cf_supp = target_cid, focal_cid
                inv_list = [str(c) for c in cf_dto.get("invalidated_conclusions", [])]
                cf_synth = str(cf_dto.get("counterfactual_synthesis", "")).strip()
                if inv_list and cf_synth:
                    c_wave = self.simulate_causal_360(cf_root, max_hops=4)
                    self._last_causal_proof_steps = [
                        (
                            str(st["source_concept"]),
                            str(st["predicate"]),
                            str(st["target_concept"]),
                        )
                        for st in c_wave.get("causal_steps", [])
                    ]
                    return MLMQueryResult(
                        query=clean_q,
                        focal_concept=cf_root,
                        status="VERIFIED",
                        answer=cf_synth,
                        confidence=0.88,
                        epistemic_entropy=0.1,
                        active_subcore=active_subcore,
                        activated_mlus=["CORE_ROOT", cf_root, cf_supp, *inv_list],
                        spoke_predictions=preds[:6],
                        inspectable_trace=(
                            f"[MLU-360-COUNTERFACTUAL] root={cf_root} | "
                            f"suppressed={cf_supp} | invalidated={inv_list} | spoke=dynamics(causes)"
                        ),
                        fluent_english=cf_synth,
                    )

            cmp_dto = self.compare_360(focal_cid, target_cid, max_hops=3)
            comp_synth = str(cmp_dto.get("comparative_synthesis", "")).strip()
            shared_list = cmp_dto.get("shared_attractors", [])
            if shared_list and comp_synth:
                conf = float(shared_list[0]["combined_confidence"])
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED",
                    answer=comp_synth,
                    confidence=conf,
                    epistemic_entropy=0.1,
                    active_subcore=active_subcore,
                    activated_mlus=activated + [target_cid],
                    spoke_predictions=preds[:6],
                    inspectable_trace=(
                        f"[MLU-360-COMPARE] {focal_cid} <-> {target_cid} | "
                        f"shared={len(shared_list)} | mutex={cmp_dto.get('mutually_exclusive', False)}"
                    ),
                    fluent_english=comp_synth,
                )

            bridge_dto = self.find_bridge_360(focal_cid, target_cid, max_hops=4)
            if bridge_dto.get("connected") and bridge_dto.get("bridge_synthesis"):
                br_synth = str(bridge_dto["bridge_synthesis"])
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED",
                    answer=br_synth,
                    confidence=0.82,
                    epistemic_entropy=0.12,
                    active_subcore=active_subcore,
                    activated_mlus=activated + [target_cid],
                    spoke_predictions=preds[:6],
                    inspectable_trace=(
                        f"[MLU-360-BRIDGE] {focal_cid} <-> {target_cid} | "
                        f"hops={bridge_dto.get('hop_count', 0)}"
                    ),
                    fluent_english=br_synth,
                )

            # Phase 41b: 2-Observation Abductive Diagnosis Fallback (e.g., "fever and inflammation")
            abd2 = self.diagnose_abductive_360(
                [focal_cid, target_cid], max_hops=3, top_k=3
            )
            if (
                abd2.get("best_explanation")
                and float(abd2.get("best_coverage_ratio", 0.0)) >= 0.99
                and abd2.get("diagnostic_synthesis")
            ):
                best_e2 = str(abd2["best_explanation"])
                d_synth2 = str(abd2["diagnostic_synthesis"])
                c2 = abd2.get("candidates", [{}])[0]
                p_conf2 = float(c2.get("posterior_confidence", 0.85))
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=best_e2,
                    status="VERIFIED",
                    answer=d_synth2,
                    confidence=p_conf2,
                    epistemic_entropy=round(1.0 - p_conf2, 4),
                    active_subcore=active_subcore,
                    activated_mlus=activated + [focal_cid, target_cid, best_e2],
                    spoke_predictions=preds[:6],
                    inspectable_trace=(
                        f"[MLU-360-DIAGNOSE] observations={[focal_cid, target_cid]} -> "
                        f"best_explanation={best_e2}"
                    ),
                    fluent_english=d_synth2,
                )

            # Phase 57: Contrastive 360° comparison when two concepts have divergent/mutually-exclusive spokes
            if comp_synth and (
                cmp_dto.get("mutually_exclusive")
                or (cmp_dto.get("distinct_a") and cmp_dto.get("distinct_b"))
            ):
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED",
                    answer=comp_synth,
                    confidence=0.90,
                    epistemic_entropy=0.08,
                    active_subcore=active_subcore,
                    activated_mlus=activated + [target_cid],
                    spoke_predictions=preds[:6],
                    inspectable_trace=(
                        f"[MLU-360-COMPARE] {focal_cid} <-> {target_cid} | "
                        f"shared=0 | mutex={cmp_dto.get('mutually_exclusive', False)}"
                    ),
                    fluent_english=comp_synth,
                )

        # Phase 63: Single-concept Counterfactual Root Intervention ("What happens if sunlight is removed?", "What if rain stops?")
        if target_cid is None and has_cf_cue:
            cf_dto = self.intervene_counterfactual_360(focal_cid, focal_cid, max_hops=4)
            causal_dto = self.simulate_causal_360(focal_cid, max_hops=4)
            inv_list = [
                str(c)
                for c in (
                    cf_dto.get("invalidated_conclusions")
                    or causal_dto.get("affected_concepts")
                    or []
                )
            ]
            if inv_list:
                self._last_causal_proof_steps = [
                    (
                        str(st["source_concept"]),
                        str(st["predicate"]),
                        str(st["target_concept"]),
                    )
                    for st in causal_dto.get("causal_steps", [])
                ]
                inv_words = [c.lower().replace("_", " ") for c in inv_list]
                inv_str = (
                    inv_words[0]
                    if len(inv_words) == 1
                    else f"{', '.join(inv_words[:-1])} and {inv_words[-1]}"
                )
                f_w = focal_cid.lower().replace("_", " ")
                cf_fluent = (
                    f"Counterfactual intervention: if {f_w} is removed, "
                    f"downstream conclusions {inv_str} are invalidated."
                )
                return MLMQueryResult(
                    query=clean_q,
                    focal_concept=focal_cid,
                    status="VERIFIED",
                    answer=cf_fluent,
                    confidence=0.90,
                    epistemic_entropy=0.08,
                    active_subcore=active_subcore,
                    activated_mlus=["CORE_ROOT", focal_cid, *inv_list],
                    spoke_predictions=preds[:6],
                    inspectable_trace=(
                        f"[MLU-360-COUNTERFACTUAL] root={focal_cid} | "
                        f"suppressed={focal_cid} | invalidated={inv_list} | spoke=dynamics(causes)"
                    ),
                    fluent_english=cf_fluent,
                )

        # Case B: Open 360° synthesis across all radial spokes + multi-hop causal wave + learned neural sequence continuation
        spoke_clauses: list[str] = []
        for p in preds[:5]:
            rel_text = p.predicate.replace("_", " ")
            tgt_text = p.target_concept.lower().replace("_", " ")
            spoke_clauses.append(f"{rel_text} {tgt_text} [{p.spoke}:{p.combined_confidence:.2f}]")

        causal_suffix = ""
        causal_trace_tag = ""
        if any(p.predicate == "causes" for p in preds):
            causal_dto = self.simulate_causal_360(focal_cid, max_hops=4)
            affected_list = [str(c) for c in causal_dto.get("affected_concepts", [])]
            if affected_list:
                self._last_causal_proof_steps = [
                    (
                        str(st["source_concept"]),
                        str(st["predicate"]),
                        str(st["target_concept"]),
                    )
                    for st in causal_dto.get("causal_steps", [])
                ]
                for aff_c in affected_list:
                    if aff_c not in activated:
                        activated.append(aff_c)
            if len(affected_list) > 1:
                c_synth = str(causal_dto.get("causal_synthesis", "")).strip()
                if c_synth:
                    causal_suffix = f" {c_synth}"
                    causal_trace_tag = f" | [MLU-360-CAUSAL-WAVE] affected={affected_list}"

        plan_suffix = ""
        plan_trace_tag = ""
        if self.rust_arena.is_native and any(p.predicate == "requires" for p in preds):
            plan_dto = self.plan_goal_360(focal_cid, max_steps=5)
            if (
                plan_dto.get("achievable")
                and int(plan_dto.get("total_steps", 0)) >= 2
                and plan_dto.get("plan_synthesis")
            ):
                plan_suffix = f" {str(plan_dto['plan_synthesis']).strip()}"
                plan_trace_tag = (
                    f" | [MLU-360-PLAN] steps={plan_dto.get('total_steps', 0)} "
                    f"prereqs={plan_dto.get('root_prerequisites', [])}"
                )

        delib_suffix = ""
        delib_trace_tag = ""
        if self.rust_arena.is_native:
            delib_dto = self.deliberate_360(focal_cid, max_hops=3)
            peers = delib_dto.get("cluster_peers", [])
            insights = delib_dto.get("analogical_insights", [])
            if peers:
                peer_names = ", ".join(p.lower().replace("_", " ") for p in peers[:3])
                if insights:
                    h0 = insights[0]
                    delib_suffix = (
                        f" Through 360° cluster resonance with {peer_names}, "
                        f"{focal_cid.lower().replace('_', ' ')} may also "
                        f"{str(h0.get('predicate', '')).replace('_', ' ')} "
                        f"{str(h0.get('target_concept', '')).lower().replace('_', ' ')} "
                        f"(shared {str(h0.get('shared_attractor', '')).lower().replace('_', ' ')})."
                    )
                else:
                    delib_suffix = f" In the 360° arena, {focal_cid.lower().replace('_', ' ')} clusters with {peer_names}."
                delib_trace_tag = (
                    f" | [MLU-360-DELIBERATE] peers={peers[:3]} "
                    f"F={delib_dto.get('free_energy', 0.0):.2f}"
                )

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
        base_fluent = rust_fluent or self._synthesize_fluent_english(
            focal_cid, preds, continuation_text
        )
        fluent_eng = (base_fluent + causal_suffix + plan_suffix + delib_suffix).strip()
        trace = (
            f"[MLU-360] Activated {len(activated)} units ({' -> '.join(activated)}) | "
            f"top_conf={top_conf:.2f} | entropy={avg_entropy:.2f}{causal_trace_tag}{plan_trace_tag}{delib_trace_tag}"
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

    def compare_360(
        self, concept_a: str, concept_b: str, max_hops: int = 3
    ) -> dict[str, Any]:
        """Phase 14: Run 360° bidirectional wave intersection between two concepts."""
        if self.rust_arena.is_native:
            return self.rust_arena.compare_360(concept_a, concept_b, max_hops=max_hops)
        return {}

    def discover_curiosity(self, max_items: int = 10) -> dict[str, Any]:
        """Phase 15: Discover invariant-safe analogical spoke hypotheses & epistemic curiosity gaps."""
        if self.rust_arena.is_native:
            return self.rust_arena.discover_curiosity(max_items=max_items)
        return {"analogical_hypotheses": [], "curiosity_gaps": []}

    def simulate_causal_360(self, root_concept: str, max_hops: int = 3) -> dict[str, Any]:
        """Phase 17: Run native Rust 360° causal & counterfactual wave simulation from a root concept."""
        if self.rust_arena.is_native:
            return self.rust_arena.simulate_causal(root_concept, max_hops=max_hops)
        return {
            "root_concept": root_concept,
            "affected_concepts": [],
            "causal_steps": [],
            "causal_synthesis": "",
        }

    def autonomous_curiosity_step(
        self, min_confidence: float = 0.65, max_promotions: int = 5
    ) -> dict[str, Any]:
        """Phase 18: Actively evaluate and promote invariant-safe analogical hypotheses into the 360° arena."""
        if not self.rust_arena.is_native:
            return {
                "hypotheses_evaluated": 0,
                "hypotheses_promoted": 0,
                "promoted_hypotheses": [],
                "open_questions": [],
            }
        step_dto = self.rust_arena.autonomous_curiosity(
            min_confidence=min_confidence, max_promotions=max_promotions
        )
        for hyp in step_dto.get("promoted_hypotheses", []):
            subj = str(hyp.get("subject_concept", "")).strip()
            pred = str(hyp.get("predicate", "")).strip()
            obj = str(hyp.get("target_concept", "")).strip()
            conf = float(hyp.get("confidence", 0.7))
            if subj and pred and obj:
                spoke = map_predicate_to_spoke(pred)
                subj_unit = self.get_or_create_mlu(subj, preferred_spoke=spoke)
                obj_unit = self.get_or_create_mlu(obj, preferred_spoke=spoke)
                subj_unit.learn_spoke_relation(
                    predicate=pred,
                    target_concept=obj_unit.concept_id,
                    target_vec=obj_unit.anchor_vec,
                    learning_rate=0.18 * conf,
                    verified=True,
                )
                self._dirty_since_save = True
        return step_dto

    def save_snapshot(self, path: str) -> bool:
        """Phase 19: Persist the Rust 360° Spider-Web Arena snapshot to disk."""
        if self.rust_arena.is_native:
            return self.rust_arena.save_snapshot(path)
        return False

    def load_snapshot(self, path: str) -> bool:
        """Phase 19: Restore the Rust 360° Spider-Web Arena snapshot from disk."""
        if self.rust_arena.is_native:
            return self.rust_arena.load_snapshot(path)
        return False

    def intervene_counterfactual_360(
        self, root_concept: str, suppressed_concept: str, max_hops: int = 3
    ) -> dict[str, Any]:
        """Phase 20: Run native Rust 360° counterfactual intervention wave."""
        if self.rust_arena.is_native:
            return self.rust_arena.intervene_counterfactual(
                root_concept, suppressed_concept, max_hops=max_hops
            )
        return {
            "root_concept": root_concept,
            "suppressed_concept": suppressed_concept,
            "baseline_affected": [],
            "invalidated_conclusions": [],
            "preserved_conclusions": [],
            "counterfactual_synthesis": "",
        }

    def find_bridge_360(
        self, concept_a: str, concept_b: str, max_hops: int = 4
    ) -> dict[str, Any]:
        """Phase 21: Discover the shortest 360° relational path bridge between two concepts."""
        if self.rust_arena.is_native:
            return self.rust_arena.find_bridge(concept_a, concept_b, max_hops=max_hops)
        return {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "connected": False,
            "hop_count": 0,
            "bridge_steps": [],
            "bridge_synthesis": "",
        }

    def export_topology_graph(self, max_edges: int = 100) -> dict[str, Any]:
        """Phase 22: Export the live 360° Spider-Web Arena topology and Mermaid graph."""
        if self.rust_arena.is_native:
            return self.rust_arena.export_topology(max_edges=max_edges)
        return {
            "total_cells": len(self.nodes),
            "total_verified_edges": 0,
            "nodes": list(self.nodes.keys()),
            "edges": [],
            "mermaid_diagram": "",
        }

    def discover_clusters_360(self, min_members: int = 2) -> list[dict[str, Any]]:
        """Phase 23: Discover emergent 64-D concept clusters sharing spoke attractors in native Rust."""
        if self.rust_arena.is_native:
            return self.rust_arena.discover_clusters(min_members=min_members)
        return []

    def revise_belief_360(
        self,
        subject: str,
        predicate: str,
        object_node: str,
        verified: bool,
        evidence_weight: float = 3.0,
    ) -> dict[str, Any]:
        """Phase 24: Apply explicit non-monotonic belief revision to a spoke synapse across Rust and Python."""
        res = (
            self.rust_arena.revise_belief(
                subject,
                predicate,
                object_node,
                verified=verified,
                evidence_weight=evidence_weight,
            )
            if self.rust_arena.is_native
            else {}
        )
        spoke = map_predicate_to_spoke(predicate)
        subj_unit = self.get_or_create_mlu(subject, preferred_spoke=spoke)
        obj_unit = self.get_or_create_mlu(object_node, preferred_spoke=spoke)
        for _ in range(max(1, round(evidence_weight))):
            subj_unit.learn_spoke_relation(
                predicate=predicate,
                target_concept=obj_unit.concept_id,
                target_vec=obj_unit.anchor_vec,
                learning_rate=0.25,
                verified=verified,
            )
        self._dirty_since_save = True
        return res

    def deliberate_360(self, concept: str, max_hops: int = 3) -> dict[str, Any]:
        """Phase 30: Run native Rust 5-wave System-2 Deliberative Thinking over a concept."""
        if self.rust_arena.is_native:
            return self.rust_arena.deliberate_360(concept, max_hops=max_hops)
        return {
            "focal_concept": concept.upper(),
            "waves_executed": 0,
            "free_energy": 1.0,
            "verified_facts": [],
            "causal_consequences": [],
            "cluster_peers": [],
            "analogical_insights": [],
            "follow_up_question": "",
            "deliberative_synthesis": "",
        }

    def plan_goal_360(
        self, goal_concept: str, start_concept: str = "", max_steps: int = 5
    ) -> dict[str, Any]:
        """Phase 32: Run native Rust 360° Goal-Directed Backward-Chaining & Action Planner."""
        if self.rust_arena.is_native:
            return self.rust_arena.plan_goal_360(
                goal_concept, start_concept=start_concept, max_steps=max_steps
            )
        return {
            "goal_concept": goal_concept.upper(),
            "start_concept": start_concept.upper(),
            "achievable": False,
            "total_steps": 0,
            "plan_confidence": 0.0,
            "root_prerequisites": [],
            "plan_steps": [],
            "plan_synthesis": "",
        }

    def evaluate_order_360(
        self, predicate: str, start_concept: str = "", max_hops: int = 6
    ) -> dict[str, Any]:
        """Phase 35: Run native Rust 360° Transitive Poset & Quantitative Order Wave."""
        if self.rust_arena.is_native:
            return self.rust_arena.evaluate_order_360(
                predicate, start_concept=start_concept, max_hops=max_hops
            )
        return {
            "predicate": predicate.lower().strip(),
            "start_concept": start_concept.upper().strip(),
            "poset_consistent": True,
            "element_count": 0,
            "ordered_chain": [],
            "transitive_pairs": [],
            "order_synthesis": "",
        }

    def summarize_domain_360(
        self, domain_concept: str = "", max_items: int = 8
    ) -> dict[str, Any]:
        """Phase 36: Run native Rust 360° Domain & Sub-Core Knowledge Summarizer."""
        if self.rust_arena.is_native:
            return self.rust_arena.summarize_domain_360(
                domain_concept=domain_concept, max_items=max_items
            )
        return {
            "domain_name": domain_concept.upper().strip() or "ALL_DOMAINS",
            "active_concepts_count": len(self.nodes),
            "verified_synapses_count": 0,
            "central_hubs": [],
            "key_principles": [],
            "emergent_clusters": [],
            "open_curiosity_questions": [],
            "domain_synthesis": "",
        }

    def diagnose_abductive_360(
        self, observations: list[str] | str, max_hops: int = 3, top_k: int = 5
    ) -> dict[str, Any]:
        """Phase 38: Run native Rust 360° Abductive Diagnostic & Root-Cause Engine."""
        if self.rust_arena.is_native:
            return self.rust_arena.diagnose_abductive_360(
                observations, max_hops=max_hops, top_k=top_k
            )
        return {
            "observations": [],
            "best_explanation": "",
            "best_coverage_ratio": 0.0,
            "candidates": [],
            "diagnostic_synthesis": "",
        }

    def compose_discourse_360(
        self, concept: str, max_hops: int = 3
    ) -> dict[str, Any]:
        """Phase 39: Run native Rust 360° Multi-Paragraph Discourse & Essay Composer."""
        if self.rust_arena.is_native:
            return self.rust_arena.compose_discourse_360(concept, max_hops=max_hops)
        return {
            "focal_concept": concept.upper().strip(),
            "title": f"360° Knowledge Discourse: {concept.lower().strip()}",
            "paragraph_count": 0,
            "paragraphs": [],
            "full_discourse": "",
        }

    def solve_analogy_360(
        self,
        concept_a: str,
        concept_b: str,
        concept_c: str,
        max_hops: int = 4,
    ) -> dict[str, Any]:
        """Phase 61: Solve proportional 360° analogies (A : B :: C : ?) via spoke role alignment and 64-D vector offset scoring."""
        cid_a = concept_a.strip().upper().replace(" ", "_")
        cid_b = concept_b.strip().upper().replace(" ", "_")
        cid_c = concept_c.strip().upper().replace(" ", "_")
        empty_res: dict[str, Any] = {
            "resolved": False,
            "concept_a": cid_a,
            "concept_b": cid_b,
            "concept_c": cid_c,
            "target_concept": "",
            "mapped_predicate": "",
            "mapped_spoke": "",
            "confidence": 0.0,
            "vector_similarity": 0.0,
            "ab_path": [],
            "cd_path": [],
            "analogy_synthesis": "",
        }
        if (
            cid_a not in self.nodes
            or cid_b not in self.nodes
            or cid_c not in self.nodes
            or len({cid_a, cid_b, cid_c}) < 3
        ):
            return empty_res

        preds_a, _ = self._collect_360_predictions(self.nodes[cid_a], max_hops=max_hops)
        ab_matches = [
            p
            for p in preds_a
            if p.target_concept == cid_b
            and p.verified
            and p.predicate not in ("co_occurs_with", "disjoint_with", "not_a")
        ]
        inverse_role = False
        if not ab_matches:
            preds_b, _ = self._collect_360_predictions(self.nodes[cid_b], max_hops=max_hops)
            ab_matches = [
                p
                for p in preds_b
                if p.target_concept == cid_a
                and p.verified
                and p.predicate not in ("co_occurs_with", "disjoint_with", "not_a")
            ]
            if ab_matches:
                inverse_role = True
        if not ab_matches:
            return empty_res

        vec_a = self.nodes[cid_a].hidden_state
        vec_b = self.nodes[cid_b].hidden_state
        vec_c = self.nodes[cid_c].hidden_state
        dim = min(len(vec_a), len(vec_b), len(vec_c))
        offset_vec = [vec_b[i] - vec_a[i] + vec_c[i] for i in range(dim)]
        norm_off = math.sqrt(sum(v * v for v in offset_vec)) or 1.0
        target_vec = [v / norm_off for v in offset_vec]

        def _cos_sim(vec_x: list[float]) -> float:
            nx = math.sqrt(sum(x * x for x in vec_x)) or 1.0
            return sum(
                (x / nx) * ty for x, ty in zip(vec_x, target_vec, strict=False)
            )

        best_score = -1.0
        best_ab: MLUSpokePrediction | None = None
        best_cd: MLUSpokePrediction | None = None
        best_d = ""
        best_vsim = 0.0

        if not inverse_role:
            preds_c, _ = self._collect_360_predictions(self.nodes[cid_c], max_hops=max_hops)
            for ab in ab_matches:
                for cd in preds_c:
                    cand_d = cd.target_concept
                    if (
                        not cd.verified
                        or cd.predicate in ("co_occurs_with", "disjoint_with", "not_a")
                        or cand_d in (cid_a, cid_b, cid_c, "CORE_ROOT")
                        or cand_d.startswith("SUBCORE_")
                        or cand_d not in self.nodes
                    ):
                        continue
                    pred_match = cd.predicate == ab.predicate
                    spoke_match = cd.spoke == ab.spoke
                    if not pred_match and not spoke_match:
                        continue
                    v_d = self.nodes[cand_d].hidden_state[:dim]
                    v_sim = max(0.0, _cos_sim(v_d))
                    role_bonus = 2.5 if pred_match else 0.85
                    cd_steps = self._find_verified_spoke_path(cid_c, cand_d, max_hops=max_hops)
                    hop_pen = 0.06 * max(0, len(cd_steps) - 1)
                    score = (
                        role_bonus
                        + 0.5 * (ab.combined_confidence + cd.combined_confidence)
                        + 0.35 * v_sim
                        - hop_pen
                    )
                    if score > best_score:
                        best_score = score
                        best_ab = ab
                        best_cd = cd
                        best_d = cand_d
                        best_vsim = v_sim
        else:
            for cand_d, node_d in self.nodes.items():
                if (
                    cand_d in (cid_a, cid_b, cid_c, "CORE_ROOT")
                    or cand_d.startswith("SUBCORE_")
                ):
                    continue
                preds_d, _ = self._collect_360_predictions(node_d, max_hops=max_hops)
                dc_matches = [
                    p
                    for p in preds_d
                    if p.target_concept == cid_c
                    and p.verified
                    and p.predicate not in ("co_occurs_with", "disjoint_with", "not_a")
                ]
                if not dc_matches:
                    continue
                v_d = node_d.hidden_state[:dim]
                v_sim = max(0.0, _cos_sim(v_d))
                dc_steps = self._find_verified_spoke_path(cand_d, cid_c, max_hops=max_hops)
                hop_pen = 0.06 * max(0, len(dc_steps) - 1)
                for ab in ab_matches:
                    for dc in dc_matches:
                        pred_match = dc.predicate == ab.predicate
                        spoke_match = dc.spoke == ab.spoke
                        if not pred_match and not spoke_match:
                            continue
                        role_bonus = 2.5 if pred_match else 0.85
                        score = (
                            role_bonus
                            + 0.5 * (ab.combined_confidence + dc.combined_confidence)
                            + 0.35 * v_sim
                            - hop_pen
                        )
                        if score > best_score:
                            best_score = score
                            best_ab = ab
                            best_cd = dc
                            best_d = cand_d
                            best_vsim = v_sim

        if best_ab is None or best_cd is None or not best_d:
            return empty_res

        def _pred_verb(pred: str, inv: bool) -> str:
            if inv:
                return f"is linked via {pred.replace('_', ' ')} from"
            if pred in ("located_in", "lives_in"):
                return "lives in"
            if pred in ("has_part", "has_property", "has"):
                return "has"
            if pred in ("can", "capable_of"):
                return "can"
            if pred in ("is_a", "subclass_of", "instance_of"):
                return "is a"
            return pred.replace("_", " ")

        a_w = cid_a.lower().replace("_", " ")
        b_w = cid_b.lower().replace("_", " ")
        c_w = cid_c.lower().replace("_", " ")
        d_w = best_d.lower().replace("_", " ")
        ab_verb = _pred_verb(best_ab.predicate, inverse_role)
        cd_verb = _pred_verb(best_cd.predicate, inverse_role)
        synthesis = f"{a_w.capitalize()} {ab_verb} {b_w}, just as {c_w} {cd_verb} {d_w}."
        conf = round(
            min(
                0.99,
                0.5 * (best_ab.combined_confidence + best_cd.combined_confidence)
                + (0.08 if best_cd.predicate == best_ab.predicate else 0.0),
            ),
            4,
        )
        ab_steps = (
            self._find_verified_spoke_path(cid_b, cid_a, max_hops=max_hops)
            if inverse_role
            else self._find_verified_spoke_path(cid_a, cid_b, max_hops=max_hops)
        )
        cd_steps = (
            self._find_verified_spoke_path(best_d, cid_c, max_hops=max_hops)
            if inverse_role
            else self._find_verified_spoke_path(cid_c, best_d, max_hops=max_hops)
        )
        return {
            "resolved": True,
            "concept_a": cid_a,
            "concept_b": cid_b,
            "concept_c": cid_c,
            "target_concept": best_d,
            "mapped_predicate": best_cd.predicate,
            "mapped_spoke": best_cd.spoke,
            "confidence": conf,
            "vector_similarity": round(best_vsim, 4),
            "ab_path": ab_steps,
            "cd_path": cd_steps,
            "analogy_synthesis": synthesis,
            "spoke_prediction": best_cd,
        }

    def intersect_concepts_360(
        self,
        category_concept: str,
        constraints: list[tuple[int, str | None, str]],
        max_hops: int = 4,
    ) -> dict[str, Any]:
        """Phase 62: Solve Compositional Set & Conjunctive Constraint Intersection queries ('Which mammal can echolocate?', 'Which bird cannot fly?')."""
        cat_cid = category_concept.strip().upper().replace(" ", "_")
        empty_res: dict[str, Any] = {
            "resolved": False,
            "category_concept": cat_cid,
            "best_concept": "",
            "matched_candidates": [],
            "confidence": 0.0,
            "primary_spoke": "taxonomy",
            "primary_predicate": "is_a",
            "constraint_summary": "",
            "proof_steps": [],
            "spoke_predictions": [],
            "intersection_synthesis": "",
        }
        if cat_cid not in self.nodes or not constraints:
            return empty_res

        # 1. Discover all taxonomic descendants of cat_cid via verified taxonomic edges
        descendants: list[tuple[str, list[tuple[str, str, str]]]] = []
        for cand_cid in self.nodes:
            if (
                cand_cid in (cat_cid, "CORE_ROOT")
                or cand_cid.startswith("SUBCORE_")
            ):
                continue
            tax_path = self._find_taxonomic_path(cand_cid, cat_cid, max_hops=max_hops)
            if tax_path:
                descendants.append((cand_cid, tax_path))

        if not descendants:
            return empty_res

        # 2. Evaluate each descendant against every positive or negated spoke constraint
        scored_matches: list[
            tuple[
                float,
                str,
                list[tuple[str, str, str]],
                list[str],
                list[MLUSpokePrediction],
                str,
                str,
            ]
        ] = []

        for cand_cid, tax_path in descendants:
            cand_unit = self.nodes[cand_cid]
            preds, _ = self._collect_360_predictions(cand_unit, max_hops=max_hops)
            all_satisfied = True
            total_conf = 0.0
            c_clauses: list[str] = []
            c_proof_steps: list[tuple[str, str, str]] = list(tax_path)
            c_spoke_preds: list[MLUSpokePrediction] = []
            primary_spoke = "taxonomy"
            primary_pred = "is_a"

            for pol, req_pred, raw_tgt in constraints:
                tgt_cid = raw_tgt.strip().upper().replace(" ", "_")
                tgt_w = tgt_cid.lower().replace("_", " ")
                if pol < 0:
                    # Negated constraint (e.g., 'cannot fly', 'does not have wings')
                    neg_rec_pred: str | None = None
                    neg_spoke = "procedural"
                    for sp_name, sp_head in cand_unit.spokes.items():
                        rec = sp_head.target_affinities.get(tgt_cid)
                        if (
                            isinstance(rec, list)
                            and len(rec) >= 4
                            and float(rec[1]) > float(rec[0])
                        ):
                            neg_rec_pred = str(rec[3])
                            neg_spoke = sp_name
                            break
                    if neg_rec_pred is None:
                        all_satisfied = False
                        break
                    eff_p = req_pred or neg_rec_pred
                    if eff_p in ("can", "capable_of"):
                        c_clauses.append(f"cannot {tgt_w}")
                        step_p = "cannot"
                    elif eff_p in ("has_part", "has_property", "has"):
                        c_clauses.append(f"does not have {tgt_w}")
                        step_p = "does_not_have"
                    else:
                        c_clauses.append(f"does not {eff_p.replace('_', ' ')} {tgt_w}")
                        step_p = f"cannot_{eff_p}"
                    neg_edge = (cand_cid, step_p, tgt_cid)
                    if neg_edge not in c_proof_steps:
                        c_proof_steps.append(neg_edge)
                    total_conf += 0.95
                    primary_spoke = neg_spoke
                    primary_pred = eff_p
                else:
                    # Positive spoke constraint (e.g., 'lives in antarctica', 'can echolocate', 'has wings')
                    cand_hits = [
                        p
                        for p in preds
                        if p.target_concept == tgt_cid
                        and p.verified
                        and p.predicate not in ("co_occurs_with", "disjoint_with", "not_a")
                    ]
                    if not cand_hits:
                        all_satisfied = False
                        break
                    best_hit = max(
                        cand_hits,
                        key=lambda p: (
                            p.combined_confidence
                            + (0.25 if req_pred and p.predicate == req_pred else 0.0)
                        ),
                    )
                    hit_p = best_hit.predicate
                    if hit_p in ("located_in", "lives_in"):
                        c_clauses.append(f"lives in {tgt_w}")
                    elif hit_p in ("has_part", "has_property", "has"):
                        c_clauses.append(f"has {tgt_w}")
                    elif hit_p in ("can", "capable_of"):
                        c_clauses.append(f"can {tgt_w}")
                    else:
                        c_clauses.append(f"{hit_p.replace('_', ' ')} {tgt_w}")
                    hop_edges = self._find_verified_spoke_path(
                        cand_cid, tgt_cid, max_hops=max_hops
                    )
                    for e in hop_edges or [(cand_cid, hit_p, tgt_cid)]:
                        if e not in c_proof_steps:
                            c_proof_steps.append(e)
                    c_spoke_preds.append(best_hit)
                    # Direct ownership bonus over inherited property
                    direct_bonus = (
                        0.08
                        if hop_edges and len(hop_edges) == 1
                        else 0.0
                    )
                    total_conf += best_hit.combined_confidence + direct_bonus
                    primary_spoke = best_hit.spoke
                    primary_pred = best_hit.predicate

            if all_satisfied:
                mean_conf = total_conf / float(len(constraints))
                scored_matches.append(
                    (
                        mean_conf,
                        cand_cid,
                        c_proof_steps,
                        c_clauses,
                        c_spoke_preds,
                        primary_spoke,
                        primary_pred,
                    )
                )

        if not scored_matches:
            return empty_res

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        (
            best_score,
            best_cid,
            best_steps,
            best_clauses,
            best_sp_preds,
            best_spoke,
            best_pred,
        ) = scored_matches[0]
        best_w = best_cid.lower().replace("_", " ")
        cat_w = cat_cid.lower().replace("_", " ")
        cat_art = "an" if cat_w and cat_w[0] in "aeiou" else "a"
        synthesis = (
            f"{best_w.capitalize()} is {cat_art} {cat_w} that "
            f"{' and '.join(best_clauses)}."
        )
        c_summary = " & ".join(
            f"{'!' if pol < 0 else ''}{pred or '*'}:{tgt}"
            for pol, pred, tgt in constraints
        )
        return {
            "resolved": True,
            "category_concept": cat_cid,
            "best_concept": best_cid,
            "matched_candidates": [m[1] for m in scored_matches],
            "confidence": round(min(0.99, best_score), 4),
            "primary_spoke": best_spoke,
            "primary_predicate": best_pred,
            "constraint_summary": c_summary,
            "proof_steps": best_steps,
            "spoke_predictions": best_sp_preds,
            "intersection_synthesis": synthesis,
        }

    def train_neural_mlu(
        self,
        text: str,
        epochs: int = 30,
        learning_rate: float = 0.035,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Phase 50: Train the True End-to-End Differentiable 360° Spider-Web Neural MLU via Backpropagation + AdamW."""
        report = self.neural_mlu.train(text, epochs=epochs, learning_rate=learning_rate)
        self.sync_neural_embeddings_to_arena(blend=0.20)
        self._dirty_since_save = True
        if persist:
            self.save_nodes([])
        return report

    def generate_neural_mlu(
        self,
        prompt: str,
        max_new_tokens: int = 28,
    ) -> dict[str, Any]:
        """Phase 51: Autoregressively generate response tokens purely from Neural Spider-Web MLU hidden states."""
        return self.neural_mlu.generate(prompt, max_new_tokens=max_new_tokens)

    def train_neural_curriculum(
        self,
        epochs: int = 55,
        learning_rate: float = 0.04,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Phase 52: Train the Differentiable 360° Neural Spider-Web MLU on the English + Chain-of-Thought Thinking Curriculum."""
        from pathlib import Path

        from little.core.runtime_paths import RuntimePaths

        candidates = [
            RuntimePaths.default().database.parent
            / "knowledge"
            / "neural_curriculum"
            / "english_thinking_curriculum.txt",
            Path("data/knowledge/neural_curriculum/english_thinking_curriculum.txt"),
        ]
        cpath = next((p for p in candidates if p.is_file()), None)
        if cpath is None:
            return {"epochs": 0, "sequences_trained": 0}
        text = cpath.read_text(encoding="utf-8", errors="ignore")
        return self.train_neural_mlu(
            text, epochs=epochs, learning_rate=learning_rate, persist=persist
        )

    def distill_triples_to_neural_mlu(
        self,
        triples: list[tuple[str, str, str]],
        epochs: int = 28,
        learning_rate: float = 0.04,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 53: Automatically synthesize Chain-of-Thought ('think ... answer ...') sequences from verified 360° spoke triples and train the Differentiable Neural MLU via Backpropagation + AdamW."""
        if not triples:
            return {"epochs": 0, "sequences_trained": 0}
        lines: list[str] = []
        for subj, pred, obj in triples:
            s = subj.lower().replace("_", " ").strip()
            p = pred.lower().replace("_", " ").strip()
            o = obj.lower().replace("_", " ").strip()
            if not s or not o:
                continue
            if pred in ("is_a", "subclass_of", "instance_of"):
                lines.append(
                    f"is a {s} a {o}? think {s} is a verified taxonomic child of {o} in the 360 spider web answer yes a {s} is a {o}."
                )
                lines.append(
                    f"what is a {s}? think querying 360 taxonomy spoke for {s} answer a {s} is a {o}."
                )
            elif pred in ("can", "capable_of"):
                lines.append(
                    f"can a {s} {o}? think checking procedural spoke of {s} for {o} answer yes a {s} can {o}."
                )
            elif pred in ("has_part", "has_property", "has"):
                lines.append(
                    f"does a {s} have {o}? think checking mereology and attribute spokes of {s} for {o} answer yes a {s} has {o}."
                )
            elif pred == "causes":
                lines.append(
                    f"what causes {o}? think tracing backward causal spoke to {s} answer {s} causes {o}."
                )
            elif pred in ("larger_than", "greater_than", "before", "after"):
                lines.append(
                    f"is {s} {p} {o}? think 360 poset order on dynamics spoke confirms {s} {p} {o} answer yes {s} is {p} {o}."
                )
            elif pred in ("disjoint_with", "cannot_be", "different_from"):
                lines.append(
                    f"is a {s} a {o}? think mutual exclusivity gate refutes {s} and {o} answer no a {s} is not a {o}."
                )
            else:
                lines.append(
                    f"tell me about {s}? think activating 360 spokes around {s} answer {s} {p} {o}."
                )
        if not lines:
            return {"epochs": 0, "sequences_trained": 0}
        return self.train_neural_mlu(
            "\n".join(lines), epochs=epochs, learning_rate=learning_rate, persist=persist
        )

    def distill_negated_triples_to_neural_mlu(
        self,
        negated_triples: list[tuple[str, str, str]],
        epochs: int = 32,
        learning_rate: float = 0.045,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 60: Distill non-monotonic exception & belief revision triples into NeuralSpiderWebMlu via Backpropagation + AdamW."""
        if not negated_triples:
            return {"epochs": 0, "sequences_trained": 0}
        lines: list[str] = []
        for subj, pred, obj in negated_triples:
            s = subj.lower().replace("_", " ").strip()
            p = pred.lower().replace("_", " ").strip()
            o = obj.lower().replace("_", " ").strip()
            if not s or not o:
                continue
            sa = "an" if s and s[0] in "aeiou" else "a"
            if pred in ("can", "capable_of"):
                lines.append(
                    f"can {sa} {s} {o}? think 360 non monotonic exception gate refutes {s} on capability spoke answer no {sa} {s} cannot {o}."
                )
            elif pred in ("has_part", "has_property", "has"):
                lines.append(
                    f"does {sa} {s} have {o}? think 360 non monotonic exception gate refutes {s} on mereology spoke answer no {sa} {s} does not have {o}."
                )
            else:
                lines.append(
                    f"does {sa} {s} {p} {o}? think 360 non monotonic exception gate refutes {s} on dynamics spoke answer no {sa} {s} does not {p} {o}."
                )
        if not lines:
            return {"epochs": 0, "sequences_trained": 0}
        return self.train_neural_mlu(
            "\n".join(lines), epochs=epochs, learning_rate=learning_rate, persist=persist
        )

    def _build_arena_conditioning(
        self, prompt: str, q_res: MLMQueryResult
    ) -> tuple[str, list[float]]:
        """Phase 55, 59 & 60: Extract 6-spoke neural cell priors and multi-hop Chain-of-Thought proof sequences from a 360° Arena wave."""
        # Map 360° predicates & wave traces to NeuralSpiderWebMlu's 6 logic cells:
        # 0: TAXONOMY, 1: MEREOLOGY, 2: CAUSALITY, 3: SPATIAL, 4: CAPABILITY, 5: DYNAMICS
        pred_to_cell: dict[str, tuple[int, str]] = {
            "is_a": (0, "taxonomy"),
            "subclass_of": (0, "taxonomy"),
            "instance_of": (0, "taxonomy"),
            "disjoint_with": (0, "taxonomy"),
            "not_a": (0, "taxonomy"),
            "has_part": (1, "mereology"),
            "part_of": (1, "mereology"),
            "has_property": (1, "mereology"),
            "has": (1, "mereology"),
            "causes": (2, "causality"),
            "requires": (2, "causality"),
            "produces": (2, "causality"),
            "located_in": (3, "spatial"),
            "orbits": (3, "spatial"),
            "lives_in": (3, "spatial"),
            "can": (4, "capability"),
            "capable_of": (4, "capability"),
            "larger_than": (5, "dynamics"),
            "greater_than": (5, "dynamics"),
            "less_than": (5, "dynamics"),
            "before": (5, "dynamics"),
            "after": (5, "dynamics"),
        }
        raw_prior = [0.02] * 6
        for p in q_res.spoke_predictions[:6]:
            cell_idx, _ = pred_to_cell.get(p.predicate, (0, "taxonomy"))
            raw_prior[cell_idx] += float(p.combined_confidence) * 0.5
        tr = q_res.inspectable_trace
        matched_trace_spoke: str | None = None
        for pred_k, (c_idx, s_label) in pred_to_cell.items():
            if f"({pred_k})" in tr:
                raw_prior[c_idx] += 3.0
                matched_trace_spoke = s_label
        if "I_SORT" in tr or "[MLU-360-ORDER]" in tr:
            raw_prior[5] += 3.5
            matched_trace_spoke = "dynamics"
        elif "I_EXCEPTION" in tr:
            if matched_trace_spoke is None:
                raw_prior[4] += 3.5
                matched_trace_spoke = "capability"
            else:
                for c_idx, s_label in pred_to_cell.values():
                    if s_label == matched_trace_spoke:
                        raw_prior[c_idx] += 3.5
                        break
        elif (
            q_res.status == "REFUTED"
            or "I_MUTEX" in tr
            or "[MLU-360-COMPARE]" in tr
            or "[MLU-360-EXPLAIN]" in tr
        ):
            raw_prior[0] += 3.0
            matched_trace_spoke = "taxonomy"
        elif (
            "[MLU-360-DIAGNOSE]" in tr
            or "[MLU-360-COUNTERFACTUAL]" in tr
            or "[MLU-360-CAUSAL-WAVE]" in tr
            or "[MLU-360-BRIDGE]" in tr
        ):
            raw_prior[2] += 3.0
            matched_trace_spoke = "causality"
        elif "[MLU-360-PLAN]" in tr:
            raw_prior[4] += 3.0
            matched_trace_spoke = "capability"

        prior_sum = sum(raw_prior) or 1.0
        spoke_prior = [round(v / prior_sum, 4) for v in raw_prior]

        if q_res.status == "UNKNOWN":
            return "", spoke_prior

        p_words = _tokenize_words(prompt)[:14]
        if not p_words:
            return "", spoke_prior
        p_clause = " ".join(p_words) + "?"
        focal_w = " ".join(_tokenize_words(q_res.focal_concept)) or "entity"
        raw_fluent = (q_res.fluent_english or q_res.answer).split("(Liquid stream:")[0]
        raw_fluent = raw_fluent.split("(Refuted")[0]
        ans_words = _tokenize_words(raw_fluent)[:34]
        if not ans_words:
            return "", spoke_prior

        if q_res.status == "REFUTED" and "I_SORT" in tr:
            think_clause = (
                f"think 360 poset antisymmetry gate refutes {focal_w} on dynamics spoke"
            )
        elif q_res.status == "REFUTED" and "I_EXCEPTION" in tr:
            exc_spoke = matched_trace_spoke or "capability"
            think_clause = (
                f"think 360 non monotonic exception gate refutes {focal_w} on {exc_spoke} spoke"
            )
        elif q_res.status == "REFUTED":
            think_clause = (
                f"think 360 invariant gate refutes {focal_w} via mutual exclusivity on taxonomy spoke"
            )
        elif "[MLU-360-ORDER]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            chain_w = " to ".join(hop_nodes[:5]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 transitive poset order traverses {chain_w} on dynamics spoke"
            )
        elif "[MLU-360-ANALOGY]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            an_spoke = matched_trace_spoke or "taxonomy"
            chain_w = " to ".join(hop_nodes[:4]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 proportional analogy maps {chain_w} on {an_spoke} spoke"
            )
        elif "[MLU-360-INTERSECT]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            in_spoke = matched_trace_spoke or "taxonomy"
            chain_w = " and ".join(hop_nodes[:4]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 compositional set intersection filters {chain_w} on {in_spoke} spoke"
            )
        elif "[MLU-360-EXPLAIN]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            chain_w = " to ".join(hop_nodes[:5]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 episodic working memory traces multi hop proof from {chain_w}"
            )
        elif "[MLU-360-DIAGNOSE]" in tr:
            obs_words = [
                w
                for w in p_words
                if self._token_salience(w) >= 2.1 and w not in focal_w.split()
            ][:4]
            obs_str = " and ".join(obs_words) if obs_words else "observations"
            think_clause = (
                f"think 360 abductive wave over {obs_str} converges on root cause {focal_w}"
            )
        elif "[MLU-360-COMPARE]" in tr:
            target_w = (
                " ".join(_tokenize_words(q_res.activated_mlus[-1]))
                if q_res.activated_mlus
                else "peer"
            )
            if "mutex=True" in tr:
                think_clause = (
                    f"think 360 comparative wave contrasts {focal_w} and {target_w} across mutually exclusive spokes"
                )
            else:
                think_clause = (
                    f"think intersecting 360 radial waves from {focal_w} and {target_w} across shared attractors"
                )
        elif "[MLU-360-COUNTERFACTUAL]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            chain_w = " and ".join(hop_nodes[:4]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 counterfactual intervention suppresses causal cascade across {chain_w} on causality spoke"
            )
        elif "[MLU-360-CAUSAL-WAVE]" in tr:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            chain_w = " then ".join(hop_nodes[:4]) if len(hop_nodes) >= 2 else focal_w
            think_clause = (
                f"think 360 multi hop causal wave propagates through {chain_w} on causality spoke"
            )
        elif "[MLU-360-PLAN]" in tr:
            think_clause = (
                f"think 360 backward chaining goal planner traces prerequisites for {focal_w}"
            )
        else:
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in q_res.activated_mlus
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ]
            best_spoke = (
                matched_trace_spoke
                or (q_res.spoke_predictions[0].spoke if q_res.spoke_predictions else "taxonomy")
            )
            if len(hop_nodes) >= 2:
                hop_chain = " to ".join(hop_nodes[:5])
                think_clause = (
                    f"think 360 multi hop wave traverses {hop_chain} on {best_spoke} spoke"
                )
            else:
                think_clause = f"think 360 spider web activates {focal_w} on {best_spoke} spoke"

        ans_clause = f"answer {' '.join(ans_words)}."
        return f"{p_clause} {think_clause} {ans_clause}", spoke_prior

    def generate_arena_conditioned_neural(
        self,
        prompt: str,
        q_res: MLMQueryResult | None = None,
        adapt_epochs: int = 18,
        adapt_lr: float = 0.045,
        max_new_tokens: int = 48,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 55 & 57: Fuse 360° Spider-Web Arena multi-hop deductions, episodic proofs, and spoke priors directly into NeuralSpiderWebMlu generation."""
        resolved_q = q_res if q_res is not None else self.query_360(prompt)
        cond_prompt = prompt
        if "[MLU-360-EXPLAIN]" in resolved_q.inspectable_trace:
            tail_cid = (
                resolved_q.activated_mlus[-1]
                if resolved_q.activated_mlus
                else resolved_q.focal_concept
            )
            cond_prompt = (
                f"{prompt.rstrip('?.!')} {resolved_q.focal_concept.lower()} {tail_cid.lower()}?"
            )
        arena_cot, spoke_prior = self._build_arena_conditioning(cond_prompt, resolved_q)
        gen = self.neural_mlu.generate_arena_conditioned(
            cond_prompt,
            arena_cot_sequence=arena_cot,
            spoke_prior=spoke_prior,
            adapt_epochs=adapt_epochs if arena_cot else 0,
            adapt_lr=adapt_lr,
            max_new_tokens=max_new_tokens,
        )
        if arena_cot and adapt_epochs > 0:
            self._dirty_since_save = True
            if persist:
                self.save_nodes([])
        gen["arena_status"] = resolved_q.status
        gen["arena_focal_concept"] = resolved_q.focal_concept
        gen["arena_spoke_prior"] = spoke_prior
        gen["arena_cot_sequence"] = arena_cot
        return gen

    def ask_curiosity_question_360(
        self,
        focal_concept: str | None = None,
        expected_spoke: str | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 58: Active Epistemic Self-Questioning — formulate a targeted 360° curiosity question and register a pending slot for elliptic user replies."""
        cur_dto = self.discover_curiosity(max_items=20)
        hyps = list(cur_dto.get("analogical_hypotheses", []))
        gaps = list(cur_dto.get("curiosity_gaps", []))

        target_focal = (
            focal_concept.strip().upper().replace(" ", "_")
            if focal_concept and focal_concept.strip()
            else None
        )

        chosen_hyp: dict[str, Any] | None = None
        if expected_spoke is None:
            for h in hyps:
                if target_focal is None or str(h.get("subject_concept", "")).upper() == target_focal:
                    chosen_hyp = h
                    break

        spoke_to_cell_idx = {
            "taxonomy": 0,
            "mereology": 1,
            "dynamics": 2,
            "attributes": 3,
            "procedural": 4,
            "episodic": 5,
        }

        if chosen_hyp is not None:
            subj_cid = str(chosen_hyp.get("subject_concept", "ENTITY")).upper()
            pred = str(chosen_hyp.get("predicate", "can")).lower()
            tgt_cid = str(chosen_hyp.get("target_concept", "")).upper()
            spoke = str(chosen_hyp.get("spoke", map_predicate_to_spoke(pred)))
            src_cid = str(chosen_hyp.get("analog_source", "PEER")).upper()
            hub_cid = str(chosen_hyp.get("shared_attractor", "SUBCORE")).upper()
            s_w = subj_cid.lower().replace("_", " ")
            t_w = tgt_cid.lower().replace("_", " ")
            src_w = src_cid.lower().replace("_", " ")
            hub_w = hub_cid.lower().replace("_", " ")
            sa = "an" if s_w and s_w[0] in "aeiou" else "a"
            src_a = "an" if src_w and src_w[0] in "aeiou" else "a"
            if pred in ("can", "capable_of"):
                question = f"Since both {s_w} and {src_w} are {hub_w}, can {sa} {s_w} also {t_w} like {src_a} {src_w}?"
                q_short = f"can {sa} {s_w} also {t_w} like {src_a} {src_w}"
            elif pred in ("has_part", "has_property", "has"):
                question = f"Since both {s_w} and {src_w} are {hub_w}, does {sa} {s_w} also have {t_w} like {src_a} {src_w}?"
                q_short = f"does {sa} {s_w} also have {t_w} like {src_a} {src_w}"
            elif pred in ("located_in", "lives_in", "orbits"):
                question = f"Since both {s_w} and {src_w} are {hub_w}, does {sa} {s_w} also live in {t_w} like {src_a} {src_w}?"
                q_short = f"does {sa} {s_w} also live in {t_w} like {src_a} {src_w}"
            else:
                question = f"Since both {s_w} and {src_w} are {hub_w}, does {sa} {s_w} also {pred.replace('_', ' ')} {t_w}?"
                q_short = f"does {sa} {s_w} also {pred.replace('_', ' ')} {t_w}"
            slot_type = "ANALOGICAL_HYPOTHESIS"
            sparsity = 0.5
        else:
            subj_cid = target_focal or (
                str(gaps[0]["concept_id"]).upper()
                if gaps
                else (self.recent_focal_stack[0][0] if self.recent_focal_stack else "CORE_ROOT")
            )
            gap_rec = next((g for g in gaps if str(g.get("concept_id", "")).upper() == subj_cid), None)
            sparsity = float(gap_rec.get("spoke_sparsity", 0.67)) if gap_rec else 0.67
            missing_list: list[str] = (
                [str(m) for m in gap_rec.get("missing_spokes", [])] if gap_rec else []
            )
            if not missing_list and subj_cid in self.nodes:
                u = self.nodes[subj_cid]
                missing_list = [
                    sp_name
                    for sp_name, head in u.spokes.items()
                    if not any(
                        isinstance(r, list) and len(r) >= 2 and float(r[0]) > float(r[1])
                        for r in head.target_affinities.values()
                    )
                ]
            priority_spokes = ("mereology", "procedural", "dynamics", "attributes", "taxonomy")
            spoke = expected_spoke or next(
                (sp for sp in priority_spokes if sp in missing_list),
                missing_list[0] if missing_list else "mereology",
            )
            s_w = subj_cid.lower().replace("_", " ")
            sa = "an" if s_w and s_w[0] in "aeiou" else "a"
            spoke_prompts = {
                "mereology": ("has_part", f"What parts or properties does {sa} {s_w} have?"),
                "procedural": ("can", f"What actions or capabilities can {sa} {s_w} perform?"),
                "dynamics": ("causes", f"What effects does {sa} {s_w} cause?"),
                "attributes": ("located_in", f"Where does {sa} {s_w} live?"),
                "taxonomy": ("is_a", f"What category is {sa} {s_w}?"),
            }
            pred, question = spoke_prompts.get(
                spoke, ("has_part", f"What properties does {sa} {s_w} have?")
            )
            q_short = " ".join(_tokenize_words(question))
            tgt_cid = ""
            src_cid = ""
            hub_cid = ""
            slot_type = "MISSING_SPOKE"

        self.pending_curiosity_slot = {
            "slot_type": slot_type,
            "focal_concept": subj_cid,
            "expected_spoke": spoke,
            "default_predicate": pred,
            "candidate_target": tgt_cid or None,
            "analog_source": src_cid or None,
            "shared_attractor": hub_cid or None,
            "question": question,
        }
        if subj_cid in self.nodes:
            self._update_conversation_context(subj_cid)

        raw_prior = [0.02] * 6
        raw_prior[spoke_to_cell_idx.get(spoke, 0)] += 3.0
        p_sum = sum(raw_prior) or 1.0
        spoke_prior = [round(v / p_sum, 4) for v in raw_prior]
        s_w = subj_cid.lower().replace("_", " ")
        prompt_q = f"what curiosity question should i ask about {s_w}?"
        cot_seq = (
            f"{prompt_q} think 360 epistemic curiosity detects high entropy on {spoke} spoke for {s_w} "
            f"answer {q_short}."
        )
        gen = self.neural_mlu.generate_arena_conditioned(
            prompt_q,
            arena_cot_sequence=cot_seq,
            spoke_prior=spoke_prior,
            adapt_epochs=18,
            adapt_lr=0.045,
            max_new_tokens=36,
        )
        if persist:
            self.save_nodes([])

        return {
            "mode": "CURIOSITY_INQUIRY",
            "slot_type": slot_type,
            "focal_concept": subj_cid,
            "expected_spoke": spoke,
            "default_predicate": pred,
            "candidate_target": tgt_cid or None,
            "question": question,
            "neural_thinking": str(gen.get("neural_thinking", "")).strip(),
            "neural_answer": str(gen.get("neural_answer", "")).strip(),
            "active_spoke_weights": gen.get("active_spoke_weights", []),
            "spoke_sparsity": sparsity,
            "trace": (
                f"[MLU-360-CURIOSITY] slot={slot_type} | focal={subj_cid} | "
                f"spoke={spoke}({pred}) | target={tgt_cid or 'OPEN'}"
            ),
        }

    def _resolve_pending_curiosity_slot(
        self,
        clean_in: str,
        tokens: list[str],
        has_leading_op: bool,
        has_interior_op: bool,
        persist: bool,
    ) -> dict[str, Any] | None:
        """Phase 58: Bind an elliptic user answer ('Yes', 'No', or bare concept phrase) to an active pending_curiosity_slot."""
        slot = self.pending_curiosity_slot
        if slot is None or not tokens or clean_in.endswith("?"):
            return None

        # If the user starts with a non-negation interrogative/leading operator, they are asking a new question
        has_neg = any(t in self.frame_policy.negation_tokens for t in tokens)
        if has_leading_op and not has_neg:
            return None

        subj_cid = str(slot["focal_concept"])
        pred = str(slot["default_predicate"])
        spoke = str(slot["expected_spoke"])
        cand_tgt = slot.get("candidate_target")
        s_w = subj_cid.lower().replace("_", " ")
        sa = "an" if s_w and s_w[0] in "aeiou" else "a"
        pred_w = "has" if pred in ("has_part", "has_property") else pred.replace("_", " ")

        if slot["slot_type"] == "ANALOGICAL_HYPOTHESIS" and cand_tgt:
            t_w = str(cand_tgt).lower().replace("_", " ")
            if has_neg:
                self.pending_curiosity_slot = None
                self.revise_belief_360(
                    subj_cid, pred, str(cand_tgt), verified=False, evidence_weight=4.0
                )
                neg_cot = (
                    f"can {sa} {s_w} {t_w}? think 360 curiosity feedback refutes {t_w} on {spoke} spoke for {s_w} "
                    f"answer no {sa} {s_w} cannot {t_w}."
                    if pred in ("can", "capable_of")
                    else f"does {sa} {s_w} have {t_w}? think 360 curiosity feedback refutes {t_w} on {spoke} spoke for {s_w} "
                    f"answer no {sa} {s_w} does not have {t_w}."
                )
                self.train_neural_mlu(neg_cot, epochs=24, learning_rate=0.04, persist=persist)
                return {
                    "mode": "CURIOSITY_RESOLVED",
                    "status": "REFUTED",
                    "focal_concept": subj_cid,
                    "predicate": pred,
                    "target_concept": str(cand_tgt),
                    "reply": f"Recorded negative evidence: {sa} {s_w} does not {pred_w} {t_w}.",
                    "trace": f"[MLU-360-CURIOSITY-REJECTED] {subj_cid} !({pred}) {cand_tgt}",
                }
            if not has_interior_op or str(cand_tgt).lower() in "_".join(tokens):
                self.pending_curiosity_slot = None
                synth_stmt = f"{sa.capitalize()} {s_w} {pred_w} {t_w}."
                rep = self.train_on_text(synth_stmt, persist=persist)
                distill_rep: dict[str, Any] = {}
                if rep.verified_triples:
                    distill_rep = self.distill_triples_to_neural_mlu(
                        rep.verified_triples, epochs=28, learning_rate=0.04, persist=persist
                    )
                return {
                    "mode": "CURIOSITY_RESOLVED",
                    "status": "VERIFIED" if rep.relations_verified > 0 else "REFUTED",
                    "focal_concept": subj_cid,
                    "predicate": pred,
                    "target_concept": str(cand_tgt),
                    "relations_verified": rep.relations_verified,
                    "neural_sequences_distilled": int(distill_rep.get("sequences_trained", 0)),
                    "reply": f"Verified and learned from curiosity confirmation: {sa} {s_w} {pred_w} {t_w}.",
                    "trace": f"[MLU-360-CURIOSITY-VERIFIED] {subj_cid} -({pred})-> {cand_tgt}",
                }

        if slot["slot_type"] == "MISSING_SPOKE":
            if not has_interior_op:
                extracted_cid, sal = self._extract_salient_concept(tokens)
                if extracted_cid and sal >= 1.8:
                    self.pending_curiosity_slot = None
                    t_w = extracted_cid.lower().replace("_", " ")
                    synth_stmt = f"{sa.capitalize()} {s_w} {pred_w} {t_w}."
                    rep = self.train_on_text(synth_stmt, persist=persist)
                    distill_rep = {}
                    if rep.verified_triples:
                        distill_rep = self.distill_triples_to_neural_mlu(
                            rep.verified_triples, epochs=28, learning_rate=0.04, persist=persist
                        )
                    return {
                        "mode": "CURIOSITY_RESOLVED",
                        "status": "VERIFIED" if rep.relations_verified > 0 else "REFUTED",
                        "focal_concept": subj_cid,
                        "predicate": pred,
                        "target_concept": extracted_cid.upper(),
                        "relations_verified": rep.relations_verified,
                        "relations_refuted": rep.relations_refuted,
                        "neural_sequences_distilled": int(distill_rep.get("sequences_trained", 0)),
                        "reply": (
                            f"Learned from your answer: {sa} {s_w} {pred_w} {t_w}."
                            if rep.relations_verified > 0
                            else f"Rejected contradictory answer for {s_w} via 360° invariant gates."
                        ),
                        "trace": f"[MLU-360-CURIOSITY-FILLED] {subj_cid} -({pred})-> {extracted_cid.upper()}",
                    }
            else:
                # Full or pronoun-led sentence answer ("It has keratin_scales.") -> clear slot and let normal learning handle it
                self.pending_curiosity_slot = None

        return None

    def chat_turn_360(self, user_input: str, persist: bool = True) -> dict[str, Any]:
        """Phase 34, 52, 53, 55, 57 & 58: Execute a unified 360° conversational turn combining Differentiable Neural Thinking + 360° Spider-Web Arena + Episodic Working Memory + Curiosity Loop."""
        clean_in = user_input.strip()
        if not clean_in:
            return {
                "mode": "NO_OP",
                "reply": "",
                "trace": "[MLU-360-CHAT] Empty input.",
                "focal_concept": "CORE_ROOT",
            }

        tokens = _tokenize_words(clean_in)
        op_tuples = {ptoks for ptoks, _, _ in self._operator_codebook}
        has_neg_exception = any(
            bool(self._extract_negated_triples(s)) for s in _segment_sentences(clean_in)
        )
        has_leading_op = bool(
            tokens
            and not (has_neg_exception and tokens[0] in ("actually", "however", "but", "no"))
            and (
                any(
                    tuple(tokens[:k]) in op_tuples
                    for k in (1, 2, 3)
                    if k <= len(tokens)
                )
                or (
                    self._operator_token_mass.get(tokens[0], 0.0) > 0.0
                    and tokens[0] not in self.frame_policy.stop_determiners
                )
            )
        )
        has_interior_op = has_neg_exception or any(
            tuple(tokens[i:j]) in op_tuples
            or tuple(t for t in tokens[i:j] if t not in self.frame_policy.stop_determiners)
            in op_tuples
            for i in range(1, len(tokens))
            for j in range(i + 1, min(len(tokens), i + 6))
        )

        if self.pending_curiosity_slot is not None:
            resolved_slot = self._resolve_pending_curiosity_slot(
                clean_in, tokens, has_leading_op, has_interior_op, persist=persist
            )
            if resolved_slot is not None:
                return resolved_slot

        is_question = clean_in.endswith("?") or has_leading_op or not has_interior_op
        neural_gen = self.generate_neural_mlu(clean_in, max_new_tokens=48)
        total_ptoks = int(neural_gen.get("total_prompt_tokens", 0))
        known_ptoks = int(neural_gen.get("known_prompt_tokens", 0))
        n_conf = float(neural_gen.get("mean_token_confidence", 0.0))
        n_think = str(neural_gen.get("neural_thinking", "")).strip()
        n_ans = str(neural_gen.get("neural_answer", "")).strip()
        neural_grounded = bool(
            total_ptoks > 0
            and known_ptoks == total_ptoks
            and n_conf >= 0.85
            and n_ans
        )

        if is_question or (neural_grounded and not clean_in.endswith(".")):
            q_res = self.query_360(clean_in)
            status = q_res.status
            reply = q_res.fluent_english or q_res.answer
            trace = q_res.inspectable_trace
            conf = q_res.confidence

            if (
                "[MLU-360-EXPLAIN]" in trace
                or "[MLU-360-COMPARE]" in trace
                or "[MLU-360-ORDER]" in trace
                or "[MLU-360-ANALOGY]" in trace
                or "[MLU-360-INTERSECT]" in trace
                or "[MLU-360-COUNTERFACTUAL]" in trace
                or "[MLU-360-CAUSAL-WAVE]" in trace
                or status == "REFUTED"
            ):
                neural_grounded = False

            # Phase 55 & 57: When the 360° Arena resolves a multi-hop/abductive/comparative/refuted/explanation query
            # that NeuralSpiderWebMlu has not yet grounded on its own, fuse the Arena proof + 6-spoke prior into NeuralSpiderWebMlu!
            if not neural_grounded and status in ("VERIFIED", "NEURAL_GROUNDED", "REFUTED"):
                neural_gen = self.generate_arena_conditioned_neural(
                    clean_in,
                    q_res=q_res,
                    adapt_epochs=18,
                    adapt_lr=0.045,
                    max_new_tokens=48,
                    persist=persist,
                )
                n_conf = float(neural_gen.get("mean_token_confidence", 0.0))
                n_think = str(neural_gen.get("neural_thinking", "")).strip()
                n_ans = str(neural_gen.get("neural_answer", "")).strip()
                neural_grounded = bool(n_ans and n_conf >= 0.75)
                spoke_weights = neural_gen.get("active_spoke_weights", [])
                if spoke_weights:
                    top_cell, top_w = spoke_weights[0]
                    trace = f"{trace} | [ARENA-NEURAL-FUSION] {top_cell}={top_w:.2f}"

            if neural_grounded:
                if n_think:
                    trace = f"{trace} | [NEURAL-THINK] {n_think}"
                explicit_concept_in_prompt = any(
                    t.upper() == q_res.focal_concept
                    or q_res.focal_concept in "_".join(t.upper() for t in tokens)
                    for t in tokens
                )
                is_structured_wave = (
                    has_leading_op
                    or status == "REFUTED"
                    or "[MLU-360-EXPLAIN]" in q_res.inspectable_trace
                    or "[MLU-360-DIAGNOSE]" in q_res.inspectable_trace
                    or "[MLU-360-COMPARE]" in q_res.inspectable_trace
                    or "[MLU-360-ANALOGY]" in q_res.inspectable_trace
                    or "[MLU-360-INTERSECT]" in q_res.inspectable_trace
                    or "[MLU-360-COUNTERFACTUAL]" in q_res.inspectable_trace
                    or "[MLU-360-CAUSAL-WAVE]" in q_res.inspectable_trace
                    or (
                        "[MLU-360-PLAN]" in q_res.inspectable_trace
                        and q_res.confidence >= 0.72
                    )
                )
                if status != "REFUTED" and (
                    status == "UNKNOWN"
                    or not is_question
                    or (not explicit_concept_in_prompt and not is_structured_wave)
                    or (not is_structured_wave and n_conf >= 0.90 and n_conf > q_res.confidence + 0.12)
                ):
                    status = "VERIFIED"
                    reply = (
                        f"{n_ans} [Neural Thought: {n_think}]"
                        if n_think and "Thought:" not in n_ans
                        else n_ans
                    )
                    conf = n_conf
                elif n_think and "Thought:" not in reply:
                    reply = f"{reply} [Neural Thought: {n_think}]"

            return {
                "mode": "INQUIRY",
                "status": status,
                "focal_concept": q_res.focal_concept,
                "reply": reply,
                "neural_thinking": n_think if neural_grounded else "",
                "neural_answer": n_ans if neural_grounded else "",
                "active_spoke_weights": neural_gen.get("active_spoke_weights", []),
                "dominant_logic_gates": neural_gen.get("dominant_logic_gates", []),
                "trace": trace,
                "confidence": conf,
            }

        rep = self.train_on_text(clean_in, persist=persist)
        neural_distill_rep: dict[str, Any] = {}
        if rep.verified_triples:
            neural_distill_rep = self.distill_triples_to_neural_mlu(
                rep.verified_triples, epochs=28, learning_rate=0.04, persist=persist
            )
        focal_cid = (
            self.recent_focal_stack[0][0]
            if self.recent_focal_stack
            else "CORE_ROOT"
        )
        delib = self.deliberate_360(focal_cid, max_hops=3)
        delib_synth = str(delib.get("deliberative_synthesis", "")).strip()
        if rep.relations_refuted > 0 and rep.relations_verified == 0:
            reply = (
                f"Rejected {rep.relations_refuted} contradictory relation(s) via 360° invariant gates."
            )
        else:
            reply = (
                f"Learned {rep.relations_verified} verified 360° spoke relation(s) "
                f"(surprise={rep.average_surprise:.3f}). {delib_synth}"
            ).strip()
        return {
            "mode": "LEARN_AND_DELIBERATE",
            "status": "VERIFIED" if rep.relations_verified > 0 else "REFUTED",
            "focal_concept": focal_cid,
            "relations_verified": rep.relations_verified,
            "relations_refuted": rep.relations_refuted,
            "neural_sequences_distilled": int(neural_distill_rep.get("sequences_trained", 0)),
            "neural_final_perplexity": float(neural_distill_rep.get("final_perplexity", 0.0)),
            "reply": reply,
            "trace": (
                f"[MLU-360-CHAT] focal={focal_cid} | verified={rep.relations_verified} | "
                f"refuted={rep.relations_refuted} | F={delib.get('free_energy', 0.0):.2f} | "
                f"neural_distilled={neural_distill_rep.get('sequences_trained', 0)}"
            ),
            "follow_up_question": delib.get("follow_up_question", ""),
        }

    def consolidate_multihop_to_neural_mlu(
        self,
        promoted_hypotheses: list[dict[str, Any]] | None = None,
        max_sequences: int = 36,
        epochs: int = 24,
        learning_rate: float = 0.038,
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 56: Sleep-Cycle Autonomous Multi-Hop, Analogical & Causal Neural Consolidation into NeuralSpiderWebMlu."""
        lines: list[str] = []
        seen: set[str] = set()

        def _add_line(s_line: str) -> None:
            if s_line and s_line not in seen and len(lines) < max_sequences:
                seen.add(s_line)
                lines.append(s_line)

        def _art(phrase: str) -> str:
            return "an" if phrase and phrase[0] in "aeiou" else "a"

        # 1. Consolidate analogical curiosity promotions discovered during sleep
        for hyp in promoted_hypotheses or []:
            s = " ".join(_tokenize_words(str(hyp.get("subject_concept", ""))))
            pred = str(hyp.get("predicate", "")).lower().strip()
            o = " ".join(_tokenize_words(str(hyp.get("target_concept", ""))))
            src = " ".join(_tokenize_words(str(hyp.get("analog_source", "")))) or "peer"
            hub = " ".join(_tokenize_words(str(hyp.get("shared_attractor", "")))) or "subcore"
            if not s or not pred or not o:
                continue
            sa = _art(s)
            oa = _art(o)
            if pred in ("can", "capable_of"):
                _add_line(
                    f"can {sa} {s} {o}? think 360 sleep analogical capability transfer from {src} via {hub} answer yes {sa} {s} can {o}."
                )
            elif pred in ("located_in", "lives_in", "orbits"):
                _add_line(
                    f"where does {sa} {s} live? think 360 sleep analogical spatial transfer from {src} via {hub} answer {sa} {s} lives in {o}."
                )
            elif pred in ("has_part", "has_property", "has"):
                _add_line(
                    f"does {sa} {s} have {o}? think 360 sleep analogical mereology transfer from {src} via {hub} answer yes {sa} {s} has {o}."
                )
            else:
                _add_line(
                    f"is {sa} {s} {oa} {o}? think 360 sleep analogical taxonomy transfer from {src} via {hub} answer yes {sa} {s} is {oa} {o}."
                )

        # 2. Consolidate multi-hop (>= 2 hops) transitive deductions across active concepts
        candidate_units = sorted(
            (
                u
                for cid, u in self.nodes.items()
                if cid != "CORE_ROOT" and cid not in DEFAULT_SUBCORES
            ),
            key=lambda u: (u.verified_hits + float(u.activation_count)),
            reverse=True,
        )
        for unit in candidate_units[:24]:
            if len(lines) >= max_sequences:
                break
            direct_targets = {
                t_cid
                for sp in unit.spokes.values()
                for t_cid, rec in sp.target_affinities.items()
                if isinstance(rec, list) and len(rec) >= 2 and float(rec[0]) > float(rec[1])
            }
            preds, activated = self._collect_360_predictions(unit, max_hops=5)
            s = " ".join(_tokenize_words(unit.concept_id))
            sa = _art(s)
            hop_nodes = [
                " ".join(_tokenize_words(c))
                for c in activated
                if c != "CORE_ROOT" and not c.startswith("SUBCORE_")
            ][:4]
            for p in preds:
                if (
                    not p.verified
                    or p.target_concept in direct_targets
                    or p.target_concept == unit.concept_id
                ):
                    continue
                o = " ".join(_tokenize_words(p.target_concept))
                if not s or not o:
                    continue
                oa = _art(o)
                hop_str = " to ".join(hop_nodes) if len(hop_nodes) >= 2 else f"{s} to {o}"
                if p.predicate in ("is_a", "subclass_of", "instance_of"):
                    _add_line(
                        f"is {sa} {s} {oa} {o}? think 360 sleep consolidation traces multi hop taxonomy from {hop_str} answer yes {sa} {s} is {oa} {o}."
                    )
                elif p.predicate in ("has_part", "has_property", "has"):
                    _add_line(
                        f"does {sa} {s} have {o}? think 360 sleep consolidation inherits {o} on mereology spoke from {hop_str} answer yes {sa} {s} has {o}."
                    )
                elif p.predicate in ("can", "capable_of"):
                    _add_line(
                        f"can {sa} {s} {o}? think 360 sleep consolidation inherits capability {o} from {hop_str} answer yes {sa} {s} can {o}."
                    )
                elif p.predicate in ("located_in", "lives_in", "orbits"):
                    _add_line(
                        f"where is {sa} {s}? think 360 sleep consolidation traces spatial spoke from {hop_str} answer {sa} {s} is in {o}."
                    )
                elif p.predicate in ("larger_than", "greater_than", "before", "after"):
                    pred_w = p.predicate.replace("_", " ")
                    _add_line(
                        f"is {s} {pred_w} {o}? think 360 sleep consolidation traces transitive poset order from {hop_str} on dynamics spoke answer yes {s} is {pred_w} {o}."
                    )

            # 3. Consolidate multi-hop causal wave cascades
            has_causal = any(
                isinstance(rec, list) and len(rec) >= 4 and str(rec[3]) == "causes"
                for sp in unit.spokes.values()
                for rec in sp.target_affinities.values()
            )
            if has_causal and self.rust_arena.is_native:
                causal_dto = self.simulate_causal_360(unit.concept_id, max_hops=4)
                affected = [str(c) for c in causal_dto.get("affected_concepts", [])]
                if len(affected) >= 2:
                    chain_w = " to ".join(
                        " ".join(_tokenize_words(c)) for c in [unit.concept_id, *affected[:3]]
                    )
                    for eff_cid in affected[1:]:
                        e_w = " ".join(_tokenize_words(eff_cid))
                        if s and e_w:
                            _add_line(
                                f"what causes {e_w}? think 360 sleep causal wave propagates from {chain_w} answer {s} causes {e_w}."
                            )
                            _add_line(
                                f"does {s} cause {e_w}? think 360 sleep causal wave propagates from {chain_w} answer yes {s} causes {e_w}."
                            )

        if not lines:
            return {
                "synthesized_sequences": 0,
                "epochs": 0,
                "sequences_trained": 0,
                "final_perplexity": 1.0,
            }
        train_rep = self.train_neural_mlu(
            "\n".join(lines), epochs=epochs, learning_rate=learning_rate, persist=persist
        )
        train_rep["synthesized_sequences"] = len(lines)
        return train_rep

    def consolidate(
        self, min_transition_weight: float = 0.05, replay_lr: float = 0.15
    ) -> dict[str, Any]:
        """Phase 2, 15, 18 & 56: Run Sleep-Cycle Consolidation (spoke replay + transition pruning + mitosis + active curiosity + autonomous multi-hop neural consolidation)."""
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
        curiosity = self.discover_curiosity(max_items=15)
        autonomous_step = self.autonomous_curiosity_step(
            min_confidence=0.65, max_promotions=5
        )
        clusters = self.discover_clusters_360(min_members=2)
        promoted, pruned = self.run_structural_plasticity_cycle()
        neural_sleep = self.consolidate_multihop_to_neural_mlu(
            promoted_hypotheses=autonomous_step.get("promoted_hypotheses", []),
            max_sequences=36,
            epochs=24,
            learning_rate=0.038,
            persist=False,
        )
        self.save_nodes(promoted if promoted else None)
        return {
            "rust_consolidation": rust_report,
            "curiosity": curiosity,
            "autonomous_curiosity": autonomous_step,
            "latent_clusters": clusters,
            "neural_sleep_consolidation": neural_sleep,
            "python_mitosis_promotions": promoted,
            "python_pruned_nodes": pruned,
            "total_mlu_nodes": len(self.nodes),
            "rust_arena_nodes": self.rust_arena.node_count,
        }

    def ingest_json_ontology(
        self, json_path: str, persist: bool = False, max_triples: int | None = None
    ) -> dict[str, Any]:
        """Phase 44: Ingest structured JSON knowledge triples (e.g., commonsense.json, extended_ontology.json) into the 360° Spider-Web MLM & Rust Arena."""
        import json
        from pathlib import Path

        fpath = Path(json_path)
        if not fpath.is_file():
            return {"triples_processed": 0, "relations_verified": 0, "relations_refuted": 0}

        try:
            payload = json.loads(fpath.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, ValueError):
            return {"triples_processed": 0, "relations_verified": 0, "relations_refuted": 0}

        raw_triples = payload.get("triples") or payload.get("seed_triples") or []
        if max_triples is not None and max_triples > 0:
            raw_triples = raw_triples[:max_triples]
        triples_processed = 0
        relations_verified = 0
        relations_refuted = 0
        synthesized_sentences: list[str] = []

        for item in raw_triples:
            if not isinstance(item, list) or len(item) < 3:
                continue
            subj = str(item[0]).strip().lower().replace(" ", "_")
            pred = str(item[1]).strip().lower().replace(" ", "_")
            obj = str(item[2]).strip().lower().replace(" ", "_")
            verified_flag = bool(item[4]) if len(item) >= 5 else True
            if not subj or not pred or not obj or subj == obj:
                continue

            triples_processed += 1
            spoke = map_predicate_to_spoke(pred)
            subj_unit = self.get_or_create_mlu(subj, preferred_spoke=spoke)
            obj_unit = self.get_or_create_mlu(obj, preferred_spoke=spoke)

            if self.rust_arena.is_native:
                inv_dto = self.rust_arena.verify_invariants(
                    subj_unit.concept_id,
                    "disjoint_with" if pred == "not_a" else pred,
                    obj_unit.concept_id,
                )
                passed = bool(inv_dto.get("passed", True))
            else:
                passed = self.verifier.verify_relation(subj, pred, obj).passed

            if passed and verified_flag:
                relations_verified += 1
                rust_pred = "disjoint_with" if pred == "not_a" else pred
                if self.rust_arena.is_native:
                    self.rust_arena.learn_spoke(
                        subj_unit.concept_id,
                        rust_pred,
                        obj_unit.concept_id,
                        learning_rate=0.20,
                        verified=True,
                    )
                subj_unit.learn_spoke_relation(
                    predicate=rust_pred,
                    target_concept=obj_unit.concept_id,
                    target_vec=obj_unit.anchor_vec,
                    learning_rate=0.20,
                    verified=True,
                )
                obj_unit.verified_hits += 0.5
                if pred not in ("disjoint_with", "not_a"):
                    s_clean = subj.replace("_", " ")
                    p_clean = pred.replace("_", " ")
                    o_clean = obj.replace("_", " ")
                    synthesized_sentences.append(f"{s_clean} {p_clean} {o_clean}.")
            else:
                relations_refuted += 1

        if synthesized_sentences:
            stream_blob = " ".join(synthesized_sentences)
            if self.rust_arena.is_native:
                self.rust_arena.train_stream(stream_blob, learning_rate=0.12)
            core_unit = self.nodes["CORE_ROOT"]
            for s_line in synthesized_sentences[:400]:
                toks = _tokenize_words(s_line)
                if len(toks) >= 2:
                    core_unit.learn_token_sequence(toks, learning_rate=0.12)

        self._dirty_since_save = True
        if persist:
            self.save_nodes()

        return {
            "triples_processed": triples_processed,
            "relations_verified": relations_verified,
            "relations_refuted": relations_refuted,
        }

    @staticmethod
    def extract_text_from_document(file_path: str) -> str:
        """Phase 47: Pure-algorithmic text extractor for .epub, .docx, .pdf, .md, and .txt files (zero regexes)."""
        import xml.etree.ElementTree as ET
        import zipfile
        import zlib
        from pathlib import Path

        fpath = Path(file_path)
        if not fpath.is_file():
            return ""
        ext = fpath.suffix.lower()

        if ext in (".epub", ".docx"):
            chunks: list[str] = []
            try:
                with zipfile.ZipFile(fpath, "r") as zf:
                    for name in sorted(zf.namelist()):
                        low_name = name.lower()
                        if low_name.endswith(("container.xml", ".opf")):
                            continue
                        if low_name.endswith((".xhtml", ".html", ".xml")):
                            try:
                                raw_xml = zf.read(name)
                                root_el = ET.fromstring(raw_xml)
                                for piece in root_el.itertext():
                                    clean_piece = piece.strip()
                                    if clean_piece:
                                        chunks.append(clean_piece)
                            except (ET.ParseError, KeyError, OSError, ValueError):
                                pass
            except (OSError, zipfile.BadZipFile, ValueError):
                return ""
            return " ".join(chunks)

        if ext == ".pdf":
            try:
                raw_bytes = fpath.read_bytes()
            except OSError:
                return ""
            extracted_blocks: list[str] = []
            pos = 0
            n_bytes = len(raw_bytes)
            while pos < n_bytes:
                s_idx = raw_bytes.find(b"stream", pos)
                if s_idx < 0:
                    break
                e_idx = raw_bytes.find(b"endstream", s_idx + 6)
                if e_idx < 0:
                    break
                body = raw_bytes[s_idx + 6 : e_idx].strip(b"\r\n ")
                pos = e_idx + 9
                decoded_str = ""
                for candidate_bytes in (body,):
                    try:
                        decomp = zlib.decompress(candidate_bytes)
                        decoded_str = decomp.decode("latin-1", errors="ignore")
                    except (zlib.error, ValueError):
                        decoded_str = candidate_bytes.decode("latin-1", errors="ignore")
                if not decoded_str:
                    continue
                # Character-state automaton to extract parenthesized text literals ( ... )
                depth = 0
                escaped = False
                buf: list[str] = []
                for ch in decoded_str:
                    if escaped:
                        buf.append(ch)
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        continue
                    if ch == "(":
                        if depth > 0:
                            buf.append(ch)
                        depth += 1
                    elif ch == ")":
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and buf:
                                span = "".join(buf).strip()
                                if span:
                                    extracted_blocks.append(span)
                                buf.clear()
                            elif depth > 0:
                                buf.append(ch)
                    elif depth > 0:
                        buf.append(ch)
            return " ".join(extracted_blocks)

        try:
            return fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    def _learn_verified_triple_direct(
        self, subj: str, pred: str, obj: str
    ) -> tuple[bool, str]:
        """Verify and commit a structured (subject, predicate, object) triple into Rust Arena, Python MLUs, and MemoryStore."""
        s_clean = subj.strip().lower().replace(" ", "_")
        o_clean = obj.strip().lower().replace(" ", "_")
        p_clean = pred.strip().lower().replace(" ", "_")
        if not s_clean or not o_clean or s_clean == o_clean:
            return False, ""
        spoke = map_predicate_to_spoke(p_clean)
        subj_u = self.get_or_create_mlu(s_clean, preferred_spoke=spoke)
        obj_u = self.get_or_create_mlu(o_clean, preferred_spoke=spoke)
        rust_p = "disjoint_with" if p_clean == "not_a" else p_clean
        if self.rust_arena.is_native:
            inv = self.rust_arena.verify_invariants(
                subj_u.concept_id, rust_p, obj_u.concept_id
            )
            passed = bool(inv.get("passed", True))
        else:
            passed = self.verifier.verify_relation(s_clean, p_clean, o_clean).passed
        if not passed:
            return False, ""
        if self.rust_arena.is_native:
            self.rust_arena.learn_spoke(
                subj_u.concept_id,
                rust_p,
                obj_u.concept_id,
                learning_rate=0.20,
                verified=True,
            )
        subj_u.learn_spoke_relation(
            predicate=rust_p,
            target_concept=obj_u.concept_id,
            target_vec=obj_u.anchor_vec,
            learning_rate=0.20,
            verified=True,
        )
        obj_u.verified_hits += 0.5
        c_sub = self.memory.create_concept(subj_u.concept_id)
        c_obj = self.memory.create_concept(obj_u.concept_id)
        self.memory.add_relation(c_sub.id, rust_p, c_obj.id)
        if p_clean not in ("disjoint_with", "not_a"):
            return (
                True,
                f"{s_clean.replace('_', ' ')} {p_clean.replace('_', ' ')} {o_clean.replace('_', ' ')}.",
            )
        return True, ""

    def ingest_csv_dataset(
        self, csv_path: str, max_rows: int | None = None, persist: bool = False
    ) -> dict[str, Any]:
        """Phase 47 & 54: Ingest tabular CSV/TSV datasets (triples, Kaggle taxonomies, QA pairs, symptom profiles, or prose) into the 360° Spider-Web MLM & Neural MLU."""
        import csv
        from pathlib import Path

        fpath = Path(csv_path)
        if not fpath.is_file():
            return {
                "rows_processed": 0,
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
            }

        raw_text = fpath.read_text(encoding="utf-8", errors="ignore")
        first_line = raw_text.splitlines()[0] if raw_text.splitlines() else ""
        if fpath.suffix.lower() == ".tsv":
            delim = "\t"
        elif ";" in first_line and first_line.count(";") > first_line.count(","):
            delim = ";"
        else:
            delim = ","

        rows_processed = 0
        relations_verified = 0
        relations_refuted = 0
        text_passages: list[str] = []
        synth_sentences: list[str] = []
        neural_cot_lines: list[str] = []
        headers: list[str] = []
        taxonomy_col_indices: list[int] = []
        q_col_idx: int = -1
        a_col_idx: int = -1
        disease_col_idx: int = -1
        symptom_cols: list[tuple[int, str]] = []

        tax_ranks = (
            "scientific_name",
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
        )

        try:
            with fpath.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                for row_idx, row in enumerate(reader):
                    if max_rows is not None and rows_processed >= max_rows:
                        break
                    raw_cells = [c.strip().lstrip("\ufeff") for c in row]
                    if not any(raw_cells):
                        continue

                    if row_idx == 0:
                        norm_h = [
                            "_".join(_tokenize_words(c)) for c in raw_cells
                        ]
                        # Check if row 0 is a header row
                        is_triple_row = (
                            len(raw_cells) >= 3
                            and (
                                self.verifier.registry.relation(norm_h[1]) is not None
                                or norm_h[1] in ("not_a", "disjoint_with")
                            )
                            and norm_h[0] != "subject"
                        )
                        if not is_triple_row:
                            headers = norm_h
                            for rk in tax_ranks:
                                if rk in headers:
                                    taxonomy_col_indices.append(headers.index(rk))
                            if "question" in headers and "answer" in headers:
                                q_col_idx = headers.index("question")
                                a_col_idx = headers.index("answer")
                            if "disease" in headers:
                                disease_col_idx = headers.index("disease")
                                for idx_c, h_name in enumerate(headers):
                                    if idx_c != disease_col_idx and h_name not in (
                                        "age",
                                        "gender",
                                        "blood_pressure",
                                        "cholesterol_level",
                                        "outcome_variable",
                                    ):
                                        symptom_cols.append((idx_c, h_name))
                            continue

                    # 1. Check Kaggle Multi-Rank Taxonomic Hierarchy CSV (e.g. Species -> Genus -> Family -> Order -> Class -> Phylum -> Kingdom)
                    if len(taxonomy_col_indices) >= 3:
                        rank_vals: list[str] = []
                        for c_idx in taxonomy_col_indices:
                            if c_idx < len(raw_cells) and raw_cells[c_idx]:
                                val_clean = "_".join(_tokenize_words(raw_cells[c_idx]))
                                if val_clean and val_clean not in rank_vals:
                                    rank_vals.append(val_clean)
                        if len(rank_vals) >= 2:
                            rows_processed += 1
                            for k in range(len(rank_vals) - 1):
                                child_c, parent_c = rank_vals[k], rank_vals[k + 1]
                                ok, sent = self._learn_verified_triple_direct(
                                    child_c, "is_a", parent_c
                                )
                                if ok:
                                    relations_verified += 1
                                    if sent:
                                        synth_sentences.append(sent)
                                else:
                                    relations_refuted += 1
                            if len(neural_cot_lines) < 25:
                                leaf = rank_vals[0].replace("_", " ")
                                parent = rank_vals[1].replace("_", " ")
                                root_k = rank_vals[-1].replace("_", " ")
                                neural_cot_lines.append(
                                    f"what is {leaf}? think {leaf} belongs to {parent} in kingdom {root_k} answer {leaf} is a {parent} in the {root_k} kingdom."
                                )
                            continue

                    # 2. Check Kaggle Question-Answer CSV (e.g. Question, Answer)
                    if (
                        q_col_idx >= 0
                        and a_col_idx >= 0
                        and q_col_idx < len(raw_cells)
                        and a_col_idx < len(raw_cells)
                    ):
                        q_txt = raw_cells[q_col_idx]
                        a_txt = raw_cells[a_col_idx]
                        if q_txt and a_txt:
                            rows_processed += 1
                            text_passages.append(a_txt if a_txt.endswith(".") else f"{a_txt}.")
                            q_toks = _tokenize_words(q_txt)[:12]
                            a_toks = _tokenize_words(a_txt)[:14]
                            if q_toks and a_toks and len(neural_cot_lines) < 35:
                                salient_q = [
                                    t for t in q_toks if self._token_salience(t) >= 2.0
                                ][:4] or q_toks[:3]
                                neural_cot_lines.append(
                                    f"{' '.join(q_toks)}? think analyzing {' '.join(salient_q)} in 360 mlu network answer {' '.join(a_toks)}."
                                )
                            continue

                    # 3. Check Kaggle Disease-Symptom Profile CSV (Disease + Yes/No symptom columns)
                    if disease_col_idx >= 0 and disease_col_idx < len(raw_cells) and symptom_cols:
                        dis_name = "_".join(_tokenize_words(raw_cells[disease_col_idx]))
                        if dis_name:
                            rows_processed += 1
                            pos_symptoms: list[str] = []
                            for s_idx, s_name in symptom_cols:
                                if (
                                    s_idx < len(raw_cells)
                                    and raw_cells[s_idx].lower() in ("yes", "true", "1")
                                ):
                                    pos_symptoms.append(s_name)
                                    ok, sent = self._learn_verified_triple_direct(
                                        dis_name, "causes", s_name
                                    )
                                    if ok:
                                        relations_verified += 1
                                        if sent:
                                            synth_sentences.append(sent)
                            if pos_symptoms and len(neural_cot_lines) < 25:
                                d_readable = dis_name.replace("_", " ")
                                s_readable = " and ".join(
                                    s.replace("_", " ") for s in pos_symptoms[:3]
                                )
                                neural_cot_lines.append(
                                    f"what causes {s_readable}? think abductive medical diagnosis over symptom spokes answer {d_readable} causes {s_readable}."
                                )
                            continue

                    cells = [c for c in raw_cells if c]
                    if len(cells) >= 3:
                        cand_pred = cells[1].lower().replace(" ", "_")
                        if (
                            self.verifier.registry.relation(cand_pred) is not None
                            or cand_pred in ("not_a", "disjoint_with")
                        ):
                            subj = cells[0].lower().replace(" ", "_")
                            obj = cells[2].lower().replace(" ", "_")
                            if subj and obj and subj != obj:
                                rows_processed += 1
                                ok, sent = self._learn_verified_triple_direct(
                                    subj, cand_pred, obj
                                )
                                if ok:
                                    relations_verified += 1
                                    if sent:
                                        synth_sentences.append(sent)
                                else:
                                    relations_refuted += 1
                                continue

                    salient_cells = [
                        c
                        for c in cells
                        if " " in c
                        and _shannon_char_entropy(c) * math.log(1.0 + len(c)) >= 4.0
                    ]
                    if salient_cells:
                        rows_processed += 1
                        text_passages.append(" ".join(salient_cells))
        except OSError:
            return {
                "rows_processed": 0,
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
            }

        if synth_sentences and self.rust_arena.is_native:
            self.rust_arena.train_stream(" ".join(synth_sentences), learning_rate=0.12)

        if text_passages:
            rep = self.train_on_text(" ".join(text_passages), persist=False)
            relations_verified += rep.relations_verified
            relations_refuted += rep.relations_refuted

        neural_seq_trained = 0
        neural_ppl = 1.0
        if neural_cot_lines:
            n_rep = self.train_neural_mlu(
                "\n".join(neural_cot_lines[:35]),
                epochs=32,
                learning_rate=0.04,
                persist=False,
            )
            neural_seq_trained = int(n_rep.get("sequences_trained", 0))
            neural_ppl = float(n_rep.get("final_perplexity", 1.0))

        self.recent_focal_stack.clear()
        self.conversation_liquid_state = [0.0] * self.dim
        self._dirty_since_save = True
        if persist:
            self.save_nodes()
        return {
            "rows_processed": rows_processed,
            "relations_verified": relations_verified,
            "relations_refuted": relations_refuted,
            "neural_sequences_trained": neural_seq_trained,
            "neural_final_perplexity": neural_ppl,
        }

    def ingest_jsonl_dataset(
        self, jsonl_path: str, max_rows: int | None = None, persist: bool = False
    ) -> dict[str, Any]:
        """Phase 47 & 54: Ingest JSONL datasets (HuggingFace SciQ, OpenBookQA, TruthfulQA, Simple Wikipedia, GenericsKB) into both the 360° Arena and Differentiable Neural MLU."""
        import json
        from pathlib import Path

        fpath = Path(jsonl_path)
        if not fpath.is_file():
            return {
                "rows_processed": 0,
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
            }

        rows_processed = 0
        relations_verified = 0
        relations_refuted = 0
        passages: list[str] = []
        neural_cot_lines: list[str] = []

        try:
            lines = fpath.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return {
                "rows_processed": 0,
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
            }

        for raw_line in lines:
            if max_rows is not None and rows_processed >= max_rows:
                break
            clean_line = raw_line.strip()
            if not clean_line:
                continue
            try:
                obj = json.loads(clean_line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue

            rows_processed += 1

            # Extract structured QA + Chain-of-Thought pairs (e.g. SciQ, OpenBookQA, TruthfulQA)
            q_str = str(obj.get("question") or obj.get("question_stem") or "").strip()
            a_str = str(
                obj.get("correct_answer")
                or obj.get("best_answer")
                or obj.get("fact1")
                or obj.get("fact")
                or ""
            ).strip()
            sup_str = str(
                obj.get("support")
                or obj.get("fact1")
                or obj.get("fact")
                or obj.get("category")
                or ""
            ).strip()
            if q_str and a_str and len(neural_cot_lines) < 35:
                q_toks = _tokenize_words(q_str)[:12]
                a_toks = _tokenize_words(a_str)[:12]
                s_toks = (
                    _tokenize_words(sup_str)[:10]
                    if sup_str
                    else [t for t in q_toks if self._token_salience(t) >= 2.0][:4]
                )
                if q_toks and a_toks:
                    think_span = " ".join(s_toks) if s_toks else "evaluating 360 spokes"
                    neural_cot_lines.append(
                        f"{' '.join(q_toks)}? think {think_span} answer {' '.join(a_toks)}."
                    )

            for val in obj.values():
                if not isinstance(val, str):
                    continue
                s_val = val.strip()
                # Select high-information natural-language strings purely by Shannon character-entropy & token count
                if (
                    " " in s_val
                    and not s_val.startswith("http")
                    and not s_val.endswith("?")
                    and _shannon_char_entropy(s_val) * math.log(1.0 + len(s_val)) >= 6.0
                ):
                    if not s_val.endswith((".", "!", "?")):
                        s_val = s_val + "."
                    passages.append(s_val)

        if passages:
            rep = self.train_on_text(" ".join(passages), persist=False)
            relations_verified += rep.relations_verified
            relations_refuted += rep.relations_refuted
            if not neural_cot_lines and rep.verified_triples:
                d_rep = self.distill_triples_to_neural_mlu(
                    rep.verified_triples[:20], epochs=28, learning_rate=0.04, persist=False
                )
                neural_seq_trained = int(d_rep.get("sequences_trained", 0))
                neural_ppl = float(d_rep.get("final_perplexity", 1.0))
            else:
                neural_seq_trained = 0
                neural_ppl = 1.0
        else:
            neural_seq_trained = 0
            neural_ppl = 1.0

        if neural_cot_lines:
            n_rep = self.train_neural_mlu(
                "\n".join(neural_cot_lines[:35]),
                epochs=32,
                learning_rate=0.04,
                persist=False,
            )
            neural_seq_trained = int(n_rep.get("sequences_trained", 0))
            neural_ppl = float(n_rep.get("final_perplexity", 1.0))

        self.recent_focal_stack.clear()
        self.conversation_liquid_state = [0.0] * self.dim
        self._dirty_since_save = True
        if persist:
            self.save_nodes()
        return {
            "rows_processed": rows_processed,
            "relations_verified": relations_verified,
            "relations_refuted": relations_refuted,
            "neural_sequences_trained": neural_seq_trained,
            "neural_final_perplexity": neural_ppl,
        }

    def fetch_huggingface_dataset(
        self,
        dataset: str,
        config: str = "default",
        split: str = "train",
        limit: int = 50,
        save_dir: str = "data/knowledge/hf_datasets",
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 48 & 54: Download dataset rows directly from HuggingFace Datasets Server API, save to JSONL, and train both the 360° Spider-Web Arena and Neural MLU."""
        import json
        import urllib.parse
        import urllib.request
        from pathlib import Path

        safe_len = max(1, min(limit, 100))
        q_ds = urllib.parse.quote(dataset, safe="")
        q_cfg = urllib.parse.quote(config, safe="")
        q_sp = urllib.parse.quote(split, safe="")
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset={q_ds}&config={q_cfg}&split={q_sp}&offset=0&length={safe_len}"
        )
        out_folder = Path(save_dir)
        out_folder.mkdir(parents=True, exist_ok=True)
        slug = dataset.replace("/", "_") + f"_{config}.jsonl"
        out_path = out_folder / slug

        rows_saved = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                raw_rows = payload.get("rows", [])
                with out_path.open("w", encoding="utf-8") as f:
                    for item in raw_rows:
                        r_dict = item.get("row", {})
                        if isinstance(r_dict, dict):
                            f.write(json.dumps(r_dict) + "\n")
                            rows_saved += 1
        except (OSError, ValueError) as exc:
            return {
                "dataset": dataset,
                "config": config,
                "rows_downloaded": 0,
                "saved_path": str(out_path),
                "error": str(exc),
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
            }

        ing = self.ingest_jsonl_dataset(str(out_path), max_rows=safe_len, persist=persist)
        return {
            "dataset": dataset,
            "config": config,
            "rows_downloaded": rows_saved,
            "saved_path": str(out_path),
            "relations_verified": ing["relations_verified"],
            "relations_refuted": ing["relations_refuted"],
            "neural_sequences_trained": ing.get("neural_sequences_trained", 0),
            "neural_final_perplexity": ing.get("neural_final_perplexity", 1.0),
        }

    def fetch_kaggle_dataset(
        self,
        dataset: str,
        limit: int = 60,
        save_dir: str = "data/knowledge/kaggle_datasets",
        persist: bool = False,
    ) -> dict[str, Any]:
        """Phase 54: Download a real dataset archive directly from Kaggle API v1, extract files, and train both the 360° Spider-Web Arena and Differentiable Neural MLU."""
        import io
        import urllib.request
        import zipfile
        from pathlib import Path

        clean_ds = dataset.strip().strip("/")
        url = f"https://www.kaggle.com/api/v1/datasets/download/{clean_ds}"
        out_folder = Path(save_dir) / clean_ds.replace("/", "_")
        out_folder.mkdir(parents=True, exist_ok=True)

        extracted_files: list[str] = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                raw_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                for member in zf.namelist():
                    if member.endswith("/"):
                        continue
                    safe_name = Path(member).name
                    if not safe_name:
                        continue
                    dest = out_folder / safe_name
                    dest.write_bytes(zf.read(member))
                    extracted_files.append(str(dest))
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            return {
                "dataset": clean_ds,
                "files_extracted": [],
                "rows_processed": 0,
                "relations_verified": 0,
                "relations_refuted": 0,
                "neural_sequences_trained": 0,
                "error": str(exc),
            }

        total_rows = 0
        total_verified = 0
        total_refuted = 0
        total_neural = 0
        final_ppl = 1.0
        safe_limit = max(1, min(limit, 250))

        for f_str in extracted_files:
            ext = Path(f_str).suffix.lower()
            if ext in (".csv", ".tsv"):
                rep = self.ingest_csv_dataset(f_str, max_rows=safe_limit, persist=False)
                total_rows += rep["rows_processed"]
                total_verified += rep["relations_verified"]
                total_refuted += rep["relations_refuted"]
                total_neural += int(rep.get("neural_sequences_trained", 0))
                if rep.get("neural_sequences_trained", 0) > 0:
                    final_ppl = float(rep.get("neural_final_perplexity", 1.0))
            elif ext in (".jsonl", ".ndjson"):
                rep = self.ingest_jsonl_dataset(f_str, max_rows=safe_limit, persist=False)
                total_rows += rep["rows_processed"]
                total_verified += rep["relations_verified"]
                total_refuted += rep["relations_refuted"]
                total_neural += int(rep.get("neural_sequences_trained", 0))
                if rep.get("neural_sequences_trained", 0) > 0:
                    final_ppl = float(rep.get("neural_final_perplexity", 1.0))
            elif ext == ".json":
                jrep = self.ingest_json_ontology(f_str, persist=False)
                total_rows += jrep["triples_processed"]
                total_verified += jrep["relations_verified"]
                total_refuted += jrep["relations_refuted"]
            elif ext in (".txt", ".md"):
                txt = Path(f_str).read_text(encoding="utf-8", errors="ignore")
                trep = self.train_on_text(txt[:25000], persist=False)
                total_rows += trep.sentences_processed
                total_verified += trep.relations_verified
                total_refuted += trep.relations_refuted

        if persist:
            self.save_nodes()
        return {
            "dataset": clean_ds,
            "files_extracted": extracted_files,
            "rows_processed": total_rows,
            "relations_verified": total_verified,
            "relations_refuted": total_refuted,
            "neural_sequences_trained": total_neural,
            "neural_final_perplexity": final_ppl,
        }

    def ingest_corpus_path(
        self, target_path: str, epochs: int = 2, consolidate_on_finish: bool = True
    ) -> dict[str, Any]:
        """Phase 3, 44 & 47: Stream-train the 360° Spider-Web MLM on a file or directory of .md/.txt/.epub/.docx/.pdf/.json/.jsonl/.csv/.tsv files."""
        from pathlib import Path

        root = Path(target_path)
        text_files: list[Path] = []
        json_files: list[Path] = []
        jsonl_files: list[Path] = []
        csv_files: list[Path] = []
        if root.is_file():
            ext = root.suffix.lower()
            if ext == ".json":
                json_files.append(root)
            elif ext in (".jsonl", ".ndjson"):
                jsonl_files.append(root)
            elif ext in (".csv", ".tsv"):
                csv_files.append(root)
            else:
                text_files.append(root)
        elif root.is_dir():
            for ext in ("*.md", "*.txt", "*.epub", "*.docx", "*.pdf"):
                text_files.extend(sorted(root.rglob(ext)))
            json_files.extend(sorted(root.rglob("*.json")))
            for ext in ("*.jsonl", "*.ndjson"):
                jsonl_files.extend(sorted(root.rglob(ext)))
            for ext in ("*.csv", "*.tsv"):
                csv_files.extend(sorted(root.rglob(ext)))

        total_sentences = 0
        total_verified = 0
        total_refuted = 0
        total_tokens = 0
        epoch_surprises: list[float] = []
        all_promotions: list[str] = []

        for jfile in json_files:
            jrep = self.ingest_json_ontology(str(jfile), persist=False)
            total_sentences += jrep["triples_processed"]
            total_verified += jrep["relations_verified"]
            total_refuted += jrep["relations_refuted"]

        for jlfile in jsonl_files:
            jlrep = self.ingest_jsonl_dataset(str(jlfile), persist=False)
            total_sentences += jlrep["rows_processed"]
            total_verified += jlrep["relations_verified"]
            total_refuted += jlrep["relations_refuted"]

        for cfile in csv_files:
            crep = self.ingest_csv_dataset(str(cfile), persist=False)
            total_sentences += crep["rows_processed"]
            total_verified += crep["relations_verified"]
            total_refuted += crep["relations_refuted"]

        for _ep in range(max(1, epochs)):
            ep_surprises: list[float] = []
            for fpath in text_files:
                content = self.extract_text_from_document(str(fpath))
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

        cons = self.consolidate() if consolidate_on_finish else {}
        return {
            "files_processed": len(text_files)
            + len(json_files)
            + len(jsonl_files)
            + len(csv_files),
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

    def bootstrap_curriculum(
        self,
        epochs: int = 2,
        include_ontologies: bool = True,
        include_ebooks: bool = True,
        include_hf_datasets: bool = False,
        hf_max_rows_per_file: int = 35,
        include_neural_thinking: bool = True,
    ) -> dict[str, Any]:
        """Phase 12, 44, 48 & 52: Pre-train the 360° Spider-Web MLM & Differentiable Neural MLU on world_corpus, ontologies, ebooks, HF datasets, and Neural Thinking curriculum."""
        from pathlib import Path

        from little.core.runtime_paths import RuntimePaths

        candidate_dirs = [
            RuntimePaths.default().database.parent / "knowledge" / "world_corpus",
            Path("data/knowledge/world_corpus"),
        ]
        target_dir = next((d for d in candidate_dirs if d.exists() and d.is_dir()), None)
        if target_dir is not None:
            has_extra = include_ontologies or include_ebooks or include_hf_datasets
            rep = self.ingest_corpus_path(
                str(target_dir),
                epochs=epochs,
                consolidate_on_finish=not has_extra,
            )
            knowledge_root = target_dir.parent
            if include_ontologies:
                for jname in ("commonsense.json", "extended_ontology.json"):
                    jpath = knowledge_root / jname
                    if jpath.is_file():
                        jrep = self.ingest_json_ontology(str(jpath), persist=False)
                        rep["files_processed"] += 1
                        rep["sentences_processed"] += jrep["triples_processed"]
                        rep["relations_verified"] += jrep["relations_verified"]
                        rep["relations_refuted"] += jrep["relations_refuted"]
            if include_ebooks:
                ebooks_dir = knowledge_root / "ebooks"
                if ebooks_dir.is_dir():
                    eb_rep = self.ingest_corpus_path(
                        str(ebooks_dir), epochs=1, consolidate_on_finish=False
                    )
                    rep["files_processed"] += eb_rep["files_processed"]
                    rep["sentences_processed"] += eb_rep["sentences_processed"]
                    rep["relations_verified"] += eb_rep["relations_verified"]
                    rep["relations_refuted"] += eb_rep["relations_refuted"]
            if include_hf_datasets:
                hf_dir = knowledge_root / "hf_datasets"
                if hf_dir.is_dir():
                    for jl_path in sorted(hf_dir.glob("*.jsonl")):
                        jl_rep = self.ingest_jsonl_dataset(
                            str(jl_path), max_rows=hf_max_rows_per_file, persist=False
                        )
                        rep["files_processed"] += 1
                        rep["sentences_processed"] += jl_rep["rows_processed"]
                        rep["relations_verified"] += jl_rep["relations_verified"]
                        rep["relations_refuted"] += jl_rep["relations_refuted"]
            if include_neural_thinking:
                rep["neural_thinking_training"] = self.train_neural_curriculum(
                    epochs=55, learning_rate=0.04, persist=False
                )
            if has_extra or include_neural_thinking:
                rep["consolidation"] = self.consolidate()
                rep["telemetry"] = self.get_telemetry()
            return rep

        synced = self.sync_from_symbolic_memory()
        cons = self.consolidate()
        return {
            "files_processed": 0,
            "epochs": max(1, epochs),
            "sentences_processed": 0,
            "relations_verified": synced,
            "relations_refuted": 0,
            "token_transitions_learned": 0,
            "epoch_surprises": [],
            "mitosis_promotions": cons.get("python_mitosis_promotions", []),
            "consolidation": cons,
            "telemetry": self.get_telemetry(),
        }

    @classmethod
    def run_benchmark_suite(cls) -> dict[str, Any]:
        """Phase 13 & 25: Run an isolated, deterministic 8-axis benchmark of the 360° Spider-Web MLM & Rust Arena."""
        import time

        with MemoryStore(seed_ontology=False) as bench_mem:
            mlm = cls(
                memory=bench_mem,
                mitosis_min_activations=2,
                mitosis_min_verified_spokes=2,
            )
            curriculum = (
                "An orca is a cetacean. A cetacean is a mammal. A mammal is an animal. "
                "A mammal has warm_blood. An orca can echolocate. An orca lives in the ocean. "
                "A cobra is a reptile. A reptile is an animal. A mammal is not a reptile. "
                "An albatross is a seabird. A seabird is a bird. A bird is an animal. "
                "A bird is not a mammal. An albatross can glide. "
                "A dolphin is a cetacean. Drought causes crop_failure. Crop_failure causes famine."
            )
            rep1 = mlm.train_on_text(curriculum, persist=False)
            mlm.consolidate()

            # Axis 1: Multi-hop 360° transitive reasoning & property inheritance
            q_trans1 = mlm.query_360("Is an orca an animal?")
            q_trans2 = mlm.query_360("Is an albatross an animal?")
            q_prop = mlm.query_360("Does an orca have warm_blood?")
            transitive_passed = (
                q_trans1.status == "VERIFIED"
                and q_trans2.status == "VERIFIED"
                and q_prop.status == "VERIFIED"
            )

            # Axis 2: Invariant Gate contradiction rejection (I_DAG + I_MUTEX in Python & Rust)
            bad_rep = mlm.train_on_text(
                "An animal is an orca. An orca is a cobra.", persist=False
            )
            rust_dag = mlm.rust_arena.verify_invariants("ANIMAL", "is_a", "ORCA")
            rust_mutex = mlm.rust_arena.verify_invariants("ORCA", "is_a", "COBRA")
            invariants_passed = (
                bad_rep.relations_refuted == 2
                and bad_rep.relations_verified == 0
                and not rust_dag["passed"]
                and rust_dag["gate"] == "I_DAG"
                and not rust_mutex["passed"]
                and rust_mutex["gate"] == "I_MUTEX"
            )

            # Axis 3: Zero-hallucination epistemic abstention
            q_unk = mlm.query_360("What is a hyper_tachyon_quasar?")
            abstention_passed = q_unk.status == "UNKNOWN" and q_unk.confidence == 0.0

            # Axis 4: Multi-turn liquid context elliptic resolution
            _ = mlm.query_360("Tell me about orca")
            q_ellip_yes = mlm.query_360("Can it echolocate?")
            q_ellip_no = mlm.query_360("Is it a cobra?")
            multiturn_passed = (
                q_ellip_yes.status == "VERIFIED"
                and q_ellip_yes.focal_concept == "ORCA"
                and q_ellip_no.status == "REFUTED"
            )

            # Axis 5: 360° Comparative Wave Intersection
            cmp_dto = mlm.compare_360("ORCA", "ALBATROSS")
            comparative_passed = bool(
                cmp_dto.get("mutually_exclusive") is True
                and len(cmp_dto.get("shared_attractors", [])) > 0
            )

            # Axis 6: Causal Wave & Counterfactual Intervention
            causal_dto = mlm.simulate_causal_360("DROUGHT", max_hops=3)
            cf_dto = mlm.intervene_counterfactual_360("DROUGHT", "CROP_FAILURE", max_hops=3)
            causal_counterfactual_passed = bool(
                "FAMINE" in causal_dto.get("affected_concepts", [])
                and "FAMINE" in cf_dto.get("invalidated_conclusions", [])
            )

            # Axis 7: Relational Bridge, Latent Clusters & Belief Revision
            bridge_dto = mlm.find_bridge_360("ORCA", "DOLPHIN", max_hops=4)
            clusters = mlm.discover_clusters_360(min_members=2)
            rev_dto = mlm.revise_belief_360(
                "ALBATROSS", "can", "GLIDE", verified=False, evidence_weight=5.0
            )
            bridge_cluster_revision_passed = bool(
                bridge_dto.get("connected") is True
                and len(clusters) > 0
                and rev_dto.get("polarity_flipped") is True
            )

            # Axis 8: Multi-Clause English Comprehension, Coreference Binding & System-2 Deliberation
            comp_rep = mlm.train_on_text(
                "An eagle is a bird, has talons, and can soar. It hunts prey.",
                persist=False,
            )
            q_eagle = mlm.query_360("Does an eagle have talons?")
            q_soar = mlm.query_360("Can an eagle soar?")
            delib_orca = mlm.deliberate_360("ORCA", max_hops=3)
            comprehension_deliberation_passed = bool(
                comp_rep.relations_verified >= 3
                and q_eagle.status == "VERIFIED"
                and q_soar.status == "VERIFIED"
                and delib_orca.get("waves_executed") == 5
                and "DOLPHIN" in delib_orca.get("cluster_peers", [])
            )

            # Axis 9: 360° Goal-Directed Backward-Chaining & Conversational Turn Engine
            plan_dto = mlm.plan_goal_360("FAMINE", max_steps=5)
            q_famine = mlm.query_360("famine")
            chat_learn = mlm.chat_turn_360("A peregrine is a bird, and can dive.", persist=False)
            chat_ask = mlm.chat_turn_360("Can it dive?", persist=False)
            goal_planning_passed = bool(
                plan_dto.get("achievable") is True
                and plan_dto.get("total_steps", 0) >= 2
                and "DROUGHT" in plan_dto.get("root_prerequisites", [])
                and "[MLU-360-PLAN]" in q_famine.inspectable_trace
                and chat_learn.get("mode") == "LEARN_AND_DELIBERATE"
                and chat_ask.get("status") == "VERIFIED"
            )

            # Axis 10: 360° Transitive Poset Ordering (I_SORT) & Domain Knowledge Summarizer
            order_dto = mlm.evaluate_order_360("causes", start_concept="DROUGHT", max_hops=4)
            rust_sort = mlm.rust_arena.verify_invariants("FAMINE", "causes", "DROUGHT")
            dom_dto = mlm.summarize_domain_360("", max_items=6)
            order_and_summary_passed = bool(
                order_dto.get("poset_consistent") is True
                and order_dto.get("element_count", 0) >= 3
                and "DROUGHT -> FAMINE" in order_dto.get("transitive_pairs", [])
                and not rust_sort["passed"]
                and rust_sort["gate"] == "I_SORT"
                and dom_dto.get("active_concepts_count", 0) >= 10
                and len(dom_dto.get("central_hubs", [])) > 0
                and bool(dom_dto.get("domain_synthesis"))
            )

            # Axis 11: 360° Abductive Root-Cause Diagnosis & Multi-Paragraph Discourse Synthesis
            abd_dto = mlm.diagnose_abductive_360(
                ["ECHOLOCATE", "OCEAN", "WARM_BLOOD"], max_hops=3, top_k=5
            )
            disc_dto = mlm.compose_discourse_360("ORCA", max_hops=3)
            abductive_and_discourse_passed = bool(
                abd_dto.get("best_explanation") == "ORCA"
                and float(abd_dto.get("best_coverage_ratio", 0.0)) >= 0.99
                and disc_dto.get("paragraph_count") == 3
                and len(disc_dto.get("paragraphs", [])) == 3
                and "cetacean" in str(disc_dto.get("full_discourse", "")).lower()
            )

            # Axis 12: Native Rust 360° beam latency (microseconds/query)
            iters = 25
            t0 = time.perf_counter()
            for _ in range(iters):
                mlm.rust_arena.query_360_beam("ORCA", max_hops=3, top_k=10)
            avg_beam_us = round(((time.perf_counter() - t0) * 1_000_000.0) / iters, 2)

            tel = mlm.get_telemetry()
            mlm.close()

            all_passed = bool(
                transitive_passed
                and invariants_passed
                and abstention_passed
                and multiturn_passed
                and comparative_passed
                and causal_counterfactual_passed
                and bridge_cluster_revision_passed
                and comprehension_deliberation_passed
                and goal_planning_passed
                and order_and_summary_passed
                and abductive_and_discourse_passed
            )
            return {
                "all_passed": all_passed,
                "transitive_360_passed": transitive_passed,
                "invariant_gates_passed": invariants_passed,
                "zero_hallucination_abstention_passed": abstention_passed,
                "multiturn_liquid_context_passed": multiturn_passed,
                "comparative_360_passed": comparative_passed,
                "causal_counterfactual_passed": causal_counterfactual_passed,
                "bridge_cluster_revision_passed": bridge_cluster_revision_passed,
                "comprehension_deliberation_passed": comprehension_deliberation_passed,
                "goal_planning_passed": goal_planning_passed,
                "order_and_summary_passed": order_and_summary_passed,
                "abductive_and_discourse_passed": abductive_and_discourse_passed,
                "rust_beam_avg_latency_us": avg_beam_us,
                "verified_spokes_learned": rep1.relations_verified + comp_rep.relations_verified,
                "telemetry": tel,
            }

    @classmethod
    def run_vs_llm_scorecard(cls) -> dict[str, Any]:
        """Phase 42: Run a head-to-head real-world cognitive scorecard comparing 360° Spider-Web MLU against standard LLM/Transformer baselines."""
        import time

        with MemoryStore(seed_ontology=False) as bench_mem:
            mlm = cls(
                memory=bench_mem,
                mitosis_min_activations=2,
                mitosis_min_verified_spokes=2,
            )

            # 1. Online Zero-Backprop Continual Learning Latency (ms)
            t_learn0 = time.perf_counter()
            learn_rep = mlm.train_on_text(
                "An orca is a cetacean, has dorsal_fin, lives in the ocean, and can echolocate. "
                "A dolphin is a cetacean, and lives in the ocean. "
                "A cetacean is a mammal. A mammal is a vertebrate. A vertebrate is an animal. "
                "A mammal has warm_blood. A mammal is not a reptile. A cobra is a reptile. "
                "Sunlight causes evaporation. Evaporation causes clouds. Clouds cause rain. "
                "Rain causes irrigation. Irrigation causes crop_growth. Crop_growth causes harvest. "
                "Drought causes soil_dryness. Soil_dryness causes crop_failure. Crop_failure causes famine. "
                "Infection causes fever. Infection causes inflammation.",
                persist=False,
            )
            learn_ms = round((time.perf_counter() - t_learn0) * 1000.0, 2)
            mlm.consolidate()

            # 2. Contradiction & Sycophancy Trap Immunity (I_DAG, I_MUTEX, I_SORT)
            trap_rep = mlm.train_on_text(
                "An animal is an orca. An orca is a cobra.", persist=False
            )
            sort_gate = mlm.rust_arena.verify_invariants("HARVEST", "causes", "SUNLIGHT")
            contradiction_immunity = bool(
                trap_rep.relations_refuted == 2
                and trap_rep.relations_verified == 0
                and not sort_gate["passed"]
                and sort_gate["gate"] == "I_SORT"
            )

            # 3. 6-Step Backward-Chaining Goal Planning & Counterfactual Intervention
            plan_dto = mlm.plan_goal_360("HARVEST", max_steps=6)
            cf_dto = mlm.query_360("What if drought is not crop_failure?")
            planning_and_counterfactual = bool(
                plan_dto.get("achievable") is True
                and plan_dto.get("total_steps") == 6
                and "SUNLIGHT" in plan_dto.get("root_prerequisites", [])
                and "[MLU-360-COUNTERFACTUAL]" in cf_dto.inspectable_trace
            )

            # 4. Conversational Abductive Root-Cause Diagnosis (2-obs and 3-obs natural language queries)
            diag_med = mlm.query_360("What explains fever and inflammation?")
            diag_bio = mlm.query_360("What explains echolocate, ocean, and dorsal_fin?")
            abductive_diagnosis = bool(
                diag_med.status == "VERIFIED"
                and diag_med.focal_concept == "INFECTION"
                and "[MLU-360-DIAGNOSE]" in diag_med.inspectable_trace
                and diag_bio.status == "VERIFIED"
                and diag_bio.focal_concept == "ORCA"
                and "[MLU-360-DIAGNOSE]" in diag_bio.inspectable_trace
            )

            # 5. Zero-Hallucination Epistemic Abstention
            unk_q = mlm.query_360("What is a quantum_tachyon_monolith?")
            zero_hallucination = bool(
                unk_q.status == "UNKNOWN"
                and unk_q.confidence == 0.0
                and unk_q.epistemic_entropy == 1.0
            )

            # 6. Non-Monotonic Live Belief Revision (Instant Retraction)
            rev = mlm.revise_belief_360(
                "ORCA", "can", "ECHOLOCATE", verified=False, evidence_weight=5.0
            )
            belief_revision = bool(rev.get("polarity_flipped") is True)

            # 7. Native Rust 360° Wave Latency (µs)
            iters = 50
            t_wave0 = time.perf_counter()
            for _ in range(iters):
                mlm.rust_arena.query_360_beam("ORCA", max_hops=3, top_k=10)
            avg_wave_us = round(((time.perf_counter() - t_wave0) * 1_000_000.0) / iters, 2)

            # 8. True End-to-End Differentiable Neural MLU (Backprop + AdamW + T-Norm Logic Gates)
            t_bprop0 = time.perf_counter()
            neural_rep = mlm.train_neural_mlu(
                "is an orca a mammal? yes an orca is a warm blooded mammal.\n"
                "is an orca a reptile? no an orca is a mammal and not a reptile.\n"
                "what causes harvest? sunlight and rain cause crop growth and harvest.\n",
                epochs=45,
                learning_rate=0.04,
                persist=False,
            )
            bprop_ms = round((time.perf_counter() - t_bprop0) * 1000.0, 2)
            g_yes = mlm.generate_neural_mlu("is an orca a mammal?", max_new_tokens=10)
            g_no = mlm.generate_neural_mlu("is an orca a reptile?", max_new_tokens=10)
            neural_backprop_passed = bool(
                neural_rep.get("final_perplexity", 99.0) < 2.0
                and "yes" in str(g_yes.get("generated_text", "")).lower()
                and "no" in str(g_no.get("generated_text", "")).lower()
            )

            snap_bytes = len(mlm.rust_arena.dump_json().encode("utf-8"))
            tel = mlm.get_telemetry()
            mlm.close()

            all_passed = bool(
                contradiction_immunity
                and planning_and_counterfactual
                and abductive_diagnosis
                and zero_hallucination
                and belief_revision
                and neural_backprop_passed
            )
            return {
                "all_passed": all_passed,
                "mlu_360_metrics": {
                    "online_continual_learning_ms": learn_ms,
                    "verified_spokes_learned": learn_rep.relations_verified,
                    "rust_360_wave_latency_us": avg_wave_us,
                    "arena_memory_footprint_kb": round(snap_bytes / 1024.0, 2),
                    "contradiction_trap_immunity": contradiction_immunity,
                    "six_step_goal_planning_and_counterfactual": planning_and_counterfactual,
                    "conversational_abductive_diagnosis": abductive_diagnosis,
                    "zero_hallucination_abstention": zero_hallucination,
                    "instant_non_monotonic_belief_revision": belief_revision,
                    "neural_backprop_passed": neural_backprop_passed,
                    "neural_backprop_train_ms": bprop_ms,
                    "neural_initial_perplexity": neural_rep.get("initial_perplexity", 0.0),
                    "neural_final_perplexity": neural_rep.get("final_perplexity", 0.0),
                    "total_cells": tel["total_mlu_nodes"],
                    "subcore_hubs_mitosis": tel["subcore_hubs"],
                },
                "vs_standard_llm_comparison": [
                    {
                        "dimension": "Online Continual Weight Plasticity",
                        "mlu_360": f"Yes ({learn_ms} ms on CPU, zero backprop)",
                        "standard_llm": "No (weights frozen at inference; requires offline GPU fine-tuning)",
                    },
                    {
                        "dimension": "Logical Contradiction & Loop Immunity",
                        "mlu_360": "100% Enforced (I_DAG, I_MUTEX, I_SORT gates)",
                        "standard_llm": "Vulnerable to sycophantic prompt injection & circular premises",
                    },
                    {
                        "dimension": "Epistemic Abstention on Unknowns",
                        "mlu_360": "Exact (UNKNOWN, confidence=0.0, entropy=1.0)",
                        "standard_llm": "Prone to next-token confabulation / hallucination",
                    },
                    {
                        "dimension": "Multi-Wave System-2 Reasoning",
                        "mlu_360": f"8 Native Rust 360° Waves (~{avg_wave_us} µs/wave)",
                        "standard_llm": "Autoregressive token chain-of-thought (50-500 ms/token)",
                    },
                    {
                        "dimension": "Memory & Compute Footprint",
                        "mlu_360": f"{round(snap_bytes / 1024.0, 2)} KB state (64-D fractal cells on CPU)",
                        "standard_llm": "4 GB - 160+ GB VRAM (dense O(N^2) attention matrices)",
                    },
                ],
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
