"""Autonomous Closed-Loop Curiosity and Self-Study Engine.

Scans missing concept features and proposes analogical candidates. Sibling analogy
is retained as a hypothesis; independent document statements can support promotion.
Graph invariants check structural consistency, not external factual truth.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.active.self_study_policy import SelfStudyPolicy
from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    ProofTrace,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.inference.engine import InferenceEngine
from little.knowledge.registry import SchemaRegistry
from little.memory.store import MemoryStore


@dataclass
class DreamReport:
    """Consolidated summary of an autonomous dreaming / self-study session."""

    steps_attempted: int
    steps_persisted: int
    steps_rejected: int
    entropy_reduction: float
    rejection_reasons: list[str] = field(default_factory=list)
    persisted_claims: list[dict[str, str]] = field(default_factory=list)
    wiki_pages_synced: int = 0
    duration_ms: float = 0.0
    details: list[CuriosityStepResult] = field(default_factory=list)
    steps_hypothesized: int = 0
    sync_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HeldOutCanaryResult:
    """Online held-out canary verification result inspired by mini-AGI (heldout.py)."""

    passed: bool
    checked_count: int
    regressions: list[str] = field(default_factory=list)


@dataclass
class SleepConsolidationReport:
    """Summary of comprehensive offline sleep and memory consolidation cycle."""

    dream_report: DreamReport
    invariants_mined: list[dict[str, str]] = field(default_factory=list)
    clusters_induced: int = 0
    turns_compressed: int = 0
    cache_refreshed: bool = False
    duration_ms: float = 0.0
    summary_message: str = ""
    errors: list[str] = field(default_factory=list)
    canary_passed: bool = True
    canary_regressions: list[str] = field(default_factory=list)


@dataclass
class CuriosityStepResult:
    """Detailed trace and verdict of a single autonomous curiosity step."""

    target_concept: str
    missing_axis: str
    hypothesis: dict[str, str]  # {"subject": S, "predicate": P, "object": O}
    status: str  # "PERSISTED", "HYPOTHESIS", "REJECTED", "NO_ANALOGY"
    gate_verdicts: dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    trace: list[str] = field(default_factory=list)
    sync_errors: list[str] = field(default_factory=list)


class AutonomousSelfStudyEngine:
    """Self-directed agent that autonomously studies, hypothesizes, and verifies missing relations."""

    def __init__(
        self,
        memory: MemoryStore,
        registry: SchemaRegistry | None = None,
        policy: SelfStudyPolicy | None = None,
        wiki: Any | None = None,
        crawler: Any | None = None,
        corpus_paths: tuple[Path, ...] | None = None,
        balance_weight: float = 0.5,
        share_decay: float = 0.85,
    ) -> None:
        self.memory = memory
        self.registry = registry or SchemaRegistry.default()
        self.policy = policy or SelfStudyPolicy.default()
        self.inference = InferenceEngine(memory, registry=self.registry)
        self.wiki = wiki
        self.crawler = crawler
        self.corpus_paths = corpus_paths
        self.last_gate_verdicts: dict[str, bool] = {}
        self.balance_weight = float(balance_weight)
        self.share_decay = float(share_decay)
        self._recent_study_share: dict[str, float] = {}

    def _axis_predicates(self, axis: str) -> set[str]:
        """Resolve radial-axis membership from the versioned relation registry."""
        return self.registry.predicates(axis=axis)

    def _hierarchy_predicates(self) -> tuple[str, ...]:
        """Return configured acyclic predicates used to ascend the concept web."""
        return tuple(
            sorted(
                self.registry.predicates(
                    axis=self.policy.hierarchy_axis,
                    acyclic=True,
                )
            )
        )

    def _ancestor_ids(self, concept_id: str, predicates: set[str] | None = None) -> list[str]:
        selected = predicates or set(self._hierarchy_predicates())
        ancestors: list[str] = []
        for predicate in sorted(selected):
            for ancestor_id in self.inference.get_ancestor_ids(
                concept_id, predicate=predicate
            ):
                if ancestor_id not in ancestors:
                    ancestors.append(ancestor_id)
        return ancestors

    def _are_concepts_disjoint(self, concept_or_id1: str, concept_or_id2: str) -> bool:
        """Check if two concepts or any of their taxonomic ancestors are disjoint."""
        c1 = self.memory.get_concept(concept_or_id1)
        c2 = self.memory.get_concept(concept_or_id2)
        id1 = c1.id if c1 else concept_or_id1
        id2 = c2.id if c2 else concept_or_id2
        conflict = self.inference._check_disjoint_conflict(id1, id2)
        return conflict is not None

    def _get_experienced_concept_names(self) -> set[str]:
        if getattr(self, "_cached_experienced_names", None) is None:
            experienced: set[str] = set()
            for experience in self.memory.list_experiences(limit=None):
                for triple in experience.extracted_triples:
                    if isinstance(triple, dict):
                        s = str(triple.get("subject", "")).strip().lower()
                        o = str(triple.get("object", "")).strip().lower()
                        if s:
                            experienced.add(s)
                        if o:
                            experienced.add(o)
            self._cached_experienced_names = experienced
        return self._cached_experienced_names

    def _experience_covers(self, concept_name: str) -> bool:
        clean_name = concept_name.strip().lower()
        return clean_name in self._get_experienced_concept_names()

    def _axis_is_covered(self, concept_name: str, axis_name: str) -> bool:
        concept = self.memory.get_concept(concept_name)
        if not concept:
            return False
        return self._axis_is_covered_fast(concept, axis_name, self._get_experienced_concept_names())

    def _axis_is_covered_fast(self, concept: Any, axis_name: str, exp_names: set[str]) -> bool:
        axis = self.policy.axis(axis_name)
        if axis is None:
            return False
        if axis.coverage == "attributes":
            return bool(concept.attributes)
        if axis.coverage == "experience":
            return concept.name.strip().lower() in exp_names
        relations = self.memory.get_relations(subject_id=concept.id)
        active_preds = {
            relation.predicate.lower()
            for relation in relations
            if relation.weight_positive > relation.weight_negative
        }
        return bool(active_preds.intersection(self._axis_predicates(axis.name)))

    def compute_knot_entropy(self, concept_name: str) -> float:
        """Compute the missing radial spoke entropy H(K) in [0.0, 1.0]."""
        concept = self.memory.get_concept(concept_name)
        if not concept:
            return 1.0

        missing_axes = sum(
            not self._axis_is_covered(concept.name, axis_name)
            for axis_name in self.policy.axis_names
        )
        return missing_axes / float(len(self.policy.axes))

    def compute_all_knot_entropies(self) -> dict[str, float]:
        """Compute radial spoke entropy for all concepts, sorted by highest entropy first."""
        concepts = self.memory.list_concepts()
        exp_names = self._get_experienced_concept_names()

        # Batch load all active relations grouped by subject_id for sub-millisecond evaluation
        active_preds_by_subj: dict[str, set[str]] = {}
        try:
            with self.memory.transaction() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT subject_id, predicate FROM relations WHERE weight_positive > weight_negative"
                )
                for sid, pred in cur.fetchall():
                    active_preds_by_subj.setdefault(sid, set()).add(pred.lower())
        except (sqlite3.Error, RuntimeError, AttributeError):
            active_preds_by_subj = {}

        axis_pred_sets = {
            ax_name: set(self._axis_predicates(ax_name))
            for ax_name in self.policy.axis_names
        }

        entropies: dict[str, float] = {}
        total_axes = float(max(1, len(self.policy.axes)))

        for c in concepts:
            c_name_clean = c.name.strip().lower()
            if c_name_clean.startswith("__") and c_name_clean.endswith("__"):
                continue
            missing_count = 0
            subj_preds = active_preds_by_subj.get(c.id, set())

            for axis_name in self.policy.axis_names:
                axis = self.policy.axis(axis_name)
                if axis is None:
                    continue
                if axis.coverage == "attributes":
                    covered = bool(c.attributes)
                elif axis.coverage == "experience":
                    covered = c_name_clean in exp_names
                else:
                    target_preds = axis_pred_sets.get(axis_name, set())
                    covered = bool(subj_preds.intersection(target_preds))
                if not covered:
                    missing_count += 1

            entropies[c.name] = missing_count / total_axes

        return dict(sorted(entropies.items(), key=lambda kv: kv[1], reverse=True))

    def identify_missing_axes(self, concept_name: str) -> list[str]:
        """Identify which specific radial axes are currently unpopulated for a concept."""
        concept = self.memory.get_concept(concept_name)
        if not concept:
            return list(self.policy.axis_names)
        return [
            axis_name
            for axis_name in self.policy.axis_names
            if not self._axis_is_covered(concept.name, axis_name)
        ]

    def propose_analogical_spoke(
        self, target_concept: str, preferred_axis: str | None = None
    ) -> tuple[str, dict[str, str], list[str]] | None:
        """Ascend taxonomic DAG to parent hypernyms and transfer candidate spokes from siblings."""
        concept = self.memory.get_concept(target_concept)
        if not concept:
            return None

        trace: list[str] = [f"Initiating analogical spoke proposal for '{target_concept}'"]
        # Ascend taxonomic DAG
        ancestor_ids = self._ancestor_ids(concept.id)
        if not ancestor_ids:
            trace.append(f"No taxonomic ancestors found for '{target_concept}'")
            return None

        missing_axes = self.identify_missing_axes(target_concept)
        if preferred_axis:
            candidate_axes = [preferred_axis]
        else:
            candidate_axes = list(missing_axes)
            for ax in self.policy.axis_names:
                if ax not in candidate_axes:
                    candidate_axes.append(ax)

        # Iterate through axes to find transferable candidate spokes
        non_transferable = self.registry.predicates(disjoint=True) | set(self._hierarchy_predicates())
        for target_axis in candidate_axes:
            allowed_preds = self._axis_predicates(target_axis) - non_transferable
            if not allowed_preds:
                continue

            best_candidate: tuple[tuple[int, int], dict[str, str], str] | None = None
            best_score = (-1, -1)

            # Root hypernyms are too broad for meaningful analogical transfer
            _ROOT_EXCLUSIONS = frozenset({"entity", "physical entity", "abstract entity", "thing", "object", "item"})

            # Find siblings under ancestors that possess spokes in the target axis
            for anc_id in ancestor_ids:
                anc = self.memory.get_concept(anc_id)
                anc_name = anc.name if anc else anc_id
                if anc and anc.name.strip().lower() in _ROOT_EXCLUSIONS:
                    continue
                trace.append(f"Examining hypernym ancestor: '{anc_name}'")

                # Siblings are concepts that also have is_a -> anc_id
                for hierarchy_predicate in self._hierarchy_predicates():
                    sib_rels = self.memory.get_relations(
                        predicate=hierarchy_predicate, object_id=anc_id
                    )
                    for sr in sib_rels:
                        if sr.weight_positive <= sr.weight_negative:
                            continue
                        if sr.subject_id == concept.id:
                            continue  # Skip self
                        # Sibling must NOT be an ancestor of target concept
                        if sr.subject_id in ancestor_ids:
                            continue

                        # Check if sibling has candidate relations matching allowed_preds first
                        cand_rels = self.memory.get_relations(subject_id=sr.subject_id)
                        matching_rels = [
                            cr for cr in cand_rels
                            if cr.predicate in allowed_preds and cr.weight_positive > cr.weight_negative
                        ]
                        if not matching_rels:
                            continue

                        sib_concept = self.memory.get_concept(sr.subject_id)
                        if not sib_concept:
                            continue

                        # Sibling must NOT be disjoint with the target concept or its ancestors
                        if self._are_concepts_disjoint(concept.id, sib_concept.id):
                            continue

                        for cr in matching_rels:
                                target_obj = self.memory.get_concept(cr.object_id)
                                if not target_obj:
                                    continue

                                # Check if target already has positive evidence for this relation
                                existing = self.memory.get_relations(
                                    subject_id=concept.id,
                                    predicate=cr.predicate,
                                    object_id=target_obj.id,
                                )
                                if not any(r.weight_positive > r.weight_negative for r in existing):
                                    hyp = {
                                        "subject": target_concept,
                                        "predicate": cr.predicate,
                                        "object": target_obj.name,
                                        "donor": sib_concept.name,
                                    }
                                    score = (
                                        1 if cr.source_experience_id else 0,
                                        cr.weight_positive - cr.weight_negative,
                                    )
                                    if score > best_score:
                                        best_score = score
                                        best_candidate = (score, hyp, sib_concept.name)

                # Prioritize closer hypernym ancestor (e.g. bird before animal)
                if best_candidate:
                    break

            if best_candidate:
                _, hyp, donor = best_candidate
                trace.append(f"Targeting radial axis: {target_axis}")
                trace.append(
                    f"Found analogical donor: '{donor}' with ({hyp['subject']} {hyp['predicate']} {hyp['object']})"
                )
                return (target_axis, hyp, trace)

        trace.append("No sibling candidates with transferable spokes found.")
        return None

    def _corpus_documents(self) -> list[tuple[Path, str]]:
        from little.core.runtime_paths import RuntimePaths
        roots = self.corpus_paths
        if roots is None:
            roots = (RuntimePaths.default().world_corpus_directory,)
            if self.wiki is not None:
                roots += (Path(self.wiki.vault_dir),)
        documents = []
        seen = set()
        for root in roots:
            root = Path(root)
            paths = sorted(root.glob("*.md")) if root.is_dir() else [root]
            for path in paths:
                if path.name.endswith(".okf.md"):
                    continue
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                documents.append((resolved, path.read_text(encoding="utf-8")))
        return documents

    def propose_corpus_spoke(self, target_concept: str) -> tuple[str, dict[str, str], list[str]] | None:
        """Extract what an actual independent document asserts about the target."""
        parser = getattr(self, "_cached_parser", None)
        if parser is None:
            from little.language.parser import configured_parser
            parser = configured_parser()
            self._cached_parser = parser
        clean_target = target_concept.strip().lower()
        cache: dict[tuple[Path, int], dict[str, list[tuple[str, CandidateClaim]]]] = getattr(
            self, "_corpus_parse_cache", {}
        )
        self._corpus_parse_cache = cache

        for path, text in self._corpus_documents():
            if path.name.endswith(".okf.md"):
                continue
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            cache_key = (path, mtime_ns)
            subject_map = cache.get(cache_key)
            if subject_map is None:
                subject_map = {}
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    for triple in parser.parse_statement(sentence):
                        if triple.is_negative:
                            continue
                        claim = CandidateClaim(
                            triple.subject,
                            triple.predicate,
                            triple.object_,
                            self.policy.hypothesis_probability,
                            positive=True,
                        )
                        subj_key = claim.subject.strip().lower()
                        subject_map.setdefault(subj_key, []).append((sentence, claim))
                cache[cache_key] = subject_map

            for sentence, claim in subject_map.get(clean_target, ()):
                if self.memory.find_relation_by_names(claim.subject, claim.predicate, claim.object):
                    continue
                hypothesis = {"subject": claim.subject, "predicate": claim.predicate,
                              "object": claim.object, "source": str(path),
                              "source_text": sentence, "probability": str(claim.probability)}
                axis = next((axis.name for axis in self.policy.axes
                             if claim.predicate in self._axis_predicates(axis.name)), self.policy.default_axis)
                return axis, hypothesis, [f"Document assertion extracted from {path}: {sentence}"]
        return None

    def verify_invariant_gates(
        self, hypothesis: dict[str, str]
    ) -> tuple[bool, str]:
        """Return the stable two-value compatibility result for a hypothesis."""
        passed, reason, verdicts = self._verify_invariant_gates_detailed(hypothesis)
        self.last_gate_verdicts = verdicts
        return passed, reason

    def _verify_invariant_gates_detailed(
        self, hypothesis: dict[str, str]
    ) -> tuple[bool, str, dict[str, bool]]:
        """Check graph/schema consistency separately from evidence for the target."""
        verifier = getattr(self, "_cached_verifier", None)
        if verifier is None:
            from little.inference.invariant_gates import DeepSeekInvariantVerifier
            verifier = DeepSeekInvariantVerifier(self.memory, self.registry)
            self._cached_verifier = verifier
        subject = hypothesis.get("subject", "").strip().lower()
        predicate = hypothesis.get("predicate", "").strip().lower()
        obj = hypothesis.get("object", "").strip().lower()
        verdicts = {gate: False for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND")}
        gate = verifier.verify_relation(subject, predicate, obj)
        if not gate.passed:
            return False, f"{gate.violated_gate} failed: {gate.error_message}", verdicts
        for name in ("I_DAG", "I_MUTEX", "I_SORT"):
            verdicts[name] = True
        existing = self.memory.find_relation_by_names(subject, predicate, obj)
        if existing and existing.weight_negative > existing.weight_positive:
            verdicts["I_MUTEX"] = False
            return False, "I_MUTEX failed: explicit negative evidence", verdicts
        donor = hypothesis.get("donor")
        if donor and self._are_concepts_disjoint(subject, donor):
            verdicts["I_MUTEX"] = False
            return False, "I_MUTEX failed: analogical donor is disjoint", verdicts
        source = hypothesis.get("source")
        sentence = hypothesis.get("source_text", "")
        if source and sentence:
            path = Path(source)
            if path.is_file() and sentence in path.read_text(encoding="utf-8"):
                verdicts["I_GROUND"] = True
        return True, ("Structural invariants passed; document assertion is sourced" if verdicts["I_GROUND"]
                      else "Structural invariants passed; target factual evidence absent, analogy remains a hypothesis"), verdicts

    def study_step(
        self, target_concept: str | None = None, preferred_axis: str | None = None
    ) -> CuriosityStepResult | None:
        """Perform one autonomous curiosity step: scan, propose, verify, and persist."""
        # 1. Select target concept with highest spoke entropy if not provided
        if not target_concept:
            entropies = self.compute_all_knot_entropies()
            cand_concepts = [c for c, h in entropies.items() if h > 0.0]
            if not cand_concepts:
                return None
            target_concept = cand_concepts[0]

        # 2. Formulate analogical or corpus hypothesis
        proposal = self.propose_corpus_spoke(target_concept)
        if not proposal:
            proposal = self.propose_analogical_spoke(target_concept, preferred_axis=preferred_axis)

        if not proposal:
            return CuriosityStepResult(
                target_concept=target_concept,
                missing_axis=self.policy.no_analogy_axis,
                hypothesis={},
                status="NO_ANALOGY",
                reason=f"No sibling analog or corpus evidence found for '{target_concept}'",
                trace=[f"Knot '{target_concept}' has unpopulated spokes, but hypernym DAG lacks donor siblings and no corpus mentions found."],
            )

        missing_axis, hyp, trace = proposal

        # 3. Verify against DeepSeek invariant gates
        passed, reason, gate_verdicts = self._verify_invariant_gates_detailed(hyp)
        trace.append(f"Invariant gate evaluation: passed={passed}, reason='{reason}'")

        if not passed:
            return CuriosityStepResult(
                target_concept=target_concept,
                missing_axis=missing_axis,
                hypothesis=hyp,
                status="REJECTED",
                gate_verdicts=gate_verdicts,
                reason=reason,
                trace=trace,
            )

        # 4. Autonomous Persistence to SQLite. The ledger creates concepts only
        # after the verified evidence is accepted, so rejected candidates leave
        # no semantic graph residue.
        frame = CandidateFrame.from_text(
            hyp.get("source_text") or f"Analogical hypothesis for {hyp['subject']} {hyp['predicate']} {hyp['object']}; donor={hyp.get('donor', 'unknown')}",
            claims=[
                CandidateClaim(
                    hyp["subject"],
                    hyp["predicate"],
                    hyp["object"],
                    probability=float(hyp.get("probability", self.policy.hypothesis_probability)),
                )
            ],
            intent="study",
            uncertainty=self.policy.hypothesis_uncertainty,
            model_id=self.policy.model_id,
            model_version=self.policy.model_version,
        )
        factual = gate_verdicts["I_GROUND"]
        status = VerificationStatus.PASSED if factual else VerificationStatus.UNKNOWN
        proof_id = None
        identity = json.dumps(hyp, sort_keys=True)
        evidence_id = "evidence_study_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        with self.memory.transaction():
            if not factual:
                donor = self.memory.get_concept(hyp.get("donor", ""))
                target = self.memory.get_concept(hyp["subject"])
                premises = [r.id for c in (donor, target) if c
                            for r in self.memory.get_relations(subject_id=c.id)
                            if r.weight_positive > r.weight_negative]
                proof_id = "proof_study_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
                if not self.memory._conn.execute("SELECT 1 FROM proof_traces WHERE proof_id=?", (proof_id,)).fetchone():
                    self.memory.evidence.save_proof(ProofTrace(proof_id, dict(hyp), premises=premises,
                        operations=[{"kind": "sibling_analogy", "donor": hyp.get("donor"),
                                     "external_truth_established": False}], status=VerificationStatus.UNKNOWN))
            evidence = self.memory.ledger.propose_relation(frame, frame.claims[0],
                source_type=SourceType.DOCUMENT if factual else SourceType.HYPOTHESIS,
                source_reference=hyp.get("source") or f"study:donor:{hyp.get('donor', 'unknown')}",
                evidence_id=evidence_id, derivation_proof_id=proof_id)
            decision = self.memory.ledger.commit_relation(evidence, VerificationResult(status, gate_verdicts, [reason]))
        if not factual:
            trace.append("Analogy retained as candidate evidence; no target relation promoted")
            return CuriosityStepResult(target_concept, missing_axis, hyp, "HYPOTHESIS", gate_verdicts, reason, trace)
        if decision.status.value != "accepted":
            return CuriosityStepResult(target_concept, missing_axis, hyp, "REJECTED", gate_verdicts,
                                       "Evidence ledger rejected document claim", trace)

        subj_c = self.memory.get_concept(hyp["subject"])
        obj_c = self.memory.get_concept(hyp["object"])
        if not subj_c or not obj_c:
            raise RuntimeError("accepted self-study evidence has no persisted concepts")

        trace.append(
            f"Autonomous study update: Persisted ({subj_c.name} {hyp['predicate']} {obj_c.name}) to SQLite store."
        )

        sync_errors = []
        if self.wiki is not None:
            for name in (subj_c.name, obj_c.name):
                try:
                    self.wiki.sync_concept(name)
                except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
                    sync_errors.append(f"Wiki sync failed for {name}: {type(exc).__name__}: {exc}")

        return CuriosityStepResult(
            target_concept=target_concept,
            missing_axis=missing_axis,
            hypothesis=hyp,
            status="PERSISTED",
            gate_verdicts=gate_verdicts,
            reason=reason,
            trace=trace,
            sync_errors=sync_errors,
        )

    def _record_study_visit(self, concept_name: str) -> None:
        """Update exponentially decayed running study share so a single hub cannot monopolize curiosity."""
        clean = concept_name.strip().lower()
        decay = self.share_decay
        for k in list(self._recent_study_share.keys()):
            self._recent_study_share[k] *= decay
            if self._recent_study_share[k] < 1e-4:
                del self._recent_study_share[k]
        self._recent_study_share[clean] = (
            self._recent_study_share.get(clean, 0.0) + (1.0 - decay)
        )

    def effective_study_priority(
        self, concept_name: str, base_entropy: float | None = None
    ) -> float:
        """Compute load-balanced curiosity priority: H_eff(C_i) = H(C_i) - balance_weight * p_bar_i."""
        clean = concept_name.strip().lower()
        h = (
            self.compute_knot_entropy(clean)
            if base_entropy is None
            else float(base_entropy)
        )
        share = self._recent_study_share.get(clean, 0.0)
        return round(h - (self.balance_weight * share), 6)

    def evaluate_heldout_canary(
        self,
        canary_claims: list[tuple[str, str, str, bool]] | None = None,
    ) -> HeldOutCanaryResult:
        """Verify core ontological invariants and optional held-out canary claims."""
        regressions: list[str] = []
        checked = 0

        # 1. Check all positive taxonomic edges for cycle-freedom and disjointness consistency
        concepts_by_id = {c.id: c.name for c in self.memory.list_concepts()}
        for pred in self._hierarchy_predicates():
            for rel in self.memory.get_relations(predicate=pred):
                if rel.weight_positive <= rel.weight_negative:
                    continue
                checked += 1
                s_name = concepts_by_id.get(rel.subject_id, rel.subject_id)
                o_name = concepts_by_id.get(rel.object_id, rel.object_id)
                # Cycle check: object should not have subject as an ancestor
                if rel.subject_id in self._ancestor_ids(rel.object_id):
                    regressions.append(
                        f"Cycle detected on {pred}: {s_name} <-> {o_name}"
                    )
                # Disjointness check: subject and object must not be mutually exclusive
                if self._are_concepts_disjoint(rel.subject_id, rel.object_id):
                    regressions.append(
                        f"Disjointness conflict on {pred}: {s_name} vs {o_name}"
                    )

        # 2. Check explicit held-out canary probes if supplied
        for subj, pred, obj, expected_pos in canary_claims or ():
            checked += 1
            rel = self.memory.find_relation_by_names(subj, pred, obj)
            if expected_pos:
                if rel is None or rel.weight_positive <= rel.weight_negative:
                    regressions.append(
                        f"Missing expected positive canary claim: ({subj}, {pred}, {obj})"
                    )
            else:
                if rel is not None and rel.weight_positive > rel.weight_negative:
                    regressions.append(
                        f"Unexpected positive canary claim: ({subj}, {pred}, {obj})"
                    )

        return HeldOutCanaryResult(
            passed=len(regressions) == 0,
            checked_count=checked,
            regressions=regressions,
        )

    def study_cycle(self, max_steps: int | None = None) -> list[CuriosityStepResult]:
        """Execute autonomous self-study loop across multiple concept knots."""
        results: list[CuriosityStepResult] = []
        attempted_concepts: set[str] = set()
        step_limit = self.policy.max_steps if max_steps is None else max_steps

        entropies = self.compute_all_knot_entropies()
        exp_names = self._get_experienced_concept_names()
        exp_anchored = sorted(
            [c for c, h in entropies.items() if 0.0 < h < 1.0 and c.strip().lower() in exp_names],
            key=lambda c: self.effective_study_priority(c, entropies[c]),
            reverse=True,
        )
        other_anchored = sorted(
            [c for c, h in entropies.items() if 0.0 < h < 1.0 and c.strip().lower() not in exp_names],
            key=lambda c: self.effective_study_priority(c, entropies[c]),
            reverse=True,
        )
        unanchored = sorted(
            [c for c, h in entropies.items() if h >= 1.0],
            key=lambda c: self.effective_study_priority(c, entropies[c]),
            reverse=True,
        )
        candidate_queue = exp_anchored + other_anchored + unanchored

        cand_idx = 0
        while len(results) < step_limit and cand_idx < len(candidate_queue):
            target = candidate_queue[cand_idx]
            cand_idx += 1
            if target in attempted_concepts:
                continue
            attempted_concepts.add(target)
            self._record_study_visit(target)
            step_res = self.study_step(target_concept=target)
            if step_res and step_res.status != "NO_ANALOGY":
                results.append(step_res)

        if len(results) < step_limit and self.wiki is not None:
            try:
                stubs = self.wiki.get_unexplored_frontier()
                for s in stubs:
                    if len(results) >= step_limit:
                        break
                    if s not in attempted_concepts:
                        attempted_concepts.add(s)
                        self._record_study_visit(s)
                        step_res = self.study_step(target_concept=s)
                        if step_res:
                            results.append(step_res)
            except (OSError, RuntimeError, ValueError, sqlite3.Error):
                pass

        return results

    def get_epistemic_gaps(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Identify concepts with highest missing radial spoke entropy and wiki stubs."""
        entropies = self.compute_all_knot_entropies()
        candidate_items = sorted(
            [(c, h) for c, h in entropies.items() if h > 0.0],
            key=lambda kv: self.effective_study_priority(kv[0], kv[1]),
            reverse=True,
        )[: top_n * 2]
        sorted_gaps = [
            {
                "concept": c,
                "entropy": round(h, 3),
                "effective_priority": round(self.effective_study_priority(c, h), 3),
                "missing_axes": self.identify_missing_axes(c),
            }
            for c, h in candidate_items
        ]
        if self.wiki is not None:
            with contextlib.suppress(OSError, RuntimeError, ValueError, sqlite3.Error):
                stubs = self.wiki.get_unexplored_frontier()
                for s in stubs:
                    if not any(g["concept"] == s for g in sorted_gaps):
                        sorted_gaps.append({
                            "concept": s,
                            "entropy": 1.0,
                            "effective_priority": round(self.effective_study_priority(s, 1.0), 3),
                            "missing_axes": list(self.policy.axis_names),
                            "is_stub": True,
                        })

        return sorted_gaps[:top_n]

    def dream(self, max_steps: int = 3) -> DreamReport:
        """Run an autonomous cognitive dreaming session: identify epistemic gaps, formulate hypotheses, verify gates, and commit."""
        start_time = time.perf_counter()
        initial_entropies = self.compute_all_knot_entropies()
        total_concepts = max(len(initial_entropies), 1)

        steps = self.study_cycle(max_steps=max_steps)

        entropy_reduction = 0.0
        persisted_claims: list[dict[str, str]] = []
        rejection_reasons: list[str] = []
        pages_synced = 0

        for s in steps:
            if s.status == "PERSISTED":
                c_before = initial_entropies.get(s.target_concept, 1.0)
                c_after = self.compute_knot_entropy(s.target_concept)
                entropy_reduction += max(0.0, c_before - c_after) / total_concepts
                persisted_claims.append(s.hypothesis)
                if self.wiki is not None:
                    pages_synced += 1
            elif s.status == "REJECTED":
                rejection_reasons.append(s.reason)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return DreamReport(
            steps_attempted=len(steps),
            steps_persisted=len(persisted_claims),
            steps_rejected=len(rejection_reasons),
            entropy_reduction=round(entropy_reduction, 4),
            rejection_reasons=rejection_reasons,
            persisted_claims=persisted_claims,
            wiki_pages_synced=pages_synced,
            duration_ms=elapsed_ms,
            details=steps,
            steps_hypothesized=sum(step.status == "HYPOTHESIS" for step in steps),
            sync_errors=[error for step in steps for error in step.sync_errors],
        )

    def consolidate_sleep_cycle(
        self,
        headroom: Any | None = None,
        paged_store: Any | None = None,
        latent_inducer: Any | None = None,
        max_dream_steps: int = 3,
    ) -> SleepConsolidationReport:
        """Execute a full multi-stage offline sleep consolidation cycle:

        1. Active dreaming & spoke entropy minimization.
        2. Latent invariant mining (I_mutex discovery).
        3. Cobweb category cluster consolidation.
        4. Episodic working memory compaction in Headroom.
        5. Paged Concept Memory cache tier refresh.
        6. Held-out canary invariant verification.
        """
        t0 = time.perf_counter()

        # Phase 1: Dreaming & Spoke Induction
        dream_rep = self.dream(max_steps=max_dream_steps)

        # Phase 2: Latent Invariant Mining (I_mutex)
        mined_invariants: list[dict[str, str]] = []
        with contextlib.suppress(sqlite3.Error, RuntimeError, ValueError, TypeError):
            mined_invariants = self._mine_latent_invariants()

        # Phase 3: Cobweb Cluster Induction
        clusters_count = 0
        inducer = latent_inducer or getattr(self, "latent_inducer", None)
        if inducer is not None:
            with contextlib.suppress(sqlite3.Error, RuntimeError, ValueError, TypeError):
                if hasattr(inducer, "induce_latent_knots"):
                    rep = inducer.induce_latent_knots()
                    clusters_count = int(
                        getattr(
                            rep,
                            "latent_knots_induced",
                            getattr(rep, "clusters_created", 0),
                        )
                    )
                elif hasattr(inducer, "induce_from_memory"):
                    res = inducer.induce_from_memory()
                    clusters_count = len(res) if isinstance(res, list) else 1

        # Phase 4: Episodic Compaction in Headroom
        compressed_count = 0
        if headroom is not None and hasattr(headroom, "_turns"):
            with contextlib.suppress(RuntimeError, ValueError, TypeError, AttributeError):
                for turn in headroom._turns[:-1]:
                    if not turn.is_compressed:
                        if hasattr(headroom, "_compress_turn"):
                            headroom._compress_turn(turn)
                        else:
                            turn.is_compressed = True
                            turn.compressed_summary = (
                                turn.raw_text[:60] + "..."
                                if len(turn.raw_text) > 60
                                else turn.raw_text
                            )
                        compressed_count += 1

        # Phase 5: Paged Store Refresh
        cache_refreshed = False
        if paged_store is not None:
            with contextlib.suppress(RuntimeError, ValueError, TypeError, AttributeError):
                paged_store.clear()
                cache_refreshed = True

        # Phase 6: Held-out Canary Verification
        canary = self.evaluate_heldout_canary()

        lat_ms = (time.perf_counter() - t0) * 1000.0
        msg = (
            f"Sleep consolidation complete: {dream_rep.steps_persisted} hypotheses verified, "
            f"{len(mined_invariants)} mutex invariants mined, {clusters_count} clusters consolidated, "
            f"{compressed_count} turns compressed in {lat_ms:.1f}ms."
        )

        return SleepConsolidationReport(
            dream_report=dream_rep,
            invariants_mined=mined_invariants,
            clusters_induced=clusters_count,
            turns_compressed=compressed_count,
            cache_refreshed=cache_refreshed,
            duration_ms=lat_ms,
            summary_message=msg,
            canary_passed=canary.passed,
            canary_regressions=canary.regressions,
        )

    def _mine_latent_invariants(self) -> list[dict[str, str]]:
        """No sound observation rule for new disjointness has been established."""
        return []
