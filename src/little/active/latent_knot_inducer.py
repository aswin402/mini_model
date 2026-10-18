"""Autonomous Latent Knot Induction & Ontology Expansion Engine.

Transforms Cobweb clusters into reified, verified Concept Knots in SQLite MemoryStore
and synchronizes with the Living OKF Knowledge Vault and TurboQuant Vector Index.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from little.active.cobweb_engine import CobwebInstance, CobwebNode, CobwebTree
from little.active.spiderweb_policy import SpiderWebGrowthPolicy
from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    ProofTrace,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.knowledge.living_wiki import LivingWiki
from little.memory.store import MemoryStore


@dataclass
class LatentInductionResult:
    """Provenance record of an autonomously induced latent concept knot."""

    latent_name: str
    constituent_concepts: list[str]
    shared_features: dict[str, list[str]]
    category_utility: float
    verification_passed: bool
    rejection_reason: str | None = None
    okf_vault_path: str | None = None
    proof_id: str | None = None
    sync_errors: list[str] = field(default_factory=list)


@dataclass
class LatentClusteringReport:
    """Summary of an autonomous ontology restructuring and latent discovery session."""

    total_concepts_analyzed: int
    clusters_formed: int
    latent_knots_induced: int
    latent_knots_rejected: int
    duration_ms: float
    inductions: list[LatentInductionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class LatentKnotInducer:
    """Autonomously discovers, names, verifies, and registers latent concept knots."""

    def __init__(
        self,
        memory: MemoryStore,
        wiki: LivingWiki | None = None,
        policy: SpiderWebGrowthPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.wiki = wiki or LivingWiki(memory=memory)
        self.policy = policy or SpiderWebGrowthPolicy.default()
        self.verifier = DeepSeekInvariantVerifier(memory=memory)

    def extract_cobweb_instances(self, concept_names: list[str] | None = None) -> list[CobwebInstance]:
        """Hydrate concepts from SQLite MemoryStore into structured CobwebInstances."""
        all_concepts = self.memory.list_concepts()
        if concept_names:
            target_set = {n.lower().strip() for n in concept_names}
            selected = [c for c in all_concepts if c.name.lower() in target_set]
        else:
            selected = all_concepts

        instances: list[CobwebInstance] = []

        for c in selected:
            features: dict[str, set[str]] = {}

            # Collect outgoing relations
            for rel in self.memory.get_relations(subject_id=c.id):
                # Only incorporate active, positive relations
                if rel.weight_positive <= rel.weight_negative:
                    continue
                pred = rel.predicate.lower().strip()
                target_concept = self.memory.get_concept(rel.object_id)
                if not target_concept:
                    # Target might be by name if id resolution differs
                    continue
                val = target_concept.name.lower().strip()

                if pred in ("is_a", "subclass_of", "instance_of"):
                    features.setdefault("is_a", set()).add(val)
                elif pred in ("has_part", "has", "consists_of"):
                    features.setdefault("has", set()).add(val)
                elif pred in ("part_of", "member_of"):
                    features.setdefault("part_of", set()).add(val)
                elif pred in ("can", "capable_of", "used_for"):
                    features.setdefault("capable_of", set()).add(val)
                elif pred in ("located_in", "lives_in"):
                    features.setdefault("located_in", set()).add(val)
                elif pred in ("eats", "diet"):
                    features.setdefault("eats", set()).add(val)
                elif pred in ("made_of", "material"):
                    features.setdefault("made_of", set()).add(val)

            # Include attributes from concept metadata
            for attr_name, attr_val in c.attributes.items():
                features.setdefault(attr_name.lower(), set()).add(str(attr_val).lower())

            if features:
                instances.append(CobwebInstance(concept_id=c.name.lower().strip(), features=features))

        return instances

    def generate_latent_name(
        self,
        constituent_concepts: list[str],
        shared_features: dict[str, list[str]],
    ) -> str:
        """Derive a natural, readable semantic name for an induced latent category."""
        # 1. Primary taxonomic hypernym
        hypernyms = shared_features.get("is_a", [])
        primary_hyper = hypernyms[0] if hypernyms else self.policy.fallback_hypernym.lower()

        # 2. Most salient differentiating capability or property
        caps = shared_features.get("capable_of", [])
        parts = shared_features.get("has", [])
        locs = shared_features.get("located_in", [])

        descriptor = ""
        if caps:
            descriptor = caps[0].replace(" ", "_")
        elif parts:
            descriptor = f"with_{parts[0].replace(' ', '_')}"
        elif locs:
            descriptor = locs[0].replace(" ", "_")

        if descriptor:
            name_candidate = f"latent_{descriptor}_{primary_hyper}".lower()
        else:
            name_candidate = f"latent_{primary_hyper}_cluster".lower()

        # Clean string
        clean_name = re.sub(r"[^a-z0-9_]+", "_", name_candidate).strip("_")
        return clean_name

    def induce_latent_knots(
        self,
        instances: list[CobwebInstance] | None = None,
        min_cluster_size: int = 2,
    ) -> LatentClusteringReport:
        """Run Cobweb clustering and induce latent concept knots with DeepSeek verification."""
        t0 = time.perf_counter()

        if instances is None:
            instances = self.extract_cobweb_instances()

        tree = CobwebTree(category_utility_threshold=self.policy.category_utility_threshold)
        for inst in instances:
            tree.insert(inst)

        clusters = tree.get_all_clusters(min_size=min_cluster_size)

        # Sort clusters by shared feature specificity first, then size descending
        sorted_clusters = sorted(
            clusters,
            key=lambda c: (
                sum(len(v) for v in c.get_shared_features(min_ratio=1.0).values()),
                len(c.concept_ids),
            ),
            reverse=True,
        )
        filtered_clusters: list[Tuple[CobwebNode, str, dict[str, list[str]]]] = []
        seen_cluster_coverages: dict[str, list[set[str]]] = {}

        for c in sorted_clusters:
            shared = c.get_shared_features(min_ratio=1.0)
            if not shared:
                continue
            name = self.generate_latent_name(sorted(c.concept_ids), shared)
            c_set = set(c.concept_ids)

            # Check if this concept set is already covered by a broader cluster with the same name
            covered = False
            for existing_set in seen_cluster_coverages.get(name, []):
                if c_set.issubset(existing_set) or (len(c_set & existing_set) / len(c_set) > 0.75):
                    covered = True
                    break

            if not covered:
                filtered_clusters.append((c, name, shared))
                seen_cluster_coverages.setdefault(name, []).append(c_set)
                if len(filtered_clusters) >= 16:
                    break

        report = LatentClusteringReport(
            total_concepts_analyzed=len(instances),
            clusters_formed=len(filtered_clusters),
            latent_knots_induced=0,
            latent_knots_rejected=0,
            duration_ms=0.0,
        )

        seen_constituent_signatures: set[str] = set()

        for cluster, latent_name, shared in filtered_clusters:
            constituents = sorted(cluster.concept_ids)
            sig = "-".join(constituents)
            if sig in seen_constituent_signatures:
                continue
            seen_constituent_signatures.add(sig)

            cu = cluster.parent.category_utility if cluster.parent is not None else cluster.category_utility

            # Invariant Verification Gate:
            # 1. Check if latent concept creates DAG cycles or mutex contradictions
            # Proposed relation: constituent is_a latent_name
            all_valid = True
            rejection_reason = None

            for c_name in constituents:
                gate_res = self.verifier.verify_relation(
                    subject=c_name,
                    predicate="is_a",
                    obj=latent_name,
                )
                if not gate_res.passed:
                    all_valid = False
                    rejection_reason = f"Gate violation for ({c_name}, is_a, {latent_name}): {gate_res.error_message}"
                    break

            if not all_valid:
                report.latent_knots_rejected += 1
                report.inductions.append(
                    LatentInductionResult(
                        latent_name=latent_name,
                        constituent_concepts=constituents,
                        shared_features=shared,
                        category_utility=cu,
                        verification_passed=False,
                        rejection_reason=rejection_reason,
                    )
                )
                continue

            # This is a defined grouping over observed graph features, not discovery
            # of a new externally true biological superclass.
            observed = {instance.concept_id: instance.features for instance in self.extract_cobweb_instances(constituents)}
            if any(name not in observed or any(not set(values).issubset(observed[name].get(feature, set()))
                                              for feature, values in shared.items()) for name in constituents):
                report.latent_knots_rejected += 1
                report.inductions.append(LatentInductionResult(latent_name, constituents, shared, cu, False,
                    rejection_reason="Supplied cluster features lack supported persisted observations"))
                continue
            signature = json.dumps([constituents, shared], sort_keys=True)
            digest = hashlib.sha256(signature.encode()).hexdigest()[:24]
            existing = self.memory.get_concept(latent_name)
            if existing and existing.attributes.get("definition_signature") != digest:
                latent_name += "_" + digest[:8]
            proof_id = "proof_cluster_" + digest
            premise_ids = sorted({relation.id for name in constituents
                                  for relation in self.memory.get_relations(subject_id=self.memory.get_concept(name).id)
                                  if relation.weight_positive > relation.weight_negative})
            definition = {"category": latent_name, "members": constituents, "shared_features": shared}
            try:
                with self.memory.transaction():
                    if not self.memory._conn.execute("SELECT 1 FROM proof_traces WHERE proof_id=?", (proof_id,)).fetchone():
                        self.memory.evidence.save_proof(ProofTrace(proof_id, definition, premises=premise_ids,
                            operations=[{"kind": "defined_cluster", "minimum_feature_ratio": 1.0,
                                         "external_truth_established": False}], status=VerificationStatus.PASSED))
                    latent = self.memory.get_or_create_concept(latent_name)
                    attributes = dict(latent.attributes)
                    attributes.update({"is_latent": True, "category_utility": cu,
                        "constituent_count": len(constituents), "definition_signature": digest,
                        "definition": definition, "definition_proof_id": proof_id})
                    self.memory.update_concept_attributes(latent.id, attributes)
                    claims = [CandidateClaim(name, "is_a", latent_name) for name in constituents]
                    claims.extend(CandidateClaim(latent_name, "is_a", parent)
                                  for parent in shared.get("is_a", []) if parent != latent_name)
                    for claim in claims:
                        gate = self.verifier.verify_relation(claim.subject, claim.predicate, claim.object)
                        if not gate.passed:
                            raise ValueError(gate.error_message)
                        text = "defined cluster: " + json.dumps(definition, sort_keys=True)
                        evidence_id = "evidence_cluster_" + hashlib.sha256(
                            json.dumps([digest, claim.subject, claim.predicate, claim.object]).encode()).hexdigest()[:24]
                        evidence = self.memory.ledger.propose_relation(CandidateFrame.from_text(text, claims=[claim]), claim,
                            source_type=SourceType.INFERENCE, source_reference=f"cluster:{digest}",
                            evidence_id=evidence_id, derivation_proof_id=proof_id)
                        self.memory.ledger.commit_relation(evidence, VerificationResult(VerificationStatus.PASSED,
                            checks={"defined_cluster_membership": True},
                            reasons=["Membership follows the recorded feature grouping definition; external truth not established"]))
            except Exception as exc:
                report.latent_knots_rejected += 1
                report.errors.append(f"Cluster persistence failed: {type(exc).__name__}: {exc}")
                report.inductions.append(LatentInductionResult(latent_name, constituents, shared, cu, False,
                    rejection_reason=str(exc)))
                continue
            okf_path = None
            sync_errors = []
            try:
                exported_path = self.wiki.okf_engine.export_concept(latent_name, output_dir=self.wiki.vault_dir)
                okf_path = str(exported_path)
            except Exception as exc:
                sync_errors.append(f"Wiki export failed for {latent_name}: {type(exc).__name__}: {exc}")
                report.errors.extend(sync_errors)

            report.latent_knots_induced += 1
            report.inductions.append(
                LatentInductionResult(
                    latent_name=latent_name,
                    constituent_concepts=constituents,
                    shared_features=shared,
                    category_utility=cu,
                    verification_passed=True,
                    okf_vault_path=okf_path,
                    proof_id=proof_id,
                    sync_errors=sync_errors,
                )
            )

        # Refresh wiki graph topology
        if report.latent_knots_induced > 0:
            try:
                self.wiki.build_graph()
            except Exception as exc:
                report.errors.append(f"Wiki graph refresh failed: {type(exc).__name__}: {exc}")

        report.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return report

    def explain_concept_cluster(self, concept_name: str) -> dict[str, Any] | None:
        """Explain why a concept belongs to its latent cluster and what traits define it."""
        c = self.memory.get_concept(concept_name.lower().strip())
        if not c:
            return None

        # Find latent superclasses
        latent_parents = []
        for rel in self.memory.get_relations(subject_id=c.id):
            if rel.predicate == "is_a":
                target = self.memory.get_concept(rel.object_id)
                if target and (target.name.startswith("latent_") or target.attributes.get("is_latent")):
                    latent_parents.append(target)

        if not latent_parents:
            return None

        parent = latent_parents[0]

        # Find peer concepts in this latent cluster
        peers = []
        for rel in self.memory.get_relations(object_id=parent.id):
            if rel.predicate == "is_a":
                peer_concept = self.memory.get_concept(rel.subject_id)
                if peer_concept and peer_concept.name != c.name:
                    peers.append(peer_concept.name)

        return {
            "concept": c.name,
            "latent_category": parent.name,
            "peer_concepts": sorted(peers),
            "category_utility": parent.attributes.get("category_utility", 0.0),
            "attributes": parent.attributes,
        }
