"""3-Tier Paged Concept Memory Store (Disk -> RAM LRU -> Resident Working Window).

Inspired by mini-AGI's dynamic paged weight hierarchy and context-driven activation
voting ("The text chooses").

Tier 1: Unbounded Disk Vault (SQLite + OKF Markdown Knowledge Vault)
Tier 2: RAM LRU Cache (High-speed OrderedDict of hydrated ConceptKnots)
Tier 3: Resident Working Set (Top-K active concepts promoted via spreading activation voting)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from little.core.concept_knot import ConceptKnot
from little.knowledge.registry import SchemaRegistry
from little.memory.memory_policy import MemoryPolicy, PagedMemoryConfig

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


@dataclass(frozen=True)
class PagedStoreTelemetry:
    """Introspectable telemetry of 3-tier memory operation."""

    resident_count: int
    lru_count: int
    resident_capacity: int
    lru_capacity: int
    resident_hits: int
    lru_hits: int
    disk_fetches: int
    evictions: int
    total_requests: int
    hit_rate: float
    resident_concepts: list[str]


class PagedConceptStore:
    """Manages 3-tier concept memory with context-driven spreading activation voting."""

    def __init__(
        self,
        memory: MemoryStore,
        policy: MemoryPolicy | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        self.memory = memory
        self._revision = getattr(memory, "semantic_revision", memory.revision)
        loaded_policy = policy or MemoryPolicy.default()
        self.paged_config: PagedMemoryConfig = getattr(
            loaded_policy, "paged_memory", PagedMemoryConfig()
        )
        self.registry: SchemaRegistry = (
            registry
            or getattr(getattr(memory, "ledger", None), "registry", None)
            or SchemaRegistry.default()
        )

        # Tier 2: LRU Cache
        self._lru_cache: OrderedDict[str, ConceptKnot] = OrderedDict()

        # Tier 3: Resident Working Set (The Text Chooses)
        self._resident_knots: dict[str, ConceptKnot] = {}

        # Spreading Activation Scores (concept_canonical_name -> activation_level)
        self._activation_scores: dict[str, float] = {}

        # Telemetry metrics
        self._resident_hits: int = 0
        self._lru_hits: int = 0
        self._disk_fetches: int = 0
        self._evictions: int = 0
        self._total_requests: int = 0

    @staticmethod
    def _canonical(name_or_id: str) -> str:
        return name_or_id.strip().upper().replace(" ", "_")

    def get_knot(self, concept_id_or_name: str) -> ConceptKnot | None:
        """Retrieve a 360° Concept Knot traversing Tier 3 -> Tier 2 -> Tier 1."""
        self._refresh_if_changed()
        self._total_requests += 1
        key = self._canonical(concept_id_or_name)

        # 1. Tier 3: Resident Working Set Hit (<0.02ms)
        if key in self._resident_knots:
            self._resident_hits += 1
            if key in self._lru_cache:
                self._lru_cache.move_to_end(key)
            return self._resident_knots[key]

        # Check alternative variants in resident set (e.g. concept ID vs canonical name)
        for r_key, knot in self._resident_knots.items():
            if r_key == key or self._canonical(knot.concept_id) == key:
                self._resident_hits += 1
                return knot

        # 2. Tier 2: RAM LRU Cache Hit (<0.1ms)
        if key in self._lru_cache:
            self._lru_hits += 1
            self._lru_cache.move_to_end(key)
            return self._lru_cache[key]

        for lru_key, knot in list(self._lru_cache.items()):
            if self._canonical(knot.concept_id) == key:
                self._lru_hits += 1
                self._lru_cache.move_to_end(lru_key)
                return knot

        # 3. Tier 1: Disk Vault Fetch (SQLite + OKF)
        self._disk_fetches += 1
        knot = ConceptKnot.from_memory(
            self.memory,
            concept_id_or_name,
            registry=self.registry,
        )
        if knot:
            self._insert_into_lru(key, knot)
        return knot

    def _insert_into_lru(self, key: str, knot: ConceptKnot) -> None:
        """Insert a knot into Tier 2 LRU cache, evicting oldest if capacity reached."""
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
            self._lru_cache[key] = knot
            return

        while len(self._lru_cache) >= self.paged_config.lru_cache_capacity:
            self._lru_cache.popitem(last=False)
            self._evictions += 1

        self._lru_cache[key] = knot

    def vote_from_tokens(
        self, tokens: list[str] | tuple[str, ...] | set[str]
    ) -> list[str]:
        """Cast context-driven activation votes over the semantic graph.

        The incoming dialogue stream or document chunks vote for concepts.
        Activation spreads along 1-hop hypernym and mereology edges, then
        the top-ranked concepts are admitted into the Tier 3 Resident Set.
        """
        self._refresh_if_changed()
        # 1. Decay previous activations exponentially
        decay = self.paged_config.activation_decay
        for k in list(self._activation_scores.keys()):
            self._activation_scores[k] *= decay
            if self._activation_scores[k] < 0.001:
                del self._activation_scores[k]

        # 2. Cast direct and spreading activation votes
        spread_factor = self.paged_config.spreading_factor
        for token in tokens:
            token_clean = token.strip().lower()
            if len(token_clean) < 2:
                continue

            c = self.memory.get_concept(token_clean)
            if not c:
                continue

            c_key = self._canonical(c.name)
            self._activation_scores[c_key] = (
                self._activation_scores.get(c_key, 0.0) + 1.0
            )

            # Spreading activation to 1-hop taxonomic hypernyms and parts
            knot = (
                self._resident_knots.get(c_key)
                or self._lru_cache.get(c_key)
                or ConceptKnot.from_memory(self.memory, c.name, registry=self.registry)
            )
            if knot:
                for hyper in knot.taxonomy_hypernyms:
                    h_key = self._canonical(hyper)
                    self._activation_scores[h_key] = (
                        self._activation_scores.get(h_key, 0.0) + spread_factor
                    )
                for part in knot.mereology_parts.keys():
                    p_key = self._canonical(part)
                    self._activation_scores[p_key] = (
                        self._activation_scores.get(p_key, 0.0) + (spread_factor * 0.5)
                    )

        # 3. Promote top-K concepts to Tier 3 Resident Working Window
        sorted_concepts = sorted(
            self._activation_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        threshold = self.paged_config.activation_threshold
        cap = self.paged_config.resident_capacity

        new_residents: dict[str, ConceptKnot] = {}
        for c_key, score in sorted_concepts[:cap]:
            if score >= threshold:
                knot = self.get_knot(c_key)
                if knot:
                    new_residents[c_key] = knot

        self._resident_knots = new_residents
        return list(self._resident_knots.keys())

    def invalidate(self, concept_id_or_name: str) -> None:
        """Invalidate concept in active tiers upon modification or learning."""
        key = self._canonical(concept_id_or_name)
        self._resident_knots.pop(key, None)
        self._lru_cache.pop(key, None)
        self._activation_scores.pop(key, None)

    def _refresh_if_changed(self) -> None:
        current_rev = getattr(self.memory, "semantic_revision", self.memory.revision)
        if current_rev != self._revision:
            self.clear()
            self._revision = current_rev

    def clear(self) -> None:
        """Clear memory cache tiers."""
        self._resident_knots.clear()
        self._lru_cache.clear()
        self._activation_scores.clear()

    @property
    def resident_concepts(self) -> list[str]:
        return list(self._resident_knots.keys())

    def get_telemetry(self) -> PagedStoreTelemetry:
        """Inspectable memory stats and hit rates."""
        cache_hits = self._resident_hits + self._lru_hits
        hit_rate = (
            cache_hits / self._total_requests if self._total_requests > 0 else 0.0
        )
        return PagedStoreTelemetry(
            resident_count=len(self._resident_knots),
            lru_count=len(self._lru_cache),
            resident_capacity=self.paged_config.resident_capacity,
            lru_capacity=self.paged_config.lru_cache_capacity,
            resident_hits=self._resident_hits,
            lru_hits=self._lru_hits,
            disk_fetches=self._disk_fetches,
            evictions=self._evictions,
            total_requests=self._total_requests,
            hit_rate=round(hit_rate, 4),
            resident_concepts=self.resident_concepts,
        )
