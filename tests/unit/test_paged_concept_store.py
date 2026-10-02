"""Unit tests for PagedConceptStore 3-tier memory and context voting."""

from pathlib import Path
from little.memory.memory_policy import MemoryPolicy, PagedMemoryConfig
from little.memory.ontology import seed_commonsense_ontology
from little.memory.paged_store import PagedConceptStore
from little.memory.store import MemoryStore


def test_paged_concept_store_tiers_and_hits(tmp_path: Path):
    db_path = tmp_path / "test_paged.db"
    store = MemoryStore(str(db_path))
    seed_commonsense_ontology(store)

    policy = MemoryPolicy(
        version="1",
        concept_confidence=1.0,
        experience_query_limit=50,
        bulk_import_batch_size=1000,
        paged_memory=PagedMemoryConfig(
            lru_cache_capacity=4,
            resident_capacity=2,
            activation_decay=0.8,
            activation_threshold=0.1,
            spreading_factor=0.5,
        ),
    )

    paged = PagedConceptStore(store, policy=policy)

    # 1. Initial lookup -> Disk fetch (Tier 1)
    knot_apple = paged.get_knot("apple")
    assert knot_apple is not None
    assert knot_apple.concept_id == "APPLE"
    assert paged.get_telemetry().disk_fetches == 1
    assert paged.get_telemetry().lru_hits == 0

    # 2. Second lookup -> RAM LRU hit (Tier 2)
    knot_apple_2 = paged.get_knot("apple")
    assert knot_apple_2 is not None
    assert paged.get_telemetry().lru_hits == 1
    assert paged.get_telemetry().disk_fetches == 1

    # 3. Context voting ("The text chooses")
    # Voting with tokens "apple" and "fruit"
    residents = paged.vote_from_tokens(["apple", "delicious", "fruit"])
    assert "APPLE" in residents or "FRUIT" in residents

    # 4. Lookup for resident concept -> Resident hit (Tier 3)
    knot_res = paged.get_knot("apple")
    assert knot_res is not None
    assert paged.get_telemetry().resident_hits == 1

    # 5. Eviction under LRU capacity (capacity = 4)
    paged.get_knot("bird")
    paged.get_knot("animal")
    paged.get_knot("plant")
    paged.get_knot("vehicle")

    telemetry = paged.get_telemetry()
    assert telemetry.lru_count <= 4
    assert telemetry.evictions >= 1

    # 6. Invalidation
    paged.invalidate("apple")
    assert "APPLE" not in paged._resident_knots
    assert "APPLE" not in paged._lru_cache
