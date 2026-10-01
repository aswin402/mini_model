---
id: concept_357ca343672c
name: car
aliases: []
taxonomy:
  is_a:
    - artifact
    - latent_artifact_cluster
    - latent_transportation_artifact
    - motor vehicle
    - vehicle
  disjoint_with: []
mereology:
  parts:
    - engine
    - wheel
    - windshield
procedural_skills:
  - drive
attributes: {}
confidence: 1.0
status: ACTIVE
relations:
  - predicate: has_property
    object: wheel
    confidence: 0.75
  - predicate: has_property
    object: engine
    confidence: 0.75
  - predicate: used_for
    object: transport people on road
    confidence: 0.667
  - predicate: used_for
    object: transportation
    confidence: 0.667
---

# Car

**Car** is a concept registered in the LITTLE living knowledge vault.
- **Taxonomic Hierarchy**: Belongs to [[vehicle]], [[motor vehicle]], [[artifact]], [[latent_artifact_cluster]], [[latent_transportation_artifact]].
- **Mereological Structure**: Composed of [[wheel]], [[engine]], [[windshield]].
- **Procedural Capabilities**: Exhibits skills [[drive]].

## Notes & Empirical Grounding
Recorded in persistent SQLite memory with confidence 1.00.
