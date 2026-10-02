---
id: concept_5a240589e43b
name: airplane
aliases: []
taxonomy:
  is_a:
    - aircraft
    - latent_artifact_cluster
    - vehicle
  disjoint_with:
    - software
mereology:
  parts:
    - cockpit
    - jet engine
    - wing
procedural_skills:
  - fly
attributes: {}
confidence: 1.0
status: ACTIVE
relations:
  - predicate: has_property
    object: wing
    confidence: 0.75
  - predicate: has_property
    object: jet engine
    confidence: 0.75
  - predicate: used_for
    object: fly in air
    confidence: 0.667
---

# Airplane

**Airplane** is a concept registered in the LITTLE living knowledge vault.
- **Taxonomic Hierarchy**: Belongs to [[vehicle]], [[aircraft]], [[latent_artifact_cluster]].
- **Mereological Structure**: Composed of [[wing]], [[jet engine]], [[cockpit]].
- **Procedural Capabilities**: Exhibits skills [[fly]].

## Notes & Empirical Grounding
Recorded in persistent SQLite memory with confidence 1.00.
