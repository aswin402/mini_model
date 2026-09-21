"""Database schema definitions and migrations for the LITTLE persistent memory store."""

SCHEMA_V1 = """
-- 1. Concepts Table (Semantic Memory)
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    category TEXT,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TEXT NOT NULL
);

-- 2. Entities Table (Specific Grounded Instances)
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    properties_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

-- 3. Relations Table (Semantic Graph Edges)
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    weight_positive INTEGER NOT NULL DEFAULT 1,
    weight_negative INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    source_experience_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(subject_id, predicate, object_id)
);

-- 4. Experiences Table (Episodic Memory Event Log)
CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    input_text TEXT NOT NULL,
    extracted_triples_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'user',
    timestamp TEXT NOT NULL
);

-- 5. Procedures Table (Procedural Memory / Learned Skills)
CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    code_body TEXT NOT NULL,
    input_signature TEXT,
    output_signature TEXT,
    created_at TEXT NOT NULL
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_concept ON entities(concept_id);
CREATE INDEX IF NOT EXISTS idx_relations_subj_pred ON relations(subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_relations_obj_pred ON relations(object_id, predicate);
CREATE INDEX IF NOT EXISTS idx_relations_pred ON relations(predicate);
CREATE INDEX IF NOT EXISTS idx_experiences_timestamp ON experiences(timestamp);
"""
