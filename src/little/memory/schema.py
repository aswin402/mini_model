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

-- 6. Constructions Table (Construction Grammar Memory)
CREATE TABLE IF NOT EXISTS constructions (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    pattern_tokens_json TEXT NOT NULL,
    slot_roles_json TEXT NOT NULL DEFAULT '{}',
    predicate_template TEXT NOT NULL,
    construction_type TEXT NOT NULL DEFAULT 'statement',
    is_negative INTEGER NOT NULL DEFAULT 0,
    is_property INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence_positive INTEGER NOT NULL DEFAULT 1,
    evidence_negative INTEGER NOT NULL DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS idx_constructions_name ON constructions(name);
CREATE INDEX IF NOT EXISTS idx_constructions_type ON constructions(construction_type);

-- 7. Evidence Ledger (Grounded Candidate and Accepted Claims)
CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT,
    value_json TEXT,
    source_type TEXT NOT NULL,
    source_reference TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    observed_at TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    support_weight INTEGER NOT NULL DEFAULT 0,
    opposition_weight INTEGER NOT NULL DEFAULT 0,
    extraction_confidence REAL NOT NULL DEFAULT 0.0,
    polarity INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'candidate',
    derivation_proof_id TEXT,
    created_at TEXT NOT NULL
);

-- 8. Proof Traces (Inspectable Derivations)
CREATE TABLE IF NOT EXISTS proof_traces (
    proof_id TEXT PRIMARY KEY,
    conclusion_json TEXT NOT NULL,
    premises_json TEXT NOT NULL DEFAULT '[]',
    operations_json TEXT NOT NULL DEFAULT '[]',
    tool_outputs_json TEXT NOT NULL DEFAULT '[]',
    verifier_results_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 9. Commit Decisions (Evidence-to-Memory Decisions)
CREATE TABLE IF NOT EXISTS commit_decisions (
    decision_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    status TEXT NOT NULL,
    checks_json TEXT NOT NULL DEFAULT '{}',
    reasons_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim
    ON evidence_records(subject_id, predicate, object_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source
    ON evidence_records(source_type);
CREATE INDEX IF NOT EXISTS idx_evidence_status
    ON evidence_records(status);
CREATE INDEX IF NOT EXISTS idx_evidence_observed_at
    ON evidence_records(observed_at);
"""
