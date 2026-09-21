"""Persistent Memory Store for LITTLE based on transactional SQLite.

Implements Episodic, Semantic, and Procedural memory tiers with inspectable
evidence tracking and local persistence.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from little.core.models import (
    Concept,
    ConceptStatus,
    Construction,
    Entity,
    Experience,
    Relation,
    Skill,
)
from little.memory.schema import SCHEMA_V1


class MemoryStore:
    """Manages persistent episodic, semantic, and procedural knowledge structures."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL;")
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.executescript(SCHEMA_V1)
        self.seed_default_constructions()

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Semantic Memory: Concepts
    # -------------------------------------------------------------------------

    def create_concept(
        self,
        name: str,
        category: str | None = None,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> Concept:
        """Create and persist a new concept. Returns existing if name already exists."""
        clean_name = name.strip().lower()
        existing = self.get_concept(clean_name)
        if existing:
            return existing

        concept = Concept.create(
            name=clean_name,
            category=category,
            aliases=aliases,
            attributes=attributes,
            confidence=confidence,
        )

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO concepts (id, name, aliases_json, category, attributes_json, confidence, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    concept.id,
                    concept.name,
                    json.dumps(concept.aliases),
                    concept.category,
                    json.dumps(concept.attributes),
                    concept.confidence,
                    concept.status.value,
                    concept.created_at,
                ),
            )
        return concept

    def get_concept(self, id_or_name: str) -> Concept | None:
        """Retrieve concept by unique ID or canonical lowercase name."""
        target = id_or_name.strip().lower()
        cursor = self._conn.execute(
            """
            SELECT id, name, aliases_json, category, attributes_json, confidence, status, created_at
            FROM concepts
            WHERE id = ? OR name = ?;
            """,
            (id_or_name, target),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return Concept(
            id=row["id"],
            name=row["name"],
            aliases=json.loads(row["aliases_json"]),
            category=row["category"],
            attributes=json.loads(row["attributes_json"]),
            confidence=row["confidence"],
            status=ConceptStatus(row["status"]),
            created_at=row["created_at"],
        )

    def get_or_create_concept(self, name: str, category: str | None = None) -> Concept:
        existing = self.get_concept(name)
        if existing:
            return existing
        return self.create_concept(name, category=category)

    def list_concepts(self) -> list[Concept]:
        cursor = self._conn.execute(
            """
            SELECT id, name, aliases_json, category, attributes_json, confidence, status, created_at
            FROM concepts
            ORDER BY name ASC;
            """
        )
        return [
            Concept(
                id=row["id"],
                name=row["name"],
                aliases=json.loads(row["aliases_json"]),
                category=row["category"],
                attributes=json.loads(row["attributes_json"]),
                confidence=row["confidence"],
                status=ConceptStatus(row["status"]),
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    def update_concept_attributes(
        self, concept_id: str, attributes: dict[str, Any]
    ) -> Concept:
        concept = self.get_concept(concept_id)
        if not concept:
            raise ValueError(f"Concept '{concept_id}' not found.")

        updated_attrs = {**concept.attributes, **attributes}
        with self._conn:
            self._conn.execute(
                "UPDATE concepts SET attributes_json = ? WHERE id = ?;",
                (json.dumps(updated_attrs), concept_id),
            )
        concept.attributes = updated_attrs
        return concept

    # -------------------------------------------------------------------------
    # Semantic Memory: Entities
    # -------------------------------------------------------------------------

    def create_entity(
        self,
        name: str,
        concept_id: str,
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        entity = Entity.create(name=name, concept_id=concept_id, properties=properties)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO entities (id, name, concept_id, properties_json, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    entity.id,
                    entity.name,
                    entity.concept_id,
                    json.dumps(entity.properties),
                    entity.created_at,
                ),
            )
        return entity

    def get_entity(self, id_or_name: str) -> Entity | None:
        cursor = self._conn.execute(
            """
            SELECT id, name, concept_id, properties_json, created_at
            FROM entities
            WHERE id = ? OR name = ?;
            """,
            (id_or_name, id_or_name.strip()),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Entity(
            id=row["id"],
            name=row["name"],
            concept_id=row["concept_id"],
            properties=json.loads(row["properties_json"]),
            created_at=row["created_at"],
        )

    # -------------------------------------------------------------------------
    # Semantic Graph: Relations & Evidence Accumulation
    # -------------------------------------------------------------------------

    def add_relation(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        positive: bool = True,
        source_experience_id: str | None = None,
    ) -> Relation:
        """Add or reinforce a typed relation edge.

        If the relation already exists, increments evidence count and recomputes
        confidence without creating duplicate rows.
        """
        clean_pred = predicate.strip().lower()

        cursor = self._conn.execute(
            """
            SELECT id, subject_id, predicate, object_id, weight_positive, weight_negative, confidence, source_experience_id, created_at
            FROM relations
            WHERE subject_id = ? AND predicate = ? AND object_id = ?;
            """,
            (subject_id, clean_pred, object_id),
        )
        row = cursor.fetchone()

        if row:
            # Relation exists: reinforce evidence
            w_pos = row["weight_positive"] + (1 if positive else 0)
            w_neg = row["weight_negative"] + (0 if positive else 1)
            total = w_pos + w_neg
            conf = round(w_pos / (total + 1.0), 4)

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE relations
                    SET weight_positive = ?, weight_negative = ?, confidence = ?
                    WHERE id = ?;
                    """,
                    (w_pos, w_neg, conf, row["id"]),
                )
            return Relation(
                id=row["id"],
                subject_id=subject_id,
                predicate=clean_pred,
                object_id=object_id,
                weight_positive=w_pos,
                weight_negative=w_neg,
                confidence=conf,
                source_experience_id=row["source_experience_id"],
                created_at=row["created_at"],
            )

        # New relation edge
        rel = Relation.create(
            subject_id=subject_id,
            predicate=clean_pred,
            object_id=object_id,
            source_experience_id=source_experience_id,
            positive=positive,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO relations (id, subject_id, predicate, object_id, weight_positive, weight_negative, confidence, source_experience_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    rel.id,
                    rel.subject_id,
                    rel.predicate,
                    rel.object_id,
                    rel.weight_positive,
                    rel.weight_negative,
                    rel.confidence,
                    rel.source_experience_id,
                    rel.created_at,
                ),
            )
        return rel

    def get_relations(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
    ) -> list[Relation]:
        """Query relations with optional filters."""
        query = "SELECT id, subject_id, predicate, object_id, weight_positive, weight_negative, confidence, source_experience_id, created_at FROM relations WHERE 1=1"
        params: list[Any] = []

        if subject_id:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate.strip().lower())
        if object_id:
            query += " AND object_id = ?"
            params.append(object_id)

        query += " ORDER BY confidence DESC;"

        cursor = self._conn.execute(query, params)
        return [
            Relation(
                id=row["id"],
                subject_id=row["subject_id"],
                predicate=row["predicate"],
                object_id=row["object_id"],
                weight_positive=row["weight_positive"],
                weight_negative=row["weight_negative"],
                confidence=row["confidence"],
                source_experience_id=row["source_experience_id"],
                created_at=row["created_at"],
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Episodic Memory: Experiences
    # -------------------------------------------------------------------------

    def add_experience(
        self,
        input_text: str,
        extracted_triples: list[dict[str, Any]] | None = None,
        source: str = "user",
    ) -> Experience:
        exp = Experience.create(
            input_text=input_text,
            extracted_triples=extracted_triples,
            source=source,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO experiences (id, input_text, extracted_triples_json, source, timestamp)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    exp.id,
                    exp.input_text,
                    json.dumps(exp.extracted_triples),
                    exp.source,
                    exp.timestamp,
                ),
            )
        return exp

    def get_experience(self, exp_id: str) -> Experience | None:
        cursor = self._conn.execute(
            """
            SELECT id, input_text, extracted_triples_json, source, timestamp
            FROM experiences
            WHERE id = ?;
            """,
            (exp_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Experience(
            id=row["id"],
            input_text=row["input_text"],
            extracted_triples=json.loads(row["extracted_triples_json"]),
            source=row["source"],
            timestamp=row["timestamp"],
        )

    def list_experiences(self, limit: int = 50) -> list[Experience]:
        cursor = self._conn.execute(
            """
            SELECT id, input_text, extracted_triples_json, source, timestamp
            FROM experiences
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,),
        )
        return [
            Experience(
                id=row["id"],
                input_text=row["input_text"],
                extracted_triples=json.loads(row["extracted_triples_json"]),
                source=row["source"],
                timestamp=row["timestamp"],
            )
            for row in cursor.fetchall()
        ]

    # -------------------------------------------------------------------------
    # Procedural Memory: Learned Procedures
    # -------------------------------------------------------------------------

    def save_procedure(
        self,
        name: str,
        code_body: str,
        input_signature: str | None = None,
        output_signature: str | None = None,
    ) -> str:
        clean_name = name.strip().upper()
        proc_id = f"proc_{clean_name.lower()}"
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO procedures (id, name, code_body, input_signature, output_signature, created_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(name) DO UPDATE SET
                    code_body = excluded.code_body,
                    input_signature = excluded.input_signature,
                    output_signature = excluded.output_signature;
                """,
                (proc_id, clean_name, code_body, input_signature, output_signature),
            )
        return proc_id

    def get_procedure(self, name: str) -> dict[str, Any] | None:
        cursor = self._conn.execute(
            "SELECT id, name, code_body, input_signature, output_signature, created_at FROM procedures WHERE name = ?;",
            (name.strip().upper(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def save_skill(self, skill: Skill) -> str:
        """Save or update a procedural Skill."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO procedures (id, name, code_body, input_signature, output_signature, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    code_body = excluded.code_body,
                    input_signature = excluded.input_signature,
                    output_signature = excluded.output_signature;
                """,
                (
                    skill.id,
                    skill.name,
                    skill.code_body,
                    json.dumps(skill.parameters),
                    skill.description,
                    skill.created_at,
                ),
            )
        return skill.id

    def get_skill(self, name: str) -> Skill | None:
        """Retrieve a procedural Skill by name."""
        row = self.get_procedure(name)
        if not row:
            return None
        params: list[str] = []
        if row["input_signature"]:
            try:
                params = json.loads(row["input_signature"])
            except (json.JSONDecodeError, TypeError, ValueError):
                params = [
                    p.strip() for p in row["input_signature"].split(",") if p.strip()
                ]
        return Skill(
            id=row["id"],
            name=row["name"],
            description=row["output_signature"] or "",
            parameters=params,
            code_body=row["code_body"],
            created_at=row["created_at"],
        )

    def list_skills(self) -> list[Skill]:
        """List all procedural skills currently stored."""
        cursor = self._conn.execute(
            "SELECT id, name, code_body, input_signature, output_signature, created_at FROM procedures ORDER BY name ASC;"
        )
        skills: list[Skill] = []
        for row in cursor.fetchall():
            params: list[str] = []
            if row["input_signature"]:
                try:
                    params = json.loads(row["input_signature"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    params = [
                        p.strip()
                        for p in row["input_signature"].split(",")
                        if p.strip()
                    ]
            skills.append(
                Skill(
                    id=row["id"],
                    name=row["name"],
                    description=row["output_signature"] or "",
                    parameters=params,
                    code_body=row["code_body"],
                    created_at=row["created_at"],
                )
            )
        return skills

    # -------------------------------------------------------------------------
    # Construction Grammar Memory: Syntactic-Semantic Constructions
    # -------------------------------------------------------------------------

    def save_construction(self, construction: Construction) -> str:
        """Save or update a Construction Grammar mapping in memory."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO constructions (
                    id, name, pattern_tokens_json, slot_roles_json, predicate_template,
                    construction_type, is_negative, is_property, confidence,
                    evidence_positive, evidence_negative, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    pattern_tokens_json = excluded.pattern_tokens_json,
                    slot_roles_json = excluded.slot_roles_json,
                    predicate_template = excluded.predicate_template,
                    construction_type = excluded.construction_type,
                    is_negative = excluded.is_negative,
                    is_property = excluded.is_property,
                    confidence = excluded.confidence,
                    evidence_positive = excluded.evidence_positive,
                    evidence_negative = excluded.evidence_negative;
                """,
                (
                    construction.id,
                    construction.name,
                    json.dumps(construction.pattern_tokens),
                    json.dumps(construction.slot_roles),
                    construction.predicate_template,
                    construction.construction_type,
                    1 if construction.is_negative else 0,
                    1 if construction.is_property else 0,
                    construction.confidence,
                    construction.evidence_positive,
                    construction.evidence_negative,
                    construction.created_at,
                ),
            )
        return construction.id

    def get_construction(self, construction_id: str) -> Construction | None:
        """Retrieve a construction by ID."""
        cursor = self._conn.execute(
            """
            SELECT id, name, pattern_tokens_json, slot_roles_json, predicate_template,
                   construction_type, is_negative, is_property, confidence,
                   evidence_positive, evidence_negative, created_at
            FROM constructions WHERE id = ?;
            """,
            (construction_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_construction(row)

    def get_construction_by_name(self, name: str) -> Construction | None:
        """Retrieve a construction by its unique name."""
        cursor = self._conn.execute(
            """
            SELECT id, name, pattern_tokens_json, slot_roles_json, predicate_template,
                   construction_type, is_negative, is_property, confidence,
                   evidence_positive, evidence_negative, created_at
            FROM constructions WHERE name = ?;
            """,
            (name.strip().lower(),),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_construction(row)

    def list_constructions(
        self, construction_type: str | None = None
    ) -> list[Construction]:
        """List all stored grammatical constructions, optionally filtered by type."""
        if construction_type:
            cursor = self._conn.execute(
                """
                SELECT id, name, pattern_tokens_json, slot_roles_json, predicate_template,
                       construction_type, is_negative, is_property, confidence,
                       evidence_positive, evidence_negative, created_at
                FROM constructions WHERE construction_type = ? ORDER BY confidence DESC, name ASC;
                """,
                (construction_type.strip().lower(),),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT id, name, pattern_tokens_json, slot_roles_json, predicate_template,
                       construction_type, is_negative, is_property, confidence,
                       evidence_positive, evidence_negative, created_at
                FROM constructions ORDER BY confidence DESC, name ASC;
                """
            )
        return [self._row_to_construction(row) for row in cursor.fetchall()]

    def add_construction_evidence(
        self, construction_id: str, positive: bool = True
    ) -> None:
        """Reinforce or penalize a construction's confidence based on real-world usage."""
        col = "evidence_positive" if positive else "evidence_negative"
        with self._conn:
            self._conn.execute(
                f"UPDATE constructions SET {col} = {col} + 1 WHERE id = ?;",
                (construction_id,),
            )
            row = self._conn.execute(
                "SELECT evidence_positive, evidence_negative FROM constructions WHERE id = ?;",
                (construction_id,),
            ).fetchone()
            if row:
                pos = row["evidence_positive"]
                neg = row["evidence_negative"]
                conf = round(pos / (pos + neg + 1.0), 4)
                self._conn.execute(
                    "UPDATE constructions SET confidence = ? WHERE id = ?;",
                    (conf, construction_id),
                )

    def seed_default_constructions(self) -> None:
        """Seed core Construction Grammar patterns into SQLite if not already present."""
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM constructions;")
        row = cursor.fetchone()
        if row and row["cnt"] > 0:
            return

        defaults = [
            # Statements: Categorical & Taxonomic
            Construction.create(
                "cxn_is_a",
                ["{X}", "is", "a", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
            ),
            Construction.create(
                "cxn_is_an",
                ["{X}", "is", "an", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
            ),
            Construction.create(
                "cxn_is_bare",
                ["{X}", "is", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
            ),
            Construction.create(
                "cxn_are_bare",
                ["{X}", "are", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
            ),
            # Statements: Negative & Disjoint
            Construction.create(
                "cxn_is_not_a",
                ["{X}", "is", "not", "a", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_is_not_an",
                ["{X}", "is", "not", "an", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_is_not",
                ["{X}", "is", "not", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_are_not",
                ["{X}", "are", "not", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_cannot_be",
                ["{X}", "cannot", "be", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_is_disjoint_with",
                ["{X}", "is", "disjoint", "with", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            Construction.create(
                "cxn_is_different_from",
                ["{X}", "is", "different", "from", "{Y}"],
                {"X": "subject", "Y": "object"},
                "disjoint_with",
            ),
            # Statements: Possession & Capability
            Construction.create(
                "cxn_has_a",
                ["{X}", "has", "a", "{Y}"],
                {"X": "subject", "Y": "object"},
                "has",
            ),
            Construction.create(
                "cxn_has_an",
                ["{X}", "has", "an", "{Y}"],
                {"X": "subject", "Y": "object"},
                "has",
            ),
            Construction.create(
                "cxn_has_bare",
                ["{X}", "has", "{Y}"],
                {"X": "subject", "Y": "object"},
                "has",
            ),
            Construction.create(
                "cxn_have_bare",
                ["{X}", "have", "{Y}"],
                {"X": "subject", "Y": "object"},
                "has",
            ),
            Construction.create(
                "cxn_owns_a",
                ["{X}", "owns", "a", "{Y}"],
                {"X": "subject", "Y": "object"},
                "owns",
            ),
            Construction.create(
                "cxn_can_be",
                ["{X}", "can", "be", "{Y}"],
                {"X": "subject", "Y": "object"},
                "can_be",
            ),
            # Statements: Mereology / Part-Whole
            Construction.create(
                "cxn_part_of",
                ["{X}", "is", "part", "of", "{Y}"],
                {"X": "subject", "Y": "object"},
                "part_of",
            ),
            Construction.create(
                "cxn_part_of_a",
                ["{X}", "is", "a", "part", "of", "{Y}"],
                {"X": "subject", "Y": "object"},
                "part_of",
            ),
            # Questions: Identity & Definition
            Construction.create(
                "q_who_are_you",
                ["who", "are", "you"],
                {},
                "__identity__",
                construction_type="question",
            ),
            Construction.create(
                "q_what_are_you",
                ["what", "are", "you"],
                {},
                "__identity__",
                construction_type="question",
            ),
            Construction.create(
                "q_what_is",
                ["what", "is", "{X}"],
                {"X": "subject"},
                "__definition__",
                construction_type="question",
            ),
            Construction.create(
                "q_who_is",
                ["who", "is", "{X}"],
                {"X": "subject"},
                "__definition__",
                construction_type="question",
            ),
            Construction.create(
                "q_tell_me_about",
                ["tell", "me", "about", "{X}"],
                {"X": "subject"},
                "__definition__",
                construction_type="question",
            ),
            # Questions: Inquiry
            Construction.create(
                "q_is_a_a",
                ["is", "{X}", "a", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
                construction_type="question",
            ),
            Construction.create(
                "q_is_a_an",
                ["is", "{X}", "an", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
                construction_type="question",
            ),
            Construction.create(
                "q_is_bare",
                ["is", "{X}", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
                construction_type="question",
            ),
            Construction.create(
                "q_are_bare",
                ["are", "{X}", "{Y}"],
                {"X": "subject", "Y": "object"},
                "is_a",
                construction_type="question",
            ),
            Construction.create(
                "q_is_part_of",
                ["is", "{X}", "part", "of", "{Y}"],
                {"X": "subject", "Y": "object"},
                "part_of",
                construction_type="question",
            ),
            Construction.create(
                "q_is_part_of_a",
                ["is", "{X}", "a", "part", "of", "{Y}"],
                {"X": "subject", "Y": "object"},
                "part_of",
                construction_type="question",
            ),
            Construction.create(
                "q_does_have",
                ["does", "{X}", "have", "{Y}"],
                {"X": "subject", "Y": "object"},
                "has",
                construction_type="question",
            ),
            Construction.create(
                "q_can_be",
                ["can", "{X}", "be", "{Y}"],
                {"X": "subject", "Y": "object"},
                "can_be",
                construction_type="question",
            ),
            # Actions: Physical & Topological
            Construction.create(
                "act_slice_into_pieces",
                ["slice", "{X}", "into", "{count}", "pieces"],
                {"X": "object", "count": "count"},
                "SLICE",
                construction_type="action",
            ),
            Construction.create(
                "act_cut_into_pieces",
                ["cut", "{X}", "into", "{count}", "pieces"],
                {"X": "object", "count": "count"},
                "SLICE",
                construction_type="action",
            ),
        ]
        for c in defaults:
            self.save_construction(c)

    def _row_to_construction(self, row: sqlite3.Row) -> Construction:
        return Construction(
            id=row["id"],
            name=row["name"],
            pattern_tokens=json.loads(row["pattern_tokens_json"]),
            slot_roles=json.loads(row["slot_roles_json"]),
            predicate_template=row["predicate_template"],
            construction_type=row["construction_type"],
            is_negative=bool(row["is_negative"]),
            is_property=bool(row["is_property"]),
            confidence=row["confidence"],
            evidence_positive=row["evidence_positive"],
            evidence_negative=row["evidence_negative"],
            created_at=row["created_at"],
        )

    # -------------------------------------------------------------------------
    # Introspection & Export (PRD NFR-004 Requirement)
    # -------------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Export the entire cognitive memory state into an inspectable JSON dictionary."""
        return {
            "concepts": [c.to_dict() for c in self.list_concepts()],
            "relations": [r.to_dict() for r in self.get_relations()],
            "experiences": [e.to_dict() for e in self.list_experiences(limit=500)],
            "skills": [s.to_dict() for s in self.list_skills()],
            "constructions": [c.to_dict() for c in self.list_constructions()],
        }
