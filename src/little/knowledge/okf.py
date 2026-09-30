"""Google OKF (Open Knowledge Format) Importer and Exporter for LITTLE.

Implements versioned Markdown + YAML frontmatter knowledge documents with
bidirectional [[Concept]] wikilinking, DeepSeek-R1 invariant verification
(I_DAG, I_MUTEX, I_SORT, I_GROUND), and atomic sync with SQLite MemoryStore.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from little.core.contracts import (
    CandidateClaim,
    EvidenceRecord,
    EvidenceStatus,
    SourceType,
    VerificationStatus,
)
from little.core.models import Concept, ConceptStatus, current_iso_timestamp, generate_id
from little.inference.invariant_gates import DeepSeekInvariantVerifier
from little.knowledge.okf_policy import OKFPolicy
from little.knowledge.registry import SchemaRegistry
from little.memory.store import MemoryStore


@dataclass(frozen=True)
class OKFWikilink:
    """A hyperlink to another concept knot within the living knowledge vault."""

    target: str
    display: str


@dataclass
class OKFDocument:
    """An in-memory representation of an OKF Markdown knowledge document."""

    frontmatter: dict[str, Any]
    body: str
    wikilinks: list[OKFWikilink] = field(default_factory=list)

    @property
    def concept_name(self) -> str:
        return str(self.frontmatter.get("name", "")).strip().lower()

    @property
    def concept_id(self) -> str:
        raw_id = self.frontmatter.get("id")
        if raw_id:
            return str(raw_id).strip()
        name = self.concept_name
        return f"CONCEPT_{name.upper().replace(' ', '_')}"


@dataclass
class OKFImportResult:
    """Outcome of parsing, verifying, and committing an OKF file."""

    concept_name: str
    file_path: str
    relations_added: int = 0
    relations_rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    passed_verification: bool = True
    new_concept_created: bool = False


def serialize_yaml_value(val: Any, indent: int = 0) -> str:
    """Deterministic, clean YAML serialization without third-party dependencies."""
    ind = "  " * indent
    if isinstance(val, dict):
        if not val:
            return "{}"
        lines = []
        for k, v in val.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{ind}{k}:")
                lines.append(serialize_yaml_value(v, indent + 1))
            else:
                lines.append(f"{ind}{k}: {serialize_yaml_value(v, 0)}")
        return "\n".join(lines)
    elif isinstance(val, (list, tuple)):
        if not val:
            return "[]"
        lines = []
        for item in val:
            if isinstance(item, dict):
                keys = list(item.keys())
                if not keys:
                    lines.append(f"{ind}- {{}}")
                    continue
                first_k = keys[0]
                lines.append(f"{ind}- {first_k}: {serialize_yaml_value(item[first_k], 0)}")
                for k in keys[1:]:
                    lines.append(f"{ind}  {k}: {serialize_yaml_value(item[k], 0)}")
            elif isinstance(item, (list, tuple)):
                lines.append(f"{ind}-")
                lines.append(serialize_yaml_value(item, indent + 1))
            else:
                lines.append(f"{ind}- {serialize_yaml_value(item, 0)}")
        return "\n".join(lines)
    elif isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (int, float)):
        return str(val)
    elif val is None:
        return "null"
    else:
        s = str(val)
        if any(c in s for c in ":\n#{}[]\"'") or s.lower() in ("true", "false", "null"):
            return json.dumps(s)
        return s


def parse_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter delimited by '---' markers into Python dictionary and markdown body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    fm_lines = []
    body_idx = -1
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_idx = idx + 1
            break
        fm_lines.append(line)

    if body_idx == -1:
        return {}, text

    body = "\n".join(lines[body_idx:]).strip()

    def parse_scalar(val_str: str) -> Any:
        v = val_str.strip()
        if not v:
            return None
        low = v.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in ("null", "none", "~"):
            return None
        if v == "[]":
            return []
        if v == "{}":
            return {}
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    def parse_block(block_lines: list[str], start_i: int, base_indent: int) -> tuple[Any, int]:
        res_dict: dict[str, Any] = {}
        res_list: list[Any] = []
        is_list = False
        i = start_i

        while i < len(block_lines):
            line = block_lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            indent = len(line) - len(line.lstrip())
            if indent < base_indent:
                break

            if stripped.startswith("- "):
                is_list = True
                content = stripped[2:].strip()
                if ":" in content and not content.startswith("{"):
                    k, v = content.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    item_dict: dict[str, Any] = {}
                    if v:
                        item_dict[k] = parse_scalar(v)
                    else:
                        sub, next_i = parse_block(block_lines, i + 1, indent + 2)
                        item_dict[k] = sub
                        i = next_i
                        res_list.append(item_dict)
                        continue

                    j = i + 1
                    while j < len(block_lines):
                        nxt = block_lines[j]
                        nxt_s = nxt.strip()
                        if not nxt_s or nxt_s.startswith("#"):
                            j += 1
                            continue
                        nxt_ind = len(nxt) - len(nxt.lstrip())
                        if nxt_ind == indent + 2 and ":" in nxt_s and not nxt_s.startswith("- "):
                            sub_k, sub_v = nxt_s.split(":", 1)
                            item_dict[sub_k.strip()] = parse_scalar(sub_v)
                            j += 1
                        else:
                            break
                    res_list.append(item_dict)
                    i = j
                    continue
                elif content:
                    res_list.append(parse_scalar(content))
                    i += 1
                else:
                    sub, next_i = parse_block(block_lines, i + 1, indent + 2)
                    res_list.append(sub)
                    i = next_i
            else:
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v:
                        res_dict[k] = parse_scalar(v)
                        i += 1
                    else:
                        sub, next_i = parse_block(block_lines, i + 1, indent + 2)
                        res_dict[k] = sub
                        i = next_i
                else:
                    i += 1

        return (res_list if is_list else res_dict), i

    parsed, _ = parse_block(fm_lines, 0, 0)
    return (parsed if isinstance(parsed, dict) else {}), body


class OKFEngine:
    """Bidirectional manager syncing Concept Knots with OKF Markdown files."""

    def __init__(
        self,
        memory: MemoryStore,
        policy: OKFPolicy | None = None,
        registry: SchemaRegistry | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or OKFPolicy.default()
        self.registry = registry or SchemaRegistry.default()
        self.verifier = DeepSeekInvariantVerifier(memory, self.registry)
        self.wikilink_re = re.compile(self.policy.wikilink_pattern)

    def parse_document(self, text: str) -> OKFDocument:
        """Parse raw text into an OKFDocument with extracted frontmatter and wikilinks."""
        fm, body = parse_yaml_frontmatter(text)
        matches = self.wikilink_re.findall(body)
        links = []
        for target, display in matches:
            tgt = target.strip().lower()
            dsp = display.strip() if display else tgt
            links.append(OKFWikilink(target=tgt, display=dsp))
        return OKFDocument(frontmatter=fm, body=body, wikilinks=links)

    def parse_file(self, file_path: str | Path) -> OKFDocument:
        """Read and parse an OKF file from disk."""
        path = Path(file_path)
        return self.parse_document(path.read_text(encoding="utf-8"))

    def export_concept(
        self,
        concept_name: str,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Export a concept and its relations from MemoryStore into an OKF document."""
        clean_name = concept_name.strip().lower()
        concept = self.memory.get_concept(clean_name)
        if not concept:
            raise KeyError(f"Concept '{clean_name}' does not exist in memory")

        out_dir = Path(output_dir or self.policy.vault_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{clean_name.replace(' ', '_')}{self.policy.file_extension}"

        # Group relations
        all_relations = self.memory.get_relations(subject_id=concept.id)
        is_a_parents = []
        disjoint_concepts = []
        parts = []
        skills = []
        custom_relations = []

        for rel in all_relations:
            if rel.subject_id != concept.id:
                continue
            obj_concept = self.memory.get_concept(rel.object_id)
            obj_name = obj_concept.name if obj_concept else rel.object_id

            if rel.predicate == "is_a":
                is_a_parents.append(obj_name)
            elif rel.predicate in ("disjoint_with", "cannot_be"):
                disjoint_concepts.append(obj_name)
            elif rel.predicate in ("has_part", "part_of", "has"):
                parts.append(obj_name)
            elif rel.predicate in ("capable_of", "can"):
                skills.append(obj_name)
            else:
                custom_relations.append({
                    "predicate": rel.predicate,
                    "object": obj_name,
                    "confidence": round(rel.confidence, 3),
                })

        # Build frontmatter
        fm: dict[str, Any] = {
            "id": concept.id,
            "name": concept.name,
            "aliases": concept.aliases,
            "taxonomy": {
                "is_a": sorted(is_a_parents),
                "disjoint_with": sorted(disjoint_concepts),
            },
            "mereology": {
                "parts": sorted(parts),
            },
            "procedural_skills": sorted(skills),
            "attributes": concept.attributes,
            "confidence": round(concept.confidence, 3),
            "status": concept.status.value,
        }
        if custom_relations:
            fm["relations"] = custom_relations

        # Build Markdown Body with [[Wikilinks]]
        title = concept.name.title()
        body_lines = [
            f"# {title}",
            "",
            f"**{title}** is a concept registered in the LITTLE living knowledge vault.",
        ]

        if is_a_parents:
            parents_links = ", ".join(f"[[{p}]]" for p in is_a_parents)
            body_lines.append(f"- **Taxonomic Hierarchy**: Belongs to {parents_links}.")
        if parts:
            parts_links = ", ".join(f"[[{p}]]" for p in parts)
            body_lines.append(f"- **Mereological Structure**: Composed of {parts_links}.")
        if skills:
            skills_links = ", ".join(f"[[{s}]]" for s in skills)
            body_lines.append(f"- **Procedural Capabilities**: Exhibits skills {skills_links}.")

        body_lines.append("")
        body_lines.append("## Notes & Empirical Grounding")
        body_lines.append(f"Recorded in persistent SQLite memory with confidence {concept.confidence:.2f}.")

        content = f"---\n{serialize_yaml_value(fm)}\n---\n\n" + "\n".join(body_lines) + "\n"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def export_vault(self, dest_dir: str | Path | None = None) -> list[Path]:
        """Export all active concepts in memory to the OKF knowledge vault."""
        out_dir = Path(dest_dir or self.policy.vault_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for concept in self.memory.list_concepts():
            if concept.status == ConceptStatus.ACTIVE:
                p = self.export_concept(concept.name, out_dir)
                paths.append(p)
        return paths

    def import_okf_file(self, file_path: str | Path) -> OKFImportResult:
        """Parse, verify against DeepSeek-R1 gates, and commit an OKF document."""
        path = Path(file_path)
        doc = self.parse_file(path)
        name = doc.concept_name
        if not name:
            return OKFImportResult(
                concept_name="",
                file_path=str(path),
                passed_verification=False,
                rejection_reasons=["Document frontmatter missing valid 'name' field"],
            )

        # 1. Ensure or update concept
        new_created = False
        concept = self.memory.get_concept(name)
        if not concept:
            aliases = list(doc.frontmatter.get("aliases", []))
            attrs = dict(doc.frontmatter.get("attributes", {}))
            conf = float(doc.frontmatter.get("confidence", 1.0))
            concept = self.memory.create_concept(
                name=name,
                aliases=aliases,
                attributes=attrs,
                confidence=conf,
            )
            new_created = True

        assert concept is not None

        # 2. Extract all candidate claims from taxonomy, mereology, skills, and relations
        candidate_claims: list[CandidateClaim] = []

        tax = doc.frontmatter.get("taxonomy", {})
        if isinstance(tax, dict):
            is_a_parents = tax.get("is_a", [])
            if isinstance(is_a_parents, list):
                for parent in is_a_parents:
                    if parent and isinstance(parent, str) and parent.strip():
                        candidate_claims.append(
                            CandidateClaim(subject=name, predicate="is_a", object=parent.strip().lower(), probability=1.0)
                        )
            disjoints = tax.get("disjoint_with", [])
            if isinstance(disjoints, list):
                for disjoint in disjoints:
                    if disjoint and isinstance(disjoint, str) and disjoint.strip():
                        candidate_claims.append(
                            CandidateClaim(subject=name, predicate="disjoint_with", object=disjoint.strip().lower(), probability=1.0)
                        )

        mereo = doc.frontmatter.get("mereology", {})
        if isinstance(mereo, dict):
            parts = mereo.get("parts", [])
            if isinstance(parts, list):
                for part in parts:
                    if part and isinstance(part, str) and part.strip():
                        candidate_claims.append(
                            CandidateClaim(subject=name, predicate="has_part", object=part.strip().lower(), probability=1.0)
                        )

        skills = doc.frontmatter.get("procedural_skills", [])
        if isinstance(skills, list):
            for skill in skills:
                if skill and isinstance(skill, str) and skill.strip():
                    candidate_claims.append(
                        CandidateClaim(subject=name, predicate="can", object=skill.strip().lower(), probability=1.0)
                    )

        for rel in doc.frontmatter.get("relations", []):
            if isinstance(rel, dict):
                p = str(rel.get("predicate", "")).strip().lower()
                o = str(rel.get("object", "")).strip().lower()
                prob = float(rel.get("confidence", 1.0))
                if p and o:
                    candidate_claims.append(
                        CandidateClaim(subject=name, predicate=p, object=o, probability=prob)
                    )

        # Also extract implicit relations from [[Wikilinks]] if not already present
        for link in doc.wikilinks:
            if link.target and link.target != name:
                # Default associative relation if not specifically asserted
                exists = any(c.object == link.target for c in candidate_claims)
                if not exists:
                    candidate_claims.append(
                        CandidateClaim(subject=name, predicate="related_to", object=link.target, probability=0.85)
                    )

        # 3. Verify each candidate claim with DeepSeek-R1 Invariant Verification Gates
        added = 0
        rejected = 0
        reasons = []

        for claim in candidate_claims:
            # Ensure target concept exists in memory so graph checks can evaluate
            target_concept = self.memory.get_or_create_concept(claim.object)

            gate_result = self.verifier.verify_relation(
                subject=claim.subject,
                predicate=claim.predicate,
                obj=claim.object,
            )

            if not gate_result.passed:
                rejected += 1
                msg = f"Rejected ({claim.subject}, {claim.predicate}, {claim.object}): {gate_result.error_message}"
                reasons.append(msg)
                continue

            # Invariant gates passed: commit relation and evidence
            self.memory.add_relation(
                subject_id=concept.id,
                predicate=claim.predicate,
                object_id=target_concept.id,
                positive=True,
            )
            # Record evidence record
            evidence_rec = EvidenceRecord.from_claim(
                claim=claim,
                source_type=SourceType.DOCUMENT,
                source_reference=str(path),
                source_text=f"OKF vault document: {path.name}",
            )
            evidence_rec.status = EvidenceStatus.ACCEPTED
            self.memory.evidence.append(evidence_rec)
            added += 1

        return OKFImportResult(
            concept_name=name,
            file_path=str(path),
            relations_added=added,
            relations_rejected=rejected,
            rejection_reasons=reasons,
            passed_verification=(rejected == 0),
            new_concept_created=new_created,
        )

    def import_vault(self, vault_dir: str | Path | None = None) -> list[OKFImportResult]:
        """Import all OKF files in a directory."""
        v_dir = Path(vault_dir or self.policy.vault_directory)
        if not v_dir.exists():
            return []
        results = []
        for file_path in sorted(v_dir.glob(f"*{self.policy.file_extension}")):
            res = self.import_okf_file(file_path)
            results.append(res)
        return results
