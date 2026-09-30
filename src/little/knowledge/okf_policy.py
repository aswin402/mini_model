"""Policy loader for Google OKF (Open Knowledge Format) files and vault management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class OKFPolicy:
    """Configures OKF frontmatter schema, file extension, and vault directories."""

    vault_directory: str
    file_extension: str
    wikilink_pattern: str
    required_frontmatter_fields: tuple[str, ...]
    supported_frontmatter_fields: tuple[str, ...]
    default_citation_source: str

    @classmethod
    def load(cls, directory: str | Path) -> OKFPolicy:
        path = Path(directory) / "okf_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.okf.v1":
            raise ValueError(f"{path} has an unsupported okf format")

        vault_dir = str(payload.get("vault_directory", "data/knowledge_vault"))
        ext = str(payload.get("file_extension", ".okf.md"))
        pattern = str(payload.get("wikilink_pattern", r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]"))
        req_fields = tuple(str(f) for f in payload.get("required_frontmatter_fields", ["id", "name", "taxonomy"]))
        sup_fields = tuple(str(f) for f in payload.get("supported_frontmatter_fields", []))
        default_src = str(payload.get("default_citation_source", "document"))

        return cls(
            vault_directory=vault_dir,
            file_extension=ext,
            wikilink_pattern=pattern,
            required_frontmatter_fields=req_fields,
            supported_frontmatter_fields=sup_fields,
            default_citation_source=default_src,
        )

    @classmethod
    def default(cls) -> OKFPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
