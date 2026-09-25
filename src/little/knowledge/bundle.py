"""Data-backed commonsense knowledge-pack loader.

The facts live in ``data/knowledge`` so changing domain knowledge does not
require editing executable Python code.  The loader keeps the historical
``get_commonsense_triples`` API used by the importer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths

Triple = tuple[str, str, str, float, bool]


def _parse_record(record: Any, path: Path, index: int) -> Triple:
    if isinstance(record, dict):
        values = (
            record.get("subject"),
            record.get("predicate"),
            record.get("object"),
            record.get("weight", 1.0),
            record.get("positive", True),
        )
    elif isinstance(record, list) and len(record) == 5:
        values = tuple(record)
    else:
        raise ValueError(
            f"{path} triple {index} must be an object or five-item array"
        )

    subject, predicate, object_, weight, positive = values
    if not all(isinstance(value, str) and value.strip() for value in values[:3]):
        raise ValueError(f"{path} triple {index} has invalid subject/predicate/object")
    if isinstance(positive, bool) is False:
        raise ValueError(f"{path} triple {index} has a non-boolean positive flag")

    return (
        subject.strip().lower(),
        predicate.strip().lower(),
        object_.strip().lower(),
        float(weight),
        positive,
    )


def load_knowledge_pack(path: str | Path) -> list[Triple]:
    """Load and validate a versioned LITTLE triple pack."""
    pack_path = Path(path)
    payload = json.loads(pack_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{pack_path} must contain a JSON object")
    if payload.get("format") != "little.knowledge.triples.v1":
        raise ValueError(f"{pack_path} has an unsupported knowledge-pack format")

    records = payload.get("triples")
    if not isinstance(records, list):
        raise TypeError(f"{pack_path} field 'triples' must be a list")
    return [_parse_record(record, pack_path, index) for index, record in enumerate(records)]


def get_commonsense_triples(path: str | Path | None = None) -> list[Triple]:
    """Return the configured commonsense pack as normalized triples."""
    return load_knowledge_pack(path or RuntimePaths.default().knowledge_pack)
