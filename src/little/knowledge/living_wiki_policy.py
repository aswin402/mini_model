"""Policy loader for the Living Graph Wiki and autonomous crawler."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class LivingWikiPolicy:
    """Configures vault directory, auto-sync triggers, and crawl limits."""

    vault_directory: str
    auto_sync_on_mutation: bool
    batch_crawl_limit: int
    verification_required: bool
    min_extraction_confidence: float

    @classmethod
    def load(cls, directory: str | Path) -> LivingWikiPolicy:
        path = Path(directory) / "living_wiki_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.living_wiki.v1":
            raise ValueError(f"{path} has an unsupported living-wiki format")

        vault_dir = str(payload.get("vault_directory", "data/knowledge_vault"))
        auto_sync = payload.get("auto_sync_on_mutation", True)
        crawl_limit = payload.get("batch_crawl_limit", 50)
        verif = payload.get("verification_required", True)
        min_conf = payload.get("min_extraction_confidence", 0.70)
        if type(auto_sync) is not bool or type(verif) is not bool:
            raise TypeError("Wiki controls must be booleans")
        if type(crawl_limit) is not int or crawl_limit < 0:
            raise ValueError("batch_crawl_limit must be a nonnegative integer")
        if (
            isinstance(min_conf, bool)
            or not isinstance(min_conf, (int, float))
            or not math.isfinite(min_conf)
            or not 0 <= min_conf <= 1
        ):
            raise ValueError("min_extraction_confidence must be between zero and one")

        return cls(
            vault_directory=vault_dir,
            auto_sync_on_mutation=auto_sync,
            batch_crawl_limit=crawl_limit,
            verification_required=verif,
            min_extraction_confidence=min_conf,
        )

    @classmethod
    def default(cls) -> LivingWikiPolicy:
        return cls.load(RuntimePaths.default().schema_directory)
