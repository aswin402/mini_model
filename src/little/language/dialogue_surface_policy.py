"""Versioned policy governing conversational surface responses, social protocols, and natural verbalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little.core.runtime_paths import RuntimePaths


def _expand_bnf_alternations(pat: str) -> set[str]:
    """Expand a BNF-style `(?:a|b)` / `(?:c)?` social pattern into normalized phrases without regex."""
    s = pat.strip().removeprefix("^").removesuffix("$")
    for suf in ("[?!.]?", "[!.]?", "[.!?]?", "[,!.]?"):
        s = s.removesuffix(suf)
    s = (
        s.replace(r"\s*,?\s*", " ")
        .replace(r"\s+", " ")
        .replace("hi+", "hi")
        .replace("hey+", "hey")
        .replace(r"(?: for [^.!?]+)?", "")
    )

    def _parse_seq(idx: int) -> tuple[list[str], int]:
        results = [""]
        n = len(s)
        while idx < n and s[idx] not in "|)":
            if s[idx : idx + 3] == "(?:":
                sub_alts, idx = _parse_alt(idx + 3)
                if idx < n and s[idx] == ")":
                    idx += 1
                is_opt = False
                if idx < n and s[idx] == "?":
                    is_opt = True
                    idx += 1
                if is_opt and "" not in sub_alts:
                    sub_alts = ["", *sub_alts]
                results = [r + a for r in results for a in sub_alts]
            elif s[idx] == "(":
                sub_alts, idx = _parse_alt(idx + 1)
                if idx < n and s[idx] == ")":
                    idx += 1
                is_opt = False
                if idx < n and s[idx] == "?":
                    is_opt = True
                    idx += 1
                if is_opt and "" not in sub_alts:
                    sub_alts = ["", *sub_alts]
                results = [r + a for r in results for a in sub_alts]
            else:
                j = idx
                while j < n and s[j] not in "|()":
                    j += 1
                lit = s[idx:j]
                if lit.endswith("?") and len(lit) >= 2:
                    # Single optional char like `s?`
                    base = lit[:-2]
                    opt_ch = lit[-2]
                    results = [r + base for r in results] + [
                        r + base + opt_ch for r in results
                    ]
                else:
                    results = [r + lit for r in results]
                idx = j
        return results, idx

    def _parse_alt(idx: int) -> tuple[list[str], int]:
        alts: list[str] = []
        while True:
            seq_res, idx = _parse_seq(idx)
            alts.extend(seq_res)
            if idx < len(s) and s[idx] == "|":
                idx += 1
            else:
                break
        return alts, idx

    raw, _ = _parse_alt(0)
    return {" ".join(x.replace(",", " ").lower().split()) for x in raw if x.strip()}


def _normalize_social_utterance(text: str) -> str | None:
    s = text.strip()
    if not s:
        return None
    if s[-1] in ".!?":
        s = s[:-1].strip()
    if any(ch in s for ch in ".!?"):
        return None
    words = s.replace(",", " ").lower().split()
    norm_words: list[str] = []
    for w in words:
        if w.startswith("hi") and len(w) > 2 and all(c == "i" for c in w[2:]):
            norm_words.append("hi")
        elif w.startswith("hey") and len(w) > 3 and all(c == "y" for c in w[3:]):
            norm_words.append("hey")
        else:
            norm_words.append(w)
    return " ".join(norm_words)


@dataclass(frozen=True)
class SocialIntent:
    name: str
    patterns: tuple[str, ...]
    responses: tuple[str, ...]


@dataclass(frozen=True)
class DialogueSurfacePolicy:
    """Policy for natural conversational surface dialogue."""

    social_intents: dict[str, SocialIntent]
    leading_greeting_regex: str
    verbalizer_templates: dict[str, str]

    @classmethod
    def load(cls, directory: str | Path) -> DialogueSurfacePolicy:
        path = Path(directory) / "dialogue_surface_policy.json"
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must contain a JSON object")
        if payload.get("format") != "little.dialogue_surface_policy.v1":
            raise ValueError(f"{path} has an unsupported dialogue-surface-policy format")

        social_dict: dict[str, SocialIntent] = {}
        raw_intents = payload.get("social_intents", {})
        for name, data in raw_intents.items():
            social_dict[name] = SocialIntent(
                name=name,
                patterns=tuple(data.get("patterns", [])),
                responses=tuple(data.get("responses", [])),
            )

        splitter = payload.get("mixed_intent_splitter", {})
        leading_rgx = splitter.get("leading_greeting_regex", "")
        templates = dict(payload.get("verbalizer_templates", {}))

        return cls(
            social_intents=social_dict,
            leading_greeting_regex=leading_rgx,
            verbalizer_templates=templates,
        )

    @classmethod
    def default(cls) -> DialogueSurfacePolicy:
        return cls.load(RuntimePaths.default().schema_directory)

    def match_social_intent(self, text: str) -> tuple[str, str] | None:
        """Match purely social inputs (greetings, pleasantries, identity) against policy."""
        norm = _normalize_social_utterance(text)
        if norm is None:
            return None
        for name, intent in self.social_intents.items():
            for pat in intent.patterns:
                phrases = _expand_bnf_alternations(pat)
                candidates = [norm]
                if "for" in pat and " for " in norm:
                    before_for, after_for = norm.split(" for ", 1)
                    candidates.append(before_for.strip())
                    for tail in (
                        "i greatly appreciate it",
                        "greatly appreciate it",
                        "i appreciate it",
                        "appreciate it",
                    ):
                        if after_for.endswith(tail):
                            candidates.append(f"{before_for.strip()} {tail}".strip())
                if any(c in phrases for c in candidates):
                    resp = intent.responses[0] if intent.responses else ""
                    return name, resp
        return None

    def decompose_leading_greeting(self, text: str) -> tuple[str, str] | None:
        """Decompose mixed utterance like 'Hello! Can an eagle fly?' into ('Hello', 'Can an eagle fly?')."""
        if not self.leading_greeting_regex:
            return None
        s = text.strip()
        if not s:
            return None
        prefix_pat = self.leading_greeting_regex
        for tail in (r"[,!.]?\s+(.+)$", r"\s+(.+)$"):
            if prefix_pat.endswith(tail):
                prefix_pat = prefix_pat[: -len(tail)]
                break
        greetings = sorted(_expand_bnf_alternations(prefix_pat), key=len, reverse=True)
        s_low = s.lower()
        for greet in greetings:
            g_words = greet.split()
            if not g_words:
                continue
            # Match g_words at start of s
            pos = 0
            matched = True
            n = len(s)
            for w_idx, gw in enumerate(g_words):
                while pos < n and s[pos].isspace():
                    pos += 1
                w_start = pos
                while pos < n and not s[pos].isspace() and s[pos] not in ",!.?":
                    pos += 1
                tok = s_low[w_start:pos]
                if (
                    gw == "hi" and tok.startswith("hi") and all(c == "i" for c in tok[2:])
                ) or (
                    gw == "hey" and tok.startswith("hey") and all(c == "y" for c in tok[3:])
                ):
                    pass
                elif tok != gw:
                    matched = False
                    break
                if w_idx < len(g_words) - 1 and (pos >= n or not s[pos].isspace()):
                    matched = False
                    break
            if not matched:
                continue
            greeting_word = s[:pos].strip()
            after_greet = pos
            if after_greet < n and s[after_greet] in ",!.":
                after_greet += 1
            if after_greet < n and s[after_greet].isspace():
                remainder = s[after_greet:].strip()
                if remainder:
                    return greeting_word, remainder
        return None

    def get_template(self, key: str, default: str = "") -> str:
        """Get verbalizer template string from loaded policy."""
        return self.verbalizer_templates.get(key, default)
