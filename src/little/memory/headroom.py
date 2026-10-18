"""Headroom Reversible Context Compression (CCR) & Working Memory Engine.

Maintains bounded active memory (<50MB RAM) via continuous salience scoring:
  Salience_i = w_recency * exp(-lambda * delta_t) + w_frequency * freq_i + w_activation * act_i

Compresses older/low-salience turns into compact semantic frames while keeping
fully reversible pointers to raw episodic experiences in SQLite MemoryStore.
"""

from __future__ import annotations

import math
import re
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from little.core.models import generate_id
from little.memory.headroom_policy import HeadroomPolicy

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


@dataclass
class WorkingTurn:
    """An episodic conversational turn in active working memory."""

    turn_id: str
    role: str  # "user" | "assistant" | "system"
    raw_text: str
    timestamp: float = field(default_factory=time.time)
    tokens: tuple[str, ...] = field(default_factory=tuple)
    concept_ids: tuple[str, ...] = field(default_factory=tuple)
    is_compressed: bool = False
    compressed_summary: str | None = None
    experience_id: str | None = None

    @property
    def token_count(self) -> int:
        if self.is_compressed:
            return max(1, len((self.compressed_summary or "").split()))
        return len(self.tokens) or max(1, len(self.raw_text.split()))


@dataclass(frozen=True)
class SalienceEntry:
    """Breakdown of multi-factor salience score."""

    key: str
    recency_score: float
    frequency_score: float
    activation_score: float
    total_salience: float


@dataclass(frozen=True)
class HeadroomStats:
    """Runtime statistics for active working memory bounds."""

    active_turns: int
    compressed_turns: int
    active_tokens: int
    active_concepts: int
    estimated_ram_kb: float
    eviction_count: int


class HeadroomContextManager:
    """Manages active conversation window with reversible compression and strict headroom limits."""

    def __init__(
        self,
        memory: MemoryStore,
        policy: HeadroomPolicy | None = None,
    ) -> None:
        self.memory = memory
        self.policy = policy or HeadroomPolicy.default()
        self._turns: list[WorkingTurn] = []
        self._concept_frequencies: dict[str, int] = {}
        self._concept_activations: dict[str, float] = {}
        self._concept_last_seen: dict[str, float] = {}
        self._eviction_count: int = 0
        self.session_id = generate_id("session")

    def _tokenize(self, text: str) -> tuple[str, ...]:
        clean = text.lower().strip()
        words = re.findall(r"[a-z0-9_]+", clean)
        return tuple(words)

    def record_turn(
        self,
        role: str,
        text: str,
        concept_ids: list[str] | tuple[str, ...] | None = None,
        experience_id: str | None = None,
    ) -> WorkingTurn:
        """Record a new conversational turn, update salience, and enforce headroom caps."""
        now = time.time()
        tokens = self._tokenize(text)
        c_ids = tuple(str(c).strip().lower() for c in (concept_ids or ()))

        experience_id = getattr(experience_id, "id", experience_id)
        changes_before = self.memory._conn.total_changes
        with self.memory.transaction():
            if experience_id is None or self.memory.get_experience(experience_id) is None:
                experience_id = self.memory.add_experience(
                    input_text=text, source=f"conversation:{role}"
                ).id
            turn = WorkingTurn(
            turn_id=generate_id("turn"),
            role=role,
            raw_text=text,
            timestamp=now,
            tokens=tokens,
            concept_ids=c_ids,
            experience_id=experience_id,
            )
            self.memory._conn.execute(
                "INSERT INTO working_turns (turn_id, session_id, role, experience_id, timestamp) VALUES (?, ?, ?, ?, ?)",
                (turn.turn_id, self.session_id, role, experience_id, now),
            )
        changes_delta = self.memory._conn.total_changes - changes_before
        self.memory._working_turn_changes = (
            getattr(self.memory, "_working_turn_changes", 0) + changes_delta
        )

        # Update concept activation statistics
        for cid in c_ids:
            self._concept_frequencies[cid] = self._concept_frequencies.get(cid, 0) + 1
            self._concept_activations[cid] = min(1.0, self._concept_activations.get(cid, 0.0) + 0.3)
            self._concept_last_seen[cid] = now

        self._turns.append(turn)
        self.enforce_headroom()
        return turn

    def compute_turn_salience(self, turn: WorkingTurn, now: float | None = None) -> SalienceEntry:
        """Compute salience score for a dialogue turn."""
        t_now = now if now is not None else time.time()
        dt = max(0.0, t_now - turn.timestamp)

        # 1. Recency: exp(-lambda * dt)
        recency = math.exp(-self.policy.recency_decay_lambda * dt)

        # 2. Concept relevance: average activation of concepts mentioned
        if turn.concept_ids:
            freq_sum = sum(min(1.0, self._concept_frequencies.get(c, 1) / 5.0) for c in turn.concept_ids)
            freq_score = freq_sum / len(turn.concept_ids)
            act_sum = sum(self._concept_activations.get(c, 0.5) for c in turn.concept_ids)
            act_score = act_sum / len(turn.concept_ids)
        else:
            freq_score = 0.2
            act_score = 0.2

        w = self.policy.salience_weights
        total = (
            w.get("recency", 0.4) * recency
            + w.get("frequency", 0.3) * freq_score
            + w.get("activation", 0.3) * act_score
        )

        return SalienceEntry(
            key=turn.turn_id,
            recency_score=round(recency, 4),
            frequency_score=round(freq_score, 4),
            activation_score=round(act_score, 4),
            total_salience=round(total, 4),
        )

    def total_active_tokens(self) -> int:
        """Calculate current active token load in working memory."""
        return sum(t.token_count for t in self._turns)

    def enforce_headroom(self) -> None:
        """Ensure active tokens and concepts remain strictly under policy limits."""
        # 1. Check token limits: compress older, low-salience turns
        while self.total_active_tokens() > self.policy.max_active_tokens and self._turns:
            # Find candidate for compression (oldest non-compressed turn, excluding last 2 turns)
            compress_candidate: WorkingTurn | None = None
            min_salience = 999.0

            eligible = self._turns[:-2] if len(self._turns) > 2 else self._turns
            for turn in eligible:
                if not turn.is_compressed:
                    sal = self.compute_turn_salience(turn).total_salience
                    if sal < min_salience:
                        min_salience = sal
                        compress_candidate = turn

            if compress_candidate:
                self._compress_turn(compress_candidate)
            else:
                # All eligible turns are compressed; evict oldest compressed turn
                self._evict_oldest_compressed()

        while len(self._turns) > self.policy.max_working_episodes:
            self._compress_turn(self._turns[0])
            self._evict_oldest_compressed()

        # This is the Python working-set estimate, not a bound on process RSS.
        while self._turns and self.stats().estimated_ram_kb > self.policy.memory_headroom_budget_mb * 1024:
            candidate = next((turn for turn in self._turns if not turn.is_compressed), None)
            if candidate is not None:
                self._compress_turn(candidate)
            else:
                self._evict_oldest_compressed()

        # 2. Check concept limits: prune lowest-salience concept entries
        if len(self._concept_frequencies) > self.policy.max_active_concepts:
            now = time.time()
            scored_concepts = []
            for cid in self._concept_frequencies:
                dt = max(0.0, now - self._concept_last_seen.get(cid, now))
                rec = math.exp(-self.policy.recency_decay_lambda * dt)
                freq = min(1.0, self._concept_frequencies[cid] / 5.0)
                act = self._concept_activations.get(cid, 0.5)
                sal = 0.4 * rec + 0.3 * freq + 0.3 * act
                scored_concepts.append((sal, cid))

            scored_concepts.sort(key=lambda item: item[0])
            to_remove = len(self._concept_frequencies) - self.policy.max_active_concepts
            for _, cid in scored_concepts[:to_remove]:
                self._concept_frequencies.pop(cid, None)
                self._concept_activations.pop(cid, None)
                self._concept_last_seen.pop(cid, None)

    def _compress_turn(self, turn: WorkingTurn) -> None:
        """Reversible Context Compression (CCR): summarize turn into structured semantic pointer."""
        c_names = ", ".join(turn.concept_ids) if turn.concept_ids else "general"
        summary = f"[{turn.role.upper()} discussed {c_names} (ref: {turn.experience_id or turn.turn_id})]"
        turn.compressed_summary = " ".join(summary.split()[: self.policy.max_active_tokens])
        turn.is_compressed = True
        turn.raw_text = ""
        turn.tokens = ()

    def _evict_oldest_compressed(self) -> None:
        """Evict oldest compressed turn from RAM to disk."""
        for i, turn in enumerate(self._turns):
            if turn.is_compressed:
                self._turns.pop(i)
                self._eviction_count += 1
                return

    def decompress_turn(self, turn_id: str) -> str:
        """Reversible Context Retrieval: restore full text from SQLite or turn cache."""
        for turn in self._turns:
            if turn.turn_id == turn_id:
                if not turn.is_compressed:
                    return turn.raw_text
                # If experience_id is available, check SQLite
                if turn.experience_id:
                    target_id = getattr(turn.experience_id, "id", turn.experience_id)
                    exp = self.memory.get_experience(target_id)
                    if exp:
                        return exp.input_text
                return turn.raw_text
        row = self.memory._conn.execute(
            "SELECT experience_id FROM working_turns WHERE turn_id=?", (turn_id,)
        ).fetchone()
        if row:
            experience = self.memory.get_experience(row["experience_id"])
            if experience is not None:
                return experience.input_text
        raise KeyError(f"Turn '{turn_id}' has no persisted experience")

    def recent_turns(self, role: str, limit: int = 3) -> list[tuple[str, str]]:
        """Return newest persisted turns in this conversation, including evictions."""
        rows = self.memory._conn.execute(
            "SELECT turn_id, experience_id FROM working_turns WHERE session_id=? AND role=? "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?", (self.session_id, role, limit),
        ).fetchall()
        return [(row["turn_id"], self.decompress_turn(row["turn_id"])) for row in rows]

    def get_active_context(self) -> list[dict[str, str]]:
        """Return formatted working memory context for reasoning engines."""
        context = []
        for turn in self._turns:
            content = turn.compressed_summary if turn.is_compressed else turn.raw_text
            context.append({"role": turn.role, "content": content or ""})
        return context

    def get_active_concepts(self) -> list[str]:
        """Return list of active concepts sorted by salience."""
        return sorted(
            self._concept_frequencies.keys(),
            key=lambda c: self._concept_activations.get(c, 0.0),
            reverse=True,
        )

    def stats(self) -> HeadroomStats:
        """Return current memory and compression metrics."""
        compressed = sum(1 for t in self._turns if t.is_compressed)
        # Approximate working memory in KB
        total_bytes = sys.getsizeof(self._turns)
        for turn in self._turns:
            total_bytes += sys.getsizeof(turn) + sys.getsizeof(turn.__dict__)
            total_bytes += sum(sys.getsizeof(value) for value in turn.__dict__.values())
            total_bytes += sum(sys.getsizeof(token) for token in turn.tokens)
        for values in (self._concept_frequencies, self._concept_activations, self._concept_last_seen):
            total_bytes += sys.getsizeof(values) + sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in values.items())
        total_kb = round(total_bytes / 1024.0, 2)

        return HeadroomStats(
            active_turns=len(self._turns),
            compressed_turns=compressed,
            active_tokens=self.total_active_tokens(),
            active_concepts=len(self._concept_frequencies),
            estimated_ram_kb=total_kb,
            eviction_count=self._eviction_count,
        )
