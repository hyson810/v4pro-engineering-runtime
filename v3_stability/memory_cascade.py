"""V3 Memory Cascade — hierarchical context lifecycle management.

The core mechanism that prevents model degradation over long sessions.
Maintains 4 memory tiers and cascade rules that keep the model's working
context clean, high-signal, and actionable.

Tiers:
  L0 — Current workspace: past 50 turns, full detail
  L1 — Recent history: 50-200 turns, decisions + outcomes
  L2 — Mid-term memory: 200-500 turns, patterns + learnings
  L3 — Long-term memory: 500+ turns, abstract insights

All compression is done locally — zero LLM token overhead.
"""

import os
import json
import time
from collections import deque
from dataclasses import dataclass, field

from shared.utils import now_ts, short_hash, detect_error_pattern, extract_keywords


@dataclass
class MemoryEntry:
    """A single record in any memory tier."""
    turn: int
    timestamp: int
    kind: str           # "action", "decision", "error", "discovery", "reflection"
    summary: str        # compressed description
    detail: str = ""    # full detail (only in L0)
    keywords: list[str] = field(default_factory=list)
    error_type: str | None = None
    parent_turn: int | None = None  # for causal linking

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "summary": self.summary,
            "detail": self.detail,
            "keywords": self.keywords,
            "error_type": self.error_type,
            "parent_turn": self.parent_turn,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**d)


class MemoryCascade:
    """Four-tier memory with automatic compression cascade."""

    def __init__(self, state_dir: str, config=None):
        self.state_dir = state_dir
        self.config = config

        # Initialize tiers
        self.l0: deque[MemoryEntry] = deque(maxlen=50)    # Current
        self.l1: list[MemoryEntry] = []                    # Recent, max 150
        self.l2: list[MemoryEntry] = []                    # Mid-term, max 300
        self.l3: list[MemoryEntry] = []                    # Long-term, unbounded

        self.turn_counter: int = 0
        self.last_error: str | None = None
        self.consecutive_same_error: int = 0
        self.error_history: list[tuple[int, str]] = []     # (turn, error_type)
        self.decisions: list[MemoryEntry] = []              # key decisions only
        self.discoveries: list[MemoryEntry] = []            # key discoveries only

        self._load()

    # ─── Public API ───────────────────────────────────────────

    def record(self, kind: str, summary: str, detail: str = "",
               error_type: str = None, parent_turn: int = None):
        """Record a new memory entry. Called after every significant model action."""
        self.turn_counter += 1

        entry = MemoryEntry(
            turn=self.turn_counter,
            timestamp=now_ts(),
            kind=kind,
            summary=summary,
            detail=detail,
            keywords=extract_keywords(summary),
            error_type=error_type or detect_error_pattern(detail or summary),
            parent_turn=parent_turn,
        )

        self.l0.append(entry)

        # Track errors
        if entry.error_type:
            self._track_error(entry)

        # Track decisions and discoveries
        if kind == "decision":
            self.decisions.append(entry)
        if kind == "discovery":
            self.discoveries.append(entry)

        # Check cascade triggers
        self._maybe_cascade()

        # Persist
        self._save()

    def reflect(self) -> str:
        """Generate a structured reflection of what's been learned.

        Returns a string suitable for injecting into the model's context.
        This costs ~200 tokens for the model to read, but saves thousands
        by preventing redundant exploration.
        """
        recent_errors = [e for e in self.l0 if e.error_type]
        recent_decisions = [e for e in self.l0 if e.kind == "decision"]
        recent_discoveries = [e for e in self.l0 if e.kind == "discovery"]

        parts = [f"## Reflection (turn {self.turn_counter})\n"]

        # Error patterns
        if recent_errors:
            error_types = set(e.error_type for e in recent_errors)
            parts.append(f"**Recent errors ({len(recent_errors)}):** "
                         f"{', '.join(error_types)}")
            if self.consecutive_same_error >= self.config.error_loop_threshold if self.config else 3:
                parts.append(f"⚠️ Same error pattern repeated "
                             f"{self.consecutive_same_error} times. "
                             f"Consider a fundamentally different approach.")

        # Key decisions
        if recent_decisions:
            parts.append(f"\n**Key decisions this phase:**")
            for d in recent_decisions[-3:]:
                parts.append(f"- [T{d.turn}] {d.summary}")

        # Discoveries
        if recent_discoveries:
            parts.append(f"\n**Discoveries:**")
            for d in recent_discoveries[-3:]:
                parts.append(f"- [T{d.turn}] {d.summary}")

        # Cross-tier insights
        if self.l2:
            parts.append(f"\n**Historical patterns (L2):** "
                         f"{len(self.l2)} entries covering "
                         f"turns {self.l2[0].turn}-{self.l2[-1].turn}")

        if self.l3:
            parts.append(f"\n**Long-term lessons (L3):** "
                         f"{len(self.l3)} accumulated insights")

        # Strategic recommendation
        if self.consecutive_same_error >= (self.config.error_loop_threshold if self.config else 3):
            parts.append(f"\n**Recommendation:** Pause. Review the root cause. "
                         f"Try a different strategy rather than fixing symptoms.")

        # Record this reflection
        reflection_text = "\n".join(parts)
        self.record(
            kind="reflection",
            summary=f"Auto-reflection at turn {self.turn_counter}",
            detail=reflection_text,
        )

        return reflection_text

    def recall(self, keywords: list[str] | str, max_results: int = 5) -> list[MemoryEntry]:
        """Search across all tiers for entries matching keywords."""
        if isinstance(keywords, str):
            keywords = extract_keywords(keywords)

        results = []
        all_entries = list(self.l0) + self.l1 + self.l2 + self.l3

        for entry in reversed(all_entries):
            text = f"{entry.summary} {' '.join(entry.keywords)}".lower()
            score = sum(1 for kw in keywords if kw.lower() in text)
            if score > 0:
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:max_results]]

    def status(self) -> dict:
        """Return current memory system status."""
        return {
            "turn": self.turn_counter,
            "tiers": {
                "L0_current": len(self.l0),
                "L1_recent": len(self.l1),
                "L2_midterm": len(self.l2),
                "L3_longterm": len(self.l3),
            },
            "total_entries": len(self.l0) + len(self.l1) + len(self.l2) + len(self.l3),
            "total_decisions": len(self.decisions),
            "total_discoveries": len(self.discoveries),
            "consecutive_same_error": self.consecutive_same_error,
            "last_error": self.last_error,
            "error_history_len": len(self.error_history),
        }

    def reset(self):
        """Reset all tiers. Use when starting a new task."""
        self.l0.clear()
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.turn_counter = 0
        self.last_error = None
        self.consecutive_same_error = 0
        self.error_history.clear()
        self.decisions.clear()
        self.discoveries.clear()
        self._save()

    # ─── Internal: Cascade Logic ─────────────────────────────

    def _maybe_cascade(self):
        """Check and execute memory cascades."""
        l0_max = self.config.cascade_l0_max_turns if self.config else 50
        l1_max = self.config.cascade_l1_max_turns if self.config else 200
        l2_max = self.config.cascade_l2_max_turns if self.config else 500

        # L0 → L1: when L0 exceeds threshold, compress oldest entries to L1
        if len(self.l0) >= l0_max:
            overflow = len(self.l0) - l0_max // 2  # keep half
            for _ in range(min(overflow, len(self.l0))):
                entry = self.l0.popleft()
                # Strip detail for L1 (keep summary + keywords)
                entry.detail = ""
                self.l1.append(entry)

        # L1 → L2: when L1 too large, distill to patterns
        if len(self.l1) >= l1_max:
            # Group L1 entries by error type and kind, keep representatives
            self._compress_l1_to_l2()

        # L2 → L3: distill to abstract insights
        if len(self.l2) >= l2_max:
            self._compress_l2_to_l3()

    def _compress_l1_to_l2(self):
        """Compress L1 (detailed recent) to L2 (patterns)."""
        # Group entries by error type
        error_groups: dict[str, list[MemoryEntry]] = {}
        for entry in self.l1:
            if entry.error_type:
                error_groups.setdefault(entry.error_type, []).append(entry)

        # Create pattern entries for repeated errors
        for err_type, entries in error_groups.items():
            if len(entries) >= 3:
                pattern = MemoryEntry(
                    turn=entries[-1].turn,
                    timestamp=now_ts(),
                    kind="pattern",
                    summary=f"Repeated error pattern: {err_type} "
                            f"({len(entries)} occurrences in L1)",
                    keywords=[err_type, "pattern", "recurring"],
                    error_type=err_type,
                )
                self.l2.append(pattern)

        # Keep only the most recent 50% of L1, delete the rest
        keep = len(self.l1) // 2
        self.l1 = self.l1[-keep:]

    def _compress_l2_to_l3(self):
        """Distill L2 (patterns) to L3 (abstract insights)."""
        if not self.l2:
            return

        # Collect all error patterns from L2
        patterns = [e for e in self.l2 if e.kind == "pattern"]
        error_summary = {}
        for p in patterns:
            if p.error_type:
                error_summary[p.error_type] = error_summary.get(p.error_type, 0) + 1

        # Create an insight entry
        top_errors = sorted(error_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        insight_text = "Top recurring issues: " + "; ".join(
            f"{err} (x{count})" for err, count in top_errors
        )

        insight = MemoryEntry(
            turn=self.l2[-1].turn,
            timestamp=now_ts(),
            kind="insight",
            summary=insight_text,
            keywords=["insight", "pattern", "root-cause"],
        )
        self.l3.append(insight)

        # Keep only most recent 30% of L2
        keep = max(len(self.l2) // 3, 10)
        self.l2 = self.l2[-keep:]

    def _track_error(self, entry: MemoryEntry):
        """Track error patterns for loop detection."""
        if entry.error_type == self.last_error:
            self.consecutive_same_error += 1
        else:
            self.consecutive_same_error = 1
            self.last_error = entry.error_type
        self.error_history.append((entry.turn, entry.error_type))

    # ─── Persistence ────────────────────────────────────────

    def _save(self):
        path = os.path.join(self.state_dir, "memory_cascade.json")
        data = {
            "turn_counter": self.turn_counter,
            "l0": [e.to_dict() for e in self.l0],
            "l1": [e.to_dict() for e in self.l1],
            "l2": [e.to_dict() for e in self.l2],
            "l3": [e.to_dict() for e in self.l3],
            "last_error": self.last_error,
            "consecutive_same_error": self.consecutive_same_error,
            "error_history": self.error_history,
            "decisions": [e.to_dict() for e in self.decisions],
            "discoveries": [e.to_dict() for e in self.discoveries],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        path = os.path.join(self.state_dir, "memory_cascade.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)

        self.turn_counter = data.get("turn_counter", 0)
        self.l0 = deque(
            [MemoryEntry.from_dict(e) for e in data.get("l0", [])],
            maxlen=50,
        )
        self.l1 = [MemoryEntry.from_dict(e) for e in data.get("l1", [])]
        self.l2 = [MemoryEntry.from_dict(e) for e in data.get("l2", [])]
        self.l3 = [MemoryEntry.from_dict(e) for e in data.get("l3", [])]
        self.last_error = data.get("last_error")
        self.consecutive_same_error = data.get("consecutive_same_error", 0)
        self.error_history = data.get("error_history", [])
        self.decisions = [MemoryEntry.from_dict(e) for e in data.get("decisions", [])]
        self.discoveries = [MemoryEntry.from_dict(e) for e in data.get("discoveries", [])]
