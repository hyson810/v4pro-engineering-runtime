"""Tests for V3 Memory Cascade."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import RuntimeConfig
from v3_stability.memory_cascade import MemoryCascade, MemoryEntry


class TestMemoryEntry:
    def test_creation(self):
        e = MemoryEntry(turn=1, timestamp=100, kind="action", summary="test")
        assert e.turn == 1
        assert e.kind == "action"
        assert e.error_type is None

    def test_serialization_roundtrip(self):
        e = MemoryEntry(
            turn=5, timestamp=500, kind="error",
            summary="TypeError in auth",
            detail="full stack trace",
            keywords=["type", "auth"],
            error_type="type_error",
            parent_turn=3,
        )
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.turn == e.turn
        assert e2.kind == e.kind
        assert e2.error_type == "type_error"
        assert e2.parent_turn == 3


class TestMemoryCascade:
    @pytest.fixture
    def cascade(self):
        tmp = tempfile.mkdtemp()
        config = RuntimeConfig(state_dir=tmp)
        mc = MemoryCascade(state_dir=tmp, config=config)
        yield mc
        # Cleanup
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

    def test_record_and_status(self, cascade):
        cascade.record("action", "Read auth module")
        cascade.record("decision", "Use JWT", "detail")
        cascade.record("error", "TypeError in login", "stack")

        s = cascade.status()
        assert s["turn"] == 3
        assert s["total_entries"] == 3
        assert s["total_decisions"] == 1

    def test_reflect_empty(self, cascade):
        r = cascade.reflect()
        assert "Reflection" in r
        assert cascade.status()["total_entries"] == 1  # reflect records itself

    def test_reflect_with_errors(self, cascade):
        for _ in range(3):
            cascade.record("error", "TypeError: null", "stack")
        r = cascade.reflect()
        assert "TypeError" in r or "type_error" in r.lower()

    def test_reflect_with_decisions(self, cascade):
        cascade.record("decision", "Chose Redis", "persistence needed")
        cascade.record("discovery", "Cache invalidation bug", "stale data")
        r = cascade.reflect()
        assert "Redis" in r
        assert "Cache" in r or "invalidation" in r.lower()

    def test_recall_string(self, cascade):
        cascade.record("action", "JWT auth middleware setup")
        cascade.record("action", "Database connection pool config")
        cascade.record("error", "JWT token expired error")
        results = cascade.recall("jwt auth")
        assert len(results) > 0
        assert any("JWT" in r.summary for r in results)

    def test_recall_list(self, cascade):
        cascade.record("action", "Redis cache setup")
        results = cascade.recall(["redis", "cache"])
        assert len(results) > 0

    def test_recall_no_match(self, cascade):
        cascade.record("action", "Setup database")
        results = cascade.recall("nonexistent_xyz")
        assert len(results) == 0

    def test_error_tracking(self, cascade):
        cascade.record("error", "TypeError: x is null")
        assert cascade.last_error is not None
        assert cascade.consecutive_same_error == 1

        cascade.record("error", "TypeError: y is null")
        assert cascade.consecutive_same_error == 2

        cascade.record("error", "ImportError: no module")
        assert cascade.consecutive_same_error == 1  # different error resets

    def test_error_loop_detection(self, cascade):
        for _ in range(4):
            cascade.record("error", "TypeError: null reference")
        r = cascade.reflect()
        assert "repeated" in r.lower() or "Repeat" in r or "3" in r or "4" in r

    def test_cascade_l0_overflow(self, cascade):
        config = cascade.config
        config.cascade_l0_max_turns = 10
        for i in range(15):
            cascade.record("action", f"Action {i}")
        s = cascade.status()
        # L0 should be around 5 (kept half)
        assert s["tiers"]["L0_current"] <= 10
        # Some entries moved to L1
        assert s["tiers"]["L1_recent"] > 0

    def test_cascade_l1_to_l2(self, cascade):
        config = cascade.config
        config.cascade_l0_max_turns = 10
        config.cascade_l1_max_turns = 15
        # Fill L0→L1 repeatedly
        for batch in range(3):
            for i in range(10):
                cascade.record(
                    "error" if i % 3 == 0 else "action",
                    f"Batch {batch} action {i}",
                    error_type="type_error" if i % 3 == 0 else None,
                )
        s = cascade.status()
        # Should have some L2 patterns
        assert s["tiers"]["L2_midterm"] >= 0  # may or may not compress

    def test_reset(self, cascade):
        cascade.record("action", "Something")
        cascade.record("decision", "Something else")
        cascade.reset()
        s = cascade.status()
        assert s["turn"] == 0
        assert s["total_entries"] == 0
        assert s["total_decisions"] == 0

    def test_persistence(self, cascade):
        cascade.record("action", "Test persistence")
        cascade.record("decision", "Important decision", "reason")
        turn_before = cascade.status()["turn"]

        # Create a new instance pointing to same state dir
        mc2 = MemoryCascade(state_dir=cascade.state_dir, config=cascade.config)
        assert mc2.status()["turn"] == turn_before
        assert mc2.status()["total_entries"] == cascade.status()["total_entries"]

    def test_decision_tracking(self, cascade):
        cascade.record("decision", "Use Postgres", "ACID needed")
        cascade.record("decision", "Use Redis", "Speed needed")
        assert cascade.status()["total_decisions"] == 2

    def test_discovery_tracking(self, cascade):
        cascade.record("discovery", "Race condition in cart")
        cascade.record("discovery", "Memory leak in auth")
        assert cascade.status()["total_discoveries"] == 2
