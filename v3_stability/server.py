"""V3 Stability MCP Server — long-running engineering session memory.

Tools:
  v3_checkpoint  — Record current phase state
  v3_reflect     — Force structured reflection
  v3_recall      — Search memory across all tiers
  v3_status      — View memory system health
  v3_decision    — Record a key decision
  v3_discovery   — Record a discovery
  v3_error       — Record an error with pattern detection
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from shared.config import RuntimeConfig
from v3_stability.memory_cascade import MemoryCascade

mcp = FastMCP("v3-stability")

config = RuntimeConfig.from_env()
memory = MemoryCascade(state_dir=config.state_dir, config=config)


@mcp.tool()
def v3_checkpoint(phase: str, summary: str, detail: str = "") -> str:
    """Record a checkpoint — current phase and what was accomplished.

    Call after completing a meaningful unit of work.
    phase: explore/plan/implement/debug/test/refactor/review/done/stuck
    summary: one-sentence summary
    detail: optional longer context (code snippets, errors, etc.)
    """
    memory.record(
        kind="checkpoint",
        summary=f"[{phase}] {summary}",
        detail=detail,
    )
    status = memory.status()
    return (
        f"Checkpoint recorded (turn {status['turn']}). "
        f"Memory tiers: L0={status['tiers']['L0_current']} "
        f"L1={status['tiers']['L1_recent']} "
        f"L2={status['tiers']['L2_midterm']} "
        f"L3={status['tiers']['L3_longterm']}"
    )


@mcp.tool()
def v3_reflect() -> str:
    """Generate structured reflection on recent activity.

    Call when stuck, repeating similar attempts, completing a major phase,
    or about to make a significant architectural decision.
    """
    return memory.reflect()


@mcp.tool()
def v3_recall(query: str, max_results: int = 5) -> str:
    """Search memory across all tiers for relevant entries.

    query: keywords or phrase
    max_results: max entries (default 5)
    """
    results = memory.recall(query, max_results=max_results)
    if not results:
        return f"No results for '{query}' across {memory.status()['total_entries']} entries."

    lines = [f"Found {len(results)} results for '{query}':"]
    for i, entry in enumerate(results):
        lines.append(f"{i+1}. [T{entry.turn}] [{entry.kind}] {entry.summary}")
        if entry.error_type:
            lines.append(f"   error pattern: {entry.error_type}")
    return "\n".join(lines)


@mcp.tool()
def v3_status() -> str:
    """View current state of the memory system."""
    s = memory.status()
    return (
        f"**V3 Memory Cascade** (turn {s['turn']})\n\n"
        f"| Tier | Entries |\n|------|--------|\n"
        f"| L0 Current | {s['tiers']['L0_current']} |\n"
        f"| L1 Recent  | {s['tiers']['L1_recent']} |\n"
        f"| L2 Midterm | {s['tiers']['L2_midterm']} |\n"
        f"| L3 Longterm| {s['tiers']['L3_longterm']} |\n\n"
        f"Decisions: {s['total_decisions']} | Discoveries: {s['total_discoveries']} | "
        f"Same-error streak: {s['consecutive_same_error']}"
    )


@mcp.tool()
def v3_decision(choice: str, reason: str, alternatives: str = "") -> str:
    """Record a key architectural or strategic decision.

    choice: what was chosen
    reason: why
    alternatives: other options considered (optional)
    """
    detail = f"Choice: {choice}\nReason: {reason}"
    if alternatives:
        detail += f"\nAlternatives: {alternatives}"
    memory.record(kind="decision", summary=f"Decided: {choice}", detail=detail)
    return f"Decision recorded: {choice}"


@mcp.tool()
def v3_discovery(insight: str, evidence: str = "") -> str:
    """Record a discovery about the codebase or problem.

    insight: what was discovered
    evidence: what led to it (optional)
    """
    detail = f"Discovery: {insight}"
    if evidence:
        detail += f"\nEvidence: {evidence}"
    memory.record(kind="discovery", summary=insight[:100], detail=detail)
    return f"Discovery recorded: {insight[:80]}"


@mcp.tool()
def v3_error(error_message: str, context: str = "") -> str:
    """Record an error with automatic pattern detection.

    error_message: the error text
    context: what was being attempted (optional)
    """
    from shared.utils import detect_error_pattern
    error_type = detect_error_pattern(error_message)
    memory.record(
        kind="error",
        summary=error_message[:120],
        detail=f"Error: {error_message}\nContext: {context}" if context else error_message,
        error_type=error_type,
    )
    msg = f"Error recorded: {error_message[:80]}"
    if error_type:
        msg += f" [pattern: {error_type}]"
    if memory.consecutive_same_error >= 3:
        msg += f"\nThis error type has repeated {memory.consecutive_same_error} times. Consider v3_reflect()."
    return msg


if __name__ == "__main__":
    mcp.run()
