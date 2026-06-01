"""V3 Stability MCP Server — injects reflection and manages memory for
long-running engineering sessions with DeepSeek V4 Pro.

Tools exposed:
  - v3_checkpoint: Record current phase state
  - v3_reflect: Force structured reflection
  - v3_recall: Search memory across all tiers
  - v3_status: View memory system health
  - v3_decision: Record a key decision
  - v3_error: Record an error with pattern detection
"""

import sys
import os

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from shared.config import RuntimeConfig
from v3_stability.memory_cascade import MemoryCascade

# ─── Init ───────────────────────────────────────────────────

config = RuntimeConfig.from_env()
memory = MemoryCascade(state_dir=config.state_dir, config=config)

server = Server("v3-stability")

# ─── Helpers ────────────────────────────────────────────────

def _phase_emoji(phase: str) -> str:
    emojis = {
        "explore": "🔍", "plan": "📋", "implement": "⚡",
        "debug": "🐛", "test": "🧪", "refactor": "🔧",
        "review": "👀", "done": "✅", "stuck": "🆘",
    }
    return emojis.get(phase.lower(), "📍")


# ─── Tools ──────────────────────────────────────────────────

@server.tool()
async def v3_checkpoint(phase: str, summary: str, detail: str = "") -> str:
    """Record a checkpoint — current phase and what was accomplished.

    Call this after completing a meaningful unit of work.
    phase: one of explore/plan/implement/debug/test/refactor/review/done/stuck
    summary: one-sentence summary of what was done
    detail: optional longer context (code snippets, error messages, etc.)
    """
    memory.record(
        kind="checkpoint",
        summary=f"[{phase}] {summary}",
        detail=detail,
    )

    status = memory.status()
    return (
        f"{_phase_emoji(phase)} Checkpoint recorded (turn {status['turn']}).\n"
        f"Memory: L0={status['tiers']['L0_current']} "
        f"L1={status['tiers']['L1_recent']} "
        f"L2={status['tiers']['L2_midterm']} "
        f"L3={status['tiers']['L3_longterm']}"
    )


@server.tool()
async def v3_reflect() -> str:
    """Generate a structured reflection on recent activity.

    Call this when:
    - You feel stuck or are repeating similar attempts
    - You've completed a major phase and want to consolidate learnings
    - You're about to make a significant architectural decision

    Returns a reflection summary that helps re-orient and avoid repeated mistakes.
    """
    reflection = memory.reflect()
    return reflection


@server.tool()
async def v3_recall(query: str, max_results: int = 5) -> str:
    """Search memory across all tiers for relevant past entries.

    query: keywords or phrase to search for
    max_results: max entries to return (default 5)
    """
    results = memory.recall(query, max_results=max_results)

    if not results:
        return f"No results found for '{query}' across {memory.status()['total_entries']} entries."

    lines = [f"Found {len(results)} results for '{query}':"]
    for i, entry in enumerate(results):
        lines.append(f"{i+1}. [T{entry.turn}] [{entry.kind}] {entry.summary}")
        if entry.error_type:
            lines.append(f"   ⚠️ error: {entry.error_type}")

    return "\n".join(lines)


@server.tool()
async def v3_status() -> str:
    """View the current state of the memory system.

    Shows turn count, tier sizes, error history, and key metrics.
    """
    status = memory.status()

    return (
        f"**V3 Memory Cascade Status** (turn {status['turn']})\n\n"
        f"| Tier | Entries |\n"
        f"|------|--------|\n"
        f"| L0 Current | {status['tiers']['L0_current']} |\n"
        f"| L1 Recent  | {status['tiers']['L1_recent']} |\n"
        f"| L2 Midterm | {status['tiers']['L2_midterm']} |\n"
        f"| L3 Longterm| {status['tiers']['L3_longterm']} |\n\n"
        f"**Metrics:**\n"
        f"- Total entries: {status['total_entries']}\n"
        f"- Key decisions: {status['total_decisions']}\n"
        f"- Discoveries: {status['total_discoveries']}\n"
        f"- Consecutive same error: {status['consecutive_same_error']}\n"
        f"- Last error: {status['last_error'] or 'none'}\n"
        f"- Error history: {status['error_history_len']} events"
    )


@server.tool()
async def v3_decision(choice: str, reason: str, alternatives: str = "") -> str:
    """Record a key architectural or strategic decision.

    choice: what was chosen
    reason: why this option was selected
    alternatives: what other options were considered (optional)
    """
    detail = f"Choice: {choice}\nReason: {reason}"
    if alternatives:
        detail += f"\nAlternatives considered: {alternatives}"

    memory.record(
        kind="decision",
        summary=f"Decided: {choice} — {reason[:80]}",
        detail=detail,
    )

    return f"📌 Decision recorded: {choice}"


@server.tool()
async def v3_discovery(insight: str, evidence: str = "") -> str:
    """Record a discovery — something learned about the codebase or problem.

    insight: what was discovered
    evidence: what led to this discovery (optional)
    """
    detail = f"Discovery: {insight}"
    if evidence:
        detail += f"\nEvidence: {evidence}"

    memory.record(
        kind="discovery",
        summary=insight[:100],
        detail=detail,
    )

    return f"💡 Discovery recorded: {insight[:80]}"


@server.tool()
async def v3_error(error_message: str, context: str = "") -> str:
    """Record an error with automatic pattern detection.

    error_message: the error text
    context: what was being attempted (optional)

    The engine automatically detects if this is part of a repeated pattern
    and will flag it in future reflections.
    """
    error_type = None
    from shared.utils import detect_error_pattern
    error_type = detect_error_pattern(error_message)

    memory.record(
        kind="error",
        summary=error_message[:120],
        detail=f"Error: {error_message}\nContext: {context}" if context else error_message,
        error_type=error_type,
    )

    msg = f"🐛 Error recorded: {error_message[:80]}"
    if error_type:
        msg += f"\nDetected pattern: **{error_type}**"
    if memory.consecutive_same_error >= 3:
        msg += f"\n⚠️ This error type has repeated {memory.consecutive_same_error} times. Consider calling v3_reflect()."

    return msg


# ─── Entry Point ────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
