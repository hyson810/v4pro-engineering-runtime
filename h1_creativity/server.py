"""H1 Creativity MCP Server — injects creative prompts for DeepSeek V4 Pro.

Tools exposed:
  - h1_inject: Get a creative divergence injection for current focus
  - h1_search: Search web + update concept graph
  - h1_graph: View concept graph status
  - h1_connect: Add a connection between concepts
  - h1_discover: Add a new discovery to the graph
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server

from shared.config import RuntimeConfig
from h1_creativity.concept_graph import ConceptGraph
from h1_creativity.walker import Walker
from h1_creativity.salience import SalienceScorer
from h1_creativity.injector import Injector

# ─── Init ───────────────────────────────────────────────────

config = RuntimeConfig.from_env()
graph = ConceptGraph(state_dir=config.state_dir)
walker = Walker(graph)
scorer = SalienceScorer()
injector = Injector()

# Bootstrap on first run
if graph.stats()["total_nodes"] == 0:
    graph.bootstrap_from_project(config.project_path)

server = Server("h1-creativity")

# Track injection cooldown
_last_injection_turn = 0
_turn_counter = 0


def _should_inject() -> bool:
    """Enforce injection cooldown."""
    global _turn_counter, _last_injection_turn
    _turn_counter += 1
    if _turn_counter - _last_injection_turn < config.injection_cooldown_turns:
        return False
    return True


# ─── Tools ──────────────────────────────────────────────────

@server.tool()
async def h1_inject(focus: str) -> str:
    """Get a creative divergence injection for a problem or module.

    The engine performs a random walk from your focus concept through
    the project knowledge graph, landing on a distant concept, and
    generates a thought-provoking question connecting the two.

    This simulates the brain's DMN (Default Mode Network) —
    spontaneous, unexpected associations that spark insight.

    focus: what you're currently working on (module name, problem, concept)
    """
    global _last_injection_turn, _turn_counter
    _turn_counter += 1

    # Ensure focus exists in graph
    if focus not in graph.nodes:
        graph.add_concept(
            name=focus, kind="module",
            summary=f"Current focus: {focus}",
            source="model_output",
        )

    focus_concept = graph.nodes[focus]

    # Walk and score
    path = walker.walk(focus, steps=config.walk_default_steps)
    distant_name = path[-1] if len(path) > 1 else graph.random_node()

    if not distant_name or distant_name not in graph.nodes:
        return "🎯 概念图还不够丰富，继续工作我会积累更多连接。试试 h1_search() 搜索一些外部参考。"

    distant_concept = graph.nodes[distant_name]
    score = scorer.score(focus_concept, distant_concept, graph)

    # Track hits
    distant_concept.hit_count += 1
    _last_injection_turn = _turn_counter

    # Generate injection
    question = injector.inject(focus_concept, distant_concept, score)

    # Show the walk path for transparency
    path_str = " → ".join(path)

    return (
        f"{question}\n\n"
        f"*Walk path: {path_str}*\n"
        f"*Score: novelty={scorer._novelty(focus_concept, distant_concept, graph):.2f} "
        f"feasibility={scorer._feasibility(focus_concept, distant_concept):.2f} "
        f"impact={scorer._impact(focus_concept, distant_concept, graph):.2f}*"
    )


@server.tool()
async def h1_search(query: str) -> str:
    """Search for external references and add them to the concept graph.

    This is the engine's way of getting "fresh stimuli" — just like
    a developer searching StackOverflow or reading docs to spark ideas.

    query: what to search for (e.g., "JWT authentication best practices")
    """
    results = []

    # Try DuckDuckGo (free, no API key needed)
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=config.search_max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")[:200],
                    "url": r.get("href", ""),
                })
    except ImportError:
        pass

    # Fallback to brave if available
    if not results:
        try:
            import requests
            api_key = os.environ.get("BRAVE_API_KEY", "")
            if api_key:
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": config.search_max_results},
                    headers={"X-Subscription-Token": api_key},
                    timeout=10,
                )
                for r in resp.json().get("web", {}).get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("description", "")[:200],
                        "url": r.get("url", ""),
                    })
        except Exception:
            pass

    if not results:
        return (
            "🔍 Search providers not available.\n\n"
            "To enable search, install one of:\n"
            "- `pip install duckduckgo-search` (free, no API key)\n"
            "- Set BRAVE_API_KEY env var for Brave Search"
        )

    # Add results to graph
    added = 0
    for r in results:
        name = r["title"][:80]
        if name not in graph.nodes:
            graph.add_concept(
                name=name, kind="external",
                summary=r["snippet"],
                source="search",
            )
            added += 1

        # Connect to project concepts with similar keywords
        graph.add_edge(name, query.split()[0] if query else "project",
                       relation="inspired_by", strength=0.3)

    lines = [f"🔍 Found {len(results)} results, added {added} new concepts:\n"]
    for r in results[:5]:
        lines.append(f"- **{r['title'][:60]}**")
        lines.append(f"  {r['snippet'][:100]}...")

    return "\n".join(lines)


@server.tool()
async def h1_graph() -> str:
    """View concept graph statistics — size, most central concepts, etc."""
    stats = graph.stats()

    return (
        f"**H1 Concept Graph**\n\n"
        f"| Metric | Value |\n"
        f"|--------|------|\n"
        f"| Nodes | {stats['total_nodes']} |\n"
        f"| Edges | {stats['total_edges']} |\n\n"
        f"**By kind:** {stats['by_kind']}\n\n"
        f"**By relation:** {stats['by_relation']}\n\n"
        f"**Most central:** {', '.join(stats['most_central'][:5])}\n\n"
        f"**Most important:** {', '.join(stats['most_important'][:5])}"
    )


@server.tool()
async def h1_connect(a: str, b: str, relation: str = "related_to") -> str:
    """Manually connect two concepts with a relationship.

    Use this when you discover a meaningful connection during development.
    This enriches the graph for future creative walks.

    a, b: the two concepts to connect
    relation: depends_on | conflicts | inspired_by | similar_to | leads_to
    """
    for name in [a, b]:
        if name not in graph.nodes:
            graph.add_concept(name, kind="module", source="manual")

    graph.add_edge(a, b, relation=relation, strength=0.7)

    return f"🔗 Connected **{a}** → **{b}** ({relation})"


@server.tool()
async def h1_discover(insight: str) -> str:
    """Record a discovery to enrich the concept graph.

    insight: what you discovered or realized
    """
    graph.add_concept(
        name=insight[:80], kind="discovery",
        summary=insight, source="model_output",
        importance=0.8,  # Discoveries are important
    )

    return f"💡 Discovery added to graph: {insight[:60]}"


# ─── Entry Point ────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
