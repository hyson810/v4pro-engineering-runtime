"""H1 Creativity MCP Server — creative divergence engine.

Tools:
  h1_inject   — Get a creative cross-domain connection for current focus
  h1_search   — Search web + update concept graph
  h1_graph    — View concept graph status
  h1_connect  — Add a connection between concepts
  h1_discover — Add a discovery to the graph
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from shared.config import RuntimeConfig
from h1_creativity.concept_graph import ConceptGraph
from h1_creativity.walker import Walker
from h1_creativity.salience import SalienceScorer
from h1_creativity.injector import Injector

mcp = FastMCP("h1-creativity")

config = RuntimeConfig.from_env()
graph = ConceptGraph(state_dir=config.state_dir)
walker = Walker(graph)
scorer = SalienceScorer()
injector = Injector()

# Bootstrap on first run
if graph.stats()["total_nodes"] == 0:
    graph.bootstrap_from_project(config.project_path)

# Cooldown tracking
_last_injection = 0
_turn = 0


@mcp.tool()
def h1_inject(focus: str) -> str:
    """Get a creative cross-domain connection for a problem or module.

    Performs a random walk through the project knowledge graph and generates
    a thought-provoking question connecting distant concepts. Simulates the
    brain's DMN (Default Mode Network) spontaneous creative associations.

    focus: what you're working on (module name, problem, concept)
    """
    global _last_injection, _turn
    _turn += 1

    if focus not in graph.nodes:
        graph.add_concept(name=focus, kind="module",
                         summary=f"Current focus: {focus}", source="model_output")

    focus_concept = graph.nodes[focus]
    path = walker.walk(focus, steps=config.walk_default_steps)
    distant_name = path[-1] if len(path) > 1 else graph.random_node()

    if not distant_name or distant_name not in graph.nodes:
        return "Concept graph needs more data. Continue working and it will accumulate connections."

    distant_concept = graph.nodes[distant_name]
    score = scorer.score(focus_concept, distant_concept, graph)
    distant_concept.hit_count += 1
    _last_injection = _turn

    question = injector.inject(focus_concept, distant_concept, score)
    path_str = " → ".join(path)
    return f"{question}\n\n*Path: {path_str}*"


@mcp.tool()
def h1_search(query: str) -> str:
    """Search for external references and add to concept graph.

    Like searching StackOverflow or docs to spark ideas.
    query: what to search for
    """
    results = []

    # Try DuckDuckGo (free, no API key)
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

    if not results:
        return (
            "Search not available. Install duckduckgo-search:\n"
            "  pip install duckduckgo-search"
        )

    added = 0
    for r in results:
        name = r["title"][:80]
        if name not in graph.nodes:
            graph.add_concept(name=name, kind="external",
                            summary=r["snippet"], source="search")
            added += 1

    lines = [f"Found {len(results)} results, added {added} concepts:\n"]
    for r in results[:5]:
        lines.append(f"- **{r['title'][:60]}**")
        lines.append(f"  {r['snippet'][:100]}...")
    return "\n".join(lines)


@mcp.tool()
def h1_graph() -> str:
    """View concept graph statistics."""
    s = graph.stats()
    return (
        f"**H1 Concept Graph**\n\n"
        f"Nodes: {s['total_nodes']} | Edges: {s['total_edges']}\n"
        f"By kind: {s['by_kind']}\n"
        f"Most central: {', '.join(s['most_central'][:5])}\n"
        f"Most important: {', '.join(s['most_important'][:5])}"
    )


@mcp.tool()
def h1_connect(a: str, b: str, relation: str = "related_to") -> str:
    """Manually connect two concepts with a relationship.

    a, b: concepts to connect
    relation: depends_on|conflicts|inspired_by|similar_to|leads_to
    """
    for name in [a, b]:
        if name not in graph.nodes:
            graph.add_concept(name, kind="module", source="manual")
    graph.add_edge(a, b, relation=relation, strength=0.7)
    return f"Connected **{a}** -> **{b}** ({relation})"


@mcp.tool()
def h1_discover(insight: str) -> str:
    """Record a discovery to enrich the concept graph."""
    graph.add_concept(name=insight[:80], kind="discovery",
                     summary=insight, source="model_output", importance=0.8)
    return f"Discovery added: {insight[:60]}"


if __name__ == "__main__":
    mcp.run()
