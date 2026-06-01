"""H1 Concept Graph — knowledge representation for creative divergence.

Maintains a graph of project concepts (modules, decisions, discoveries,
external references) and supports random-walk-based creative exploration.

All operations are local Python — zero LLM token overhead.
"""

import os
import json
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field

from shared.utils import now_ts, short_hash, extract_keywords


@dataclass
class Concept:
    """A node in the concept graph."""
    name: str
    kind: str              # module, decision, discovery, external, question
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    importance: float = 0.5    # 0-1, dynamic
    hit_count: int = 0         # times visited by random walk
    created_at: int = field(default_factory=now_ts)
    source: str = "manual"     # manual, search, model_output, injection

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind,
            "keywords": self.keywords, "summary": self.summary,
            "importance": self.importance, "hit_count": self.hit_count,
            "created_at": self.created_at, "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Concept":
        return cls(**d)


@dataclass
class Edge:
    """A relationship between two concepts."""
    source: str
    target: str
    relation: str          # depends_on, conflicts, inspired_by, similar_to, leads_to
    strength: float = 0.5  # 0-1
    created_at: int = field(default_factory=now_ts)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "target": self.target,
            "relation": self.relation, "strength": self.strength,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)


class ConceptGraph:
    """Knowledge graph for creative exploration.

    Nodes = concepts (modules, decisions, discoveries, external refs)
    Edges = relationships with typed relations and strengths
    """

    def __init__(self, state_dir: str, max_nodes: int = 500):
        self.state_dir = state_dir
        self.max_nodes = max_nodes
        self.nodes: dict[str, Concept] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self._load()

    # ─── CRUD ───────────────────────────────────────────────

    def add_concept(self, name: str, kind: str = "module",
                    keywords: list[str] | None = None,
                    summary: str = "", source: str = "manual",
                    importance: float = 0.5) -> Concept:
        """Add or update a concept node."""
        if name in self.nodes:
            c = self.nodes[name]
            if keywords:
                c.keywords = list(set(c.keywords + keywords))
            if summary:
                c.summary = summary
            c.importance = max(c.importance, importance)
            return c

        if keywords is None:
            keywords = extract_keywords(name + " " + summary)

        c = Concept(
            name=name, kind=kind, keywords=keywords,
            summary=summary, importance=importance, source=source,
        )
        self.nodes[name] = c

        # Prune if over limit
        if len(self.nodes) > self.max_nodes:
            self._prune()

        self._save()
        return c

    def add_edge(self, source: str, target: str, relation: str = "related_to",
                 strength: float = 0.5):
        """Add or strengthen an edge between two concepts."""
        key = (source, target)
        if key in self.edges:
            self.edges[key].strength = min(1.0, self.edges[key].strength + 0.1)
        else:
            # Ensure both nodes exist
            for name in [source, target]:
                if name not in self.nodes:
                    self.add_concept(name, kind="unknown")

            self.edges[key] = Edge(
                source=source, target=target,
                relation=relation, strength=strength,
            )
        self._save()

    def get_neighbors(self, name: str) -> list[Concept]:
        """Get all concepts directly connected to this one."""
        neighbors = []
        for (s, t), edge in self.edges.items():
            if s == name and t in self.nodes:
                neighbors.append(self.nodes[t])
            elif t == name and s in self.nodes:
                neighbors.append(self.nodes[s])
        return neighbors

    def get_distant(self, name: str, min_distance: int = 3) -> list[Concept]:
        """Get concepts at least min_distance hops away."""
        # BFS to compute distances
        distances = {name: 0}
        queue = [name]
        while queue:
            current = queue.pop(0)
            for neighbor in self.get_neighbors(current):
                if neighbor.name not in distances:
                    distances[neighbor.name] = distances[current] + 1
                    queue.append(neighbor.name)

        return [
            self.nodes[n] for n, d in distances.items()
            if d >= min_distance and n in self.nodes
        ]

    def semantic_distance(self, a: str, b: str) -> float:
        """Approximate semantic distance using keyword overlap.

        Returns 0 (identical) to 1 (completely different).
        Zero-cost: no embedding API needed.
        """
        node_a = self.nodes.get(a)
        node_b = self.nodes.get(b)
        if not node_a or not node_b:
            return 1.0

        set_a = set(node_a.keywords)
        set_b = set(node_b.keywords)

        if not set_a or not set_b:
            return 1.0

        jaccard = len(set_a & set_b) / len(set_a | set_b)
        return 1.0 - jaccard

    def centrality(self, name: str) -> float:
        """Degree centrality normalized to [0,1]."""
        degree = len(self.get_neighbors(name))
        max_degree = max(
            (len(self.get_neighbors(n)) for n in self.nodes), default=1
        )
        return degree / max(max_degree, 1)

    def max_distance(self) -> float:
        """Maximum semantic distance between any two nodes."""
        return 1.0  # By definition with Jaccard

    def random_node(self) -> str | None:
        """Pick a random node, weighted by importance."""
        if not self.nodes:
            return None
        names = list(self.nodes.keys())
        weights = [self.nodes[n].importance for n in names]
        return random.choices(names, weights=weights, k=1)[0]

    # ─── Bootstrap ──────────────────────────────────────────

    def bootstrap_from_project(self, project_path: str):
        """Scan project directory to create initial module nodes."""
        import os as _os

        root_depth = len(project_path.rstrip("/").split("/"))
        for dirpath, dirnames, _filenames in _os.walk(project_path):
            # Skip hidden and common ignores
            dirnames[:] = [d for d in dirnames
                          if not d.startswith('.')
                          and d not in ('node_modules', '__pycache__',
                                        '.git', 'venv', 'dist', 'build',
                                        'target', '.next', 'coverage')]

            depth = len(dirpath.rstrip("/").split("/")) - root_depth
            if depth > 3:  # Don't go too deep
                continue

            for d in dirnames:
                full = _os.path.join(dirpath, d)
                mod_depth = len(full.rstrip("/").split("/")) - root_depth
                if mod_depth <= 2:
                    self.add_concept(
                        name=d, kind="module",
                        keywords=extract_keywords(d),
                        summary=f"Project module: {d}",
                        source="bootstrap",
                    )

        self._save()

    def stats(self) -> dict:
        """Graph statistics."""
        kinds = defaultdict(int)
        for c in self.nodes.values():
            kinds[c.kind] += 1

        relations = defaultdict(int)
        for e in self.edges.values():
            relations[e.relation] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "by_kind": dict(kinds),
            "by_relation": dict(relations),
            "most_central": sorted(
                self.nodes.keys(),
                key=lambda n: self.centrality(n),
                reverse=True,
            )[:5],
            "most_important": sorted(
                self.nodes.keys(),
                key=lambda n: self.nodes[n].importance,
                reverse=True,
            )[:5],
        }

    # ─── Internal ───────────────────────────────────────────

    def _prune(self):
        """Remove lowest-importance nodes to stay under max_nodes."""
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: (x[1].importance, -x[1].hit_count),
        )
        to_remove = len(self.nodes) - self.max_nodes
        for name, _ in sorted_nodes[:to_remove]:
            del self.nodes[name]
            # Also remove related edges
            self.edges = {
                k: v for k, v in self.edges.items()
                if k[0] != name and k[1] != name
            }

    def _save(self):
        path = os.path.join(self.state_dir, "concept_graph.json")
        data = {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges.values()],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        path = os.path.join(self.state_dir, "concept_graph.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)

        self.nodes = {
            k: Concept.from_dict(v) for k, v in data.get("nodes", {}).items()
        }
        self.edges = {
            (e["source"], e["target"]): Edge.from_dict(e)
            for e in data.get("edges", [])
        }
