"""Tests for H1 Concept Graph."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h1_creativity.concept_graph import ConceptGraph, Concept, Edge


class TestConcept:
    def test_creation(self):
        c = Concept(name="auth", kind="module", keywords=["jwt", "token"])
        assert c.name == "auth"
        assert c.importance == 0.5
        assert c.hit_count == 0

    def test_serialization(self):
        c = Concept(name="db", kind="module", keywords=["pg", "sql"],
                   summary="Database layer", importance=0.8, source="bootstrap")
        d = c.to_dict()
        c2 = Concept.from_dict(d)
        assert c2.name == c.name
        assert c2.importance == 0.8
        assert c2.source == "bootstrap"


class TestEdge:
    def test_creation(self):
        e = Edge(source="a", target="b", relation="depends_on", strength=0.7)
        assert e.relation == "depends_on"
        assert e.strength == 0.7

    def test_serialization(self):
        e = Edge(source="x", target="y", relation="inspired_by")
        d = e.to_dict()
        e2 = Edge.from_dict(d)
        assert e2.source == e.source


class TestConceptGraph:
    @pytest.fixture
    def graph(self):
        tmp = tempfile.mkdtemp()
        g = ConceptGraph(state_dir=tmp)
        yield g
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

    def test_add_concept(self, graph):
        c = graph.add_concept("auth", kind="module",
                             keywords=["jwt", "token"])
        assert "auth" in graph.nodes
        assert graph.nodes["auth"].kind == "module"

    def test_add_duplicate_concept(self, graph):
        graph.add_concept("auth", keywords=["jwt"])
        graph.add_concept("auth", keywords=["token"], importance=0.9)
        c = graph.nodes["auth"]
        assert "jwt" in c.keywords
        assert "token" in c.keywords
        assert c.importance == 0.9

    def test_add_edge(self, graph):
        graph.add_concept("a")
        graph.add_concept("b")
        graph.add_edge("a", "b", relation="depends_on", strength=0.8)
        assert ("a", "b") in graph.edges
        assert graph.edges[("a", "b")].strength == 0.8

    def test_add_edge_auto_creates_nodes(self, graph):
        graph.add_edge("x", "y", relation="leads_to")
        assert "x" in graph.nodes
        assert "y" in graph.nodes

    def test_add_edge_strengthens_existing(self, graph):
        graph.add_edge("a", "b", strength=0.5)
        graph.add_edge("a", "b", strength=0.5)  # same edge
        assert graph.edges[("a", "b")].strength == 0.6

    def test_get_neighbors(self, graph):
        graph.add_concept("center")
        graph.add_concept("a")
        graph.add_concept("b")
        graph.add_edge("center", "a")
        graph.add_edge("center", "b")
        neighbors = graph.get_neighbors("center")
        assert len(neighbors) == 2
        names = {n.name for n in neighbors}
        assert names == {"a", "b"}

    def test_get_neighbors_empty(self, graph):
        graph.add_concept("lonely")
        assert len(graph.get_neighbors("lonely")) == 0

    def test_semantic_distance_identical(self, graph):
        graph.add_concept("a", keywords=["jwt", "auth", "token"])
        graph.add_concept("b", keywords=["jwt", "auth", "token"])
        d = graph.semantic_distance("a", "b")
        assert d == 0.0

    def test_semantic_distance_different(self, graph):
        graph.add_concept("a", keywords=["jwt", "auth"])
        graph.add_concept("b", keywords=["redis", "cache"])
        d = graph.semantic_distance("a", "b")
        assert d > 0.9

    def test_semantic_distance_missing(self, graph):
        assert graph.semantic_distance("nonexistent", "also_nonexistent") == 1.0

    def test_centrality(self, graph):
        graph.add_concept("center")
        for name in "abcde":
            graph.add_concept(name)
            graph.add_edge("center", name)
        assert graph.centrality("center") == 1.0
        assert graph.centrality("a") == 0.2  # 1/5

    def test_random_node(self, graph):
        graph.add_concept("a", importance=0.1)
        graph.add_concept("b", importance=1.0)
        # Run many times — "b" should be picked more often
        picks = [graph.random_node() for _ in range(100)]
        assert picks.count("b") > picks.count("a")

    def test_random_node_empty(self, graph):
        assert graph.random_node() is None

    def test_stats(self, graph):
        graph.add_concept("a", kind="module")
        graph.add_concept("b", kind="external")
        graph.add_edge("a", "b", relation="inspired_by")
        s = graph.stats()
        assert s["total_nodes"] == 2
        assert s["total_edges"] == 1
        assert s["by_kind"]["module"] == 1

    def test_prune(self, graph):
        graph.max_nodes = 5
        for i in range(10):
            graph.add_concept(f"node_{i}", importance=0.1 * (i + 1))
        assert len(graph.nodes) <= 5
        # Most important nodes should survive
        assert "node_9" in graph.nodes  # highest importance

    def test_persistence(self, graph):
        graph.add_concept("persist", keywords=["test"])
        graph.add_edge("persist", "other")

        g2 = ConceptGraph(state_dir=graph.state_dir)
        assert "persist" in g2.nodes
        assert ("persist", "other") in g2.edges

    def test_empty_graph_stats(self, graph):
        s = graph.stats()
        assert s["total_nodes"] == 0
        assert s["total_edges"] == 0

    def test_bidirectional_neighbor(self, graph):
        graph.add_concept("a")
        graph.add_concept("b")
        graph.add_edge("a", "b")  # a → b
        # Both should see each other
        assert len(graph.get_neighbors("a")) == 1
        assert len(graph.get_neighbors("b")) == 1
