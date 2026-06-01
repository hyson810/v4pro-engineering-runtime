"""Tests for H1 Walker and Salience scorer."""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h1_creativity.concept_graph import ConceptGraph
from h1_creativity.walker import Walker
from h1_creativity.salience import SalienceScorer
from h1_creativity.injector import Injector


class TestWalker:
    @pytest.fixture
    def graph(self):
        tmp = tempfile.mkdtemp()
        g = ConceptGraph(state_dir=tmp)
        # Build a test graph
        nodes = ["auth", "database", "cache", "api", "frontend", "queue", "search"]
        for n in nodes:
            g.add_concept(n, keywords=[n])
        g.add_edge("auth", "database", strength=0.8)
        g.add_edge("database", "cache", strength=0.7)
        g.add_edge("cache", "api", strength=0.6)
        g.add_edge("api", "frontend", strength=0.9)
        g.add_edge("auth", "api", strength=0.5)
        g.add_edge("database", "queue", strength=0.4)
        g.add_edge("queue", "search", strength=0.3)
        yield g
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

    @pytest.fixture
    def walker(self, graph):
        return Walker(graph)

    def test_walk_returns_path(self, walker):
        path = walker.walk("auth", steps=3)
        assert len(path) == 4  # start + 3 steps
        assert path[0] == "auth"

    def test_walk_single_step(self, walker):
        path = walker.walk("auth", steps=1)
        assert len(path) == 2

    def test_walk_unknown_start(self, walker):
        path = walker.walk("nonexistent", steps=3)
        assert path == ["nonexistent"]

    def test_adaptive_steps_low(self, walker):
        steps = walker.adaptive_steps(0.0)
        assert steps == 2

    def test_adaptive_steps_high(self, walker):
        steps = walker.adaptive_steps(1.0)
        assert steps == 10

    def test_creative_leap(self, walker):
        target = walker.creative_leap("auth")
        # target can be None or a string
        if target:
            assert target in walker.graph.nodes


class TestSalience:
    @pytest.fixture
    def graph(self):
        tmp = tempfile.mkdtemp()
        g = ConceptGraph(state_dir=tmp)
        g.add_concept("JWT", keywords=["jwt", "auth", "token"])
        g.add_concept("Redis", keywords=["redis", "cache", "session"])
        g.add_concept("Kafka", keywords=["kafka", "mq", "events"])
        g.add_concept("OAuth", keywords=["oauth", "auth", "sso"])
        g.add_edge("JWT", "Redis")
        g.add_edge("Redis", "Kafka")
        yield g
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)

    @pytest.fixture
    def scorer(self):
        return SalienceScorer()

    def test_score_range(self, scorer, graph):
        jwt = graph.nodes["JWT"]
        redis = graph.nodes["Redis"]
        s = scorer.score(jwt, redis, graph)
        assert 0.0 <= s <= 1.0

    def test_similar_concepts_score_higher(self, scorer, graph):
        jwt = graph.nodes["JWT"]
        oauth = graph.nodes["OAuth"]  # both auth-related

        jwt_kafka = graph.nodes["Kafka"]  # completely different

        score_similar = scorer.score(jwt, oauth, graph)
        score_different = scorer.score(jwt, jwt_kafka, graph)

        # Similar concepts should have higher feasibility
        # (not necessarily higher total score — novelty is lower)
        f_similar = scorer._feasibility(jwt, oauth)
        f_different = scorer._feasibility(jwt, jwt_kafka)
        assert f_similar > f_different

    def test_novelty_is_sigmoid(self, scorer, graph):
        jwt = graph.nodes["JWT"]
        redis = graph.nodes["Redis"]
        n = scorer._novelty(jwt, redis, graph)
        assert 0.0 <= n <= 1.0

    def test_impact_minimum(self, scorer, graph):
        jwt = graph.nodes["JWT"]
        redis = graph.nodes["Redis"]
        i = scorer._impact(jwt, redis, graph)
        assert i >= 0.1


class TestInjector:
    @pytest.fixture
    def injector(self):
        return Injector()

    def test_inject_returns_styled_string(self, injector):
        from h1_creativity.concept_graph import Concept
        focus = Concept(name="JWT认证", kind="module", keywords=["jwt"])
        distant = Concept(name="事件溯源", kind="external", keywords=["events"])
        result = injector.inject(focus, distant, 0.75)
        assert "JWT" in result or "事件溯源" in result
        assert "0.75" in result

    def test_inject_high_score(self, injector):
        from h1_creativity.concept_graph import Concept
        focus = Concept(name="auth", kind="module", keywords=["auth"])
        distant = Concept(name="graphql", kind="external", keywords=["api"])
        result = injector.inject(focus, distant, 0.85)
        assert "高价值" in result or "0.85" in result

    def test_inject_low_score(self, injector):
        from h1_creativity.concept_graph import Concept
        focus = Concept(name="auth", kind="module")
        distant = Concept(name="random", kind="external")
        result = injector.inject(focus, distant, 0.25)
        assert "边缘" in result or "0.25" in result

    def test_inject_batch(self, injector):
        from h1_creativity.concept_graph import Concept
        a = Concept(name="A", kind="module")
        b = Concept(name="B", kind="module")
        c = Concept(name="C", kind="module")
        results = injector.inject_batch([
            (a, b, 0.3),
            (a, c, 0.8),
            (b, c, 0.5),
        ])
        assert len(results) <= 3
