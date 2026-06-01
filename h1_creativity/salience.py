"""H1 Salience Scorer — evaluates creative connections.

Rates the value of a connection between two concepts:
  Salience = Novelty × Feasibility × Impact

All three factors are computed locally from graph structure —
zero token cost.
"""

import math

from h1_creativity.concept_graph import ConceptGraph, Concept


class SalienceScorer:
    """Scores the creative value of connecting two concepts."""

    def score(self, focus: Concept, distant: Concept,
              graph: ConceptGraph) -> float:
        """Compute salience as product of three factors.

        Returns 0 (worthless) to 1 (breakthrough idea).
        """
        novelty = self._novelty(focus, distant, graph)
        feasibility = self._feasibility(focus, distant)
        impact = self._impact(focus, distant, graph)

        return novelty * feasibility * impact

    def _novelty(self, a: Concept, b: Concept, graph: ConceptGraph) -> float:
        """How unexpected is this connection?

        Measured by semantic distance (keyword Jaccard) in [0,1].
        Higher = more surprising = more potential for insight.
        """
        raw = graph.semantic_distance(a.name, b.name)
        # Apply sigmoid-like scaling: push mid-range distances up
        # (connections that are neither obvious nor nonsensical)
        return 1.0 / (1.0 + math.exp(-10 * (raw - 0.5)))

    def _feasibility(self, a: Concept, b: Concept) -> float:
        """Can this connection actually be implemented?

        Based on keyword overlap — more shared vocabulary = more feasible.
        """
        set_a = set(a.keywords)
        set_b = set(b.keywords)
        union = set_a | set_b
        if not union:
            return 0.3  # Unknown domains have medium feasibility

        overlap = len(set_a & set_b)
        # Base from overlap, with a floor of 0.2 so no connection is impossible
        return 0.2 + 0.8 * (overlap / len(union))

    def _impact(self, a: Concept, b: Concept, graph: ConceptGraph) -> float:
        """How much would this change things if implemented?

        Based on the graph centrality of both concepts.
        """
        cent_a = graph.centrality(a.name)
        cent_b = graph.centrality(b.name)
        # Average centrality, with a minimum floor so
        # even peripheral concepts can have impact
        return max(0.1, (cent_a + cent_b) / 2)
