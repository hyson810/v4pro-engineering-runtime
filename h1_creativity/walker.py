"""H1 Walker — random walk over concept graph for creative divergence.

Simulates the DMN (Default Mode Network) by randomly traversing the concept
graph, favoring novel, distant connections. Walk distance adapts to the
current "stuckness" level — more stuck → wander further.
"""

import random

from h1_creativity.concept_graph import ConceptGraph


class Walker:
    """Random walker over the concept graph."""

    def __init__(self, graph: ConceptGraph):
        self.graph = graph

    def walk(self, start: str, steps: int = 4,
             strategy: str = "adaptive") -> list[str]:
        """Walk from start node for `steps` hops.

        Transfer probability at each step:
          P(next) = α·edge_strength + β·(1 - semantic_distance) + γ·novelty

        where novelty = 1 / (hit_count + 1) — favors unexplored nodes.
        """
        if start not in self.graph.nodes:
            return [start]

        path = [start]
        current = start

        for _ in range(steps):
            neighbors = self.graph.get_neighbors(current)

            # If no neighbors, jump to a random node (creative leap)
            if not neighbors:
                random_target = self.graph.random_node()
                if random_target and random_target != current:
                    path.append(random_target)
                    current = random_target
                continue

            # Compute transition probabilities
            probs = []
            for n in neighbors:
                # Edge strength: how strong is the direct connection?
                edge = self.graph.edges.get((current, n.name)) or \
                       self.graph.edges.get((n.name, current))
                edge_w = edge.strength if edge else 0.1

                # Semantic proximity: how related are start and this node?
                dist_w = 1.0 - self.graph.semantic_distance(path[0], n.name)

                # Novelty: prefer nodes we haven't visited much
                novel_w = 1.0 / (n.hit_count + 1)

                # Adaptive weights
                prob = 0.2 * edge_w + 0.3 * dist_w + 0.5 * novel_w
                probs.append(prob)

            # Normalize
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            # Select next node
            current = random.choices(
                [n.name for n in neighbors], weights=probs, k=1
            )[0]
            path.append(current)

        return path

    def adaptive_steps(self, stuckness: float) -> int:
        """More stuck → more steps → more divergent thinking.

        stuckness: 0 (everything's fine) to 1 (completely stuck)
        """
        return 2 + int(stuckness * 8)  # 2 to 10 steps

    def creative_leap(self, start: str) -> str | None:
        """A long random jump — force a completely unexpected connection.

        Walks 6-10 steps to land on something far from the start.
        """
        steps = random.randint(6, 10)
        path = self.walk(start, steps=steps)
        return path[-1] if path else None
