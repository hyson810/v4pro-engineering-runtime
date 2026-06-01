"""H1 Injector — generates creative prompts for model injection.

Translates a scored concept connection into a one-sentence "crazy question"
that stimulates divergent thinking, costing the model only ~20 tokens to read.
"""

import random

from h1_creativity.concept_graph import Concept


class Injector:
    """Generates minimal-token creative prompts from concept pairs."""

    TEMPLATES = [
        # Cross-domain inspiration
        "如果把 {distant} 的思路用在 {focus} 上会怎样？",
        "{distant} 领域有没有技巧可以搬到 {focus} 来？",

        # Reframing
        "有没有可能 {focus} 本质上和 {distant} 是同一个问题？",
        "如果不用现有的 {focus} 方案，用 {distant} 的方式重新设计会怎样？",

        # Hybrid innovation
        "把 {focus} 和 {distant} 结合起来，会不会产生第三种更好的方案？",
        "有没有人试过用 {distant} 解决 {focus} 的问题？",

        # Constraint-breaking
        "抛开所有假设：{focus} 如果像 {distant} 一样运作会变成什么样？",
        "如果 {focus} 和 {distant} 是同一样东西，我们会怎么做？",

        # Reversal
        "如果我们反过来看：不是用 {distant} 优化 {focus}，而是让 {focus} 变得像 {distant}？",
        "{distant} 的成功模式能不能直接套在 {focus} 上？",
    ]

    def inject(self, focus: Concept, distant: Concept, score: float) -> str:
        """Generate a creative injection.

        Returns a single sentence designed to spark divergent thinking.
        Costs the model ~20 tokens to process.
        """
        template = random.choice(self.TEMPLATES)
        question = template.format(focus=focus.name, distant=distant.name)

        # Add metadata so the model can gauge relevance
        if score > 0.7:
            prefix = "[💡 高价值联想"
        elif score > 0.5:
            prefix = "[✨ 有趣连接"
        else:
            prefix = "[🔍 边缘联想"

        return f"{prefix} | 新颖度: {score:.2f}] {question}"

    def inject_batch(self, pairs: list[tuple[Concept, Concept, float]],
                     max_results: int = 3) -> list[str]:
        """Generate injections for multiple concept pairs, sorted by score."""
        scored = []
        for focus, distant, score in pairs:
            if score > 0.2:  # Minimum threshold
                injection = self.inject(focus, distant, score)
                scored.append((score, injection))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [inj for _, inj in scored[:max_results]]
