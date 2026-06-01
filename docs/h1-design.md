# 搜索驱动发散引擎 — 详细设计

> 2026-06-01 | 与用户共创

---

## 一、架构总览

```
divergence_engine/
├── concept_graph.py      # 概念图数据结构 + 序列化
├── search_provider.py    # 搜索接口（可插拔）
├── walker.py             # 随机游走算法
├── salience.py           # 突显打分
├── injector.py           # 注入生成
├── engine.py             # 主循环
└── state.json            # 图持久化
```

## 二、概念图

### 数据结构

```python
@dataclass
class Concept:
    name: str              # "JWT认证中间件"
    kind: str              # "module"|"decision"|"discovery"|"question"|"external"
    keywords: list[str]    # ["auth", "middleware", "token", "jwt"]
    summary: str           # 一句话描述
    importance: float      # 0-1, 动态更新
    created_at: int        # 创建时间戳（第几轮）
    hit_count: int         # 被随机游走访问次数
    source: str            # "search"|"model_output"|"user"|"injection"

@dataclass  
class Edge:
    source: str            # 概念 A
    target: str            # 概念 B
    relation: str          # "depends_on"|"conflicts"|"inspired_by"|"similar_to"|"leads_to"
    strength: float        # 0-1
    created_at: int

class ConceptGraph:
    nodes: dict[str, Concept]
    edges: dict[tuple[str,str], Edge]
    
    def add_concept(self, ...)
    def add_edge(self, ...)
    def get_neighbors(self, name, depth=1) -> list[Concept]
    def get_distant_concepts(self, name, min_distance=3) -> list[Concept]
    def centrality(self, name) -> float         # 图的中心性
    def semantic_distance(self, a, b) -> float  # 概念间的语义距离
    def prune(self, max_nodes=500)              # 节点上限，LRU 淘汰
```

### 概念来源

| 来源 | 提取方式 | 示例 |
|------|---------|------|
| 项目结构 | 扫描文件树 → 模块节点 | `auth/`, `cart/`, `db.py` |
| 模型决策 | 解析模型输出 → 决策节点 | "选了 Redis 而非 Memcached" |
| Web 搜索 | 搜索结果标题/摘要 → 外部节点 | "JWT best practices 2026" |
| 错误日志 | 错误类型 → 发现节点 | "race condition in cart" |
| 注入反馈 | 注入后被接受的 → 提升 importance | "事件溯源方案可行" |

## 三、搜索提供者

```python
class SearchProvider(ABC):
    """可插拔搜索后端"""
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str

class BraveSearch(SearchProvider): ...
class TavilySearch(SearchProvider): ...
class GoogleSearch(SearchProvider): ...
```

### 搜索触发策略

```python
# 不每轮都搜，而是在以下时机触发：

TRIGGERS = {
    "error_loop": "连续3轮同样的错误",       # 模型在打转
    "low_confidence": "假设验证置信度<0.3",   # 模型不确定
    "new_domain": "进入新模块/新概念区域",    # 探索陌生领域
    "stuck": "连续10轮无进展",               # 卡住了
    "diverge": "随机游走命中了外部概念",      # DMN 触发的联想
    "periodic": "每50轮自动搜索一次",         # 保持新鲜感
}
```

## 四、随机游走算法

### 基础游走

```python
class Walker:
    def walk(self, graph: ConceptGraph, start: str, steps: int = 4,
             strategy: str = "adaptive") -> list[str]:
        """
        从 start 出发，随机游走 steps 步。
        
        每步的选择概率：
        P(下一个节点) = α × edge_strength + β × (1 - semantic_distance) + γ × novelty
        """
        path = [start]
        current = start
        
        for i in range(steps):
            neighbors = graph.get_neighbors(current, depth=1)
            if not neighbors:
                break
            
            # 计算转移概率
            probs = []
            for n in neighbors:
                edge = graph.edges.get((current, n.name), None)
                edge_w = edge.strength if edge else 0.1
                dist_w = 1 - graph.semantic_distance(path[0], n.name)
                novel_w = 1 / (n.hit_count + 1)  # 越少被访问，越新奇
                
                prob = 0.3 * edge_w + 0.3 * dist_w + 0.4 * novel_w
                probs.append(prob)
            
            # 归一化
            probs = [p / sum(probs) for p in probs]
            current = random.choices([n.name for n in neighbors], weights=probs)[0]
            path.append(current)
        
        return path
```

### 游走策略

| 策略 | 步数 | 用途 |
|------|------|------|
| **short_jump** | 1-2 步 | 局部联想，相关概念间的启发 |
| **medium_leap** | 3-5 步 | 跨领域联想，寻求新鲜视角 |
| **long_shot** | 6-10 步 | 疯狂模式，完全出乎意料的连接 |
| **adaptive** | 动态 | 根据当前"卡住程度"自动调整步数 |

### 动态步数调整

```python
def adaptive_steps(self, stuckness: float) -> int:
    """
    stuckness: 0(顺利) → 1(完全卡住)
    越卡，游走越远
    """
    base = 2
    max_extra = 8
    return base + int(stuckness * max_extra)
```

## 五、突显打分

### 三因子模型

```python
class SalienceScorer:
    def score(self, focus: Concept, distant: Concept, graph: ConceptGraph) -> float:
        """
        Salience = 新颖性 × 可行性 × 冲击力
        """
        novelty = self._novelty(focus, distant, graph)
        feasibility = self._feasibility(focus, distant)
        impact = self._impact(focus, distant, graph)
        
        return novelty * feasibility * impact
    
    def _novelty(self, a, b, graph) -> float:
        """
        语义距离 / 最大距离，归一化到 [0,1]
        距离越远 → 越新颖
        """
        raw = graph.semantic_distance(a.name, b.name)
        return min(raw / graph.max_distance(), 1.0)
    
    def _feasibility(self, a, b) -> float:
        """
        基于关键词重叠 + 技术栈兼容性
        完全不搭 → 0, 高度相关 → 1
        """
        overlap = len(set(a.keywords) & set(b.keywords))
        total = len(set(a.keywords) | set(b.keywords))
        return overlap / max(total, 1) if total > 0 else 0.1
    
    def _impact(self, a, b, graph) -> float:
        """
        如果这个联想被实现，影响范围多大？
        = 两个概念的中心性之和 / 2
        """
        return (graph.centrality(a.name) + graph.centrality(b.name)) / 2
```

### 打分阈值

```python
INJECTION_THRESHOLD = 0.35   # 低于此分不注入
SILENT_THRESHOLD = 0.15      # 低于此分不记录
```

## 六、注入器

### 注入模板

```python
class Injector:
    TEMPLATES = [
        "如果把 {a} 的思路用在 {b} 上会怎样？",
        "{a} 和 {b} 有没有可能是一回事？",
        "有没有人试过用 {a} 解决 {b} 的问题？",
        "如果 {a} 和 {b} 结合起来，会不会产生第三种方案？",
        "我注意到 {b}。有没有可能这跟 {a} 有关？",
        "{a} 领域有没有什么技巧可以直接搬到 {b}？",
        "抛开现有假设：如果不用 {b}，用 {a} 的思路重新设计会怎样？",
    ]
    
    def inject(self, focus: Concept, distant: Concept, score: float) -> str:
        template = random.choice(self.TEMPLATES)
        question = template.format(a=distant.name, b=focus.name)
        return f"[💡 发散引擎 | 新奇度: {score:.2f}] {question}"
```

### 注入节制

```python
# 不是每轮都注入，避免干扰正常开发
INJECTION_COOLDOWN = 5       # 至少间隔 5 轮
MAX_INJECTIONS_PER_SESSION = 50  # 单次会话最多 50 次注入
```

## 七、主循环

```python
class DivergenceEngine:
    def __init__(self):
        self.graph = ConceptGraph.load("state.json")
        self.searcher = BraveSearch()
        self.walker = Walker()
        self.scorer = SalienceScorer()
        self.injector = Injector()
        self.round = 0
        self.stuck_counter = 0
        self.last_error = None
    
    def run_cycle(self, 
                  current_focus: str,      # 模型当前在做什么
                  last_action_result: str,  # 上一次操作结果
                  model_context_summary: str # 模型最近在关注什么
                 ) -> Injection | None:
        """
        返回 None 表示本轮不需要注入。
        返回 Injection 表示有一个值得关注的联想。
        """
        self.round += 1
        
        # ① 更新概念图
        self._extract_concepts(current_focus, last_action_result)
        
        # ② 检测是否需要搜索
        if self._should_search(last_action_result):
            results = self.searcher.search(current_focus)
            for r in results:
                self.graph.add_concept(
                    name=r.title,
                    kind="external",
                    keywords=self._extract_keywords(r.snippet),
                    summary=r.snippet[:200],
                    source="search"
                )
        
        # ③ 计算"卡住程度"
        stuckness = self._compute_stuckness(last_action_result)
        self.stuck_counter = (self.stuck_counter + 1) if stuckness > 0.5 else 0
        
        # ④ 随机游走
        focus_concept = self.graph.nodes.get(current_focus) or self._create_concept(current_focus)
        steps = self.walker.adaptive_steps(stuckness)
        path = self.walker.walk(self.graph, focus_concept.name, steps)
        distant_name = path[-1]
        distant_concept = self.graph.nodes[distant_name]
        
        # ⑤ 打分
        score = self.scorer.score(focus_concept, distant_concept, self.graph)
        
        # ⑥ 决定是否注入
        if (score > INJECTION_THRESHOLD and 
            self.round - self.last_injection_round > INJECTION_COOLDOWN):
            injection = self.injector.inject(focus_concept, distant_concept, score)
            self.last_injection_round = self.round
            distant_concept.hit_count += 1
            
            # ⑦ 持久化
            self.graph.save("state.json")
            return injection
        
        return None
    
    def _compute_stuckness(self, result: str) -> float:
        """检测模型是否在打转"""
        signals = 0
        if "error" in result.lower() or "failed" in result.lower():
            signals += 1
        if result == self.last_error:   # 同样的错误
            signals += 1
        # 更多启发式...
        return min(signals / 5, 1.0)
    
    def _should_search(self, result: str) -> bool:
        """判断是否需要触发搜索"""
        return (
            self.stuck_counter >= 3 or
            self.round % 50 == 0 or
            "error" in result.lower()
        )
```

## 八、与 Claude Code 集成

```bash
# hook 方式集成：每次模型工具调用后，引擎检查是否需要注入
# ~/.claude/settings.json

{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "",
        "command": "python3 ~/divergence_engine/engine.py --focus \"$CLAUDE_LAST_TOOL\" --result \"$CLAUDE_TOOL_OUTPUT\""
      }
    ]
  }
}
```

当引擎返回一个 injection 时，它被追加到模型的上下文窗口。模型看到它，就像大脑收到 DMN 的随机联想一样——可以选择采纳，也可以忽略。

## 九、零 Token 承诺

| 操作 | Token | 耗时 |
|------|-------|------|
| 概念图读取 | 0 | < 1ms |
| 随机游走 4 步 | 0 | < 5ms |
| 突显打分 | 0 | < 1ms |
| Web 搜索 (按需，非每轮) | 0 | 200-500ms |
| 注入一句话 | **~20** | 0ms |
| **每轮总成本** | **≤ 20 token** | **< 10ms** |
