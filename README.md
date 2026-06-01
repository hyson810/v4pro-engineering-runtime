# V4Pro Engineering Runtime

> 让 DeepSeek V4 Pro 跑得更久、产出更惊艳

## 这是什么？

DeepSeek V4 Pro 拥有 1M 上下文和 Codeforces 3206 的代码生成能力，但缺乏 GLM 5.1 级别的长程自主工程稳定性。

本项目通过两个 MCP Server 在推理层补足这个差距：

- **V3 长程工程引擎** — 防止模型在长时间任务中"迷失方向"
- **H1 发散引擎** — 激发创造力，让产品超出预期

两个引擎**不修改模型权重**，纯推理层增强，几乎不消耗额外 token。

## 快速开始

### 1. 安装

```bash
git clone https://github.com/YOUR_USER/v4pro-engineering-runtime.git
cd v4pro-engineering-runtime
pip install -e .
```

可选：安装搜索支持
```bash
pip install duckduckgo-search
```

### 2. 配置 Claude Code

将 `.mcp.json.example` 复制到你的项目目录，重命名为 `.mcp.json`：

```bash
cp .mcp.json.example /your/project/.mcp.json
```

编辑 `cwd` 指向本项目的安装路径。

### 3. 使用

引擎会随 Claude Code 自动启动。在对话中：

```
# V3 稳定性
v3_checkpoint(phase="implement", summary="完成 JWT 认证中间件")
v3_reflect()     # 卡住时：强制反思，发现盲点
v3_recall("jwt") # 搜索历史记忆
v3_status()      # 查看记忆系统状态

# H1 创造力
h1_inject("JWT认证中间件")  # 获取跨领域灵感
h1_search("JWT best practices 2026")  # 搜索外部参考
h1_graph()       # 查看概念图状态
h1_connect("JWT", "Redis", "depends_on")  # 手动添加概念连接
```

## 工作原理

### V3 — 分层记忆管理

```
L0 当前工作区 (50轮, 全细节)
  ↓ 自动压缩
L1 近期历史 (150轮, 决策+结果)
  ↓ 模式提取
L2 中期记忆 (300轮, 模式+教训)
  ↓ 抽象蒸馏
L3 长期记忆 (无限, 核心洞察)
```

- 自动检测错误循环（连续 3 次同类错误 → 触发反思）
- 每轮决策被追踪，因果链可回溯
- 全部本地 Python 完成，零 LLM token 开销

### H1 — 搜索驱动发散

```
概念图随机游走 → 跨领域联想 → 突显打分 → 一句话注入模型
```

模拟人脑 DMN（默认模式网络）的自发创意机制。

## 文档

- [V3 详细设计](https://github.com/YOUR_USER/v4pro-engineering-runtime/blob/main/docs/v3-design.md)
- [H1 详细设计](https://github.com/YOUR_USER/v4pro-engineering-runtime/blob/main/docs/h1-design.md)
- [神经科学理论依据](https://github.com/YOUR_USER/v4pro-engineering-runtime/blob/main/docs/neuroscience.md)

## 许可证

MIT — 自由使用、修改、分发。
