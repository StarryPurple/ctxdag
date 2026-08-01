# ContextDAG

ContextDAG 为多 Agent 系统的上下文共享提供一种"内容寻址 + 按需物化"的运行时抽象：Agent 团队共享的上下文被组织为节点 DAG，节点是自包含的语义单元，边是显式声明的依赖；Agent 之间通过语义指针（`node-id`）引用而非拷贝全文，运行时按依赖闭包为每个 Agent 物化其实际需要的上下文。

本仓库当前阶段的目标是**完整的协议层**（prompt 级、跨模型、免训练、纯 API 模型可用），具体包括：

- `Node` 数据模型与内容寻址指纹；
- `Registry`：不可变、无环、依赖必须先于节点存在的 DAG 存储；
- 物化器：传递闭包计算 + 规范拓扑序渲染 + 稳定序列化（同一节点在任何闭包中字节一致，最大化共享前缀以命中 prefix cache）；
- 协议指令：`<ref=...>` 前瞻依赖声明与 `<require=...>` 缺页兜底；
- `Session`：面向 Agent 的运行时 API（物化、校验、注册、缺页扩展、deref）。

设计文档见 `docs/contextdag-docs/`（方向、开题报告与研究计划），协议层实现见 `src/contextdag/`。

## 快速开始

```python
from contextdag import Session, Node

session = Session()

# 注册根节点（无依赖）
analysis = session.complete(
    content="架构评审结论：将通信层拆为独立包，协议与传输解耦。",
    refs=[],
)

# 声明式引用：子任务只读依赖闭包，不拷贝全文
context = session.materialize(refs=[analysis.id])
impl = session.complete(
    content="按评审结论实现 protocol 包，对外暴露 Session API。",
    refs=[analysis.id],
)
```

## 开发

```bash
python3 -m unittest discover -s tests -v   # 运行全部测试
```

实现只依赖 Python 标准库。

## 路线图

1. **协议层**（本阶段）：数据模型、闭包物化、规范渲染、`ref`/`require` 指令、Agent Session API。
2. **测量研究**：subagent 冷启动开销量化（token、TTFT、传话失真率），覆盖 orchestrator-worker / 黑板 / 流水线三种编排模式。
3. **KV 下放层**：node-id 寻址的 segment KV 复用（Mini-SGLang / vLLM），作为同一抽象在有引擎控制权时的加速选项。
4. **RL 训练线**：模型学会自声明 `<node>`/`<ref>` 结构（见 `RESEARCH-PLAN.md`）。
