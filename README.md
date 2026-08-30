# ContextDAG

ContextDAG 为多 Agent 系统的上下文共享提供一种"内容寻址 + 依赖感知"的运行时抽象：Agent 团队共享的上下文被组织为节点 DAG，节点是自包含的语义单元，边是显式声明的依赖；Agent 之间通过语义指针（`node-id`）引用而非拷贝全文，运行时按依赖集为每个 Agent 展开其实际需要的上下文。

本仓库当前阶段的目标是**完整的协议层**（prompt 级、跨模型、免训练、纯 API 模型可用），包括：

- `Node` 数据模型与内容寻址指纹（`content`/`refs`/`meta` 参与哈希，同内容必同 id）；
- `Registry`：不可变、无环、依赖必须先于节点存在的 DAG 存储；
- 展开器：传递依赖集 + 规范拓扑序渲染 + 稳定序列化（依赖集在前、目录在后，最大化共享前缀以命中 prefix cache）；
- 协议指令：`<ref=...>` 前瞻依赖声明与 `<require=...>` 缺页请求（require 即节点边界）；
- 目录与摘要：两级可见性（全文可见 / 目录可见）、侧表摘要（不进 meta，保护内容寻址）、质量分级与懒加载摘要服务接口；
- `Session`：面向 Agent 的运行时 API（注册、展开、缺页请求、读取）与测量计数（缺页次数、require 拒绝数、目录字符开销）。

设计文档见 `docs/overview.md` 与 `docs/protocol.md`（协议设计全书），实现见 `src/contextdag/`。

## 快速开始

```python
from contextdag import Session

session = Session()

# 注册根节点（无依赖）
brief = session.register(
    content="客户要求：退款订单 #X（$120），并换货为型号 A。",
    refs=[],
)

# 展开依赖集：子任务只读依赖闭包，不拷贝全文
ctx = session.expand(refs=[brief.id])
print(ctx.text)
# <node=4cd5f05386b7b6ab>
# 客户要求：退款订单 #X（$120），并换货为型号 A。

# 模型回复后注册新节点，显式声明对 brief 的依赖
reply = session.register(
    content="已核实订单：金额 $120，符合退款条件。",
    refs=[brief.id],
)

# 缺页请求：中途发现还需要目录中的其他节点
ctx2 = session.require("9a3f6c1f...")   # page_faults + 1
```

默认目录 = 最近注册的 1024 个节点（`Session(catalog_size=...)` 可调）；目录条目按
"域 + 触发条件"的决策辅助格式渲染，摘要来源依次为侧表 → `meta.summary` → 首句启发式。

## 开发

环境由 uv 管理（Python 3.12，见 `.python-version` 与 `uv.lock`）：

```bash
uv sync                      # 安装依赖（协议层本身只依赖标准库）
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v   # 运行全部测试
PYTHONPATH=src .venv/bin/python bench/bench_hit_rate.py            # 离线命中率基准（radix 缓存代理）
PYTHONPATH=src .venv/bin/python bench/bench_hit_rate.py --dataset taubench --limit 10   # 真实测试集：tau-bench 轨迹
PYTHONPATH=src .venv/bin/python bench/bench_hit_rate.py --dataset longbench --limit 10  # 真实测试集：LongBench 长文档
PYTHONPATH=src .venv/bin/python demo/chat.py --backend scripted --verbose  # 端到端冒烟（无需模型）
PYTHONPATH=src .venv/bin/python demo/chat.py --backend local --verbose      # 本地 vLLM（需 GPU）

本地真实引擎（vLLM + Qwen3-4B-AWQ）：

```bash
uv sync --extra engine
scripts/start_vllm.sh
```

`--backend local` 默认连接 `http://127.0.0.1:30000/v1`，可通过
`LOCAL_BASE_URL` 和 `LOCAL_MODEL_NAME` 环境变量覆盖。
启动脚本标准参数为 `max-model-len=8192`、`gpu-memory-utilization=0.8`、
`--enforce-eager`，并开启 prefix caching、`prompt_tokens_details` 与
Qwen3 工具调用解析。
```

`demo/chat.py` 支持 `--backend scripted|api|local`、`--catalog-size`（默认 1024）、
`--summary-service`（演示懒加载摘要）与 `--verbose`（查看模型看到的上下文、原始输出与解析结果）。

## 路线图

1. **协议层**（当前阶段）：节点、展开、`ref`/`require`、目录与摘要、测量计数。
2. **测量与调参**：用缺页/require 命中率、拒绝率、目录 token 开销校准 `catalog_size`，并回答"小目录 vs 检索层"的权衡。
3. **编排层能力**：用户显式指定节点（pinned，必选候选）、embedding 候选预筛/排序（大规模目录）。
4. **KV 下放层**：拥有引擎控制权时，按 node-id 寻址的 segment KV 复用（radix / 分页式 cache）。
5. **RL 训练线**（远期）：模型学会自主声明节点结构，由测量数据驱动。
