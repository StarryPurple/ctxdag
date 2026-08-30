# 研究计划

本文件记录 ContextDAG 的总体开发路线与当前阶段主线。协议与设计详见 `docs/overview.md` 与 `docs/protocol.md`；文献清单见 [INDEX.md](./INDEX.md)。

## 1. 问题与三层模型

目标：多 Agent 协作中"谁需要谁的什么结果"显式、可验证；每个 Agent 只看到自己需要的上下文，省去重复表达的 token，长任务中按需激活相关上下文。

实现分三层——**不是二选一，是同一路线的三个阶段**：

| 层 | 内容 | 对标物 | 是否动模型 |
|---|---|---|---|
| 协议层（文字层） | 内容寻址节点 + 依赖集展开 + `ref`/`require` 指令 + 目录/摘要 | A-Mem、Anthropic Context Management、渐进式披露 | 否，任何 API 模型可用 |
| 引擎层（KV 层） | 节点/段粒度 KV 复用（替代/增强 RadixAttention 的前缀树缓存） | SGLang RadixAttention、HiCache、SparseX、Mooncake | 否，但需本地权重 + 自建/改造引擎 |
| 模型层（激活层） | 长任务中选择性激活相关上下文（稀疏注意力） | NSA、Selective Attention、SPA、DELTA | 是，需训练/微调或改 attention mask |

## 2. 关键决策（2025-08 重开时定）

1. **方向**：协议层方向正确且有充分对标；"指针语义 + 选择性激活"应分三层落地，而非直接改模型。
2. **进入下一层的门槛是测量数据**：先用协议层 + 测量线量化"重复表达 token 占比"与"选择性激活收益"，数据支持才深入引擎层/模型层。
3. **硬件**：宿主机有 RTX 4060 8GB，开发环境（Agent 容器）不能直接访问 GPU；由宿主机运行真实引擎（vLLM），Agent 通过 HTTP 调用。协议层 + 离线测量用 CPU/API 即可；模型层训练再按需租 24G 单卡起步。
4. **测试纪律**：树状测试结构（L0 小测试 → L1 合并 → L2+ 递归合并，允许重叠），详见 AGENTS.md。
5. **工程组织**：单仓库分层（`src/contextdag` 协议 / `engine/` 引擎 / `train/` 模型层）；Python 用 uv 管理；保留可与人交互展示的对象（统计报表、交互式 demo）。
6. **真实性优先**：离线代理（如 `Accountant`）只用于早期筛选和相对比较，不能替代真实引擎验证；进入引擎层前必须用真实引擎（不限定 SGLang，可以是 vLLM、自研或其它方案）测量真实缓存命中。模型层后续也会包含缓存实现优化，因此先建立真实引擎验证是合理路径。
7. **当前真实引擎采用 vLLM**：本机 8GB GPU 使用 Qwen3-4B-AWQ + vLLM，开启 prefix caching 做真实质量评测与缓存命中验证；SGLang 作为引擎层研究的参考对象。

## 3. 当前阶段：协议层（已完成）与测量线（进行中）

### 3.1 协议层现状

- 节点/注册表/展开器/指令解析/目录摘要/测量计数全部实现，87 个测试通过。
- 设计文档：`docs/overview.md`（总览与术语）、`docs/protocol.md`（协议设计全书）。

### 3.2 测量线现状

- 离线命中率基准：`bench/bench_hit_rate.py`（radix-cache 代理核算 token 复用；该指标仅用于早期筛选，不作为真实缓存证据）。
- Level-1 质量评测：`bench/eval_longbench.py`（full / protocol / pruned / no-headers 四条件对比），尚未有正式结果，需按固定方法重跑。
- Level-2 任务成功率：`bench/eval_taubench.py`（oracle / baseline / protocol 三条件），**尚未跑出结果**。

### 3.3 测量线待办

1. 重跑并扩大 LongBench 样本（每数据集 20+），固定评测方法；
2. 跑通 tau-bench Level-2（scripted 冒烟 → local/API 真实）；
3. 校准 `catalog_size`（目录 token 开销 vs require 触发质量）；
4. 输出决策文档：回答"小目录 vs 检索层"与"是否值得进入引擎层"两个问题。

## 4. 后续阶段（由数据门槛触发）

- **引擎层**：节点粒度 KV 复用（HiCache / SparseX 方向）。先写接口契约（协议渲染文字 + node-id 的解析协议），再用真实引擎实现并测量真实缓存命中，不把离线代理当作结论。
- **模型层**：选择性激活（NSA 式训练线或 Selective Attention 式掩码控制），远期；模型层可与引擎层的缓存实现协同，先建立真实引擎验证是合理路径。
