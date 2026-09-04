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
3. **硬件**：协议层与离线测量在 CPU 环境开发；真实模型实验使用独立 GPU 实例，通过 OpenAI 兼容 HTTP 接口调用。模型层训练根据测量结果再确定显存需求。
4. **测试纪律**：树状测试结构（L0 小测试 → L1 合并 → L2+ 递归合并，允许重叠），详见 AGENTS.md。
5. **工程组织**：单仓库分层（`src/contextdag` 协议 / `engine/` 引擎 / `train/` 模型层）；Python 用 uv 管理；保留可与人交互展示的对象（统计报表、交互式 demo）。
6. **真实性优先**：离线代理（如 `Accountant`）只用于早期筛选和相对比较，不能替代真实引擎验证；进入引擎层前必须用真实引擎（不限定 SGLang，可以是 vLLM、自研或其它方案）测量真实缓存命中。模型层后续也会包含缓存实现优化，因此先建立真实引擎验证是合理路径。
7. **当前真实引擎采用 vLLM**：已验证环境使用 RTX 4090D、Qwen2.5-14B-Instruct-AWQ 与 vLLM 0.10.1，开启 prefix caching 和逐请求 token 明细；SGLang 作为引擎层研究的参考对象。

## 3. 当前阶段：Context Compiler 测量线

系统主线以 [context-compiler.md](./context-compiler.md) 为契约：协议层提供内容身份、依赖闭包和运行时授权；编译器把这些语义对象布局成稳定控制面、共享前缀和可路由缓存描述符；推理引擎负责 KV 执行。研究主张必须以真实引擎的质量、TTFT、吞吐、实际 prefill 与授权审计共同验证。

### 3.1 协议层现状

- 节点/注册表/展开器、结构化特征侧表、启发式与严格 JSON 提取器、多字段词法索引、控制动作、原生工具与标签传输、测量计数全部实现，137 个测试通过。
- 设计文档：`docs/overview.md`（总览与术语）、`docs/protocol.md`（协议设计全书）。

### 3.2 测量线现状

- Measurement v1：统一记录代码版本、运行配置、逐样本观测与聚合结果，方法见 `measurement-v1.md`。
- 选择—恢复闭环：`bench/eval_recovery.py` 比较完整上下文、位置裁剪、关键词初选和 `require` 恢复；50 条参数扫描确认恢复显著优于只初选，但第二轮无净收益，完整结果见 `measurement-results.md`。
- 离线命中率基准：`bench/bench_hit_rate.py`（radix-cache 代理核算 token 复用；该指标仅用于早期筛选，不作为真实缓存证据）。
- Level-1 质量评测：`bench/eval_longbench.py`（full / protocol / pruned / no-headers 四条件对比）。100 条正式样本中，固定前半段条件输入 token 减少 60.8%，但质量差异的 95% 配对区间跨零；详见 `measurement-results.md`。
- Level-2 任务成功率：`bench/eval_taubench.py`（oracle / baseline / protocol 三条件）。10 条 retail 任务的模型条件均未成功，诊断复跑确认工具链可完成任务；当前结果不足以比较质量。
- 真实缓存验证：独立冷启动的 vLLM 实验中，protocol 在三个场景都提高前缀缓存命中率；稳定前缀 `messages` 恢复也将二次调用命中率从 16.97% 提高到 22.53%，但质量显著低于重渲染传输。两者都是共享前缀证据，不是节点级 KV 证据。

### 3.3 测量线待办

1. 定义可重放的 Agent/DAG trace 格式，并采集 vLLM 的实际 prefill、TTFT、延迟和缓存观测；
2. 将现有 tags、动态 tools 与固定 grammar tools 作为 H1 基线，扩大独立冷启动样本并预先声明质量容忍区间；
3. 实现布局编译器和 cache descriptor，以共享祖先、分叉状态和低重叠请求验证 H2；
4. 接入 SGLang，确认 H1/H2 不依赖 vLLM 的模板或缓存实现；
5. 扩大 tau-bench 任务数与重复种子，记录分类型失败并比较任务成功率；
6. 在单副本结论稳定后，构造多副本负载并验证闭包亲和路由 H3。

## 4. 后续阶段（由数据门槛触发）

- **引擎层**：节点粒度 KV 复用（HiCache / SparseX 方向）。先写接口契约（协议渲染文字 + node-id 的解析协议），再用真实引擎实现并测量真实缓存命中，不把离线代理当作结论。
- **模型层**：选择性激活（NSA 式训练线或 Selective Attention 式掩码控制），远期；模型层可与引擎层的缓存实现协同，先建立真实引擎验证是合理路径。
