# 研究文档索引

本目录收录 ContextDAG 的研究计划、文献索引与提案。研究文档用中文书写；文件名与代码标识符用英文。论文按 arXiv ID 引用，新增文献必须在此登记后才能在其他文档引用。

## 目录

| 文档 | 说明 |
|---|---|
| [RESEARCH-PLAN.md](./RESEARCH-PLAN.md) | 三层开发路线（协议层 / 引擎层 / 模型层）与测量线主线 |
| [context-compiler.md](./context-compiler.md) | 面向 LLM serving 的 ContextDAG 编译器命题、系统边界与评测契约 |
| [measurement-v1.md](./measurement-v1.md) | 统一实验报告契约、执行顺序与判读边界 |
| [measurement-results.md](./measurement-results.md) | 当前正式测量结果、统计边界与阶段决策 |
| INDEX.md（本文件） | 研究文档索引 + 文献清单 |

## 文献索引

### 上下文复用与 KV 缓存（引擎层对标）

- **SGLang / RadixAttention** — *SGLang: Efficient Execution of Structured Language Model Programs*，NeurIPS 2024，arXiv:2312.07104。前缀树 KV 缓存：只对共享前缀友好；中间/任意位置的共享块（DAG 结构）不命中。[SGLang 文档](https://docs.sglang.io/)
- **SparseX** — *SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving*，arXiv:2606.01751。段级 KV 缓存共享，引擎层直接对标。
- **HiCache / Mooncake** — SGLang/Mooncake 的分层 KV cache 与 KVCache-centric 调度。[HiCache 设计文档](https://docs.sglang.io/docs/advanced_features/hicache_design) / [Mooncake](https://github.com/kvcache-ai/Mooncake)

### 选择性激活 / 稀疏注意力（模型层对标）

- **NSA** — *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*，DeepSeek，arXiv:2502.11089。原生稀疏注意力，需训练。
- **Selective Attention** — *Selective Attention: Enhancing Transformer through Principled Context Control*，ICLR 2024，arXiv:2410.02703。推理期可控注意力掩码。
- **Long-Context Generalization with Sparse Attention** — arXiv:2506.16640。
- **HGCA** — *Hybrid GPU-CPU Attention for Long Context LLM Inference*，arXiv:2507.03153。选择性关注关键中间结果。
- **DELTA** — *Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning*，ACL Findings 2026。[论文页](https://aclanthology.org/2026.findings-acl.558/)

### 内容寻址记忆与 Agent 上下文（协议层对标）

- **A-Mem** — *A-Mem: Agentic Memory for LLM Agents*，arXiv:2502.12110。内容寻址的记忆对象 + 记忆图。
- **Context Management / 渐进式披露** — Anthropic Context Management、Claude Skills 的"短描述 + 按需加载"模式。
- **Prompt Caching 经济性** — *Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads*，arXiv:2607.19214；[Prompt Caching 综述](https://futureagi.com/blog/understanding-prompt-caching-for-faster-ai-responses/)
