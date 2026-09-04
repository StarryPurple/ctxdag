# Measurement v1

## 目标

Measurement v1 用同一数据契约验证 ContextDAG 的三个独立效果：输入 token 减少、任务质量保持和真实引擎缓存收益。三类结果不得相互替代；离线 `Accountant` 仅表示前缀复用代理，不构成真实 KV 缓存证据。

## 报告契约

正式评测输出 `contextdag.measurement.v1` JSON，默认写入 `bench/results/`。报告包含：

- `revision`：commit 与工作区是否存在已跟踪修改；
- `runtime`、`backend`、`config`：复现实验所需环境；
- `observations`：逐样本或逐轮原始观测；
- `aggregates`：按实验条件聚合的质量、token 或缓存指标。

运行中断时采用原子写入，已有 JSON 保持可读取。正式报告应保留完整原始观测；论文式结论另写文档，不修改原始数据。

## 执行顺序

先运行无需模型的管线检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python bench/eval_longbench.py \
  --backend scripted --tokenizer char --limit 2
PYTHONPATH=src python bench/eval_taubench.py \
  --backend scripted --tokenizer char --limit 2
```

scripted 输出只验证加载、状态推进、评分和报告生成。质量实验使用固定 tokenizer 与同一模型依次运行全部条件；LongBench 每个目标数据集至少 20 条。tau-bench 先确认 oracle，再比较 baseline 与 protocol。选择—恢复闭环先比较 full、protocol-full、fixed-half、keyword-initial 与 keyword-recovery：

```bash
PYTHONPATH=src python bench/eval_recovery.py \
  --backend scripted --tokenizer char --limit 2 --selection-ratio 0.5 \
  --transport tools
```

恢复评测支持 `--transport flattened`（每轮重渲染完整上下文）、`--transport messages`（文本标签兼容路径）与 `--transport tools`（强制在 `require_context` 与 `return_answer` 间选择，并以 tool result 返回节点）。后两者用于检验 API/引擎的真实 prefix cache；追加消息不会重复已可见节点。该实验分别记录初始与恢复后的质量、逻辑与 API token、cached token、completion token、延迟、新增节点 token、`require` 轮数、请求节点摘要和失败类型。参数矩阵由 `bench/run_recovery_sweep.py` 执行；同一服务内顺序运行的缓存数据只用于诊断，正式缓存比较仍需独立冷启动。`fixed-half` 只是位置裁剪基线，不是 oracle relevance；缺少可靠相关性标注时不得将其解释为理论上限。真实缓存实验由开启 prefix caching 和指标端点的 vLLM 执行：

```bash
PYTHONPATH=src python bench/bench_real_cache.py \
  --workflow refund_policy --baseline
```

## 判读边界

进入引擎层前必须同时观察到：protocol 显著减少输入 token、质量下降处于预设容忍区间、目录与缺页开销未抵消收益，以及真实引擎测得可复现的缓存或延迟收益。vLLM 的 prefix-cache 命中只证明共享前缀复用；非前缀节点复用仍需节点/段级引擎实验验证。
