# MaskAttn 真实模型测量

## 环境与实验口径

Qwen2.5-14B-Instruct-AWQ，RTX 4090 D 24GB，PyTorch 2.7.1+cu126；独立实验目录使用 Transformers 4.51.3、AutoAWQ 0.2.9、tokenizers 0.21.4、accelerate 1.14.0。计算为 FP16 激活、AWQ 权重与 eager attention。`bench/eval_mask_attn.py` 直接调用模型，不经过 vLLM。

四条件采用完全相同的分块 token：普通因果 mask/全局位置、DAG mask/全局位置、DAG mask/依赖相对位置全量计算、DAG mask/依赖相对位置分支 KV 复用。每个证据分支只依赖共享根，回答依赖全部分支。该依赖图是默认实验划分，不代表标注的语义依赖。

分支 KV 单独计算后在每层拼接，回答只计算新增 token。额外前缀对照从同一次 DAG 全量计算提取前缀 KV，再计算回答，以检查分段执行本身的数值误差。

## 质量与一致性

| 指标 | 合成事实探针，12 条 | 2WikiMQA 前 10 条 |
|---|---:|---:|
| DAG 全量/复用生成文本一致 | 12/12 | 10/10 |
| DAG 全量/复用首 token 一致 | 12/12 | 10/10 |
| DAG 全量/复用最大 logit 误差 | 0.453125 | 0.5078125 |
| 同次全量前缀对照最大 logit 误差 | 0.23046875 | 0.2578125 |
| 四条件严格 exact match | 均为 100% | 均为 0% |

2WikiMQA 每条只保留前 2,048 个正文 token，最多生成 32 token；合成探针最多生成 16 token。答案按完整生成文本评分，未对长解释进行答案提取。部分问答证据被截断，不能与 8,000-token 协议评测直接比较。

2WikiMQA 平均 token F1：普通因果 0.2112、DAG 全局位置 0.1992、DAG 相对位置全量 0.1285、DAG 相对位置复用 0.1285。该小样本支持复用路径与 DAG 重算输出一致的观察，不支持 DAG mask 或相对位置策略质量无损。尚未用统计检验证明质量差异。

非零 logits/KV 误差仍需诊断。普通前缀对照同样存在误差，说明分段执行也影响数值；不能因此认定跨分支复用的全部误差都是安全舍入。AWQ 的 GEMM 路径可随输入长度变化，需进一步控制量化 kernel、dtype 和矩阵形状。

另取 3 条合成探针，在相同输入与形状下重复全量前向，最大 logit 差异分别为 0.21875、0.375、0.203125，首 token 仍一致。这确认当前执行路径存在数值非确定性；具体来源尚未定位，不能仅凭设置随机种子保证逐次 logits 一致。对应原始报告为 `bench/results/maskattn-repeat-control-n3.json`，附代码指纹、模型配置及聚合数据。

## 成本口径

| GPU 同步后的平均前向时间 | 合成探针 | 2WikiMQA |
|---|---:|---:|
| 普通因果全量 | 95.62 ms | 787.17 ms |
| DAG 相对位置全量 | 95.61 ms | 797.67 ms |
| 根与分支缓存构建，各前向之和 | 225.48 ms | 716.05 ms |
| 已有分支缓存的拼接 | 1.25 ms | 1.58 ms |
| 已有缓存后的回答前向 | 75.87 ms | 52.85 ms |

时间是固定顺序、单次每样本的诊断测量，没有置信区间；不包括全部 CPU 规划、缓存切片/复制和解码成本，不是 serving TTFT、吞吐或端到端加速。稠密 eager mask 不会跳过被屏蔽的矩阵运算。

2WikiMQA 共 21,291 个输入 token，其中 20,860 个前缀 token 的 KV 在回答前已构建，回答阶段只新增 431 token；97.98% 是 warm 查询阶段的复用占比，不是整个工作流 token 减少。缓存构建仍处理全部证据。峰值 allocated 显存约 11.64 GiB，包含诊断用缓存副本。

## 复现与下一步

原始结果在 `bench/results/maskattn-prefix-control-n12.json` 与 `bench/results/maskattn-2wiki-n10.json`，本地保留已执行脚本快照 `sandbox/maskattn/eval_mask_attn_executed.py`。结果与快照不自动随 Git 发布。

```bash
PYTHONPATH=sandbox/maskattn-deps:src /root/autodl-tmp/ctxdag-venv/bin/python bench/eval_mask_attn.py \
  --model /root/autodl-tmp/models/Qwen2.5-14B-Instruct-AWQ \
  --samples 10 --data data/longbench/data/2wikimqa_e.jsonl \
  --context-tokens 2048 --max-new-tokens 32 \
  --out bench/results/maskattn-2wiki-n10.json
```

后续先定位同形状重复前向的非确定性，在更可控的 kernel 或非量化模型上验证数值容忍范围；再在证据完整的任务上扩大质量评估。需要对照普通前缀缓存，并用多查询共享分支的负载摊销构建成本，才能判断 DAG 复用的额外收益。
