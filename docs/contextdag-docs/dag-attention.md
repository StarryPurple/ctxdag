# DAG Attention 正确性实验

## 实现与边界

`bench/dag_attention.py` 是标准库 CPU 数值参考：随机权重、三层单头残差 attention、位置特征与输出 logits。它执行实际 Q/K/V 运算，验证缓存语义；没有预训练、RoPE、MLP、归一化或高效稀疏 kernel，不能据此判断回答质量或 serving 性能。

节点内使用因果可见性，跨节点只允许读取显式祖先。回答节点把所需证据分支声明为父节点。节点起始位置取父节点结束位置的最大值，节点内连续递增；独立分支可共享位置编号。该位置策略的预训练分布偏移必须在真实模型上测量。

缓存保存在单个模型实例中，身份包含节点 token、位置与递归父节点身份。缓存保存每层 K/V 和隐藏状态；重用节点跳过投影与 attention，新节点仍计算对可见祖先的 attention。该参考没有显存管理、淘汰、跨模型共享或持久化缓存。

## 运行与观测

```bash
python3 bench/dag_attention.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

固定随机种子 7，先计算 root/left，再加入 right 和 answer：

| 观测 | 最大 logit 绝对误差 |
|---|---:|
| 复用 4 个 token 对比 DAG 全量重算 | 0.0 |
| DAG mask + 依赖相对位置：插入无关分支 | 0.0 |
| 普通因果 mask + 依赖相对位置：插入分支 | 0.156678 |
| DAG mask + 全局位置：插入分支 | 0.565312 |

这些数字仅描述该确定性参考。测试还覆盖祖先变化使后代缓存失效、分支重排与汇合回答重算一致、非法依赖拒绝。误差阈值为 1e-12，不要求不同计算后端逐 bit 一致。

## 后续真实模型验证

在支持显式 attention mask 和 position ids 的预训练模型上，分别比较普通因果全量重算、DAG 全量重算、DAG 节点缓存。前两者评估质量变化，后两者验证 logits 正确性；同时保留普通全局位置作为位置策略消融。需要固定模型修订、tokenizer、dtype、位置配置与实际 token 序列，不能把节点身份直接当成引擎 KV 缓存键。

CPU 参考已完成；真实 14B 模型的初步数值与问答观察见 [maskattn-results.md](./maskattn-results.md)。GPU serving 延迟与吞吐仍需专门测量。该实验不构成新颖性证明。
