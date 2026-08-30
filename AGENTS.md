# Repository Guidelines

## Overview

本仓库实现并文档化 ContextDAG——面向多 Agent LLM 系统的"内容寻址 + 依赖感知"上下文运行时。开发按三层组织（详见 `docs/contextdag-docs/RESEARCH-PLAN.md`）：

1. **协议层（当前阶段）**：数据模型、闭包展开器、prompt 级 `ref`/`require` 指令、目录与摘要、测量计数；
2. **引擎层（规划）**：节点/段粒度 KV 复用（对标 HiCache / SparseX 方向），需要本地权重 + 自建/改造推理引擎；
3. **模型层（远期）**：长任务中选择性激活相关上下文（稀疏注意力，NSA / Selective Attention 方向），训练线。

进入引擎层/模型层的门槛是**测量数据**：先用协议层 + 测量线量化"重复表达 token 占比"与"选择性激活收益"，数据支持才深入。

## Design Principles

- 设计时优先考虑结构的**理论真实性**，不要为了结构完整、可执行、便于演示而首先牺牲真实性。
- 如果评估后认为不得不舍弃部分真实性，必须**立即与用户沟通并讨论**，选择尽可能符合用户需求、真实性更高的设计。
- 代理指标（如离线 `Accountant` 模拟缓存）只能用于早期筛选和相对比较，不能替代真实引擎验证，也不应被当作真实性证据。

## Project Structure & Module Organization

- `src/contextdag/` — 协议层实现（Python 标准库）：`node.py`（数据模型、内容寻址指纹）、`registry.py`（不可变、无环 DAG 存储）、`expand.py`（规范拓扑序渲染 + 目录渲染）、`tags.py`（`ref`/`require`/`<summary>` 指令解析）、`summary.py`（侧表摘要 + 懒加载摘要服务）、`session.py`（Agent 面向 API）、`accounting.py`（token 复用核算）、`agent.py`（对话包装）。
- `tests/` — `unittest` 套件，镜像包结构；测试按树状组织（见 Testing Guidelines）。
- `bench/` — 离线命中率基准与 LongBench/tau-bench 评测脚本。
- `demo/` — 可交互演示：`chat.py`（交互式对话 REPL）、`run_demo.py`（端到端调试视图）。
- `docs/` — `overview.md`、`protocol.md`（协议设计全书）与 `contextdag-docs/`（研究计划、文献索引、提案）。
- `README.md` — 项目总览与快速开始。

## Development Workflow

Python 由 **uv 管理**（Python 3.12，见 `.python-version` 与 `uv.lock`）。协议层本身只依赖标准库；bench/demo 的第三方依赖（datasets、huggingface-hub、requests、tokenizers 等）由 uv 统一安装。

```bash
uv sync                                                        # 安装依赖
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v   # 运行全部测试
```

不要将结果写入 `./tmp/`。最终结果按项目传统位置放置（如 `docs/`、`bench/` 或对应的输出目录）；执行过程中产生的可复用中间内容（例如加载的模型、缓存、生成物）统一放入 `./sandbox/`，便于人工审查和后续复用。系统级临时文件仍可放在 `/tmp`。提交前确保全套测试通过。

## Coding Style & Naming Conventions

- Python 遵循 PEP 8：4 空格缩进、`snake_case` 函数、`PascalCase` 类、公开 API 带类型注解。公开名称从 `contextdag/__init__.py` 导出并列入 `__all__`。
- 节点 id 为 16 位十六进制内容寻址前缀（见 `docs/protocol.md`）。
- Markdown 为文档主格式；`docs/contextdag-docs/` 下文件保持清晰的 `#`/`##`/`###` 标题结构。
- 文件名规范：索引与计划文档用全大写（`INDEX.md`、`RESEARCH-PLAN.md`），支撑笔记用小写 kebab-case（如 `background-primer.md`）。
- 研究文档用中文书写；文件名与代码标识符用英文。论文按 arXiv ID 引用，并保持 `INDEX.md` 更新。

## Documentation & Versioning

- 不要在文档、代码注释或长期保留的说明中留存“历史版本、旧接口、已废弃行为、变更记录”等历史信息；历史由 git 保存。
- 不要写“不再使用 xxx”“已移除 xxx”这类历史性描述；文档只保留当前正在采用/现在开始采用的内容。
- 需要查看演进时使用 `git log`、`git blame` 或 diff，不在正文中维护 changelog 或版本对照。
- 每个文档/注释只描述当前版本的真实状态，保持内容干净；迁移说明、弃用原因等一次性信息放入 commit message 或 PR 描述。

## Testing Guidelines

测试使用标准库 `unittest`。测试文件命名 `test_<module>.py`，镜像 `tests/` 下的包结构。

**树状测试结构（必须遵守）**：

- **L0 叶子**：每个最小 feature 一个/几个独立小测试，单独验证该 feature；
- **L1 合并**：类型相近的几个小 feature 合并为一个较大的 feature test——组合逻辑必须真实（共享状态、依赖顺序、失败路径），不是简单拼接断言；
- **L2+ 递归合并**：相近的 L1 测试继续按类型合并进更大的 feature test，逐层上卷到端到端场景；
- **允许重叠**：涵盖相同或相近 feature 集合的测试可在不同层级、不同侧重下重复存在多个（每层断言重点不同：L0 验证正确性、L1 验证组合语义、L2 验证系统行为）。

不轻信已有测试的断言：编写/修改测试前先读实现，以实际行为为准。新行为必须有配套测试。

## Commit & Pull Request Guidelines

提交用简短祈使句摘要（如 "Add closure materializer with canonical rendering"），按逻辑分组（文档、实现、测试可分多次提交）。PR 说明改了什么、为什么，并引用相关 issue 或文档。

## Security & Configuration Tips

不提交模型权重、API 密钥等机密；凭据放环境变量或未跟踪的本地文件。`.gitignore` 排除编辑器/Agent 工作目录与 Python 产物。
