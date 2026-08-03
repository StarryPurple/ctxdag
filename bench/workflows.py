"""Scripted multi-agent workflows for hit-rate benchmarking.

Every workflow returns its ``ExpandedContext`` list plus per-turn
full-context baselines (all registered nodes) for comparison.
"""

from __future__ import annotations

from contextdag import Session


class BenchSession:
    """Session wrapper that records every expansion and a full-context
    baseline snapshot (all registered nodes at that moment)."""

    def __init__(self, session: Session | None = None) -> None:
        self.session = session or Session()
        self.ctxs: list = []
        self.baselines: list[str] = []

    def register(self, *args, **kwargs):
        return self.session.register(*args, **kwargs)

    def register_declared(self, *args, **kwargs):
        return self.session.register_declared(*args, **kwargs)

    def expand(self, *args, **kwargs):
        ctx = self.session.expand(*args, **kwargs)
        self.ctxs.append(ctx)
        self.baselines.append(self.session.full_context())
        return ctx

    @property
    def registry(self):
        return self.session.registry


def sequential_chat(session: BenchSession, turns: int = 4) -> None:
    """Conversational chain: every turn expands the growing prefix."""
    prev: str | None = None
    for i in range(turns):
        user = session.register(
            content=(
                f"用户问题 {i}：请基于前面的对话与共享资料，说明当前进展、"
                "尚未解决的问题，以及下一步需要谁配合。"
            ),
            refs=(prev,) if prev else (),
        )
        session.expand(refs=[user.id])
        reply = session.register_declared(
            f"<ref={user.id}> 第 {i} 轮回答：已完成阶段 {i} 的核对，"
            "结论记录在案，等待下一轮输入。",
            default_refs=[user.id],
        )
        prev = reply.id if reply else None


def shared_knowledge(
    session: BenchSession,
    shared: int = 6,
    agents: int = 5,
    refs_per_agent: int = 3,
    spawn_children: int = 2,
) -> None:
    """Agents share a brief and overlapping knowledge nodes, then spawn
    children that share their parent's prefix before diverging."""
    brief = session.register(
        content=(
            "任务简报：请基于共享知识库完成一次综合评估，覆盖各领域的现状、"
            "主要风险与改进建议，并在最后给出优先级排序。"
        )
    )
    kn = [
        session.register(
            content=(
                f"共享知识节点 {j}：领域 {j} 的关键事实。最近一轮数据更新显示"
                f"领域 {j} 的指标从基准值持续上升，主要驱动因素包括外部需求"
                "变化与内部流程调整；建议在下一阶段保持观察并准备应对预案。"
            )
        )
        for j in range(shared)
    ]
    parents = []
    for a in range(agents):
        refs = [brief.id] + [kn[(a + j) % shared].id for j in range(refs_per_agent)]
        session.expand(refs=refs)
        node = session.register(
            content=(
                f"Agent {a} 的评估结论：基于所引用领域的现状，确认风险点集中在"
                "数据一致性与流程衔接，建议按优先级逐步推进整改并复测。"
            ),
            refs=refs,
        )
        parents.append(node)
    for a, parent in enumerate(parents):
        for c in range(spawn_children):
            session.expand(refs=[parent.id])
            session.register(
                content=(
                    f"子任务 {a}-{c} 的输出：针对 {a} 号评估中的第 {c + 1} 条建议"
                    "给出实施方案、负责人与时间线，并标注依赖的前置节点。"
                ),
                refs=[parent.id],
            )


def refund_policy(session: BenchSession) -> None:
    """The refund/policy multi-worker scenario with a require page fault."""
    brief = session.register(
        content="客户要求：退款订单 #X（$120），并换货为型号 A。"
    )
    policy = session.register(
        content="换货政策：30 天内可换新款；退款需在确认订单后发起。",
        meta={"summary": "换货与退款政策；判断退款资格、换货条件时使用"},
    )
    session.expand(refs=[brief.id], candidates=[policy.id])
    n1 = session.register_declared(
        "已核实订单：金额 $120，符合退款条件。",
        default_refs=[brief.id],
    )
    session.expand(refs=[n1.id], candidates=[policy.id])
    n2 = session.register_declared(
        f"需要确认换货约束。<require={policy.id}> 继续执行换货。",
        default_refs=[n1.id],
    )
    session.expand(refs=[n1.id, n2.id])
    session.register_declared(
        "综合结论：退款 $120 并换货为型号 A。",
        default_refs=[n1.id, n2.id],
    )
