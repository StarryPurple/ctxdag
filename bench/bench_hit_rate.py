"""Offline hit-rate benchmark: how much token reuse does the protocol give?

Usage:
  PYTHONPATH=src .venv/bin/python bench/bench_hit_rate.py
  PYTHONPATH=src .venv/bin/python bench/bench_hit_rate.py --workflow refund_policy

The cached-token metric is a radix-cache proxy: each context is compared
against every previously seen context and the longest shared prefix counts
as reused.  Baselines use the full context (all registered nodes) at each
turn.  Requires a tokenizer; falls back to per-character ids when the
model tokenizer is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from contextdag import Accountant, render_node  # noqa: E402

from adapters import longbench, taubench  # noqa: E402
from workflows import (  # noqa: E402
    BenchSession,
    refund_policy,
    sequential_chat,
    shared_knowledge,
)


def load_tokenize(path: str | None):
    if path and Path(path).exists():
        try:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(path)
            return lambda s: tokenizer.encode(s).ids
        except Exception as exc:
            print(f"[bench] tokenizer load failed ({exc}); using char ids")
    print("[bench] no tokenizer found; using per-character ids")
    return lambda s: list(s)


def run_workflow(name: str, build: callable, tokenize) -> None:
    bench = BenchSession()
    build(bench)
    acct = Accountant(tokenize)
    registry = bench.registry
    for ctx in bench.ctxs:
        marker = "\n\n<目录 可申请范围>"
        catalog = ctx.text.split(marker, 1)[1] if marker in ctx.text else None
        acct.account(
            ctx.text,
            order=ctx.order,
            node_blocks=lambda nid: render_node(registry.get(nid)),
            catalog_text=catalog,
        )

    baseline_tokens = sum(len(tokenize(b)) for b in bench.baselines)
    dep_tokens = acct.total_prompt_tokens
    savings = 1 - dep_tokens / baseline_tokens if baseline_tokens else 0.0

    print(f"\n=== {name}（{len(bench.ctxs)} 次展开）===")
    print("轮次 | prompt tok | 命中 | 新增 | 命中率 | 目录 tok")
    for t in acct.turns:
        print(
            f"{t.turn:>4} | {t.prompt_tokens:>10} | {t.cached_tokens:>4} "
            f"| {t.new_tokens:>4} | {t.hit_rate:>6.1%} | {t.catalog_tokens:>8}"
        )
    print(
        f"合计: 发送 {dep_tokens} tok, 复用 {acct.total_cached_tokens} "
        f"({acct.overall_hit_rate:.1%}), 新增 {dep_tokens - acct.total_cached_tokens}"
    )
    catalog_total = sum(t.catalog_tokens for t in acct.turns)
    print(
        f"其中目录开销: {catalog_total} tok "
        f"({catalog_total / dep_tokens:.1%} of 发送量)"
    )
    print(f"基线(全量上下文): {baseline_tokens} tok → 依赖集模式节省 {savings:.1%}")
    top = sorted(acct.nodes.items(), key=lambda kv: (-kv[1].appearances, kv[0]))
    if top:
        print("节点复用 Top:")
        for nid, stat in top[:5]:
            print(f"  {nid} 出现 {stat.appearances} 次, 块 {stat.block_tokens} tok")


def run_dataset(
    name: str,
    iter_samples: callable,
    tokenize,
    limit: int,
) -> None:
    """Replay real-dataset samples into one shared cache (a persistent
    engine cache spanning sessions) and report aggregate reuse."""
    acct = Accountant(tokenize)
    baseline_total = 0
    samples = 0
    for builder in iter_samples(limit=limit):
        bench = BenchSession()
        builder(bench)
        registry = bench.registry
        for ctx, base in zip(bench.ctxs, bench.baselines):
            marker = "\n\n<目录 可申请范围>"
            catalog = ctx.text.split(marker, 1)[1] if marker in ctx.text else None
            acct.account(
                ctx.text,
                order=ctx.order,
                node_blocks=lambda nid: render_node(registry.get(nid)),
                catalog_text=catalog,
            )
            baseline_total += len(tokenize(base))
        samples += 1

    dep_tokens = acct.total_prompt_tokens
    savings = 1 - dep_tokens / baseline_total if baseline_total else 0.0
    catalog_total = sum(t.catalog_tokens for t in acct.turns)
    mean_hit = (
        sum(t.hit_rate for t in acct.turns) / len(acct.turns) if acct.turns else 0.0
    )
    print(f"\n=== {name}（{samples} 样本, {len(acct.turns)} 次展开, 共享缓存）===")
    print(
        f"发送 {dep_tokens} tok, 复用 {acct.total_cached_tokens} "
        f"(总体 {acct.overall_hit_rate:.1%}, 每轮平均 {mean_hit:.1%})"
    )
    print(f"目录开销: {catalog_total} tok ({catalog_total / dep_tokens:.1%} of 发送量)")
    print(f"基线(全量上下文): {baseline_total} tok → 依赖集模式节省 {savings:.1%}")
    top = sorted(acct.nodes.items(), key=lambda kv: (-kv[1].appearances, kv[0]))
    if top:
        print("节点复用 Top:")
        for nid, stat in top[:5]:
            print(f"  {nid} 出现 {stat.appearances} 次, 块 {stat.block_tokens} tok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        default="data/models/qwen3-4b-awq/tokenizer.json",
    )
    parser.add_argument(
        "--workflow",
        choices=["sequential_chat", "shared_knowledge", "refund_policy"],
        default=None,
    )
    parser.add_argument(
        "--dataset",
        choices=["taubench", "longbench"],
        default=None,
        help="在真实测试集上运行（共享缓存跨样本）",
    )
    parser.add_argument("--limit", type=int, default=10, help="每数据集样本数")
    args = parser.parse_args()
    tokenize = load_tokenize(args.tokenizer)
    if args.dataset == "taubench":
        run_dataset(
            f"tau-bench（gpt-4o-retail，前 {args.limit} 条轨迹）",
            taubench.iter_samples,
            tokenize,
            args.limit,
        )
        return
    if args.dataset == "longbench":
        run_dataset(
            f"LongBench（multi_news_e，前 {args.limit} 篇文档，渐进阅读）",
            longbench.iter_samples,
            tokenize,
            args.limit,
        )
        return
    workflows = {
        "sequential_chat": (sequential_chat, "顺序对话链（前缀持续增长）"),
        "shared_knowledge": (shared_knowledge, "共享知识 + agent spawn 树"),
        "refund_policy": (refund_policy, "退款/换货多 worker + require 缺页"),
    }
    for name, (fn, label) in workflows.items():
        if args.workflow and args.workflow != name:
            continue
        run_workflow(label, fn, tokenize)


if __name__ == "__main__":
    main()
