"""Real-engine cache validation through an OpenAI-compatible server (vLLM).

Unlike ``Accountant``, this benchmark sends the actual rendered contexts to a
real serving engine and reads engine-reported cache metrics.

It reads two data sources when available:
  1. OpenAI usage.prompt_tokens_details.cached_tokens (vLLM needs
     --enable-prompt-tokens-details for this field)
  2. vLLM Prometheus /metrics prefix_cache counters (always available with
     --enable-prefix-caching)

Usage:
  PYTHONPATH=src .venv/bin/python bench/bench_real_cache.py --workflow refund_policy
  PYTHONPATH=src .venv/bin/python bench/bench_real_cache.py --workflow sequential_chat --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from workflows import (  # noqa: E402
    BenchSession,
    refund_policy,
    sequential_chat,
    shared_knowledge,
)

WORKFLOWS = {
    "sequential_chat": (sequential_chat, "顺序对话链（前缀持续增长）"),
    "shared_knowledge": (shared_knowledge, "共享知识 + agent spawn 树"),
    "refund_policy": (refund_policy, "退款/换货多 worker + require 缺页"),
}


def _server_root(base: str) -> str:
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _read_cache_counters(metrics_url: str) -> tuple[int, int]:
    """Return (queries_total, hits_total) from vLLM /metrics."""
    resp = requests.get(metrics_url, timeout=10)
    resp.raise_for_status()
    text = resp.text
    patterns = [
        (r"^vllm:prefix_cache_queries_total\{(.*)\} (\d+)",
         r"^vllm:prefix_cache_hits_total\{(.*)\} (\d+)"),
        (r"^vllm:gpu_prefix_cache_queries_total\{(.*)\} (\d+)",
         r"^vllm:gpu_prefix_cache_hits_total\{(.*)\} (\d+)"),
    ]
    for query_re, hit_re in patterns:
        q = re.search(query_re, text, re.MULTILINE)
        h = re.search(hit_re, text, re.MULTILINE)
        if q and h:
            return int(q.group(2)), int(h.group(2))
    return 0, 0


def _chat(base: str, model: str, prompt: str, max_tokens: int = 1) -> dict:
    """Send one prompt to an OpenAI-compatible chat endpoint and return usage."""
    resp = requests.post(
        f"{base}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,
    )
    body = resp.json()
    if "error" in body or "detail" in body:
        raise RuntimeError(
            f"server error: {body.get('error') or body.get('detail')}"
        )
    usage = body.get("usage", {}) or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "cached_tokens": int(details.get("cached_tokens", 0) or 0)
        if details else 0,
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def run_sequence(
    name: str,
    prompts: list[str],
    base: str,
    model: str,
    metrics_url: str,
) -> tuple[list[dict], int, int]:
    before_q, before_h = _read_cache_counters(metrics_url)
    stats = []
    for i, prompt in enumerate(prompts, start=1):
        info = _chat(base, model, prompt)
        info["turn"] = i
        info["text_len"] = len(prompt)
        stats.append(info)
        hit = info["cached_tokens"] / info["prompt_tokens"] if info["prompt_tokens"] else 0.0
        print(
            f"  [{name}] turn {i}: prompt={info['prompt_tokens']} "
            f"api_cached={info['cached_tokens']} api_hit={hit:.1%}",
            flush=True,
        )
    after_q, after_h = _read_cache_counters(metrics_url)
    metric_queries = after_q - before_q
    metric_hits = after_h - before_h
    return stats, metric_queries, metric_hits


def summarize(
    name: str,
    stats: list[dict],
    metric_queries: int,
    metric_hits: int,
) -> dict:
    prompt = sum(s["prompt_tokens"] for s in stats)
    api_cached = sum(s["cached_tokens"] for s in stats)
    # Prefer per-request API cached_tokens when the server exposes it;
    # otherwise fall back to Prometheus prefix-cache counters.
    if api_cached > 0:
        cached = api_cached
        source = "api"
    else:
        cached = metric_hits
        source = "metrics"
    return {
        "name": name,
        "n_turns": len(stats),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cache_source": source,
        "metric_queries": metric_queries,
        "metric_hits": metric_hits,
        "hit_rate": cached / prompt if prompt else 0.0,
        "turns": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=os.environ.get(
            "LOCAL_BASE_URL", "http://127.0.0.1:30000/v1"
        ),
        help="OpenAI-compatible base URL, default LOCAL_BASE_URL or local vLLM",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "LOCAL_MODEL_NAME", "data/models/qwen3-4b-awq"
        ),
        help="model name served by the endpoint",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="vLLM Prometheus metrics URL; default derived from --base",
    )
    parser.add_argument(
        "--workflow",
        choices=list(WORKFLOWS),
        default="refund_policy",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also send full-context baselines for comparison",
    )
    parser.add_argument(
        "--out",
        default="results/bench_real_cache.json",
    )
    args = parser.parse_args()

    metrics_url = args.metrics_url or f"{_server_root(args.base)}/metrics"
    build, label = WORKFLOWS[args.workflow]
    bench = BenchSession()
    build(bench)

    print(f"\n=== 真实引擎缓存验证: {label} ===")
    protocol_stats, p_q, p_h = run_sequence(
        "protocol", [ctx.text for ctx in bench.ctxs], args.base, args.model, metrics_url
    )
    result = {
        "workflow": args.workflow,
        "metrics_url": metrics_url,
        "protocol": summarize(args.workflow, protocol_stats, p_q, p_h),
    }

    if args.baseline:
        baseline_stats, b_q, b_h = run_sequence(
            "baseline", list(bench.baselines), args.base, args.model, metrics_url
        )
        result["baseline"] = summarize("baseline", baseline_stats, b_q, b_h)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n--- 汇总 ---")
    for key in ("protocol", "baseline"):
        if key in result:
            s = result[key]
            print(
                f"{key}: {s['n_turns']} 次展开, prompt={s['prompt_tokens']}, "
                f"cached={s['cached_tokens']} ({s['cache_source']}), "
                f"hit_rate={s['hit_rate']:.1%}"
            )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
