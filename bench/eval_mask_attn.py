"""Pretrained-model DAG mask and branch-cache experiment (GPU required)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contextdag import MaskAttn, Session


def summarize(rows: list[dict]) -> dict:
    """Aggregate observed samples without inferring speedup or equivalence."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    conditions = tuple(rows[0]["outputs"])
    total_tokens = sum(r["input_tokens"] for r in rows)
    result = {
        "n": n,
        "exact_match": {c: sum(r["exact_match"][c] for r in rows) / n for c in conditions},
        "reuse_next_token_agreement": sum(r["reuse_vs_full"]["next_token_equal"] for r in rows) / n,
        "reuse_output_agreement": sum(r["outputs"]["dag_full"] == r["outputs"]["dag_reuse"] for r in rows) / n,
        "reuse_max_logit_error": max(r["reuse_vs_full"]["max_logit_error"] for r in rows),
        "prefix_control_max_logit_error": max(r["same_full_prefix_vs_full"]["max_logit_error"] for r in rows),
        "input_tokens": total_tokens,
        "warm_query_tokens": sum(r["new_query_tokens"] for r in rows),
        "cached_token_fraction": sum(r["cached_prefix_tokens"] for r in rows) / total_tokens,
        "mean_seconds": {k: sum(r[k] for r in rows) / n for k in (
            "dag_full_seconds", "causal_seconds", "cache_build_seconds",
            "cache_assembly_seconds", "warm_query_seconds")},
    }
    if all("f1" in r for r in rows):
        result["mean_f1"] = {c: sum(r["f1"][c] for r in rows) / n for c in conditions}
    return result


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--data", help="Optional LongBench JSONL; synthetic probes otherwise")
    parser.add_argument("--context-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.samples < 1 or args.max_new_tokens < 1 or args.context_tokens < 2:
        parser.error("samples and max-new-tokens must be positive")
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, device_map="cuda", torch_dtype=torch.float16,
        attn_implementation="eager",
    ).eval()
    report = {
        "experiment": "maskattn_pretrained_branch_reuse",
        "scope": "synthetic correctness and fact-lookup probe; not LongBench or serving throughput",
        "model": args.model, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "dtype": str(model.dtype), "attention": "eager", "seed": 0,
        "observations": [],
        "config": vars(args),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/contextdag/mask_attn.py").read_bytes()
        ).hexdigest(),
        "model_config": model.config.to_dict(),
    }
    if args.data:
        report["scope"] = "LongBench truncated-context subset; two default branches, not annotated semantic dependencies"
        with open(args.data) as handle:
            records = [json.loads(line) for line in handle][:args.samples]
    else:
        records = [None] * args.samples
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(target)

    def forward(ids, positions, mask, past=None):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=torch.tensor([ids], device="cuda"),
                position_ids=torch.tensor([positions], device="cuda"),
                attention_mask=mask, past_key_values=past, use_cache=True,
            )
        torch.cuda.synchronize()
        return out, time.perf_counter() - started

    def clone_cache(parts):
        return DynamicCache.from_legacy_cache(tuple(
            (torch.cat([p[layer][0] for p in parts], dim=2),
             torch.cat([p[layer][1] for p in parts], dim=2))
            for layer in range(len(parts[0]))))

    def extract(cache, start, end):
        return tuple((k[:, :, start:end].clone(), v[:, :, start:end].clone())
                     for k, v in cache.to_legacy_cache())

    def decode(out, next_position):
        generated = []
        cache = out.past_key_values
        logits = out.logits[:, -1]
        for _ in range(args.max_new_tokens):
            token = int(logits.argmax(-1))
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            mask = torch.zeros((1, 1, 1, cache.get_seq_length() + 1),
                               device="cuda", dtype=model.dtype)
            out, _ = forward([token], [next_position], mask, cache)
            next_position += 1
            cache, logits = out.past_key_values, out.logits[:, -1]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def metrics(a, b):
        a, b = a.float(), b.float()
        delta = (a-b).abs()
        return {"max_logit_error": float(delta.max()),
                "mean_logit_error": float(delta.mean()),
                "next_token_equal": bool(a.argmax(-1).eq(b.argmax(-1)).all())}

    colors = ["amber", "violet", "silver", "green", "orange", "blue"]
    for sample, record in enumerate(records):
        s = Session()
        root = s.register("<|im_start|>system\nAnswer the question using the records. "
                          "Return only the requested code.\n<|im_end|>\n<|im_start|>user\nRecords:\n")
        code = colors[sample % len(colors)]
        left = s.register(f"Record A: Item {sample} has code {code}.\n", refs=[root.id])
        right = s.register(f"Record B: Item {sample+100} has code black. "
                           + "This is a separate record. " * (1 + sample % 3) + "\n", refs=[root.id])
        answer = s.register(f"Question: What is the code of Item {sample}?\n"
                            "<|im_end|>\n<|im_start|>assistant\n", refs=[left.id, right.id])
        order = [root.id, right.id, left.id, answer.id]
        tokens = {nid: tokenizer.encode(s.read(nid), add_special_tokens=False) for nid in order}
        if record is not None:
            s = Session()
            root = s.register("<|im_start|>system\nAnswer using the records. Return only a short answer."
                              "\n<|im_end|>\n<|im_start|>user\nRecords:\n")
            context_ids = tokenizer.encode(record["context"], add_special_tokens=False)
            retained = context_ids[:args.context_tokens]
            middle = len(retained) // 2
            # Preserve identical token blocks across every condition.
            right = s.register(tokenizer.decode(retained[:middle]), refs=[root.id])
            left = s.register(tokenizer.decode(retained[middle:]), refs=[root.id])
            answer = s.register("\nQuestion: " + record["input"] +
                                "\n<|im_end|>\n<|im_start|>assistant\n", refs=[right.id, left.id])
            order = [root.id, right.id, left.id, answer.id]
            tokens = {nid: tokenizer.encode(s.read(nid), add_special_tokens=False) for nid in order}
            tokens[right.id], tokens[left.id] = retained[:middle], retained[middle:]
            code = record["answers"]
        plan = MaskAttn.compile(s.registry, order, tokens)
        mask = torch.tensor(plan.dense_mask(additive=True), device="cuda", dtype=model.dtype)[None, None]
        lengths = [len(tokens[n]) for n in order]
        root_end, right_end, left_end = lengths[0], sum(lengths[:2]), sum(lengths[:3])
        # Warm the execution path; each measured full condition creates fresh KV.
        if sample == 0:
            warm, _ = forward(plan.input_ids, plan.position_ids, mask)
            del warm
        full, full_seconds = forward(plan.input_ids, plan.position_ids, mask)
        full_logits = full.logits[:, -1].clone()
        repeat, repeat_seconds = forward(plan.input_ids, plan.position_ids, mask)
        repeat_metrics = metrics(full_logits, repeat.logits[:, -1])
        del repeat
        full_kv = extract(full.past_key_values, 0, len(plan.input_ids))
        # Prefix control: reuse KVs produced in the SAME full DAG computation.
        prefix = tuple((k[:, :, :left_end].clone(), v[:, :, :left_end].clone())
                       for k, v in full_kv)
        ordinary, ordinary_seconds = forward(
            tokens[answer.id], plan.position_ids[left_end:], mask[:, :, left_end:, :],
            clone_cache([prefix]))
        prefix_metrics = metrics(full_logits, ordinary.logits[:, -1])
        del ordinary, prefix
        full_text = decode(full, plan.position_ids[-1] + 1)
        del full
        # Build root and each branch independently. Left never sees right.
        root_out, root_seconds = forward(tokens[root.id], plan.position_ids[:root_end],
                                        mask[:, :, :root_end, :root_end])
        root_kv = extract(root_out.past_key_values, 0, root_end)
        del root_out
        branch_parts = {}
        branch_seconds = 0.0
        for nid in [right.id, left.id]:
            local = MaskAttn.compile(s.registry, [root.id, nid], tokens)
            local_mask = torch.tensor(local.dense_mask(additive=True), device="cuda", dtype=model.dtype)[None, None]
            out, seconds = forward(tokens[nid], local.position_ids[root_end:],
                                   local_mask[:, :, root_end:, :], clone_cache([root_kv]))
            branch_parts[nid] = extract(out.past_key_values, root_end, len(local.input_ids))
            branch_seconds += seconds
            del out
        # Measure assembly separately from the forward; no server/TTFT claim.
        torch.cuda.synchronize()
        started = time.perf_counter()
        assembled = clone_cache([root_kv, branch_parts[right.id], branch_parts[left.id]])
        torch.cuda.synchronize()
        assembly_seconds = time.perf_counter() - started
        reused, reuse_seconds = forward(tokens[answer.id], plan.position_ids[left_end:],
                                        mask[:, :, left_end:, :], assembled)
        reuse_logits = reused.logits[:, -1].clone()
        reuse_kv = extract(reused.past_key_values, 0, len(plan.input_ids))
        kv_error = max(float((x.float()-y.float()).abs().max())
                       for p, q in zip(full_kv, reuse_kv) for x, y in zip(p, q))
        reuse_text = decode(reused, plan.position_ids[-1] + 1)
        del reused, assembled, reuse_kv, root_kv, branch_parts
        global_pos = tuple(range(len(plan.input_ids)))
        global_out, global_seconds = forward(plan.input_ids, global_pos, mask)
        global_metrics = metrics(full_logits, global_out.logits[:, -1])
        global_text = decode(global_out, len(plan.input_ids))
        del global_out
        causal_mask = torch.full_like(mask, float("-inf"))
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal, causal_seconds = forward(plan.input_ids, global_pos, causal_mask)
        causal_metrics = metrics(full_logits, causal.logits[:, -1])
        causal_text = decode(causal, len(plan.input_ids))
        del causal, full_kv
        row = {
            "sample": sample, "expected": code, "input_tokens": len(plan.input_ids),
            "cached_prefix_tokens": left_end, "new_query_tokens": lengths[-1],
            "dag_full_seconds": full_seconds, "cache_build_seconds": root_seconds + branch_seconds,
            "cache_assembly_seconds": assembly_seconds, "warm_query_seconds": reuse_seconds,
            "dag_global_seconds": global_seconds, "causal_seconds": causal_seconds,
            "reuse_vs_full": metrics(full_logits, reuse_logits), "kv_max_error": kv_error,
            "same_full_prefix_vs_full": prefix_metrics,
            "same_full_prefix_query_seconds": ordinary_seconds,
            "repeated_full_vs_full": repeat_metrics,
            "repeated_full_seconds": repeat_seconds,
            "global_vs_dag": global_metrics, "causal_vs_dag": causal_metrics,
            "outputs": {"dag_full": full_text, "dag_reuse": reuse_text,
                        "dag_global": global_text, "causal": causal_text},
            "blocks": [{"id": n, "refs": list(s.registry.get(n).refs),
                        "text": s.read(n), "tokens": tokens[n]} for n in order],
            "position_ids": list(plan.position_ids),
        }
        expected = code if isinstance(code, list) else [code]
        row["exact_match"] = {k: v.strip().lower() in [a.strip().lower() for a in expected]
                              for k, v in row["outputs"].items()}
        if record is not None:
            from eval_metrics import f1_score
            row["sample_id"] = record.get("_id")
            row["original_context_tokens"] = len(context_ids)
            row["retained_context_tokens"] = len(retained)
            row["f1"] = {k: max(f1_score(v, a) for a in expected)
                         for k, v in row["outputs"].items()}
        report["observations"].append(row)
        report["aggregates"] = summarize(report["observations"])
        save()
        print(json.dumps({k: row[k] for k in ["sample", "reuse_vs_full", "kv_max_error", "exact_match"]}), flush=True)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save()


if __name__ == "__main__":
    main()
