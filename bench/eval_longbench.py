"""Level-1 correctness: does the protocol rendering preserve answer quality?

Same LongBench questions answered under three context conditions:
  full     - raw document as-is (vanilla baseline)
  protocol - document split into section nodes, rendered as a dependency set
  pruned   - dependency set restricted to the first half of sections
  no-headers - sections joined without node headers (isolates header cost)

Metrics follow LongBench conventions (F1 / ROUGE-L / accuracy per dataset).

Usage:
  PYTHONPATH=src .venv/bin/python bench/eval_longbench.py --backend scripted --limit 2
  PYTHONPATH=src .venv/bin/python bench/eval_longbench.py --backend local --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from contextdag import Session  # noqa: E402

from adapters.longbench import LONGBENCH_PATH, split_sections  # noqa: E402
from eval_metrics import score_prediction, strip_think  # noqa: E402
from measurement import mean as arithmetic_mean, new_report, save_report  # noqa: E402
from model_backends import load_tokenize, make_text_model  # noqa: E402


def truncate(text: str, tokenize, decode, max_tokens: int) -> str:
    ids = tokenize(text)
    if len(ids) <= max_tokens:
        return text
    return decode(ids[:max_tokens])


def question_for(rec: dict) -> str:
    return rec.get("input") or rec.get("question", "")


def document_for(rec: dict, tokenize, decode, max_tokens: int) -> str:
    """Keep the question and instruction outside document truncation."""
    suffix = "\n\n" + question_for(rec) + "\n\n" + instruction_for(rec)
    document_budget = max(0, max_tokens - len(tokenize(suffix)))
    normalized = rec["context"].replace("NEWLINE_CHAR", "\n")
    return truncate(normalized, tokenize, decode, document_budget)


def protocol_context(rec: dict, tokenize, decode, max_tokens: int, prune: bool = False):
    """Document split into nodes; question node references the sections."""
    session = Session()
    document = document_for(rec, tokenize, decode, max_tokens)
    sections = split_sections(document, 8)
    if prune:
        sections = sections[: max(1, len(sections) // 2)]
    section_nodes = [session.register(content=section) for section in sections]
    question = session.register(
        content=question_for(rec),
        refs=[n.id for n in section_nodes],
    )
    ctx = session.expand(refs=[question.id])
    return ctx.text


def instruction_for(rec: dict) -> str:
    """Match the answer instruction to the record's language so the model
    doesn't default to Chinese for English tasks."""
    text = (rec.get("input") or "") + rec["context"][:200]
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if has_cjk:
        return "请直接给出答案，不要输出思考过程。"
    return "Answer directly in English, without any thinking process."


RETRY_INSTRUCTION = "只输出最终答案本身，不要任何解释、思考或前后缀。"


def no_headers_context(rec: dict, tokenize, decode, max_tokens: int) -> str:
    """Sections joined as plain text (same sectioning as protocol, no node
    headers), to separate 'sectioning' from 'node headers'."""
    document = document_for(rec, tokenize, decode, max_tokens)
    body = "\n\n".join(split_sections(document, 8))
    return body + "\n\n" + question_for(rec)


def run_condition(
    name: str,
    build_prompt,
    records,
    model,
    tokenize,
    decode,
    max_tokens,
    on_record=None,
):
    scores = []
    outputs = []
    n = len(records)
    t0 = time.time()
    for i, rec in enumerate(records):
        prompt = build_prompt(rec, tokenize, decode, max_tokens)
        prompt_tokens = len(tokenize(prompt))
        raw = model(prompt)
        prediction = strip_think(raw)
        retries = 0
        while not prediction.strip() and retries < 2:
            raw = model(prompt + "\n\n" + RETRY_INSTRUCTION)
            prediction = strip_think(raw)
            retries += 1
        answers = rec["answers"] if isinstance(rec["answers"], list) else [rec["answers"]]
        score = score_prediction(rec["dataset"], prediction, answers)
        scores.append(score)
        outputs.append(
            {
                "_id": rec.get("_id"),
                "dataset": rec["dataset"],
                "score": score,
                "prediction": prediction[:200],
                "raw_prediction": raw[:500],
                "retries": retries,
                "prompt_tokens": prompt_tokens,
            }
        )
        if on_record is not None:
            on_record(name, i, n, score, time.time() - t0, scores, outputs)
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, scores, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["local", "api", "scripted"], default="scripted"
    )
    parser.add_argument(
        "--datasets",
        default="multi_news_e,2wikimqa_e,hotpotqa_e,triviaqa_e,qasper_e",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--out", default="bench/results/eval_longbench.json")
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    tokenize, decode = load_tokenize(args.tokenizer)
    model = make_text_model(args.backend, max_tokens=args.max_output_tokens)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    records = []
    for name in datasets:
        with open(LONGBENCH_PATH.format(dataset=name)) as f:
            for line in f:
                records.append(json.loads(line))
                if len([r for r in records if r["dataset"] == name]) >= args.limit:
                    break

    report = new_report(
        "longbench_quality",
        args.backend,
        {
            "datasets": datasets,
            "limit_per_dataset": args.limit,
            "max_prompt_tokens": args.max_tokens,
            "max_output_tokens": args.max_output_tokens,
            "tokenizer": args.tokenizer,
            "conditions": ["full", "protocol", "pruned", "no-headers"],
        },
        model=os.environ.get("LOCAL_MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or ("scripted" if args.backend == "scripted" else None),
    )
    conditions = {
        "full": lambda rec, t, d, m: document_for(rec, t, d, m)
        + "\n\n"
        + question_for(rec)
        + "\n\n"
        + instruction_for(rec),
        "protocol": lambda rec, t, d, m: protocol_context(rec, t, d, m)
        + "\n\n"
        + instruction_for(rec),
        "pruned": lambda rec, t, d, m: protocol_context(rec, t, d, m, prune=True)
        + "\n\n"
        + instruction_for(rec),
        "no-headers": lambda rec, t, d, m: no_headers_context(rec, t, d, m)
        + "\n\n"
        + instruction_for(rec),
    }
    completed = []
    for cond, builder in conditions.items():
        def on_record(cond, i, n, score, elapsed, scores, outputs):
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f"[{time.strftime('%H:%M:%S')}] {cond:>8} "
                f"record {i + 1}/{n} score={score:.4f} "
                f"elapsed={elapsed:.0f}s eta~{eta:.0f}s",
                flush=True,
            )
            report["aggregates"][cond] = {
                "mean_score": arithmetic_mean(scores),
                "n": len(scores),
                "prompt_tokens": sum(row["prompt_tokens"] for row in outputs),
            }
            report["observations"] = completed + [
                {"condition": cond, **row} for row in outputs
            ]
            _save()

        def _save():
            save_report(report, args.out)

        mean, scores, outputs = run_condition(
            cond,
            builder,
            records,
            model,
            tokenize,
            decode,
            args.max_tokens,
            on_record=on_record,
        )
        condition_rows = [{"condition": cond, **row} for row in outputs]
        completed.extend(condition_rows)
        report["observations"] = list(completed)
        report["aggregates"][cond] = {
            "mean_score": mean,
            "n": len(scores),
            "prompt_tokens": sum(row["prompt_tokens"] for row in outputs),
        }
        _save()
        print(f"[{cond}] n={len(scores)} mean={mean:.4f}")

    save_report(report, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
