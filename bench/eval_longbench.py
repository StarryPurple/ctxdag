"""Level-1 correctness: does the protocol rendering preserve answer quality?

Same LongBench questions answered under three context conditions:
  full     - raw document as-is (vanilla baseline)
  protocol - document split into section nodes, rendered as a dependency set
  pruned   - dependency set restricted to the first half of sections

Metrics follow LongBench conventions (F1 / ROUGE-L / accuracy per dataset).

Usage:
  PYTHONPATH=src .venv/bin/python bench/eval_longbench.py --backend scripted --limit 2
  PYTHONPATH=src .venv/bin/python bench/eval_longbench.py --backend local --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from contextdag import Session  # noqa: E402

from adapters.longbench import LONGBENCH_PATH, split_sections  # noqa: E402
from eval_metrics import score_prediction, strip_think  # noqa: E402
from model_backends import load_tokenize, make_text_model  # noqa: E402


def truncate(text: str, tokenize, decode, max_tokens: int) -> str:
    ids = tokenize(text)
    if len(ids) <= max_tokens:
        return text
    return decode(ids[:max_tokens])


def protocol_context(rec: dict, tokenize, decode, max_tokens: int, prune: bool = False):
    """Document split into nodes; question node references the sections."""
    session = Session()
    sections = split_sections(rec["context"], 8)
    if prune:
        sections = sections[: max(1, len(sections) // 2)]
    section_nodes = [
        session.register(content=truncate(sec, tokenize, decode, max_tokens))
        for sec in sections
    ]
    question = session.register(
        content=rec.get("input") or rec.get("question", ""),
        refs=[n.id for n in section_nodes],
    )
    ctx = session.expand(refs=[question.id])
    return ctx.text


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
        prediction = strip_think(model(prompt))
        answers = rec["answers"] if isinstance(rec["answers"], list) else [rec["answers"]]
        score = score_prediction(rec["dataset"], prediction, answers)
        scores.append(score)
        outputs.append(
            {
                "_id": rec.get("_id"),
                "dataset": rec["dataset"],
                "score": score,
                "prediction": prediction[:200],
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
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--out", default="tmp/eval_longbench.json")
    parser.add_argument("--tokenizer", default="data/models/qwen3-4b-awq/tokenizer.json")
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

    Path("tmp").mkdir(exist_ok=True)
    report = {"backend": args.backend, "conditions": {}}
    ANSWER_INSTRUCTION = "请直接给出答案，不要输出任何思考过程。"
    conditions = {
        "full": lambda rec, t, d, m: truncate(
            rec["context"] + "\n\n" + (rec.get("input") or "") + "\n\n" + ANSWER_INSTRUCTION,
            t,
            d,
            m,
        ),
        "protocol": lambda rec, t, d, m: protocol_context(rec, t, d, m)
        + "\n\n"
        + ANSWER_INSTRUCTION,
        "pruned": lambda rec, t, d, m: protocol_context(rec, t, d, m, prune=True)
        + "\n\n"
        + ANSWER_INSTRUCTION,
    }
    for cond, builder in conditions.items():
        def on_record(cond, i, n, score, elapsed, scores, outputs):
            eta = elapsed / (i + 1) * (n - i - 1)
            print(
                f"[{time.strftime('%H:%M:%S')}] {cond:>8} "
                f"record {i + 1}/{n} score={score:.4f} "
                f"elapsed={elapsed:.0f}s eta~{eta:.0f}s",
                flush=True,
            )
            report["conditions"][cond] = {
                "mean": sum(scores) / len(scores),
                "n": len(scores),
                "scores": scores,
                "outputs": outputs,
            }
            _save()

        def _save():
            with open(args.out, "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

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
        report["conditions"][cond] = {
            "mean": mean,
            "n": len(scores),
            "scores": scores,
            "outputs": outputs,
        }
        _save()
        print(f"[{cond}] n={len(scores)} mean={mean:.4f}")

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
