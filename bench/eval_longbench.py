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
    sections = split_sections(rec["context"], 8)
    body = "\n\n".join(
        truncate(sec, tokenize, decode, max_tokens) for sec in sections
    )
    return truncate(
        body + "\n\n" + (rec.get("input") or ""),
        tokenize,
        decode,
        max_tokens,
    )


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
    parser.add_argument("--out", default="results/eval_longbench.json")
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = {"backend": args.backend, "conditions": {}}
    conditions = {
        "full": lambda rec, t, d, m: truncate(
            rec["context"]
            + "\n\n"
            + (rec.get("input") or "")
            + "\n\n"
            + instruction_for(rec),
            t,
            d,
            m,
        ),
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
