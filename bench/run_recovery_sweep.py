"""Run the selection/recovery parameter matrix as separate reports."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_case(args, conditions: str, ratio: float, rounds: int, name: str) -> None:
    output = Path(args.out_dir) / f"{name}.json"
    command = [
        sys.executable,
        "bench/eval_recovery.py",
        "--backend",
        args.backend,
        "--datasets",
        args.datasets,
        "--limit",
        str(args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--selection-ratio",
        str(ratio),
        "--max-require-rounds",
        str(rounds),
        "--seed",
        str(args.seed),
        "--conditions",
        conditions,
        "--out",
        str(output),
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    print(f"running {name}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["local", "api", "scripted"], default="scripted"
    )
    parser.add_argument(
        "--datasets",
        default="multi_news_e,2wikimqa_e,hotpotqa_e,triviaqa_e,qasper_e",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--ratios", default="0.25,0.5,0.75")
    parser.add_argument("--rounds", default="1,2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out-dir", default="bench/results/recovery-sweep")
    args = parser.parse_args()

    ratios = [float(value) for value in args.ratios.split(",")]
    rounds = [int(value) for value in args.rounds.split(",")]
    if any(not 0 <= value <= 1 for value in ratios):
        parser.error("--ratios values must be between 0 and 1")
    if any(value < 1 for value in rounds):
        parser.error("--rounds values must be positive")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_case(args, "full,protocol-full,fixed-half", 0.5, 0, "baselines")
    for ratio in ratios:
        label = str(int(ratio * 100))
        run_case(
            args,
            "keyword-initial",
            ratio,
            0,
            f"keyword-initial-r{label}",
        )
        for max_rounds in rounds:
            run_case(
                args,
                "keyword-recovery",
                ratio,
                max_rounds,
                f"keyword-recovery-r{label}-m{max_rounds}",
            )


if __name__ == "__main__":
    main()
