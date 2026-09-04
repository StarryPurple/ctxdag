"""Evaluate bounded node selection and protocol-level require recovery."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from time import perf_counter

sys.path.insert(0, "src")
sys.path.insert(0, "bench")

from contextdag import (  # noqa: E402
    KeywordSelector,
    SelectionCandidate,
    Session,
    render_node,
    summary_of,
)

from adapters.longbench import LONGBENCH_PATH, split_sections  # noqa: E402
from eval_longbench import document_for, instruction_for, question_for  # noqa: E402
from eval_metrics import score_prediction, strip_think  # noqa: E402
from measurement import mean, new_report, save_report, session_counters  # noqa: E402
from model_backends import (
    load_tokenize,
    make_agent_model,
    make_chat_model,
    make_text_model,
)  # noqa: E402
from recovery import (
    RecoveryResult,
    run_message_recovery,
    run_recovery,
    run_tool_recovery,
)  # noqa: E402

def tool_instruction(session: Session, policy: str = "balanced") -> str:
    """Describe the control decision without exposing tag syntax."""
    if not session.current_context.catalog:
        return ""
    if policy == "strict":
        return (
            "\n\nYou must make one control tool call. Catalog summaries are "
            "routing hints, not answer evidence. Call return_answer only if "
            "the full visible nodes explicitly contain every fact needed for "
            "the answer. Otherwise, if any catalog entry could contain a "
            "needed fact or a linking fact, call require_context with its "
            "authorized node id before answering."
        )
    return (
        "\n\nCatalog summaries are routing hints, not answer evidence. "
        "Do not answer from a summary alone. If a catalog entry may contain "
        "evidence needed for the question that is absent from full visible "
        "nodes, call require_context with its authorized node id. Answer "
        "directly only when the full visible nodes contain sufficient evidence."
    )


def require_instruction(session: Session) -> str:
    """List only tags that are legal in the current catalog."""
    choices = " ".join(
        f"<require={node_id}>" for node_id in session.current_context.catalog
    )
    if not choices:
        return ""
    return (
        "\n\nAnswer directly from the visible nodes. If evidence is missing, choose "
        "the matching catalog summary and output exactly one of these complete "
        f"tags, with no other text: {choices}"
    )


CONDITIONS = (
    "full",
    "protocol-full",
    "fixed-half",
    "keyword-initial",
    "keyword-recovery",
)


def build_session(
    rec: dict,
    tokenize,
    decode,
    max_tokens: int,
    condition: str,
    selection_ratio: float,
    search_source: str,
):
    """Build one condition while keeping all document sections addressable."""

    session = Session(catalog_size=0)
    document = document_for(rec, tokenize, decode, max_tokens)
    sections = split_sections(document, 8)
    nodes = [session.register(section) for section in sections]
    catalog = []
    for position, (node, section) in enumerate(zip(nodes, sections)):
        summary = summary_of(section, max_chars=240)
        session.summaries.set(node.id, summary, source="service")
        catalog.append(
            SelectionCandidate(
                node_id=node.id,
                summary=summary,
                token_count=len(tokenize(render_node(node))),
                position=position,
                search_text=section if search_source == "fulltext" else summary,
            )
        )

    if condition == "protocol-full":
        selected = [node.id for node in nodes]
    elif condition == "fixed-half":
        selected = [node.id for node in nodes[: max(1, len(nodes) // 2)]]
    else:
        budget = int(sum(item.token_count for item in catalog) * selection_ratio)
        selected = KeywordSelector().select(question_for(rec), catalog, budget)

    question = session.register(question_for(rec), refs=selected)
    remaining = [
        item.node_id for item in catalog if item.node_id not in set(selected)
    ]
    session.expand(
        refs=[question.id],
        candidates=remaining if condition.startswith("keyword-") else (),
    )
    return session, selected


def direct_result(prompt: str, model, tokenize) -> RecoveryResult:
    started = perf_counter()
    output = model(prompt)
    latency_seconds = perf_counter() - started
    tokens = len(tokenize(prompt))
    usage = getattr(model, "last_usage", {}) or {}
    return RecoveryResult(
        initial_output=output,
        final_output=output,
        initial_prompt_tokens=tokens,
        recovery_prompt_tokens=0,
        total_prompt_tokens=tokens,
        require_rounds=0,
        requested_node_ids=(),
        required_node_ids=(),
        incremental_node_tokens=0,
        calls=(
            {
                "logical_prompt_tokens": tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
            },
        ),
        failure_type=None,
    )


def direct_message_result(prompt: str, model, tokenize) -> RecoveryResult:
    started = perf_counter()
    output = model([{"role": "user", "content": prompt}])
    latency_seconds = perf_counter() - started
    tokens = len(tokenize(prompt))
    usage = getattr(model, "last_usage", {}) or {}
    return RecoveryResult(
        initial_output=output,
        final_output=output,
        initial_prompt_tokens=tokens,
        recovery_prompt_tokens=0,
        total_prompt_tokens=tokens,
        require_rounds=0,
        requested_node_ids=(),
        required_node_ids=(),
        incremental_node_tokens=0,
        calls=(
            {
                "logical_prompt_tokens": tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
                "message_count": 1,
            },
        ),
        failure_type=None,
    )


def direct_tool_result(prompt: str, model, tokenize) -> RecoveryResult:
    started = perf_counter()
    message = model([{"role": "user", "content": prompt}], [])
    latency_seconds = perf_counter() - started
    output = str(message.get("content") or "")
    tokens = len(tokenize(prompt))
    usage = getattr(model, "last_usage", {}) or {}
    return RecoveryResult(
        initial_output=output,
        final_output=output,
        initial_prompt_tokens=tokens,
        recovery_prompt_tokens=0,
        total_prompt_tokens=tokens,
        require_rounds=0,
        requested_node_ids=(),
        required_node_ids=(),
        incremental_node_tokens=0,
        calls=(
            {
                "logical_prompt_tokens": tokens,
                "api_prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "latency_seconds": latency_seconds,
                "message_count": 1,
            },
        ),
        failure_type=None,
    )


def evaluate_record(
    condition: str,
    rec: dict,
    model,
    tokenize,
    decode,
    max_tokens: int,
    selection_ratio: float,
    max_require_rounds: int,
    search_source: str = "summary",
    transport: str = "flattened",
    tool_policy: str = "balanced",
) -> dict:
    instruction = "\n\n" + instruction_for(rec)
    if condition == "full":
        document = document_for(rec, tokenize, decode, max_tokens)
        prompt = document + "\n\n" + question_for(rec) + instruction
        if transport == "tools":
            result = direct_tool_result(prompt, model, tokenize)
        elif transport == "messages":
            result = direct_message_result(prompt, model, tokenize)
        else:
            result = direct_result(prompt, model, tokenize)
        selected_node_ids = []
        summaries = {}
        counters = {}
    else:
        session, selected_node_ids = build_session(
            rec,
            tokenize,
            decode,
            max_tokens,
            condition,
            selection_ratio,
            search_source,
        )
        suffix = instruction
        if condition.startswith("keyword-"):
            if transport == "tools":
                suffix = lambda: tool_instruction(session, tool_policy) + instruction
            else:
                suffix = lambda: require_instruction(session) + instruction
        if transport == "tools":
            result = run_tool_recovery(
                session,
                model,
                tokenize,
                suffix,
                max_require_rounds=max_require_rounds,
                allow_recovery=condition == "keyword-recovery",
            )
        elif transport == "messages":
            result = run_message_recovery(
                session,
                model,
                tokenize,
                suffix,
                lambda: (
                    "\n\nOriginal question:\n"
                    + question_for(rec)
                    + require_instruction(session)
                    + instruction
                ),
                max_require_rounds=max_require_rounds,
                allow_recovery=condition == "keyword-recovery",
            )
        else:
            result = run_recovery(
                session,
                model,
                tokenize,
                suffix,
                max_require_rounds=max_require_rounds,
                allow_recovery=condition == "keyword-recovery",
            )
        summaries = session.summaries.entries()
        counters = session_counters(session)

    initial_prediction = strip_think(result.initial_output)
    final_prediction = strip_think(result.final_output)
    answers = rec["answers"] if isinstance(rec["answers"], list) else [rec["answers"]]
    initial_score = score_prediction(rec["dataset"], initial_prediction, answers)
    final_score = score_prediction(rec["dataset"], final_prediction, answers)
    return {
        "_id": rec.get("_id"),
        "dataset": rec["dataset"],
        "transport": transport,
        "initial_node_ids": selected_node_ids,
        "initial_node_summaries": {
            node_id: summaries.get(node_id) for node_id in selected_node_ids
        },
        "requested_node_ids": list(result.requested_node_ids),
        "requested_node_summaries": {
            node_id: summaries.get(node_id) for node_id in result.requested_node_ids
        },
        "required_node_ids": list(result.required_node_ids),
        "require_rounds": result.require_rounds,
        "initial_prompt_tokens": result.initial_prompt_tokens,
        "recovery_prompt_tokens": result.recovery_prompt_tokens,
        "total_prompt_tokens": result.total_prompt_tokens,
        "incremental_node_tokens": result.incremental_node_tokens,
        "calls": list(result.calls),
        "initial_score": initial_score,
        "final_score": final_score,
        "initial_prediction": initial_prediction[:200],
        "final_prediction": final_prediction[:200],
        "failure_type": result.failure_type,
        "session": counters,
    }


def sum_reported(rows: list[dict], key: str):
    values = [
        call[key]
        for row in rows
        for call in row["calls"]
        if call.get(key) is not None
    ]
    return sum(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_initial_score": mean(row["initial_score"] for row in rows),
        "mean_final_score": mean(row["final_score"] for row in rows),
        "initial_prompt_tokens": sum(row["initial_prompt_tokens"] for row in rows),
        "recovery_prompt_tokens": sum(row["recovery_prompt_tokens"] for row in rows),
        "total_prompt_tokens": sum(row["total_prompt_tokens"] for row in rows),
        "api_prompt_tokens": sum_reported(rows, "api_prompt_tokens"),
        "cached_tokens": sum_reported(rows, "cached_tokens"),
        "completion_tokens": sum_reported(rows, "completion_tokens"),
        "incremental_node_tokens": sum(
            row["incremental_node_tokens"] for row in rows
        ),
        "total_latency_seconds": sum(
            call["latency_seconds"] for row in rows for call in row["calls"]
        ),
        "require_attempt_rate": mean(
            bool(row["requested_node_ids"]) for row in rows
        ),
        "require_execution_rate": mean(
            row["require_rounds"] > 0 for row in rows
        ),
        "recovery_success_rate": mean(
            row["final_score"] > row["initial_score"]
            for row in rows
            if row["require_rounds"] > 0
        ),
        "failure_types": {
            failure: sum(row["failure_type"] == failure for row in rows)
            for failure in sorted(
                {row["failure_type"] for row in rows if row["failure_type"]}
            )
        },
    }


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
    parser.add_argument("--selection-ratio", type=float, default=0.5)
    parser.add_argument("--max-require-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument(
        "--search-source", choices=["summary", "fulltext"], default="summary"
    )
    parser.add_argument(
        "--transport", choices=["flattened", "messages", "tools"], default="flattened"
    )
    parser.add_argument(
        "--tool-policy", choices=["balanced", "strict"], default="balanced"
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out", default="bench/results/eval_recovery.json")
    args = parser.parse_args()
    if not 0 <= args.selection_ratio <= 1:
        parser.error("--selection-ratio must be between 0 and 1")
    selected_conditions = [
        item.strip() for item in args.conditions.split(",") if item.strip()
    ]
    unknown = set(selected_conditions) - set(CONDITIONS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")

    tokenize, decode = load_tokenize(args.tokenizer)
    if args.transport == "tools":
        make_model = make_agent_model
    elif args.transport == "messages":
        make_model = make_chat_model
    else:
        make_model = make_text_model
    model = make_model(
        args.backend, max_tokens=args.max_output_tokens, seed=args.seed
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    records = []
    for dataset in datasets:
        with open(LONGBENCH_PATH.format(dataset=dataset)) as handle:
            for line in handle:
                records.append(json.loads(line))
                if sum(row["dataset"] == dataset for row in records) >= args.limit:
                    break

    report = new_report(
        "selection_recovery",
        args.backend,
        {
            "datasets": datasets,
            "limit_per_dataset": args.limit,
            "max_prompt_tokens": args.max_tokens,
            "max_output_tokens": args.max_output_tokens,
            "selection_ratio": args.selection_ratio,
            "search_source": args.search_source,
            "transport": args.transport,
            "tool_policy": args.tool_policy,
            "max_require_rounds": args.max_require_rounds,
            "tokenizer": args.tokenizer,
            "conditions": selected_conditions,
            "temperature": 0.0,
            "seed": args.seed,
        },
        model=os.environ.get("LOCAL_MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or ("scripted" if args.backend == "scripted" else None),
    )

    for condition in selected_conditions:
        rows = []
        started = time.time()
        for index, rec in enumerate(records):
            row = evaluate_record(
                condition,
                rec,
                model,
                tokenize,
                decode,
                args.max_tokens,
                args.selection_ratio,
                args.max_require_rounds,
                args.search_source,
                args.transport,
                args.tool_policy,
            )
            rows.append(row)
            report["observations"].append({"condition": condition, **row})
            report["aggregates"][condition] = aggregate(rows)
            save_report(report, args.out)
            print(
                f"[{condition}] {index + 1}/{len(records)} "
                f"score={row['final_score']:.4f} elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
