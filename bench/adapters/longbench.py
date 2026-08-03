"""LongBench long documents with progressive (overlapping) readers.

A document is split into sections registered as root nodes; readers then
read cumulatively (reader i reads sections 0..i), so every later context
shares the full earlier prefix -- the classic long-background reuse shape.
"""

from __future__ import annotations

import json
import re

LONGBENCH_PATH = "data/longbench/data/{dataset}.jsonl"
DEFAULT_DATASETS = [
    "multi_news_e",
    "2wikimqa_e",
    "hotpotqa_e",
    "triviaqa_e",
    "gov_report_e",
    "qasper_e",
    "multifieldqa_en_e",
    "passage_count_e",
    "trec_e",
    "samsum_e",
]


def iter_samples(
    dataset: str | None = None,
    limit: int = 5,
    max_sections: int = 8,
    datasets=None,
):
    """Yield progressive-reading builders; all default datasets when
    ``dataset`` is omitted."""
    names = [dataset] if dataset else (datasets or DEFAULT_DATASETS)
    for name in names:
        with open(LONGBENCH_PATH.format(dataset=name)) as f:
            records = [json.loads(line) for line in f][:limit]
        for rec in records:
            yield build_from_document(split_sections(rec["context"], max_sections))


def split_sections(context: str, max_sections: int) -> list[str]:
    normalized = context.replace("NEWLINE_CHAR", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if len(parts) <= max_sections:
        return parts or [context]
    # merge the tail so we stay within max_sections
    merged = parts[: max_sections - 1]
    merged.append("\n\n".join(parts[max_sections - 1 :]))
    return merged


def build_from_document(sections: list[str]):
    def build(session) -> None:
        section_nodes = [
            session.register(content=sec, refs=()) for sec in sections
        ]
        prev: str | None = None
        for i, sec in enumerate(section_nodes):
            refs = [sec.id] + ([prev] if prev else [])
            session.expand(refs=[prev, sec.id] if prev else [sec.id])
            agent = session.register(
                content=(
                    f"第 {i + 1} 段阅读小结：提取关键事实、数字与结论，"
                    "并说明与上一段结论的衔接。"
                ),
                refs=refs,
            )
            prev = agent.id

    return build
