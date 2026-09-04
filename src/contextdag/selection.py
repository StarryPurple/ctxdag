"""Deterministic node selectors for bounded initial context."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

_TERM_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class SelectionCandidate:
    """A catalog entry available to an initial-context selector."""

    node_id: str
    summary: str
    token_count: int
    position: int
    search_text: str | None = None


class NodeSelector(Protocol):
    """Choose visible node ids without exceeding the supplied token budget."""

    def select(
        self,
        query: str,
        catalog: list[SelectionCandidate],
        budget_tokens: int,
    ) -> list[str]:
        ...


def _within_budget(
    candidates: list[SelectionCandidate],
    budget_tokens: int,
) -> list[str]:
    selected: list[str] = []
    used = 0
    for candidate in candidates:
        if candidate.token_count > budget_tokens - used:
            continue
        selected.append(candidate.node_id)
        used += candidate.token_count
    return selected


class RecentSelector:
    """Select the newest entries that fit, returning registration order."""

    def select(
        self,
        query: str,
        catalog: list[SelectionCandidate],
        budget_tokens: int,
    ) -> list[str]:
        del query
        newest = sorted(catalog, key=lambda item: item.position, reverse=True)
        selected = set(_within_budget(newest, max(0, budget_tokens)))
        return [item.node_id for item in catalog if item.node_id in selected]


class KeywordSelector:
    """Rank summaries with a small deterministic BM25-style lexical score."""

    def select(
        self,
        query: str,
        catalog: list[SelectionCandidate],
        budget_tokens: int,
    ) -> list[str]:
        if not catalog or budget_tokens <= 0:
            return []
        query_terms = _terms(query)
        documents = [
            _terms(item.search_text or item.summary) for item in catalog
        ]
        document_frequency = {
            term: sum(term in document for document in documents)
            for term in query_terms
        }
        ranked = []
        for candidate, document in zip(catalog, documents):
            score = 0.0
            for term in query_terms:
                frequency = document.count(term)
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1 + (len(catalog) - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                score += inverse_frequency * frequency / (frequency + 1.2)
            ranked.append((score, candidate.position, candidate))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2].node_id))
        return _within_budget(
            [candidate for _, _, candidate in ranked],
            budget_tokens,
        )


def _terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TERM_RE.finditer(text)]
