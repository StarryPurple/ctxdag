"""Indexes over rebuildable node features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .features import FeatureRecord
from .selection import KeywordSelector, SelectionCandidate


@dataclass(frozen=True)
class IndexedNode:
    """Feature record plus stable selection cost and ordering metadata."""

    record: FeatureRecord
    token_count: int
    position: int


class FeatureIndex(Protocol):
    """Replaceable semantic lookup over feature records."""

    def upsert(
        self,
        record: FeatureRecord,
        token_count: int,
        position: int,
    ) -> None:
        ...

    def search(self, query: str, budget_tokens: int) -> list[str]:
        ...


class LexicalFeatureIndex:
    """Deterministic BM25 routing across every textual feature field."""

    def __init__(self) -> None:
        self._entries: dict[str, IndexedNode] = {}

    def upsert(
        self,
        record: FeatureRecord,
        token_count: int,
        position: int,
    ) -> None:
        if token_count < 0:
            raise ValueError("token count must not be negative")
        self._entries[record.node_id] = IndexedNode(
            record=record,
            token_count=token_count,
            position=position,
        )

    def candidates(self) -> list[SelectionCandidate]:
        return [
            self._candidate(entry)
            for entry in sorted(
                self._entries.values(), key=lambda item: item.position
            )
        ]

    def search(self, query: str, budget_tokens: int) -> list[str]:
        return KeywordSelector().select(
            query,
            self.candidates(),
            budget_tokens,
        )

    @staticmethod
    def _candidate(entry: IndexedNode) -> SelectionCandidate:
        features = entry.record.features
        search_parts = (
            features.summary,
            *features.topics,
            *features.entities,
            *features.claims,
            *features.keywords,
            features.node_type,
            features.task_state or "",
        )
        return SelectionCandidate(
            node_id=entry.record.node_id,
            summary=features.summary,
            token_count=entry.token_count,
            position=entry.position,
            search_text="\n".join(part for part in search_parts if part),
        )
