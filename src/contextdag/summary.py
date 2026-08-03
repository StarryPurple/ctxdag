"""Side-table summaries and the lazy summary-service hook.

Summaries live outside node content/meta: ``meta`` participates in content
addressing, so LLM-written summaries must not affect node ids.  The store
keeps the best available source per node (reader > explicit > service), and
the service protocol is the pluggable decision-aid generator (small local
model, API, or scripted stand-in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SummaryService(Protocol):
    """One-shot decision-aid summary generator (no KV reuse between tasks)."""

    def summarize(self, content: str, node_id: str) -> str:
        """Return a short "domain + when to use it" description."""
        ...


@dataclass(frozen=True)
class _Entry:
    text: str
    source: str


# Lower rank wins: higher-quality sources replace lower ones on update.
_SOURCE_RANK = {"reader": 0, "explicit": 1, "service": 2, "meta": 3}


class SummaryStore:
    """``node_id -> summary`` map; external to node content/meta (id-safe)."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._entries

    def get(self, node_id: str) -> str | None:
        """Best stored summary, or None when absent."""
        entry = self._entries.get(node_id)
        return entry.text if entry else None

    def set(self, node_id: str, text: str, source: str) -> None:
        """Store the best source; equal rank replaces (allows refresh)."""
        text = text.strip()
        if not text:
            return
        existing = self._entries.get(node_id)
        if existing is None or _SOURCE_RANK.get(
            source, 9
        ) <= _SOURCE_RANK.get(existing.source, 9):
            self._entries[node_id] = _Entry(text=text, source=source)

    def entries(self) -> dict[str, str]:
        """Snapshot of all stored summaries (id -> text)."""
        return {nid: entry.text for nid, entry in self._entries.items()}
