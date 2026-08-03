"""Token-level reuse accounting for expanded contexts.

A proxy for engine prefix-cache (radix) hits: an unlimited cache of every
context seen so far, and each new context's cached tokens are the longest
prefix it shares with any previously seen context.  Tokenizer-agnostic:
pass any ``tokenize(text) -> list[int]`` callable (e.g. an HF fast
tokenizer).  Per-node stats are approximate at block boundaries because
each block is tokenized separately; headline numbers use whole-text
tokenization and are exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

Tokenizer = Callable[[str], list[int]]


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(frozen=True)
class TurnStats:
    turn: int
    prompt_tokens: int
    cached_tokens: int
    new_tokens: int
    catalog_tokens: int

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


@dataclass
class NodeStat:
    block_tokens: int
    appearances: int = 0


@dataclass
class Accountant:
    tokenize: Tokenizer
    _seen: list[tuple[int, ...]] = field(default_factory=list, init=False)
    turns: list[TurnStats] = field(default_factory=list, init=False)
    nodes: dict[str, NodeStat] = field(default_factory=dict, init=False)

    def account(
        self,
        text: str,
        order: Sequence[str] = (),
        node_blocks: Callable[[str], str] | None = None,
        catalog_text: str | None = None,
    ) -> TurnStats:
        """Record one expanded context and return its reuse stats."""
        ids = tuple(self.tokenize(text))
        cached = 0
        for prev in self._seen:
            cached = max(cached, lcp_len(ids, prev))
        self._seen.append(ids)
        stats = TurnStats(
            turn=len(self.turns) + 1,
            prompt_tokens=len(ids),
            cached_tokens=cached,
            new_tokens=len(ids) - cached,
            catalog_tokens=len(self.tokenize(catalog_text)) if catalog_text else 0,
        )
        self.turns.append(stats)
        if node_blocks is not None:
            for nid in order:
                stat = self.nodes.setdefault(nid, NodeStat(block_tokens=0))
                if stat.block_tokens == 0:
                    stat.block_tokens = len(self.tokenize(node_blocks(nid)))
                stat.appearances += 1
        return stats

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.turns)

    @property
    def total_cached_tokens(self) -> int:
        return sum(t.cached_tokens for t in self.turns)

    @property
    def overall_hit_rate(self) -> float:
        total = self.total_prompt_tokens
        return self.total_cached_tokens / total if total else 0.0
