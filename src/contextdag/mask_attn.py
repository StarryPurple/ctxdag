"""Dependency-aware attention plans; dense masks are correctness references."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from .node import DependencyError
from .registry import Registry


@dataclass(frozen=True)
class MaskAttn:
    """A token layout with ancestor visibility and stable DAG positions.

    Build from an explicit authorized closure. True means attention is allowed;
    additive masks use zero for allowed entries and negative infinity otherwise.
    This object does not execute an attention kernel or authorize retrieval.
    """

    node_order: tuple[str, ...]
    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    token_nodes: tuple[str, ...]
    ancestors: tuple[frozenset[str], ...]

    @classmethod
    def compile(
        cls, registry: Registry, order: Sequence[str],
        tokens: Mapping[str, Sequence[int]],
    ) -> "MaskAttn":
        """Require a complete topological closure; never load hidden ancestors.

        Tokens must be the exact block token sequences sent to the model.
        Independent tokenization is not equivalent to tokenizing concatenated
        text. Special tokens and role/template boundaries belong in the plan.
        """
        order = tuple(order)
        ancestry: dict[str, frozenset[str]] = {}
        ends: dict[str, int] = {}
        input_ids, positions, owners, visible = [], [], [], []
        for nid in order:
            if nid in ancestry:
                raise ValueError("duplicate node in attention layout")
            node = registry.get(nid)
            if any(parent not in ancestry for parent in node.refs):
                raise DependencyError("layout must include parents before children")
            ids = tuple(tokens[nid])
            if not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError("each node requires nonempty nonnegative token ids")
            deps = set(node.refs)
            for parent in node.refs:
                deps.update(ancestry[parent])
            ancestry[nid] = frozenset(deps)
            start = max((ends[p] for p in node.refs), default=0)
            ends[nid] = start + len(ids)
            input_ids.extend(ids)
            positions.extend(range(start, ends[nid]))
            owners.extend([nid] * len(ids))
            visible.extend([ancestry[nid]] * len(ids))
        return cls(order, tuple(input_ids), tuple(positions), tuple(owners),
                   tuple(visible))

    def allows(self, query: int, key: int) -> bool:
        """Check physical token indices; future tokens in a node stay hidden."""
        if not (0 <= query < len(self.input_ids) and 0 <= key < len(self.input_ids)):
            raise IndexError("attention index out of range")
        return (self.token_nodes[key] in self.ancestors[query] or
                (self.token_nodes[key] == self.token_nodes[query] and key <= query))

    def dense_mask(self, *, additive: bool = False) -> tuple[tuple, ...]:
        """Materialize an O(n^2) reference mask, not a sparse execution plan."""
        def entry(i, j):
            allowed = self.allows(i, j)
            return (0.0 if allowed else float("-inf")) if additive else allowed
        return tuple(tuple(entry(i, j) for j in range(len(self.input_ids)))
                     for i in range(len(self.input_ids)))
