"""Dependency-set expansion: canonical, prefix-stable context rendering."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from .node import Node, NodeError
from .registry import Registry
from .summary import SummaryStore
from .tags import first_sentence


@dataclass(frozen=True)
class ExpandedContext:
    """Result of expanding a dependency set.

    ``text`` is the rendered context.  ``positions`` maps each included
    node id to its 1-based position inside ``text``.  ``catalog`` lists
    the ids visible in the 可申请范围 section (empty when none).
    """

    text: str
    order: tuple[str, ...] = ()
    positions: dict[str, int] = field(default_factory=dict)
    catalog: tuple[str, ...] = ()
    catalog_chars: int = 0


def render_node(node: Node, placeholder: bool = False) -> str:
    """Stable per-node rendering; byte-identical in every dependency set."""
    header = f"<node={node.id}>"
    if placeholder:
        return f"{header} [depth limit: content elided]"
    return f"{header}\n{node.content}"


def topological_order(
    nodes: dict[str, Node],
    registration: Sequence[str] | None = None,
) -> list[Node]:
    """Deterministic topo order: dependencies first, same-depth ties by
    registration order (early-registered shared roots lead the prefix),
    falling back to node id for unknown nodes."""
    rank = (
        {nid: i for i, nid in enumerate(registration)}
        if registration is not None
        else {}
    )

    def key(nid: str) -> tuple[int, str]:
        return (rank.get(nid, 1 << 30), nid)

    indegree = {nid: len(n.refs) for nid, n in nodes.items()}
    children: dict[str, list[str]] = {nid: [] for nid in nodes}
    for nid, node in nodes.items():
        for dep in node.refs:
            children[dep].append(nid)
    ready = deque(sorted((nid for nid, deg in indegree.items() if deg == 0), key=key))
    order: list[Node] = []
    while ready:
        nid = ready.popleft()
        order.append(nodes[nid])
        for child in sorted(children[nid], key=key):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise NodeError("dependency cycle detected in registry")
    return order


def depth_of(registry: Registry, refs: list[str]) -> dict[str, int]:
    """Distance in dependency edges from the nearest ref root."""
    depth: dict[str, int] = {}
    pending = deque((r, 0) for r in refs)
    while pending:
        nid, d = pending.popleft()
        if nid in depth and depth[nid] <= d:
            continue
        depth[nid] = d
        for dep in registry.get(nid).refs:
            pending.append((dep, d + 1))
    return depth


def render_catalog(
    registry: Registry,
    ids: tuple[str, ...],
    summaries: SummaryStore | None = None,
) -> str:
    lines = ["<目录 可申请范围>"]
    for nid in ids:
        node = registry.get(nid)
        summary = summaries.get(nid) if summaries is not None else None
        if summary is None:
            summary = node.meta.get("summary")
        if summary is None:
            summary = first_sentence(node.content)
        lines.append(f"<node={nid}> {summary}")
    lines.append("</目录>")
    return "\n".join(lines)


class Expander:
    """Renders dependency sets as canonical, prefix-stable context text."""

    def __init__(
        self,
        registry: Registry,
        summaries: SummaryStore | None = None,
    ) -> None:
        self._registry = registry
        self._summaries = summaries

    def expand(
        self,
        refs: list[str] | tuple[str, ...] = (),
        max_depth: int = 0,
        candidates: list[str] | tuple[str, ...] | None = None,
    ) -> ExpandedContext:
        """Expand the union of the refs' dependency sets, plus optional catalog."""
        refs = list(refs or [])
        for ref in refs:
            self._registry.get(ref)

        nodes: dict[str, Node] = {}
        for ref in refs:
            for node in self._registry.dependency_set(ref):
                nodes[node.id] = node

        catalog_ids = tuple(
            sorted(nid for nid in (candidates or ()) if nid not in nodes)
        )
        if not nodes and not catalog_ids:
            return ExpandedContext(text="", order=(), positions={}, catalog=())

        order = topological_order(nodes, self._registry.registration_order())
        depth = depth_of(self._registry, refs)
        blocks: list[str] = []
        positions: dict[str, int] = {}
        for pos, node in enumerate(order, start=1):
            positions[node.id] = pos
            d = depth.get(node.id, 0)
            blocks.append(render_node(node, placeholder=0 < max_depth < d))

        text = "\n\n".join(blocks)
        catalog_chars = 0
        if catalog_ids:
            catalog_text = render_catalog(
                self._registry, catalog_ids, self._summaries
            )
            catalog_chars = len(catalog_text)
            text = f"{text}\n\n{catalog_text}" if text else catalog_text
        return ExpandedContext(
            text=text,
            order=tuple(n.id for n in order),
            positions=positions,
            catalog=catalog_ids,
            catalog_chars=catalog_chars,
        )
