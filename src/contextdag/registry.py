"""Immutable, cycle-free DAG store for context nodes."""

from __future__ import annotations

from .node import DependencyError, Node


class Registry:
    """Content-addressed registry of nodes.

    Invariants enforced on registration:

    - every declared dependency must already be registered
      (no forward references, no cycles by construction);
    - a node may not reference itself.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._order: list[str] = []

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def get(self, node_id: str) -> Node:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise DependencyError(f"unknown node {node_id!r}") from None

    def register(self, node: Node) -> Node:
        for dep in node.refs:
            if dep not in self._nodes:
                raise DependencyError(
                    f"node {node.id!r} references unregistered node {dep!r}"
                )
            if dep == node.id:
                raise DependencyError(f"node {node.id!r} cannot reference itself")
        self._nodes[node.id] = node
        self._order.append(node.id)
        return node

    def recent(self, count: int) -> tuple[str, ...]:
        """Ids of the most recently registered nodes, in registration order."""
        if count <= 0:
            return ()
        return tuple(self._order[-count:])

    def registration_order(self) -> tuple[str, ...]:
        """All node ids in registration order (stable; nodes are immutable)."""
        return tuple(self._order)

    def nodes(self) -> list[Node]:
        return sorted(self._nodes.values(), key=lambda n: n.id)

    def dependency_set(self, node_id: str) -> list[Node]:
        """All reachable nodes (direct + indirect dependencies, incl. self)."""
        node = self.get(node_id)
        seen: dict[str, Node] = {node.id: node}

        def walk(n: Node) -> None:
            for dep in n.refs:
                if dep in seen:
                    continue
                ancestor = self.get(dep)
                seen[dep] = ancestor
                walk(ancestor)

        walk(node)
        return sorted(seen.values(), key=lambda n: n.id)
