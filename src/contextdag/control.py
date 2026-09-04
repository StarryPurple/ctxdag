"""Transport-independent runtime control actions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NODE_ID = re.compile(r"^[0-9a-f]{16}$")


class ControlError(ValueError):
    """Raised when a control action or its transport envelope is invalid."""


@dataclass(frozen=True)
class RequireContext:
    node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ControlError("require_context needs at least one node id")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ControlError("require_context node ids must be unique")
        if any(
            not isinstance(node_id, str) or not _NODE_ID.fullmatch(node_id)
            for node_id in self.node_ids
        ):
            raise ControlError("require_context contains an invalid node id")


@dataclass(frozen=True)
class SearchContext:
    query: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ControlError("search_context query must not be blank")


@dataclass(frozen=True)
class ExpandCatalog:
    topic_id: str

    def __post_init__(self) -> None:
        if not self.topic_id.strip():
            raise ControlError("expand_catalog topic id must not be blank")


@dataclass(frozen=True)
class ReturnAnswer:
    text: str
