"""Node data model and content-addressed ids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable


class NodeError(ValueError):
    """Base error for invalid node construction or references."""


class DependencyError(NodeError):
    """Raised when a declared dependency does not exist or is invalid."""


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def fingerprint(
    content: str,
    refs: Iterable[str],
    meta: dict | None = None,
) -> str:
    """Content address for a node.

    ``refs`` participates as a set (sorted before hashing), so declaration
    order never changes the id.  Meta participates: any change to any
    input produces a new node.
    """
    payload = _stable_json(
        {
            "content": content,
            "refs": sorted(set(refs)),
            "meta": _stable_json(meta or {}),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Node:
    """A self-contained content unit with declared direct dependencies."""

    content: str
    refs: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return fingerprint(self.content, self.refs, self.meta)
