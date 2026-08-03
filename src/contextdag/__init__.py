"""ContextDAG: content-addressed, dependency-aware agent context runtime."""

from .agent import ContextAgent
from .expand import ExpandedContext, Expander, render_node, topological_order
from .node import DependencyError, Node, NodeError, fingerprint
from .registry import Registry
from .session import Session
from .summary import SummaryService, SummaryStore
from .tags import (
    Directives,
    extract_summary,
    first_sentence,
    parse_directives,
    split_at_requires,
    strip_tags,
    summary_of,
)

__all__ = [
    "ContextAgent",
    "DependencyError",
    "Directives",
    "ExpandedContext",
    "Expander",
    "Node",
    "NodeError",
    "Registry",
    "Session",
    "SummaryService",
    "SummaryStore",
    "extract_summary",
    "fingerprint",
    "first_sentence",
    "parse_directives",
    "render_node",
    "split_at_requires",
    "strip_tags",
    "summary_of",
    "topological_order",
]
