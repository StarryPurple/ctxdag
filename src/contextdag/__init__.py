"""ContextDAG: content-addressed, dependency-aware agent context runtime."""

from .agent import ContextAgent
from .accounting import Accountant, NodeStat, TurnStats, lcp_len
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
    "Accountant",
    "ContextAgent",
    "DependencyError",
    "Directives",
    "ExpandedContext",
    "Expander",
    "Node",
    "NodeStat",
    "NodeError",
    "Registry",
    "Session",
    "SummaryService",
    "SummaryStore",
    "TurnStats",
    "extract_summary",
    "fingerprint",
    "first_sentence",
    "lcp_len",
    "parse_directives",
    "render_node",
    "split_at_requires",
    "strip_tags",
    "summary_of",
    "topological_order",
]
