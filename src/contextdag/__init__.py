"""ContextDAG: content-addressed, dependency-aware agent context runtime."""

from .agent import ContextAgent
from .accounting import Accountant, NodeStat, TurnStats, lcp_len
from .expand import ExpandedContext, Expander, render_node, topological_order
from .features import (
    FeatureExtractor,
    FeatureRecord,
    FeatureStore,
    HeuristicFeatureExtractor,
    ModelFeatureExtractor,
    NodeFeatures,
)
from .control import (
    ControlError,
    ExpandCatalog,
    RequireContext,
    ReturnAnswer,
    SearchContext,
)
from .node import DependencyError, Node, NodeError, fingerprint
from .index import FeatureIndex, IndexedNode, LexicalFeatureIndex
from .registry import Registry
from .mask_attn import MaskAttn
from .selection import KeywordSelector, NodeSelector, RecentSelector, SelectionCandidate
from .session import Session
from .summary import SummaryService, SummaryStore
from .transport import ToolRequest, ToolTransport
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
    "ControlError",
    "DependencyError",
    "Directives",
    "ExpandedContext",
    "Expander",
    "ExpandCatalog",
    "FeatureExtractor",
    "FeatureRecord",
    "FeatureStore",
    "FeatureIndex",
    "HeuristicFeatureExtractor",
    "ModelFeatureExtractor",
    "MaskAttn",
    "KeywordSelector",
    "IndexedNode",
    "LexicalFeatureIndex",
    "Node",
    "NodeSelector",
    "NodeStat",
    "NodeError",
    "NodeFeatures",
    "RecentSelector",
    "Registry",
    "RequireContext",
    "ReturnAnswer",
    "SearchContext",
    "SelectionCandidate",
    "Session",
    "SummaryService",
    "SummaryStore",
    "ToolRequest",
    "ToolTransport",
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
