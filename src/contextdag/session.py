"""Agent-facing session: register, expand, require, read, full context."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .expand import ExpandedContext, Expander, render_node
from .features import FeatureExtractor, FeatureRecord, FeatureStore
from .node import DependencyError, Node
from .registry import Registry
from .summary import SummaryService, SummaryStore
from .tags import extract_summary, parse_directives, split_at_requires, strip_tags


class Session:
    """Entry point of the protocol.

    Typical flow: register a root, expand its dependency set into the
    context text the model sees, generate, register the reply (parsing
    any tags), expand again for the next agent, and use ``require`` for
    page-fault expansion.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        catalog_size: int = 1024,
        summaries: SummaryStore | None = None,
        summary_service: SummaryService | None = None,
        features: FeatureStore | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        self.registry = registry or Registry()
        self.catalog_size = catalog_size
        self.summaries = summaries or SummaryStore()
        self.summary_service = summary_service
        self.features = features or FeatureStore()
        self.feature_extractor = feature_extractor
        self._expander = Expander(self.registry, self.summaries)
        self._current: ExpandedContext | None = None
        self._last_candidates: tuple[str, ...] | None = None
        self._last_max_depth = 0
        self.page_faults = 0
        self.requires_issued = 0
        self.requires_rejected = 0
        self.catalog_chars = 0
        self.expands = 0

    @property
    def current_context(self) -> ExpandedContext | None:
        """Last expanded context, or None before the first expansion."""
        return self._current

    @property
    def current_node_ids(self) -> tuple[str, ...]:
        """Ids in the last expanded context (what the agent saw)."""
        return self._current.order if self._current else ()

    # -- node registration ------------------------------------------------

    def register(
        self,
        content: str,
        refs: list[str] | tuple[str, ...] = (),
        meta: dict | None = None,
        grounded: bool = False,
    ) -> Node:
        """Register a node; an explicit ``<summary>`` block is extracted into
        the side table and never enters node content (no newline edits)."""
        refs = tuple(refs)
        if grounded:
            self._check_grounded(refs, self.current_node_ids)
        content, explicit = extract_summary(content)
        node = Node(content=content, refs=refs, meta=meta or {})
        self.registry.register(node)
        if explicit is not None:
            self.summaries.set(node.id, explicit, source="explicit")
        return node

    def index_node(
        self,
        node_id: str,
        extractor: FeatureExtractor | None = None,
    ) -> FeatureRecord:
        """Extract and store rebuildable features outside the write path."""
        selected = extractor or self.feature_extractor
        if selected is None:
            raise RuntimeError("no feature extractor configured")
        node = self.registry.get(node_id)
        features = selected.extract(node)
        record = FeatureRecord(
            node_id=node.id,
            features=features,
            extractor=selected.name,
            source=selected.source,
        )
        self.features.set(record)
        self.summaries.set(node.id, features.summary, source="service")
        return record

    def index_nodes(
        self,
        node_ids: list[str] | tuple[str, ...],
        extractor: FeatureExtractor | None = None,
        max_workers: int = 1,
    ) -> tuple[FeatureRecord, ...]:
        """Build side-table features concurrently, outside node registration.

        Results retain first-request order and are committed only after every
        extraction succeeds, so a failed batch cannot leave a partial index.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        selected = extractor or self.feature_extractor
        if selected is None:
            raise RuntimeError("no feature extractor configured")
        unique_ids = tuple(dict.fromkeys(node_ids))
        nodes = tuple(self.registry.get(node_id) for node_id in unique_ids)

        def build(node: Node) -> FeatureRecord:
            return FeatureRecord(
                node_id=node.id,
                features=selected.extract(node),
                extractor=selected.name,
                source=selected.source,
            )

        if max_workers == 1:
            records = tuple(build(node) for node in nodes)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                records = tuple(executor.map(build, nodes))
        for record in records:
            self.features.set(record)
            self.summaries.set(
                record.node_id, record.features.summary, source="service"
            )
        return records

    def register_declared(
        self,
        text: str,
        default_refs: list[str] | tuple[str, ...] = (),
        meta: dict | None = None,
        grounded: bool = False,
    ) -> Node | None:
        """Register from agent output; require tags split the stream.

        Segment 0 keeps the declared refs; each later segment depends on
        the previous node plus its require target.  Returns the last node.
        """
        directives = parse_directives(text)
        self.page_faults += len(directives.requires)
        self.requires_issued += len(directives.requires)
        for req in directives.requires:
            if req not in self.registry:
                self.requires_rejected += 1
        initial = tuple(directives.refs) or tuple(default_refs)
        last: Node | None = None
        prev_req: str | None = None
        for index, (segment, req) in enumerate(split_at_requires(text)):
            content = strip_tags(segment)
            content, explicit = extract_summary(content)
            if not content.strip():
                if index == 0 and req is not None:
                    initial = initial + (req,)  # require at stream start
                continue
            if index == 0 or last is None:
                refs = initial
            else:
                refs = tuple(
                    dict.fromkeys((last.id, prev_req) if prev_req else (last.id,))
                )
            last = self.register(content=content, refs=refs, meta=meta, grounded=grounded)
            if explicit is not None:
                self.summaries.set(last.id, explicit, source="explicit")
            prev_req = req
        return last

    def _check_grounded(self, refs: tuple[str, ...], known: tuple[str, ...]) -> None:
        known_set = set(known)
        for ref in refs:
            if ref not in known_set:
                raise DependencyError(
                    f"grounded ref {ref!r} is not in the current expanded context"
                )

    # -- expansion --------------------------------------------------------

    def expand(
        self,
        refs: list[str] | tuple[str, ...] = (),
        max_depth: int = 0,
        candidates: list[str] | tuple[str, ...] | None = None,
        grounded: bool = False,
    ) -> ExpandedContext:
        """Expand a dependency set (refs must be registered).

        When ``candidates`` is omitted, the catalog defaults to the most
        recently registered ``catalog_size`` nodes (see ``Session``).
        """
        refs = list(refs or [])
        if grounded:
            self._check_grounded(tuple(refs), self.current_node_ids)
        if candidates is None:
            candidates = self.registry.recent(self.catalog_size)
        return self._do_expand(refs, max_depth=max_depth, candidates=candidates)

    def _do_expand(
        self,
        refs: list[str],
        max_depth: int,
        candidates: list[str] | tuple[str, ...] | None,
    ) -> ExpandedContext:
        candidates = tuple(candidates or ())
        self._fill_missing_summaries(candidates)
        self._current = self._expander.expand(
            refs, max_depth=max_depth, candidates=candidates
        )
        self._last_candidates = candidates
        self._last_max_depth = max_depth
        self.expands += 1
        self.catalog_chars += self._current.catalog_chars
        return self._current

    def _fill_missing_summaries(self, candidate_ids: tuple[str, ...]) -> None:
        """Lazily generate summaries for candidates that have none."""
        if self.summary_service is None:
            return
        for nid in candidate_ids:
            if nid in self.summaries:
                continue
            node = self.registry.get(nid)
            if node.meta.get("summary"):
                continue
            try:
                text = self.summary_service.summarize(node.content, nid)
            except Exception:
                continue  # heuristic fallback at render time
            self.summaries.set(nid, text, source="service")

    def require(self, node_id: str, max_depth: int = 0) -> ExpandedContext:
        """Page-fault request: add ``node_id`` to the closure and re-expand."""
        self.requires_issued += 1
        if node_id not in self.registry:
            self.requires_rejected += 1
            raise DependencyError(f"require {node_id!r} does not resolve to a node")
        if self._last_candidates is not None and node_id not in self._last_candidates:
            self.requires_rejected += 1
            raise DependencyError(
                f"require {node_id!r} is not in the current catalog"
            )
        refs = list(self.current_node_ids)
        if node_id in refs:
            return self._current
        if node_id not in refs:
            refs.append(node_id)
        self.page_faults += 1
        return self._do_expand(
            refs, max_depth=max_depth, candidates=self._last_candidates
        )

    def require_many(
        self,
        node_ids: list[str] | tuple[str, ...],
        max_depth: int = 0,
    ) -> ExpandedContext:
        """Atomically validate and load several authorized nodes."""
        requested = tuple(dict.fromkeys(node_ids))
        self.requires_issued += len(requested)
        invalid = [node_id for node_id in requested if node_id not in self.registry]
        unauthorized = [
            node_id
            for node_id in requested
            if self._last_candidates is not None
            and node_id not in self._last_candidates
        ]
        if invalid or unauthorized:
            self.requires_rejected += len(set(invalid + unauthorized))
            rejected = (invalid + unauthorized)[0]
            raise DependencyError(f"require {rejected!r} is not authorized")
        refs = list(self.current_node_ids)
        added = [node_id for node_id in requested if node_id not in refs]
        if not added:
            if self._current is None:
                raise RuntimeError("session must be expanded before require")
            return self._current
        refs.extend(added)
        self.page_faults += len(added)
        return self._do_expand(
            refs, max_depth=max_depth, candidates=self._last_candidates
        )

    # -- reads ------------------------------------------------------------

    def read(self, node_id: str) -> str:
        """Render a single node's full content."""
        return render_node(self.registry.get(node_id))

    def full_context(self) -> str:
        """Full-history fallback: render every registered node."""
        return "\n\n".join(render_node(n) for n in self.registry.nodes())
