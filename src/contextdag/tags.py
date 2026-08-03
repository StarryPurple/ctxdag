"""Prompt-level protocol directives and text helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

NODE_ID_RE = r"[0-9a-f]{16}"
_REF_RE = re.compile(rf"<ref=\[?([0-9a-f,\s]+)\]?>")
_REQUIRE_RE = re.compile(rf"<require=({NODE_ID_RE})>")
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)
_TAG_RE = re.compile(rf"<ref=[^>]*>|<require={NODE_ID_RE}>")


@dataclass(frozen=True)
class Directives:
    refs: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.refs or self.requires)


def parse_directives(text: str) -> Directives:
    """Extract all ref/require directives from agent output (deterministic)."""
    refs: list[str] = []
    for match in _REF_RE.finditer(text):
        refs.extend(p.strip() for p in match.group(1).split(",") if p.strip())
    requires = tuple(_REQUIRE_RE.findall(text))
    return Directives(refs=tuple(dict.fromkeys(refs)), requires=requires)


def strip_tags(text: str) -> str:
    """Remove ref/require tags without touching any other bytes."""
    return _TAG_RE.sub("", text)


def extract_summary(text: str) -> tuple[str, str | None]:
    """Pull the first explicit ``<summary>...</summary>`` block out of text.

    Returns ``(content_without_tag, summary)``.  The tag is producer
    self-description and never enters node content; an empty summary leaves
    the text untouched.
    """
    match = _SUMMARY_RE.search(text)
    if not match:
        return text, None
    summary = match.group(1).strip()
    if not summary:
        return text, None
    content = text[: match.start()] + text[match.end() :]
    return content, summary


def split_at_requires(text: str) -> list[tuple[str, str | None]]:
    """Split at require tags; each item is (segment, require_id_after)."""
    segments: list[tuple[str, str | None]] = []
    pos = 0
    for match in _REQUIRE_RE.finditer(text):
        segments.append((text[pos : match.start()], match.group(1)))
        pos = match.end()
    segments.append((text[pos:], None))
    return segments


def summary_of(content: str, meta: dict | None = None, max_chars: int = 80) -> str:
    """Short descriptor: explicit meta.summary first, else first sentence."""
    explicit = (meta or {}).get("summary")
    if explicit:
        return str(explicit)
    return first_sentence(content, max_chars=max_chars)


def first_sentence(text: str, max_chars: int = 80) -> str:
    """Heuristic extraction: first line or first sentence, bounded."""
    t = text.strip()
    if not t:
        return ""
    cut = len(t)
    for sep in ("\n", "。", "！", "？", ". ", "! ", "? "):
        idx = t.find(sep)
        if idx != -1:
            cut = min(cut, idx + (1 if sep != "\n" else 0))
    return t[: min(cut, max_chars)].strip()
