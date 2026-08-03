"""Conversational agent wrapper: chat over the protocol session."""

from __future__ import annotations

from typing import Callable

from .expand import ExpandedContext
from .node import Node
from .session import Session

ModelFn = Callable[[str], str]


class ContextAgent:
    """A chat agent whose turns flow through the protocol.

    Each user message becomes a node that chains to the previous agent
    reply; the agent's answer is parsed (refs/requires) and registered.
    ``model`` is a callable(prompt: str) -> str (local engine, API, or
    scripted).  ``catalog_ids`` lists nodes visible in the 可申请范围
    section of every expanded context.
    """

    def __init__(
        self,
        model: ModelFn,
        session: Session | None = None,
        name: str = "agent",
        instruction: str | None = None,
        catalog_ids: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.model = model
        self.session = session or Session()
        self.name = name
        self.instruction = instruction
        self.catalog_ids = tuple(catalog_ids)
        self.last_user_node: Node | None = None
        self.last_agent_node: Node | None = None
        self.last_context: ExpandedContext | None = None

    def say(self, message: str) -> str:
        """One user turn: register, expand, generate, parse, register."""
        refs = (self.last_agent_node.id,) if self.last_agent_node is not None else ()
        user_node = self.session.register(content=message, refs=refs)
        self.last_user_node = user_node

        candidates = list(self.catalog_ids) if self.catalog_ids else None
        ctx = self.session.expand(refs=[user_node.id], candidates=candidates)
        self.last_context = ctx

        prompt = ctx.text
        if self.instruction:
            prompt = f"{prompt}\n\n{self.instruction}"
        raw = self.model(prompt)

        node = self.session.register_declared(
            raw,
            default_refs=[user_node.id],
        )
        self.last_agent_node = node
        return raw
