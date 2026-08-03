"""tau-bench historical trajectories replayed as conversation chains.

Each trajectory becomes a session: the shared system policy is the root
node (identical content across trajectories -> identical content address),
then every user/assistant/tool message chains onto the previous node.
"""

from __future__ import annotations

import json

TAU_FILES = {
    "gpt-4o-retail": "data/tau-bench/historical_trajectories/gpt-4o-retail.json",
    "gpt-4o-airline": "data/tau-bench/historical_trajectories/gpt-4o-airline.json",
    "sonnet-35-new-retail": "data/tau-bench/historical_trajectories/sonnet-35-new-retail.json",
    "sonnet-35-new-airline": "data/tau-bench/historical_trajectories/sonnet-35-new-airline.json",
}


def iter_samples(limit: int = 10, path: str | None = None, domains=None):
    """Yield trajectory replay builders; all domains when ``path`` omitted."""
    paths = [path] if path else [
        TAU_FILES[d] for d in (domains or list(TAU_FILES))
    ]
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for rec in data[:limit]:
            yield build_from_trajectory(rec["traj"])


def build_from_trajectory(traj: list[dict]):
    def build(session) -> None:
        prev: str | None = None
        for msg in traj:
            content = msg.get("content")
            if content is None:
                content = json.dumps(
                    {k: v for k, v in msg.items() if k != "role"},
                    ensure_ascii=False,
                )
            content = str(content).strip()
            if not content:
                continue
            node = session.register(content=content, refs=(prev,) if prev else ())
            session.expand(refs=[node.id])
            prev = node.id

    return build
