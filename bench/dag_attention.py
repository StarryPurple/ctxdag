"""CPU numerical reference for DAG-masked attention; not a serving benchmark."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    name: str
    tokens: tuple[int, ...]
    parents: tuple[str, ...] = ()


def layout(blocks: list[Block]):
    """Validate a topological layout and assign dependency-relative positions."""
    ancestors, ends, positions = {}, {}, {}
    for block in blocks:
        if block.name in ancestors or not block.tokens:
            raise ValueError("duplicate or empty block")
        if any(parent not in ancestors for parent in block.parents):
            raise ValueError("parents must precede their children")
        ancestors[block.name] = set(block.parents)
        for parent in block.parents:
            ancestors[block.name].update(ancestors[parent])
        start = max((ends[p] for p in block.parents), default=0)
        positions[block.name] = tuple(range(start, start + len(block.tokens)))
        ends[block.name] = start + len(block.tokens)
    return ancestors, positions


class ReferenceAttention:
    """Deterministic multi-layer, single-head residual attention with logits.

    Weights are random, not pretrained. Cache entries contain every layer's
    keys/values and final hidden states. Keys include exact ancestor identities,
    token sequences and positions. A cache belongs to one model instance.
    """

    def __init__(self, width: int = 8, layers: int = 3, seed: int = 7):
        if width < 1 or layers < 1:
            raise ValueError("positive width and layers required")
        self.width = width
        rng = random.Random(seed)
        def matrix(rows=width):
            return tuple(tuple(rng.uniform(-0.3, 0.3) for _ in range(width))
                         for _ in range(rows))
        self.weights = tuple((matrix(), matrix(), matrix(), matrix())
                             for _ in range(layers))
        self.head = matrix(16)
        self.cache = {}

    @staticmethod
    def project(matrix, vector):
        return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)

    def run(self, blocks: list[Block], *, reuse=False, causal=False,
            global_positions=False):
        if reuse and (causal or global_positions):
            raise ValueError("reuse requires DAG mask and stable positions")
        ancestors, positions = layout(blocks)
        rows = []
        signatures = {}
        for block in blocks:
            signatures[block.name] = (
                block.name, block.tokens, positions[block.name],
                tuple(sorted(signatures[p] for p in block.parents)),
            )
            for offset, token in enumerate(block.tokens):
                pos = len(rows) if global_positions else positions[block.name][offset]
                rows.append((block.name, offset, token, pos))
        hidden = [tuple(math.sin((token + 1) * (i + 1))
                        + math.cos(pos / (100 ** (i / self.width)))
                        for i in range(self.width)) for _, _, token, pos in rows]
        hits = {b.name for b in blocks if reuse and signatures[b.name] in self.cache}
        layer_kv = []
        for layer, (wq, wk, wv, wo) in enumerate(self.weights):
            keys, values = [], []
            for index, (name, offset, _, _) in enumerate(rows):
                if name in hits:
                    k, v = self.cache[signatures[name]][0][layer][offset]
                else:
                    k = self.project(wk, hidden[index])
                    v = self.project(wv, hidden[index])
                keys.append(k)
                values.append(v)
            layer_kv.append(list(zip(keys, values)))
            updated = []
            for i, (name, offset, _, _) in enumerate(rows):
                if name in hits:
                    updated.append(self.cache[signatures[name]][1][layer][offset])
                    continue
                q = self.project(wq, hidden[i])
                visible = [j for j, (other, local, _, _) in enumerate(rows)
                           if (j <= i if causal else
                               other in ancestors[name] or
                               (other == name and local <= offset))]
                scores = [sum(a*b for a, b in zip(q, keys[j])) / math.sqrt(self.width)
                          for j in visible]
                weights = [math.exp(s - max(scores)) for s in scores]
                total = sum(weights)
                mixed = tuple(sum(w * values[j][d] for w, j in zip(weights, visible))
                              / total for d in range(self.width))
                output = self.project(wo, mixed)
                updated.append(tuple(a + b for a, b in zip(hidden[i], output)))
            hidden = updated
            if layer == 0:
                states = []
            states.append(hidden)
        if reuse:
            for block in blocks:
                indices = [i for i, row in enumerate(rows) if row[0] == block.name]
                self.cache[signatures[block.name]] = (
                    [[kv[i] for i in indices] for kv in layer_kv],
                    [[state[i] for i in indices] for state in states],
                )
        logits = {block.name: [self.project(self.head, hidden[i])
                              for i, row in enumerate(rows) if row[0] == block.name]
                  for block in blocks}
        return logits, sum(len(b.tokens) for b in blocks if b.name in hits)


def difference(a, b):
    return max(abs(x-y) for u, v in zip(a, b) for x, y in zip(u, v))


def experiment():
    root = Block("root", (1, 2))
    left = Block("left", (3, 4), ("root",))
    right = Block("right", (5, 6, 7), ("root",))
    answer = Block("answer", (8,), ("left", "right"))
    model = ReferenceAttention()
    model.run([root, left], reuse=True)
    blocks = [root, right, left, answer]
    cached, hits = model.run(blocks, reuse=True)
    full, _ = model.run(blocks)
    isolated, _ = model.run([root, left])
    causal, _ = model.run(blocks, causal=True)
    causal_isolated, _ = model.run([root, left], causal=True)
    shifted, _ = model.run(blocks, global_positions=True)
    return {
        "kind": "random_weight_cpu_attention_correctness_only",
        "reused_tokens": hits,
        "cached_vs_full_max_logit_error": max(difference(cached[n], full[n]) for n in full),
        "dag_branch_insertion_error": difference(full["left"], isolated["left"]),
        "causal_branch_insertion_error": difference(causal["left"], causal_isolated["left"]),
        "global_position_branch_insertion_error": difference(shifted["left"], isolated["left"]),
    }


if __name__ == "__main__":
    print(json.dumps(experiment(), indent=2))
