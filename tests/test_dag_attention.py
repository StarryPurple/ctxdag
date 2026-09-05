"""Leaf, composition and scenario tests for the attention reference."""

import sys
import unittest

sys.path.insert(0, "bench")
from dag_attention import Block, ReferenceAttention, difference, experiment, layout


class LayoutLeafTest(unittest.TestCase):
    def test_missing_parent_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            layout([Block("a", (1,), ("missing",))])
        with self.assertRaises(ValueError):
            layout([Block("a", (1,)), Block("a", (2,))])

    def test_positions_follow_longest_dependency_path(self):
        ancestors, positions = layout([
            Block("a", (1, 2)), Block("b", (3,), ("a",)),
            Block("c", (4, 5, 6), ("a",)), Block("d", (7,), ("b", "c"))])
        self.assertEqual(positions["d"], (5,))
        self.assertEqual(ancestors["d"], {"a", "b", "c"})


class CacheCompositionTest(unittest.TestCase):
    def test_changed_ancestor_invalidates_descendants(self):
        model = ReferenceAttention()
        child = Block("b", (3,), ("a",))
        model.run([Block("a", (1,)), child], reuse=True)
        blocks = [Block("a", (2,)), child]
        cached, hits = model.run(blocks, reuse=True)
        full, _ = model.run(blocks)
        self.assertEqual(hits, 0)
        self.assertLess(difference(cached["b"], full["b"]), 1e-12)

    def test_branch_reordering_preserves_answer(self):
        a = Block("a", (1,))
        b = Block("b", (2,), ("a",))
        c = Block("c", (3, 4), ("a",))
        d = Block("d", (5,), ("b", "c"))
        model = ReferenceAttention()
        first, _ = model.run([a, b, c, d], reuse=True)
        second, hits = model.run([a, c, b, d], reuse=True)
        full, _ = model.run([a, c, b, d])
        self.assertEqual(hits, 5)
        self.assertLess(difference(first["d"], second["d"]), 1e-12)
        self.assertLess(difference(full["d"], second["d"]), 1e-12)


class AttentionScenarioTest(unittest.TestCase):
    def test_reuse_and_negative_controls(self):
        report = experiment()
        self.assertEqual(report["reused_tokens"], 4)
        self.assertLess(report["cached_vs_full_max_logit_error"], 1e-12)
        self.assertLess(report["dag_branch_insertion_error"], 1e-12)
        self.assertGreater(report["causal_branch_insertion_error"], 1e-6)
        self.assertGreater(report["global_position_branch_insertion_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
