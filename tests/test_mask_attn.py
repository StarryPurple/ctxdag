"""Tree-structured tests of registry-backed attention visibility."""

import unittest

from contextdag import DependencyError, MaskAttn, Session


class MaskLeafTest(unittest.TestCase):
    def test_internal_causality_and_additive_convention(self):
        s = Session()
        a = s.register("a")
        plan = MaskAttn.compile(s.registry, [a.id], {a.id: [1, 2]})
        self.assertEqual(plan.dense_mask(), ((True, False), (True, True)))
        self.assertEqual(plan.dense_mask(additive=True)[0], (0.0, float("-inf")))
        with self.assertRaises(IndexError):
            plan.allows(-1, 0)

    def test_invalid_tokens_and_duplicate_layout(self):
        s = Session()
        a = s.register("a")
        for ids in ([], [True], [-1]):
            with self.assertRaises(ValueError):
                MaskAttn.compile(s.registry, [a.id], {a.id: ids})
        with self.assertRaises(ValueError):
            MaskAttn.compile(s.registry, [a.id, a.id], {a.id: [1]})


class MaskClosureTest(unittest.TestCase):
    def test_incomplete_closure_is_rejected(self):
        s = Session()
        a = s.register("a")
        b = s.register("b", refs=[a.id])
        with self.assertRaises(DependencyError):
            MaskAttn.compile(s.registry, [b.id], {b.id: [2]})

    def test_branch_visibility_and_join(self):
        s = Session()
        a = s.register("a")
        b = s.register("b", refs=[a.id])
        c = s.register("c", refs=[a.id])
        d = s.register("answer", refs=[b.id, c.id])
        ids = {a.id: [1], b.id: [2], c.id: [3, 4], d.id: [5]}
        p = MaskAttn.compile(s.registry, [a.id, b.id, c.id, d.id], ids)
        self.assertEqual(p.position_ids, (0, 1, 1, 2, 3))
        self.assertFalse(p.allows(2, 1))
        self.assertTrue(p.allows(2, 0))
        self.assertTrue(all(p.dense_mask()[-1]))
        reordered = MaskAttn.compile(s.registry, [a.id, c.id, b.id, d.id], ids)
        self.assertEqual(reordered.position_ids[3], p.position_ids[1])


class MaskSessionScenarioTest(unittest.TestCase):
    def test_require_expands_visibility_without_exposing_other_catalog_nodes(self):
        s = Session()
        root = s.register("root")
        evidence = s.register("evidence", refs=[root.id])
        hidden = s.register("unrequested")
        s.expand([root.id], candidates=[evidence.id, hidden.id])
        tokens = {root.id: [1], evidence.id: [2], hidden.id: [3]}
        initial = MaskAttn.compile(s.registry, s.current_node_ids, tokens)
        s.require_many([evidence.id])
        answer = s.register("answer", refs=s.current_node_ids, grounded=True)
        tokens[answer.id] = [4]
        s.expand([answer.id])
        plan = MaskAttn.compile(s.registry, s.current_node_ids, tokens)
        self.assertEqual(initial.input_ids, (1,))
        self.assertNotIn(hidden.id, plan.node_order)
        self.assertEqual(plan.input_ids, (1, 2, 4))
        self.assertTrue(all(plan.dense_mask()[-1]))
