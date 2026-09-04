"""Tests for the session workflow."""

import unittest

from contextdag import DependencyError, Session


class SessionTest(unittest.TestCase):
    def test_register_read_full_context(self):
        session = Session()
        node = session.register("hello")
        self.assertIn(node.id, session.read(node.id))
        self.assertIn("hello", session.read(node.id))
        self.assertIn("hello", session.full_context())

    def test_content_stored_verbatim(self):
        session = Session()
        node = session.register("line1\nline2\n")
        self.assertEqual(
            session.read(node.id),
            f"<node={node.id}>\nline1\nline2\n",
        )

    def test_expand_chain_and_prefix_stability(self):
        session = Session()
        a = session.register("aaa")
        b = session.register("bbbb", refs=[a.id])
        ctx1 = session.expand(refs=[a.id], candidates=())
        ctx2 = session.expand(refs=[b.id], candidates=())
        self.assertTrue(ctx2.text.startswith(ctx1.text))

    def test_require_from_catalog_counts_and_reexpands(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        ctx = session.expand(refs=[a.id], candidates=[x.id])
        self.assertIn("<目录 可申请范围>", ctx.text)
        ctx2 = session.require(x.id)
        self.assertEqual(session.page_faults, 1)
        self.assertIn("policy", ctx2.text)
        self.assertIn(x.id, ctx2.order)

    def test_requiring_an_already_loaded_node_is_idempotent(self):
        session = Session()
        node = session.register("loaded")
        first = session.expand(refs=[node.id], candidates=[node.id])

        second = session.require(node.id)

        self.assertIs(second, first)
        self.assertEqual(session.requires_issued, 1)
        self.assertEqual(session.page_faults, 0)
        self.assertEqual(session.expands, 1)

    def test_require_outside_catalog_rejected(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        session.expand(refs=[a.id], candidates=[x.id])
        y = session.register("other")
        with self.assertRaises(DependencyError):
            session.require(y.id)

    def test_require_unknown_rejected(self):
        session = Session()
        session.expand(refs=[])
        with self.assertRaises(DependencyError):
            session.require("ffff0000ffff0000")

    def test_default_catalog_is_recent_nodes(self):
        session = Session(catalog_size=2)
        a = session.register("aaa")
        b = session.register("bbb")
        c = session.register("ccc")
        ctx = session.expand(refs=[a.id])
        self.assertEqual(ctx.catalog, tuple(sorted((b.id, c.id))))
        self.assertNotIn(a.id, ctx.catalog)  # already expanded
        self.assertIn("<目录 可申请范围>", ctx.text)

    def test_catalog_size_zero_disables_default_catalog(self):
        session = Session(catalog_size=0)
        session.register("aaa")
        ctx = session.expand(refs=[])
        self.assertEqual(ctx.catalog, ())
        self.assertNotIn("<目录 可申请范围>", ctx.text)

    def test_grounded_register_rejects_out_of_context(self):
        session = Session()
        a = session.register("aaa")
        b = session.register("bbb")
        session.expand(refs=[a.id])
        with self.assertRaises(DependencyError):
            session.register("x", refs=[b.id], grounded=True)
        node = session.register("x", refs=[a.id], grounded=True)
        self.assertEqual(node.refs, (a.id,))

    def test_grounded_expand(self):
        session = Session()
        a = session.register("aaa")
        b = session.register("bbb")
        session.expand(refs=[a.id])
        with self.assertRaises(DependencyError):
            session.expand(refs=[b.id], grounded=True)

    def test_register_declared_require_splits_nodes(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        last = session.register_declared(
            "need it.<require=%s> then continue" % x.id,
            default_refs=[a.id],
        )
        nodes = session.registry.nodes()
        self.assertEqual(len(nodes), 4)  # a, x, segment1, segment2
        seg1 = next(n for n in nodes if n.content == "need it.")
        seg2 = next(n for n in nodes if n.content == " then continue")
        self.assertEqual(seg1.refs, (a.id,))
        self.assertEqual(seg2.refs, (seg1.id, x.id))
        self.assertEqual(last.id, seg2.id)
        self.assertEqual(session.page_faults, 1)

    def test_register_declared_require_at_start(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        last = session.register_declared(
            "<require=%s> content" % x.id,
            default_refs=[a.id],
        )
        self.assertEqual(last.refs, (a.id, x.id))
        self.assertEqual(session.page_faults, 1)

    def test_register_declared_refs_override_defaults(self):
        session = Session()
        a = session.register("aaa")
        b = session.register("bbb")
        node = session.register_declared(
            f"<ref={b.id}> content",
            default_refs=[a.id],
        )
        self.assertEqual(node.refs, (b.id,))
        self.assertNotIn("ref=", node.content)

    def test_register_declared_unknown_require_rejected(self):
        session = Session()
        a = session.register("aaa")
        with self.assertRaises(DependencyError):
            session.register_declared(
                "text<require=ffff0000ffff0000>more",
                default_refs=[a.id],
            )

    def test_register_extracts_explicit_summary(self):
        session = Session()
        node = session.register("正文<summary>政策摘要</summary>")
        self.assertEqual(node.content, "正文")
        self.assertEqual(session.summaries.get(node.id), "政策摘要")
        ctx = session.expand(refs=[], candidates=[node.id])
        self.assertIn("政策摘要", ctx.text)
        self.assertNotIn("<summary>", node.content)

    def test_register_declared_extracts_summary(self):
        session = Session()
        a = session.register("aaa")
        node = session.register_declared(
            f"<ref={a.id}> 正文<summary>决策摘要</summary>"
        )
        self.assertEqual(node.content, " 正文")
        self.assertEqual(session.summaries.get(node.id), "决策摘要")

    def test_summary_service_fills_catalog_lazily_and_caches(self):
        calls: list[str] = []

        class Svc:
            def summarize(self, content: str, node_id: str) -> str:
                calls.append(node_id)
                return f"摘要：{content[:6]}"

        session = Session(summary_service=Svc())
        a = session.register("订单内容很长很长的正文")
        ctx = session.expand(refs=[], candidates=[a.id])
        self.assertEqual(len(calls), 1)
        self.assertIn("摘要：订单内容很长", ctx.text)
        session.expand(refs=[], candidates=[a.id])
        self.assertEqual(len(calls), 1)  # cached, not regenerated

    def test_summary_service_failure_falls_back_to_heuristic(self):
        class Boom:
            def summarize(self, content: str, node_id: str) -> str:
                raise RuntimeError("no network")

        session = Session(summary_service=Boom())
        a = session.register("首句就是摘要内容。第二句忽略。")
        ctx = session.expand(refs=[], candidates=[a.id])
        self.assertIn("首句就是摘要内容。", ctx.text)

    def test_metrics_counters(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        session.expand(refs=[a.id], candidates=[x.id])
        self.assertEqual(session.expands, 1)
        self.assertGreater(session.catalog_chars, 0)
        session.require(x.id)
        self.assertEqual(session.requires_issued, 1)
        self.assertEqual(session.expands, 2)
        with self.assertRaises(DependencyError):
            session.require("ffff0000ffff0000")
        self.assertEqual(session.requires_issued, 2)
        self.assertEqual(session.requires_rejected, 1)

    def test_register_declared_require_metrics(self):
        session = Session()
        a = session.register("aaa")
        x = session.register("policy")
        session.register_declared(
            f"text<require={x.id}>more",
            default_refs=[a.id],
        )
        self.assertEqual(session.requires_issued, 1)
        self.assertEqual(session.requires_rejected, 0)
        with self.assertRaises(DependencyError):
            session.register_declared(
                "t<require=ffff0000ffff0000>m",
                default_refs=[a.id],
            )
        self.assertEqual(session.requires_rejected, 1)


if __name__ == "__main__":
    unittest.main()
