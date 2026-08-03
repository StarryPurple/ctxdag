"""Tests for dependency-set expansion and canonical rendering."""

import unittest

from contextdag import ExpandedContext, Expander, Node, Registry, SummaryStore, render_node


def build_registry() -> Registry:
    registry = Registry()
    a = registry.register(Node(content="aaa"))
    b = registry.register(Node(content="bbbb", refs=(a.id,)))
    c = registry.register(Node(content="ccccc", refs=(a.id,)))
    return registry, a, b, c


class ExpandTest(unittest.TestCase):
    def test_order_is_canonical(self):
        registry, a, b, c = build_registry()
        ctx = Expander(registry).expand([c.id])
        self.assertEqual(ctx.order, (a.id, c.id))

    def test_union_order_and_positions(self):
        registry, a, b, c = build_registry()
        ctx = Expander(registry).expand([b.id, c.id])
        self.assertEqual(ctx.order[0], a.id)
        self.assertEqual(set(ctx.order), {a.id, b.id, c.id})
        self.assertEqual(ctx.positions[a.id], 1)

    def test_stable_serialization_across_closures(self):
        registry, a, b, c = build_registry()
        ctx1 = Expander(registry).expand([b.id])
        ctx2 = Expander(registry).expand([b.id, c.id])
        block = render_node(registry.get(a.id))
        self.assertIn(block, ctx1.text)
        self.assertIn(block, ctx2.text)
        # byte-identical in both contexts
        self.assertEqual(
            ctx1.text.split("\n\n")[0],
            ctx2.text.split("\n\n")[0],
        )

    def test_block_separator_and_verbatim_content(self):
        registry = Registry()
        a = registry.register(Node(content="line1\nline2"))
        b = registry.register(Node(content="next", refs=(a.id,)))
        ctx = Expander(registry).expand([b.id])
        self.assertIn("<node=", ctx.text)
        self.assertIn("line1\nline2", ctx.text)
        self.assertEqual(ctx.text, f"{render_node(a)}\n\n{render_node(b)}")

    def test_catalog_rendering(self):
        registry = Registry()
        a = registry.register(Node(content="aaa"))
        x = registry.register(Node(content="policy text", meta={"summary": "换货政策"}))
        ctx = Expander(registry).expand([a.id], candidates=[x.id])
        self.assertIn("<目录 可申请范围>", ctx.text)
        self.assertIn(f"<node={x.id}> 换货政策", ctx.text)
        self.assertEqual(ctx.catalog, (x.id,))

    def test_catalog_excludes_nodes_already_in_dependency_set(self):
        registry = Registry()
        a = registry.register(Node(content="aaa"))
        x = registry.register(Node(content="policy", meta={"summary": "政策"}))
        ctx = Expander(registry).expand([a.id], candidates=[a.id, x.id])
        self.assertIn(x.id, ctx.catalog)
        self.assertNotIn(a.id, ctx.catalog)
        self.assertNotIn(f"<node={a.id}>", ctx.text.split("<目录 可申请范围>")[1])

    def test_catalog_uses_side_table_summary(self):
        registry = Registry()
        a = registry.register(Node(content="aaa"))
        x = registry.register(Node(content="policy"))
        store = SummaryStore()
        store.set(x.id, "侧表摘要", source="explicit")
        ctx = Expander(registry, store).expand([a.id], candidates=[x.id])
        self.assertIn(f"<node={x.id}> 侧表摘要", ctx.text)
        self.assertGreater(ctx.catalog_chars, 0)

    def test_catalog_without_store_falls_back_to_meta_then_heuristic(self):
        registry = Registry()
        a = registry.register(Node(content="aaa"))
        x = registry.register(Node(content="政策正文很长", meta={"summary": "政策"}))
        y = registry.register(Node(content="首句即摘要。其余忽略。"))
        ctx = Expander(registry).expand([a.id], candidates=[x.id, y.id])
        self.assertIn(f"<node={x.id}> 政策", ctx.text)
        self.assertIn(f"<node={y.id}> 首句即摘要。", ctx.text)

    def test_depth_placeholder(self):
        registry = Registry()
        a = registry.register(Node(content="aaa"))
        b = registry.register(Node(content="bbbb", refs=(a.id,)))
        c = registry.register(Node(content="ccccc", refs=(b.id,)))
        ctx = Expander(registry).expand([c.id], max_depth=1)
        self.assertIn("[depth limit: content elided]", ctx.text)
        self.assertIn(render_node(registry.get(b.id)), ctx.text)
        self.assertIn(render_node(registry.get(c.id)), ctx.text)

    def test_empty_expand(self):
        ctx = Expander(Registry()).expand([])
        self.assertIsInstance(ctx, ExpandedContext)
        self.assertEqual(ctx.text, "")


if __name__ == "__main__":
    unittest.main()
