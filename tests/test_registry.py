"""Tests for the immutable DAG store."""

import unittest

from contextdag import DependencyError, Node, Registry


class RegistryTest(unittest.TestCase):
    def test_register_rejects_unknown_ref(self):
        registry = Registry()
        with self.assertRaises(DependencyError):
            registry.register(Node(content="x", refs=("ffff0000ffff0000",)))

    def test_register_rejects_self_reference(self):
        registry = Registry()
        node = Node(content="x")
        registry.register(node)
        # A self-reference cannot be expressed: the id derives from refs,
        # so referencing one's own id would change the id (no fixed point).
        self.assertNotEqual(Node(content="x", refs=(node.id,)).id, node.id)

    def test_dependency_set_is_transitive(self):
        registry = Registry()
        c = registry.register(Node(content="c"))
        b = registry.register(Node(content="b", refs=(c.id,)))
        a = registry.register(Node(content="a", refs=(b.id,)))
        ids = {n.id for n in registry.dependency_set(a.id)}
        self.assertEqual(ids, {a.id, b.id, c.id})

    def test_get_unknown_raises(self):
        registry = Registry()
        with self.assertRaises(DependencyError):
            registry.get("ffff0000ffff0000")

    def test_nodes_sorted_by_id(self):
        registry = Registry()
        nodes = [Node(content=f"n{i}") for i in range(5)]
        for node in nodes:
            registry.register(node)
        ids = [n.id for n in registry.nodes()]
        self.assertEqual(ids, sorted(ids))

    def test_recent_returns_last_n_in_registration_order(self):
        registry = Registry()
        nodes = [registry.register(Node(content=f"n{i}")) for i in range(5)]
        self.assertEqual(
            registry.recent(2),
            (nodes[3].id, nodes[4].id),
        )
        self.assertEqual(registry.recent(0), ())
        self.assertEqual(registry.recent(99), tuple(n.id for n in nodes))


if __name__ == "__main__":
    unittest.main()
