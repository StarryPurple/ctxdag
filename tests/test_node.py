"""Tests for the node data model and content addressing."""

import unittest

from contextdag import Node, fingerprint


class NodeTest(unittest.TestCase):
    def test_fingerprint_is_order_insensitive_on_refs(self):
        self.assertEqual(
            fingerprint("x", ["a", "b"]),
            fingerprint("x", ["b", "a"]),
        )

    def test_fingerprint_is_content_sensitive(self):
        self.assertNotEqual(
            fingerprint("x", ["a"]),
            fingerprint("y", ["a"]),
        )

    def test_fingerprint_is_ref_sensitive(self):
        self.assertNotEqual(
            fingerprint("x", ["a"]),
            fingerprint("x", ["a", "b"]),
        )

    def test_meta_participates_in_id(self):
        self.assertNotEqual(
            fingerprint("x", [], {"summary": "s"}),
            fingerprint("x", [], {}),
        )

    def test_node_id_property(self):
        node = Node(content="hello", refs=(), meta={})
        self.assertEqual(node.id, fingerprint("hello", [], {}))
        self.assertEqual(len(node.id), 16)

    def test_same_input_same_id(self):
        a = Node(content="hello", refs=(), meta={})
        b = Node(content="hello", refs=(), meta={})
        self.assertEqual(a.id, b.id)


if __name__ == "__main__":
    unittest.main()
