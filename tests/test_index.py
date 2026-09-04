"""Tree-structured tests for indexes over semantic node features."""

import unittest

from contextdag import FeatureRecord, LexicalFeatureIndex, Node, NodeFeatures


def record(content, summary, **features):
    node = Node(content)
    return FeatureRecord(
        node.id,
        NodeFeatures(summary=summary, **features),
        extractor="test",
        source="model",
    )


class FeatureIndexLeafTest(unittest.TestCase):
    def test_negative_node_cost_is_rejected(self):
        index = LexicalFeatureIndex()
        with self.assertRaises(ValueError):
            index.upsert(record("a", "summary"), -1, 0)

    def test_upsert_replaces_features_without_duplicate_candidate(self):
        index = LexicalFeatureIndex()
        first = record("same", "old summary")
        replacement = FeatureRecord(
            first.node_id,
            NodeFeatures("new summary"),
            extractor="new-model",
            source="model",
        )

        index.upsert(first, 5, 0)
        index.upsert(replacement, 5, 0)

        candidates = index.candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].summary, "new summary")


class FeatureIndexScenarioTest(unittest.TestCase):
    def test_search_uses_all_semantic_feature_fields(self):
        index = LexicalFeatureIndex()
        cache = record(
            "cache",
            "runtime measurement",
            topics=("inference",),
            entities=("vLLM",),
            claims=("prefix reuse increased",),
            keywords=("cached_tokens",),
        )
        unrelated = record(
            "other",
            "restaurant policy",
            topics=("retail",),
        )
        index.upsert(cache, 10, 0)
        index.upsert(unrelated, 10, 1)

        selected = index.search("Which vLLM result increased cached_tokens?", 10)

        self.assertEqual(selected, [cache.node_id])

    def test_search_respects_full_node_token_budget(self):
        index = LexicalFeatureIndex()
        expensive = record("expensive", "cache cache")
        affordable = record("affordable", "cache")
        index.upsert(expensive, 11, 0)
        index.upsert(affordable, 5, 1)

        selected = index.search("cache", 10)

        self.assertEqual(selected, [affordable.node_id])


if __name__ == "__main__":
    unittest.main()
