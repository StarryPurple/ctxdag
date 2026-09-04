"""Tree-structured tests for rebuildable node features."""

import unittest

from contextdag import (
    FeatureRecord,
    FeatureStore,
    HeuristicFeatureExtractor,
    ModelFeatureExtractor,
    Node,
    NodeFeatures,
    Session,
)


class FeatureLeafTest(unittest.TestCase):
    def test_features_require_a_nonblank_summary(self):
        with self.assertRaises(ValueError):
            NodeFeatures(summary="  ")

    def test_heuristic_extractor_is_deterministic_and_structured(self):
        node = Node(
            "Cache reuse improves latency. Cache reuse needs measurement.",
            meta={
                "type": "experiment_result",
                "topics": ["cache", "evaluation"],
                "task_state": "completed",
            },
        )
        extractor = HeuristicFeatureExtractor()

        first = extractor.extract(node)
        second = extractor.extract(node)

        self.assertEqual(first, second)
        self.assertEqual(first.keywords[0], "cache")
        self.assertEqual(first.node_type, "experiment_result")
        self.assertEqual(first.topics, ("cache", "evaluation"))
        self.assertEqual(first.task_state, "completed")

    def test_model_extractor_validates_and_converts_json(self):
        output = (
            '{"summary":"cache result","topics":["cache"],'
            '"entities":["vLLM"],"claims":["hits increased"],'
            '"keywords":["prefix"],"node_type":"experiment_result",'
            '"task_state":"completed","language":"en"}'
        )
        extractor = ModelFeatureExtractor(lambda prompt: output, name="model-a")

        features = extractor.extract(Node("untrusted content"))

        self.assertEqual(features.summary, "cache result")
        self.assertEqual(features.topics, ("cache",))
        self.assertEqual(features.entities, ("vLLM",))
        self.assertEqual(features.node_type, "experiment_result")
        self.assertIn("untrusted content", extractor.prompt(Node("untrusted content")))

    def test_model_extractor_rejects_invalid_schema(self):
        malformed = ModelFeatureExtractor(lambda prompt: "not json")
        unknown = ModelFeatureExtractor(
            lambda prompt: '{"summary":"ok","unexpected":true}'
        )
        with self.assertRaises(ValueError):
            malformed.extract(Node("content"))
        with self.assertRaises(ValueError):
            unknown.extract(Node("content"))

    def test_empty_node_has_an_indexable_fallback(self):
        features = HeuristicFeatureExtractor().extract(Node(""))
        self.assertEqual(features.summary, "(empty node)")


class FeatureStoreTest(unittest.TestCase):
    def test_better_source_replaces_heuristic(self):
        node = Node("content")
        store = FeatureStore()
        heuristic = FeatureRecord(
            node.id,
            NodeFeatures("fallback"),
            extractor="heuristic-v1",
            source="heuristic",
        )
        model = FeatureRecord(
            node.id,
            NodeFeatures("semantic summary"),
            extractor="model-a",
            source="model",
        )

        store.set(heuristic)
        store.set(model)
        store.set(heuristic)

        self.assertEqual(store.get(node.id), model)

    def test_entries_returns_a_snapshot(self):
        node = Node("content")
        record = FeatureRecord(
            node.id,
            NodeFeatures("summary"),
            extractor="model-a",
        )
        store = FeatureStore()
        store.set(record)

        snapshot = store.entries()
        snapshot.clear()

        self.assertEqual(store.get(node.id), record)


class FeatureIndexScenarioTest(unittest.TestCase):
    def test_background_indexing_populates_features_and_catalog_summary(self):
        extractor = HeuristicFeatureExtractor()
        session = Session(feature_extractor=extractor)
        node = session.register(
            "Prefix caching reuses stable prompt blocks.",
            meta={"topics": ["cache"]},
        )
        original_id = node.id

        record = session.index_node(node.id)
        context = session.expand(refs=[], candidates=[node.id])

        self.assertEqual(node.id, original_id)
        self.assertIs(session.features.get(node.id), record)
        self.assertIn(record.features.summary, context.text)
        self.assertEqual(record.features.topics, ("cache",))

    def test_batch_indexing_deduplicates_and_commits_after_extraction(self):
        session = Session(feature_extractor=HeuristicFeatureExtractor())
        first = session.register("first routing document")
        second = session.register("second routing document")

        records = session.index_nodes(
            [first.id, second.id, first.id], max_workers=2
        )

        self.assertEqual(
            [record.node_id for record in records], [first.id, second.id]
        )
        self.assertIn(first.id, session.features)
        self.assertIn(second.id, session.features)
        self.assertEqual(
            session.summaries.get(first.id), records[0].features.summary
        )

    def test_batch_indexing_does_not_commit_a_partial_batch(self):
        class Extractor:
            name = "failing"
            source = "model"

            def extract(self, node):
                if node.content == "bad":
                    raise ValueError("model output invalid")
                return NodeFeatures("good summary")

        session = Session(feature_extractor=Extractor())
        good = session.register("good")
        bad = session.register("bad")

        with self.assertRaises(ValueError):
            session.index_nodes([good.id, bad.id], max_workers=2)

        self.assertNotIn(good.id, session.features)
        self.assertNotIn(bad.id, session.features)

    def test_batch_indexing_rejects_invalid_worker_count(self):
        session = Session(feature_extractor=HeuristicFeatureExtractor())

        with self.assertRaises(ValueError):
            session.index_nodes([], max_workers=0)

    def test_registration_does_not_run_extractor(self):
        class Extractor:
            name = "counting"
            source = "model"

            def __init__(self):
                self.calls = 0

            def extract(self, node):
                self.calls += 1
                return NodeFeatures("summary")

        extractor = Extractor()
        session = Session(feature_extractor=extractor)

        node = session.register("foreground write")

        self.assertEqual(extractor.calls, 0)
        session.index_node(node.id)
        self.assertEqual(extractor.calls, 1)


if __name__ == "__main__":
    unittest.main()
