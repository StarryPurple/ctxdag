"""Tests for the side-table summary store."""

import unittest

from contextdag import SummaryStore


class SummaryStoreTest(unittest.TestCase):
    def test_service_fills_missing(self):
        store = SummaryStore()
        store.set("nid", "摘要", source="service")
        self.assertEqual(store.get("nid"), "摘要")

    def test_better_source_wins(self):
        store = SummaryStore()
        store.set("nid", "service 摘要", source="service")
        store.set("nid", "explicit 摘要", source="explicit")
        self.assertEqual(store.get("nid"), "explicit 摘要")

    def test_reader_beats_explicit(self):
        store = SummaryStore()
        store.set("nid", "explicit 摘要", source="explicit")
        store.set("nid", "reader 摘要", source="reader")
        self.assertEqual(store.get("nid"), "reader 摘要")

    def test_equal_rank_refreshes(self):
        store = SummaryStore()
        store.set("nid", "旧摘要", source="service")
        store.set("nid", "新摘要", source="service")
        self.assertEqual(store.get("nid"), "新摘要")

    def test_blank_ignored(self):
        store = SummaryStore()
        store.set("nid", "  \n ", source="service")
        self.assertIsNone(store.get("nid"))
        self.assertNotIn("nid", store)

    def test_unknown_source_never_overrides_known(self):
        store = SummaryStore()
        store.set("nid", "explicit 摘要", source="explicit")
        store.set("nid", "weird 摘要", source="unknown")
        self.assertEqual(store.get("nid"), "explicit 摘要")


if __name__ == "__main__":
    unittest.main()
