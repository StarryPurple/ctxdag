"""Tests for the token-reuse accounting (radix-cache proxy)."""

import unittest

from contextdag import Accountant, lcp_len


def char_ids(text: str) -> list[int]:
    return [ord(c) for c in text]


class AccountingTest(unittest.TestCase):
    def test_lcp_len(self):
        self.assertEqual(lcp_len([1, 2, 3], [1, 2, 3, 4]), 3)
        self.assertEqual(lcp_len([1, 2], [9, 2]), 0)
        self.assertEqual(lcp_len([], [1]), 0)

    def test_cached_is_longest_common_prefix(self):
        acct = Accountant(char_ids)
        first = acct.account("abc")
        second = acct.account("abcd")
        self.assertEqual(first.cached_tokens, 0)
        self.assertEqual(second.cached_tokens, 3)
        self.assertEqual(second.new_tokens, 1)
        self.assertAlmostEqual(second.hit_rate, 0.75)

    def test_cached_against_any_seen_context(self):
        acct = Accountant(char_ids)
        acct.account("abcd")
        acct.account("wxyz")
        third = acct.account("wxy")
        self.assertEqual(third.cached_tokens, 3)  # prefix of "wxyz"

    def test_catalog_tokens_counted(self):
        acct = Accountant(char_ids)
        stats = acct.account("abc", catalog_text="catalog!")
        self.assertEqual(stats.catalog_tokens, 8)
        # catalog_text is the tail slice of ``text`` (sub-count), not extra
        self.assertEqual(stats.prompt_tokens, 3)

    def test_totals_and_overall_hit_rate(self):
        acct = Accountant(char_ids)
        acct.account("abc")
        acct.account("abcd")
        self.assertEqual(acct.total_prompt_tokens, 7)
        self.assertEqual(acct.total_cached_tokens, 3)
        self.assertAlmostEqual(acct.overall_hit_rate, 3 / 7)

    def test_node_stats_cached_per_node(self):
        acct = Accountant(char_ids)
        blocks = {"n1": "block-one"}
        acct.account("block-one", order=["n1"], node_blocks=lambda nid: blocks[nid])
        acct.account("block-one!", order=["n1"], node_blocks=lambda nid: blocks[nid])
        stat = acct.nodes["n1"]
        self.assertEqual(stat.block_tokens, 9)  # tokenized once
        self.assertEqual(stat.appearances, 2)

    def test_empty_text(self):
        acct = Accountant(char_ids)
        stats = acct.account("")
        self.assertEqual(stats.prompt_tokens, 0)
        self.assertEqual(stats.hit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
