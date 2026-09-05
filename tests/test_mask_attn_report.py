"""Aggregation tests for measured MaskAttn results."""

from copy import deepcopy
import sys
import unittest

sys.path.insert(0, "bench")
from eval_mask_attn import summarize


class MaskReportTest(unittest.TestCase):
    def test_empty_report_does_not_claim_success(self):
        self.assertEqual(summarize([]), {"n": 0})

    def test_token_weighting_and_output_agreement_differ_from_quality(self):
        row = {
            "outputs": {"dag_full": "a", "dag_reuse": "a"},
            "exact_match": {"dag_full": False, "dag_reuse": False},
            "reuse_vs_full": {"next_token_equal": True, "max_logit_error": 0.2},
            "same_full_prefix_vs_full": {"max_logit_error": 0.1},
            "input_tokens": 10, "new_query_tokens": 2, "cached_prefix_tokens": 8,
            "dag_full_seconds": 1, "causal_seconds": 1,
            "cache_build_seconds": 2, "cache_assembly_seconds": 0.1,
            "warm_query_seconds": 0.5,
        }
        second = deepcopy(row)
        second.update(input_tokens=30, new_query_tokens=12, cached_prefix_tokens=18)
        second["outputs"]["dag_reuse"] = "b"
        result = summarize([row, second])
        self.assertEqual(result["cached_token_fraction"], 26 / 40)
        self.assertEqual(result["warm_query_tokens"], 14)
        self.assertEqual(result["reuse_output_agreement"], 0.5)
        self.assertEqual(result["exact_match"]["dag_full"], 0)
        self.assertEqual(result["mean_seconds"]["cache_build_seconds"], 2)
