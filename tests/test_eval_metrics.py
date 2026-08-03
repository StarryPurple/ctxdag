"""Tests for answer-quality metrics."""

import sys
import unittest

sys.path.insert(0, "bench")

from eval_metrics import (  # noqa: E402
    accuracy,
    exact_match,
    f1_score,
    normalize_answer,
    rouge_l_f1,
    score_prediction,
)


class EvalMetricsTest(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer("The Answer!?"), "answer")
        self.assertEqual(normalize_answer("  Hello   World "), "hello world")

    def test_f1(self):
        self.assertEqual(f1_score("a b c", "a b c"), 1.0)
        self.assertEqual(f1_score("a b", "x y"), 0.0)
        self.assertGreater(f1_score("a b c", "a b"), 0.0)
        self.assertLess(f1_score("a b c", "a b"), 1.0)

    def test_exact_match(self):
        self.assertEqual(exact_match("Answer", "answer"), 1.0)
        self.assertEqual(exact_match("a", "b"), 0.0)

    def test_rouge_l(self):
        self.assertEqual(rouge_l_f1("a b c", "a b c"), 1.0)
        self.assertEqual(rouge_l_f1("a b c", "x y z"), 0.0)
        self.assertGreater(rouge_l_f1("a b c d", "a b c"), 0.0)

    def test_accuracy(self):
        self.assertEqual(accuracy("yes", ["yes", "no"]), 1.0)
        self.assertEqual(accuracy("maybe", ["yes", "no"]), 0.0)

    def test_score_prediction_mapping(self):
        self.assertAlmostEqual(score_prediction("triviaqa_e", "the answer", ["the answer"]), 1.0)
        self.assertAlmostEqual(score_prediction("multi_news_e", "a b", ["a b"]), 1.0)
        self.assertAlmostEqual(score_prediction("trec_e", "yes", ["yes"]), 1.0)


if __name__ == "__main__":
    unittest.main()
