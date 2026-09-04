"""Tests for the shared benchmark report schema."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "bench")

from measurement import (  # noqa: E402
    SCHEMA_VERSION,
    mean,
    new_report,
    save_report,
    session_counters,
)
from eval_longbench import document_for, instruction_for, run_condition  # noqa: E402
from model_backends import load_tokenize  # noqa: E402
from bench_real_cache import summarize  # noqa: E402


class MeasurementLeafTest(unittest.TestCase):
    def test_mean_handles_values_and_empty_input(self):
        self.assertEqual(mean([1, 2, 3]), 2.0)
        self.assertEqual(mean([]), 0.0)

    def test_session_counters_uses_stable_names(self):
        session = type(
            "FakeSession",
            (),
            {
                "expands": 2,
                "page_faults": 1,
                "requires_issued": 3,
                "requires_rejected": 1,
                "catalog_chars": 42,
            },
        )()
        self.assertEqual(
            session_counters(session),
            {
                "expands": 2,
                "page_faults": 1,
                "requires_issued": 3,
                "requires_rejected": 1,
                "catalog_chars": 42,
            },
        )


class MeasurementIntegrationTest(unittest.TestCase):
    def test_document_budget_preserves_question_and_instruction(self):
        tokenize, decode = load_tokenize("char")
        record = {
            "context": "x" * 200,
            "input": "where?",
            "dataset": "triviaqa_e",
        }
        limit = 80
        document = document_for(record, tokenize, decode, limit)
        suffix = "\n\nwhere?\n\n" + instruction_for(record)

        self.assertEqual(len(document) + len(suffix), limit)
        self.assertTrue((document + suffix).endswith(instruction_for(record)))

    def test_real_cache_summary_includes_latency(self):
        stats = [
            {"prompt_tokens": 100, "cached_tokens": 25, "latency_seconds": 0.2},
            {"prompt_tokens": 100, "cached_tokens": 75, "latency_seconds": 0.4},
        ]
        aggregate = summarize("protocol", stats, 200, 100)

        self.assertEqual(aggregate["hit_rate"], 0.5)
        self.assertAlmostEqual(aggregate["mean_latency_seconds"], 0.3)
        self.assertAlmostEqual(aggregate["total_latency_seconds"], 0.6)

    def test_condition_records_prompt_tokens_and_score(self):
        tokenize, decode = load_tokenize("char")
        records = [
            {"_id": "sample", "dataset": "triviaqa_e", "answers": ["answer"]}
        ]
        result = run_condition(
            "full",
            lambda rec, tok, dec, limit: "answer prompt",
            records,
            lambda prompt: "answer",
            tokenize,
            decode,
            100,
        )

        mean_score, scores, observations = result
        self.assertEqual(mean_score, 1.0)
        self.assertEqual(scores, [1.0])
        self.assertEqual(observations[0]["prompt_tokens"], len("answer prompt"))


class MeasurementReportTest(unittest.TestCase):
    def test_report_can_be_saved_and_resumed_as_valid_json(self):
        report = new_report(
            "smoke", "scripted", {"limit": 2}, model="deterministic"
        )
        report["observations"].append({"sample_id": "one", "score": 1.0})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            save_report(report, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
        self.assertEqual(loaded["experiment"], "smoke")
        self.assertEqual(loaded["backend"]["model"], "deterministic")
        self.assertIn("commit", loaded["revision"])
        self.assertEqual(loaded["observations"][0]["sample_id"], "one")


if __name__ == "__main__":
    unittest.main()
