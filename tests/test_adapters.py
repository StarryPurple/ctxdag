"""Smoke tests for the real-dataset benchmark adapters."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "bench")

from adapters import longbench, taubench  # noqa: E402
from workflows import BenchSession  # noqa: E402


class AdaptersTest(unittest.TestCase):
    def test_taubench_replay_builds_a_chain(self):
        trajectory = {"traj": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "checking"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.json"
            path.write_text(json.dumps([trajectory]), encoding="utf-8")
            builder = next(taubench.iter_samples(limit=1, path=str(path)))
            bench = BenchSession()
            builder(bench)
        self.assertGreater(len(bench.ctxs), 5)  # multi-turn trajectory

    def test_longbench_progressive_reads(self):
        record = {"context": "first\n\nsecond\n\nthird"}
        with tempfile.TemporaryDirectory() as directory:
            template = str(Path(directory) / "{dataset}.jsonl")
            Path(template.format(dataset="fixture")).write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            original = longbench.LONGBENCH_PATH
            try:
                longbench.LONGBENCH_PATH = template
                builder = next(longbench.iter_samples(
                    dataset="fixture", limit=1, max_sections=8
                ))
            finally:
                longbench.LONGBENCH_PATH = original
            bench = BenchSession()
            builder(bench)
        self.assertGreaterEqual(len(bench.ctxs), 2)  # doc split into sections

    def test_split_sections_handles_newline_char(self):
        parts = longbench.split_sections(
            "Passage 1: a\nNEWLINE_CHAR NEWLINE_CHAR Passage 2: b\n\nPassage 3: c",
            8,
        )
        self.assertEqual(parts, ["Passage 1: a", "Passage 2: b", "Passage 3: c"])


if __name__ == "__main__":
    unittest.main()
