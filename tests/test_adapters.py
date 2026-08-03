"""Smoke tests for the real-dataset benchmark adapters."""

import sys
import unittest

sys.path.insert(0, "bench")

from adapters import longbench, taubench  # noqa: E402
from workflows import BenchSession  # noqa: E402


class AdaptersTest(unittest.TestCase):
    def test_taubench_replay_builds_a_chain(self):
        builder = next(taubench.iter_samples(limit=1))
        bench = BenchSession()
        builder(bench)
        self.assertGreater(len(bench.ctxs), 5)  # multi-turn trajectory

    def test_longbench_progressive_reads(self):
        builder = next(longbench.iter_samples(limit=1))
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
