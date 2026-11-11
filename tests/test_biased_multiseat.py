"""多席 harness / ladder で biased:<kind> 席を座らせる(W2-C-BIASEDSEAT)。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from naeco.ai.biased_agent import BIASED_KINDS  # noqa: E402
from naeco.ai.eval import harness  # noqa: E402

import ladder  # noqa: E402


def _lineup(n, kinds):
    base = ["biased:%s" % k for k in kinds]
    return (base + ["heuristic", "random"])[:n]


class BiasedSeatTests(unittest.TestCase):

    def test_parse(self):
        for kind in BIASED_KINDS:
            self.assertEqual(
                harness.parse_multi_agent_spec("biased:" + kind),
                ("biased", kind))

    def test_unknown_kind_is_error(self):
        for spec in ("biased:nope", "biased:", "biased"):
            with self.assertRaises(ValueError):
                harness.parse_multi_agent_spec(spec)

    def test_other_specs_unchanged(self):
        self.assertEqual(harness.parse_multi_agent_spec("heuristic"),
                         ("heuristic", ""))
        with self.assertRaises(ValueError):
            harness.parse_multi_agent_spec("checkpoint:x.json")

    def _run(self, n, lineup):
        cond = harness.multi_conditions(n, lineup, "train", 0, 2,
                                        "test-biased", None)
        run_id = harness.multi_run_id(cond)
        tasks = harness.cyclic_tasks(run_id, "train", 0, 2, lineup, n,
                                     models=cond.get("models"))
        return harness.run_cyclic_games(tasks, workers=1)

    def test_n3_all_biased_kinds_play_to_end_and_deterministic(self):
        for kind in BIASED_KINDS:
            lineup = _lineup(3, [kind])
            first = self._run(3, lineup)
            second = self._run(3, lineup)
            self.assertEqual(len(first), 6)       # 2 clusters x 3 shifts
            self.assertEqual(first, second)

    def test_n4_biased_seats_play_to_end_and_deterministic(self):
        lineup = ["biased:avoid_hiring", "biased:mine_first",
                  "biased:school_first", "heuristic"]
        first = self._run(4, lineup)
        second = self._run(4, lineup)
        self.assertEqual(len(first), 8)
        self.assertEqual(first, second)

    def test_ladder_diagnostic_accepts_biased(self):
        summary, records = ladder.run_multi_diagnostic(
            _lineup(3, ["school_first"]), 3, 0, 1, 1)
        self.assertEqual(len(summary["rows"]), 3)
        self.assertTrue(records)


if __name__ == "__main__":
    unittest.main()
