"""B2 学習用相手プールと固定ベンチマーク群のテスト(Phase 3 設計書 §4.5 / §7)。

「歯」の狙い: 抽選が "pool" stream 由来であること・weight 比例・学習用プールと
固定ベンチマーク群の型分離・不正入力の拒否を、実装から1つ外すと落ちる形で
固定する。torch は import しない(標準ライブラリのみ)。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.engine.state import derive_seed  # noqa: E402
from naeco.train import pool  # noqa: E402


def checkpoint_entry(name, weight=1.0, path=None):
    if path is None:
        path = "meta/%s.json" % name
    return pool.PoolEntry(name, "checkpoint", path, weight)


class SampleDeterminismTests(unittest.TestCase):

    def setUp(self):
        self.entries = [
            checkpoint_entry("gen007", 1.0),
            checkpoint_entry("gen006", 2.0),
            pool.PoolEntry("heuristic", "heuristic", "", 3.0),
            pool.PoolEntry("random", "random", "", 4.0),
        ]
        self.op = pool.OpponentPool(self.entries)

    def test_same_coordinates_are_deterministic(self):
        first = self.op.sample(20260811, 0)
        for _ in range(20):
            self.assertEqual(self.op.sample(20260811, 0), first)
        again = pool.OpponentPool(self.entries)
        self.assertEqual(again.sample(20260811, 0), first)

    def test_sample_returns_one_of_the_pool_entries(self):
        for index in range(50):
            picked = self.op.sample(20260811, index)
            self.assertIn(picked, self.entries)


class WeightProportionTests(unittest.TestCase):

    SAMPLES = 30000
    ROOT_SEED = 2026081102

    def setUp(self):
        self.entries = [
            checkpoint_entry("gen007", 1.0),
            checkpoint_entry("gen006", 2.0),
            pool.PoolEntry("heuristic", "heuristic", "", 3.0),
        ]
        self.op = pool.OpponentPool(self.entries)
        self.expected_ratios = {
            "gen007": 1.0 / 6.0,
            "gen006": 2.0 / 6.0,
            "heuristic": 3.0 / 6.0,
        }

    def test_empirical_distribution_follows_weights(self):
        counts = dict.fromkeys(self.expected_ratios, 0)
        for index in range(self.SAMPLES):
            counts[self.op.sample(self.ROOT_SEED, index).name] += 1
        for name, expected in self.expected_ratios.items():
            observed = counts[name] / float(self.SAMPLES)
            self.assertAlmostEqual(observed, expected, delta=0.01)


class PoolStreamTests(unittest.TestCase):

    def setUp(self):
        self.entries = [
            checkpoint_entry("gen007", 1.0),
            pool.PoolEntry("random", "random", "", 1.0),
        ]
        self.op = pool.OpponentPool(self.entries)

    def test_sample_derives_seed_from_pool_stream(self):
        calls = []

        def spy(root_seed, stream_kind, episode_index, seat_variant=0):
            calls.append((root_seed, stream_kind, episode_index, seat_variant))
            return derive_seed(
                root_seed, stream_kind, episode_index, seat_variant)

        with mock.patch.object(pool, "derive_seed", side_effect=spy):
            picked = self.op.sample(1234, 17)
        self.assertEqual(calls, [(1234, "pool", 17, 0)])
        self.assertIn(picked, self.entries)

    def test_changing_stream_kind_changes_selection(self):
        expected = self.op.sample(4321, 9)

        def seat_stream(root_seed, stream_kind, episode_index, seat_variant=0):
            return derive_seed(
                root_seed, "seat", episode_index, seat_variant)

        with mock.patch.object(pool, "derive_seed", side_effect=seat_stream):
            changed = self.op.sample(4321, 9)
        self.assertNotEqual(changed, expected)


class TrainingPoolTests(unittest.TestCase):

    def test_mixed_ratio_is_configurable(self):
        latest = checkpoint_entry("gen010", 3.0)
        past = [
            checkpoint_entry("gen009", 2.0),
            checkpoint_entry("gen008", 1.0),
        ]
        op = pool.training_pool(
            latest, past, heuristic_weight=4.0, random_weight=2.0)
        expected = {
            "gen010": 3.0 / 12.0,
            "gen009": 2.0 / 12.0,
            "gen008": 1.0 / 12.0,
            "heuristic": 4.0 / 12.0,
            "random": 2.0 / 12.0,
        }
        counts = dict.fromkeys(expected, 0)
        for index in range(60000):
            counts[op.sample(9876, index).name] += 1
        for name, ratio in expected.items():
            observed = counts[name] / 60000.0
            self.assertAlmostEqual(observed, ratio, delta=0.01)

    def test_zero_past_generations_still_works(self):
        op = pool.training_pool(
            checkpoint_entry("gen005", 1.0), [], heuristic_weight=1.0)
        sampled = {op.sample(1111, index).name for index in range(3000)}
        self.assertEqual(sampled, {"gen005", "heuristic"})

    def test_rejects_non_checkpoint_latest(self):
        bad = pool.PoolEntry("heuristic", "heuristic", "", 1.0)
        with self.assertRaises(ValueError):
            pool.training_pool(bad, [])

    def test_rejects_non_checkpoint_past_generation(self):
        latest = checkpoint_entry("gen005")
        bad = pool.PoolEntry("random", "random", "", 1.0)
        with self.assertRaises(ValueError):
            pool.training_pool(latest, [bad])


class BenchmarkSuiteTests(unittest.TestCase):

    def setUp(self):
        self.generations = [
            checkpoint_entry("gen009"),
            checkpoint_entry("gen007"),
            pool.PoolEntry("heuristic", "heuristic", "", 1.0),
        ]

    def test_suite_is_not_opponent_pool(self):
        suite = pool.benchmark_suite(self.generations)
        self.assertIsInstance(suite, tuple)
        self.assertNotIsInstance(suite, pool.OpponentPool)
        self.assertFalse(hasattr(suite, "sample"))

    def test_suite_preserves_fixed_order(self):
        first = pool.benchmark_suite(self.generations)
        second = pool.benchmark_suite(list(self.generations))
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(entry.name for entry in first),
            ("gen009", "gen007", "heuristic"))


class ValidationTests(unittest.TestCase):

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            pool.PoolEntry("alien", "alien", "", 1.0)

    def test_non_positive_weight_raises(self):
        for weight in (0.0, -1.0):
            with self.assertRaises(ValueError):
                pool.PoolEntry(
                    "gen007", "checkpoint", "meta/gen007.json", weight)

    def test_checkpoint_without_path_raises(self):
        with self.assertRaises(ValueError):
            pool.PoolEntry("gen007", "checkpoint", "", 1.0)

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError):
            pool.OpponentPool([])

    def test_empty_benchmark_suite_raises(self):
        with self.assertRaises(ValueError):
            pool.benchmark_suite([])

    def test_non_pool_entry_raises(self):
        with self.assertRaises(ValueError):
            pool.OpponentPool(["gen007"])


if __name__ == "__main__":
    unittest.main()
