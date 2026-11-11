"""`scripts/power_search.py` の検出力計算・fallback・CLI のテスト。

必要ペア数は標準ライブラリだけから決定論的に決まる。ここでは
ceil / 境界 / train JSONL からの sigma 推定 / 不足・無効時の 0.5 fallback /
CLI 出力を固定する(sol SOL-P25G-007)。
"""
import contextlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.power_search import (  # noqa: E402
    FALLBACK_SIGMA, classify_records, collect_pair_scores, main,
    normal_quantile, required_pairs, sigma_from_records,
)


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def write_records(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


class RequiredPairsTests(unittest.TestCase):

    def test_default_is_two_sided_alpha_005_delta_005_power_080(self):
        # (1.959964 + 0.841621) * 0.5 / 0.05 = 28.01585 -> ceil 785。
        self.assertEqual(required_pairs(), 785)
        self.assertEqual(required_pairs(0.05, 0.5, 0.05, 0.80, 2), 785)

    def test_result_is_ceil_not_floor(self):
        raw = ((normal_quantile(0.975) + normal_quantile(0.8)) * 0.5 / 0.05) ** 2
        self.assertGreater(raw, 784.0)
        self.assertLess(raw, 785.0)
        self.assertEqual(required_pairs(), 785)

    def test_exact_integer_stays_exact(self):
        sigma = 0.05 * math.sqrt(100.0) / (normal_quantile(0.975)
                                           + normal_quantile(0.8))
        self.assertEqual(required_pairs(0.05, sigma), 100)

    def test_two_sided_is_more_conservative_than_one_sided(self):
        self.assertEqual(required_pairs(0.05, 0.5, 0.05, 0.80, 1), 619)
        self.assertGreater(required_pairs(0.05, 0.5, 0.05, 0.80, 2),
                           required_pairs(0.05, 0.5, 0.05, 0.80, 1))

    def test_deterministic(self):
        first = [required_pairs(0.05, 0.2, 0.05, 0.80, 2) for unused in range(3)]
        second = [required_pairs(0.05, 0.2, 0.05, 0.80, 2) for unused in range(3)]
        self.assertEqual(first, second)

    def test_sigma_shrinks_required_pairs(self):
        self.assertLess(required_pairs(0.05, 0.1199),
                        required_pairs(0.05, 0.5))


class BoundaryTests(unittest.TestCase):

    def test_invalid_delta(self):
        for delta in (0.0, -0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                required_pairs(delta=delta)

    def test_invalid_sigma(self):
        for sigma in (0.0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                required_pairs(sigma=sigma)

    def test_invalid_alpha_and_power(self):
        for alpha in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                required_pairs(alpha=alpha)
        for power in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                required_pairs(power=power)

    def test_invalid_sides(self):
        for sides in (0, 3, -1):
            with self.assertRaises(ValueError):
                required_pairs(sides=sides)


class RecordsSigmaTests(unittest.TestCase):

    def test_sigma_matches_statistics_stdev(self):
        scores_a = [1.0, 1.0, 0.5, 0.0, 1.0, 1.0]
        scores_b = [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [
                {"label": "a", "left": "pimc", "right": "heuristic",
                 "budget": 4500, "pair_scores": scores_a},
                {"label": "b", "left": "ismcts", "right": "heuristic",
                 "budget": 4500, "pair_scores": scores_b},
            ])
            all_scores = scores_a + scores_b
            self.assertEqual(sigma_from_records(path),
                             statistics.stdev(all_scores))
            self.assertEqual(
                sigma_from_records(path, matchup="ismcts-vs-heuristic"),
                statistics.stdev(scores_b))
            self.assertEqual(
                sigma_from_records(path, label="a", budget=4500),
                statistics.stdev(scores_a))
        finally:
            os.unlink(path)

    def test_no_matching_rows_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [
                {"label": "a", "left": "pimc", "right": "heuristic",
                 "budget": 4500, "pair_scores": [1.0, 0.5]},
            ])
            self.assertIsNone(sigma_from_records(path, matchup="not-there"))
            self.assertIsNone(sigma_from_records(path, budget=1500))
        finally:
            os.unlink(path)

    def test_single_score_is_insufficient(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [
                {"label": "a", "pair_scores": [0.5]},
            ])
            self.assertIsNone(sigma_from_records(path))
        finally:
            os.unlink(path)

    def test_missing_or_invalid_records_return_none(self):
        self.assertIsNone(sigma_from_records("no-such-file.jsonl"))
        cases = [
            "not json",
            json.dumps({"pair_scores": [1.5]}),
            json.dumps({"pair_scores": ["1.0"]}),
            json.dumps({"pair_scores": [float("nan")]}),
            json.dumps({"pair_scores": []}),
            json.dumps({"label": "x"}),
        ]
        for content in cases:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                             delete=False) as handle:
                handle.write(content)
                path = handle.name
            try:
                self.assertIsNone(sigma_from_records(path))
            finally:
                os.unlink(path)


class FallbackTests(unittest.TestCase):

    def test_cli_without_records_uses_fallback(self):
        code, out, err = run_cli(["--delta", "0.05", "--power", "0.80"])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("fallback upper bound", out)
        self.assertIn("required_pairs=785", out)

    def test_cli_missing_records_falls_back(self):
        code, out, err = run_cli([
            "--records", "no-such-file.jsonl",
            "--matchup", "pimc-vs-heuristic",
            "--label", "default", "--budget", "4500"])
        self.assertEqual(code, 0)
        self.assertIn("fallback", out)
        self.assertIn("required_pairs=785", out)

    def test_cli_invalid_records_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            handle.write(json.dumps({
                "label": "default", "left": "pimc", "right": "heuristic",
                "budget": 4500, "pair_scores": [2.0]}) + "\n")
            path = handle.name
        try:
            with self.assertRaises(SystemExit):
                run_cli(["--records", path, "--matchup", "pimc-vs-heuristic",
                         "--label", "default", "--budget", "4500"])
        finally:
            os.unlink(path)

    def test_cli_corrupt_records_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            handle.write("not json\n")
            path = handle.name
        try:
            with self.assertRaises(SystemExit):
                run_cli(["--records", path, "--matchup", "pimc-vs-heuristic",
                         "--label", "default", "--budget", "4500"])
        finally:
            os.unlink(path)

    def test_cli_records_requires_all_config_filters(self):
        scores = [1.0, 0.5, 0.0, 1.0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [{"label": "default", "left": "pimc",
                                  "right": "heuristic", "budget": 4500,
                                  "pair_scores": scores}])
            for argv in (["--records", path],
                         ["--records", path, "--matchup", "pimc-vs-heuristic"],
                         ["--records", path, "--label", "default",
                          "--budget", "4500"]):
                with self.assertRaises(SystemExit):
                    run_cli(argv)
        finally:
            os.unlink(path)

    def test_cli_no_matching_rows_still_falls_back(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [{"label": "default", "left": "pimc",
                                  "right": "heuristic", "budget": 4500,
                                  "pair_scores": [1.0, 0.5]}])
            code, out, err = run_cli([
                "--records", path, "--matchup", "ismcts-vs-heuristic",
                "--label", "other", "--budget", "4500"])
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        self.assertIn("fallback", out)
        self.assertIn("required_pairs=785", out)

    def test_cli_valid_records_uses_sample_stdev(self):
        scores = [1.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 1.0]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [
                {"label": "default", "left": "ismcts", "right": "heuristic",
                 "budget": 4500, "pair_scores": scores},
            ])
            expected = required_pairs(0.05, statistics.stdev(scores))
            code, out, err = run_cli([
                "--records", path, "--matchup", "ismcts-vs-heuristic",
                "--label", "default", "--budget", "4500"])
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        self.assertIn("sample stdev", out)
        self.assertIn("required_pairs=%d" % expected, out)

    def test_explicit_sigma_wins_over_records(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [{"pair_scores": [0.0, 1.0]}])
            code, out, err = run_cli(["--records", path, "--sigma", "0.5"])
        finally:
            os.unlink(path)
        self.assertEqual(code, 0)
        self.assertIn("explicit --sigma", out)
        self.assertIn("required_pairs=785", out)


class CliTests(unittest.TestCase):

    def test_cli_rejects_boundary_violations(self):
        for argv in (["--delta", "0"], ["--sigma", "0"], ["--alpha", "1"],
                     ["--power", "1"], ["--sides", "3"]):
            with self.assertRaises(SystemExit):
                main(argv)

    def test_cli_prints_two_sided_criterion(self):
        code, out, err = run_cli(["--delta", "0.05"])
        self.assertEqual(code, 0)
        self.assertIn("(two-sided)", out)
        self.assertIn("z_alpha=1.9600", out)
        self.assertIn("z_beta=0.8416", out)

    def test_cli_output_is_deterministic(self):
        first = run_cli([])
        second = run_cli([])
        self.assertEqual(first, second)
        self.assertIn("required_pairs=785", first[1])

    def test_classify_records_distinguishes_failure_modes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            path = handle.name
        try:
            write_records(path, [{"label": "a", "left": "pimc",
                                  "right": "heuristic", "budget": 4500,
                                  "pair_scores": [1.0, 0.5]}])
            self.assertEqual(classify_records(path, "pimc-vs-heuristic",
                                              "a", 4500), "valid")
            self.assertEqual(classify_records(path, "ismcts-vs-heuristic",
                                              "a", 4500), "no_match")
            self.assertEqual(classify_records("no-such.jsonl"), "missing")
        finally:
            os.unlink(path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            handle.write("broken\n")
            corrupt = handle.name
        try:
            self.assertEqual(classify_records(corrupt), "corrupt")
        finally:
            os.unlink(corrupt)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as handle:
            handle.write("[]\n")
            non_dict = handle.name
        try:
            self.assertEqual(classify_records(non_dict), "corrupt")
            self.assertIsNone(sigma_from_records(non_dict))
        finally:
            os.unlink(non_dict)


if __name__ == "__main__":
    unittest.main()
