"""Phase 9 C1: deterministic fitting, monotonic maps, ECE and safe loading."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from naeco.ai.calibration import Identity, Platt, ece, ece_with_bins, fit_platt, load_calibrator
from scripts.fit_calibration import build_report, collect_pairs

REPORT_INPUT = "docs/eval-runs/phase7-step7/calib-gen1300.json"
INPUT = ROOT / REPORT_INPUT
ARTIFACT = ROOT / "docs/eval-runs/phase9/calibration-gen1300.json"

# Mac-generated artifact vs a recomputation on another platform (csi/Linux):
# finite floats can drift in the last bits, so compare them with a strict
# abs_tol=rel_tol=1e-12. Strings, ints, counts, seed ranges, hashes, provenance
# and schema stay exactly equal, and NaN is never accepted.
_FLOAT_REL_TOL = 1e-12
_FLOAT_ABS_TOL = 1e-12


def build_report_from_root(relative_input=REPORT_INPUT):
    """Reproduce the artifact with the recorded ROOT-relative source path.

    build_report stores os.fspath(input_path) as source.file, and the artifact
    records the relative path used when it was generated. Resolving that path
    against ROOT (instead of the process cwd) lets the reproduction work from
    any test cwd while source.file still matches the artifact exactly.
    """
    cwd = os.getcwd()
    os.chdir(str(ROOT))
    try:
        return build_report(relative_input)
    finally:
        os.chdir(cwd)


def assert_report_close(case, actual, expected):
    """Recursively compare a report to the artifact, tolerating only float drift."""
    if isinstance(expected, dict):
        case.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_report_close(case, actual[key], expected[key])
    elif isinstance(expected, list):
        case.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_report_close(case, left, right)
    elif isinstance(expected, float):
        case.assertIs(type(actual), float)
        case.assertTrue(math.isfinite(actual) and math.isfinite(expected),
                        "non-finite float: %r != %r" % (actual, expected))
        case.assertTrue(
            math.isclose(actual, expected, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL),
            "float drift beyond 1e-12: %r != %r" % (actual, expected))
    else:
        case.assertEqual(actual, expected)


def diagnostic():
    """Synthetic rows only: distinct fit/holdout values and actor perspectives."""
    games = []
    for index in range(502800, 503200):
        games.append({"group": "train", "index": index,
                      "utilities": [-1, 1], "rows": [
                          {"actor": 0, "pending": False, "u0": -1.5, "u1": 0.6},
                          {"actor": 1, "pending": False, "u0": 0.8, "u1": 1.4},
                          {"actor": 0, "pending": True, "u0": 0.4, "u1": 0.4},
                      ]})
    return {"summary": {"group": "train", "meta": "gen1300.meta.json"},
            "games": games}


class FitTests(unittest.TestCase):
    def test_recovers_known_coefficients_within_005(self):
        # Deterministic binomial counts, not random draws or soft-label fitting.
        for a, b in ((1.23, -0.29), (0.7, 0.4), (2.0, -0.6)):
            with self.subTest(a=a, b=b):
                pairs = []
                for x in (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2):
                    p = 1.0 / (1.0 + math.exp(-x))
                    wins = round(2000 / (1.0 + math.exp(-(a * x + b))))
                    pairs.extend([(p, 1)] * wins + [(p, 0)] * (2000 - wins))
                fitted = fit_platt(pairs)
                self.assertLess(abs(fitted.a - a), 0.05)
                self.assertLess(abs(fitted.b - b), 0.05)
                self.assertEqual(fitted, fit_platt(iter(pairs)))

    def test_half_targets_are_supported(self):
        fitted = fit_platt([(0.2, 0), (0.5, 0.5), (0.8, 1)] * 20)
        self.assertAlmostEqual(fitted.b, 0, places=7)
        self.assertAlmostEqual(fitted.apply(0.5), 0.5, places=7)

    def test_rejects_empty_degenerate_and_invalid_input(self):
        for pairs in ([], [(0.5, 0), (0.5, 1)], [(0.3, 0.2)],
                      [(float("nan"), 0)], [(1.1, 1)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                fit_platt(pairs)
        for iters in (0, -1, True, 1.5):
            with self.subTest(iters=iters), self.assertRaises(ValueError):
                fit_platt([(0.2, 0), (0.8, 1)], iters=iters)  # type: ignore[arg-type]


class MapTests(unittest.TestCase):
    def test_monotonicity_and_logit_clipping(self):
        calibrator = Platt(1.23, -0.29)
        inputs = [i / 1000.0 for i in range(1001)]
        outputs = [calibrator.apply(p) for p in inputs]
        self.assertTrue(all(left < right for left, right in zip(outputs, outputs[1:])))
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in outputs))
        self.assertEqual(calibrator.apply(0), calibrator.apply(1e-6))
        self.assertEqual(calibrator.apply(1), calibrator.apply(1 - 1e-6))
        self.assertAlmostEqual(calibrator.apply(0.5), 1 / (1 + math.exp(0.29)))

    def test_identity_preserves_endpoints_exactly_and_describes(self):
        for p in (0, 0.13, 0.5, 1):
            self.assertEqual(Identity().apply(p), p)
        self.assertEqual(Identity().describe(),
                         {"kind": "identity", "a": 1.0, "b": 0.0, "source": None})
        self.assertEqual(Platt(1.2, -0.3).describe(),
                         {"kind": "platt", "a": 1.2, "b": -0.3, "source": None})

    def test_invalid_coefficients_and_probabilities(self):
        for a, b in ((0, 0), (-1, 0), (True, 0), (1, float("inf")),
                     (float("nan"), 0), ("1", 0)):
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                Platt(a, b)  # type: ignore[arg-type]
        for p in (float("nan"), float("inf"), -0.1, 1.1, True):
            for calibrator in (Identity(), Platt(1, 0)):
                with self.subTest(p=p, calibrator=calibrator), self.assertRaises(ValueError):
                    calibrator.apply(p)

    def test_extreme_logits_do_not_overflow(self):
        self.assertEqual(Platt(1000, 1000).apply(1), 1)
        self.assertEqual(Platt(1000, -1000).apply(0), 0)


class EceTests(unittest.TestCase):
    def test_hand_computation_and_upper_endpoint(self):
        # Buckets 0, 1, 9: contributions 0, |0.29-0.5|/5, |1.9-2|/5.
        self.assertAlmostEqual(ece([(0, 0), (0.1, 0), (0.19, 0.5),
                                   (0.9, 1), (1, 1)]), 0.062)
        self.assertAlmostEqual(ece([(0.2, 0), (0.3, 1)], bins=1), 0.25)
        self.assertEqual(ece([]), 0)
        self.assertEqual(ece([(0, 0), (1, 1)]), 0)

    def test_shared_diagnostic_aggregation_keeps_frozen_values_and_bins(self):
        from scripts.value_calibration import _ece

        self.assertIs(_ece, ece_with_bins)
        pairs = [(0, 0), (0.1, 0), (0.19, 0.5), (0.9, 1), (1, 1)]
        expected = (0.06200000000000001, [
            {"bin": 0, "n": 1, "mean_p": 0.0, "mean_y": 0.0},
            {"bin": 1, "n": 2, "mean_p": 0.14500000000000002, "mean_y": 0.25},
            {"bin": 9, "n": 2, "mean_p": 0.95, "mean_y": 1.0},
        ])
        self.assertEqual(_ece(pairs), expected)
        self.assertEqual(ece(pairs), expected[0])
        self.assertEqual(ece_with_bins(iter(pairs)), expected)
        self.assertEqual(ece_with_bins([]), (0.0, []))
        self.assertEqual(ece_with_bins([(0, 0), (1, 1)], bins=1),
                         (0.0, [{"bin": 0, "n": 2, "mean_p": 0.5, "mean_y": 0.5}]))

    def test_saved_diagnostic_ece_and_tables_reproduce(self):
        from scripts.value_calibration import analyze

        raw = json.loads(INPUT.read_text())
        actual = analyze(raw["games"])
        for mapping in ("mapped", "antisymmetric"):
            for metric in ("ece", "bins"):
                assert_report_close(self, actual[mapping][metric],
                                    raw["summary"][mapping][metric])

    def test_bin_boundaries_and_half_targets_with_custom_width(self):
        # Exactly 0.25 starts bucket 1, 0.75 bucket 3, and 1 stays in bucket 3.
        pairs = [(0, 0), (0.125, 0.5), (0.25, 0), (0.75, 1), (1, 0.5)]
        self.assertEqual(ece_with_bins(pairs, bins=4), (0.175, [
            {"bin": 0, "n": 2, "mean_p": 0.0625, "mean_y": 0.25},
            {"bin": 1, "n": 1, "mean_p": 0.25, "mean_y": 0.0},
            {"bin": 3, "n": 2, "mean_p": 0.875, "mean_y": 0.75},
        ]))

    def test_rejects_invalid_bins(self):
        for bins in (0, -1, True, 2.5):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                ece([(0.5, 1)], bins)  # type: ignore[arg-type]


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "calibration.json"
        self.sha = "a" * 64
        self.raw = {"schema": 1, "kind": "platt", "a": 1.23, "b": -0.29,
                    "source": {"file_sha256": "b" * 64, "group": "train"},
                    "champion_weights_sha256": self.sha}

    def save(self, raw=None):
        self.path.write_text(json.dumps(self.raw if raw is None else raw))

    def assert_identity(self, reason_contains):
        calibrator, reason = load_calibrator(self.path, self.sha)
        self.assertIsInstance(calibrator, Identity)
        assert reason is not None
        self.assertIn(reason_contains, reason)
        self.assertEqual(calibrator.apply(0), 0)
        self.assertEqual(calibrator.apply(1), 1)

    def test_missing_file_returns_identity_and_reason(self):
        self.assert_identity("missing")

    def test_corrupt_json_and_invalid_encoding_return_identity_and_reason(self):
        self.path.write_text("{broken")
        self.assert_identity("corrupt")
        self.path.write_bytes(b"\xff")
        self.assert_identity("corrupt")

    def test_sha_mismatch_returns_identity_and_reason(self):
        self.raw["champion_weights_sha256"] = "c" * 64
        self.save()
        self.assert_identity("SHA-256 mismatch")

    def test_invalid_schema_coefficients_and_digests_return_identity(self):
        cases = [("schema", True), ("schema", 2), ("kind", "isotonic"),
                 ("a", -1), ("a", True), ("a", "1"), ("b", float("nan")),
                 ("b", float("inf")), ("source", []),
                 ("champion_weights_sha256", "not-a-digest")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                raw = dict(self.raw)
                raw[key] = value
                self.save(raw)
                self.assert_identity("corrupt")
        for raw in ([], {}, {k: v for k, v in self.raw.items() if k != "b"}):
            self.save(raw)
            self.assert_identity("corrupt")

    def test_valid_file_returns_platt_with_source(self):
        self.save()
        calibrator, reason = load_calibrator(self.path, self.sha)
        self.assertIsNone(reason)
        self.assertEqual(calibrator, Platt(1.23, -0.29, self.raw["source"]))
        self.assertEqual(calibrator.describe()["source"], self.raw["source"])


class ReportTests(unittest.TestCase):
    def test_split_pending_filter_actor_perspective_and_d2_tail_clipping(self):
        fit, holdout = collect_pairs(diagnostic())
        self.assertEqual(len(fit), 600)
        self.assertEqual(len(holdout), 200)
        self.assertEqual(fit, [(0, 0), (1, 1)] * 300)
        self.assertEqual(holdout, [(0, 0), (1, 1)] * 100)

    def test_rejects_nontrain_duplicate_missing_games_and_wrong_model(self):
        changes = [lambda d: d["games"][0].update(group="dev"),
                   lambda d: d["summary"].update(group="accept"),
                   lambda d: d["summary"].update(meta="gen532.meta.json"),
                   lambda d: d["games"].pop(),
                   lambda d: d["games"].append(copy.deepcopy(d["games"][0]))]
        for change in changes:
            raw = diagnostic()
            change(raw)
            with self.subTest(change=change), self.assertRaises(ValueError):
                collect_pairs(raw)

    def test_heldout_targets_do_not_change_fit_coefficients(self):
        raw = diagnostic()
        for game in raw["games"]:
            game["rows"][0]["u0"] = -0.6
            game["rows"][1]["u1"] = 0.6
        fit_before, holdout_before = collect_pairs(raw)
        for game in raw["games"]:
            if game["index"] >= 503100:
                game["utilities"] = [1, -1]
        fit_after, holdout_after = collect_pairs(raw)
        self.assertEqual(fit_before, fit_after)
        self.assertNotEqual(holdout_before, holdout_after)
        self.assertEqual(fit_platt(fit_before), fit_platt(fit_after))

    def test_artifact_reproduces_and_both_ece_splits_improve(self):
        report = build_report_from_root()
        assert_report_close(self, report, json.loads(ARTIFACT.read_text()))
        self.assertEqual(report["source"]["file_sha256"],
                         hashlib.sha256(INPUT.read_bytes()).hexdigest())
        for name in ("fit", "holdout"):
            self.assertLess(report["ece"][name]["after"], report["ece"][name]["before"])
            self.assertLess(report["brier"][name]["after"], report["brier"][name]["before"])
        calibrator, reason = load_calibrator(ARTIFACT, report["champion_weights_sha256"])
        self.assertIsNone(reason)
        self.assertIsInstance(calibrator, Platt)

    def test_cli_runs_with_no_site_packages_and_blocks_torch_and_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.json"
            code = """
import importlib.abc
import runpy
import sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy') or fullname.startswith('naeco.engine'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
sys.argv = ['scripts/fit_calibration.py', '--in', sys.argv[1], '--out', sys.argv[2]]
runpy.run_path('scripts/fit_calibration.py', run_name='__main__')
"""
            result = subprocess.run([sys.executable, "-S", "-c", code, str(INPUT), str(output)],
                                    cwd=str(ROOT), capture_output=True, text=True,
                                    env=dict(os.environ, OMP_NUM_THREADS="1"))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            assert_report_close(self, report["a"], json.loads(ARTIFACT.read_text())["a"])

    def test_source_has_only_stdlib_and_calibration_imports(self):
        allowed = {"json", "math", "dataclasses", "typing", "argparse", "hashlib",
                   "os", "sys", "pathlib", "naeco.ai.calibration"}
        for path in (ROOT / "naeco/ai/calibration.py", ROOT / "scripts/fit_calibration.py"):
            tree = ast.parse(path.read_text(), feature_version=(3, 9))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in allowed for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)


if __name__ == "__main__":
    unittest.main()
