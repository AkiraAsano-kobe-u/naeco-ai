"""compare_search の hash seed 契約・runtime 引き渡し・max 集約テスト。"""
import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import compare_search  # noqa: E402  (import 時に 4 thread env を "1" へ設定する)
import search_records  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "compare_search.py")


def _thread_env():
    return dict((name, "1") for name in search_records.THREAD_ENV_NAMES)


def _cli_env(**overrides):
    env = dict(os.environ)
    env.update(_thread_env())
    env.pop("PYTHONHASHSEED", None)
    env.pop(compare_search._HASH_SEED_BOOTSTRAP_MARKER, None)
    env.update(overrides)
    return env


def _run_cli(argv, env):
    return subprocess.run([sys.executable, SCRIPT] + argv, env=env,
                          capture_output=True, text=True, timeout=180)


class HashSeedCliContract(unittest.TestCase):

    def test_unset_reexec_and_manifest_records_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            proc = _run_cli([
                "--stage", "2", "--pairs", "1", "--budget", "300",
                "--start-index", "0", "--records", out, "--execute",
            ], _cli_env())
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest_path = out + ".manifest.jsonl"
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.loads(handle.readline())
            self.assertEqual(manifest["python_hash_seed"], "0")
            self.assertEqual(manifest["thread_env"], _thread_env())

    def test_unset_check_determinism_children_inherit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det.json")
            proc = _run_cli([
                "--stage", "2", "--pairs", "1", "--budget", "300",
                "--check-determinism", "--group", "train",
                "--start-index", "100", "--matchups", "pimc-vs-heuristic",
                "--workers", "2", "--records", report, "--execute",
            ], _cli_env())
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["python_hash_seed"], "0")
            self.assertTrue(payload["all_labels_matched"])
            self.assertEqual(payload["python_version"], platform.python_version())
            self.assertEqual(payload["python_executable"], sys.executable)

    def test_explicit_zero_does_not_reexec(self):
        proc = _run_cli(["--help"], _cli_env(PYTHONHASHSEED="0"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage", proc.stdout)

    def test_explicit_nonzero_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            proc = _run_cli([
                "--stage", "2", "--pairs", "1", "--budget", "300",
                "--start-index", "0", "--records", out, "--execute",
            ], _cli_env(PYTHONHASHSEED="1"))
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("PYTHONHASHSEED", proc.stderr)
            self.assertFalse(os.path.exists(out))

    def test_bootstrap_marker_prevents_infinite_restart(self):
        proc = _run_cli(["--help"], _cli_env(
            **{compare_search._HASH_SEED_BOOTSTRAP_MARKER: "1"}))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("拒否", proc.stderr)

    def test_bootstrap_reexec_process_sees_zero_and_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = os.path.join(tmp, "probe.py")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write(
                    "import os, sys\n"
                    "sys.path.insert(0, %r)\n"
                    "sys.path.insert(0, %r)\n"
                    "from scripts import compare_search\n"
                    "compare_search.bootstrap_python_hash_seed()\n"
                    "print('PYTHONHASHSEED=%%s' %% os.environ.get('PYTHONHASHSEED'))\n"
                    "print('MARKER=%%s' %% os.environ.get(\n"
                    "    compare_search._HASH_SEED_BOOTSTRAP_MARKER))\n"
                    % (os.path.join(ROOT, "scripts"), ROOT))
            proc = subprocess.run([sys.executable, probe],
                                  env=_cli_env(), capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("PYTHONHASHSEED=0", proc.stdout)
            self.assertIn("MARKER=1", proc.stdout)


class EnvironmentContract(unittest.TestCase):

    def test_environment_includes_hash_seed_and_absolute_runtime(self):
        previous = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "0"
        try:
            environment = compare_search._environment(1, _thread_env())
            self.assertEqual(environment["python_hash_seed"], "0")
            self.assertEqual(environment["python_executable"], sys.executable)
            self.assertEqual(environment["thread_env"], _thread_env())
        finally:
            if previous is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous

    def test_ensure_rejects_unset_without_bootstrap(self):
        previous = os.environ.get("PYTHONHASHSEED")
        os.environ.pop("PYTHONHASHSEED", None)
        try:
            with self.assertRaises(SystemExit):
                compare_search._ensure_python_hash_seed_env()
        finally:
            if previous is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous

    def test_ensure_rejects_explicit_nonzero(self):
        previous = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "1"
        try:
            with self.assertRaises(SystemExit):
                compare_search._ensure_python_hash_seed_env()
        finally:
            if previous is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous

    def test_main_rejects_unset_without_bootstrap(self):
        previous = os.environ.get("PYTHONHASHSEED")
        os.environ.pop("PYTHONHASHSEED", None)
        try:
            with self.assertRaises(SystemExit):
                compare_search.main(["--help"])
        finally:
            if previous is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous

    def test_main_rejects_explicit_nonzero(self):
        previous = os.environ.get("PYTHONHASHSEED")
        os.environ["PYTHONHASHSEED"] = "1"
        try:
            with self.assertRaises(SystemExit):
                compare_search.main(["--help"])
        finally:
            if previous is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous


class DeadlineMaxAggregation(unittest.TestCase):

    def test_sum_run_totals_uses_max_for_max_metric(self):
        first = compare_search._empty_stats_acc()
        first["deadline_overrun_max_ms"] = 5.0
        first["deadline_overrun_total_ms"] = 4.0
        first["deadline_overrun_count"] = 2
        second = compare_search._empty_stats_acc()
        second["deadline_overrun_max_ms"] = 20.0
        second["deadline_overrun_total_ms"] = 10.0
        second["deadline_overrun_count"] = 1
        total = compare_search._sum_run_totals(
            [(None, first, None), (None, second, None)])
        self.assertEqual(total["deadline_overrun_max_ms"], 20.0)
        self.assertEqual(total["deadline_overrun_total_ms"], 14.0)
        self.assertEqual(total["deadline_overrun_count"], 3)

    def test_run_comparison_row_deadline_max_is_max(self):
        original = compare_search._run_pair

        def fake_run_pair(task):
            index = task[1]
            acc = compare_search._empty_stats_acc()
            acc["deadline_overrun_max_ms"] = 5.0 if index == 0 else 20.0
            acc["deadline_overrun_total_ms"] = 2.0 if index == 0 else 4.0
            acc["deadline_overrun_count"] = 1
            acc["elapsed_ms"] = [1.0, 1.0]
            acc["rss"] = None
            acc["thread_env"] = _thread_env()
            acc["search_metrics"] = {}
            games = [1.0, 1.0] if index == 0 else [0.0, 0.0]
            return games, acc, []

        compare_search._run_pair = fake_run_pair
        try:
            row, unused_games = compare_search.run_comparison(
                "dev", 0, 2, "pimc", "heuristic", 300, 2, 30, 1.41)
        finally:
            compare_search._run_pair = original
        self.assertEqual(row["deadline_overrun_max_ms"], 20.0)
        self.assertEqual(row["deadline_overrun_total_ms"], 6.0)
        self.assertEqual(row["deadline_overrun_count"], 2)


class DeterminismReport(unittest.TestCase):

    def test_report_records_runtime_for_standalone_audit(self):
        environment = {
            "python_version": "3.9.21",
            "python_executable": "/usr/bin/python3",
            "source_digest": "test-digest",
        }
        report = compare_search._determinism_report(
            "train", 400, 1, 300, None, 2, 30, 1.41, 2,
            {"pimc-vs-heuristic"}, _thread_env(), environment)
        self.assertEqual(report["python_version"], "3.9.21")
        self.assertEqual(report["python_executable"], "/usr/bin/python3")
        self.assertEqual(report["source_digest"], "test-digest")
        self.assertIn("python_hash_seed", report)
        # worker=1/N の manifest 比較で差を検出する契約を維持するため、
        # runtime field を決定論比較の除外対象へ追加しない。
        self.assertNotIn("python_version", report["excluded_fields"])
        self.assertNotIn("python_executable", report["excluded_fields"])


class ReplayAuditCli(unittest.TestCase):

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_replay_audit_requires_check_determinism(self):
        with self.assertRaises(SystemExit):
            compare_search.main([
                "--stage", "2", "--pairs", "1", "--start-index", "3000",
                "--replay-audit", "--execute",
            ])


if __name__ == "__main__":
    unittest.main()
