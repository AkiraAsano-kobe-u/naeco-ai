"""compare_search の完了ペア逐次ジャーナルと --resume の回帰テスト。"""
import os
import sys
import tempfile
import threading
import time
import unittest
from multiprocessing.pool import ThreadPool
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import compare_search  # noqa: E402
import search_records  # noqa: E402
from search_records import SearchRecordError  # noqa: E402

LABEL = "pimc-vs-heuristic"


class Killed(BaseException):
    """強制停止の代用(通常の except Exception では拾われない)。"""


def _environment():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    env = compare_search._environment(
        1, dict((n, "1") for n in search_records.THREAD_ENV_NAMES))
    return env


def _run(out, pairs=3, start_index=0, budget=60, resume=False, stop_after=None,
         environment=None):
    real = compare_search._run_pair
    calls = {"n": 0}

    def flaky(task):
        if stop_after is not None and calls["n"] >= stop_after:
            raise Killed()
        calls["n"] += 1
        return real(task)

    compare_search._run_pair = flaky
    try:
        return compare_search.run_stage_two(
            pairs, budget, 2, 30, 1.41, out, group="dev",
            start_index=start_index, only={LABEL}, resume=resume,
            environment=environment or _environment())
    finally:
        compare_search._run_pair = real


def _bundle(out):
    return search_records.load_validated_bundles(out)[LABEL]


class JournalResume(unittest.TestCase):

    def test_kill_then_resume_equals_single_run(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            whole = os.path.join(tmp, "whole.jsonl")
            _run(whole, environment=env)
            cut = os.path.join(tmp, "cut.jsonl")
            with self.assertRaises(Killed):
                _run(cut, stop_after=2, environment=env)
            journal = os.path.join(
                cut + "-journal", LABEL + ".pairs.jsonl")
            with open(journal, encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), 1 + 2)  # header+2
            # 確定前なので bundle はまだ無い
            self.assertFalse(os.path.isdir(cut + "-bundle"))
            _run(cut, resume=True, environment=env)
            self.assertFalse(os.path.exists(journal))
            a, b = _bundle(whole), _bundle(cut)
            self.assertEqual(
                search_records.find_deterministic_diffs(
                    a["summary"], b["summary"], a["games"], b["games"]), [])
            self.assertEqual(a["summary"]["pair_scores"],
                             b["summary"]["pair_scores"])
            self.assertEqual(
                [(g["index"], g["seat_variant"], g["final_scores"])
                 for g in a["games"]],
                [(g["index"], g["seat_variant"], g["final_scores"])
                 for g in b["games"]])

    def test_resume_runs_only_missing_pairs(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "r.jsonl")
            with self.assertRaises(Killed):
                _run(out, stop_after=2, environment=env)
            seen = []
            real = compare_search._run_pair

            def spy(task):
                seen.append(task[1])
                return real(task)
            compare_search._run_pair = spy
            try:
                compare_search.run_stage_two(
                    3, 60, 2, 30, 1.41, out, group="dev", start_index=0,
                    only={LABEL}, resume=True, environment=env)
            finally:
                compare_search._run_pair = real
            self.assertEqual(seen, [2])

    def test_torn_last_line_is_discarded(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "t.jsonl")
            with self.assertRaises(Killed):
                _run(out, stop_after=2, environment=env)
            journal = os.path.join(out + "-journal", LABEL + ".pairs.jsonl")
            with open(journal, "a", encoding="utf-8") as handle:
                handle.write('{"kind": "pair", "index": 2, "ga')  # 書込途中
            _run(out, resume=True, environment=env)
            self.assertEqual(len(_bundle(out)["games"]), 6)

    def test_resume_rejects_changed_conditions(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "c.jsonl")
            with self.assertRaises(Killed):
                _run(out, stop_after=1, environment=env)
            for kwargs in ({"start_index": 10}, {"budget": 61},
                           {"pairs": 4}):
                with self.assertRaises(SearchRecordError, msg=str(kwargs)):
                    _run(out, resume=True, environment=env, **kwargs)
            other = dict(env, python_hash_seed="random")
            with self.assertRaises(SearchRecordError):
                _run(out, resume=True, environment=other)

    def test_fresh_run_refuses_leftover_journal(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "f.jsonl")
            with self.assertRaises(Killed):
                _run(out, stop_after=1, environment=env)
            with self.assertRaises(SearchRecordError):
                _run(out, environment=env)

    def test_tampered_journal_entry_is_rejected(self):
        env = _environment()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "x.jsonl")
            with self.assertRaises(Killed):
                _run(out, stop_after=1, environment=env)
            journal = os.path.join(out + "-journal", LABEL + ".pairs.jsonl")
            with open(journal, encoding="utf-8") as handle:
                lines = handle.readlines()
            with open(journal, "a", encoding="utf-8") as handle:
                handle.write(lines[1])          # 同じ index の重複
            with self.assertRaises(SearchRecordError):
                _run(out, resume=True, environment=env)


if __name__ == "__main__":
    unittest.main()

def _wait_for_pair_line(path, timeout):
    """journal に pair 行が書かれるまで待つ(テスト内の完了順同期用)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                if '"kind": "pair"' in handle.read():
                    return True
        except OSError:
            pass
        time.sleep(0.001)
    return False


class CompletionOrderJournal(unittest.TestCase):

    def test_later_pair_is_journaled_before_earlier_pair_fails(self):
        """先頭ペアが長引いて失敗しても、先に完了した後続ペアは journal に残る。"""
        with tempfile.TemporaryDirectory() as tmp:
            journal = search_records.PairJournal(
                os.path.join(tmp, "o.jsonl"), "probe", "run", 0, 2)
            ready = threading.Event()
            finished = []

            def worker(task):
                if task[1] == 0:
                    self.assertTrue(ready.wait(10))
                    # ready は worker の return 前に立つため、後続ペアが親で
                    # journal に書かれたことを観測してから失敗させる。
                    self.assertTrue(_wait_for_pair_line(journal.path, 10))
                    raise RuntimeError("earlier pair failed")
                finished.append(task[1])
                ready.set()
                return ([1.0, 0.0], {}, [])

            pool = ThreadPool(2)
            try:
                with mock.patch.object(compare_search, "_run_pair", worker):
                    with self.assertRaises(RuntimeError):
                        compare_search.run_comparison(
                            "dev", 0, 2, "pimc", "heuristic", 60, 2, 30,
                            1.41, pool=pool, journal=journal)
            finally:
                pool.terminate()
                pool.join()
            self.assertEqual(finished, [1])
            self.assertEqual(sorted(journal.open_resume()), [1])


class ResumedMetrics(unittest.TestCase):

    def test_full_restore_does_not_inflate_throughput(self):
        """全ペア復元時は今回の仕事が無いので、遷移/秒を水増しして出さない。"""
        with tempfile.TemporaryDirectory() as tmp:
            journal = search_records.PairJournal(
                os.path.join(tmp, "m.jsonl"), "probe", "run", 0, 1)
            row1, unused = compare_search.run_comparison(
                "dev", 0, 1, "pimc", "heuristic", 60, 2, 30, 1.41,
                journal=journal)
            self.assertGreater(row1["transitions_per_second"], 0)
            row2, unused = compare_search.run_comparison(
                "dev", 0, 1, "pimc", "heuristic", 60, 2, 30, 1.41,
                journal=journal, resume=True)
            self.assertIsNone(row2["transitions_per_second"])

    def test_partial_resume_counts_only_pairs_run_now(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = search_records.PairJournal(
                os.path.join(tmp, "p.jsonl"), "probe", "run", 0, 2)
            full, unused = compare_search.run_comparison(
                "dev", 0, 2, "pimc", "heuristic", 60, 2, 30, 1.41)
            compare_search.run_comparison(
                "dev", 0, 1, "pimc", "heuristic", 60, 2, 30, 1.41,
                journal=search_records.PairJournal(
                    os.path.join(tmp, "p.jsonl"), "probe", "run", 0, 2))
            moments = iter([100.0, 102.0])
            with mock.patch.object(compare_search.time, "perf_counter",
                                   lambda: next(moments)):
                row, unused = compare_search.run_comparison(
                    "dev", 0, 2, "pimc", "heuristic", 60, 2, 30, 1.41,
                    journal=journal, resume=True)
            # 分子は今回走らせた 1 ペア(index 1)の遷移だけ。elapsed=2s。
            only_second = compare_search._run_pair(
                ("dev", 1, "pimc", "heuristic", 60, 2, 30, 1.41, None,
                 False))
            ran = only_second[1]["transitions"]
            self.assertAlmostEqual(row["transitions_per_second"], ran / 2.0)
            self.assertLess(ran, full["decisions"] * 10 ** 9)
