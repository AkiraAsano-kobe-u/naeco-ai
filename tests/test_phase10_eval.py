"""Phase 10 C3: n 局巡回クラスタ評価(D4)・record v2・多人数 seed 台帳のテスト。

歯の狙い:
- 3/4 席の heuristic/random 対戦が完走し、クラスタ内の全局が同一配札・各 agent が全席を 1 回ずつ担当する。
- 勝利指標は ranking[target_seat] == 0。順位・効用は保存物(scores と終局時の開始席)から再計算できる。
- 集計・CI・合否を保存物だけから再計算でき、worker=1/4 で records が byte 一致する。
- 欠け・重複・別 run・巡回でない席配置は合否計算を拒否する。
- D4 の検出力(786 → 800、Bonferroni)を固定値として検査する。
- 台帳: 2 人昇格予約列・消費済み帯・既存多人数予約との重複、登録外の対戦、二重起動を拒否する。
- n=2 の legacy 経路は変更前の golden(本カード着手前の HEAD で採取)と bit 一致する。
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import eval_agent  # noqa: E402
import ladder  # noqa: E402
from naeco.ai.eval import harness  # noqa: E402
from naeco.engine.cards import load_cards  # noqa: E402

AGENTS3 = ["heuristic", "random", "random"]
AGENTS4 = ["heuristic", "random", "random", "random"]
_GROUP = "train"
_START = 9200000          # 本テスト専用(train、判定ではない。台帳の他用途と重ならない)


def _run(agents, start=_START, clusters=3, workers=1):
    n = len(agents)
    cond = harness.multi_conditions(n, agents, _GROUP, start, clusters,
                                    "test", 0.5)
    run_id = harness.multi_run_id(cond)
    tasks = harness.cyclic_tasks(run_id, _GROUP, start, clusters, agents, n)
    return cond, run_id, harness.run_cyclic_games(tasks, workers=workers)


class CyclicClusterTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runs = {n: _run(a) for n, a in ((3, AGENTS3), (4, AGENTS4))}

    def test_games_complete_with_same_deal_and_balanced_seats(self):
        for n, (cond, run_id, records) in self.runs.items():
            self.assertEqual(len(records), 3 * n)
            for c in range(3):
                games = [r for r in records if r["index"] == _START + c]
                self.assertEqual([g["seat_variant"] for g in games],
                                 list(range(n)))
                self.assertEqual(len(set(g["initial_state_hash"]
                                         for g in games)), 1)
                self.assertEqual(len(set(json.dumps(g["initial_deck"])
                                         for g in games)), 1)
                # 各 agent が全席を 1 回ずつ
                for j in range(n):
                    self.assertEqual(sorted(g["order"][j] for g in games),
                                     list(range(n)))
                for g in games:
                    self.assertEqual(g["n"], n)
                    self.assertEqual(len(g["scores"]), n)
                    self.assertEqual(sorted(g["ranking"]), list(range(n)))

    def test_cyclic_order_matches_env_seat_order(self):
        from naeco.train import env
        for n in (3, 4):
            for k in range(n):
                self.assertEqual(harness.cyclic_seat_order(n, k),
                                 env._seat_order(0, 0, k, n))

    def test_win_metric_is_rank_zero_not_positive_utility(self):
        for n, (cond, run_id, records) in self.runs.items():
            for r in records:
                self.assertEqual(r["target_win"],
                                 1 if r["ranking"][r["target_seat"]] == 0 else 0)
                self.assertEqual(r["target_rank"], r["ranking"][r["target_seat"]])
        # 4 席の 2 位は効用 +1/3 > 0 だが勝ちではない
        rec = dict(self.runs[4][2][0])
        self.assertAlmostEqual(1.0 - 2.0 * 1 / 3, 1.0 / 3)

    def test_record_v2_fields(self):
        rec = self.runs[3][2][0]
        for key in ("protocol", "run_id", "group", "index", "n", "seat_variant",
                    "order", "target_seat", "seats", "env_seed", "setup_seed",
                    "initial_deck", "initial_hands", "initial_money",
                    "start_player", "initial_state_hash", "cards_sha256",
                    "cards_data_version", "api_version", "rules_version",
                    "obs_version", "reward_version", "scores", "ranking",
                    "target_win", "final_start_player", "utilities"):
            self.assertIn(key, rec)
        self.assertEqual(rec["reward_version"], "2.0.0")
        self.assertEqual(rec["cards_sha256"], load_cards().sha256)
        for seat in rec["seats"]:
            self.assertEqual(set(seat), {"seat", "agent_index", "agent",
                                         "model_sha256", "policy_seed",
                                         "search"})

    def test_summary_recomputes_from_saved_records(self):
        for n, (cond, run_id, records) in self.runs.items():
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "r.jsonl")
                harness.write_multi_records(records, path)
                loaded = harness.read_multi_records(path)
            a = harness.summarize_multi(records, cond, run_id, threshold=0.5)
            b = harness.summarize_multi(loaded, cond, run_id, threshold=0.5)
            self.assertEqual(a, b)
            self.assertEqual(a["games"], 3 * n)
            self.assertEqual(a["clusters"], 3)
            self.assertEqual(len(a["cluster_results"]), 3)
            self.assertEqual(sum(a["target_rank_histogram"]), 3 * n)
            values = [c["value"] for c in a["cluster_results"]]
            lo, hi = harness.cluster_bootstrap(
                tuple(values), harness.BOOTSTRAP_SAMPLES, harness.BOOTSTRAP_SEED)
            self.assertEqual((a["ci_low"], a["ci_high"]), (lo, hi))
            self.assertEqual(a["passed"], lo > 0.5)

    def test_workers_one_and_four_give_identical_bytes(self):
        cond, run_id, one = _run(AGENTS4, start=_START + 50, clusters=2,
                                 workers=1)
        _c, _r, four = _run(AGENTS4, start=_START + 50, clusters=2, workers=4)
        to_bytes = lambda rs: "".join(harness.record_line(r) for r in rs)
        self.assertEqual(to_bytes(one), to_bytes(four))

    def test_rejects_missing_duplicate_foreign_and_noncyclic(self):
        cond, run_id, records = self.runs[3]
        args = (run_id, cond["start_index"], cond["clusters"], 3, AGENTS3)
        with self.assertRaisesRegex(ValueError, "missing"):
            harness.multi_cluster_results(records[:-1], *args)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            harness.multi_cluster_results(records + [records[0]], *args)
        other = dict(records[0], run_id="x" * 64)
        with self.assertRaisesRegex(ValueError, "another run"):
            harness.multi_cluster_results([other] + records[1:], *args)
        swapped = [dict(r) for r in records]
        swapped[0]["seat_agents"] = [1, 0, 2]
        with self.assertRaisesRegex(ValueError, "cyclic"):
            harness.multi_cluster_results(swapped, *args)
        tampered = [dict(r) for r in records]
        tampered[0]["target_win"] = 1 - tampered[0]["target_win"]
        tampered[0]["ranking"] = list(reversed(tampered[0]["ranking"]))
        with self.assertRaisesRegex(ValueError, "ranking"):
            harness.multi_cluster_results(tampered, *args)

    def test_two_seat_and_model_agents_are_refused(self):
        with self.assertRaises(ValueError):
            harness.multi_conditions(2, ["heuristic", "random"], _GROUP, 0, 1,
                                     "x", 0.5)
        with self.assertRaises(ValueError):
            harness.multi_conditions(3, ["heuristic", "random"], _GROUP, 0, 1,
                                     "x", 0.5)
        with self.assertRaises(ValueError):
            harness.multi_conditions(3, ["checkpoint:x.meta.json", "random",
                                         "random"], _GROUP, 0, 1, "x", 0.5)


class D4PowerTests(unittest.TestCase):

    def test_fixed_values(self):
        self.assertEqual(harness.required_clusters(), 786)
        self.assertEqual(harness.D4_CLUSTERS, 800)
        self.assertEqual(harness.D4_COMPLETION_THRESHOLDS, {3: 0.60, 4: 0.50})
        rec = harness.d4_power_record(0.60)
        self.assertAlmostEqual(rec["alternative"], 0.65)
        self.assertGreaterEqual(rec["approximate_power"], 0.80)
        self.assertLess(harness.approximate_power(786 - 100), 0.80)
        self.assertEqual(rec["bonferroni_judgements"], 2)
        self.assertLessEqual(rec["family_alpha_bound"], 0.05)
        self.assertAlmostEqual(rec["alpha_one_sided_per_judgement"], 0.025,
                               places=3)


_LEDGER_TEMPLATE = """# test ledger
<!-- CONSUMED-DEV-BANDS:BEGIN -->
0-499 7000-7799 7800-8599 9000-9799 10000-10799 13200-13999 14000-14799
<!-- CONSUMED-DEV-BANDS:END -->

<!-- MULTI-SEAT-RESERVED:BEGIN -->
<!-- MULTI-SEAT-RESERVED:END -->

<!-- MULTI-SEAT-CONSUMED:BEGIN -->
<!-- MULTI-SEAT-CONSUMED:END -->
"""


class LedgerTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER_TEMPLATE)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _entry(self, rid, group, lo, clusters, matchups=(AGENTS3,), n=3,
               threshold=0.5):
        return eval_agent.build_reservation(rid, n, matchups, group, lo,
                                            lo + clusters - 1, "test",
                                            threshold)

    def test_two_seat_reserve_sequence_and_consumed_bands_are_refused(self):
        for group, lo in (("dev", 14800), ("dev", 16400), ("dev", 7800),
                          ("dev", 9000), ("dev", 400), ("dev", 50000),
                          ("accept", 0)):
            with self.assertRaises(eval_agent.SeedLedgerError, msg=(group, lo)):
                eval_agent.register_multi(self._entry("r", group, lo, 800),
                                          ledger_path=self.ledger)
        eval_agent.register_multi(self._entry("ok", "accept", 800, 800),
                                  ledger_path=self.ledger)

    def test_multi_dev_region_is_the_only_free_dev_area_above_10800(self):
        # 2026-10-09 裁定: 多人数専用 dev 30000-39999。両端の外は 2 人列のまま拒否。
        self.assertEqual(eval_agent.MULTI_DEV_REGION, (30000, 39999))
        for lo in (29201, 39201, 40000):
            with self.assertRaises(eval_agent.SeedLedgerError, msg=lo):
                eval_agent.register_multi(self._entry("x%d" % lo, "dev", lo, 800),
                                          ledger_path=self.ledger)
        for rid, lo in (("b3", 30000), ("b4", 30800), ("p3", 31600),
                        ("p4", 32400), ("last", 39200)):
            eval_agent.register_multi(
                eval_agent.build_hold(rid, 3, "dev", lo, lo + 799, "t", None),
                ledger_path=self.ledger)
        with self.assertRaises(eval_agent.SeedLedgerError):   # 領域内でも重複は拒否
            eval_agent.register_multi(
                eval_agent.build_hold("dup", 4, "dev", 30400, 31199, "t", None),
                ledger_path=self.ledger)

    def test_formal_needs_800_clusters(self):
        with self.assertRaises(eval_agent.SeedLedgerError):
            self._entry("r", "accept", 800, 799)

    def test_reservation_overlap_and_unregistered_matchup_and_double_launch(self):
        eval_agent.register_multi(
            self._entry("a", _GROUP, _START + 100, 2,
                        matchups=(AGENTS3, ["random", "heuristic", "random"])),
            ledger_path=self.ledger)
        with self.assertRaises(eval_agent.SeedLedgerError):
            eval_agent.register_multi(self._entry("b", _GROUP, _START + 101, 2),
                                      ledger_path=self.ledger)
        with self.assertRaises(eval_agent.SeedLedgerError):   # id reuse
            eval_agent.register_multi(self._entry("a", _GROUP, _START + 900, 2),
                                      ledger_path=self.ledger)
        records = os.path.join(self.dir, "r.jsonl")
        summary = os.path.join(self.dir, "s.json")
        with self.assertRaisesRegex(eval_agent.SeedLedgerError,
                                    "not pre-registered"):
            eval_agent.run_multi("a", 3, ["random", "random", "heuristic"],
                                 _GROUP, _START + 100, 2, "test", 0.5,
                                 records, summary, ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "differ"):
            eval_agent.run_multi("a", 3, AGENTS3, _GROUP, _START + 100, 2,
                                 "test", 0.6, records, summary,
                                 ledger_path=self.ledger)
        out = eval_agent.run_multi("a", 3, AGENTS3, _GROUP, _START + 100, 2,
                                   "test", 0.5, records, summary,
                                   ledger_path=self.ledger)
        self.assertEqual(out["games"], 6)
        fresh = eval_agent.recompute_multi(summary)
        self.assertEqual(fresh["ci_low"], out["ci_low"])
        same, unused = eval_agent.replay_multi(summary, workers=1)
        self.assertTrue(same)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError,
                                    "already consumed"):
            eval_agent.run_multi("a", 3, AGENTS3, _GROUP, _START + 100, 2,
                                 "test", 0.5, records + "2", summary + "2",
                                 ledger_path=self.ledger)
        # 事前固定した 2 つ目の対戦は同帯で起動できる
        out2 = eval_agent.run_multi(
            "a", 3, ["random", "heuristic", "random"], _GROUP, _START + 100, 2,
            "test", 0.5, records + "3", summary + "3", ledger_path=self.ledger)
        self.assertEqual(out2["games"], 6)

    def test_resume_fills_only_missing_games_deterministically(self):
        eval_agent.register_multi(self._entry("r", _GROUP, _START + 200, 2,
                                              matchups=(AGENTS4,), n=4),
                                  ledger_path=self.ledger)
        records = os.path.join(self.dir, "r.jsonl")
        summary = os.path.join(self.dir, "s.json")
        eval_agent.run_multi("r", 4, AGENTS4, _GROUP, _START + 200, 2, "test",
                             0.5, records, summary, ledger_path=self.ledger)
        with open(records, "rb") as fh:
            full = fh.read()
        # 中断を模す: summary を消し、最後の 3 局と途中書きの行を残す
        os.remove(summary)
        lines = full.splitlines(True)
        with open(records, "wb") as fh:
            fh.write(b"".join(lines[:5]) + lines[5][:40])
        partial = [json.loads(line) for line in lines[:5]]
        cond = harness.multi_conditions(4, AGENTS4, _GROUP, _START + 200, 2,
                                        "test", 0.5)
        with self.assertRaisesRegex(ValueError, "missing"):   # 部分集合で判定しない
            harness.summarize_multi(partial, cond, harness.multi_run_id(cond),
                                    threshold=0.5)
        eval_agent.run_multi("r", 4, AGENTS4, _GROUP, _START + 200, 2, "test",
                             0.5, records, summary, resume=True,
                             ledger_path=self.ledger)
        with open(records, "rb") as fh:
            self.assertEqual(fh.read(), full)

    def test_hold_cannot_launch_and_is_finalised_once(self):
        hold = eval_agent.build_hold("h", 3, "accept", 800, 1599, "final", 0.6)
        eval_agent.register_multi(hold, ledger_path=self.ledger)
        records = os.path.join(self.dir, "r.jsonl")
        summary = os.path.join(self.dir, "s.json")
        with self.assertRaisesRegex(eval_agent.SeedLedgerError,
                                    "not pre-registered"):
            eval_agent.run_multi("h", 3, AGENTS3, "accept", 800, 800, "final",
                                 0.6, records, summary, ledger_path=self.ledger)
        with self.assertRaises(eval_agent.SeedLedgerError):   # threshold change
            eval_agent.register_multi(
                eval_agent.build_reservation("h", 3, [AGENTS3], "accept", 800,
                                             1599, "final", 0.5),
                ledger_path=self.ledger)
        eval_agent.register_multi(
            eval_agent.build_reservation("h", 3, [AGENTS3], "accept", 800, 1599,
                                         "final", 0.6),
            ledger_path=self.ledger)
        with self.assertRaises(eval_agent.SeedLedgerError):   # second finalise
            eval_agent.register_multi(
                eval_agent.build_reservation("h", 3, [AGENTS3], "accept", 800,
                                             1599, "final", 0.6),
                ledger_path=self.ledger)

    def test_real_ledger_parses_and_promote_next_band_is_unchanged(self):
        import promote
        unused, reserved, consumed, two_seat = eval_agent.read_multi_ledger()
        ids = set(e["id"] for e in reserved)
        self.assertTrue({"p10-accept-3p", "p10-accept-4p", "p10-bench-3p",
                         "p10-bench-4p", "p10-promote1-3p",
                         "p10-promote1-4p"} <= ids)
        bands = dict((e["id"], (e["group"], e["lo"], e["hi"], e["num_players"]))
                     for e in reserved)
        self.assertEqual(bands["p10-bench-3p"], ("dev", 30000, 30799, 3))
        self.assertEqual(bands["p10-bench-4p"], ("dev", 30800, 31599, 4))
        self.assertEqual(bands["p10-promote1-3p"], ("dev", 31600, 32399, 3))
        self.assertEqual(bands["p10-promote1-4p"], ("dev", 32400, 33199, 4))
        consumed_dev = promote._load_consumed_dev_bands()
        self.assertEqual(promote._next_reserved_direct_band(consumed_dev),
                         14800)


class SavedSmokeArtefactTests(unittest.TestCase):
    """docs/eval-runs/phase10/c3 の保存物だけから集計・CI・合否を再計算する。"""

    def test_saved_smoke_recomputes(self):
        base = os.path.join(_ROOT, "docs", "eval-runs", "phase10", "c3")
        for name, n in (("smoke-3p-train-870000-870019.json", 3),
                        ("smoke-4p-train-870100-870119.json", 4)):
            fresh = eval_agent.recompute_multi(os.path.join(base, name))
            self.assertEqual(fresh["num_players"], n)
            self.assertEqual(fresh["games"], 20 * n)
            self.assertEqual(sum(fresh["target_wins_by_seat"]), fresh["wins"])


class LadderMultiTests(unittest.TestCase):

    def test_multi_diagnostic_has_no_elo_and_rotates_targets(self):
        summary, records = ladder.run_multi_diagnostic(
            ["heuristic", "random", "random"], 3, _START + 300, 2, 1)
        self.assertNotIn("elo", summary)
        self.assertEqual([r["target"] for r in summary["rows"]],
                         ["heuristic", "random", "random"])
        self.assertEqual(len(records), 3 * 3 * 2)

    def test_multi_ladder_refuses_formal_groups(self):
        with self.assertRaises(SystemExit):
            ladder.main(["heuristic", "random", "random", "--num-players", "3",
                         "--group", "dev", "--start-index", "0",
                         "--out", "/nonexistent/x.json", "--execute"])


# 本カード着手前の HEAD(4daf579)で採取した n=2 legacy の golden。
# 採取: harness.evaluate(6, train, 9100000, B=1000) / eval_agent.run_evaluation(
# heuristic, random, train, 9100010, 4) / ladder.run([random, heuristic], train,
# 9100020, 3)(seconds / wall_seconds / mean_game_seconds を除く)。
_LEGACY_GOLDEN = {
    "harness_records_sha":
        "1ad31c46a58cfe85a613007a585997a3bc81425f4e6c08db7e16fcc4a90c32dc",
    "eval_agent_records_sha":
        "94d42e37dfc8709c21ebfc7c03f79ef63298d81fe4aa58fb99761405f7ccc5b1",
    "ladder_summary_sha":
        "c288b2bfd8d15b46e45402e676dfd25e4d0e45e341be2487651b9b0524e65086",
    "ladder_results_sha":
        "80307a973f8a4e59b4c49a258c03d6c0b9150c9344494d28117104989ee9c8e5",
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class TwoSeatLegacyGoldenTests(unittest.TestCase):

    def test_harness_records_and_ci_bit_identical(self):
        r = harness.evaluate(pairs=6, group="train", start_index=9100000,
                             bootstrap_samples=1000, bootstrap_seed=2026072102)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jsonl")
            harness.write_game_records(r, p)
            with open(p, "rb") as fh:
                self.assertEqual(_sha(fh.read()),
                                 _LEGACY_GOLDEN["harness_records_sha"])

    def test_eval_agent_records_bit_identical(self):
        e = eval_agent.run_evaluation("heuristic", "random", "train", 9100010,
                                      4, workers=1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.jsonl")
            harness.write_game_records(e, p)
            with open(p, "rb") as fh:
                self.assertEqual(_sha(fh.read()),
                                 _LEGACY_GOLDEN["eval_agent_records_sha"])

    def test_ladder_two_seat_bit_identical(self):
        s, results = ladder.run(["random", "heuristic"], "train", 9100020, 3, 1)
        for r in results:
            r.pop("seconds")
        s.pop("wall_seconds")
        for p in s["pairings"]:
            p.pop("mean_game_seconds")
        self.assertEqual(_sha(json.dumps(s, sort_keys=True).encode()),
                         _LEGACY_GOLDEN["ladder_summary_sha"])
        self.assertEqual(_sha(json.dumps(results, sort_keys=True).encode()),
                         _LEGACY_GOLDEN["ladder_results_sha"])

    def test_two_seat_cli_rejects_multi_only_flags(self):
        with self.assertRaises(SystemExit):
            eval_agent.main(["heuristic", "random", "random"])
        with self.assertRaises(SystemExit):
            eval_agent.main(["heuristic", "random", "--clusters", "3"])


class ThresholdlessBenchTests(unittest.TestCase):
    """R12-001: explicit None survives finalisation, run, recompute and resume."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER_TEMPLATE)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _cli(self, agents, rid, start, clusters, group="train", threshold="none"):
        return list(agents) + [
            "--num-players", str(len(agents)), "--group", group,
            "--start-index", str(start), "--clusters", str(clusters),
            "--reservation", rid, "--purpose", "bench", "--ledger", self.ledger,
            "--threshold", threshold, "--execute",
        ]

    def _register(self, agents, rid, start, threshold=None):
        eval_agent.register_multi(
            eval_agent.build_reservation(rid, len(agents), [agents], "train",
                                         start, start, "bench", threshold),
            ledger_path=self.ledger)

    def _assert_saved(self, summary, path, threshold):
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved, summary)
        self.assertEqual(saved["threshold"], threshold)
        self.assertEqual(saved["conditions"]["threshold"], threshold)
        self.assertEqual(saved["power"], harness.d4_power_record(threshold, 1))
        if threshold is None:
            self.assertIsNone(saved["passed"])
        else:
            self.assertEqual(saved["passed"], saved["ci_low"] > threshold)
        fresh = eval_agent.recompute_multi(path)
        self.assertEqual(fresh, {k: saved[k] for k in eval_agent.RECOMPUTED_KEYS})

    def test_cli_finalises_none_holds_without_changing_numeric_default(self):
        for agents, rid, start in ((AGENTS3, "h3", 30000),
                                   (AGENTS4, "h4", 30800)):
            with self.subTest(n=len(agents)):
                eval_agent.register_multi(
                    eval_agent.build_hold(rid, len(agents), "dev", start,
                                          start + 799, "bench", None),
                    ledger_path=self.ledger)
                argv = self._cli(agents, rid, start, 800, group="dev")
                # Omitting the explicit None still uses 0.50: never silently
                # inherit a hold and hide a change to registered conditions.
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    eval_agent.main(argv[:-3] + ["--execute", "--register"])
                self.assertEqual(eval_agent.main(argv + ["--register"]), 0)
                unused, reservations, consumed, unused_two = \
                    eval_agent.read_multi_ledger(self.ledger)
                entry = next(r for r in reservations if r["id"] == rid)
                self.assertIsNone(entry["threshold"])
                self.assertIsNone(entry["power"])
                self.assertEqual(entry["matchups"], [agents])
                self.assertEqual(consumed, [])

    def test_cli_bench_run_and_recompute_exit_success_without_ci_judgement(self):
        import contextlib
        import io
        for agents, rid, start in ((AGENTS3, "b3", _START + 400),
                                   (AGENTS4, "b4", _START + 401)):
            with self.subTest(n=len(agents)):
                self._register(agents, rid, start)
                records = os.path.join(self.dir, rid + ".jsonl")
                summary_path = os.path.join(self.dir, rid + ".json")
                argv = self._cli(agents, rid, start, 1)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(eval_agent.main(argv + [
                        "--records", records, "--summary", summary_path]), 0)
                    self.assertEqual(eval_agent.main(argv + [
                        "--recompute", summary_path]), 0)
                self.assertIn("no CI threshold", stdout.getvalue())
                self.assertNotIn("pass (CI low >", stdout.getvalue())
                with open(summary_path, encoding="utf-8") as fh:
                    summary = json.load(fh)
                self._assert_saved(summary, summary_path, None)
                self.assertEqual(len(harness.read_multi_records(records)),
                                 len(agents))

    def test_api_summary_recomputes_for_none_and_numeric_thresholds(self):
        for i, (agents, threshold) in enumerate((
                (AGENTS3, None), (AGENTS4, None),
                (AGENTS3, 0.60), (AGENTS4, 0.50))):
            with self.subTest(n=len(agents), threshold=threshold):
                rid = "api%d" % i
                start = _START + 410 + i
                self._register(agents, rid, start, threshold)
                records = os.path.join(self.dir, rid + ".jsonl")
                summary_path = os.path.join(self.dir, rid + ".json")
                summary = eval_agent.run_multi(
                    rid, len(agents), agents, "train", start, 1, "bench",
                    threshold, records, summary_path, ledger_path=self.ledger)
                self._assert_saved(summary, summary_path, threshold)

    def test_none_resume_finishes_interrupted_and_complete_records(self):
        from unittest import mock
        for agents, rid, start in ((AGENTS3, "resume3", _START + 420),
                                   (AGENTS4, "resume4", _START + 421)):
            with self.subTest(n=len(agents)):
                self._register(agents, rid, start)
                records = os.path.join(self.dir, rid + ".jsonl")
                summary_path = os.path.join(self.dir, rid + ".json")
                args = (rid, len(agents), agents, "train", start, 1, "bench",
                        None, records, summary_path)
                play = harness._run_cyclic_task

                def interrupt(task):
                    if task[3] == 1:
                        raise RuntimeError("test interruption")
                    return play(task)

                with mock.patch.object(harness, "_run_cyclic_task",
                                       side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        eval_agent.run_multi(*args, ledger_path=self.ledger)
                self.assertFalse(os.path.exists(summary_path))
                with open(records, "rb") as fh:
                    prefix = fh.read()
                with open(records, "ab") as fh:
                    fh.write(b'{"incomplete":')
                with mock.patch.object(harness, "_run_cyclic_task",
                                       wraps=play) as resumed_play:
                    summary = eval_agent.run_multi(
                        *args, resume=True, ledger_path=self.ledger)
                self.assertEqual(resumed_play.call_count, len(agents) - 1)
                self.assertEqual(summary["provenance"]["resumed_games"],
                                 len(agents) - 1)
                self._assert_saved(summary, summary_path, None)
                self.assertTrue(eval_agent.replay_multi(summary_path)[0])
                with open(records, "rb") as fh:
                    full = fh.read()
                self.assertTrue(full.startswith(prefix))
                # Old R12 failure left all games but no summary. That case
                # must resume without replaying any completed game as well.
                os.remove(summary_path)
                with mock.patch.object(harness, "_run_cyclic_task",
                                       wraps=play) as completed_play:
                    summary = eval_agent.run_multi(
                        *args, resume=True, ledger_path=self.ledger)
                completed_play.assert_not_called()
                self._assert_saved(summary, summary_path, None)
                with open(records, "rb") as fh:
                    self.assertEqual(fh.read(), full)

    def test_numeric_cli_holds_keep_power_and_reject_none(self):
        for agents, rid, start, threshold in (
                (AGENTS3, "accept3", 800, 0.60),
                (AGENTS4, "accept4", 1600, 0.50),
                (AGENTS3, "promote3", 31600, 1.0 / 3),
                (AGENTS4, "promote4", 32400, 0.25)):
            with self.subTest(rid=rid):
                group = "accept" if rid.startswith("accept") else "dev"
                eval_agent.register_multi(
                    eval_agent.build_hold(rid, len(agents), group, start,
                                          start + 799, "bench", threshold),
                    ledger_path=self.ledger)
                argv = self._cli(agents, rid, start, 800, group=group)
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    eval_agent.main(argv + ["--register"])
                argv = self._cli(agents, rid, start, 800, group=group,
                                 threshold=str(threshold))
                self.assertEqual(eval_agent.main(argv + ["--register"]), 0)
                unused, reservations, unused_c, unused_two = \
                    eval_agent.read_multi_ledger(self.ledger)
                entry = next(r for r in reservations if r["id"] == rid)
                self.assertEqual(entry["threshold"], threshold)
                self.assertEqual(entry["power"],
                                 harness.d4_power_record(threshold, 800))
                self.assertEqual(entry["power"]["required_clusters"], 786)
                self.assertEqual(entry["power"]["clusters"], 800)
                self.assertEqual(entry["power"]["bonferroni_judgements"], 2)

    def test_none_power_is_not_applicable(self):
        self.assertIsNone(harness.d4_power_record(None))
        self.assertIsNone(harness.d4_power_record(None, 1))

    def test_invalid_numeric_thresholds_and_two_seat_none_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "only for --num-players"):
            eval_agent.main(["heuristic", "random", "--threshold", "none"])
        for threshold in ("0", "1", "nan", "inf", "invalid"):
            with self.subTest(threshold=threshold):
                with self.assertRaises(SystemExit):
                    eval_agent.main(self._cli(AGENTS3, "bad", _START + 430, 1,
                                             threshold=threshold))
        unused, reservations, consumed, unused_two = \
            eval_agent.read_multi_ledger(self.ledger)
        self.assertEqual(reservations, [])
        self.assertEqual(consumed, [])


if __name__ == "__main__":
    unittest.main()
