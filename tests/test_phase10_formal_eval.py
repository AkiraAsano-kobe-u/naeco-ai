"""B-C6-fix (t_4cf080a7): formal multi-seat evaluation with learned models.

What these tests pin:
- n=3,4 cyclic evaluation seats `checkpoint:META` (per-player-count greedy) and
  `strong:META` (same model + ISMCTS with the fixed D5 setting H=2,
  c_puct=1.5, T=4500, NetEvaluator). A meta whose num_players differs from the
  seat count is refused.
- records carry model_sha256 / search / policy_seed per seat; the summary
  carries the model SHAs; the strong policy seed is derived from the cluster
  coordinates, and workers=1 and 4 give byte-identical records.
- model SHAs are part of the run conditions, the ledger reservation and the
  claim. A model swapped after registration or during a run is refused.
- promote.py has a formal D5 path for n>=3 (challenger vs champion x(n-1),
  CI low > 1/n, bench >= champion-3pt, rollback beyond 5pt, 800 clusters,
  ledger reservations, no double consumption). n=2 is untouched.
- the frozen initial champion files match initial-champion.json and are
  reproduced by the be36450 construction (seed 20260812).
Train seeds only (index 9300000+); no dev/accept band is consumed.
"""
import contextlib
import importlib.util
import io
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
import promote  # noqa: E402
from naeco.ai.eval import harness  # noqa: E402
from naeco.engine.cards import load_cards  # noqa: E402
from naeco.engine.state import derive_seed, eval_seed  # noqa: E402

# Tests that build or play learned models need torch; without it (CI:
# pip install -e .[dev]) only those are skipped.
requires_torch = unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "torch が無い環境")

MODELS = os.path.join(_ROOT, "docs", "eval-runs", "phase10", "models")
INITIAL_RECORD = os.path.join(MODELS, "initial-champion.json")


def _meta(n, name):
    return os.path.join(MODELS, "n%d" % n, "%s.meta.json" % name)


_START = 9300000
_LEDGER = """# test ledger
<!-- CONSUMED-DEV-BANDS:BEGIN -->
0-499 7000-7799 7800-8599 9000-9799 10000-10799 13200-13999 14000-14799
<!-- CONSUMED-DEV-BANDS:END -->

<!-- MULTI-SEAT-RESERVED:BEGIN -->
<!-- MULTI-SEAT-RESERVED:END -->

<!-- MULTI-SEAT-CONSUMED:BEGIN -->
<!-- MULTI-SEAT-CONSUMED:END -->
"""


def _copy_model(src_meta, directory, name):
    dst = os.path.join(directory, name + ".meta.json")
    shutil.copyfile(src_meta, dst)
    shutil.copyfile(harness.weights_path_for(src_meta),
                    harness.weights_path_for(dst))
    return dst


def _run(agents, start, clusters=1, workers=1):
    n = len(agents)
    cond = harness.multi_conditions(n, agents, "train", start, clusters,
                                    "test", None)
    run_id = harness.multi_run_id(cond)
    tasks = harness.cyclic_tasks_for(cond, run_id)
    return cond, run_id, harness.run_cyclic_games(tasks, workers=workers)


class SpecTests(unittest.TestCase):

    def test_model_specs_are_accepted_and_malformed_refused(self):
        self.assertEqual(harness.parse_multi_agent_spec("heuristic"),
                         ("heuristic", ""))
        self.assertEqual(harness.parse_multi_agent_spec("checkpoint:a.meta.json"),
                         ("checkpoint", "a.meta.json"))
        self.assertEqual(harness.parse_multi_agent_spec("strong:b.meta.json"),
                         ("strong", "b.meta.json"))
        for bad in ("strong", "checkpoint:", "strong:x.pt", "gumbel:x.meta.json",
                    "heuristic:x", ""):
            with self.assertRaises(ValueError, msg=bad):
                harness.parse_multi_agent_spec(bad)

    def test_player_count_mismatch_is_refused(self):
        for n, other in ((3, 4), (4, 3)):
            agents = ["checkpoint:" + _meta(other, "gen199")] + \
                ["heuristic"] * (n - 1)
            with self.assertRaisesRegex(ValueError, "seat"):
                harness.multi_conditions(n, agents, "train", _START, 1, "t",
                                         None)
            agents = ["strong:" + _meta(other, "initial")] + \
                ["heuristic"] * (n - 1)
            with self.assertRaisesRegex(ValueError, "seat"):
                harness.multi_conditions(n, agents, "train", _START, 1, "t",
                                         None)
        with self.assertRaises(ValueError):    # two-seat gen1300 never sits here
            harness.multi_conditions(
                3, ["checkpoint:" + os.path.join(
                    _ROOT, "docs", "eval-runs", "phase7-champion",
                    "gen1300.meta.json"), "heuristic", "heuristic"],
                "train", _START, 1, "t", None)

    def test_heuristic_only_conditions_have_no_model_keys(self):
        cond = harness.multi_conditions(3, ["heuristic", "random", "random"],
                                        "train", 870000, 20, "phase10-c3-smoke",
                                        0.5)
        self.assertNotIn("models", cond)
        self.assertNotIn("strong_search", cond)

    @requires_torch
    def test_strong_setting_is_fixed_and_built_from_cluster_seed(self):
        n = 3
        spec = "strong:" + _meta(n, "gen199")
        models = harness.multi_models(n, [spec, "heuristic", "heuristic"])
        agent, shas, search = harness._make_multi_agent(
            spec, 12345, load_cards(), n, models)
        from naeco.ai.search.ismcts import IsmctsAgent
        from naeco.train.search_evaluator import NetEvaluator
        from naeco.train.agent import PolicyAgent
        self.assertIsInstance(agent, IsmctsAgent)
        self.assertEqual(agent._budget_transitions, 4500)
        self.assertEqual(agent._c_puct, 1.5)
        self.assertEqual(agent._horizon, 2)
        self.assertIsNone(agent._budget_seconds)
        self.assertEqual(agent._root_policy, "puct")
        self.assertEqual(agent._root_seed, 12345)
        self.assertIsInstance(agent._evaluator, NetEvaluator)
        self.assertIsInstance(agent._policy_agent, PolicyAgent)
        self.assertEqual(search, dict(harness.STRONG_SEARCH, root_seed=12345))
        self.assertEqual(shas, harness.model_file_shas(_meta(n, "gen199")))
        # D5 fixed values; no argument of the public entry points changes them.
        self.assertEqual((harness.STRONG_SEARCH["horizon"],
                          harness.STRONG_SEARCH["c_puct"],
                          harness.STRONG_SEARCH["budget_transitions"]),
                         (2, 1.5, 4500))
        import inspect
        for fn in (harness.multi_conditions, harness.play_cyclic_game,
                   eval_agent.run_multi):
            params = set(inspect.signature(fn).parameters)
            self.assertFalse(params & {"budget", "budget_transitions", "c_puct",
                                       "horizon", "search"}, fn.__name__)
        checkpoint, unused, none_search = harness._make_multi_agent(
            "checkpoint:" + _meta(n, "gen199"), 7, load_cards(), n,
            harness.multi_models(n, ["checkpoint:" + _meta(n, "gen199"),
                                     "heuristic", "heuristic"]))
        self.assertIsInstance(checkpoint, PolicyAgent)
        self.assertEqual(checkpoint._mode, "greedy")
        self.assertIsNone(none_search)


@requires_torch
class StrongAndCheckpointRecordTests(unittest.TestCase):
    """Real n3/n4 models; one cluster each, workers 1 and 4."""

    @classmethod
    def setUpClass(cls):
        cls.lineups = {
            3: ["strong:" + _meta(3, "gen199"), "checkpoint:" + _meta(3, "gen199"),
                "heuristic"],
            4: ["strong:" + _meta(4, "gen199"), "checkpoint:" + _meta(4, "initial"),
                "heuristic", "heuristic"],
        }
        cls.runs = {}
        for n, agents in cls.lineups.items():
            start = _START + n
            one = _run(agents, start, workers=1)
            four = _run(agents, start, workers=4)
            cls.runs[n] = (one, four)

    def test_workers_one_and_four_are_byte_identical(self):
        for n, ((c1, r1, one), (c4, r4, four)) in self.runs.items():
            self.assertEqual(r1, r4)
            self.assertEqual("".join(harness.record_line(r) for r in one),
                             "".join(harness.record_line(r) for r in four),
                             "n=%d" % n)
            self.assertEqual(len(one), n)

    def test_records_carry_sha_search_and_cluster_seed(self):
        for n, ((cond, run_id, records), unused) in self.runs.items():
            agents = self.lineups[n]
            for rec in records:
                root = eval_seed("train", rec["index"])
                for seat in rec["seats"]:
                    seed = derive_seed(root, "policy", rec["index"], seat["seat"])
                    self.assertEqual(seat["policy_seed"], seed)
                    kind, path = harness.parse_multi_agent_spec(seat["agent"])
                    if kind == "heuristic":
                        self.assertIsNone(seat["model_sha256"])
                        self.assertIsNone(seat["search"])
                        continue
                    self.assertEqual(seat["model_sha256"],
                                     harness.model_file_shas(path))
                    if kind == "strong":
                        self.assertEqual(seat["search"]["budget_transitions"],
                                         4500)
                        self.assertEqual(seat["search"]["c_puct"], 1.5)
                        self.assertEqual(seat["search"]["horizon"], 2)
                        self.assertEqual(seat["search"]["root_seed"], seed)
                    else:
                        self.assertIsNone(seat["search"])
            # every agent (incl. strong) sat at every seat once
            seats_of_strong = sorted(
                s["seat"] for r in records for s in r["seats"]
                if s["agent"] == agents[0])
            self.assertEqual(seats_of_strong, list(range(n)))
            summary = harness.summarize_multi(records, cond, run_id)
            self.assertEqual(set(summary["models"]),
                             set(a for a in agents if ":" in a))
            self.assertEqual(summary["strong_search"], harness.STRONG_SEARCH)
            self.assertEqual(summary["games"], n)

    def test_tampered_provenance_is_refused(self):
        (cond, run_id, records), unused = self.runs[3]
        for field, value in (("policy_seed", 1), ("search", None),
                             ("model_sha256", {"meta": "0", "weights": "0"})):
            bad = json.loads(json.dumps(records))
            strong_seat = [s for s in bad[0]["seats"]
                           if s["agent"].startswith("strong:")][0]
            strong_seat[field] = value
            with self.assertRaises(ValueError, msg=field):
                harness.summarize_multi(bad, cond, run_id)


class ModelShaGuardTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER)
        self.model = _copy_model(_meta(3, "initial"), self.dir, "m")
        self.other = _copy_model(_meta(3, "gen199"), self.dir, "o")
        self.agents = ["checkpoint:" + self.model, "heuristic", "heuristic"]

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _swap(self):
        for suffix in (".meta.json", ".weights.pt"):
            shutil.copyfile(self.other[:-len(".meta.json")] + suffix,
                            self.model[:-len(".meta.json")] + suffix)

    def test_run_id_and_reservation_bind_the_model_sha(self):
        cond = harness.multi_conditions(3, self.agents, "train", _START + 10, 1,
                                        "t", None)
        entry = eval_agent.build_reservation("r", 3, [self.agents], "train",
                                             _START + 10, _START + 10, "t", None)
        key = " ".join(self.agents)
        self.assertEqual(entry["model_sha256"][key],
                         {"checkpoint:" + self.model:
                          harness.model_file_shas(self.model)})
        eval_agent.register_multi(entry, ledger_path=self.ledger)
        self._swap()
        cond2 = harness.multi_conditions(3, self.agents, "train", _START + 10,
                                         1, "t", None)
        self.assertNotEqual(harness.multi_run_id(cond),
                            harness.multi_run_id(cond2))
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "SHA"):
            eval_agent.run_multi("r", 3, self.agents, "train", _START + 10, 1,
                                 "t", None, os.path.join(self.dir, "x.jsonl"),
                                 os.path.join(self.dir, "x.json"),
                                 ledger_path=self.ledger)

    def test_model_swapped_before_play_is_refused(self):
        cond = harness.multi_conditions(3, self.agents, "train", _START + 11, 1,
                                        "t", None)
        self._swap()
        tasks = harness.cyclic_tasks_for(cond, harness.multi_run_id(cond))
        harness._NET_CACHE.clear()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            harness.run_cyclic_games(tasks[:1])
        with self.assertRaisesRegex(RuntimeError, "changed"):
            harness.verify_models_unchanged(cond)

    @requires_torch
    def test_cli_runs_checkpoint_seat_and_recomputes(self):
        start = _START + 12
        base = self.agents + ["--num-players", "3", "--group", "train",
                              "--start-index", str(start), "--clusters", "1",
                              "--reservation", "cli", "--purpose", "t",
                              "--threshold", "none", "--ledger", self.ledger]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(eval_agent.main(base + ["--register", "--execute"]),
                             0)
            self.assertEqual(eval_agent.main(base), 0)       # prepared only
            records = os.path.join(self.dir, "c.jsonl")
            summary = os.path.join(self.dir, "c.json")
            self.assertEqual(eval_agent.main(base + [
                "--records", records, "--summary", summary, "--execute"]), 0)
        self.assertIn("models=", out.getvalue())
        fresh = eval_agent.recompute_multi(summary)
        self.assertEqual(fresh["games"], 3)
        with open(summary, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual(saved["models"]["checkpoint:" + self.model]["sha256"],
                         harness.model_file_shas(self.model))
        same, unused = eval_agent.replay_multi(summary)
        self.assertTrue(same)
        with self.assertRaises(SystemExit):           # one launch per matchup
            eval_agent.main(base + ["--records", records + "2", "--summary",
                                    summary + "2", "--execute"])


class D5FormalPathTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _conditions(self, n, group="train", bench=None, direct=None,
                    clusters=1, champion=None, challenger=None, initial=None):
        bench = bench if bench is not None else _START + 100 + n * 10
        direct = direct if direct is not None else _START + 101 + n * 10
        return {
            "schema": promote.MULTI_D5_SCHEMA,
            "num_players": n,
            "champion": champion or _meta(n, "initial"),
            "challenger": challenger or _meta(n, "gen199"),
            "bench": {"reservation": "b%d" % n, "group": group,
                      "start_index": bench, "clusters": clusters,
                      "purpose": "phase10-bench-%dp" % n,
                      "entries": [{"name": "heuristic", "kind": "heuristic"},
                                  {"name": "initial", "kind": "checkpoint",
                                   "path": initial or _meta(n, "initial")}]},
            "direct": {"reservation": "p%d" % n, "group": group,
                       "start_index": direct, "clusters": clusters,
                       "purpose": "phase10-promote1-%dp" % n},
            "promote_threshold": 1.0 / n,
            "promote_tolerance": 0.03,
            "regression_tolerance": 0.05,
            "out_dir": os.path.join(self.dir, "out%d" % n),
        }

    def _write(self, cond, name="cond.json"):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cond, fh)
        return path

    def _main(self, path, n, stage, out, *extra):
        cond = json.load(open(path, encoding="utf-8"))
        return promote.main([
            "--num-players", str(n), "--conditions", path,
            "--seed-ledger", self.ledger, "--challenger", cond["challenger"],
            "--champion", cond["champion"], "--out", out, "--stage", stage,
        ] + list(extra))

    def test_matchups_follow_d5(self):
        for n in (3, 4):
            m = promote._d5_matchups(self._conditions(n))
            ch = "checkpoint:" + _meta(n, "gen199")
            cp = "checkpoint:" + _meta(n, "initial")
            self.assertEqual(m["direct"], [("direct", [ch] + [cp] * (n - 1))])
            self.assertEqual(dict(m["bench"]), {
                "bench-champion-heuristic": [cp] + ["heuristic"] * (n - 1),
                "bench-champion-initial": [cp] + [cp] * (n - 1),
                "bench-challenger-heuristic": [ch] + ["heuristic"] * (n - 1),
                "bench-challenger-initial": [ch] + [cp] * (n - 1),
            })

    def test_formal_validation_rejects_deviations(self):
        n = 3
        good = self._conditions(n, group="dev", bench=30000, direct=31600,
                                clusters=800)
        eval_agent.register_multi(eval_agent.build_hold(
            "b3", 3, "dev", 30000, 30799, "phase10-bench-3p", None),
            ledger_path=self.ledger)
        eval_agent.register_multi(eval_agent.build_hold(
            "p3", 3, "dev", 31600, 32399, "phase10-promote1-3p", 1.0 / 3),
            ledger_path=self.ledger)
        eval_agent.register_multi(eval_agent.build_hold(
            "p3b", 3, "dev", 32400, 33199, "phase10-promote2-3p", 1.0 / 3),
            ledger_path=self.ledger)
        promote._validate_d5(good, ledger_path=self.ledger)
        bad_cases = []
        c = json.loads(json.dumps(good)); c["direct"]["clusters"] = 799
        bad_cases.append(("clusters", c))
        c = json.loads(json.dumps(good)); c["bench"]["group"] = "train"
        bad_cases.append(("group", c))
        c = json.loads(json.dumps(good)); c["promote_threshold"] = 0.34
        bad_cases.append(("threshold", c))
        c = json.loads(json.dumps(good)); c["promote_tolerance"] = 0.04
        bad_cases.append(("tolerance", c))
        c = json.loads(json.dumps(good)); c["direct"]["reservation"] = "p3b"
        c["direct"]["start_index"] = 32400
        c["direct"]["purpose"] = "phase10-promote2-3p"
        bad_cases.append(("skipped promotion band", c))
        c = json.loads(json.dumps(good)); c["direct"]["reservation"] = "nope"
        bad_cases.append(("no reservation", c))
        c = json.loads(json.dumps(good))
        c["bench"]["entries"][1]["path"] = _meta(3, "gen199")
        bad_cases.append(("initial is not the frozen champion", c))
        c = json.loads(json.dumps(good))
        c["bench"]["entries"] = c["bench"]["entries"][:1]
        bad_cases.append(("bench entries", c))
        c = json.loads(json.dumps(good)); c["challenger"] = c["champion"]
        bad_cases.append(("same model", c))
        c = json.loads(json.dumps(good)); c["champion"] = _meta(4, "initial")
        bad_cases.append(("wrong player count", c))
        for name, cond in bad_cases:
            with self.assertRaises(ValueError, msg=name):
                promote._validate_d5(cond, ledger_path=self.ledger)

    def test_real_ledger_accepts_the_registered_bands(self):
        for n, bench, direct in ((3, 30000, 31600), (4, 30800, 32400)):
            cond = self._conditions(n, group="dev", bench=bench, direct=direct,
                                    clusters=800)
            cond["bench"]["reservation"] = "p10-bench-%dp" % n
            cond["bench"]["purpose"] = "phase10-initial-bench-%dp" % n
            cond["direct"]["reservation"] = "p10-promote1-%dp" % n
            promote._validate_d5(cond)       # reads docs/eval-seeds.md only
            plan = promote.d5_register(cond, execute=False)
            self.assertEqual([p["status"] for p in plan],
                             ["would register", "would register"])
            self.assertEqual(len(plan[0]["matchups"]), 4)
            self.assertEqual(len(plan[1]["matchups"]), 1)

    @requires_torch
    def test_informal_train_run_bench_then_promote(self):
        n = 3
        cond = self._conditions(n)
        path = self._write(cond)
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet), \
                contextlib.redirect_stderr(quiet):
            # formal mode refuses train bands
            self.assertNotEqual(0, self._safe_main(path, n, "register", "x"))
            eval_agent.register_multi(eval_agent.build_hold(
                "b3", 3, "train", cond["bench"]["start_index"],
                cond["bench"]["start_index"], cond["bench"]["purpose"], None),
                ledger_path=self.ledger)
            eval_agent.register_multi(eval_agent.build_hold(
                "p3", 3, "train", cond["direct"]["start_index"],
                cond["direct"]["start_index"], cond["direct"]["purpose"],
                1.0 / 3), ledger_path=self.ledger)
            self.assertEqual(0, self._main(path, n, "register", "x",
                                           "--allow-informal", "--execute"))
            self.assertEqual(0, self._main(path, n, "register", "x",
                                           "--allow-informal", "--execute"))
            out_bench = os.path.join(self.dir, "bench.json")
            out_promote = os.path.join(self.dir, "promote.json")
            # promote before bench is refused (champion record missing)
            self.assertEqual(1, self._main(path, n, "promote", out_promote,
                                           "--allow-informal", "--execute"))
            self.assertEqual(0, self._main(path, n, "bench", out_bench,
                                           "--allow-informal", "--execute"))
            self.assertEqual(0, self._main(path, n, "promote", out_promote,
                                           "--allow-informal", "--execute"))
        with open(out_promote, encoding="utf-8") as fh:
            result = json.load(fh)
        self.assertFalse(result["formal"])
        self.assertFalse(result["certified"])
        self.assertEqual(result["direct"]["agents"],
                         ["checkpoint:" + cond["challenger"]] +
                         ["checkpoint:" + cond["champion"]] * 2)
        self.assertEqual(result["direct"]["threshold"], 1.0 / 3)
        self.assertEqual(result["direct"]["passed"],
                         result["direct"]["ci_low"] > 1.0 / 3)
        expect = promote.decide(
            result["direct"]["passed"],
            result["benchmark"]["challenger_record"],
            result["benchmark"]["champion_record"], 0.03, 0.05)
        self.assertEqual(result["decision"], expect)
        self.assertEqual(set(result["benchmark"]["champion_record"]),
                         {"heuristic", "initial"})
        for row in result["benchmark"]["champion_runs"]:
            self.assertTrue(row["reused_existing_summary"])
        unused, reserved, consumed, unused_two = \
            eval_agent.read_multi_ledger(self.ledger)
        self.assertEqual(len(consumed), 5)       # 4 bench matchups + direct
        # double consumption: drop a summary and relaunch without --resume
        os.remove(os.path.join(cond["out_dir"], "direct.json"))
        os.remove(os.path.join(cond["out_dir"], "direct.jsonl"))
        with contextlib.redirect_stdout(quiet), \
                contextlib.redirect_stderr(quiet):
            self.assertEqual(1, self._main(path, n, "promote",
                                           out_promote + "2",
                                           "--allow-informal", "--execute"))
        self.assertIn("already consumed", quiet.getvalue())

    def _safe_main(self, *args):
        try:
            return self._main(*args)
        except SystemExit as exc:
            return exc.code or 1

    def test_decision_rules(self):
        champ = {"heuristic": 0.50, "initial": 0.60}
        self.assertEqual(promote.decide(True, {"heuristic": 0.47,
                                               "initial": 0.57}, champ,
                                        0.03, 0.05), "promote")
        self.assertEqual(promote.decide(True, {"heuristic": 0.4699,
                                               "initial": 0.60}, champ,
                                        0.03, 0.05), "reject")
        self.assertEqual(promote.decide(False, {"heuristic": 0.60,
                                                "initial": 0.70}, champ,
                                        0.03, 0.05), "reject")
        self.assertEqual(promote.decide(True, {"heuristic": 0.4499,
                                               "initial": 0.70}, champ,
                                        0.03, 0.05), "rollback")

    def test_two_seat_cli_rejects_d5_flags(self):
        with self.assertRaises(SystemExit):
            promote.main(["--conditions", "x.json", "--challenger", "a.meta.json",
                          "--champion", "b.meta.json", "--out", "o.json",
                          "--stage", "bench"])


class InitialChampionTests(unittest.TestCase):

    def test_record_matches_files_and_loads_per_player_count(self):
        with open(INITIAL_RECORD, encoding="utf-8") as fh:
            record = json.load(fh)
        self.assertFalse(record["certified"])
        self.assertFalse(record["identity_with_main_run"]["proven"])
        for n in (3, 4):
            entry = record["models"][str(n)]
            shas = harness.model_file_shas(_meta(n, "initial"))
            self.assertEqual(shas, {"meta": entry["meta_sha256"],
                                    "weights": entry["weights_sha256"]})
            self.assertEqual(record["regeneration_runs"]["csi_run1"][str(n)],
                             record["regeneration_runs"]["csi_run2"][str(n)])
            self.assertEqual(entry["weights_sha256"],
                             record["verify2_comparison"][str(n)]
                             ["initial.weights.pt"]["sha256"])
            harness.check_model_for_players(_meta(n, "initial"), n)
            with self.assertRaises(ValueError):
                harness.check_model_for_players(_meta(n, "initial"), 7 - n)

    @requires_torch
    def test_be36450_construction_reproduces_the_weights(self):
        from naeco.train import encoding_multi
        from naeco.train import net as net_mod
        from naeco.train import ppo
        for n in (3, 4):
            ppo.set_num_threads(1)
            ppo.manual_seed(20260812)
            net = net_mod.PolicyValueNet(
                encoding_multi.obs_dim(n), encoding_multi.action_dim(),
                hidden=(256, 256), action_hidden=(128,))
            saved = ppo.load_state_dict(
                harness.weights_path_for(_meta(n, "initial")))
            fresh = net.state_dict()
            self.assertEqual(set(saved), set(fresh))
            for key in fresh:
                self.assertTrue(ppo.torch.equal(saved[key], fresh[key]),
                                "n=%d %s" % (n, key))


if __name__ == "__main__":
    unittest.main()
