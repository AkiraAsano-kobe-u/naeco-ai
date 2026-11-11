"""Phase 12b C1 (t_1cc2cca7): phase12b-multi/v1 evaluation protocol.

Pinned here:
- thresholds (G1 1/n, G2 8/15 and 19/40), alpha 0.0125 / 97.5% CI, 1000 clusters,
  G1/G2 power ~0.8214 (800 would be ~0.7214) and the separate R4 CI sensitivity
  ~0.3654 (sigma .5, true diff 0, 1000 clusters, alpha .0125, margin .03);
- 95% and 97.5% CIs from a fixture of saved records, recomputed by hand from
  the bootstrap means; paired d (R4 / d_E) from saved records;
- refusal of duplicate bands, partial clusters, duplicate games, mixing the
  old phase10-cyclic/v1 protocol, unfrozen holds, unregistered matchups;
- workers 1 and 4 give byte-identical records;
- the old Phase 10 D4 path and saved records still recompute bit-identically.
Train seeds only (index 19950000+ in temp ledgers); no dev/accept is consumed.
"""
import importlib.util
import json
import math
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
from naeco.ai.eval import harness  # noqa: E402
from naeco.ai.eval import phase12b as p  # noqa: E402

requires_torch = unittest.skipUnless(
    importlib.util.find_spec("torch") is not None, "torch が無い環境")

_START = 19950000
_LEDGER = """# test ledger
<!-- CONSUMED-DEV-BANDS:BEGIN -->
0-499 7000-7799 7800-8599 9000-9799 10000-10799 13200-13999 14000-14799
<!-- CONSUMED-DEV-BANDS:END -->

<!-- MULTI-SEAT-RESERVED:BEGIN -->
<!-- MULTI-SEAT-RESERVED:END -->

<!-- MULTI-SEAT-CONSUMED:BEGIN -->
<!-- MULTI-SEAT-CONSUMED:END -->
"""
MODELS = os.path.join(_ROOT, "docs", "eval-runs", "phase10", "models")
SMOKE = os.path.join(_ROOT, "docs", "eval-runs", "phase12b", "c1")


def _meta(n, name):
    return os.path.join(MODELS, "n%d" % n, "%s.meta.json" % name)


def _fake_records(cond, run_id, wins_by_cluster):
    """Saved-record fixture: n records per cluster consistent with harness checks.

    wins_by_cluster[c] is the set of k whose target wins. Scores are chosen so
    the ranking follows the RUL-SCORE-002 recomputation.
    """
    n = cond["num_players"]
    out = []
    for c, wins in enumerate(wins_by_cluster):
        index = cond["start_index"] + c
        agents = p.lineup_for(cond, index)
        for k in range(n):
            seat_agents = harness.cyclic_seat_order(n, k)
            order = [seat_agents.index(j) for j in range(n)]
            target = order[0]
            winner = target if k in wins else (target + 1) % n
            scores = [10 - ((s - winner) % n) for s in range(n)]
            ranking = [0] * n
            for place, pid in enumerate(sorted(range(n),
                                               key=lambda q: -scores[q])):
                ranking[pid] = place
            seats = []
            from naeco.engine.state import derive_seed, eval_seed
            root = eval_seed(cond["group"], index)
            for seat in range(n):
                seats.append({"seat": seat, "agent_index": seat_agents[seat],
                              "agent": agents[seat_agents[seat]],
                              "model_sha256": None,
                              "policy_seed": derive_seed(root, "policy", index,
                                                         seat),
                              "search": None})
            out.append({
                "schema": harness.MULTI_RECORD_SCHEMA, "protocol": p.PROTOCOL,
                "run_id": run_id, "group": cond["group"], "index": index,
                "n": n, "seat_variant": k, "agents": agents, "order": order,
                "seat_agents": seat_agents, "target_seat": target,
                "seats": seats, "env_seed": 1, "setup_seed": 2,
                "start_player": 0, "initial_state_hash": "h%d" % index,
                "final_start_player": 0, "scores": scores, "ranking": ranking,
                "utilities": [1.0 - 2.0 * r / (n - 1) for r in ranking],
                "target_rank": ranking[target], "target_score": scores[target],
                "target_win": 1 if ranking[target] == 0 else 0,
                "decisions": 1})
    return out


def _bench_cond(challenger, opponents, n=3, clusters=4, start=_START):
    return p.conditions(n, "R4", challenger, opponents, "train", start,
                        clusters, "fixture", "fixture")


class ThresholdAndPowerTests(unittest.TestCase):

    def test_thresholds_are_the_registered_constants(self):
        from fractions import Fraction as F
        self.assertEqual(p.g1_threshold(3), F(1, 3))
        self.assertEqual(p.g1_threshold(4), F(1, 4))
        self.assertEqual(p.g2_threshold(3), F(8, 15))
        self.assertEqual(p.g2_threshold(4), F(19, 40))
        # back to n=2 the same excess rate is the Phase 7 0.65
        self.assertEqual(F(1, 2) + p.G2_EXCESS * F(1, 2), F(13, 20))
        self.assertEqual(p.ALPHA_ONE_SIDED, F(1, 80))
        self.assertEqual(p.CI_PERCENTILES["97.5"], (0.0125, 0.9875))
        self.assertEqual(p.FORMAL_CLUSTERS, 1000)
        for n in (2, 5, "3"):
            with self.assertRaises(p.ProtocolError):
                p.baseline(n)

    def test_power_and_r4_sensitivity(self):
        self.assertEqual(p.required_clusters(), 951)
        self.assertAlmostEqual(p.approximate_power(1000), 0.8214, places=4)
        self.assertAlmostEqual(p.approximate_power(800), 0.7214, places=4)
        self.assertAlmostEqual(p.r4_ci_sensitivity(), 0.3654, places=4)
        # hand formula: Phi(0.03/(0.5/sqrt(1000)) - z_{1-1/80})
        z = (0.03 / (0.5 / math.sqrt(1000))) - p.Z_ALPHA
        self.assertAlmostEqual(p.r4_ci_sensitivity(),
                               0.5 * (1 + math.erf(z / math.sqrt(2))), 12)
        rec = p.r4_sensitivity_record()
        self.assertIn("excludes the point-estimate", rec["note"])


class FixtureRecomputeTests(unittest.TestCase):
    """CIs and paired diffs from saved-record fixtures (no game is played)."""

    def setUp(self):
        self.opp = ["heuristic", "random"]
        self.ca = _bench_cond("heuristic", self.opp)
        self.cb = _bench_cond("random", self.opp)
        self.ra, self.rb = p.run_id_for(self.ca), p.run_id_for(self.cb)
        self.wins_a = [{0, 1, 2}, {0, 1}, {1}, set()]
        self.wins_b = [{0}, {0}, set(), {2}]
        self.rec_a = _fake_records(self.ca, self.ra, self.wins_a)
        self.rec_b = _fake_records(self.cb, self.rb, self.wins_b)

    def test_opponent_order_is_balanced_by_cluster_index(self):
        self.assertEqual(self.ca["opponent_orders"],
                         [["heuristic", "random"], ["random", "heuristic"]])
        self.assertEqual(p.lineup_for(self.ca, _START), ["heuristic",
                                                         "heuristic", "random"])
        self.assertEqual(p.lineup_for(self.ca, _START + 1),
                         ["heuristic", "random", "heuristic"])
        same = p.conditions(3, "R4", "random", ["heuristic", "heuristic"],
                            "train", _START, 1, "x", "x")
        self.assertEqual(same["opponent_orders"], [["heuristic", "heuristic"]])

    def test_summary_cis_match_hand_bootstrap(self):
        s = p.summarize(self.rec_a, self.ca, self.ra)
        values = [len(w) / 3.0 for w in self.wins_a]
        self.assertAlmostEqual(s["point_estimate"], sum(values) / 4)
        means = sorted(p.bootstrap_means(values))
        def pct(q):
            i = (len(means) - 1) * q
            lo = int(i)
            return means[lo] + (means[min(lo + 1, len(means) - 1)] -
                                means[lo]) * (i - lo)
        self.assertEqual(s["ci"]["95"], [pct(0.025), pct(0.975)])
        self.assertEqual(s["ci"]["97.5"], [pct(0.0125), pct(0.9875)])
        # 95% equals the legacy harness bootstrap on the same values
        self.assertEqual(tuple(s["ci"]["95"]),
                         harness.cluster_bootstrap(values, 10000, 2026072102))
        self.assertLessEqual(s["ci"]["97.5"][0], s["ci"]["95"][0])

    def test_g_gate_uses_the_97_5_lower_bound(self):
        cond = p.conditions(3, "G1", "checkpoint:" + _meta(3, "initial"),
                            ["strong:" + _meta(3, "gen199")] * 2, "train",
                            _START, 4, "fixture", "fixture") \
            if os.path.exists(_meta(3, "initial")) else None
        if cond is None:
            self.skipTest("phase10 models are absent")
        run_id = p.run_id_for(cond)
        recs = _fake_records(cond, run_id, [{0, 1, 2}] * 3 + [{0}])
        for r in recs:            # model seats carry their SHA/search
            for seat in r["seats"]:
                info = cond["models"][seat["agent"]]
                seat["model_sha256"] = info["sha256"]
                if info["kind"] == "strong":
                    seat["search"] = dict(harness.STRONG_SEARCH,
                                          root_seed=seat["policy_seed"])
        s = p.summarize(recs, cond, run_id)
        self.assertEqual(s["gate"]["threshold"], "1/3")
        self.assertEqual(s["gate"]["passed"], s["ci"]["97.5"][0] > 1 / 3.0)

    def test_paired_r4_and_exploit_from_saved_records(self):
        d = p.paired("R4", self.rec_a, self.ca, self.ra, self.rec_b, self.cb,
                     self.rb)
        diffs = [len(a) / 3.0 - len(b) / 3.0
                 for a, b in zip(self.wins_a, self.wins_b)]
        self.assertEqual([x["diff"] for x in d["cluster_diffs"]], diffs)
        self.assertAlmostEqual(d["point_estimate"], sum(diffs) / 4)
        self.assertEqual(d["ci"], p.intervals(diffs))
        self.assertEqual(d["gate"]["passed"],
                         d["point_estimate"] >= 0 and
                         d["ci"]["97.5"][0] > -0.03)
        self.assertEqual(d["gate"]["dev_reject"], d["point_estimate"] < -0.05)
        with self.assertRaises(p.ProtocolError):          # wrong kind
            p.paired("EXPLOIT", self.rec_a, self.ca, self.ra, self.rec_b,
                     self.cb, self.rb)
        # EXPLOIT needs model seats; its paired d_E is covered by the saved smoke.
        with self.assertRaises(p.ProtocolError):
            p.conditions(3, "EXPLOIT", "heuristic", ["random", "random"],
                         "train", _START, 2, "fixture", "fixture")

    def test_paired_rejects_mismatched_deals_and_benches(self):
        bad = json.loads(json.dumps(self.rec_b))
        bad[0]["initial_state_hash"] = "other"
        for r in bad:
            if r["index"] == _START:
                r["initial_state_hash"] = "other"
        with self.assertRaisesRegex(p.ProtocolError, "initial_state_hash"):
            p.paired("R4", self.rec_a, self.ca, self.ra, bad, self.cb, self.rb)
        other = _bench_cond("random", ["random", "random"])
        with self.assertRaisesRegex(p.ProtocolError, "same bench"):
            p.paired("R4", self.rec_a, self.ca, self.ra,
                     _fake_records(other, p.run_id_for(other), self.wins_b),
                     other, p.run_id_for(other))
        shorter = _bench_cond("random", self.opp, clusters=3)
        with self.assertRaises(p.ProtocolError):
            p.paired("R4", self.rec_a, self.ca, self.ra,
                     _fake_records(shorter, p.run_id_for(shorter),
                                   self.wins_b[:3]),
                     shorter, p.run_id_for(shorter))

    def test_partial_duplicate_and_old_protocol_records_are_refused(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            p.summarize(self.rec_a[:-1], self.ca, self.ra)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.summarize(self.rec_a + self.rec_a[:1], self.ca, self.ra)
        old = json.loads(json.dumps(self.rec_a))
        old[0]["protocol"] = harness.MULTI_PROTOCOL
        with self.assertRaisesRegex(ValueError, "phase12b-multi/v1"):
            p.summarize(old, self.ca, self.ra)
        swapped = json.loads(json.dumps(self.rec_a))
        for r in swapped:
            if r["index"] == _START + 1:
                r["agents"] = p.lineup_for(self.ca, _START)
        with self.assertRaisesRegex(ValueError, "agent list"):
            p.summarize(swapped, self.ca, self.ra)
        legacy = dict(self.ca, protocol=harness.MULTI_PROTOCOL)
        with self.assertRaises(ValueError):
            p.summarize(self.rec_a, legacy, p.run_id_for(legacy))
        # the old D4 summarizer refuses the new records as well
        with self.assertRaises(ValueError):
            harness.multi_cluster_results(self.rec_a, self.ra, _START, 4, 3,
                                          self.ca["canonical_agents"])

    def test_decide_without_frozen_holds_is_no_judgement(self):
        g = {"schema": p.SUMMARY_SCHEMA, "num_players": 3, "group": "train",
             "gate": {"passed": True}, "bootstrap_samples": p.BOOTSTRAP_SAMPLES,
             "bootstrap_seed": p.BOOTSTRAP_SEED}
        g1, g2 = dict(g, role="G1"), dict(g, role="G2")
        r4 = p.paired("R4", self.rec_a, self.ca, self.ra, self.rec_b, self.cb,
                      self.rb)
        for args in ((g1, g2, [], []), (g1, g2, [r4], [])):
            d = p.decide(3, *args)
            self.assertFalse(d["complete"])
            self.assertFalse(d["all_conditions"])
            self.assertIsNone(d["G1"])
            self.assertIn("frozen", d["refusals"][0])
        with self.assertRaises(p.ProtocolError):
            p.decide(2, g1, g2, [r4], [])


_CHK3 = "checkpoint:" + _meta(3, "initial")
_CAND3 = "strong:" + _meta(3, "initial")
_B3 = "strong:" + _meta(3, "gen199")
_BENCHES3 = (["random", "random"], ["heuristic", "heuristic"])


def _full_matchups(benches=_BENCHES3):
    """One frozen judgement: G1, G2, each bench for candidate and B, exploiter
    against the candidate table and the B table (n=3, candidate=initial,
    B=gen199; models are the saved Phase 10 ones, this is not a CPU result)."""
    b, c = [_B3] * 2, [_CAND3] * 2
    out = [{"role": "G1", "challenger": _CHK3, "opponents": b},
           {"role": "G2", "challenger": _CAND3, "opponents": b}]
    for bench in benches:
        for ch in (_CAND3, _B3):
            out.append({"role": "R4", "challenger": ch, "opponents": list(bench)})
    for opp in (c, b):
        out.append({"role": "EXPLOIT", "challenger": _CHK3, "opponents": opp})
    return out


def _freeze(ledger, rid, group, lo, hi, kind, matchups):
    eval_agent.register_p12b(eval_agent.build_p12b_hold(
        rid, 3, group, lo, hi, kind, "fixture-" + rid), ledger_path=ledger)
    return eval_agent.freeze_p12b(rid, matchups, ledger_path=ledger)


def _stub_summary(row, m, passed=True):
    """decide input for one matchup (the gate value is given, not computed)."""
    cond = p.conditions_from_hold(row, m["role"], m["challenger"],
                                  m["opponents"])
    return {"schema": p.SUMMARY_SCHEMA, "protocol": p.PROTOCOL,
            "role": m["role"], "num_players": cond["num_players"],
            "group": cond["group"], "run_id": p.run_id_for(cond),
            "challenger": cond["challenger"], "opponents": cond["opponents"],
            "start_index": cond["start_index"], "end_index": cond["end_index"],
            "reservation_id": cond["reservation_id"],
            "bootstrap_samples": p.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": p.BOOTSTRAP_SEED, "gate": {"passed": passed}}


def _stub_pair(kind, sa, sb, passed=True):
    return {"schema": p.PAIRED_SCHEMA, "protocol": p.PROTOCOL, "kind": kind,
            "num_players": sa["num_players"], "group": sa["group"],
            "run_a": sa["run_id"], "run_b": sb["run_id"],
            "challenger_a": sa["challenger"], "challenger_b": sb["challenger"],
            "opponents_a": sa["opponents"], "opponents_b": sb["opponents"],
            "point_estimate": 0.0, "ci": {"95": [0.0, 0.0],
                                          "97.5": [0.0, 0.0]},
            "bootstrap_samples": p.BOOTSTRAP_SAMPLES,
            "bootstrap_seed": p.BOOTSTRAP_SEED,
            "gate": {"passed": passed, "dev_reject": False}}


def _inputs(holds):
    """Complete decide inputs for every frozen matchup of the holds."""
    s = {}
    for row in holds:
        for m in row["matchups"]:
            s[eval_agent._p12b_matchup_key(m)] = (m, _stub_summary(row, m))
    by_role = lambda role: [(m, x) for m, x in s.values() if m["role"] == role]
    g1 = by_role("G1")[0][1]
    g2 = by_role("G2")[0][1]
    r4 = []
    for m, x in by_role("R4"):
        if m["challenger"] == _CAND3:
            other = [y for mm, y in by_role("R4") if mm["challenger"] == _B3
                     and mm["opponents"] == m["opponents"]][0]
            r4.append(_stub_pair("R4", x, other))
    ex = []
    for m, x in by_role("EXPLOIT"):
        if m["opponents"] == [_CAND3] * 2:
            other = [y for mm, y in by_role("EXPLOIT")
                     if mm["challenger"] == m["challenger"]
                     and mm["opponents"] == [_B3] * 2][0]
            ex.append(_stub_pair("EXPLOIT", x, other))
    return g1, g2, r4, ex


@unittest.skipUnless(os.path.exists(_meta(3, "initial")),
                     "phase10 models are absent")
class FrozenDecisionTests(unittest.TestCase):
    """C2 review B1/B2: decide matches the frozen holds exactly (fail-closed).

    Temp ledgers only; nothing is launched and no dev/accept band is consumed.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER)
        full = _full_matchups()
        self.accept = _freeze(self.ledger, "fx-accept-3p", "accept", 2400,
                              3399, "ACCEPT", full)
        self.dev = [
            _freeze(self.ledger, "fx-dev-g-3p", "dev", 33200, 34199, "G",
                    [m for m in full if m["role"] in ("G1", "G2")]),
            _freeze(self.ledger, "fx-dev-r4-3p", "dev", 35200, 36199, "R4",
                    [m for m in full if m["role"] == "R4"]),
            _freeze(self.ledger, "fx-dev-ex-3p", "dev", 37200, 38199, "EXPLOIT",
                    [m for m in full if m["role"] == "EXPLOIT"])]
        self.train = _freeze(self.ledger, "fx-train-3p", "train", _START,
                             _START, "SMOKE", full)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _decide(self, inputs, holds):
        return p.decide(3, *inputs, holds=holds)

    def test_complete_accept_and_dev_judgements(self):
        d = self._decide(_inputs([self.accept]), [self.accept])
        self.assertEqual(d["refusals"], [])
        self.assertTrue(d["complete"] and d["all_conditions"] and d["formal"])
        self.assertEqual((d["group"], len(d["r4_benches"]),
                          len(d["exploiters"])), ("accept", 2, 1))
        # dev: three holds with different reservation ids are one judgement
        d = self._decide(_inputs(self.dev), self.dev)
        self.assertEqual(d["refusals"], [])
        self.assertTrue(d["all_conditions"])
        self.assertEqual(d["reservation_ids"],
                         ["fx-dev-ex-3p", "fx-dev-g-3p", "fx-dev-r4-3p"])
        # the G2 -> R4 order is kept on a complete input
        g1, g2, r4, ex = _inputs([self.accept])
        d = self._decide((g1, dict(g2, gate={"passed": False}), r4, ex),
                         [self.accept])
        self.assertTrue(d["complete"])
        self.assertIsNone(d["R4"])
        self.assertFalse(d["all_conditions"])

    def test_b1_each_missing_extra_or_duplicate_pair_is_refused(self):
        g1, g2, r4, ex = _inputs([self.accept])
        cases = []
        for i in range(len(r4)):
            cases.append(("missing frozen R4",
                          (g1, g2, r4[:i] + r4[i + 1:], ex)))
        for i in range(len(ex)):
            cases.append(("missing frozen EXPLOIT",
                          (g1, g2, r4, ex[:i] + ex[i + 1:])))
        cases.append(("duplicate R4", (g1, g2, r4 + r4[:1], ex)))
        cases.append(("duplicate EXPLOIT", (g1, g2, r4, ex + ex[:1])))
        mixed = dict(r4[0], run_b=r4[1]["run_b"])
        cases.append(("extra R4", (g1, g2, r4 + [mixed], ex)))
        cases.append(("is not a EXPLOIT", (g1, g2, r4, ex + r4[:1])))
        cases.append(("missing frozen R4", (g1, g2, [], [])))
        for reason, args in cases:
            d = self._decide(args, [self.accept])
            self.assertFalse(d["complete"], reason)
            self.assertFalse(d["all_conditions"], reason)
            self.assertIsNone(d["R4"], reason)
            self.assertTrue(any(reason in r for r in d["refusals"]),
                            (reason, d["refusals"]))
        # dev: dropping the whole exploiter hold or any one hold is refused
        dev_inputs = _inputs(self.dev)
        for i in range(3):
            d = self._decide(dev_inputs, self.dev[:i] + self.dev[i + 1:])
            self.assertFalse(d["complete"])
            self.assertIn("needs exactly the holds", d["refusals"][0])

    def test_b2_other_candidate_stage_or_size_is_refused(self):
        g1, g2, r4, ex = _inputs([self.accept])
        b = [_B3] * 2
        swaps = {
            # another candidate (gen199) on the same accept band
            "model": _stub_summary(self.accept, {"role": "G2",
                                                 "challenger": _B3,
                                                 "opponents": b}),
            # the frozen G2 matchup but from the dev stage
            "group": _stub_summary(self.dev[0], {"role": "G2",
                                                 "challenger": _CAND3,
                                                 "opponents": b}),
            # the frozen G2 matchup but a 1-cluster train run
            "size": _stub_summary(self.train, {"role": "G2",
                                               "challenger": _CAND3,
                                               "opponents": b}),
        }
        for what, other in swaps.items():
            d = self._decide((g1, other, r4, ex), [self.accept])
            self.assertFalse(d["complete"], what)
            self.assertFalse(d["all_conditions"], what)
            self.assertTrue(any("G2 summary" in r for r in d["refusals"]),
                            (what, d["refusals"]))
        # an R4 pair of another candidate is an extra pair (and leaves a gap)
        tr = _inputs([self.train])
        d = self._decide((g1, g2, [tr[2][0]] + r4[1:], ex), [self.accept])
        self.assertFalse(d["complete"])
        self.assertTrue(any(r.startswith("extra R4") for r in d["refusals"]))
        # holds of two stages, or a train hold for a formal input, are refused
        d = self._decide((g1, g2, r4, ex), [self.accept, self.train])
        self.assertIn("different evaluation stages", d["refusals"][0])
        d = self._decide((g1, g2, r4, ex), [self.train])
        self.assertFalse(d["complete"])

    def test_inconsistent_frozen_holds_are_refused(self):
        full = _full_matchups()
        bad = {
            "no exploiter": [m for m in full if m["role"] != "EXPLOIT"],
            "G2 of another candidate": [m for m in full if m["role"] != "G2"] +
            [{"role": "G2", "challenger": _B3, "opponents": [_B3] * 2}],
            "R4 without B": [m for m in full if not (
                m["role"] == "R4" and m["challenger"] == _B3 and
                m["opponents"] == ["random", "random"])],
        }
        for i, (what, matchups) in enumerate(sorted(bad.items())):
            row = _freeze(self.ledger, "fx-bad-%d" % i, "train",
                          _START + 10 + i, _START + 10 + i, "SMOKE", matchups)
            if what == "no exploiter":     # train smoke may omit exploiters,
                row = dict(row, group="accept", kind="ACCEPT", lo=2400,
                           hi=3399)        # a formal judgement may not
            with self.assertRaises(p.ProtocolError, msg=what):
                p.expected_matchups(3, [row])
        unfrozen = eval_agent.build_p12b_hold("u", 3, "accept", 2400, 3399,
                                              "ACCEPT", "u")
        with self.assertRaisesRegex(p.ProtocolError, "not frozen"):
            p.expected_matchups(3, [unfrozen])
        with self.assertRaisesRegex(p.ProtocolError, "SHA-256"):
            row = json.loads(json.dumps(self.accept))
            key = sorted(row["model_sha256"])[0]
            spec = sorted(row["model_sha256"][key])[0]
            row["model_sha256"][key][spec]["meta"] = "0" * 64
            p.expected_matchups(3, [row])

    def test_cli_decide_needs_the_holds(self):
        """`phase12b decide` reads --hold from the ledger; exit 3 if incomplete."""
        g1, g2, r4, ex = _inputs([self.accept])
        rows = eval_agent.p12b_decision_holds(["fx-accept-3p"],
                                              ledger_path=self.ledger)
        self.assertEqual(rows, [self.accept])
        self.assertTrue(p.decide(3, g1, g2, r4, ex, holds=rows)["complete"])
        self.assertEqual(eval_agent.p12b_decision_holds(
            [], ledger_path=self.ledger), [])
        with self.assertRaises(eval_agent.SeedLedgerError):
            eval_agent.p12b_decision_holds(["nope"], ledger_path=self.ledger)


def _model_records(cond, run_id, wins):
    recs = _fake_records(cond, run_id, wins)
    for r in recs:
        for seat in r["seats"]:
            info = (cond.get("models") or {}).get(seat["agent"])
            if info:
                seat["model_sha256"] = info["sha256"]
                if info["kind"] == "strong":
                    seat["search"] = dict(harness.STRONG_SEARCH,
                                          root_seed=seat["policy_seed"])
    return recs


@unittest.skipUnless(os.path.exists(_meta(3, "initial")),
                     "phase10 models are absent")
class ReviewFixtureTests(unittest.TestCase):
    """The C2 reviewer's formal 1000-cluster fixture (docs/reviews/
    phase12b-pretrain.md): G1/G2 all wins and a single random-bench R4 pair,
    no exploiters. Before the fix decide returned group=accept and
    all_conditions=True for it (B1) and for a train gen199 G2 swapped in (B2).
    Synthetic records, no game is played, no ledger band is consumed."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        ledger = os.path.join(cls.dir, "eval-seeds.md")
        with open(ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER)
        cls.hold = _freeze(ledger, "fx-accept-3p", "accept", 2400, 3399,
                           "ACCEPT", _full_matchups())
        b = [_B3] * 2

        def run(row, role, ch, opp, wins):
            cond = p.conditions_from_hold(row, role, ch, opp)
            rid = p.run_id_for(cond)
            return _model_records(cond, rid, wins), cond, rid

        allwins = [{0, 1, 2}] * 1000
        cls.g1 = p.summarize(*run(cls.hold, "G1", _CHK3, b, allwins))
        cls.g2 = p.summarize(*run(cls.hold, "G2", _CAND3, b, allwins))
        a = run(cls.hold, "R4", _CAND3, ["random"] * 2, [{0}] * 1000)
        bb = run(cls.hold, "R4", _B3, ["random"] * 2, [{0}] * 1000)
        cls.pair = p.paired("R4", *(a + bb))
        train = _freeze(ledger, "other-candidate", "train", _START, _START,
                        "SMOKE", [{"role": "G2", "challenger": _B3,
                                   "opponents": b}])
        cls.other = p.summarize(*run(train, "G2", _B3, b, [{0, 1, 2}]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir)

    def test_b1_formal_fixture_with_one_bench_and_no_exploiter(self):
        self.assertEqual((self.g1["clusters"], self.g1["group"]),
                         (1000, "accept"))
        self.assertTrue(self.g1["gate"]["passed"] and
                        self.g2["gate"]["passed"] and
                        self.pair["gate"]["passed"])
        # the reviewer's exact call (no frozen expectation)
        d = p.decide(3, self.g1, self.g2, [self.pair], [])
        self.assertFalse(d["all_conditions"], d)
        self.assertFalse(d["complete"])
        # against the frozen accept hold: heuristic bench and exploiter missing
        d = p.decide(3, self.g1, self.g2, [self.pair], [], holds=[self.hold])
        self.assertFalse(d["all_conditions"])
        self.assertEqual(sorted(d["refusals"]), [
            "missing frozen EXPLOIT pair for %r" % _CHK3,
            "missing frozen R4 pair for ('heuristic', 'heuristic')"])

    def test_b2_train_gen199_g2_is_not_mixed_into_accept(self):
        d = p.decide(3, self.g1, self.other, [self.pair], [])
        self.assertFalse(d["all_conditions"], d)
        d = p.decide(3, self.g1, self.other, [self.pair], [],
                     holds=[self.hold])
        self.assertFalse(d["all_conditions"])
        self.assertTrue(any(r.startswith("G2 summary is from group 'train'")
                            for r in d["refusals"]), d["refusals"])
        self.assertTrue(any("not the frozen G2 matchup" in r
                            for r in d["refusals"]))


class ConditionTests(unittest.TestCase):

    def test_roles_formal_size_and_n2_are_enforced(self):
        with self.assertRaisesRegex(p.ProtocolError, "1000"):
            p.conditions(3, "R4", "heuristic", ["random", "random"], "dev",
                         33200, 800, "x", "x")
        with self.assertRaises(p.ProtocolError):
            p.conditions(2, "R4", "heuristic", ["random"], "train", 0, 1, "x",
                         "x")
        with self.assertRaisesRegex(p.ProtocolError, "G1"):
            p.conditions(3, "G1", "heuristic", ["random", "random"], "train",
                         0, 1, "x", "x")
        with self.assertRaisesRegex(p.ProtocolError, "G2"):
            p.conditions(3, "G2", "heuristic", ["random", "random"], "train",
                         0, 1, "x", "x")
        cond = p.conditions(3, "R4", "heuristic", ["random", "random"],
                            "train", 0, 1, "x", "x")
        self.assertEqual(cond["protocol"], p.PROTOCOL)
        self.assertNotIn("agents", cond)       # canonical_agents + orders only


class LedgerTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = os.path.join(self.dir, "eval-seeds.md")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(_LEDGER)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def _hold(self, rid="h3", lo=_START, hi=_START + 1, n=3):
        return eval_agent.register_p12b(
            eval_agent.build_p12b_hold(rid, n, "train", lo, hi, "SMOKE", "t"),
            ledger_path=self.ledger)

    def test_holds_overlap_and_kind_rules(self):
        self._hold()
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "overlaps"):
            self._hold("h3b", _START + 1, _START + 2)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "exists"):
            self._hold("h3", _START + 10, _START + 11)
        for args in (("x", 3, "dev", 33200, 33999, "G", "t"),     # 800
                     ("x", 3, "dev", 33200, 34199, "ACCEPT", "t"),
                     ("x", 3, "accept", 2400, 3399, "G", "t"),
                     ("x", 3, "train", 0, 9, "G", "t"),
                     ("x", 3, "dev", 20000, 20999, "G", "t"),   # outside region
                     ("x", 2, "train", 0, 9, "SMOKE", "t")):
            with self.assertRaises(eval_agent.SeedLedgerError, msg=args):
                eval_agent.build_p12b_hold(*args)
        # Phase 10 consumed dev bands and the two-seat accept band are refused
        e = eval_agent.build_p12b_hold("y", 3, "dev", 32000, 32999, "G", "t")
        with open(self.ledger, encoding="utf-8") as fh:
            text = fh.read()
        text = text.replace(
            "<!-- MULTI-SEAT-CONSUMED:END -->",
            '{"group":"dev","hi":32399,"id":"p10","lo":31600,"run_id":"r"}\n'
            "<!-- MULTI-SEAT-CONSUMED:END -->")
        with open(self.ledger, "w", encoding="utf-8") as fh:
            fh.write(text)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "overlaps"):
            eval_agent.register_p12b(e, ledger_path=self.ledger)
        a = eval_agent.build_p12b_hold("z", 3, "accept", 0, 999, "ACCEPT", "t")
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "overlaps"):
            eval_agent.register_p12b(a, ledger_path=self.ledger)

    def test_unfrozen_hold_cannot_launch_and_freeze_is_once(self):
        self._hold()
        opp = ["heuristic", "random"]
        cond = eval_agent.p12b_conditions_for("h3", "R4", "random", opp,
                                              ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "unfrozen"):
            eval_agent.claim_p12b(cond, p.run_id_for(cond),
                                  ledger_path=self.ledger)
        m = [{"role": "R4", "challenger": "random", "opponents": opp}]
        eval_agent.freeze_p12b("h3", m, ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "once"):
            eval_agent.freeze_p12b("h3", m, ledger_path=self.ledger)
        other = eval_agent.p12b_conditions_for("h3", "R4", "heuristic", opp,
                                               ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "not frozen"):
            eval_agent.claim_p12b(other, p.run_id_for(other),
                                  ledger_path=self.ledger)
        rid = p.run_id_for(cond)
        eval_agent.claim_p12b(cond, rid, ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "consumed"):
            eval_agent.claim_p12b(cond, rid, ledger_path=self.ledger)
        self.assertEqual(eval_agent.claim_p12b(
            cond, rid, resume=True, ledger_path=self.ledger)["run_id"], rid)
        with self.assertRaises(eval_agent.SeedLedgerError):
            eval_agent.claim_p12b(cond, "other", resume=True,
                                  ledger_path=self.ledger)

    def test_old_and_new_protocol_rows_are_not_mixed(self):
        self._hold()
        cond = harness.multi_conditions(3, ["random", "heuristic", "random"],
                                        "train", _START, 2, "t", None)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "protocol"):
            eval_agent.claim_multi("h3", cond, harness.multi_run_id(cond),
                                   ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "phase12b"):
            eval_agent.register_multi(eval_agent.build_reservation(
                "h3", 3, [["random", "heuristic", "random"]], "train", _START,
                _START + 1, "t", None), ledger_path=self.ledger)
        eval_agent.register_multi(eval_agent.build_reservation(
            "old", 3, [["random", "heuristic", "random"]], "train",
            _START + 10, _START + 11, "t", None), ledger_path=self.ledger)
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "not a"):
            eval_agent.p12b_conditions_for("old", "R4", "random",
                                           ["heuristic", "random"],
                                           ledger_path=self.ledger)

    def test_run_recompute_replay_workers_1_and_4(self):
        """Heuristic/random bench, 2 clusters: end-to-end and byte determinism."""
        self._hold(n=3)
        opp = ["heuristic", "random"]
        m = [{"role": "R4", "challenger": "random", "opponents": opp},
             {"role": "R4", "challenger": "heuristic", "opponents": opp}]
        eval_agent.freeze_p12b("h3", m, ledger_path=self.ledger)
        out = {}
        for ch, w in (("random", 1), ("heuristic", 4)):
            rec = os.path.join(self.dir, ch + ".jsonl")
            summ = os.path.join(self.dir, ch + ".json")
            eval_agent.run_p12b("h3", "R4", ch, opp, rec, summ, workers=w,
                                ledger_path=self.ledger)
            out[ch] = summ
            self.assertEqual(p.recompute(summ)["games"], 6)
        for w in (1, 4):
            self.assertTrue(eval_agent.replay_p12b(out["random"], workers=w)[0])
        with self.assertRaisesRegex(eval_agent.SeedLedgerError, "exists"):
            eval_agent.run_p12b("h3", "R4", "random", opp,
                                os.path.join(self.dir, "x.jsonl"),
                                out["random"], ledger_path=self.ledger)
        paired_path = os.path.join(self.dir, "paired.json")
        eval_agent.paired_p12b("R4", out["heuristic"], out["random"],
                               out_path=paired_path)
        fresh = eval_agent.recompute_paired_p12b(paired_path)
        self.assertEqual(fresh["clusters"], 2)
        # tamper with saved records -> refused
        rec = os.path.join(self.dir, "random.jsonl")
        with open(rec, "a", encoding="utf-8") as fh:
            fh.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            p.recompute(out["random"])


class LegacyRegressionTests(unittest.TestCase):
    """Old protocol / n2 saved results recompute bit-identically."""

    def test_phase10_saved_summaries_recompute(self):
        base = os.path.join(_ROOT, "docs", "eval-runs", "phase10")
        for rel in ("c3/smoke-3p-train-870000-870019.json",
                    "c3/smoke-4p-train-870100-870119.json"):
            path = os.path.join(base, rel)
            fresh = eval_agent.recompute_multi(path)
            saved = json.load(open(path, encoding="utf-8"))
            for key in ("point_estimate", "ci_low", "ci_high", "passed"):
                self.assertEqual(fresh[key], saved[key], (rel, key))
            self.assertEqual(saved["protocol"], harness.MULTI_PROTOCOL)

    def test_two_seat_saved_pairs_recompute(self):
        path = os.path.join(_ROOT, "docs", "eval-runs", "phase8",
                            "accept-0-799")
        records = [json.loads(l) for l in open(path + ".jsonl",
                                               encoding="utf-8")]
        saved = json.load(open(path + ".json", encoding="utf-8"))
        pairing = saved["pairings"][0]
        by = {}
        for r in records:
            if r["left"] == pairing["left"] and r["right"] == pairing["right"]:
                by[(r["index"], r["seat_variant"])] = r["result"]
        games = [(by[(i, 0)], by[(i, 1)]) for i in range(
            saved["start_index"], saved["end_index"] + 1)]
        unused_w, unused_d, unused_l, pr, point = harness.summarize_pairs(games)
        lo, hi = harness.cluster_bootstrap(pr, 10000, 2026072102)
        self.assertEqual((point, lo, hi), (pairing["point_estimate"],
                                           pairing["ci_low"],
                                           pairing["ci_high"]))

    def test_old_d4_constants_unchanged(self):
        self.assertEqual(harness.D4_CLUSTERS, 800)
        self.assertEqual(harness.MULTI_PROTOCOL, "phase10-cyclic/v1")
        self.assertEqual(harness.required_clusters(), 786)


class RealLedgerTests(unittest.TestCase):

    def test_registered_holds_are_unfrozen_and_match_design(self):
        unused, reserved, consumed, unused_two = eval_agent.read_multi_ledger()
        rows = dict((e["id"], e) for e in reserved
                    if e.get("protocol") == p.PROTOCOL)
        for rid, n, group, lo, hi, kind, purpose in eval_agent.P12B_HOLDS:
            row = rows[rid]
            self.assertEqual((row["num_players"], row["group"], row["lo"],
                              row["hi"], row["kind"], row["purpose"]),
                             (n, group, lo, hi, kind, purpose))
            self.assertEqual(row["hi"] - row["lo"] + 1, 1000)
            self.assertFalse(row["frozen"], rid)
            self.assertEqual(row["matchups"], [])
            self.assertAlmostEqual(row["power"]["approximate_power"], 0.8214, 4)
            self.assertAlmostEqual(
                row["r4_sensitivity"]["ci_condition_pass_probability"], 0.3654,
                4)
            self.assertFalse(any(c.get("id") == rid for c in consumed))


@requires_torch
class SavedSmokeTests(unittest.TestCase):
    """docs/eval-runs/phase12b/c1 train smoke recomputes from saved files."""

    def test_saved_smoke_recomputes(self):
        for n in (3, 4):
            d = os.path.join(SMOKE, "n%d" % n)
            if not os.path.exists(os.path.join(d, "decision.json")):
                self.skipTest("smoke artefacts absent")
            for name in ("g1", "g2", "r4-cand", "r4-b", "ex-cand", "ex-b"):
                fresh = p.recompute(os.path.join(d, name + ".json"))
                self.assertEqual(fresh["num_players"], n)
                self.assertEqual(fresh["protocol"], p.PROTOCOL)
            for name in ("r4-paired", "ex-paired"):
                eval_agent.recompute_paired_p12b(os.path.join(d, name + ".json"))


if __name__ == "__main__":
    unittest.main()
