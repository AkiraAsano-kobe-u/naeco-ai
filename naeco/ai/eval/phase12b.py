"""Phase 12b: 3・4 人戦の G 型判定 protocol `phase12b-multi/v1`(opt-in)。

正本は docs/phase12b-design.md D1/D4 と docs/phase12b-protocol.md。
Phase 10 D4 の `phase10-cyclic/v1`(800 クラスタ・95% CI)と 2 人戦の旧経路は
そのまま残し、ここは別名の新 protocol として実装する。既存 record schema
(`multi-eval-record/v2`)を使い、record の `protocol` 欄で新旧を区別する。

- クラスタ: 1 配札 = n 局巡回(D4 継承)。異種の他家は「他家の並べ方」を
  クラスタ index 由来で巡らせて均等化してから巡回する(差は高々 1 クラスタ)。
- 正式(dev/accept)は 1000 クラスタ固定。95% 両側 CI を必ず報告し、合否は
  97.5% 両側 percentile クラスタ bootstrap(B=10,000、seed=2026072102)。
- G1: CI 下限 > 1/n。G2: 生勝率 CI 下限 > 1/n + 0.30(1 − 1/n)(n3 8/15、n4 0.475)。
- R4: 同配札・同他家・同席写像の paired 勝率差 d=p_cand−p_B、点推定 >= 0 かつ
  97.5% CI 下限 > −0.03(−0.05 未満なら dev で棄却)。搾取者 d_E は点推定 <= 0 かつ
  97.5% CI 上限 < +0.03。穴の線(診断)は搾取者勝率の 95% CI 下限 > 1/n + 0.05。

stdlib のみ。torch は import しない(モデル席は harness が net.py 経由で遅延読込)。
"""
import hashlib
import itertools
import json
import math
import os
import random
from fractions import Fraction
from statistics import NormalDist

from naeco.ai.eval import harness
from naeco.engine.cards import load_cards
from naeco.engine.state import GROUPS

PROTOCOL = "phase12b-multi/v1"
SUMMARY_SCHEMA = "phase12b-summary/v1"
PAIRED_SCHEMA = "phase12b-paired/v1"
DECISION_SCHEMA = "phase12b-decision/v1"
PLAYER_COUNTS = (3, 4)
FORMAL_GROUPS = ("dev", "accept")
FORMAL_CLUSTERS = 1000
BOOTSTRAP_SAMPLES = harness.BOOTSTRAP_SAMPLES     # 10,000
BOOTSTRAP_SEED = harness.BOOTSTRAP_SEED           # 2026072102

# 4 主判定(G1/G2 × n3/n4)の family α<=0.05 → 片側 0.0125 → 97.5% 両側 CI。
FAMILY_ALPHA = Fraction(5, 100)
FAMILY_JUDGEMENTS = 4
ALPHA_ONE_SIDED = FAMILY_ALPHA / FAMILY_JUDGEMENTS            # 1/80
# percentile は浮動小数の (1-level)/2 を使わず定数で固定する(0.025 の誤差を避ける)。
CI_PERCENTILES = {"95": (0.025, 0.975), "97.5": (0.0125, 0.9875)}
REPORT_CI = "95"
JUDGE_CI = "97.5"

# 検出力(正規近似。bootstrap の有限標本保証ではない)。
POWER_DELTA = 0.05
POWER_SIGMA = 0.5
TARGET_POWER = 0.80
Z_ALPHA = NormalDist().inv_cdf(1.0 - float(ALPHA_ONE_SIDED))   # 2.2414...
Z_BETA = NormalDist().inv_cdf(TARGET_POWER)                    # 0.8416...

G2_EXCESS = Fraction(3, 10)
R4_MARGIN = Fraction(3, 100)
R4_DEV_REJECT = Fraction(5, 100)
EXPLOIT_MARGIN = Fraction(3, 100)
HOLE_MARGIN = Fraction(5, 100)

ROLES = ("G1", "G2", "R4", "EXPLOIT")
HOLD_KINDS = {
    # 帯の種別 → その帯で起動できる role(同帯共有は登録した対戦列のみ)。
    "G": ("G1", "G2"),
    "R4": ("R4",),
    "EXPLOIT": ("EXPLOIT",),
    "ACCEPT": ("G1", "G2", "R4", "EXPLOIT"),
    "SMOKE": ("G1", "G2", "R4", "EXPLOIT"),
}
OPPONENT_ORDER_RULE = ("opponent order = distinct permutations of the canonical "
                       "opponent list in lexicographic index order, chosen by "
                       "cluster index modulo their count; then the n-game cyclic "
                       "rotation")


class ProtocolError(ValueError):
    """phase12b-multi/v1 の条件・保存物が規則に合わない(fail-closed)。"""


# ------------------------------------------------------------------ 閾値と検出力

def _check_n(num_players):
    if type(num_players) is not int or num_players not in PLAYER_COUNTS:
        raise ProtocolError("phase12b-multi/v1 is for n in %s (n=2 keeps the "
                            "legacy pair runner), got %r"
                            % (PLAYER_COUNTS, num_players))


def baseline(num_players):
    _check_n(num_players)
    return Fraction(1, num_players)


def g1_threshold(num_players):
    return baseline(num_players)


def g2_threshold(num_players):
    """偶然基準からの正規化超過率 q=(p−1/n)/(1−1/n) > 0.30 の生勝率表現。"""
    b = baseline(num_players)
    return b + G2_EXCESS * (1 - b)


def excess_rate(p, num_players):
    b = float(baseline(num_players))
    return (p - b) / (1.0 - b)


def _fraction_text(value):
    return "%d/%d" % (value.numerator, value.denominator)


def required_clusters(delta=POWER_DELTA, sigma=POWER_SIGMA, z_alpha=Z_ALPHA,
                      z_beta=Z_BETA):
    return int(math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2))


def approximate_power(clusters, delta=POWER_DELTA, sigma=POWER_SIGMA,
                      z_alpha=Z_ALPHA):
    if clusters < 1:
        raise ProtocolError("clusters must be positive")
    return NormalDist().cdf(delta * math.sqrt(clusters) / sigma - z_alpha)


def r4_ci_sensitivity(clusters=FORMAL_CLUSTERS, sigma=POWER_SIGMA,
                      true_diff=0.0, margin=float(R4_MARGIN), z_alpha=Z_ALPHA):
    """R4 の CI 条件だけ(97.5% 下限 > −margin)を満たす近似確率。

    G1/G2 の検出力とは別物。点推定 >= 0 の条件や全ベンチ AND を含む
    全体の通過率ではない。σ はクラスタ単位の paired 差の標準偏差の仮定値。
    """
    se = sigma / math.sqrt(clusters)
    return NormalDist().cdf((true_diff + margin) / se - z_alpha)


def g_power_record(num_players, clusters=FORMAL_CLUSTERS):
    return {
        "G1_threshold": _fraction_text(g1_threshold(num_players)),
        "G2_threshold": _fraction_text(g2_threshold(num_players)),
        "G2_excess_rate": _fraction_text(G2_EXCESS),
        "delta": POWER_DELTA,
        "sigma": POWER_SIGMA,
        "alpha_one_sided": float(ALPHA_ONE_SIDED),
        "family_alpha": float(FAMILY_ALPHA),
        "family_judgements": "G1/G2 x n3/n4 (Bonferroni)",
        "z_alpha": Z_ALPHA,
        "z_beta": Z_BETA,
        "target_power": TARGET_POWER,
        "required_clusters": required_clusters(),
        "clusters": clusters,
        "approximate_power": approximate_power(clusters),
        "power_at_800_not_used": approximate_power(800),
        "note": "normal approximation for G1/G2 only; not a finite-sample "
                "bootstrap guarantee and not the R4 power",
    }


def r4_sensitivity_record(clusters=FORMAL_CLUSTERS):
    return {
        "sigma_paired_cluster_diff": POWER_SIGMA,
        "true_diff": 0.0,
        "clusters": clusters,
        "alpha_one_sided": float(ALPHA_ONE_SIDED),
        "margin": float(R4_MARGIN),
        "ci_condition_pass_probability": r4_ci_sensitivity(clusters),
        "note": "probability that only the CI condition (97.5% low > -0.03) "
                "holds at true difference 0; excludes the point-estimate "
                "condition and the AND over all benches; separate from the "
                "G1/G2 power",
    }


def gates_for(kind, num_players):
    """帯の種別ごとの事前固定判定(台帳の hold 行に保存する)。"""
    _check_n(num_players)
    if kind not in HOLD_KINDS:
        raise ProtocolError("unknown hold kind %r" % (kind,))
    g1 = {"statistic": "target win rate (ranking[target_seat]==0)",
          "pass": "ci97.5_low > %s" % _fraction_text(g1_threshold(num_players)),
          "threshold": _fraction_text(g1_threshold(num_players)),
          "lineup": "checkpoint:candidate (greedy) + strong:B x (n-1)"}
    g2 = {"statistic": "target win rate (ranking[target_seat]==0)",
          "pass": "ci97.5_low > %s (excess q > 3/10)"
                  % _fraction_text(g2_threshold(num_players)),
          "threshold": _fraction_text(g2_threshold(num_players)),
          "lineup": "strong:candidate (new net + frozen search) + strong:B x (n-1)",
          "requires": "nothing; R4 is evaluated only after G2 passes"}
    r4 = {"statistic": "paired d = p_candidate - p_B per bench (same deal, "
                       "opponents, seat map)",
          "pass": "point >= 0 and ci97.5_low > -3/100 for every bench (AND)",
          "dev_reject": "point < -5/100",
          "order": "after G2 passed, same alpha 0.0125, intersection-union"}
    exploit = {"statistic": "paired d_E = p_E(vs candidate) - p_E(vs B) per "
                            "frozen exploiter",
               "pass": "point <= 0 and ci97.5_high < 3/100 (added to R4)",
               "hole_line_diagnostic": "ci95_low of exploiter win rate > 1/n + "
                                       "5/100",
               "note": "undetected means undetected within the search budget"}
    table = {"G1": g1, "G2": g2, "R4": r4, "EXPLOIT": exploit}
    return dict((role, table[role]) for role in HOLD_KINDS[kind])


# ------------------------------------------------------------- 卓・条件・タスク

def opponent_orders(opponents):
    """他家の並べ方(重複を除いた順列、index タプルの辞書順)。"""
    opponents = list(opponents)
    out = []
    for perm in itertools.permutations(range(len(opponents))):
        order = [opponents[i] for i in perm]
        if order not in out:
            out.append(order)
    return out


def lineup_for(conditions, index):
    """クラスタ index の正準 agent 列(先頭が評価対象)。"""
    orders = conditions["opponent_orders"]
    return [conditions["challenger"]] + list(orders[index % len(orders)])


def _kind(spec):
    return harness.parse_multi_agent_spec(spec)[0]


def _validate_role(role, challenger, opponents):
    if role not in ROLES:
        raise ProtocolError("unknown role %r (expected %s)" % (role, ROLES))
    same_strong = len(set(opponents)) == 1 and _kind(opponents[0]) == "strong"
    if role == "G1" and (_kind(challenger) != "checkpoint" or not same_strong):
        raise ProtocolError("G1 is checkpoint:candidate (greedy) vs the same "
                            "strong:B in every other seat")
    if role == "G2" and (_kind(challenger) != "strong" or not same_strong):
        raise ProtocolError("G2 is strong:candidate vs the same strong:B in "
                            "every other seat")
    if role == "EXPLOIT" and (_kind(challenger) != "checkpoint" or
                              not same_strong):
        raise ProtocolError("EXPLOIT is a frozen checkpoint exploiter vs one "
                            "strong target in every other seat")


def conditions(num_players, role, challenger, opponents, group, start_index,
               clusters, reservation_id, purpose, cards=None):
    """実行前に固定する条件(run_id の材料)。"""
    _check_n(num_players)
    opponents = list(opponents)
    if len(opponents) != num_players - 1:
        raise ProtocolError("need exactly %d opponents" % (num_players - 1))
    _validate_role(role, challenger, opponents)
    if group not in GROUPS:
        raise ProtocolError("unknown seed group %r" % (group,))
    if group in FORMAL_GROUPS and clusters != FORMAL_CLUSTERS:
        raise ProtocolError("formal (%s) phase12b-multi/v1 judgements use "
                            "exactly %d clusters" % (group, FORMAL_CLUSTERS))
    if not reservation_id or not isinstance(reservation_id, str):
        raise ProtocolError("reservation_id must be a non-empty string")
    base = harness.multi_conditions(num_players, [challenger] + opponents,
                                    group, start_index, clusters, purpose,
                                    None, cards=cards)
    canonical = base.pop("agents")
    base.pop("threshold")
    base.update({
        "protocol": PROTOCOL,
        "role": role,
        "reservation_id": reservation_id,
        "challenger": challenger,
        "opponents": opponents,
        "canonical_agents": canonical,
        "opponent_orders": opponent_orders(opponents),
        "opponent_order_rule": OPPONENT_ORDER_RULE,
        "ci_report": REPORT_CI,
        "ci_judge": JUDGE_CI,
        "ci_percentiles": dict((k, list(v)) for k, v in CI_PERCENTILES.items()),
        "gates": gates_for("SMOKE", num_players)[role],
    })
    return base


def hold_view(row):
    """台帳 hold 行のうち条件と照合する欄(人数は登録行が正本)。"""
    if row.get("protocol") != PROTOCOL:
        raise ProtocolError("ledger row %r is not %s" % (row.get("id"),
                                                         PROTOCOL))
    return {"num_players": row["num_players"], "group": row["group"],
            "start_index": row["lo"], "end_index": row["hi"],
            "purpose": row["purpose"]}


def conditions_from_hold(row, role, challenger, opponents):
    """台帳 hold 行から条件を作る(帯・人数・用途は行が決め、引数で変えられない)。"""
    v = hold_view(row)
    return conditions(v["num_players"], role, challenger, list(opponents),
                      v["group"], v["start_index"],
                      v["end_index"] - v["start_index"] + 1, row["id"],
                      v["purpose"])


def players_of(obj):
    """summary / conditions の人数(表示用)。"""
    return obj["num_players"]


def run_id_for(cond):
    return harness.multi_run_id(cond)


def tasks_for(cond, run_id, skip=()):
    skip = set(skip)
    out = []
    for c in range(cond["clusters"]):
        index = cond["start_index"] + c
        lineup = tuple(lineup_for(cond, index))
        for k in range(cond["num_players"]):
            if (index, k) not in skip:
                out.append((run_id, cond["group"], index, k, lineup,
                            cond["num_players"], cond.get("models"), PROTOCOL))
    return out


def run_task(task):
    run_id, group, index, k, agents, n, models, protocol = task
    return harness.play_cyclic_game(load_cards(), run_id, group, index, k,
                                    agents, n, models=models,
                                    protocol=protocol)


def run_games(tasks, workers=1):
    """(index, k) 順の record。worker 数・完了順は結果に影響しない。"""
    if workers < 1:
        raise ProtocolError("workers must be >= 1")
    if workers == 1 or len(tasks) <= 1:
        records = [run_task(t) for t in tasks]
    else:
        import multiprocessing
        pool = multiprocessing.get_context("spawn").Pool(processes=workers)
        try:
            records = list(pool.map(run_task, tasks, chunksize=1))
        finally:
            pool.close()
            pool.join()
    return sorted(records, key=lambda r: (r["index"], r["seat_variant"]))


# ------------------------------------------------------------------- 統計

def bootstrap_means(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """harness.cluster_bootstrap と同じ乱数消費のクラスタ復元抽出平均列。"""
    rng = random.Random(seed)
    count = len(values)
    if count == 0:
        raise ProtocolError("no clusters")
    means = []
    for unused in range(samples):
        total = 0.0
        for unused_inner in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    return means


def intervals(values, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    means = bootstrap_means(values, samples, seed)
    out = {}
    for level, (lo, hi) in sorted(CI_PERCENTILES.items()):
        out[level] = [harness._percentile(means, lo),
                      harness._percentile(means, hi)]
    return out


def _cluster_values(records, cond, run_id):
    if cond.get("protocol") != PROTOCOL:
        raise ProtocolError("conditions are not %s (got %r)"
                            % (PROTOCOL, cond.get("protocol")))
    values, rows = harness.multi_cluster_results(
        records, run_id, cond["start_index"], cond["clusters"],
        cond["num_players"], cond["canonical_agents"], protocol=PROTOCOL,
        agents_for=lambda index: lineup_for(cond, index))
    harness._check_seat_provenance(records, cond)
    return values, rows


def summarize(records, cond, run_id, samples=BOOTSTRAP_SAMPLES,
              seed=BOOTSTRAP_SEED):
    """1 対戦(1 run)の集計。保存物(records + 条件)だけから再計算できる。"""
    if run_id_for(cond) != run_id:
        raise ProtocolError("run_id does not match the conditions")
    n = cond["num_players"]
    values, rows = _cluster_values(records, cond, run_id)
    cis = intervals(values, samples, seed)
    point = sum(values) / len(values)
    games = cond["clusters"] * n
    ranks = [rank for r in rows for rank in r["target_ranks"]]
    seat_wins = [0] * n
    for rec in records:
        seat_wins[rec["target_seat"]] += rec["target_win"]
    role = cond["role"]
    gate = {"role": role}
    if role in ("G1", "G2"):
        threshold = g1_threshold(n) if role == "G1" else g2_threshold(n)
        gate.update({"threshold": _fraction_text(threshold),
                     "passed": cis[JUDGE_CI][0] > float(threshold),
                     "excess_rate_point": excess_rate(point, n),
                     "excess_rate_ci97.5_low": excess_rate(cis[JUDGE_CI][0], n)})
    elif role == "EXPLOIT":
        line = baseline(n) + HOLE_MARGIN
        gate.update({"hole_line": _fraction_text(line),
                     "hole_detected": cis[REPORT_CI][0] > float(line),
                     "E_proxy": max(0.0, point - float(baseline(n))),
                     "diagnostic_only": True})
    else:
        gate.update({"passed": None,
                     "note": "R4 is judged on the paired difference"})
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "run_id": run_id,
        "role": role,
        "reservation_id": cond["reservation_id"],
        "num_players": n,
        "challenger": cond["challenger"],
        "opponents": cond["opponents"],
        "group": cond["group"],
        "start_index": cond["start_index"],
        "end_index": cond["end_index"],
        "clusters": cond["clusters"],
        "games": games,
        "wins": sum(r["target_wins"] for r in rows),
        "point_estimate": point,
        "ci": cis,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "baseline_1_over_n": _fraction_text(baseline(n)),
        "gate": gate,
        "target_rank_histogram": [ranks.count(p) for p in range(n)],
        "target_mean_rank": sum(ranks) / float(len(ranks)),
        "target_wins_by_seat": seat_wins,
        "cluster_results": [
            {"index": r["index"], "value": r["cluster_value"],
             "target_ranks": r["target_ranks"],
             "initial_state_hash": r["initial_state_hash"]} for r in rows],
    }


def _pairing_checks(kind, cond_a, cond_b):
    for key in ("num_players", "group", "start_index", "end_index", "clusters",
                "cards_sha256", "api_version", "rules_version", "obs_version",
                "reservation_id"):
        if cond_a.get(key) != cond_b.get(key):
            raise ProtocolError("paired runs differ at %s (%r vs %r)"
                                % (key, cond_a.get(key), cond_b.get(key)))
    if kind == "R4":
        if cond_a["role"] != "R4" or cond_b["role"] != "R4":
            raise ProtocolError("R4 pairs two R4 runs")
        if cond_a["opponents"] != cond_b["opponents"] or \
                cond_a["opponent_orders"] != cond_b["opponent_orders"]:
            raise ProtocolError("R4 pairs need the same bench opponents/order")
        if cond_a["challenger"] == cond_b["challenger"]:
            raise ProtocolError("R4 pairs a candidate with a different B")
    elif kind == "EXPLOIT":
        if cond_a["role"] != "EXPLOIT" or cond_b["role"] != "EXPLOIT":
            raise ProtocolError("d_E pairs two EXPLOIT runs")
        if cond_a["challenger"] != cond_b["challenger"]:
            raise ProtocolError("d_E pairs the same frozen exploiter")
        if cond_a["opponents"] == cond_b["opponents"]:
            raise ProtocolError("d_E pairs candidate-table vs B-table")
    else:
        raise ProtocolError("unknown paired kind %r" % (kind,))


def paired(kind, records_a, cond_a, run_a, records_b, cond_b, run_b,
           samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """paired 差 d = 値(a) − 値(b) をクラスタ単位で作り bootstrap する。

    R4: a=候補 strong、b=B strong(同ベンチ卓)。EXPLOIT(d_E): a=搾取者 vs 候補卓、
    b=同搾取者 vs B 卓。同 index の配札 hash・席写像・他家配置を照合し、
    欠損・重複・SHA 不一致・protocol 違いは拒否する。
    """
    _pairing_checks(kind, cond_a, cond_b)
    if run_id_for(cond_a) != run_a or run_id_for(cond_b) != run_b:
        raise ProtocolError("run_id does not match the conditions")
    va, rows_a = _cluster_values(records_a, cond_a, run_a)
    vb, rows_b = _cluster_values(records_b, cond_b, run_b)
    by_a = dict(((r["index"], r["seat_variant"]), r) for r in records_a)
    by_b = dict(((r["index"], r["seat_variant"]), r) for r in records_b)
    if set(by_a) != set(by_b):
        raise ProtocolError("paired runs cover different (index, k)")
    for key in sorted(by_a):
        ra, rb = by_a[key], by_b[key]
        for field in ("initial_state_hash", "env_seed", "setup_seed",
                      "seat_agents", "order", "target_seat", "start_player"):
            if ra[field] != rb[field]:
                raise ProtocolError("paired games differ at %s: index %d k=%d"
                                    % (field, key[0], key[1]))
        if kind == "R4":
            for sa, sb in zip(ra["seats"], rb["seats"]):
                if sa["agent_index"] != 0 and sa["agent"] != sb["agent"]:
                    raise ProtocolError("paired bench opponent differs at "
                                        "index %d k=%d seat %d"
                                        % (key[0], key[1], sa["seat"]))
    diffs = [a - b for a, b in zip(va, vb)]
    cis = intervals(diffs, samples, seed)
    point = sum(diffs) / len(diffs)
    if kind == "R4":
        passed = point >= 0 and cis[JUDGE_CI][0] > -float(R4_MARGIN)
        gate = {"pass_rule": "point >= 0 and ci97.5_low > -3/100",
                "passed": passed,
                "dev_reject": point < -float(R4_DEV_REJECT)}
    else:
        passed = point <= 0 and cis[JUDGE_CI][1] < float(EXPLOIT_MARGIN)
        gate = {"pass_rule": "point <= 0 and ci97.5_high < 3/100",
                "passed": passed}
    return {
        "schema": PAIRED_SCHEMA,
        "protocol": PROTOCOL,
        "kind": kind,
        "num_players": cond_a["num_players"],
        "group": cond_a["group"],
        "start_index": cond_a["start_index"],
        "end_index": cond_a["end_index"],
        "clusters": cond_a["clusters"],
        "run_a": run_a,
        "run_b": run_b,
        "challenger_a": cond_a["challenger"],
        "challenger_b": cond_b["challenger"],
        "opponents_a": cond_a["opponents"],
        "opponents_b": cond_b["opponents"],
        "p_a": sum(va) / len(va),
        "p_b": sum(vb) / len(vb),
        "point_estimate": point,
        "ci": cis,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "gate": gate,
        "cluster_diffs": [{"index": ra["index"], "diff": d}
                          for ra, d in zip(rows_a, diffs)],
    }


def matchup_key(m):
    """台帳 hold の model_sha256 の鍵(eval_agent._p12b_matchup_key と同形)。"""
    return "%s|%s|%s" % (m["role"], m["challenger"], ",".join(m["opponents"]))


# hold の組み合わせ: 評価段(group)ごとに、判定に要る hold の種別の集合。
# dev は G/R4/EXPLOIT の 3 帯(正当に別の reservation_id)、accept は 1 帯。
DECISION_HOLD_KINDS = {
    "dev": ("EXPLOIT", "G", "R4"),
    "accept": ("ACCEPT",),
    "train": ("SMOKE",),
}


def expected_matchups(num_players, holds):
    """凍結 hold 行から、判定に必要な対戦列とその run_id を作る(fail-closed)。

    返り値: {"group", "formal", "reservation_ids", "candidate", "base",
    "G1": run_id, "G2": run_id, "R4": {bench: (run_cand, run_B)},
    "EXPLOIT": {exploiter: (run_vs_cand, run_vs_B)}}。
    候補 P は G1 の checkpoint:P と G2 の strong:P で同一、B は G1/G2 の他家
    strong:B。R4 の各ベンチは strong:P と strong:B の 2 対戦、搾取者は
    同じ checkpoint が [strong:P]×(n−1) と [strong:B]×(n−1) の 2 対戦を凍結して
    いなければならない。run_id は保存条件(帯・人数・用途・モデル SHA・
    正準卓・予約 ID)の SHA なので、候補・段・規模の混入は run_id 不一致になる。
    """
    _check_n(num_players)
    holds = list(holds or [])
    if not holds:
        raise ProtocolError("decide needs the frozen ledger hold rows of this "
                            "judgement (the expected benches/exploiters)")
    for row in holds:
        hold_view(row)
        if row.get("num_players") != num_players:
            raise ProtocolError("hold %s is for n=%r, not %d"
                                % (row.get("id"), row.get("num_players"),
                                   num_players))
        if not row.get("frozen") or not row.get("matchups"):
            raise ProtocolError("hold %s is not frozen (no expected matchups)"
                                % row.get("id"))
        if row.get("search") != harness.STRONG_SEARCH:
            raise ProtocolError("hold %s froze another search configuration"
                                % row.get("id"))
    ids = [row["id"] for row in holds]
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate hold rows %s" % ids)
    groups = set(row["group"] for row in holds)
    if len(groups) != 1:
        raise ProtocolError("holds of different evaluation stages are mixed: "
                            "%s" % sorted(groups))
    group = groups.pop()
    kinds = tuple(sorted(row["kind"] for row in holds))
    if DECISION_HOLD_KINDS.get(group) != kinds:
        raise ProtocolError("a %s judgement needs exactly the holds of kinds "
                            "%s (got %s)" % (group, DECISION_HOLD_KINDS.get(group),
                                             kinds))
    by_role = {}
    for row in holds:
        for m in row["matchups"]:
            if m["role"] not in HOLD_KINDS[row["kind"]]:
                raise ProtocolError("hold %s froze role %s outside its kind"
                                    % (row["id"], m["role"]))
            cond = conditions_from_hold(row, m["role"], m["challenger"],
                                        m["opponents"])
            shas = dict((s, v["sha256"])
                        for s, v in (cond.get("models") or {}).items())
            if row.get("model_sha256", {}).get(matchup_key(m)) != shas:
                raise ProtocolError("model SHA-256 of %s differs from the "
                                    "frozen hold %s" % (matchup_key(m),
                                                        row["id"]))
            by_role.setdefault(m["role"], []).append(
                (m["challenger"], list(m["opponents"]), run_id_for(cond)))
    for role in ("G1", "G2"):
        if len(by_role.get(role, [])) != 1:
            raise ProtocolError("the holds must freeze exactly one %s matchup "
                                "(got %d)" % (role, len(by_role.get(role, []))))
    g1_ch, g1_opp, g1_run = by_role["G1"][0]
    g2_ch, g2_opp, g2_run = by_role["G2"][0]
    cand = harness.parse_multi_agent_spec(g1_ch)[1]
    if g2_ch != "strong:" + cand:
        raise ProtocolError("G1 and G2 freeze different candidates (%s vs %s)"
                            % (g1_ch, g2_ch))
    if g1_opp != g2_opp:
        raise ProtocolError("G1 and G2 freeze different B tables")
    base = g1_opp[0]
    if base == g2_ch:
        raise ProtocolError("candidate and B are the same model")
    cand_table = [g2_ch] * (num_players - 1)
    base_table = [base] * (num_players - 1)
    benches = {}
    for ch, opp, run in by_role.get("R4", []):
        if ch not in (g2_ch, base):
            raise ProtocolError("R4 matchup %s is neither the candidate nor B"
                                % ch)
        slot = benches.setdefault(tuple(opp), [None, None])
        slot[0 if ch == g2_ch else 1] = run
    exploiters = {}
    for ch, opp, run in by_role.get("EXPLOIT", []):
        if opp not in (cand_table, base_table):
            raise ProtocolError("EXPLOIT matchup %s does not face the "
                                "candidate or the B table" % ch)
        slot = exploiters.setdefault(ch, [None, None])
        slot[0 if opp == cand_table else 1] = run
    for what, table in (("R4 bench", benches), ("exploiter", exploiters)):
        for key, slot in sorted(table.items()):
            if None in slot:
                raise ProtocolError("frozen %s %s lacks its candidate/B "
                                    "counterpart" % (what, key))
    formal = group in FORMAL_GROUPS
    if not benches:
        raise ProtocolError("the holds freeze no R4 bench")
    if formal and not exploiters:
        raise ProtocolError("a formal judgement needs the frozen exploiter "
                            "union (none frozen)")
    return {"group": group, "formal": formal, "reservation_ids": sorted(ids),
            "candidate": g2_ch, "base": base, "G1": g1_run, "G2": g2_run,
            "R4": dict((k, tuple(v)) for k, v in benches.items()),
            "EXPLOIT": dict((k, tuple(v)) for k, v in exploiters.items())}


def _match_pairs(kind, pairs, expected, refusals):
    """paired 結果を凍結対戦列へ 1 対 1 で対応させる。

    欠損・余剰(凍結外)・重複は refusals に理由を足す。対応できた結果を
    凍結列の順で返す。
    """
    by_runs = dict((v, k) for k, v in expected.items())
    seen = {}
    for p in pairs:
        key = by_runs.get((p.get("run_a"), p.get("run_b")))
        if key is None:
            refusals.append("extra %s pair %s vs %s (run %s/%s) is not a frozen "
                            "matchup of these holds"
                            % (kind, p.get("challenger_a"), p.get("challenger_b"),
                               p.get("run_a"), p.get("run_b")))
        elif key in seen:
            refusals.append("duplicate %s pair for %r" % (kind, key))
        else:
            seen[key] = p
    for key in sorted(set(expected) - set(seen)):
        refusals.append("missing frozen %s pair for %r" % (kind, key))
    return [seen[k] for k in sorted(seen)]


def _input_refusals(num_players, g1, g2, r4_pairs, exploit_pairs, holds):
    """判定入力と凍結 hold の対応検査。理由の列と対応済み paired を返す。"""
    refusals = []
    try:
        expected = expected_matchups(num_players, holds)
    except (ProtocolError, ValueError, KeyError, TypeError) as exc:
        return ["frozen expectation unavailable: %s" % exc], None, [], []
    for name, s in (("G1", g1), ("G2", g2)):
        if not isinstance(s, dict) or s.get("schema") != SUMMARY_SCHEMA or \
                s.get("role") != name or s.get("num_players") != num_players:
            refusals.append("%s summary is missing or of another role/n" % name)
            continue
        if s.get("group") != expected["group"]:
            refusals.append("%s summary is from group %r, the holds are %r"
                            % (name, s.get("group"), expected["group"]))
        if s.get("run_id") != expected[name]:
            refusals.append("%s summary (%s vs %s, %s %r-%r, reservation %r) is "
                            "not the frozen %s matchup of %s"
                            % (name, s.get("challenger"), s.get("opponents"),
                               s.get("group"), s.get("start_index"),
                               s.get("end_index"), s.get("reservation_id"),
                               name, expected["reservation_ids"]))
    lists = {"R4": [], "EXPLOIT": []}
    for kind, given in (("R4", r4_pairs), ("EXPLOIT", exploit_pairs)):
        for p in given:
            if not isinstance(p, dict) or p.get("schema") != PAIRED_SCHEMA or \
                    p.get("num_players") != num_players or p.get("kind") != kind:
                refusals.append("a %s entry is not a %s paired result for n=%d"
                                % (kind, kind, num_players))
            else:
                lists[kind].append(p)
    for s in [x for x in (g1, g2) if isinstance(x, dict)] + \
            lists["R4"] + lists["EXPLOIT"]:
        if s.get("bootstrap_samples") != BOOTSTRAP_SAMPLES or \
                s.get("bootstrap_seed") != BOOTSTRAP_SEED:
            refusals.append("bootstrap B/seed of %s differ from the protocol"
                            % (s.get("run_id") or s.get("run_a")))
    r4 = _match_pairs("R4", lists["R4"], expected["R4"], refusals)
    ex = _match_pairs("EXPLOIT", lists["EXPLOIT"], expected["EXPLOIT"],
                      refusals)
    return refusals, expected, r4, ex


def decide(num_players, g1, g2, r4_pairs, exploit_pairs, holds=None):
    """人数ごとの判定。G1 と G2→R4(固定順序・全ベンチ AND)の枝を保つ。

    holds は判定対象の凍結台帳 hold 行(dev は G/R4/EXPLOIT の 3 行、accept は
    ACCEPT 1 行、train smoke は SMOKE 1 行)。G1/G2 summary と r4_pairs /
    exploit_pairs は、その hold が凍結した対戦列と run_id で完全一致しなければ
    ならない。凍結 hold が無い・欠損・余剰・重複・別候補/B・別段・別規模なら
    fail-closed: G1/G2/R4 は None、all_conditions=False、refusals に理由。
    """
    _check_n(num_players)
    refusals, expected, r4_pairs, exploit_pairs = _input_refusals(
        num_players, g1, g2, list(r4_pairs or []), list(exploit_pairs or []),
        holds)
    if refusals:
        return {
            "schema": DECISION_SCHEMA,
            "protocol": PROTOCOL,
            "num_players": num_players,
            "group": expected["group"] if expected else None,
            "formal": expected["formal"] if expected else None,
            "reservation_ids": expected["reservation_ids"] if expected else [],
            "complete": False,
            "refusals": refusals,
            "G1": None,
            "G2": None,
            "R4": None,
            "all_conditions": False,
            "dev_reject": None,
            "r4_benches": [],
            "exploiters": [],
            "note": "inputs do not match the frozen matchups; no judgement "
                    "(fail-closed)",
        }
    g1_pass = bool(g1["gate"]["passed"])
    g2_pass = bool(g2["gate"]["passed"])
    if g2_pass:
        r4_all = all(p["gate"]["passed"] for p in r4_pairs) and \
            all(p["gate"]["passed"] for p in exploit_pairs)
    else:
        r4_all = None          # 固定順序: G2 不合格なら R4 は判定しない
    return {
        "schema": DECISION_SCHEMA,
        "protocol": PROTOCOL,
        "num_players": num_players,
        "group": expected["group"],
        "formal": expected["formal"],
        "reservation_ids": expected["reservation_ids"],
        "complete": True,
        "refusals": [],
        "G1": g1_pass,
        "G2": g2_pass,
        "R4": r4_all,
        "all_conditions": bool(g1_pass and g2_pass and r4_all),
        "dev_reject": any(p["gate"].get("dev_reject") for p in r4_pairs),
        "r4_benches": [{"challenger_a": p["challenger_a"],
                        "opponents": p["opponents_a"],
                        "point": p["point_estimate"],
                        "ci97.5": p["ci"][JUDGE_CI],
                        "passed": p["gate"]["passed"]} for p in r4_pairs],
        "exploiters": [{"exploiter": p["challenger_a"],
                        "point": p["point_estimate"],
                        "ci97.5": p["ci"][JUDGE_CI],
                        "passed": p["gate"]["passed"]} for p in exploit_pairs],
        "note": "G3/G4 and exploiter hole lines are diagnostics and are not "
                "part of this decision",
    }


# ----------------------------------------------------------- 保存物の I/O と再計算

def sha256_path(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, sort_keys=True, ensure_ascii=True, indent=1)
        fh.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_run(summary_path):
    """summary JSON から (records, conditions, run_id, saved) を読む(SHA 照合)。"""
    saved = read_json(summary_path)
    if saved.get("schema") != SUMMARY_SCHEMA or \
            saved.get("protocol") != PROTOCOL:
        raise ProtocolError("%s is not a %s summary" % (summary_path, PROTOCOL))
    records_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                                saved["records_path"])
    if sha256_path(records_path) != saved["records_sha256"]:
        raise ProtocolError("records SHA-256 differs from the summary")
    cond = saved["conditions"]
    if run_id_for(cond) != saved["run_id"]:
        raise ProtocolError("run_id does not match the saved conditions")
    return harness.read_multi_records(records_path), cond, saved["run_id"], saved


RECOMPUTED_KEYS = (
    "schema", "protocol", "run_id", "role", "reservation_id", "num_players",
    "challenger", "opponents", "group", "start_index", "end_index", "clusters",
    "games", "wins", "point_estimate", "ci", "bootstrap_samples",
    "bootstrap_seed", "baseline_1_over_n", "gate", "target_rank_histogram",
    "target_mean_rank", "target_wins_by_seat", "cluster_results",
)


def recompute(summary_path):
    records, cond, run_id, saved = load_run(summary_path)
    fresh = summarize(records, cond, run_id)
    diffs = [k for k in RECOMPUTED_KEYS if fresh[k] != saved.get(k)]
    if diffs:
        raise ProtocolError("recomputed summary differs at %s" % diffs)
    return fresh


def paired_from_summaries(kind, summary_a, summary_b):
    ra, ca, ia, unused = load_run(summary_a)
    rb, cb, ib, unused = load_run(summary_b)
    return paired(kind, ra, ca, ia, rb, cb, ib)
