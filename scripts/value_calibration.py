"""Phase 7 段1: value ヘッドの較正診断。

チェックポイントの greedy 自己対戦を `train` 群の診断帯で回し、**各決定点で
両席の value(効用空間 u)**を記録して、終局の勝敗と突き合わせる。

出すもの:
- ゼロ和からのずれ: mean(u0 + u1)、|u0 + u1| の平均
- 較正: p=(u+1)/2(D-2 の写像)と反対称化 p=0.5+(u_a-u_b)/4 の
  ECE(10 bins)/ Brier / 対数損失。手番側の視点で集計する
- ラウンド別の平均バイアス(u_actor − 実際の効用)

診断専用で判定には使わない。両席の value を取るために `GameState` から
`observe(state, pid)` を両席分作るが、これはスクリプトの側であり、
エージェント(`PolicyAgent`)は自席の観測しか受け取らない。
torch は import しない(`net.infer_values` に委ねる)。
"""
import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _entry in (_SCRIPT_DIR, _REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import argparse
import json
import math
import multiprocessing
import time

import eval_agent
from naeco.ai.calibration import ece_with_bins as _ece
from naeco.engine import game as game_mod
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe
from naeco.engine.state import Phase, derive_seed, eval_seed
from naeco.train import encoding as enc
from naeco.train.agent import PolicyAgent

_NET = None
_META = None


def _net():
    global _NET
    if _NET is None:
        _NET = eval_agent._make_checkpoint_agent(_META, 0)._net
    return _NET


def _init(meta_path):
    global _META
    _META = meta_path


def _play(task):
    group, index = task
    cards = load_cards()
    net = _net()
    root = eval_seed(group, index)
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game_mod.new_game(2, env_seed, cards, setup_seed=setup_seed)
    agents = [PolicyAgent(net, mode="greedy", seed=derive_seed(root, "policy", index, v))
              for v in (0, 1)]
    rows = []
    decision_no = 0
    while state.phase != Phase.GAME_OVER:
        who = game_mod.actor(state)
        obs_pair = [observe(state, pid, cards) for pid in (0, 1)]
        u0, u1 = net.infer_values(
            [enc.encode_observation(o, cards) for o in obs_pair])
        obs = obs_pair[who]
        rows.append({
            "decision": decision_no, "round": obs.round_number,
            "actor": who, "pending": obs.pending is not None,
            "u0": u0, "u1": u1,
        })
        action = agents[who].act(obs)
        state = game_mod.step(state, action, cards).state
        decision_no += 1
    utilities = game_mod.terminal_utilities(state, cards)
    return {"group": group, "index": index, "utilities": list(utilities),
            "rows": rows}


def analyze(games):
    sum_u = []
    mapped = []          # p=(u+1)/2 for actor
    antisym = []         # p=0.5+(u_a-u_b)/4 for actor
    by_round = {}
    for g in games:
        util = g["utilities"]
        for r in g["rows"]:
            a = r["actor"]
            ua, ub = (r["u0"], r["u1"]) if a == 0 else (r["u1"], r["u0"])
            y = 1.0 if util[a] > 0 else 0.0 if util[a] < 0 else 0.5
            sum_u.append(r["u0"] + r["u1"])
            mapped.append((max(0.0, min(1.0, (ua + 1) / 2)), y))
            antisym.append((max(0.0, min(1.0, 0.5 + (ua - ub) / 4)), y))
            rb = by_round.setdefault(r["round"], [])
            rb.append(ua - util[a])
    def brier(pairs):
        return sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    def logloss(pairs):
        eps = 1e-6
        return -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                    for p, y in pairs) / len(pairs)
    ece_m, tab_m = _ece(mapped)
    ece_a, tab_a = _ece(antisym)
    return {
        "decisions": len(sum_u),
        "games": len(games),
        "mean_u0_plus_u1": sum(sum_u) / len(sum_u),
        "mean_abs_u0_plus_u1": sum(abs(x) for x in sum_u) / len(sum_u),
        "mapped": {"ece": ece_m, "brier": brier(mapped), "logloss": logloss(mapped),
                   "bins": tab_m},
        "antisymmetric": {"ece": ece_a, "brier": brier(antisym),
                          "logloss": logloss(antisym), "bins": tab_a},
        "round_bias": {str(k): sum(v) / len(v) for k, v in sorted(by_round.items())},
    }


# ---------------------------------------------------------------------------
# Phase 12b G3: 3・4 人戦の「順位効用 ECE」(docs/phase12b-design.md D2)。
#
# ログ方策(--log-meta の greedy、全席)で train 帯の局を回し、各決定点で全席の
# 標準観測 observe(state, seat) を作って、B(--meta)と候補(--compare-meta)の
# 両方で同じ観測から value を推論する。予測 v=(u+1)/2、教師 1−r/(n−1)。
# 勝利確率の ECE とは呼ばない。真の他家情報は入力にしない(各席の標準観測のみ)。
# 全席は各決定点で同数。ラウンドは最少ラウンドの件数に揃えて seed 付きで抽出する。
# n=2 の経路(上)は変更しない。
# ---------------------------------------------------------------------------

G3_SCHEMA = "phase12b-g3/v1"
G3_SAMPLE_SEED = 2026101103

_MULTI = {}


def _init_multi(n, metas):
    _MULTI.clear()
    _MULTI.update({"n": n, "metas": list(metas), "nets": None})


def _multi_nets():
    if _MULTI["nets"] is None:
        from naeco.ai.eval import harness
        nets = []
        for meta in _MULTI["metas"]:
            shas = harness.model_file_shas(meta)
            nets.append(harness._load_net(meta, _MULTI["n"], shas))
        _MULTI["nets"] = nets
    return _MULTI["nets"]


def _play_multi(task):
    """1 局。rows は決定点ごとの全席 u(各 net)。ログ方策は nets[0] 以外でもよい。"""
    group, index = task
    n = _MULTI["n"]
    cards = load_cards()
    nets = _multi_nets()
    log_net = nets[-1]
    root = eval_seed(group, index)
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game_mod.new_game(n, env_seed, cards, setup_seed=setup_seed)
    agents = [PolicyAgent(log_net, mode="greedy",
                          seed=derive_seed(root, "policy", index, seat))
              for seat in range(n)]
    rows = []
    decision_no = 0
    while state.phase != Phase.GAME_OVER:
        who = game_mod.actor(state)
        obs_all = [observe(state, seat, cards) for seat in range(n)]
        encoded = [enc.encode_observation(o, cards) for o in obs_all]
        values = [net.infer_values(encoded) for net in nets[:-1]]
        rows.append({"decision": decision_no,
                     "round": obs_all[who].round_number, "actor": who,
                     "u": values})
        state = game_mod.step(state, agents[who].act(obs_all[who]), cards).state
        decision_no += 1
    ranking = game_mod.ranking(state, cards)
    return {"group": group, "index": index, "ranking": list(ranking),
            "rows": rows}


def g3_observations(games, n, sample_seed=G3_SAMPLE_SEED):
    """(round, seat) 層で同数に揃えた観測。各要素 (round, seat, [u per net], target)。"""
    import random as _random
    by_round = {}
    for g in games:
        for r in g["rows"]:
            by_round.setdefault(r["round"], []).append((g, r))
    if not by_round:
        raise ValueError("no decisions")
    per_round = min(len(v) for v in by_round.values())
    rng = _random.Random(sample_seed)
    out = []
    for rnd in sorted(by_round):
        items = sorted(by_round[rnd],
                       key=lambda gr: (gr[0]["index"], gr[1]["decision"]))
        chosen = sorted(rng.sample(range(len(items)), per_round))
        for i in chosen:
            g, r = items[i]
            for seat in range(n):
                target = 1.0 - g["ranking"][seat] / float(n - 1)
                out.append((rnd, seat, [u[seat] for u in r["u"]], target,
                            [sum(u) for u in r["u"]]))
    return out, per_round


def analyze_multi(games, n, labels, sample_seed=G3_SAMPLE_SEED):
    """ラベル(B / 候補)ごとの順位効用 ECE / Brier / ラウンド別 bias / mean|Σu|。

    教師 1−r/(n−1) は {0, 1/(n−1), …, 1} を取るので、ECE は等幅 10 bin の
    |mean_p − mean_target| を件数重みで足す(calibration.ece_with_bins の
    {0,0.5,1} 教師制約は使わない)。
    """
    obs, per_round = g3_observations(games, n, sample_seed)
    result = {"schema": G3_SCHEMA, "num_players": n, "games": len(games),
              "observations": len(obs), "per_round_decisions": per_round,
              "sample_seed": sample_seed,
              "metric": "rank-utility ECE (v=(u+1)/2 vs 1-r/(n-1)); not a "
                        "win-probability ECE",
              "models": {}}
    for k, label in enumerate(labels):
        pairs = [(max(0.0, min(1.0, (o[2][k] + 1.0) / 2.0)), o[3]) for o in obs]
        buckets = [[] for _ in range(10)]
        for p, y in pairs:
            buckets[min(9, int(p * 10))].append((p, y))
        ece = 0.0
        table = []
        for b, items in enumerate(buckets):
            if items:
                mp = sum(p for p, _ in items) / len(items)
                my = sum(y for _, y in items) / len(items)
                ece += abs(mp - my) * len(items) / len(pairs)
                table.append({"bin": b, "n": len(items), "mean_v": mp,
                              "mean_target": my})
        by_round = {}
        for o, (p, y) in zip(obs, pairs):
            by_round.setdefault(o[0], []).append(p - y)
        sums = [abs(o[4][k]) for o in obs if o[1] == 0]
        result["models"][label] = {
            "rank_utility_ece": ece,
            "rank_utility_brier": sum((p - y) ** 2 for p, y in pairs)
            / len(pairs),
            "mean_abs_sum_u": sum(sums) / len(sums),
            "round_bias": dict((str(r), sum(v) / len(v))
                               for r, v in sorted(by_round.items())),
            "bins": table,
        }
    if len(labels) == 2:
        b, c = (result["models"][x] for x in labels)
        result["targets"] = {
            "ece_half_of_B": c["rank_utility_ece"] <= b["rank_utility_ece"] / 2,
            "sum_third_of_B": c["mean_abs_sum_u"] <= b["mean_abs_sum_u"] / 3,
            "note": "G3 is a diagnostic; not part of G1/G2/R4",
        }
    return result


def main_multi(args):
    from naeco.ai.eval import harness
    n = args.num_players
    if args.group != "train":
        raise SystemExit("multi-seat G3 is a train diagnostic only")
    metas = [args.meta] + ([args.compare_meta] if args.compare_meta else [])
    labels = ["B"] + (["candidate"] if args.compare_meta else [])
    log_meta = args.log_meta or args.meta
    for meta in metas + [log_meta]:
        try:
            harness.check_model_for_players(meta, n)
        except ValueError as exc:
            raise SystemExit("refused: %s" % exc)
    if os.path.exists(args.out):
        raise SystemExit("output already exists: %s" % args.out)
    shas = dict((m, harness.model_file_shas(m)) for m in metas + [log_meta])
    if not args.execute:
        print("prepared: n=%d metas=%s log=%s train %d-%d" % (
            n, metas, log_meta, args.start_index,
            args.start_index + args.games - 1))
        return 0
    tasks = [(args.group, args.start_index + i) for i in range(args.games)]
    started = time.monotonic()
    init = (n, metas + [log_meta])
    if args.workers == 1:
        _init_multi(*init)
        games = [_play_multi(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_init_multi, initargs=init)
        try:
            games = pool.map(_play_multi, tasks, chunksize=1)
        finally:
            pool.close()
            pool.join()
    for m, s in shas.items():
        if harness.model_file_shas(m) != s:
            raise SystemExit("model %s changed during the run" % m)
    summary = analyze_multi(games, n, labels)
    summary.update({"metas": dict(zip(labels, metas)), "log_meta": log_meta,
                    "model_sha256": shas, "group": args.group,
                    "start_index": args.start_index,
                    "end_index": args.start_index + args.games - 1,
                    "workers": args.workers,
                    "wall_seconds": round(time.monotonic() - started, 1)})
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"summary": summary, "games": games}, fh, sort_keys=True,
                  ensure_ascii=True)
        fh.write("\n")
    print(json.dumps(dict((k, v["rank_utility_ece"])
                          for k, v in summary["models"].items()),
                     sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="value 較正診断")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--group", default="train")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-players", type=int, choices=(2, 3, 4), default=2,
                        help="3/4: Phase 12b G3 rank-utility calibration")
    parser.add_argument("--compare-meta", default=None,
                        help="n>=3: candidate meta inferred on the same logs")
    parser.add_argument("--log-meta", default=None,
                        help="n>=3: greedy policy that plays the logged games "
                             "(default --meta)")
    args = parser.parse_args(argv)
    if args.num_players != 2:
        return main_multi(args)
    if args.compare_meta or args.log_meta:
        raise SystemExit("--compare-meta/--log-meta are for --num-players 3|4")
    if not args.execute:
        print("prepared: meta=%s group=%s start=%d games=%d" % (
            args.meta, args.group, args.start_index, args.games))
        return 0
    tasks = [(args.group, args.start_index + i) for i in range(args.games)]
    started = time.monotonic()
    if args.workers == 1:
        _init(args.meta)
        games = [_play(t) for t in tasks]
    else:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(args.workers, initializer=_init,
                                    initargs=(args.meta,))
        try:
            games = pool.map(_play, tasks, chunksize=1)
        finally:
            pool.close()
            pool.join()
    summary = analyze(games)
    summary.update({"meta": args.meta, "group": args.group,
                    "start_index": args.start_index,
                    "end_index": args.start_index + args.games - 1,
                    "wall_seconds": round(time.monotonic() - started, 1)})
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"summary": summary, "games": games}, fh, sort_keys=True,
                  ensure_ascii=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("round_bias",)}, indent=1, sort_keys=True))
    print("round_bias:", summary["round_bias"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
