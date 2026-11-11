"""Phase 7 段1: Elo ラダー(総当たりの席交換ペア対局 + Bradley-Terry 推定)。

エージェント指定は `random` / `heuristic` / `checkpoint:<meta>`(greedy)/
`search:<meta>[@<budget>]`(Phase 5 認定構成の SO-ISMCTS、S2+L1)/
`gumbel:<meta>[@<budget>]`(段2 の Gumbel 根選択)の5種。
全ペアリングは**同じ index 帯(同じ配札)**で対局するので、相手ごとの勝率が
配札の違いに左右されない。

- 診断(`train` 帯)にも、Phase 7 の G1/G2 のような**正式判定(`dev` 帯)**にも使う。
  正式判定では `promote.py` と同じ来歴を残す(Sol ゲート 2026-09-05 の指摘):
  既存出力ファイルを拒否する / checkpoint・search エージェントのモデル(meta と weights)を
  実行前に SHA-256 で凍結し実行後に再検査する / 起動時 HEAD・seed 台帳の SHA-256・開始と終了の
  時刻・実行条件・`formal`(group=="dev")を出力 JSON に保存する。
- worker 数を変えても結果は一致する(seed は group / index / seat_variant
  だけから派生し、net は worker ごとに1回だけロードする)。
- torch は import しない。重みのロードは `naeco/train/net.py` の
  `load_state_dict_from_path` に委ねる。
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
import hashlib
import itertools
import json
import math
import multiprocessing
import platform
import subprocess
import time

import eval_agent
from naeco.ai.eval.harness import (
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap, summarize_pairs,
)
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.random_agent import RandomAgent
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.engine.cards import load_cards
from naeco.engine.state import GROUPS, derive_seed, eval_seed
from naeco.train.agent import PolicyAgent
from naeco.train.env import play_episode
from naeco.train.search_evaluator import NetEvaluator

DEFAULT_BUDGET = 4500
C_PUCT = 1.5
HORIZON = 2

_NETS = {}


def parse_spec(spec):
    """spec -> (kind, meta_path, budget)。"""
    kind, sep, rest = spec.partition(":")
    if kind in ("random", "heuristic"):
        if sep:
            raise ValueError("%s takes no argument: %r" % (kind, spec))
        return (kind, "", 0)
    if kind == "checkpoint":
        if not rest:
            raise ValueError("checkpoint spec requires a meta path")
        return (kind, rest, 0)
    if kind in ("search", "gumbel"):
        path, at, budget = rest.partition("@")
        if not path:
            raise ValueError("%s spec requires a meta path" % kind)
        return (kind, path, int(budget) if at else DEFAULT_BUDGET)
    raise ValueError("unknown agent spec %r" % (spec,))


def _get_net(meta_path):
    """メタのパスごとに net を1回だけロードして保持する(eval_agent と同じ検証)。"""
    key = os.path.abspath(meta_path)
    net = _NETS.get(key)
    if net is None:
        agent = eval_agent._make_checkpoint_agent(meta_path, 0)
        net = agent._net
        _NETS[key] = net
    return net


def make_agent(spec, seed, cards):
    kind, path, budget = parse_spec(spec)
    if kind == "random":
        return RandomAgent(seed)
    if kind == "heuristic":
        return HeuristicAgent(seed)
    net = _get_net(path)
    if kind == "checkpoint":
        return PolicyAgent(net, mode="greedy", seed=seed)
    return IsmctsAgent(seed, budget, c_puct=C_PUCT, horizon=HORIZON,
                       cards=cards, evaluator=NetEvaluator(net),
                       policy_agent=PolicyAgent(net, mode="greedy", seed=0),
                       root_policy="gumbel" if kind == "gumbel" else "puct")


def _play(task):
    """1局。例外はどの局で起きたかが分かるように task 付きで返す。"""
    try:
        return _play_one(task)
    except Exception as exc:          # noqa: BLE001 - 診断ツール。局を特定して再送出する
        import traceback
        return {"error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(), "task": list(task)}


def _play_one(task):
    group, index, seat_variant, left, right = task
    cards = load_cards()
    root = eval_seed(group, index)
    left_seed = derive_seed(root, "policy", index, seat_variant)
    right_seed = derive_seed(root, "policy", index, 1 - seat_variant)
    agents = [make_agent(left, left_seed, cards),
              make_agent(right, right_seed, cards)]
    started = time.monotonic()
    trajectory = play_episode(group, index, seat_variant, agents,
                              cards=cards, record_vectors=False)
    if trajectory.discarded:
        raise RuntimeError("discarded %s/%d v=%d: %s" % (
            group, index, seat_variant, trajectory.discard_reason))
    utility = trajectory.terminal_rewards[seat_variant]
    outcome = 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5
    return {
        "group": group, "index": index, "seat_variant": seat_variant,
        "left": left, "right": right, "result": outcome,
        "decisions": len(trajectory.decisions),
        "initial_state_hash": trajectory.initial_state_hash,
        "seconds": round(time.monotonic() - started, 3),
    }


def _pairing(records, left, right, start_index, pairs):
    pair_games = []
    for offset in range(pairs):
        r0 = records[(left, right, start_index + offset, 0)]
        r1 = records[(left, right, start_index + offset, 1)]
        if r0["initial_state_hash"] != r1["initial_state_hash"]:
            raise AssertionError("pair %d did not share a deal" % offset)
        pair_games.append((r0["result"], r1["result"]))
    wins, draws, losses, pair_results, point = summarize_pairs(pair_games)
    lo, hi = cluster_bootstrap(pair_results, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    secs = [records[(left, right, start_index + o, v)]["seconds"]
            for o in range(pairs) for v in (0, 1)]
    return {
        "left": left, "right": right, "pairs": pairs,
        "wins": wins, "draws": draws, "losses": losses,
        "point_estimate": point, "ci_low": lo, "ci_high": hi,
        "mean_game_seconds": sum(secs) / len(secs),
    }


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def freeze_models(agents):
    """checkpoint / search / gumbel エージェントのモデルを SHA-256 で凍結する。"""
    frozen = {}
    for spec in agents:
        kind, meta, unused_budget = parse_spec(spec)
        if not meta:
            continue
        weights = eval_agent.checkpoint_weights_path(meta)
        frozen[spec] = {
            "meta_path": os.path.abspath(meta),
            "meta_sha256": _sha256_file(meta),
            "weights_path": os.path.abspath(weights),
            "weights_sha256": _sha256_file(weights),
        }
    return frozen


def verify_frozen(frozen):
    """凍結値と現在のファイルを再検査する。1つでも違えば RuntimeError(fail-closed)。"""
    for spec, entry in frozen.items():
        now_meta = _sha256_file(entry["meta_path"])
        now_weights = _sha256_file(entry["weights_path"])
        if now_meta != entry["meta_sha256"] or now_weights != entry["weights_sha256"]:
            raise RuntimeError(
                "model changed during evaluation: %s (meta %s -> %s, weights %s -> %s)"
                % (spec, entry["meta_sha256"][:12], now_meta[:12],
                   entry["weights_sha256"][:12], now_weights[:12]))


def _git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True,
            text=True, timeout=30)
        if out.returncode == 0:
            head = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=_REPO_ROOT,
                capture_output=True, text=True, timeout=30)
            if dirty.returncode == 0 and dirty.stdout.strip():
                head += "-dirty"
            return head
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable"


def provenance(agents, group, start_index, pairs, workers, ledger_path=None):
    """実行前に確定する来歴(HEAD・台帳 SHA・条件・凍結モデル・開始時刻)。"""
    ledger = ledger_path or os.path.join(_REPO_ROOT, "docs", "eval-seeds.md")
    return {
        "git_commit": _git_head(),
        "seed_ledger": {"path": os.path.abspath(ledger),
                        "sha256": _sha256_file(ledger) if os.path.exists(ledger) else None},
        "conditions": {"agents": list(agents), "group": group,
                       "start_index": start_index,
                       "end_index": start_index + pairs - 1, "pairs": pairs,
                       "workers": workers, "c_puct": C_PUCT, "horizon": HORIZON,
                       "default_budget": DEFAULT_BUDGET},
        # 判定用の群は dev(昇格・G1/G2)と accept(最終受け入れ)。train は診断。
        "formal": group in ("dev", "accept"),
        "frozen_models": freeze_models(agents),
        "python_version": platform.python_version(),
        "host": platform.node(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def bradley_terry(agents, pairings, iterations=2000):
    """Bradley-Terry の MM 推定。戻り値は Elo(平均 0)。"""
    idx = {name: i for i, name in enumerate(agents)}
    n = len(agents)
    wins = [[0.0] * n for _ in range(n)]      # wins[i][j] = i が j に勝った数
    for p in pairings:
        i, j = idx[p["left"]], idx[p["right"]]
        w = p["wins"] + 0.5 * p["draws"]
        l = p["losses"] + 0.5 * p["draws"]
        wins[i][j] += w
        wins[j][i] += l
    strength = [1.0] * n
    for _ in range(iterations):
        new = []
        for i in range(n):
            num = sum(wins[i][j] for j in range(n) if j != i)
            den = 0.0
            for j in range(n):
                if j == i:
                    continue
                games = wins[i][j] + wins[j][i]
                if games > 0:
                    den += games / (strength[i] + strength[j])
            new.append(max(num / den, 1e-9) if den > 0 else strength[i])
        mean_log = sum(math.log(s) for s in new) / n
        strength = [s / math.exp(mean_log) for s in new]
    return {name: 400.0 * math.log10(strength[idx[name]]) for name in agents}


def run(agents, group, start_index, pairs, workers):
    tasks = []
    for left, right in itertools.combinations(agents, 2):
        for offset in range(pairs):
            for variant in (0, 1):
                tasks.append((group, start_index + offset, variant, left, right))
    started = time.monotonic()
    if workers == 1:
        results = [_play(t) for t in tasks]
    else:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(processes=workers)
        try:
            results = pool.map(_play, tasks, chunksize=1)
        finally:
            pool.close()
            pool.join()
    errors = [r for r in results if "error" in r]
    if errors:
        for r in errors:
            print("ERROR task=%s %s" % (r["task"], r["error"]))
        print(errors[0]["traceback"])
        raise RuntimeError("%d games failed (first: task=%s)" % (
            len(errors), errors[0]["task"]))
    records = {(r["left"], r["right"], r["index"], r["seat_variant"]): r
               for r in results}
    pairings = [_pairing(records, l, r, start_index, pairs)
                for l, r in itertools.combinations(agents, 2)]
    elo = bradley_terry(agents, pairings)
    return {
        "agents": list(agents), "group": group,
        "start_index": start_index, "end_index": start_index + pairs - 1,
        "pairs_per_pairing": pairs, "pairings": pairings, "elo": elo,
        "games": len(results),
        "mean_decisions_per_game": sum(r["decisions"] for r in results) / len(results),
        "wall_seconds": round(time.monotonic() - started, 1),
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
    }, results


def run_multi_diagnostic(agents, num_players, start_index, clusters, workers):
    """3・4 人戦の診断表(D5: 二者 Elo を流用しない。勝率/順位だけを並べる)。

    正準リスト `agents`(長さ n)を巡回させ、各 agent を 1 回ずつ評価対象
    (正準位置 0)にした n 局巡回クラスタ評価を、同じ train 帯で行う。
    判定ではないので train 群だけを受け付け、台帳の多人数予約は使わない。
    """
    from naeco.ai.eval import harness
    agents = list(agents)
    if len(agents) != num_players:
        raise ValueError("need exactly %d agents for n=%d" % (num_players,
                                                               num_players))
    rows = []
    records_out = []
    for shift in range(num_players):
        lineup = agents[shift:] + agents[:shift]
        cond = harness.multi_conditions(num_players, lineup, "train",
                                        start_index, clusters,
                                        "ladder-diagnostic", None)
        run_id = harness.multi_run_id(cond)
        tasks = harness.cyclic_tasks(run_id, "train", start_index, clusters,
                                     lineup, num_players,
                                     models=cond.get("models"))
        records = harness.run_cyclic_games(tasks, workers=workers)
        summary = harness.summarize_multi(records, cond, run_id)
        records_out.extend(records)
        rows.append({
            "target": lineup[0], "lineup": lineup, "run_id": run_id,
            "clusters": clusters, "games": summary["games"],
            "win_rate": summary["point_estimate"],
            "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
            "baseline_1_over_n": summary["baseline_1_over_n"],
            "mean_rank": summary["target_mean_rank"],
            "rank_histogram": summary["target_rank_histogram"],
            "mean_score": summary["target_mean_score"],
        })
    return {
        "kind": "multi-seat diagnostic (no Elo; D5)",
        "num_players": num_players, "agents": agents, "group": "train",
        "start_index": start_index, "end_index": start_index + clusters - 1,
        "clusters": clusters, "rows": rows,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }, records_out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 7 Elo ラダー")
    parser.add_argument("agents", nargs="+", help="エージェント指定(2つ以上)")
    parser.add_argument(
        "--num-players", type=int, choices=(2, 3, 4), default=2,
        help="3/4: 多人数の勝率・順位診断表(Elo なし、train のみ。D5)")
    parser.add_argument("--group", choices=sorted(GROUPS), default="train")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True, help="集計 JSON の出力先")
    parser.add_argument("--records", default=None, help="1局1行 JSONL の出力先")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.num_players != 2:
        if args.group != "train":
            raise SystemExit(
                "multi-seat ladder is a train diagnostic; formal n>=3 "
                "judgements use scripts/eval_agent.py --num-players with a "
                "ledger reservation")
        for path in (args.out, args.records):
            if path and os.path.exists(path):
                raise SystemExit("output already exists: %s" % path)
        if not args.execute:
            print("prepared: n=%d %s train start=%d clusters=%d workers=%d"
                  % (args.num_players, args.agents, args.start_index,
                     args.pairs, args.workers))
            return 0
        try:
            summary, records = run_multi_diagnostic(
                args.agents, args.num_players, args.start_index, args.pairs,
                args.workers)
        except ValueError as exc:
            raise SystemExit(str(exc))
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, sort_keys=True, ensure_ascii=True, indent=1)
            fh.write("\n")
        if args.records:
            from naeco.ai.eval import harness
            with open(args.records, "w", encoding="utf-8", newline="\n") as fh:
                for rec in records:
                    fh.write(harness.record_line(rec))
        for row in summary["rows"]:
            print("%-10s win %.4f [%.4f, %.4f] (1/n %.3f) mean rank %.3f %s"
                  % (row["target"], row["win_rate"], row["ci_low"],
                     row["ci_high"], row["baseline_1_over_n"],
                     row["mean_rank"], row["rank_histogram"]))
        return 0
    if len(args.agents) < 2:
        raise SystemExit("at least two agents are required")
    for spec in args.agents:
        parse_spec(spec)
    for path in (args.out, args.records):
        if path and os.path.exists(path):
            raise SystemExit(
                "output already exists (refusing to overwrite; a judgment is run once): %s"
                % path)
    if not args.execute:
        print("prepared: %s group=%s start=%d pairs=%d workers=%d"
              % (args.agents, args.group, args.start_index, args.pairs,
                 args.workers))
        return 0
    prov = provenance(args.agents, args.group, args.start_index, args.pairs,
                      args.workers)
    summary, results = run(args.agents, args.group, args.start_index,
                           args.pairs, args.workers)
    verify_frozen(prov["frozen_models"])
    prov["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prov["frozen_models_verified_after_run"] = True
    summary["provenance"] = prov
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, sort_keys=True, ensure_ascii=True, indent=1)
        fh.write("\n")
    if args.records:
        with open(args.records, "w", encoding="utf-8", newline="\n") as fh:
            for r in results:
                fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True))
                fh.write("\n")
    for p in summary["pairings"]:
        print("%-40s vs %-40s %.4f [%.4f, %.4f] (%d/%d/%d) %.2fs/game" % (
            p["left"], p["right"], p["point_estimate"], p["ci_low"],
            p["ci_high"], p["wins"], p["draws"], p["losses"],
            p["mean_game_seconds"]))
    for name, rating in sorted(summary["elo"].items(), key=lambda kv: -kv[1]):
        print("Elo %8.1f  %s" % (rating, name))
    print("games=%d mean_decisions=%.1f wall=%.1fs" % (
        summary["games"], summary["mean_decisions_per_game"],
        summary["wall_seconds"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
