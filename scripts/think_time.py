"""Phase 8: 「強」候補の思考時間をローカル基準機で測る(設計 D-4 の UX 条件: 平均 2 秒 / p95 5 秒)。

単一プロセス・他負荷なしで、`search:<meta>@T`(SO-ISMCTS、PUCT、H=2、評価器 = 同じネット)と
greedy(同じネット)の対局を `train` 帯で N 局打ち、探索した決定点ごとの壁時計
(`IsmctsAgent.stats.elapsed_ms`)を集計する。委譲(pending / 一択)は探索しないので含めない。
判定には使わない(診断)。torch は import しない。
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
import platform
import time

import ladder
from naeco.engine.cards import load_cards
from naeco.engine.state import derive_seed, eval_seed
from naeco.train.env import play_episode


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main(argv=None):
    parser = argparse.ArgumentParser(description="強候補の思考時間(ローカル基準機)")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1500, 4500])
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--group", default="train")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("prepared", args)
        return 0
    cards = load_cards()
    report = {"meta": args.meta, "host": platform.node(), "python": platform.python_version(),
              "group": args.group, "start_index": args.start_index, "games": args.games,
              "budgets": {}}
    for budget in args.budgets:
        per_decision = []
        per_game = []
        for i in range(args.games):
            index = args.start_index + i
            root = eval_seed(args.group, index)
            search = ladder.make_agent("search:%s@%d" % (args.meta, budget),
                                       derive_seed(root, "policy", index, 0), cards)
            greedy = ladder.make_agent("checkpoint:%s" % args.meta,
                                       derive_seed(root, "policy", index, 1), cards)
            started = time.monotonic()
            traj = play_episode(args.group, index, i % 2, [search, greedy] if i % 2 == 0
                                else [greedy, search], cards=cards, record_vectors=False)
            secs = time.monotonic() - started
            per_decision.extend(search.stats.elapsed_ms)
            per_game.append({"index": index, "seconds": round(secs, 2),
                             "search_decisions": search.stats.decisions,
                             "delegated": search.stats.delegated_calls,
                             "decisions_total": len(traj.decisions)})
        summary = {
            "search_decisions": len(per_decision),
            "mean_ms": sum(per_decision) / len(per_decision) if per_decision else None,
            "p50_ms": _percentile(per_decision, 0.5),
            "p95_ms": _percentile(per_decision, 0.95),
            "max_ms": max(per_decision) if per_decision else None,
            "games": per_game,
        }
        report["budgets"][str(budget)] = summary
        print("T=%d: search decisions %d, mean %.0f ms, p50 %.0f ms, p95 %.0f ms, max %.0f ms" % (
            budget, summary["search_decisions"], summary["mean_ms"], summary["p50_ms"],
            summary["p95_ms"], summary["max_ms"]))
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, ensure_ascii=True, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
