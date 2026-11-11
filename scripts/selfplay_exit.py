"""Phase 7 段2/4: Expert Iteration 用の自己対戦データ収集(torch を import しない)。

学習側(探索つき)を「教師」として対局し、教師の**探索した決定点**ごとに
観測ベクトル・合法手ベクトル・根の統計(改善方策 π'、訪問分布、completed Q、
根 value)を記録する。終局後に各決定点へ勝敗 z(記録席の効用 ±1)、最終得点差、
両者の得点を付与する。

- 対戦カード: 記録席 = `search`(または `gumbel`)エージェント、相手席 = `--opponent`
  (`checkpoint:<meta>` / `search:<meta>` / `gumbel:<meta>` / `heuristic` / `random`)。
  `--both-sides` なら両席とも教師にし、両席の決定点を記録する(純自己対戦)。
- Playout cap randomization(KataGo): 決定点ごとに確率 `--p-full` で高予算
  (`--budget`)で探索して方策ターゲットを記録し、それ以外は低予算(`--budget-low`)
  で探索して**方策ターゲットを記録しない**(価値ターゲットだけ)。抽選は
  `derive_seed(root, "playoutcap", index, decision)` 由来で決定論的。
- pending(下位決定)は policy_agent へ委譲され探索しないので、`kind="pending"`
  として観測だけ記録する(価値ターゲット用)。
- 出力は gzip JSONL(1決定点1行)。worker 数を変えても内容は一致する。
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
import gzip
import json
import multiprocessing
import random
import time

import ladder
from naeco.engine import game as game_mod
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe
from naeco.engine.state import Phase, derive_seed, eval_seed
from naeco.train import encoding as enc
from naeco.train.search_evaluator import NetEvaluator
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.train.agent import PolicyAgent

DATA_VERSION = "exit-1"


def _teacher(spec_kind, meta, budget, seed, cards, root_policy_kwargs):
    net = ladder._get_net(meta)
    return IsmctsAgent(seed, budget, c_puct=ladder.C_PUCT, horizon=ladder.HORIZON,
                       cards=cards, evaluator=NetEvaluator(net),
                       policy_agent=PolicyAgent(net, mode="greedy", seed=0),
                       root_policy="gumbel" if spec_kind == "gumbel" else "puct",
                       **root_policy_kwargs)


def _play(task):
    (group, index, teacher_kind, meta, budget, budget_low, p_full,
     opponent, both_sides, seat_variant, root_policy_kwargs) = task
    cards = load_cards()
    root = eval_seed(group, index)
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game_mod.new_game(2, env_seed, cards, setup_seed=setup_seed)
    initial_hash = None
    try:
        from naeco.engine.state import state_hash
        initial_hash = state_hash(state)
    except ImportError:
        pass
    teacher_seat = seat_variant
    seeds = [derive_seed(root, "policy", index, v) for v in (0, 1)]
    agents = [None, None]
    teachers = {}
    for seat in (0, 1):
        if seat == teacher_seat or both_sides:
            full = _teacher(teacher_kind, meta, budget, seeds[seat], cards,
                            root_policy_kwargs)
            low = (_teacher(teacher_kind, meta, budget_low, seeds[seat], cards,
                            root_policy_kwargs) if budget_low else None)
            teachers[seat] = (full, low)
            agents[seat] = full
        else:
            agents[seat] = ladder.make_agent(opponent, seeds[seat], cards)
    cap_rng = random.Random(derive_seed(root, "playoutcap", index))
    rows = []
    decision_no = 0
    started = time.monotonic()
    while state.phase != Phase.GAME_OVER:
        who = game_mod.actor(state)
        obs = observe(state, who, cards)
        legal = obs.legal_actions
        if who in teachers:
            full, low = teachers[who]
            use_full = (low is None) or (cap_rng.random() < p_full)
            agent = full if use_full else low
            searched = obs.pending is None and len(legal) > 1
            action = agent.act(obs)
            row = {
                "index": index, "decision": decision_no, "seat": who,
                "round": obs.round_number,
                "kind": "search" if searched else "pending",
                "full": bool(use_full),
                "obs": [round(x, 6) for x in enc.encode_observation(obs, cards)],
            }
            if searched:
                rec = agent.last_root
                order = rec["action_list"]
                row["actions"] = [[round(x, 6) for x in enc.encode_action(obs, a, cards)]
                                  for a in order]
                row["chosen"] = rec["chosen_index"]
                row["root_value"] = rec["root_value"]
                row["visits"] = rec["visits"]
                if use_full:
                    row["policy"] = [round(p, 6) for p in rec["improved_policy"]]
                    row["completed_q"] = [round(q, 6) for q in rec["completed_q"]]
                    row["prior"] = [round(p, 6) for p in rec["prior"]]
            else:
                row["chosen"] = legal.index(action)
            rows.append(row)
        else:
            action = agents[who].act(obs)
        state = game_mod.step(state, action, cards).state
        decision_no += 1
    utilities = game_mod.terminal_utilities(state, cards)
    scores = game_mod.final_scores(state, cards)
    for row in rows:
        seat = row["seat"]
        row["z"] = float(utilities[seat])
        row["score_self"] = int(scores[seat])
        row["score_opp"] = int(scores[1 - seat])
        row["score_diff"] = int(scores[seat] - scores[1 - seat])
    return {
        "index": index, "teacher_seat": teacher_seat, "both_sides": both_sides,
        "utilities": list(utilities), "scores": list(scores),
        "decisions": decision_no, "rows": rows,
        "initial_state_hash": initial_hash,
        "seconds": round(time.monotonic() - started, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Expert Iteration 自己対戦データ収集")
    parser.add_argument("--meta", required=True, help="教師ネットのメタ(checkpoint)")
    parser.add_argument("--teacher", choices=("search", "gumbel"), default="search")
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--budget-low", type=int, default=0,
                        help="低予算探索の遷移数(0 で playout cap randomization 無効)")
    parser.add_argument("--p-full", type=float, default=0.25)
    parser.add_argument("--opponent", default=None,
                        help="相手席の指定(既定: 教師と同じネットの greedy)")
    parser.add_argument("--both-sides", action="store_true")
    parser.add_argument("--group", default="train")
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gumbel-m", type=int, default=8)
    parser.add_argument("--c-visit", type=float, default=50.0)
    parser.add_argument("--c-scale", type=float, default=1.0)
    parser.add_argument("--out", required=True, help="gzip JSONL の出力先")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    opponent = args.opponent or ("checkpoint:%s" % args.meta)
    ladder.parse_spec(opponent)
    kw = {"gumbel_m": args.gumbel_m, "c_visit": args.c_visit,
          "c_scale": args.c_scale}
    tasks = []
    for i in range(args.games):
        index = args.start_index + i
        tasks.append((args.group, index, args.teacher, args.meta, args.budget,
                      args.budget_low, args.p_full, opponent, args.both_sides,
                      i % 2, kw))
    if not args.execute:
        print("prepared: %d games %s..%d teacher=%s budget=%d low=%d p_full=%.2f opp=%s"
              % (args.games, args.group, args.start_index + args.games - 1,
                 args.teacher, args.budget, args.budget_low, args.p_full, opponent))
        return 0
    started = time.monotonic()
    if args.workers == 1:
        games = [_play(t) for t in tasks]
    else:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(args.workers)
        try:
            games = pool.map(_play, tasks, chunksize=1)
        finally:
            pool.close()
            pool.join()
    n_rows = n_policy = 0
    header = {
        "data_version": DATA_VERSION, "meta": args.meta, "teacher": args.teacher,
        "budget": args.budget, "budget_low": args.budget_low, "p_full": args.p_full,
        "opponent": opponent, "both_sides": args.both_sides, "group": args.group,
        "start_index": args.start_index, "games": args.games,
        "obs_encoder_version": enc.OBS_ENCODER_VERSION,
        "action_encoder_version": enc.ACTION_ENCODER_VERSION,
        "obs_dim": enc.obs_dim(2), "action_dim": enc.action_dim(),
        "root_policy_kwargs": kw,
    }
    with gzip.open(args.out, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps({"header": header}, sort_keys=True))
        fh.write("\n")
        for g in games:
            for row in g["rows"]:
                fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
                fh.write("\n")
                n_rows += 1
                if "policy" in row:
                    n_policy += 1
    wins = sum(1 for g in games if g["utilities"][g["teacher_seat"]] > 0)
    print("games=%d teacher_wins=%d rows=%d policy_rows=%d mean_game_s=%.2f wall=%.1fs"
          % (len(games), wins, n_rows, n_policy,
             sum(g["seconds"] for g in games) / len(games),
             time.monotonic() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
