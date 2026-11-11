"""性能ベンチマーク(PLAN.md Phase 1 完了条件 / 設計書 §11)。

1局時間・step/秒・clone時間・serialize時間・状態サイズを計測して表示する。
遅ければこの時点で表現を見直す(numpy化 / in-place+undo)判断材料にする。

    python scripts/benchmark.py [--games N] [--players 2]
"""
import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.engine import game                             # noqa: E402
from naeco.engine.cards import load_cards                 # noqa: E402
from naeco.engine.state import Phase, clone, serialize, state_hash  # noqa: E402

CARDS = load_cards()


def play_measured(n, seed):
    rng = random.Random(seed ^ 0xABCDEF)
    s = game.new_game(n, seed, CARDS)
    steps = 0
    t0 = time.perf_counter()
    while s.phase != Phase.GAME_OVER:
        la = game.legal_actions(s, CARDS)
        s = game.step(s, rng.choice(la), CARDS).state
        steps += 1
    dt = time.perf_counter() - t0
    return dt, steps, s


def bench_clone(s, iters=20000):
    t0 = time.perf_counter()
    for _ in range(iters):
        clone(s)
    return (time.perf_counter() - t0) / iters


def bench_serialize(s, iters=5000):
    t0 = time.perf_counter()
    for _ in range(iters):
        serialize(s)
    return (time.perf_counter() - t0) / iters


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--players", type=int, default=2)
    args = ap.parse_args(argv)

    game_times = []
    steps_total = 0
    last = None
    for seed in range(args.games):
        dt, steps, s = play_measured(args.players, seed)
        game_times.append(dt)
        steps_total += steps
        last = s

    total_time = sum(game_times)
    mid = game.new_game(args.players, 0, CARDS)
    # 代表的な中盤局面を1つ作って clone/serialize を測る
    rng = random.Random(7)
    for _ in range(60):
        if mid.phase == Phase.GAME_OVER:
            break
        mid = game.step(mid, rng.choice(game.legal_actions(mid, CARDS)), CARDS).state

    clone_us = bench_clone(mid) * 1e6
    ser_us = bench_serialize(mid) * 1e6
    size = len(serialize(last))
    size_nohist = len(serialize(last, include_history=False))

    print("=== ナショナルエコノミー エンジン性能ベンチ ===")
    print("環境: Python %s, %d人戦, %d局(ランダム方策)"
          % (sys.version.split()[0], args.players, args.games))
    print("-" * 52)
    print("%-26s %10.3f ms   (目標 <=20)"
          % ("1局あたり(平均)", statistics.mean(game_times) * 1e3))
    print("%-26s %10.3f ms"
          % ("1局あたり(p95)", sorted(game_times)[int(len(game_times) * 0.95)] * 1e3))
    print("%-26s %10.0f     (目標 >=5000)"
          % ("step/秒", steps_total / total_time))
    print("%-26s %10.2f us   (目標 <=50)" % ("clone", clone_us))
    print("%-26s %10.2f us   (目標 <=200)" % ("serialize", ser_us))
    print("%-26s %10d bytes(目標 <=4096)" % ("状態サイズ(履歴含む)", size))
    print("%-26s %10d bytes" % ("状態サイズ(履歴除く)", size_nohist))
    print("-" * 52)
    print("平均step数/局 %.1f" % (steps_total / args.games))


if __name__ == "__main__":
    main()
