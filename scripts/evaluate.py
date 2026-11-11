"""固定 seed 群で HeuristicAgent vs RandomAgent を評価する CLI。

    python scripts/evaluate.py --pairs 500 --group dev --start-index 0 \
        --records docs/eval-runs/phase2-dev-0-499.jsonl
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.ai.eval.harness import (  # noqa: E402
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, evaluate, passes_threshold,
    write_game_records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ナショナルエコノミー CPU 評価")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--group", choices=("train", "dev", "accept"), default="dev")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--records", default=None,
                        help="対応表(1局1行JSONL)の出力先。PLAN.md §5 の証跡。")
    args = parser.parse_args(argv)
    result = evaluate(args.pairs, args.group, args.start_index,
                      BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    end_index = args.start_index + args.pairs - 1
    print("Phase 2: HeuristicAgent vs RandomAgent")
    print("group=%s index=%d-%d pairs=%d games=%d" %
          (result.group, result.start_index, end_index, result.pairs, result.games))
    print("wins/draws/losses=%d/%d/%d" %
          (result.wins, result.draws, result.losses))
    print("point_estimate=%.4f  95%% CI=[%.4f, %.4f]" %
          (result.point_estimate, result.ci_low, result.ci_high))
    print("bootstrap=B%d seed=%d" %
          (result.bootstrap_samples, result.bootstrap_seed))
    if args.records:
        directory = os.path.dirname(args.records)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        write_game_records(result, args.records)
        print("records=%s (%d rows)" % (args.records, len(result.game_records)))
    passed = passes_threshold(result, args.threshold)
    print("pass (CI low > %.2f): %s" % (args.threshold, passed))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
