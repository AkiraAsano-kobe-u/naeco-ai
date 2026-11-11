"""Phase 2.5 再ゲートの検出力設計(SOL-P25G-007)を決定論的に計算する。

ペア勝率(0 / 0.5 / 1)の平均を 0.5 と比較するために必要なペア数を、
標準ライブラリのみ(statistics.NormalDist)で求める。CLI と import の両方で使える。

- 有意水準: 既定は **両側 alpha=0.05**。primary(①②)は片側 alpha=0.05 の
  Holm 補正だが、secondary(③ PIMC vs IS-MCTS)の両側 alpha=0.05 を保守基準として
  `--sides 2` を既定にする。
- 検出力: 既定 0.80。
- 実用最小差 delta: 既定 0.05。
- sigma_pair: `--sigma`、または records の該当行 `pair_scores` から標本標準偏差
  (ddof=1)を取得できる。ファイル欠落・該当行なし・標本が1個以下の場合は
  [0,1] のペア結果に対する保守上界 **0.5** へ fallback する。
  **JSON 破損・非 dict 行・pair_scores の欠落/空/非数/[0,1]外はエラー**であり、
  黙って正常データとして扱わない。

必要ペア数の式(正規近似):

    n = ceil(((z_{1-alpha/sides} + z_{1-beta}) * sigma / delta) ** 2)

実行例:

    python scripts/power_search.py --delta 0.05 --power 0.80
    python scripts/power_search.py --records docs/eval-runs/phase2.5-stage1-train.jsonl \
        --matchup ismcts-vs-heuristic --delta 0.05 --power 0.80
"""
import argparse
import json
import math
import statistics
from statistics import NormalDist

FALLBACK_SIGMA = 0.5


def normal_quantile(probability):
    """標準正規分布の下側 probability 分位点。"""
    return NormalDist().inv_cdf(probability)


def required_pairs(delta=0.05, sigma=FALLBACK_SIGMA, alpha=0.05, power=0.80,
                   sides=2):
    """H0: ペア勝率 = 0.5 の検定に必要なペア数を ceil で返す。

    引数はすべて実数で、delta / sigma / alpha / power / sides の境界違反は
    ValueError で拒否する。戻り値は決定論的な int。
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a positive finite number")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1")
    if not (0.0 < power < 1.0):
        raise ValueError("power must be strictly between 0 and 1")
    if sides not in (1, 2):
        raise ValueError("sides must be 1 (one-sided) or 2 (two-sided)")

    z_alpha = normal_quantile(1.0 - alpha / float(sides))
    z_beta = normal_quantile(power)          # z_{1-beta}
    numerator = (z_alpha + z_beta) * sigma
    return int(math.ceil((numerator / delta) ** 2))


def _row_matches(row, matchup, label, budget):
    if label is not None and row.get("label") != label:
        return False
    if budget is not None and row.get("budget") != budget:
        return False
    if matchup is None:
        return True
    if row.get("label") == matchup:
        return True
    left, right = row.get("left"), row.get("right")
    return (left is not None and right is not None
            and "%s-vs-%s" % (left, right) == matchup)


def collect_pair_scores(path, matchup=None, label=None, budget=None):
    """records から該当行の `pair_scores` をファイル順に連結して返す。

    読み込み失敗・JSON 破損・`pair_scores` 欠落・空・[0,1] 外・非数の値が
    1つでもある・該当行が無い場合は None を返す。呼び出し側は
    FALLBACK_SIGMA へ fallback する。
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError):
        return None

    scores = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            return None
        if not isinstance(row, dict):
            return None
        if not _row_matches(row, matchup, label, budget):
            continue
        pair_scores = row.get("pair_scores")
        if not isinstance(pair_scores, list) or not pair_scores:
            return None
        for value in pair_scores:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                return None
        scores.extend(pair_scores)
    return scores if scores else None


def classify_records(path, matchup=None, label=None, budget=None):
    """records を strict に検査し、sigma 取得失敗の理由を返す。

    - "missing": ファイルが読めない(該当なしと同じく fallback 可)
    - "corrupt": JSON 破損行がある(黙って fallback しない)
    - "invalid": pair_scores 欠落・空・[0,1] 外・非数がある(同上)
    - "no_match": 有効だが該当行が無い(fallback 可)
    - "valid": 該当行が有効
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError):
        return "missing"
    matched = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError:
            return "corrupt"
        if not isinstance(row, dict):
            return "corrupt"
        if not _row_matches(row, matchup, label, budget):
            continue
        matched += 1
        pair_scores = row.get("pair_scores")
        if not isinstance(pair_scores, list) or not pair_scores:
            return "invalid"
        for value in pair_scores:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "invalid"
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                return "invalid"
    return "no_match" if matched == 0 else "valid"


def sigma_from_records(path, matchup=None, label=None, budget=None):
    """該当行の pair_scores から標本標準偏差(ddof=1)を返す。

    取得できない・標本が1個以下・無効な場合は None(→ fallback)。
    """
    scores = collect_pair_scores(path, matchup=matchup, label=label,
                                 budget=budget)
    if scores is None or len(scores) < 2:
        return None
    return statistics.stdev(scores)


def _source_line(sigma, args, from_records):
    if args.sigma is not None:
        return "explicit --sigma"
    if from_records:
        details = "records=%s" % args.records
        if args.matchup is not None:
            details += " matchup=%s" % args.matchup
        if args.label is not None:
            details += " label=%s" % args.label
        if args.budget is not None:
            details += " budget=%d" % args.budget
        return "sample stdev of pair_scores (%s)" % details
    return "fallback upper bound for [0,1] pair scores"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2.5 検出力設計: 必要ペア数を決定論的に計算する")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="実用最小差(既定 0.05)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="対差分の標準偏差。省略時は records から推定し、"
                             "得られなければ 0.5 へ fallback")
    parser.add_argument("--records", default=None,
                        help="train/dev JSONL(行ごとに pair_scores を持つ)")
    parser.add_argument("--matchup", default=None,
                        help="records から選ぶ対戦(label または left-vs-right)")
    parser.add_argument("--label", default=None,
                        help="records から選ぶ label")
    parser.add_argument("--budget", type=int, default=None,
                        help="records から選ぶ budget(T)値")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="有意水準(既定 0.05)")
    parser.add_argument("--power", type=float, default=0.80,
                        help="検出力 1-beta(既定 0.80)")
    parser.add_argument("--sides", type=int, default=2, choices=(1, 2),
                        help="検定の片側/両側。両側 alpha=0.05 が保守基準")
    args = parser.parse_args(argv)

    sigma = args.sigma
    from_records = False
    if sigma is None:
        if args.records is not None:
            if (args.matchup is None or args.label is None
                    or args.budget is None):
                raise SystemExit(
                    "--records 使用時は --matchup / --label / --budget を"
                    "すべて指定してください(複数 config の曖昧な連結を防ぐ)")
            reason = classify_records(args.records, args.matchup, args.label,
                                      args.budget)
            if reason in ("corrupt", "invalid"):
                raise SystemExit(
                    "records が検査に不合格です(%s): %s"
                    % (reason, args.records))
            sigma = sigma_from_records(args.records, args.matchup, args.label,
                                       args.budget)
            from_records = sigma is not None
        if sigma is None:
            sigma = FALLBACK_SIGMA

    try:
        pairs = required_pairs(args.delta, sigma, args.alpha, args.power,
                               args.sides)
    except ValueError as exc:
        raise SystemExit(str(exc))

    side = "one-sided" if args.sides == 1 else "two-sided"
    print("sigma=%.4f  source: %s" % (sigma, _source_line(sigma, args,
                                                          from_records)))
    print("delta=%.4f  alpha=%.4f (%s)  power=%.4f"
          % (args.delta, args.alpha, side, args.power))
    print("z_alpha=%.4f  z_beta=%.4f"
          % (normal_quantile(1.0 - args.alpha / float(args.sides)),
             normal_quantile(args.power)))
    print("required_pairs=%d" % pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
