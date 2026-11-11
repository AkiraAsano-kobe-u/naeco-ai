"""Phase 2.5 段階2の記録から、事前登録した採用基準で方式を決定する。

`docs/eval-seeds.md`「判定宣言: Phase 2.5」の状態機械をそのまま実装したものである。
**実行(compare_search.py)と判定(本スクリプト)を分ける**ことで、結果を見てから
基準を動かせないようにする。判定前に入力記録を厳格検証する:
required 3 label の一意存在、pair_scores 個数、details 2*pairs、重複なし、
config/manifest/run_id 一致、details からの W/D/L・point・pair_scores 再計算、
ペア内 seed/初期 state 整合。旧形式は `--legacy` を明示したときだけ読む。
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import search_records  # noqa: E402
from search_records import (  # noqa: E402
    REQUIRED_LABELS, SearchRecordError, load_validated_bundles,
    validate_legacy_records,
)

ALPHA = 0.05        # 片側
NULL = 0.5          # 席交換ペアなので探索なし同士の期待勝率は厳密に 0.5

PROTOCOLS = {
    "phase2.5-dev": {
        "group": "dev", "start_index": 6000, "pairs": 785,
        "budget": 4500, "budget_seconds": None,
        "horizon": 2, "scale": 30, "c": 1.41,
        "python_version_prefix": "3.9.",
        "python_hash_seed": "0",
        "python_executable": "/usr/bin/python3",
    },
    "phase2.5-wallclock": {
        "group": "train", "start_index": 1100, "pairs": 200,
        "budget": 1000000, "budget_seconds": 2.0,
        "horizon": 2, "scale": 30, "c": 1.41,
        "python_version_prefix": "3.9.",
        "python_hash_seed": "0",
        "python_executable": "/usr/bin/python3",
    },
}
PROTOCOL_KEYS = ("group", "start_index", "pairs", "budget", "budget_seconds",
                 "horizon", "scale", "c")


def _sign_counts(pair_scores):
    """ペアスコアを H0=0.5 基準で勝ち/負け/引き分けに数える。"""
    wins = losses = draws = 0
    for score in pair_scores:
        if score > NULL:
            wins += 1
        elif score < NULL:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


def one_sided_p(pair_scores):
    """H0: ペア勝率 = 0.5、H1: > 0.5 の片側 p 値(exact 符号検定)。

    採用理由: pair_scores は {0, 0.5, 1} の離散値で、H0 下では非引き分け
    ペアの勝敗が公平コインになる。このため条件付き符号検定
    X ~ Binomial(n, 1/2)(n = 勝ち数 + 負け数)が exact かつ決定論的で、
    B=10,000 bootstrap の 1e-4 下限に張り付かない。
    引き分け(0.5)ペアは H0 の下で勝敗と独立で情報を持たないため、n から
    除外する(標準的な符号検定の引き分け処理)。
    """
    wins, losses, unused_draws = _sign_counts(pair_scores)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return tail / float(1 << n)


def holm_two(p_values):
    """2仮説の Holm 手続き。棄却された仮説のキー集合を返す。"""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rejected = set()
    if ordered[0][1] <= ALPHA / 2:
        rejected.add(ordered[0][0])
        if ordered[1][1] <= ALPHA:
            rejected.add(ordered[1][0])
    return rejected


def _fmt_p(p):
    """exact 値を bootstrap の分解能下限(1e-4)と誤認させない出力整形。"""
    if p <= 0.0:
        return "0"
    if p < 1e-4:
        return "%.3e" % p
    return "%.5f" % p


def _unique_required_labels(rows):
    labels = [row.get("label") for row in rows]
    if any(label is None for label in labels):
        raise SearchRecordError("summary row lacks label")
    duplicates = sorted(set(label for label in labels
                            if labels.count(label) > 1))
    if duplicates:
        raise SearchRecordError(
            "duplicate labels must not be overwritten: %r" % duplicates)
    found = set(labels)
    if found != set(REQUIRED_LABELS):
        raise SearchRecordError(
            "required unique labels %r not exactly present: %r"
            % (REQUIRED_LABELS, sorted(found)))
    by_label = dict((row["label"], row) for row in rows)
    return [by_label[label] for label in REQUIRED_LABELS]


def load_records(records, legacy=False, protocol=None):
    """記録を検証して required 3 label の summary 行を返す。"""
    if legacy and protocol is not None:
        raise SearchRecordError("--legacy と --protocol は併用できません")
    if legacy:
        validate_legacy_records(records)
        with open(records, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = _unique_required_labels(rows)
    else:
        if protocol not in PROTOCOLS:
            raise SearchRecordError(
                "新形式では --protocol が必須: %s"
                % (sorted(PROTOCOLS),))
        bundles = load_validated_bundles(records)
        rows = _unique_required_labels(
            [bundle["summary"] for bundle in bundles.values()])
        _check_protocol(rows, PROTOCOLS[protocol])
    _check_shared_config(rows, legacy=legacy)
    return rows


def _check_protocol(rows, expected):
    """事前登録 protocol と records の設定を厳格に照合する。"""
    for row in rows:
        for key in PROTOCOL_KEYS:
            value = row.get(key)
            if value != expected[key]:
                raise SearchRecordError(
                    "protocol mismatch on %s: %r != %r"
                    % (key, value, expected[key]))
        python_version = row.get("manifest", {}).get("python_version")
        prefix = expected.get("python_version_prefix")
        if prefix and (not isinstance(python_version, str)
                       or not python_version.startswith(prefix)):
            raise SearchRecordError(
                "protocol requires python_version %r, got %r"
                % (prefix, python_version))
        hash_seed = row.get("manifest", {}).get("python_hash_seed")
        expected_seed = expected.get("python_hash_seed")
        if expected_seed is not None and hash_seed != expected_seed:
            raise SearchRecordError(
                "protocol requires python_hash_seed %r, got %r"
                % (expected_seed, hash_seed))
        executable = row.get("manifest", {}).get("python_executable")
        expected_exe = expected.get("python_executable")
        if expected_exe is not None and executable != expected_exe:
            raise SearchRecordError(
                "protocol requires python_executable %r, got %r"
                % (expected_exe, executable))


def _check_shared_config(rows, legacy=False):
    """3 label 間で共通すべき設定・版・source_digest を検証する。

    legacy は top-level の版・source_digest を比較する。新形式は summary の
    `manifest` 内の値だけが正であり、top-level の None を比較しない。
    対戦固有の left/right/key/label/run_id と、環境・worker 数に依存する
    workers/thread_env/git_revision などは比較対象から除外する。
    """
    config_keys = ("group", "start_index", "pairs", "budget", "budget_seconds",
                   "horizon", "scale", "c")
    manifest_keys = ("cards_sha256", "cards_data_version", "api_version",
                     "rules_version", "obs_version", "source_digest",
                     "python_version", "python_executable",
                     "python_hash_seed")
    baseline = {}
    for key in config_keys + manifest_keys:
        if legacy:
            baseline[key] = rows[0].get(key)
        else:
            baseline[key] = rows[0].get("manifest", {}).get(key)
    for row in rows[1:]:
        for key in config_keys + manifest_keys:
            if legacy:
                value = row.get(key)
            else:
                value = row.get("manifest", {}).get(key)
            if value != baseline[key]:
                raise SearchRecordError(
                    "labels disagree on %s: %r != %r"
                    % (key, value, baseline[key]))


def decide(rows):
    """事前登録した状態機械で採用方式を決める。"""
    by_label = dict((row["label"], row) for row in rows)
    required = REQUIRED_LABELS
    missing = [label for label in required if label not in by_label]
    if missing:
        raise SystemExit("記録が足りない: %r" % (missing,))

    primary = dict(
        (name, one_sided_p(by_label["%s-vs-heuristic" % name]["pair_scores"]))
        for name in ("pimc", "ismcts"))
    rejected = holm_two(primary)
    trace = ["検定: exact 符号検定(条件付き二項)。引き分け(0.5)ペアは非情報として除外。"
             "bootstrap B=10000 の下限 1e-4 は使用しない。",
             "primary p値: pimc=%s ismcts=%s (片側 α=%.2f, Holm)"
             % (_fmt_p(primary["pimc"]), _fmt_p(primary["ismcts"]), ALPHA),
             "棄却: %s" % (sorted(rejected) or "なし",)]

    if not rejected:
        return "none", trace + ["→ この計算予算では探索の優位を示せなかった"]
    if len(rejected) == 1:
        winner = sorted(rejected)[0]
        return winner, trace + ["→ 採用: %s" % winner]

    # ③は**両側** α で判定する(sol SOL-P25G-004)。
    # 両方向の片側検定をそれぞれ α で試すと、方向を事前固定していない以上、
    # 第一種過誤が最大 2α(= 0.10)まで膨らむ。min を2倍して両側化する。
    head = by_label["pimc-vs-ismcts"]
    forward = one_sided_p(head["pair_scores"])
    reverse = one_sided_p([1.0 - score for score in head["pair_scores"]])
    two_sided = min(1.0, 2.0 * min(forward, reverse))
    trace.append("secondary(pimc vs ismcts) 片側 p: 順方向=%s 逆方向=%s "
                 "→ 両側 p=%s (α=%.2f) 点推定=%.4f"
                 % (_fmt_p(forward), _fmt_p(reverse), _fmt_p(two_sided),
                    ALPHA, head["point_estimate"]))
    if two_sided <= ALPHA:
        if head["point_estimate"] > NULL:
            return "pimc", trace + ["→ 直接対決で有意差あり → 採用: pimc"]
        if head["point_estimate"] < NULL:
            return "ismcts", trace + ["→ 直接対決で有意差あり → 採用: ismcts"]
        trace.append("→ 直接対決は有意だが point_estimate=0.5 のため方向を"
                     "確定せずタイブレークへ")
    else:
        trace.append("→ 直接対決は不確定。タイブレークへ")
    for key, label in (("p95_think_ms", "p95 思考時間"), ("peak_rss_bytes", "ピークRSS")):
        left = by_label["pimc-vs-heuristic"].get(key)
        right = by_label["ismcts-vs-heuristic"].get(key)
        if left is None or right is None or left == right:
            trace.append("  %s: 同値または未計測 → 次へ" % label)
            continue
        winner = "pimc" if left < right else "ismcts"
        return winner, trace + ["  %s: pimc=%s ismcts=%s → 採用: %s"
                                % (label, left, right, winner)]
    return "pimc", trace + ["  すべて同値 → 事前固定の既定順により 採用: pimc"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 2.5 方式選定の判定")
    parser.add_argument("records")
    parser.add_argument("--legacy", action="store_true",
                        help="bundle なしの旧 JSONL 形式を読む(歴史的成果物のみ)")
    parser.add_argument("--protocol",
                        choices=("phase2.5-dev", "phase2.5-wallclock",
                                 ), default=None,
                        help="新形式の事前登録 protocol。phase2.5-dev は "
                             "dev/start6000/pairs785/budget4500、"
                             "phase2.5-wallclock は train/start1100/pairs200/"
                             "budget1000000/seconds2.0")
    args = parser.parse_args(argv)
    if args.legacy and args.protocol is not None:
        raise SystemExit("--legacy と --protocol は併用できません")
    if not args.legacy and args.protocol is None:
        raise SystemExit(
            "新形式では --protocol が必須: phase2.5-dev / phase2.5-wallclock")
    rows = load_records(args.records, legacy=args.legacy,
                        protocol=args.protocol)
    for row in rows:
        print("%-22s 勝率=%.4f CI=[%.4f,%.4f] 勝/分/敗=%d/%d/%d"
              % (row["label"], row["point_estimate"], row["ci_low"],
                 row["ci_high"], row["wins"], row["draws"], row["losses"]))
    print()
    choice, trace = decide(rows)
    for line in trace:
        print(line)
    print()
    print("採用方式: %s" % choice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
