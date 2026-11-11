#!/usr/bin/env python3
"""昇格・退行判定 CLI(Phase 3 設計書 §7 / docs/eval-seeds.md Phase 3 判定宣言)。

    python scripts/promote.py \
        --conditions runs/phase3/promote-conditions.json \
        --challenger runs/phase3/gen011.meta.json \
        --champion runs/phase3/gen010.meta.json \
        --out docs/eval-runs/phase3-promote-1.json \
        --records docs/eval-runs/phase3-promote-dev-7800-8599 \
        --workers 4 --execute

- **昇格**(§7.1、次の両方を満たすときだけ):
  1. 現行チャンピオンとの直接対戦(dev、800ペア、席交換)で 95%CI 下限 > 0.50
     (**厳密不等号**。`>=` にしない。`harness.passes_threshold` 再利用)
  2. 固定ベンチマーク群(ヒューリスティック + 指定過去世代)への勝率点推定が、
     チャンピオンの記録値 − 3pt 以上(エントリごとに判定)
- **退行**(§7.2): 固定ベンチマーク勝率がチャンピオン比 5pt **超**低下
  → ロールバック。
- **判定条件の固定**: 相手・混合比率・対局 seed・ペア数・閾値・チャンピオン
  記録は `--conditions` の JSON で**実行前に読み込んで固定**する。実行後に
  ファイルが変更されたことは sha256 の照合で検出し、その場合は判定を出さずに
  失敗する。
- **モデルの固定(§18.2)**: 判定開始時に challenger / champion / ベンチの
  過去世代(checkpoint エントリ)の meta と weights を SHA-256 で凍結し、
  判定終了時に再検査する。1 つでも置換されていたら RuntimeError で
  fail-closed(判定を出さない)。凍結値は出力 JSON の `frozen_models` に
  記録される。
- **正式条件(§18.2)**: 既定モードでは正式条件(direct dev 7800-8599 /
  benchmark dev 9000-9799 / 各 800 ペア / promote 0.50 / 0.03 / 0.05)を
  強制し、逸脱は拒否する。逸脱は明示的な `--allow-informal` でのみ許され、
  その場合の出力は `formal: false` となり「非正式」と明記される。
- **seed 分離**(§7.3): 判定は `dev` seed のみ。`train` / `accept` は拒否
  (学習用プールは `train` seed で別経路)。
- **対局と統計は `eval_agent` 経由**: チェックポイント読込は必ず
  `checkpoint.load(..., expected=checkpoint.expected_meta(...))`(§15.1)を
  渡し、model_arch は eval_agent の arch 先読みの厳格検証(キー集合の完全
  一致と int 型の強制。§17.1)と重み state_dict とのキー集合・形状照合
  (§15.2)で検証される。昇格・退行の評価も同じ経路を使うため、型偽装・
  余分なキー・キー欠落・形状矛盾のあるチェックポイントは拒否される。統計は `harness` の `cluster_bootstrap` / `summarize_pairs` /
  `passes_threshold` を再利用する。
- torch は直接 import しない。`--execute` を付けない限り実行しない。
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import platform
import sys
import time

# 数値ライブラリを読むより前に固定し、spawn した worker へも同じ環境を渡す。
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_agent  # noqa: E402
from naeco.ai.eval.harness import (  # noqa: E402
    passes_threshold, write_game_records,
)
from naeco.engine import state as state_mod  # noqa: E402
from naeco.engine.cards import load_cards  # noqa: E402
from naeco.engine.observe import OBS_VERSION  # noqa: E402
from naeco.train import encoding as enc  # noqa: E402
from naeco.train import reward as reward_mod  # noqa: E402

DECISION_SCHEMA_VERSION = "phase3-promote/v1"

# 正式条件(docs/eval-seeds.md Phase 3 予約 / 設計書 §18.2)。変更禁止。
FORMAL_DIRECT_START_INDEX = 7800     # dev 7800-8599(昇格判定 第1回)
FORMAL_BENCH_START_INDEX = 9000      # dev 9000-9799(固定ベンチマーク群)
# 昇格判定 第2回以降の予約帯(docs/eval-seeds.md:1296 / :1326)。
# 台帳は「`dev` 10800 以降の 800 刻み」を第2回以降として予約している。
# 帯を 7800 に固定したままだと、台帳を守る限り**2回目の正式昇格が実行できない**
# (消費済み帯を使うか台帳に違反するかの二択になる。Sol ゲート3回目の指摘)。
#
# **緩めているのは帯だけである。** ペア数・閾値・許容差・ベンチマーク帯は
# 凍結したままなので、「結果を見てから条件を有利に変える」余地は生まれない。
# 選べるのは台帳が**事前に**予約した並びの1つだけで、任意の index は拒否する。
FORMAL_DIRECT_SECOND_ROUND_START = 10800
FORMAL_DIRECT_ROUND_STRIDE = 800
# 台帳の正本。`promote.py` はここから消費済み帯を読み、**次の未消費の予約帯**
# だけを正式条件として認める。予約列に載っているだけでは足りない
# (載っている帯を全部同時に認めると、飛ばし・再利用・事後選択を防げない。
#  Sol ゲート4回目の指摘)。
SEED_LEDGER_PATH = os.path.join(REPO_ROOT, "docs", "eval-seeds.md")
_LEDGER_BEGIN = "<!-- CONSUMED-DEV-BANDS:BEGIN -->"
_LEDGER_END = "<!-- CONSUMED-DEV-BANDS:END -->"
# 予約列を探す上限。これを超えても未消費帯が見つからないのは台帳の異常。
_MAX_RESERVED_ROUNDS = 128
FORMAL_PAIRS = 800
FORMAL_PROMOTE_THRESHOLD = 0.50
FORMAL_PROMOTE_TOLERANCE = 0.03
FORMAL_REGRESSION_TOLERANCE = 0.05

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)


@dataclasses.dataclass(frozen=True)
class Conditions:
    """実行前に読み込んで固定する判定条件(§7.1 / §7.2)。"""

    direct_group: str
    direct_start_index: int
    direct_pairs: int
    bench_group: str
    bench_start_index: int
    bench_pairs: int
    promote_threshold: float
    promote_tolerance: float
    regression_tolerance: float
    benchmark_entries: tuple          # (name, kind, path) のタプル
    champion_benchmark: dict          # name -> 点推定(チャンピオンの記録値)
    raw: dict                         # 元の JSON(出力へ埋め込む)


def _read_conditions_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _conditions_sha256(data):
    return hashlib.sha256(data).hexdigest()


def _validate_group(block, label):
    group = block.get("group")
    if group != "dev":
        raise ValueError(
            "%s.group must be 'dev' (judgment uses the dev seed pool; "
            "train is reserved for learning, accept for final acceptance): "
            "{!r}".format(group))
    return group


def _validate_index(block, label):
    value = block.get("start_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            "%s.start_index must be a non-negative integer, got {!r}"
            .format(label, value))
    return value


def _validate_pairs(block, label):
    value = block.get("pairs")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(
            "%s.pairs must be a positive integer, got {!r}".format(label, value))
    return value


def _validate_conditions(parsed):
    """条件 JSON を検証し、frozen な Conditions を返す。"""
    if not isinstance(parsed, dict):
        raise ValueError("conditions must be a JSON object")
    direct = parsed.get("direct")
    benchmark = parsed.get("benchmark")
    if not isinstance(direct, dict) or not isinstance(benchmark, dict):
        raise ValueError("conditions requires 'direct' and 'benchmark' objects")
    direct_group = _validate_group(direct, "direct")
    direct_start_index = _validate_index(direct, "direct")
    direct_pairs = _validate_pairs(direct, "direct")
    bench_group = _validate_group(benchmark, "benchmark")
    bench_start_index = _validate_index(benchmark, "benchmark")
    bench_pairs = _validate_pairs(benchmark, "benchmark")

    threshold = parsed.get("promote_threshold", 0.50)
    promote_tolerance = parsed.get("promote_tolerance", 0.03)
    regression_tolerance = parsed.get("regression_tolerance", 0.05)
    for name, value in (
            ("promote_threshold", threshold),
            ("promote_tolerance", promote_tolerance),
            ("regression_tolerance", regression_tolerance)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(
                "{} must be a positive finite number, got {!r}".format(
                    name, value))
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            "promote_threshold must be in (0, 1), got %g" % threshold)
    if regression_tolerance <= promote_tolerance:
        raise ValueError(
            "regression_tolerance (%g) must be greater than "
            "promote_tolerance (%g)" % (regression_tolerance,
                                        promote_tolerance))

    entries = parsed.get("benchmark_entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmark_entries must be a non-empty list")
    names = []
    normalized = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                "benchmark_entries[%d] must be an object" % i)
        name = entry.get("name")
        kind = entry.get("kind")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "benchmark_entries[%d].name must be a non-empty string" % i)
        if name in names:
            raise ValueError("duplicate benchmark entry name: %s" % name)
        if kind == "heuristic":
            path = ""
        elif kind == "checkpoint":
            path = entry.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(
                    "checkpoint benchmark entry %s requires a path" % name)
        else:
            raise ValueError(
                "benchmark entry %s has unknown kind {!r}".format(name, kind))
        names.append(name)
        normalized.append((name, kind, path))

    champion = parsed.get("champion_benchmark")
    if not isinstance(champion, dict):
        raise ValueError("champion_benchmark must be an object")
    if not champion:
        raise ValueError(
            "champion_benchmark must be a non-empty object: "
            "昇格条件②と退行判定の比較対象(チャンピオンの記録値)が無い")
    # エントリと記録値の一対一対応: 欠落も未知キーもこの集合一致検査で弾く。
    if set(champion) != set(names):
        raise ValueError(
            "champion_benchmark keys must equal benchmark entry names: "
            "{} vs {}".format(sorted(champion), sorted(names)))
    champion_record = {}
    for name, value in champion.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)):
            raise ValueError(
                "champion_benchmark[%s] must be a finite number" % name)
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                "champion_benchmark[%s] must be a win rate in [0, 1]" % name)
        champion_record[name] = float(value)

    return Conditions(
        direct_group=direct_group, direct_start_index=direct_start_index,
        direct_pairs=direct_pairs, bench_group=bench_group,
        bench_start_index=bench_start_index, bench_pairs=bench_pairs,
        promote_threshold=float(threshold),
        promote_tolerance=float(promote_tolerance),
        regression_tolerance=float(regression_tolerance),
        benchmark_entries=tuple(normalized),
        champion_benchmark=champion_record,
        raw=parsed,
    )


def _is_reserved_direct_band(index):
    """direct 帯が台帳の予約列にあるか(docs/eval-seeds.md)。

    第1回は dev 7800、第2回以降は dev 10800 から 800 刻み。
    **これは必要条件でしかない。** 予約列に載っている帯を全部認めると、
    消費済みの帯を選び直したり、次を飛ばして先の帯を使ったりできてしまう。
    正式条件は `_next_reserved_direct_band()` との一致を要求する。
    """
    if index == FORMAL_DIRECT_START_INDEX:
        return True
    if index < FORMAL_DIRECT_SECOND_ROUND_START:
        return False
    return (index - FORMAL_DIRECT_SECOND_ROUND_START) \
        % FORMAL_DIRECT_ROUND_STRIDE == 0


def _parse_consumed_dev_bands(text):
    """台帳の機械可読ブロックから消費済み dev 帯を読む。

    ブロックが無い / 空 / 解析できないときは ValueError で**実行を拒否する**
    (fail-closed)。黙って「消費済みは無い」と解釈すると、消費済みの帯を
    正式条件として通してしまう。
    """
    start = text.find(_LEDGER_BEGIN)
    end = text.find(_LEDGER_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            "seed 台帳に %s / %s のブロックが無い(%s)。"
            "昇格判定は台帳との突合なしには実行できない。"
            % (_LEDGER_BEGIN, _LEDGER_END, SEED_LEDGER_PATH))
    body = text[start + len(_LEDGER_BEGIN):end]
    bands = []
    for lo, hi in re.findall(r"(\d+)\s*-\s*(\d+)", body):
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise ValueError("seed 台帳の帯が逆順である: %d-%d" % (lo, hi))
        bands.append((lo, hi))
    if not bands:
        raise ValueError(
            "seed 台帳の消費済みブロックが空である(%s)。" % SEED_LEDGER_PATH)
    return bands


def _load_consumed_dev_bands(path=None):
    """台帳ファイルを読んで消費済み dev 帯を返す。"""
    path = path or SEED_LEDGER_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ValueError("seed 台帳を読めない(%s): %s" % (path, exc))
    return _parse_consumed_dev_bands(text)


def _next_reserved_direct_band(consumed, pairs=FORMAL_PAIRS):
    """予約列のうち、台帳でまだ消費されていない**最初の**帯を返す。

    これが正式条件で使える唯一の direct 帯である。次を飛ばすことも、
    消費済みへ戻ることもできない。
    """
    candidates = [FORMAL_DIRECT_START_INDEX] + [
        FORMAL_DIRECT_SECOND_ROUND_START + k * FORMAL_DIRECT_ROUND_STRIDE
        for k in range(_MAX_RESERVED_ROUNDS)
    ]
    for start in candidates:
        stop = start + pairs - 1
        if not any(lo <= stop and start <= hi for lo, hi in consumed):
            return start
    raise ValueError(
        "予約列 %d 個を調べたが未消費の帯が無い。台帳を確認すること。"
        % len(candidates))


def _seed_ledger_record(ledger_path=None):
    """判定 JSON に残す台帳の証跡(パス・sha256・次の未消費帯)。"""
    path = ledger_path or SEED_LEDGER_PATH
    with open(path, "rb") as fh:
        raw = fh.read()
    consumed = _parse_consumed_dev_bands(raw.decode("utf-8"))
    return {
        "path": os.path.relpath(path, REPO_ROOT),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "consumed_dev_bands": len(consumed),
        "expected_direct_start_index":
            _next_reserved_direct_band(consumed, FORMAL_PAIRS),
    }


def _conditions_are_formal(conditions, ledger_path=None):
    """条件が正式条件(設計書 §18.2)と一致するかを返す。

    正式条件は: direct が **台帳でまだ消費されていない最初の予約帯**の 800 ペア /
    benchmark が dev 9000-9799 の 800 ペア / promote_threshold 0.50 /
    promote_tolerance 0.03 / regression_tolerance 0.05(docs/eval-seeds.md)。

    台帳が読めないときは例外を投げる(fail-closed)。「読めなかったから
    予約列だけで判断する」は、消費済み帯を通す穴になる。
    """
    expected_direct = _next_reserved_direct_band(
        _load_consumed_dev_bands(ledger_path), FORMAL_PAIRS)
    return (
        conditions.direct_group == "dev"
        and conditions.direct_start_index == expected_direct
        and conditions.direct_pairs == FORMAL_PAIRS
        and conditions.bench_group == "dev"
        and conditions.bench_start_index == FORMAL_BENCH_START_INDEX
        and conditions.bench_pairs == FORMAL_PAIRS
        and conditions.promote_threshold == FORMAL_PROMOTE_THRESHOLD
        and conditions.promote_tolerance == FORMAL_PROMOTE_TOLERANCE
        and conditions.regression_tolerance == FORMAL_REGRESSION_TOLERANCE
    )


def _require_formal_conditions(conditions, ledger_path=None):
    """正式条件からの逸脱を検出し、逸脱があれば ValueError(設計書 §18.2)。

    既定(正式)モードでは逸脱を拒否する。逸脱は明示的な --allow-informal
    フラグでのみ許され、その場合は出力に「非正式」と明記される。
    """
    deviations = []
    expected_direct = _next_reserved_direct_band(
        _load_consumed_dev_bands(ledger_path), FORMAL_PAIRS)
    if conditions.direct_group != "dev" or \
            conditions.direct_start_index != expected_direct or \
            conditions.direct_pairs != FORMAL_PAIRS:
        deviations.append(
            "direct must be dev %d-%d (800 pairs) — これは docs/eval-seeds.md の "
            "消費済みブロックから求めた**次の未消費の予約帯**である。"
            "got group=%s start_index=%d pairs=%d。"
            "帯を飛ばすことも、消費済みへ戻ることもできない。"
            "帯を進めるには、使った帯を先に台帳へ追記すること"
            % (expected_direct, expected_direct + FORMAL_PAIRS - 1,
               conditions.direct_group, conditions.direct_start_index,
               conditions.direct_pairs))
    if conditions.bench_group != "dev" or \
            conditions.bench_start_index != FORMAL_BENCH_START_INDEX or \
            conditions.bench_pairs != FORMAL_PAIRS:
        deviations.append(
            "benchmark must be dev %d-%d (800 pairs), got group=%s "
            "start_index=%d pairs=%d"
            % (FORMAL_BENCH_START_INDEX,
               FORMAL_BENCH_START_INDEX + FORMAL_PAIRS - 1,
               conditions.bench_group, conditions.bench_start_index,
               conditions.bench_pairs))
    if conditions.promote_threshold != FORMAL_PROMOTE_THRESHOLD:
        deviations.append(
            "promote_threshold must be %g, got %g"
            % (FORMAL_PROMOTE_THRESHOLD, conditions.promote_threshold))
    if conditions.promote_tolerance != FORMAL_PROMOTE_TOLERANCE:
        deviations.append(
            "promote_tolerance must be %g, got %g"
            % (FORMAL_PROMOTE_TOLERANCE, conditions.promote_tolerance))
    if conditions.regression_tolerance != FORMAL_REGRESSION_TOLERANCE:
        deviations.append(
            "regression_tolerance must be %g, got %g"
            % (FORMAL_REGRESSION_TOLERANCE,
               conditions.regression_tolerance))
    if deviations:
        raise ValueError(
            "formal judgment conditions violated (docs/eval-seeds.md Phase 3 "
            "予約): %s. 逸脱は --allow-informal でのみ許される。"
            % "; ".join(deviations))


def _load_conditions(path, allow_informal=False, ledger_path=None):
    """判定条件を実行前に読み込んで固定する(sha256 込みで返す)。

    allow_informal=False(既定)では正式条件(設計書 §18.2)を強制し、逸脱は
    ValueError で拒否する。逸脱を許すのは明示的な --allow-informal フラグだけ。

    ledger_path はテスト用の差し替え口。既定は docs/eval-seeds.md。
    """
    data = _read_conditions_bytes(path)
    digest = _conditions_sha256(data)
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(
            "conditions {} is not valid UTF-8 JSON: {}".format(path, exc))
    conditions = _validate_conditions(parsed)
    if not allow_informal:
        _require_formal_conditions(conditions, ledger_path)
    return conditions, digest


def _verify_conditions_unchanged(path, expected_sha256):
    """実行後に条件ファイルが変更されていないことを確認する。"""
    current = _conditions_sha256(_read_conditions_bytes(path))
    if current != expected_sha256:
        raise RuntimeError(
            "conditions file changed during the run: {} (sha256 {} -> {}). "
            "判定条件は実行前に読み込んで固定する契約のため、判定を出さずに中止する。"
            .format(path, expected_sha256, current))


def _file_sha256(path, label):
    """ファイルの SHA-256 を返す。無ければ RuntimeError(fail-closed)。"""
    if not os.path.isfile(path):
        raise RuntimeError(
            "model file missing: %s (%s). 判定を出さずに中止する。"
            % (path, label))
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _freeze_model_files(conditions, challenger_meta, champion_meta):
    """判定開始時に、昇格対象となる全モデルの meta / weights を凍結する。

    対象は challenger / champion / ベンチマークの過去世代(checkpoint エント
    リ)。heuristic エントリはモデルを持たないため対象外。eval_agent は対局
    タスクごとにチェックポイントを再読込するため、評価中に同じパスの重みが
    置換されると複数モデルの結果が混在しても検出できない。これを判定終了時
    の再検査で検出できるよう、実行前に SHA-256 を取得して固定する
    (設計書 §18.2)。
    """
    frozen = {}
    for label, meta_path in (("challenger", challenger_meta),
                             ("champion", champion_meta)):
        weights_path = eval_agent.checkpoint_weights_path(meta_path)
        frozen[label] = {
            "meta_path": meta_path,
            "weights_path": weights_path,
            "meta": _file_sha256(meta_path, label + " meta"),
            "weights": _file_sha256(weights_path, label + " weights"),
        }
    for name, kind, path in conditions.benchmark_entries:
        if kind != "checkpoint":
            continue
        label = "bench:" + name
        weights_path = eval_agent.checkpoint_weights_path(path)
        frozen[label] = {
            "meta_path": path,
            "weights_path": weights_path,
            "meta": _file_sha256(path, label + " meta"),
            "weights": _file_sha256(weights_path, label + " weights"),
        }
    return frozen


def _verify_models_unchanged(frozen):
    """判定終了時に、凍結したモデルが途中で置換されていないことを再検査する。

    1つでも SHA-256 が変わっていたら RuntimeError で失敗させる(fail-closed。
    判定を出さずに中止)。eval_agent が対局タスクごとにチェックポイントを再
    読込するため、重みが置換されると複数モデルの結果が混在したまま検出でき
    ない。この再検査が無い実装へ戻すとテストが落ちる。
    """
    for label, files in frozen.items():
        for file_kind in ("meta", "weights"):
            path = files[file_kind + "_path"]
            expected = files[file_kind]
            current = _file_sha256(path, label + " " + file_kind)
            if current != expected:
                raise RuntimeError(
                    "model file changed during the run: %s (%s, sha256 %s -> "
                    "%s). 判定中にモデルが置換されたため、判定を出さずに中止する。"
                    % (path, label, expected, current))


def benchmark_floor_ok(challenger_benchmark, champion_benchmark,
                       promote_tolerance):
    """昇格②: 全ベンチエントリで「チャンピオンの記録値 − 3pt 以上」。"""
    return all(
        challenger_benchmark[name] >= champion_benchmark[name] - promote_tolerance
        for name in champion_benchmark)


def regression_detected(challenger_benchmark, champion_benchmark,
                        regression_tolerance):
    """退行: いずれかのエントリでチャンピオン比 5pt 超の低下。"""
    return any(
        challenger_benchmark[name] < champion_benchmark[name] - regression_tolerance
        for name in champion_benchmark)


def decide(direct_pass, challenger_benchmark, champion_benchmark,
           promote_tolerance, regression_tolerance):
    """昇格・退行の判定。戻り値は "promote" / "reject" / "rollback"。

    - "rollback" は退行条件(5pt 超低下)が少しでも検出されたとき最優先。
    - "promote" は直接対戦の合格(CI 下限 > 閾値の厳密不等号)と昇格②
      (全エントリで −3pt 以上)の**両方**を満たすときだけ。
    """
    missing = sorted(set(champion_benchmark) - set(challenger_benchmark))
    if missing:
        raise ValueError(
            "challenger benchmark missing entries: %s" % missing)
    if regression_detected(challenger_benchmark, champion_benchmark,
                           regression_tolerance):
        return "rollback"
    if direct_pass and benchmark_floor_ok(
            challenger_benchmark, champion_benchmark, promote_tolerance):
        return "promote"
    return "reject"


def _versions():
    cards = load_cards()
    return {
        "rules_version": state_mod.RULES_VERSION,
        "api_version": state_mod.API_VERSION,
        "obs_version": OBS_VERSION,
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "obs_encoder_version": enc.OBS_ENCODER_VERSION,
        "action_encoder_version": enc.ACTION_ENCODER_VERSION,
        "reward_version": reward_mod.REWARD_VERSION,
    }


class _RecordsHolder(object):
    """write_game_records に渡す最小の result 形(対応表の連結用)。"""

    def __init__(self, game_records):
        self.game_records = game_records


def _write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")


def run(args, conditions, conditions_sha256):
    ledger_path = getattr(args, "seed_ledger", None)
    """昇格・退行判定の実行本体。判定結果を --out へ書き、stdout へ要約する。"""
    challenger = "checkpoint:%s" % args.challenger
    champion = "checkpoint:%s" % args.champion

    # 判定開始時に全モデル(challenger / champion / ベンチの過去世代)の meta と
    # weights を SHA-256 で凍結し、判定終了時に再検査する(設計書 §18.2)。
    frozen_models = _freeze_model_files(
        conditions, args.challenger, args.champion)

    direct = eval_agent.run_evaluation(
        challenger, champion,
        conditions.direct_group, conditions.direct_start_index,
        conditions.direct_pairs, workers=args.workers)
    print("direct: %s vs %s group=%s index=%d..%d pairs=%d "
          "point=%.4f CI=[%.4f, %.4f] W/D/L=%d/%d/%d"
          % (challenger, champion, conditions.direct_group,
             conditions.direct_start_index,
             conditions.direct_start_index + conditions.direct_pairs - 1,
             conditions.direct_pairs, direct.point_estimate, direct.ci_low,
             direct.ci_high, direct.wins, direct.draws, direct.losses),
          flush=True)

    bench_results = []
    for name, kind, path in conditions.benchmark_entries:
        spec = "heuristic" if kind == "heuristic" else "checkpoint:%s" % path
        result = eval_agent.run_evaluation(
            challenger, spec,
            conditions.bench_group, conditions.bench_start_index,
            conditions.bench_pairs, workers=args.workers)
        bench_results.append((name, kind, path, spec, result))
        print("bench: %s vs %s group=%s index=%d..%d pairs=%d "
              "point=%.4f CI=[%.4f, %.4f] W/D/L=%d/%d/%d"
              % (challenger, spec, conditions.bench_group,
                 conditions.bench_start_index,
                 conditions.bench_start_index + conditions.bench_pairs - 1,
                 conditions.bench_pairs, result.point_estimate,
                 result.ci_low, result.ci_high, result.wins, result.draws,
                 result.losses),
              flush=True)

    _verify_conditions_unchanged(args.conditions, conditions_sha256)
    _verify_models_unchanged(frozen_models)

    direct_pass = passes_threshold(direct, conditions.promote_threshold)
    challenger_benchmark = dict(
        (name, result.point_estimate)
        for name, unused_kind, unused_path, unused_spec, result
        in bench_results)
    champion_benchmark = dict(conditions.champion_benchmark)
    decision = decide(
        direct_pass, challenger_benchmark, champion_benchmark,
        conditions.promote_tolerance, conditions.regression_tolerance)

    formal = _conditions_are_formal(conditions, ledger_path)
    if not formal:
        print("非正式: この判定は正式条件(direct dev %d-%d / benchmark dev "
              "%d-%d / 各800ペア / promote 0.50 / 0.03 / 0.05)から逸脱して"
              "おり、Phase 3 の正式な昇格・退行判定ではない。"
              % (FORMAL_DIRECT_START_INDEX,
                 FORMAL_DIRECT_START_INDEX + FORMAL_PAIRS - 1,
                 FORMAL_BENCH_START_INDEX,
                 FORMAL_BENCH_START_INDEX + FORMAL_PAIRS - 1),
              flush=True)

    records_direct = records_bench = None
    if args.records:
        records_direct = args.records + "-direct.jsonl"
        records_bench = args.records + "-bench.jsonl"
        for path in (records_direct, records_bench):
            parent = os.path.dirname(os.path.abspath(path))
            if parent and not os.path.isdir(parent):
                os.makedirs(parent)
        write_game_records(direct, records_direct)
        bench_records = []
        for name, unused_kind, unused_path, unused_spec, result \
                in bench_results:
            bench_records.extend(result.game_records)
        write_game_records(_RecordsHolder(bench_records), records_bench)
        print("records=%s (%d rows), %s (%d rows)"
              % (records_direct, len(direct.game_records), records_bench,
                 len(bench_records)))

    payload = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": conditions.raw,
        # どの台帳を使ったかを残す。--seed-ledger で差し替えても、
        # パスと sha256 が判定 JSON に載るので後から検証できる。
        "seed_ledger": _seed_ledger_record(ledger_path),
        "conditions_sha256": conditions_sha256,
        "formal": formal,
        "frozen_models": dict(
            (label,
             {"meta": files["meta"], "weights": files["weights"]})
            for label, files in sorted(frozen_models.items())),
        "challenger": challenger,
        "champion": champion,
        "versions": _versions(),
        "python_version": platform.python_version(),
        "thread_env": dict((name, os.environ.get(name))
                           for name in _THREAD_ENV_NAMES),
        "direct": {
            "left": challenger,
            "right": champion,
            "group": direct.group,
            "start_index": direct.start_index,
            "pairs": direct.pairs,
            "games": direct.games,
            "wins": direct.wins,
            "draws": direct.draws,
            "losses": direct.losses,
            "point_estimate": direct.point_estimate,
            "ci_low": direct.ci_low,
            "ci_high": direct.ci_high,
            "bootstrap_samples": direct.bootstrap_samples,
            "bootstrap_seed": direct.bootstrap_seed,
            "passes_threshold": direct_pass,
        },
        "benchmark": {
            "champion_record": champion_benchmark,
            "challenger_record": challenger_benchmark,
            "promote_tolerance": conditions.promote_tolerance,
            "regression_tolerance": conditions.regression_tolerance,
            "promotion_floor_ok": benchmark_floor_ok(
                challenger_benchmark, champion_benchmark,
                conditions.promote_tolerance),
            "regression_detected": regression_detected(
                challenger_benchmark, champion_benchmark,
                conditions.regression_tolerance),
            "per_entry": [
                {
                    "name": name,
                    "kind": kind,
                    "path": path,
                    "left": challenger,
                    "right": spec,
                    "group": result.group,
                    "start_index": result.start_index,
                    "pairs": result.pairs,
                    "games": result.games,
                    "wins": result.wins,
                    "draws": result.draws,
                    "losses": result.losses,
                    "point_estimate": result.point_estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                }
                for name, kind, path, spec, result in bench_results
            ],
        },
        "decision": decision,
        "new_champion_record": challenger_benchmark,
        "records": {"direct": records_direct, "bench": records_bench},
    }
    _write_json(args.out, payload)
    print("decision: %s" % decision, flush=True)
    return decision


def run_multi_smoke(args):
    """D5 pipeline smoke on train only; never certifies or replaces champion."""
    from naeco.train import net as net_mod
    from naeco.train.agent import PolicyAgent
    from naeco.train.env import play_episode
    from naeco.ai.heuristic_agent import HeuristicAgent
    from naeco.ai.eval.harness import cluster_bootstrap
    n = args.num_players
    if not args.allow_informal:
        raise ValueError("multi smoke requires --allow-informal; formal evaluations use registered D4 bundles")
    raw = _read_conditions_bytes(args.conditions)
    conditions = json.loads(raw)
    if set(conditions) != {"schema", "num_players", "group", "start_index", "clusters"}:
        raise ValueError("invalid multi smoke conditions keys")
    if conditions["schema"] != "phase10-promote-smoke/v1" or conditions["num_players"] != n or conditions["group"] != "train":
        raise ValueError("multi smoke requires matching player count and train group")
    if type(conditions["start_index"]) is not int or conditions["start_index"] < 2000 or type(conditions["clusters"]) is not int or not 1 <= conditions["clusters"] <= 20:
        raise ValueError("invalid diagnostic range")
    frozen = {}
    for path in (args.challenger, args.champion):
        for target in (path, eval_agent.checkpoint_weights_path(path)):
            frozen[target] = _file_sha256(target, "model")
    challenger, _ = net_mod.load_checkpoint(args.challenger, n)
    champion, _ = net_mod.load_checkpoint(args.champion, n)
    if not args.execute:
        return {"formal": False, "conditions": conditions, "frozen_models": frozen}
    if os.path.exists(args.out):
        raise ValueError("output already exists")
    records = []
    rates = {}
    for label, target, opponent in (
            ("direct", challenger, champion),
            ("challenger_heuristic", challenger, None),
            ("champion_heuristic", champion, None),
            ("challenger_initial", challenger, champion),
            ("champion_initial", champion, champion)):
        clusters = []
        for index in range(conditions["start_index"], conditions["start_index"] + conditions["clusters"]):
            wins = []
            for variant in range(n):
                agents = [PolicyAgent(target, mode="greedy", seed=0)]
                agents.extend(HeuristicAgent(0) if opponent is None else
                              PolicyAgent(opponent, mode="greedy", seed=0)
                              for _ in range(n - 1))
                trajectory = play_episode("train", index, variant, agents,
                                          record_vectors=False, learner_index=0)
                if trajectory.discarded:
                    raise RuntimeError("discarded promotion smoke game")
                won = trajectory.terminal_rewards[trajectory.learner_seat] == 1.0
                wins.append(float(won))
                records.append({"label": label, "index": index, "k": variant,
                                "seats": trajectory.seats, "target_seat": trajectory.learner_seat,
                                "initial_hash": trajectory.initial_state_hash,
                                "utilities": trajectory.terminal_rewards, "target_win": won})
            clusters.append(sum(wins) / n)
        ci = cluster_bootstrap(clusters, 10000, 2026072102)
        rates[label] = {"win_rate": sum(clusters) / len(clusters), "ci": list(ci)}
    for path, expected in frozen.items():
        if _file_sha256(path, "model") != expected:
            raise RuntimeError("model changed during smoke")
    if _read_conditions_bytes(args.conditions) != raw:
        raise RuntimeError("conditions changed during smoke")
    floor = all(rates["challenger_" + label]["win_rate"] >=
                rates["champion_" + label]["win_rate"] - .03
                for label in ("heuristic", "initial"))
    rollback = any(rates["challenger_" + label]["win_rate"] <
                   rates["champion_" + label]["win_rate"] - .05
                   for label in ("heuristic", "initial"))
    result = {"schema": "phase10-promote-smoke/v1", "formal": False,
              "certified": False, "conditions": conditions,
              "conditions_sha256": _conditions_sha256(raw), "frozen_models": frozen,
              "rates": rates, "records": records,
              "diagnostic_decision": "rollback" if rollback else
              ("promote" if rates["direct"]["ci"][0] > 1.0 / n and floor else "reject")}
    _write_json(args.out, result)
    return result


# ---------------------------------------------------------------------------
# Phase 10 D5: 3・4 人戦の正式昇格(B-C6-fix t_4cf080a7)。
#
# 対戦は 候補 1 席 vs champion×(n−1)(direct)と、固定ベンチ
# [X, heuristic×(n−1)] / [X, 凍結初期 policy×(n−1)](X = champion / 候補)。
# すべて D4 の n 局巡回クラスタ(eval_agent.run_multi)で、帯は台帳
# (docs/eval-seeds.md の MULTI-SEAT 欄)の予約を経由し、各対戦は 1 回だけ起動
# できる(claim_multi が消費を記録し、重複起動・登録外の対戦・モデル差し替えを
# 拒否する)。判定:
#   rollback : いずれかのベンチで 候補 < champion − 5pt
#   promote  : direct の CI 下限 > 1/n かつ 全ベンチで 候補 ≥ champion − 3pt
#   reject   : それ以外
# 席のモデルは人数別 greedy(checkpoint:)。強(strong:)は最終 accept 用。
# n=2 の経路(上の run / _load_conditions)は一切変えない。
# ---------------------------------------------------------------------------

MULTI_D5_SCHEMA = "phase10-promote/v1"
MULTI_D5_DECISION_SCHEMA = "phase10-promote-decision/v1"
MULTI_D5_CLUSTERS = 800
MULTI_D5_PROMOTE_TOLERANCE = 0.03
MULTI_D5_REGRESSION_TOLERANCE = 0.05
MULTI_D5_BENCH_NAMES = ("heuristic", "initial")
MULTI_D5_STAGES = ("register", "bench", "promote")
INITIAL_CHAMPION_PATH = os.path.join(
    REPO_ROOT, "docs", "eval-runs", "phase10", "models", "initial-champion.json")
_D5_KEYS = ("schema", "num_players", "champion", "challenger", "bench",
            "direct", "promote_threshold", "promote_tolerance",
            "regression_tolerance", "out_dir")
_D5_BAND_KEYS = ("reservation", "group", "start_index", "clusters", "purpose")


def _d5_threshold(n):
    return 1.0 / n


def _read_initial_champion(n, path=None):
    """initial-champion.json の n 席エントリ(凍結初期 policy の SHA)。無ければ拒否。"""
    path = path or INITIAL_CHAMPION_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError("initial champion record unreadable (%s): %s"
                         % (path, exc))
    entry = (record.get("models") or {}).get(str(n))
    if not isinstance(entry, dict):
        raise ValueError("initial champion record has no %d-seat entry" % n)
    return entry


def _d5_bench_specs(conditions):
    """[(name, spec)]。heuristic と凍結初期 policy(greedy)。"""
    out = []
    for entry in conditions["bench"]["entries"]:
        if entry["kind"] == "heuristic":
            out.append((entry["name"], "heuristic"))
        else:
            out.append((entry["name"], "checkpoint:" + entry["path"]))
    return out


def _d5_matchups(conditions):
    """{"bench": [(label, agents)], "direct": [(label, agents)]}(正準 agent リスト)。"""
    n = conditions["num_players"]
    champion = "checkpoint:" + conditions["champion"]
    challenger = "checkpoint:" + conditions["challenger"]
    bench = []
    for role, target in (("champion", champion), ("challenger", challenger)):
        for name, spec in _d5_bench_specs(conditions):
            bench.append(("bench-%s-%s" % (role, name),
                          [target] + [spec] * (n - 1)))
    direct = [("direct", [challenger] + [champion] * (n - 1))]
    return {"bench": bench, "direct": direct}


def _validate_d5(parsed, allow_informal=False, ledger_path=None,
                 initial_record_path=None):
    """D5 条件を検査する。正式モードは D5/D4 の固定値と台帳の予約を強制する。"""
    from naeco.ai.eval import harness
    if not isinstance(parsed, dict) or set(parsed) != set(_D5_KEYS):
        raise ValueError("D5 conditions keys must be exactly %s"
                         % (sorted(_D5_KEYS),))
    if parsed["schema"] != MULTI_D5_SCHEMA:
        raise ValueError("schema must be %s" % MULTI_D5_SCHEMA)
    n = parsed["num_players"]
    if type(n) is not int or n not in harness.MULTI_PLAYER_COUNTS:
        raise ValueError("num_players must be 3 or 4")
    for label in ("champion", "challenger"):
        path = parsed[label]
        if not isinstance(path, str) or not path.endswith(".meta.json"):
            raise ValueError("%s must be a checkpoint .meta.json path" % label)
        harness.check_model_for_players(path, n)
        harness.model_file_shas(path)
    if os.path.abspath(parsed["champion"]) == \
            os.path.abspath(parsed["challenger"]):
        raise ValueError("challenger and champion must differ")
    for label in ("bench", "direct"):
        block = parsed[label]
        keys = set(_D5_BAND_KEYS) | ({"entries"} if label == "bench" else set())
        if not isinstance(block, dict) or set(block) != keys:
            raise ValueError("%s keys must be exactly %s" % (label,
                                                             sorted(keys)))
        if type(block["start_index"]) is not int or \
                type(block["clusters"]) is not int or block["clusters"] < 1:
            raise ValueError("%s band must be integers" % label)
    entries = parsed["bench"]["entries"]
    if not isinstance(entries, list) or \
            [e.get("name") for e in entries if isinstance(e, dict)] != \
            list(MULTI_D5_BENCH_NAMES):
        raise ValueError("bench entries must be exactly %s (D5: heuristic×(n−1)"
                         " and the frozen initial policy×(n−1))"
                         % (list(MULTI_D5_BENCH_NAMES),))
    if entries[0] != {"name": "heuristic", "kind": "heuristic"}:
        raise ValueError("bench entry 'heuristic' must be "
                         "{name: heuristic, kind: heuristic}")
    initial = entries[1]
    if set(initial) != {"name", "kind", "path"} or \
            initial["kind"] != "checkpoint":
        raise ValueError("bench entry 'initial' must be "
                         "{name, kind: checkpoint, path}")
    harness.check_model_for_players(initial["path"], n)
    threshold = parsed["promote_threshold"]
    if threshold != _d5_threshold(n):
        raise ValueError("promote_threshold must be 1/n = %r (D5), got %r"
                         % (_d5_threshold(n), threshold))
    if parsed["promote_tolerance"] != MULTI_D5_PROMOTE_TOLERANCE or \
            parsed["regression_tolerance"] != MULTI_D5_REGRESSION_TOLERANCE:
        raise ValueError("tolerances must be %g / %g (D5)"
                         % (MULTI_D5_PROMOTE_TOLERANCE,
                            MULTI_D5_REGRESSION_TOLERANCE))
    if not isinstance(parsed["out_dir"], str) or not parsed["out_dir"]:
        raise ValueError("out_dir must be a non-empty path")
    if allow_informal:
        return parsed
    # ---- 正式モード: dev・800 クラスタ・多人数 dev 領域・台帳の予約 ----
    shas = harness.model_file_shas(initial["path"])
    frozen = _read_initial_champion(n, initial_record_path)
    if shas != {"meta": frozen.get("meta_sha256"),
                "weights": frozen.get("weights_sha256")}:
        raise ValueError(
            "bench 'initial' %s is not the frozen initial champion recorded "
            "in initial-champion.json (%r vs %r)" % (initial["path"], shas,
                                                    frozen))
    region_lo, region_hi = eval_agent.MULTI_DEV_REGION
    unused, reserved, consumed, unused_two = eval_agent.read_multi_ledger(
        ledger_path)
    for label, want_threshold in (("bench", None), ("direct", threshold)):
        block = parsed[label]
        lo = block["start_index"]
        hi = lo + block["clusters"] - 1
        if block["group"] != "dev" or block["clusters"] != MULTI_D5_CLUSTERS:
            raise ValueError("%s must be dev with %d clusters (D4/D5)"
                             % (label, MULTI_D5_CLUSTERS))
        if not (region_lo <= lo and hi <= region_hi):
            raise ValueError("%s band %d-%d is outside the multi-seat dev "
                             "region %d-%d" % (label, lo, hi, region_lo,
                                               region_hi))
        matches = [e for e in reserved if e.get("id") == block["reservation"]]
        if len(matches) != 1:
            raise ValueError("%s reservation %r is not in the ledger"
                             % (label, block["reservation"]))
        res = matches[0]
        want = {"num_players": n, "group": "dev", "lo": lo, "hi": hi,
                "purpose": block["purpose"], "threshold": want_threshold}
        diffs = [k for k, v in want.items() if res.get(k) != v]
        if diffs:
            raise ValueError("%s conditions differ from ledger reservation "
                             "%s at %s" % (label, res["id"], diffs))
    # 昇格帯は「同領域内で連続未使用 800 帯を順に」: この人数の未消費の昇格予約の
    # うち最も若い帯でなければならない(飛ばし・事後選択の拒否)。
    pattern = re.compile(r"^phase10-promote\d+-%dp$" % n)
    used = set(c.get("id") for c in consumed)
    open_promote = sorted(
        (e["lo"], e["id"]) for e in reserved
        if e.get("num_players") == n and e.get("group") == "dev"
        and pattern.match(str(e.get("purpose"))) and e["id"] not in used)
    direct_id = parsed["direct"]["reservation"]
    if not pattern.match(parsed["direct"]["purpose"]):
        raise ValueError("direct purpose must look like phase10-promoteK-%dp"
                         % n)
    if direct_id not in used and (not open_promote or
                                  open_promote[0][1] != direct_id):
        raise ValueError("direct band must be the lowest unconsumed %d-seat "
                         "promotion reservation (%r)" % (n, open_promote[:1]))
    return parsed


def _load_d5_conditions(path, allow_informal=False, ledger_path=None,
                        initial_record_path=None):
    data = _read_conditions_bytes(path)
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("conditions %s is not valid UTF-8 JSON: %s"
                         % (path, exc))
    return (_validate_d5(parsed, allow_informal, ledger_path,
                         initial_record_path), _conditions_sha256(data))


def _d5_paths(conditions, label):
    base = os.path.join(conditions["out_dir"], label)
    return base + ".jsonl", base + ".json"


def d5_register(conditions, execute=False, ledger_path=None):
    """ベンチ帯と昇格帯の hold を、事前固定の対戦(とモデル SHA)で確定する。

    同じ対戦・SHA で既に確定済みなら何もしない(冪等)。それ以外の既存登録は
    eval_agent.register_multi が拒否する(hold の確定は 1 回だけ)。
    """
    n = conditions["num_players"]
    matchups = _d5_matchups(conditions)
    unused, reserved, unused_c, unused_two = eval_agent.read_multi_ledger(
        ledger_path)
    out = []
    for label, threshold in (("bench", None),
                             ("direct", conditions["promote_threshold"])):
        block = conditions[label]
        lo = block["start_index"]
        hi = lo + block["clusters"] - 1
        entry = eval_agent.build_reservation(
            block["reservation"], n, [m for _l, m in matchups[label]],
            block["group"], lo, hi, block["purpose"], threshold)
        current = [e for e in reserved if e.get("id") == block["reservation"]]
        if current and current[0].get("matchups") == entry["matchups"] and \
                current[0].get("model_sha256") == entry["model_sha256"]:
            out.append({"id": entry["id"], "status": "already registered"})
            continue
        if execute:
            eval_agent.register_multi(entry, ledger_path=ledger_path)
            out.append({"id": entry["id"], "status": "registered",
                        "matchups": entry["matchups"]})
        else:
            out.append({"id": entry["id"], "status": "would register",
                        "matchups": entry["matchups"],
                        "model_sha256": entry["model_sha256"]})
    return out


def _d5_run(conditions, label, agents, block, threshold, workers, resume,
            ledger_path):
    """1 対戦を run_multi で実行する。summary が既にあれば再計算で検証して再利用。"""
    records, summary_path = _d5_paths(conditions, label)
    if os.path.exists(summary_path):
        fresh = eval_agent.recompute_multi(summary_path)
        with open(summary_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        if saved["conditions"]["agents"] != list(agents) or \
                saved["reservation_id"] != block["reservation"]:
            raise ValueError("%s exists for another matchup/reservation"
                             % summary_path)
        return saved, fresh, True
    summary = eval_agent.run_multi(
        block["reservation"], conditions["num_players"], list(agents),
        block["group"], block["start_index"], block["clusters"],
        block["purpose"], threshold, records, summary_path, workers=workers,
        # --resume は「起動済みで records がある対戦」だけを再開し、未起動の
        # 対戦は新規に claim する(stage 全体を中断点から続けるため)。
        resume=resume and os.path.exists(records), ledger_path=ledger_path)
    return summary, eval_agent.recompute_multi(summary_path), False


def _d5_summary_row(label, agents, saved, reused):
    records = saved["records_path"]
    return {
        "label": label, "agents": list(agents),
        "reservation_id": saved["reservation_id"], "run_id": saved["run_id"],
        "group": saved["group"], "start_index": saved["start_index"],
        "end_index": saved["end_index"], "clusters": saved["clusters"],
        "games": saved["games"], "wins": saved["wins"],
        "point_estimate": saved["point_estimate"], "ci_low": saved["ci_low"],
        "ci_high": saved["ci_high"], "threshold": saved["threshold"],
        "passed": saved["passed"], "records_sha256": saved["records_sha256"],
        "records_path": records, "models": saved.get("models"),
        "reused_existing_summary": reused,
    }


def run_d5(conditions, conditions_path, conditions_sha256, stage, out,
           workers=1, resume=False, formal=True, ledger_path=None):
    """D5 の stage を実行する(bench: champion のベンチ / promote: 候補ベンチ+direct+判定)。"""
    from naeco.ai.eval import harness
    n = conditions["num_players"]
    matchups = _d5_matchups(conditions)
    ledger_file = ledger_path or eval_agent.SEED_LEDGER_PATH
    frozen = {}
    for label, path in (("champion", conditions["champion"]),
                        ("challenger", conditions["challenger"]),
                        ("bench:initial",
                         conditions["bench"]["entries"][1]["path"])):
        frozen[label] = harness.model_file_shas(path)
    if not os.path.isdir(conditions["out_dir"]):
        os.makedirs(conditions["out_dir"])
    champion_rows = []
    for label, agents in matchups["bench"]:
        if not label.startswith("bench-champion-"):
            continue
        if stage == "bench":
            saved, unused_fresh, reused = _d5_run(
                conditions, label, agents, conditions["bench"], None, workers,
                resume, ledger_path)
        else:
            unused_records, summary_path = _d5_paths(conditions, label)
            if not os.path.exists(summary_path):
                raise ValueError("champion bench %s is missing; run --stage "
                                 "bench first" % summary_path)
            eval_agent.recompute_multi(summary_path)
            with open(summary_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved["conditions"]["agents"] != agents:
                raise ValueError("%s is for another matchup" % summary_path)
            reused = True
        champion_rows.append(_d5_summary_row(label, agents, saved, reused))
    if stage == "bench":
        for label, path in (("champion", conditions["champion"]),
                            ("challenger", conditions["challenger"]),
                            ("bench:initial",
                             conditions["bench"]["entries"][1]["path"])):
            if harness.model_file_shas(path) != frozen[label]:
                raise RuntimeError("model %s changed during the bench" % path)
        payload = {
            "schema_version": MULTI_D5_DECISION_SCHEMA,
            "stage": "bench",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "formal": formal, "certified": False, "num_players": n,
            "conditions": conditions, "conditions_sha256": conditions_sha256,
            "frozen_models": frozen, "champion_bench": champion_rows,
            "champion_record": dict((r["label"].rsplit("-", 1)[1],
                                     r["point_estimate"])
                                    for r in champion_rows),
        }
        _write_json(out, payload)
        return payload
    challenger_rows = []
    for label, agents in matchups["bench"]:
        if label.startswith("bench-challenger-"):
            saved, unused_fresh, reused = _d5_run(
                conditions, label, agents, conditions["bench"], None, workers,
                resume, ledger_path)
            challenger_rows.append(_d5_summary_row(label, agents, saved,
                                                   reused))
    label, agents = matchups["direct"][0]
    saved, unused_fresh, reused = _d5_run(
        conditions, label, agents, conditions["direct"],
        conditions["promote_threshold"], workers, resume, ledger_path)
    direct_row = _d5_summary_row(label, agents, saved, reused)
    _verify_conditions_unchanged(conditions_path, conditions_sha256)
    for label_, path in (("champion", conditions["champion"]),
                         ("challenger", conditions["challenger"]),
                         ("bench:initial",
                          conditions["bench"]["entries"][1]["path"])):
        if harness.model_file_shas(path) != frozen[label_]:
            raise RuntimeError("model %s changed during the run; no decision"
                               % path)
    name_of = lambda row: row["label"].rsplit("-", 1)[1]
    champion_record = dict((name_of(r), r["point_estimate"])
                           for r in champion_rows)
    challenger_record = dict((name_of(r), r["point_estimate"])
                             for r in challenger_rows)
    direct_pass = direct_row["ci_low"] > conditions["promote_threshold"]
    if direct_row["passed"] is not direct_pass:
        raise AssertionError("direct passed flag disagrees with ci_low > 1/n")
    decision = decide(direct_pass, challenger_record, champion_record,
                      conditions["promote_tolerance"],
                      conditions["regression_tolerance"])
    payload = {
        "schema_version": MULTI_D5_DECISION_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "formal": formal,
        "certified": False,
        "num_players": n,
        "conditions": conditions,
        "conditions_sha256": conditions_sha256,
        "seed_ledger": {
            "path": os.path.relpath(os.path.abspath(ledger_file), REPO_ROOT),
            "sha256": _file_sha256(ledger_file, "seed ledger"),
        },
        "frozen_models": frozen,
        "agent_mode": "greedy checkpoint per player count (strong is for the "
                      "final accept only)",
        "champion": "checkpoint:" + conditions["champion"],
        "challenger": "checkpoint:" + conditions["challenger"],
        "direct": direct_row,
        "benchmark": {
            "champion_runs": champion_rows,
            "challenger_runs": challenger_rows,
            "champion_record": champion_record,
            "challenger_record": challenger_record,
            "promote_tolerance": conditions["promote_tolerance"],
            "regression_tolerance": conditions["regression_tolerance"],
            "promotion_floor_ok": benchmark_floor_ok(
                challenger_record, champion_record,
                conditions["promote_tolerance"]),
            "regression_detected": regression_detected(
                challenger_record, champion_record,
                conditions["regression_tolerance"]),
        },
        "rule": "rollback if any bench < champion-0.05; promote iff direct "
                "ci_low > 1/n and every bench >= champion-0.03; else reject",
        "decision": decision,
        "new_champion": ("checkpoint:" + conditions["challenger"]
                         if decision == "promote"
                         else "checkpoint:" + conditions["champion"]),
        "new_champion_record": (challenger_record if decision == "promote"
                                else champion_record),
        "versions": dict(_versions_multi(n)),
        "python_version": platform.python_version(),
        "thread_env": dict((name, os.environ.get(name))
                           for name in _THREAD_ENV_NAMES),
    }
    _write_json(out, payload)
    return payload


def _versions_multi(n):
    from naeco.train import encoding_multi
    cards = load_cards()
    contract = encoding_multi.contract_for(n)
    return {
        "rules_version": state_mod.RULES_VERSION,
        "api_version": state_mod.API_VERSION,
        "obs_version": OBS_VERSION,
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "obs_encoder_version": contract.obs_encoder_version,
        "action_encoder_version": contract.action_encoder_version,
        "reward_version": reward_mod.reward_version(n),
    }


def main_d5(args):
    """--num-players 3|4 で conditions が phase10-promote/v1 のときの入口。"""
    if args.stage not in MULTI_D5_STAGES:
        raise SystemExit("--stage must be one of %s for the D5 path"
                         % ", ".join(MULTI_D5_STAGES))
    try:
        conditions, digest = _load_d5_conditions(
            args.conditions, allow_informal=args.allow_informal,
            ledger_path=args.seed_ledger)
    except ValueError as exc:
        raise SystemExit("conditions: %s" % exc)
    if conditions["num_players"] != args.num_players:
        raise SystemExit("--num-players %d differs from conditions (%d)"
                         % (args.num_players, conditions["num_players"]))
    if os.path.abspath(args.challenger) != \
            os.path.abspath(conditions["challenger"]) or \
            os.path.abspath(args.champion) != \
            os.path.abspath(conditions["champion"]):
        raise SystemExit("--challenger/--champion must equal the conditions "
                         "(fixed before the run)")
    formal = not args.allow_informal
    try:
        if args.stage == "register":
            result = d5_register(conditions, execute=args.execute,
                                 ledger_path=args.seed_ledger)
            print(json.dumps({"formal": formal, "register": result},
                             sort_keys=True))
            return 0
        if not args.execute:
            print("prepared: D5 n=%d stage=%s conditions=%s sha256=%s "
                  "matchups=%s workers=%d formal=%s"
                  % (conditions["num_players"], args.stage, args.conditions,
                     digest, json.dumps(_d5_matchups(conditions)),
                     args.workers, formal))
            return 0
        if os.path.exists(args.out):
            raise SystemExit("記録先が既に存在する: %s" % args.out)
        result = run_d5(conditions, args.conditions, digest, args.stage,
                        args.out, workers=args.workers, resume=args.resume,
                        formal=formal, ledger_path=args.seed_ledger)
    except (eval_agent.SeedLedgerError, ValueError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    if args.stage == "bench":
        for row in result["champion_bench"]:
            print("bench %s: point=%.4f CI=[%.4f, %.4f] reused=%s"
                  % (row["label"], row["point_estimate"], row["ci_low"],
                     row["ci_high"], row["reused_existing_summary"]))
        return 0
    print("direct: point=%.4f CI=[%.4f, %.4f] > 1/n=%.4f: %s"
          % (result["direct"]["point_estimate"], result["direct"]["ci_low"],
             result["direct"]["ci_high"], conditions["promote_threshold"],
             result["direct"]["passed"]))
    print("bench champion=%s challenger=%s"
          % (json.dumps(result["benchmark"]["champion_record"], sort_keys=True),
             json.dumps(result["benchmark"]["challenger_record"],
                        sort_keys=True)))
    print("decision: %s (formal=%s)" % (result["decision"], formal))
    return 0


def _is_d5_conditions(path):
    try:
        with open(path, "rb") as fh:
            parsed = json.loads(fh.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(parsed, dict) and parsed.get("schema") == MULTI_D5_SCHEMA


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="promote.py",
        description="昇格・退行判定(Phase 3 設計書 §7)。判定条件は "
                    "--conditions で実行前に読み込んで固定する")
    parser.add_argument("--num-players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--conditions", required=True, metavar="PATH",
                        help="判定条件 JSON(直接対戦・ベンチマーク・閾値・"
                             "チャンピオン記録。実行前に sha256 で固定)")
    parser.add_argument("--seed-ledger", metavar="PATH", default=None,
                        help="消費済み seed 台帳(既定: docs/eval-seeds.md)。"
                             "正式条件の direct 帯はこの台帳から求めた"
                             "**次の未消費の予約帯**と一致しなければならない。"
                             "差し替えた場合もパスと sha256 が判定 JSON に残る")
    parser.add_argument("--challenger", required=True, metavar="META",
                        help="昇格候補のチェックポイント(メタ .meta.json)")
    parser.add_argument("--champion", required=True, metavar="META",
                        help="現行チャンピオンのチェックポイント(メタ)")
    parser.add_argument("--out", required=True, metavar="PATH",
                        help="判定結果 JSON の出力先")
    parser.add_argument(
        "--records", default=None, metavar="PREFIX",
        help="対応表 JSONL の接頭辞(接頭辞-direct.jsonl と 接頭辞-bench.jsonl)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--allow-informal", action="store_true",
        help="正式条件(800ペア / 0.50 / 0.03 / 0.05 / dev 7800-8599・"
             "9000-9799)からの逸脱を許す。出力に「非正式」と明記される"
             "(設計書 §18.2)")
    parser.add_argument(
        "--execute", action="store_true",
        help="実行を明示する。無ければ準備表示のみで終了")
    parser.add_argument(
        "--stage", choices=MULTI_D5_STAGES, default=None,
        help="n>=3 の正式 D5 経路(conditions schema phase10-promote/v1)の段階: "
             "register(ベンチ帯・昇格帯の hold を対戦とモデル SHA で確定)/ "
             "bench(champion の固定ベンチ)/ promote(候補ベンチ + 直接対戦 + 判定)")
    parser.add_argument(
        "--resume", action="store_true",
        help="n>=3 D5: 起動済み対戦の欠けた局だけを同じ run で再開する")
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1, got %d" % args.workers)
    if args.num_players == 2 and (args.stage or args.resume):
        raise SystemExit("--stage / --resume are only for --num-players 3|4")
    if args.num_players > 2 and _is_d5_conditions(args.conditions):
        if args.stage is None:
            raise SystemExit("D5 conditions need --stage register|bench|promote")
        return main_d5(args)
    if args.num_players > 2 and (args.stage or args.resume):
        raise SystemExit("--stage / --resume need phase10-promote/v1 conditions")
    if args.num_players > 2:
        try:
            result = run_multi_smoke(args)
            print(json.dumps(result if not args.execute else
                             {"formal": False, "out": args.out,
                              "diagnostic_decision": result["diagnostic_decision"]}))
            return 0
        except (ValueError, RuntimeError) as exc:
            print("error: %s" % exc, file=sys.stderr)
            return 1
    try:
        conditions, conditions_sha256 = _load_conditions(
            args.conditions, allow_informal=args.allow_informal,
            ledger_path=args.seed_ledger)
    except ValueError as exc:
        raise SystemExit("conditions: %s" % exc)
    for label, path in (("--challenger", args.challenger),
                        ("--champion", args.champion)):
        try:
            eval_agent.checkpoint_weights_path(path)
            eval_agent.parse_agent_spec("checkpoint:" + path)
        except ValueError as exc:
            raise SystemExit("%s: %s" % (label, exc))
    if not args.execute:
        print("prepared: conditions=%s sha256=%s challenger=%s champion=%s "
              "direct=%s index=%d pairs=%d bench=%s index=%d pairs=%d "
              "entries=%d workers=%d formal=%s"
              % (args.conditions, conditions_sha256, args.challenger,
                 args.champion, conditions.direct_group,
                 conditions.direct_start_index, conditions.direct_pairs,
                 conditions.bench_group, conditions.bench_start_index,
                 conditions.bench_pairs, len(conditions.benchmark_entries),
                 args.workers, _conditions_are_formal(conditions)))
        return 0
    for path in ([args.out] + (
            [args.records + "-direct.jsonl", args.records + "-bench.jsonl"]
            if args.records else [])):
        if os.path.exists(path):
            raise SystemExit(
                "記録先が既に存在する: %s / 追記すると『1回だけ実行』の証跡が"
                "壊れる。別名を指定するか退避すること。" % path)
    try:
        run(args, conditions, conditions_sha256)
    except (RuntimeError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
