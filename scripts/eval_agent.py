#!/usr/bin/env python3
"""任意2エージェントの席交換ペア対局評価 CLI(Phase 3 設計書 §4.5 の評価側 / §7)。

    python scripts/eval_agent.py heuristic random --group dev \
        --start-index 7800 --pairs 800 --workers 4 \
        --records docs/eval-runs/phase3-promote-dev-7800-8599.jsonl --execute

- エージェント指定は `random` / `heuristic` / `checkpoint:メタのパス` の3種。
  checkpoint はメタ JSON のパスを与えると `PolicyAgent(mode="greedy")` を
  作る。読込は必ず `checkpoint.load(..., expected=checkpoint.expected_meta(...))`
  を渡す(設計書 §15.1。渡さない経路は無く、テストで固定する)。
- model_arch は保存メタから読むが(設計書 §15.2)、`_read_model_arch` が
  キー集合の完全一致(余分も欠落も拒否)と型(obs_dim / action_dim は
  `type(x) is int`、hidden / action_hidden は list で全要素 int)を厳格に
  検証する(§17.1)。検証を通った arch をそのまま(正規化せず)使って
  PolicyValueNet を構築し、重み state_dict のキー集合と各テンソル形状が
  一致することも検証してから使う。偽装した model_arch はここで拒否される
  (この検証を経ない読込経路を作らない)。
- num_players は評価プロトコル側の値(2人戦)の定数を使い、保存メタから
  読まない(設計書 §15.2)。
- 対局駆動は `naeco/train/env.py` の `play_episode` を使い、`seat_variant`
  0 と 1 の2局で1ペアとする(同一配札・席交換。設計書 §6.3)。
- 統計処理は `naeco/ai/eval/harness.py` の `cluster_bootstrap` /
  `summarize_pairs` / `passes_threshold` を再利用し、新規実装しない。
- 対応表は `GameRecord` を継承したレコードに §4.5 の版情報を追加し、
  `write_game_records` で1局1行の JSONL として保存する。
- 並列は multiprocessing(spawn)。worker 数を変えても結果は完全一致する
  (seed は論理座標だけから派生し、worker index を混ぜない)。
- torch は直接 import しない。checkpoint エージェントの構築だけが関数内の
  遅延 import で `naeco/train/net.py`(torch 境界)に到達する。
- `--execute` を付けない限り実行しない(`scripts/compare_search.py` と同じ
  規律)。合否は CI 下限による厳密不等号 >(`passes_threshold` 再利用)。
- `--record-eval-to META` で、評価結果(相手・group・index 範囲・ペア数・
  勝分敗・点推定・95%CI・対応表の SHA-256・合否)を評価対象の checkpoint
  メタの `eval_results`(B 表キー)へ追記できる(設計書 §18.1)。追記先は
  評価対象の checkpoint エージェントのメタでなければならない。
"""
import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
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

from naeco.ai.eval.harness import (  # noqa: E402
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, GameRecord, cluster_bootstrap,
    passes_threshold, summarize_pairs, write_game_records,
)
from naeco.ai.heuristic_agent import HeuristicAgent  # noqa: E402
from naeco.ai.random_agent import RandomAgent  # noqa: E402
from naeco.engine import game  # noqa: E402
from naeco.engine.cards import load_cards  # noqa: E402
from naeco.engine.observe import OBS_VERSION  # noqa: E402
from naeco.engine.state import (  # noqa: E402
    API_VERSION, GROUPS, RULES_VERSION, derive_seed, eval_seed, state_hash,
)
from naeco.train import encoding as enc  # noqa: E402
from naeco.train import reward as reward_mod  # noqa: E402
from naeco.train.agent import PolicyAgent  # noqa: E402
from naeco.train.env import play_episode  # noqa: E402

NUM_PLAYERS = 2
CHECKPOINT_META_SUFFIX = ".meta.json"
CHECKPOINT_WEIGHTS_SUFFIX = ".weights.pt"
AGENT_KINDS = ("random", "heuristic", "checkpoint")


def _fix_thread_env():
    """4つのスレッド環境変数を "1" へ固定する(設計書 §8.4)。"""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


_fix_thread_env()


@dataclasses.dataclass(frozen=True)
class EvalGameRecord(GameRecord):
    """対応表の1局行。GameRecord に §4.5 の記録要件(版)を追加する。

    `write_game_records` は dataclasses.asdict で書くため、このサブクラスを
    そのまま書き出せる(harness の記録経路を新規実装しない)。
    """

    left: str
    right: str
    obs_version: str
    obs_encoder_version: str
    action_encoder_version: str
    reward_version: str


@dataclasses.dataclass(frozen=True)
class AgentEvaluationResult:
    """1回の評価の集計。harness.EvaluationResult と同じ形で持つ。"""

    left: str
    right: str
    group: str
    start_index: int
    pairs: int
    games: int
    wins: int
    draws: int
    losses: int
    point_estimate: float
    ci_low: float
    ci_high: float
    bootstrap_samples: int
    bootstrap_seed: int
    pair_results: tuple
    initial_state_hashes: tuple
    game_records: tuple


def parse_agent_spec(spec):
    """エージェント指定を (kind, path) へ正規化する。

    `random` / `heuristic` / `checkpoint:メタのパス` の3種を受け付ける。
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError("agent spec must be a non-empty string")
    kind, sep, path = spec.partition(":")
    if sep and kind != "checkpoint":
        raise ValueError(
            "only checkpoint specs take a path: {!r}".format(spec))
    if kind == "checkpoint":
        if not path:
            raise ValueError(
                "checkpoint spec requires a meta path: {!r}".format(spec))
        return (kind, path)
    if kind not in AGENT_KINDS:
        raise ValueError(
            "unknown agent spec {!r} (valid: {})".format(
                spec, ", ".join(AGENT_KINDS)))
    return (kind, "")


def checkpoint_weights_path(meta_path):
    """メタのパスから重み(.pt)のパスを導出する(train_ppo の命名規約)。"""
    if meta_path.endswith(CHECKPOINT_META_SUFFIX):
        prefix = meta_path[:-len(CHECKPOINT_META_SUFFIX)]
        return prefix + CHECKPOINT_WEIGHTS_SUFFIX
    raise ValueError(
        "checkpoint meta path must end with {!r}: {!r}".format(
            CHECKPOINT_META_SUFFIX, meta_path))


_MODEL_ARCH_KEYS = ("obs_dim", "action_dim", "hidden", "action_hidden")


def _read_model_arch(meta_path):
    """expected=expected_meta(...) を組むために model_arch を厳格に検証する。

    評価・昇格経路では保存メタから読んだ model_arch を expected の材料に
    するため(設計書 §15.2)、ここで型偽装を素通しすると自己比較になって
    しまう。そこで model_arch は次を**すべて**満たすことだけを受理する
    (設計書 §17.1 第一):

    - キー集合が {"obs_dim", "action_dim", "hidden", "action_hidden"} と
      完全一致する(余分なキーも欠落も拒否)。
    - obs_dim / action_dim は `type(x) is int`(bool は int の派生なので
      type の完全一致で判定する)。
    - hidden / action_hidden は list であり、全要素が `type(x) is int`。

    1つでも違えば IncompatibleCheckpoint で拒否する。**正規化はしない**
    (§17.1 第二) — 検証を通った値をそのまま PolicyValueNet の構築へ渡す。
    正規化(int へのキャスト)は不正な入力を黙って受け入れることと同義であり
    設計書 §16 の「正規化しない」方針に反する。

    この読み取りは互換性の受入を行わない。互換性の照合は必ず
    `checkpoint.load(meta_path, expected=checkpoint.expected_meta(...))` が
    行い、model_arch 自体は読んだ arch で PolicyValueNet を構築して重みの
    state_dict と照合する(設計書 §15.2)。model_arch は expected を組むのに
    必要なため、メタ JSON から arch だけを先に取り出す。
    """
    from naeco.train import checkpoint as checkpoint_mod
    with open(meta_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    arch = raw.get("model_arch")
    if not isinstance(arch, dict):
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: {} model_arch is not an object"
            .format(meta_path))
    if set(arch) != set(_MODEL_ARCH_KEYS):
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: {} model_arch keys must be exactly "
            "{} (got {})".format(
                meta_path, sorted(_MODEL_ARCH_KEYS), sorted(arch)))
    for key in ("obs_dim", "action_dim"):
        if type(arch[key]) is not int:
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: {} model_arch.{} must be an int "
                "(got {})".format(
                    meta_path, key, type(arch[key]).__name__))
    for key in ("hidden", "action_hidden"):
        value = arch[key]
        if not isinstance(value, list) or \
                any(type(item) is not int for item in value):
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: {} model_arch.{} must be a list "
                "of ints (got {!r})".format(meta_path, key, value))
    return arch


def _verify_weights_match_arch(net, state_dict, meta_path):
    """読んだ model_arch で構築した net と重み state_dict を照合する(§15.2)。

    保存メタの model_arch はキー集合と型を _read_model_arch で検証済みだが
    (§17.1)、重みとの一致は未検証の値なので、その arch で PolicyValueNet
    を構築し、重み state_dict のキー集合と各テンソル形状が一致することを
    検証する。偽装した model_arch は重みと矛盾するため IncompatibleCheckpoint
    で拒否される(形状まで一致する偽装は実質的に同じアーキテクチャであり、
    互換性に影響しない)。この検証を経ない読込経路は作らない。
    """
    from naeco.train import checkpoint as checkpoint_mod
    expected = net.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: %s weights do not match model_arch "
            "(missing keys %s, unexpected keys %s)"
            % (meta_path, missing, unexpected))
    for name in expected:
        expected_shape = tuple(expected[name].shape)
        actual_shape = tuple(state_dict[name].shape)
        if actual_shape != expected_shape:
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: %s weights do not match model_arch "
                "(%s has shape %s, model_arch requires %s)"
                % (meta_path, name, actual_shape, expected_shape))


def _make_checkpoint_agent(meta_path, seed):
    """メタのパスから greedy の PolicyAgent を作る。

    torch はここで初めて遅延 import される(`naeco/train/net.py` が torch
    境界。本モジュールは torch を直接 import しない)。`checkpoint.load` へは
    必ず `expected=checkpoint.expected_meta(...)` を渡す(§15.1)。

    model_arch は保存メタから読んでよいが(§15.2)、`_read_model_arch` が
    キー集合の完全一致と型を厳格に検証する(§17.1。型偽装・余分なキー・
    欠落はこの先読みで拒否される)。検証を通った arch を**そのまま**(
    正規化しない)使って PolicyValueNet を構築し、重み state_dict のキー
    集合と各テンソル形状が一致することも検証する。偽装した model_arch は
    重みと矛盾するため `_verify_weights_match_arch` で拒否される。
    num_players は評価プロトコル側の値(NUM_PLAYERS = 2)を渡し、ファイル
    から読まない(§15.2)。
    """
    from naeco.train import checkpoint as checkpoint_mod
    weights_path = checkpoint_weights_path(meta_path)
    arch = _read_model_arch(meta_path)
    checkpoint_mod.load(
        meta_path,
        expected=checkpoint_mod.expected_meta(NUM_PLAYERS, arch))
    if not os.path.exists(weights_path):
        raise ValueError(
            "checkpoint {} has meta but weights {} is missing".format(
                meta_path, weights_path))
    from naeco.train import net as net_mod
    torch_module = net_mod.torch          # net.py が import した torch
    # §17.1 第二: 正規化(int へのキャスト)はしない。_read_model_arch の
    # 厳格検証を通った値をそのまま使う。正規化は不正な入力を黙って受け入れる。
    net = net_mod.PolicyValueNet(
        arch["obs_dim"], arch["action_dim"],
        hidden=tuple(arch["hidden"]),
        action_hidden=tuple(arch["action_hidden"]))
    state_dict = torch_module.load(weights_path, map_location="cpu")
    _verify_weights_match_arch(net, state_dict, meta_path)
    net.load_state_dict(state_dict)
    net.eval()
    return PolicyAgent(net, mode="greedy", seed=seed)


def _make_agent(spec, seed):
    """spec からエージェントを作る。seed は論理座標由来(worker 非依存)。"""
    kind, path = parse_agent_spec(spec)
    if kind == "random":
        return RandomAgent(seed)
    if kind == "heuristic":
        return HeuristicAgent(seed)
    return _make_checkpoint_agent(path, seed)


def _play_game(cards, group, index, seat_variant, left, right):
    """席交換ペアの1局を play_episode で駆動し、(結果, 対応表1行)を返す。

    初期配札の記録は play_episode と同じ seed 派生で new_game を再現して
    取得し、Trajectory.initial_state_hash との一致を検証する(記録と実対局が
    食い違ったらエンジンの契約違反として失敗する)。
    """
    root = eval_seed(group, index)
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game.new_game(NUM_PLAYERS, env_seed, cards, setup_seed=setup_seed)
    initial_hash = state_hash(state)
    initial_deck = tuple(state.deck)
    initial_hands = tuple(tuple(p.hand_buildings) for p in state.players)
    initial_money = tuple(p.money for p in state.players)
    start_player = state.start_player

    left_seed = derive_seed(root, "policy", index, seat_variant)
    right_seed = derive_seed(root, "policy", index, 1 - seat_variant)
    agents = [_make_agent(left, left_seed), _make_agent(right, right_seed)]
    trajectory = play_episode(
        group, index, seat_variant, agents, cards=cards,
        record_vectors=False)
    if trajectory.discarded:
        raise RuntimeError(
            "game discarded by engine bug ({} / {} seat_variant={}): {}".format(
                group, index, seat_variant, trajectory.discard_reason))
    if trajectory.initial_state_hash != initial_hash:
        raise AssertionError(
            "recorded initial deal does not match play_episode ({}/{})"
            .format(group, index))
    if len(trajectory.terminal_rewards) != NUM_PLAYERS:
        raise AssertionError(
            "terminal_rewards must have length {}: {!r}".format(
                NUM_PLAYERS, trajectory.terminal_rewards))
    utility = trajectory.terminal_rewards[seat_variant]
    outcome = 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5
    record = EvalGameRecord(
        group=group, index=index, seat_variant=seat_variant,
        env_seed=env_seed, setup_seed=setup_seed,
        policy_seed=left_seed, opponent_seed=right_seed,
        target_seat=seat_variant, opponent_seat=1 - seat_variant,
        start_player=start_player,
        initial_deck=initial_deck, initial_hands=initial_hands,
        initial_money=initial_money, initial_state_hash=initial_hash,
        cards_sha256=cards.sha256, cards_data_version=cards.data_version,
        api_version=API_VERSION, rules_version=RULES_VERSION,
        result=outcome,
        left=left, right=right,
        obs_version=OBS_VERSION,
        obs_encoder_version=enc.OBS_ENCODER_VERSION,
        action_encoder_version=enc.ACTION_ENCODER_VERSION,
        reward_version=reward_mod.REWARD_VERSION,
    )
    return outcome, record


def _run_game_task(task):
    """spawn worker で1局を実行する。task は論理座標だけを含む。"""
    group, index, seat_variant, left, right = task
    cards = load_cards()
    return _play_game(cards, group, index, seat_variant, left, right)


def run_evaluation(left, right, group, start_index, pairs, workers=1):
    """指定2エージェントを席交換ペアで評価し、集計結果を返す。

    worker 数を変えても結果は完全一致する(各局の seed は group / index /
    seat_variant だけから派生する)。
    """
    parse_agent_spec(left)
    parse_agent_spec(right)
    if group not in GROUPS:
        raise ValueError("unknown seed group: {!r}".format(group))
    if pairs < 1:
        raise ValueError("pairs must be positive, got %d" % pairs)
    if start_index < 0:
        raise ValueError("start_index must be >= 0, got %d" % start_index)
    if workers < 1:
        raise ValueError("workers must be >= 1, got %d" % workers)
    tasks = [
        (group, start_index + offset, variant, left, right)
        for offset in range(pairs)
        for variant in (0, 1)
    ]
    if workers == 1:
        results = [_run_game_task(task) for task in tasks]
    else:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(processes=workers)
        try:
            results = list(pool.map(_run_game_task, tasks))
        finally:
            pool.close()
            pool.join()
    pair_games = []
    hashes = []
    records = []
    for offset in range(pairs):
        outcome0, record0 = results[2 * offset]
        outcome1, record1 = results[2 * offset + 1]
        if record0.initial_state_hash != record1.initial_state_hash:
            raise AssertionError(
                "paired games did not share an initial deal (index %d)"
                % (start_index + offset))
        pair_games.append((outcome0, outcome1))
        hashes.append((record0.initial_state_hash,
                       record1.initial_state_hash))
        records.append(record0)
        records.append(record1)
    wins, draws, losses, pair_results, point = summarize_pairs(pair_games)
    ci_low, ci_high = cluster_bootstrap(
        pair_results, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    return AgentEvaluationResult(
        left=left, right=right, group=group, start_index=start_index,
        pairs=pairs, games=pairs * 2, wins=wins, draws=draws, losses=losses,
        point_estimate=point, ci_low=ci_low, ci_high=ci_high,
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        pair_results=pair_results, initial_state_hashes=tuple(hashes),
        game_records=tuple(records),
    )


def append_eval_result(meta_path, result, threshold, records_path=None):
    """評価結果をチェックポイントのメタの eval_results へ追記する(設計書 §18.1)。

    PLAN.md Phase 3 の再現メタデータ「評価結果」を満たす経路。記録内容は
    相手の指定・group・index 範囲・ペア数・勝分敗・点推定・95%CI・対応表の
    SHA-256・合否判定。

    - meta_path はこの評価の checkpoint エージェント(左か右)のメタでなければ
      ならない(それ以外は ValueError)。
    - 読込は必ず `checkpoint.load(..., expected=expected_meta(...))` を渡す
      (設計書 §15.1)。`eval_results` は B 表キーなので、追記しても A 表13キー
      は不変(テストで固定)。
    - eval_results が null なら新規 list として扱い、list なら追記し、それ以外
      は ValueError(fail-closed)。
    戻り値は追記したエントリ(dict)。
    """
    from naeco.train import checkpoint as checkpoint_mod
    left_kind, left_path = parse_agent_spec(result.left)
    right_kind, right_path = parse_agent_spec(result.right)
    recorded = opponent = None
    if left_kind == "checkpoint" and \
            os.path.abspath(left_path) == os.path.abspath(meta_path):
        recorded, opponent = result.left, result.right
    elif right_kind == "checkpoint" and \
            os.path.abspath(right_path) == os.path.abspath(meta_path):
        recorded, opponent = result.right, result.left
    if recorded is None:
        raise ValueError(
            "record target %s is not one of the evaluated checkpoint agents "
            "(%s / %s)" % (meta_path, result.left, result.right))
    weights_path = checkpoint_weights_path(meta_path)
    arch = _read_model_arch(meta_path)
    meta = checkpoint_mod.load(
        meta_path,
        expected=checkpoint_mod.expected_meta(NUM_PLAYERS, arch))
    if not os.path.exists(weights_path):
        raise ValueError(
            "checkpoint {} has meta but weights {} is missing".format(
                meta_path, weights_path))
    records_sha256 = None
    if records_path and os.path.exists(records_path):
        with open(records_path, "rb") as fh:
            records_sha256 = hashlib.sha256(fh.read()).hexdigest()
    entry = {
        "recorded_agent": recorded,
        "opponent": opponent,
        "group": result.group,
        "start_index": result.start_index,
        "end_index": result.start_index + result.pairs - 1,
        "pairs": result.pairs,
        "games": result.games,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "point_estimate": result.point_estimate,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "bootstrap_samples": result.bootstrap_samples,
        "bootstrap_seed": result.bootstrap_seed,
        "threshold": threshold,
        "passed": passes_threshold(result, threshold),
        "records_sha256": records_sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = meta.get("eval_results")
    if results is None:
        results = []
    if not isinstance(results, list):
        raise ValueError(
            "eval_results must be null or a list, got %s"
            % type(results).__name__)
    results.append(entry)
    meta["eval_results"] = results
    checkpoint_mod.save(meta_path, meta)
    return entry


# ---------------------------------------------------------------------------
# Phase 10 D4: 3・4 人戦の n 局巡回クラスタ評価と多人数 seed 登録。
#
# n=2 の経路(上の run_evaluation / main の既定)は一切変えない。
# 台帳(docs/eval-seeds.md)の 2 人用 CONSUMED-DEV-BANDS と promote.py の
# 次帯検査には書き込まない。多人数は専用の予約欄と消費欄を別に持つ。
# ---------------------------------------------------------------------------

SEED_LEDGER_PATH = os.path.join(REPO_ROOT, "docs", "eval-seeds.md")
MULTI_RESERVED_BEGIN = "<!-- MULTI-SEAT-RESERVED:BEGIN -->"
MULTI_RESERVED_END = "<!-- MULTI-SEAT-RESERVED:END -->"
MULTI_CONSUMED_BEGIN = "<!-- MULTI-SEAT-CONSUMED:BEGIN -->"
MULTI_CONSUMED_END = "<!-- MULTI-SEAT-CONSUMED:END -->"
_RESERVATION_KEYS = ("id", "num_players", "group", "lo", "hi", "purpose",
                     "matchups", "threshold", "model_sha256", "power",
                     "registered_at")
# accept 群の 2 人戦消費(Phase 8 最終受け入れ、eval-seeds.md)。
TWO_SEAT_ACCEPT_CONSUMED = ((0, 799),)
# 多人数専用の dev 領域(2026-10-09 オーケストレータ裁定、phase10-design D4 改訂行)。
# 2 人昇格予約列とは独立。多人数の dev 帯はこの領域内でだけ 2 人列と衝突しない。
MULTI_DEV_REGION = (30000, 39999)


class SeedLedgerError(ValueError):
    """台帳との突合に失敗した(fail-closed)。"""


def _block(text, begin, end):
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start or \
            text.count(begin) != 1 or text.count(end) != 1:
        raise SeedLedgerError(
            "seed 台帳に %s / %s のブロックが 1 組だけ存在しなければならない"
            % (begin, end))
    return start + len(begin), stop


def _block_entries(text, begin, end):
    lo, hi = _block(text, begin, end)
    entries = []
    for line in text[lo:hi].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise SeedLedgerError("台帳の多人数欄に JSON でない行: %r (%s)"
                                  % (line[:80], exc))
        if not isinstance(entry, dict):
            raise SeedLedgerError("台帳の多人数欄の行は object: %r" % line[:80])
        for key in ("group", "lo", "hi"):
            if key not in entry:
                raise SeedLedgerError("台帳の行に %s が無い: %r" % (key, line[:80]))
        if type(entry["lo"]) is not int or type(entry["hi"]) is not int or \
                entry["hi"] < entry["lo"] or entry["group"] not in GROUPS:
            raise SeedLedgerError("台帳の帯が不正: %r" % line[:80])
        entries.append(entry)
    return entries


def read_multi_ledger(path=None):
    """(text, reserved, consumed, two_seat_bands) を返す。

    two_seat_bands は 2 人戦の消費済み/予約済み dev 帯(promote.py と同じ
    解釈): CONSUMED-DEV-BANDS、昇格第1回 7800、固定ベンチ 9000-9799、
    第2回以降 10800 から 800 刻み(上限なし)。
    """
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    import promote  # 遅延 import(promote は本モジュールを import する)
    path = path or SEED_LEDGER_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise SeedLedgerError("seed 台帳を読めない(%s): %s" % (path, exc))
    reserved = _block_entries(text, MULTI_RESERVED_BEGIN, MULTI_RESERVED_END)
    consumed = _block_entries(text, MULTI_CONSUMED_BEGIN, MULTI_CONSUMED_END)
    try:
        dev_consumed = promote._parse_consumed_dev_bands(text)
    except ValueError as exc:
        raise SeedLedgerError(str(exc))
    stride = promote.FORMAL_DIRECT_ROUND_STRIDE
    two_seat = [("dev", lo, hi, "two-seat consumed (CONSUMED-DEV-BANDS)")
                for lo, hi in dev_consumed]
    two_seat.append(("dev", promote.FORMAL_DIRECT_START_INDEX,
                     promote.FORMAL_DIRECT_START_INDEX + stride - 1,
                     "two-seat promote round 1"))
    two_seat.append(("dev", promote.FORMAL_BENCH_START_INDEX,
                     promote.FORMAL_BENCH_START_INDEX + stride - 1,
                     "two-seat fixed benchmark"))
    # 台帳は「dev 10800 以降の 800 刻み」を上限なしで 2 人昇格に予約している
    # (eval-seeds.md Phase 3/4 節)。上限は付けない(D4「旧 2 人予約列を
    # 無断で消費しない」)。2026-10-09 の裁定(t_da064f08 comment)で多人数専用
    # dev 領域 MULTI_DEV_REGION だけを多人数側の登録対象として切り出した。
    # 2 人列そのもの(promote._next_reserved_direct_band)は変えない。
    region_lo, region_hi = MULTI_DEV_REGION
    two_seat.append((
        "dev", promote.FORMAL_DIRECT_SECOND_ROUND_START, region_lo - 1,
        "two-seat promote reserve sequence (dev 10800 + 800k, unbounded)"))
    two_seat.append((
        "dev", region_hi + 1, sys.maxsize,
        "two-seat promote reserve sequence (dev 10800 + 800k, unbounded)"))
    for lo, hi in TWO_SEAT_ACCEPT_CONSUMED:
        two_seat.append(("accept", lo, hi, "two-seat consumed (Phase 8 accept)"))
    return text, reserved, consumed, two_seat


def ledger_conflicts(group, lo, hi, reserved, consumed, two_seat,
                     ignore_id=None):
    """[lo, hi] と重なる予約・消費の一覧(空なら登録可能)。"""
    hits = []
    for g, a, b, why in two_seat:
        if g == group and a <= hi and lo <= b:
            hits.append("%s %d-%d (%s)" % (g, a, b, why))
    for kind, entries in (("multi-reserved", reserved),
                          ("multi-consumed", consumed)):
        for e in entries:
            if e.get("id") is not None and e.get("id") == ignore_id:
                continue
            if e["group"] == group and e["lo"] <= hi and lo <= e["hi"]:
                hits.append("%s %d-%d (%s %s)" % (
                    e["group"], e["lo"], e["hi"], kind,
                    e.get("id") or e.get("source")))
    return hits


def _insert_line(path, text, end_marker, line):
    pos = text.find(end_marker)
    new = text[:pos] + line + "\n" + text[pos:]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    os.replace(tmp, path)


def _model_shas(agents):
    """モデルを持つ agent(checkpoint:/strong:)の meta/weights SHA-256。

    heuristic/random だけの対戦は空 dict(既存の予約行と同じ形)。予約時と起動時に
    同じ関数で求めて照合するので、登録後にモデルを差し替えると claim が拒否する。
    """
    from naeco.ai.eval import harness
    out = {}
    for spec in agents:
        kind, path = harness.parse_multi_agent_spec(spec)
        if kind in harness.MULTI_MODEL_KINDS:
            out[spec] = harness.model_file_shas(path)
    return out


def build_reservation(reservation_id, num_players, matchups, group, lo, hi,
                      purpose, threshold):
    """予約行を作る。matchups は同帯で許す対戦(正準 agent リスト)の事前固定列。

    D4「同帯使用は事前固定した対戦間だけ」: 1 帯に複数の対戦を登録でき、
    各対戦は 1 回だけ起動できる。登録外の対戦は同帯で起動できない。
    """
    from naeco.ai.eval import harness
    if not reservation_id or not isinstance(reservation_id, str):
        raise SeedLedgerError("reservation id must be a non-empty string")
    clusters = hi - lo + 1
    matchups = [list(m) for m in matchups]
    if not matchups:
        raise SeedLedgerError("at least one matchup is required")
    if len(set(tuple(m) for m in matchups)) != len(matchups):
        raise SeedLedgerError("duplicate matchup in reservation")
    for agents in matchups:
        harness.multi_conditions(num_players, agents, group, lo, clusters,
                                 purpose, threshold)
    if group in ("dev", "accept") and clusters != harness.D4_CLUSTERS:
        raise SeedLedgerError(
            "formal (%s) multi-seat judgements use exactly %d clusters (D4)"
            % (group, harness.D4_CLUSTERS))
    return {
        "id": reservation_id,
        "num_players": num_players,
        "group": group,
        "lo": lo,
        "hi": hi,
        "purpose": purpose,
        "matchups": matchups,
        "threshold": threshold,
        "model_sha256": dict((" ".join(m), _model_shas(m)) for m in matchups),
        "power": harness.d4_power_record(threshold, clusters)
        if threshold is not None else None,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_hold(reservation_id, num_players, group, lo, hi, purpose, threshold):
    """帯だけを先に押さえる予約(matchups 空・起動不可)。

    D4 の seed 計画は C3 で帯を登録し、対戦相手とモデル SHA は C6 で凍結する。
    hold は claim できない(登録済み対戦が無いため)。対戦の確定は
    register_multi が同じ id・帯・人数・用途・閾値のときだけ置き換えで受け付ける。
    """
    from naeco.ai.eval import harness
    if not reservation_id or not isinstance(reservation_id, str):
        raise SeedLedgerError("reservation id must be a non-empty string")
    if type(num_players) is not int or \
            num_players not in harness.MULTI_PLAYER_COUNTS:
        raise SeedLedgerError("num_players must be 3 or 4")
    if group not in GROUPS:
        raise SeedLedgerError("unknown seed group: %r" % (group,))
    if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo:
        raise SeedLedgerError("invalid band %r-%r" % (lo, hi))
    clusters = hi - lo + 1
    if group in ("dev", "accept") and clusters != harness.D4_CLUSTERS:
        raise SeedLedgerError(
            "formal (%s) multi-seat judgements use exactly %d clusters (D4)"
            % (group, harness.D4_CLUSTERS))
    return {
        "id": reservation_id,
        "num_players": num_players,
        "group": group,
        "lo": lo,
        "hi": hi,
        "purpose": purpose,
        "matchups": [],
        "threshold": threshold,
        "model_sha256": {},
        "power": harness.d4_power_record(threshold, clusters)
        if threshold is not None else None,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def register_multi(entry, ledger_path=None):
    """D4 条件と検出力を実行前に予約欄へ登録する。重複は拒否する。"""
    path = ledger_path or SEED_LEDGER_PATH
    text, reserved, consumed, two_seat = read_multi_ledger(path)
    if set(entry) != set(_RESERVATION_KEYS):
        raise SeedLedgerError("reservation keys must be %s" %
                              (sorted(_RESERVATION_KEYS),))
    line = json.dumps(entry, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))
    same = [e for e in reserved if e.get("id") == entry["id"]]
    if same:
        # 帯だけを先に押さえた hold 行(matchups 空)は、同じ帯・人数・用途・
        # 閾値のまま対戦とモデルを確定する形でだけ置き換えられる(C6 用)。
        hold = same[0]
        fixed = ("num_players", "group", "lo", "hi", "purpose", "threshold")
        if hold.get("protocol") is not None:
            raise SeedLedgerError(
                "reservation %s is a %s row; freeze it with the phase12b "
                "command" % (entry["id"], hold.get("protocol")))
        if len(same) != 1 or hold.get("matchups") != [] or \
                any(hold.get(k) != entry[k] for k in fixed):
            raise SeedLedgerError(
                "reservation id already exists: %s" % entry["id"])
        if any(c.get("id") == entry["id"] for c in consumed):
            raise SeedLedgerError("hold %s was already consumed" % entry["id"])
        hits = ledger_conflicts(entry["group"], entry["lo"], entry["hi"],
                                reserved, consumed, two_seat,
                                ignore_id=entry["id"])
        if hits:
            raise SeedLedgerError("hold now overlaps: %s" % "; ".join(hits))
        old = json.dumps(hold, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"))
        lo, hi = _block(text, MULTI_RESERVED_BEGIN, MULTI_RESERVED_END)
        body = text[lo:hi]
        if body.count(old) != 1:
            raise SeedLedgerError("hold line for %s is not canonical JSON; "
                                  "refusing to rewrite" % entry["id"])
        new = text[:lo] + body.replace(old, line) + text[hi:]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new)
        os.replace(tmp, path)
        return entry
    hits = ledger_conflicts(entry["group"], entry["lo"], entry["hi"],
                            reserved, consumed, two_seat)
    if hits:
        raise SeedLedgerError(
            "band %s %d-%d overlaps consumed/reserved bands: %s"
            % (entry["group"], entry["lo"], entry["hi"], "; ".join(hits)))
    _insert_line(path, text, MULTI_RESERVED_END, line)
    return entry


def claim_multi(reservation_id, conditions, run_id, resume=False,
                ledger_path=None):
    """予約と条件を照合し、起動前に消費欄へ記録する(中断しても消費扱い)。

    resume=True は同じ run_id の消費記録が既にあるときだけ許す(欠けた局の
    決定論的再開)。別 run_id での再使用・未予約帯・条件不一致は拒否する。
    """
    path = ledger_path or SEED_LEDGER_PATH
    text, reserved, consumed, two_seat = read_multi_ledger(path)
    matches = [e for e in reserved if e.get("id") == reservation_id]
    if len(matches) != 1:
        raise SeedLedgerError("no unique reservation %r in the ledger"
                              % reservation_id)
    res = matches[0]
    if res.get("protocol") is not None:
        # Phase 12b の行(phase12b-multi/v1)を旧 D4 経路で起動させない。
        raise SeedLedgerError(
            "reservation %s belongs to protocol %s; the Phase 10 D4 runner "
            "refuses it" % (reservation_id, res.get("protocol")))
    agents = list(conditions["agents"])
    from naeco.ai.eval import harness
    expect = harness.conditions_registration_view(conditions)
    for key, value in expect.items():
        if res.get(key) != value:
            raise SeedLedgerError(
                "run conditions differ from the registration %s: %s=%r "
                "(registered %r)" % (reservation_id, key, value, res.get(key)))
    if agents not in res.get("matchups", []):
        raise SeedLedgerError(
            "matchup %r is not pre-registered for %s (registered %r)"
            % (agents, reservation_id, res.get("matchups")))
    key = " ".join(agents)
    if (res.get("model_sha256") or {}).get(key) != _model_shas(agents):
        raise SeedLedgerError("model SHA-256 differs from the registration")
    hits = ledger_conflicts(res["group"], res["lo"], res["hi"], reserved,
                            [], two_seat, ignore_id=reservation_id)
    if hits:
        raise SeedLedgerError("registered band now overlaps: %s"
                              % "; ".join(hits))
    claims = [e for e in consumed
              if e["group"] == res["group"] and e["lo"] <= res["hi"]
              and res["lo"] <= e["hi"]]
    foreign = [c for c in claims if c.get("id") != reservation_id]
    if foreign:
        raise SeedLedgerError(
            "band overlaps consumption of another reservation: %s"
            % ", ".join(str(c.get("id")) for c in foreign))
    mine = [c for c in claims if c.get("matchup") == agents]
    if resume:
        if len(mine) != 1 or mine[0].get("run_id") != run_id:
            raise SeedLedgerError(
                "resume requires an existing consumption of the same run_id")
        return mine[0]
    if mine:
        raise SeedLedgerError(
            "matchup %r on %s %d-%d is already consumed (run %s); a launched "
            "band counts as consumed even if it failed. Resume the same run or "
            "register an unused band." % (agents, res["group"], res["lo"],
                                          res["hi"], mine[0].get("run_id")))
    claim = {"id": reservation_id, "group": res["group"], "lo": res["lo"],
             "hi": res["hi"], "matchup": agents, "run_id": run_id,
             "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime())}
    _insert_line(path, text, MULTI_CONSUMED_END,
                 json.dumps(claim, sort_keys=True, ensure_ascii=True,
                            separators=(",", ":")))
    return claim


def _sha256_path(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_multi(reservation_id, num_players, agents, group, start_index,
              clusters, purpose, threshold, records_path, summary_path,
              workers=1, resume=False, ledger_path=None):
    """登録済み帯で巡回クラスタ評価を実行し、records と summary を保存する。"""
    from naeco.ai.eval import harness
    conditions = harness.multi_conditions(
        num_players, agents, group, start_index, clusters, purpose, threshold)
    run_id = harness.multi_run_id(conditions)
    if not resume:
        for p in (records_path, summary_path):
            if os.path.exists(p):
                raise SeedLedgerError(
                    "output already exists (a judgement is run once): %s" % p)
    elif os.path.exists(summary_path):
        raise SeedLedgerError("summary already exists; nothing to resume: %s"
                              % summary_path)
    ledger_file = ledger_path or SEED_LEDGER_PATH
    ledger_sha_before = _sha256_path(ledger_file)
    claim = claim_multi(reservation_id, conditions, run_id, resume=resume,
                        ledger_path=ledger_path)
    existing = []
    if resume and os.path.exists(records_path):
        # 中断で途中まで書かれた最終行(改行なし)だけを捨てる。完了局は残す。
        with open(records_path, "rb") as fh:
            raw = fh.read()
        if raw and not raw.endswith(b"\n"):
            with open(records_path, "wb") as fh:
                fh.write(raw[:raw.rfind(b"\n") + 1])
        existing = harness.read_multi_records(records_path)
        if any(r.get("run_id") != run_id for r in existing):
            raise SeedLedgerError("records belong to another run_id")
    done = set((r["index"], r["seat_variant"]) for r in existing)
    tasks = harness.cyclic_tasks_for(conditions, run_id, skip=done)
    directory = os.path.dirname(os.path.abspath(records_path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    started = time.monotonic()
    # 1 局ごとに追記(中断しても完了局は残る)→ 最後に (index,k) 順で正準化。
    with open(records_path, "a", encoding="utf-8", newline="\n") as fh:
        if workers == 1:
            for task in tasks:
                fh.write(harness.record_line(harness._run_cyclic_task(task)))
                fh.flush()
        else:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=workers)
            try:
                for rec in pool.imap_unordered(harness._run_cyclic_task, tasks,
                                               chunksize=1):
                    fh.write(harness.record_line(rec))
                    fh.flush()
            finally:
                pool.close()
                pool.join()
    records = harness.read_multi_records(records_path)
    harness.write_multi_records(records, records_path)
    # 判定前にモデル SHA を再検査する(途中差し替えは判定を出さない)。
    harness.verify_models_unchanged(conditions)
    summary = harness.summarize_multi(records, conditions, run_id,
                                      threshold=threshold)
    summary["conditions"] = conditions
    summary["reservation_id"] = reservation_id
    summary["power"] = harness.d4_power_record(threshold, clusters)
    summary["records_path"] = os.path.relpath(
        os.path.abspath(records_path), os.path.dirname(
            os.path.abspath(summary_path)))
    summary["records_sha256"] = _sha256_path(records_path)
    summary["formal"] = group in ("dev", "accept")
    summary["provenance"] = {
        "claim": claim,
        "seed_ledger_sha256_before_claim": ledger_sha_before,
        "resumed": bool(resume),
        "resumed_games": len(tasks) if resume else 0,
        "workers": workers,
        "wall_seconds": round(time.monotonic() - started, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(summary_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, sort_keys=True, ensure_ascii=True, indent=1)
        fh.write("\n")
    return summary


# 保存物から再計算したとき一致を要求する集計キー(実行環境の記録は除く)。
RECOMPUTED_KEYS = (
    "schema", "protocol", "run_id", "num_players", "agents", "group",
    "start_index", "end_index", "clusters", "games", "wins", "draws", "losses",
    "point_estimate", "ci_low", "ci_high", "bootstrap_samples",
    "bootstrap_seed", "threshold", "passed", "baseline_1_over_n",
    "target_rank_histogram", "target_mean_rank", "target_mean_score",
    "target_wins_by_seat", "cluster_results", "limitations",
)


def recompute_multi(summary_path):
    """summary JSON と records JSONL だけから集計・CI・合否を再計算して照合する。"""
    from naeco.ai.eval import harness
    with open(summary_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    records_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                                saved["records_path"])
    if _sha256_path(records_path) != saved["records_sha256"]:
        raise ValueError("records SHA-256 differs from the summary")
    conditions = saved["conditions"]
    if harness.multi_run_id(conditions) != saved["run_id"]:
        raise ValueError("run_id does not match the saved conditions")
    records = harness.read_multi_records(records_path)
    fresh = harness.summarize_multi(records, conditions, saved["run_id"],
                                    threshold=conditions["threshold"])
    diffs = [k for k in RECOMPUTED_KEYS if fresh[k] != saved[k]]
    if diffs:
        raise ValueError("recomputed summary differs at %s" % diffs)
    return fresh


def replay_multi(summary_path, workers=1):
    """保存済み条件から全局を再実行し、保存 records と byte 一致を検査する。

    同じ run_id の決定論的再実行(新しい判定ではない)なので台帳に触れない。
    worker 数を変えて呼べば worker=1/4 の一致検査になる。
    """
    from naeco.ai.eval import harness
    with open(summary_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    records_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                                saved["records_path"])
    with open(records_path, "rb") as fh:
        saved_bytes = fh.read()
    cond = saved["conditions"]
    run_id = harness.multi_run_id(cond)
    if run_id != saved["run_id"]:
        raise ValueError("run_id does not match the saved conditions")
    tasks = harness.cyclic_tasks_for(cond, run_id)
    records = harness.run_cyclic_games(tasks, workers=workers)
    fresh = "".join(harness.record_line(r) for r in records).encode("utf-8")
    return fresh == saved_bytes, hashlib.sha256(fresh).hexdigest()


def main_multi(args):
    num_players = args.num_players
    agents = [args.left, args.right] + list(args.more)
    if args.replay:
        same, digest = replay_multi(args.replay, workers=args.workers)
        print("replayed %s with workers=%d: records sha256=%s byte_equal=%s"
              % (args.replay, args.workers, digest, same))
        return 0 if same else 1
    if args.hold:
        if args.clusters is None or args.reservation is None or \
                args.purpose is None:
            raise SystemExit("--hold needs --clusters, --reservation, --purpose")
        lo = args.start_index
        hi = lo + args.clusters - 1
        try:
            entry = build_hold(args.reservation, num_players, args.group, lo,
                               hi, args.purpose, args.hold_threshold)
            _t, reserved, consumed, two_seat = read_multi_ledger(args.ledger)
            hits = ledger_conflicts(args.group, lo, hi, reserved, consumed,
                                    two_seat)
            if not args.execute:
                print("check %s %d-%d: %s" % (
                    args.group, lo, hi,
                    "free" if not hits else "CONFLICT " + "; ".join(hits)))
                return 0 if not hits else 3
            register_multi(entry, ledger_path=args.ledger)
        except (SeedLedgerError, ValueError) as exc:
            raise SystemExit("refused: %s" % exc)
        print("held: %s" % json.dumps(entry, sort_keys=True))
        return 0
    if args.recompute:
        fresh = recompute_multi(args.recompute)
        print("recomputed from saved records only: %s" % args.recompute)
        print("n=%d clusters=%d games=%d wins=%d point=%.4f CI=[%.4f, %.4f] "
              "passed=%s" % (fresh["games"] // fresh["clusters"],
                             fresh["clusters"],
                             fresh["games"], fresh["wins"],
                             fresh["point_estimate"], fresh["ci_low"],
                             fresh["ci_high"], fresh["passed"]))
        return 0
    if len(agents) != num_players:
        raise SystemExit("--num-players %d needs %d agent specs (target first)"
                         % (num_players, num_players))
    if args.clusters is None or args.reservation is None or \
            args.purpose is None:
        raise SystemExit("--clusters, --reservation and --purpose are required "
                         "for n>=3")
    lo = args.start_index
    hi = lo + args.clusters - 1
    if args.matchup and not args.register:
        raise SystemExit("--matchup is only used with --register")
    if args.register:
        matchups = [agents] + [m.split(",") for m in args.matchup]
        try:
            entry = build_reservation(args.reservation, num_players, matchups,
                                      args.group, lo, hi, args.purpose,
                                      args.threshold)
            if args.execute:
                register_multi(entry, ledger_path=args.ledger)
                print("registered: %s" % json.dumps(entry, sort_keys=True))
            else:
                _t, reserved, consumed, two_seat = read_multi_ledger(
                    args.ledger)
                hits = ledger_conflicts(args.group, lo, hi, reserved, consumed,
                                        two_seat)
                print("check %s %d-%d: %s" % (
                    args.group, lo, hi,
                    "free" if not hits else "CONFLICT " + "; ".join(hits)))
                print("power: %s" % json.dumps(entry["power"], sort_keys=True))
                return 0 if not hits else 3
        except (SeedLedgerError, ValueError) as exc:
            raise SystemExit("refused: %s" % exc)
        return 0
    if not args.execute:
        from naeco.ai.eval import harness
        try:
            cond = harness.multi_conditions(
                num_players, agents, args.group, lo, args.clusters,
                args.purpose, args.threshold)
        except ValueError as exc:
            raise SystemExit("refused: %s" % exc)
        print("prepared: n=%d %s group=%s %d-%d clusters=%d workers=%d "
              "reservation=%s run_id=%s models=%s"
              % (num_players, agents, args.group, lo, hi, args.clusters,
                 args.workers, args.reservation, harness.multi_run_id(cond),
                 json.dumps(cond.get("models") or {}, sort_keys=True)))
        return 0
    if not args.records or not args.summary:
        raise SystemExit("--records and --summary are required for n>=3")
    try:
        summary = run_multi(
            args.reservation, num_players, agents, args.group, lo,
            args.clusters, args.purpose, args.threshold, args.records,
            args.summary, workers=args.workers, resume=args.resume,
            ledger_path=args.ledger)
    except (SeedLedgerError, ValueError) as exc:
        raise SystemExit("refused: %s" % exc)
    print("n=%d %s group=%s index=%d-%d clusters=%d games=%d"
          % (num_players, agents, args.group, lo, hi, summary["clusters"],
             summary["games"]))
    print("target wins=%d point=%.4f 95%% CI=[%.4f, %.4f] (1/n=%.4f)"
          % (summary["wins"], summary["point_estimate"], summary["ci_low"],
             summary["ci_high"], summary["baseline_1_over_n"]))
    print("rank histogram=%s mean rank=%.3f wins by seat=%s"
          % (summary["target_rank_histogram"], summary["target_mean_rank"],
             summary["target_wins_by_seat"]))
    if args.threshold is None:
        print("benchmark point estimate=%.4f; no CI threshold (D5)"
              % summary["point_estimate"])
        return 0
    print("pass (CI low > %.4f): %s" % (args.threshold, summary["passed"]))
    return 0 if summary["passed"] else 1


def _parse_threshold(value):
    """Keep the numeric default; thresholdless benches require explicit 'none'."""
    return None if value == "none" else float(value)


# ---------------------------------------------------------------------------
# Phase 12b: `phase12b-multi/v1`(opt-in。docs/phase12b-protocol.md)。
#
# 旧 D4(上の run_multi / build_hold、800 クラスタ・95%)と n=2 は無変更。
# 予約・消費行は同じ MULTI-SEAT 欄に置き(帯の重複検査を共有する)、行の
# `protocol` 欄で区別する。旧経路は protocol 付きの行を拒否し、本経路は
# protocol の無い行を拒否する。入口は `eval_agent.py phase12b <command>`。
# ---------------------------------------------------------------------------

P12B_ROW_KEYS = ("id", "protocol", "num_players", "group", "lo", "hi", "kind",
                 "purpose", "gates", "search", "power", "r4_sensitivity",
                 "matchups", "model_sha256", "frozen", "registered_at",
                 "frozen_at")

# 設計 D1 の dev/accept hold(実行前登録。帯・用途・判定を固定し対戦は未凍結)。
P12B_HOLDS = (
    ("p12b-dev-g-3p", 3, "dev", 33200, 34199, "G", "phase12b-dev-G1G2-3p"),
    ("p12b-dev-g-4p", 4, "dev", 34200, 35199, "G", "phase12b-dev-G1G2-4p"),
    ("p12b-dev-r4-3p", 3, "dev", 35200, 36199, "R4", "phase12b-dev-R4-3p"),
    ("p12b-dev-r4-4p", 4, "dev", 36200, 37199, "R4", "phase12b-dev-R4-4p"),
    ("p12b-dev-ex-3p", 3, "dev", 37200, 38199, "EXPLOIT",
     "phase12b-dev-exploit-3p"),
    ("p12b-dev-ex-4p", 4, "dev", 38200, 39199, "EXPLOIT",
     "phase12b-dev-exploit-4p"),
    ("p12b-accept-3p", 3, "accept", 2400, 3399, "ACCEPT",
     "phase12b-accept-G1G2R4-3p"),
    ("p12b-accept-4p", 4, "accept", 3400, 4399, "ACCEPT",
     "phase12b-accept-G1G2R4-4p"),
)


def _p12b():
    from naeco.ai.eval import phase12b
    return phase12b


def _p12b_matchup_key(m):
    return "%s|%s|%s" % (m["role"], m["challenger"], ",".join(m["opponents"]))


def build_p12b_hold(reservation_id, num_players, group, lo, hi, kind, purpose):
    """phase12b-multi/v1 の hold 行(対戦未凍結。claim できない)。"""
    p = _p12b()
    from naeco.ai.eval import harness
    if not reservation_id or not isinstance(reservation_id, str):
        raise SeedLedgerError("reservation id must be a non-empty string")
    try:
        gates = p.gates_for(kind, num_players)
    except p.ProtocolError as exc:
        raise SeedLedgerError(str(exc))
    if group not in GROUPS:
        raise SeedLedgerError("unknown seed group: %r" % (group,))
    if type(lo) is not int or type(hi) is not int or lo < 0 or hi < lo:
        raise SeedLedgerError("invalid band %r-%r" % (lo, hi))
    clusters = hi - lo + 1
    if group in p.FORMAL_GROUPS:
        if clusters != p.FORMAL_CLUSTERS:
            raise SeedLedgerError("formal (%s) phase12b holds use exactly %d "
                                  "clusters" % (group, p.FORMAL_CLUSTERS))
        if kind == "SMOKE":
            raise SeedLedgerError("SMOKE holds are train only")
        if group == "dev" and kind == "ACCEPT":
            raise SeedLedgerError("ACCEPT holds are accept only")
        if group == "accept" and kind != "ACCEPT":
            raise SeedLedgerError("accept bands take the ACCEPT kind")
        region_lo, region_hi = MULTI_DEV_REGION
        if group == "dev" and not (region_lo <= lo and hi <= region_hi):
            raise SeedLedgerError("dev band %d-%d is outside the multi-seat "
                                  "dev region %d-%d" % (lo, hi, region_lo,
                                                        region_hi))
    elif kind != "SMOKE":
        raise SeedLedgerError("train bands take the SMOKE kind only")
    return {
        "id": reservation_id,
        "protocol": p.PROTOCOL,
        "num_players": num_players,
        "group": group,
        "lo": lo,
        "hi": hi,
        "kind": kind,
        "purpose": purpose,
        "gates": gates,
        "search": dict(harness.STRONG_SEARCH),
        "power": p.g_power_record(num_players, clusters),
        "r4_sensitivity": p.r4_sensitivity_record(clusters),
        "matchups": [],
        "model_sha256": {},
        "frozen": False,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_at": None,
    }


def _p12b_rows(reserved, reservation_id):
    rows = [e for e in reserved if e.get("id") == reservation_id]
    if len(rows) != 1:
        raise SeedLedgerError("no unique reservation %r in the ledger"
                              % reservation_id)
    row = rows[0]
    if row.get("protocol") != _p12b().PROTOCOL:
        raise SeedLedgerError(
            "reservation %s is not a %s row (old protocol rows are not mixed)"
            % (reservation_id, _p12b().PROTOCOL))
    return row


def _write_ledger(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _canonical_line(entry):
    return json.dumps(entry, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))


def register_p12b(entry, ledger_path=None):
    """hold 行を予約欄へ登録する。帯の重複(旧 2 人列・Phase 10・既存行)は拒否。"""
    path = ledger_path or SEED_LEDGER_PATH
    text, reserved, consumed, two_seat = read_multi_ledger(path)
    if set(entry) != set(P12B_ROW_KEYS):
        raise SeedLedgerError("phase12b row keys must be %s"
                              % (sorted(P12B_ROW_KEYS),))
    if entry["frozen"] or entry["matchups"]:
        raise SeedLedgerError("register a hold first; freeze it separately")
    if any(e.get("id") == entry["id"] for e in reserved + consumed):
        raise SeedLedgerError("reservation id already exists: %s" % entry["id"])
    hits = ledger_conflicts(entry["group"], entry["lo"], entry["hi"], reserved,
                            consumed, two_seat)
    if hits:
        raise SeedLedgerError("band %s %d-%d overlaps consumed/reserved bands: "
                              "%s" % (entry["group"], entry["lo"], entry["hi"],
                                      "; ".join(hits)))
    _insert_line(path, text, MULTI_RESERVED_END, _canonical_line(entry))
    return entry


def freeze_p12b(reservation_id, matchups, ledger_path=None):
    """hold の対戦列・モデル SHA・探索構成を一度だけ凍結する(帯/判定は不変)。

    matchups は [{"role", "challenger", "opponents"}]。役割は hold の kind が
    許すものだけ、形は phase12b.conditions が検査する。凍結済み・消費済みは拒否。
    """
    p = _p12b()
    path = ledger_path or SEED_LEDGER_PATH
    text, reserved, consumed, two_seat = read_multi_ledger(path)
    hold = _p12b_rows(reserved, reservation_id)
    if hold["frozen"] or hold["matchups"]:
        raise SeedLedgerError("reservation %s is already frozen (once only)"
                              % reservation_id)
    if any(c.get("id") == reservation_id for c in consumed):
        raise SeedLedgerError("reservation %s was already consumed"
                              % reservation_id)
    clean = []
    for m in matchups:
        if set(m) != {"role", "challenger", "opponents"}:
            raise SeedLedgerError("matchup keys are role/challenger/opponents")
        m = {"role": m["role"], "challenger": m["challenger"],
             "opponents": list(m["opponents"])}
        if m["role"] not in p.HOLD_KINDS[hold["kind"]]:
            raise SeedLedgerError("role %s is not allowed on a %s hold"
                                  % (m["role"], hold["kind"]))
        try:
            p.conditions_from_hold(hold, m["role"], m["challenger"],
                                   m["opponents"])
        except ValueError as exc:
            raise SeedLedgerError("matchup %r refused: %s" % (m, exc))
        clean.append(m)
    if not clean:
        raise SeedLedgerError("freeze needs at least one matchup")
    keys = [_p12b_matchup_key(m) for m in clean]
    if len(set(keys)) != len(keys):
        raise SeedLedgerError("duplicate matchup")
    hits = ledger_conflicts(hold["group"], hold["lo"], hold["hi"], reserved,
                            consumed, two_seat, ignore_id=reservation_id)
    if hits:
        raise SeedLedgerError("hold now overlaps: %s" % "; ".join(hits))
    frozen = dict(hold)
    frozen["matchups"] = clean
    frozen["model_sha256"] = dict(
        (k, _model_shas([m["challenger"]] + m["opponents"]))
        for k, m in zip(keys, clean))
    frozen["frozen"] = True
    frozen["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    old = _canonical_line(hold)
    lo, hi = _block(text, MULTI_RESERVED_BEGIN, MULTI_RESERVED_END)
    body = text[lo:hi]
    if body.count(old) != 1:
        raise SeedLedgerError("hold line for %s is not canonical JSON; refusing "
                              "to rewrite" % reservation_id)
    _write_ledger(path, text[:lo] + body.replace(old, _canonical_line(frozen))
                  + text[hi:])
    return frozen


def p12b_conditions_for(reservation_id, role, challenger, opponents,
                        ledger_path=None):
    """台帳の行から条件を作る(帯・人数・用途は行が正本。引数で変えられない)。"""
    p = _p12b()
    unused, reserved, unused_c, unused_t = read_multi_ledger(ledger_path)
    row = _p12b_rows(reserved, reservation_id)
    return p.conditions_from_hold(row, role, challenger, opponents)


def p12b_decision_holds(reservation_ids, ledger_path=None):
    """decide が照合する凍結 hold 行(台帳が正本)。未指定は空(判定側で拒否)。"""
    unused, reserved, unused_c, unused_t = read_multi_ledger(ledger_path)
    return [_p12b_rows(reserved, rid) for rid in reservation_ids]


def claim_p12b(cond, run_id, resume=False, ledger_path=None):
    """凍結済み行の登録対戦だけを起動前に消費欄へ記録する(未凍結 hold は拒否)。"""
    p = _p12b()
    path = ledger_path or SEED_LEDGER_PATH
    text, reserved, consumed, two_seat = read_multi_ledger(path)
    rid = cond["reservation_id"]
    row = _p12b_rows(reserved, rid)
    if not row["frozen"] or not row["matchups"]:
        raise SeedLedgerError(
            "reservation %s is an unfrozen hold (no frozen models/matchups); "
            "refusing to launch" % rid)
    if cond.get("protocol") != p.PROTOCOL:
        raise SeedLedgerError("conditions are not %s" % p.PROTOCOL)
    for key, value in p.hold_view(row).items():
        if cond.get(key) != value:
            raise SeedLedgerError("run conditions differ from %s at %s (%r vs "
                                  "registered %r)" % (rid, key, cond.get(key),
                                                      value))
    m = {"role": cond["role"], "challenger": cond["challenger"],
         "opponents": list(cond["opponents"])}
    if m not in row["matchups"]:
        raise SeedLedgerError("matchup %r is not frozen for %s" % (m, rid))
    key = _p12b_matchup_key(m)
    if row["model_sha256"].get(key) != \
            _model_shas([m["challenger"]] + m["opponents"]):
        raise SeedLedgerError("model SHA-256 differs from the frozen hold")
    if (cond.get("models") or {}) and dict(
            (s, v["sha256"]) for s, v in cond["models"].items()) != \
            row["model_sha256"][key]:
        raise SeedLedgerError("conditions carry other model SHA-256")
    hits = ledger_conflicts(row["group"], row["lo"], row["hi"], reserved, [],
                            two_seat, ignore_id=rid)
    if hits:
        raise SeedLedgerError("registered band now overlaps: %s"
                              % "; ".join(hits))
    claims = [c for c in consumed if c["group"] == row["group"]
              and c["lo"] <= row["hi"] and row["lo"] <= c["hi"]]
    foreign = [c for c in claims if c.get("id") != rid]
    if foreign:
        raise SeedLedgerError("band overlaps consumption of another "
                              "reservation: %s"
                              % ", ".join(str(c.get("id")) for c in foreign))
    mine = [c for c in claims if c.get("matchup_key") == key]
    if resume:
        if len(mine) != 1 or mine[0].get("run_id") != run_id:
            raise SeedLedgerError("resume requires an existing consumption of "
                                  "the same run_id")
        return mine[0]
    if mine:
        raise SeedLedgerError(
            "matchup %s on %s is already consumed (run %s); a launched band "
            "counts as consumed even if it failed" % (key, rid,
                                                      mine[0].get("run_id")))
    claim = {"id": rid, "protocol": p.PROTOCOL, "group": row["group"],
             "lo": row["lo"], "hi": row["hi"], "matchup_key": key,
             "matchup": [m["challenger"]] + m["opponents"],
             "role": m["role"], "run_id": run_id,
             "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _insert_line(path, text, MULTI_CONSUMED_END, _canonical_line(claim))
    return claim


def run_p12b(reservation_id, role, challenger, opponents, records_path,
             summary_path, workers=1, resume=False, ledger_path=None):
    """凍結済み対戦を 1 回だけ実行し、records と summary(95%/97.5%)を保存する。"""
    p = _p12b()
    from naeco.ai.eval import harness
    cond = p12b_conditions_for(reservation_id, role, challenger, opponents,
                               ledger_path=ledger_path)
    run_id = p.run_id_for(cond)
    if not resume:
        for path in (records_path, summary_path):
            if os.path.exists(path):
                raise SeedLedgerError("output already exists (a judgement is "
                                      "run once): %s" % path)
    elif os.path.exists(summary_path):
        raise SeedLedgerError("summary already exists; nothing to resume: %s"
                              % summary_path)
    ledger_file = ledger_path or SEED_LEDGER_PATH
    ledger_sha_before = _sha256_path(ledger_file)
    claim = claim_p12b(cond, run_id, resume=resume, ledger_path=ledger_path)
    existing = []
    if resume and os.path.exists(records_path):
        with open(records_path, "rb") as fh:
            raw = fh.read()
        if raw and not raw.endswith(b"\n"):
            with open(records_path, "wb") as fh:
                fh.write(raw[:raw.rfind(b"\n") + 1])
        existing = harness.read_multi_records(records_path)
        if any(r.get("run_id") != run_id for r in existing):
            raise SeedLedgerError("records belong to another run_id")
    done = set((r["index"], r["seat_variant"]) for r in existing)
    tasks = p.tasks_for(cond, run_id, skip=done)
    directory = os.path.dirname(os.path.abspath(records_path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    started = time.monotonic()
    with open(records_path, "a", encoding="utf-8", newline="\n") as fh:
        if workers == 1:
            for task in tasks:
                fh.write(harness.record_line(p.run_task(task)))
                fh.flush()
        else:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=workers)
            try:
                for rec in pool.imap_unordered(p.run_task, tasks, chunksize=1):
                    fh.write(harness.record_line(rec))
                    fh.flush()
            finally:
                pool.close()
                pool.join()
    records = harness.read_multi_records(records_path)
    harness.write_multi_records(records, records_path)
    harness.verify_models_unchanged(cond)
    summary = p.summarize(records, cond, run_id)
    summary["conditions"] = cond
    summary["records_path"] = os.path.relpath(
        os.path.abspath(records_path),
        os.path.dirname(os.path.abspath(summary_path)))
    summary["records_sha256"] = _sha256_path(records_path)
    summary["formal"] = cond["group"] in p.FORMAL_GROUPS
    summary["provenance"] = {
        "claim": claim,
        "seed_ledger_sha256_before_claim": ledger_sha_before,
        "resumed": bool(resume),
        "resumed_games": len(tasks) if resume else 0,
        "workers": workers,
        "wall_seconds": round(time.monotonic() - started, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p.write_json(summary_path, summary)
    return summary


def replay_p12b(summary_path, workers=1):
    """保存条件から全局を再実行し保存 records と byte 照合(台帳に触れない)。"""
    p = _p12b()
    from naeco.ai.eval import harness
    saved = p.read_json(summary_path)
    records_path = os.path.join(os.path.dirname(os.path.abspath(summary_path)),
                                saved["records_path"])
    with open(records_path, "rb") as fh:
        saved_bytes = fh.read()
    cond = saved["conditions"]
    run_id = p.run_id_for(cond)
    if run_id != saved["run_id"]:
        raise ValueError("run_id does not match the saved conditions")
    records = p.run_games(p.tasks_for(cond, run_id), workers=workers)
    fresh = "".join(harness.record_line(r) for r in records).encode("utf-8")
    return fresh == saved_bytes, hashlib.sha256(fresh).hexdigest()


PAIRED_RECOMPUTED_KEYS = (
    "schema", "protocol", "kind", "num_players", "group", "start_index",
    "end_index", "clusters", "run_a", "run_b", "challenger_a", "challenger_b",
    "opponents_a", "opponents_b", "p_a", "p_b", "point_estimate", "ci",
    "bootstrap_samples", "bootstrap_seed", "gate", "cluster_diffs")


def paired_p12b(kind, summary_a, summary_b, out_path=None):
    p = _p12b()
    result = p.paired_from_summaries(kind, summary_a, summary_b)
    if out_path:
        base = os.path.dirname(os.path.abspath(out_path))
        result["summary_a"] = os.path.relpath(os.path.abspath(summary_a), base)
        result["summary_b"] = os.path.relpath(os.path.abspath(summary_b), base)
        result["summary_a_sha256"] = _sha256_path(summary_a)
        result["summary_b_sha256"] = _sha256_path(summary_b)
        if os.path.exists(out_path):
            raise SeedLedgerError("output already exists: %s" % out_path)
        p.write_json(out_path, result)
    return result


def recompute_paired_p12b(path):
    """保存した paired 結果を元 summary/records だけから再計算して照合する。"""
    p = _p12b()
    saved = p.read_json(path)
    base = os.path.dirname(os.path.abspath(path))
    sa = os.path.join(base, saved["summary_a"])
    sb = os.path.join(base, saved["summary_b"])
    if _sha256_path(sa) != saved["summary_a_sha256"] or \
            _sha256_path(sb) != saved["summary_b_sha256"]:
        raise ValueError("paired source summary SHA-256 differs")
    fresh = p.paired_from_summaries(saved["kind"], sa, sb)
    diffs = [k for k in PAIRED_RECOMPUTED_KEYS if fresh[k] != saved.get(k)]
    if diffs:
        raise ValueError("recomputed paired result differs at %s" % diffs)
    return fresh


def main_p12b(argv):
    p = _p12b()
    parser = argparse.ArgumentParser(
        prog="eval_agent.py phase12b",
        description="Phase 12b phase12b-multi/v1 (1000 clusters, 97.5%% "
                    "judgement CI, paired R4/d_E)")
    sub = parser.add_subparsers(dest="command")
    sub.required = True
    hp = sub.add_parser("hold", help="register the D1 dev/accept holds "
                        "(--execute to write) or one --id band")
    hp.add_argument("--id", default=None)
    hp.add_argument("--num-players", type=int, choices=(3, 4))
    hp.add_argument("--group", choices=sorted(GROUPS))
    hp.add_argument("--lo", type=int)
    hp.add_argument("--hi", type=int)
    hp.add_argument("--kind", choices=sorted(p.HOLD_KINDS))
    hp.add_argument("--purpose")
    fp = sub.add_parser("freeze", help="freeze matchups of one hold (once)")
    fp.add_argument("--id", required=True)
    fp.add_argument("--matchups", required=True,
                    help="JSON file: [{role, challenger, opponents}]")
    rp = sub.add_parser("run", help="launch one frozen matchup (consumes)")
    rp.add_argument("--id", required=True)
    rp.add_argument("--role", required=True, choices=p.ROLES)
    rp.add_argument("challenger")
    rp.add_argument("opponents", nargs="+")
    rp.add_argument("--records", required=True)
    rp.add_argument("--summary", required=True)
    rp.add_argument("--workers", type=int, default=1)
    rp.add_argument("--resume", action="store_true")
    cp = sub.add_parser("recompute", help="recompute a summary or paired file")
    cp.add_argument("path")
    yp = sub.add_parser("replay", help="replay a run; byte-compare records")
    yp.add_argument("summary")
    yp.add_argument("--workers", type=int, default=1)
    pp = sub.add_parser("paired", help="paired d from two saved summaries")
    pp.add_argument("--kind", required=True, choices=("R4", "EXPLOIT"))
    pp.add_argument("--a", required=True)
    pp.add_argument("--b", required=True)
    pp.add_argument("--out", default=None)
    dp = sub.add_parser("decide", help="G1, G2->R4 AND from saved files")
    dp.add_argument("--num-players", type=int, required=True, choices=(3, 4))
    dp.add_argument("--g1", required=True)
    dp.add_argument("--g2", required=True)
    dp.add_argument("--r4", action="append", default=[])
    dp.add_argument("--exploit", action="append", default=[])
    dp.add_argument("--hold", action="append", default=[],
                    help="frozen hold id(s) of this judgement: dev = the G, "
                         "R4 and EXPLOIT holds, accept = the ACCEPT hold; the "
                         "inputs must match their frozen matchups exactly")
    dp.add_argument("--out", default=None)
    sub.add_parser("power", help="print the G1/G2 power and R4 sensitivity")
    for q in (hp, fp, rp):
        q.add_argument("--execute", action="store_true")
    for q in (hp, fp, rp, sub.choices["recompute"], dp):
        q.add_argument("--ledger", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "power":
            for n in p.PLAYER_COUNTS:
                print(json.dumps({"n": n, "G": p.g_power_record(n)},
                                 sort_keys=True))
            print(json.dumps({"R4": p.r4_sensitivity_record()},
                             sort_keys=True))
            return 0
        if args.command == "hold":
            if args.id is not None:
                rows = [(args.id, args.num_players, args.group, args.lo,
                         args.hi, args.kind, args.purpose)]
            else:
                rows = P12B_HOLDS
            entries = [build_p12b_hold(*row) for row in rows]
            unused, reserved, consumed, two_seat = read_multi_ledger(
                args.ledger)
            status = 0
            for e in entries:
                hits = ledger_conflicts(e["group"], e["lo"], e["hi"], reserved,
                                        consumed, two_seat)
                print("check %s %s %d-%d: %s" % (
                    e["id"], e["group"], e["lo"], e["hi"],
                    "free" if not hits else "CONFLICT " + "; ".join(hits)))
                status = status or (3 if hits else 0)
            if status or not args.execute:
                return status
            for e in entries:
                register_p12b(e, ledger_path=args.ledger)
                print("held: %s" % e["id"])
            return 0
        if args.command == "freeze":
            with open(args.matchups, encoding="utf-8") as fh:
                matchups = json.load(fh)
            if not args.execute:
                print("prepared freeze %s: %d matchups"
                      % (args.id, len(matchups)))
                return 0
            row = freeze_p12b(args.id, matchups, ledger_path=args.ledger)
            print("frozen: %s %s" % (args.id,
                                     json.dumps(row["model_sha256"],
                                                sort_keys=True)))
            return 0
        if args.command == "run":
            cond = p12b_conditions_for(args.id, args.role, args.challenger,
                                       args.opponents, ledger_path=args.ledger)
            if not args.execute:
                print("prepared: %s %s run_id=%s %s %d-%d"
                      % (args.id, args.role, p.run_id_for(cond), cond["group"],
                         cond["start_index"], cond["end_index"]))
                return 0
            s = run_p12b(args.id, args.role, args.challenger, args.opponents,
                         args.records, args.summary, workers=args.workers,
                         resume=args.resume, ledger_path=args.ledger)
            print("%s n=%d %s point=%.4f CI95=%s CI97.5=%s gate=%s"
                  % (args.role, p.players_of(s), s["group"],
                     s["point_estimate"], s["ci"]["95"], s["ci"]["97.5"],
                     json.dumps(s["gate"], sort_keys=True)))
            return 0
        if args.command == "recompute":
            saved = p.read_json(args.path)
            if saved.get("schema") == p.PAIRED_SCHEMA:
                fresh = recompute_paired_p12b(args.path)
            else:
                fresh = p.recompute(args.path)
            print("recomputed from saved records only: %s point=%.6f "
                  "CI97.5=%s gate=%s" % (args.path, fresh["point_estimate"],
                                         fresh["ci"]["97.5"],
                                         json.dumps(fresh["gate"],
                                                    sort_keys=True)))
            return 0
        if args.command == "replay":
            same, digest = replay_p12b(args.summary, workers=args.workers)
            print("replayed %s workers=%d sha256=%s byte_equal=%s"
                  % (args.summary, args.workers, digest, same))
            return 0 if same else 1
        if args.command == "paired":
            r = paired_p12b(args.kind, args.a, args.b, out_path=args.out)
            print("%s d=%.6f CI97.5=%s gate=%s"
                  % (args.kind, r["point_estimate"], r["ci"]["97.5"],
                     json.dumps(r["gate"], sort_keys=True)))
            return 0
        if args.command == "decide":
            g1 = p.recompute(args.g1)
            g2 = p.recompute(args.g2)
            r4 = [recompute_paired_p12b(x) for x in args.r4]
            ex = [recompute_paired_p12b(x) for x in args.exploit]
            d = p.decide(args.num_players, g1, g2, r4, ex,
                         holds=p12b_decision_holds(args.hold,
                                                   ledger_path=args.ledger))
            if args.out:
                if os.path.exists(args.out):
                    raise SeedLedgerError("output already exists: %s"
                                          % args.out)
                p.write_json(args.out, d)
            print(json.dumps(d, sort_keys=True))
            # 凍結対戦列と一致しない入力は判定なし(fail-closed)を exit 3 で示す。
            return 0 if d["complete"] else 3
    except (SeedLedgerError, ValueError) as exc:
        raise SystemExit("refused: %s" % exc)
    return 2


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "phase12b":
        return main_p12b(list(argv[1:]))
    return _main_legacy(argv)


def _main_legacy(argv=None):
    parser = argparse.ArgumentParser(
        description="任意2エージェントの席交換ペア対局評価"
                    "(--num-players 3/4 は Phase 10 の n 局巡回クラスタ)")
    parser.add_argument(
        "left", metavar="LEFT",
        help="エージェント指定: random / heuristic / checkpoint:メタのパス")
    parser.add_argument("right", metavar="RIGHT", help="同上(対戦相手)")
    parser.add_argument(
        "more", metavar="MORE", nargs="*",
        help="n>=3 のときの残りの相手(正準順。先頭 LEFT が評価対象)")
    parser.add_argument("--num-players", type=int, choices=(2, 3, 4),
                        default=2)
    parser.add_argument("--clusters", type=int, default=None,
                        help="n>=3: 巡回クラスタ数(1 クラスタ = n 局)")
    parser.add_argument("--reservation", default=None,
                        help="n>=3: 台帳の多人数予約 id")
    parser.add_argument("--purpose", default=None, help="n>=3: 用途")
    parser.add_argument("--register", action="store_true",
                        help="n>=3: 予約の登録(--execute なしは重複検査のみ)")
    parser.add_argument("--matchup", action="append", default=[],
                        help="n>=3 --register: 同帯で許す追加の対戦"
                             "(カンマ区切りの正準リスト。複数可)")
    parser.add_argument("--resume", action="store_true",
                        help="n>=3: 同 run の欠けた局だけ再開")
    parser.add_argument("--summary", default=None,
                        help="n>=3: 集計 JSON の出力先")
    parser.add_argument("--recompute", default=None, metavar="SUMMARY",
                        help="n>=3: 保存物だけから再計算して照合")
    parser.add_argument("--ledger", default=None,
                        help="seed 台帳(既定 docs/eval-seeds.md)")
    parser.add_argument("--replay", default=None, metavar="SUMMARY",
                        help="n>=3: 保存条件から全局を再実行し records と byte 照合")
    parser.add_argument("--hold", action="store_true",
                        help="n>=3: 対戦未確定の帯 hold を登録(--execute なしは検査のみ)")
    parser.add_argument("--hold-threshold", type=float, default=None,
                        help="n>=3 --hold: 帯の合格閾値(ベンチは省略)")
    parser.add_argument(
        "--group", choices=("train", "dev", "accept"), default="dev",
        help="seed group(docs/eval-seeds.md の予約範囲を使うこと)")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=800)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threshold", type=_parse_threshold, default=0.50,
                        help="CI 合格閾値(既定 0.50)。n>=3 の固定ベンチは none を明示")
    parser.add_argument(
        "--record-eval-to", default=None, metavar="META",
        help="評価結果をこのチェックポイントのメタの eval_results へ追記する"
             "(評価対象の checkpoint メタのパス。--execute 時にのみ書込。"
             "設計書 §18.1)")
    parser.add_argument(
        "--records", default=None,
        help="対応表(1局1行JSONL)の出力先")
    parser.add_argument(
        "--execute", action="store_true",
        help="実行を明示する。無ければ準備表示のみで終了")
    args = parser.parse_args(argv)
    if args.num_players != 2:
        if args.record_eval_to:
            raise SystemExit("--record-eval-to is two-seat only")
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1, got %d" % args.workers)
        if args.threshold is not None and not (0.0 < args.threshold < 1.0):
            raise SystemExit(
                "--threshold must be in (0, 1) or none, got %g" % args.threshold)
        if args.start_index < 0:
            raise SystemExit(
                "--start-index must be >= 0, got %d" % args.start_index)
        return main_multi(args)
    if args.threshold is None:
        raise SystemExit("--threshold none is only for --num-players 3|4")
    multi_only = [name for name in ("clusters", "reservation", "purpose",
                                    "summary", "recompute", "ledger",
                                    "replay", "hold_threshold")
                  if getattr(args, name) is not None]
    if args.more or multi_only or args.register or args.resume or \
            args.hold or args.matchup:
        raise SystemExit("extra agents / %s are only for --num-players 3|4"
                         % ", ".join(multi_only or ["--register/--resume"]))
    try:
        parse_agent_spec(args.left)
        parse_agent_spec(args.right)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.record_eval_to:
        try:
            checkpoint_weights_path(args.record_eval_to)
        except ValueError as exc:
            raise SystemExit(str(exc))
        checkpoint_paths = []
        for spec in (args.left, args.right):
            kind, path = parse_agent_spec(spec)
            if kind == "checkpoint":
                checkpoint_paths.append(os.path.abspath(path))
        if os.path.abspath(args.record_eval_to) not in checkpoint_paths:
            raise SystemExit(
                "--record-eval-to は評価対象の checkpoint エージェントのメタ"
                "でなければならない: %s (評価: %s / %s)"
                % (args.record_eval_to, args.left, args.right))
    if args.pairs < 1:
        raise SystemExit("--pairs must be >= 1, got %d" % args.pairs)
    if args.start_index < 0:
        raise SystemExit(
            "--start-index must be >= 0, got %d" % args.start_index)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1, got %d" % args.workers)
    if not (0.0 < args.threshold < 1.0):
        raise SystemExit(
            "--threshold must be in (0, 1), got %g" % args.threshold)
    threads = ",".join("%s=%s" % (name, os.environ.get(name))
                       for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS"))
    if not args.execute:
        print("prepared: %s vs %s group=%s start_index=%d pairs=%d "
              "workers=%d threshold=%g thread_env=%s record_to=%s"
              % (args.left, args.right, args.group, args.start_index,
                 args.pairs, args.workers, args.threshold, threads,
                 args.record_eval_to))
        return 0
    if args.records and os.path.exists(args.records):
        raise SystemExit(
            "記録先が既に存在する: %s / 追記すると『1回だけ実行』の証跡が壊れる。"
            "別名を指定するか退避すること。" % args.records)
    if args.records:
        directory = os.path.dirname(os.path.abspath(args.records))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
    try:
        result = run_evaluation(
            args.left, args.right, args.group, args.start_index,
            args.pairs, workers=args.workers)
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    end_index = args.start_index + args.pairs - 1
    print("%s vs %s" % (args.left, args.right))
    print("group=%s index=%d-%d pairs=%d games=%d"
          % (result.group, result.start_index, end_index,
             result.pairs, result.games))
    print("wins/draws/losses=%d/%d/%d"
          % (result.wins, result.draws, result.losses))
    print("point_estimate=%.4f  95%% CI=[%.4f, %.4f]"
          % (result.point_estimate, result.ci_low, result.ci_high))
    print("bootstrap=B%d seed=%d"
          % (result.bootstrap_samples, result.bootstrap_seed))
    if args.records:
        write_game_records(result, args.records)
        print("records=%s (%d rows)"
              % (args.records, len(result.game_records)))
    if args.record_eval_to:
        append_eval_result(
            args.record_eval_to, result, args.threshold,
            records_path=args.records)
        print("recorded eval result into %s" % args.record_eval_to)
    passed = passes_threshold(result, args.threshold)
    print("pass (CI low > %.4f): %s" % (args.threshold, passed))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
