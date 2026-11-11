#!/usr/bin/env python3
"""PPO 自己対戦学習 CLI(Phase 3 設計書 §2 / §4 / §8 / §13.2 / §14.2 / §14.3)。

csi-enshu の切断耐性ジョブ(`scripts/csi_job.sh`、仕様 `docs/csi-jobs.md`)から
起動する。csi_job.sh のランナーは JOB_DIR を作業ディレクトリにして exec する
ため、本スクリプトはカレントディレクトリ(リポジトリ root 基準の相対パス)に
依存しない。リポジトリ root と既定のチェックポイント場所は `__file__` から
解決し、`sys.path` へ root を入れてから naeco を import する。

起動例(リポジトリと JOB_DIR は絶対・展開済みパス):

    bash "$repo/scripts/csi_job.sh" start "$job" -- \
        bash -c 'cd "$1" && exec /usr/bin/python3 scripts/train_ppo.py \
            --checkpoint-dir "$2/runs/phase3" \
            --generations 20 --episodes 64 --workers 4' \
        bash "$repo" "$repo"

`cd "$1"` の有無に関わらず動作する(スクリプトは __file__ から root を解決する)。

学習ループの性質
----------------
- **SIGTERM trap / deadline 中断**: SIGTERM を受け取ったら現在の世代の完了
  を待ってチェックポイントを保存してから終了する。ロールアウト中に SIGTERM
  で spawn ワーカーが死ぬと multiprocessing の map が戻らないことがあるため、
  ロールアウトは daemon スレッドで実行して main スレッドをブロックしない。
  TERM 検知時は保存後に `os._exit(0)` で確実に終了する(atexit の pool 後始末
  には依存しない)。`--max-seconds` の deadline もロールアウト実行中に検査し、
  達したら同じ経路(保存 → `os._exit(0)`)で終了する(残留リスク #5)。
- **世代ローテーション**: 保存後に古い世代(meta / weights / optimizer)を
  `--keep-latest` の上限で整理する。
- **中断からの再開**: 最新世代のチェックポイントを読み、学習ステップ数・
  世代番号・次のエピソード index に加えて **Adam の状態**
  (`optimizer.state_dict()`、`.optimizer.pt` ファイル)を引き継ぐ(設計書
  §14.3)。読込は必ず
  `checkpoint.load(..., expected=checkpoint.expected_meta(...))` を渡す
  (設計書 §15.1)。期待値なしの load 経路は存在せず、テストで固定する。
  model_arch はこのプロセスの net から来る(保存メタから読まない)ため
  A 表照合が真の照合になり、重みは `net.load_state_dict` がキー・形状を
  厳密検査する(設計書 §15.2 の検証を経ない読込経路を作らない)。
  optimizer 状態の無い旧チェックポイントからの再開は警告を出したうえで
  新規作成を許す(既存の成果物を読めなくしない)。
- **seed 分離**: 自己対戦は `train` グループの index 2000 以降だけを使う
  (`docs/eval-seeds.md` Phase 3 予約)。dev / accept は使わない(テストで固定)。
- **破棄局**: `Trajectory.discarded` の局は学習データに混ぜず、件数を
  カウンタとしてログに残す(§4.3 / core-api §9.4)。
- **learner の決定だけを学習**: 相手(過去世代 / HeuristicAgent /
  RandomAgent)の手は学習データへ入れない(設計書 14.1)。`_make_agents` は
  正準順 `[learner, opponent]` を作り、rollout へ `learner_index=0` を渡す。
  env が席置換後の実際の席を `Trajectory.learner_seat` へ記録し、
  `trajectories_to_batch` がその席の決定だけをバッチへ入れる。
- **チェックポイント排他ロック(Phase 4)**: run() は torch import と
  チェックポイント読込の前に checkpoint_dir の排他ロック(.lock)を取得する。
  別ジョブが同一 --checkpoint-dir を保持中なら明確なログを出して非ゼロで
  終了し、チェックポイントを一切書き込まない。ロックは fcntl.flock で、
  プロセス死亡(SIGKILL / os._exit 含む)時にカーネルが自動解放するため、
  残置ロックで後続の再開ジョブを妨げない。CLI 引数は追加しない(常時有効)。
- **待機上限**: `--max-seconds` は壁時計の絶対 deadline で表現する(§8.3)。
  反復回数で書かない。deadline は世代境界(while ループ先頭)とロールアウト
  実行中の両方で検査する(残留リスク #5)。
- **停止条件**: `--generations` はこのプロセスで完了させる世代数(0 は
  無制限。再開時は再開世代からの追加数)、`--max-seconds` は壁時計の絶対
  deadline(0 は無制限)。組み合わせの挙動は次の通り。
  - `--generations` だけ指定: 指定世代数を完了した時点で保存して停止する。
  - `--max-seconds` だけ指定: deadline に達した時点で保存して停止する。
  - 両方指定: 先に満たされた方(世代数と deadline の早い方)で停止する。
  - どちらも指定しない: SIGTERM を受け取るまで走り続け、SIGTERM 時は
    安全な地点でチェックポイントを保存してから終了する。
- **再現メタデータ(§18.1)**: 保存メタの B 表キー `git_commit` に必ず
  リポジトリの HEAD を記録する(クリーンなら `<40桁 sha>`、変更があれば
  `<sha>-dirty`、git が使えなければ `"unavailable"`)。None のままに
  しない(テストで固定)。
"""
import argparse
import gc
import json
import logging
import math
import multiprocessing
import multiprocessing.pool
import os
import platform
import random
import re
import signal
import subprocess
import sys
import threading
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from naeco.ai.heuristic_agent import HeuristicAgent  # noqa: E402
from naeco.ai.random_agent import RandomAgent  # noqa: E402
from naeco.engine.state import derive_seed, eval_seed  # noqa: E402
from naeco.train import checkpoint as checkpoint_mod  # noqa: E402
from naeco.train import encoding as enc  # noqa: E402
from naeco.train import pool as pool_mod  # noqa: E402
from naeco.train.agent import PolicyAgent  # noqa: E402
from naeco.train.env import rollout  # noqa: E402

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# 自己対戦 seed の契約(docs/eval-seeds.md Phase 3 予約)。変更禁止。
TRAIN_GROUP = "train"
MIN_TRAIN_INDEX = 2000

# Legacy default; multi-seat runs pass their count explicitly.
NUM_PLAYERS = 2

# 正準順 [learner, opponent] のうち学習対象エージェントの位置(設計書 14.1)。
# env の席置換(seat_variant / 席ランダム化)で実際の席が決まるため、ここでは
# 位置だけを指定し、席の解決は env が Trajectory.learner_seat へ記録する。
LEARNER_INDEX = 0

GEN_PREFIX = "gen"
GEN_META_SUFFIX = ".meta.json"
GEN_WEIGHTS_SUFFIX = ".weights.pt"
GEN_OPTIMIZER_SUFFIX = ".optimizer.pt"
_GEN_META_RE = re.compile(r"^gen(\d+)\.meta\.json$")
_GEN_WEIGHTS_RE = re.compile(r"^gen(\d+)\.weights\.pt$")
_GEN_OPTIMIZER_RE = re.compile(r"^gen(\d+)\.optimizer\.pt$")

DEFAULT_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "runs", "phase3")

_logger = logging.getLogger("train_ppo")

# SIGTERM の受信フラグ。ハンドラはフラグを立てるだけにし、保存と終了は
# 学習ループの安全な地点で行う(シグナルハンドラ内でファイル書込をしない)。
_TERMINATE = {"flag": False}


def _fix_thread_env():
    """4つのスレッド環境変数を "1" へ固定する(設計書 §8.4 / docs/csi-jobs.md)。"""
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


_fix_thread_env()


def _handle_sigterm(signum, frame):
    """SIGTERM を受けたらフラグを立てる。保存と終了は安全な地点で行う。"""
    _TERMINATE["flag"] = True


def _install_sigterm_handler():
    signal.signal(signal.SIGTERM, _handle_sigterm)


def _validate_start_index(index):
    """自己対戦 index が train 2000 以降であることを検証する(§6 / §7.3)。"""
    index = int(index)
    if index < MIN_TRAIN_INDEX:
        raise ValueError(
            "self-play episode index must be >= %d "
            "(docs/eval-seeds.md Phase 3 予約: train %d 以降), got %d"
            % (MIN_TRAIN_INDEX, MIN_TRAIN_INDEX, index))
    return index


def _episode_root(index):
    """エピソードの root seed。group は TRAIN_GROUP 固定(§7.3 の分離)。"""
    return eval_seed(TRAIN_GROUP, index)


def _make_specs(start_index, num_episodes, agents_factory):
    """学習エピソードの rollout spec 列 `(group, index, seat_variant, agents)`。

    group は TRAIN_GROUP 固定、index は start_index からの連番、seat_variant
    は 0 固定(席ランダム化は randomize_seats が "seat" ストリームで行う)。
    dev / accept を学習に使う経路は存在しない(テストで固定)。
    """
    return [
        (TRAIN_GROUP, start_index + k, 0, agents_factory(start_index + k))
        for k in range(num_episodes)
    ]


def _select_usable(trajectories):
    """破棄局を学習データから除く。戻り値は `(usable, discarded)` の2 list。

    破棄局はログのカウンタにのみ残し、trajectories_to_batch へ渡さない
    (設計書 §4.3 / core-api §9.4)。
    """
    usable = [traj for traj in trajectories if not traj.discarded]
    discarded = [traj for traj in trajectories if traj.discarded]
    return usable, discarded


def _make_agents(learner_net, entry, entry_nets, index):
    """1エピソードの agents を正準順 `[learner, opponent]` で返す。

    - learner は常に現行 net の `PolicyAgent(mode="sample")`。
    - opponent はプール抽選の entry に従う。checkpoint は過去世代の net を
      sample モードで使い、heuristic / random は該当エージェントを使う。
    - sample 用 RNG の seed はエピソード座標だけから派生する。env の席置換が
      "seat" ストリームの variant=0 を消費するため、agent 側は variant 1
      (learner) / 2 (opponent) を使い衝突させない(設計書 §6.3)。
    - 戻り値の先頭が learner であることが rollout の `learner_index=0` と
      `trajectories_to_batch` の絞り込みの前提(設計書 14.1)。順序を変えると
      相手の手が学習データへ混入する。
    """
    root = _episode_root(index)
    learner = PolicyAgent(
        learner_net, mode="sample", seed=derive_seed(root, "seat", index, 1))
    if entry.kind == "heuristic":
        opponent = HeuristicAgent()
    elif entry.kind == "random":
        opponent = RandomAgent(seed=derive_seed(root, "seat", index, 2))
    elif entry.name == "latest":
        opponent = PolicyAgent(
            learner_net, mode="sample",
            seed=derive_seed(root, "seat", index, 2))
    else:
        opponent = PolicyAgent(
            entry_nets[entry.name], mode="sample",
            seed=derive_seed(root, "seat", index, 2))
    return [learner, opponent]


def _git_commit():
    """リポジトリの HEAD と working tree の状態を再現メタデータとして返す。

    - クリーン: `<40桁 sha>`。
    - dirty: `<40桁 sha>-dirty`(working tree が変更されていることが分かる)。
    - git が使えない環境(バイナリ無し / リポジトリでない / タイムアウト等):
      `"unavailable"`。

    `None` を返さない(設計書 §18.1。checkpoint の B 表キー `git_commit` を
    `None` のままにしない)。
    """
    try:
        head = subprocess.run(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        if head.returncode != 0 or not head.stdout.strip():
            return "unavailable"
        sha = head.stdout.strip()
        status = subprocess.run(
            ["git", "-C", REPO_ROOT, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
        if status.returncode != 0 or status.stdout.strip():
            return "%s-dirty" % sha
        return sha
    except Exception:
        return "unavailable"


def _meta_extra(args, config, generation, training_steps,
                next_episode_index, start_index, torch_version):
    """checkpoint の B 表(記録)キーと再現メタデータを組み立てる。"""
    return {
        "git_commit": _git_commit(),
        "torch_version": torch_version,
        "root_seed": args.seed,
        "hyperparameters": {
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_eps": config.clip_eps,
            "value_coef": config.value_coef,
            "entropy_coef": config.entropy_coef,
            "lr": config.lr,
            "epochs": config.epochs,
            "minibatch_size": config.minibatch_size,
            "max_grad_norm": config.max_grad_norm,
            "hidden": list(args.hidden),
            "action_hidden": list(args.action_hidden),
            "episodes_per_generation": args.episodes_per_generation,
            "workers": args.workers,
            "heuristic_weight": args.heuristic_weight,
            "random_weight": args.random_weight,
            # Multi-seat records live under the B-table key "hyperparameters"
            # (checkpoint extra accepts B-table keys only); n=2 is unchanged.
            **({"device": "cpu", "pool_manifest": {
                "num_players": args.num_players, "latest": 0.5,
                "past": 0.25, "heuristic": 0.25, "past_generations": 5,
                "missing_past": "latest", "frozen_at": "generation_start"},
                "rng_contract": {
                    "pool": "derive_seed(episode_root,pool,index,opponent_index)",
                    "policy": "derive_seed(episode_root,seat,index,agent_index+1)",
                    "shuffle": "derive_seed(root_seed,ppo,epoch)"}}
               if getattr(args, "num_players", 2) > 2 else {}),
        },
        "training_steps": training_steps,
        "generation": generation,
        "next_episode_index": next_episode_index,
        "start_index": start_index,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _gen_meta_path(checkpoint_dir, generation):
    return os.path.join(
        checkpoint_dir, "%s%03d%s" % (GEN_PREFIX, generation, GEN_META_SUFFIX))


def _gen_weights_path(checkpoint_dir, generation):
    return os.path.join(
        checkpoint_dir,
        "%s%03d%s" % (GEN_PREFIX, generation, GEN_WEIGHTS_SUFFIX))


def _gen_optimizer_path(checkpoint_dir, generation):
    return os.path.join(
        checkpoint_dir,
        "%s%03d%s" % (GEN_PREFIX, generation, GEN_OPTIMIZER_SUFFIX))


def _list_generations(checkpoint_dir):
    """完走した世代番号の昇順 list(meta がある世代だけ)。"""
    if not os.path.isdir(checkpoint_dir):
        return []
    generations = []
    for name in os.listdir(checkpoint_dir):
        match = _GEN_META_RE.match(name)
        if match:
            generations.append(int(match.group(1)))
    return sorted(generations)


def _find_latest_generation(checkpoint_dir):
    """最新の完走世代番号を返す。無ければ None。"""
    generations = _list_generations(checkpoint_dir)
    return generations[-1] if generations else None


def _save_checkpoint(checkpoint_dir, generation, net, optimizer, extra, num_players=2):
    """世代チェックポイントを保存する。

    重み(.weights.pt)と Adam 状態(.optimizer.pt)を先に書き、メタ JSON を
    最後に書く(設計書 §14.3)。クラッシュで中断された世代はメタが無いため、
    再開時・ローテーション時に不完全として扱える。

    重みファイルは従来どおり裸の `net.state_dict()` のまま保存し、optimizer
    状態は別ファイルに置く。これは eval_agent / promote の既存の重み読込経路
    (裸の state dict を `net.load_state_dict` へ渡す)を optimizer 形式へ
    依存させないための形式分離。optimizer 状態は中断再開の連続実行との一致
    にだけ使い、他スクリプトは読まない。
    torch 操作は `naeco.train.ppo` の包み関数(`save_state_dict`)だけを
    使う(torch を import してよいのは net.py / ppo.py だけ)。ppo は
    torch 依存のため、保存を実行するこの関数の内側で遅延 import する
    (--help 等の torch 不要経路を torch 無し環境で動かすため)。
    """
    from naeco.train import ppo as ppo_mod
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta_path = _gen_meta_path(checkpoint_dir, generation)
    weights_path = _gen_weights_path(checkpoint_dir, generation)
    optimizer_path = _gen_optimizer_path(checkpoint_dir, generation)
    ppo_mod.save_state_dict(net.state_dict(), weights_path)
    ppo_mod.save_state_dict(optimizer.state_dict(), optimizer_path)
    meta = checkpoint_mod.current_meta(
        num_players, net.arch_dict(), extra=extra)
    checkpoint_mod.save(meta_path, meta)
    return meta_path


def _load_checkpoint(checkpoint_dir, generation, model_arch, num_players=2):
    """世代のメタ・重み・Adam 状態を読み、互換性を検証して返す。

    checkpoint.load へは必ず expected=expected_meta(...) を渡す(設計書
    §15.1 / §14.2)。期待値なしの load 経路は本スクリプトに存在せず、テストで
    固定する。model_arch はこのプロセスの net(現在の実装)から来るため、
    保存メタの model_arch は A 表照合で真に検証される(自己比較にならない)。
    重みは呼び出し側が `net.load_state_dict` へ渡す際にキー集合と各テンソル
    形状が厳密検査される(設計書 §15.2 の「検証を経ない読込経路を作らない」
    を満たす)。
    戻り値は `(meta, state_dict, optimizer_state)`。optimizer_state は
    `.optimizer.pt` の無い旧チェックポイントでは None(設計書 §14.3)。

    torch 操作は `naeco.train.ppo` の包み関数(`load_state_dict`)だけを
    使う(torch を import してよいのは net.py / ppo.py だけ)。ppo は
    torch 依存のため、読込を実行するこの関数の内側で遅延 import する
    (--help 等の torch 不要経路を torch 無し環境で動かすため)。
    """
    from naeco.train import ppo as ppo_mod
    meta_path = _gen_meta_path(checkpoint_dir, generation)
    weights_path = _gen_weights_path(checkpoint_dir, generation)
    optimizer_path = _gen_optimizer_path(checkpoint_dir, generation)
    meta = checkpoint_mod.load(
        meta_path,
        expected=checkpoint_mod.expected_meta(num_players, model_arch))
    if not os.path.exists(weights_path):
        raise ValueError(
            "checkpoint %s has meta but weights %s is missing"
            % (meta_path, weights_path))
    state_dict = ppo_mod.load_state_dict(weights_path)
    optimizer_state = None
    if os.path.exists(optimizer_path):
        optimizer_state = ppo_mod.load_state_dict(optimizer_path)
    return meta, state_dict, optimizer_state


def _prune_checkpoints(checkpoint_dir, keep_latest, logger=None):
    """古い世代を保持数上限で整理する(ディスクを埋めない)。

    - 保持上限を超えた世代の meta / weights / optimizer を削除する。
    - メタの無い weights / optimizer(クラッシュで meta 書込前に中断した
      世代)は、世代番号に関わらず不完全なチェックポイントとして削除する。
    戻り値は削除した世代番号の list。
    """
    keep_latest = int(keep_latest)
    if keep_latest < 1:
        raise ValueError("keep-latest must be >= 1, got %d" % keep_latest)
    if not os.path.isdir(checkpoint_dir):
        return []
    removed = []
    generations = _list_generations(checkpoint_dir)
    for generation in generations[:-keep_latest]:
        for path in (_gen_meta_path(checkpoint_dir, generation),
                     _gen_weights_path(checkpoint_dir, generation),
                     _gen_optimizer_path(checkpoint_dir, generation)):
            if os.path.exists(path):
                os.remove(path)
        removed.append(generation)
        if logger is not None:
            logger.info("pruned generation %03d", generation)
    for name in sorted(os.listdir(checkpoint_dir)):
        match = _GEN_WEIGHTS_RE.match(name)
        if not match:
            continue
        generation = int(match.group(1))
        if not os.path.exists(_gen_meta_path(checkpoint_dir, generation)):
            for path in (_gen_weights_path(checkpoint_dir, generation),
                         _gen_optimizer_path(checkpoint_dir, generation)):
                if os.path.exists(path):
                    os.remove(path)
            removed.append(generation)
            if logger is not None:
                logger.info(
                    "pruned orphan weights of generation %03d", generation)
    # 防御: optimizer ファイルが meta と weights の両方を欠く孤児として残る
    # 経路は通常の保存順(weights → optimizer → meta)では生じないが、万一
    # 残っていればこの掃除で取り除く(removed への重複追加はしない)。
    for name in sorted(os.listdir(checkpoint_dir)):
        match = _GEN_OPTIMIZER_RE.match(name)
        if not match:
            continue
        generation = int(match.group(1))
        if not os.path.exists(_gen_meta_path(checkpoint_dir, generation)) \
                and not os.path.exists(
                    _gen_weights_path(checkpoint_dir, generation)):
            os.remove(os.path.join(checkpoint_dir, name))
            if generation not in removed:
                removed.append(generation)
            if logger is not None:
                logger.info(
                    "pruned orphan optimizer of generation %03d", generation)
    return removed


# ===========================================================================
# Phase 12b(opt-in。--phase12b-mode を指定したときだけ有効。旧経路は不変)
# ===========================================================================
PHASE12B_MODES = ("A", "B", "exploiter")
PHASE12B_EXIT_BLOCK_EXHAUSTED = 3
PHASE12B_EXIT_INVALID_DISCARD = 4     # 不合法/InvariantError の discard が1局でもある
PHASE12B_EXIT_NONFINITE = 5           # 更新統計が NaN/inf
PHASE12B_EXIT_CHILDREN_REMAIN = 6     # 期限/TERM 後に子プロセスの収束を確認できない
PHASE12B_FAILURE_FILE = "phase12b-failure.json"
SNAPSHOT_DIR = "snapshots"
FORK_DIR = "fork"
PFSP_DIR = "pfsp"
MAX_SNAPSHOTS = 8


def _sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha_list(text, names):
    parts = (text or "").split(":")
    if len(parts) != len(names) or not all(
            re.match(r"^[0-9a-f]{64}$", p) for p in parts):
        raise ValueError("expected %s as %d colon-separated 64-hex SHA-256"
                         % (":".join(names), len(names)))
    return dict(zip(names, parts))


def _write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _converge_children(logger, grace=10.0):
    """spawn worker pool と resource_tracker を収束させ、残存数を返す。

    deadline/TERM では rollout の daemon スレッドが pool.map で待ったままなので、
    親だけ os._exit すると worker と resource_tracker が PPID=1 で残る。
    1) 稼働中の Pool を terminate()(再生成ハンドラも止まる)
    2) 残った multiprocessing 子を terminate→join
    3) resource_tracker を停止して回収
    SIGKILL は使わない。戻り値は確認できた残存子プロセス数(0 が収束)。
    """
    # Pool.terminate() は inqueue の書込ロックを取る。グループ TERM で死んだ
    # worker がそのロックを握ったままだと永久にブロックする(W2-C-TERMHANG)。
    # 状態は terminate() 冒頭で TERMINATE になり再生成は止まるので、daemon
    # スレッドで時間制限付きに呼び、戻らなければ放置して子を直接止める。
    limit = time.monotonic() + grace
    stuck = []
    for obj in gc.get_objects():
        if type(obj) is multiprocessing.pool.Pool:
            def _terminate_pool(pool=obj):
                try:
                    pool.terminate()
                except Exception as exc:  # 残存確認で最終判定する
                    logger.warning("pool terminate failed: %r", exc)
            thread = threading.Thread(target=_terminate_pool, daemon=True)
            thread.start()
            thread.join(timeout=min(2.0, max(0.1, limit - time.monotonic())))
            if thread.is_alive():
                stuck.append(thread)
    if stuck:
        logger.warning("pool terminate did not return (%d); terminating "
                       "workers directly", len(stuck))
    # spawn 途中・再生成途中の worker が一度の snapshot から漏れるため、
    # 子が尽きるか grace が切れるまで繰り返す。
    children = []
    seen = set()
    while True:
        current = multiprocessing.active_children()
        if not current:
            # Pool の handler が直後に worker を再生成する競合があるため、
            # 短い猶予を置いて空のままかを再確認する。
            time.sleep(0.3)
            current = multiprocessing.active_children()
        if not current or time.monotonic() >= limit:
            break
        for child in current:
            if child.pid not in seen:
                seen.add(child.pid)
                children.append(child)
            child.terminate()
        for child in current:
            child.join(timeout=max(0.1, min(1.0, limit - time.monotonic())))
    # _stop() は tracker の終了を waitpid で無期限に待つ。pipe の write 端を
    # 握った worker が残ると戻らないので、これも時間制限付きにする
    # (親は直後に os._exit するため、tracker は EOF で自然に終わる)。
    def _stop_tracker():
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception as exc:
            logger.warning("resource_tracker stop failed: %r", exc)
    tracker_thread = threading.Thread(target=_stop_tracker, daemon=True)
    tracker_thread.start()
    tracker_thread.join(timeout=min(3.0, max(0.5, limit - time.monotonic())))
    if tracker_thread.is_alive():
        logger.warning("resource_tracker stop did not return; continuing")
    # 停止待ちの間に再生成された worker があれば最後にもう一度止める。
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=1.0)
    remaining = multiprocessing.active_children()
    logger.info("child convergence: terminated=%d remaining=%d", len(children),
                len(remaining))
    return len(remaining)


def _exit_converged(logger, code=0):
    """deadline/TERM の中断終了。子を収束させてから親を終える(残存があれば非0)。"""
    remaining = _converge_children(logger)
    sys.stdout.flush()
    sys.stderr.flush()
    if remaining:
        logger.error("children remain after convergence (%d); exiting %d",
                     remaining, PHASE12B_EXIT_CHILDREN_REMAIN)
        os._exit(PHASE12B_EXIT_CHILDREN_REMAIN)
    os._exit(code)


class Phase12bFailure(Exception):
    """Phase 12b の即停止(不合法 discard / NaN)。世代を更新・保存扱いにしない。"""

    def __init__(self, kind, code, detail):
        Exception.__init__(self, "%s: %s" % (kind, detail.get("message", "")))
        self.kind = kind
        self.code = code
        self.detail = detail


def _discard_failure(trajs, where):
    discards = [{"index": t.index, "seat_variant": t.seat_variant,
                 "reason": t.discard_reason} for t in trajs if t.discarded]
    return Phase12bFailure(
        "invalid_discard", PHASE12B_EXIT_INVALID_DISCARD,
        {"message": "%d of %d episodes discarded in %s"
                    % (len(discards), len(trajs), where),
         "where": where, "discards": discards, "episodes": len(trajs)})


def _record_phase12b_failure(checkpoint_dir, logger, failure, generation, age,
                             next_episode_index):
    """回収側が拒否できる失敗記録。checkpoint/世代ログには触れない。"""
    record = {"schema": "phase12b-failure/v1", "kind": failure.kind,
              "exit_code": failure.code, "generation_file": generation,
              "age_before": age, "next_episode_index": next_episode_index,
              "detail": failure.detail,
              "update_applied": False, "checkpoint_saved": False}
    _write_json_atomic(os.path.join(checkpoint_dir, PHASE12B_FAILURE_FILE), record)
    logger.error("phase12b FAILURE kind=%s exit=%d generation=%d: %s; no update, "
                 "no checkpoint saved", failure.kind, failure.code, generation,
                 failure.detail.get("message"))


def _sibling(meta_path, suffix):
    if not meta_path.endswith(GEN_META_SUFFIX):
        raise ValueError("model path must end with %s: %s"
                         % (GEN_META_SUFFIX, meta_path))
    return meta_path[:-len(GEN_META_SUFFIX)] + suffix


def _verify_files(paths_shas, what):
    for name, (path, sha) in sorted(paths_shas.items()):
        if not os.path.isfile(path):
            raise ValueError("%s %s missing: %s" % (what, name, path))
        actual = _sha256_file(path)
        if actual != sha:
            raise ValueError("%s %s SHA mismatch: %s expected %s got %s"
                             % (what, name, path, sha, actual))


def _trajectory_sha256(trajectories):
    """順序付き trajectory 列の内容 hash(worker 数・完了順に依存しない)。"""
    import hashlib
    digest = hashlib.sha256()
    for t in trajectories:
        row = [t.group, t.index, t.seat_variant, list(t.seats), t.learner_seat,
               t.discarded, list(t.terminal_rewards), t.initial_state_hash,
               [[d.player_id, d.chosen_index, d.reward, d.turn_advanced]
                for d in t.decisions]]
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class _CanonicalPolicyAgent(PolicyAgent):
    """同一エンコードの合法手に必ず同一 logit を与える PolicyAgent(Phase 12b 専用)。

    torch の batched 推論は、同じ入力行でも batch 内の位置(メモリ配置)により
    1 ulp 異なる値を返すことがあり、同じ建物の別コピー売却など「エンコードが
    完全に同じ合法手」で greedy の同点解消が process 状態依存になる(C3 で
    worker 1/4 不一致として実測)。一意な行だけを推論して元の並びへ戻すことで、
    同一エンコードの手は構成上ビット同一の logit になり sort_key の同点解消が効く。
    """

    def _forward(self, obs_vec, action_vecs):
        rows = []
        position = {}
        mapping = []
        for vec in action_vecs:
            key = tuple(vec)
            if key not in position:
                position[key] = len(rows)
                rows.append(vec)
            mapping.append(position[key])
        logits, values = PolicyAgent._forward(self, obs_vec, rows)
        return [logits[i] for i in mapping], values


def _strong_agent(net, seed, cards):
    """凍結 strong(Phase 10 B と同じ固定探索: harness.STRONG_SEARCH)。"""
    from naeco.ai.eval.harness import STRONG_SEARCH
    from naeco.ai.search.ismcts import IsmctsAgent
    from naeco.train.search_evaluator import NetEvaluator
    return IsmctsAgent(
        root_seed=seed,
        budget_transitions=STRONG_SEARCH["budget_transitions"],
        c_puct=STRONG_SEARCH["c_puct"], horizon=STRONG_SEARCH["horizon"],
        cards=cards, evaluator=NetEvaluator(net),
        policy_agent=PolicyAgent(net, mode="greedy", seed=0),
        root_policy=STRONG_SEARCH["root_policy"])


class Phase12bRun(object):
    """fork/lr 段/保存座標/snapshot/PFSP/exploiter の状態と検証を一か所に持つ。

    meta の B 表 `hyperparameters.phase12b` に記録し、再開時はそこから読む
    (CLI 既定値や --start-index だけで fork/lr が成立したとみなさない)。
    """

    def __init__(self, args, logger, checkpoint_dir, num_players, model_arch,
                 net_mod, ppo_mod, obs_dim, action_dim):
        self.args = args
        self.logger = logger
        self.dir = checkpoint_dir
        self.n = num_players
        self.model_arch = model_arch
        self.net_mod = net_mod
        self.ppo = ppo_mod
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mode = args.phase12b_mode
        self.fork = None
        self.init = None
        self.target = None
        self.lr_history = []
        self.pool_manifest = None
        self.pool_manifest_sha = None
        self.windows = {}
        self.cards = None
        self.deadline = None

    # ----------------------------------------------------------- utilities
    def _new_net(self):
        return self.net_mod.PolicyValueNet(
            self.obs_dim, self.action_dim, hidden=tuple(self.args.hidden),
            action_hidden=tuple(self.args.action_hidden))

    def _load_model(self, meta_path, shas):
        """SHA 照合 → checkpoint 互換照合(expected_meta) → 重みを読み net を返す。"""
        _verify_files({"meta": (meta_path, shas["meta"]),
                       "weights": (_sibling(meta_path, GEN_WEIGHTS_SUFFIX),
                                   shas["weights"])}, "model")
        meta = checkpoint_mod.load(
            meta_path, expected=checkpoint_mod.expected_meta(
                self.n, self.model_arch))
        net = self._new_net()
        net.load_state_dict(self.ppo.load_state_dict(
            _sibling(meta_path, GEN_WEIGHTS_SUFFIX)))
        return net, meta

    def optimizer_lrs(self, optimizer):
        return [float(g["lr"]) for g in optimizer.param_groups]

    def _apply_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            group["lr"] = lr
        actual = self.optimizer_lrs(optimizer)
        if any(x != lr for x in actual):
            raise RuntimeError("optimizer lr %r != manifest lr %r" % (actual, lr))
        self.logger.info("phase12b optimizer_lr actual=%s manifest=%r",
                         ",".join(repr(x) for x in actual), lr)

    # -------------------------------------------------------------- setup
    def start(self, net, optimizer, latest_generation):
        """fork/init(新規)または通常 resume。(generation, steps, next_index, age)。"""
        args = self.args
        if self.mode == "B":
            self._load_pool_manifest()
        if self.mode == "exploiter":
            shas = _parse_sha_list(args.exploit_target_sha256, ("meta", "weights"))
            self.target = {"path": os.path.abspath(args.exploit_target),
                           "sha256": shas}
            self.target_net, unused = self._load_model(self.target["path"], shas)
        if latest_generation is None:
            return self._fresh(net, optimizer)
        return self._resume(net, optimizer, latest_generation)

    def _fresh(self, net, optimizer):
        args = self.args
        if self.mode in ("A", "B"):
            self._fork(net, optimizer)
        else:
            shas = _parse_sha_list(args.init_sha256, ("meta", "weights"))
            path = os.path.abspath(args.init_from)
            init_net, init_meta = self._load_model(path, shas)
            net.load_state_dict(init_net.state_dict())
            self.init = {"path": path, "sha256": shas,
                         "source_generation": init_meta.get("generation")}
            self.logger.info("phase12b exploiter init weights from %s (fresh Adam)",
                             path)
        if args.lr_stage != "start":
            raise ValueError("a fresh phase12b run must use --lr-stage start")
        if self.fork is not None and args.lr != self.fork["source_lr"]:
            raise ValueError("start lr %r must equal the fork checkpoint lr %r"
                             % (args.lr, self.fork["source_lr"]))
        self.lr_history = [{"from_age": 0, "lr": args.lr, "stage": "start"}]
        self._apply_lr(optimizer, args.lr)
        start = args.start_index
        steps = int(self.fork["source_training_steps"]) if self.fork else 0
        self.logger.info(
            "phase12b fresh %s run: next_episode_index=%d (block head; fork "
            "source coordinate %s not reused) max_index=%d", self.mode, start,
            self.fork["source_next_episode_index"] if self.fork else "n/a",
            args.max_index)
        return 0, steps, start, 0

    def _fork(self, net, optimizer):
        args = self.args
        src = os.path.abspath(args.fork_from)
        shas = _parse_sha_list(args.fork_sha256, ("meta", "weights", "optimizer"))
        sources = {"meta": src,
                   "weights": _sibling(src, GEN_WEIGHTS_SUFFIX),
                   "optimizer": _sibling(src, GEN_OPTIMIZER_SUFFIX)}
        _verify_files(dict((k, (sources[k], shas[k])) for k in sources),
                      "fork input")
        fork_dir = os.path.join(self.dir, FORK_DIR)
        os.makedirs(fork_dir, exist_ok=True)
        import shutil
        copies = {}
        for key, path in sources.items():
            dst = os.path.join(fork_dir, os.path.basename(path))
            if not os.path.exists(dst):
                shutil.copyfile(path, dst)
                os.chmod(dst, 0o444)
            copies[key] = dst
        _verify_files(dict((k, (copies[k], shas[k])) for k in copies),
                      "fork copy")
        meta = checkpoint_mod.load(
            copies["meta"], expected=checkpoint_mod.expected_meta(
                self.n, self.model_arch))
        net.load_state_dict(self.ppo.load_state_dict(copies["weights"]))
        opt_state = self.ppo.load_state_dict(copies["optimizer"])
        optimizer.load_state_dict(opt_state)
        restored = self.optimizer_lrs(optimizer)
        source_lr = float(meta["hyperparameters"]["lr"])
        if any(x != source_lr for x in restored):
            raise ValueError("fork optimizer lr %r != fork meta lr %r"
                             % (restored, source_lr))
        if not optimizer.state:
            raise ValueError("fork optimizer has no Adam moment state")
        source_next = int(meta["next_episode_index"])
        if args.start_index < source_next:
            raise ValueError("fork --start-index %d must be a new block above "
                             "the source coordinate %d"
                             % (args.start_index, source_next))
        self.fork = {"source_meta_path": src, "copy_dir": FORK_DIR,
                     "sha256": shas, "source_generation": int(meta["generation"]),
                     "source_training_steps": int(meta["training_steps"]),
                     "source_next_episode_index": source_next,
                     "source_lr": source_lr,
                     "absolute_generation_offset": int(meta["generation"]) + 1,
                     "adam_state_tensors": len(optimizer.state)}
        self.logger.info(
            "phase12b fork from %s generation=%d lr=%r adam_state=%d "
            "(weights + Adam inherited)", src, self.fork["source_generation"],
            source_lr, len(optimizer.state))

    def _resume(self, net, optimizer, latest_generation):
        args = self.args
        meta, state_dict, opt_state = _load_checkpoint(
            self.dir, latest_generation, self.model_arch, self.n)
        block = meta["hyperparameters"].get("phase12b")
        if not isinstance(block, dict):
            raise ValueError("checkpoint %s is not a phase12b run; refusing to "
                             "resume it in phase12b mode" % self.dir)
        if block["mode"] != self.mode:
            raise ValueError("resume mode %s != saved mode %s"
                             % (self.mode, block["mode"]))
        if block["block"] != {"start_index": args.start_index,
                              "max_index": args.max_index}:
            raise ValueError("resume block %r != saved block %r"
                             % ({"start_index": args.start_index,
                                 "max_index": args.max_index}, block["block"]))
        if self.mode in ("A", "B"):
            shas = _parse_sha_list(args.fork_sha256,
                                   ("meta", "weights", "optimizer"))
            if block["fork"]["sha256"] != shas:
                raise ValueError("resume fork SHA differs from the saved fork")
            self.fork = block["fork"]
            fork_dir = os.path.join(self.dir, FORK_DIR)
            _verify_files(dict(
                (k, (os.path.join(fork_dir, os.path.basename(
                    _sibling(self.fork["source_meta_path"], suffix))), shas[k]))
                for k, suffix in (("meta", GEN_META_SUFFIX),
                                  ("weights", GEN_WEIGHTS_SUFFIX),
                                  ("optimizer", GEN_OPTIMIZER_SUFFIX))),
                "retained fork copy")
        else:
            self.init = block["init"]
            if block["target"] != self.target:
                raise ValueError("resume exploit target differs from the saved target")
        if block.get("pool_manifest_sha256") != self.pool_manifest_sha:
            raise ValueError("resume pool manifest SHA differs")
        if opt_state is None:
            raise ValueError("phase12b resume requires the saved Adam state")
        net.load_state_dict(state_dict)
        optimizer.load_state_dict(opt_state)
        # age = 完了した更新(世代)数。phase12b は更新のない保存をしないので
        # 世代ファイル n は常に age n+1(保存値と番号の両方で照合する)。
        age = int(block["age"])
        if age != latest_generation + 1:
            raise ValueError("saved age %d inconsistent with generation file %d "
                             "(file n must hold age n+1)" % (age, latest_generation))
        history = list(block["lr_history"])
        if args.lr_stage == "start":
            if len(history) != 1 or args.lr != history[0]["lr"]:
                raise ValueError("--lr-stage start resume needs lr %r and no "
                                 "extension (history %r)" % (history[0]["lr"], history))
        else:
            if self.mode == "exploiter":
                raise ValueError("exploiter runs have no lr extension")
            half = history[0]["lr"] / 2.0
            if args.lr != half:
                raise ValueError("--lr-stage extend lr must be %r (half of %r)"
                                 % (half, history[0]["lr"]))
            if len(history) == 1:
                history.append({"from_age": age, "lr": args.lr, "stage": "extend"})
                self.logger.info("phase12b lr extension applied once at age %d: "
                                 "%r -> %r", age, history[0]["lr"], args.lr)
            elif len(history) == 2 and history[1]["lr"] == args.lr:
                self.logger.info("phase12b lr extension already applied at age "
                                 "%d; not halving again", history[1]["from_age"])
            else:
                raise ValueError("unexpected lr history %r" % (history,))
        self.lr_history = history
        self._apply_lr(optimizer, args.lr)
        self.logger.info(
            "phase12b resume (saved coordinates): generation=%d age=%d "
            "next_episode_index=%d", latest_generation + 1, age,
            int(meta["next_episode_index"]))
        # 保存と snapshot の間で落ちた場合だけ、保存済み世代から補完する。
        self.ensure_snapshot(age, latest_generation)
        return (latest_generation + 1, int(meta["training_steps"]),
                _validate_start_index(int(meta["next_episode_index"])), age)

    # --------------------------------------------------------- meta record
    def meta_block(self, age, optimizer):
        block = {"mode": self.mode, "age": age,
                 "block": {"start_index": self.args.start_index,
                           "max_index": self.args.max_index},
                 "lr_history": self.lr_history,
                 "optimizer_lr": self.optimizer_lrs(optimizer),
                 "snapshots_every": self.args.snapshot_every,
                 "pool_manifest_sha256": self.pool_manifest_sha}
        if self.fork is not None:
            block["fork"] = self.fork
            block["absolute_generation"] = \
                self.fork["absolute_generation_offset"] + age - 1
        if self.mode == "exploiter":
            block["init"] = self.init
            block["target"] = self.target
        return block

    def check_lr(self, optimizer):
        actual = self.optimizer_lrs(optimizer)
        expected = self.lr_history[-1]["lr"]
        if any(x != expected for x in actual):
            raise RuntimeError("optimizer lr drifted: %r != %r" % (actual, expected))

    # ----------------------------------------------------------- snapshots
    def ensure_snapshot(self, age, gen):
        """age が snapshot 間隔の倍数なら、世代ファイル gen を snapshots/ へ複製。

        gen はその age を完了した直後に保存した世代ファイル番号。prune 対象外の
        別ディレクトリに meta/weights/optimizer を置き、index.json に相対 age と
        ファイル番号・(fork 時は)絶対世代と SHA を記録する。
        """
        every = self.args.snapshot_every
        if age <= 0 or age % every:
            return
        import shutil
        snap_dir = os.path.join(self.dir, SNAPSHOT_DIR)
        os.makedirs(snap_dir, exist_ok=True)
        index_path = os.path.join(snap_dir, "index.json")
        index = {"snapshots": []}
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as fh:
                index = json.loads(fh.read())
        if any(s["age"] == age for s in index["snapshots"]):
            return
        if len(index["snapshots"]) >= MAX_SNAPSHOTS:
            raise RuntimeError("snapshot cap %d reached at age %d"
                               % (MAX_SNAPSHOTS, age))
        with open(_gen_meta_path(self.dir, gen), encoding="utf-8") as fh:
            saved_age = json.loads(fh.read())["hyperparameters"]["phase12b"]["age"]
        if saved_age != age:
            raise RuntimeError("generation file %d holds age %d, not %d"
                               % (gen, saved_age, age))
        row = {"age": age, "generation_file": gen, "sha256": {}}
        if self.fork is not None:
            row["absolute_generation"] = \
                self.fork["absolute_generation_offset"] + age - 1
        for key, src in (("meta", _gen_meta_path(self.dir, gen)),
                         ("weights", _gen_weights_path(self.dir, gen)),
                         ("optimizer", _gen_optimizer_path(self.dir, gen))):
            suffix = {"meta": GEN_META_SUFFIX, "weights": GEN_WEIGHTS_SUFFIX,
                      "optimizer": GEN_OPTIMIZER_SUFFIX}[key]
            dst = os.path.join(snap_dir, "age%04d%s" % (age, suffix))
            shutil.copyfile(src, dst)
            row["sha256"][key] = _sha256_file(dst)
            if row["sha256"][key] != _sha256_file(src):
                raise RuntimeError("snapshot copy changed %s" % src)
        index["snapshots"].append(row)
        _write_json_atomic(index_path, index)
        self.logger.info("phase12b snapshot age=%d saved (never pruned)", age)

    def snapshot_rows(self):
        path = os.path.join(self.dir, SNAPSHOT_DIR, "index.json")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return json.loads(fh.read())["snapshots"]

    # ----------------------------------------------------------------- PFSP
    def _load_pool_manifest(self):
        if not self.args.pool_manifest:
            raise ValueError("--phase12b-mode B requires --pool-manifest")
        path = os.path.abspath(self.args.pool_manifest)
        self.pool_manifest_sha = _sha256_file(path)
        with open(path, encoding="utf-8") as fh:
            manifest = json.loads(fh.read())
        pool_mod.validate_pool_manifest(manifest, self.n)
        self.entry_nets = {}
        for entry in manifest["entries"]:
            entry_path = entry["path"]
            if not os.path.isabs(entry_path):
                entry_path = os.path.join(REPO_ROOT, entry_path)
            self.entry_nets[entry["name"]], unused = self._load_model(
                entry_path, entry["sha256"])
        pfsp = manifest["pfsp"]
        tables = len(manifest["pfsp_tables"])
        windows_max = (pfsp["diag_max_index"] - pfsp["diag_start_index"] + 1) // (
            tables * pfsp["clusters"])
        if windows_max < 1:
            raise ValueError("PFSP diagnostic block too small for one window")
        self.pool_manifest = manifest
        self.logger.info("phase12b pool manifest %s sha256=%s entries=%d "
                         "pfsp_tables=%d biased_tables=%d", path,
                         self.pool_manifest_sha, len(manifest["entries"]),
                         tables, len(manifest["biased_tables"]))

    def _past_set(self, age):
        """window 開始時に凍結する過去方策集合(fork 入力 + age 以下の snapshot)。"""
        past = {}
        fork_meta = os.path.join(self.dir, FORK_DIR, os.path.basename(
            self.fork["source_meta_path"]))
        net, unused = self._load_model(fork_meta, {
            "meta": self.fork["sha256"]["meta"],
            "weights": self.fork["sha256"]["weights"]})
        past["fork"] = (net, self.fork["sha256"]["weights"])
        for row in self.snapshot_rows():
            if row["age"] <= age:
                meta = os.path.join(self.dir, SNAPSHOT_DIR,
                                    "age%04d%s" % (row["age"], GEN_META_SUFFIX))
                net, unused = self._load_model(meta, {
                    "meta": row["sha256"]["meta"],
                    "weights": row["sha256"]["weights"]})
                past["age%04d" % row["age"]] = (net, row["sha256"]["weights"])
        return past

    def window_for(self, age, net, workers):
        """age の属する PFSP window の凍結重みと卓 sampler を返す(無ければ計測)。"""
        pfsp = self.pool_manifest["pfsp"]
        interval = pfsp["interval_generations"]
        window = age // interval
        if window in self.windows:
            return self.windows[window]
        start_age = window * interval
        past = self._past_set(start_age)
        os.makedirs(os.path.join(self.dir, PFSP_DIR), exist_ok=True)
        path = os.path.join(self.dir, PFSP_DIR, "window%03d.json" % window)
        past_shas = dict((k, v[1]) for k, v in sorted(past.items()))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                record = json.loads(fh.read())
            if record["past_set"] != past_shas:
                raise ValueError("PFSP window %d past set changed on resume" % window)
            self.logger.info("phase12b PFSP window %d reloaded (frozen weights)",
                             window)
        else:
            if age != start_age:
                raise ValueError("PFSP window %d record missing mid-window "
                                 "(age %d)" % (window, age))
            record = self._measure_window(window, start_age, past, net, workers)
            record["past_set"] = past_shas
            _write_json_atomic(path, record)
        sampler = pool_mod.TableSampler(
            self.n, self.pool_manifest["pfsp_tables"], record["weights"],
            self.pool_manifest["biased_tables"], sorted(past))
        self.windows = {window: (record, sampler, past)}
        return self.windows[window]

    def _measure_window(self, window, start_age, past, net, workers):
        """window 開始時の learner(greedy)を各 PFSP 凍結卓と巡回クラスタで対戦。"""
        pfsp = self.pool_manifest["pfsp"]
        tables = self.pool_manifest["pfsp_tables"]
        clusters = pfsp["clusters"]
        base = pfsp["diag_start_index"] + window * len(tables) * clusters
        hi = base + len(tables) * clusters - 1
        if hi > pfsp["diag_max_index"]:
            raise ValueError("PFSP diagnostic block exhausted at window %d" % window)
        win_rates = []
        per_table = []
        started = time.monotonic()
        for t, table in enumerate(tables):
            specs = []
            for c in range(clusters):
                index = base + t * clusters + c
                for k in range(self.n):
                    agents = self.table_agents(net, table["opponents"], index,
                                               past, learner_mode="greedy")
                    specs.append((TRAIN_GROUP, index, k, agents))
            trajs = _rollout_or_terminate(specs, workers, self.logger,
                                          self.deadline, randomize_seats=False)
            if trajs is None:
                # 計測途中の SIGTERM/期限: 何も保存せずに終了(再開時に同じ帯で再計測)。
                self.logger.warning("PFSP window %d measurement interrupted; "
                                    "exiting without a record", window)
                _exit_converged(self.logger)
            if any(x.discarded for x in trajs):
                raise _discard_failure(trajs, "pfsp window %d table %s"
                                       % (window, table["id"]))
            usable = [x for x in trajs if not x.discarded]
            wins = sum(1 for x in usable
                       if x.terminal_rewards[x.learner_seat] == 1.0)
            rate = wins / float(len(usable)) if usable else None
            win_rates.append(rate)
            per_table.append({"id": table["id"], "games": len(usable),
                              "wins": wins, "win_rate": rate,
                              "index_lo": base + t * clusters,
                              "index_hi": base + (t + 1) * clusters - 1,
                              "trajectory_sha256": _trajectory_sha256(trajs)})
        weights = pool_mod.pfsp_weights(win_rates, self.n)
        record = {"schema": "phase12b-pfsp-window/v1", "window": window,
                  "start_age": start_age, "group": TRAIN_GROUP,
                  "clusters": clusters, "index_lo": base, "index_hi": hi,
                  "learner_mode": "greedy", "tables": per_table,
                  "x": [pool_mod.pfsp_x(p, self.n) for p in win_rates],
                  "weights": weights, "seconds": time.monotonic() - started}
        self.logger.info("phase12b PFSP window %d measured: win_rates=%s weights=%s",
                         window, win_rates, ["%.4f" % w for w in weights])
        return record

    def table_agents(self, net, opponents, index, past, learner_mode="sample"):
        """正準順 [learner, 席記号...] の agents。seed はエピソード座標だけから。"""
        from naeco.ai.biased_agent import BiasedAgent
        from naeco.ai.random_agent import RandomAgent as _Random
        root = _episode_root(index)
        learner = _CanonicalPolicyAgent(net, mode=learner_mode,
                              seed=derive_seed(root, "seat", index, 1))
        learner.name = "learner"
        agents = [learner]
        for j, token in enumerate(opponents, start=1):
            seed = derive_seed(root, "seat", index, j + 1)
            if token == "@past":
                # 計測卓の @past は席ごとに座標だけから凍結集合を一様抽選
                # (学習卓は TableSampler が解決済みなのでここへは来ない)。
                names = sorted(past)
                if not names:
                    raise ValueError("@past seat needs a frozen past set")
                pick = random.Random(derive_seed(root, "pfsp-past", index, j))
                token = names[pick.randrange(len(names))]
            if token == "@latest":
                agent = _CanonicalPolicyAgent(net, mode="sample", seed=seed)
            elif token == "heuristic":
                agent = HeuristicAgent(seed)
            elif token == "random":
                agent = _Random(seed=seed)
            elif token.startswith("biased:"):
                agent = BiasedAgent(token.split(":", 1)[1], seed=seed)
            elif token in past:
                agent = _CanonicalPolicyAgent(past[token][0], mode="sample", seed=seed)
            else:
                agent = _CanonicalPolicyAgent(self.entry_nets[token], mode="sample", seed=seed)
            agent.name = token
            agents.append(agent)
        return agents

    def exploiter_agents(self, net, index):
        if self.cards is None:
            from naeco.engine import cards as cards_mod
            self.cards = cards_mod.load_cards()
        root = _episode_root(index)
        learner = _CanonicalPolicyAgent(net, mode="sample",
                              seed=derive_seed(root, "seat", index, 1))
        learner.name = "learner"
        agents = [learner]
        for j in range(1, self.n):
            agent = _strong_agent(self.target_net,
                                  derive_seed(root, "seat", index, j + 1),
                                  self.cards)
            agent.name = "target-strong"
            agents.append(agent)
        return agents


def _run_rollout(specs, workers, holder, randomize_seats=True):
    """daemon スレッドで rollout を実行し、結果か例外を holder に記録する。"""
    try:
        holder["trajectories"] = rollout(
            specs, workers=workers, randomize_seats=randomize_seats,
            learner_index=LEARNER_INDEX)
    except Exception:
        holder["error"] = sys.exc_info()


def _rollout_or_terminate(specs, workers, logger, deadline=None,
                          randomize_seats=True):
    """rollout を実行する。SIGTERM 中または deadline 到達時は None を返す。

    共有計算機の stop は PGID 全体へ SIGTERM を送るため、spawn ワーカーが
    死んで pool.map が戻らないことがある。rollout を daemon スレッドで実行し
    main スレッドをブロックしないことで、TERM を検知して確実に保存・終了
    できる(呼び出し側が保存後に os._exit する)。待機は壁時計 deadline の
    ポーリング(反復回数で書かない)。

    deadline は time.monotonic() 基準の絶対 deadline(秒)。None は無制限。
    SIGTERM フラグが立った場合、または deadline に達した場合
    (time.monotonic() >= deadline)は、rollout の完了を待たずに None を
    返す。呼び出し側は _TERMINATE["flag"] の有無で理由を判別する。
    """
    holder = {}
    thread = threading.Thread(
        target=_run_rollout, args=(specs, workers, holder, randomize_seats),
        daemon=True)
    thread.start()
    while thread.is_alive() and not _TERMINATE["flag"]:
        if deadline is not None and time.monotonic() >= deadline:
            break
        thread.join(timeout=0.5)
    if _TERMINATE["flag"]:
        logger.warning("SIGTERM received during rollout")
        return None
    if deadline is not None and time.monotonic() >= deadline:
        logger.warning("wall-clock deadline reached during rollout")
        return None
    if "error" in holder:
        _exc_type, exc, traceback = holder["error"]
        raise exc.with_traceback(traceback)
    return holder["trajectories"]


def _check_phase12b_args(parser, args, raw_argv):
    """Phase 12b の明示指定を検査する(CLI 既定値で fork/lr を成立させない)。"""
    if args.num_players not in (3, 4):
        parser.error("--phase12b-mode is for 3/4 players")
    if not any(a == "--lr" or a.startswith("--lr=") for a in raw_argv):
        parser.error("--phase12b-mode requires an explicit --lr")
    if not any(a == "--start-index" or a.startswith("--start-index=")
               for a in raw_argv):
        parser.error("--phase12b-mode requires an explicit --start-index")
    if args.max_index is None or args.max_index < args.start_index:
        parser.error("--phase12b-mode requires --max-index >= --start-index")
    if args.target_age is None or args.target_age < 1:
        parser.error("--phase12b-mode requires --target-age >= 1")
    if args.lr_stage is None:
        parser.error("--phase12b-mode requires --lr-stage start|extend")
    if args.snapshot_every < 1:
        parser.error("--snapshot-every must be >= 1")
    if args.generations < 1 or args.max_seconds <= 0:
        parser.error("--phase12b-mode requires both --generations>=1 and "
                     "--max-seconds>0 (bounded job)")
    forked = args.phase12b_mode in ("A", "B")
    need = {"fork_from": forked, "fork_sha256": forked,
            "pool_manifest": args.phase12b_mode == "B",
            "init_from": not forked, "init_sha256": not forked,
            "exploit_target": not forked, "exploit_target_sha256": not forked}
    for key, wanted in need.items():
        present = getattr(args, key) is not None
        if wanted and not present:
            parser.error("--phase12b-mode %s requires --%s"
                         % (args.phase12b_mode, key.replace("_", "-")))
        if present and not wanted:
            parser.error("--%s is not used by --phase12b-mode %s"
                         % (key.replace("_", "-"), args.phase12b_mode))
    if args.phase12b_mode == "exploiter" and args.lr_stage != "start":
        parser.error("exploiter runs have no lr extension")
    try:
        if forked:
            _parse_sha_list(args.fork_sha256, ("meta", "weights", "optimizer"))
        else:
            _parse_sha_list(args.init_sha256, ("meta", "weights"))
            _parse_sha_list(args.exploit_target_sha256, ("meta", "weights"))
    except ValueError as exc:
        parser.error(str(exc))


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="train_ppo.py",
        description=(
            "PPO 自己対戦学習 CLI(Phase 3 設計書 §2 / §4 / §8 / §13.2 / "
            "§14.2 / §14.3)。"
            "scripts/csi_job.sh から起動する(docs/csi-jobs.md)。自己対戦 seed は"
            "train グループの 2000 以降のみを使い、dev / accept は使わない"
            "(docs/eval-seeds.md Phase 3 予約)。"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-players", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--checkpoint-dir", metavar="DIR", default=DEFAULT_CHECKPOINT_DIR,
        help="チェックポイントの保存先(既定: リポジトリ内 runs/phase3)")
    parser.add_argument(
        "--start-index", type=int, metavar="N", default=MIN_TRAIN_INDEX,
        help="自己対戦エピソードの開始 index(train 2000 以降)")
    parser.add_argument(
        "--episodes", type=int, metavar="N", default=64,
        dest="episodes_per_generation",
        help="1世代あたりのエピソード数")
    parser.add_argument(
        "--generations", type=int, metavar="N", default=0,
        help="このプロセスで完了させる世代数(0 は無制限。再開時は再開世代からの追加数)")
    parser.add_argument(
        "--max-seconds", type=float, metavar="SECONDS", default=0.0,
        help="壁時計の絶対 deadline(0 は無制限)。上限は反復回数ではなく時刻で判定する")
    parser.add_argument(
        "--workers", type=int, metavar="N", default=4,
        help="rollout の並列プロセス数")
    parser.add_argument(
        "--keep-latest", type=int, metavar="N", default=5,
        help="世代ローテーションで保持する最新世代数")
    parser.add_argument(
        "--seed", type=int, metavar="N", default=20260812,
        help="学習 root seed(初期重みと ppo シャッフル。checkpoint に記録)")
    parser.add_argument(
        "--heuristic-weight", type=float, metavar="W", default=1.0,
        help="相手プールの HeuristicAgent 抽選重み(0 以下で不使用)")
    parser.add_argument(
        "--random-weight", type=float, metavar="W", default=0.0,
        help="相手プールの RandomAgent 抽選重み(0 以下で不使用)")
    parser.add_argument(
        "--hidden", type=int, nargs="+", metavar="N", default=[256, 256],
        help="obs MLP の幅")
    parser.add_argument(
        "--action-hidden", type=int, nargs="+", metavar="N", default=[128],
        help="action MLP の幅")
    for name, default in (
            ("gamma", 0.997), ("gae-lambda", 0.95), ("clip-eps", 0.2),
            ("value-coef", 0.5), ("entropy-coef", 0.01), ("lr", 3e-4)):
        parser.add_argument(
            "--%s" % name, type=float, metavar="V", default=default,
            help="PPO ハイパーパラメータ(既定は設計書 §4.8)")
    parser.add_argument(
        "--epochs", type=int, metavar="N", default=4,
        help="PPO 更新の epoch 数")
    parser.add_argument(
        "--minibatch-size", type=int, metavar="N", default=256,
        help="PPO のミニバッチサイズ")
    parser.add_argument(
        "--max-grad-norm", type=float, metavar="V", default=0.5,
        help="勾配クリップ上限")
    parser.add_argument(
        "--profile-dir", metavar="DIR", default=None,
        help="opt-in timing and fixed-input evidence directory (Phase 10 D7)")
    group = parser.add_argument_group(
        "Phase 12b (opt-in; absent = unchanged Phase 10 behaviour)")
    group.add_argument("--phase12b-mode", choices=PHASE12B_MODES, default=None,
                       help="A: gen199 fork + legacy 50/25/25 pool; B: gen199 "
                       "fork + 50/35/15 PFSP tables; exploiter: learner 1 seat "
                       "vs frozen strong target")
    group.add_argument("--max-index", type=int, default=None, metavar="N",
                       help="last train index of the registered block (required)")
    group.add_argument("--target-age", type=int, default=None, metavar="N",
                       help="stop when the run has completed N generations in "
                       "total (survives SIGTERM/resume; A1=400, A2=800)")
    group.add_argument("--fork-from", metavar="META",
                       help="A/B fresh run: gen199 meta (weights/optimizer siblings)")
    group.add_argument("--fork-sha256", metavar="META:WEIGHTS:OPTIMIZER")
    group.add_argument("--lr-stage", choices=("start", "extend"), default=None,
                       help="start = fork lr; extend = half once (resume only)")
    group.add_argument("--snapshot-every", type=int, default=100, metavar="N")
    group.add_argument("--pool-manifest", metavar="JSON",
                       help="B: phase12b-pool/v1 manifest")
    group.add_argument("--init-from", metavar="META",
                       help="exploiter fresh run: initial weights (fresh Adam)")
    group.add_argument("--init-sha256", metavar="META:WEIGHTS")
    group.add_argument("--exploit-target", metavar="META",
                       help="exploiter: frozen target net used as strong n-1 seats")
    group.add_argument("--exploit-target-sha256", metavar="META:WEIGHTS")
    args = parser.parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if args.phase12b_mode is not None:
        _check_phase12b_args(parser, args, raw_argv)
    elif any(getattr(args, k) is not None for k in (
            "max_index", "target_age", "fork_from", "fork_sha256", "lr_stage",
            "pool_manifest",
            "init_from", "init_sha256", "exploit_target",
            "exploit_target_sha256")):
        parser.error("phase12b options need --phase12b-mode")
    if args.num_players > 2 and (args.heuristic_weight != 1.0 or args.random_weight != 0.0 or args.keep_latest != 5):
        parser.error("multi-player D5 fixes pool categories and keeps five generations")
    try:
        args.start_index = _validate_start_index(args.start_index)
    except ValueError as exc:
        parser.error(str(exc))
    for flag, value in (
            ("--episodes", args.episodes_per_generation),
            ("--workers", args.workers),
            ("--keep-latest", args.keep_latest),
            ("--epochs", args.epochs),
            ("--minibatch-size", args.minibatch_size)):
        if value < 1:
            parser.error("%s must be >= 1, got %d" % (flag, value))
    # --generations は 0(無制限)を既定値とするため、>= 1 検証には含めない。
    # 負値だけを拒否する。
    if args.generations < 0:
        parser.error(
            "--generations must be >= 0 (0 は無制限), got %d"
            % args.generations)
    if args.max_seconds < 0:
        parser.error("--max-seconds must be >= 0, got %g" % args.max_seconds)
    for flag, value in (
            ("--heuristic-weight", args.heuristic_weight),
            ("--random-weight", args.random_weight),
            ("--max-grad-norm", args.max_grad_norm)):
        if value < 0:
            parser.error("%s must be >= 0, got %g" % (flag, value))
    if not args.hidden or any(width < 1 for width in args.hidden):
        parser.error("--hidden must be a non-empty list of positive widths")
    if not args.action_hidden or any(width < 1 for width in args.action_hidden):
        parser.error(
            "--action-hidden must be a non-empty list of positive widths")
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    return args


def run(args, logger=None):
    """学習ループ本体。戻り値はプロセス終了コード(0 は正常/安全終了)。

    torch import とチェックポイント読込の前に checkpoint_dir の排他ロック
    (fcntl.flock、checkpoint.CheckpointLock)を取得する(Phase 4)。取得に
    失敗したら(別ジョブが同一 --checkpoint-dir を保持中)明確なログを出して
    非ゼロを返し、チェックポイントを一切書き込まない。ロックは正常終了・
    SIGTERM 経路の両方で release() され、SIGTERM 中の os._exit(0) 経路でも
    カーネルが自動解放する。
    """
    _fix_thread_env()
    _TERMINATE["flag"] = False
    if logger is None:
        logger = _logger
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    lock = checkpoint_mod.CheckpointLock(checkpoint_dir)
    try:
        lock.acquire()
    except checkpoint_mod.CheckpointLockError as exc:
        logger.error(
            "cannot acquire exclusive lock %s: %s; another train_ppo job "
            "may be running with the same --checkpoint-dir; refusing to "
            "start and writing no checkpoint", lock.path, exc)
        return 1
    logger.info("acquired exclusive checkpoint lock: %s", lock.path)
    try:
        return _run_locked(args, logger)
    finally:
        lock.release()
        logger.info("checkpoint lock released: %s", lock.path)


def _run_locked(args, logger):
    """run() が排他ロックを取得した後に実行する学習ループ本体。

    torch import・チェックポイント読込・保存はすべてこの本体で行う。
    run() が try/finally でロックを解放するため、この本体が return で
    抜ける経路(SIGTERM 保存後・deadline 到達・世代数完了)ではロックは
    確実に release() される。ロールアウト中の SIGTERM で os._exit(0) する
    経路だけは finally を走らないが、fcntl.flock はプロセス死亡時にカーネ
    ルが自動解放するため残置しない。
    """
    num_players = getattr(args, "num_players", 2)
    start_index = _validate_start_index(args.start_index)

    from naeco.train import net as net_mod
    from naeco.train import ppo as ppo_mod
    # torch 操作はすべて ppo.py の包み関数経由で行う(torch を import して
    # よいのは net.py / ppo.py だけ)。set_num_threads(1) は ppo import 時にも
    # 固定されるが、run 時点でも明示する(設計書 §8.4)。
    ppo_mod.set_num_threads(1)
    ppo_mod.manual_seed(args.seed)

    config = ppo_mod.PPOConfig(
        gamma=args.gamma, gae_lambda=args.gae_lambda, clip_eps=args.clip_eps,
        value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        lr=args.lr, epochs=args.epochs,
        minibatch_size=args.minibatch_size, max_grad_norm=args.max_grad_norm)
    encoder = enc
    if num_players > 2:
        from naeco.train import encoding_multi
        encoder = encoding_multi
    obs_dim = encoder.obs_dim(num_players)
    action_dim = encoder.action_dim()
    net = net_mod.PolicyValueNet(
        obs_dim, action_dim, hidden=tuple(args.hidden),
        action_hidden=tuple(args.action_hidden))
    model_arch = net.arch_dict()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(
        "python=%s torch=%s repo_root=%s group=%s start_index=%d workers=%d",
        platform.python_version(), ppo_mod.torch_version(), REPO_ROOT,
        TRAIN_GROUP, start_index, args.workers)
    logger.info(
        "thread_env OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s "
        "OPENBLAS_NUM_THREADS=%s NUMEXPR_NUM_THREADS=%s",
        os.environ.get("OMP_NUM_THREADS"), os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"))

    if getattr(args, "phase12b_mode", None) is not None:
        return _run_phase12b(args, logger, net, net_mod, ppo_mod, config,
                             model_arch, checkpoint_dir, num_players, obs_dim,
                             action_dim)

    latest_generation = _find_latest_generation(checkpoint_dir)
    generation = 0
    training_steps = 0
    next_episode_index = start_index
    # Adam は resume ブロックの前で作成し、旧チェックポイント(optimizer 状態
    # 無し)のときは警告のうえこの新規状態のまま使う(設計書 §14.3)。
    optimizer = ppo_mod.new_adam_optimizer(net.parameters(), config.lr)
    if latest_generation is not None:
        meta, state_dict, optimizer_state = _load_checkpoint(
            checkpoint_dir, latest_generation, model_arch, num_players)
        net.load_state_dict(state_dict)
        generation = int(meta["generation"]) + 1
        training_steps = int(meta["training_steps"])
        next_episode_index = _validate_start_index(
            int(meta["next_episode_index"]))
        logger.info(
            "resumed from generation %03d: generation=%d training_steps=%d "
            "next_episode_index=%d",
            latest_generation, generation, training_steps, next_episode_index)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            logger.info(
                "optimizer state restored from %s",
                _gen_optimizer_path(checkpoint_dir, latest_generation))
        else:
            logger.warning(
                "optimizer state not found in %s; creating a fresh Adam "
                "optimizer (old checkpoint without optimizer state)",
                _gen_optimizer_path(checkpoint_dir, latest_generation))
    else:
        logger.info(
            "fresh run: checkpoint_dir=%s start_index=%d",
            checkpoint_dir, start_index)

    deadline = None
    if args.max_seconds > 0:
        deadline = time.monotonic() + args.max_seconds
        logger.info("wall-clock deadline at +%.1fs", args.max_seconds)

    initial_generation = generation
    total_discarded = 0

    def save_current(reason):
        extra = _meta_extra(
            args, config, generation, training_steps,
            next_episode_index, start_index, ppo_mod.torch_version())
        _save_checkpoint(checkpoint_dir, generation, net, optimizer, extra, num_players)
        logger.info(
            "%s: checkpoint saved (generation=%d training_steps=%d)",
            reason, generation, training_steps)

    while True:
        if _TERMINATE["flag"]:
            logger.warning("SIGTERM received; saving checkpoint and exiting")
            save_current("SIGTERM")
            return 0
        if deadline is not None and time.monotonic() >= deadline:
            logger.info(
                "wall-clock deadline reached; saving checkpoint and exiting")
            save_current("wall-clock deadline")
            return 0
        if args.generations > 0 and \
                generation - initial_generation >= args.generations:
            break

        gen = generation
        gen_start = time.monotonic()
        profile = None
        profile_overhead = 0.0
        if getattr(args, "profile_dir", None):
            from colab.profile import Recorder
            profile = Recorder(args.profile_dir, gen)
            evidence_start = time.monotonic()
            profile.initial(net, optimizer, ppo_mod)
            profile_overhead += time.monotonic() - evidence_start
        past_generations = _list_generations(checkpoint_dir)
        past_entries = []
        entry_nets = {}
        for past_gen in past_generations:
            meta, state_dict, _optimizer_state = _load_checkpoint(
                checkpoint_dir, past_gen, model_arch, num_players)
            past_net = net_mod.PolicyValueNet(
                obs_dim, action_dim, hidden=tuple(args.hidden),
                action_hidden=tuple(args.action_hidden))
            past_net.load_state_dict(state_dict)
            name = "%s%03d" % (GEN_PREFIX, past_gen)
            entry_nets[name] = past_net
            past_entries.append(pool_mod.PoolEntry(
                name, "checkpoint", _gen_meta_path(checkpoint_dir, past_gen),
                1.0))
        latest_entry = pool_mod.PoolEntry(
            "latest", "checkpoint", _gen_meta_path(checkpoint_dir, gen), 1.0)
        opponent_pool = pool_mod.training_pool(
            latest_entry, past_entries,
            heuristic_weight=args.heuristic_weight,
            random_weight=args.random_weight)

        def agents_factory(index):
            if num_players == 2:
                entry = opponent_pool.sample(_episode_root(index), index)
                return _make_agents(net, entry, entry_nets, index)
            root = _episode_root(index)
            learner = PolicyAgent(net, mode="sample",
                                  seed=derive_seed(root, "seat", index, 1))
            learner.name = "learner"
            agents = [learner]
            for j in range(1, num_players):
                entry = opponent_pool.sample_multi(root, index, j)
                seed = derive_seed(root, "seat", index, j + 1)
                if entry.kind == "heuristic":
                    agent = HeuristicAgent(seed)
                else:
                    opponent_net = net if entry.name == "latest" else entry_nets[entry.name]
                    agent = PolicyAgent(opponent_net, mode="sample", seed=seed)
                agent.name = entry.name
                agents.append(agent)
            return agents

        specs = _make_specs(
            next_episode_index, args.episodes_per_generation, agents_factory)
        logger.info(
            "generation=%d episodes=%d index=%d..%d",
            gen, len(specs), specs[0][1], specs[-1][1])
        rollout_start = time.monotonic()
        trajectories = _rollout_or_terminate(
            specs, args.workers, logger, deadline)
        if trajectories is None:
            # SIGTERM がロールアウト中に届いた、または deadline に達した。
            # ワーカーが死んで pool.map が戻らないことがあるため、保存後に
            # os._exit で確実に終了する(atexit の pool 後始末には依存しない)。
            if _TERMINATE["flag"]:
                reason = "SIGTERM during rollout"
            else:
                reason = "wall-clock deadline during rollout"
            save_current(reason)
            _exit_converged(logger)
        rollout_seconds = time.monotonic() - rollout_start
        elapsed = time.monotonic() - gen_start
        if profile is not None:
            evidence_start = time.monotonic()
            profile.rollout(trajectories)
            profile_overhead += time.monotonic() - evidence_start
        diagnostic_seconds = 0.0
        usable, discarded = _select_usable(trajectories)
        if num_players > 2:
            diagnostic_start = time.monotonic()
            diagnostics = ppo_mod.value_diagnostics(usable, net)
            diagnostic_seconds = time.monotonic() - diagnostic_start
            diagnostic_path = os.path.join(checkpoint_dir, "gen%03d.diagnostics.json" % gen)
            with open(diagnostic_path + ".tmp", "w", encoding="utf-8") as stream:
                json.dump({"group": TRAIN_GROUP, "num_players": num_players,
                           "evaluation_seconds": diagnostic_seconds,
                           "value_by_actor_pending": diagnostics,
                           "episodes": [{"index": t.index, "seats": t.seats,
                                         "learner_seat": t.learner_seat}
                                        for t in usable]}, stream, indent=2)
            os.replace(diagnostic_path + ".tmp", diagnostic_path)
        total_discarded += len(discarded)
        decisions_total = sum(len(traj.decisions) for traj in trajectories)
        stats = None
        batch_seconds = 0.0
        update_seconds = 0.0
        if usable:
            batch_start = time.monotonic()
            batch = ppo_mod.trajectories_to_batch(usable, net, config)
            batch_seconds = time.monotonic() - batch_start
            if profile is not None:
                evidence_start = time.monotonic()
                profile.batch(batch, ppo_mod)
                profile_overhead += time.monotonic() - evidence_start
            update_start = time.monotonic()
            stats = ppo_mod.update(
                net, optimizer, batch, config, seed=args.seed)
            update_seconds = time.monotonic() - update_start
            training_steps += 1
        else:
            logger.error(
                "generation=%d: all %d episodes discarded; update skipped",
                gen, len(trajectories))
        next_episode_index += args.episodes_per_generation
        checkpoint_start = time.monotonic()
        save_current("generation %d completed" % gen)
        checkpoint_seconds = time.monotonic() - checkpoint_start
        removed = _prune_checkpoints(checkpoint_dir, args.keep_latest, logger)
        if removed:
            logger.info(
                "rotated out generations: %s",
                ",".join("%03d" % g for g in removed))
        decisions_per_second = (
            decisions_total / elapsed if elapsed > 0 else 0.0)
        if stats is not None:
            logger.info(
                "generation=%d training_steps=%d episodes=%d discarded=%d "
                "discarded_total=%d decisions=%d decisions_per_second=%.1f "
                "elapsed=%.2fs policy_loss=%.6f value_loss=%.6f "
                "entropy=%.6f total_loss=%.6f clip_frac=%.4f approx_kl=%.6f",
                gen, training_steps, len(trajectories), len(discarded),
                total_discarded, decisions_total, decisions_per_second,
                elapsed, stats["policy_loss"], stats["value_loss"],
                stats["entropy"], stats["total_loss"], stats["clip_frac"],
                stats["approx_kl"])
        else:
            logger.error(
                "generation=%d training_steps=%d episodes=%d discarded=%d "
                "discarded_total=%d decisions=%d decisions_per_second=%.1f "
                "elapsed=%.2fs",
                gen, training_steps, len(trajectories), len(discarded),
                total_discarded, decisions_total, decisions_per_second,
                elapsed)
        if profile is not None:
            profile.finish(
                {"rollout": rollout_seconds, "batch_preparation": batch_seconds,
                 "update": update_seconds, "checkpoint": checkpoint_seconds,
                 "instrumentation": profile_overhead,
                 "generation": time.monotonic() - gen_start - diagnostic_seconds},
                {"model": _gen_weights_path(checkpoint_dir, gen),
                 "optimizer": _gen_optimizer_path(checkpoint_dir, gen),
                 "checkpoint_meta": _gen_meta_path(checkpoint_dir, gen)},
                _meta_extra(args, config, gen, training_steps,
                            next_episode_index, start_index,
                            ppo_mod.torch_version()),
                stats, ppo_mod.torch_version())
        generation += 1

    logger.info(
        "training complete: generations=%d training_steps=%d",
        generation, training_steps)
    return 0


def _legacy_multi_agents(net, index, num_players, opponent_pool, entry_nets):
    """Phase 10 の 50/25/25 卓(手法 A の対照条件)。旧ループと同じ組立て。"""
    root = _episode_root(index)
    learner = _CanonicalPolicyAgent(net, mode="sample",
                          seed=derive_seed(root, "seat", index, 1))
    learner.name = "learner"
    agents = [learner]
    for j in range(1, num_players):
        entry = opponent_pool.sample_multi(root, index, j)
        seed = derive_seed(root, "seat", index, j + 1)
        if entry.kind == "heuristic":
            agent = HeuristicAgent(seed)
        else:
            opponent_net = net if entry.name == "latest" else entry_nets[entry.name]
            agent = _CanonicalPolicyAgent(opponent_net, mode="sample", seed=seed)
        agent.name = entry.name
        agents.append(agent)
    return agents


def _run_phase12b(args, logger, net, net_mod, ppo_mod, config, model_arch,
                  checkpoint_dir, num_players, obs_dim, action_dim):
    """Phase 12b の学習ループ(A/B/exploiter)。旧ループとは別関数で持つ。

    座標・age・lr はすべて checkpoint の `hyperparameters.phase12b` に保存し、
    SIGTERM/deadline の中断では未消費 index を進めない(再開時に同じ局を再生)。
    """
    p12 = Phase12bRun(args, logger, checkpoint_dir, num_players, model_arch,
                      net_mod, ppo_mod, obs_dim, action_dim)
    optimizer = ppo_mod.new_adam_optimizer(net.parameters(), args.lr)
    latest_generation = _find_latest_generation(checkpoint_dir)
    generation, training_steps, next_episode_index, age = p12.start(
        net, optimizer, latest_generation)
    deadline = time.monotonic() + args.max_seconds
    p12.deadline = deadline
    logger.info("phase12b mode=%s target_age=%d generations<=%d max_seconds=%.1f",
                args.phase12b_mode, args.target_age, args.generations,
                args.max_seconds)
    log_path = os.path.join(checkpoint_dir, "phase12b-generations.jsonl")
    initial_generation = generation
    state = {"age": age}

    def save_current(reason):
        extra = _meta_extra(args, config, generation, training_steps,
                            next_episode_index, args.start_index,
                            ppo_mod.torch_version())
        hyper = extra["hyperparameters"]
        if args.phase12b_mode == "B":
            hyper["pool_manifest"] = {
                "schema": pool_mod.POOL_SCHEMA, "num_players": num_players,
                "sha256": p12.pool_manifest_sha,
                "categories": dict(pool_mod.TABLE_CATEGORIES)}
        elif args.phase12b_mode == "exploiter":
            hyper["pool_manifest"] = {
                "kind": "exploiter", "num_players": num_players,
                "learner_seats": 1, "opponents": "target strong x (n-1)"}
        hyper["phase12b"] = p12.meta_block(state["age"], optimizer)
        _save_checkpoint(checkpoint_dir, generation, net, optimizer, extra,
                         num_players)
        logger.info("%s: checkpoint saved (generation=%d age=%d "
                    "next_episode_index=%d training_steps=%d)", reason,
                    generation, state["age"], next_episode_index, training_steps)

    def fail(failure):
        _record_phase12b_failure(checkpoint_dir, logger, failure, generation,
                                 state["age"], next_episode_index)
        return failure.code

    # 世代境界で毎回保存するので、中断時に追加保存はしない(更新のない世代
    # ファイルを作らず、未消費 index も進めない)。再開は最後の保存から。
    while True:
        if _TERMINATE["flag"]:
            logger.warning("SIGTERM received; last generation checkpoint is "
                           "current (age=%d next_episode_index=%d); exiting",
                           state["age"], next_episode_index)
            return 0
        if time.monotonic() >= deadline:
            logger.info("wall-clock deadline reached; last checkpoint is current "
                        "(age=%d); exiting", state["age"])
            return 0
        if state["age"] >= args.target_age:
            logger.info("phase12b target age %d reached", args.target_age)
            break
        if generation - initial_generation >= args.generations:
            break
        last = next_episode_index + args.episodes_per_generation - 1
        if last > args.max_index:
            logger.error("phase12b block exhausted: next generation needs "
                         "%d..%d beyond max_index %d; not starting it",
                         next_episode_index, last, args.max_index)
            return PHASE12B_EXIT_BLOCK_EXHAUSTED
        gen = generation
        gen_start = time.monotonic()
        categories = {}
        if args.phase12b_mode == "A":
            entry_nets = {}
            past_entries = []
            for past_gen in _list_generations(checkpoint_dir):
                unused_meta, state_dict, unused_opt = _load_checkpoint(
                    checkpoint_dir, past_gen, model_arch, num_players)
                past_net = p12._new_net()
                past_net.load_state_dict(state_dict)
                name = "%s%03d" % (GEN_PREFIX, past_gen)
                entry_nets[name] = past_net
                past_entries.append(pool_mod.PoolEntry(
                    name, "checkpoint", _gen_meta_path(checkpoint_dir, past_gen),
                    1.0))
            opponent_pool = pool_mod.training_pool(
                pool_mod.PoolEntry("latest", "checkpoint",
                                   _gen_meta_path(checkpoint_dir, gen), 1.0),
                past_entries)

            def agents_factory(index):
                return _legacy_multi_agents(net, index, num_players,
                                            opponent_pool, entry_nets)
        elif args.phase12b_mode == "B":
            try:
                record, sampler, past = p12.window_for(state["age"], net,
                                                       args.workers)
            except Phase12bFailure as failure:
                return fail(failure)

            def agents_factory(index):
                category, table_id, seats = sampler.sample(_episode_root(index),
                                                           index)
                categories[table_id] = categories.get(table_id, 0) + 1
                return p12.table_agents(net, seats, index, past)
        else:
            def agents_factory(index):
                return p12.exploiter_agents(net, index)
        specs = _make_specs(next_episode_index, args.episodes_per_generation,
                            agents_factory)
        logger.info("generation=%d age=%d episodes=%d index=%d..%d", gen,
                    state["age"], len(specs), specs[0][1], specs[-1][1])
        trajectories = _rollout_or_terminate(specs, args.workers, logger, deadline)
        if trajectories is None:
            logger.warning("%s; generation %d discarded unconsumed, resume "
                           "replays index %d", "SIGTERM during rollout"
                           if _TERMINATE["flag"] else "deadline during rollout",
                           gen, next_episode_index)
            _exit_converged(logger)
        usable, discarded = _select_usable(trajectories)
        if discarded:
            # D3: 不合法/InvariantError は1局でも即停止。更新も保存もしない。
            return fail(_discard_failure(trajectories, "generation %d" % gen))
        p12.check_lr(optimizer)
        batch = ppo_mod.trajectories_to_batch(usable, net, config)
        stats = ppo_mod.update(net, optimizer, batch, config, seed=args.seed)
        if not all(math.isfinite(float(stats[k]))
                   for k in ("policy_loss", "value_loss")):
            return fail(Phase12bFailure(
                "nonfinite", PHASE12B_EXIT_NONFINITE,
                {"message": "non-finite update statistics",
                 "policy_loss": repr(stats["policy_loss"]),
                 "value_loss": repr(stats["value_loss"])}))
        p12.check_lr(optimizer)
        training_steps += 1
        state["age"] += 1
        index_lo = next_episode_index
        next_episode_index += args.episodes_per_generation
        save_current("generation %d completed" % gen)
        weights_sha = _sha256_file(_gen_weights_path(checkpoint_dir, gen))
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "age": state["age"], "generation_file": gen,
                "index_lo": index_lo, "index_hi": next_episode_index - 1,
                "episodes": len(trajectories), "discarded": len(discarded),
                "trajectory_sha256": _trajectory_sha256(trajectories),
                "weights_sha256": weights_sha,
                "optimizer_lr": p12.optimizer_lrs(optimizer),
                "tables": categories, "seconds": time.monotonic() - gen_start,
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"]}, sort_keys=True) + "\n")
        _prune_checkpoints(checkpoint_dir, args.keep_latest, logger)
        p12.ensure_snapshot(state["age"], gen)
        logger.info("generation=%d age=%d done discarded=%d elapsed=%.2fs "
                    "policy_loss=%.6f value_loss=%.6f", gen, state["age"],
                    len(discarded), time.monotonic() - gen_start,
                    stats["policy_loss"], stats["value_loss"])
        generation += 1
    logger.info("phase12b training complete: age=%d generation_files=%d "
                "next_episode_index=%d", state["age"], generation,
                next_episode_index)
    return 0


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout)
    args = _parse_args(argv)
    _install_sigterm_handler()
    try:
        return run(args)
    except checkpoint_mod.IncompatibleCheckpoint as exc:
        _logger.error("checkpoint incompatible: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
