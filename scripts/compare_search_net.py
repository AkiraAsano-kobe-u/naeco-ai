"""Phase 5 完了判定用の対局ハーネス(探索あり vs 探索なし、M3/M4)。

search-on(IsmctsAgent(root_seed, budget, cards, evaluator=NetEvaluator(net),
policy_agent=PolicyAgent(net, greedy))) と search-off(PolicyAgent(net, greedy))
の**同一配札・席交換ペア対局**を実行し、`scripts/search_records.py` の bundle
形式(summary 1行 + games details + manifest)で記録する。

- label は "searchon-vs-searchoff" の1つだけを自前で書く。既存の
  decide_search.py は使わない。search_records.py の REQUIRED_LABEL_MATCHUPS /
  REQUIRED_LABELS は変更しない(Phase 2.5 用)。
- CI(ペア単位 bootstrap、B=10000 / seed=2026072102)は summary へ
  ci_low / ci_high / point_estimate として記録する。**判定文(CI 下限 >
  0.50 の合否)は出力しない**。判定は実行後にオーケストレーターが行う。
- net は spawn worker ごとに1回だけロードする(initializer でグローバルへ
  保持し、`_play` 相当の関数がグローバルから読む)。task ごとに net を
  pickle しない。
- 本スクリプトは torch を一切 import しない。重みのロードは net.py の
  `load_state_dict_from_path`(torch.load(map_location="cpu") の包み)へ
  委ねるため、torch の無い /usr/bin/python3 でも本スクリプトを import
  できる。
"""
import os
import sys

# 数値ライブラリを読むより前に固定し、spawn した worker へも同じ環境を渡す。
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _entry in (_SCRIPT_DIR, _REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import tempfile
import time

import search_records
from search_records import (
    assert_thread_env, build_manifest, environment_info,
    find_deterministic_diffs, load_validated_bundles, regenerate_aggregates,
    validate_resume_state, write_bundle, write_report, PairJournal,
)

from naeco.ai import value as value_mod
from naeco.ai.eval.harness import (
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap,
)
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.engine import game
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe
from naeco.engine.observe import OBS_VERSION
from naeco.engine.state import (
    API_VERSION, RULES_VERSION, Phase, decode_rng, derive_seed, encode_rng,
    eval_seed, state_hash,
)
from naeco.train.agent import PolicyAgent
from naeco.train.search_evaluator import NetEvaluator

PYTHON_HASH_SEED = "0"
_HASH_SEED_BOOTSTRAP_MARKER = "NAECO_PYTHON_HASH_SEED_BOOTSTRAPPED"

# このハーネスが書く label と左右の腕(自前で1つだけ書く)。
LABEL = "searchon-vs-searchoff"
LEFT = "search_on"
RIGHT = "search_off"

# 探索腕の固定設定(設計書 phase5-design.md D-1/D-2/D-3)。
DEFAULT_BUDGET = 4500
HORIZON = 2
SCALE = 30
C = 1.41

CHECKPOINT_META_NAME = "gen532.meta.json"
CHECKPOINT_WEIGHTS_NAME = "gen532.weights.pt"


def _hash_seed_contract_error(value):
    return (
        "PYTHONHASHSEED=%r は compare_search_net の正式契約外です。"
        "未設定のまま起動すると CLI が同じ sys.executable/argv で "
        "PYTHONHASHSEED=%s へ一度だけ自己再execし、明示指定は '%s' のみ許可します。"
        % (value, PYTHON_HASH_SEED, PYTHON_HASH_SEED))


def _hash_seed_missing_error():
    return (
        "PYTHONHASHSEED が未設定です。正式CLIは __main__ の bootstrap が同じ "
        "sys.executable/argv で PYTHONHASHSEED=%s へ一度だけ自己execしてから main を "
        "呼びます。import 経由で main を直接呼ぶ場合、現在 process の hash seed を "
        "後付けで固定したことにはできないため、PYTHONHASHSEED=%s を明示してから "
        "呼んでください。" % (PYTHON_HASH_SEED, PYTHON_HASH_SEED))


def _ensure_python_hash_seed_env():
    """pool 生成前に、現在 process が hash seed 0 で実行済みであることを確認する。

    未設定や非0を後付けで修正すると、現在 interpreter の hash seed と manifest の
    記録が食い違う。ブートストラップ済み process だけが "0" を見るため、それ以外は
    拒否する(compare_search と同じ契約)。
    """
    value = os.environ.get("PYTHONHASHSEED")
    if value == PYTHON_HASH_SEED:
        return PYTHON_HASH_SEED
    if value is None:
        raise SystemExit(_hash_seed_missing_error())
    raise SystemExit(_hash_seed_contract_error(value))


def bootstrap_python_hash_seed(argv=None):
    """正式CLIの hash seed ブートストラップ。

    未設定なら同一の `sys.executable`/argv で一度だけ自己execし、現在プロセス
    の hash seed を実効固定する。既に "0" なら再execしない。明示された非0値は
    拒否し、marker が残ったまま未設定に戻る異常時も無限再起動しない。
    """
    value = os.environ.get("PYTHONHASHSEED")
    if value == PYTHON_HASH_SEED:
        return
    if value is not None:
        raise SystemExit(_hash_seed_contract_error(value))
    if os.environ.get(_HASH_SEED_BOOTSTRAP_MARKER):
        raise SystemExit(
            "PYTHONHASHSEED が未設定のまま bootstrap marker が残っています。"
            "自己execを無限に行わないため拒否します。")
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = PYTHON_HASH_SEED
    env[_HASH_SEED_BOOTSTRAP_MARKER] = "1"
    argv = list(sys.argv if argv is None else argv)
    os.execve(sys.executable, [sys.executable] + argv, env)


def peak_rss_bytes():
    """ピークワーキングセットを返す。取得できない環境では None(推定しない)。"""
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def _versions(cards):
    return {
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "api_version": API_VERSION,
        "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
    }


def _environment(workers, thread_env):
    """manifest 側が runtime を検証できる環境 dict を渡す。"""
    _ensure_python_hash_seed_env()
    environment = environment_info(workers, thread_env)
    environment["python_executable"] = sys.executable
    return environment


def _config(group, start_index, pairs, left, right, budget, budget_seconds,
            horizon, scale, c):
    return {
        "group": group, "start_index": start_index, "pairs": pairs,
        "left": left, "right": right, "budget": budget,
        "budget_seconds": budget_seconds, "horizon": horizon,
        "scale": scale, "c": c,
    }


# ---------------------------------------------------------------------------
# net のロード(spawn worker ごとに1回だけ。task ごとに pickle しない)
# ---------------------------------------------------------------------------

_NET = None

_MODEL_ARCH_KEYS = ("obs_dim", "action_dim", "hidden", "action_hidden")


def _read_model_arch(meta_path):
    """gen532.meta.json から model_arch を厳格に検証して読む。

    キー集合の完全一致と型(bool 偽装・float 偽装・欠落)を拒否する
    (eval_agent._read_model_arch と同じ規律。正規化はしない)。
    """
    from naeco.train import checkpoint as checkpoint_mod
    with open(meta_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    arch = raw.get("model_arch")
    if not isinstance(arch, dict):
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: %s model_arch is not an object"
            % meta_path)
    if set(arch) != set(_MODEL_ARCH_KEYS):
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: %s model_arch keys must be exactly "
            "%s (got %s)" % (meta_path, sorted(_MODEL_ARCH_KEYS),
                             sorted(arch)))
    for key in ("obs_dim", "action_dim"):
        if type(arch[key]) is not int:
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: %s model_arch.%s must be an int "
                "(got %s)" % (meta_path, key, type(arch[key]).__name__))
    for key in ("hidden", "action_hidden"):
        value = arch[key]
        if not isinstance(value, list) or \
                any(type(item) is not int for item in value):
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: %s model_arch.%s must be a list "
                "of ints (got %r)" % (meta_path, key, value))
    return arch


def _load_net(checkpoint_dir):
    """gen532 の PolicyValueNet を checkpoint ディレクトリからロードする。

    本スクリプトは torch を import せず、重みのロードは net.py の
    `load_state_dict_from_path` へ委ねる。PolicyValueNet は
    obs_dim=enc.obs_dim(2) / action_dim=enc.action_dim() で構築し、
    gen532.meta.json の model_arch と一致することを検証する(設計書 §5
    D-5 の受入条件。obs_dim=569 / action_dim=109 は gen532 model_arch と
    照合する)。重みは gen532.weights.pt から読み、state_dict のキー・形状を
    照合して load_state_dict + eval() する。
    """
    from naeco.train import checkpoint as checkpoint_mod
    from naeco.train import encoding as enc
    from naeco.train import net as net_mod

    meta_path = os.path.join(checkpoint_dir, CHECKPOINT_META_NAME)
    weights_path = os.path.join(checkpoint_dir, CHECKPOINT_WEIGHTS_NAME)
    arch = _read_model_arch(meta_path)
    checkpoint_mod.load(
        meta_path, expected=checkpoint_mod.expected_meta(2, arch))
    if not os.path.isfile(weights_path):
        raise ValueError(
            "checkpoint %s has meta but weights %s is missing"
            % (meta_path, weights_path))
    obs_dim = enc.obs_dim(2)
    action_dim = enc.action_dim()
    if arch["obs_dim"] != obs_dim or arch["action_dim"] != action_dim:
        raise ValueError(
            "checkpoint incompatible: %s model_arch obs_dim/action_dim "
            "(%d/%d) does not match encoder obs_dim(2)/action_dim() (%d/%d)"
            % (meta_path, arch["obs_dim"], arch["action_dim"],
               obs_dim, action_dim))
    net = net_mod.PolicyValueNet(
        obs_dim, action_dim, hidden=(256, 256), action_hidden=(128,))
    state_dict = net_mod.load_state_dict_from_path(weights_path)
    expected = net.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "checkpoint incompatible: %s weights do not match model_arch "
            "(missing keys %s, unexpected keys %s)"
            % (weights_path, missing, unexpected))
    for name in expected:
        expected_shape = tuple(expected[name].shape)
        actual_shape = tuple(state_dict[name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                "checkpoint incompatible: %s weights do not match model_arch "
                "(%s has shape %s, model_arch requires %s)"
                % (weights_path, name, actual_shape, expected_shape))
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _net_factory_spec(net_factory):
    """テスト用 net_factory を spawn 子へ渡せる spec へ変換する。"""
    module = sys.modules.get(net_factory.__module__)
    file_path = None
    if module is not None:
        file_path = getattr(module, "__file__", None)
    return ("callable", net_factory.__module__, net_factory.__qualname__,
            file_path)


def _resolve_net_factory(spec):
    """spawn 子の中で net_factory を import 解決して呼び出す。"""
    unused_kind, module_name, qualname, file_path = spec
    module = None
    if module_name:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            module = None
    if module is None:
        if not file_path or not os.path.isfile(file_path):
            raise ImportError(
                "cannot import net factory module %r (file %r)"
                % (module_name, file_path))
        module_spec = importlib.util.spec_from_file_location(
            module_name or "net_factory", file_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    factory = module
    for part in qualname.split("."):
        factory = getattr(factory, part)
    return factory()


def _build_net(net_spec):
    kind = net_spec[0]
    if kind == "checkpoint":
        return _load_net(net_spec[1])
    if kind == "callable":
        return _resolve_net_factory(net_spec)
    raise ValueError("unknown net spec: %r" % (net_spec,))


def _init_worker(net_spec):
    """spawn worker の初期化。net を worker ごとに1回だけロードして保持する。

    task ごとに net を pickle しないための worker 側の受け口。`_play` 相当の
    関数はグローバル `_NET` から読む。
    """
    global _NET
    _NET = _build_net(net_spec)


def _make_pool(processes, checkpoint_dir=None, net_factory=None):
    """spawn プールを作る。net は initializer 経由で worker 側がロードする。"""
    if net_factory is None:
        net_spec = ("checkpoint", os.path.abspath(checkpoint_dir))
    else:
        net_spec = _net_factory_spec(net_factory)
    context = multiprocessing.get_context("spawn")
    return context.Pool(processes=processes, initializer=_init_worker,
                        initargs=(net_spec,))


def _prepare_net(checkpoint_dir=None, net_factory=None):
    """pool を使わない実行(in-process)の net を1回だけ用意する。"""
    global _NET
    if _NET is None:
        _NET = (net_factory() if net_factory is not None
                else _load_net(checkpoint_dir))


# ---------------------------------------------------------------------------
# 対局
# ---------------------------------------------------------------------------


def _make_agent(name, seed, budget, horizon, c, cards, budget_seconds=None):
    if name == RIGHT:                       # search_off: 全決定 greedy
        return PolicyAgent(_NET, mode="greedy", seed=0)
    if name == LEFT:                        # search_on: 木探索 + net 委譲
        return IsmctsAgent(seed, budget, c=c, horizon=horizon, cards=cards,
                           budget_seconds=budget_seconds,
                           evaluator=NetEvaluator(_NET),
                           policy_agent=PolicyAgent(_NET, mode="greedy",
                                                    seed=0))
    raise ValueError("unknown agent: %s" % name)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values):
    return (sum(values) / len(values)) if values else None


def _on_arm_metrics(stats):
    """search_on の1局ぶん探索統計(stats.elapsed_ms は決定点ごとの ms)。

    net 呼出総数は探索内の葉評価(`evaluations`)と pending / 一択局面の
    policy_agent 委譲(`delegated_calls`)の合計。委譲は1決定で1回
    policy_agent.act() = 1回の net 呼出なので、
    total_net_calls = evaluations + delegated_calls が net 呼出総数になる。
    """
    elapsed = list(stats.elapsed_ms)
    return {
        "transitions": stats.transitions,
        "nodes_created": stats.nodes_created,
        "determinizations": stats.determinizations,
        "evaluations": stats.evaluations,     # 探索内(木内の葉評価)の net 呼出
        "delegated_calls": stats.delegated_calls,  # 委譲先 policy_agent の net 呼出
        "total_net_calls": stats.evaluations + stats.delegated_calls,
        "elapsed_ms_mean": _mean(elapsed),
        "elapsed_ms_p95": _percentile(elapsed, 0.95),
    }


def _off_arm_metrics(decisions, elapsed):
    """search_off の1局ぶん探索統計。遷移は無く、net 呼出 = 決定数。

    elapsed は決定ごとに実測した思考時間(ms)の list。決定が無いときは
    `_mean` / `_percentile` が空 list に None を返す。
    """
    return {
        "transitions": 0,
        "nodes_created": 0,
        "determinizations": 0,
        "evaluations": decisions,             # = 決定数(全決定で1 net 呼出)
        "elapsed_ms_mean": _mean(elapsed),
        "elapsed_ms_p95": _percentile(elapsed, 0.95),
    }


def _empty_stats_acc():
    return dict(transitions=0, nodes_created=0, determinizations=0,
                evaluations=0, delegated_calls=0, decisions=0, elapsed_ms=[])


def _play(cards, group, root, index, seat_variant, left, right, budget,
          horizon, scale, c, budget_seconds, acc, replay_audit=False):
    """席交換ペアの1局。結果と**1局1行の対応表**(PLAN §5)を返す。

    seed 導出は compare_search._play と同じ:
    root=eval_seed(group,index), env_seed=derive_seed(root,"env",index),
    setup_seed=derive_seed(root,"setup",index),
    policy_seed=derive_seed(root,"policy",index,seat)。ペアの2局は同一配札・
    席交換(seat_variant 0/1)。

    `replay_audit=True` のとき、決定ごとの行動列・各 step 後の state_hash・
    RNG state を `replay` として記録し、worker=1/N の完全一致検証に使う。
    """
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game.new_game(2, env_seed, cards, setup_seed=setup_seed)
    initial_hash = state_hash(state)
    initial_deck = list(state.deck)
    initial_hands = [list(p.hand_buildings) for p in state.players]
    initial_money = [p.money for p in state.players]
    start_player = state.start_player

    left_seat = seat_variant
    names = [right, right]
    names[left_seat] = left
    policy_seeds = [derive_seed(root, "policy", index, seat)
                    for seat in range(2)]
    agents = [_make_agent(name, policy_seeds[seat], budget, horizon, c, cards,
                          budget_seconds)
              for seat, name in enumerate(names)]
    decisions_by_seat = [0, 0]
    elapsed_by_seat = [[], []]
    replay = None
    if replay_audit:
        replay = {"actions": [], "state_hashes": [], "rng_states": []}
    while state.phase != Phase.GAME_OVER:
        actor = game.actor(state)
        observation = observe(state, actor, cards)
        if names[actor] == RIGHT:
            # search_off 腕は決定ごとの実測思考時間(ms)を記録する。
            started = time.perf_counter()
            action = agents[actor].act(observation)
            elapsed_by_seat[actor].append(
                (time.perf_counter() - started) * 1000.0)
        else:
            action = agents[actor].act(observation)
        decisions_by_seat[actor] += 1
        if replay is not None:
            replay["actions"].append([
                action.kind.value, action.target_instance_id,
                action.instance_id, action.goods_count,
            ])
        state = game.step(state, action, cards).state
        if replay is not None:
            replay["state_hashes"].append(state_hash(state))
            replay["rng_states"].append(
                encode_rng(decode_rng(state.rng_state)).hex())
    # 両腕の探索統計。search_on は IsmctsAgent.stats、search_off は
    # 決定数(net 呼出 = 決定数)と実測思考時間(ms)を記録する。
    on_stats = None
    off_decisions = 0
    off_elapsed = []
    for seat, name in enumerate(names):
        if name == LEFT:
            on_stats = agents[seat].stats
        else:
            off_decisions = decisions_by_seat[seat]
            off_elapsed = elapsed_by_seat[seat]
    search_metrics = {
        LEFT: _on_arm_metrics(on_stats),
        RIGHT: _off_arm_metrics(off_decisions, off_elapsed),
    }
    # summary 集計用の acc にも累積する(elapsed_ms は決定点ごとの生値)。
    on_acc = acc["search_metrics"][LEFT]
    on_acc["transitions"] += on_stats.transitions
    on_acc["nodes_created"] += on_stats.nodes_created
    on_acc["determinizations"] += on_stats.determinizations
    on_acc["evaluations"] += on_stats.evaluations
    on_acc["delegated_calls"] += on_stats.delegated_calls
    on_acc["decisions"] += on_stats.decisions
    on_acc["elapsed_ms"].extend(list(on_stats.elapsed_ms))
    off_acc = acc["search_metrics"][RIGHT]
    off_acc["evaluations"] += off_decisions
    off_acc["decisions"] += off_decisions
    off_acc["elapsed_ms"].extend(off_elapsed)

    utility = game.terminal_utilities(state, cards)[left_seat]
    result = 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5
    record = {
        "group": group, "index": index, "seat_variant": seat_variant,
        "left": left, "right": right,
        "left_seat": left_seat, "right_seat": 1 - left_seat,
        "result_left": result,
        "final_scores": list(game.final_scores(state, cards)),
        "env_seed": env_seed, "setup_seed": setup_seed,
        "policy_seed_seat0": policy_seeds[0],
        "policy_seed_seat1": policy_seeds[1],
        "start_player": start_player, "initial_money": initial_money,
        "initial_deck": initial_deck, "initial_hands": initial_hands,
        "initial_state_hash": initial_hash,
        "cards_sha256": cards.sha256, "cards_data_version": cards.data_version,
        "api_version": API_VERSION, "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
        "budget": budget, "budget_seconds": budget_seconds,
        "horizon": horizon, "scale": scale, "c": c,
        "search_metrics": search_metrics,
    }
    if replay is not None:
        record["replay"] = replay
    return result, record


def _run_pair(task):
    """1ペア(同一配札の席交換2局)。worker はこの単位で並列化する。"""
    (group, index, left, right, budget, horizon, scale, c,
     budget_seconds, replay_audit) = task
    value_mod.SCALE = scale               # spawn した worker でも毎回設定する
    cards = load_cards()
    thread_env = assert_thread_env()
    acc = {"rss": None, "thread_env": thread_env,
           "search_metrics": {LEFT: _empty_stats_acc(),
                              RIGHT: _empty_stats_acc()}}
    root = eval_seed(group, index)
    played = [_play(cards, group, root, index, variant, left, right, budget,
                    horizon, scale, c, budget_seconds, acc, replay_audit)
              for variant in (0, 1)]
    acc["rss"] = peak_rss_bytes()
    return [result for result, unused in played], acc, \
        [record for unused, record in played]


def _run_indexed(task):
    """imap_unordered 用: 結果に index を付け、完了順に受け取れるようにする。"""
    return task[1], _run_pair(task)


def _method_metrics(arm, total):
    """方式別の探索統計を summary 用の metrics として組み立てる。"""
    elapsed = total["elapsed_ms"]
    return {
        "method": arm,
        "transitions": total["transitions"],
        "nodes_created": total["nodes_created"],
        "determinizations": total["determinizations"],
        "evaluations": total["evaluations"],
        "delegated_calls": total["delegated_calls"],
        "total_net_calls": total["evaluations"] + total["delegated_calls"],
        "decisions": total["decisions"],
        "elapsed_ms_mean": _mean(elapsed),
        "elapsed_ms_p95": _percentile(elapsed, 0.95),
    }


def run_comparison(group, start_index, pairs, budget, horizon, scale, c,
                   pool=None, replay_audit=False, checkpoint_dir=None,
                   net_factory=None, journal=None, resume=False):
    """席交換ペア対局を実行し、summary row と game records を返す。

    `journal`(PairJournal)があれば完了ペアを 1 ペアごとに追記保存し、
    resume=True なら条件一致のジャーナルの完了済みペアを再利用する。

    pool が None なら in-process で `checkpoint_dir` から net を1回ロードする
    (テストは `net_factory` でフェイク net を注入できる)。pool を使う場合は
    `_make_pool` の initializer が net を worker 側でロード済みであること。
    """
    if pool is None:
        _prepare_net(checkpoint_dir, net_factory)
    tasks = [(group, start_index + offset, LEFT, RIGHT, budget, horizon,
              scale, c, None, replay_audit) for offset in range(pairs)]
    done = {}
    if journal is not None:
        done = journal.open_resume() if resume else (journal.open_new() or {})
        if done:
            print("[resume] %s: ジャーナルから %d/%d ペアを復元"
                  % (journal.key, len(done), pairs), flush=True)
    todo = [task for task in tasks if task[1] not in done]
    started = time.perf_counter()
    by_index = dict((index, (entry["games"], entry["acc"], entry["records"]))
                    for index, entry in done.items())
    stream = (pool.imap_unordered(_run_indexed, todo) if pool   # 完了順
              else (_run_indexed(task) for task in todo))
    for index, result in stream:
        if journal is not None:
            journal.append(index, result[0], result[1], result[2])
        by_index[index] = result
    results = [by_index[task[1]] for task in tasks]
    elapsed = time.perf_counter() - started

    pair_scores, wins, draws, losses = [], 0, 0, 0
    game_records = []
    totals = {LEFT: _empty_stats_acc(), RIGHT: _empty_stats_acc()}
    rss_values = []
    for games, acc, records in results:
        game_records.extend(records)
        for outcome in games:
            wins += outcome == 1.0
            draws += outcome == 0.5
            losses += outcome == 0.0
        pair_scores.append(sum(games) / 2.0)
        for arm, metric in (acc.get("search_metrics") or {}).items():
            into = totals[arm]
            for key in ("transitions", "nodes_created", "determinizations",
                        "evaluations", "delegated_calls", "decisions"):
                into[key] += metric[key]
            into["elapsed_ms"].extend(metric["elapsed_ms"])
        if acc["rss"] is not None:
            rss_values.append(acc["rss"])
    low, high = cluster_bootstrap(pair_scores, BOOTSTRAP_SAMPLES,
                                  BOOTSTRAP_SEED)
    worker_envs = [acc["thread_env"] for unused_games, acc, unused_records
                   in results]
    search_metrics = dict(
        (arm, _method_metrics(arm, totals[arm]))
        for arm in (LEFT, RIGHT))
    on = totals[LEFT]
    row = {
        "group": group, "start_index": start_index, "pairs": pairs,
        "left": LEFT, "right": RIGHT, "budget": budget,
        "budget_seconds": None,
        "horizon": horizon, "scale": scale, "c": c,
        "wins": wins, "draws": draws, "losses": losses,
        "point_estimate": (sum(pair_scores) / len(pair_scores)
                           if pair_scores else 0.0),
        "ci_low": low, "ci_high": high, "pair_scores": pair_scores,
        "decisions": on["decisions"],
        "transitions": on["transitions"],
        "evaluations": on["evaluations"],
        "delegated_calls": on["delegated_calls"],
        "total_net_calls": on["evaluations"] + on["delegated_calls"],
        "mean_think_ms": _mean(on["elapsed_ms"]),
        "p95_think_ms": _percentile(on["elapsed_ms"], 0.95),
        "wall_seconds": elapsed,
        "peak_rss_bytes": max(rss_values) if rss_values else None,
        "worker_thread_envs": worker_envs,
        "search_metrics": search_metrics,
    }
    return row, game_records


def _finalize_bundle(out_path, key, label, row, records, config, versions,
                     environment):
    """summary/detail 双方へ共通 manifest を載せ、label 単位で原子確定する。"""
    manifest = build_manifest(key, label, config, versions, environment)
    row["key"] = key
    row["label"] = label
    row["run_id"] = manifest["run_id"]
    row["manifest"] = manifest
    for name in search_records.CONFIG_KEYS:
        row[name] = config[name]
    for record in records:
        for name in search_records.CONFIG_KEYS:
            record[name] = config[name]
        record["key"] = key
        record["label"] = label
        record["run_id"] = manifest["run_id"]
        record["manifest"] = manifest
    write_bundle(out_path, key, row, records, manifest)


def _report(tag, row):
    # 判定文(CI 下限 > 0.50 の合否)は出力しない。判定はオーケストレーターが
    # 実行後に bundle の ci_low から行う。net呼出 は search_on の net 呼出
    # 総数(探索内 evaluations + 委譲 delegated_calls)。
    print("[%s] %-10s %-6s T=%-5d 勝率=%.3f CI=[%.3f,%.3f] 遷移=%.0f "
          "net呼出=%.0f %.0fs"
          % (tag, LABEL, row["left"], row["budget"],
             row["point_estimate"], row["ci_low"], row["ci_high"],
             row["transitions"], row["total_net_calls"],
             row["wall_seconds"]),
          flush=True)


def _checkpoint_fingerprint(checkpoint_dir):
    """実際にロードする meta と重みファイルの内容ハッシュ(パスは含めない)。"""
    digest = hashlib.sha256()
    for name in (CHECKPOINT_META_NAME, CHECKPOINT_WEIGHTS_NAME):
        path = os.path.join(checkpoint_dir, name)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            # 実運用では _validate_checkpoint_dir が先に存在を保証する。
            # ここに来るのは loader を差し替えたテスト等だけなので、
            # 別パスは別モデル扱いになるようパスを指紋へ入れる。
            data = b"missing:" + os.path.abspath(path).encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def run_phase5_comparison(group, start_index, pairs, budget, horizon, scale,
                          c, out_path, pool=None, resume=False,
                          environment=None, replay_audit=False,
                          checkpoint_dir=None, net_factory=None):
    """bundle 最終化込みの1ラベル実行。resume 時は検証済み bundle をスキップ。"""
    key = LABEL
    config = _config(group, start_index, pairs, LEFT, RIGHT, budget, None,
                     horizon, scale, c)
    if out_path and resume:
        expected = {key: build_manifest(key, LABEL, config,
                                        _versions(load_cards()), environment)}
        existing = validate_resume_state(out_path, [key], expected)
        if key in existing:
            print("[phase5] %s は検証済み bundle 済み。スキップする。" % key,
                  flush=True)
            return existing[key]["summary"]
    journal = None
    if out_path:
        manifest = build_manifest(key, LABEL, config, _versions(load_cards()),
                                  environment)
        # 重みの内容ハッシュを header 条件へ入れる(別モデルの resume を拒否)。
        # net_factory 注入(テスト用)では checkpoint が無いので None。
        fingerprint = (_checkpoint_fingerprint(checkpoint_dir)
                       if checkpoint_dir and net_factory is None else None)
        journal = PairJournal(out_path, key, manifest["run_id"], start_index,
                              pairs, replay_audit, fingerprint)
    row, game_records = run_comparison(group, start_index, pairs, budget,
                                       horizon, scale, c, pool, replay_audit,
                                       checkpoint_dir, net_factory, journal,
                                       resume)
    _finalize_bundle(out_path, key, LABEL, row, game_records, config,
                     _versions(load_cards()), environment)
    if journal is not None:
        journal.discard()
    _report("phase5", row)
    return row


# ---------------------------------------------------------------------------
# 決定論検査
# ---------------------------------------------------------------------------

# 決定論比較から外す path の明示 allowlist。search_records の既定に加えて、
# games / summary の search_metrics 内 elapsed_ms(壁時計由来)も外す。
DETERMINISM_EXCLUDED_PATHS = search_records.DETERMINISM_EXCLUDED_PATHS + (
    ("games", "*", "search_metrics", "*", "elapsed_ms_mean"),
    ("games", "*", "search_metrics", "*", "elapsed_ms_p95"),
    ("summary", "search_metrics", "*", "elapsed_ms_mean"),
    ("summary", "search_metrics", "*", "elapsed_ms_p95"),
)


def _dotted_path(path):
    """path タプルを report 用のドット記法へ変換する(`*` は `[]`)。"""
    return ".".join("[]" if part == "*" else str(part) for part in path)


DETERMINISM_EXCLUDED = tuple(_dotted_path(path)
                             for path in DETERMINISM_EXCLUDED_PATHS)


def _determinism_report(group, start_index, pairs, budget, budget_seconds,
                        horizon, scale, c, workers, thread_env, environment,
                        replay_audit, checkpoint_dir):
    """決定論検査の formal report を組み立てる。

    report 単独で実行 runtime を監査できるよう python_version と
    python_executable を環境情報から保存する(compare_search と同じ契約)。
    """
    return {
        "schema_version": "phase5-determinism/v1",
        "group": group,
        "start_index": start_index,
        "pairs": pairs,
        "config": {
            "budget": budget,
            "budget_seconds": budget_seconds,
            "horizon": horizon,
            "scale": scale,
            "c": c,
        },
        "workers": workers,
        "label": LABEL,
        "checkpoint_dir": (os.path.abspath(checkpoint_dir)
                           if checkpoint_dir else None),
        "excluded_fields": sorted(DETERMINISM_EXCLUDED),
        "replay_audit": replay_audit,
        "thread_env": thread_env,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_version": environment["python_version"],
        "python_executable": environment["python_executable"],
        "source_digest": environment["source_digest"],
        "all_labels_matched": True,
    }


def run_determinism_check(pairs, budget, horizon, scale, c, start_index,
                          records, workers, group="dev", replay_audit=False,
                          checkpoint_dir=None, net_factory=None):
    """worker=1 と worker=N を同じ設定で実行し、決定論的 field を比較する。

    成功時は指定 `records` へ原子 JSON report を保存する。既存ファイルは拒否。
    worker はすべて同一 hash seed "0" で spawn する(hash seed の固定は CLI
    ブートストラップが通常 run と共通に行い、ここでは契約値を確認する)。
    """
    _ensure_python_hash_seed_env()
    if pairs is None:
        raise SystemExit("--check-determinism には --pairs の明示が必要")
    if checkpoint_dir is not None:
        _validate_checkpoint_dir(checkpoint_dir)
    if os.path.exists(records):
        raise SystemExit("決定論検査の報告先が既に存在する: %s" % records)
    thread_env = dict((name, os.environ.get(name)) for name in
                      search_records.THREAD_ENV_NAMES)
    with tempfile.TemporaryDirectory(prefix="det-check-") as tmp:
        path_one = os.path.join(tmp, "worker1.jsonl")
        path_many = os.path.join(tmp, "worker%d.jsonl" % workers)
        env_one = _environment(1, thread_env)
        env_many = _environment(workers, thread_env)
        pool_one = _make_pool(1, checkpoint_dir, net_factory)
        try:
            run_phase5_comparison(group, start_index, pairs, budget, horizon,
                                  scale, c, path_one, pool=pool_one,
                                  environment=env_one,
                                  replay_audit=replay_audit)
        finally:
            pool_one.close()
            pool_one.join()
        pool_many = _make_pool(workers, checkpoint_dir, net_factory)
        try:
            run_phase5_comparison(group, start_index, pairs, budget, horizon,
                                  scale, c, path_many, pool=pool_many,
                                  environment=env_many,
                                  replay_audit=replay_audit)
        finally:
            pool_many.close()
            pool_many.join()
        one = load_validated_bundles(path_one)
        many = load_validated_bundles(path_many)
        if replay_audit:
            for bundle in (one[LABEL], many[LABEL]):
                for game in bundle["games"]:
                    if not isinstance(game.get("replay"), dict):
                        raise SystemExit(
                            "--replay-audit が有効なのに %s に replay 記録"
                            "がない" % LABEL)
        diffs = find_deterministic_diffs(
            one[LABEL]["summary"], many[LABEL]["summary"],
            one[LABEL]["games"], many[LABEL]["games"],
            excluded=DETERMINISM_EXCLUDED_PATHS)
        if diffs:
            raise SystemExit("決定論不一致 %s:\n%s" % (LABEL, "\n".join(diffs)))
        print("[det] %s: worker=1 と worker=%d の決定論的 field は一致"
              % (LABEL, workers), flush=True)
    report = _determinism_report(
        group, start_index, pairs, budget, None, horizon, scale, c,
        workers, thread_env, env_one, replay_audit, checkpoint_dir)
    write_report(records, report)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_checkpoint_dir(checkpoint_dir):
    """gen532.meta.json + gen532.weights.pt の存在を要求する。"""
    meta_path = os.path.join(checkpoint_dir, CHECKPOINT_META_NAME)
    weights_path = os.path.join(checkpoint_dir, CHECKPOINT_WEIGHTS_NAME)
    if not os.path.isfile(meta_path) or not os.path.isfile(weights_path):
        raise SystemExit(
            "--checkpoint-dir に %s と %s が必要です: %s"
            % (CHECKPOINT_META_NAME, CHECKPOINT_WEIGHTS_NAME, checkpoint_dir))


def main(argv=None):
    _ensure_python_hash_seed_env()
    parser = argparse.ArgumentParser(
        description="Phase 5: 探索あり(search-on) vs 探索なし(search-off) "
                    "の対局ハーネス")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="gen532.meta.json と gen532.weights.pt を"
                             "含むディレクトリ")
    parser.add_argument("--group", choices=("train", "dev"), default="dev")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                        help="1決定点あたりの状態遷移数(既定 4500)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--records", default=None,
                        help="出力 bundle パス")
    parser.add_argument("--resume", action="store_true",
                        help="既存 bundle の続きへ追記する。manifest 一致・"
                             "完全性を検証済みの bundle はスキップする")
    parser.add_argument("--execute", action="store_true",
                        help="実際に対局を実行する")
    parser.add_argument("--regenerate", action="store_true",
                        help="bundle から summary / -games / .manifest を"
                             "再生成する")
    parser.add_argument("--check-determinism", action="store_true",
                        help="worker=1 と worker=N で同じ対戦を実行し、"
                             "決定論的 field を比較する")
    parser.add_argument("--replay-audit", action="store_true",
                        help="--check-determinism と併用し、行動列・各 step "
                             "後の state_hash・RNG state まで完全一致を"
                             "検証する(既定 off)")
    args = parser.parse_args(argv)

    threads = ",".join("%s=%s" % (name, os.environ[name])
                       for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS"))
    if args.regenerate:
        if not args.records:
            raise SystemExit("--regenerate には --records が必要")
        bundles = load_validated_bundles(args.records)
        regenerate_aggregates(args.records, bundles)
        print("regenerated aggregates from %d bundles" % len(bundles))
        return 0
    if args.replay_audit and not args.check_determinism:
        raise SystemExit("--replay-audit は --check-determinism と併用してください")
    if not args.execute:
        print("prepared: label=%s workers=%d thread_env=%s"
              % (LABEL, args.workers, threads))
        return 0
    print("thread_env=%s workers=%d" % (threads, args.workers), flush=True)

    if args.check_determinism:
        if args.workers < 2:
            raise SystemExit("--check-determinism には --workers >= 2 が必要")
        if not args.records:
            raise SystemExit("--check-determinism には --records が必要")
        if args.pairs is None:
            raise SystemExit("--check-determinism には --pairs の明示が必要")
        if args.start_index is None:
            raise SystemExit("--check-determinism には --start-index の明示が必要")
        if args.pairs <= 0:
            raise SystemExit("--pairs は正の整数が必要")
        if args.start_index < 0:
            raise SystemExit("--start-index は 0 以上が必要")
        if args.budget <= 0:
            raise SystemExit("--budget は正の整数が必要")
        return run_determinism_check(
            args.pairs, args.budget, HORIZON, SCALE, C,
            args.start_index, args.records, args.workers, args.group,
            replay_audit=args.replay_audit,
            checkpoint_dir=args.checkpoint_dir)

    if args.pairs is None:
        raise SystemExit("--execute には --pairs の明示が必要")
    if args.start_index is None:
        raise SystemExit("--execute には --start-index の明示が必要")
    if args.pairs <= 0:
        raise SystemExit("--pairs は正の整数が必要")
    if args.start_index < 0:
        raise SystemExit("--start-index は 0 以上が必要")
    if args.budget <= 0:
        raise SystemExit("--budget は正の整数が必要")
    if args.resume and not args.records:
        raise SystemExit("--resume には --records が必要")
    if not args.resume and args.records:
        candidates = [args.records,
                      args.records.replace(".jsonl", "") + "-games.jsonl",
                      args.records + ".manifest.jsonl"]
        for candidate in candidates:
            if os.path.exists(candidate):
                raise SystemExit(
                    "記録先が既に存在する: %s / "
                    "追記すると『1回だけ実行』の証跡が壊れる。"
                    "別名を指定するか退避すること。中断した実行の続きなら "
                    "--resume を明示する(検証済み bundle はスキップされる)。"
                    % candidate)
        bundle_dir = args.records + "-bundle"
        if os.path.isdir(bundle_dir) and os.listdir(bundle_dir):
            raise SystemExit("bundle ディレクトリが既に存在する: %s"
                             % bundle_dir)
    _validate_checkpoint_dir(args.checkpoint_dir)

    thread_env = dict((name, os.environ.get(name)) for name in
                      search_records.THREAD_ENV_NAMES)
    environment = _environment(args.workers, thread_env)
    pool = _make_pool(args.workers, args.checkpoint_dir) \
        if args.workers > 1 else None
    try:
        run_phase5_comparison(
            args.group, args.start_index, args.pairs, args.budget,
            HORIZON, SCALE, C, args.records, pool, args.resume, environment,
            replay_audit=args.replay_audit,
            checkpoint_dir=args.checkpoint_dir)
        if args.records:
            regenerate_aggregates(args.records,
                                  load_validated_bundles(args.records))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return 0


if __name__ == "__main__":
    bootstrap_python_hash_seed()
    raise SystemExit(main())
