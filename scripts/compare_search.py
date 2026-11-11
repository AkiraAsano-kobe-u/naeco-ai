"""Phase 2.5 の予算感度(段階1)と本比較(段階2)を実行する。

設計書 `docs/phase2.5-design.md` §6.4 / §7 の事前登録値に従う。
実験は `--execute` を明示したときだけ走る。label 単位の結果は
`scripts/search_records.write_bundle` で原子的に確定し、
summary / -games / .manifest は bundle から再生成する。
"""
import os
import sys

# 数値ライブラリを読むより前に固定し、spawn した worker へも同じ環境を渡す。
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import multiprocessing
import tempfile
import time

import search_records
from search_records import (
    PairJournal, aggregate_audit, assert_thread_env, build_manifest, environment_info,
    find_deterministic_diffs, load_validated_bundles, regenerate_aggregates,
    validate_audit, validate_resume_state, write_bundle, write_report,
)

from naeco.ai import value as value_mod
from naeco.ai.eval.harness import (
    BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap,
)
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.ai.search.pimc import PimcAgent
from naeco.engine import game
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe
from naeco.engine.state import (
    API_VERSION, RULES_VERSION, Phase, decode_rng, derive_seed, encode_rng,
    eval_seed, state_hash,
)
from naeco.engine.observe import OBS_VERSION

PYTHON_HASH_SEED = "0"
_HASH_SEED_BOOTSTRAP_MARKER = "NAECO_PYTHON_HASH_SEED_BOOTSTRAPPED"


def _hash_seed_contract_error(value):
    return (
        "PYTHONHASHSEED=%r は compare_search の正式契約外です。"
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
    拒否する。
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

TRAIN_PAIRS = 50
TRAIN_INDEX = 0
DEV_PAIRS = 500
DEV_INDEX = 2000

# 1決定点あたりの状態遷移数(設計書 §7.1。1標本 = 28〜32遷移の実測にもとづく)。
BUDGETS = (500, 1500, 4500)
MID_BUDGET = 1500

# 既定値から一変数ずつ動かす座標降下一巡(設計書 §6.3)。`K` は出力値なので走査しない。
DEFAULT_CONFIG = ("default", 2, 30, 1.41)
COORDINATE_CONFIGS = (
    ("horizon-1", 1, 30, 1.41),
    ("horizon-3", 3, 30, 1.41),
    ("scale-20", 2, 20, 1.41),
    ("scale-50", 2, 50, 1.41),
    ("ucb-0.7", 2, 30, 0.70),      # UCB 係数は ISMCTS にしか効かない
    ("ucb-2.0", 2, 30, 2.00),
)
_UCB_ONLY = ("ucb-0.7", "ucb-2.0")


def peak_rss_bytes():
    """ピークワーキングセットを返す。取得できない環境では None(推定しない)。"""
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        # restype / argtypes を明示する。省くと 64bit で疑似ハンドルが 32bit へ
        # 切り詰められ、GetProcessMemoryInfo が常に失敗する(戻り値 0)。
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:
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


def _agent(name, seed, budget, horizon, c, cards, budget_seconds=None):
    if name == "heuristic":
        return HeuristicAgent(seed)
    if name == "pimc":
        return PimcAgent(seed, budget, horizon=horizon, cards=cards,
                         budget_seconds=budget_seconds)
    if name == "ismcts":
        return IsmctsAgent(seed, budget, c=c, horizon=horizon, cards=cards,
                           budget_seconds=budget_seconds)
    raise ValueError("unknown agent: %s" % name)


def _merge_stats(into, agent):
    """探索エージェントの統計を足し込む(ヒューリスティックは統計を持たない)。"""
    stats = getattr(agent, "stats", None)
    if stats is None:
        return
    for key in ("decisions", "transitions", "sampler_calls", "nodes_created",
                "determinizations", "truncated_samples", "clip_saturated",
                "evaluations", "total_nodes",
                "decisions_with_zero_sample_actions", "deadline_overrun_count"):
        into[key] += int(getattr(stats, key, 0) or 0)
    into["search_elapsed_ms"] += float(
        getattr(stats, "search_elapsed_ms", 0.0) or 0.0)
    into["deadline_overrun_total_ms"] += float(
        getattr(stats, "deadline_overrun_total_ms", 0.0) or 0.0)
    into["deadline_overrun_max_ms"] = max(
        into["deadline_overrun_max_ms"],
        float(getattr(stats, "deadline_overrun_max_ms", 0.0) or 0.0))
    into["elapsed_ms"].extend(list(getattr(stats, "elapsed_ms", ())))
    into["samples_per_action"].extend(
        list(getattr(stats, "samples_per_action", ())))
    into["transitions_per_action"].extend(
        list(getattr(stats, "transitions_per_action", ())))
    into["max_sample_gap"].extend(
        list(getattr(stats, "max_sample_gap", ())))
    into["zero_sample_actions"].extend(
        list(getattr(stats, "zero_sample_actions", ())))
    into["deadline_overruns_ms"].extend(
        list(getattr(stats, "deadline_overruns_ms", ())))


def _empty_stats_acc():
    return dict(decisions=0, transitions=0, sampler_calls=0, nodes_created=0,
                total_nodes=0, determinizations=0, truncated_samples=0,
                clip_saturated=0, evaluations=0, search_elapsed_ms=0.0,
                elapsed_ms=[], samples_per_action=[],
                transitions_per_action=[], max_sample_gap=[],
                zero_sample_actions=[], decisions_with_zero_sample_actions=0,
                deadline_overruns_ms=[], deadline_overrun_count=0,
                deadline_overrun_total_ms=0.0, deadline_overrun_max_ms=0.0)


def _merge_acc(into, source):
    """方式別に蓄積した統計 acc を、別の acc へ足し込む。"""
    for key in ("decisions", "transitions", "sampler_calls", "nodes_created",
                "total_nodes", "determinizations", "truncated_samples",
                "clip_saturated", "evaluations",
                "decisions_with_zero_sample_actions", "deadline_overrun_count"):
        into[key] += int(source.get(key, 0) or 0)
    into["search_elapsed_ms"] += float(
        source.get("search_elapsed_ms", 0.0) or 0.0)
    into["deadline_overrun_total_ms"] += float(
        source.get("deadline_overrun_total_ms", 0.0) or 0.0)
    into["deadline_overrun_max_ms"] = max(
        into["deadline_overrun_max_ms"],
        float(source.get("deadline_overrun_max_ms", 0.0) or 0.0))
    for name in ("elapsed_ms", "samples_per_action", "transitions_per_action",
                 "max_sample_gap", "zero_sample_actions",
                 "deadline_overruns_ms"):
        into[name].extend(list(source.get(name, ())))


def _play(cards, group, root, index, seat_variant, left, right, budget,
          horizon, scale, c, budget_seconds, acc, replay_audit=False):
    """席交換ペアの1局。結果と**1局1行の対応表**(PLAN §5)を返す。

    集計値だけでは、席の均衡・同一配札・seed 対応・版固定を保存物から独立に
    監査できない(sol SOL-P25G-002)。Phase 2 の `GameRecord` と同じ水準で残す。
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
    policy_seeds = [derive_seed(root, "policy", index, seat) for seat in range(2)]
    agents = [_agent(name, policy_seeds[seat], budget, horizon, c, cards,
                     budget_seconds)
              for seat, name in enumerate(names)]
    replay = None
    if replay_audit:
        replay = {"actions": [], "state_hashes": [], "rng_states": []}
    while state.phase != Phase.GAME_OVER:
        actor = game.actor(state)
        action = agents[actor].act(observe(state, actor, cards))
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
    # 直接対決では両方式の探索統計が必要。方式別へ全探索 agent を集約し、
    # トップレベルは従来どおり left 方式の互換 field として残す。
    for seat, name in enumerate(names):
        if name in ("pimc", "ismcts"):
            method_acc = acc["search_metrics"].setdefault(
                name, _empty_stats_acc())
            _merge_stats(method_acc, agents[seat])
    _merge_stats(acc, agents[left_seat])
    utility = game.terminal_utilities(state, cards)[left_seat]
    result = 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5
    record = {
        "group": group, "index": index, "seat_variant": seat_variant,
        "left": left, "right": right,
        "left_seat": left_seat, "right_seat": 1 - left_seat,
        "result_left": result,
        "final_scores": list(game.final_scores(state, cards)),
        "env_seed": env_seed, "setup_seed": setup_seed,
        "policy_seed_seat0": policy_seeds[0], "policy_seed_seat1": policy_seeds[1],
        "start_player": start_player, "initial_money": initial_money,
        "initial_deck": initial_deck, "initial_hands": initial_hands,
        "initial_state_hash": initial_hash,
        "cards_sha256": cards.sha256, "cards_data_version": cards.data_version,
        "api_version": API_VERSION, "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
        "budget": budget, "budget_seconds": budget_seconds,
        "horizon": horizon, "scale": scale, "c": c,
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
    acc = _empty_stats_acc()
    acc["search_metrics"] = {}
    root = eval_seed(group, index)
    played = [_play(cards, group, root, index, variant, left, right, budget,
                    horizon, scale, c, budget_seconds, acc, replay_audit)
              for variant in (0, 1)]
    acc["rss"] = peak_rss_bytes()
    acc["thread_env"] = thread_env
    return [result for result, unused in played], acc, [rec for unused, rec in played]


def _run_indexed(task):
    """imap_unordered 用: 結果に index を付け、完了順に受け取れるようにする。"""
    return task[1], _run_pair(task)


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _sum_run_totals(results):
    """ペア acc の数値統計を集約する。max 指標は max で畳む。"""
    total = dict(decisions=0, transitions=0, sampler_calls=0, nodes_created=0,
                 total_nodes=0, determinizations=0, truncated_samples=0,
                 clip_saturated=0, evaluations=0, search_elapsed_ms=0.0,
                 decisions_with_zero_sample_actions=0,
                 deadline_overrun_count=0, deadline_overrun_total_ms=0.0,
                 deadline_overrun_max_ms=0.0)
    for unused_games, acc, unused_records in results:
        for key in total:
            value = acc.get(key, 0.0)
            if key == "deadline_overrun_max_ms":
                total[key] = max(total[key], float(value or 0.0))
            else:
                total[key] += value
    return total


def run_comparison(group, start_index, pairs, left, right, budget, horizon,
                   scale, c, pool=None, budget_seconds=None,
                   replay_audit=False, journal=None, resume=False):
    """席交換ペア対局を実行し、勝率と計測値をまとめて返す。

    `journal`(PairJournal)があれば完了ペアを 1 ペアごとに追記保存する。
    `resume=True` なら条件一致のジャーナルから完了済みペアを読み、不足分だけ走らせる。
    """
    tasks = [(group, start_index + offset, left, right, budget, horizon, scale,
              c, budget_seconds, replay_audit)
             for offset in range(pairs)]
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
    if pool:
        stream = pool.imap_unordered(_run_indexed, todo)   # 完了順に受け取る
    else:
        stream = (_run_indexed(task) for task in todo)
    for index, result in stream:
        if journal is not None:
            journal.append(index, result[0], result[1], result[2])
        by_index[index] = result
    results = [by_index[task[1]] for task in tasks]
    elapsed = time.perf_counter() - started
    # elapsed は今回実行した不足ペアだけの時間。速度の分子も同じペアに揃える
    # (復元分を含めると resume 後の速度が水増しされる)。
    ran_transitions = _sum_run_totals(
        [by_index[task[1]] for task in todo])["transitions"]

    pair_scores, wins, draws, losses = [], 0, 0, 0
    total = _sum_run_totals(results)
    think_ms, rss_values = [], []
    game_records = []
    method_totals = {}
    for games, acc, records in results:
        game_records.extend(records)
        for outcome in games:
            wins += outcome == 1.0
            draws += outcome == 0.5
            losses += outcome == 0.0
        pair_scores.append(sum(games) / 2.0)
        think_ms.extend(acc["elapsed_ms"])
        if acc["rss"] is not None:
            rss_values.append(acc["rss"])
        for method, metric in (acc.get("search_metrics") or {}).items():
            into = method_totals.setdefault(method, _empty_stats_acc())
            _merge_acc(into, metric)
    low, high = cluster_bootstrap(pair_scores, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
    decisions = total["decisions"] or 1
    search_seconds = total["search_elapsed_ms"] / 1000.0
    worker_envs = [acc["thread_env"] for unused_games, acc, unused_records
                   in results]
    audit = aggregate_audit([acc for unused_games, acc, unused_records in results])
    search_metrics = dict(
        (method, _method_metrics(method, total_metric, worker_envs))
        for method, total_metric in sorted(method_totals.items()))
    row = {
        "group": group, "start_index": start_index, "pairs": pairs,
        "left": left, "right": right, "budget": budget,
        "budget_seconds": budget_seconds,
        "horizon": horizon, "scale": scale, "c": c,
        "wins": wins, "draws": draws, "losses": losses,
        "point_estimate": sum(pair_scores) / len(pair_scores),
        "ci_low": low, "ci_high": high, "pair_scores": pair_scores,
        "mean_think_ms": (sum(think_ms) / len(think_ms)) if think_ms else None,
        "p95_think_ms": _percentile(think_ms, 0.95),
        "decisions": total["decisions"],
        "transitions_per_second": (ran_transitions / elapsed
                                   if todo and elapsed else None),
        "sampler_calls_per_decision": total["sampler_calls"] / decisions,
        "determinizations_per_decision": total["determinizations"] / decisions,
        "nodes_per_decision": total["nodes_created"] / decisions,
        "nodes_per_second": (total["total_nodes"] / search_seconds
                             if search_seconds else None),
        "truncated_samples": total["truncated_samples"],
        "peak_rss_bytes": max(rss_values) if rss_values else None,
        "clip_saturation_rate": (total["clip_saturated"] / total["evaluations"]
                                 if total["evaluations"] else None),
        "wall_seconds": elapsed,
        "evaluations": total["evaluations"],
        "clip_saturated": total["clip_saturated"],
        "max_sample_gap": audit["max_sample_gap"],
        "zero_sample_actions": audit["zero_sample_actions"],
        "decisions_with_zero_sample_actions":
            audit["decisions_with_zero_sample_actions"],
        "action_transition_distribution":
            audit["action_transition_distribution"],
        "deadline_overrun_count": audit["deadline_overrun_count"],
        "deadline_overrun_total_ms": audit["deadline_overrun_total_ms"],
        "deadline_overrun_max_ms": audit["deadline_overrun_max_ms"],
        "worker_thread_envs": worker_envs,
        "search_metrics": search_metrics,
    }
    return row, game_records


def _method_metrics(method, total, worker_envs):
    """方式別の探索統計を summary 用の metrics として組み立てる。"""
    decisions = total["decisions"] or 1
    think_ms = total["elapsed_ms"]
    search_seconds = total["search_elapsed_ms"] / 1000.0
    audit = aggregate_audit([total])
    return {
        "method": method,
        "decisions": total["decisions"],
        "transitions": total["transitions"],
        "sampler_calls": total["sampler_calls"],
        "determinizations": total["determinizations"],
        "nodes_created": total["nodes_created"],
        "total_nodes": total["total_nodes"],
        "search_elapsed_ms": total["search_elapsed_ms"],
        "nodes_per_second": (total["total_nodes"] / search_seconds
                             if search_seconds else None),
        "mean_think_ms": (sum(think_ms) / len(think_ms)) if think_ms else None,
        "p95_think_ms": _percentile(think_ms, 0.95),
        "truncated_samples": total["truncated_samples"],
        "evaluations": total["evaluations"],
        "clip_saturated": total["clip_saturated"],
        "clip_saturation_rate": (total["clip_saturated"] / total["evaluations"]
                                 if total["evaluations"] else None),
        "max_sample_gap": audit["max_sample_gap"],
        "zero_sample_actions": audit["zero_sample_actions"],
        "decisions_with_zero_sample_actions":
            audit["decisions_with_zero_sample_actions"],
        "action_transition_distribution":
            audit["action_transition_distribution"],
        "deadline_overrun_count": audit["deadline_overrun_count"],
        "deadline_overrun_total_ms": audit["deadline_overrun_total_ms"],
        "deadline_overrun_max_ms": audit["deadline_overrun_max_ms"],
        "worker_thread_envs": worker_envs,
    }


def _stage_one_tasks(budgets):
    """(ラベル, 方式, 予算, horizon, scale, c) を実行順で列挙する。"""
    label, horizon, scale, c = DEFAULT_CONFIG
    for method in ("pimc", "ismcts"):
        for budget in budgets:
            yield (label, method, budget, horizon, scale, c)
    for label, horizon, scale, c in COORDINATE_CONFIGS:
        for method in ("pimc", "ismcts"):
            if label in _UCB_ONLY and method != "ismcts":
                continue
            yield (label, method, MID_BUDGET, horizon, scale, c)


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


def _journal(out_path, key, manifest, replay_audit):
    """--records 指定時だけ、label 単位の完了ペアジャーナルを返す。"""
    if not out_path:
        return None
    config = manifest
    return PairJournal(out_path, key, manifest["run_id"],
                       config["start_index"], config["pairs"], replay_audit)


def _report(tag, row):
    print("[%s] %-10s %-6s T=%-5d 勝率=%.3f CI=[%.3f,%.3f] 思考 平均%.0fms/p95%.0fms "
          "K/決定点=%.1f %.0fs"
          % (tag, row.get("label", ""), row["left"], row["budget"],
             row["point_estimate"], row["ci_low"], row["ci_high"],
             row["mean_think_ms"] or 0, row["p95_think_ms"] or 0,
             row["determinizations_per_decision"], row["wall_seconds"]),
          flush=True)


def _stage_one_plan(pairs, budgets, budget_seconds, environment):
    cards = load_cards()
    versions = _versions(cards)
    planned = []
    for label, method, budget, horizon, scale, c in _stage_one_tasks(budgets):
        key = "%s/%s/%s" % (label, method, budget)
        config = _config("train", TRAIN_INDEX, pairs, method, "heuristic",
                         budget, budget_seconds, horizon, scale, c)
        planned.append((key, label, method, budget, horizon, scale, c,
                        config, versions, environment))
    return planned


def _validate_row_audits(row, budget_seconds):
    """対局に参加した全探索方式へ監査規則を適用する。"""
    metrics = row.get("search_metrics")
    if not metrics:
        validate_audit(row.get("left"), budget_seconds, row)
        return
    for method, metric in metrics.items():
        validate_audit(method, budget_seconds, metric)


def run_stage_one(pairs, budgets, out_path, pool=None, budget_seconds=None,
                  resume=False, environment=None, replay_audit=False):
    rows = []
    planned = _stage_one_plan(pairs, budgets, budget_seconds, environment)
    expected = dict((key, build_manifest(key, label, config, versions, env))
                    for key, label, unused_m, unused_b, unused_h, unused_s,
                    unused_c, config, versions, env in planned)
    existing = validate_resume_state(out_path, [item[0] for item in planned],
                                     expected) if resume else {}
    for (key, label, method, budget, horizon, scale, c, config, versions,
         environment_item) in planned:
        if key in existing:
            print("[stage1] %s は検証済み bundle 済み。スキップする。" % key,
                  flush=True)
            rows.append(existing[key]["summary"])
            continue
        journal = _journal(out_path, key, expected[key], replay_audit)
        row, game_records = run_comparison("train", TRAIN_INDEX, pairs, method,
                                           "heuristic", budget, horizon, scale,
                                           c, pool, budget_seconds,
                                           replay_audit, journal, resume)
        _validate_row_audits(row, budget_seconds)
        _finalize_bundle(out_path, key, label, row, game_records, config,
                         versions, environment_item)
        if journal is not None:
            journal.discard()
        _report("stage1", row)
        rows.append(row)
    return rows


STAGE_TWO_MATCHUPS = (("pimc", "heuristic"), ("ismcts", "heuristic"),
                      ("pimc", "ismcts"))
STAGE_TWO_LABELS = tuple("%s-vs-%s" % (left, right)
                         for left, right in STAGE_TWO_MATCHUPS)
# 壁時計交差確認の transition safety cap。これ未満の遷移予算は
# 時間予算だけに任せると PIMC の最低1標本保証が破れるため拒否する。
WALLCLOCK_MIN_TRANSITION_CAP = 1000000


def run_stage_two(pairs, budget, horizon, scale, c, out_path, pool=None,
                  budget_seconds=None, group="dev", start_index=DEV_INDEX,
                  only=None, resume=False, environment=None,
                  replay_audit=False):
    rows = []
    planned = []
    for left, right in STAGE_TWO_MATCHUPS:
        label = "%s-vs-%s" % (left, right)
        if only and label not in only:
            continue
        config = _config(group, start_index, pairs, left, right, budget,
                         budget_seconds, horizon, scale, c)
        planned.append((label, left, right, config))
    expected = dict(
        (label, build_manifest(label, label, config,
                               _versions(load_cards()), environment))
        for label, unused_left, unused_right, config in planned)
    existing = validate_resume_state(out_path, [item[0] for item in planned],
                                     expected) if resume else {}
    for label, left, right, config in planned:
        if label in existing:
            print("[stage2] %s は検証済み bundle 済み。スキップする。" % label,
                  flush=True)
            rows.append(existing[label]["summary"])
            continue
        journal = _journal(out_path, label, expected[label], replay_audit)
        row, game_records = run_comparison(group, start_index, pairs, left,
                                           right, budget, horizon, scale, c,
                                           pool, budget_seconds, replay_audit,
                                           journal, resume)
        _validate_row_audits(row, budget_seconds)
        _finalize_bundle(out_path, label, label, row, game_records, config,
                         _versions(load_cards()), environment)
        if journal is not None:
            journal.discard()
        _report("stage2", row)
        rows.append(row)
    return rows


def _determinism_report(group, start_index, pairs, budget, budget_seconds,
                        horizon, scale, c, workers, labels, thread_env,
                        environment, replay_audit=False):
    """決定論検査の formal report を組み立てる。

    report 単独で実行 runtime を監査できるよう `python_version` と
    `python_executable` を環境情報から保存する。worker=1/N の manifest 比較
    (`find_deterministic_diffs`)ではこれらの field は除外されないため、
    両実行の runtime が食い違えば report 作成前に不一致として拒否される。
    """
    return {
        "schema_version": "phase2.5-determinism/v1",
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
        "labels": sorted(labels),
        "excluded_fields": sorted(search_records.DETERMINISM_EXCLUDED),
        "replay_audit": replay_audit,
        "thread_env": thread_env,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "python_version": environment["python_version"],
        "python_executable": environment["python_executable"],
        "source_digest": environment["source_digest"],
        "all_labels_matched": True,
    }


def run_determinism_check(pairs, budget, horizon, scale, c, budget_seconds,
                          start_index, matchups, records, workers,
                          group="dev", replay_audit=False):
    """worker=1 と worker=N を同じ設定で実行し、決定論的 field を比較する。

    成功時は指定 `records` へ原子 JSON report を保存する。既存ファイルは拒否。
    探索コアの set/dict 反復は PYTHONHASHSEED に依存するため、比較 worker を
    すべて同一 hash seed "0" で spawn する。hash seed の固定は CLI ブート
    ストラップが通常 run と共通に行い、ここでは契約値を pool 生成前に確認する。
    """
    _ensure_python_hash_seed_env()
    if pairs is None:
        raise SystemExit("--check-determinism には --pairs の明示が必要")
    labels = matchups or list(STAGE_TWO_LABELS)
    unknown = sorted(set(labels) - set(STAGE_TWO_LABELS))
    if unknown:
        raise SystemExit(
            "未知の matchup: %r (有効: %s)" % (unknown, list(STAGE_TWO_LABELS)))
    if os.path.exists(records):
        raise SystemExit("決定論検査の報告先が既に存在する: %s" % records)
    thread_env = dict((name, os.environ.get(name)) for name in
                      search_records.THREAD_ENV_NAMES)
    with tempfile.TemporaryDirectory(prefix="det-check-") as tmp:
        path_one = os.path.join(tmp, "worker1.jsonl")
        path_many = os.path.join(tmp, "worker%d.jsonl" % workers)
        multiprocessing.set_start_method("spawn", force=True)
        env_one = _environment(1, thread_env)
        env_many = _environment(workers, thread_env)
        pool_one = multiprocessing.Pool(processes=1)
        try:
            run_stage_two(pairs, budget, horizon, scale, c, path_one,
                          pool=pool_one, budget_seconds=budget_seconds,
                          group=group, start_index=start_index, only=labels,
                          environment=env_one, replay_audit=replay_audit)
        finally:
            pool_one.close()
            pool_one.join()
        pool_many = multiprocessing.Pool(processes=workers)
        try:
            run_stage_two(pairs, budget, horizon, scale, c, path_many,
                          pool=pool_many, budget_seconds=budget_seconds,
                          group=group, start_index=start_index, only=labels,
                          environment=env_many, replay_audit=replay_audit)
        finally:
            pool_many.close()
            pool_many.join()
        one = load_validated_bundles(path_one)
        many = load_validated_bundles(path_many)
        if replay_audit:
            for label in labels:
                for bundle in (one[label], many[label]):
                    for game in bundle["games"]:
                        if not isinstance(game.get("replay"), dict):
                            raise SystemExit(
                                "--replay-audit が有効なのに %s に replay 記録"
                                "がない" % label)
        for label in labels:
            diffs = find_deterministic_diffs(
                one[label]["summary"], many[label]["summary"],
                one[label]["games"], many[label]["games"])
            if diffs:
                raise SystemExit("決定論不一致 %s:\n%s" % (label, "\n".join(diffs)))
            print("[det] %s: worker=1 と worker=%d の決定論的 field は一致"
                  % (label, workers), flush=True)
    report = _determinism_report(
        group, start_index, pairs, budget, budget_seconds, horizon, scale, c,
        workers, labels, thread_env, env_one, replay_audit)
    write_report(records, report)
    return 0


def main(argv=None):
    _ensure_python_hash_seed_env()
    parser = argparse.ArgumentParser(description="Phase 2.5 search comparison")
    parser.add_argument("--stage", choices=("1", "2"), default="1")
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--budget", type=int, default=None,
                        help="1決定点あたりの状態遷移数(既定 1500)。"
                             "--budget-seconds 使用時は transition safety cap "
                             "として 1,000,000 以上を明示する)")
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--group", choices=("train", "dev"), default="dev",
                        help="stage2 で使う seed group(壁時計/並列証跡は train も可)")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument("--c", type=float, default=1.41)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--records", default=None)
    parser.add_argument("--matchups", nargs="+", default=None,
                        help="stage2 で実行する対戦ラベル(既定は全3対戦)")
    parser.add_argument("--resume", action="store_true",
                        help="既存 bundle の続きへ追記する。manifest 一致・完全性を"
                             "検証済みの label だけスキップする")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--regenerate", action="store_true",
                        help="bundle から summary / -games / .manifest を再生成する")
    parser.add_argument("--check-determinism", action="store_true",
                        help="worker=1 と worker=N で同じ対戦を実行し、"
                             "決定論的 field を比較する")
    parser.add_argument("--replay-audit", action="store_true",
                        help="--check-determinism と併用し、行動列・各 step 後"
                             "の state_hash・RNG state まで完全一致を検証する"
                             "(既定 off)")
    args = parser.parse_args(argv)

    threads = ",".join("%s=%s" % (name, os.environ[name])
                       for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
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
        print("prepared: stage=%s workers=%d thread_env=%s"
              % (args.stage, args.workers, threads))
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
        if args.budget_seconds is not None:
            raise SystemExit("--check-determinism では --budget-seconds を禁止")
        unknown = (set(args.matchups) if args.matchups else set()) - \
            set(STAGE_TWO_LABELS)
        if unknown:
            raise SystemExit(
                "未知の matchup: %r (有効: %s)"
                % (sorted(unknown), list(STAGE_TWO_LABELS)))
        if args.pairs <= 0:
            raise SystemExit("--pairs は正の整数が必要")
        if args.start_index < 0:
            raise SystemExit("--start-index は 0 以上が必要")
        budget = args.budget if args.budget is not None else MID_BUDGET
        return run_determinism_check(
            args.pairs, budget, args.horizon, args.scale,
            args.c, args.budget_seconds,
            args.start_index,
            set(args.matchups) if args.matchups else None,
            args.records, args.workers, args.group,
            replay_audit=args.replay_audit)

    if args.pairs is not None and args.pairs <= 0:
        raise SystemExit("--pairs は正の整数が必要")
    if args.budgets and any(value <= 0 for value in args.budgets):
        raise SystemExit("--budgets は正の整数のみ")
    if args.budget is not None and args.budget <= 0:
        raise SystemExit("--budget は正の整数が必要")
    if args.start_index is not None and args.start_index < 0:
        raise SystemExit("--start-index は 0 以上が必要")
    if args.budget_seconds is not None:
        if args.stage == "1":
            raise SystemExit("--budget-seconds は stage1 では禁止")
        if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
            raise SystemExit("--budget-seconds は有限の正の値が必要")
    if args.resume and not args.records:
        raise SystemExit("--resume には --records が必要")
    if args.stage == "2":
        if args.start_index is None:
            raise SystemExit(
                "stage2 では --start-index の明示が必要"
                "(消費済み DEV_INDEX=2000 への暗黙 fallback は禁止)")
        unknown = (set(args.matchups) if args.matchups else set()) - \
            set(STAGE_TWO_LABELS)
        if unknown:
            raise SystemExit(
                "未知の matchup: %r (有効: %s)"
                % (sorted(unknown), list(STAGE_TWO_LABELS)))

    budget = args.budget if args.budget is not None else MID_BUDGET
    if args.budget_seconds is not None:
        if args.budget is None:
            raise SystemExit(
                "--budget-seconds 使用時は --budget を transition safety cap "
                "として明示してください(既定 1500 では壁時計比較を許可しない)")
        if budget < WALLCLOCK_MIN_TRANSITION_CAP:
            raise SystemExit(
                "--budget-seconds 使用時は --budget %d 以上が必要"
                % WALLCLOCK_MIN_TRANSITION_CAP)

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
            raise SystemExit("bundle ディレクトリが既に存在する: %s" % bundle_dir)

    thread_env = dict((name, os.environ.get(name)) for name in
                      search_records.THREAD_ENV_NAMES)
    environment = _environment(args.workers, thread_env)
    pool = None
    if args.workers > 1:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(processes=args.workers)
    try:
        if args.stage == "1":
            run_stage_one(args.pairs or TRAIN_PAIRS, tuple(args.budgets),
                          args.records, pool, args.budget_seconds, args.resume,
                          environment)
        else:
            run_stage_two(args.pairs or DEV_PAIRS, budget, args.horizon,
                          args.scale, args.c, args.records, pool,
                          args.budget_seconds, args.group,
                          args.start_index,
                          set(args.matchups) if args.matchups else None,
                          args.resume, environment)
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
