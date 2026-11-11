"""Phase 5 D-7: 完全 replay による決定論検証(phase5-design.md §7 / handoff D-7)。

`scripts/compare_search.py --replay-audit` は worker=1/N の**記録の一致**しか
検証しない。ここでは「記録した行動列と seed から独立に再導出して中間 state と
RNG state が再現する」こと(handoff D-7 ②)を3点のテストで固定する:

1. 遷移関数の決定性(独立 replay): 決定論的エージェント(HeuristicAgent(0) 両席)で
   1局を最後まで進め、決定ごとの action(4要素 [kind.value, target_instance_id,
   instance_id, goods_count])・各 step 後の state_hash・RNG state
   (encode_rng(decode_rng(state.rng_state)).hex())を記録する。その後、
   env_seed / setup_seed(compare_search._play と同じ derive_seed)と記録した
   action 列**だけ**から game.new_game + game.step で独立に再構成し、各 step
   後の state_hash と RNG state が記録と一致することを検証する。
2. 探索エージェントの決定性(フル1局): IsmctsAgent(root_seed, budget_transitions,
   cards, evaluator=フェイク評価器)を同じ seed で2回独立に構築し、同じ配札・
   同じ席で1局を最後まで進めて、全決定の action 列と各 step 後の state_hash が
   完全一致することを検証する。フェイク評価器は Observation の legal_actions
   から一様 priors と固定 value(0.5)を返す stdlib 実装。探索前後で根 state が
   不変であること(clone 汚染の不在)も同じテストで確認する(設計書 §7)。
3. PYTHONHASHSEED 非依存: 1 の replay 検証を PYTHONHASHSEED=0 / 137 の
   subprocess で実行し、両方で同じ state_hash / RNG state 列になることを検証
   する(Phase 2.5 の N01 再発防止。serialize_observation は sort_keys=True で
   既に決定論的だが、実測で固定する)。

stdlib のみ。torch は import しない(net はフェイク評価器で代替)。
"""
import json
import os
import subprocess
import sys
import unittest

# 単体実行(discover -s tests)と `python -c 'import tests.test_replay'`
# (プロジェクト root から)の両方で helpers / naeco を解決できるよう、
# tests ディレクトリとプロジェクト root を sys.path へ載せる(既存テストの作法)。
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
for _entry in (_TESTS_DIR, _PROJECT_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import helpers as H  # noqa: E402
from naeco.ai.heuristic_agent import HeuristicAgent  # noqa: E402
from naeco.ai.search.ismcts import IsmctsAgent  # noqa: E402
from naeco.engine import game  # noqa: E402
from naeco.engine.observe import observe  # noqa: E402
from naeco.engine.state import (  # noqa: E402
    Action, ActionKind, Phase, decode_rng, derive_seed, encode_rng, eval_seed,
    state_hash,
)

# 較正帯 train 500000–500999(handoff-phase5.md §3)の先頭2 index をテスト固定の
# root / index に使う。テストは対局記録を残さないため seed 帯は消費しない。
_ROOT_REPLAY = eval_seed("train", 500000)
_INDEX_REPLAY = 500000
_ROOT_SEARCH = eval_seed("train", 500001)
_INDEX_SEARCH = 500001

# 探索エージェントの1決定点あたり遷移予算(小さい T で足りる: 設計書 §7)。
_SEARCH_BUDGET = 300

# テスト3で比較する PYTHONHASHSEED の2値(compare_search 契約値 0 と非0の代表)。
_HASH_SEEDS = ("0", "137")

# 1局の原子決定数に使う安全上限。フルゲームは数百に満たないため、これを超えたら
# ループ異常として打ち切る(テストが無限に回らないようにする)。
_MAX_DECISIONS = 2000


def _rng_hex(state):
    """RNG state の正準 hex 表現(compare_search._play と同じ)。"""
    return encode_rng(decode_rng(state.rng_state)).hex()


def _play_and_record(root, index):
    """決定論的エージェント(HeuristicAgent(0) 両席)で1局を最後まで進めて記録する。

    compare_search._play と同じ seed 導出と記録形式(4要素 action 列・各 step 後
    の state_hash・RNG state hex)を使う。返り値の dict だけで独立 replay が
    再構成できるように、env_seed / setup_seed も記録に含める。
    """
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game.new_game(2, env_seed, H.CARDS, setup_seed=setup_seed)
    agents = [HeuristicAgent(0), HeuristicAgent(0)]
    record = {
        "env_seed": env_seed,
        "setup_seed": setup_seed,
        "initial_state_hash": state_hash(state),
        "actions": [],
        "state_hashes": [],
        "rng_states": [],
    }
    while state.phase != Phase.GAME_OVER:
        if len(record["actions"]) >= _MAX_DECISIONS:
            raise AssertionError("game did not terminate within %d decisions"
                                 % _MAX_DECISIONS)
        actor = game.actor(state)
        action = agents[actor].act(observe(state, actor, H.CARDS))
        record["actions"].append([
            action.kind.value, action.target_instance_id,
            action.instance_id, action.goods_count,
        ])
        state = game.step(state, action, H.CARDS).state
        record["state_hashes"].append(state_hash(state))
        record["rng_states"].append(_rng_hex(state))
    return record


def _replay_record(record):
    """(env_seed / setup_seed, 行動列) だけから独立に再構成する。

    エージェントは使わない。4要素 action 列から Action を再構成し、
    game.new_game + game.step だけで各 step 後の中間 state を再現する。
    """
    state = game.new_game(2, record["env_seed"], H.CARDS,
                          setup_seed=record["setup_seed"])
    replay = {
        "initial_state_hash": state_hash(state),
        "state_hashes": [],
        "rng_states": [],
    }
    for kind, target_instance_id, instance_id, goods_count in record["actions"]:
        action = Action(kind=ActionKind(kind),
                        target_instance_id=target_instance_id,
                        instance_id=instance_id,
                        goods_count=goods_count)
        state = game.step(state, action, H.CARDS).state
        replay["state_hashes"].append(state_hash(state))
        replay["rng_states"].append(_rng_hex(state))
    return replay


def _assert_replay_matches(record, replay):
    """記録と独立再構成が一致しない最初の箇所を特定して AssertionError にする。"""
    if replay["initial_state_hash"] != record["initial_state_hash"]:
        raise AssertionError(
            "initial_state_hash mismatch: recorded %s != replayed %s"
            % (record["initial_state_hash"], replay["initial_state_hash"]))
    n = len(record["state_hashes"])
    if len(replay["state_hashes"]) != n or len(replay["rng_states"]) != n:
        raise AssertionError(
            "replay length mismatch: recorded %d steps, replayed %d/%d"
            % (n, len(replay["state_hashes"]), len(replay["rng_states"])))
    for i in range(n):
        if replay["state_hashes"][i] != record["state_hashes"][i]:
            raise AssertionError(
                "state_hash mismatch at step %d/%d" % (i + 1, n))
        if replay["rng_states"][i] != record["rng_states"][i]:
            raise AssertionError(
                "rng_state mismatch at step %d/%d" % (i + 1, n))


class TransitionDeterminismTests(unittest.TestCase):
    """設計書 §7(1): 記録した行動列 + seed からの独立 replay が一致する。"""

    def test_independent_replay_reproduces_every_intermediate_state(self):
        record = _play_and_record(_ROOT_REPLAY, _INDEX_REPLAY)
        self.assertGreaterEqual(len(record["actions"]), 20)   # フルゲームの形
        replay = _replay_record(record)
        # 各 step 後の state_hash と RNG state が記録と一致する(不一致は
        # _assert_replay_matches が最初の差分位置を報告する)。
        _assert_replay_matches(record, replay)


class _FakeEvaluator(object):
    """D-7 用フェイク評価器: legal_actions 上の一様 priors と固定 value(0.5)。

    Observation だけを受け取り、state へは触れない(D-6 の情報境界に沿う)。
    stdlib のみ。torch は使わない。
    """

    def __init__(self, value=0.5):
        self.value = float(value)

    def evaluate(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        prob = 1.0 / len(legal)
        return [(action, prob) for action in legal], self.value


def _play_full_game_with_search(make_search_agent):
    """席0=IsmctsAgent(フェイク評価器)、席1=HeuristicAgent(0) で1局を進める。

    全決定の action 列・各 step 後の state_hash 列・探索統計(決定数, 評価器
    呼出数)を返す。各決定の前後で根 state の state_hash が不変であることを
    確認する(設計書 §7「探索が clone を汚染しない」)。
    """
    env_seed = derive_seed(_ROOT_SEARCH, "env", _INDEX_SEARCH)
    setup_seed = derive_seed(_ROOT_SEARCH, "setup", _INDEX_SEARCH)
    state = game.new_game(2, env_seed, H.CARDS, setup_seed=setup_seed)
    search_agent = make_search_agent()
    agents = [search_agent, HeuristicAgent(0)]
    actions = []
    state_hashes = []
    while state.phase != Phase.GAME_OVER:
        if len(actions) >= _MAX_DECISIONS:
            raise AssertionError("game did not terminate within %d decisions"
                                 % _MAX_DECISIONS)
        who = game.actor(state)
        before = state_hash(state)
        action = agents[who].act(observe(state, who, H.CARDS))
        if state_hash(state) != before:
            raise AssertionError(
                "search mutated the root state at decision %d (actor %d)"
                % (len(actions), who))
        actions.append(action)
        state = game.step(state, action, H.CARDS).state
        state_hashes.append(state_hash(state))
    return (actions, state_hashes,
            search_agent.stats.decisions, search_agent.stats.evaluations)


class SearchAgentDeterminismTests(unittest.TestCase):
    """設計書 §7(2): 同じ seed の2つの IsmctsAgent がフル1局で完全一致する。"""

    def test_two_independent_agents_play_identical_full_games(self):
        def make():
            return IsmctsAgent(_ROOT_SEARCH, _SEARCH_BUDGET, cards=H.CARDS,
                               evaluator=_FakeEvaluator(0.5))

        first = _play_full_game_with_search(make)
        second = _play_full_game_with_search(make)
        first_actions, first_hashes, first_decisions, first_evaluations = first
        second_actions, second_hashes, second_decisions, second_evaluations = second
        self.assertGreater(first_decisions, 0)      # 探索が実際に走っている
        self.assertGreater(first_evaluations, 0)    # フェイク評価器の経路を使った
        self.assertGreaterEqual(len(first_actions), 20)   # フルゲームの形
        self.assertEqual(first_decisions, second_decisions)
        self.assertEqual(first_evaluations, second_evaluations)
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first_hashes, second_hashes)


def _replay_trace():
    """記録 + 独立 replay の全 state_hash / RNG state 列を JSON 化する。

    PYTHONHASHSEED 比較の子プロセスが呼ぶ。記録と replay が一致しない場合は
    AssertionError を送出し、親は非零終了として検知する。
    """
    record = _play_and_record(_ROOT_REPLAY, _INDEX_REPLAY)
    replay = _replay_record(record)
    _assert_replay_matches(record, replay)
    return {
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "num_decisions": len(record["actions"]),
        "initial_state_hash": record["initial_state_hash"],
        "state_hashes": record["state_hashes"],
        "rng_states": record["rng_states"],
        "replayed_state_hashes": replay["state_hashes"],
        "replayed_rng_states": replay["rng_states"],
    }


def _run_replay_trace(hash_seed):
    """PYTHONHASHSEED=hash_seed の子プロセスで replay trace を取り JSON で返す。"""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    script = os.path.abspath(__file__)
    try:
        proc = subprocess.run(
            [sys.executable, script, "--dump-replay-trace"],
            env=env, cwd=_PROJECT_ROOT, capture_output=True, text=True,
            check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AssertionError(
            "replay trace subprocess failed under PYTHONHASHSEED=%s:\n"
            "stdout:\n%s\nstderr:\n%s"
            % (hash_seed, getattr(exc, "stdout", None),
               getattr(exc, "stderr", None)))
    return json.loads(proc.stdout)


class PythonHashSeedIndependenceTests(unittest.TestCase):
    """設計書 §7(3): replay は PYTHONHASHSEED に依存しない(N01 再発防止)。"""

    def test_replay_is_identical_across_hash_seeds(self):
        traces = [_run_replay_trace(seed) for seed in _HASH_SEEDS]
        for trace, seed in zip(traces, _HASH_SEEDS):
            # 子プロセスへ注入が効いていることを先に確認する。
            self.assertEqual(trace["hash_seed"], seed)
        first, second = traces
        self.assertEqual(first["num_decisions"], second["num_decisions"])
        for key in ("initial_state_hash", "state_hashes", "rng_states",
                    "replayed_state_hashes", "replayed_rng_states"):
            self.assertEqual(first[key], second[key],
                             "trace key %r differs across PYTHONHASHSEED" % key)


if __name__ == "__main__":
    if "--dump-replay-trace" in sys.argv:
        print(json.dumps(_replay_trace(), sort_keys=True))
        raise SystemExit(0)
    unittest.main()
