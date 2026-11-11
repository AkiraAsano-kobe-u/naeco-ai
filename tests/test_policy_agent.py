"""B3 推論用 PolicyAgent のテスト(Phase 3 設計書 §4.6)。

「歯」の狙い: 情報境界テストは game import / GameState 参照を、同点テストは
sort_key を外すと落ちる形で固定する。sample の決定論は自前 RNG を使わず
グローバル乱数へ戻すと落ちる。実対局テストは返り値が合法手の要素であること
を全決定点で検査する。フェイク net は PolicyValueNet の形状契約
(B, A_max, action_dim / B, A_max)を強制し、agent のレイアウトが net と
食い違うと単体テストで ValueError になる。実 net 統合テストは torch がある
環境だけ実行し、play_episode 完走と sample の seed 再現を固定する。
"""
import dataclasses
import ast
import inspect
import sys
import unittest

from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Action, ActionKind, Phase
from naeco.train import agent as agent_mod
from naeco.train.agent import PolicyAgent

import helpers as H

try:
    import torch
except ImportError:
    torch = None

HAS_TORCH = torch is not None

if HAS_TORCH:
    from naeco.train import encoding as enc
    from naeco.train import env as train_env
    from naeco.train import net as net_mod
else:
    enc = None
    net_mod = None
    train_env = None


def _module_import_names(source):
    """モジュールソースの import 文から対象名だけを取り出す。"""
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            names.extend(alias.name for alias in node.names)
    return names


class FakePolicyNet(object):
    """PolicyValueNet(設計書 §4.7)と同じ形状契約を強制する決定論フェイク。

    ``forward(obs_batch, action_batch, action_mask) -> (logits, values)`` で、
    ``action_batch`` は (B, A_max, action_dim) の3次元、``action_mask`` は
    (B, A_max) の2次元を要求し、形状が違えば ``ValueError`` を投げる
    (agent 側がレイアウトを誤っても単体テストで検出される)。スコアは
    位置インデックスで決まるので、テストが期待する選択を直接制御できる。
    """

    def __init__(self, scores=(), default=0.0):
        self._scores = tuple(float(x) for x in scores)
        self._default = float(default)

    def forward(self, obs_batch, action_batch, action_mask):
        batch_size = self._require_shape(action_batch, 3, "action_batch")
        self._require_shape(action_mask, 2, "action_mask")
        if len(obs_batch) != batch_size:
            raise ValueError(
                "obs_batch and action_batch must share batch size, "
                "got %d and %d" % (len(obs_batch), batch_size))
        for i, row in enumerate(action_batch):
            if len(action_mask[i]) != len(row):
                raise ValueError(
                    "action_batch and action_mask must share A_max, "
                    "got %d and %d" % (len(row), len(action_mask[i])))
            if not any(action_mask[i]):
                raise ValueError(
                    "every batch row must have at least one legal action "
                    "(all-False action_mask rows are not supported)")
        logits = [
            [self._scores[j] if j < len(self._scores) else self._default
             for j in range(len(row))]
            for row in action_batch
        ]
        return logits, [0.0] * len(logits)

    @staticmethod
    def _require_shape(batch, dims, name):
        """batch が dims 次元の矩形 list であることを要求し、行数を返す。

        PolicyValueNet.forward の形状検査(action_batch 3次元 /
        action_mask 2次元)を Python list で再現する。違えば ValueError。
        """
        if not isinstance(batch, (list, tuple)) or len(batch) == 0:
            raise ValueError(
                "%s must be a non-empty %d-D batch" % (name, dims))
        widths = None
        for row in batch:
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                raise ValueError(
                    "%s must be a rectangular %d-D batch" % (name, dims))
            if widths is None:
                widths = len(row)
            elif len(row) != widths:
                raise ValueError(
                    "%s must be a rectangular %d-D batch" % (name, dims))
            if dims == 3:
                inner = None
                for vec in row:
                    if (not isinstance(vec, (list, tuple))
                            or len(vec) == 0):
                        raise ValueError(
                            "%s must be a rectangular %d-D batch"
                            % (name, dims))
                    if inner is None:
                        inner = len(vec)
                    elif len(vec) != inner:
                        raise ValueError(
                            "%s must be a rectangular %d-D batch"
                            % (name, dims))
        return len(batch)


def observation_with(actions):
    """legal_actions だけ差し替えた観測を作る。"""
    base = observe(H.new(2, 17), 0, H.CARDS)
    return dataclasses.replace(base, legal_actions=tuple(actions))


def select_build(instance_id):
    return Action(ActionKind.SELECT_BUILD, instance_id=instance_id)


class FakeNetShapeContractTests(unittest.TestCase):
    """フェイク net が PolicyValueNet の形状契約を強制すること。

    「歯」の狙い: 旧レイアウト(action_batch=(L, action_dim), mask=(L,))を
    agent が渡すとここで ValueError になり、形状不整合が単体テストで
    検出される。
    """

    def test_accepts_batched_layout(self):
        net = FakePolicyNet([0.0, 1.0])
        logits, values = net.forward(
            [[0.0] * 8],
            [[[0.0] * 4, [0.0] * 4]],
            [[True, True]])
        self.assertEqual(logits, [[0.0, 1.0]])
        self.assertEqual(values, [0.0])

    def test_rejects_legacy_per_action_row_layout(self):
        # (L, action_dim) の action_batch と (L,) の mask は ValueError。
        net = FakePolicyNet()
        with self.assertRaises(ValueError):
            net.forward(
                [[0.0] * 8, [0.0] * 8],
                [[0.0] * 4, [0.0] * 4],
                [True, True])

    def test_rejects_flat_action_batch(self):
        net = FakePolicyNet()
        with self.assertRaises(ValueError):
            net.forward([[0.0] * 8], [[0.0] * 4, [0.0] * 4], [True, True])

    def test_rejects_batch_size_mismatch(self):
        net = FakePolicyNet()
        with self.assertRaises(ValueError):
            net.forward(
                [[0.0] * 8, [0.0] * 8],
                [[[0.0] * 4]],
                [[True]])

    def test_rejects_mask_row_length_mismatch(self):
        net = FakePolicyNet()
        with self.assertRaises(ValueError):
            net.forward(
                [[0.0] * 8],
                [[[0.0] * 4, [0.0] * 4]],
                [[True]])

    def test_rejects_all_false_mask_row(self):
        net = FakePolicyNet()
        with self.assertRaises(ValueError):
            net.forward(
                [[0.0] * 8],
                [[[0.0] * 4]],
                [[False]])


class InformationBoundaryTests(unittest.TestCase):
    """PolicyAgent が Observation 以外に触れないことを固定する。"""

    def test_signatures_match_section_4_6(self):
        init_parameters = list(inspect.signature(PolicyAgent.__init__).parameters)
        self.assertEqual(init_parameters, ["self", "net", "mode", "seed"])
        act_parameters = list(inspect.signature(PolicyAgent.act).parameters)
        self.assertEqual(act_parameters, ["self", "observation"])

    def test_source_does_not_mention_gamestate(self):
        source = inspect.getsource(PolicyAgent)
        self.assertNotIn("GameState", source)

    def test_module_does_not_import_engine_game(self):
        source = inspect.getsource(agent_mod)
        imports = _module_import_names(source)
        self.assertNotIn("naeco.engine.game", imports)
        self.assertNotIn("game", imports)

    def test_module_header_does_not_import_torch_or_net(self):
        # ヘッダーだけでなくソース全体をスキャンする: torch 境界(設計書 §2)
        # により agent.py に ``import torch`` が一切現れてはならない。
        source = inspect.getsource(agent_mod)
        self.assertNotIn("import torch", source)
        self.assertNotIn("naeco.train import net", source)
        self.assertNotIn("naeco.train.net", source)

    def test_source_has_no_try_import_torch(self):
        # torch を触る入口自体(旧 _try_import_torch)が無いことを全行固定。
        source = inspect.getsource(agent_mod)
        self.assertNotIn("_try_import_torch", source)


class GreedySelectionTests(unittest.TestCase):
    """mode="greedy" の最高スコア選択と sort_key 同点解消。"""

    def test_picks_the_highest_score_action(self):
        mine = select_build(H.wp("base:mine"))
        cafe = select_build(H.wp("base:cafe"))
        farm = select_build(H.wp("base:farm"))
        obs = observation_with((mine, cafe, farm))
        agent = PolicyAgent(FakePolicyNet([0.0, 3.0, 1.0]))
        self.assertEqual(agent.act(obs), cafe)

    def test_ties_resolve_by_sort_key_ascending(self):
        # 全 logits が同点でも、sort_key の最小(base:cafe)が選ばれる。
        mine = select_build(H.wp("base:mine"))
        cafe = select_build(H.wp("base:cafe"))
        farm = select_build(H.wp("base:farm"))
        obs = observation_with((mine, cafe, farm))
        agent = PolicyAgent(FakePolicyNet([2.0, 2.0, 2.0]))
        self.assertEqual(agent.act(obs), cafe)

    def test_ties_ignore_tuple_insertion_order(self):
        # sort_key 解消が無いと tuple の先頭(挿入順)が勝つ実装はここで落ちる。
        cafe = select_build(H.wp("base:cafe"))
        mine = select_build(H.wp("base:mine"))
        obs = observation_with((mine, cafe))
        agent = PolicyAgent(FakePolicyNet([1.0, 1.0]))
        self.assertEqual(agent.act(obs), cafe)

    def test_ties_across_kinds_use_kind_value(self):
        place = Action(ActionKind.PLACE_WORKER,
                       target_instance_id=H.wp("base:mine"))
        build = select_build(H.wp("base:cafe"))
        obs = observation_with((build, place))
        agent = PolicyAgent(FakePolicyNet([0.0, 0.0]))
        # "place_worker" < "select_build" なので place が勝つ。
        self.assertEqual(agent.act(obs), place)

    def test_greedy_is_deterministic_across_instances(self):
        mine = select_build(H.wp("base:mine"))
        cafe = select_build(H.wp("base:cafe"))
        obs = observation_with((mine, cafe))
        self.assertEqual(
            PolicyAgent(FakePolicyNet([0.0, 1.0]), seed=5).act(obs),
            PolicyAgent(FakePolicyNet([0.0, 1.0]), seed=5).act(obs))


class SampleDeterminismTests(unittest.TestCase):
    """mode="sample" が同じ seed で完全に同じ行動列を返すこと。"""

    def _play(self, player_seed_base):
        net = FakePolicyNet()   # 全合法手同点 → softmax は一様
        actions = []
        for game_seed in (41, 42, 43):
            state = H.new(2, game_seed)
            agents = [
                PolicyAgent(net, mode="sample", seed=player_seed_base + p)
                for p in range(2)
            ]
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                observation = observe(state, who, H.CARDS)
                action = agents[who].act(observation)
                actions.append(action)
                state = game.step(state, action, H.CARDS).state
        return actions

    def test_same_seed_replays_identical_sequence(self):
        first = self._play(1000)
        second = self._play(1000)
        self.assertGreaterEqual(len(first), 30)
        self.assertEqual(first, second)

    def test_sample_actually_samples_instead_of_greedy_pick(self):
        mine = select_build(H.wp("base:mine"))
        cafe = select_build(H.wp("base:cafe"))
        obs = observation_with((mine, cafe))
        # 一様2択の最初の draw で seed=0 は index 1(cafe)を引く。
        agent = PolicyAgent(FakePolicyNet([0.0, 0.0]), mode="sample", seed=0)
        self.assertEqual(agent.act(obs), cafe)


class ErrorHandlingTests(unittest.TestCase):
    """legal_actions の空 / None は黙って None を返さず ValueError。"""

    def test_empty_legal_actions_raise_value_error(self):
        obs = observation_with(())
        with self.assertRaises(ValueError):
            PolicyAgent(FakePolicyNet()).act(obs)

    def test_none_legal_actions_raise_value_error(self):
        obs = dataclasses.replace(
            observe(H.new(2, 5), 0, H.CARDS), legal_actions=None)
        with self.assertRaises(ValueError):
            PolicyAgent(FakePolicyNet()).act(obs)

    def test_unknown_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            PolicyAgent(FakePolicyNet(), mode="argmax")


class LegalActionMembershipTests(unittest.TestCase):
    """実対局の全決定点で返り値が legal_actions の要素であること。"""

    def _assert_all_decisions_legal(self, mode):
        configs = ((2, range(6)), (3, range(3)), (4, range(2)))
        checked = 0
        for n, seeds in configs:
            for seed in seeds:
                state = H.new(n, 2000 + seed)
                agents = [
                    PolicyAgent(FakePolicyNet(), mode=mode, seed=1000 + p)
                    for p in range(n)
                ]
                steps = 0
                while state.phase != Phase.GAME_OVER:
                    who = game.actor(state)
                    observation = observe(state, who, H.CARDS)
                    action = agents[who].act(observation)
                    self.assertIn(action, observation.legal_actions)
                    state = game.step(state, action, H.CARDS).state
                    checked += 1
                    steps += 1
                    self.assertLess(steps, 6000)
        self.assertGreater(checked, 100)

    def test_greedy_returns_legal_members(self):
        self._assert_all_decisions_legal("greedy")

    def test_sample_returns_legal_members(self):
        self._assert_all_decisions_legal("sample")


class TorchBoundaryTests(unittest.TestCase):
    """フェイク net だけの経路が torch 不在でも動くこと。"""

    def test_fake_net_drives_act_without_torch(self):
        mine = select_build(H.wp("base:mine"))
        cafe = select_build(H.wp("base:cafe"))
        obs = observation_with((mine, cafe))
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None
        try:
            agent = PolicyAgent(FakePolicyNet([0.0, 1.0]))
            self.assertEqual(agent.act(obs), cafe)
            sample = PolicyAgent(FakePolicyNet(), mode="sample", seed=0)
            self.assertEqual(sample.act(obs), cafe)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved


class _RecordingAgent(object):
    """返した行動を全部記録するラッパー。行動列の再現を直接検査するため。"""

    def __init__(self, agent, recorder):
        self._agent = agent
        self._recorder = recorder

    def act(self, observation):
        action = self._agent.act(observation)
        self._recorder.append(action)
        return action


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class RealPolicyValueNetIntegrationTests(unittest.TestCase):
    """本物の PolicyValueNet で play_episode を1局完走できること。

    「歯」の狙い: agent のレイアウトが net の契約(B, A_max, action_dim /
    B, A_max)と食い違うと、ここで ValueError になって落ちる。sample の
    同一 seed 再現は、自前 RNG を使わずグローバル乱数へ戻すと落ちる。
    """

    @staticmethod
    def _make_net():
        torch.manual_seed(0)
        return net_mod.PolicyValueNet(enc.obs_dim(2), enc.action_dim())

    def test_greedy_completes_two_player_episode(self):
        net = self._make_net()
        trajectory = train_env.play_episode(
            "train", 4100, 0,
            [PolicyAgent(net, mode="greedy"),
             PolicyAgent(net, mode="greedy")],
            cards=H.CARDS)
        self.assertFalse(trajectory.discarded)
        self.assertGreater(len(trajectory.decisions), 0)
        self.assertEqual(len(trajectory.terminal_rewards), 2)
        self.assertTrue(all(u in (1.0, -1.0)
                            for u in trajectory.terminal_rewards))

    def test_sample_same_seed_replays_identical_action_sequence(self):
        def run():
            net = self._make_net()
            actions = []
            agents = [
                _RecordingAgent(
                    PolicyAgent(net, mode="sample", seed=7000 + p), actions)
                for p in range(2)
            ]
            trajectory = train_env.play_episode(
                "train", 4101, 0, agents, cards=H.CARDS)
            return trajectory, actions

        first_trajectory, first_actions = run()
        second_trajectory, second_actions = run()
        self.assertFalse(first_trajectory.discarded)
        self.assertGreaterEqual(len(first_actions), 30)
        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first_trajectory, second_trajectory)


if __name__ == "__main__":
    unittest.main()
