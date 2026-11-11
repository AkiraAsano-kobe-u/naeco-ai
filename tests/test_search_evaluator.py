"""M2: 評価器プロトコルと SO-ISMCTS 組込のテスト(D-1 / D-2 / D-5 / D-6)。

「歯」の狙い:
- D-2 写像: 効用空間 u の [0,1] 勝率への写像 p=clip((u+1)/2,0,1) はアダプタ
  (`naeco/train/search_evaluator.py`)の1箇所だけにあり、u=+1 → 1.0 /
  u=-1 → 0.0 / 範囲外は clip される。
- ゼロ和ベクトル: 探索側が絶対席順の [p, 1-p] を組み立てる(actor の席で
  順序が反転する)。
- D-1 委譲: pending / 一択では policy_agent.act が呼ばれ、探索 stats は
  増えない。
- D-6 情報境界: 根の観測と整合する範囲で真の隠れ状態(相手手札・山札)を
  入れ替えても、評価器使用時の着手は不変。評価器が受け取るのは
  observe(determinized_state, actor) の観測だけ(legal_actions が入るのは
  actor 観測だけ)。
- 終端効用: terminal_utilities の ±1 を (u+1)/2 で [0,1] へ写像する。
- 後方互換: evaluator=None / policy_agent=None の既定では従来と同一挙動
  (ヒューリスティック委譲 + static_value)。
- S2: NetEvaluator の priors は policy ヘッド logits の数値安定 softmax
  (合計 1.0、argmax が logits 最大)。探索は PUCT(c_puct × prior、FPU=0.5、
  sqrt(availability) 補正)で未展開も含めて全合法手を選択し、ルート prior は
  探索前に事前計算される(最初の selection から全合法手に prior がある)。
  evaluator=None では従来 UCB + unexpanded[0] のまま(後方互換)。
"""
import dataclasses
import inspect
import math
import sys
import unittest

from naeco.engine import game
from naeco.engine.observe import observe, serialize_observation
from naeco.engine.state import Phase
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.search.common import SearchStats
from naeco.ai.search.evaluator import Evaluator, UniformEvaluator
from naeco.ai.search.ismcts import IsmctsAgent, _Node, evaluate_with_evaluator
from naeco.train import search_evaluator as se_mod
from naeco.train.search_evaluator import (
    NetEvaluator, utility_to_win_probability,
)

import helpers as H


def _root(seed=44):
    """配置手(pending なし)の観測を返す(test_search と同じ構図)。"""
    state = H.new(2, seed=seed)
    return state, observe(state, game.actor(state), H.CARDS)


class _FakeEvaluator(Evaluator):
    """記録付きフェイク評価器。value は固定値。priors は一様。"""

    def __init__(self, value=0.5):
        self.value = float(value)
        self.observations = []

    def evaluate(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        self.observations.append(observation)
        prob = 1.0 / len(legal)
        return [(action, prob) for action in legal], self.value


class _RecordingPolicyAgent(object):
    """呼出を記録し、常に legal_actions[0] を返すフェイク委譲先。"""

    def __init__(self):
        self.calls = []

    def act(self, observation):
        self.calls.append(observation)
        return observation.legal_actions[0]


class _FakeValueNet(object):
    """値だけ返すフェイク。PolicyValueNet と同じ (logits, values) 契約。

    logits を省略すると全 0(softmax 後は一様)になる。logits を渡すと
    そのまま先頭行として返す(長さは legal_actions 数と一致させること)。
    `infer` を持たないので、アダプタは forward を list で呼ぶ経路を使う。
    """

    def __init__(self, u, logits=None):
        self._u = float(u)
        self._logits = logits

    def forward(self, obs_batch, action_batch, action_mask):
        if self._logits is None:
            logits = [0.0] * len(action_batch[0])
        else:
            logits = list(self._logits)
        return [logits], [self._u]


class _FakeInferNet(object):
    """infer と forward の両方を持つフェイク。

    PolicyValueNet と同じく `infer` を持ち、呼出引数と forward の使用有無を
    記録する。アダプタが infer 経路を使うことの確認用。
    """

    def __init__(self, u=0.0):
        self._u = float(u)
        self.infer_calls = []
        self.forward_calls = 0

    def infer(self, obs_list, action_list, mask_list):
        self.infer_calls.append((obs_list, action_list, mask_list))
        logits = [0.0] * len(action_list[0])
        return [logits], [self._u]

    def forward(self, obs_batch, action_batch, action_mask):
        self.forward_calls += 1
        return [obs_batch], [self._u]


class UtilityMappingTests(unittest.TestCase):
    """D-2: 効用空間 u → [0,1] 勝率の写像はアダプタの1箇所だけ。"""

    def test_win_utility_maps_to_one(self):
        self.assertEqual(utility_to_win_probability(1.0), 1.0)
        self.assertEqual(utility_to_win_probability(0.99), 0.995)

    def test_loss_utility_maps_to_zero(self):
        self.assertEqual(utility_to_win_probability(-1.0), 0.0)
        self.assertEqual(utility_to_win_probability(-0.5), 0.25)

    def test_mid_utility_maps_to_half(self):
        self.assertEqual(utility_to_win_probability(0.0), 0.5)

    def test_out_of_range_utilities_are_clipped(self):
        self.assertEqual(utility_to_win_probability(1.5), 1.0)
        self.assertEqual(utility_to_win_probability(7.0), 1.0)
        self.assertEqual(utility_to_win_probability(-2.0), 0.0)
        self.assertEqual(utility_to_win_probability(-100.0), 0.0)


class NetEvaluatorTests(unittest.TestCase):
    """torch アダプタの契約(D-5)。フェイク net + torch 不在で検証する。"""

    def test_maps_utility_values(self):
        for u, expected in ((1.0, 1.0), (-1.0, 0.0), (0.0, 0.5),
                            (2.5, 1.0), (-3.0, 0.0), (0.2, 0.6)):
            unused_state, obs = _root()
            priors, value = NetEvaluator(_FakeValueNet(u)).evaluate(obs)
            self.assertAlmostEqual(value, expected)

    def test_priors_are_uniform_over_legal_actions(self):
        unused_state, obs = _root()
        priors, unused_value = NetEvaluator(_FakeValueNet(0.0)).evaluate(obs)
        self.assertEqual(len(priors), len(obs.legal_actions))
        self.assertAlmostEqual(sum(p for unused_a, p in priors), 1.0)
        self.assertEqual(
            sorted((a for a, unused_p in priors), key=lambda a: a.sort_key()),
            sorted(obs.legal_actions, key=lambda a: a.sort_key()))
        self.assertAlmostEqual(
            {a: p for a, p in priors}[obs.legal_actions[0]],
            1.0 / len(obs.legal_actions))

    def test_works_without_torch(self):
        unused_state, obs = _root()
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None
        try:
            priors, value = NetEvaluator(_FakeValueNet(-1.0)).evaluate(obs)
            self.assertEqual(value, 0.0)
            self.assertEqual(len(priors), len(obs.legal_actions))
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_logits_row_contract_mismatches_raise(self):
        """S2: logits の行数 / 長さ / 有限性の契約違反は ValueError。"""
        unused_state, obs = _root()
        net = NetEvaluator(_FakeValueNet(0.0))
        with self.assertRaises(ValueError):
            net._logits_row([[0.0] * len(obs.legal_actions),
                             [0.0] * len(obs.legal_actions)],
                            len(obs.legal_actions))
        with self.assertRaises(ValueError):
            net._logits_row([[0.0] * (len(obs.legal_actions) + 1)],
                            len(obs.legal_actions))
        with self.assertRaises(ValueError):
            net._logits_row([[float("nan")] * len(obs.legal_actions)],
                            len(obs.legal_actions))
        with self.assertRaises(ValueError):
            net._logits_row([[float("inf")] * len(obs.legal_actions)],
                            len(obs.legal_actions))

    def test_no_torch_import_anywhere(self):
        """torch 境界: ソース全体に `import torch` が一切現れない。

        トップレベル判定ではなく全行スキャン(関数内遅延 import も禁止)。
        `_try_import_torch` のような torch を触る入口自体が無いことも
        assert する。tensor 変換と no_grad は net.infer が担う。
        """
        source = inspect.getsource(se_mod)
        self.assertNotIn("import torch", source)
        self.assertNotIn("_try_import_torch", source)

    def test_evaluate_prefers_infer_over_forward(self):
        """infer を持つ net は infer 経路で呼ばれ、forward は使われない。"""
        unused_state, obs = _root()
        net = _FakeInferNet(0.25)
        priors, value = NetEvaluator(net).evaluate(obs)
        self.assertEqual(value, 0.625)
        self.assertEqual(net.forward_calls, 0)
        self.assertEqual(len(net.infer_calls), 1)
        obs_batch, action_batch, mask = net.infer_calls[0]
        # list のまま渡される(batch=1、mask は全 True)。
        self.assertEqual(len(obs_batch), 1)
        self.assertEqual(len(action_batch), 1)
        self.assertEqual(len(action_batch[0]), len(obs.legal_actions))
        self.assertEqual(mask, [[True] * len(obs.legal_actions)])
        self.assertEqual(len(priors), len(obs.legal_actions))

    def test_uniform_evaluator_priors_sum_to_one(self):
        unused_state, obs = _root()
        priors, value = UniformEvaluator(0.5).evaluate(obs)
        self.assertEqual(value, 0.5)
        self.assertEqual(len(priors), len(obs.legal_actions))
        self.assertAlmostEqual(sum(p for unused_a, p in priors), 1.0)
        self.assertEqual([a for a, unused_p in priors],
                         list(obs.legal_actions))


class NetEvaluatorPriorTests(unittest.TestCase):
    """S2: priors は policy ヘッド logits の数値安定 softmax。"""

    def test_softmax_priors_follow_logits(self):
        unused_state, obs = _root()
        logits = [0.1 * (i + 1) for i in range(len(obs.legal_actions))]
        priors, unused_value = NetEvaluator(_FakeValueNet(0.0, logits)).evaluate(obs)
        self.assertEqual(len(priors), len(obs.legal_actions))
        # legal_actions と 1:1 の順序で返る。
        self.assertEqual([a for a, unused_p in priors],
                         list(obs.legal_actions))
        probs = [p for unused_a, p in priors]
        self.assertAlmostEqual(sum(probs), 1.0)
        # argmax は logits 最大と一致。
        self.assertEqual(max(range(len(probs)), key=lambda i: probs[i]),
                         max(range(len(logits)), key=lambda i: logits[i]))
        # 一様ではない(logits が一様でないとき softmax が効いている)。
        self.assertGreater(max(probs) - min(probs), 1e-6)

    def test_softmax_priors_are_numerically_stable(self):
        """大きな logits でも exp が溢れない(数値安定版)。"""
        unused_state, obs = _root()
        logits = [1000.0 * (i + 1) for i in range(len(obs.legal_actions))]
        priors, unused_value = NetEvaluator(_FakeValueNet(0.0, logits)).evaluate(obs)
        probs = [p for unused_a, p in priors]
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertEqual(max(range(len(probs)), key=lambda i: probs[i]),
                         max(range(len(logits)), key=lambda i: logits[i]))

    def test_zero_logits_are_uniform(self):
        """全 0 logits の softmax は一様(旧テストとの互換)。"""
        unused_state, obs = _root()
        priors, unused_value = NetEvaluator(_FakeValueNet(0.0)).evaluate(obs)
        probs = [p for unused_a, p in priors]
        self.assertAlmostEqual(sum(probs), 1.0)
        for prob in probs:
            self.assertAlmostEqual(prob, 1.0 / len(obs.legal_actions))


class _PriorBiasedEvaluator(Evaluator):
    """最初の呼出(ルート prior の事前計算)だけ強く偏った priors を返すフェイク。

    favorite(legal の sort_key 最大)へ prior 0.999 を与え、以降の呼出(展開)は
    一様にする。value は常に 0.5 で、選択結果を prior だけに鋭敏にする。
    0.999/0.001 の偏りなら、探索中に他手の sqrt(availability) 補正が favorite
    の Q+prior 項を追い越す交点は千回超の訪問後で、予算 400 では到達しない。
    """

    def __init__(self):
        self.observations = []
        self._first = True

    def evaluate(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        self.observations.append(observation)
        if len(legal) == 1:
            return [(legal[0], 1.0)], 0.5
        if self._first:
            self._first = False
            favorite = max(legal, key=lambda a: a.sort_key())
            probs = [0.999 if a == favorite else 0.001 / (len(legal) - 1)
                     for a in legal]
        else:
            prob = 1.0 / len(legal)
            probs = [prob] * len(legal)
        return list(zip(legal, probs)), 0.5


class PuctSearchTests(unittest.TestCase):
    """S2: 評価器使用時は未展開も含めて PUCT で選択する(evaluator=None は従来 UCB)。"""

    def test_search_with_c_puct_returns_legal_action_and_is_deterministic(self):
        unused_state, obs = _root(seed=44)
        first = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=1.5,
                            evaluator=_FakeEvaluator(0.5))
        second = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=1.5,
                             evaluator=_FakeEvaluator(0.5))
        action = first.act(obs)
        self.assertIn(action, obs.legal_actions)
        self.assertEqual(action, second.act(obs))
        self.assertEqual(first.stats.samples_per_action,
                         second.stats.samples_per_action)
        self.assertEqual(first.stats.transitions, second.stats.transitions)
        self.assertEqual(first.stats.evaluations, second.stats.evaluations)

    @staticmethod
    def _node_with_priors(priors, visits=10):
        node = _Node(actor=0)
        node.visits = visits
        node.priors = dict(priors)
        for action in priors:
            node.edge_visits[action] = 3
            node.availability[action] = 5
            node.value_sum[action] = [0.3, 0.3]
        return node

    def test_puct_score_prefers_higher_prior(self):
        """同 value / 同訪問でも prior が大きい action の score が上。"""
        node = self._node_with_priors({1: 0.8, 2: 0.2})
        agent = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=1.5)
        self.assertGreater(agent._puct(node, 1, node.actor), agent._puct(node, 2, node.actor))

    def test_puct_score_matches_formula(self):
        node = self._node_with_priors({1: 0.8, 2: 0.2})
        agent = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=1.5)
        visits = node.edge_visits[1]
        expected = (node.value_sum[1][node.actor] / visits
                    + 1.5 * 0.8 * math.sqrt(node.availability[1])
                    / (1.0 + visits))
        self.assertAlmostEqual(agent._puct(node, 1, node.actor), expected)

    def test_unvisited_action_uses_fpu_and_prior(self):
        """未訪問は +inf ではなく FPU=0.5 で同一尺度に乗る(S2)。"""
        node = _Node(actor=0)
        node.availability[1] = 1
        node.availability[2] = 1
        node.priors = {1: 0.8, 2: 0.2}
        agent = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=1.5)
        self.assertNotEqual(agent._puct(node, 1, node.actor), float("inf"))
        # edge_visits=0 → Q=FPU=0.5、prior 項は sqrt(availability)/(1+0)。
        expected = 0.5 + 1.5 * 0.8 * math.sqrt(1.0) / 1.0
        self.assertAlmostEqual(agent._puct(node, 1, node.actor), expected)
        # prior の高い未訪問手のほうが高いスコア(優先展開される)。
        self.assertGreater(agent._puct(node, 1, node.actor), agent._puct(node, 2, node.actor))

    def test_puct_indexes_value_by_current_actor_not_node_actor(self):
        """歯(Phase 7 段1 の実クラッシュ): 展開時の決定化で終端だった子は
        `node.actor=None` を持つが、別の決定化では同じ経路が非終端に到達する。
        選択は現在の決定化 state の手番で value_sum を引かなければならない
        (node.actor で引くと `list indices must be integers` で落ちる)。"""
        node = _Node(actor=None)
        node.priors = {1: 0.6, 2: 0.4}
        for action in (1, 2):
            node.edge_visits[action] = 2
            node.availability[action] = 4
            node.value_sum[action] = [0.9, 0.1]
        agent = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=1.5)
        # 手番 0 の視点: value 0.45 + prior 項。手番 1 の視点: value 0.05 + prior 項。
        self.assertAlmostEqual(
            agent._puct(node, 1, 0),
            0.45 + 1.5 * 0.6 * math.sqrt(4.0) / 3.0)
        self.assertAlmostEqual(
            agent._puct(node, 1, 1),
            0.05 + 1.5 * 0.6 * math.sqrt(4.0) / 3.0)

    def test_c_puct_scales_exploration_term_only(self):
        node = self._node_with_priors({1: 0.8})
        small = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=1.0)
        large = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5), c_puct=5.0)
        self.assertGreater(large._puct(node, 1, node.actor), small._puct(node, 1, node.actor))
        value_term = node.value_sum[1][node.actor] / node.edge_visits[1]
        self.assertAlmostEqual(small._puct(node, 1, node.actor) - value_term,
                               (large._puct(node, 1, node.actor) - value_term) / 5.0)

    def test_c_puct_is_ignored_without_evaluator(self):
        """後方互換: evaluator=None では c_puct は従来 UCB と同一着手。"""
        unused_state, obs = _root(seed=44)
        plain = IsmctsAgent(9, 400, cards=H.CARDS)
        odd = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=99.0)
        self.assertEqual(plain.act(obs), odd.act(obs))
        self.assertEqual(plain.stats.transitions, odd.stats.transitions)
        self.assertEqual(plain.stats.nodes_created, odd.stats.nodes_created)
        self.assertEqual(plain.stats.samples_per_action,
                         odd.stats.samples_per_action)

    def test_puct_uses_prior_but_not_expansion_value(self):
        """展開時の value は seed に使わない(value は葉評価のみ)。"""
        unused_state, obs = _root(seed=44)
        agent = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=1.5,
                            evaluator=_FakeEvaluator(0.5))
        node = self._node_with_priors({1: 0.5})
        visits = node.edge_visits[1]
        before = node.value_sum[1][node.actor]
        # 展開時の value を変えても _puct の score は不変(value 項は
        # バックアップされた value_sum だけを見る)。
        self.assertEqual(agent._puct(node, 1, node.actor),
                         before / visits
                         + 1.5 * 0.5 * math.sqrt(node.availability[1])
                         / (1.0 + visits))

    def test_prior_affects_selection_end_to_end(self):
        """S2: prior が選択・展開順に効く(歯: unexpanded[0] 強制展開の再発防止)。

        ルート prior だけ favorite(sort_key 最大)へ 0.999 を振るフェイク評価器
        を使う。c_puct=1.5 なら最初の展開から favorite が選ばれ、最終着手も
        favorite になる。c_puct=0.0 の対照では prior 項が消え Q=FPU 一定の
        ため sort_key 最小へ退化する(prior が選択に効かなければ両者は一致)。
        """
        unused_state, obs = _root(seed=44)
        favorite = max(obs.legal_actions, key=lambda a: a.sort_key())
        plain = min(obs.legal_actions, key=lambda a: a.sort_key())
        self.assertNotEqual(favorite, plain)
        biased = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=1.5,
                             evaluator=_PriorBiasedEvaluator())
        no_prior = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=0.0,
                               evaluator=_PriorBiasedEvaluator())
        self.assertEqual(biased.act(obs), favorite)
        self.assertEqual(no_prior.act(obs), plain)

    def test_root_priors_are_computed_before_search(self):
        """S2: 探索前にルート観測で evaluate が1回呼ばれる(事前計算)。"""
        unused_state, obs = _root(seed=44)
        evaluator = _FakeEvaluator(0.5)
        agent = IsmctsAgent(9, 400, cards=H.CARDS, evaluator=evaluator)
        agent.act(obs)
        self.assertGreater(len(evaluator.observations), 0)
        # 最初の呼出はルート観測そのもの(探索ループの前)。
        self.assertEqual(serialize_observation(evaluator.observations[0]),
                         serialize_observation(obs))
        self.assertGreater(agent.stats.evaluations, 0)


class ZeroSumVectorTests(unittest.TestCase):
    """D-2: 探索側が絶対席順の 2人ゼロ和ベクトル [p, 1-p] を組み立てる。"""

    def _leaf_value(self, actor, p):
        state = H.new(2, seed=44)
        H.turn(state, actor)
        self.assertTrue(H.legal(state))     # 開始局面は配置手がある(OPEN-010)
        evaluator = _FakeEvaluator(p)
        stats = SearchStats()
        value = evaluate_with_evaluator(evaluator, state, H.CARDS, stats)
        return value, evaluator, stats

    def test_actor_seat_zero_vector(self):
        value, evaluator, stats = self._leaf_value(0, 0.25)
        self.assertEqual(value, [0.25, 0.75])
        self.assertAlmostEqual(sum(value), 1.0)
        self.assertEqual(stats.evaluations, 1)
        self.assertEqual(len(evaluator.observations), 1)

    def test_actor_seat_one_vector_is_reversed(self):
        value, evaluator, unused_stats = self._leaf_value(1, 0.25)
        self.assertEqual(value, [0.75, 0.25])
        self.assertAlmostEqual(sum(value), 1.0)

    def test_evaluator_receives_actor_observation_only(self):
        unused_value, evaluator, unused_stats = self._leaf_value(0, 0.5)
        obs = evaluator.observations[0]
        # observe() は actor 視点のときだけ legal_actions を埋める。評価器が
        # 受け取るのは決定化済み state の actor 観測だけ(D-6)。
        self.assertEqual(obs.player_id, 0)
        self.assertIsNotNone(obs.legal_actions)
        self.assertEqual(obs.num_players, 2)

    def test_terminal_utilities_map_to_win_probabilities(self):
        state = H.new(2, seed=8)
        while state.phase != Phase.GAME_OVER:
            state = game.step(state, H.legal(state)[0], H.CARDS).state
        evaluator = _FakeEvaluator(0.5)
        stats = SearchStats()
        value = evaluate_with_evaluator(evaluator, state, H.CARDS, stats)
        utilities = game.terminal_utilities(state, H.CARDS)
        self.assertEqual(value, [(u + 1) / 2 for u in utilities])
        self.assertAlmostEqual(sum(value), 1.0)
        # 終端は評価器を呼ばず、計上もしない(Phase 5: 実測 net 呼出数の正本)。
        self.assertEqual(evaluator.observations, [])
        self.assertEqual(stats.evaluations, 0)


class DelegationTests(unittest.TestCase):
    """D-1: pending / 一択は policy_agent へ委譲し、探索 stats を増やさない。"""

    def _pending_observation(self):
        state = H.new()
        H.own(state, 0, [H.wp("base:factory")])
        H.set_hand(state, 0, [], goods=2)
        state = H.place(state, 0, H.wp("base:factory"))
        self.assertIsNotNone(state.pending)
        return observe(state, state.pending.player_id, H.CARDS)

    def test_pending_decision_delegates_to_policy_agent(self):
        obs = self._pending_observation()
        delegate = _RecordingPolicyAgent()
        agent = IsmctsAgent(9, 400, cards=H.CARDS, policy_agent=delegate)
        action = agent.act(obs)
        self.assertEqual(action, obs.legal_actions[0])
        self.assertEqual(delegate.calls, [obs])
        self.assertEqual(agent.stats.decisions, 0)
        self.assertEqual(agent.stats.transitions, 0)
        self.assertEqual(agent.stats.sampler_calls, 0)
        self.assertEqual(agent.stats.evaluations, 0)

    def test_single_action_decision_delegates_to_policy_agent(self):
        unused_state, obs = _root(seed=44)
        single = dataclasses.replace(obs, legal_actions=(obs.legal_actions[0],))
        delegate = _RecordingPolicyAgent()
        agent = IsmctsAgent(9, 400, cards=H.CARDS, policy_agent=delegate)
        action = agent.act(single)
        self.assertEqual(action, single.legal_actions[0])
        self.assertEqual(delegate.calls, [single])
        self.assertEqual(agent.stats.decisions, 0)
        self.assertEqual(agent.stats.transitions, 0)

    def test_pending_default_still_delegates_to_heuristic(self):
        """後方互換: policy_agent=None は従来どおりヒューリスティック委譲。"""
        obs = self._pending_observation()
        expected = HeuristicAgent(0).act(obs)
        agent = IsmctsAgent(9, 400, cards=H.CARDS)
        self.assertEqual(agent.act(obs), expected)
        self.assertEqual(agent.stats.decisions, 0)


class InformationBoundaryTests(unittest.TestCase):
    """D-6: 根の観測と整合する隠れ状態の置換で着手が不変。"""

    @staticmethod
    def _swap_hidden(state):
        """相手(非手番)の手札1枚と山札の先頭1枚を交換する(公開情報は不変)。

        手札の内容と山札の順序は非公開なので、観測は完全に一致したまま
        隠れ状態だけが変わる。
        """
        actor = game.actor(state)
        other = 1 - actor
        hand = list(state.players[other].hand_buildings)
        deck = list(state.deck)
        if not hand or not deck:
            raise AssertionError("expected a non-empty hidden hand and deck")
        hand_card = hand.pop(0)
        deck_card = deck.pop(0)
        hand.append(deck_card)
        deck.append(hand_card)
        state.players[other].hand_buildings = tuple(sorted(hand))
        state.deck = tuple(deck)
        return state

    def test_action_is_invariant_to_hidden_state_swap(self):
        state_a = H.new(2, seed=7)
        state_b = self._swap_hidden(H.new(2, seed=7))
        obs_a = observe(state_a, game.actor(state_a), H.CARDS)
        obs_b = observe(state_b, game.actor(state_b), H.CARDS)
        # 置換は「根の観測と整合」: 公開情報は完全に一致する。
        self.assertEqual(serialize_observation(obs_a),
                         serialize_observation(obs_b))
        # 隠れ状態は本当に違う(交換できている)。
        self.assertNotEqual(state_a.players[1].hand_buildings,
                            state_b.players[1].hand_buildings)
        evaluator = _FakeEvaluator(0.37)
        agent = IsmctsAgent(9, 400, cards=H.CARDS, evaluator=evaluator)
        self.assertEqual(agent.act(obs_a), agent.act(obs_b))
        # 評価器が見た観測も両実行で完全に一致する(決定化は根の観測だけから
        # 作られるため、評価器が隠れ状態へ触れていれば不一致になる)。
        total = len(evaluator.observations)
        self.assertGreater(total, 0)
        first = evaluator.observations[:total // 2]
        second = evaluator.observations[total // 2:]
        self.assertEqual(len(first), len(second))
        self.assertEqual(
            [serialize_observation(o) for o in first],
            [serialize_observation(o) for o in second])
        for seen in evaluator.observations:
            self.assertIsNotNone(seen.legal_actions)


class EvaluatorSearchTests(unittest.TestCase):
    """evaluator 使用時の探索: value 経路が static_value を使わない。"""

    def test_search_uses_evaluator_and_counts_evaluations(self):
        unused_state, obs = _root(seed=44)
        evaluator = _FakeEvaluator(0.25)
        agent = IsmctsAgent(9, 400, cards=H.CARDS, evaluator=evaluator)
        action = agent.act(obs)
        self.assertIn(action, obs.legal_actions)
        self.assertGreater(agent.stats.evaluations, 0)
        # static 経路(evaluate_static)は evaluator 使用時に通らない。
        self.assertEqual(agent.stats.clip_saturated, 0)
        for seen in evaluator.observations:
            self.assertIsNotNone(seen.legal_actions)
            self.assertEqual(seen.num_players, 2)

    def test_search_with_evaluator_is_deterministic(self):
        unused_state, obs = _root(seed=44)
        first = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5))
        second = IsmctsAgent(9, 400, cards=H.CARDS,
                             evaluator=_FakeEvaluator(0.5))
        self.assertEqual(first.act(obs), second.act(obs))
        self.assertEqual(first.stats.samples_per_action,
                         second.stats.samples_per_action)
        self.assertEqual(first.stats.transitions, second.stats.transitions)

    def test_budget_accounting_is_unchanged_with_evaluator(self):
        """遷移構成は変えない(D-3): 評価器は予算消費に影響しない。"""
        unused_state, obs = _root(seed=44)
        agent = IsmctsAgent(9, 400, cards=H.CARDS,
                            evaluator=_FakeEvaluator(0.5))
        agent.act(obs)
        self.assertEqual(agent.stats.transitions, 400)
        self.assertGreater(agent.stats.determinizations, 1)
        self.assertEqual(agent.stats.nodes_created, agent.stats.total_nodes)

    def test_backward_compatible_defaults(self):
        """後方互換: evaluator=None / policy_agent=None は従来と同一。"""
        unused_state, obs = _root(seed=44)
        plain = IsmctsAgent(9, 400, cards=H.CARDS)
        explicit = IsmctsAgent(9, 400, cards=H.CARDS,
                               evaluator=None, policy_agent=None)
        self.assertEqual(plain.act(obs), explicit.act(obs))
        self.assertEqual(plain.stats.transitions, explicit.stats.transitions)
        self.assertEqual(plain.stats.nodes_created,
                         explicit.stats.nodes_created)
        self.assertEqual(plain.stats.evaluations, explicit.stats.evaluations)


if __name__ == "__main__":
    unittest.main()
