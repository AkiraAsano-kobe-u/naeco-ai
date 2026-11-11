"""Phase 10 B-C4: 3/4 席の葉評価・max^n backup・greedy/ISMCTS の終局(D3)。

手計算 fixture:
- 葉: 各席の観測だけを評価器へ渡し、u を clip 後 (u+1)/2 にした値を絶対席順に
  並べる。終端は順位線形効用 (u+1)/2。合法手ゼロは n 個の 0.5。
- backup: 経路上の各辺へ n 成分すべてを加算する。選択は現在の決定化 state の
  actor 成分だけを使う(PUCT/FPU/c_puct は 2 席と同じ式)。
- static 葉(軽量 fallback): 潜在得点と他席平均の差を SCALE で [0,1] に clip。
"""
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.ai import value as value_mod
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.search import ismcts
from naeco.ai.search.common import Budget, SearchStats, evaluate_static
from naeco.ai.search.evaluator import Evaluator, UniformEvaluator
from naeco.ai.search.ismcts import IsmctsAgent, _Node, evaluate_with_evaluator
from naeco.ai.search.pimc import PimcAgent
from naeco.engine import cards as cards_mod
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Phase
from naeco.train import encoding_multi
from naeco.train.search_evaluator import NetEvaluator

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None
HAS_TORCH = torch is not None

CARDS = cards_mod.load_cards()


def _play_to_end(num_players, seed, agents=None):
    state = game.new_game(num_players, seed, CARDS)
    agents = agents or [HeuristicAgent(s) for s in range(num_players)]
    result = None
    while state.phase != Phase.GAME_OVER:
        actor = game.actor(state)
        obs = observe(state, actor, CARDS)
        result = game.step(state, agents[actor].act(obs), CARDS)
        state = result.state
    return state, result


class SeatValueEvaluator(Evaluator):
    """席ごとに固定の [0,1] 値を返す評価器。受け取った観測を記録する。"""

    def __init__(self, values):
        self.values = list(values)
        self.seen = []

    def evaluate(self, observation):
        legal = observation.legal_actions
        prob = 1.0 / len(legal)
        return [(a, prob) for a in legal], self.values[observation.player_id]

    def evaluate_value(self, observation):
        self.seen.append(observation)
        return self.values[observation.player_id]


class SequenceValueNet(object):
    """infer_values が呼出順に u を返すフェイク net(arch_dict は持たない)。"""

    def __init__(self, utilities):
        self.utilities = list(utilities)
        self.inputs = []

    def infer_values(self, obs_list):
        self.inputs.append(obs_list[0])
        return [self.utilities[len(self.inputs) - 1]]


class LeafVectorFixture(unittest.TestCase):

    def test_each_seat_is_evaluated_from_its_own_observation(self):
        for n, values in ((3, [0.1, 0.6, 0.35]), (4, [0.9, 0.2, 0.4, 0.7])):
            state = game.new_game(n, 12, CARDS)
            evaluator = SeatValueEvaluator(values)
            stats = SearchStats()
            leaf = evaluate_with_evaluator(evaluator, state, CARDS, stats)
            self.assertEqual(leaf, values)            # 絶対席順そのまま
            self.assertEqual(stats.evaluations, n)    # 各席の実呼出を数える
            self.assertEqual([o.player_id for o in evaluator.seen],
                             list(range(n)))
            actor = game.actor(state)
            for obs in evaluator.seen:
                # 観測だけが渡る(真の hidden state は渡らない)。非手番席は
                # 合法手を持たない観測でも評価できる。
                self.assertEqual(obs, observe(state, obs.player_id, CARDS))
                if obs.player_id != actor:
                    self.assertIsNone(obs.legal_actions)

    def test_net_evaluator_maps_clip_and_seat_order(self):
        """u = [0.6, -0.2, 1.4] → p = [0.8, 0.4, 1.0](clip)。"""
        state = game.new_game(3, 8, CARDS)
        net = SequenceValueNet([0.6, -0.2, 1.4])
        stats = SearchStats()
        leaf = evaluate_with_evaluator(NetEvaluator(net), state, CARDS, stats)
        self.assertEqual(len(leaf), 3)
        for got, want in zip(leaf, [0.8, 0.4, 1.0]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(stats.evaluations, 3)
        for pid, vec in enumerate(net.inputs):
            self.assertEqual(vec, encoding_multi.encode_observation(
                observe(state, pid, CARDS)))

    def test_terminal_leaf_uses_rank_linear_utilities(self):
        """3 席 +1/0/−1 → 1/0.5/0、4 席 +1/+1/3/−1/3/−1 → 1/(2/3)/(1/3)/0。"""
        for n, mapped in ((3, [1.0, 0.5, 0.0]),
                          (4, [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])):
            state, unused_result = _play_to_end(n, 40 + n)
            ranking = game.ranking(state, CARDS)
            evaluator = SeatValueEvaluator([0.0] * n)
            stats = SearchStats()
            leaf = evaluate_with_evaluator(evaluator, state, CARDS, stats)
            for seat in range(n):
                self.assertAlmostEqual(leaf[seat], mapped[ranking[seat]],
                                       places=12)
            self.assertEqual(stats.evaluations, 0)
            self.assertEqual(evaluator.seen, [])

    def test_no_legal_action_leaf_is_neutral(self):
        state = game.new_game(4, 3, CARDS)
        evaluator = SeatValueEvaluator([0.9] * 4)
        stats = SearchStats()
        with mock.patch.object(ismcts.game, "legal_actions", return_value=[]):
            leaf = evaluate_with_evaluator(evaluator, state, CARDS, stats)
        self.assertEqual(leaf, [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(stats.evaluations, 0)

    def test_two_seat_leaf_is_unchanged(self):
        state = game.new_game(2, 4, CARDS)
        stats = SearchStats()
        leaf = evaluate_with_evaluator(UniformEvaluator(0.3), state, CARDS,
                                       stats)
        actor = game.actor(state)
        self.assertEqual(leaf, [0.3 if pid == actor else 0.7
                                for pid in range(2)])
        self.assertEqual(stats.evaluations, 1)


class ActorBackupFixture(unittest.TestCase):

    def test_puct_selects_by_the_current_actor_component(self):
        """value_sum[a] は全席分。席 1 の手番では成分 1 だけで比べる。

        visits=1, availability=2, prior=0.5, c_puct=1.5:
        bonus = 1.5 * 0.5 * sqrt(2) / 2 = 0.53033...
        A: [0.9, 0.1, 0.5] / B: [0.2, 0.7, 0.5]
        actor=0 → A=1.43033 > B=0.73033、actor=1 → B=1.23033 > A=0.63033。
        """
        agent = IsmctsAgent(0, 10, cards=CARDS,
                            evaluator=UniformEvaluator())
        node = _Node(1)
        a, b = "A", "B"
        node.priors = {a: 0.5, b: 0.5}
        node.value_sum = {a: [0.9, 0.1, 0.5], b: [0.2, 0.7, 0.5]}
        for act in (a, b):
            node.edge_visits[act] = 1
            node.availability[act] = 2
        bonus = 1.5 * 0.5 * math.sqrt(2) / 2.0
        self.assertAlmostEqual(agent._puct(node, a, 0), 0.9 + bonus, places=12)
        self.assertAlmostEqual(agent._puct(node, b, 0), 0.2 + bonus, places=12)
        self.assertAlmostEqual(agent._puct(node, a, 1), 0.1 + bonus, places=12)
        self.assertAlmostEqual(agent._puct(node, b, 1), 0.7 + bonus, places=12)
        # FPU は 2 席と同じ 0.5
        node.edge_visits["C"] = 0
        node.availability["C"] = 2
        node.priors["C"] = 0.0
        self.assertEqual(agent._puct(node, "C", 2), 0.5)

    def test_backup_adds_every_seat_component_along_the_path(self):
        """定数葉 v=[0.1,0.6,0.35] で k 反復 → 各辺の value_sum = visits * v。"""
        values = [0.1, 0.6, 0.35]
        state = game.new_game(3, 21, CARDS)
        actor = game.actor(state)
        observation = observe(state, actor, CARDS)
        agent = IsmctsAgent(3, 10 ** 6, cards=CARDS,
                            evaluator=SeatValueEvaluator(values))
        root = _Node(actor)
        priors, unused = agent._evaluator.evaluate(observation)
        root.priors = dict(priors)
        budget = Budget(2000).start()
        for iteration in range(12):
            agent._simulate(root, observation, 77, iteration, budget)

        def check(node):
            for action, child in node.children.items():
                visits = node.edge_visits[action]
                self.assertGreater(visits, 0)
                for seat in range(3):
                    self.assertAlmostEqual(node.value_sum[action][seat],
                                           visits * values[seat], places=9)
                check(child)

        self.assertEqual(sum(root.edge_visits.values()), 12)
        self.assertEqual(root.visits, 12)
        check(root)


class StaticFallbackFixture(unittest.TestCase):

    def _with_potentials(self, n, potentials):
        state = game.new_game(n, 2, CARDS)
        with mock.patch.object(value_mod, "_potentials",
                               return_value=list(potentials)):
            return value_mod.static_value(state, CARDS)

    def test_hand_computed_three_and_four_seat_values(self):
        # SCALE=30: own = clip(0.5 + (v_i - mean(others)) / 60)
        got = self._with_potentials(3, [10, 4, 1])
        for a, b in zip(got, [0.625, 0.475, 0.4]):
            self.assertAlmostEqual(a, b, places=12)
        got = self._with_potentials(3, [30, 0, 0])
        for a, b in zip(got, [1.0, 0.25, 0.25]):
            self.assertAlmostEqual(a, b, places=12)
        got = self._with_potentials(4, [12, 0, 6, 0])
        for a, b in zip(got, [0.6666666666666667, 0.4, 0.5333333333333333,
                              0.4]):
            self.assertAlmostEqual(a, b, places=12)
        got = self._with_potentials(4, [100, 0, 0, 0])
        self.assertEqual(got[0], 1.0)

    def test_two_seat_static_value_is_unchanged(self):
        got = self._with_potentials(2, [10, 4])
        self.assertAlmostEqual(got[0], 0.6, places=12)
        self.assertAlmostEqual(got[1], 0.4, places=12)

    def test_terminal_static_value_and_clip_count(self):
        state, unused = _play_to_end(3, 9)
        ranking = game.ranking(state, CARDS)
        got = value_mod.static_value(state, CARDS)
        for seat in range(3):
            self.assertAlmostEqual(got[seat], [1.0, 0.5, 0.0][ranking[seat]])
        stats = SearchStats()
        live = game.new_game(3, 2, CARDS)
        with mock.patch.object(value_mod, "_potentials",
                               return_value=[100, 0, 0]):
            from naeco.ai.search import common
            with mock.patch.object(common, "static_value",
                                   side_effect=value_mod.static_value):
                evaluate_static(live, CARDS, stats)
        self.assertEqual(stats.clip_saturated, 1)


class MultiSeatSearchPlaysLegally(unittest.TestCase):
    """評価器なし(static 葉)の ISMCTS / PIMC が 3/4 席で合法手のまま終局する。"""

    def test_static_ismcts_and_pimc_finish(self):
        for n in (3, 4):
            agents = [IsmctsAgent(5 + n, 80, cards=CARDS),
                      PimcAgent(9 + n, 40, cards=CARDS)]
            agents += [HeuristicAgent(s) for s in range(2, n)]
            state = game.new_game(n, 60 + n, CARDS)
            result = None
            while state.phase != Phase.GAME_OVER:
                actor = game.actor(state)
                obs = observe(state, actor, CARDS)
                action = agents[actor].act(obs)
                self.assertIn(action, obs.legal_actions)
                result = game.step(state, action, CARDS)
                state = result.state
            self.assertEqual(len(result.utilities), n)
            self.assertAlmostEqual(sum(result.utilities), 0.0, places=9)
            self.assertGreater(agents[0].stats.decisions, 0)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class UntrainedPolicyGames(unittest.TestCase):
    """人数別初期 checkpoint を保存/読込し、greedy / ISMCTS が合法手で終局する。"""

    def _load_initial(self, n, td):
        from naeco.train import net as net_mod
        meta_path = os.path.join(td, "init%d.meta.json" % n)
        net_mod.save_checkpoint(net_mod.build_for_players(n, seed=10 + n), n,
                                meta_path)
        net, meta = net_mod.load_checkpoint(meta_path, n)
        self.assertEqual(meta["num_players"], n)
        return net

    def _game(self, n, net, seed):
        from naeco.train.agent import PolicyAgent
        search = IsmctsAgent(seed, 150, cards=CARDS,
                             evaluator=NetEvaluator(net),
                             policy_agent=PolicyAgent(net, "greedy"))
        greedy = PolicyAgent(net, "greedy")
        agents = [search] + [greedy] * (n - 1)
        state = game.new_game(n, seed, CARDS)
        actions = []
        result = None
        while state.phase != Phase.GAME_OVER:
            actor = game.actor(state)
            obs = observe(state, actor, CARDS)
            action = agents[actor].act(obs)
            self.assertIn(action, obs.legal_actions)
            actions.append(action.sort_key())
            result = game.step(state, action, CARDS)
            state = result.state
        return actions, result, search.stats

    def test_greedy_and_ismcts_finish_with_legal_actions(self):
        for n in (3, 4):
            with tempfile.TemporaryDirectory() as td:
                net = self._load_initial(n, td)
            actions, result, stats = self._game(n, net, 31)
            self.assertTrue(result.terminal)
            self.assertEqual(len(result.utilities), n)
            self.assertAlmostEqual(sum(result.utilities), 0.0, places=9)
            self.assertGreater(stats.decisions, 0)
            # 多人数の葉は各席を推論するので評価回数 >= 決定数
            self.assertGreaterEqual(stats.evaluations, stats.decisions)
            again, unused_result, unused_stats = self._game(n, net, 31)
            self.assertEqual(actions, again)   # 固定 seed で決定論的


if __name__ == "__main__":
    unittest.main()
