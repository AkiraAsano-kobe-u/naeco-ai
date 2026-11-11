"""Phase 7 段2: Gumbel 根選択(root_policy="gumbel")のテスト。

歯の狙い:
- 決定論: 同じ seed / 観測 → 同じ着手・同じ last_root。
- 改善方策は全合法手にわたる確率分布(和 1、長さ = 合法手数)。
- c_scale=0 なら σ=0 で改善方策は prior と一致する(σ 経路の歯)。
- gumbel_scale=0 かつ m=1 なら候補は prior 最大の1手だけで、根の標本は
  その手にしか付かない(forced 根選択の歯)。
- 訪問済みの completed_q は value_sum/visits と一致し、未訪問は v_mix。
- PUCT 経路(既定)も評価器があれば last_root を記録し、評価器なしでは None。
- 評価器なしで root_policy="gumbel" は ValueError。
"""
import math
import unittest

from naeco.ai.search.evaluator import Evaluator
from naeco.ai.search.ismcts import IsmctsAgent, _Node
from naeco.engine import game
from naeco.engine.observe import observe

import helpers as H


def _root(seed=44):
    state = H.new(2, seed=seed)
    return state, observe(state, game.actor(state), H.CARDS)


class _RankedEvaluator(Evaluator):
    """priors は sort_key 昇順で単調減少(先頭が最大)。value は固定。"""

    def __init__(self, value=0.5):
        self.value = float(value)

    def evaluate(self, observation):
        legal = sorted(observation.legal_actions, key=lambda a: a.sort_key())
        if not legal:
            raise ValueError("no legal actions")
        raw = [1.0 / (i + 1) for i in range(len(legal))]
        norm = sum(raw)
        return [(a, r / norm) for a, r in zip(legal, raw)], self.value


class GumbelRootTests(unittest.TestCase):

    def _agent(self, **kw):
        base = dict(cards=H.CARDS, c_puct=1.5, evaluator=_RankedEvaluator(0.5),
                    root_policy="gumbel")
        base.update(kw)
        return IsmctsAgent(9, 400, **base)

    def test_requires_evaluator(self):
        with self.assertRaises(ValueError):
            IsmctsAgent(9, 400, cards=H.CARDS, root_policy="gumbel")
        with self.assertRaises(ValueError):
            IsmctsAgent(9, 400, cards=H.CARDS, root_policy="bogus")

    def test_deterministic_and_records_distribution(self):
        unused, obs = _root(44)
        first, second = self._agent(), self._agent()
        a1, a2 = first.act(obs), second.act(obs)
        self.assertIn(a1, obs.legal_actions)
        self.assertEqual(a1, a2)
        self.assertEqual(first.last_root, second.last_root)
        rec = first.last_root
        self.assertEqual(len(rec["improved_policy"]), len(obs.legal_actions))
        self.assertAlmostEqual(sum(rec["improved_policy"]), 1.0, places=9)
        self.assertTrue(all(p >= 0.0 for p in rec["improved_policy"]))
        self.assertEqual(rec["action_list"][rec["chosen_index"]], a1)
        self.assertEqual(sum(rec["visits"]), sum(
            first.stats.samples_per_action[0].values()))
        self.assertLessEqual(first.stats.transitions, 400)

    def test_c_scale_zero_reduces_improved_policy_to_prior(self):
        unused, obs = _root(44)
        agent = self._agent(c_scale=0.0)
        agent.act(obs)
        rec = agent.last_root
        for p, prior in zip(rec["improved_policy"], rec["prior"]):
            self.assertAlmostEqual(p, prior, places=9)

    def test_noise_zero_and_m_one_forces_prior_argmax(self):
        unused, obs = _root(44)
        agent = self._agent(gumbel_scale=0.0, gumbel_m=1)
        chosen = agent.act(obs)
        rec = agent.last_root
        best = max(range(len(rec["prior"])), key=lambda i: rec["prior"][i])
        self.assertEqual(rec["chosen_index"], best)
        self.assertEqual(chosen, rec["action_list"][best])
        # 根の標本はその1手にしか付かない(forced 根選択)。
        samples = agent.stats.samples_per_action[0]
        nonzero = [k for k, v in samples.items() if v > 0]
        self.assertEqual(nonzero, [str(chosen.sort_key())])
        self.assertGreater(samples[str(chosen.sort_key())], 0)

    def test_completed_q_visited_and_unvisited(self):
        agent = self._agent()
        root = _Node(actor=0)
        root.priors = {1: 0.5, 2: 0.3, 3: 0.2}
        root.edge_visits[1] = 4
        root.value_sum[1] = [2.8, 1.2]        # Q(1) = 0.7
        root.edge_visits[2] = 1
        root.value_sum[2] = [0.2, 0.8]        # Q(2) = 0.2
        completed, max_visits = agent._completed_q(root, [1, 2, 3], 0, 0.5)
        self.assertEqual(max_visits, 4)
        self.assertAlmostEqual(completed[1], 0.7)
        self.assertAlmostEqual(completed[2], 0.2)
        mixed = (0.5 * 0.7 + 0.3 * 0.2) / 0.8
        v_mix = (0.5 + 5 * mixed) / 6.0
        self.assertAlmostEqual(completed[3], v_mix)
        self.assertAlmostEqual(agent._sigma(0.7, 4), (50.0 + 4) * 1.0 * 0.7)

    def test_puct_default_records_root_when_evaluator_present(self):
        unused, obs = _root(44)
        agent = IsmctsAgent(9, 400, cards=H.CARDS, c_puct=1.5,
                            evaluator=_RankedEvaluator(0.5))
        chosen = agent.act(obs)
        rec = agent.last_root
        self.assertIsNotNone(rec)
        self.assertEqual(rec["action_list"][rec["chosen_index"]], chosen)
        self.assertEqual(rec["candidates"], len(obs.legal_actions))
        self.assertAlmostEqual(sum(rec["improved_policy"]), 1.0, places=9)
        # 訪問数最大の手が chosen(PUCT 経路の選択規則は不変)。
        best = max(range(len(rec["visits"])),
                   key=lambda i: (rec["visits"][i], -i))
        self.assertEqual(rec["visits"][rec["chosen_index"]], rec["visits"][best])

    def test_puct_without_evaluator_leaves_last_root_none(self):
        unused, obs = _root(44)
        agent = IsmctsAgent(9, 400, cards=H.CARDS)
        agent.act(obs)
        self.assertIsNone(agent.last_root)

    def test_gumbel_noise_is_seeded_by_decision(self):
        agent = self._agent()
        noise_a = agent._gumbel_noise(123, [1, 2, 3])
        noise_b = agent._gumbel_noise(123, [1, 2, 3])
        noise_c = agent._gumbel_noise(124, [1, 2, 3])
        self.assertEqual(noise_a, noise_b)
        self.assertNotEqual(noise_a, noise_c)
        self.assertTrue(all(math.isfinite(v) for v in noise_a.values()))


if __name__ == "__main__":
    unittest.main()
