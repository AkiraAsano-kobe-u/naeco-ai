"""Phase 2.5 の PIMC / SO-ISMCTS の契約(設計書 §6, §8.3)。"""
import os
import unittest

import helpers as H
from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.sampler import sample_hidden_state
from naeco.ai.search.common import Budget, SearchStats, rollout_from
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.ai.search.pimc import PimcAgent
from naeco.ai.value import static_value
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Phase, state_hash


def _root(seed=44, players=2):
    """配置手(pending なし)の観測を返す。"""
    state = H.new(players, seed=seed)
    return state, observe(state, game.actor(state), H.CARDS)


def _agents(budget):
    return (PimcAgent(9, budget, cards=H.CARDS),
            IsmctsAgent(9, budget, cards=H.CARDS))


class Contract(unittest.TestCase):

    def test_returns_a_legal_action(self):
        unused_state, obs = _root()
        for agent in _agents(400):
            self.assertIn(agent.act(obs), obs.legal_actions)

    def test_is_deterministic_for_the_same_seed_and_observation(self):
        unused_state, obs = _root()
        for first, second in zip(_agents(400), _agents(400)):
            self.assertEqual(first.act(obs), second.act(obs))

    def test_does_not_mutate_the_source_state(self):
        state, obs = _root()
        before = state_hash(state)
        for agent in _agents(400):
            agent.act(obs)
            self.assertEqual(state_hash(state), before)

    def test_pending_decisions_are_delegated_to_the_heuristic(self):
        """D-5: 探索するのは配置手だけ。下位決定はヒューリスティックに委譲する。"""
        state = H.new()
        H.own(state, 0, [H.wp("base:factory")])
        H.set_hand(state, 0, [], goods=2)
        state = H.place(state, 0, H.wp("base:factory"))
        self.assertIsNotNone(state.pending)
        obs = observe(state, state.pending.player_id, H.CARDS)
        expected = HeuristicAgent(0).act(obs)
        for agent in _agents(400):
            self.assertEqual(agent.act(obs), expected)
            self.assertEqual(agent.stats.decisions, 0)   # 探索していない


class BudgetAccounting(unittest.TestCase):
    """予算は状態遷移数で厳密に数える(D-6 / D-13)。"""

    def test_budget_is_consumed_exactly(self):
        unused_state, obs = _root()
        for budget in (200, 401, 1000):
            for agent in _agents(budget):
                agent.act(obs)
                self.assertEqual(agent.stats.transitions, budget)

    def test_budget_exhaustion_at_every_stage(self):
        """木降下中・展開直後・ロールアウト中のどこで尽きても壊れない。"""
        unused_state, obs = _root()
        smallest = len(obs.legal_actions)
        for budget in range(smallest, smallest + 40):
            for agent in _agents(budget):
                self.assertIn(agent.act(obs), obs.legal_actions)
                self.assertEqual(agent.stats.transitions, budget)

    def test_a_full_game_runs_without_exhausting_the_budget_mid_step(self):
        """実戦の全決定点を通す(小予算の単発テストでは出ない経路がある)。

        `_resolve_pending` が最後の1遷移で pending を解決し終えると残予算 0 のまま
        木降下へ戻り、`budget.step` が例外を投げていた(Opus 一次レビュー)。
        """
        for make in (lambda: PimcAgent(5, 300, cards=H.CARDS),
                     lambda: IsmctsAgent(5, 300, cards=H.CARDS)):
            state = H.new(2, seed=44)
            agent, opponent = make(), HeuristicAgent(0)
            agents = [agent, opponent]
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                action = agents[who].act(observe(state, who, H.CARDS))
                state = game.step(state, action, H.CARDS).state
            self.assertGreater(agent.stats.decisions, 0)

    def test_a_budget_below_the_action_count_is_rejected(self):
        unused_state, obs = _root()
        with self.assertRaises(ValueError):
            PimcAgent(9, len(obs.legal_actions) - 1, cards=H.CARDS).act(obs)

    def test_pimc_samples_every_action_within_one(self):
        """標本数の差は最大1(P25-014 / P25-025)。"""
        unused_state, obs = _root()
        agent = PimcAgent(9, 2000, cards=H.CARDS)
        agent.act(obs)
        counts = list(agent.stats.samples_per_action[0].values())
        self.assertGreater(min(counts), 0)
        self.assertLessEqual(max(counts) - min(counts), 1)
        self.assertLessEqual(agent.stats.max_sample_gap[0], 1)
        self.assertEqual(agent.stats.zero_sample_actions[0], 0)
        self.assertEqual(agent.stats.decisions_with_zero_sample_actions, 0)

    def test_pimc_derives_the_determinization_count_from_the_budget(self):
        """K は固定値ではなく予算から決まる。予算を増やせば決定化も増える。"""
        unused_state, obs = _root()
        small = PimcAgent(9, 400, cards=H.CARDS)
        large = PimcAgent(9, 4000, cards=H.CARDS)
        small.act(obs)
        large.act(obs)
        self.assertGreater(large.stats.determinizations,
                           small.stats.determinizations)
        self.assertEqual(small.stats.sampler_calls, small.stats.determinizations)

    def test_a_full_rollout_costs_far_more_than_a_handful_of_transitions(self):
        """局所予算方式が壊れていた根拠の回帰。

        1標本は十数〜数十遷移を使う。1標本ぶんに満たない予算を枠へ配る方式だと
        全標本が途中打切りになり、決定論的な方策では2標本目以降が
        同一ロールアウトの再計算になっていた(Opus 一次レビュー)。
        """
        unused_state, obs = _root()
        determinization = sample_hidden_state(obs, 1, H.CARDS)
        action = sorted(obs.legal_actions, key=lambda a: a.sort_key())[0]
        stats = SearchStats()
        budget = Budget(10 ** 6)
        unused_value, used = rollout_from(determinization, action, budget,
                                          H.CARDS, HeuristicAgent(0), 2, stats)
        self.assertGreater(used, 5)
        self.assertEqual(stats.truncated_samples, 0)


class IsmctsInternals(unittest.TestCase):

    def test_grows_a_tree_over_several_determinizations(self):
        unused_state, obs = _root()
        agent = IsmctsAgent(9, 2000, cards=H.CARDS)
        agent.act(obs)
        self.assertGreater(agent.stats.nodes_created, 1)
        self.assertGreater(agent.stats.determinizations, 1)


class ValueFunction(unittest.TestCase):

    def test_vector_sums_to_one_and_stays_in_range(self):
        value = static_value(H.new(2, seed=3), H.CARDS)
        self.assertEqual(len(value), 2)
        self.assertAlmostEqual(value[0] + value[1], 1.0)
        self.assertTrue(0.0 <= value[0] <= 1.0)

    def test_terminal_value_matches_the_utility_mapping(self):
        """終端も非終端と同一尺度 [0,1](P25-026)。"""
        state = H.new(2, seed=8)
        while state.phase != Phase.GAME_OVER:
            state = game.step(state, H.legal(state)[0], H.CARDS).state
        value = static_value(state, H.CARDS)
        utilities = game.terminal_utilities(state, H.CARDS)
        for pid in range(2):
            self.assertAlmostEqual(value[pid], (utilities[pid] + 1) / 2)


class ThreadControl(unittest.TestCase):

    def test_compare_script_sets_thread_limits_before_imports(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "scripts", "compare_search.py")
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        first_import = source.index("import argparse")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            self.assertLess(source.index(name), first_import)


class _FakeClock(object):
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _AdvancingClock(object):
    def __init__(self, delta):
        self.now = 0.0
        self.delta = delta

    def __call__(self):
        self.now += self.delta
        return self.now


class WallClockBudget(unittest.TestCase):
    """固定壁時計での停止(設計 D-6 の交差確認軸。sol SOL-P25G-003)。"""

    class _FakeClock(object):
        def __init__(self):
            self.now = 0.0

        def __call__(self):
            return self.now

    def test_time_budget_stops_the_search(self):
        clock = self._FakeClock()
        budget = Budget(10 ** 9, seconds=1.0, clock=clock).start()
        self.assertEqual(budget.remaining, 10 ** 9)
        clock.now = 0.99
        self.assertEqual(budget.remaining, 10 ** 9)
        clock.now = 1.0                       # 期限到達
        self.assertEqual(budget.remaining, 0)
        # 時間切れは軟らかい期限。step 自体は落とさない(実行中の preempt は不可)。
        state = H.new()
        budget.step(state, H.legal(state)[0], H.CARDS)
        # 硬い上限は遷移数だけ。
        hard = Budget(1).start()
        hard.step(state, H.legal(state)[0], H.CARDS)
        with self.assertRaises(RuntimeError):
            hard.step(state, H.legal(state)[0], H.CARDS)

    def test_transition_budget_is_unaffected_without_seconds(self):
        budget = Budget(5).start()
        self.assertIsNone(budget.seconds)
        self.assertEqual(budget.remaining, 5)

    def test_agents_accept_a_time_budget_and_stay_legal(self):
        unused_state, obs = _root()
        for agent in (PimcAgent(9, 10 ** 7, cards=H.CARDS, budget_seconds=0.20),
                      IsmctsAgent(9, 10 ** 7, cards=H.CARDS, budget_seconds=0.20)):
            action = agent.act(obs)
            self.assertIn(action, obs.legal_actions)
            # 遷移予算ではなく時間で止まっている(遷移は上限まで使い切っていない)。
            self.assertLess(agent.stats.transitions, 10 ** 7)
            self.assertGreater(agent.stats.transitions, 0)


class DeadlineAudit(unittest.TestCase):
    """期限超過は Budget で最初の境界に1回だけ集約し、SearchStats へ載せる。"""

    def test_overrun_is_recorded_once_with_totals_and_list(self):
        clock = _FakeClock()
        budget = Budget(100, seconds=1.0, clock=clock).start()
        clock.now = 1.5
        self.assertEqual(budget.check(), 0)
        self.assertEqual(budget.deadline_overrun_count, 1)
        self.assertEqual(budget.deadline_overrun_total_ms, 500.0)
        self.assertEqual(budget.deadline_overrun_max_ms, 500.0)
        self.assertEqual(budget.deadline_overruns_ms, [500.0])
        # 2回目以降は二重計上しない。
        self.assertEqual(budget.check(), 0)
        self.assertEqual(budget.deadline_overrun_count, 1)

    def test_cap_shares_the_deadline_without_double_counting(self):
        clock = _FakeClock()
        budget = Budget(100, seconds=1.0, clock=clock).start()
        clock.now = 1.25
        sub = budget.cap(10)
        self.assertEqual(sub.check(), 0)
        self.assertEqual(budget.deadline_overrun_count, 1)
        self.assertEqual(budget.deadline_overrun_total_ms, 250.0)
        self.assertEqual(budget.check(), 0)
        self.assertEqual(budget.deadline_overrun_count, 1)

    def test_rollout_used_counts_spent_not_unused_cap_after_deadline(self):
        """壁時計切れ時も消費遷移数は spent 差で計る(未消費分を計上しない)。"""
        unused_state, obs = _root()
        determinization = sample_hidden_state(obs, 1, H.CARDS)
        action = sorted(obs.legal_actions, key=lambda a: a.sort_key())[0]
        stats = SearchStats()
        budget = Budget(10 ** 6, seconds=2.5, clock=_AdvancingClock(1.0)).start()
        unused_value, used = rollout_from(determinization, action, budget,
                                          H.CARDS, HeuristicAgent(0), 2, stats)
        self.assertEqual(used, budget.spent)
        self.assertGreater(used, 0)
        self.assertLess(used, 10 ** 6)
        self.assertEqual(budget.deadline_overrun_count, 1)
        self.assertEqual(budget.deadline_overrun_total_ms, 500.0)

    def test_stats_merge_deadline_overruns(self):
        from naeco.ai.search.common import merge_deadline_overruns
        clock = _FakeClock()
        budget = Budget(100, seconds=1.0, clock=clock).start()
        clock.now = 1.25
        self.assertEqual(budget.check(), 0)
        stats = SearchStats()
        merge_deadline_overruns(budget, stats)
        self.assertEqual(stats.deadline_overrun_count, 1)
        self.assertEqual(stats.deadline_overrun_total_ms, 250.0)
        self.assertEqual(stats.deadline_overrun_max_ms, 250.0)
        self.assertEqual(stats.deadline_overruns_ms, [250.0])


class WallClockSearch(unittest.TestCase):
    """PIMC は壁時計でも初巡を無制限に続けない(sol SOL-P25G-003)。"""

    def test_pimc_stops_after_deadline_in_first_pass(self):
        unused_state, obs = _root()
        agent = PimcAgent(9, 10 ** 7, cards=H.CARDS, budget_seconds=1.5,
                          clock=_AdvancingClock(0.6))
        action = agent.act(obs)
        self.assertIn(action, obs.legal_actions)
        # start 1回 + ループ境界の clock 呼出で期限を越え、1標本だけで停止する。
        self.assertEqual(agent.stats.sampler_calls, 1)
        self.assertEqual(agent.stats.determinizations, 1)
        self.assertEqual(agent.stats.deadline_overrun_count, 1)
        self.assertAlmostEqual(agent.stats.deadline_overrun_total_ms, 300.0)
        self.assertAlmostEqual(agent.stats.deadline_overrun_max_ms, 300.0)
        self.assertAlmostEqual(agent.stats.deadline_overruns_ms[0], 300.0)
        self.assertGreater(agent.stats.zero_sample_actions[0], 0)
        self.assertEqual(agent.stats.decisions_with_zero_sample_actions, 1)
        self.assertEqual(agent.stats.truncated_samples, 1)

    def test_pimc_zero_sample_wall_clock_is_explicitly_flagged(self):
        """期限が最初の境界で切れた場合も合法手を返し、0標本を明示する。"""
        unused_state, obs = _root()
        agent = PimcAgent(9, 10 ** 7, cards=H.CARDS, budget_seconds=1.0,
                          clock=_AdvancingClock(1.2))
        action = agent.act(obs)
        self.assertIn(action, obs.legal_actions)
        self.assertEqual(agent.stats.sampler_calls, 0)
        self.assertEqual(agent.stats.determinizations, 0)
        self.assertEqual(agent.stats.decisions_with_zero_sample_actions, 1)
        self.assertEqual(agent.stats.zero_sample_actions[0],
                         len(obs.legal_actions))
        self.assertEqual(agent.stats.deadline_overrun_count, 1)
        self.assertAlmostEqual(agent.stats.deadline_overrun_total_ms, 200.0)


class IsmctsWallClockSearch(unittest.TestCase):
    """サンプラー中に期限を跨いでも、ISMCTS は早期停止として落ちずに終える。"""

    def test_sampler_crossing_deadline_returns_legal_fallback_without_crash(self):
        unused_state, obs = _root()
        agent = IsmctsAgent(9, 10 ** 7, cards=H.CARDS, budget_seconds=1.0,
                            clock=_AdvancingClock(0.6))
        action = agent.act(obs)
        self.assertIn(action, obs.legal_actions)
        self.assertEqual(agent.stats.decisions, 1)
        self.assertEqual(agent.stats.sampler_calls, 1)
        self.assertEqual(agent.stats.determinizations, 1)
        self.assertEqual(agent.stats.deadline_overrun_count, 1)
        self.assertAlmostEqual(agent.stats.deadline_overrun_total_ms, 200.0)
        self.assertAlmostEqual(agent.stats.deadline_overrun_max_ms, 200.0)
        self.assertAlmostEqual(agent.stats.deadline_overruns_ms[0], 200.0)
        self.assertEqual(agent.stats.zero_sample_actions[0],
                         len(obs.legal_actions))
        self.assertEqual(agent.stats.decisions_with_zero_sample_actions, 1)
        self.assertEqual(agent.stats.truncated_samples, 1)


class SearchStatsAudit(unittest.TestCase):
    """nodes/s と行動別標本監査の集計元を公開する。"""

    def test_nodes_and_elapsed_fields_are_aggregated(self):
        unused_state, obs = _root()
        pimc = PimcAgent(9, 400, cards=H.CARDS)
        pimc.act(obs)
        self.assertEqual(pimc.stats.search_elapsed_ms,
                         sum(pimc.stats.elapsed_ms))
        self.assertEqual(pimc.stats.total_nodes, pimc.stats.nodes_created)
        self.assertEqual(pimc.stats.total_nodes, 0)
        self.assertEqual(len(pimc.stats.max_sample_gap), 1)
        self.assertEqual(len(pimc.stats.zero_sample_actions), 1)

        ismcts = IsmctsAgent(9, 2000, cards=H.CARDS)
        ismcts.act(obs)
        self.assertEqual(ismcts.stats.search_elapsed_ms,
                         sum(ismcts.stats.elapsed_ms))
        self.assertEqual(ismcts.stats.total_nodes, ismcts.stats.nodes_created)
        self.assertGreater(ismcts.stats.total_nodes, 1)
        self.assertEqual(len(ismcts.stats.samples_per_action[0]),
                         len(obs.legal_actions))
        self.assertEqual(len(ismcts.stats.transitions_per_action[0]),
                         len(obs.legal_actions))

    def test_transition_budget_injected_clock_keeps_action_deterministic(self):
        unused_state, obs = _root()
        clock = _FakeClock()
        for agent in (PimcAgent(9, 400, cards=H.CARDS, clock=clock),
                      IsmctsAgent(9, 400, cards=H.CARDS, clock=clock)):
            reference = type(agent)(9, 400, cards=H.CARDS)
            expected = reference.act(obs)
            self.assertEqual(agent.act(obs), expected)


if __name__ == "__main__":
    unittest.main()
