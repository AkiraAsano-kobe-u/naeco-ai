"""Phase 2 の観測ベース・ヒューリスティック方策テスト。"""
import dataclasses
import inspect
import unittest

from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Action, ActionKind, DecisionKind, PendingDecision, Phase

import helpers as H


class HeuristicAgentTests(unittest.TestCase):

    def _observation_with(self, actions):
        base = observe(H.new(2, 17), 0, H.CARDS)
        # dataclasses.replace を使う。全フィールドを手で並べると Observation に
        # フィールドが増えるたびに本テストが壊れる(OBS_VERSION 0.3.0 で実際に壊れた)。
        return dataclasses.replace(base, legal_actions=tuple(actions))

    def test_returns_a_member_of_legal_actions(self):
        for seed in range(12):
            state = H.new(2, seed)
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                observation = observe(state, who, H.CARDS)
                action = HeuristicAgent(seed + who).act(observation)
                self.assertIn(action, observation.legal_actions)
                state = game.step(state, action, H.CARDS).state

    def test_multiple_seeds_finish_without_exception(self):
        for seed in range(16):
            state = H.new(2, seed)
            agents = (HeuristicAgent(seed), HeuristicAgent(seed + 1000))
            steps = 0
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                action = agents[who].act(observe(state, who, H.CARDS))
                state = game.step(state, action, H.CARDS).state
                steps += 1
                self.assertLess(steps, 6000)

    def test_all_intermediate_decision_kinds_are_traversed(self):
        agent = HeuristicAgent(9)

        # 大工の建設対象選択から、支払い札の廃棄選択までを実際に進める。
        state = H.new(2, 20)
        H.set_hand(state, 0, (H.wp("base:farm"), H.wp("base:cafe")))
        state = H.place(state, 0, H.wp("base:carpenter"))
        build = agent.act(observe(state, 0, H.CARDS))
        self.assertEqual(build.kind, ActionKind.SELECT_BUILD)
        state = game.step(state, build, H.CARDS).state
        discard = agent.act(observe(state, 0, H.CARDS))
        self.assertEqual(discard.kind, ActionKind.SELECT_DISCARD)

        # 設計事務所の公開札からの保持選択。
        state = H.new(2, 21)
        H.own(state, 0, (H.wp("base:design-office"),))
        state = H.place(state, 0, H.wp("base:design-office"))
        self.assertEqual(agent.act(observe(state, 0, H.CARDS)).kind,
                         ActionKind.SELECT_KEEP)

        # 賃金不足の売却選択も pending として観測経由で渡す。
        state = H.new(2, 22)
        H.own(state, 0, (H.wp("base:cafe"),))
        state.phase = Phase.END
        state.pending = PendingDecision(
            DecisionKind.INSOLVENCY_SELL, 0, "test", 0)
        self.assertEqual(agent.act(observe(state, 0, H.CARDS)).kind,
                         ActionKind.SELL_BUILDING)

    def test_same_seed_and_observation_sequence_is_deterministic(self):
        state = H.new(2, 33)
        observations = []
        while state.phase != Phase.GAME_OVER:
            who = game.actor(state)
            observation = observe(state, who, H.CARDS)
            observations.append(observation)
            state = game.step(state, observation.legal_actions[0], H.CARDS).state
        first = HeuristicAgent(88)
        second = HeuristicAgent(88)
        self.assertEqual([first.act(obs) for obs in observations],
                         [second.act(obs) for obs in observations])

    def test_information_boundary_is_observation_only(self):
        parameters = list(inspect.signature(HeuristicAgent.act).parameters)
        self.assertEqual(parameters, ["self", "observation"])
        source = inspect.getsource(HeuristicAgent)
        self.assertNotIn("GameState", source)
        # 相手の公開ビューには hand_size しか無く、内訳を要求しない。
        observation = self._observation_with((
            Action(ActionKind.PLACE_WORKER, target_instance_id=H.wp("base:mine")),
        ))
        self.assertEqual(HeuristicAgent(1).act(observation),
                         observation.legal_actions[0])


class QuarryPreferenceTests(unittest.TestCase):
    """採石場が一度も選ばれない問題(sol H)の回帰。

    採石場(1枚ドロー + 開始プレイヤーマーカー)と鉱山(1枚ドローのみ)を同点に
    すると、同点解決の sort_key 昇順で `base:mine` が必ず勝ち、採石場が死に手に
    なる。鉱山は占有無制限なので毎ターン候補に残り続ける。
    """

    def _pick(self, round_number, start_player_offset):
        base = observe(H.new(2, 5), 0, H.CARDS)
        obs = dataclasses.replace(
            base, round_number=round_number,
            start_player_offset=start_player_offset,
            legal_actions=(
                Action(ActionKind.PLACE_WORKER,
                       target_instance_id=H.wp("base:mine")),
                Action(ActionKind.PLACE_WORKER,
                       target_instance_id=H.wp("base:quarry")),
            ),
        )
        return HeuristicAgent(3).act(obs).target_instance_id

    def test_the_quarry_beats_the_mine_when_the_marker_still_matters(self):
        self.assertEqual(self._pick(1, 1), H.wp("base:quarry"))

    def test_the_quarry_is_still_taken_to_deny_it_while_holding_the_marker(self):
        self.assertEqual(self._pick(1, 0), H.wp("base:quarry"))

    def test_the_marker_is_worthless_in_the_last_round(self):
        """R9 で採石場を取っても start_player は更新されない(_finish_round)。"""
        self.assertEqual(self._pick(9, 1), H.wp("base:mine"))

    def test_self_play_actually_uses_the_quarry(self):
        used = 0
        for seed in range(12):
            state = H.new(2, seed)
            agents = {p: HeuristicAgent(p) for p in range(2)}
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                action = agents[who].act(observe(state, who, H.CARDS))
                if (action.kind == ActionKind.PLACE_WORKER
                        and "quarry" in action.target_instance_id):
                    used += 1
                state = game.step(state, action, H.CARDS).state
        self.assertGreater(used, 0)

    def test_the_start_player_marker_actually_moves_in_self_play(self):
        moved = False
        for seed in range(12):
            state = H.new(2, seed)
            agents = {p: HeuristicAgent(p) for p in range(2)}
            first = state.start_player
            while state.phase != Phase.GAME_OVER:
                who = game.actor(state)
                state = game.step(
                    state, agents[who].act(observe(state, who, H.CARDS)),
                    H.CARDS).state
                if state.start_player != first:
                    moved = True
        self.assertTrue(moved)


if __name__ == "__main__":
    unittest.main()
