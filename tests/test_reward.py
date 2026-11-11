"""報酬モジュールのテスト(Phase 3 設計書 §4.2 / core-api §9)。"""
import random
import unittest

import helpers as H
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Phase
from naeco.train import reward

STEP_LIMIT = 6000


def _potentials(state, n):
    """各席の観測から score_diff_potentials の該当席成分を集めたタプル。"""
    return tuple(
        reward.score_diff_potentials(observe(state, pid, H.CARDS), H.CARDS)[pid]
        for pid in range(n)
    )


def _random_steps(seed, n):
    """終端までランダムに step した StepResult を yield する。"""
    rng = random.Random(seed)
    state = H.new(n, seed)
    steps = 0
    while state.phase != Phase.GAME_OVER:
        actions = game.legal_actions(state, H.CARDS)
        result = game.step(state, rng.choice(actions), H.CARDS)
        yield result
        state = result.state
        steps += 1
        if steps >= STEP_LIMIT:
            raise AssertionError("game did not terminate (seed=%d n=%d)" % (seed, n))


def _insolvency_pending_state():
    """賃金支払い不能で強制売却 pending が立った2人局面を作る。"""
    state = H.new(2, 0)
    H.set_hand(state, 0, buildings=(), goods=0)
    H.own(state, 0, ["base:farm#00", "base:farm#01"])
    H.set_hand(state, 1, buildings=(), goods=0)
    state.players[0].money = 0
    state.players[0].workers_waiting = 3
    state.players[0].workers_training = 1   # 賃金 2×4 を農場2枚(6+6)で払えない
    state.players[1].money = 0
    state.players[1].workers_waiting = 0
    state.phase = Phase.PLACE
    state.current_player = 0
    state.pending = None
    while state.phase == Phase.PLACE:
        action = next(a for a in game.legal_actions(state, H.CARDS)
                      if a.kind.value == "place_worker")
        state = game.step(state, action, H.CARDS).state
    pending = state.pending
    if pending is None or pending.kind.value != "insolvency_sell":
        raise AssertionError("expected insolvency_sell pending")
    return state


class RewardModuleTests(unittest.TestCase):

    def test_none_shaping_all_steps_zero(self):
        """既定 shaping="none" では数十局の全 step が長さ N のゼロタプル。"""
        config = reward.RewardConfig()
        counts = {2: 20, 3: 10, 4: 10}
        for n, games in counts.items():
            for seed in range(games):
                for result in _random_steps(seed + n * 1000, n):
                    self.assertEqual(
                        reward.step_rewards((0.0,) * n, result, config, H.CARDS),
                        (0.0,) * n,
                    )

    def test_turn_advanced_false_is_zero_with_score_diff(self):
        """turn_advanced=False の step は score_diff でも報酬 0(歯 #3)。"""
        config = reward.RewardConfig(shaping="score_diff", shaping_scale=1.0)
        seen_pending_step = False
        for seed in range(12):
            rng = random.Random(seed)
            state = H.new(2, seed)
            prev = _potentials(state, 2)
            while state.phase != Phase.GAME_OVER:
                actions = game.legal_actions(state, H.CARDS)
                result = game.step(state, rng.choice(actions), H.CARDS)
                if not result.turn_advanced:
                    seen_pending_step = True
                    self.assertEqual(
                        reward.step_rewards(prev, result, config, H.CARDS),
                        (0.0, 0.0),
                    )
                if result.turn_advanced:
                    prev = _potentials(result.state, 2)
                state = result.state
        self.assertTrue(seen_pending_step, "no pending step sampled; test is vacuous")

        # 決定的な歯: 売却は価値中立(所持金と所有建物が1:1)でΦは変化しない。
        # そこで「Φ(s') と異なる参照」を意図的に渡し、報酬0がΦ不変による見せかけ
        # ではなく turn_advanced ガードによることを検証する。ガードを外すと
        # この参照差がそのまま shaped 報酬として返ってくる。
        state = _insolvency_pending_state()
        sell_action = next(a for a in game.legal_actions(state, H.CARDS)
                           if a.kind.value == "sell_building")
        prev = _potentials(state, 2)
        result = game.step(state, sell_action, H.CARDS)
        self.assertFalse(result.turn_advanced)
        fake_prev = tuple(value + 0.25 for value in prev)
        self.assertEqual(
            reward.step_rewards(fake_prev, result, config, H.CARDS), (0.0, 0.0))

        # 同じ fake_prev を手番が進む step に渡すと非ゼロになること(配管の確認)。
        # これで「常にゼロを返す実装」と「ガードでゼロを返す実装」を区別できる。
        turn_state = H.new(2, 0)
        turn_result = None
        for action in game.legal_actions(turn_state, H.CARDS):
            candidate = game.step(turn_state, action, H.CARDS)
            if candidate.turn_advanced:
                turn_result = candidate
                break
        self.assertIsNotNone(turn_result)
        self.assertNotEqual(
            reward.step_rewards(fake_prev, turn_result, config, H.CARDS),
            (0.0, 0.0))

    def test_terminal_rewards_match_terminal_utilities(self):
        """terminal_rewards は game.terminal_utilities と完全一致。"""
        for seed in range(8):
            rng = random.Random(seed)
            state = H.new(2, seed + 500)
            while state.phase != Phase.GAME_OVER:
                actions = game.legal_actions(state, H.CARDS)
                state = game.step(state, rng.choice(actions), H.CARDS).state
            self.assertEqual(
                reward.terminal_rewards(state, H.CARDS),
                game.terminal_utilities(state, H.CARDS),
            )
            self.assertEqual(len(reward.terminal_rewards(state, H.CARDS)), 2)
            self.assertTrue(all(u in (1.0, -1.0)
                                for u in reward.terminal_rewards(state, H.CARDS)))

    def test_score_diff_telescopes_without_changing_terminal(self):
        """score_diff の shaped 合計は Φ(s_T)−Φ(s_0) に telescope し、主報酬は不変。"""
        config = reward.RewardConfig(shaping="score_diff", shaping_scale=1.0)
        for seed in range(8):
            rng = random.Random(seed)
            state = H.new(2, seed + 900)
            start = _potentials(state, 2)
            prev = start
            shaped_total = [0.0, 0.0]
            while state.phase != Phase.GAME_OVER:
                actions = game.legal_actions(state, H.CARDS)
                result = game.step(state, rng.choice(actions), H.CARDS)
                shaped = reward.step_rewards(prev, result, config, H.CARDS)
                for pid in range(2):
                    shaped_total[pid] += shaped[pid]
                if result.turn_advanced:
                    prev = _potentials(result.state, 2)
                state = result.state
            terminal = _potentials(state, 2)
            for pid in range(2):
                self.assertAlmostEqual(
                    shaped_total[pid], terminal[pid] - start[pid], places=6)
            self.assertEqual(
                reward.terminal_rewards(state, H.CARDS),
                game.terminal_utilities(state, H.CARDS),
            )

    def test_unknown_shaping_raises_value_error(self):
        """未知 shaping は step 内容と無関係に ValueError。"""
        state = H.new(2, 0)
        result = game.step(state, game.legal_actions(state, H.CARDS)[0], H.CARDS)
        with self.assertRaises(ValueError):
            reward.step_rewards(
                (0.0, 0.0), result,
                reward.RewardConfig(shaping="bogus"), H.CARDS)
        with self.assertRaises(ValueError):
            reward.step_rewards(
                (0.0, 0.0), result,
                reward.RewardConfig(shaping="SCORE_DIFF"), H.CARDS)

    def test_multi_player_terminal_matches_engine(self):
        """N>=3 の順位線形効用もエンジンへ委譲する。"""
        for n in (3, 4):
            for result in _random_steps(n, n):
                if result.terminal:
                    self.assertEqual(reward.terminal_rewards(result.state, H.CARDS),
                                     result.utilities)
                    assert result.utilities is not None
                    self.assertEqual(len(result.utilities), n)

    def test_reward_version_is_selected_by_player_count(self):
        self.assertEqual(reward.REWARD_VERSION, "1.0.0")
        self.assertEqual(reward.reward_version(2), "1.0.0")
        self.assertEqual(reward.reward_version(3), "2.0.0")
        self.assertEqual(reward.reward_version(4), "2.0.0")
        for invalid in (0, 1, 5, True, False, 2.0, "3", None):
            with self.assertRaises(ValueError):
                reward.reward_version(invalid)

    def test_multi_player_score_diff_shaping_is_rejected(self):
        """N=3,4 の shaping は none に限定し、score_diff は拒否する。"""
        for n in (3, 4):
            state = H.new(n)
            result = game.step(state, game.legal_actions(state, H.CARDS)[0], H.CARDS)
            with self.assertRaises(ValueError):
                reward.step_rewards(
                    (0.0,) * n, result,
                    reward.RewardConfig(shaping="score_diff", shaping_scale=1.0),
                    H.CARDS)

    def test_score_diff_potentials_contract(self):
        """Φ は長さ N・平均0・席順タプルで、所持金変化に反応する。"""
        state = H.new(2, 0)
        potentials = _potentials(state, 2)
        self.assertEqual(len(potentials), 2)
        self.assertAlmostEqual(sum(potentials), 0.0, places=9)
        state.players[0].money += 10
        moved = _potentials(state, 2)
        # 推定スコア +10 → 平均 +5 なので Φ0 +0.05 / Φ1 −0.05(正規化 /100)。
        self.assertAlmostEqual(moved[0] - potentials[0], 0.05, places=6)
        self.assertAlmostEqual(moved[1] - potentials[1], -0.05, places=6)

    def test_prev_potentials_length_mismatch_raises(self):
        """score_diff の参照ポテンシャルは長さ N でなければ ValueError。"""
        state = H.new(2, 0)
        result = None
        for action in game.legal_actions(state, H.CARDS):
            candidate = game.step(state, action, H.CARDS)
            if candidate.turn_advanced:
                result = candidate
                break
        self.assertIsNotNone(result)
        with self.assertRaises(ValueError):
            reward.step_rewards(
                (0.0,),
                result,
                reward.RewardConfig(shaping="score_diff", shaping_scale=1.0),
                H.CARDS,
            )

    def test_score_diff_zero_scale_is_zero(self):
        """既定 shaping_scale=0.0 のまま有効化しても shaped 報酬は 0。"""
        state = H.new(2, 0)
        result = None
        for action in game.legal_actions(state, H.CARDS):
            candidate = game.step(state, action, H.CARDS)
            if candidate.turn_advanced:
                result = candidate
                break
        self.assertIsNotNone(result)
        self.assertEqual(
            reward.step_rewards(
                _potentials(state, 2), result,
                reward.RewardConfig(shaping="score_diff"), H.CARDS),
            (0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
