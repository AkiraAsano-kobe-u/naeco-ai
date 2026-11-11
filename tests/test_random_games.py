"""全人数でのランダム大量対戦: 不変条件 + 終了可能性(PLAN.md Phase 1)。

環境変数 NAECO_GAMES で1人数あたりの局数を上書きできる(既定は軽量。CIなし前提)。
"""
import os
import random
import unittest

import helpers as H
from naeco.engine import game, invariants
from naeco.engine.state import DecisionKind, Phase

GAMES = int(os.environ.get("NAECO_GAMES", "40"))
STEP_LIMIT = 6000


class MassRandomGames(unittest.TestCase):

    def _play_one(self, n, seed):
        rng = random.Random(seed ^ (n * 0x9E3779B1))
        s = H.new(n, seed)
        steps = 0
        while s.phase != Phase.GAME_OVER:
            la = game.legal_actions(s, H.CARDS)
            # 保証1b(継続可能性): pending 中は常に非空。
            self.assertTrue(la, "empty legal_actions: phase=%r pending=%r"
                            % (s.phase, s.pending))
            # 保証2(行き詰まりがない): 配置フェーズで待機ワーカーがあれば手がある。
            if s.phase == Phase.PLACE and s.pending is None:
                self.assertTrue(s.players[s.current_player].workers_waiting > 0)
            a = rng.choice(la)
            s = game.step(s, a, H.CARDS).state
            invariants.check(s, H.CARDS, "always")
            if s.pending is None and s.phase != Phase.GAME_OVER:
                invariants.check(s, H.CARDS, "after_resolution")
            steps += 1
            self.assertLess(steps, STEP_LIMIT, "game did not terminate")
        # 終端の契約(§4)
        sc = game.final_scores(s, H.CARDS)
        rk = game.ranking(s, H.CARDS, sc)
        self.assertEqual(sorted(rk), list(range(n)))     # 厳密な全順序
        return s, steps

    def test_two_player_games_terminate_with_invariants(self):
        for seed in range(GAMES):
            self._play_one(2, seed)

    def test_three_player_games_terminate_with_invariants(self):
        for seed in range(GAMES):
            self._play_one(3, seed)

    def test_four_player_games_terminate_with_invariants(self):
        for seed in range(GAMES):
            self._play_one(4, seed)

    def test_continuation_guarantee_holds_inside_every_pending(self):
        """pending が非 None かつ legal_actions が空、が一度も起きないこと(§3.5)。"""
        rng = random.Random(123)
        for n in (2, 3, 4):
            for seed in range(GAMES // 2 + 1):
                s = H.new(n, seed + 500)
                while s.phase != Phase.GAME_OVER:
                    la = game.legal_actions(s, H.CARDS)
                    if s.pending is not None:
                        self.assertTrue(la)
                    s = game.step(s, rng.choice(la), H.CARDS).state


if __name__ == "__main__":
    unittest.main()
