"""席交換評価の決定論とプロトコルのテスト。"""
import dataclasses
import unittest

from naeco.ai import eval as eval_pkg
from naeco.ai.eval import harness as harness_mod
from naeco.ai.eval.harness import (
    GameRecord, cluster_bootstrap, evaluate, passes_threshold, summarize_pairs)
from naeco.engine.state import derive_seed, eval_seed


class EvaluationHarnessTests(unittest.TestCase):

    def test_small_evaluation_is_exactly_reproducible(self):
        first = evaluate(pairs=20, group="train", start_index=100,
                         bootstrap_samples=1000, bootstrap_seed=456)
        second = evaluate(pairs=20, group="train", start_index=100,
                          bootstrap_samples=1000, bootstrap_seed=456)
        self.assertEqual(first, second)

    def test_paired_games_have_the_same_initial_deal(self):
        result = evaluate(pairs=4, group="train", start_index=200,
                          bootstrap_samples=100, bootstrap_seed=7)
        for left, right in result.initial_state_hashes:
            self.assertEqual(left, right)

    def test_threshold_is_strictly_greater_than(self):
        result = evaluate(pairs=1, group="train", start_index=300,
                          bootstrap_samples=10, bootstrap_seed=4)
        equal_low = dataclasses.replace(result, ci_low=0.90)
        above_low = dataclasses.replace(result, ci_low=0.9000001)
        self.assertFalse(passes_threshold(equal_low, 0.90))
        self.assertTrue(passes_threshold(above_low, 0.90))


class ClusterBootstrapTests(unittest.TestCase):
    """統計処理そのもの(局単位への退行・引き分け集計・CI 値)の検査。"""

    def test_bootstrap_resamples_pairs_not_games(self):
        # 全ペアが厳密に 0.5(席交換の一方だけ勝つ完全相関)なら、
        # ペア単位リサンプルの平均は常に 0.5 で分散ゼロ → CI は退化して [0.5, 0.5]。
        # もし局単位でリサンプルしていれば {0,1} が混ざり CI は必ず広がる。
        # このテストは局単位 bootstrap への誤変更を検出する歯である。
        pair_results = [0.5] * 100
        low, high = cluster_bootstrap(pair_results, samples=2000, seed=1)
        self.assertEqual(low, 0.5)
        self.assertEqual(high, 0.5)

    def test_evaluate_feeds_pair_results_not_flattened_games(self):
        # `_play_game` をスタブ化し、各ペアを (勝, 敗) に固定する。ペア値は全て 0.5。
        # evaluate がペア単位で bootstrap するなら CI は [0.5, 0.5] に退化する。
        # 局結果 [1,0,1,0,...] を平坦化して bootstrap すれば CI は必ず広がるので、
        # このテストは evaluate 側の「局への平坦化」退行を検出する。
        real_play = harness_mod._play_game

        def fake_play_game(cards, group, index, seat_variant, env_seed,
                           target_seed, opponent_seed, target_seat, setup_seed):
            return GameRecord(
                group=group, index=index, seat_variant=seat_variant,
                env_seed=env_seed, setup_seed=setup_seed,
                policy_seed=target_seed,
                opponent_seed=opponent_seed, target_seat=target_seat,
                opponent_seat=1 - target_seat, start_player=0,
                initial_deck=(), initial_hands=(), initial_money=(5, 6),
                initial_state_hash="x" * 64,
                cards_sha256="c", cards_data_version="v",
                api_version="t", rules_version="t",
                result=1.0 if seat_variant == 0 else 0.0)

        harness_mod._play_game = fake_play_game
        try:
            result = evaluate(pairs=100, group="train", start_index=700,
                              bootstrap_samples=2000, bootstrap_seed=1)
        finally:
            harness_mod._play_game = real_play
        self.assertEqual((result.ci_low, result.ci_high), (0.5, 0.5))
        self.assertEqual(result.point_estimate, 0.5)
        self.assertEqual((result.wins, result.draws, result.losses),
                         (100, 0, 100))

    def test_bootstrap_ci_matches_pinned_values(self):
        # 既知入力・固定 seed に対する CI の回帰ピン(統計経路の意図せぬ変化を検出)。
        pair_results = [1.0, 1.0, 1.0, 0.5, 0.0]
        low, high = cluster_bootstrap(pair_results, samples=5000, seed=42)
        self.assertAlmostEqual(low, 0.30, places=2)
        self.assertAlmostEqual(high, 1.00, places=2)

    def test_all_wins_gives_degenerate_ci(self):
        low, high = cluster_bootstrap([1.0] * 50, samples=1000, seed=7)
        self.assertEqual((low, high), (1.0, 1.0))

    def test_summarize_counts_draws_as_half(self):
        # 引き分け(0.5)を 0.5 勝として勝/分/敗・点推定・ペア勝率へ反映する経路。
        pair_games = [(1.0, 0.5), (0.0, 0.0), (1.0, 1.0)]
        wins, draws, losses, pair_results, point = summarize_pairs(pair_games)
        self.assertEqual((wins, draws, losses), (3, 1, 2))
        self.assertEqual(pair_results, (0.75, 0.0, 1.0))
        self.assertAlmostEqual(point, (0.75 + 0.0 + 1.0) / 3.0)


class HarnessWiringTests(unittest.TestCase):
    """ハーネスが席ごとに異なる方策 seed を渡し、ペアで環境 seed を共有すること。"""

    def test_pairs_use_distinct_policy_seed_per_seat_variant(self):
        captured = {"heuristic": [], "random": []}
        real_heuristic = harness_mod.HeuristicAgent
        real_random = harness_mod.RandomAgent

        class RecordingHeuristic(real_heuristic):
            def __init__(self, seed):
                captured["heuristic"].append(seed)
                super(RecordingHeuristic, self).__init__(seed)

        class RecordingRandom(real_random):
            def __init__(self, seed):
                captured["random"].append(seed)
                super(RecordingRandom, self).__init__(seed)

        harness_mod.HeuristicAgent = RecordingHeuristic
        harness_mod.RandomAgent = RecordingRandom
        try:
            result = evaluate(pairs=1, group="train", start_index=500,
                              bootstrap_samples=10, bootstrap_seed=1)
        finally:
            harness_mod.HeuristicAgent = real_heuristic
            harness_mod.RandomAgent = real_random

        # 1ペア = 2局。評価対象(Heuristic)は席替えで別 seed を受け取る。
        self.assertEqual(len(captured["heuristic"]), 2)
        self.assertNotEqual(captured["heuristic"][0], captured["heuristic"][1])
        # 期待値は seat_variant を反映した derive_seed と一致する。
        root = eval_seed("train", 500)
        expected = [derive_seed(root, "policy", 500, 0),
                    derive_seed(root, "policy", 500, 1)]
        self.assertEqual(captured["heuristic"], expected)
        # 対戦相手 RandomAgent の seed も席ごとに配線されている。
        self.assertEqual(captured["random"], expected)
        # 対応表の policy_seed / opponent_seed も seat_variant 由来値と一致する。
        for rec in result.game_records:
            seat_seed = derive_seed(root, "policy", 500, rec.seat_variant)
            self.assertEqual(rec.policy_seed, seat_seed)
            self.assertEqual(rec.opponent_seed, seat_seed)
        # 対応表の env_seed はペアの2局で一致する。
        env_seeds = set(r.env_seed for r in result.game_records)
        self.assertEqual(len(env_seeds), 1)

    def test_game_records_capture_seed_seat_correspondence(self):
        result = evaluate(pairs=2, group="train", start_index=600,
                          bootstrap_samples=10, bootstrap_seed=1)
        self.assertEqual(len(result.game_records), 4)
        for rec in result.game_records:
            self.assertIn(rec.result, (0.0, 0.5, 1.0))
            self.assertEqual(rec.opponent_seat, 1 - rec.target_seat)
            self.assertEqual(rec.target_seat, rec.seat_variant)
            # 初期山札・初期手札が明示的に保存されている(集計値だけでない)。
            self.assertTrue(rec.initial_deck)
            self.assertEqual(len(rec.initial_hands), 2)
            self.assertTrue(all(rec.initial_hands))
            self.assertEqual(len(rec.cards_sha256), 64)
        # ペアの2局は同一配札(初期山札・初期手札が一致)。
        by_index = {}
        for rec in result.game_records:
            by_index.setdefault(rec.index, []).append(rec)
        for recs in by_index.values():
            self.assertEqual(recs[0].initial_deck, recs[1].initial_deck)
            self.assertEqual(recs[0].initial_hands, recs[1].initial_hands)


if __name__ == "__main__":
    unittest.main()
