"""A3 自己対戦エピソード駆動と並列ロールアウトのテスト(Phase 3 設計書 §4.3)。

- 歯の狙い:
- 同一 seed → 同一軌跡 / worker=1 対 8 の完全一致 … seed 派生が論理座標のみ
  であることを検出する(worker index・PID 混入への歯)。
- ペアの席交換 … 既定の席割当が決定論的巡回で、N=2 では `seat_variant`
  0/1 が必ず交換になることを 200 index で全数検査する(§6.3)。
- 席ランダム化 … 学習経路 `randomize_seats=True` だけが
  `derive_seed(..., "seat", ...)` からの置換を使い、再現することを比較する。
- ペアの初期局面一致 … `"env"` / `"setup"` の席非依存を検出する。
- learner の席記録 … `learner_index` で指定した学習対象エージェントが、席置換
  (seat_variant / 席ランダム化)の後にどの席にいるかを `learner_seat` へ記録
  することを検出する。席を入れ替えても正しく learner を特定できる(設計書
  14.1)。
- LegalityError 破棄 … 握り潰し・カウンタ未設定への歯。
"""
import random
import unittest

import helpers as H
from naeco.ai.random_agent import RandomAgent
from naeco.engine.state import (
    Action, ActionKind, derive_seed, eval_seed,
)
from naeco.train import encoding as enc
from naeco.train import env as train_env
from naeco.train import reward


class _NamedRandomAgent(object):
    """決定論的ランダム方策。`name` 属性が識別子になる。"""

    def __init__(self, name, seed):
        self.name = name
        self._rng = random.Random(seed)

    def act(self, observation):
        return self._rng.choice(observation.legal_actions)


class _SpyAgent(object):
    """合法手数を決定順に記録する決定論的方策。"""

    def __init__(self, name, seed, recorder):
        self.name = name
        self._rng = random.Random(seed)
        self._recorder = recorder

    def act(self, observation):
        self._recorder.append(len(observation.legal_actions))
        return self._rng.choice(observation.legal_actions)


class _IllegalAgent(object):
    """配置フェーズでは常に非合法な SELL_BUILDING を返すエージェント。"""

    def act(self, observation):
        return Action(ActionKind.SELL_BUILDING)


class PlayEpisodeTests(unittest.TestCase):

    def test_same_seed_same_trajectory(self):
        """同一 (group, index, seat_variant) なら軌跡全体が完全一致。"""
        first = train_env.play_episode(
            "train", 2000, 0,
            [_NamedRandomAgent("A", 1), _NamedRandomAgent("B", 2)],
            cards=H.CARDS)
        second = train_env.play_episode(
            "train", 2000, 0,
            [_NamedRandomAgent("A", 1), _NamedRandomAgent("B", 2)],
            cards=H.CARDS)
        self.assertEqual(first, second)
        self.assertEqual(first.initial_state_hash, second.initial_state_hash)

    def test_seat_randomization_uses_seat_stream(self):
        """学習経路(randomize_seats=True)の席割当は seat ストリーム由来で再現する。"""
        root = eval_seed("train", 2002)
        for variant in (0, 1):
            order = list(range(2))
            random.Random(derive_seed(root, "seat", 2002, variant)).shuffle(order)
            expected = tuple("AB"[pid] for pid in order)
            first = train_env.play_episode(
                "train", 2002, variant,
                [_NamedRandomAgent("A", 3), _NamedRandomAgent("B", 4)],
                cards=H.CARDS, randomize_seats=True)
            second = train_env.play_episode(
                "train", 2002, variant,
                [_NamedRandomAgent("A", 3), _NamedRandomAgent("B", 4)],
                cards=H.CARDS, randomize_seats=True)
            self.assertEqual(first.seats, expected)
            self.assertEqual(second.seats, expected)

    def test_pair_seats_swapped_for_all_indices(self):
        """N=2 の 200 index 全数で seat_variant 0/1 が必ず席交換になる。

        1件でも同一席割当があれば失敗する。席割当はゲーム本体より先に
        決まるため、破棄局になった index でも検査は有効。
        """
        agents = [_NamedRandomAgent("A", 300), _NamedRandomAgent("B", 301)]
        for index in range(3000, 3200):
            variant0 = train_env.play_episode(
                "train", index, 0, agents, cards=H.CARDS,
                record_vectors=False)
            variant1 = train_env.play_episode(
                "train", index, 1, agents, cards=H.CARDS,
                record_vectors=False)
            self.assertEqual(
                variant0.seats, ("A", "B"),
                "index %d: seat_variant=0 must be identity" % index)
            self.assertEqual(
                variant1.seats, ("B", "A"),
                "index %d: seat_variant=1 must swap seats" % index)
            self.assertNotEqual(
                variant0.seats, variant1.seats,
                "index %d: pair seats must be exchanged" % index)

    def test_seat_variant_cycles_for_three_and_four_players(self):
        """N=3/4 は決定論的巡回で、variant 0..N-1 で各 agent が各席に1回。"""
        for num_players in (3, 4):
            index = 3200 + num_players
            root = eval_seed("train", index)
            orders = []
            for variant in range(num_players):
                order = train_env._seat_order(
                    root, index, variant, num_players)
                self.assertEqual(
                    sorted(order), list(range(num_players)),
                    "N=%d variant=%d must be a permutation"
                    % (num_players, variant))
                orders.append(order)
            for seat in range(num_players):
                occupants = [
                    orders[variant][seat]
                    for variant in range(num_players)
                ]
                self.assertEqual(
                    sorted(occupants), list(range(num_players)),
                    "N=%d seat %d must host each agent exactly once"
                    % (num_players, seat))
            for variant in range(1, num_players):
                expected = [
                    (seat + variant) % num_players
                    for seat in range(num_players)
                ]
                self.assertEqual(
                    orders[variant], expected,
                    "N=%d variant=%d must be a cyclic rotation"
                    % (num_players, variant))

    def test_randomize_seats_is_opt_in_and_reproducible(self):
        """ランダム化は opt-in で seat ストリーム由来・再現可能、既定は不使用。"""
        root = eval_seed("train", 3300)
        agents = [_NamedRandomAgent("A", 500), _NamedRandomAgent("B", 501)]
        for variant in (0, 1):
            order = list(range(2))
            random.Random(derive_seed(root, "seat", 3300, variant)).shuffle(order)
            expected_random = tuple("AB"[pid] for pid in order)
            expected_cycle = ("A", "B") if variant == 0 else ("B", "A")
            first = train_env.play_episode(
                "train", 3300, variant, agents, cards=H.CARDS,
                randomize_seats=True)
            second = train_env.play_episode(
                "train", 3300, variant, agents, cards=H.CARDS,
                randomize_seats=True)
            default = train_env.play_episode(
                "train", 3300, variant, agents, cards=H.CARDS)
            self.assertEqual(first.seats, expected_random)
            self.assertEqual(second.seats, expected_random)
            self.assertEqual(default.seats, expected_cycle)

    def test_pair_games_share_initial_deal(self):
        """ペア(seat_variant 0/1)は同一配札・同一開始席(initial_state_hash 一致)。"""
        agents = [_NamedRandomAgent("A", 5), _NamedRandomAgent("B", 6)]
        variant0 = train_env.play_episode(
            "train", 2003, 0, agents, cards=H.CARDS)
        variant1 = train_env.play_episode(
            "train", 2003, 1, agents, cards=H.CARDS)
        self.assertEqual(variant0.initial_state_hash,
                         variant1.initial_state_hash)

    def test_legality_error_discards_and_counts(self):
        """LegalityError は破棄局になり、カウンタと ERROR ログが立つ。"""
        before = train_env.discard_counts().get("LegalityError", 0)
        with self.assertLogs("naeco.train.env", level="ERROR") as captured:
            trajectory = train_env.play_episode(
                "train", 2004, 0,
                [_IllegalAgent(), _NamedRandomAgent("B", 7)],
                cards=H.CARDS)
        after = train_env.discard_counts().get("LegalityError", 0)
        self.assertTrue(trajectory.discarded)
        self.assertIn("illegal action", trajectory.discard_reason)
        self.assertEqual(after, before + 1)
        self.assertTrue(any(
            "discarded" in record.getMessage() for record in captured.records))

    def test_record_vectors_lengths_and_chosen_index(self):
        """obs_vec / action_vecs は合法手と同順で、chosen_index は有効範囲。"""
        recorder = []
        agents = [_SpyAgent("A", 8, recorder), _SpyAgent("B", 9, recorder)]
        trajectory = train_env.play_episode(
            "train", 2005, 0, agents, cards=H.CARDS)
        self.assertFalse(trajectory.discarded)
        self.assertEqual(len(trajectory.decisions), len(recorder))
        for decision, legal_count in zip(trajectory.decisions, recorder):
            self.assertEqual(len(decision.action_vecs), legal_count)
            self.assertEqual(len(decision.obs_vec), enc.obs_dim(2))
            self.assertTrue(0 <= decision.chosen_index < legal_count)

    def test_intermediate_decision_reward_zero(self):
        """turn_advanced=False の決定には shaped 有効時も報酬 0。"""
        config = reward.RewardConfig(shaping="score_diff", shaping_scale=1.0)
        saw_pending = False
        for seed in range(16):
            trajectory = train_env.play_episode(
                "train", 2010 + seed, 0,
                [_NamedRandomAgent("A", seed),
                 _NamedRandomAgent("B", seed + 50)],
                cards=H.CARDS, reward_config=config)
            self.assertFalse(trajectory.discarded)
            for decision in trajectory.decisions:
                if not decision.turn_advanced:
                    saw_pending = True
                    self.assertEqual(decision.reward, 0.0)
        self.assertTrue(saw_pending, "no pending decision sampled; test is vacuous")

    def test_record_vectors_false_is_lightweight(self):
        """record_vectors=False でも軽量経路が終端まで動く。"""
        trajectory = train_env.play_episode(
            "train", 2007, 0,
            [_NamedRandomAgent("A", 10), _NamedRandomAgent("B", 11)],
            cards=H.CARDS, record_vectors=False)
        self.assertFalse(trajectory.discarded)
        self.assertTrue(trajectory.decisions)
        for decision in trajectory.decisions:
            self.assertEqual(decision.obs_vec, ())
            self.assertEqual(decision.action_vecs, ())
            self.assertTrue(0 <= decision.chosen_index)
        self.assertEqual(len(trajectory.terminal_rewards), 2)
        self.assertTrue(all(u in (1.0, -1.0)
                            for u in trajectory.terminal_rewards))


class LearnerSeatTrackingTests(unittest.TestCase):
    """設計書 14.1: play_episode / rollout が学習対象席を記録する。

    歯の狙い:
    - learner_seat を seat_variant や席ランダム化の**前**の正準位置で記録
      (例: 常に 0)すると、席交換側で正しい席を指さず落ちる。
    - learner_index を無視して常に None にする実装へ戻すと落ちる。
    - learner_index の範囲検査を外すと落ちる。
    """

    @staticmethod
    def _agents():
        return [
            _NamedRandomAgent("learner", 1),
            _NamedRandomAgent("opponent", 2),
        ]

    def test_default_learner_seat_is_none(self):
        """learner_index を指定しないときは learner_seat が None(既定)。"""
        traj = train_env.play_episode(
            "train", 2300, 0, self._agents(), cards=H.CARDS)
        self.assertIsNone(traj.learner_seat)

    def test_learner_seat_tracked_for_both_seat_variants(self):
        """seat_variant 0/1(席交換)の両方で、learner(正準順の先頭)の席を
        置換後の実際の席として記録する。N=2 の決定論的巡回では variant 0 が
        席0、variant 1 が席1。"""
        v0 = train_env.play_episode(
            "train", 2301, 0, self._agents(), cards=H.CARDS,
            learner_index=0)
        v1 = train_env.play_episode(
            "train", 2301, 1, self._agents(), cards=H.CARDS,
            learner_index=0)
        self.assertEqual((v0.learner_seat, v1.learner_seat), (0, 1))
        self.assertEqual(v0.seats, ("learner", "opponent"))
        self.assertEqual(v1.seats, ("opponent", "learner"))
        for traj in (v0, v1):
            self.assertEqual(traj.seats[traj.learner_seat], "learner")
        self.assertNotEqual(v0.learner_seat, v1.learner_seat)

    def test_learner_seat_with_randomize_seats(self):
        """席ランダム化(randomize_seats=True)でも、seat ストリーム由来の置換の
        後に learner の席を正しく記録する。"""
        index = 2302
        root = eval_seed("train", index)
        for variant in (0, 1):
            order = list(range(2))
            random.Random(derive_seed(root, "seat", index, variant)).shuffle(
                order)
            traj = train_env.play_episode(
                "train", index, variant, self._agents(), cards=H.CARDS,
                randomize_seats=True, learner_index=0)
            self.assertEqual(
                traj.learner_seat, order.index(0),
                "seat_variant=%d: 置換後の席が正しくない" % variant)
            self.assertEqual(traj.seats[traj.learner_seat], "learner")

    def test_learner_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            train_env.play_episode(
                "train", 2303, 0, self._agents(), cards=H.CARDS,
                learner_index=2)
        with self.assertRaises(ValueError):
            train_env.play_episode(
                "train", 2303, 0, self._agents(), cards=H.CARDS,
                learner_index=-1)

    def test_rollout_forwards_learner_index(self):
        """rollout の learner_index が全 spec へ伝わり、worker 数に関わらず
        learner_seat が一致する。"""
        specs = [
            ("train", 2310 + i, i % 2, self._agents())
            for i in range(4)
        ]
        one = train_env.rollout(specs, workers=1, cards=H.CARDS,
                                learner_index=0)
        many = train_env.rollout(specs, workers=2, cards=H.CARDS,
                                 learner_index=0)
        self.assertEqual(len(one), 4)
        for traj, other in zip(one, many):
            self.assertFalse(traj.discarded)
            expected = 0 if traj.seat_variant == 0 else 1
            self.assertEqual(traj.learner_seat, expected)
            self.assertEqual(traj.seats[traj.learner_seat], "learner")
            self.assertEqual(traj.learner_seat, other.learner_seat)


class RolloutTests(unittest.TestCase):

    @staticmethod
    def _specs(count):
        return [
            ("train", 2100 + i, i % 2,
             [_NamedRandomAgent("A", 20 + i), _NamedRandomAgent("B", 120 + i)])
            for i in range(count)
        ]

    def _assert_trajectories_equal(self, first, second):
        """巨大な float ベクトルを difflib の diff へ流さずに全フィールド比較する。"""
        self.assertEqual(len(first), len(second))
        for i, (left, right) in enumerate(zip(first, second)):
            for field in ("group", "index", "seat_variant", "seats",
                          "terminal_rewards", "initial_state_hash",
                          "discarded", "discard_reason", "learner_seat"):
                self.assertTrue(
                    getattr(left, field) == getattr(right, field),
                    "trajectory %d field %s" % (i, field))
            self.assertEqual(len(left.decisions), len(right.decisions))
            for j, (dl, dr) in enumerate(zip(left.decisions, right.decisions)):
                self.assertEqual(dl.player_id, dr.player_id,
                                 "trajectory %d decision %d" % (i, j))
                self.assertTrue(
                    dl.obs_vec == dr.obs_vec,
                    "trajectory %d decision %d obs_vec" % (i, j))
                self.assertTrue(
                    dl.action_vecs == dr.action_vecs,
                    "trajectory %d decision %d action_vecs" % (i, j))
                self.assertEqual(dl.chosen_index, dr.chosen_index)
                self.assertEqual(dl.reward, dr.reward)
                self.assertEqual(dl.turn_advanced, dr.turn_advanced)

    def test_rollout_worker_count_independent(self):
        """worker=1 と worker=8 は全 Trajectory の全フィールドが完全一致。"""
        one = train_env.rollout(self._specs(10), workers=1, cards=H.CARDS)
        many = train_env.rollout(self._specs(10), workers=8, cards=H.CARDS)
        self._assert_trajectories_equal(one, many)
        self.assertTrue(all(not t.discarded for t in one))

    def test_rollout_aggregates_worker_discard_counters(self):
        """spawn ワーカーの破棄も親側のカウンタへ集約される。"""
        before = train_env.discard_counts().get("LegalityError", 0)
        results = train_env.rollout(
            [("train", 2200, 0,
              [_IllegalAgent(), RandomAgent(3)])],
            workers=2, cards=H.CARDS)
        after = train_env.discard_counts().get("LegalityError", 0)
        self.assertTrue(results[0].discarded)
        self.assertEqual(after, before + 1)


if __name__ == "__main__":
    unittest.main()
