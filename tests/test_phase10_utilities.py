"""Phase 10 D1/B1: rank utilities and byte-stable two-seat transitions."""
import hashlib
import itertools
import json
import random
import unittest
from unittest import mock

import helpers as H
from naeco.engine import game
from naeco.engine.state import Phase, clone, deserialize, serialize, state_hash
from naeco.train import reward


# Captured from the pre-C2 engine on Python 3.9. Includes every serialized state,
# RNG, action and StepResult field for the full episode, not just final scores.
LEGACY_EPISODE_SHA256 = {
    (0, 0): "c7c8b7a1f9a837777b195857ff9daeb25d872fb6457589234a9a02ccf96934dc",
    (0, 1): "08e66560c555e000851b02ecf0fa89da5de219e2f11eb3dec6ad1deab673e484",
    (1, 0): "80b20feccd70d3448bdb86814db175f64bad4b4c3cf14c5f9bbf2c3ef730c482",
    (1, 1): "5fc0d2af419eab59f39c9447af8cc5add72b3ccbb39b94a25403b33f3c3f4537",
    (7, 0): "fb838a8be9b65f4283eeddb084079f0f5a653a783017bec8bb6d41fd00c31ac6",
    (7, 1): "ae84e24f37e18f4cd8e72b5ba10803fb19e7d144d581b65ba5e233b8881a4651",
    (42, 0): "5b140fbea299dbed7668d79c6c9ebb2fcd202dc8174df07b910b782dffb3b948",
    (42, 1): "e8e62cd4b945dbbf23ba4eba5dd6545e2c837625b8d3c3e4442f5edcff2e613e",
    (123, 0): "d14670b93381df03d0b14ddf385ff61818d7edb0f1abd9b7c90bcb0fa48fcbe8",
    (123, 1): "0b63d2b411cb7dc885d4af2cd141307a644a08b1f8c1ace0345eccfca8aad860",
    (999, 0): "38fc7dcbe9503cad4f539b80374828d8476d1c76c612f9756c742e6c5853e7a2",
    (999, 1): "d7909fea9ea4e0d4ac15c03166cea236976a650cf52b5d122795bcc7ceb5b4de",
}


def _legacy_episode_digest(seed, start_player, verify=False):
    state = game.new_game(2, seed, H.CARDS, start_player=start_player)
    rng = random.Random(seed)
    digest = hashlib.sha256()
    digest.update(serialize(state))
    actions = []
    hashes = []
    for _ in range(6000):
        before = serialize(state)
        action = rng.choice(game.legal_actions(state, H.CARDS))
        result = game.step(state, action, H.CARDS)
        if verify:
            assert serialize(state) == before, "step mutated its input"
            assert game.step(clone(state), action, H.CARDS) == result
            assert serialize(deserialize(before, H.CARDS)) == before
            assert state_hash(state) == hashlib.sha256(before).hexdigest()
        actions.append(action)
        hashes.append(state_hash(result.state))
        digest.update(serialize(result.state))
        digest.update(json.dumps({
            "action": [action.kind.value, action.target_instance_id,
                       action.instance_id, action.goods_count],
            "terminal": result.terminal,
            "scores": result.scores,
            "ranking": result.ranking,
            "utilities": result.utilities,
            "turn_advanced": result.turn_advanced,
            "info": result.info,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        state = result.state
        if result.terminal:
            break
    else:
        raise AssertionError("episode did not terminate")
    if verify:
        replay = game.new_game(2, seed, H.CARDS, start_player=start_player)
        for action, expected_hash in zip(actions, hashes):
            replay = game.step(replay, action, H.CARDS).state
            assert state_hash(replay) == expected_hash, "replay diverged"
        assert serialize(replay) == serialize(state)
    return digest.hexdigest()


def _scored_state(scores, start_player):
    state = H.new(len(scores), start_player=start_player)
    state.phase = Phase.GAME_OVER
    for player, score in zip(state.players, scores):
        player.money = score
    return state


class MultiUtilitiesTests(unittest.TestCase):
    def test_all_rank_permutations(self):
        expected_by_rank = {3: (1.0, 0.0, -1.0),
                            4: (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)}
        for n in (3, 4):
            for ranks in itertools.permutations(range(n)):
                for start in range(n):
                    with self.subTest(n=n, ranks=ranks, start=start):
                        state = _scored_state(tuple(n - r for r in ranks), start)
                        self.assertEqual(game.ranking(state, H.CARDS), ranks)
                        utilities = game.terminal_utilities(state, H.CARDS)
                        self.assertIsInstance(utilities, tuple)
                        self.assertEqual(len(utilities), n)
                        for pid, rank in enumerate(ranks):
                            self.assertIsInstance(utilities[pid], float)
                            self.assertAlmostEqual(utilities[pid],
                                                   expected_by_rank[n][rank])
                        self.assertAlmostEqual(sum(utilities), 0.0)
                        self.assertEqual(reward.terminal_rewards(state, H.CARDS),
                                         utilities)

    def test_all_tie_patterns_and_start_seats(self):
        # Every ordered partition of seats into tied score groups is represented
        # by a vector over n distinct scores (including all-equal scores).
        for n in (2, 3, 4):
            for scores in itertools.product(range(n), repeat=n):
                for start in range(n):
                    with self.subTest(n=n, scores=scores, start=start):
                        state = _scored_state(scores, start)
                        # Resolve each score group in clockwise start-seat order,
                        # independently of the engine's sorting implementation.
                        order = []
                        clockwise = [(start + offset) % n for offset in range(n)]
                        for score in sorted(set(scores), reverse=True):
                            order.extend(pid for pid in clockwise
                                         if scores[pid] == score)
                        ranks = tuple(order.index(pid) for pid in range(n))
                        self.assertEqual(game.ranking(state, H.CARDS), ranks)
                        utilities = game.terminal_utilities(state, H.CARDS)
                        for pid, rank in enumerate(ranks):
                            self.assertEqual(utilities[pid], 1.0 - 2 * rank / (n - 1))
                        self.assertEqual(sum(r == 0 for r in ranks), 1)
                        self.assertAlmostEqual(sum(utilities), 0.0)

                        # Exercise the actual terminal step for the same score
                        # partition: only the last worker remains in round nine.
                        ending = H.set_round(H.new(n, start_player=start), 9)
                        last = (start + n - 1) % n
                        ending.current_player = last
                        for pid, player in enumerate(ending.players):
                            player.money = scores[pid] + 10  # wage: 2 * $5
                            player.workers_waiting = 0
                            player.workers_training = 2
                        ending.players[last].workers_waiting = 1
                        ending.players[last].workers_training = 1
                        H.check(ending)
                        result = game.step(ending, next(
                            a for a in game.legal_actions(ending, H.CARDS)
                            if a.target_instance_id == H.wp("base:mine")), H.CARDS)
                        self.assertTrue(result.terminal)
                        self.assertEqual(result.scores, scores)
                        self.assertEqual(result.ranking, ranks)
                        self.assertEqual(result.utilities, utilities)
                        H.check(result.state)

    def test_explicit_ranking_is_used_without_recomputing(self):
        for n in (2, 3, 4):
            state = H.new(n)
            ranks = tuple(reversed(range(n)))
            with mock.patch.object(game, "ranking", side_effect=AssertionError):
                utilities = game.terminal_utilities(state, H.CARDS, ranks)
            self.assertEqual(utilities,
                             tuple(1.0 - 2 * r / (n - 1) for r in ranks))

    def test_real_episodes_have_terminal_utilities_only(self):
        for n in (2, 3, 4):
            for seed in range(8):
                state = game.new_game(n, seed, H.CARDS)
                rng = random.Random(seed)
                for _ in range(6000):
                    before = serialize(state)
                    action = rng.choice(game.legal_actions(state, H.CARDS))
                    result = game.step(state, action, H.CARDS)
                    self.assertEqual(serialize(state), before)
                    state = result.state
                    if not result.terminal:
                        self.assertIsNone(result.scores)
                        self.assertIsNone(result.ranking)
                        self.assertIsNone(result.utilities)
                        continue
                    self.assertEqual(result.scores, game.final_scores(state, H.CARDS))
                    self.assertEqual(result.ranking, game.ranking(state, H.CARDS))
                    self.assertEqual(result.utilities,
                                     game.terminal_utilities(state, H.CARDS))
                    assert result.utilities is not None
                    self.assertEqual(len(result.utilities), n)
                    self.assertAlmostEqual(sum(result.utilities), 0.0)
                    H.check(state)
                    break
                else:
                    self.fail("episode did not terminate")

    def test_two_player_full_episode_golden_clone_replay_hash(self):
        self.assertEqual(len(LEGACY_EPISODE_SHA256), 12)
        for (seed, start), expected in LEGACY_EPISODE_SHA256.items():
            with self.subTest(seed=seed, start=start):
                self.assertEqual(_legacy_episode_digest(seed, start, verify=True),
                                 expected)


if __name__ == "__main__":
    unittest.main()
