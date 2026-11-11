"""clone / serialize / state_hash / replay の決定論(設計書 §2, PLAN.md Phase 1)。"""
import random
import unittest

import helpers as H
from naeco.engine import game
from naeco.engine.state import (
    RULES_VERSION, clone, decode_rng, derive_seed, encode_rng, eval_seed,
    serialize, deserialize, state_hash, Phase, RNG_BLOB_LEN,
)


def random_playout(num_players, seed, limit=5000):
    """ランダム方策で1局回し、(最終状態, 行動列) を返す。"""
    rng = random.Random(seed ^ 0x5EED)
    s = H.new(num_players, seed)
    actions = []
    while s.phase != Phase.GAME_OVER:
        la = game.legal_actions(s, H.CARDS)
        assert la, "no legal actions"
        a = rng.choice(la)
        actions.append(a)
        s = game.step(s, a, H.CARDS).state
        assert len(actions) < limit
    return s, actions


class RngEncoding(unittest.TestCase):
    """§2.5: pickle を正準形にしない。"""

    def test_roundtrip_and_reencode_stability(self):
        r = random.Random(12345)
        for _ in range(50):
            r.random()
        blob = encode_rng(r)
        self.assertEqual(len(blob), RNG_BLOB_LEN)
        r2 = decode_rng(blob)
        self.assertEqual(encode_rng(r2), blob)
        self.assertEqual([r.random() for _ in range(5)],
                         [r2.random() for _ in range(5)])

    def test_index_is_preserved(self):
        """index は第3要素ではなく internal の末尾にある(設計書の実測メモ)。"""
        r = random.Random(7)
        r.random()                                  # index を 624 でない値にする
        self.assertEqual(decode_rng(encode_rng(r)).getstate(), r.getstate())

    def test_gauss_state_is_rejected(self):
        r = random.Random(7)
        r.gauss(0, 1)                               # gauss_next が float になる
        with self.assertRaises(ValueError):
            encode_rng(r)

    def test_bad_blob_length_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_rng(b"\x00" * 10)


class SeedDerivation(unittest.TestCase):
    """§2.5 / §10.1。"""

    def test_env_stream_ignores_seat_variant(self):
        a = derive_seed(99, "env", 3, seat_variant=0)
        b = derive_seed(99, "env", 3, seat_variant=1)
        self.assertEqual(a, b)

    def test_other_streams_honour_seat_variant(self):
        for kind in ("sampler", "search", "policy"):
            self.assertNotEqual(derive_seed(99, kind, 3, 0),
                                derive_seed(99, kind, 3, 1))

    def test_paired_games_share_the_identical_initial_deal(self):
        """★ ペアの2局の初期 deck が完全一致すること(§2.5)。"""
        root = eval_seed("dev", 17)
        env = derive_seed(root, "env", 5)
        s0 = H.new(2, env)
        s1 = H.new(2, derive_seed(root, "env", 5, seat_variant=1))
        self.assertEqual(env, derive_seed(root, "env", 5, seat_variant=1))
        self.assertEqual(s0.deck, s1.deck)
        self.assertEqual([p.hand_buildings for p in s0.players],
                         [p.hand_buildings for p in s1.players])

    def test_eval_seed_groups_are_separated_and_typos_rejected(self):
        self.assertNotEqual(eval_seed("train", 0), eval_seed("dev", 0))
        self.assertNotEqual(eval_seed("dev", 0), eval_seed("accept", 0))
        with self.assertRaises(ValueError):
            eval_seed("devv", 0)


class CloneIsolation(unittest.TestCase):
    """§2.2: cloneした状態を進めても元に影響しない(探索が本局を汚染しない土台)。"""

    def test_step_does_not_mutate_its_input(self):
        rng = random.Random(3)
        s = H.new(2, 4)
        for _ in range(120):
            if s.phase == Phase.GAME_OVER:
                break
            before = state_hash(s)
            snapshot = clone(s)
            a = rng.choice(H.legal(s))
            nxt = game.step(s, a, H.CARDS).state
            self.assertEqual(state_hash(s), before)       # 入力は不変
            self.assertEqual(state_hash(snapshot), before)
            s = nxt

    def test_clone_deep_copies_the_mutable_containers(self):
        s = H.new(2, 1)
        c = clone(s)
        c.workplaces[H.wp("base:mine")].occupants = (0,)
        c.slash_and_burn_load["x"] = 1
        c.players[0].money += 100
        self.assertEqual(s.workplaces[H.wp("base:mine")].occupants, ())
        self.assertEqual(s.slash_and_burn_load, {})
        self.assertNotEqual(s.players[0].money, c.players[0].money)

    def test_illegal_action_leaves_the_state_untouched(self):
        from naeco.engine.state import Action, ActionKind, LegalityError
        s = H.new(2, 1)
        before = state_hash(s)
        bogus = Action(ActionKind.SELL_BUILDING, instance_id=H.wp("base:mansion"))
        with self.assertRaises(LegalityError):
            game.step(s, bogus, H.CARDS)
        self.assertEqual(state_hash(s), before)


class Serialization(unittest.TestCase):
    """§2.2 / §2.3。"""

    def test_roundtrip_preserves_the_state_hash(self):
        rng = random.Random(11)
        s = H.new(3, 8)
        for _ in range(200):
            if s.phase == Phase.GAME_OVER:
                break
            blob = serialize(s)
            back = deserialize(blob, H.CARDS)
            self.assertEqual(state_hash(back), state_hash(s))
            self.assertEqual(serialize(back), blob)
            s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state

    def test_roundtrip_survives_a_pending_decision(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        s = H.step(s, H.build_action(H.wp("base:farm")))
        self.assertIsNotNone(s.pending)
        back = deserialize(serialize(s), H.CARDS)
        self.assertEqual(back.pending, s.pending)
        self.assertEqual(H.legal(back), H.legal(s))

    def test_version_mismatch_is_rejected(self):
        s = H.new()
        blob = serialize(s)
        # 版文字列は定数から取る(直書きすると版を上げたときに静かに no-op になる)。
        current = ('"rules_version":"%s"' % RULES_VERSION).encode("ascii")
        self.assertIn(current, blob)
        tampered = blob.replace(current, b'"rules_version":"0.9.9"')
        self.assertNotEqual(tampered, blob)
        with self.assertRaises(ValueError):
            deserialize(tampered, H.CARDS)

    def test_cards_hash_mismatch_is_rejected(self):
        s = H.new()
        s2 = clone(s)
        s2.cards_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            deserialize(serialize(s2), H.CARDS)

    def test_deserialize_checks_invariants(self):
        from naeco.engine.state import InvariantError
        s = H.new()
        s2 = clone(s)
        s2.players[0].hand_buildings = ()          # 建物3実体を消す
        with self.assertRaises(InvariantError):
            deserialize(serialize(s2), H.CARDS)

    def test_serialized_size_is_recorded(self):
        """§11 の暫定目標(≤4KB)に対する実測。未達なら表現を見直す判断材料。"""
        s, _ = random_playout(2, 5)
        size = len(serialize(s))
        self.assertLess(size, 16 * 1024)           # 明らかな暴走だけを弾く
        self.assertGreater(size, RNG_BLOB_LEN)


class Replay(unittest.TestCase):
    """同一seed・同一行動列 → 同一結果(PLAN.md Phase 1 の回帰テスト)。"""

    def test_same_seed_same_actions_give_the_same_state_hash(self):
        for n in (2, 3, 4):
            final, actions = random_playout(n, 21 + n)
            s = H.new(n, 21 + n)
            for a in actions:
                s = game.step(s, a, H.CARDS).state
            self.assertEqual(state_hash(s), state_hash(final))

    def test_replay_matches_step_by_step(self):
        final, actions = random_playout(2, 33)
        a_hashes = []
        s = H.new(2, 33)
        for a in actions:
            s = game.step(s, a, H.CARDS).state
            a_hashes.append(state_hash(s))
        b_hashes = []
        s = H.new(2, 33)
        for a in actions:
            s = game.step(s, a, H.CARDS).state
            b_hashes.append(state_hash(s))
        self.assertEqual(a_hashes, b_hashes)

    def test_new_game_is_a_pure_function_of_the_seed(self):
        self.assertEqual(state_hash(H.new(2, 777)), state_hash(H.new(2, 777)))
        self.assertNotEqual(state_hash(H.new(2, 777)), state_hash(H.new(2, 778)))

    def test_state_hash_is_environment_independent(self):
        """★ Windows ローカルと csi-enshu で同一値になることの回帰ピン。

        値が変わったら、状態表現か正準化規則を変えたということである。
        意図した変更なら本テストの期待値を更新し、その理由を記録すること。

        更新履歴:
          - 2026-07-21 `rules_version` 0.3.0 → 0.3.1(OPEN-027: 手札の消費財枚数を
            derived-public に区分)。`rules_version` は serialize ヘッダに載るため
            state_hash が変わる。**状態表現と正準化規則そのものは無変更**であり、
            変わったのはヘッダの版文字列だけである(sol P25-019)。
            旧値: d3884ab91858a533d300c489a362f368e127ac115868d032830e4eb9cbd1a6fe
        """
        self.assertEqual(
            state_hash(H.new(2, 0)),
            "4c4f7d37cc4d9054566c2d994417953635136bfb09785258d7c03b46924ed6e4",
        )


class LegalActionOrdering(unittest.TestCase):
    """§3.6: 返り値は seed 非依存の安定順。"""

    def test_ordering_is_stable_and_sorted(self):
        rng = random.Random(2)
        s = H.new(2, 6)
        for _ in range(150):
            if s.phase == Phase.GAME_OVER:
                break
            la = H.legal(s)
            self.assertEqual([a.sort_key() for a in la],
                             sorted(a.sort_key() for a in la))
            self.assertEqual(la, H.legal(s))
            s = game.step(s, rng.choice(la), H.CARDS).state


if __name__ == "__main__":
    unittest.main()
