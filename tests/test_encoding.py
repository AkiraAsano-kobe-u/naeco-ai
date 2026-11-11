"""A1 観測・行動エンコーダのテスト(Phase 3 設計書 §5)。

「歯」の狙い: 席回転対称性は絶対席の混入を、player_id 非参照テストは
環境制御メタデータの参照を、実体交換テストは instance_id そのものの混入を
それぞれ検出する。エンコーダ実装には game 由来の経路を import しない。
"""
import dataclasses
import math
import os
import random
import tempfile
import unittest

import helpers as H
from naeco.engine import game, invariants, ops
from naeco.engine.observe import observe
from naeco.engine.state import (
    Action, ActionKind, DecisionKind, Phase, clone,
)
from naeco.train import encoding as enc


def _rotate_seats(s, k):
    """席 i の内容を席 (i+k)%n へ移した等価な局面を作る(§7.4)。"""
    n = s.num_players
    s2 = clone(s)
    shift = lambda pid: None if pid is None else (pid + k) % n

    players = [None] * n
    for i, p in enumerate(s.players):
        q = p.copy()
        q.player_id = (i + k) % n
        players[(i + k) % n] = q
    s2.players = tuple(players)

    s2.start_player = shift(s.start_player)
    s2.next_start_player = shift(s.next_start_player)
    s2.current_player = shift(s.current_player)

    for instance_id, w in s.workplaces.items():
        w2 = s2.workplaces[instance_id]
        w2.owner = shift(w.owner)
        w2.occupants = tuple(shift(x) for x in w.occupants)

    s2.public_knowledge = tuple(
        dataclasses.replace(e, holder=shift(e.holder))
        for e in s.public_knowledge
    )
    if s.pending is not None:
        s2.pending = dataclasses.replace(
            s.pending, player_id=shift(s.pending.player_id)
        )
    s2.draws_since_reshuffle = tuple(
        s.draws_since_reshuffle[(i - k) % n] for i in range(n)
    )
    return s2


class DimensionPin(unittest.TestCase):
    """現在値を pin し、意図しない次元変化を検出する(§4.1)。"""

    def test_current_dimensions_are_pinned(self):
        self.assertEqual(enc.obs_dim(2), 569)
        self.assertEqual(enc.obs_dim(3), 688)
        self.assertEqual(enc.obs_dim(4), 807)
        self.assertEqual(enc.action_dim(), 109)

    def test_dimensions_are_derived_from_tables(self):
        # 1席増えると、他家ブロック + 職場の他家分 + 知識のholder枠 + ドロー履歴
        # が増える。表から計算していなければこの関係は成り立たない。
        delta = (4 + len(enc.CARD_ID_ORDER) + 3 + 1 + len(enc.TAG_ORDER)
                 + len(enc.WORKPLACE_LOGICAL_ID_ORDER)
                 + len(enc.CARD_ID_ORDER) + 1)
        self.assertEqual(enc.obs_dim(3) - enc.obs_dim(2), delta)
        self.assertEqual(enc.obs_dim(4) - enc.obs_dim(3), delta)


class TableDerivation(unittest.TestCase):
    """ID 表が cards.json / enum から決定論的に導出されていること。"""

    def test_id_tables_match_cards_data_and_enum_order(self):
        self.assertEqual(len(enc.CARD_ID_ORDER), 39)
        self.assertEqual(enc.CARD_ID_ORDER, tuple(sorted(H.CARDS.defs)))
        self.assertEqual(len(enc.TAG_ORDER), 5)
        self.assertEqual(
            enc.TAG_ORDER,
            tuple(sorted({t for cd in H.CARDS.defs.values() for t in cd.tags})),
        )
        self.assertEqual(
            enc.ACTION_KIND_ORDER,
            tuple(sorted(k.value for k in ActionKind)),
        )
        self.assertEqual(
            enc.DECISION_KIND_ORDER,
            tuple(sorted(k.value for k in DecisionKind)),
        )
        self.assertEqual(
            enc.PHASE_ORDER,
            tuple(sorted(p.value for p in Phase)),
        )
        self.assertEqual(enc.CARDS_SHA256, H.CARDS.sha256)


class RandomGameDimensionContract(unittest.TestCase):
    """実局面の全決定点で len が obs_dim / action_dim に一致する。"""

    def test_length_contract_holds_at_every_decision_point(self):
        rng = random.Random(11)
        checked_actions = 0
        for n, games in ((2, 20), (3, 8), (4, 4)):
            for seed in range(games):
                s = H.new(n, 1000 + seed)
                steps = 0
                while s.phase != Phase.GAME_OVER:
                    for viewer in range(n):
                        obs = observe(s, viewer, H.CARDS)
                        self.assertEqual(
                            len(enc.encode_observation(obs)), enc.obs_dim(n)
                        )
                    la = H.legal(s)
                    self.assertTrue(la)
                    actor_obs = observe(s, game.actor(s), H.CARDS)
                    self.assertIsNotNone(actor_obs.legal_actions)
                    for a in la:
                        avec = enc.encode_action(actor_obs, a, H.CARDS)
                        self.assertEqual(len(avec), enc.action_dim())
                        checked_actions += 1
                    s = game.step(s, rng.choice(la), H.CARDS).state
                    steps += 1
                    self.assertLess(steps, 6000)
        self.assertGreater(checked_actions, 1000)


class FiniteBoundedOutput(unittest.TestCase):
    """出力に NaN / inf がなく、値域が概ね [-10, 10] に収まる。"""

    def _assert_finite_bounded(self, vec):
        for x in vec:
            self.assertTrue(math.isfinite(x), "non-finite feature %r" % x)
            self.assertGreaterEqual(x, -10.0)
            self.assertLessEqual(x, 10.0)

    def test_random_observations_and_actions_stay_finite_and_bounded(self):
        rng = random.Random(17)
        for n, games in ((2, 4), (3, 2), (4, 2)):
            for seed in range(games):
                s = H.new(n, 500 + seed)
                while s.phase != Phase.GAME_OVER:
                    for viewer in range(n):
                        self._assert_finite_bounded(
                            enc.encode_observation(observe(s, viewer, H.CARDS))
                        )
                    actor_obs = observe(s, game.actor(s), H.CARDS)
                    for a in H.legal(s):
                        self._assert_finite_bounded(
                            enc.encode_action(actor_obs, a, H.CARDS)
                        )
                    s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state


class SeatRotationSymmetry(unittest.TestCase):
    """★ 席回転対称性(§7.4)。絶対席が混入していたら落ちる。"""

    def _assert_rotation_invariant(self, s):
        n = s.num_players
        for k in range(1, n):
            s2 = _rotate_seats(s, k)
            invariants.check(s2, H.CARDS, "always")
            for viewer in range(n):
                a = observe(s, viewer, H.CARDS)
                b = observe(s2, (viewer + k) % n, H.CARDS)
                self.assertEqual(
                    enc.encode_observation(a),
                    enc.encode_observation(b),
                    "observation encoding is not seat-rotation invariant "
                    "(n=%d, k=%d, viewer=%d)" % (n, k, viewer),
                )
                if a.legal_actions is not None:
                    self.assertIsNotNone(b.legal_actions)
                    self.assertEqual(len(a.legal_actions), len(b.legal_actions))
                    for i in range(len(a.legal_actions)):
                        self.assertEqual(
                            enc.encode_action(a, a.legal_actions[i], H.CARDS),
                            enc.encode_action(
                                b, b.legal_actions[i], H.CARDS
                            ),
                        )

    def test_initial_positions(self):
        for n in (2, 3, 4):
            self._assert_rotation_invariant(H.new(n, n * 11))

    def test_played_positions(self):
        rng = random.Random(4)
        for n in (2, 3, 4):
            s = H.new(n, n * 7)
            for _ in range(25):
                if s.phase == Phase.GAME_OVER:
                    break
                self._assert_rotation_invariant(s)
                s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state

    def test_after_reshuffle_with_uneven_draws(self):
        s = H.new(3)
        H.set_hand(s, 0, [H.wp("base:cafe")], goods=1)
        H.set_hand(s, 1, [H.wp("base:orchard")], goods=1)
        H.set_hand(s, 2, [H.wp("base:farm")], goods=1)
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 1, H.wp("base:mine"))
        self.assertEqual(s.draws_since_reshuffle, (0, 1, 0))
        self._assert_rotation_invariant(s)


class PlayerIdIndependence(unittest.TestCase):
    """obs.player_id を差し替えても出力が変わらない(§8.2)。"""

    def test_replacing_player_id_does_not_change_output(self):
        rng = random.Random(3)
        for n in (2, 3, 4):
            s = H.new(n, 20 + n)
            for _ in range(12):
                for viewer in range(n):
                    obs = observe(s, viewer, H.CARDS)
                    base_obs = enc.encode_observation(obs)
                    alt_obs = dataclasses.replace(obs, player_id=99)
                    self.assertEqual(base_obs, enc.encode_observation(alt_obs))
                    if obs.legal_actions is not None:
                        alt_legal = dataclasses.replace(obs, player_id=99)
                        for a in obs.legal_actions:
                            self.assertEqual(
                                enc.encode_action(obs, a, H.CARDS),
                                enc.encode_action(alt_legal, a, H.CARDS),
                            )
                if s.phase == Phase.GAME_OVER:
                    break
                s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state

    def test_encoder_source_never_mentions_player_id(self):
        with open(os.path.abspath(enc.__file__), "r",
                  encoding="utf-8") as fh:
            source = fh.read()
        self.assertNotIn("player_id", source)


class InstanceExchangeability(unittest.TestCase):
    """同一 card_id の実体番号は特徴に影響しない(§5.2 禁止事項)。"""

    def test_observation_counts_only_card_id(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:farm", 0), H.wp("base:factory", 0)])
        H.set_hand(s, 0, [H.wp("base:farm", 1), H.wp("base:cafe", 0)],
                   goods=1)
        obs = observe(s, 0, H.CARDS)
        base = enc.encode_observation(obs)
        alt = dataclasses.replace(
            obs,
            my_owned=("base:farm#03", "base:factory#02"),
            my_hand_buildings=("base:farm#05", "base:cafe#01"),
        )
        self.assertEqual(base, enc.encode_observation(alt))

    def test_actions_with_same_card_id_encode_identically(self):
        s = H.new()
        obs = observe(s, 0, H.CARDS)
        a0 = Action(ActionKind.SELECT_BUILD, instance_id="base:farm#00")
        a1 = Action(ActionKind.SELECT_BUILD, instance_id="base:farm#01")
        self.assertEqual(
            enc.encode_action(obs, a0, H.CARDS),
            enc.encode_action(obs, a1, H.CARDS),
        )


class RuntimeWorkplaceValues(unittest.TestCase):
    """★ 職場特徴は obs.workplaces の実行時値(WorkplaceView)から作る(§13.1)。

    歯: base:orchard の cards.json 静的定義は owner_only だが、売却(TX-011)後は
    any_player かつ公共職場になる。静的定義(初期値)へ戻すとこのクラスのテストが落ちる。
    """

    @staticmethod
    def _workplace_block_start(n):
        """職場ブロックの開始位置(§5.2 のブロック構成を定数表から計算)。"""
        global_size = 6 + len(enc.PHASE_ORDER) + 2
        self_size = 2 + 2 * len(enc.CARD_ID_ORDER) + 3 + 1 + len(enc.TAG_ORDER)
        other_size = 4 + len(enc.CARD_ID_ORDER) + 3 + 1 + len(enc.TAG_ORDER)
        return (global_size + self_size + (n - 1) * other_size
                + len(enc.CARD_ID_ORDER))

    @staticmethod
    def _logical_pos(card_id):
        lid = H.CARDS.logical_id_of(H.wp(card_id))
        return enc.WORKPLACE_LOGICAL_ID_ORDER.index(lid)

    def _sold_orchard_state(self):
        """base:orchard#00 を所有→売却(TX-011)した2人局面。"""
        s = H.new(2)
        H.own(s, 0, [H.wp("base:orchard")])
        ops.sell_building(s, H.CARDS, 0, H.wp("base:orchard"))
        return s

    def test_encode_action_target_uses_runtime_values(self):
        """売却済み果樹園へ PLACE_WORKER: 公共・any_player・exclusive・存在 1。

        静的定義へ戻すと public=0 / usable_by=[1,0] になり落ちる(歯)。
        """
        obs = observe(self._sold_orchard_state(), 0, H.CARDS)
        vec = enc.encode_action(
            obs,
            Action(ActionKind.PLACE_WORKER,
                   target_instance_id=H.wp("base:orchard")),
            H.CARDS,
        )
        kind = len(enc.ACTION_KIND_ORDER)
        start = kind + len(enc.CARD_ID_ORDER)
        self.assertEqual(
            vec[kind + enc.CARD_ID_ORDER.index("base:orchard")], 1.0)
        self.assertEqual(vec[start], 1.0)                       # 公共
        self.assertEqual(vec[start + 1:start + 3], [0.0, 1.0])  # any_player
        self.assertEqual(vec[start + 3:start + 5], [1.0, 0.0])  # exclusive
        self.assertEqual(vec[start + 5], 1.0)                   # 存在フラグ

    def test_encode_action_round_face_is_public_from_runtime(self):
        """ラウンド表面(カード種別は round)も実行時は公共職場。

        静的定義の card_type 判定へ戻すと public=0 になり落ちる(歯)。
        """
        s = H.new(2)
        H.reveal_round(s, 1)
        round_iid = H.CARDS.instances_by_card["base:round-1"][0]
        obs = observe(s, 0, H.CARDS)
        self.assertIn(round_iid, [w.instance_id for w in obs.workplaces])
        vec = enc.encode_action(
            obs,
            Action(ActionKind.PLACE_WORKER, target_instance_id=round_iid),
            H.CARDS,
        )
        kind = len(enc.ACTION_KIND_ORDER)
        start = kind + len(enc.CARD_ID_ORDER)
        self.assertEqual(
            vec[kind + enc.CARD_ID_ORDER.index("base:round-1")], 1.0)
        self.assertEqual(vec[start], 1.0)                       # 公共(実行時)
        self.assertEqual(vec[start + 1:start + 3], [0.0, 1.0])  # any_player
        self.assertEqual(vec[start + 3:start + 5], [1.0, 0.0])  # exclusive
        self.assertEqual(vec[start + 5], 1.0)                   # 存在フラグ

    def test_encode_action_target_absent_uses_presence_flag(self):
        """target が obs.workplaces に無い(山札内の職場)場合は実行時値 0 + 存在 0。

        静的定義へフォールバックすると usable_by=[1,0] になり落ちる(歯)。
        """
        obs = observe(H.new(2), 0, H.CARDS)
        farm = H.wp("base:farm")
        self.assertNotIn(farm, [w.instance_id for w in obs.workplaces])
        vec = enc.encode_action(
            obs, Action(ActionKind.PLACE_WORKER, target_instance_id=farm),
            H.CARDS,
        )
        kind = len(enc.ACTION_KIND_ORDER)
        start = kind + len(enc.CARD_ID_ORDER)
        self.assertEqual(
            vec[kind + enc.CARD_ID_ORDER.index("base:farm")], 1.0)
        self.assertEqual(vec[start], 0.0)                       # 公共 0
        self.assertEqual(vec[start + 1:start + 3], [0.0, 0.0])  # usable_by 未知
        self.assertEqual(vec[start + 3:start + 5], [0.0, 0.0])  # 占有モード未知
        self.assertEqual(vec[start + 5], 0.0)                   # 存在フラグ 0
        # target_instance_id 自体が無い行動も存在フラグ 0
        vec_none = enc.encode_action(
            obs, Action(ActionKind.SELECT_BUILD, instance_id=farm), H.CARDS)
        self.assertEqual(vec_none[start + 5], 0.0)

    def test_observation_workplace_block_counts_runtime_usable_by(self):
        """職場ブロックが実行時 usable_by 別の実体数を logical_id 単位で集計する。

        歯: 売却後も静的定義(owner_only)で集計すると any_player 枠が 0 のまま
        になり落ちる。instance_id は特徴に含めず、同一 logical_id の実体を合算する。
        """
        s = H.new(2)
        H.own(s, 0, [H.wp("base:orchard")])
        H.own(s, 1, [H.wp("base:orchard", 1)])
        pos = (self._workplace_block_start(2)
               + self._logical_pos("base:orchard") * 6)  # stride = n + 4
        # 所有中: 自分の所有 1 / owner_only 2 / any_player 0
        vec = enc.encode_observation(observe(s, 0, H.CARDS))
        self.assertEqual(vec[pos + 3], 1.0 / 4.0)
        self.assertEqual(vec[pos + 4], 2.0 / 4.0)
        self.assertEqual(vec[pos + 5], 0.0)
        # 両方売却(TX-011): 自分の所有 0 / owner_only 0 / any_player 2
        ops.sell_building(s, H.CARDS, 0, H.wp("base:orchard"))
        ops.sell_building(s, H.CARDS, 1, H.wp("base:orchard", 1))
        vec2 = enc.encode_observation(observe(s, 0, H.CARDS))
        self.assertEqual(vec2[pos + 3], 0.0)
        self.assertEqual(vec2[pos + 4], 0.0)
        self.assertEqual(vec2[pos + 5], 2.0 / 4.0)


class AllKindsCoverage(unittest.TestCase):
    """全 DecisionKind × 全 ActionKind の局面で encode_action が落ちない。"""

    SYNTHETIC_ACTIONS = (
        Action(ActionKind.PLACE_WORKER, target_instance_id=H.wp("base:mine")),
        Action(ActionKind.SELECT_BUILD, instance_id=H.wp("base:farm")),
        Action(ActionKind.SELECT_DISCARD, instance_id=H.wp("base:farm")),
        Action(ActionKind.SELECT_DISCARD, goods_count=2),
        Action(ActionKind.SELECT_KEEP, instance_id=H.wp("base:farm")),
        Action(ActionKind.SELL_BUILDING, instance_id=H.wp("base:farm")),
    )

    @staticmethod
    def _enrich_opponent(s, opp):
        H.set_hand(s, opp, [H.wp("base:cafe"), H.wp("base:orchard")],
                   goods=2)

    def _pending_state_for(self, kind):
        if kind == DecisionKind.BUILD_TARGET:
            s = H.new()
            H.set_hand(s, 0, [H.wp("base:factory"), H.wp("base:mansion"),
                              H.wp("base:railroad")], goods=2)
            self._enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:carpenter"))
        if kind == DecisionKind.BUILD_COST:
            s = H.new()
            H.set_hand(s, 0, [H.wp("base:factory"), H.wp("base:mansion"),
                              H.wp("base:railroad")], goods=2)
            self._enrich_opponent(s, 1)
            s = H.place(s, 0, H.wp("base:carpenter"))
            return H.step(s, H.build_action(H.wp("base:factory")))
        if kind == DecisionKind.DISCARD_N:
            s = H.new()
            H.own(s, 0, [H.wp("base:factory")])
            H.set_hand(s, 0, [H.wp("base:mansion"), H.wp("base:railroad")],
                       goods=2)
            self._enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:factory"))
        if kind == DecisionKind.DESIGN_OFFICE_KEEP:
            s = H.new()
            H.own(s, 0, [H.wp("base:design-office")])
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=1)
            self._enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:design-office"))
        if kind == DecisionKind.HAND_LIMIT_DISCARD:
            s = H.new()
            s.players[0].money = 20
            s.players[1].money = 20
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=7)
            self._enrich_opponent(s, 1)
            s.phase = Phase.END
            s.end_phase_cursor = 0
            game._advance(s, H.CARDS)
            return s
        if kind == DecisionKind.INSOLVENCY_SELL:
            s = H.new()
            s.players[0].money = 0
            s.players[1].money = 20
            H.own(s, 0, [H.wp("base:farm"), H.wp("base:factory")])
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=1)
            self._enrich_opponent(s, 1)
            s.phase = Phase.END
            s.end_phase_cursor = 0
            game._advance(s, H.CARDS)
            return s
        raise AssertionError("unhandled kind %r" % kind)

    def test_every_decision_kind_and_action_kind_encodes(self):
        for kind in DecisionKind:
            s = self._pending_state_for(kind)
            self.assertIsNotNone(s.pending)
            self.assertEqual(s.pending.kind, kind)
            for viewer in range(s.num_players):
                obs = observe(s, viewer, H.CARDS)
                for a in self.SYNTHETIC_ACTIONS:
                    self.assertEqual(
                        len(enc.encode_action(obs, a, H.CARDS)),
                        enc.action_dim(),
                    )
            actor_obs = observe(s, s.pending.player_id, H.CARDS)
            self.assertTrue(actor_obs.legal_actions)
            for a in actor_obs.legal_actions:
                self.assertEqual(
                    len(enc.encode_action(actor_obs, a, H.CARDS)),
                    enc.action_dim(),
                )

    def test_place_phase_legal_actions_encode(self):
        s = H.new()
        obs = observe(s, game.actor(s), H.CARDS)
        self.assertTrue(obs.legal_actions)
        kinds = set(a.kind for a in obs.legal_actions)
        self.assertIn(ActionKind.PLACE_WORKER, kinds)
        for a in obs.legal_actions:
            self.assertEqual(
                len(enc.encode_action(obs, a, H.CARDS)), enc.action_dim()
            )


class EncoderHash(unittest.TestCase):
    """encoder_sha256() は実装・定数表が変わると変わる(§3.2)。"""

    def test_returns_two_distinct_hex_digests(self):
        obs_sha, action_sha = enc.encoder_sha256()
        for h in (obs_sha, action_sha):
            self.assertEqual(len(h), 64)
            int(h, 16)
        self.assertNotEqual(obs_sha, action_sha)

    def test_hash_changes_when_normalization_changes(self):
        before = enc.encoder_sha256()
        original = enc._NORMALIZATION
        try:
            enc._NORMALIZATION = dict(original, count=11.0)
            changed = enc.encoder_sha256()
        finally:
            enc._NORMALIZATION = original
        self.assertNotEqual(before[0], changed[0])
        self.assertNotEqual(before[1], changed[1])
        self.assertEqual(enc.encoder_sha256(), before)

    def test_hash_changes_when_source_changes(self):
        before = enc.encoder_sha256()
        with open(os.path.abspath(enc.__file__), "rb") as fh:
            source = fh.read()
        original_file = enc.__file__
        try:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "encoding.py")
                with open(path, "wb") as fh:
                    fh.write(b"# tooth-check: implementation change\n" + source)
                enc.__file__ = path
                changed = enc.encoder_sha256()
        finally:
            enc.__file__ = original_file
        self.assertNotEqual(before[0], changed[0])
        self.assertNotEqual(before[1], changed[1])
        self.assertEqual(enc.encoder_sha256(), before)


if __name__ == "__main__":
    unittest.main()
