"""情報境界と漏洩テスト(設計書 §6, §7.4)。

漏洩テストは「p にとって真に隠れている部分」だけを入れ替えた s' を作り、
observe(s,p) == observe(s',p) を要求する。内訳(建物/消費財)まで振り直すのが要点。

整合性判定とスクランブルは `tests/hidden.py` へ移設した(Phase 2.5)。
隠れ状態サンプラーの受入オラクルと同一の実装を使うためである。
"""
import dataclasses
import random
import unittest

import helpers as H
from hidden import hand_internals_differ, is_consistent_hidden, scramble_hidden
from naeco.engine import game, invariants
from naeco.engine.observe import observe, serialize_observation
from naeco.engine.state import DecisionKind, KnowledgeKind, Phase, clone


class BoundaryStructure(unittest.TestCase):

    def test_legal_actions_only_go_to_the_actor(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:factory")])
        H.set_hand(s, 0, [], goods=2)
        s = H.place(s, 0, H.wp("base:factory"))           # pending: actor=0
        self.assertIsNotNone(observe(s, 0, H.CARDS).legal_actions)
        self.assertIsNone(observe(s, 1, H.CARDS).legal_actions)

    def test_opponent_hand_is_only_a_count(self):
        s = H.new()
        obs = observe(s, 0, H.CARDS)
        self.assertEqual(obs.others[0].hand_size, 3)
        self.assertFalse(hasattr(obs.others[0], "hand_buildings"))

    def test_deck_is_only_a_size(self):
        s = H.new()
        obs = observe(s, 0, H.CARDS)
        self.assertIsInstance(obs.deck_size, int)

    def test_reserved_targets_hidden_from_opponents(self):
        """§7.2.1: ニコイチの建設対象と constraint_cost は本人のみ。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        s = H.step(s, H.build_action(H.wp("base:farm")))  # 1枚予約済み
        self.assertIsNotNone(s.pending.reserved_targets)
        me = observe(s, 0, H.CARDS).pending
        opp = observe(s, 1, H.CARDS).pending
        self.assertEqual(me.reserved_targets, (H.wp("base:farm"),))
        self.assertIsNone(opp.reserved_targets)
        self.assertIsNone(opp.constraint_cost)

    def test_build_cost_remaining_hidden_from_opponents(self):
        """★ §7.2.1: remaining は非公開の建設対象コストから導出されるので他家に None。"""
        s = H.new()
        H.set_hand(s, 0, [H.wp("base:factory")], goods=2)  # cost 2
        s = H.place(s, 0, H.wp("base:carpenter"))          # 初期公共職場
        s = H.step(s, H.build_action(H.wp("base:factory")))
        self.assertEqual(s.pending.kind, DecisionKind.BUILD_COST)
        self.assertEqual(observe(s, 0, H.CARDS).pending.remaining, 2)
        self.assertIsNone(observe(s, 1, H.CARDS).pending.remaining)

    def test_discard_n_remaining_is_public(self):
        """DISCARD_N の remaining は職場の params.discard(公開)由来なので公開。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:factory")])
        H.set_hand(s, 0, [], goods=2)
        s = H.place(s, 0, H.wp("base:factory"))
        self.assertEqual(observe(s, 1, H.CARDS).pending.remaining, 2)

    def test_workplace_and_start_player_are_relative(self):
        """§9.3: 絶対席を観測に含めない。"""
        s = H.new(4)
        s.start_player = 2
        obs = observe(s, 3, H.CARDS)
        self.assertEqual(obs.start_player_offset, (2 - 3) % 4)
        self.assertTrue(all(0 <= o.seat_offset < 4 for o in obs.others))


class LeakageTests(unittest.TestCase):
    """serialize(observe(s,p)) == serialize(observe(s',p)) を全観測者で要求する。

    スクランブル後の状態が**整合している**ことも毎回検査する(sol Phase1 #1)。
    非整合な状態同士の比較では漏洩を検出できないため、これが前提条件になる。
    """

    def _assert_no_leak(self, s, viewer, seed=0):
        """no-leak を検査し、**他家手札の内訳/実体が変化したか**(山札順を除く)を返す。"""
        s2 = scramble_hidden(s, viewer, seed)
        # 前提: s2 は履歴上あり得る整合状態(でなければ比較に意味がない)。
        self.assertTrue(is_consistent_hidden(s2, s, viewer),
                        "scrambled state is inconsistent")
        a = serialize_observation(observe(s, viewer, H.CARDS))
        b = serialize_observation(observe(s2, viewer, H.CARDS))
        self.assertEqual(a, b,
                         "observation leaks hidden state to player %d" % viewer)
        return hand_internals_differ(s, s2, viewer)

    def _assert_hand_scramble_reachable(self, s, viewer, seeds=16):
        """no-leak を保ちつつ、少なくとも1 seed で他家手札の内訳/実体が変わること。

        sol Phase1 再確認 #1: deck-only フォールバックに退行していないことを、
        各 (局面, viewer) で個別に保証する。
        """
        changed = False
        for seed in range(seeds):
            changed |= self._assert_no_leak(s, viewer, seed)
        self.assertTrue(
            changed,
            "no seed changed opponent hand internals for viewer %d "
            "(scramble may be silently deck-only)" % viewer)

    def test_no_leak_in_random_positions(self):
        rng = random.Random(0)
        changed_any = False
        for n in (2, 3, 4):
            s = H.new(n, n * 3)
            for _ in range(120):
                if s.phase == Phase.GAME_OVER:
                    break
                for viewer in range(n):
                    for seed in range(3):
                        changed_any |= self._assert_no_leak(s, viewer, seed)
                s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state
        # テストに歯があること: スクランブルが実際に**手札の内訳/実体**を変えた
        # 局面が存在する(山札順だけの変化ではない)。
        self.assertTrue(changed_any, "scramble never changed any hidden hand internals")

    def test_no_leak_mid_two_for_one(self):
        """constraint_cost の漏洩経路。必ずケースに含める(§7.4)。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        s = H.step(s, H.build_action(H.wp("base:farm")))
        for viewer in (0, 1):
            for seed in range(5):
                self._assert_no_leak(s, viewer, seed)

    def test_no_leak_mid_build_cost(self):
        """remaining の漏洩経路。建設対象のコストだけ変えた2状態で他家観測が一致。"""
        base = H.new()
        # コストの異なる建設対象で他家の観測が一致すること(コストが割れない)。
        H.set_hand(base, 0, [H.wp("base:factory")], goods=5)  # cost 2
        s_cheap = H.place(base, 0, H.wp("base:carpenter"))
        s_cheap = H.step(s_cheap, H.build_action(H.wp("base:factory")))

        base2 = H.new()
        H.set_hand(base2, 0, [H.wp("base:steel-mill")], goods=5)  # cost 4
        s_exp = H.place(base2, 0, H.wp("base:carpenter"))
        s_exp = H.step(s_exp, H.build_action(H.wp("base:steel-mill")))

        o1 = serialize_observation(observe(s_cheap, 1, H.CARDS))
        o2 = serialize_observation(observe(s_exp, 1, H.CARDS))
        self.assertEqual(o1, o2)

    def test_no_leak_after_reshuffle(self):
        """§7.4 もう一つの必須ケース: 山札再構成の直後。"""
        s = H.new()
        H.set_hand(s, 0, [])
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 0, H.wp("base:mine"))              # TX-007 が発火
        self.assertEqual(s.reshuffle_seq, 1)
        for viewer in range(2):
            for seed in range(5):
                self._assert_no_leak(s, viewer, seed)

    def test_no_leak_after_reshuffle_during_design_office(self):
        """§7.4 必須ケース: 山札再構成の直後に設計事務所で公開中(sol P25-003)。

        公開した実体は `temporary_public`(公開ゾーン)へ移るので、§7.3 の失効規則により
        `IN_DECK_SINCE` は**その時点で**消えていなければならない。残っていると、
        同じ実体が「公開ゾーンにある」と「山札か適格な手札にある」を同時に主張し、
        正当な局面が非整合になる(Phase 2.5 のサンプラーが C4 と C5 を同時に
        満たせなくなる)。
        """
        s = H.new()
        H.own(s, 0, [H.wp("base:design-office")])
        H.set_hand(s, 0, [])
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 0, H.wp("base:design-office"))     # TX-007 -> TX-022
        self.assertEqual(s.reshuffle_seq, 1)
        self.assertEqual(s.pending.kind, DecisionKind.DESIGN_OFFICE_KEEP)
        self.assertTrue(s.temporary_public)

        # 公開ゾーンへ移った実体の知識エントリは残っていない。
        since = set(e.instance_id for e in s.public_knowledge
                    if e.kind == KnowledgeKind.IN_DECK_SINCE)
        self.assertEqual(since & set(s.temporary_public), set(),
                         "revealed instances must not keep IN_DECK_SINCE")
        # 正当な局面が整合と判定されること(修正前はここが False だった)。
        self.assertTrue(is_consistent_hidden(s, s, 0))
        for viewer in range(2):
            for seed in range(5):
                self._assert_no_leak(s, viewer, seed)

    def test_public_knowledge_is_not_scrambled(self):
        """§7.4: public_knowledge は viewer が知ってよいので入れ替えない。"""
        s = H.new()
        H.own(s, 1, [H.wp("base:design-office")])
        H.set_hand(s, 1, [])
        s = H.place(s, 1, H.wp("base:design-office"))
        opts = H.legal(s)
        s = H.step(s, opts[0])                            # 1 が公開札を1枚取得
        self.assertTrue(any(e.kind == KnowledgeKind.IN_HAND
                            for e in s.public_knowledge))
        # viewer=0 から見ても、その既知カードは s と s' で一致する。
        for seed in range(5):
            self._assert_no_leak(s, 0, seed)

    def _pending_state_for(self, kind):
        """各 DecisionKind の途中状態を構築して返す(全 kind を網羅するため)。

        両観測者から見て他家手札を非自明に入れ替えられるよう、actor・非actorとも
        混在した手札を持たせ、山札にも材料を残す(deck-only 退行を排除できる局面)。
        """
        def enrich_opponent(s, opp):
            H.set_hand(s, opp, [H.wp("base:cafe"), H.wp("base:orchard")], goods=2)

        if kind == DecisionKind.BUILD_TARGET:
            s = H.new()
            H.set_hand(s, 0, [H.wp("base:factory"), H.wp("base:mansion"),
                              H.wp("base:railroad")], goods=2)
            enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:carpenter"))
        if kind == DecisionKind.BUILD_COST:
            s = H.new()
            H.set_hand(s, 0, [H.wp("base:factory"), H.wp("base:mansion"),
                              H.wp("base:railroad")], goods=2)
            enrich_opponent(s, 1)
            s = H.place(s, 0, H.wp("base:carpenter"))
            return H.step(s, H.build_action(H.wp("base:factory")))  # factory 予約
        if kind == DecisionKind.DISCARD_N:
            s = H.new()
            H.own(s, 0, [H.wp("base:factory")])
            H.set_hand(s, 0, [H.wp("base:mansion"), H.wp("base:railroad")], goods=2)
            enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:factory"))
        if kind == DecisionKind.DESIGN_OFFICE_KEEP:
            s = H.new()
            H.own(s, 0, [H.wp("base:design-office")])
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=1)
            enrich_opponent(s, 1)
            return H.place(s, 0, H.wp("base:design-office"))
        if kind == DecisionKind.HAND_LIMIT_DISCARD:
            s = H.new()
            s.players[0].money = 20
            s.players[1].money = 20
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=7)
            enrich_opponent(s, 1)
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
            enrich_opponent(s, 1)
            s.phase = Phase.END
            s.end_phase_cursor = 0
            game._advance(s, H.CARDS)
            return s
        raise AssertionError("unhandled kind %r" % kind)

    def test_no_leak_for_every_decision_kind_and_viewer(self):
        """全 DecisionKind × 全観測者で漏洩がなく、かつ他家手札が実際に変わること(§7.4)。

        各 (kind, viewer) について no-leak を満たしつつ、少なくとも1 seed で
        他家手札の実体/配分が変化する(deck-only 退行では通らない)。
        """
        for kind in DecisionKind:
            s = self._pending_state_for(kind)
            self.assertIsNotNone(s.pending)
            self.assertEqual(s.pending.kind, kind)
            for viewer in range(s.num_players):
                self._assert_hand_scramble_reachable(s, viewer)

    def test_no_leak_three_player_build_cost_all_viewers(self):
        """3人局面でも全観測者で漏洩なし+他家手札が変化する(sol 再確認 #1)。"""
        s = H.new(3)
        H.set_hand(s, 0, [H.wp("base:factory"), H.wp("base:mansion"),
                          H.wp("base:railroad")], goods=2)
        H.set_hand(s, 1, [H.wp("base:cafe"), H.wp("base:orchard")], goods=2)
        H.set_hand(s, 2, [H.wp("base:steel-mill")], goods=3)
        s = H.place(s, 0, H.wp("base:carpenter"))
        s = H.step(s, H.build_action(H.wp("base:factory")))
        self.assertEqual(s.pending.kind, DecisionKind.BUILD_COST)
        for viewer in range(3):
            self._assert_hand_scramble_reachable(s, viewer)


class EncoderMetadataGuard(unittest.TestCase):
    """§8.2: encode 側が player_id(絶対席)に触れないことのピン。"""

    def test_observation_serialization_ignores_absolute_seat(self):
        s = H.new(4)
        a = serialize_observation(observe(s, 1, H.CARDS))
        b = serialize_observation(observe(s, 1, H.CARDS))
        self.assertEqual(a, b)
        self.assertNotIn(b'"player_id"', a)


def _rotate_seats(s, k):
    """席 i の内容を席 (i+k)%n へ移した等価な局面を作る。

    盤面としては同じゲームを席番号だけ付け替えたものなので、観測者も同じだけ
    ずらせば**正準 Observation は一致しなければならない**。
    """
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
    # end_phase_cursor は開始席からの相対位置なので回転で変わらない。

    for instance_id, w in s.workplaces.items():
        w2 = s2.workplaces[instance_id]
        w2.owner = shift(w.owner)
        w2.occupants = tuple(shift(x) for x in w.occupants)

    s2.public_knowledge = tuple(
        dataclasses.replace(e, holder=shift(e.holder)) for e in s.public_knowledge
    )
    if s.pending is not None:
        s2.pending = dataclasses.replace(s.pending, player_id=shift(s.pending.player_id))
    s2.draws_since_reshuffle = tuple(
        s.draws_since_reshuffle[(i - k) % n] for i in range(n)
    )
    return s2


class SeatRotationSymmetry(unittest.TestCase):
    """★ 席回転対称性(§7.4、sol P25-010)。

    同一 viewer どうしの入替え比較では、`draws_since_reshuffle` を**絶対席順**で
    持つ誤実装を検出できない(入替えの前後で同じ値になるため)。状態と席番号を
    同時に回転して正準 Observation を突き合わせる必要がある。
    """

    def _assert_rotation_invariant(self, s):
        n = s.num_players
        for k in range(1, n):
            s2 = _rotate_seats(s, k)
            invariants.check(s2, H.CARDS, "always")
            for viewer in range(n):
                a = serialize_observation(observe(s, viewer, H.CARDS))
                b = serialize_observation(observe(s2, (viewer + k) % n, H.CARDS))
                self.assertEqual(a, b,
                                 "observation is not seat-rotation invariant "
                                 "(n=%d, k=%d, viewer=%d)" % (n, k, viewer))

    def test_initial_positions(self):
        for n in (2, 3, 4):
            self._assert_rotation_invariant(H.new(n, n * 11))

    def test_played_positions(self):
        rng = random.Random(4)
        for n in (2, 3, 4):
            s = H.new(n, n * 7)
            for _ in range(40):
                if s.phase == Phase.GAME_OVER:
                    break
                self._assert_rotation_invariant(s)
                s = game.step(s, rng.choice(H.legal(s)), H.CARDS).state

    def test_after_reshuffle_with_uneven_draws(self):
        """引いた枚数が席ごとに違う局面(絶対席順の誤実装が最も出やすい)。"""
        s = H.new(3)
        H.set_hand(s, 0, [H.wp("base:cafe")], goods=1)
        H.set_hand(s, 1, [H.wp("base:orchard")], goods=1)
        H.set_hand(s, 2, [H.wp("base:farm")], goods=1)
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 1, H.wp("base:mine"))
        self.assertEqual(s.draws_since_reshuffle, (0, 1, 0))
        self._assert_rotation_invariant(s)


class PublicScalarsAddedIn030(unittest.TestCase):
    """OBS_VERSION 0.3.0 で追加した公開スカラーの具体値(sol P25-010)。"""

    def test_draws_since_reshuffle_is_observer_relative(self):
        s = H.new(3)
        H.set_hand(s, 0, [])
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 1, H.wp("base:mine"))          # 席1だけが1枚引く
        self.assertEqual(s.draws_since_reshuffle, (0, 1, 0))
        # index = seat_offset。席1から見れば自分が offset 0。
        self.assertEqual(observe(s, 0, H.CARDS).draws_since_reshuffle, (0, 1, 0))
        self.assertEqual(observe(s, 1, H.CARDS).draws_since_reshuffle, (1, 0, 0))
        self.assertEqual(observe(s, 2, H.CARDS).draws_since_reshuffle, (0, 0, 1))

    def test_reshuffle_seq_before_and_after(self):
        s = H.new()
        self.assertEqual(observe(s, 0, H.CARDS).reshuffle_seq, 0)
        H.set_hand(s, 0, [])
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        s = H.place(s, 0, H.wp("base:mine"))
        self.assertEqual(observe(s, 0, H.CARDS).reshuffle_seq, 1)

    def test_next_start_player_offset_tracks_the_quarry(self):
        s = H.new(3)
        self.assertEqual(observe(s, 0, H.CARDS).next_start_player_offset, 0)
        s = H.place(s, 2, H.wp("base:quarry"))       # TX-023: 席2が予約
        self.assertEqual(s.next_start_player, 2)
        for viewer in range(3):
            self.assertEqual(observe(s, viewer, H.CARDS).next_start_player_offset,
                             (2 - viewer) % 3)

    def test_end_phase_cursor_boundaries(self):
        s = H.new()
        self.assertEqual(observe(s, 0, H.CARDS).end_phase_cursor, 0)
        s.end_phase_cursor = s.num_players
        self.assertEqual(observe(s, 0, H.CARDS).end_phase_cursor, s.num_players)

    def test_opponent_goods_are_derived_public(self):
        """RULES.md OPEN-027: 消費財枚数は公開、建物枚数は合計との差で定まる。"""
        s = H.new()
        H.set_hand(s, 1, [H.wp("base:farm")], goods=4)
        opp = observe(s, 0, H.CARDS).others[0]
        self.assertEqual(opp.hand_goods, 4)
        self.assertEqual(opp.hand_size, 5)
        self.assertEqual(opp.hand_buildings_count, 1)

    def test_cards_identity_is_exposed(self):
        s = H.new()
        obs = observe(s, 0, H.CARDS)
        self.assertEqual(obs.cards_sha256, s.cards_sha256)
        self.assertEqual(obs.cards_data_version, s.cards_data_version)


if __name__ == "__main__":
    unittest.main()
