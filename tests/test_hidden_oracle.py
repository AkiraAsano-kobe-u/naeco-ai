"""`tests/hidden.py` の整合性オラクル自体のテスト(設計書 §7.4 / §7.5)。

このオラクルは漏洩テストの前提条件であり、Phase 2.5 の隠れ状態サンプラーの
受入オラクル(C1〜C6)でもある。**オラクルに歯があること**を独立に固定する。
甘いオラクルは、サンプラーが履歴上あり得ない局面を返しても検出できない。

sol P25-018: 「制約に歯がある」ことを示すには、**その制約でだけ落ちる**ことを
示さなければならない。別の条件(C1 の観測不一致など)で先に落ちているテストは、
制約が無くても通ってしまうため証明にならない。そこで `failing_condition` を使い
**どの条件で落ちたか**を突き合わせる。
"""
import unittest

import helpers as H
import hidden
from naeco.engine import ops
from naeco.engine.state import (
    ActionKind, KnowledgeEntry, KnowledgeKind, clone,
)


def _reshuffled_two_player():
    """山札を空にして鉱山で1枚引く → TX-007 が発火し席0だけが1枚引いた状態。"""
    s = H.new()
    H.set_hand(s, 0, [])
    s.building_discard = tuple(sorted(s.deck))
    s.deck = ()
    return H.place(s, 0, H.wp("base:mine"))


def _reshuffled_three_player():
    """3人戦。再構成後に席1・席2がそれぞれ1枚だけ引いた状態を作る。

    鉱山は occupancy_mode="unlimited" なので複数プレイヤーが同一ラウンドに置ける。
    """
    s = H.new(3)
    # 各家に再構成前からの手札(= 非 IN_DECK_SINCE)を持たせる。
    H.set_hand(s, 0, [H.wp("base:cafe")], goods=1)
    H.set_hand(s, 1, [H.wp("base:orchard")], goods=1)
    H.set_hand(s, 2, [H.wp("base:farm")], goods=1)
    s.building_discard = tuple(sorted(s.deck))
    s.deck = ()
    s = H.place(s, 1, H.wp("base:mine"))      # TX-007 発火 + 席1が1枚
    s = H.place(s, 2, H.wp("base:mine"))      # 席2が1枚
    return s


def _reshuffled_two_player_double_draw():
    """席0が再構成後に2枚引き、うち1枚を捨て札(公開ゾーン)へ出した状態。

    `draws_since_reshuffle[0] == 2` かつ手札の `IN_DECK_SINCE` は1枚になるので、
    C4b に1枚ぶんの余裕がある。C4c だけを落とすケースを作るために使う。
    捨て札への移動は公開ゾーンへの移動なので、知識エントリは失効する(§7.3)。
    """
    s = H.new()
    H.set_hand(s, 0, [H.wp("base:cafe")], goods=1)
    s.building_discard = tuple(sorted(s.deck))
    s.deck = ()
    s = H.place(s, 0, H.wp("base:mine"))      # TX-007 発火 + 1枚
    s = H.place(s, 0, H.wp("base:mine"))      # 鉱山は占有無制限なのでもう1枚
    since = hidden.deck_since_instances(s)
    drawn = sorted(i for i in s.players[0].hand_buildings if i in since)
    assert len(drawn) == 2, drawn
    ops.discard_building_from_hand(s, H.CARDS, 0, drawn[0], ActionKind.SELECT_DISCARD)
    return s


class RealStatesAreConsistent(unittest.TestCase):

    def test_two_player_reshuffled(self):
        s = _reshuffled_two_player()
        self.assertEqual(s.reshuffle_seq, 1)
        self.assertIsNone(hidden.failing_condition(s, s, 0))

    def test_three_player_reshuffled(self):
        s = _reshuffled_three_player()
        self.assertEqual(s.reshuffle_seq, 1)
        self.assertEqual(s.draws_since_reshuffle, (0, 1, 1))
        for viewer in range(3):
            self.assertIsNone(hidden.failing_condition(s, s, viewer))


class DrawCapacityConstraint(unittest.TestCase):
    """C4b: `#(IN_DECK_SINCE ∩ hand_j) <= draws_since_reshuffle[j]`(sol P25-002)。"""

    def test_violation_is_caught_by_c4b_and_not_by_an_earlier_condition(self):
        """席1と席2の手札を**枚数を変えずに**入れ替え、C4b だけで落とす。

        席1は1枚しか引いていないのに IN_DECK_SINCE を2枚持つ状態になる。
        手札枚数・消費財枚数・山札は不変なので C1 は通り、移した実体は
        「引いた家の手札」にあるので C4a も通る。
        """
        s = _reshuffled_three_player()
        since = hidden.deck_since_instances(s)

        p1, p2 = s.players[1], s.players[2]
        p1_since = [i for i in p1.hand_buildings if i in since]
        p1_other = [i for i in p1.hand_buildings if i not in since]
        p2_since = [i for i in p2.hand_buildings if i in since]
        self.assertEqual(len(p1_since), 1, "seat 1 drew exactly one card")
        self.assertTrue(p1_other, "seat 1 must hold a pre-reshuffle building")
        self.assertEqual(len(p2_since), 1, "seat 2 drew exactly one card")

        give, take = p1_other[0], p2_since[0]
        bad = clone(s)
        bad.players[1].hand_buildings = tuple(sorted(
            [i for i in p1.hand_buildings if i != give] + [take]))
        bad.players[2].hand_buildings = tuple(sorted(
            [i for i in p2.hand_buildings if i != take] + [give]))

        # 枚数は完全に保存されている(C1 が先に落ちないことの前提)。
        self.assertEqual(len(bad.players[1].hand_buildings), len(p1.hand_buildings))
        self.assertEqual(len(bad.players[2].hand_buildings), len(p2.hand_buildings))
        self.assertEqual(bad.players[1].hand_goods, p1.hand_goods)

        # 個別条件: C4a は通り、C4b だけが落ちる。
        self.assertTrue(hidden.check_candidate_zones(bad))
        self.assertFalse(hidden.check_draw_capacity(bad))
        # 端から端まで: 最初に落ちる条件が C4b であること。
        self.assertEqual(hidden.failing_condition(bad, s, 0), "C4b draw capacity")

    def test_scramble_respects_capacity(self):
        s = _reshuffled_three_player()
        since = hidden.deck_since_instances(s)
        for viewer in range(3):
            for seed in range(12):
                s2 = hidden.scramble_hidden(s, viewer, seed)
                for j in range(s2.num_players):
                    held = sum(1 for i in s2.players[j].hand_buildings if i in since)
                    self.assertLessEqual(held, s2.draws_since_reshuffle[j])


class DeckSubsetConstraint(unittest.TestCase):
    """C4c: 再構成後の山札は IN_DECK_SINCE 実体だけから成る(sol P25-017)。

    `ops.reshuffle_deck` は捨て札全体に印を付け、以後カードが山札へ入る経路は
    次の再構成だけである。容量制約(C4b)は必要条件にすぎない。
    """

    def test_real_states_satisfy_it(self):
        for s in (_reshuffled_two_player(), _reshuffled_three_player()):
            self.assertTrue(hidden.check_deck_is_reshuffled_subset(s))

    def test_violation_is_caught_by_c4c_and_not_by_an_earlier_condition(self):
        """統合オラクル経由で C4c だけが落ちること(sol P25-027)。

        単に山札の S と手札の非 S を交換すると、受け取る席の S 枚数が増えて
        **C4c より先に C4b で落ちる**。そこで「2枚引いたが S を1枚しか持たない」
        正当な基準状態を作り、C4b に余裕を残したうえで交換する。
        `check_*` を直接呼ぶだけのテストでは、`CONDITIONS` から C4c を外す退行を
        検出できないので、`failing_condition` で突き合わせる。
        """
        s = _reshuffled_two_player_double_draw()
        since = hidden.deck_since_instances(s)
        p0 = s.players[0]
        held = [i for i in p0.hand_buildings if i in since]
        outsider = [i for i in p0.hand_buildings if i not in since]
        self.assertEqual(s.draws_since_reshuffle[0], 2)
        self.assertEqual(len(held), 1, "one drawn card was discarded to the public zone")
        self.assertTrue(outsider, "seat 0 must hold a pre-reshuffle building")

        bad = clone(s)
        give, take = outsider[0], bad.deck[0]
        bad.deck = (give,) + bad.deck[1:]                 # 山札に非 S が入る
        bad.players[0].hand_buildings = tuple(sorted(
            [i for i in p0.hand_buildings if i != give] + [take]))

        # C4b には余裕がある(2枚引いて S は 2枚 <= 2)。
        self.assertTrue(hidden.check_draw_capacity(bad))
        self.assertTrue(hidden.check_candidate_zones(bad))
        self.assertFalse(hidden.check_deck_is_reshuffled_subset(bad))
        # 観測者は席1(他家)にする。席0自身から見ると自分の手札内容が見えるので
        # C1 が先に落ちてしまい、C4c の検査に到達しない。
        self.assertEqual(hidden.failing_condition(bad, s, 1), "C4c deck subset")

    def test_real_state_keeps_the_deck_within_the_reshuffled_set(self):
        self.assertTrue(
            hidden.check_deck_is_reshuffled_subset(_reshuffled_two_player_double_draw()))

    def test_scramble_keeps_deck_within_the_reshuffled_set(self):
        s = _reshuffled_three_player()
        since = hidden.deck_since_instances(s)
        for viewer in range(3):
            for seed in range(12):
                s2 = hidden.scramble_hidden(s, viewer, seed)
                self.assertTrue(all(i in since for i in s2.deck))


class KnowledgeShape(unittest.TestCase):

    def test_duplicate_entries_are_rejected(self):
        s = _reshuffled_two_player()
        bad = clone(s)
        bad.public_knowledge = s.public_knowledge + (s.public_knowledge[0],)
        self.assertFalse(hidden.check_knowledge_shape(bad))

    def test_stale_generation_is_rejected(self):
        s = _reshuffled_two_player()
        bad = clone(s)
        bad.public_knowledge = tuple(
            KnowledgeEntry(e.instance_id, e.kind, e.holder, 99, e.round_number, e.source)
            if e.kind == KnowledgeKind.IN_DECK_SINCE else e
            for e in s.public_knowledge)
        self.assertFalse(hidden.check_knowledge_shape(bad))


    def test_in_deck_since_before_any_reshuffle_is_rejected(self):
        """`reshuffle_seq == 0` に IN_DECK_SINCE は存在しえない(sol P25-027)。

        `ops.reshuffle_deck` が唯一の生成元であり、そこで seq を 1 以上へ進めるため。
        """
        s = H.new()
        self.assertEqual(s.reshuffle_seq, 0)
        self.assertTrue(hidden.check_knowledge_shape(s))
        bad = clone(s)
        bad.public_knowledge = (
            KnowledgeEntry(s.deck[0], KnowledgeKind.IN_DECK_SINCE, None, 0, 1, "TX-007"),
        )
        self.assertFalse(hidden.check_knowledge_shape(bad))

    def test_in_hand_entry_must_not_carry_a_generation(self):
        s = H.new()
        bad = clone(s)
        bad.public_knowledge = (
            KnowledgeEntry(s.players[0].hand_buildings[0], KnowledgeKind.IN_HAND,
                           0, 1, 1, "EFF-design-office-01"),
        )
        self.assertFalse(hidden.check_knowledge_shape(bad))


class PublicZonesAndCounts(unittest.TestCase):
    """C5 / C6 を観測一致とは独立に検査する(sol P25-018)。"""

    def test_public_zone_change_is_caught(self):
        s = _reshuffled_two_player()
        bad = clone(s)
        moved = bad.deck[0]
        bad.deck = bad.deck[1:]
        bad.building_discard = tuple(sorted(bad.building_discard + (moved,)))
        self.assertFalse(hidden.check_public_zones(bad, s))
        self.assertTrue(hidden.check_public_zones(clone(s), s))

    def test_goods_count_change_is_caught(self):
        s = _reshuffled_two_player()
        bad = clone(s)
        bad.players[1].hand_goods += 1
        self.assertFalse(hidden.check_hand_counts(bad, s))
        self.assertTrue(hidden.check_hand_counts(clone(s), s))


class TemporaryPublicKnowledge(unittest.TestCase):
    """公開ゾーンへ移った実体の知識は即座に失効する(§7.3、sol P25-003)。"""

    def _design_office_after_reshuffle(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:design-office")])
        H.set_hand(s, 0, [])
        s.building_discard = tuple(sorted(s.deck))
        s.deck = ()
        return H.place(s, 0, H.wp("base:design-office"))

    def test_revealed_instances_drop_their_knowledge(self):
        s = self._design_office_after_reshuffle()
        since = hidden.deck_since_instances(s)
        self.assertTrue(s.temporary_public)
        self.assertEqual(since & set(s.temporary_public), set())
        self.assertIsNone(hidden.failing_condition(s, s, 0))

    def test_stale_knowledge_would_be_inconsistent(self):
        """失効させ忘れた状態は非整合と判定される(オラクルの歯の確認)。"""
        s = self._design_office_after_reshuffle()
        stale = clone(s)
        stale.public_knowledge = s.public_knowledge + tuple(
            KnowledgeEntry(i, KnowledgeKind.IN_DECK_SINCE, None,
                           s.reshuffle_seq, s.round_number, "TX-007")
            for i in sorted(s.temporary_public))
        self.assertFalse(hidden.is_consistent_hidden(stale, stale, 0))


if __name__ == "__main__":
    unittest.main()
