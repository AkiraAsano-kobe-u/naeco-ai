"""不変条件が壊れた状態を確実に棄却することの検査(sol Phase1 #4)。

各検査が「検出すべき破損」を実際に検出することを、破損状態を注入して確認する。
検査に穴があると deserialize が不正状態を受理してしまう。
"""
import unittest

import helpers as H
from naeco.engine import invariants
from naeco.engine.state import InvariantError, WorkplaceState, clone


def assert_rejects(testcase, s):
    with testcase.assertRaises(InvariantError):
        invariants.check(s, H.CARDS, "always")


class OccupancyGuards(unittest.TestCase):

    def test_unlimited_workplace_rejects_out_of_range_player(self):
        """INV-005: 鉱山(無制限)でも占有者IDは範囲内でなければならない。"""
        s = H.new(2)
        s.workplaces[H.wp("base:mine")].occupants = (5,)
        assert_rejects(self, s)

    def test_exclusive_workplace_rejects_double_occupancy(self):
        s = H.new(2)
        s.workplaces[H.wp("base:quarry")].occupants = (0, 1)
        assert_rejects(self, s)


class WorkplaceMetadataGuards(unittest.TestCase):

    def test_rejects_key_instance_id_mismatch(self):
        s = H.new(2)
        w = s.workplaces[H.wp("base:mine")]
        s.workplaces[H.wp("base:mine")] = WorkplaceState(
            "base:quarry#00", w.logical_id, None, "any_player", w.occupancy_mode, ())
        assert_rejects(self, s)

    def test_rejects_wrong_occupancy_mode(self):
        s = H.new(2)
        w = s.workplaces[H.wp("base:quarry")]
        w.occupancy_mode = "unlimited"          # 採石場は exclusive のはず
        assert_rejects(self, s)

    def test_rejects_owner_out_of_range(self):
        s = H.new(2)
        H.own(s, 0, [H.wp("base:farm")])
        s.workplaces[H.wp("base:farm")].owner = 9
        assert_rejects(self, s)

    def test_rejects_owned_workplace_missing_from_workplaces(self):
        s = H.new(2)
        H.own(s, 0, [H.wp("base:farm")])
        del s.workplaces[H.wp("base:farm")]
        assert_rejects(self, s)


class CarpenterGuards(unittest.TestCase):

    def test_rejects_duplicate_carpenter_in_out_of_box(self):
        s = H.new(2)                              # 2人戦: 箱外に大工2実体
        dup = s.out_of_box[0]
        s.out_of_box = tuple(sorted((dup, dup)))  # 同一実体を重複させる
        assert_rejects(self, s)

    def test_rejects_non_carpenter_in_out_of_box(self):
        s = H.new(2)
        s.out_of_box = tuple(sorted(s.out_of_box + ("base:farm#00",)))
        assert_rejects(self, s)


class ContinuationGuard(unittest.TestCase):

    def test_rejects_pending_with_no_legal_actions(self):
        """保証1b: pending があるのに合法手が空、という状態を棄却する。"""
        from naeco.engine.state import DecisionKind, PendingDecision
        s = H.new(2)
        H.set_hand(s, 0, [])
        # 手札が空なのに BUILD_TARGET を要求 → 候補ゼロ = 継続不能。
        s.pending = PendingDecision(
            kind=DecisionKind.BUILD_TARGET, player_id=0,
            source_effect_id="EFF-carpenter-01", remaining=1)
        assert_rejects(self, s)


class BuildingConservationGuard(unittest.TestCase):

    def test_rejects_missing_building(self):
        s = H.new(2)
        s.players[0].hand_buildings = ()          # 建物3実体を消す
        assert_rejects(self, s)

    def test_rejects_duplicated_building(self):
        s = H.new(2)
        p = s.players[0]
        # 手札の1枚を山札にも重複させる(64実体の保存則違反)。
        dup = p.hand_buildings[0]
        s.deck = tuple(sorted(s.deck + (dup,)))
        assert_rejects(self, s)


if __name__ == "__main__":
    unittest.main()
