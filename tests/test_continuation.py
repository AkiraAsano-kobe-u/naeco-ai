"""継続可能性(保証1b, §3.5)の網羅的検査(sol Phase1 #3 / 再確認 #3)。

ランダム対戦の確率的網羅では BUILD_COST の全 k、建物/消費財構成、ニコイチ全組合せを
保証できない。合法な k 集合を**独立に再計算して直接比較**し、ニコイチは**期待ペア集合と
完全一致**することを確認する。DFS は実装が返した行動しか辿れないため、合法 k の欠落は
集合比較でしか検出できない。
"""
import itertools
import unittest

import helpers as H
from naeco.engine import game, invariants
from naeco.engine.state import (
    ActionKind, DecisionKind, Phase, clone, state_hash,
)

# 建設対象(印刷コスト 1..5)。大工(discount=0)なので remaining = cost。
TARGET_BY_COST = {
    1: "base:farm",
    2: "base:factory",
    3: "base:plantation",
    4: "base:steel-mill",
    5: "base:automobile-factory",
}
# 捨て札原資に使う予備建物(実体は互いに異なる。捨てるだけなのでコストは無関係)。
SPARES = ["base:mansion", "base:headquarters", "base:railroad", "base:law-office",
          "base:agricultural-coop", "base:warehouse", "base:company-housing"]


def explore(s, max_nodes=20000):
    """pending 解決までの全行動列をDFSで列挙し、各中間ノードを検査する。"""
    seen = set()
    resolved = set()
    stack = [s]
    while stack:
        cur = stack.pop()
        key = state_hash(cur)
        if key in seen:
            continue
        seen.add(key)
        assert len(seen) < max_nodes, "state space too large"
        invariants.check(cur, H.CARDS, "always")
        if cur.pending is None:
            resolved.add(key)
            continue
        la = game.legal_actions(cur, H.CARDS)
        assert la, "continuation broken: empty legal_actions at %r" % (cur.pending,)
        for a in la:
            stack.append(game.step(cur, a, H.CARDS).state)
    return resolved


class BuildCostKEnumeration(unittest.TestCase):
    """BUILD_COST の合法 k 集合を独立計算と直接比較する(remaining=1..5 全域)。"""

    def test_legal_goods_counts_match_kmin_kmax_exactly(self):
        checked = 0
        for cost in range(1, 6):
            target = H.wp(TARGET_BY_COST[cost])
            for n_spare in range(0, cost + 2):
                for goods in range(0, cost + 2):
                    total = 1 + n_spare + goods           # 建設対象 + 予備建物 + 消費財
                    if total - 1 < cost:
                        continue                          # 原資不足では配置自体が非合法
                    spares = [H.wp(SPARES[i]) for i in range(n_spare)]
                    s = H.new()
                    H.set_hand(s, 0, [target] + spares, goods=goods)
                    s = H.place(s, 0, H.wp("base:carpenter"))
                    s = H.step(s, H.build_action(target))
                    self.assertEqual(s.pending.kind, DecisionKind.BUILD_COST)
                    self.assertEqual(s.pending.remaining, cost)

                    # 独立計算: 建物だけで払えない分は消費財必須。
                    eligible = n_spare                    # 予約(対象)を除く手札建物
                    k_min = max(0, cost - eligible)
                    k_max = min(goods, cost)
                    expected = set(range(k_min, k_max + 1))

                    la = H.legal(s)
                    self.assertTrue(all(a.kind == ActionKind.SELECT_DISCARD
                                        and a.instance_id is None for a in la),
                                    "first stage must be goods-count choices only")
                    got = set(a.goods_count for a in la)
                    self.assertEqual(got, expected,
                                     "cost=%d spare=%d goods=%d" % (cost, n_spare, goods))
                    # どの合法 k を選んでも解決に至る(行き詰まりなし)。
                    self.assertGreaterEqual(len(explore(s)), 1)
                    checked += 1
        self.assertGreater(checked, 30)        # 実際に多数の構成を検査したことのピン

    def test_building_selection_is_ascending_suffix(self):
        """R1b: k 決定後、建物は instance_id 昇順でのみ、かつ need 枚選べる範囲に絞る。"""
        s = H.new()
        # コスト4を建物のみで払う: 予約対象 + 予備4枚、消費財0。
        spares = [H.wp(SPARES[i]) for i in range(4)]
        H.set_hand(s, 0, [H.wp("base:steel-mill")] + spares, goods=0)
        s = H.place(s, 0, H.wp("base:carpenter"))
        s = H.step(s, H.build_action(H.wp("base:steel-mill")))
        s = H.step(s, H.discard_goods(0))                 # k=0(消費財なし)
        # remaining=4, 適格予備4枚 → 昇順で全4枚を順に取るしかない。
        cur = s
        picks = []
        while cur.pending is not None:
            la = H.legal(cur)
            self.assertTrue(all(a.instance_id is not None for a in la))
            ids = [a.instance_id for a in la]
            self.assertEqual(ids, sorted(ids))            # 昇順のみ
            picks.append(ids[0])
            cur = H.step(cur, la[0])
        self.assertEqual(sorted(picks), sorted(spares))


class TwoForOneEnumeration(unittest.TestCase):
    """ニコイチ: 1枚目/2枚目候補集合と到達ペア集合が期待と完全一致する。"""

    def _reachable_pairs(self, s):
        """全選択列を辿り、建設された2枚(ニコイチ本体を除く)の集合を集める。"""
        two_for_one = None
        for i in s.players[0].owned_buildings:
            if i.startswith("base:two-for-one"):
                two_for_one = i
        base_owned = set(s.players[0].owned_buildings)
        pairs = set()
        seen = set()
        stack = [s]
        while stack:
            cur = stack.pop()
            key = state_hash(cur)
            if key in seen:
                continue
            seen.add(key)
            if cur.pending is None:
                built = frozenset(set(cur.players[0].owned_buildings) - base_owned)
                if built:
                    pairs.add(built)
                continue
            for a in H.legal(cur):
                stack.append(H.step(cur, a))
        return pairs

    def test_same_cost_triple_reaches_all_three_pairs(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1),
                          H.wp("base:farm", 2)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        # 1枚目候補 = 「より大きいIDの同コスト相方あり」= farm#00, farm#01(farm#02は不可)
        self.assertEqual(sorted(a.instance_id for a in H.legal(s)),
                         [H.wp("base:farm"), H.wp("base:farm", 1)])
        farms = [H.wp("base:farm"), H.wp("base:farm", 1), H.wp("base:farm", 2)]
        expected = set(frozenset(p) for p in itertools.combinations(farms, 2))
        self.assertEqual(self._reachable_pairs(s), expected)

    def test_second_candidate_set_is_strictly_greater_ids(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1),
                          H.wp("base:farm", 2)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        for first in (H.wp("base:farm"), H.wp("base:farm", 1)):
            s2 = H.step(s, H.build_action(first))
            got = sorted(a.instance_id for a in H.legal(s2))
            expected = sorted(f for f in [H.wp("base:farm", 1), H.wp("base:farm", 2)]
                              if f > first)
            self.assertEqual(got, expected)

    def test_mixed_costs_reach_only_same_cost_pairs(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        # コスト1の農場2枚 + コスト2の工場2枚 + 消費財2。
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1),
                          H.wp("base:factory"), H.wp("base:factory", 1)], goods=2)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        firsts = sorted(a.instance_id for a in H.legal(s))
        self.assertEqual(firsts, [H.wp("base:factory"), H.wp("base:farm")])
        expected = {
            frozenset((H.wp("base:farm"), H.wp("base:farm", 1))),
            frozenset((H.wp("base:factory"), H.wp("base:factory", 1))),
        }
        self.assertEqual(self._reachable_pairs(s), expected)


class HandLimitContinuation(unittest.TestCase):
    """手札上限調整: 超過分を必ず捨て切れる。"""

    def test_hand_limit_discard_resolves_for_various_excess(self):
        for goods in range(6, 10):
            s = H.new()
            s.players[0].money = 30
            s.players[1].money = 30
            H.set_hand(s, 0, [H.wp("base:mansion")], goods=goods)
            s.phase = Phase.END
            s.end_phase_cursor = 0
            game._advance(s, H.CARDS)
            if s.pending is None:
                continue
            self.assertEqual(s.pending.kind, DecisionKind.HAND_LIMIT_DISCARD)
            self.assertGreaterEqual(len(explore(s)), 1)


if __name__ == "__main__":
    unittest.main()
