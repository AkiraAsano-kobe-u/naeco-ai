"""カード1種につき1テスト以上(PLAN.md Phase 1 完了条件)。

建物24種 + 公共職場12種 + 消費財 + 未払い賃金 = 38種を網羅する。
テスト名の接頭辞はカードIDに対応させ、抜けを目視で確認できるようにする。
"""
import unittest

import helpers as H
from naeco.engine import game
from naeco.engine.state import Phase


class BuildingEffects(unittest.TestCase):
    """RULES.md 第11章 建物24種。"""

    # --- base:farm ---------------------------------------------------------
    def test_farm_gains_two_goods(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:farm")])
        s = H.place(s, 0, H.wp("base:farm"))
        self.assertEqual(s.players[0].hand_goods, 2)      # EFF-farm-01
        H.check(s)

    # --- base:slash-and-burn ----------------------------------------------
    def test_slash_and_burn_loads_then_collects_next_round(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:slash-and-burn")])
        s = H.place(s, 0, H.wp("base:slash-and-burn"))
        self.assertEqual(s.slash_and_burn_load[H.wp("base:slash-and-burn")], 5)
        self.assertEqual(s.players[0].hand_goods, 0)      # この時点では獲得しない
        H.check(s)

    def test_slash_and_burn_collect_moves_card_to_discard(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:slash-and-burn")])
        s.slash_and_burn_load[H.wp("base:slash-and-burn")] = 5
        goods_before = s.players[0].hand_goods
        game._do_round_start(s, H.CARDS)                  # TX-026 + TX-021
        self.assertEqual(s.players[0].hand_goods, goods_before + 5)
        self.assertIn(H.wp("base:slash-and-burn"), s.building_discard)
        self.assertNotIn(H.wp("base:slash-and-burn"), s.players[0].owned_buildings)
        self.assertEqual(s.slash_and_burn_load, {})
        H.check(s)

    # --- base:cafe ---------------------------------------------------------
    def test_cafe_takes_five_from_household(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:cafe")])
        s.household = 5
        money = s.players[0].money
        s = H.place(s, 0, H.wp("base:cafe"))
        self.assertEqual(s.household, 0)
        self.assertEqual(s.players[0].money, money + 5)
        H.check(s)

    def test_cafe_illegal_when_household_short(self):
        """RUL-HOUSEHOLD-001 / OPEN-003: 部分獲得はなく、配置自体ができない。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:cafe")])
        s.household = 4
        H.turn(s, 0)
        targets = [a.target_instance_id for a in H.legal(s)]
        self.assertNotIn(H.wp("base:cafe"), targets)

    # --- base:design-office ------------------------------------------------
    def test_design_office_keeps_one_discards_rest(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:design-office")])
        H.set_hand(s, 0, [])
        s = H.place(s, 0, H.wp("base:design-office"))
        self.assertEqual(len(s.temporary_public), 5)      # TX-022
        opts = H.legal(s)
        self.assertEqual(len(opts), 5)
        kept = opts[0].instance_id
        s = H.step(s, opts[0])
        self.assertEqual(s.temporary_public, ())          # after_resolution で空
        self.assertIn(kept, s.players[0].hand_buildings)
        self.assertEqual(len(s.building_discard), 4)
        # 一時公開されたカードは既知情報として残る(RUL-INFO-001, §7.3)
        self.assertEqual([e.instance_id for e in s.public_knowledge], [kept])
        H.check(s, "after_resolution")

    # --- base:factory ------------------------------------------------------
    def test_factory_discards_two_then_draws_four(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:factory")])
        H.set_hand(s, 0, [], goods=2)
        s = H.place(s, 0, H.wp("base:factory"))
        s = H.step(s, H.discard_goods(2))
        self.assertEqual(s.players[0].hand_goods, 0)
        self.assertEqual(len(s.players[0].hand_buildings), 4)
        H.check(s, "after_resolution")

    # --- base:orchard ------------------------------------------------------
    def test_orchard_tops_hand_up_to_four(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:orchard")])
        H.set_hand(s, 0, [H.wp("base:mansion")])          # 手札1枚
        s = H.place(s, 0, H.wp("base:orchard"))
        self.assertEqual(s.players[0].hand_total, 4)
        self.assertEqual(s.players[0].hand_goods, 3)
        H.check(s)

    def test_orchard_is_a_legal_no_op_at_four_cards(self):
        """OPEN-007: 手札4枚以上なら空振り。配置は合法。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:orchard")])
        H.set_hand(s, 0, [], goods=4)
        s = H.place(s, 0, H.wp("base:orchard"))
        self.assertEqual(s.players[0].hand_goods, 4)
        H.check(s)

    # --- base:construction-company ----------------------------------------
    def test_construction_company_applies_discount_one(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:construction-company")])
        H.set_hand(s, 0, [H.wp("base:farm")], goods=2)    # farm cost 1 -> 0
        s = H.place(s, 0, H.wp("base:construction-company"))
        s = H.step(s, H.build_action(H.wp("base:farm")))
        self.assertIn(H.wp("base:farm"), s.players[0].owned_buildings)
        self.assertEqual(s.players[0].hand_goods, 2)      # 割引でコスト0(OPEN-020)
        self.assertIsNone(s.pending)
        H.check(s, "after_resolution")

    # --- base:warehouse ----------------------------------------------------
    def test_warehouse_raises_hand_limit_by_four(self):
        from naeco.engine.invariants import hand_limit
        s = H.new()
        self.assertEqual(hand_limit(s, H.CARDS, 0), 5)
        H.own(s, 0, [H.wp("base:warehouse")])
        self.assertEqual(hand_limit(s, H.CARDS, 0), 9)    # EFF-warehouse-01

    # --- base:company-housing ---------------------------------------------
    def test_company_housing_raises_worker_cap_by_one(self):
        from naeco.engine.invariants import worker_cap
        s = H.new()
        self.assertEqual(worker_cap(s, H.CARDS, 0), 5)
        H.own(s, 0, [H.wp("base:company-housing")])
        self.assertEqual(worker_cap(s, H.CARDS, 0), 6)
        H.own(s, 0, [H.wp("base:company-housing", 1)])
        self.assertEqual(worker_cap(s, H.CARDS, 0), 7)    # 物理7を超えない

    # --- base:law-office ---------------------------------------------------
    def test_law_office_forgives_five_unpaid_wages(self):
        s = H.new()
        s.players[0].unpaid_wages = 7
        base = game.final_scores(s, H.CARDS)[0]
        H.own(s, 0, [H.wp("base:law-office")])
        after = game.final_scores(s, H.CARDS)[0]
        # 8(法律事務所の価値) + 5枚免除 * 3点
        self.assertEqual(after - base, 8 + 5 * 3)

    # --- base:plantation ---------------------------------------------------
    def test_plantation_gains_three_goods(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:plantation")])
        s = H.place(s, 0, H.wp("base:plantation"))
        self.assertEqual(s.players[0].hand_goods, 3)
        H.check(s)

    # --- base:restaurant ---------------------------------------------------
    def test_restaurant_discards_one_for_fifteen(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:restaurant")])
        H.set_hand(s, 0, [], goods=1)
        s.household = 15
        money = s.players[0].money
        s = H.place(s, 0, H.wp("base:restaurant"))
        s = H.step(s, H.discard_goods(1))
        self.assertEqual(s.players[0].money, money + 15)
        self.assertEqual(s.household, 0)
        H.check(s, "after_resolution")

    # --- base:pioneer ------------------------------------------------------
    def test_pioneer_builds_a_farm_tagged_card_for_free(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:pioneer")])
        H.set_hand(s, 0, [H.wp("base:plantation"), H.wp("base:mansion")])
        s = H.place(s, 0, H.wp("base:pioneer"))
        # 候補は farm 属性のみ(大農園は farm、邸宅は違う)
        self.assertEqual([a.instance_id for a in H.legal(s)],
                         [H.wp("base:plantation")])
        s = H.step(s, H.build_action(H.wp("base:plantation")))
        self.assertIn(H.wp("base:plantation"), s.players[0].owned_buildings)
        self.assertIn(H.wp("base:mansion"), s.players[0].hand_buildings)  # 支払いなし
        H.check(s, "after_resolution")

    # --- base:realtor ------------------------------------------------------
    def test_realtor_scores_three_per_owned_building(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:realtor"), H.wp("base:farm"), H.wp("base:mansion")])
        # 価値 10+6+28 = 44、不動産屋 3棟*3 = 9
        self.assertEqual(game.final_scores(s, H.CARDS)[0],
                         s.players[0].money + 44 + 9)

    # --- base:agricultural-coop -------------------------------------------
    def test_agricultural_coop_scores_three_per_hand_good(self):
        s = H.new()
        H.set_hand(s, 0, [], goods=4)
        H.own(s, 0, [H.wp("base:agricultural-coop")])
        self.assertEqual(game.final_scores(s, H.CARDS)[0],
                         s.players[0].money + 12 + 4 * 3)

    # --- base:general-contractor ------------------------------------------
    def test_general_contractor_builds_then_draws_two(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:general-contractor")])
        H.set_hand(s, 0, [H.wp("base:farm")], goods=1)    # farm cost 1
        s = H.place(s, 0, H.wp("base:general-contractor"))
        s = H.step(s, H.build_action(H.wp("base:farm")))
        s = H.step(s, H.discard_goods(1))
        self.assertIn(H.wp("base:farm"), s.players[0].owned_buildings)
        self.assertEqual(len(s.players[0].hand_buildings), 2)   # 建設後に2枚ドロー
        H.check(s, "after_resolution")

    def test_general_contractor_illegal_without_buildable_card(self):
        """OPEN-021: ドロー目的のみの配置は不可。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:general-contractor")])
        H.set_hand(s, 0, [])
        H.turn(s, 0)
        self.assertNotIn(H.wp("base:general-contractor"),
                         [a.target_instance_id for a in H.legal(s)])

    # --- base:steel-mill ---------------------------------------------------
    def test_steel_mill_draws_three(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:steel-mill")])
        H.set_hand(s, 0, [])
        s = H.place(s, 0, H.wp("base:steel-mill"))
        self.assertEqual(len(s.players[0].hand_buildings), 3)
        H.check(s)

    # --- base:chemical-plant -----------------------------------------------
    def test_chemical_plant_draws_four_on_empty_hand(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:chemical-plant")])
        H.set_hand(s, 0, [])
        s = H.place(s, 0, H.wp("base:chemical-plant"))
        self.assertEqual(len(s.players[0].hand_buildings), 4)
        H.check(s)

    def test_chemical_plant_draws_two_otherwise(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:chemical-plant")])
        H.set_hand(s, 0, [], goods=1)                     # 消費財も手札に数える
        s = H.place(s, 0, H.wp("base:chemical-plant"))
        self.assertEqual(len(s.players[0].hand_buildings), 2)
        H.check(s)

    # --- base:mansion ------------------------------------------------------
    def test_mansion_is_value_only(self):
        s = H.new()
        base = game.final_scores(s, H.CARDS)[0]
        H.own(s, 0, [H.wp("base:mansion")])
        self.assertEqual(game.final_scores(s, H.CARDS)[0] - base, 28)
        self.assertNotIn(H.wp("base:mansion"), s.workplaces)   # 施設は配置不能

    # --- base:labor-union --------------------------------------------------
    def test_labor_union_scores_six_per_employed_worker(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:labor-union")])
        self.assertEqual(s.employed(0), 2)
        self.assertEqual(game.final_scores(s, H.CARDS)[0],
                         s.players[0].money + 0 + 2 * 6)

    # --- base:automobile-factory ------------------------------------------
    def test_automobile_factory_discards_three_then_draws_seven(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:automobile-factory")])
        H.set_hand(s, 0, [], goods=3)
        s = H.place(s, 0, H.wp("base:automobile-factory"))
        s = H.step(s, H.discard_goods(3))
        self.assertEqual(len(s.players[0].hand_buildings), 7)
        H.check(s, "after_resolution")

    # --- base:two-for-one --------------------------------------------------
    def test_two_for_one_builds_two_same_cost_for_one_payment(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        # farm と slash-and-burn はどちらもコスト1
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1)], goods=1)
        s = H.place(s, 0, H.wp("base:two-for-one"))
        # 1枚目の候補は「より大きいIDの同コスト相方が存在する」ものだけ(R2)
        self.assertEqual([a.instance_id for a in H.legal(s)], [H.wp("base:farm")])
        s = H.step(s, H.build_action(H.wp("base:farm")))
        self.assertEqual([a.instance_id for a in H.legal(s)], [H.wp("base:farm", 1)])
        s = H.step(s, H.build_action(H.wp("base:farm", 1)))
        s = H.step(s, H.discard_goods(1))                 # コスト1回分だけ
        self.assertEqual(set(s.players[0].owned_buildings),
                         {H.wp("base:two-for-one"), H.wp("base:farm"),
                          H.wp("base:farm", 1)})
        H.check(s, "after_resolution")

    def test_two_for_one_illegal_without_affordable_pair(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:two-for-one")])
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:farm", 1)])  # 原資なし
        H.turn(s, 0)
        self.assertNotIn(H.wp("base:two-for-one"),
                         [a.target_instance_id for a in H.legal(s)])

    # --- base:headquarters -------------------------------------------------
    def test_headquarters_scores_six_per_locked_building(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:headquarters"), H.wp("base:mansion"),
                     H.wp("base:farm")])
        # locked: 本社ビル・邸宅の2つ(農場は locked でない)
        self.assertEqual(game.final_scores(s, H.CARDS)[0],
                         s.players[0].money + (20 + 28 + 6) + 2 * 6)

    # --- base:railroad -----------------------------------------------------
    def test_railroad_scores_eight_per_factory_building(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:railroad"), H.wp("base:factory"),
                     H.wp("base:steel-mill")])
        self.assertEqual(game.final_scores(s, H.CARDS)[0],
                         s.players[0].money + (18 + 12 + 20) + 2 * 8)


class PublicWorkplaceEffects(unittest.TestCase):
    """RULES.md 第5章 公共職場12種。"""

    # --- base:quarry -------------------------------------------------------
    def test_quarry_draws_one_and_moves_start_player_marker(self):
        s = H.new()
        H.set_hand(s, 1, [])
        s = H.place(s, 1, H.wp("base:quarry"))
        self.assertEqual(len(s.players[1].hand_buildings), 1)
        self.assertEqual(s.next_start_player, 1)          # TX-023
        self.assertEqual(s.start_player, 0)               # 次ラウンドから有効
        H.check(s)

    # --- base:mine ---------------------------------------------------------
    def test_mine_is_unlimited_and_always_placeable(self):
        s = H.new()
        H.set_hand(s, 0, [])
        s = H.place(s, 0, H.wp("base:mine"))
        self.assertEqual(len(s.players[0].hand_buildings), 1)
        # 占有無制限(INV-005 唯一の例外) → 他家も同じラウンドに置ける
        s = H.place(s, 1, H.wp("base:mine"))
        self.assertEqual(s.workplaces[H.wp("base:mine")].occupants, (0, 1))
        H.check(s)

    def test_mine_placeable_even_when_deck_and_discard_are_empty(self):
        """保証2(行き詰まりがない)の保証者。OPEN-010 / OPEN-011。

        山札・捨て札が完全に枯渇していても鉱山は合法手として残り続ける。
        """
        s = H.new()
        # 建物64実体をすべてプレイヤー0の手札に集め、山札と捨て札を空にする。
        s.players[1].hand_buildings = ()
        s.players[0].hand_buildings = tuple(sorted(H.CARDS.building_instances))
        s.deck = ()
        s.building_discard = ()
        H.check(s)
        H.turn(s, 0)
        targets = [a.target_instance_id for a in H.legal(s)]
        self.assertIn(H.wp("base:mine"), targets)
        s = H.place(s, 0, H.wp("base:mine"))
        self.assertEqual(len(s.players[0].hand_buildings), 64)   # 空振りドロー
        H.check(s)

    # --- base:school -------------------------------------------------------
    def test_school_hires_one_training_worker(self):
        s = H.new()
        s = H.place(s, 0, H.wp("base:school"))
        self.assertEqual(s.players[0].workers_training, 1)   # RUL-WORKER-003
        H.check(s)

    def test_school_is_a_legal_no_op_at_worker_cap(self):
        """OPEN-018: 雇用上限時は空振り(配置合法)。"""
        s = H.new()
        s.players[0].workers_waiting = 5
        s = H.place(s, 0, H.wp("base:school"))
        self.assertEqual(s.players[0].workers_training, 0)
        H.check(s)

    # --- base:carpenter ----------------------------------------------------
    def test_carpenter_builds_at_printed_cost(self):
        s = H.new()
        H.set_hand(s, 0, [H.wp("base:factory")], goods=2)     # cost 2
        s = H.place(s, 0, H.wp("base:carpenter"))
        s = H.step(s, H.build_action(H.wp("base:factory")))
        s = H.step(s, H.discard_goods(2))
        self.assertIn(H.wp("base:factory"), s.players[0].owned_buildings)
        self.assertEqual(s.players[0].hand_goods, 0)
        H.check(s, "after_resolution")

    def test_carpenter_instances_are_independent(self):
        """§1.2.1: 大工はカード実体ごとに1人(RUL-WORKER-001)。"""
        s = H.new(4)
        self.assertEqual(
            len([i for i in s.workplaces if i.startswith("base:carpenter")]), 3)
        for pid in (0, 1):
            H.set_hand(s, pid, [H.wp("base:farm", pid)], goods=1)
        s = H.place(s, 0, H.wp("base:carpenter", 0))
        s = H.step(s, H.build_action(H.wp("base:farm", 0)))
        s = H.step(s, H.discard_goods(1))
        s = H.place(s, 1, H.wp("base:carpenter", 1))
        self.assertIsNotNone(s.pending)
        H.check(s)

    # --- ラウンドカード表面(露店〜万博) ---------------------------------
    def _household_workplace(self, round_no, discard, amount):
        s = H.new()
        H.set_round(s, round_no + 1)
        H.set_hand(s, 0, [], goods=discard)
        s.household = amount
        money = s.players[0].money
        s = H.place(s, 0, H.wp("base:round-%d" % round_no))
        s = H.step(s, H.discard_goods(discard))
        self.assertEqual(s.players[0].money, money + amount)
        self.assertEqual(s.household, 0)
        self.assertEqual(s.players[0].hand_goods, 0)
        H.check(s, "after_resolution")
        return s

    def test_stall_discards_one_for_six(self):
        self._household_workplace(1, 1, 6)

    def test_market_discards_two_for_twelve(self):
        self._household_workplace(2, 2, 12)

    def test_supermarket_discards_three_for_eighteen(self):
        self._household_workplace(4, 3, 18)

    def test_department_store_discards_four_for_twentyfour(self):
        self._household_workplace(6, 4, 24)

    def test_expo_discards_five_for_thirty(self):
        self._household_workplace(8, 5, 30)

    def test_stall_illegal_when_household_short(self):
        s = H.new()
        H.set_round(s, 2)
        H.set_hand(s, 0, [], goods=1)
        s.household = 5                                   # $6 に足りない
        H.turn(s, 0)
        self.assertNotIn(H.wp("base:round-1"),
                         [a.target_instance_id for a in H.legal(s)])

    # --- base:high-school --------------------------------------------------
    def test_high_school_hires_up_to_four(self):
        s = H.new()
        H.set_round(s, 4)
        s = H.place(s, 0, H.wp("base:round-3"))
        # 初期2人 → 4人まで一括雇用(OPEN-014)。研修中(OPEN-023)
        self.assertEqual(s.players[0].workers_training, 2)
        self.assertEqual(s.employed(0), 4)
        H.check(s)

    # --- base:university ---------------------------------------------------
    def test_university_hires_up_to_five(self):
        s = H.new()
        H.set_round(s, 6)
        s = H.place(s, 0, H.wp("base:round-5"))
        self.assertEqual(s.employed(0), 5)
        H.check(s)

    def test_university_is_capped_by_worker_cap(self):
        s = H.new()
        H.set_round(s, 6)
        s.players[0].workers_waiting = 5                  # 上限5に到達済み
        s = H.place(s, 0, H.wp("base:round-5"))
        self.assertEqual(s.employed(0), 5)
        H.check(s)

    # --- base:vocational-school -------------------------------------------
    def test_vocational_school_hires_ready_worker(self):
        """TX-027: 研修中ではなく待機状態(当ラウンドから配置可能)。"""
        s = H.new()
        H.set_round(s, 8)
        before = s.players[0].workers_waiting
        s = H.place(s, 0, H.wp("base:round-7"))
        self.assertEqual(s.players[0].workers_waiting, before - 1 + 1)
        self.assertEqual(s.players[0].workers_training, 0)
        H.check(s)


class TokenEffects(unittest.TestCase):
    """消費財と未払い賃金(instance_id を持たず枚数で保持する。OPEN-024)。"""

    # --- base:consumer-goods -----------------------------------------------
    def test_consumer_goods_are_payable_and_countable(self):
        s = H.new()
        H.set_hand(s, 0, [H.wp("base:farm")], goods=1)
        s = H.place(s, 0, H.wp("base:carpenter"))
        s = H.step(s, H.build_action(H.wp("base:farm")))
        # 消費財を建設コストに使える(EFF-consumer-goods-01)
        s = H.step(s, H.discard_goods(1))
        self.assertEqual(s.players[0].hand_goods, 0)
        self.assertIn(H.wp("base:farm"), s.players[0].owned_buildings)
        H.check(s, "after_resolution")

    def test_consumer_goods_have_no_instance_ids(self):
        s = H.new()
        self.assertEqual(H.CARDS.instances_by_card["base:consumer-goods"], ())
        self.assertEqual(H.CARDS.instances_by_card["base:unpaid-wage"], ())

    # --- base:unpaid-wage --------------------------------------------------
    def test_unpaid_wage_costs_three_points_each(self):
        s = H.new()
        base = game.final_scores(s, H.CARDS)[0]
        s.players[0].unpaid_wages = 4
        self.assertEqual(game.final_scores(s, H.CARDS)[0], base - 12)


if __name__ == "__main__":
    unittest.main()
