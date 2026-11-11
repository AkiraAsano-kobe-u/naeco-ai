"""賃金・借金・終了の境界テスト(PLAN.md Phase 1 完了条件)。"""
import unittest

import helpers as H
from naeco.engine import game
from naeco.engine.state import DecisionKind, Phase


def run_end_phase(s):
    """PHASE-END を開始し、pending が立つまで(または完了まで)進める。"""
    s.phase = Phase.END
    s.end_phase_cursor = 0
    game._advance(s, H.CARDS)
    return s


class WageTests(unittest.TestCase):
    """RUL-WAGE-001 / RUL-WAGE-002。"""

    def test_wage_rate_table_matches_rules(self):
        expected = [2, 2, 3, 3, 3, 4, 4, 5, 5]
        self.assertEqual([H.CARDS.wage_rate(r) for r in range(1, 10)], expected)

    def test_wage_counts_training_workers(self):
        """TERM-005: 研修中も賃金計算の対象。"""
        s = H.new()
        s.players[0].workers_training = 1
        s.players[0].money = 20
        s.players[1].money = 20
        before = s.players[0].money
        s = run_end_phase(s)
        # R1 賃金率$2 × 雇用3人 = $6
        self.assertEqual(before - s.players[0].money, 6)

    def test_wage_goes_to_household(self):
        s = H.new()
        s.players[0].money = 20
        s.players[1].money = 20
        s = run_end_phase(s)
        self.assertEqual(s.household, 4 + 4)      # 2人 × ($2 × 2人)


class InsolvencyTests(unittest.TestCase):
    """RUL-INSOLV-001 / RUL-BUILD-002。"""

    def test_selling_is_forced_and_only_the_choice_of_which(self):
        """§3.4 G: 「売らない」という選択肢はない。"""
        s = H.new()
        s.players[0].money = 0
        s.players[1].money = 20
        H.own(s, 0, [H.wp("base:farm"), H.wp("base:factory")])
        s = run_end_phase(s)
        self.assertIsNotNone(s.pending)
        self.assertEqual(s.pending.kind, DecisionKind.INSOLVENCY_SELL)
        kinds = set(a.kind for a in H.legal(s))
        self.assertEqual(kinds, {H.legal(s)[0].kind})     # SELL_BUILDING のみ
        self.assertEqual(len(H.legal(s)), 2)              # 打ち切り行動はない

    def test_locked_buildings_cannot_be_sold(self):
        s = H.new()
        s.players[0].money = 0
        s.players[1].money = 20
        H.own(s, 0, [H.wp("base:mansion"), H.wp("base:farm")])   # 邸宅は locked
        s = run_end_phase(s)
        self.assertEqual([a.instance_id for a in H.legal(s)], [H.wp("base:farm")])

    def test_sale_makes_the_building_a_public_workplace(self):
        s = H.new()
        s.players[0].money = 0
        s.players[1].money = 20
        H.own(s, 0, [H.wp("base:farm")])
        s = run_end_phase(s)
        s = H.step(s, H.sell(H.wp("base:farm")))
        w = s.workplaces[H.wp("base:farm")]
        self.assertIsNone(w.owner)                        # TX-011
        self.assertEqual(w.usable_by, "any_player")       # RUL-BUILD-002
        self.assertNotIn(H.wp("base:farm"), s.players[0].owned_buildings)
        H.check(s)

    def test_selling_stops_once_the_wage_is_covered(self):
        """RUL-BUILD-002: 金銭が足りるなら売却してはならない。"""
        s = H.new()
        s.players[0].money = 0
        s.players[1].money = 20
        H.own(s, 0, [H.wp("base:farm"), H.wp("base:factory")])   # 価値 6 と 12
        s = run_end_phase(s)
        s = H.step(s, H.sell(H.wp("base:factory")))       # $12 > 賃金$4
        self.assertIsNone(s.pending)
        self.assertIn(H.wp("base:farm"), s.players[0].owned_buildings)
        self.assertEqual(s.players[0].money, 12 - 4)

    def test_unpaid_wages_issued_when_nothing_left_to_sell(self):
        """RUL-INSOLV-001 (2) / TX-015。"""
        s = H.new()
        s.players[0].money = 1
        s.players[1].money = 20
        s = run_end_phase(s)
        self.assertEqual(s.players[0].money, 0)           # 全額を家計へ
        self.assertEqual(s.players[0].unpaid_wages, 3)    # 賃金$4 - 所持$1
        H.check(s)

    def test_unpaid_wages_are_counted_but_money_is_not_created(self):
        """INV-001: TX-015 はカード発行であり金銭を生成しない。"""
        s = H.new()
        s.players[0].money = 1
        s.players[1].money = 20
        total_before = s.household + sum(p.money for p in s.players)
        s = run_end_phase(s)
        total_after = s.household + sum(p.money for p in s.players)
        self.assertEqual(total_before, total_after)


class HandLimitTests(unittest.TestCase):
    """RUL-ROUND-002 / INV-006 / OPEN-019。"""

    def test_hand_limit_applies_only_at_round_end(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:steel-mill")])
        H.set_hand(s, 0, [], goods=5)
        s = H.place(s, 0, H.wp("base:steel-mill"))        # 手札8枚になる
        self.assertEqual(s.players[0].hand_total, 8)
        H.check(s)                                        # 手番中の超過は合法

    def test_hand_limit_discard_pending_at_round_end(self):
        s = H.new()
        s.players[0].money = 20
        s.players[1].money = 20
        H.set_hand(s, 0, [], goods=8)
        s = run_end_phase(s)
        self.assertEqual(s.pending.kind, DecisionKind.HAND_LIMIT_DISCARD)
        self.assertEqual(s.pending.remaining, 3)          # 8 - 5
        s = H.step(s, H.discard_goods(3))
        self.assertEqual(s.players[0].hand_goods, 5)
        H.check(s, "after_round")

    def test_warehouse_extends_the_limit_at_round_end(self):
        s = H.new()
        s.players[0].money = 20
        s.players[1].money = 20
        H.own(s, 0, [H.wp("base:warehouse")])
        H.set_hand(s, 0, [], goods=9)
        s = run_end_phase(s)
        self.assertIsNone(s.pending)                      # 上限 5+4=9 なので調整なし
        self.assertEqual(s.players[0].hand_goods, 9)

    def test_goods_count_toward_the_hand_limit(self):
        s = H.new()
        s.players[0].money = 20
        s.players[1].money = 20
        H.set_hand(s, 0, [H.wp("base:farm")], goods=6)    # 合計7枚
        s = run_end_phase(s)
        self.assertEqual(s.pending.remaining, 2)


class GameEndTests(unittest.TestCase):
    """RUL-GAMEEND-001 / RUL-SCORE-001 / RUL-SCORE-002。"""

    def test_round_nine_end_terminates_the_game(self):
        s = H.new()
        H.set_round(s, 9)
        s.players[0].money = 50
        s.players[1].money = 50
        s = run_end_phase(s)
        self.assertEqual(s.phase, Phase.GAME_OVER)
        self.assertEqual(H.legal(s), [])
        self.assertIsNone(game.actor(s))

    def test_round_nine_does_not_flip_a_round_card(self):
        """RUL-RCARD-001 / OPEN-005: カード9の表面は使用しない。"""
        s = H.new()
        H.set_round(s, 9)
        s.players[0].money = 50
        s.players[1].money = 50
        s = run_end_phase(s)
        self.assertIn(H.wp("base:round-9"), s.round_deck)
        self.assertNotIn(H.wp("base:round-9"), s.workplaces)

    def test_tiebreak_favours_the_start_player(self):
        """RUL-SCORE-002: 引き分けは存在しない(§9.3)。"""
        s = H.new()
        s.players[0].money = 10
        s.players[1].money = 10
        s.start_player = 1
        self.assertEqual(game.final_scores(s, H.CARDS), (10, 10))
        self.assertEqual(game.ranking(s, H.CARDS), (1, 0))
        s.start_player = 0
        self.assertEqual(game.ranking(s, H.CARDS), (0, 1))

    def test_tiebreak_is_a_strict_total_order_for_all_places(self):
        s = H.new(4)
        for p in s.players:
            p.money = 7
        s.start_player = 2
        self.assertEqual(sorted(game.ranking(s, H.CARDS)), [0, 1, 2, 3])
        self.assertEqual(game.ranking(s, H.CARDS), (2, 3, 0, 1))

    def test_utilities_are_defined_for_all_supported_players(self):
        s = H.new()
        s.players[0].money = 20
        self.assertEqual(game.terminal_utilities(s, H.CARDS), (1.0, -1.0))
        for n in (3, 4):
            sn = H.new(n)
            utilities = game.terminal_utilities(sn, H.CARDS)
            self.assertEqual(len(utilities), n)
            self.assertEqual(utilities,
                             tuple(1.0 - 2 * r / (n - 1)
                                   for r in game.ranking(sn, H.CARDS)))

    def test_step_terminates_normally_for_three_and_four_players(self):
        """§9.2: N>=3 も長さ N の順位線形 utilities を返して終端する。"""
        import random
        for n in (3, 4):
            rng = random.Random(n)
            s = H.new(n, seed=n)
            r = None
            while s.phase != Phase.GAME_OVER:
                r = game.step(s, rng.choice(H.legal(s)), H.CARDS)
                s = r.state
            assert r is not None
            self.assertTrue(r.terminal)
            self.assertIsNotNone(r.scores)
            self.assertIsNotNone(r.ranking)
            self.assertEqual(r.utilities, game.terminal_utilities(s, H.CARDS))
            assert r.utilities is not None
            self.assertEqual(len(r.utilities), n)

    def test_hand_cards_do_not_score_except_via_the_coop(self):
        """RUL-SCORE-001: 手札は得点にならない(農協の効果対象を除く)。"""
        s = H.new()
        base = game.final_scores(s, H.CARDS)[0]
        H.set_hand(s, 0, [H.wp("base:mansion")], goods=3)
        self.assertEqual(game.final_scores(s, H.CARDS)[0], base)


class SlashAndBurnRoundNineTests(unittest.TestCase):
    """OPEN-026: R9に配置した焼畑は所有建物のまま終了する(★明示テスト)。"""

    def _round_nine_with_slash_and_burn(self):
        s = H.new()
        H.set_round(s, 9)
        H.own(s, 0, [H.wp("base:slash-and-burn")])
        s = H.place(s, 0, H.wp("base:slash-and-burn"))
        self.assertEqual(s.slash_and_burn_load[H.wp("base:slash-and-burn")], 5)
        s.players[0].money = 50
        s.players[1].money = 50
        return run_end_phase(s)

    def test_card_stays_owned_and_load_is_returned_to_the_pile(self):
        s = self._round_nine_with_slash_and_burn()
        self.assertEqual(s.phase, Phase.GAME_OVER)
        self.assertIn(H.wp("base:slash-and-burn"), s.players[0].owned_buildings)
        self.assertNotIn(H.wp("base:slash-and-burn"), s.building_discard)
        self.assertEqual(s.slash_and_burn_load, {})       # 積載は山へ戻る
        H.check(s)

    def test_realtor_counts_the_round_nine_slash_and_burn(self):
        s = self._round_nine_with_slash_and_burn()
        before = game.final_scores(s, H.CARDS)[0]
        H.own(s, 0, [H.wp("base:realtor")])
        after = game.final_scores(s, H.CARDS)[0]
        # 不動産屋の価値10 + (焼畑・不動産屋の2棟)×3点 = 10 + 6
        self.assertEqual(after - before, 10 + 2 * 3)

    def test_headquarters_counts_the_round_nine_slash_and_burn(self):
        s = self._round_nine_with_slash_and_burn()
        before = game.final_scores(s, H.CARDS)[0]
        H.own(s, 0, [H.wp("base:headquarters")])
        after = game.final_scores(s, H.CARDS)[0]
        # 本社ビルの価値20 + locked 2棟(焼畑・本社ビル)×6点
        self.assertEqual(after - before, 20 + 2 * 6)

    def test_both_together_are_worth_nine_more_points(self):
        """設計書 §12: 両方所有していれば9点差になる。取りこぼしてはならない。"""
        from naeco.engine.state import clone
        sab = H.wp("base:slash-and-burn")
        s = self._round_nine_with_slash_and_burn()
        H.own(s, 0, [H.wp("base:realtor"), H.wp("base:headquarters")])
        with_sab = game.final_scores(s, H.CARDS)[0]

        # 焼畑を捨て札へ移した(= TX-021 が誤って発火した)場合との差を数える。
        s2 = clone(s)
        s2.players[0].owned_buildings = tuple(
            i for i in s2.players[0].owned_buildings if i != sab)
        s2.workplaces.pop(sab, None)
        s2.building_discard = tuple(sorted(s2.building_discard + (sab,)))
        H.check(s2)
        without_sab = game.final_scores(s2, H.CARDS)[0]

        # 焼畑の価値は0なので、差は不動産屋+3 と 本社ビル+6 の9点ちょうど
        self.assertEqual(with_sab - without_sab, 9)

    def test_loaded_goods_do_not_score_for_the_coop(self):
        """OPEN-022: 積載中の消費財は農協得点の対象外。"""
        s = H.new()
        H.set_round(s, 9)
        H.own(s, 0, [H.wp("base:slash-and-burn"), H.wp("base:agricultural-coop")])
        H.set_hand(s, 0, [])
        s = H.place(s, 0, H.wp("base:slash-and-burn"))
        s.players[0].money = 50
        s.players[1].money = 50
        s = run_end_phase(s)
        self.assertEqual(s.players[0].hand_goods, 0)
        base = s.players[0].money + 0 + 12                # 焼畑0 + 農協12
        self.assertEqual(game.final_scores(s, H.CARDS)[0], base)


class DeckTests(unittest.TestCase):
    """RUL-DECK-001 / TX-007 / OPEN-011。"""

    def test_reshuffle_when_the_deck_runs_out(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:steel-mill")])
        H.set_hand(s, 0, [])
        rest = s.deck
        s.deck = rest[:2]                                 # 山札2枚のみ
        s.building_discard = tuple(sorted(rest[2:]))
        s = H.place(s, 0, H.wp("base:steel-mill"))        # 3枚引く
        self.assertEqual(len(s.players[0].hand_buildings), 3)
        self.assertEqual(s.reshuffle_seq, 1)
        self.assertEqual(s.building_discard, ())
        H.check(s)

    def test_reshuffle_records_deck_membership_as_public_knowledge(self):
        """§7.3: TX-007 で知識を削除してはならない(IN_DECK_SINCE として保持)。"""
        from naeco.engine.state import KnowledgeKind
        s = H.new()
        H.set_hand(s, 0, [])                              # 鉱山は初期公共職場
        pooled = tuple(sorted(s.deck))
        s.building_discard = pooled
        s.deck = ()
        s = H.place(s, 0, H.wp("base:mine"))
        kinds = set(e.kind for e in s.public_knowledge)
        self.assertEqual(kinds, {KnowledgeKind.IN_DECK_SINCE})
        self.assertEqual(
            sorted(e.instance_id for e in s.public_knowledge), sorted(pooled))
        self.assertEqual(s.draws_since_reshuffle[0], 1)

    def test_draw_is_a_no_op_when_deck_and_discard_are_exhausted(self):
        """OPEN-011: 引ける枚数だけ引き、不足は無視する。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:steel-mill")])
        s.players[1].hand_buildings = ()
        s.players[0].hand_buildings = tuple(
            sorted(set(H.CARDS.building_instances) - {H.wp("base:steel-mill")}))
        s.deck = ()
        s.building_discard = ()
        n = len(s.players[0].hand_buildings)
        s = H.place(s, 0, H.wp("base:steel-mill"))
        self.assertEqual(len(s.players[0].hand_buildings), n)
        H.check(s)


class StartPlayerSetupTests(unittest.TestCase):
    """RUL-SETUP-001: 開始プレイヤーは公平に決める(2026-07-21 修正)。"""

    def test_start_player_is_drawn_from_the_seed_not_fixed_to_seat_zero(self):
        seen = set(game.new_game(2, seed, H.CARDS).start_player
                   for seed in range(60))
        self.assertEqual(seen, {0, 1})
        seen3 = set(game.new_game(3, seed, H.CARDS).start_player
                    for seed in range(60))
        self.assertEqual(seen3, {0, 1, 2})

    def test_the_draw_is_deterministic_for_a_given_seed(self):
        for seed in range(12):
            self.assertEqual(game.new_game(2, seed, H.CARDS).start_player,
                             game.new_game(2, seed, H.CARDS).start_player)

    def test_the_draw_does_not_consume_the_environment_rng(self):
        """§2.4: 環境RNGは初期シャッフルと TX-007 しか消費しない。

        抽選の有無・指定値で `rng_state` が動くと、以降の山札再構成列がずれる。
        """
        for seed in range(20):
            drawn = game.new_game(2, seed, H.CARDS)
            for fixed in (0, 1):
                pinned = game.new_game(2, seed, H.CARDS, start_player=fixed)
                self.assertEqual(drawn.rng_state, pinned.rng_state, seed)
                self.assertEqual(drawn.deck, pinned.deck, seed)

    def test_the_setup_stream_is_seat_independent(self):
        """席交換ペアの2局は同一配札**かつ同一開始席**でなければならない(§2.5)。"""
        from naeco.engine.state import SEAT_INDEPENDENT_STREAMS, derive_seed
        self.assertIn("setup", SEAT_INDEPENDENT_STREAMS)
        for root in (0, 1, 12345):
            self.assertEqual(derive_seed(root, "setup", 7, seat_variant=0),
                             derive_seed(root, "setup", 7, seat_variant=1))
        # 方策ストリームは席ごとに変えてよい(むしろ変えるべき)。
        self.assertNotEqual(derive_seed(0, "policy", 7, seat_variant=0),
                            derive_seed(0, "policy", 7, seat_variant=1))

    def test_the_harness_derives_setup_from_the_evaluation_root(self):
        """§10.1: 1局の各ストリームは derive_seed(eval_seed(group,index), kind, ...)。

        開始席の seed も env_seed の再利用ではなく root から派生していること。
        """
        from naeco.ai.eval.harness import evaluate
        from naeco.engine.state import derive_seed, eval_seed
        result = evaluate(pairs=2, group="train", start_index=900,
                          bootstrap_samples=10, bootstrap_seed=1)
        for rec in result.game_records:
            root = eval_seed(rec.group, rec.index)
            expected = derive_seed(root, "setup", rec.index)
            self.assertEqual(rec.setup_seed, expected)
            self.assertNotEqual(rec.setup_seed, rec.env_seed)
            self.assertEqual(rec.start_player,
                             game.draw_start_player(2, expected))

    def test_paired_games_share_the_start_player(self):
        from naeco.ai.eval.harness import evaluate
        result = evaluate(pairs=3, group="train", start_index=920,
                          bootstrap_samples=10, bootstrap_seed=1)
        by_index = {}
        for rec in result.game_records:
            by_index.setdefault(rec.index, []).append(rec)
        for recs in by_index.values():
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0].setup_seed, recs[1].setup_seed)
            self.assertEqual(recs[0].start_player, recs[1].start_player)
            self.assertEqual(recs[0].initial_deck, recs[1].initial_deck)
            self.assertEqual(recs[0].initial_money, recs[1].initial_money)

    def test_an_explicit_setup_seed_selects_the_start_player(self):
        for setup_seed in range(40):
            expected = game.draw_start_player(2, setup_seed)
            s = game.new_game(2, 0, H.CARDS, setup_seed=setup_seed)
            self.assertEqual(s.start_player, expected)
            # setup_seed は環境RNGにも配札にも影響しない。
            self.assertEqual(s.deck, game.new_game(2, 0, H.CARDS).deck)
            self.assertEqual(s.rng_state, game.new_game(2, 0, H.CARDS).rng_state)

    def test_the_deal_does_not_depend_on_the_start_player_draw(self):
        """抽選は山札シャッフルの後に引くので、配札は seed だけで決まる。"""
        for seed in range(8):
            a = game.new_game(2, seed, H.CARDS, start_player=0)
            b = game.new_game(2, seed, H.CARDS, start_player=1)
            self.assertEqual(a.deck, b.deck)
            self.assertEqual([p.hand_buildings for p in a.players],
                             [p.hand_buildings for p in b.players])

    def test_initial_money_follows_the_start_player(self):
        """$5 は席0ではなく**開始プレイヤー**が受け取る(RUL-SETUP-001 の表)。"""
        for n in (2, 3, 4):
            for start in range(n):
                s = game.new_game(n, 0, H.CARDS, start_player=start)
                self.assertEqual(s.players[start].money, 5)
                for pid in range(n):
                    self.assertEqual(s.players[pid].money,
                                     5 + (pid - start) % n)

    def test_the_start_player_takes_the_first_turn(self):
        for start in (0, 1):
            s = game.new_game(2, 0, H.CARDS, start_player=start)
            self.assertEqual(s.current_player, start)
            self.assertEqual(s.next_start_player, start)

    def test_an_out_of_range_start_player_is_rejected(self):
        for bad in (-1, 2, 99):
            with self.assertRaises(ValueError):
                game.new_game(2, 0, H.CARDS, start_player=bad)

    def test_the_marker_is_kept_when_nobody_uses_the_quarry(self):
        """採石場を誰も使わなければ開始プレイヤーは動かない(TX-023)。"""
        s = H.new(2, 3, start_player=1)
        s.next_start_player = 1
        s.round_number = 4
        game._finish_round(s, H.CARDS)
        self.assertEqual(s.start_player, 1)
        self.assertEqual(s.next_start_player, 1)


class TurnAdvancedTests(unittest.TestCase):
    """§3.1: turn_advanced は「入力した行動の手番が完了したか」を表す(sol K)。"""

    def test_intermediate_decisions_do_not_advance_the_turn(self):
        s = H.new()
        H.set_hand(s, 0, [H.wp("base:farm"), H.wp("base:cafe")])
        result = H.result(H.turn(s, 0), _place(H.wp("base:carpenter")))
        self.assertFalse(result.turn_advanced)      # 建設対象の選択が残る

    def test_a_completed_placement_advances_even_when_the_end_phase_pends(self):
        """最後のワーカーを置いて手番が完了した step は、終了処理の pending が
        立っても turn_advanced=True でなければならない。"""
        s = H.new()
        H.set_hand(s, 0, [], goods=9)              # 手札上限を超えさせる
        H.set_hand(s, 1, [])
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        result = H.result(H.turn(s, 0), _place(H.wp("base:mine")))
        self.assertIsNotNone(result.state.pending)
        self.assertEqual(result.state.pending.kind,
                         DecisionKind.HAND_LIMIT_DISCARD)
        self.assertTrue(result.turn_advanced)

    def test_a_continuing_forced_sale_does_not_advance_the_turn(self):
        """RUL-INSOLV-001: 1回売っても不足が残れば同じ決定が続く(sol 4回目)。"""
        s = H.new()
        # 価値6の農場×2。賃金は $2 × 雇用5人 = $10 で、1枚売っても足りない。
        H.own(s, 0, [H.wp("base:farm", 0), H.wp("base:farm", 1)])
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].money = 0
        s.players[0].workers_waiting = 0
        s.players[0].workers_training = 5
        s.players[1].workers_waiting = 0
        s.players[1].money = 50
        s = run_end_phase(s)
        self.assertEqual(s.pending.kind, DecisionKind.INSOLVENCY_SELL)
        first = H.result(s, H.sell(H.wp("base:farm", 0)))
        # 不足が残るので同じ RUL-INSOLV-001 が再び立つ = 手番は進んでいない。
        self.assertIsNotNone(first.state.pending)
        self.assertEqual(first.state.pending.kind, DecisionKind.INSOLVENCY_SELL)
        self.assertEqual(first.state.pending.player_id, 0)
        self.assertFalse(first.turn_advanced)
        # 2枚目を売れば賃金を満たし、連鎖が終わる。
        second = H.result(first.state, H.sell(H.wp("base:farm", 1)))
        self.assertTrue(second.turn_advanced)

    def test_a_sale_for_another_player_is_not_a_continuation(self):
        """別プレイヤーの支払い不能は別の決定。手番は進んでいる。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:farm", 0)])
        H.own(s, 1, [H.wp("base:farm", 1)])
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        for p in s.players:
            p.money = 0
            p.workers_waiting = 0
            p.workers_training = 2
        s = run_end_phase(s)
        self.assertEqual(s.pending.player_id, 0)
        result = H.result(s, H.sell(H.wp("base:farm", 0)))
        self.assertEqual(result.state.pending.player_id, 1)
        self.assertTrue(result.turn_advanced)

    def test_every_placement_that_resolves_immediately_advances_the_turn(self):
        from naeco.ai.heuristic_agent import HeuristicAgent
        from naeco.engine.observe import observe
        from naeco.engine.state import ActionKind
        for seed in range(6):
            s = H.new(2, seed)
            agents = {p: HeuristicAgent(p) for p in range(2)}
            while s.phase != Phase.GAME_OVER:
                who = game.actor(s)
                was_place = s.pending is None
                action = agents[who].act(observe(s, who, H.CARDS))
                result = H.result(s, action)
                if was_place and action.kind == ActionKind.PLACE_WORKER:
                    # 配置が即時完結したなら、その step は必ず手番を進めている。
                    if result.state.pending is None or (
                            result.state.pending.player_id != who):
                        self.assertTrue(result.turn_advanced)
                s = result.state


class LedgerTests(unittest.TestCase):
    """StepResult.info["tx"] は実際に起きた取引を網羅する(sol J)。"""

    def test_immediate_effects_record_their_transactions(self):
        s = H.place(H.new(), 0, H.wp("base:mine"))          # 直前局面は使わない
        cases = [
            ("base:mine", ["TX-005", "TX-006"]),
            ("base:quarry", ["TX-005", "TX-006", "TX-023"]),
            ("base:school", ["TX-005", "TX-024"]),
        ]
        for card_id, expected in cases:
            result = H.result(H.turn(H.new(), 0), _place(H.wp(card_id)))
            self.assertEqual(sorted(set(result.info["tx"])), sorted(set(expected)),
                             card_id)

    def test_consumer_goods_gain_is_recorded(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:farm")])
        result = H.result(H.turn(s, 0), _place(H.wp("base:farm")))
        self.assertIn("TX-020", result.info["tx"])

    def test_end_phase_transactions_are_recorded(self):
        """賃金支払い・未払い発行・ワーカー回収・ラウンド公開・研修解除まで載る。"""
        s = H.new()
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        s.players[1].workers_training = 1      # 次ラウンド開始で TX-004 が起きる
        s.players[0].money = 10                # 払える → TX-013
        s.players[1].money = 0                 # 払えない → TX-015
        result = H.result(H.turn(s, 0), _place(H.wp("base:mine")))
        recorded = set(result.info["tx"])
        for tx_id in ("TX-016", "TX-013", "TX-015", "TX-018", "TX-004"):
            self.assertIn(tx_id, recorded, tx_id)

    def test_hand_limit_discard_uses_tx_017_exactly_once(self):
        """複数枚捨てても取引は1件(RULES.md 第12章 TX-017)。"""
        s = H.new()
        H.set_hand(s, 0, [], goods=9)
        H.set_hand(s, 1, [])
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        s = H.result(H.turn(s, 0), _place(H.wp("base:mine"))).state
        self.assertEqual(s.pending.kind, DecisionKind.HAND_LIMIT_DISCARD)
        recorded = []
        while s.pending is not None \
                and s.pending.kind == DecisionKind.HAND_LIMIT_DISCARD:
            result = H.result(s, H.legal(s)[0])
            recorded.extend(result.info["tx"])
            s = result.state
        self.assertEqual(recorded.count("TX-017"), 1)
        self.assertEqual(recorded.count("TX-009"), 0)

    def test_a_multi_card_payment_records_tx_009_exactly_once(self):
        """RUL-BUILD-001: コスト支払いの手札廃棄は TX-009 の1回のみで表す。"""
        s = H.new()
        # 製鉄所(コスト4)を大工で建設し、消費財と建物を混ぜて4枚払う。
        H.set_hand(s, 0, [H.wp("base:steel-mill"), H.wp("base:farm"),
                          H.wp("base:cafe")], goods=2)
        s = H.place(s, 0, H.wp("base:carpenter"))
        recorded = list(H.result(s, H.build_action(H.wp("base:steel-mill"))
                                 ).info["tx"])
        s = H.step(s, H.build_action(H.wp("base:steel-mill")))
        while s.pending is not None:
            result = H.result(s, H.legal(s)[0])
            recorded.extend(result.info["tx"])
            s = result.state
        self.assertEqual(recorded.count("TX-009"), 1)
        self.assertEqual(recorded.count("TX-010"), 1)
        self.assertIn(H.wp("base:steel-mill"), s.players[0].owned_buildings)

    def test_design_office_records_tx_022_exactly_once(self):
        """TX-022 は「山札→一時公開」の1件。保持選択で二重計上しない。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:design-office")])
        recorded = list(H.result(H.turn(s, 0),
                                 _place(H.wp("base:design-office"))).info["tx"])
        s = H.place(s, 0, H.wp("base:design-office"))
        self.assertEqual(s.pending.kind, DecisionKind.DESIGN_OFFICE_KEEP)
        result = H.result(s, H.keep(s.temporary_public[0]))
        recorded.extend(result.info["tx"])
        self.assertEqual(recorded.count("TX-022"), 1)

    def test_a_zero_cost_build_records_no_discard(self):
        """開拓民はコスト0なので TX-009 は起きない(OPEN-020)。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:pioneer")])
        H.set_hand(s, 0, [H.wp("base:farm")])
        s = H.place(s, 0, H.wp("base:pioneer"))
        result = H.result(s, H.build_action(H.wp("base:farm")))
        self.assertNotIn("TX-009", result.info["tx"])
        self.assertIn("TX-010", result.info["tx"])


class LedgerEventTests(unittest.TestCase):
    """info["events"] は公開情報の実額を持つ(UI が差分から逆算しないため)。"""

    def test_wage_payment_reports_the_actual_amount(self):
        s = H.new()
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        s.players[1].workers_training = 1
        s.players[0].money = 10
        s.players[1].money = 10
        result = H.result(H.turn(s, 0), _place(H.wp("base:mine")))
        wages = [e for e in result.info["events"] if e["kind"] == "wage_paid"]
        self.assertEqual(len(wages), 2)
        for e in wages:
            self.assertEqual(e["amount"], 2)        # R1 は $2 × 雇用1人

    def test_a_gain_and_a_wage_in_one_step_are_reported_separately(self):
        """所持金の純減から賃金を逆算すると相殺されて誤る(sol 指摘)。"""
        s = H.new()
        H.own(s, 0, [H.wp("base:cafe")])
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.household = 20
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        before = s.players[0].money
        result = H.result(H.turn(s, 0), _place(H.wp("base:cafe")))
        kinds = [e["kind"] for e in result.info["events"]]
        self.assertIn("household_taken", kinds)
        self.assertIn("wage_paid", kinds)
        taken = [e for e in result.info["events"]
                 if e["kind"] == "household_taken"][0]
        paid = [e for e in result.info["events"]
                if e["kind"] == "wage_paid" and e["player_id"] == 0][0]
        self.assertEqual(taken["amount"], 5)
        self.assertEqual(paid["amount"], 2)
        # 所持金の純増減は +3。ここから賃金$2 は復元できない。
        self.assertEqual(result.state.players[0].money - before, 3)

    def test_a_zero_payment_still_records_tx_013(self):
        """RUL-INSOLV-001 は「所持金の全額を家計へ払う」経路にも TX-013 を置く。
        所持金0でもその1件は立つ(TX-013 の定義に非零条件はない)。"""
        s = H.new()
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].money = 0                # 売却可能建物なし・所持金0
        s.players[1].money = 0
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        s.players[1].workers_training = 1
        result = H.result(H.turn(s, 0), _place(H.wp("base:mine")))
        self.assertEqual(result.info["tx"].count("TX-013"), 2)
        self.assertEqual(result.info["tx"].count("TX-015"), 2)
        paid = [e for e in result.info["events"] if e["kind"] == "wage_paid"]
        self.assertEqual([e["amount"] for e in paid], [0, 0])

    def test_a_sale_reports_the_actual_proceeds(self):
        s = H.new()
        # 邸宅などの locked は売却できない(RUL-BUILD-002)ので製鉄所(価値20)を使う。
        H.own(s, 0, [H.wp("base:steel-mill")])
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].money = 0
        s.players[0].workers_waiting = 0
        s.players[0].workers_training = 2
        s.players[1].workers_waiting = 0
        s = run_end_phase(s)
        self.assertEqual(s.pending.kind, DecisionKind.INSOLVENCY_SELL)
        result = H.result(s, H.sell(H.wp("base:steel-mill")))
        sold = [e for e in result.info["events"]
                if e["kind"] == "building_sold"]
        self.assertEqual(len(sold), 1)
        self.assertEqual(sold[0]["amount"], 20)
        self.assertEqual(sold[0]["instance_id"], H.wp("base:steel-mill"))

    def test_unpaid_wages_report_the_issued_count(self):
        s = H.new()
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[0].money = 0
        s.players[1].money = 0
        s.players[0].workers_waiting = 1
        s.players[1].workers_waiting = 0
        s.players[1].workers_training = 1
        result = H.result(H.turn(s, 0), _place(H.wp("base:mine")))
        issued = [e for e in result.info["events"]
                  if e["kind"] == "unpaid_wages_issued"]
        self.assertEqual(sorted(e["count"] for e in issued), [2, 2])


class PlacementReasonTests(unittest.TestCase):
    """RUL-PLACE-001: 置けない理由を説明できる(ユーザー報告 #1/#2/#4)。"""

    def test_a_built_workplace_stays_on_the_board_but_reports_why_it_is_blocked(self):
        """建設直後に手札が枯れると建設系職場は置けない(OPEN-021)が、
        盤面からは消えない。"""
        from naeco.engine.observe import observe
        s = H.new()
        H.own(s, 0, [H.wp("base:construction-company")])
        H.set_hand(s, 0, [])
        obs = observe(H.turn(s, 0), 0, H.CARDS)
        view = [w for w in obs.workplaces
                if w.instance_id == H.wp("base:construction-company")]
        self.assertEqual(len(view), 1)          # 盤面には在る
        self.assertFalse(view[0].can_place)
        self.assertIn("建物がない", view[0].block_reason)

    def test_a_market_that_is_revealed_but_unaffordable_reports_the_household(self):
        from naeco.engine.observe import observe
        s = H.set_round(H.new(), 3)
        H.set_hand(s, 0, [])
        s.household = 10
        obs = observe(H.turn(s, 0), 0, H.CARDS)
        market = [w for w in obs.workplaces
                  if w.instance_id == H.CARDS.instances_by_card["base:round-2"][0]]
        self.assertEqual(len(market), 1)        # R3 の盤面に市場は存在する
        self.assertFalse(market[0].can_place)
        self.assertIn("手札が2枚必要", market[0].block_reason)

    def test_a_sold_building_becomes_a_public_workplace_immediately(self):
        from naeco.engine.observe import observe
        s = H.new()
        H.own(s, 1, [H.wp("base:farm")])
        H.set_hand(s, 0, [])
        H.set_hand(s, 1, [])
        s.players[1].money = 0
        s.players[1].workers_waiting = 0
        s.players[1].workers_training = 2     # 雇用2人 → 賃金$4 を払えない
        s.players[0].workers_waiting = 0
        s = run_end_phase(s)
        self.assertEqual(s.pending.kind, DecisionKind.INSOLVENCY_SELL)
        s = H.step(s, H.sell(H.wp("base:farm")))
        s.players[0].workers_waiting = 1          # 次ラウンドの配置手番にする
        obs = observe(H.turn(s, 0), 0, H.CARDS)
        farm = [w for w in obs.workplaces
                if w.instance_id == H.wp("base:farm")]
        self.assertEqual(len(farm), 1)
        self.assertIsNone(farm[0].owner_offset)         # 公共になった
        self.assertTrue(farm[0].can_place)

    def test_block_reason_is_absent_outside_the_viewers_placement_turn(self):
        """他家の観測に自分の手札由来の差分を出さない(§7.2.1)。"""
        from naeco.engine.observe import observe
        s = H.new()
        H.set_hand(s, 0, [])
        obs = observe(H.turn(s, 0), 1, H.CARDS)     # 手番でない席1から見る
        self.assertTrue(all(w.block_reason is None for w in obs.workplaces))
        self.assertTrue(all(not w.can_place for w in obs.workplaces))


class ScoreBreakdownTests(unittest.TestCase):

    def test_breakdown_totals_match_final_scores(self):
        from naeco.ai.heuristic_agent import HeuristicAgent
        from naeco.engine.observe import observe
        for seed in range(6):
            s = H.new(2, seed)
            agents = {p: HeuristicAgent(p) for p in range(2)}
            while s.phase != Phase.GAME_OVER:
                who = game.actor(s)
                s = H.step(s, agents[who].act(observe(s, who, H.CARDS)))
            breakdown = game.score_breakdown(s, H.CARDS)
            self.assertEqual(tuple(b["total"] for b in breakdown),
                             game.final_scores(s, H.CARDS))
            for pid, b in enumerate(breakdown):
                self.assertEqual(
                    b["total"],
                    b["building_value"] + b["money"] + b["end_game_points"]
                    + b["unpaid_penalty"])

    def test_law_office_forgiveness_appears_in_the_breakdown(self):
        s = H.new()
        H.own(s, 0, [H.wp("base:law-office")])
        s.players[0].unpaid_wages = 7
        b = game.score_breakdown(s, H.CARDS)[0]
        self.assertEqual(b["unpaid_wages"], 7)
        self.assertEqual(b["forgiven_unpaid_wages"], 5)
        self.assertEqual(b["unpaid_penalty"], -6)       # 免除外2枚 × -3


def _place(instance_id):
    from naeco.engine.state import Action, ActionKind
    return Action(ActionKind.PLACE_WORKER, target_instance_id=instance_id)


if __name__ == "__main__":
    unittest.main()
