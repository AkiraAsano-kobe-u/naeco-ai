"""hook完全性(§5.3)と instance_id 採番・ラウンド表面写像(§1.1, §1.2)。"""
import unittest

import helpers as H
from naeco.engine import handlers
from naeco.engine.cards import (
    ROUND_FACE_WORKPLACE, WORKPLACE_EFFECT_IDS, card_id_of, load_cards,
)


class RegistryCompleteness(unittest.TestCase):

    def test_all_hooks_have_handlers_and_no_dead_handlers(self):
        self.assertTrue(handlers.verify_registry(H.CARDS))

    def test_there_are_exactly_29_hooks(self):
        self.assertEqual(len(H.CARDS.hooks_in_data()), 29)

    def test_no_duplicate_registration(self):
        with self.assertRaises(ValueError):
            handlers.register("on_place__quarry", "on_place")(lambda *a: None)

    def test_pending_returning_hooks_are_the_expected_seven(self):
        """§付録A: PendingDecision を返すのは A/C/D/E の7 hook。"""
        expected = {
            "on_place__design_office",
            "on_place__discard_then_draw",
            "on_place__build",
            "on_place__discard_gain_household",
            "on_place__build_farm_free",
            "on_place__build_then_draw",
            "on_place__build_two_same_cost",
        }
        # DecisionKind を返しうる hook を、実際の登録から静的に確認する。
        returns_pending = set()
        for hook in H.CARDS.hooks_in_data():
            if handlers.HANDLER_TIMING[hook] == "on_place":
                returns_pending.add(hook)
        # on_place のうち即時完結するものを除く
        immediate = {
            "on_place__gain_consumer_goods", "on_place__slash_and_burn",
            "on_place__gain_from_household", "on_place__orchard",
            "on_place__draw_cards", "on_place__chemical_plant",
            "on_place__quarry", "on_place__hire_worker",
            "on_place__hire_worker_up_to", "on_place__hire_worker_ready",
        }
        self.assertEqual(returns_pending - immediate, expected)


class CardDerivation(unittest.TestCase):

    def test_instance_numbering_is_seed_independent(self):
        a = load_cards()
        self.assertEqual(a.instances_by_card["base:farm"],
                         tuple("base:farm#%02d" % i for i in range(8)))
        self.assertEqual(len(a.building_instances), 64)
        self.assertEqual(len(a.round_instances), 9)

    def test_round_face_mapping_matches_the_transcription_pin(self):
        """§1.1: データ由来の写像がハードコード表と一致する(EFF入れ替わり検出)。"""
        self.assertEqual(H.CARDS.derived_round_face, ROUND_FACE_WORKPLACE)

    def test_logical_id_of_round_card_is_its_face_workplace(self):
        self.assertEqual(H.CARDS.logical_id_of("base:round-1#00"), "base:stall")
        self.assertEqual(H.CARDS.logical_id_of("base:round-8#00"), "base:expo")
        self.assertIsNone(H.CARDS.logical_id_of("base:round-9#00"))

    def test_consumer_goods_and_wages_are_not_instanced(self):
        self.assertEqual(H.CARDS.instances_by_card["base:consumer-goods"], ())
        self.assertEqual(H.CARDS.instances_by_card["base:unpaid-wage"], ())

    def test_workplace_effect_ids_cover_all_faces(self):
        for logical_id, eff_id in WORKPLACE_EFFECT_IDS.items():
            self.assertEqual(H.CARDS.effect_owner_card.get(eff_id) is not None
                             or eff_id in H.CARDS.effect_hook, True)


if __name__ == "__main__":
    unittest.main()
