"""Phase 2.5 の隠れ状態サンプラーの受入検査。"""
import dataclasses
import hashlib
import os
import random
import subprocess
import sys
import unittest
from unittest import mock

import helpers as H
import hidden
from naeco.ai import sampler
from naeco.ai.sampler import (
    SamplerError, _unrank_allocation, _validate_constructed_zones,
    sample_hidden_state, total_allocations,
)
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import (
    ActionKind, DecisionKind, KnowledgeKind, clone, derive_seed, encode_rng,
    state_hash,
)
from test_hidden_oracle import TemporaryPublicKnowledge
from test_observe import LeakageTests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AllocationUnranking(unittest.TestCase):
    """小さな人工ビンで rank と割当ての全単射を直接確認する。"""

    def test_every_valid_allocation_appears_exactly_once(self):
        s_items = ("s0", "s1", "s2")
        other_items = ("o0",)
        total = total_allocations(s_items, other_items, 1, (1, 2), (1, 1))
        seen = set()
        for rank in range(total):
            deck, hands = _unrank_allocation(s_items, other_items, 1,
                                              (1, 2), (1, 1), rank)
            self.assertEqual(len(deck), 1)
            self.assertEqual(len(hands[0]), 1)
            self.assertEqual(len(hands[1]), 2)
            key = (tuple(sorted(deck)), tuple(hands[0]), tuple(hands[1]))
            self.assertNotIn(key, seen)
            seen.add(key)
        self.assertEqual(len(seen), total)


class HiddenStateSampler(unittest.TestCase):
    def _check(self, state, viewer, seed=19):
        obs = observe(state, viewer, H.CARDS)
        sampled = sample_hidden_state(obs, seed, H.CARDS)
        self.assertIsNone(hidden.failing_condition(sampled, state, viewer))
        return sampled

    def test_initial_states_for_two_three_and_four_players(self):
        for players in (2, 3, 4):
            state = H.new(players, seed=players)
            self._check(state, game.actor(state), players)

    def test_every_decision_kind_can_be_restored_for_its_actor(self):
        factory = LeakageTests()
        for kind in DecisionKind:
            state = factory._pending_state_for(kind)
            self.assertEqual(state.pending.kind, kind)
            self._check(state, state.pending.player_id, 100 + list(DecisionKind).index(kind))

    def test_empty_deck_and_reshuffled_design_office(self):
        state = H.new()
        state.building_discard = tuple(sorted(state.deck))
        state.deck = ()
        self._check(state, game.actor(state), 41)
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        self.assertGreater(state.reshuffle_seq, 0)
        self._check(state, state.pending.player_id, 42)

    def test_determinism_and_seed_derived_environment_rng(self):
        state = H.new(2, seed=13)
        viewer = game.actor(state)
        obs = observe(state, viewer, H.CARDS)
        first = sample_hidden_state(obs, 123, H.CARDS)
        second = sample_hidden_state(obs, 123, H.CARDS)
        self.assertEqual(state_hash(first), state_hash(second))
        self.assertEqual(first.rng_state,
                         encode_rng(random.Random(derive_seed(123, "det_env", 0))))
        # C9: seed=123 / "det_env" / episode=0 の golden blob SHA-256。
        self.assertEqual(hashlib.sha256(first.rng_state).hexdigest(),
                         "44996f8e29f8daad2a75fb404f3652c826c6d7a20545dca394a1f175a3668ae6")
        self.assertEqual(first.rng_state, second.rng_state)

    def test_only_the_two_permitted_random_constructors_are_used(self):
        state = H.new(2, seed=21)
        obs = observe(state, game.actor(state), H.CARDS)
        original = random.Random
        seeds = []

        def recorded(seed=None):
            seeds.append(seed)
            return original(seed)

        before = random.getstate()
        with mock.patch.object(sampler.random, "Random", side_effect=recorded), \
                mock.patch.object(random, "SystemRandom", side_effect=AssertionError), \
                mock.patch.object(os, "urandom", side_effect=AssertionError):
            sample_hidden_state(obs, 77, H.CARDS)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(seeds, [77, derive_seed(77, "det_env", 0)])

    def test_hash_seed_independent_across_subprocesses(self):
        """同一(obs, seed)が PYTHONHASHSEED 別でも同一 hidden state になる。

        `_unrank_allocation` が set を返すと、反復順序が str ハッシュに依存して
        同一プロセス内の比較では検出できない。set は境界から返さず、deck は
        shuffle 前に確定順序へ変換する(N01)。初期状態と再シャッフル
        (reshuffle_seq>0 / public knowledge)の両経路を別 subprocess で比較する。
        """
        script = (
            "import sys;"
            "import helpers as H;"
            "from naeco.engine import game;"
            "from naeco.engine.cards import load_cards;"
            "from naeco.engine.observe import observe;"
            "from naeco.engine.state import state_hash;"
            "from naeco.ai.sampler import sample_hidden_state;"
            "from test_hidden_oracle import TemporaryPublicKnowledge;"
            "cards=load_cards();"
            "s0=H.new(2, seed=13);"
            "o0=observe(s0, game.actor(s0), cards);"
            "h0=state_hash(sample_hidden_state(o0, 123, cards));"
            "s1=TemporaryPublicKnowledge()._design_office_after_reshuffle();"
            "o1=observe(s1, s1.pending.player_id, cards);"
            "h1=state_hash(sample_hidden_state(o1, 456, cards));"
            "print(h0); print(h1)"
        )
        results = []
        for value in ("0", "1", "2"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = value
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env,
                cwd=os.path.join(ROOT, "tests"),
                timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results.append(tuple(proc.stdout.strip().splitlines()))
        self.assertEqual(len(set(results)), 1, results)

    def test_hash_seed_independent_for_list_field_inputs(self):
        """list 形の全 tuple 宣言フィールドでも PYTHONHASHSEED 別で一致する。"""
        script = (
            "import dataclasses, sys\n"
            "import helpers as H\n"
            "from naeco.engine import game\n"
            "from naeco.engine.cards import load_cards\n"
            "from naeco.engine.observe import observe\n"
            "from naeco.engine.state import state_hash\n"
            "from naeco.ai.sampler import sample_hidden_state\n"
            "from test_hidden_oracle import TemporaryPublicKnowledge\n"
            "cards = load_cards()\n"
            "def as_lists(o):\n"
            "    others = tuple(dataclasses.replace(\n"
            "        v, owned=list(v.owned), workers=list(v.workers))\n"
            "        for v in o.others)\n"
            "    workplaces = tuple(dataclasses.replace(\n"
            "        w, occupant_offsets=list(w.occupant_offsets))\n"
            "        for w in o.workplaces)\n"
            "    p = o.pending\n"
            "    if p is not None:\n"
            "        p = dataclasses.replace(\n"
            "            p, reserved_targets=list(p.reserved_targets),\n"
            "            selected_discards=list(p.selected_discards))\n"
            "    return dataclasses.replace(\n"
            "        o, my_hand_buildings=list(o.my_hand_buildings),\n"
            "        my_owned=list(o.my_owned), my_workers=list(o.my_workers),\n"
            "        others=list(others), building_discard=list(o.building_discard),\n"
            "        workplaces=list(workplaces),\n"
            "        slash_and_burn_load=[list(r) for r in o.slash_and_burn_load],\n"
            "        temporary_public=list(o.temporary_public),\n"
            "        public_knowledge=[list(r) for r in o.public_knowledge],\n"
            "        pending=p, legal_actions=list(o.legal_actions),\n"
            "        draws_since_reshuffle=list(o.draws_since_reshuffle))\n"
            "s0 = H.new(2, seed=13)\n"
            "o0 = observe(s0, game.actor(s0), cards)\n"
            "print(state_hash(sample_hidden_state(as_lists(o0), 123, cards)))\n"
            "s1 = TemporaryPublicKnowledge()._design_office_after_reshuffle()\n"
            "o1 = observe(s1, s1.pending.player_id, cards)\n"
            "print(state_hash(sample_hidden_state(as_lists(o1), 456, cards)))\n"
        )
        results = []
        for value in ("0", "1", "2"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = value
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env,
                cwd=os.path.join(ROOT, "tests"),
                timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results.append(tuple(proc.stdout.strip().splitlines()))
        self.assertEqual(len(set(results)), 1, results)

    def test_precondition_and_infeasible_observation(self):
        state = H.new()
        observer = 1 - game.actor(state)
        with self.assertRaises(ValueError):
            sample_hidden_state(observe(state, observer, H.CARDS), 1, H.CARDS)
        obs = observe(state, game.actor(state), H.CARDS)
        broken = obs.__class__(**dict(obs.__dict__, deck_size=obs.deck_size + 1))
        with self.assertRaises(SamplerError):
            sample_hidden_state(broken, 1, H.CARDS)


class InputValidation(unittest.TestCase):
    """壊れた観測を production 側で拒否する(sol SOL-P25G-001)。

    `invariants.check` は public_knowledge を見ないので、知識集合が壊れた観測は
    構成後の不変条件検査をすり抜ける。サンプラー自身が入口で弾く必要がある。
    """

    def _obs(self, **overrides):
        state = H.new(2, seed=5)
        obs = observe(state, game.actor(state), H.CARDS)
        return dataclasses.replace(obs, **overrides) if overrides else obs

    def _opponent_obs(self, workers):
        obs = self._obs()
        return dataclasses.replace(
            obs, others=(dataclasses.replace(obs.others[0], workers=workers),))

    def _pending_obs(self):
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        return observe(state, state.pending.player_id, H.CARDS)

    def _pending_hand_obs(self):
        """予約札が手札に残る BUILD_COST pending の観測を返す。"""
        state = H.new()
        H.set_hand(state, 0, [H.wp("base:factory")], goods=2)
        state = H.place(state, 0, H.wp("base:carpenter"))
        state = H.step(state, H.build_action(H.wp("base:factory")))
        return observe(state, state.pending.player_id, H.CARDS)

    def _entry(self, instance_id, kind, holder, sequence):
        return (instance_id, kind.value, holder, sequence, 1, "TX-007")

    def _entry_with_round(self, instance_id, kind, holder, sequence, round_number):
        return (instance_id, kind.value, holder, sequence, round_number, "TX-007")

    def test_obs_version_mismatch_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(obs_version="0.2.0"), 1, H.CARDS)

    def test_card_data_mismatch_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(cards_sha256="0" * 64), 1, H.CARDS)

    def test_duplicate_knowledge_entry_is_rejected(self):
        entry = self._entry("base:farm#00", KnowledgeKind.IN_HAND, 1, None)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(entry, entry)), 1, H.CARDS)

    def test_in_deck_since_before_any_reshuffle_is_rejected(self):
        obs = self._obs()
        self.assertEqual(obs.reshuffle_seq, 0)
        entry = self._entry("base:farm#00", KnowledgeKind.IN_DECK_SINCE, None, 0)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_stale_generation_is_rejected(self):
        obs = self._obs(reshuffle_seq=2)
        entry = self._entry("base:farm#00", KnowledgeKind.IN_DECK_SINCE, None, 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_in_hand_with_a_generation_is_rejected(self):
        entry = self._entry("base:farm#00", KnowledgeKind.IN_HAND, 1, 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(entry,)), 1, H.CARDS)

    def test_in_hand_without_a_holder_is_rejected(self):
        entry = self._entry("base:farm#00", KnowledgeKind.IN_HAND, None, None)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(entry,)), 1, H.CARDS)

    def test_in_deck_since_with_a_holder_is_rejected(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry("base:farm#00", KnowledgeKind.IN_DECK_SINCE, 1, 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_string_player_count_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(num_players="2"), 1, H.CARDS)

    def test_boolean_reshuffle_seq_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(reshuffle_seq=True), 1, H.CARDS)

    def test_string_round_number_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(round_number="1"), 1, H.CARDS)

    def test_boolean_draw_count_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(draws_since_reshuffle=(True, 0)),
                                1, H.CARDS)

    def test_player_id_bool_and_out_of_range_are_rejected(self):
        obs = self._obs()
        for bad in (True, -1, 2, 3):
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, player_id=bad), 1, H.CARDS)

    def test_string_end_phase_cursor_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(end_phase_cursor="0"), 1, H.CARDS)

    def test_string_my_money_is_rejected(self):
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(my_money="10"), 1, H.CARDS)

    def test_string_opponent_money_is_rejected(self):
        obs = self._obs()
        view = obs.others[0]
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, others=(dataclasses.replace(view, money="10"),)),
                1, H.CARDS)

    def test_boolean_in_hand_holder_is_rejected(self):
        entry = self._entry("base:farm#00", KnowledgeKind.IN_HAND, True, None)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(entry,)), 1, H.CARDS)

    def test_string_in_hand_holder_is_rejected(self):
        entry = self._entry("base:farm#00", KnowledgeKind.IN_HAND, "x", None)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(entry,)), 1, H.CARDS)

    def test_boolean_in_deck_sequence_is_rejected(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry("base:farm#00", KnowledgeKind.IN_DECK_SINCE,
                            None, True)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_string_pending_remaining_is_rejected(self):
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        obs = observe(state, state.pending.player_id, H.CARDS)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, pending=dataclasses.replace(obs.pending, remaining="1")),
                1, H.CARDS)

    def test_pending_negative_and_impossible_values_are_rejected(self):
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        obs = observe(state, state.pending.player_id, H.CARDS)
        for field in ("remaining", "discount"):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, pending=dataclasses.replace(
                        obs.pending, **{field: -1})),
                    1, H.CARDS)
        for field in ("discarded_goods", "constraint_cost"):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, pending=dataclasses.replace(
                        obs.pending, **{field: -1})),
                    1, H.CARDS)
        for bad in (True, -1, 2, 3):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, pending=dataclasses.replace(
                        obs.pending, actor_offset=bad)),
                    1, H.CARDS)

    def test_pending_source_effect_id_is_validated(self):
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        obs = observe(state, state.pending.player_id, H.CARDS)
        for bad in ("no:such-effect", 1, None):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, pending=dataclasses.replace(
                        obs.pending, source_effect_id=bad)),
                    1, H.CARDS)

    def test_pending_kind_must_be_a_decision_kind(self):
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        obs = observe(state, state.pending.player_id, H.CARDS)
        for bad in ("design_office_keep", "no:such-kind", 1):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, pending=dataclasses.replace(obs.pending, kind=bad)),
                    1, H.CARDS)

    def test_row_with_five_elements_is_rejected(self):
        row = self._entry("base:farm#00", KnowledgeKind.IN_HAND, 1, None)[:5]
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(row,)), 1, H.CARDS)

    def test_row_with_seven_elements_is_rejected(self):
        row = self._entry("base:farm#00", KnowledgeKind.IN_HAND, 1, None) + ("extra",)
        with self.assertRaises(SamplerError):
            sample_hidden_state(self._obs(public_knowledge=(row,)), 1, H.CARDS)

    def test_unknown_instance_id_is_rejected(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry("no:such-card#00", KnowledgeKind.IN_DECK_SINCE, None, 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_non_building_instance_id_is_rejected(self):
        """実体 ID だが building ではない(carpenter は public workplace)。"""
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry("base:carpenter#00", KnowledgeKind.IN_DECK_SINCE, None, 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_zero_round_number_is_rejected(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry_with_round(
            "base:farm#00", KnowledgeKind.IN_DECK_SINCE, None, 1, 0)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_future_round_number_is_rejected(self):
        obs = self._obs()
        entry = self._entry_with_round(
            "base:farm#00", KnowledgeKind.IN_HAND, 1, None,
            obs.round_number + 1)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_boolean_round_number_is_rejected(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry_with_round(
            "base:farm#00", KnowledgeKind.IN_DECK_SINCE, None, 1, True)
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, public_knowledge=(entry,)), 1, H.CARDS)

    def test_public_zone_in_deck_since_is_rejected(self):
        """失効し忘れた IN_DECK_SINCE が公開ゾーン(捨て札)を指す観測を拒否する。"""
        state = H.new()
        H.set_hand(state, 0, [])
        state.building_discard = tuple(sorted(state.deck))
        state.deck = ()
        state = H.place(state, 0, H.wp("base:mine"))
        self.assertGreater(state.reshuffle_seq, 0)
        moved = state.deck[0]
        bad = clone(state)
        bad.deck = bad.deck[1:]
        bad.building_discard = tuple(sorted(bad.building_discard + (moved,)))
        obs = observe(bad, game.actor(bad), H.CARDS)
        self.assertIn(moved, obs.building_discard)
        with self.assertRaises(SamplerError):
            sample_hidden_state(obs, 1, H.CARDS)

    def test_public_zone_in_hand_is_rejected(self):
        """IN_HAND が公開ゾーンへ移動済みの実体を指す観測を拒否する。"""
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        legal = game.legal_actions(state, H.CARDS)
        keep = [a for a in legal if a.kind == ActionKind.SELECT_KEEP][0]
        state = game.step(state, keep, H.CARDS).state
        self.assertTrue(any(e.kind == KnowledgeKind.IN_HAND
                           for e in state.public_knowledge))
        kept = keep.instance_id
        bad = clone(state)
        hand = list(bad.players[0].hand_buildings)
        hand.remove(kept)
        bad.players[0].hand_buildings = tuple(sorted(hand))
        bad.building_discard = tuple(sorted(bad.building_discard + (kept,)))
        obs = observe(bad, game.actor(bad), H.CARDS)
        with self.assertRaises(SamplerError):
            sample_hidden_state(obs, 1, H.CARDS)

    def test_constructed_zone_check_rejects_missing_in_hand(self):
        """production の構成後検査が IN_HAND の実在ゾーン不一致を拒否する。"""
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        legal = game.legal_actions(state, H.CARDS)
        keep = [a for a in legal if a.kind == ActionKind.SELECT_KEEP][0]
        state = game.step(state, keep, H.CARDS).state
        sampled = sample_hidden_state(
            observe(state, game.actor(state), H.CARDS), 1, H.CARDS)
        kept = keep.instance_id
        bad = clone(sampled)
        hand = list(bad.players[0].hand_buildings)
        hand.remove(kept)
        bad.players[0].hand_buildings = tuple(sorted(hand))
        bad.deck = bad.deck + (kept,)
        with self.assertRaises(SamplerError):
            _validate_constructed_zones(bad)

    def test_constructed_zone_check_rejects_public_in_deck_since(self):
        """production の構成後検査が失効済み IN_DECK_SINCE を拒否する。"""
        state = H.new()
        H.set_hand(state, 0, [])
        state.building_discard = tuple(sorted(state.deck))
        state.deck = ()
        state = H.place(state, 0, H.wp("base:mine"))
        sampled = sample_hidden_state(
            observe(state, game.actor(state), H.CARDS), 1, H.CARDS)
        moved = sampled.deck[0]
        bad = clone(sampled)
        bad.deck = bad.deck[1:]
        bad.building_discard = tuple(sorted(bad.building_discard + (moved,)))
        with self.assertRaises(SamplerError):
            _validate_constructed_zones(bad)

    def test_round_trip_holds_for_every_sampled_state(self):
        """C1 を production 側で毎回確認していること(戻り値が観測を再現する)。"""
        for seed in range(6):
            state = H.new(2, seed=seed)
            viewer = game.actor(state)
            obs = observe(state, viewer, H.CARDS)
            sampled = sample_hidden_state(obs, 1000 + seed, H.CARDS)
            self.assertEqual(observe(sampled, viewer, H.CARDS), obs)

    def test_my_workers_wrong_length_is_rejected(self):
        for value in ((), (1,), (1, 2), (1, 2, 3, 4)):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._obs(my_workers=value), 1, H.CARDS)

    def test_my_workers_bad_element_is_rejected(self):
        for value in (("1", 0, 0), (1.0, 0, 0), (True, 0, 0),
                      (-1, 0, 0), (8, 0, 0)):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._obs(my_workers=value), 1, H.CARDS)

    def test_my_workers_non_sequence_is_rejected(self):
        for value in ("2,0,0", 3, None):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._obs(my_workers=value), 1, H.CARDS)

    def test_opponent_workers_wrong_length_is_rejected(self):
        for value in ((), (1,), (1, 2), (1, 2, 3, 4)):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._opponent_obs(value), 1, H.CARDS)

    def test_opponent_workers_bad_element_is_rejected(self):
        for value in (("1", 0, 0), (1.0, 0, 0), (True, 0, 0),
                      (-1, 0, 0), (8, 0, 0)):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._opponent_obs(value), 1, H.CARDS)

    def test_opponent_workers_non_sequence_is_rejected(self):
        for value in ("2,0,0", 3, None):
            with self.assertRaises(SamplerError):
                sample_hidden_state(self._opponent_obs(value), 1, H.CARDS)

    def test_pending_reserved_targets_non_sequence_is_rejected(self):
        for value in ("ab", {"base:farm#00"}, None, 1):
            obs = self._pending_obs()
            obs = dataclasses.replace(
                obs, pending=dataclasses.replace(
                    obs.pending, reserved_targets=value))
            with self.assertRaises(SamplerError):
                sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_reserved_targets_bad_element_is_rejected(self):
        for value in (("no:such-card#00",), ("base:carpenter#00",),
                      (1,), ("base:farm#00", "no:such-card#00")):
            obs = self._pending_obs()
            obs = dataclasses.replace(
                obs, pending=dataclasses.replace(
                    obs.pending, reserved_targets=value))
            with self.assertRaises(SamplerError):
                sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_reserved_targets_duplicate_is_rejected(self):
        first = sorted(H.CARDS.building_instances)[0]
        obs = self._pending_obs()
        obs = dataclasses.replace(
            obs, pending=dataclasses.replace(
                obs.pending, reserved_targets=(first, first)))
        with self.assertRaises(SamplerError):
            sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_reserved_targets_unsorted_is_rejected(self):
        first, second = sorted(H.CARDS.building_instances)[:2]
        self.assertLess(first, second)
        obs = self._pending_obs()
        obs = dataclasses.replace(
            obs, pending=dataclasses.replace(
                obs.pending, reserved_targets=(second, first)))
        with self.assertRaises(SamplerError):
            sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_selected_discards_non_sequence_is_rejected(self):
        for value in ("ab", {"base:farm#00"}, None, 1):
            obs = self._pending_obs()
            obs = dataclasses.replace(
                obs, pending=dataclasses.replace(
                    obs.pending, selected_discards=value))
            with self.assertRaises(SamplerError):
                sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_selected_discards_bad_element_is_rejected(self):
        for value in (("no:such-card#00",), (1,),
                      ("base:farm#00", "no:such-card#00")):
            obs = self._pending_obs()
            obs = dataclasses.replace(
                obs, pending=dataclasses.replace(
                    obs.pending, selected_discards=value))
            with self.assertRaises(SamplerError):
                sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_selected_discards_unsorted_is_rejected(self):
        first, second = sorted(H.CARDS.building_instances)[:2]
        self.assertLess(first, second)
        obs = self._pending_obs()
        obs = dataclasses.replace(
            obs, pending=dataclasses.replace(
                obs.pending, selected_discards=(second, first)))
        with self.assertRaises(SamplerError):
            sample_hidden_state(obs, 1, H.CARDS)

    def test_pending_non_view_is_rejected(self):
        obs = self._pending_obs()
        for bad in ("pending", {"kind": "build_cost"}, (1, 2)):
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, pending=bad), 1, H.CARDS)

    def test_obs_must_be_an_observation(self):
        for bad in ({"legal_actions": ()}, (), "obs", None, 3):
            with self.assertRaises(SamplerError):
                sample_hidden_state(bad, 1, H.CARDS)

    def test_others_element_must_be_an_opponent_view(self):
        obs = self._obs()
        for bad in ((1, 10, 3, 0, (2, 0, 0), 0),
                    {"seat_offset": 1, "money": 10}, "view"):
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, others=(bad,)), 1, H.CARDS)

    def test_workplaces_element_must_be_a_workplace_view(self):
        obs = self._obs()
        for bad in (("base:mine#00", "base:mine", None, "any_player",
                     "unlimited", ()),
                    {"instance_id": "base:mine#00"}, "view"):
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, workplaces=(bad,)), 1, H.CARDS)

    def test_workplace_instance_id_is_validated(self):
        obs = self._obs()
        for bad in (1, "base:warehouse#00", "base:mine#99", "no:such#00"):
            view = dataclasses.replace(obs.workplaces[0], instance_id=bad)
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, workplaces=(view,)), 1, H.CARDS)

    def test_workplace_card_fields_are_validated(self):
        obs = self._obs()
        base = obs.workplaces[0]
        bad = (
            dataclasses.replace(base, logical_id="base:wrong"),
            dataclasses.replace(base, occupancy_mode="unlimited"),
            dataclasses.replace(base, usable_by="owner_only"),
            dataclasses.replace(base, can_place=1),
            dataclasses.replace(base, block_reason=1),
            dataclasses.replace(base, occupant_offsets=(0, 9)),
        )
        for view in bad:
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, workplaces=(view,)), 1, H.CARDS)

    def test_sequence_fields_reject_non_sequence_containers(self):
        obs = self._obs()
        bad = {
            "my_hand_buildings": {"base:farm#00"},
            "my_owned": {"base:farm#00"},
            "building_discard": {"base:farm#00"},
            "temporary_public": {"base:farm#00"},
            "draws_since_reshuffle": {1, 2},
            "legal_actions": "not-actions",
        }
        for name, value in bad.items():
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, **{name: value}), 1, H.CARDS)

    def test_generator_sequence_inputs_are_rejected(self):
        obs = self._obs()
        bad = {
            "my_hand_buildings": iter(obs.my_hand_buildings),
            "draws_since_reshuffle": iter(obs.draws_since_reshuffle),
            "legal_actions": iter(obs.legal_actions),
        }
        for name, value in bad.items():
            with self.assertRaises(SamplerError):
                sample_hidden_state(
                    dataclasses.replace(obs, **{name: value}), 1, H.CARDS)

    def test_slash_and_burn_load_dict_is_rejected(self):
        obs = self._obs()
        with self.assertRaises(SamplerError):
            sample_hidden_state(dataclasses.replace(
                obs, slash_and_burn_load={"base:slash-and-burn#00": 5}),
                1, H.CARDS)

    def test_slash_and_burn_load_bad_rows_are_rejected(self):
        obs = self._obs()
        bad = (
            ("base:slash-and-burn#00", 5),          # row ではなく 1 ペア
            (("base:slash-and-burn#00",),),         # 長さ1
            (("base:slash-and-burn#00", 5, 0),),    # 長さ3
            ((1, 5),),                              # key 非文字列
            (("base:slash-and-burn#00", True),),    # 値 bool
            (("base:slash-and-burn#00", "5"),),     # 値 非int
            (("base:slash-and-burn#00", 0),),       # 値 0
            (("base:slash-and-burn#00", -1),),      # 値 負
            (("base:slash-and-burn#99", 5),),       # 未知実体
            (("base:warehouse#00", 5),),            # 焼畑以外の実体
            (("base:slash-and-burn#00", 5),
             ("base:slash-and-burn#00", 5)),        # 重複 key
        )
        for value in bad:
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, slash_and_burn_load=value), 1, H.CARDS)

    def test_public_knowledge_source_is_validated(self):
        obs = self._obs(reshuffle_seq=1)
        entry = self._entry("base:farm#00", KnowledgeKind.IN_DECK_SINCE, None, 1)
        for bad in ("no:such-source", 1, None):
            with self.assertRaises(SamplerError):
                sample_hidden_state(dataclasses.replace(
                    obs, public_knowledge=(
                        entry[:-1] + (bad,),)), 1, H.CARDS)

    def _as_lists(self, obs):
        others = tuple(dataclasses.replace(
            view, owned=list(view.owned), workers=list(view.workers))
            for view in obs.others)
        workplaces = tuple(dataclasses.replace(
            view, occupant_offsets=list(view.occupant_offsets))
            for view in obs.workplaces)
        pending = obs.pending
        if pending is not None:
            pending = dataclasses.replace(
                pending,
                reserved_targets=list(pending.reserved_targets),
                selected_discards=list(pending.selected_discards))
        return dataclasses.replace(
            obs,
            my_hand_buildings=list(obs.my_hand_buildings),
            my_owned=list(obs.my_owned),
            my_workers=list(obs.my_workers),
            others=list(others),
            building_discard=list(obs.building_discard),
            workplaces=list(workplaces),
            slash_and_burn_load=[list(row) for row in obs.slash_and_burn_load],
            temporary_public=list(obs.temporary_public),
            public_knowledge=[list(row) for row in obs.public_knowledge],
            pending=pending,
            legal_actions=None if obs.legal_actions is None
            else list(obs.legal_actions),
            draws_since_reshuffle=list(obs.draws_since_reshuffle))

    def test_all_sequence_field_lists_preserve_seed_and_state_hash(self):
        states = []
        state = H.new(2, seed=44)
        states.append((state, game.actor(state)))
        state = TemporaryPublicKnowledge()._design_office_after_reshuffle()
        states.append((state, state.pending.player_id))
        state = H.new()
        H.own(state, 0, [H.wp("base:slash-and-burn")])
        state = H.place(state, 0, H.wp("base:slash-and-burn"))
        self.assertTrue(state.slash_and_burn_load)
        states.append((state, game.actor(state)))
        for state, viewer in states:
            obs = observe(state, viewer, H.CARDS)
            list_obs = self._as_lists(obs)
            for seed in (321, 654):
                expected = state_hash(sample_hidden_state(obs, seed, H.CARDS))
                actual = state_hash(
                    sample_hidden_state(list_obs, seed, H.CARDS))
                self.assertEqual(actual, expected)

    def test_worker_lists_are_normalized_for_round_trip(self):
        obs = self._obs()
        obs = dataclasses.replace(
            obs, my_workers=[2, 0, 0],
            others=(dataclasses.replace(obs.others[0], workers=[2, 0, 0]),))
        sampled = sample_hidden_state(obs, 1, H.CARDS)
        expected = dataclasses.replace(
            obs, my_workers=(2, 0, 0),
            others=(dataclasses.replace(obs.others[0], workers=(2, 0, 0)),))
        self.assertEqual(observe(sampled, obs.player_id, H.CARDS), expected)

    def test_pending_sequence_lists_are_normalized_for_round_trip(self):
        obs = self._pending_hand_obs()
        keep = obs.pending.reserved_targets[0]
        obs = dataclasses.replace(
            obs, pending=dataclasses.replace(
                obs.pending, reserved_targets=[keep], selected_discards=[]))
        sampled = sample_hidden_state(obs, 1, H.CARDS)
        expected = dataclasses.replace(
            obs, pending=dataclasses.replace(
                obs.pending, reserved_targets=(keep,), selected_discards=()))
        self.assertEqual(observe(sampled, obs.player_id, H.CARDS), expected)


def _multi_seat_samples(num_players, games=2, every=4):
    """3/4 席の自己対戦(heuristic/random 混在)から actor 観測と元局面を集める。

    pending 中・再シャッフル後は毎決定、それ以外は `every` 決定ごとに採る。
    すべて決定論的(new_game の seed と agent の seed は固定)。
    """
    from naeco.ai.heuristic_agent import HeuristicAgent
    from naeco.ai.random_agent import RandomAgent
    from naeco.engine.state import Phase
    out = []
    for game_index in range(games):
        state = game.new_game(num_players, 500 + game_index, H.CARDS)
        agents = [HeuristicAgent(seat) if seat % 2 == 0
                  else RandomAgent(game_index * 10 + seat)
                  for seat in range(num_players)]
        step = 0
        while state.phase != Phase.GAME_OVER:
            actor = game.actor(state)
            obs = observe(state, actor, H.CARDS)
            if (step % every == 0 or state.pending is not None
                    or state.reshuffle_seq > 0):
                out.append((clone(state), actor, obs, 1000 + step))
            state = game.step(state, agents[actor].act(obs), H.CARDS).state
            step += 1
    return out


class MultiSeatSampler(unittest.TestCase):
    """Phase 10 D3 / B3: 3・4 席の各観測席で C1〜C9 を検査する。

    実装は既存の複数 bin への非復元分配のまま(相手ごとの独立サンプリングは
    しない)。ここでは公開知識・既知手札・予約対象・焼畑負荷・山札・pending を
    含む局面が実際に網羅されたことも数え、網羅が欠けたら落ちる。
    """

    @classmethod
    def setUpClass(cls):
        cls.samples = dict((n, _multi_seat_samples(n)) for n in (3, 4))

    def test_c1_to_c6_and_constraints_for_every_observer_seat(self):
        for n, rows in self.samples.items():
            coverage = set()
            for state, viewer, obs, seed in rows:
                with self.subTest(n=n, viewer=viewer, seed=seed):
                    sampled = sample_hidden_state(obs, seed, H.CARDS)
                    self.assertIsNone(
                        hidden.failing_condition(sampled, state, viewer))
                    # C1(player_id を含む完全一致)
                    restored = observe(sampled, viewer, H.CARDS)
                    self.assertEqual(restored.player_id, viewer)
                    self.assertEqual(restored, obs)
                coverage.add("viewer%d" % viewer)
                if obs.pending is not None:
                    coverage.add("pending")
                    if obs.pending.reserved_targets:
                        coverage.add("reserved")
                if any(row[1] == KnowledgeKind.IN_HAND.value
                       for row in obs.public_knowledge):
                    coverage.add("known_hand")
                if any(row[1] == KnowledgeKind.IN_DECK_SINCE.value
                       for row in obs.public_knowledge):
                    coverage.add("deck_since")
                if obs.reshuffle_seq > 0:
                    coverage.add("reshuffled")
                if obs.temporary_public:
                    coverage.add("temporary_public")
                if obs.slash_and_burn_load:
                    coverage.add("slash_and_burn")
                if obs.deck_size == 0:
                    coverage.add("empty_deck")
            required = set(["viewer%d" % seat for seat in range(n)]) | set([
                "pending", "reserved", "known_hand", "deck_since",
                "reshuffled", "temporary_public", "slash_and_burn"])
            self.assertTrue(required <= coverage,
                            (n, sorted(required - coverage)))

    def test_hidden_hands_actually_vary_across_seeds(self):
        """非復元分配が他家の手札を実際に入れ替えている(山札順だけではない)。"""
        for n, rows in self.samples.items():
            unused_state, unused_viewer, obs, unused_seed = rows[0]
            hands = set()
            for seed in range(8):
                sampled = sample_hidden_state(obs, seed, H.CARDS)
                hands.add(tuple(tuple(sorted(p.hand_buildings))
                                for p in sampled.players))
                # 他家の手札は互いに素(同じ実体が二つの手札に入らない)
                seen = set()
                for player in sampled.players:
                    self.assertFalse(seen & set(player.hand_buildings))
                    seen.update(player.hand_buildings)
            self.assertGreater(len(hands), 1, n)

    def test_c7_same_seed_same_state_and_c9_rng(self):
        for n, rows in self.samples.items():
            for unused_state, viewer, obs, seed in rows[::7]:
                first = sample_hidden_state(obs, seed, H.CARDS)
                second = sample_hidden_state(obs, seed, H.CARDS)
                self.assertEqual(state_hash(first), state_hash(second))
                self.assertEqual(
                    first.rng_state,
                    encode_rng(random.Random(derive_seed(seed, "det_env", 0))))
                # clone/serialize round-trip を壊さない
                self.assertEqual(state_hash(clone(first)), state_hash(first))
                self.assertEqual(observe(clone(first), viewer, H.CARDS), obs)

    def test_c8_environment_rng_is_not_read(self):
        for n, rows in self.samples.items():
            state, unused_viewer, obs, seed = rows[len(rows) // 2]
            rng_before = state.rng_state
            original = random.Random
            seeds = []

            def recorded(value=None):
                seeds.append(value)
                return original(value)

            before = random.getstate()
            with mock.patch.object(sampler.random, "Random",
                                   side_effect=recorded), \
                    mock.patch.object(random, "SystemRandom",
                                      side_effect=AssertionError), \
                    mock.patch.object(os, "urandom",
                                      side_effect=AssertionError):
                sample_hidden_state(obs, seed, H.CARDS)
            self.assertEqual(random.getstate(), before)
            self.assertEqual(seeds, [seed, derive_seed(seed, "det_env", 0)])
            self.assertEqual(state.rng_state, rng_before)

    def test_hash_seed_independent_for_three_and_four_seats(self):
        script = (
            "import helpers as H\n"
            "from naeco.engine import game\n"
            "from naeco.engine.observe import observe\n"
            "from naeco.engine.state import state_hash\n"
            "from naeco.ai.sampler import sample_hidden_state\n"
            "for n in (3, 4):\n"
            "    s = game.new_game(n, 77, H.CARDS)\n"
            "    o = observe(s, game.actor(s), H.CARDS)\n"
            "    print(state_hash(sample_hidden_state(o, 5, H.CARDS)))\n"
        )
        results = []
        for value in ("0", "1"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = value
            proc = subprocess.run(
                [sys.executable, "-c", script], capture_output=True,
                text=True, env=env, cwd=os.path.join(ROOT, "tests"),
                timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            results.append(tuple(proc.stdout.strip().splitlines()))
        self.assertEqual(len(set(results)), 1, results)

    def test_impossible_multi_seat_observations_are_rejected(self):
        for n in (3, 4):
            state = game.new_game(n, 21, H.CARDS)
            obs = observe(state, game.actor(state), H.CARDS)
            forged = [
                # 人数の偽装: 他家ビューや draws の長さと食い違う
                ("num_players up", dict(num_players=n + 1)),
                ("num_players down", dict(num_players=n - 1)),
                ("num_players 5", dict(num_players=5)),
                ("num_players bool", dict(num_players=True)),
                ("num_players float", dict(num_players=float(n))),
                ("drop opponent", dict(others=obs.others[:-1])),
                ("draws length", dict(
                    draws_since_reshuffle=obs.draws_since_reshuffle[:-1])),
                ("player_id range", dict(player_id=n)),
                # 版の偽装
                ("obs_version", dict(obs_version="0.2.0")),
                ("cards sha", dict(cards_sha256="0" * 64)),
                # 隠れ枚数の不整合
                ("deck size", dict(deck_size=obs.deck_size + 1)),
                ("opponent hand", dict(others=tuple(
                    dataclasses.replace(view, hand_size=view.hand_size + 1)
                    if view.seat_offset == n - 1 else view
                    for view in obs.others))),
                ("duplicate offset", dict(others=tuple(
                    dataclasses.replace(view, seat_offset=1)
                    for view in obs.others))),
            ]
            for name, change in forged:
                with self.subTest(n=n, name=name):
                    with self.assertRaises(SamplerError):
                        sample_hidden_state(
                            dataclasses.replace(obs, **change), 1, H.CARDS)

    def test_two_seat_strict_checks_still_reject(self):
        """旧 2 席の厳格検査は維持(多人数対応で緩めていない)。"""
        state = H.new(2, seed=5)
        obs = observe(state, game.actor(state), H.CARDS)
        for change in (dict(num_players=3), dict(obs_version="0.2.0"),
                       dict(draws_since_reshuffle=(0, 0, 0))):
            with self.subTest(change=change):
                with self.assertRaises(SamplerError):
                    sample_hidden_state(dataclasses.replace(obs, **change),
                                        1, H.CARDS)
