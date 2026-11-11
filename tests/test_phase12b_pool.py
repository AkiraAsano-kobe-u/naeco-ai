"""Phase 12b C3: PFSP 卓 sampler・pool manifest・偏り方策(stdlib 部分)。"""
import dataclasses
import hashlib
import json

import pytest

from naeco.ai import biased_agent
from naeco.ai.biased_agent import BiasedAgent, check_card_ids
from naeco.engine import game, observe
from naeco.engine.cards import card_id_of, load_cards
from naeco.engine.state import ActionKind, derive_seed
from naeco.train import pool

# HEAD 9271626(C3 変更前)で採取した旧経路の golden。
LEGACY_SAMPLE_MULTI_SHA = \
    "0a74191d2f2872af5cdb5925903e3a57cc0975f217e8177c5f37d8a6a958a34f"
LEGACY_SAMPLE_SHA = \
    "1f4a4768b171a739edd9b86c24d1585dff725272b35dbba256a128b5dce6c0c0"


def _legacy_pool():
    latest = pool.PoolEntry("latest", "checkpoint", "latest", 1.0)
    past = [pool.PoolEntry("gen%03d" % i, "checkpoint", "p%d" % i, 1.0)
            for i in range(7)]
    return pool.training_pool(latest, past)


def test_legacy_sample_multi_and_sample_golden_unchanged():
    p = _legacy_pool()
    draws = [p.sample_multi(123, i, j).name for i in range(300) for j in (1, 2, 3)]
    two = [p.sample(77, i).name for i in range(300)]
    assert hashlib.sha256(json.dumps(draws).encode()).hexdigest() == \
        LEGACY_SAMPLE_MULTI_SHA
    assert hashlib.sha256(json.dumps(two).encode()).hexdigest() == \
        LEGACY_SAMPLE_SHA


def test_pfsp_weights_match_hand_calculation():
    # n=3, p=(0.5, 0.2, 0.0, None): x=(1.0, 0.6, 0.0, 0.5)
    # raw=(max(.05,0)=.05, .4^2=.16, 1.0, .5^2=.25), total=1.46
    w = pool.pfsp_weights([0.5, 0.2, 0.0, None], 3)
    expected = [0.05 / 1.46, 0.16 / 1.46, 1.0 / 1.46, 0.25 / 1.46]
    assert w == pytest.approx(expected, abs=1e-15)
    # n=4, p=0.1 -> x=0.4 -> 0.36; p=0.3 -> x=1.0 (clip) -> floor 0.05
    w4 = pool.pfsp_weights([0.1, 0.3], 4)
    assert w4 == pytest.approx([0.36 / 0.41, 0.05 / 0.41], abs=1e-15)
    assert pool.pfsp_x(None, 4) == 0.5
    with pytest.raises(ValueError):
        pool.pfsp_weights([1.2], 3)
    with pytest.raises(ValueError):
        pool.pfsp_weights([], 3)


def _manifest(n=3):
    return {
        "schema": pool.POOL_SCHEMA, "num_players": n,
        "categories": {"latest": 0.5, "pfsp": 0.35, "biased": 0.15},
        "entries": [{"name": "initial", "kind": "checkpoint",
                     "path": "x/initial.meta.json",
                     "sha256": {"meta": "a" * 64, "weights": "b" * 64}}],
        "pfsp_tables": [
            {"id": "past-past", "opponents": ["@past"] * (n - 1)},
            {"id": "past-heur", "opponents": ["@past"] + ["heuristic"] * (n - 2)},
            {"id": "initial", "opponents": ["initial"] * (n - 1)}],
        "biased_tables": [
            {"id": "school", "opponents": ["biased:school_first"] * (n - 1)},
            {"id": "mine", "opponents": ["biased:mine_first"] * (n - 1)}],
        "pfsp": {"interval_generations": 100, "clusters": 64,
                 "diag_start_index": 20100000, "diag_max_index": 20199999,
                 "floor": 0.05, "unmeasured_x": 0.5},
        "tie_break": "Action.sort_key ascending",
        "seed_domain": "derive_seed(train root, pfsp-table, index)"}


def test_manifest_validation_rejects_tampering():
    pool.validate_pool_manifest(_manifest(3), 3)
    pool.validate_pool_manifest(_manifest(4), 4)
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(_manifest(3), 4)
    bad = _manifest()
    bad["categories"] = {"latest": 0.5, "pfsp": 0.25, "biased": 0.25}
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)
    bad = _manifest()
    bad["extra"] = 1
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)
    bad = _manifest()
    bad["pfsp_tables"][0]["opponents"] = ["@past"]
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)
    bad = _manifest()
    bad["pfsp_tables"][2]["opponents"] = ["ghost", "ghost"]
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)
    bad = _manifest()
    bad["biased_tables"][0]["opponents"] = ["heuristic", "biased:mine_first"]
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)
    bad = _manifest()
    bad["pfsp"]["floor"] = 0.01
    with pytest.raises(ValueError):
        pool.validate_pool_manifest(bad, 3)


def _sampler(n=3, weights=(0.5, 0.3, 0.2)):
    m = _manifest(n)
    return pool.TableSampler(n, m["pfsp_tables"], list(weights),
                             m["biased_tables"], ["age0100", "fork"])


def test_table_categories_and_pfsp_weights_frequency():
    s = _sampler()
    counts = {"latest": 0, "pfsp": 0, "biased": 0}
    tables = {}
    for i in range(20000):
        category, table_id, seats = s.sample(11, i)
        counts[category] += 1
        if category == "pfsp":
            tables[table_id] = tables.get(table_id, 0) + 1
        assert len(seats) == 2 and "@past" not in seats
    assert 0.49 < counts["latest"] / 20000 < 0.51
    assert 0.34 < counts["pfsp"] / 20000 < 0.36
    assert 0.14 < counts["biased"] / 20000 < 0.16
    total = float(sum(tables.values()))
    assert abs(tables["past-past"] / total - 0.5) < 0.02
    assert abs(tables["initial"] / total - 0.2) < 0.02


def test_table_sampler_depends_only_on_coordinates():
    a = _sampler()
    b = _sampler()
    forward = [a.sample(5, i) for i in range(500)]
    backward = [b.sample(5, i) for i in reversed(range(500))][::-1]
    assert forward == backward
    assert [a.sample(6, i) for i in range(50)] != forward[:50]


def _states(n, count=40):
    cards = load_cards()
    for seed in range(count):
        state = game.new_game(n, derive_seed(seed, "env", 0), cards,
                              setup_seed=derive_seed(seed, "setup", 0))
        yield cards, state


def test_biased_card_ids_exist_in_cards_json():
    cards = load_cards()
    assert check_card_ids(cards)
    hooks = dict((cid, [e["hook"] for e in cards.defs[cid].effects])
                 for cid in biased_agent.HIRING_CARD_IDS)
    assert all(any(h in biased_agent.HIRING_HOOKS for h in v) for v in hooks.values())
    with pytest.raises(ValueError):
        BiasedAgent("unknown")


def test_biased_agents_follow_bias_and_heuristic_fallback():
    seen = {"school_first": 0, "mine_first": 0, "avoid_hiring": 0}
    for n in (3, 4):
        for cards, state in _states(n):
            st = state
            for unused in range(30):
                who = game.actor(st)
                if who is None:
                    break
                obs = observe.observe(st, who, cards)
                legal = obs.legal_actions
                for kind in seen:
                    action = BiasedAgent(kind).act(obs)
                    assert action in legal
                    places = [a for a in legal
                              if a.kind == ActionKind.PLACE_WORKER and obs.pending is None]
                    ids = [card_id_of(a.target_instance_id) for a in places]
                    if kind == "school_first" and "base:school" in ids:
                        assert card_id_of(action.target_instance_id) == "base:school"
                        seen[kind] += 1
                    if kind == "mine_first" and "base:mine" in ids:
                        assert card_id_of(action.target_instance_id) == "base:mine"
                        seen[kind] += 1
                    if kind == "avoid_hiring" and places and any(
                            i in biased_agent.HIRING_CARD_IDS for i in ids) and any(
                            i not in biased_agent.HIRING_CARD_IDS for i in ids):
                        assert card_id_of(action.target_instance_id) not in \
                            biased_agent.HIRING_CARD_IDS
                        seen[kind] += 1
                st = game.step(st, BiasedAgent("avoid_hiring").act(obs), cards).state
    assert all(v > 0 for v in seen.values()), seen


def test_biased_agent_reads_only_observation_and_is_deterministic():
    for cards, state in _states(3, 5):
        obs = observe.observe(state, game.actor(state), cards)
        a = BiasedAgent("mine_first", seed=1).act(obs)
        b = BiasedAgent("mine_first", seed=99).act(obs)
        assert a == b
        # 観測を書き換えれば結果も観測だけから決まる(合法手を絞ると従う)
        narrowed = dataclasses.replace(obs, legal_actions=(obs.legal_actions[-1],))
        assert BiasedAgent("school_first").act(narrowed) == obs.legal_actions[-1]
