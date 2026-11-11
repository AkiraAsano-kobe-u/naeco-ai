"""Phase 12b C3: 探索モード C(他家を方策混合サンプル)と旧探索 golden。"""
import hashlib
import json
import sys
import types

import pytest

from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.biased_agent import BiasedAgent
from naeco.ai.search import ismcts as ismcts_mod
from naeco.ai.search.evaluator import (
    ActionPolicy, DeterministicPolicy, EvaluatorPolicy, PolicyMixture,
    UniformEvaluator, checked_probabilities)
from naeco.ai.search.ismcts import IsmctsAgent
from naeco.engine import game, observe
from naeco.engine.cards import load_cards
from naeco.engine.state import derive_seed

# HEAD 9271626(C3 変更前)で採取した 2 人探索の golden。
LEGACY_ISMCTS_N2_SHA = \
    "bc9c3a18a706f143be31e9598b693843d428c2fefa245277fac39919fcbae828"


def _new(n, seed):
    cards = load_cards()
    return cards, game.new_game(n, derive_seed(seed, "env", 0), cards,
                                setup_seed=derive_seed(seed, "setup", 0))


def test_two_player_search_golden_unchanged():
    rows = []
    for seed in range(4):
        cards, state = _new(2, seed)
        for evaluator in (None, UniformEvaluator(0.5)):
            st = state
            picks = []
            for step in range(6):
                obs = observe.observe(st, game.actor(st), cards)
                agent = IsmctsAgent(seed * 10 + step, 60, cards=cards,
                                    evaluator=evaluator)
                action = agent.act(obs)
                picks.append([list(action.sort_key()), agent.stats.transitions,
                              agent.stats.determinizations])
                st = game.step(st, action, cards).state
            rows.append(picks)
    digest = hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()
    assert digest == LEGACY_ISMCTS_N2_SHA


def _mixture():
    return PolicyMixture([
        ("heuristic", DeterministicPolicy(HeuristicAgent(0)), 0.5),
        ("uniform", EvaluatorPolicy(UniformEvaluator(0.5)), 0.3),
        ("mine", DeterministicPolicy(BiasedAgent("mine_first")), 0.2)])


def _root_obs(n, seed):
    cards, state = _new(n, seed)
    st = state
    while True:
        obs = observe.observe(st, game.actor(st), cards)
        if obs.pending is None and len(obs.legal_actions) > 1:
            return cards, obs
        st = game.step(st, HeuristicAgent(0).act(obs), cards).state


@pytest.mark.parametrize("n", [3, 4])
def test_mode_c_is_deterministic_and_reports_costs(n):
    cards, obs = _root_obs(n, 3)
    picks = []
    for unused in range(2):
        agent = IsmctsAgent(7, 1500, cards=cards, evaluator=UniformEvaluator(0.5),
                            opponent_mixture=_mixture())
        action = agent.act(obs)
        picks.append((action, agent.last_mixture_id, agent.stats.transitions,
                      agent.stats.evaluations, agent.opponent_policy_calls))
        assert action in obs.legal_actions
        assert agent.last_mixture_id in ("heuristic", "uniform", "mine")
        assert agent.opponent_policy_calls > 0
        assert agent.stats.transitions <= 1500
    assert picks[0] == picks[1]


def test_mode_c_mixture_choice_varies_by_decision_and_uses_weights():
    cards, unused = _root_obs(3, 0)
    counts = {}
    mixture = _mixture()
    for seed in range(4000):
        cid = mixture.sample_component(seed)[0]
        counts[cid] = counts.get(cid, 0) + 1
    assert abs(counts["heuristic"] / 4000.0 - 0.5) < 0.03
    assert abs(counts["mine"] / 4000.0 - 0.2) < 0.03


def test_mode_c_rejects_two_players_and_bad_inputs():
    cards, obs2 = _root_obs(2, 1)
    agent = IsmctsAgent(1, 50, cards=cards, evaluator=UniformEvaluator(0.5),
                        opponent_mixture=_mixture())
    with pytest.raises(ValueError):
        agent.act(obs2)
    with pytest.raises(ValueError):
        IsmctsAgent(1, 50, cards=cards, opponent_mixture=[("x", None, 1.0)])
    with pytest.raises(ValueError):
        PolicyMixture([])
    with pytest.raises(ValueError):
        PolicyMixture([("a", DeterministicPolicy(HeuristicAgent(0)), 0.0)])
    with pytest.raises(ValueError):
        PolicyMixture([("a", DeterministicPolicy(HeuristicAgent(0)), 1.0),
                       ("a", DeterministicPolicy(HeuristicAgent(0)), 1.0)])


class _Broken(ActionPolicy):
    def __init__(self, how):
        self.how = how

    def action_probabilities(self, observation):
        legal = list(observation.legal_actions)
        p = [1.0 / len(legal)] * len(legal)
        if self.how == "sum":
            p[0] += 0.5
        elif self.how == "negative":
            p[0] = -p[0]
            p[1] += 2 * (1.0 / len(legal))
        elif self.how == "nan":
            p[0] = float("nan")
        elif self.how == "missing":
            return list(zip(legal[1:], [1.0 / (len(legal) - 1)] * (len(legal) - 1)))
        elif self.how == "duplicate":
            return list(zip(legal, p)) + [(legal[0], 0.0)]
        return list(zip(legal, p))


@pytest.mark.parametrize("how", ["sum", "negative", "nan", "missing", "duplicate"])
def test_broken_probabilities_are_rejected(how):
    cards, obs = _root_obs(3, 2)
    with pytest.raises(ValueError):
        checked_probabilities(_Broken(how), obs)
    agent = IsmctsAgent(3, 1500, cards=cards, evaluator=UniformEvaluator(0.5),
                        opponent_mixture=PolicyMixture([("b", _Broken(how), 1.0)]))
    with pytest.raises(ValueError):
        agent.act(obs)


class _Spy(ActionPolicy):
    def __init__(self):
        self.seen = []

    def action_probabilities(self, observation):
        self.seen.append(observation)
        legal = observation.legal_actions
        return [(a, 1.0 / len(legal)) for a in legal]


def test_mode_c_gives_opponent_policy_only_actor_observations():
    cards, obs = _root_obs(3, 5)
    spy = _Spy()
    agent = IsmctsAgent(5, 1500, cards=cards, evaluator=UniformEvaluator(0.5),
                        opponent_mixture=PolicyMixture([("spy", spy, 1.0)]))
    agent.act(obs)
    assert spy.seen
    for seen in spy.seen:
        assert type(seen).__name__ == "Observation"
        assert seen.player_id != obs.player_id
        assert seen.legal_actions
    # 方策の呼出は根の手番席以外だけ。
    assert agent.opponent_policy_calls == len(spy.seen)


def test_mode_c_search_imports_no_torch():
    assert "torch" not in sys.modules or not any(
        getattr(m, "__name__", "").startswith("naeco.ai.search") and
        "torch" in getattr(m, "__dict__", {})
        for m in list(sys.modules.values()) if isinstance(m, types.ModuleType))
    import naeco.ai.search.evaluator as ev
    assert "torch" not in ev.__dict__ and "torch" not in ismcts_mod.__dict__


def test_default_path_without_mixture_is_unchanged_for_three_players():
    cards, obs = _root_obs(3, 4)
    a = IsmctsAgent(4, 200, cards=cards, evaluator=UniformEvaluator(0.5))
    b = IsmctsAgent(4, 200, cards=cards, evaluator=UniformEvaluator(0.5),
                    opponent_mixture=None)
    assert a.act(obs) == b.act(obs)
    assert a.stats.transitions == b.stats.transitions
    assert a.opponent_policy_calls == 0 and a.last_mixture_id is None
