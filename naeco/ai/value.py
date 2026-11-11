"""探索用の軽量なロールアウト評価。"""
from types import SimpleNamespace

from naeco.engine import game
from naeco.engine.state import Phase


A_HAND = 0.5
A_GOODS = 0.5
A_WORKER = 0.4
A_UNPAID = 3.0
SCALE = 30


def rollout_view(state, cards, pid):
    """HeuristicAgent._score が参照する観測フィールドだけを返す。"""
    player = state.players[pid]
    return SimpleNamespace(
        round_number=state.round_number,
        start_player_offset=(state.start_player - pid) % state.num_players,
        my_money=player.money,
        my_hand_buildings=player.hand_buildings,
        my_hand_goods=player.hand_goods,
        my_owned=player.owned_buildings,
        my_workers=(player.workers_waiting, player.workers_training,
                    state.placed_count(pid)),
        my_unpaid_wages=player.unpaid_wages,
    )


def _potentials(state, cards, scores):
    """席ごとの潜在得点(2 席の既存式と同じ項・同じ計算順)。"""
    values = []
    for pid in range(state.num_players):
        player = state.players[pid]
        hand_potential = sum(
            max(0, (cards.printed_value(instance_id) or 0)
                - (cards.build_cost(instance_id) or 0))
            for instance_id in player.hand_buildings
        )
        remaining = 9 - state.round_number
        values.append(
            scores[pid]["total"]
            + A_HAND * hand_potential
            + A_GOODS * player.hand_goods
            + A_WORKER * state.employed(pid) * remaining
            - A_UNPAID * player.unpaid_wages
        )
    return values


def static_value(state, cards):
    """終端または切詰め局面を [0, 1] の価値ベクトル(絶対席順)で評価する。

    2 席は既存の [first, 1-first](計算順も従来どおり)。3/4 席(Phase 10 D3)は
    終端なら D1 の順位線形効用を (u+1)/2 へ写し、非終端は席ごとの潜在得点と
    他席平均との差を SCALE で [0,1] に clip する軽量 fallback(認定強には使わない)。
    """
    if state.num_players not in (2, 3, 4):
        raise NotImplementedError("static_value is defined for 2..4 players")
    n = state.num_players
    if state.phase == Phase.GAME_OVER:
        utilities = game.terminal_utilities(state, cards)
        return [(utilities[pid] + 1) / 2 for pid in range(n)]
    scores = game.score_breakdown(state, cards)
    values = _potentials(state, cards, scores)
    if n == 2:
        first = 0.5 + (values[0] - values[1]) / (2.0 * SCALE)
        first = max(0.0, min(1.0, first))
        return [first, 1.0 - first]
    out = []
    total = sum(values)
    for pid in range(n):
        others_mean = (total - values[pid]) / (n - 1)
        own = 0.5 + (values[pid] - others_mean) / (2.0 * SCALE)
        out.append(max(0.0, min(1.0, own)))
    return out
