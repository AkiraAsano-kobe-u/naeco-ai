"""テスト用の局面構築ヘルパー。

不変条件(特に INV-002a 建物64実体の保存則)を壊さずに手札・所有建物を
差し替えるため、カードは必ずどこかのゾーンから移動させる。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.engine import game, invariants, ops           # noqa: E402
from naeco.engine.cards import load_cards                # noqa: E402
from naeco.engine.state import (                         # noqa: E402
    Action, ActionKind, Phase, WorkplaceState,
)

CARDS = load_cards()


def new(num_players=2, seed=0, start_player=0):
    """局面テスト用の初期状態。

    開始プレイヤーは既定で**席0に固定**する。本番の `new_game` は RUL-SETUP-001
    に従い seed から公平に抽選するが、シナリオテストは「席0が先手」という固定の
    幾何を前提に書かれており、抽選を混ぜると検証したい効果と無関係な差分が入る。
    抽選そのものは tests/test_boundaries.py の開始プレイヤー抽選テストで検証する。
    """
    return game.new_game(num_players, seed, CARDS, start_player=start_player)


def _extract(s, instance_ids):
    """指定実体を山札または各手札から取り除く(建物の総数は変えない)。"""
    want = list(instance_ids)
    for instance_id in want:
        if instance_id in s.deck:
            s.deck = tuple(x for x in s.deck if x != instance_id)
            continue
        moved = False
        for p in s.players:
            if instance_id in p.hand_buildings:
                hand = list(p.hand_buildings)
                hand.remove(instance_id)
                p.hand_buildings = tuple(hand)
                moved = True
                break
        if not moved:
            raise AssertionError("%s is not in deck or any hand" % instance_id)


def set_hand(s, pid, buildings=(), goods=0):
    """手札を丸ごと置き換える。元の手札は山札の末尾へ戻す。"""
    p = s.players[pid]
    s.deck = s.deck + p.hand_buildings
    p.hand_buildings = ()
    _extract(s, buildings)
    p.hand_buildings = tuple(sorted(buildings))
    p.hand_goods = goods
    return s


def own(s, pid, buildings):
    """建物を所有状態にする(職場なら workplaces にも登録する)。"""
    _extract(s, buildings)
    p = s.players[pid]
    p.owned_buildings = tuple(sorted(p.owned_buildings + tuple(buildings)))
    for instance_id in buildings:
        cd = CARDS.of_instance(instance_id)
        if cd.workplace is not None:
            s.workplaces[instance_id] = WorkplaceState(
                instance_id, CARDS.logical_id_of(instance_id), pid,
                "owner_only", cd.occupancy_mode, ())
    return s


def reveal_round(s, n):
    """ラウンドカード n の表面を公共職場として公開する(TX-018 と同じ扱い)。"""
    instance_id = CARDS.instances_by_card["base:round-%d" % n][0]
    if instance_id in s.workplaces:
        return s
    s.round_deck = tuple(x for x in s.round_deck if x != instance_id)
    cd = CARDS.of_instance(instance_id)
    s.workplaces[instance_id] = WorkplaceState(
        instance_id, CARDS.logical_id_of(instance_id), None,
        "any_player", cd.occupancy_mode, ())
    return s


def set_round(s, n):
    """ラウンド n の配置フェーズ相当にする(表面 1..n-1 を公開済みにする)。"""
    for k in range(1, n):
        reveal_round(s, k)
    s.round_number = n
    return s


def turn(s, pid):
    s.phase = Phase.PLACE
    s.current_player = pid
    return s


def place(s, pid, instance_id):
    """pid が instance_id へワーカーを置く(合法性はエンジンが検証する)。"""
    turn(s, pid)
    return step(s, Action(ActionKind.PLACE_WORKER, target_instance_id=instance_id))


def step(s, action):
    return game.step(s, action, CARDS).state


def result(s, action):
    return game.step(s, action, CARDS)


def legal(s):
    return game.legal_actions(s, CARDS)


def build_action(instance_id):
    return Action(ActionKind.SELECT_BUILD, instance_id=instance_id)


def discard_goods(k):
    return Action(ActionKind.SELECT_DISCARD, goods_count=k)


def discard_building(instance_id):
    return Action(ActionKind.SELECT_DISCARD, instance_id=instance_id)


def keep(instance_id):
    return Action(ActionKind.SELECT_KEEP, instance_id=instance_id)


def sell(instance_id):
    return Action(ActionKind.SELL_BUILDING, instance_id=instance_id)


def wp(card_id, seq=0):
    return "%s#%02d" % (card_id, seq)


def check(s, scope="always"):
    invariants.check(s, CARDS, scope)
    return s


# --- torch の有無 -------------------------------------------------------
#
# 中・強(gen532)を構築するテストは torch を要する。torch の無い機械は
# **正常な利用形態**である(README のとおり弱は標準ライブラリだけで遊べる)。
# そこで fail ではなく skip にする。fail のままだと、clone してテストを
# 回した人が「壊れている」と誤解する。import せずに find_spec で判定する
# のは、弱だけの経路で torch を読み込まない規律に合わせるため。
import importlib.util as _importlib_util
import unittest as _unittest

TORCH_AVAILABLE = _importlib_util.find_spec("torch") is not None

requires_torch = _unittest.skipUnless(
    TORCH_AVAILABLE,
    "torch が無い環境。中・強の構築には torch が要る(弱は torch 無しで動く)")
