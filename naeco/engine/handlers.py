"""効果ハンドラの登録制(設計書 §5)。

cards.json の hook 名 → 実装。データ側には式もPython名も持ち込まない(§5.2)。
新しい挙動が要るときは params を増やすのではなく hook を増やす。

timing 別の返り値契約(core-api.md §5.1 と一致。sol Phase1 #5):
  on_place        (state, ctx, params) -> Optional[PendingDecision]  # None=即時完結
  on_round_start  (state, ctx, params) -> None
  passive         呼ばれない(恒常効果。参照側が所有枚数を数える)
  on_game_end     (state, ctx, params) -> Tuple[int, int]
                    = (加算得点, 免除する未払い賃金枚数)。得点系は (points, 0)、
                      法律事務所のみ (0, forgive)。final_scores が集計する。
"""
import dataclasses
from typing import Callable, Dict, Optional

from . import cards as cards_mod
from . import ops
from .invariants import worker_cap
from .state import DecisionKind, EffectContextError, PendingDecision

HANDLERS = {}        # type: Dict[str, Callable]
HANDLER_TIMING = {}  # type: Dict[str, str]


def register(hook, timing):
    """デコレータ。同一 hook の二重登録はエラー(§5.1)。"""
    def deco(fn):
        if hook in HANDLERS:
            raise ValueError("duplicate handler registration: %s" % hook)
        HANDLERS[hook] = fn
        HANDLER_TIMING[hook] = timing
        return fn
    return deco


@dataclasses.dataclass(frozen=True)
class EffectContext:
    controller: int
    effect_id: str
    source_instance_id: str
    logical_id: str
    cards: object
    # TX-* の記録先(StepResult.info["tx"])。None なら記録しない(sol J)。
    tx: Optional[list] = None


# ============================================================ on_place

@register("on_place__gain_consumer_goods", "on_place")
def _gain_consumer_goods(state, ctx, params):
    """農場(2枚) / 大農園(3枚)。TX-020。"""
    ops.gain_goods(state, ctx.controller, params["amount"], ctx.tx)
    return None


@register("on_place__slash_and_burn", "on_place")
def _slash_and_burn(state, ctx, params):
    """焼畑: 消費財5枚をカード上へ置く(この時点では獲得しない)。TX-025。"""
    key = ctx.source_instance_id
    state.slash_and_burn_load[key] = state.slash_and_burn_load.get(key, 0) + params["amount"]
    ops.tx(ctx.tx, "TX-025")
    return None


@register("on_place__gain_from_household", "on_place")
def _gain_from_household(state, ctx, params):
    """珈琲店: 家計から$5。配置前提で家計残額を保証済み(OPEN-003)。"""
    ops.take_from_household(state, ctx.controller, params["amount"], ctx.tx)
    return None


@register("on_place__design_office", "on_place")
def _design_office(state, ctx, params):
    """設計事務所: 最大5枚公開し1枚を手札へ。公開0枚なら pending を作らず空振り。"""
    revealed = ops.reveal_to_temporary_public(state, ctx.cards, params["amount"], ctx.tx)
    if not revealed:
        return None
    return PendingDecision(
        kind=DecisionKind.DESIGN_OFFICE_KEEP,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=1,
    )


@register("on_place__discard_then_draw", "on_place")
def _discard_then_draw(state, ctx, params):
    """工場(2捨て4引き) / 自動車工場(3捨て7引き)。捨て札の完了後に引く。"""
    return PendingDecision(
        kind=DecisionKind.DISCARD_N,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=params["discard"],
    )


@register("on_place__orchard", "on_place")
def _orchard(state, ctx, params):
    """果樹園: 手札が4枚未満の間、消費財で4枚にする。4枚以上なら空振り(OPEN-007)。"""
    p = state.players[ctx.controller]
    ops.gain_goods(state, ctx.controller,
                   max(0, params["target_total"] - p.hand_total), ctx.tx)
    return None


@register("on_place__build", "on_place")
def _build(state, ctx, params):
    """大工(discount=0) / 建設会社(discount=1)。"""
    return PendingDecision(
        kind=DecisionKind.BUILD_TARGET,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=1,
        discount=params.get("discount", 0),
    )


@register("on_place__discard_gain_household", "on_place")
def _discard_gain_household(state, ctx, params):
    """露店〜万博・レストラン: n枚捨てて家計から定額を得る。"""
    return PendingDecision(
        kind=DecisionKind.DISCARD_N,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=params["discard"],
    )


@register("on_place__build_farm_free", "on_place")
def _build_farm_free(state, ctx, params):
    """開拓民: farm属性の建物1枚をコスト0で建設する。"""
    return PendingDecision(
        kind=DecisionKind.BUILD_TARGET,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=1,
    )


@register("on_place__build_then_draw", "on_place")
def _build_then_draw(state, ctx, params):
    """ゼネコン: 通常建設1回の後に2枚引く。継続は source_effect_id で分岐する(§3.3)。"""
    return PendingDecision(
        kind=DecisionKind.BUILD_TARGET,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=1,
    )


@register("on_place__draw_cards", "on_place")
def _draw_cards(state, ctx, params):
    """製鉄所(3枚) / 鉱山(1枚)。"""
    ops.draw(state, ctx.cards, ctx.controller, params["amount"], ctx.tx)
    return None


@register("on_place__chemical_plant", "on_place")
def _chemical_plant(state, ctx, params):
    """化学工場: 解決開始時の手札が0枚なら4枚、その他は2枚引く。"""
    p = state.players[ctx.controller]
    ops.draw(state, ctx.cards, ctx.controller, 4 if p.hand_total == 0 else 2, ctx.tx)
    return None


@register("on_place__build_two_same_cost", "on_place")
def _build_two_same_cost(state, ctx, params):
    """ニコイチ建設: 同一コスト2枚を、コスト1回分の支払いで建設する。"""
    return PendingDecision(
        kind=DecisionKind.BUILD_TARGET,
        player_id=ctx.controller,
        source_effect_id=ctx.effect_id,
        remaining=2,
    )


@register("on_place__quarry", "on_place")
def _quarry(state, ctx, params):
    """採石場: 1枚引き、開始プレイヤーマーカーを得る(次ラウンドから)。TX-023。"""
    ops.draw(state, ctx.cards, ctx.controller, params["draw"], ctx.tx)
    state.next_start_player = ctx.controller
    ops.tx(ctx.tx, "TX-023")
    return None


@register("on_place__hire_worker", "on_place")
def _hire_worker(state, ctx, params):
    """学校: 研修中ワーカー1人。上限時は空振り(OPEN-018)。"""
    ops.hire_training(state, ctx.cards, ctx.controller, params["amount"], ctx.tx)
    return None


@register("on_place__hire_worker_up_to", "on_place")
def _hire_worker_up_to(state, ctx, params):
    """高等学校(4人) / 大学(5人): 目標人数までの差分を研修中で一括雇用(OPEN-014)。

    OPEN-023 は provisional。裁定が変われば本ハンドラだけを差し替える。
    """
    pid = ctx.controller
    target = min(params["target_total"], worker_cap(state, ctx.cards, pid))
    ops.hire_training(state, ctx.cards, pid,
                      max(0, target - state.employed(pid)), ctx.tx)
    return None


@register("on_place__hire_worker_ready", "on_place")
def _hire_worker_ready(state, ctx, params):
    """専門学校: 待機状態で雇用し当ラウンドから配置できる(TX-027)。"""
    ops.hire_ready(state, ctx.cards, ctx.controller, params["amount"], ctx.tx)
    return None


# ====================================================== on_round_start

@register("on_round_start__slash_and_burn_collect", "on_round_start")
def _slash_and_burn_collect(state, ctx, params):
    """焼畑: カード上の消費財を所有者の手札へ移し、焼畑を建物捨て札へ(TX-026, TX-021)。

    R9 に配置した場合は次ラウンドがないため本効果は発火しない。焼畑は所有建物の
    まま終了し、不動産屋・本社ビルの得点対象に数える(OPEN-026)。
    """
    key = ctx.source_instance_id
    load = state.slash_and_burn_load.pop(key, 0)
    # カード上→手札の移動なので TX-020 ではなく TX-026(RULES.md 第12章)。
    ops.gain_goods(state, ctx.controller, load, ctx.tx, tx_id="TX-026")
    ops.remove_owned_building_to_discard(state, ctx.cards, ctx.controller, key,
                                         ctx.tx)
    return None


# ============================================================== passive
# 恒常効果は呼び出されない。参照側(worker_cap / hand_limit / 得点)が
# 所有枚数を数える。登録は完全性検証(§5.3)のために必要。

@register("passive__hand_limit_bonus", "passive")
def _passive_hand_limit_bonus(state, ctx, params):
    raise EffectContextError("passive hook is never invoked: hand_limit_bonus")


@register("passive__worker_cap_bonus", "passive")
def _passive_worker_cap_bonus(state, ctx, params):
    raise EffectContextError("passive hook is never invoked: worker_cap_bonus")


@register("passive__none", "passive")
def _passive_none(state, ctx, params):
    raise EffectContextError("passive hook is never invoked: none")


@register("passive__consumer_goods", "passive")
def _passive_consumer_goods(state, ctx, params):
    raise EffectContextError("passive hook is never invoked: consumer_goods")


# ========================================================== on_game_end
# 返り値は (得点, 免除する未払い賃金枚数)。TX-019。

@register("on_game_end__forgive_unpaid_wages", "on_game_end")
def _forgive_unpaid_wages(state, ctx, params):
    """法律事務所: 未払い賃金最大5枚を減点対象から除外。"""
    return (0, params["amount"])


@register("on_game_end__score_per_building", "on_game_end")
def _score_per_building(state, ctx, params):
    """不動産屋: 所有建物数×3点(自身を含む)。R9の焼畑も数える(OPEN-026)。"""
    owned = state.players[ctx.controller].owned_buildings
    return (params["points_per"] * len(owned), 0)


@register("on_game_end__score_per_consumer_good", "on_game_end")
def _score_per_consumer_good(state, ctx, params):
    """農協: 手札中の消費財枚数×3点。焼畑上の積載は対象外(OPEN-022)。"""
    return (params["points_per"] * state.players[ctx.controller].hand_goods, 0)


@register("on_game_end__score_per_worker", "on_game_end")
def _score_per_worker(state, ctx, params):
    """労働組合: 雇用ワーカー数×6点。"""
    return (params["points_per"] * state.employed(ctx.controller), 0)


@register("on_game_end__score_per_locked_building", "on_game_end")
def _score_per_locked_building(state, ctx, params):
    """本社ビル: 所有 locked 建物数×6点。R9の焼畑も locked として数える(OPEN-026)。"""
    owned = state.players[ctx.controller].owned_buildings
    n = sum(1 for i in owned if "locked" in ctx.cards.tags_of(i))
    return (params["points_per"] * n, 0)


@register("on_game_end__score_per_factory", "on_game_end")
def _score_per_factory(state, ctx, params):
    """鉄道: 所有 factory 属性建物数×8点。"""
    owned = state.players[ctx.controller].owned_buildings
    n = sum(1 for i in owned if "factory" in ctx.cards.tags_of(i))
    return (params["points_per"] * n, 0)


@register("on_game_end__unpaid_wage", "on_game_end")
def _unpaid_wage(state, ctx, params):
    """未払い賃金: 免除対象外1枚につき3点減算。得点関数から直接参照する。"""
    raise EffectContextError("unpaid_wage is scored from final_scores directly")


# ------------------------------------------------------------- 完全性検証

def verify_registry(cards=None):
    """§5.3: cards.json の hook と HANDLERS の突き合わせ。不一致は例外。"""
    cards = cards or cards_mod.load_cards()
    in_data = cards.hooks_in_data()
    missing = sorted(set(in_data) - set(HANDLERS))
    if missing:
        raise ValueError("hooks in cards.json without handler: %r" % missing)
    dead = sorted(set(HANDLERS) - set(in_data))
    if dead:
        raise ValueError("registered handlers not used by cards.json: %r" % dead)
    for hook, occurrences in in_data.items():
        timings = set(eff["timing"] for (_card_id, eff) in occurrences)
        if len(timings) != 1:
            raise ValueError("hook %s has mixed timings: %r" % (hook, sorted(timings)))
        timing = timings.pop()
        if HANDLER_TIMING[hook] != timing:
            raise ValueError("hook %s timing mismatch: data=%s handler=%s"
                             % (hook, timing, HANDLER_TIMING[hook]))
        keysets = set(
            frozenset((eff.get("params") or {}).keys())
            for (_card_id, eff) in occurrences
        )
        if len(keysets) != 1:
            raise ValueError("hook %s has inconsistent params keys: %r"
                             % (hook, sorted(sorted(k) for k in keysets)))
    return True
