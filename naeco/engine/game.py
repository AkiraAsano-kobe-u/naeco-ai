"""legal_actions / step / フェーズ駆動 / 得点(設計書 §3, §4, §9)。

`step` は純関数的である: 入力 state を破壊せず、clone した作業用状態を進めて返す。
中間決定(pending)は手番を消費しない(`turn_advanced=False`, 報酬0)。
"""
import dataclasses
import random
from typing import Any, Dict, List, Optional, Tuple

from . import cards as cards_mod
from . import ops
from .cards import card_id_of
from .handlers import HANDLERS, EffectContext
from .invariants import hand_limit, worker_cap
from .state import (
    Action, ActionKind, DecisionKind, GameState, LegalityError, PendingDecision,
    Phase, PlayerState, WorkplaceState, clone, decode_rng, derive_seed,
    encode_rng,
)


@dataclasses.dataclass
class StepResult:
    state: GameState
    terminal: bool
    scores: Optional[Tuple[int, ...]]
    ranking: Optional[Tuple[int, ...]]
    utilities: Optional[Tuple[float, ...]]
    turn_advanced: bool
    info: Dict[str, Any]


# ================================================================== 初期化

def draw_start_player(num_players, setup_seed):
    """セットアップRNGから開始プレイヤーを抽選する(RUL-SETUP-001, §2.5)。

    派生は呼び出し側で行い、ここは「seed → 席」の写像だけを担う。評価ハーネスは
    `derive_seed(root, "setup", index)` を自分で派生して渡すため、期待値を
    テストから独立に再現できる。
    """
    return random.Random(setup_seed).randrange(num_players)


def new_game(num_players, seed, cards=None, start_player=None, setup_seed=None):
    """RUL-SETUP-001 / RUL-SETUP-002。最初の選択地点まで自動処理して返す。

    `start_player=None` のとき、開始プレイヤーは seed から決定論的に**公平に抽選**
    する(RUL-SETUP-001「公平な方法で席順と開始プレイヤーを決め」)。席0固定は
    初期所持金($5が開始プレイヤー)と同点時優先権(RUL-SCORE-002)を席0へ恒久的に
    与えてしまうため、仕様違反である。

    抽選には**環境RNGを使わない**。§2.4 は環境RNG(`state.rng_state`)の消費を
    「初期シャッフルと山札再構成(TX-007)のみ」に限定しており、ここで
    `rng.randrange` を消費すると以降の再構成列がずれる。さらに `start_player` を
    明示指定した場合としない場合で `rng_state` が食い違ってしまう。そこで独立した
    セットアップRNG(§2.5 の席非依存ストリーム `"setup"`)から引く。

    `setup_seed` を渡せばその seed で抽選する。評価ハーネスのように
    `derive_seed(eval_seed(group, index), "setup", index)` を正しい root と
    episode_index から派生したい場合に使う(§10.1)。省略時は `seed` を root、
    `episode_index=0` として派生する単局用の既定値になる。
    """
    cards = cards or cards_mod.load_cards()
    if num_players not in (2, 3, 4):
        raise ValueError("num_players must be 2..4")
    if start_player is None:
        if setup_seed is None:
            setup_seed = derive_seed(seed, "setup", 0)
        start_player = draw_start_player(num_players, setup_seed)
    elif not (0 <= start_player < num_players):
        raise ValueError("start_player must be in 0..%d" % (num_players - 1))

    rng = random.Random(seed)
    deck = list(cards.building_instances)
    rng.shuffle(deck)

    workplaces = {}
    for card_id in ("base:quarry", "base:mine", "base:school"):
        instance_id = cards.instances_by_card[card_id][0]
        cd = cards.get(card_id)
        workplaces[instance_id] = WorkplaceState(
            instance_id, card_id, None, "any_player", cd.occupancy_mode, ())
    carpenters = cards.instances_by_card["base:carpenter"]
    n_carp = cards_mod.carpenter_count(num_players)
    for instance_id in carpenters[:n_carp]:
        workplaces[instance_id] = WorkplaceState(
            instance_id, "base:carpenter", None, "any_player", "exclusive", ())
    out_of_box = tuple(sorted(carpenters[n_carp:]))

    players = []
    for pid in range(num_players):
        hand = tuple(sorted(deck[:cards_mod.INITIAL_HAND]))
        deck = deck[cards_mod.INITIAL_HAND:]
        players.append(PlayerState(
            player_id=pid,
            # 初期所持金は「開始プレイヤーから時計回りに $5,$6,…」(RUL-SETUP-001)。
            money=cards_mod.initial_money((pid - start_player) % num_players),
            hand_buildings=hand,
            hand_goods=0,
            owned_buildings=(),
            workers_waiting=cards_mod.INITIAL_WORKERS,
            workers_training=0,
            unpaid_wages=0,
        ))

    state = GameState(
        num_players=num_players,
        round_number=1,
        phase=Phase.PLACE,
        start_player=start_player,
        next_start_player=start_player,
        current_player=start_player,
        end_phase_cursor=0,
        players=tuple(players),
        deck=tuple(deck),
        building_discard=(),
        round_deck=tuple(cards.round_instances),
        out_of_box=out_of_box,
        household=0,
        workplaces=workplaces,
        slash_and_burn_load={},
        temporary_public=(),
        public_knowledge=(),
        reshuffle_seq=0,
        draws_since_reshuffle=tuple(0 for _ in range(num_players)),
        pending=None,
        rng_state=encode_rng(rng),
        history=(),
        cards_sha256=cards.sha256,
        cards_data_version=cards.data_version,
    )
    _advance(state, cards)
    return state


# ============================================================== 合法手

def actor(state):
    """いま選択する主体。pending があればその player_id。終端では None。"""
    if state.phase == Phase.GAME_OVER:
        return None
    if state.pending is not None:
        return state.pending.player_id
    return state.current_player


def legal_actions(state, cards=None):
    """手番プレイヤー、または pending の主体の原子選択を安定順で返す(§3.6)。"""
    cards = cards or cards_mod.load_cards()
    if state.phase == Phase.GAME_OVER:
        return []
    if state.pending is not None:
        acts = _pending_actions(state, cards, state.pending)
    elif state.phase == Phase.PLACE:
        acts = _place_actions(state, cards, state.current_player)
    else:
        acts = []
    acts.sort(key=Action.sort_key)
    return acts


def _place_actions(state, cards, pid):
    if state.players[pid].workers_waiting <= 0:
        return []
    out = []
    for instance_id in sorted(state.workplaces):
        w = state.workplaces[instance_id]
        if _can_place(state, cards, pid, w):
            out.append(Action(ActionKind.PLACE_WORKER, target_instance_id=instance_id))
    return out


def _can_place(state, cards, pid, w):
    """RUL-PLACE-001 の境界を判定式にしたもの(§3.5)。"""
    return placement_block_reason(state, cards, pid, w) is None


def placement_block_reason(state, cards, pid, w):
    """`pid` が職場 `w` に置けない理由を返す。置けるなら None(RUL-PLACE-001)。

    判定に使うのは **pid 自身の手札**と公開情報(占有・所有・家計)だけなので、
    pid 本人へ返しても非公開情報は漏れない(§7.2.1)。UI が「盤面には在るが今は
    置けない」を理由付きで説明するために公開している。
    """
    if w.occupancy_mode != "unlimited" and w.occupants:
        return "この職場は今ラウンドすでに使われている"      # INV-005 占有
    if w.usable_by == "owner_only" and w.owner != pid:
        return "他プレイヤーの所有職場は使えない"            # TERM-002
    eff_id = cards.on_place_effect.get(card_id_of(w.instance_id))
    if eff_id is None:
        return "ワーカーを置けない施設"
    hook = cards.effect_hook[eff_id]
    params = cards.effect_params[eff_id]
    p = state.players[pid]

    if hook == "on_place__discard_gain_household":
        if p.hand_total < params["discard"]:
            return "手札が%d枚必要(いま%d枚)" % (params["discard"], p.hand_total)
        if state.household < params["amount"]:
            return "家計に$%d必要(いま$%d)" % (params["amount"], state.household)
        return None
    if hook == "on_place__gain_from_household":
        if state.household < params["amount"]:
            return "家計に$%d必要(いま$%d)" % (params["amount"], state.household)
        return None
    if hook == "on_place__discard_then_draw":
        if p.hand_total < params["discard"]:
            return "手札が%d枚必要(いま%d枚)" % (params["discard"], p.hand_total)
        return None
    if hook in ("on_place__build", "on_place__build_then_draw"):
        d = params.get("discount", 0)
        if not any(_can_build(cards, p, b, d) for b in p.hand_buildings):
            return _build_shortfall_reason(cards, p, d)
        return None
    if hook == "on_place__build_farm_free":
        if not any("farm" in cards.tags_of(b) for b in p.hand_buildings):
            return "手札に農場系(farm)の建物がない"
        return None
    if hook == "on_place__build_two_same_cost":
        if not _two_for_one_first_candidates(cards, p):
            return "同じコストの建物2枚と、そのコスト1回分の支払い札がない"
        return None
    # 獲得系の空振りは配置を禁止しない(OPEN-010)
    return None


def _build_shortfall_reason(cards, p, discount):
    """建設できない理由を、手札の枚数で説明する(OPEN-021)。"""
    if not p.hand_buildings:
        return "手札に建物がないので建設できない"
    cheapest = min(max(0, cards.build_cost(b) - discount) for b in p.hand_buildings)
    return ("建設には対象1枚とは別に%d枚の支払い札が要る(支払いに使えるのは%d枚)"
            % (cheapest, p.hand_total - 1))


def _can_build(cards, p, instance_id, discount, extra_reserved=0):
    """建設対象自身とニコイチの2枚目は捨て札原資に使えない(§3.5)。"""
    cost = max(0, cards.build_cost(instance_id) - discount)
    return p.hand_total - 1 - extra_reserved >= cost


def _two_for_one_first_candidates(cards, p):
    """ニコイチ1枚目: 「**より大きいID**の同コスト相方」を要求する(R2との整合)。"""
    out = []
    for b in p.hand_buildings:
        c = cards.build_cost(b)
        if p.hand_total - 2 < c:
            continue
        if any(b2 > b and cards.build_cost(b2) == c for b2 in p.hand_buildings):
            out.append(b)
    return out


# -------------------------------------------------------- pending の合法手

def _pending_actions(state, cards, pend):
    if pend.kind == DecisionKind.BUILD_TARGET:
        return _build_target_actions(state, cards, pend)
    if pend.kind in (DecisionKind.BUILD_COST, DecisionKind.DISCARD_N,
                     DecisionKind.HAND_LIMIT_DISCARD):
        return _discard_actions(state, cards, pend)
    if pend.kind == DecisionKind.DESIGN_OFFICE_KEEP:
        return [Action(ActionKind.SELECT_KEEP, instance_id=i)
                for i in state.temporary_public]
    if pend.kind == DecisionKind.INSOLVENCY_SELL:
        return [Action(ActionKind.SELL_BUILDING, instance_id=i)
                for i in _sellable(state, cards, pend.player_id)]
    raise AssertionError("unhandled pending kind: %r" % (pend.kind,))


def _build_target_actions(state, cards, pend):
    p = state.players[pend.player_id]
    eff = pend.source_effect_id
    if eff == "EFF-pioneer-01":
        cands = [b for b in p.hand_buildings if "farm" in cards.tags_of(b)]
    elif eff == "EFF-two-for-one-01":
        if not pend.reserved_targets:
            cands = _two_for_one_first_candidates(cards, p)
        else:
            first = pend.reserved_targets[0]
            cands = [b for b in p.hand_buildings
                     if b > first and cards.build_cost(b) == pend.constraint_cost]
    else:
        cands = [b for b in p.hand_buildings
                 if _can_build(cards, p, b, pend.discount)]
    return [Action(ActionKind.SELECT_BUILD, instance_id=b) for b in cands]


def _discard_actions(state, cards, pend):
    """R1a(消費財の枚数を一括で決める) → R1b(建物を instance_id 昇順)。"""
    p = state.players[pend.player_id]
    eligible = [b for b in p.hand_buildings if b not in pend.reserved_targets]

    if pend.discarded_goods is None:
        # 第1段階。建物だけでは足りない分は消費財必須(k_min)。
        k_min = max(0, pend.remaining - len(eligible))
        k_max = min(p.hand_goods, pend.remaining)
        return [Action(ActionKind.SELECT_DISCARD, goods_count=k)
                for k in range(k_min, k_max + 1)]

    need = pend.remaining - pend.discarded_goods - len(pend.selected_discards)
    if need <= 0:
        return []
    last = pend.selected_discards[-1] if pend.selected_discards else None
    cands = [b for b in eligible if last is None or b > last]
    # 継続可能性(保証1b): 選んだ後も昇順で need-1 枚を選べる候補だけを残す。
    cands = cands[:len(cands) - need + 1]
    return [Action(ActionKind.SELECT_DISCARD, instance_id=b) for b in cands]


def _sellable(state, cards, pid):
    """RUL-BUILD-002: locked は選べない。"""
    return [i for i in state.players[pid].owned_buildings
            if "locked" not in cards.tags_of(i)]


# ================================================================== step

def step(state, action, cards=None):
    """非合法手は LegalityError を送出する(状態は変更しない)。"""
    cards = cards or cards_mod.load_cards()
    legal = legal_actions(state, cards)
    if action not in legal:
        raise LegalityError("illegal action %r (legal: %r)" % (action, legal))

    s = clone(state)
    tx = ops.Ledger()
    pending_before = state.pending

    if action.kind == ActionKind.PLACE_WORKER:
        _apply_place(s, cards, action, tx)
    elif action.kind == ActionKind.SELECT_BUILD:
        _apply_select_build(s, cards, action, tx)
    elif action.kind == ActionKind.SELECT_DISCARD:
        _apply_select_discard(s, cards, action, tx)
    elif action.kind == ActionKind.SELECT_KEEP:
        _apply_select_keep(s, cards, action, tx)
    elif action.kind == ActionKind.SELL_BUILDING:
        _apply_sell(s, cards, action, tx)
    else:
        raise AssertionError("unhandled action kind")

    # 中間決定は手番を消費しない(§3.1)。判定は「入力した Action が属する原子決定の
    # 連鎖が完了したか」であって、返却状態に pending があるか否かではない。
    # (a) _apply_* の直後にまだ pending が残っていれば、その決定は未完了。
    #     判定を _advance の後に採ると、最後のワーカーを配置して手番が完了した直後に
    #     _advance が終了処理へ進んで**別の** pending を立てた場合に、完了した手番を
    #     未完了と誤判定する。
    turn_advanced = s.pending is None

    _advance(s, cards, tx)

    # (b) 逆に、_advance が**同じ決定の続き**を立て直すこともある。強制売却は
    #     1回売っても不足が残れば同じ RUL-INSOLV-001 の pending が再び立つ
    #     (_apply_sell が一旦 None にして終了処理へ再評価させる設計)。これは
    #     連鎖の継続なので手番は進んでいない。
    if turn_advanced and _is_same_decision(pending_before, s.pending):
        turn_advanced = False

    terminal = s.phase == Phase.GAME_OVER
    scores = ranking_ = utilities = None
    if terminal:
        scores = final_scores(s, cards)
        ranking_ = ranking(s, cards, scores)
        utilities = terminal_utilities(s, cards, ranking_)
    # events は公開情報だけの構造化ログ。UI が金額を差分から逆算しなくてよい。
    info = {"tx": tx.tx_ids, "events": tuple(tx.events)}
    return StepResult(s, terminal, scores, ranking_, utilities, turn_advanced, info)


def _is_same_decision(before, after):
    """2つの pending が「同一の決定の継続」か(§3.1 の turn_advanced 判定用)。

    同じプレイヤーの、同じ発生源・同じ種類の決定なら継続とみなす。別プレイヤーの
    同種決定(次の席の支払い不能など)や、同じプレイヤーでも別段階の決定
    (手札上限調整 → 賃金支払い不能)は継続ではない。
    """
    if before is None or after is None:
        return False
    return (before.player_id == after.player_id
            and before.kind == after.kind
            and before.source_effect_id == after.source_effect_id)


def _apply_place(s, cards, action, tx):
    pid = s.current_player
    instance_id = action.target_instance_id
    w = s.workplaces[instance_id]
    s.players[pid].workers_waiting -= 1
    w.occupants = w.occupants + (pid,)
    ops.log(s, pid, ActionKind.PLACE_WORKER, target=instance_id)
    ops.tx(tx, "TX-005")

    eff_id = cards.on_place_effect[card_id_of(instance_id)]
    pend = _run_effect(s, cards, eff_id, pid, instance_id, tx)
    s.pending = pend
    if pend is None:
        _end_turn(s)


def _run_effect(s, cards, eff_id, controller, source_instance_id, tx=None):
    ctx = EffectContext(
        controller=controller,
        effect_id=eff_id,
        source_instance_id=source_instance_id,
        logical_id=cards.logical_id_of(source_instance_id),
        cards=cards,
        tx=tx,
    )
    hook = cards.effect_hook[eff_id]
    return HANDLERS[hook](s, ctx, cards.effect_params[eff_id])


def _end_turn(s):
    """配置の解決が完了した。次の手番へ(RUL-TURN-002)。"""
    s.current_player = (s.current_player + 1) % s.num_players


# ---------------------------------------------------------- 各選択の適用

def _apply_select_build(s, cards, action, tx):
    pend = s.pending
    b = action.instance_id
    pid = pend.player_id

    if pend.source_effect_id == "EFF-pioneer-01":
        ops.build(s, cards, pid, b, tx)                   # コスト0(OPEN-020)
        s.pending = None
        _end_turn(s)
        return

    if pend.source_effect_id == "EFF-two-for-one-01":
        if not pend.reserved_targets:
            s.pending = dataclasses.replace(
                pend, reserved_targets=(b,), remaining=1,
                constraint_cost=cards.build_cost(b))
            return
        # 2枚目。R2 により reserved_targets[0] < reserved_targets[1]。
        reserved = pend.reserved_targets + (b,)
        cost = pend.constraint_cost
        if cost == 0:
            _finish_build(s, cards, pend.source_effect_id, pid, reserved, tx)
            return
        s.pending = dataclasses.replace(
            pend, kind=DecisionKind.BUILD_COST, remaining=cost,
            reserved_targets=reserved, selected_discards=(), discarded_goods=None)
        return

    cost = max(0, cards.build_cost(b) - pend.discount)
    if cost == 0:
        _finish_build(s, cards, pend.source_effect_id, pid, (b,), tx)
        return
    s.pending = dataclasses.replace(
        pend, kind=DecisionKind.BUILD_COST, remaining=cost,
        reserved_targets=(b,), selected_discards=(), discarded_goods=None)


def _finish_build(s, cards, eff_id, pid, reserved, tx):
    """建設順は instance_id 昇順で固定する(R3)。"""
    for b in sorted(reserved):
        ops.build(s, cards, pid, b, tx)
    if eff_id == "EFF-general-contractor-01":
        # 継続処理は source_effect_id による分岐で行う(§3.3。follow_up は持たない)
        ops.draw(s, cards, pid, cards.effect_params[eff_id]["draw"], tx)
    s.pending = None
    _end_turn(s)


def _apply_select_discard(s, cards, action, tx):
    pend = s.pending
    pid = pend.player_id

    if action.instance_id is None:
        ops.discard_goods_from_hand(s, pid, action.goods_count,
                                    ActionKind.SELECT_DISCARD)
        pend = dataclasses.replace(pend, discarded_goods=action.goods_count)
    else:
        ops.discard_building_from_hand(s, cards, pid, action.instance_id,
                                       ActionKind.SELECT_DISCARD)
        pend = dataclasses.replace(
            pend, selected_discards=pend.selected_discards + (action.instance_id,))

    s.pending = pend
    done = (pend.discarded_goods is not None
            and pend.discarded_goods + len(pend.selected_discards) >= pend.remaining)
    if done:
        _finish_discard(s, cards, pend, tx)


def _finish_discard(s, cards, pend, tx):
    pid = pend.player_id
    eff_id = pend.source_effect_id

    # 廃棄は「支払い1回」で1件だけ記録する。捨てた枚数ぶん記録してはならない
    # (RUL-BUILD-001「コスト支払いの手札廃棄は TX-009 の1回のみで表す」)。
    # 手札上限の調整だけは TX-017(RULES.md 第12章)。
    if pend.remaining > 0:
        ops.tx(tx, "TX-017" if pend.kind == DecisionKind.HAND_LIMIT_DISCARD
               else "TX-009")

    if pend.kind == DecisionKind.BUILD_COST:
        _finish_build(s, cards, eff_id, pid, pend.reserved_targets, tx)
        return

    if pend.kind == DecisionKind.HAND_LIMIT_DISCARD:
        s.pending = None                # 終了処理のループへ戻る
        return

    # DISCARD_N: 効果ごとの後続処理
    hook = cards.effect_hook[eff_id]
    params = cards.effect_params[eff_id]
    if hook == "on_place__discard_then_draw":
        ops.draw(s, cards, pid, params["draw"], tx)
    elif hook == "on_place__discard_gain_household":
        ops.take_from_household(s, pid, params["amount"], tx)
    else:
        raise AssertionError("unexpected DISCARD_N source: %s" % eff_id)
    s.pending = None
    _end_turn(s)


def _apply_select_keep(s, cards, action, tx):
    pend = s.pending
    pid = pend.player_id
    kept = action.instance_id
    rest = [i for i in s.temporary_public if i != kept]

    p = s.players[pid]
    p.hand_buildings = tuple(sorted(p.hand_buildings + (kept,)))
    # 公開されたので既知情報として残る(RUL-INFO-001, §7.3)
    ops.knowledge_in_hand(s, kept, pid, pend.source_effect_id)
    ops.log(s, pid, ActionKind.SELECT_KEEP, revealed=kept)

    s.building_discard = tuple(sorted(s.building_discard + tuple(rest)))
    ops.knowledge_drop(s, rest)
    s.temporary_public = ()
    # TX-022 は「山札→一時公開」の記録であり、公開時に一度だけ計上済み
    # (ops.reveal_to_temporary_public)。ここで再計上すると二重になる。
    s.pending = None
    _end_turn(s)


def _apply_sell(s, cards, action, tx):
    """RUL-INSOLV-001 (1): 「どれを売るか」だけが決定。打ち切り行動はない(§3.4 G)。"""
    pend = s.pending
    pid = pend.player_id
    ops.sell_building(s, cards, pid, action.instance_id, tx)
    s.pending = None                    # 終了処理のループが不足を再評価する


# ============================================== フェーズ駆動(自動処理)

def _advance(s, cards, tx=None):
    """次に選択が必要な地点まで自動処理を進める(付録B)。"""
    while True:
        if s.pending is not None:
            return
        if s.phase == Phase.GAME_OVER:
            return
        if s.phase == Phase.ROUND_START:
            _do_round_start(s, cards, tx)
            continue
        if s.phase == Phase.PLACE:
            for i in range(s.num_players):
                pid = (s.current_player + i) % s.num_players
                if s.players[pid].workers_waiting > 0:
                    s.current_player = pid
                    return
            s.phase = Phase.END
            s.end_phase_cursor = 0
            continue
        if s.phase == Phase.END:
            if _do_end_phase(s, cards, tx):
                return                   # pending が立った
            continue
        raise AssertionError("unhandled phase: %r" % (s.phase,))


def _do_round_start(s, cards, tx=None):
    """PHASE-ROUND-START: 研修解除と焼畑回収(TX-004, TX-026, TX-021)。"""
    for i in range(s.num_players):
        pid = (s.start_player + i) % s.num_players
        p = s.players[pid]
        ops.release_training(s, pid, tx)
        for instance_id in sorted(s.slash_and_burn_load):
            if instance_id in p.owned_buildings:
                _run_effect(s, cards, "EFF-slash-and-burn-02", pid, instance_id, tx)
    s.phase = Phase.PLACE
    s.current_player = s.start_player


def _do_end_phase(s, cards, tx=None):
    """RUL-ROUND-002 を開始プレイヤーから1人ずつ完全解決する(OPEN-002)。

    pending を立てたら True を返す。各人 (1)回収 (2)手札調整 (3)賃金 の順。
    再入時はどの段階かを状態から判定できる:
      手札超過が残っていれば (2)、そうでなければ (3)。
    """
    while s.end_phase_cursor < s.num_players:
        pid = (s.start_player + s.end_phase_cursor) % s.num_players
        p = s.players[pid]

        ops.recall_workers(s, pid, tx)                  # (1) TX-016

        excess = p.hand_total - hand_limit(s, cards, pid)
        if excess > 0:                                  # (2) TX-017
            s.pending = PendingDecision(
                kind=DecisionKind.HAND_LIMIT_DISCARD,
                player_id=pid,
                source_effect_id="RUL-ROUND-002",
                remaining=excess,
            )
            return True

        wage = cards.wage_rate(s.round_number) * s.employed(pid)   # (3) TX-013
        if p.money >= wage:
            ops.pay_to_household(s, pid, wage, tx)
            s.end_phase_cursor += 1
            continue
        sellable = _sellable(s, cards, pid)
        if sellable:                                    # RUL-INSOLV-001 (1)
            s.pending = PendingDecision(
                kind=DecisionKind.INSOLVENCY_SELL,
                player_id=pid,
                source_effect_id="RUL-INSOLV-001",
                remaining=wage - p.money,
            )
            return True
        # (2) 全額を家計へ払い、残不足$1につき未払い賃金1枚(TX-015)
        shortage = wage - p.money
        ops.pay_to_household(s, pid, p.money, tx)
        ops.issue_unpaid_wages(s, pid, shortage, tx)
        s.end_phase_cursor += 1

    _finish_round(s, cards, tx)
    return False


def _finish_round(s, cards, tx=None):
    if s.round_number >= 9:
        # R9: 焼畑の積載消費財を山へ戻す。カード本体は所有建物に残す(OPEN-026)。
        s.slash_and_burn_load = {}
        s.phase = Phase.GAME_OVER
        return
    # TX-018: 最上段のラウンドカードをめくり公共職場へ
    top = s.round_deck[0]
    s.round_deck = s.round_deck[1:]
    cd = cards.of_instance(top)
    s.workplaces[top] = WorkplaceState(
        instance_id=top,
        logical_id=cards.logical_id_of(top),
        owner=None,
        usable_by="any_player",
        occupancy_mode=cd.occupancy_mode,
        occupants=(),
    )
    ops.tx(tx, "TX-018")
    s.round_number += 1
    # 採石場(TX-023)で予約された開始プレイヤーがここで有効になる。誰も採石場を
    # 使わなければ next_start_player は据え置かれ、マーカーは保持される。
    s.start_player = s.next_start_player
    s.end_phase_cursor = 0
    s.phase = Phase.ROUND_START


# ============================================================ 得点・順位

def score_breakdown(state, cards=None):
    """RUL-SCORE-001 の各項を分解して返す(表示・検証用)。

    合計は `final_scores` と常に一致する。UI が内訳を再計算して食い違うことを
    避けるため、算出はここに一本化する。
    """
    cards = cards or cards_mod.load_cards()
    penalty_per_wage = cards.effect_params["EFF-unpaid-wage-01"]["points_per"]
    out = []
    for pid in range(state.num_players):
        p = state.players[pid]
        building_value = sum(cards.printed_value(i) or 0
                             for i in p.owned_buildings)
        points = 0
        forgiven = 0
        for instance_id in p.owned_buildings:
            for eff in cards.of_instance(instance_id).effects:
                if eff["timing"] != "on_game_end":
                    continue
                ctx = EffectContext(pid, eff["effect_id"], instance_id,
                                    cards.logical_id_of(instance_id), cards)
                got, forgive = HANDLERS[eff["hook"]](state, ctx, eff["params"])
                points += got
                forgiven += forgive
        forgiven = min(forgiven, p.unpaid_wages)
        unforgiven = p.unpaid_wages - forgiven
        penalty = penalty_per_wage * unforgiven
        out.append({
            "building_value": building_value,
            "money": p.money,
            "end_game_points": points,
            "unpaid_wages": p.unpaid_wages,
            "forgiven_unpaid_wages": forgiven,
            "unpaid_penalty": -penalty,
            "total": building_value + p.money + points - penalty,
        })
    return tuple(out)


def final_scores(state, cards=None):
    """RUL-SCORE-001。TX-019。"""
    return tuple(b["total"] for b in score_breakdown(state, cards))


def ranking(state, cards=None, scores=None):
    """RUL-SCORE-002 のタイブレークを適用した順位(0=1位)。同順位は生じない。"""
    cards = cards or cards_mod.load_cards()
    scores = scores if scores is not None else final_scores(state, cards)
    n = state.num_players
    order = sorted(
        range(n),
        key=lambda pid: (-scores[pid], (pid - state.start_player) % n),
    )
    out = [0] * n
    for place, pid in enumerate(order):
        out[pid] = place
    return tuple(out)


def terminal_utilities(state, cards=None, ranking_=None):
    """Absolute-seat utility tuple (core-api §9.2 / Phase 10 D1).

    Preserve the two-seat win/loss branch; three/four seats use linear ranks.
    Ties have already been broken by ranking (RUL-SCORE-002).
    """
    cards = cards or cards_mod.load_cards()
    ranking_ = ranking_ if ranking_ is not None else ranking(state, cards)
    if state.num_players == 2:
        return tuple(1.0 if r == 0 else -1.0 for r in ranking_)
    return tuple(1.0 - 2 * r / (state.num_players - 1) for r in ranking_)
