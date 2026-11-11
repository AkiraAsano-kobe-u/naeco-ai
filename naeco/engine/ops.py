"""状態遷移のプリミティブ(RULES.md 第12章の TX-* に対応)。

ここに載る関数は**作業用コピーを in-place で書き換える**。純関数性は `step()` が
入口で `clone()` することで担保する(設計書 §4)。
"""
from typing import List, Optional, Tuple

from . import cards as cards_mod
from .state import (
    HistoryEntry, KnowledgeEntry, KnowledgeKind, WorkplaceState,
    decode_rng, encode_rng,
)


# ---------------------------------------------------------------- TX 台帳

class Ledger(object):
    """1 step ぶんの取引記録。`StepResult.info` の中身になる。

    - `tx_ids`: RULES.md 第12章の TX-* の並び(発生順)
    - `events`: 公開情報だけを持つ構造化イベント。UI が金額を推測せずに
      実況できるようにするためのもの(sol 指摘: 所持金の純減から賃金額を
      逆算すると、同一 step 内の家計獲得・売却と相殺されて誤る)
    """

    def __init__(self):
        self.tx_ids = []
        self.events = []


def tx(sink, *transaction_ids):
    """TX-* を記録する。`sink` が None なら何もしない。

    取引の記録は**それを実行した関数の中で**行う(sol J)。呼び出し側で append
    すると、即時効果や自動処理の経路がまるごと抜け落ちる。
    ただし「1取引 = 1記録」の粒度は RULES.md に従う。たとえばコスト支払いの
    手札廃棄は捨てた枚数によらず TX-009 の1回である(RUL-BUILD-001)。
    """
    if sink is not None:
        sink.tx_ids.extend(transaction_ids)


def event(sink, kind, **fields):
    """公開情報だけの構造化イベントを積む(金額・枚数は実額)。"""
    if sink is not None:
        fields["kind"] = kind
        sink.events.append(fields)


# ------------------------------------------------------------------ 履歴

def log(state, actor, kind, target=None, revealed=None, goods=0):
    """公開ゾーンへの移動と配置だけを履歴に残す(§7.2.2)。"""
    state.history = state.history + (
        HistoryEntry(state.round_number, actor, kind, target, revealed, goods),
    )


# -------------------------------------------------------------- 公開知識

def knowledge_drop(state, instance_ids):
    """実体が公開ゾーンへ移ったので知識エントリを削除する(§7.3 失効規則)。"""
    drop = set(instance_ids)
    if not drop:
        return
    state.public_knowledge = tuple(
        e for e in state.public_knowledge if e.instance_id not in drop
    )


def knowledge_in_hand(state, instance_id, holder, source):
    knowledge_drop(state, [instance_id])
    state.public_knowledge = state.public_knowledge + (
        KnowledgeEntry(instance_id, KnowledgeKind.IN_HAND, holder, None,
                       state.round_number, source),
    )


# ------------------------------------------------------- 山札 (TX-006/007)

def reshuffle_deck(state, cards, tx_sink=None):
    """TX-007: 建物捨て札全体をシャッフルして山札にする。

    捨て札の中身は**全員が見た公開事実**なので、山札に入ったことを
    IN_DECK_SINCE として保持する(削除してはならない。§7.3)。
    """
    if not state.building_discard:
        return
    tx(tx_sink, "TX-007")
    rng = decode_rng(state.rng_state)
    pool = list(state.building_discard)      # 正準順(昇順)から始める
    rng.shuffle(pool)
    state.rng_state = encode_rng(rng)
    state.deck = tuple(pool)
    state.building_discard = ()
    state.reshuffle_seq += 1
    state.draws_since_reshuffle = tuple(0 for _ in range(state.num_players))
    kept = tuple(
        e for e in state.public_knowledge if e.kind != KnowledgeKind.IN_DECK_SINCE
    )
    added = tuple(
        KnowledgeEntry(i, KnowledgeKind.IN_DECK_SINCE, None, state.reshuffle_seq,
                       state.round_number, "TX-007")
        for i in sorted(pool)
    )
    state.public_knowledge = kept + added


def _pop_deck(state, cards, tx_sink=None):
    """山札の上を1枚返す。空なら再構成を試み、それでも空なら None(OPEN-011)。"""
    if not state.deck:
        reshuffle_deck(state, cards, tx_sink)
    if not state.deck:
        return None
    top = state.deck[0]
    state.deck = state.deck[1:]
    return top


def draw(state, cards, pid, count, tx_sink=None):
    """TX-006: 引ける枚数だけ引く。返り値は実際に引いた実体の一覧。"""
    drawn = []
    for _ in range(count):
        top = _pop_deck(state, cards, tx_sink)
        if top is None:
            break
        drawn.append(top)
        counts = list(state.draws_since_reshuffle)
        counts[pid] += 1
        state.draws_since_reshuffle = tuple(counts)
    if drawn:
        p = state.players[pid]
        p.hand_buildings = tuple(sorted(p.hand_buildings + tuple(drawn)))
        tx(tx_sink, "TX-006")
    return drawn


def reveal_to_temporary_public(state, cards, count, tx_sink=None):
    """TX-022: 設計事務所の公開。ドローと同じ再構成規則に従う(OPEN-011)。

    公開した実体は `temporary_public`(公開ゾーン)へ移るので、知識エントリは
    **この時点で**失効させる(§7.3 の失効規則「公開ゾーンへ移った実体は削除」)。
    選択完了まで持ち越すと、山札再構成の直後に設計事務所を使った局面で
    `IN_DECK_SINCE` が残存し、同じ実体が「公開ゾーンにある」と「山札か適格な手札に
    ある」を同時に主張する。Phase 2.5 のサンプラーはこの局面で C4 と C5 を同時に
    満たせず、正当な観測に対して失敗する(sol P25-003)。
    """
    revealed = []
    for _ in range(count):
        top = _pop_deck(state, cards, tx_sink)
        if top is None:
            break
        revealed.append(top)
    if revealed:
        state.temporary_public = tuple(sorted(state.temporary_public + tuple(revealed)))
        knowledge_drop(state, revealed)
        tx(tx_sink, "TX-022")
    return revealed


# ------------------------------------------------ 手札の廃棄 (TX-009/017)

def discard_building_from_hand(state, cards, pid, instance_id, actor_kind):
    """手札の建物を捨て札へ。

    TX の記録はここでは**行わない**。廃棄1枚ごとではなく「支払い1回」で
    TX-009 / TX-017 を1件だけ記録するのが正本の粒度であり(RUL-BUILD-001
    「コスト支払いの手札廃棄は TX-009 の1回のみで表す」)、記録は決定の完了を
    知っている `game._finish_discard` が行う。
    """
    p = state.players[pid]
    hand = list(p.hand_buildings)
    hand.remove(instance_id)
    p.hand_buildings = tuple(hand)
    state.building_discard = tuple(sorted(state.building_discard + (instance_id,)))
    knowledge_drop(state, [instance_id])
    log(state, pid, actor_kind, revealed=instance_id)


def discard_goods_from_hand(state, pid, count, actor_kind):
    """消費財は山へ合流して個体性を失う(INV-007)。枚数だけ減らす。

    TX の記録は `discard_building_from_hand` と同じ理由でここでは行わない。
    """
    if count <= 0:
        return
    p = state.players[pid]
    p.hand_goods -= count
    log(state, pid, actor_kind, goods=count)


def gain_goods(state, pid, count, tx_sink=None, tx_id="TX-020"):
    """TX-020: 消費財取得。山は無限資源(virtual_issue)。

    焼畑の回収だけは「カード上→手札」であり TX-026 なので、呼び出し側が
    `tx_id` を差し替える(RULES.md EFF-slash-and-burn-02)。
    """
    if count > 0:
        state.players[pid].hand_goods += count
        tx(tx_sink, tx_id)


# ----------------------------------------------------- 建設・売却 (TX-010/011)

def build(state, cards, pid, instance_id, tx_sink=None):
    """TX-010: 手札の建物を自分の場へ移す。職場なら owner_only の職場になる。"""
    tx(tx_sink, "TX-010")
    p = state.players[pid]
    hand = list(p.hand_buildings)
    hand.remove(instance_id)
    p.hand_buildings = tuple(hand)
    p.owned_buildings = tuple(sorted(p.owned_buildings + (instance_id,)))
    knowledge_drop(state, [instance_id])
    cd = cards.of_instance(instance_id)
    if cd.workplace is not None:
        state.workplaces[instance_id] = WorkplaceState(
            instance_id=instance_id,
            logical_id=cards.logical_id_of(instance_id),
            owner=pid,
            usable_by="owner_only",
            occupancy_mode=cd.occupancy_mode,
            occupants=(),
        )
    from .state import ActionKind
    log(state, pid, ActionKind.SELECT_BUILD, revealed=instance_id)


def sell_building(state, cards, pid, instance_id, tx_sink=None):
    """TX-011 + TX-012: 建物を公共職場化し、価値と同額をサプライから得る。"""
    tx(tx_sink, "TX-011", "TX-012")
    event(tx_sink, "building_sold", player_id=pid, instance_id=instance_id,
          amount=cards.printed_value(instance_id) or 0)
    p = state.players[pid]
    owned = list(p.owned_buildings)
    owned.remove(instance_id)
    p.owned_buildings = tuple(owned)
    p.money += cards.printed_value(instance_id) or 0
    cd = cards.of_instance(instance_id)
    state.workplaces[instance_id] = WorkplaceState(
        instance_id=instance_id,
        logical_id=cards.logical_id_of(instance_id),
        owner=None,
        usable_by="any_player",
        occupancy_mode=cd.occupancy_mode,
        occupants=(),
    )
    from .state import ActionKind
    log(state, pid, ActionKind.SELL_BUILDING, revealed=instance_id)


def remove_owned_building_to_discard(state, cards, pid, instance_id, tx_sink=None):
    """TX-021: 焼畑除去。所有建物から建物捨て札へ。"""
    tx(tx_sink, "TX-021")
    p = state.players[pid]
    owned = list(p.owned_buildings)
    owned.remove(instance_id)
    p.owned_buildings = tuple(owned)
    state.workplaces.pop(instance_id, None)
    state.building_discard = tuple(sorted(state.building_discard + (instance_id,)))
    knowledge_drop(state, [instance_id])


# --------------------------------------------------------------- 家計・賃金

def take_from_household(state, pid, amount, tx_sink=None):
    """TX-014: 家計から指定額の全額を得る(部分獲得はない。OPEN-003)。"""
    state.household -= amount
    state.players[pid].money += amount
    tx(tx_sink, "TX-014")
    event(tx_sink, "household_taken", player_id=pid, amount=amount)


def pay_to_household(state, pid, amount, tx_sink=None):
    """TX-013: 所持金から家計へ払う(賃金支払い)。

    実支払額が $0 でも記録する。RULES.md の TX-013 に非零条件はなく、
    RUL-INSOLV-001 は「所持金の全額を家計へ払う」経路の台帳に TX-013 を
    明示しているため、所持金0での支払いも取引として1件立つ。表示側が
    金額0を出したくない場合は `events` の `amount` で判断する。
    """
    state.players[pid].money -= amount
    state.household += amount
    tx(tx_sink, "TX-013")
    event(tx_sink, "wage_paid", player_id=pid, amount=amount)


def issue_unpaid_wages(state, pid, count, tx_sink=None):
    """TX-015: 不足$1につき未払い賃金1枚をサプライから受け取る。"""
    if count <= 0:
        return
    state.players[pid].unpaid_wages += count
    tx(tx_sink, "TX-015")
    event(tx_sink, "unpaid_wages_issued", player_id=pid, count=count)


# ------------------------------------------------------------------ ワーカー

def hire_training(state, cards, pid, count, tx_sink=None):
    """TX-024: 研修中ワーカーを雇う。上限を超えない分だけ(OPEN-018)。"""
    from .invariants import worker_cap
    room = worker_cap(state, cards, pid) - state.employed(pid)
    n = max(0, min(count, room))
    state.players[pid].workers_training += n
    if n:
        tx(tx_sink, "TX-024")
    return n


def hire_ready(state, cards, pid, count, tx_sink=None):
    """TX-027: 専門学校。研修中ではなく待機状態で追加する(当ラウンドから配置可)。"""
    from .invariants import worker_cap
    room = worker_cap(state, cards, pid) - state.employed(pid)
    n = max(0, min(count, room))
    state.players[pid].workers_waiting += n
    if n:
        tx(tx_sink, "TX-027")
    return n


def release_training(state, pid, tx_sink=None):
    """TX-004: 研修中ワーカーを通常(待機)状態へ表返す。"""
    p = state.players[pid]
    if p.workers_training:
        p.workers_waiting += p.workers_training
        p.workers_training = 0
        tx(tx_sink, "TX-004")


def recall_workers(state, pid, tx_sink=None):
    """TX-016: 当該プレイヤーの配置ワーカーを待機へ戻す。"""
    n = 0
    for w in state.workplaces.values():
        if pid in w.occupants:
            c = w.occupants.count(pid)
            w.occupants = tuple(x for x in w.occupants if x != pid)
            n += c
    state.players[pid].workers_waiting += n
    if n:
        tx(tx_sink, "TX-016")
    return n
