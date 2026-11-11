"""Observation と情報境界(設計書 §6, §7)。

非公開は2つだけ: (1)自分以外の手札**内容**(枚数は公開)、(2)山札の順序・内容。
ただし「値そのものが非公開か」ではなく「**非公開の値から導出されるか**」で
判定しなければならない(§7.2.1 の `remaining` が典型)。
"""
import dataclasses
import json
from typing import Any, Dict, List, Optional, Tuple

from . import cards as cards_mod
from . import game as game_mod
from .state import (
    Action, ActionKind, DecisionKind, KnowledgeEntry, Phase,
)

# 0.2.0: WorkplaceView.block_reason を追加(2026-07-21)。観測の内容が増えたので
# 版を上げる。serialize_observation の出力も変わるため 0.1.0 とは比較できない。
# 0.3.0: Phase 2.5(2026-07-21)。`sample_hidden_state` が obs だけから GameState を
# 再構成できるようにするため、公開情報を追加した(設計書 §7.2、RULES.md OPEN-027):
#   - OpponentView.hand_goods … derived-public。公開イベントから一意に定まる
#   - next_start_player_offset / end_phase_cursor / reshuffle_seq /
#     draws_since_reshuffle … いずれも卓上で数えられる公開情報
#   - cards_sha256 / cards_data_version … カードデータの取り違え検出用
OBS_VERSION = "0.3.0"


@dataclasses.dataclass(frozen=True)
class OpponentView:
    seat_offset: int
    money: int
    hand_size: int              # 建物+消費財の合計
    # 消費財枚数は derived-public(RULES.md OPEN-027)。増加経路は農場・大農園
    # (定数)・果樹園(公開の手札合計から決まる)・焼畑回収(公開の積載枚数)、
    # 減少経路は捨て札(枚数が公開)、初期値0。よって公開情報の関数である。
    # 非公開のままなのは「手札にある建物がどの実体か」。
    hand_goods: int
    owned: Tuple[str, ...]
    workers: Tuple[int, int, int]   # waiting / training / placed
    unpaid_wages: int

    @property
    def hand_buildings_count(self):
        """建物枚数も derived-public(合計 − 消費財枚数)。"""
        return self.hand_size - self.hand_goods


@dataclasses.dataclass(frozen=True)
class WorkplaceView:
    instance_id: str
    logical_id: str
    owner_offset: Optional[int]
    usable_by: str
    occupancy_mode: str
    occupant_offsets: Tuple[int, ...]
    can_place: bool
    # 置けない理由(観測者自身が手番のときのみ。置けるなら None)。
    # 観測者自身の手札と公開情報だけから決まるので非公開情報は含まない(§7.2.1)。
    block_reason: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class PendingView:
    kind: DecisionKind
    actor_offset: int
    source_effect_id: str
    remaining: Optional[int]                      # 非公開由来なら他家に None
    discount: int
    discarded_goods: Optional[int]
    constraint_cost: Optional[int]                # 本人のみ
    reserved_targets: Optional[Tuple[str, ...]]   # 本人のみ
    selected_discards: Optional[Tuple[str, ...]]  # 全員に公開(捨て札は公開ゾーン)


@dataclasses.dataclass(frozen=True)
class Observation:
    obs_version: str
    player_id: int
    num_players: int
    round_number: int
    phase: Phase
    start_player_offset: int
    current_player_offset: int
    household: int
    my_money: int
    my_hand_buildings: Tuple[str, ...]
    my_hand_goods: int
    my_owned: Tuple[str, ...]
    my_workers: Tuple[int, int, int]
    my_unpaid_wages: int
    others: Tuple[OpponentView, ...]
    building_discard: Tuple[str, ...]
    workplaces: Tuple[WorkplaceView, ...]
    deck_size: int
    round_deck_size: int
    slash_and_burn_load: Tuple[Tuple[str, int], ...]
    temporary_public: Tuple[str, ...]
    public_knowledge: Tuple[Tuple[Any, ...], ...]
    pending: Optional[PendingView]
    legal_actions: Optional[Tuple[Action, ...]]
    # --- 0.3.0 追加。すべて公開情報(§7.2)---
    next_start_player_offset: int           # TX-023 の予約。卓上のマーカー
    end_phase_cursor: int                   # PHASE-END の処理位置(席IDではない)
    reshuffle_seq: int                      # 山札再構成の回数
    # index = seat_offset(0 = 自分)。**絶対席順で持ってはならない**(§9.3)。
    draws_since_reshuffle: Tuple[int, ...]
    cards_sha256: str
    cards_data_version: str


# `remaining` が非公開値から導出される DecisionKind(§7.2.1)。
# BUILD_COST の remaining は max(0, cost(reserved_target) - discount) であり、
# discount は職場から既知なので、公開すると手札にある建設対象のコストが割れる。
_REMAINING_PRIVATE_KINDS = frozenset([DecisionKind.BUILD_COST])


def _offset(pid, viewer, n):
    return (pid - viewer) % n


def observe(state, player_id, cards=None):
    """player_id から見た観測。非公開情報を一切含まない。"""
    cards = cards or cards_mod.load_cards()
    n = state.num_players
    me = state.players[player_id]
    acting = game_mod.actor(state)

    others = []
    for off in range(1, n):
        pid = (player_id + off) % n
        p = state.players[pid]
        others.append(OpponentView(
            seat_offset=off,
            money=p.money,
            hand_size=p.hand_total,
            hand_goods=p.hand_goods,          # derived-public(OPEN-027)
            owned=tuple(sorted(p.owned_buildings)),
            workers=(p.workers_waiting, p.workers_training, state.placed_count(pid)),
            unpaid_wages=p.unpaid_wages,
        ))

    # 配置可否を評価してよいのは「観測者自身が配置手番にいる」ときだけ。
    # 同じゲートを block_reason にも掛けることで、他家の観測に自分の手札由来の
    # 差分が出ない(§7.4 の漏洩テストが要求する性質)。
    my_place_turn = (
        state.phase == Phase.PLACE
        and state.pending is None
        and state.current_player == player_id
        and me.workers_waiting > 0
    )

    wviews = []
    for instance_id in sorted(state.workplaces):
        w = state.workplaces[instance_id]
        reason = None
        if my_place_turn:
            reason = game_mod.placement_block_reason(state, cards, player_id, w)
        wviews.append(WorkplaceView(
            instance_id=w.instance_id,
            logical_id=w.logical_id,
            owner_offset=None if w.owner is None else _offset(w.owner, player_id, n),
            usable_by=w.usable_by,
            occupancy_mode=w.occupancy_mode,
            occupant_offsets=tuple(_offset(x, player_id, n) for x in w.occupants),
            can_place=my_place_turn and reason is None,
            block_reason=reason,
        ))

    pend_view = None
    if state.pending is not None:
        pend = state.pending
        is_self = pend.player_id == player_id
        remaining = pend.remaining
        if not is_self and pend.kind in _REMAINING_PRIVATE_KINDS:
            remaining = None
        pend_view = PendingView(
            kind=pend.kind,
            actor_offset=_offset(pend.player_id, player_id, n),
            source_effect_id=pend.source_effect_id,
            remaining=remaining,
            discount=pend.discount,
            discarded_goods=pend.discarded_goods,
            constraint_cost=pend.constraint_cost if is_self else None,
            reserved_targets=pend.reserved_targets if is_self else None,
            selected_discards=pend.selected_discards,
        )

    # 合法手には actor の手札実体が含まれるので、本人にしか渡さない(§7.2.1)。
    legal = None
    if acting == player_id:
        legal = tuple(game_mod.legal_actions(state, cards))

    knowledge = tuple(
        (
            e.instance_id,
            e.kind.value,
            None if e.holder is None else _offset(e.holder, player_id, n),
            e.reshuffle_seq,
            e.round_number,
            e.source,
        )
        for e in state.public_knowledge
    )

    return Observation(
        obs_version=OBS_VERSION,
        player_id=player_id,
        num_players=n,
        round_number=state.round_number,
        phase=state.phase,
        start_player_offset=_offset(state.start_player, player_id, n),
        current_player_offset=_offset(state.current_player, player_id, n),
        household=state.household,
        my_money=me.money,
        my_hand_buildings=tuple(sorted(me.hand_buildings)),
        my_hand_goods=me.hand_goods,
        my_owned=tuple(sorted(me.owned_buildings)),
        my_workers=(me.workers_waiting, me.workers_training,
                    state.placed_count(player_id)),
        my_unpaid_wages=me.unpaid_wages,
        others=tuple(others),
        building_discard=tuple(sorted(state.building_discard)),
        workplaces=tuple(wviews),
        deck_size=len(state.deck),
        round_deck_size=len(state.round_deck),
        slash_and_burn_load=tuple(
            (k, state.slash_and_burn_load[k])
            for k in sorted(state.slash_and_burn_load)
        ),
        temporary_public=tuple(sorted(state.temporary_public)),
        public_knowledge=knowledge,
        pending=pend_view,
        legal_actions=legal,
        next_start_player_offset=_offset(state.next_start_player, player_id, n),
        end_phase_cursor=state.end_phase_cursor,
        reshuffle_seq=state.reshuffle_seq,
        # 観測者相対に並べ替える。絶対席順のまま渡すと §9.3 に反し、しかも
        # 同一観測者どうしの比較では検出できない(席回転対称性テストが要る)。
        draws_since_reshuffle=tuple(
            state.draws_since_reshuffle[(player_id + off) % n] for off in range(n)
        ),
        cards_sha256=state.cards_sha256,
        cards_data_version=state.cards_data_version,
    )


def serialize_observation(obs):
    """漏洩テスト(§7.4)で2つの観測を厳密比較するための正準化。"""
    body = {
        "obs_version": obs.obs_version,
        "num_players": obs.num_players,
        "round_number": obs.round_number,
        "phase": obs.phase.value,
        "start_player_offset": obs.start_player_offset,
        "current_player_offset": obs.current_player_offset,
        "household": obs.household,
        "my": [obs.my_money, list(obs.my_hand_buildings), obs.my_hand_goods,
               list(obs.my_owned), list(obs.my_workers), obs.my_unpaid_wages],
        "others": [
            [o.seat_offset, o.money, o.hand_size, o.hand_goods, list(o.owned),
             list(o.workers), o.unpaid_wages]
            for o in obs.others
        ],
        "building_discard": list(obs.building_discard),
        "workplaces": [
            [w.instance_id, w.logical_id, w.owner_offset, w.usable_by,
             w.occupancy_mode, list(w.occupant_offsets), w.can_place,
             w.block_reason]
            for w in obs.workplaces
        ],
        "deck_size": obs.deck_size,
        "round_deck_size": obs.round_deck_size,
        "slash_and_burn_load": [list(x) for x in obs.slash_and_burn_load],
        "temporary_public": list(obs.temporary_public),
        "public_knowledge": [list(x) for x in obs.public_knowledge],
        "pending": None if obs.pending is None else [
            obs.pending.kind.value, obs.pending.actor_offset,
            obs.pending.source_effect_id, obs.pending.remaining,
            obs.pending.discount, obs.pending.discarded_goods,
            obs.pending.constraint_cost,
            None if obs.pending.reserved_targets is None
            else list(obs.pending.reserved_targets),
            None if obs.pending.selected_discards is None
            else list(obs.pending.selected_discards),
        ],
        "legal_actions": None if obs.legal_actions is None else [
            list(a.sort_key()) for a in obs.legal_actions
        ],
        "next_start_player_offset": obs.next_start_player_offset,
        "end_phase_cursor": obs.end_phase_cursor,
        "reshuffle_seq": obs.reshuffle_seq,
        "draws_since_reshuffle": list(obs.draws_since_reshuffle),
        "cards_sha256": obs.cards_sha256,
        "cards_data_version": obs.cards_data_version,
    }
    # player_id は環境制御用のメタデータであり、観測の内容ではない(§8.2)。
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")
