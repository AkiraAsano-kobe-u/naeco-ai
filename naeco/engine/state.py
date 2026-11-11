"""GameState と clone / serialize / state_hash(設計書 §2)。

正準化規則は §2.3 の表に従う。RNG の正準エンコードは §2.5。
"""
import dataclasses
import enum
import hashlib
import json
import random
import struct
from typing import Any, Dict, List, Optional, Tuple

from . import cards as cards_mod

API_VERSION = "0.1.0"
# 0.3.1: OPEN-027(他家の手札の消費財枚数を derived-public に区分)。状態遷移は変わらないが、
# RUL-INFO-001 が許す情報集合が変わるため、ルール裁定版として上げる(sol P25-019)。
# これにより serialize ヘッダが変わり state_hash も変わる。0.3.0 で生成した
# docs/eval-runs/*.jsonl の initial_state_hash は歴史的成果物として保持する
# (勝敗・配札そのものは不変。docs/eval-seeds.md の注記を参照)。
RULES_VERSION = "0.3.1"
SERIAL_MAGIC = b"NAECO1"

RNG_BLOB_LEN = 2 + 624 * 4 + 4


class Zone(enum.Enum):
    SUPPLY = "supply"
    DECK = "deck"
    HAND = "hand"
    BUILDING_DISCARD = "building_discard"
    OWNED_BUILDING = "owned_building"
    PUBLIC_WORKPLACE = "public_workplace"
    ON_CARD = "on_card"
    TEMPORARY_PUBLIC = "temporary_public"
    ROUND_DECK = "round_deck"
    OUT_OF_BOX = "out_of_box"


class Phase(enum.Enum):
    ROUND_START = "round_start"
    PLACE = "place"
    END = "end"
    GAME_OVER = "game_over"


class ActionKind(enum.Enum):
    PLACE_WORKER = "place_worker"
    SELECT_BUILD = "select_build"
    SELECT_DISCARD = "select_discard"
    SELECT_KEEP = "select_keep"
    SELL_BUILDING = "sell_building"


class DecisionKind(enum.Enum):
    BUILD_TARGET = "build_target"
    BUILD_COST = "build_cost"
    DISCARD_N = "discard_n"
    DESIGN_OFFICE_KEEP = "design_office_keep"
    HAND_LIMIT_DISCARD = "hand_limit_discard"
    INSOLVENCY_SELL = "insolvency_sell"


class KnowledgeKind(enum.Enum):
    IN_HAND = "in_hand"
    IN_DECK_SINCE = "in_deck_since"


class LegalityError(Exception):
    """legal_actions() にない Action を step() に渡した。エンジンのバグを示す。

    状態は変更されない。学習ループでは局を破棄する(設計書 §9.4)。
    """


class InvariantError(Exception):
    """不変条件違反。エンジンのバグ、または壊れた状態の注入を示す。"""


class EffectContextError(Exception):
    """呼ばれてはならない文脈でハンドラが呼ばれた。エンジンのバグを示す。"""


@dataclasses.dataclass(frozen=True)
class Action:
    kind: ActionKind
    target_instance_id: Optional[str] = None
    instance_id: Optional[str] = None
    goods_count: int = 0

    def sort_key(self):
        return (
            self.kind.value,
            self.target_instance_id or "",
            self.instance_id or "",
            self.goods_count,
        )


@dataclasses.dataclass(frozen=True)
class KnowledgeEntry:
    instance_id: str
    kind: KnowledgeKind
    holder: Optional[int]
    reshuffle_seq: Optional[int]
    round_number: int
    source: str


@dataclasses.dataclass(frozen=True)
class HistoryEntry:
    round_number: int
    actor: int
    kind: ActionKind
    target_instance_id: Optional[str]
    revealed_instance_id: Optional[str]
    goods_count: int


@dataclasses.dataclass(frozen=True)
class PendingDecision:
    kind: DecisionKind
    player_id: int
    source_effect_id: str
    remaining: int
    discount: int = 0
    reserved_targets: Tuple[str, ...] = ()
    selected_discards: Tuple[str, ...] = ()
    discarded_goods: Optional[int] = None   # None=未選択 / 0以上=選択済み(§3.2.1)
    constraint_cost: Optional[int] = None


@dataclasses.dataclass
class WorkplaceState:
    instance_id: str
    logical_id: str
    owner: Optional[int]
    usable_by: str            # "any_player" | "owner_only"(実行時に変わる。§5.4)
    occupancy_mode: str       # "exclusive" | "unlimited"
    occupants: Tuple[int, ...] = ()

    def copy(self):
        return WorkplaceState(
            self.instance_id,
            self.logical_id,
            self.owner,
            self.usable_by,
            self.occupancy_mode,
            self.occupants,
        )


@dataclasses.dataclass
class PlayerState:
    player_id: int
    money: int
    hand_buildings: Tuple[str, ...]
    hand_goods: int
    owned_buildings: Tuple[str, ...]
    workers_waiting: int
    workers_training: int
    unpaid_wages: int

    def copy(self):
        return PlayerState(
            self.player_id,
            self.money,
            self.hand_buildings,
            self.hand_goods,
            self.owned_buildings,
            self.workers_waiting,
            self.workers_training,
            self.unpaid_wages,
        )

    @property
    def hand_total(self):
        return len(self.hand_buildings) + self.hand_goods


@dataclasses.dataclass
class GameState:
    num_players: int
    round_number: int
    phase: Phase
    start_player: int
    next_start_player: int
    current_player: int
    end_phase_cursor: int
    players: Tuple[PlayerState, ...]
    deck: Tuple[str, ...]
    building_discard: Tuple[str, ...]
    round_deck: Tuple[str, ...]
    out_of_box: Tuple[str, ...]
    household: int
    workplaces: Dict[str, WorkplaceState]
    slash_and_burn_load: Dict[str, int]
    temporary_public: Tuple[str, ...]
    public_knowledge: Tuple[KnowledgeEntry, ...]
    reshuffle_seq: int
    draws_since_reshuffle: Tuple[int, ...]
    pending: Optional[PendingDecision]
    rng_state: bytes
    history: Tuple[HistoryEntry, ...]
    cards_sha256: str
    cards_data_version: str

    # --- 便宜アクセサ ---
    def player(self, pid):
        return self.players[pid]

    def placed_count(self, pid):
        """配置済みワーカー数は状態に持たず occupants から導出する(§2.3)。"""
        n = 0
        for w in self.workplaces.values():
            n += w.occupants.count(pid)
        return n

    def employed(self, pid):
        p = self.players[pid]
        return p.workers_waiting + p.workers_training + self.placed_count(pid)


def clone(state):
    # type: (GameState) -> GameState
    """完全な深いコピー(RNG状態を含む)。§2.2。

    タプル・不変オブジェクトは共有してよい。可変なのは2つの辞書と
    PlayerState / WorkplaceState だけなので、そこだけを複製する。
    """
    wp = {}
    for k, v in state.workplaces.items():
        wp[k] = WorkplaceState(
            v.instance_id, v.logical_id, v.owner, v.usable_by,
            v.occupancy_mode, v.occupants,
        )
    return GameState(
        num_players=state.num_players,
        round_number=state.round_number,
        phase=state.phase,
        start_player=state.start_player,
        next_start_player=state.next_start_player,
        current_player=state.current_player,
        end_phase_cursor=state.end_phase_cursor,
        players=tuple(p.copy() for p in state.players),
        deck=state.deck,
        building_discard=state.building_discard,
        round_deck=state.round_deck,
        out_of_box=state.out_of_box,
        household=state.household,
        workplaces=wp,
        slash_and_burn_load=dict(state.slash_and_burn_load),
        temporary_public=state.temporary_public,
        public_knowledge=state.public_knowledge,
        reshuffle_seq=state.reshuffle_seq,
        draws_since_reshuffle=state.draws_since_reshuffle,
        pending=state.pending,
        rng_state=state.rng_state,
        history=state.history,
        cards_sha256=state.cards_sha256,
        cards_data_version=state.cards_data_version,
    )


# ---------------------------------------------------------------- RNG (§2.5)

def encode_rng(rng):
    """random.Random の状態を固定長バイト列へ写す(pickle を正準形にしない)。"""
    version, internal, gauss_next = rng.getstate()
    if version != 3 or len(internal) != 625:
        raise ValueError("unsupported RNG state")
    if gauss_next is not None:
        raise ValueError("environment RNG must not use gauss()")
    words, index = internal[:624], internal[624]
    return (struct.pack("<H", version)
            + b"".join(struct.pack("<I", w) for w in words)
            + struct.pack("<I", index))


def decode_rng(data):
    if len(data) != RNG_BLOB_LEN:
        raise ValueError("bad RNG blob length")
    version = struct.unpack("<H", data[:2])[0]
    words = struct.unpack("<624I", data[2:2 + 624 * 4])
    index = struct.unpack("<I", data[2 + 624 * 4:])[0]
    r = random.Random()
    r.setstate((version, tuple(words) + (index,), None))
    return r


# 席交換ペアの2局で**同一でなければならない**ストリーム(§2.5)。
#   "env"   … 初期シャッフル。配札が変わるとペアの意味が消える
#   "setup" … 開始プレイヤーの抽選。席割当だけを交換するので開始席も共通にする
# ここに載るストリームは seat_variant を無視する。
SEAT_INDEPENDENT_STREAMS = frozenset(["env", "setup"])

# 後方互換の別名(旧称)。
ENV_STREAMS = SEAT_INDEPENDENT_STREAMS


def derive_seed(root_seed, stream_kind, episode_index, seat_variant=0):
    """論理的な座標だけから決定論的に seed を派生する(§2.5)。"""
    variant = 0 if stream_kind in SEAT_INDEPENDENT_STREAMS else seat_variant
    payload = "naeco/v1/{}/{}/{}/{}".format(
        root_seed, stream_kind, episode_index, variant).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


GROUPS = frozenset(["train", "dev", "accept"])


def eval_seed(group, index):
    """評価seedの3群分離(§10.1)。"""
    if group not in GROUPS:
        raise ValueError("unknown seed group: {!r}".format(group))
    payload = "naeco/v1/{}/{}".format(group, index).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# -------------------------------------------------------- serialize (§2.2/2.3)

def _canonical_body(state):
    """正準化表(§2.3)に従って JSON 化可能な構造へ落とす。"""
    return {
        "num_players": state.num_players,
        "round_number": state.round_number,
        "phase": state.phase.value,
        "start_player": state.start_player,
        "next_start_player": state.next_start_player,
        "current_player": state.current_player,
        "end_phase_cursor": state.end_phase_cursor,
        # ordered: 席順固定
        "players": [
            [
                p.player_id,
                p.money,
                list(sorted(p.hand_buildings)),      # set-like
                p.hand_goods,
                list(sorted(p.owned_buildings)),     # set-like
                p.workers_waiting,
                p.workers_training,
                p.unpaid_wages,
            ]
            for p in state.players
        ],
        "deck": list(state.deck),                    # ordered
        "round_deck": list(state.round_deck),        # ordered
        "building_discard": list(sorted(state.building_discard)),   # set-like
        "out_of_box": list(sorted(state.out_of_box)),               # set-like
        "temporary_public": list(sorted(state.temporary_public)),   # set-like
        "household": state.household,
        # set-like: キー昇順の疎辞書
        "workplaces": [
            [
                w.instance_id, w.logical_id, w.owner, w.usable_by,
                w.occupancy_mode, list(w.occupants),   # occupants は ordered
            ]
            for w in (state.workplaces[k] for k in sorted(state.workplaces))
        ],
        # set-like: 値0のキーは持たない(§2.3)
        "slash_and_burn_load": [
            [k, state.slash_and_burn_load[k]]
            for k in sorted(state.slash_and_burn_load)
        ],
        # ordered: 発生順
        "public_knowledge": [
            [e.instance_id, e.kind.value, e.holder, e.reshuffle_seq,
             e.round_number, e.source]
            for e in state.public_knowledge
        ],
        "reshuffle_seq": state.reshuffle_seq,
        "draws_since_reshuffle": list(state.draws_since_reshuffle),
        "pending": None if state.pending is None else [
            state.pending.kind.value,
            state.pending.player_id,
            state.pending.source_effect_id,
            state.pending.remaining,
            state.pending.discount,
            list(state.pending.reserved_targets),
            list(state.pending.selected_discards),
            state.pending.discarded_goods,
            state.pending.constraint_cost,
        ],
        # ordered: 発生順
        "history": [
            [h.round_number, h.actor, h.kind.value, h.target_instance_id,
             h.revealed_instance_id, h.goods_count]
            for h in state.history
        ],
    }


def serialize(state, include_history=True):
    # type: (GameState, bool) -> bytes
    header = {
        "api_version": API_VERSION,
        "rules_version": RULES_VERSION,
        "cards_data_version": state.cards_data_version,
        "cards_sha256": state.cards_sha256,
    }
    body = _canonical_body(state)
    if not include_history:
        body.pop("history")
    hdr = json.dumps(header, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return b"".join([
        SERIAL_MAGIC,
        struct.pack("<I", len(hdr)), hdr,
        state.rng_state,
        struct.pack("<I", len(payload)), payload,
    ])


def deserialize(data, cards=None, check_invariants=True):
    # type: (bytes, Any, bool) -> GameState
    cards = cards or cards_mod.load_cards()
    if data[:len(SERIAL_MAGIC)] != SERIAL_MAGIC:
        raise ValueError("bad magic")
    pos = len(SERIAL_MAGIC)
    (hdr_len,) = struct.unpack("<I", data[pos:pos + 4])
    pos += 4
    header = json.loads(data[pos:pos + hdr_len].decode("utf-8"))
    pos += hdr_len
    rng_state = data[pos:pos + RNG_BLOB_LEN]
    pos += RNG_BLOB_LEN
    (body_len,) = struct.unpack("<I", data[pos:pos + 4])
    pos += 4
    body = json.loads(data[pos:pos + body_len].decode("utf-8"))

    if header["api_version"] != API_VERSION:
        raise ValueError("api_version mismatch: %r" % header["api_version"])
    if header["rules_version"] != RULES_VERSION:
        raise ValueError("rules_version mismatch: %r" % header["rules_version"])
    if header["cards_data_version"] != cards.data_version:
        raise ValueError("cards_data_version mismatch")
    if header["cards_sha256"] != cards.sha256:
        raise ValueError("cards.json hash mismatch")

    players = tuple(
        PlayerState(
            player_id=r[0], money=r[1], hand_buildings=tuple(r[2]),
            hand_goods=r[3], owned_buildings=tuple(r[4]),
            workers_waiting=r[5], workers_training=r[6], unpaid_wages=r[7],
        )
        for r in body["players"]
    )
    workplaces = {}
    for r in body["workplaces"]:
        workplaces[r[0]] = WorkplaceState(r[0], r[1], r[2], r[3], r[4], tuple(r[5]))
    pend = body["pending"]
    pending = None
    if pend is not None:
        pending = PendingDecision(
            kind=DecisionKind(pend[0]), player_id=pend[1],
            source_effect_id=pend[2], remaining=pend[3], discount=pend[4],
            reserved_targets=tuple(pend[5]), selected_discards=tuple(pend[6]),
            discarded_goods=pend[7], constraint_cost=pend[8],
        )
    state = GameState(
        num_players=body["num_players"],
        round_number=body["round_number"],
        phase=Phase(body["phase"]),
        start_player=body["start_player"],
        next_start_player=body["next_start_player"],
        current_player=body["current_player"],
        end_phase_cursor=body["end_phase_cursor"],
        players=players,
        deck=tuple(body["deck"]),
        building_discard=tuple(body["building_discard"]),
        round_deck=tuple(body["round_deck"]),
        out_of_box=tuple(body["out_of_box"]),
        household=body["household"],
        workplaces=workplaces,
        slash_and_burn_load=dict((k, v) for k, v in body["slash_and_burn_load"]),
        temporary_public=tuple(body["temporary_public"]),
        public_knowledge=tuple(
            KnowledgeEntry(e[0], KnowledgeKind(e[1]), e[2], e[3], e[4], e[5])
            for e in body["public_knowledge"]
        ),
        reshuffle_seq=body["reshuffle_seq"],
        draws_since_reshuffle=tuple(body["draws_since_reshuffle"]),
        pending=pending,
        rng_state=rng_state,
        history=tuple(
            HistoryEntry(h[0], h[1], ActionKind(h[2]), h[3], h[4], h[5])
            for h in body.get("history", [])
        ),
        cards_sha256=header["cards_sha256"],
        cards_data_version=header["cards_data_version"],
    )
    if check_invariants:
        from . import invariants
        invariants.check(state, cards, scope="deserialize")
    return state


def state_hash(state):
    # type: (GameState) -> str
    return hashlib.sha256(serialize(state)).hexdigest()
