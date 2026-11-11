"""観測・行動エンコーダ(Phase 3 設計書 §5)。

`Observation` と `data/cards.json` だけから計算する。環境制御メタデータ(絶対席)と
実体番号を特徴に含めない。標準ライブラリのみを使う(§2)。
職場の公共フラグ・usable_by・占有モードは §13.1 に従い `obs.workplaces` の
`WorkplaceView`(実行時値)から取る。cards.json の `workplace.usable_by` は初期値に
すぎず特徴に使わない(core-api §5.4)。
"""
import hashlib
import json
import os

from naeco.engine import cards as cards_mod
from naeco.engine.cards import card_id_of
from naeco.engine.state import ActionKind, DecisionKind, Phase

OBS_ENCODER_VERSION = "1.0.0"
ACTION_ENCODER_VERSION = "1.0.0"

# --- ID 表(§5.1)。cards.json と enum から決定論的に導出する。 ---
_CARDS = cards_mod.load_cards()
CARD_ID_ORDER = tuple(sorted(_CARDS.defs))
TAG_ORDER = tuple(sorted(
    {tag for cd in _CARDS.defs.values() for tag in cd.tags}
))
ACTION_KIND_ORDER = tuple(sorted(k.value for k in ActionKind))
DECISION_KIND_ORDER = tuple(sorted(k.value for k in DecisionKind))
PHASE_ORDER = tuple(sorted(p.value for p in Phase))
# 職場ブロックの論理ID順。round 表面は cards.json から導出した面 ID を使う。
WORKPLACE_LOGICAL_ID_ORDER = tuple(sorted(
    {_CARDS.logical_id_of(iid) for iid in _CARDS.instances
     if _CARDS.of_instance(iid).workplace is not None}
    | {face for face in _CARDS.derived_round_face.values() if face is not None}
))
CARDS_SHA256 = _CARDS.sha256

_CARD_POS = {cid: i for i, cid in enumerate(CARD_ID_ORDER)}
_TAG_POS = {tag: i for i, tag in enumerate(TAG_ORDER)}
_ACTION_KIND_POS = {v: i for i, v in enumerate(ACTION_KIND_ORDER)}
_DECISION_KIND_POS = {v: i for i, v in enumerate(DECISION_KIND_ORDER)}
_PHASE_POS = {v: i for i, v in enumerate(PHASE_ORDER)}
_WORKPLACE_POS = {lid: i for i, lid in enumerate(WORKPLACE_LOGICAL_ID_ORDER)}


def _load_printed_points():
    """cards.json の printed_points は CardDef が公開していないため直接読む。"""
    with open(cards_mod.DATA_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {rec["card_id"]: rec.get("printed_points")
            for rec in raw["cards"]}


_PRINTED_POINTS = _load_printed_points()

# 正規化定数表(§5.2 / §5.3)。encoder_sha256() の対象。
_NORMALIZATION = {
    "round": 9.0,
    "money": 10.0,
    "money_cap": 10.0,
    "count": 10.0,
    "workplace": 4.0,
    "knowledge": 5.0,
    "draws": 20.0,
    "cost": 10.0,
    "value": 10.0,
    "effect": 10.0,
    "goods": 10.0,
    "remaining": 10.0,
    "lengths": 10.0,
}


def _money(value, nm):
    # 家計・所持金はゲームを通じて無制限に積み上がるため /10 に加えて上限で切る。
    return min(value / nm["money"], nm["money_cap"])


def _card_counts(instances, size):
    counts = [0.0] * size
    for iid in instances:
        counts[_CARD_POS[card_id_of(iid)]] += 1.0
    return counts


def _tag_counts(instances, cards, size):
    counts = [0.0] * size
    for iid in instances:
        for tag in cards.tags_of(iid):
            counts[_TAG_POS[tag]] += 1.0
    return counts


def obs_dim(num_players):
    """§5.2 のブロック構成を定数表から計算する。"""
    n = num_players
    global_size = 6 + len(PHASE_ORDER) + 2
    self_size = 2 + 2 * len(CARD_ID_ORDER) + 3 + 1 + len(TAG_ORDER)
    other_size = 4 + len(CARD_ID_ORDER) + 3 + 1 + len(TAG_ORDER)
    # 1論理IDあたり: 空き / 自分 / 他家(n-1) / 自分の所有 / owner_only / any_player
    workplace_size = len(WORKPLACE_LOGICAL_ID_ORDER) * (n + 4)
    knowledge_size = len(CARD_ID_ORDER) * (n + 1) + len(CARD_ID_ORDER)
    decision_size = 1 + len(DECISION_KIND_ORDER) + 2 + 1 + 2 + 2 + 1 + 1 + 1
    return (
        global_size
        + self_size
        + (n - 1) * other_size
        + len(CARD_ID_ORDER)
        + workplace_size
        + knowledge_size
        + 1 + len(CARD_ID_ORDER)
        + n
        + decision_size
    )


def encode_observation(obs, cards=None):
    """観測を float の list へ変換する(§5.2)。"""
    cards = cards_mod.load_cards() if cards is None else cards
    n = obs.num_players
    nm = _NORMALIZATION
    out = []

    # グローバル
    out.append(obs.round_number / nm["round"])
    out.append(cards.wage_rate(obs.round_number) / nm["money"])
    out.append(_money(obs.household, nm))
    out.append(obs.deck_size / nm["count"])
    out.append(obs.round_deck_size / nm["count"])
    out.append(obs.reshuffle_seq / nm["count"])
    phase = [0.0] * len(PHASE_ORDER)
    phase[_PHASE_POS[obs.phase.value]] = 1.0
    out.extend(phase)
    out.append(1.0 if obs.start_player_offset == 0 else 0.0)
    out.append(1.0 if obs.next_start_player_offset == 0 else 0.0)

    # 自分
    out.append(_money(obs.my_money, nm))
    out.append(obs.my_hand_goods / nm["count"])
    out.extend(_card_counts(obs.my_hand_buildings, len(CARD_ID_ORDER)))
    out.extend(_card_counts(obs.my_owned, len(CARD_ID_ORDER)))
    out.extend(x / nm["count"] for x in obs.my_workers)
    out.append(obs.my_unpaid_wages / nm["count"])
    out.extend(_tag_counts(obs.my_owned, cards, len(TAG_ORDER)))

    # 他家(offset 昇順)
    for other in obs.others:
        out.append(_money(other.money, nm))
        out.append(other.hand_size / nm["count"])
        out.append(other.hand_goods / nm["count"])
        out.append(other.hand_buildings_count / nm["count"])
        out.extend(_card_counts(other.owned, len(CARD_ID_ORDER)))
        out.extend(x / nm["count"] for x in other.workers)
        out.append(other.unpaid_wages / nm["count"])
        out.extend(_tag_counts(other.owned, cards, len(TAG_ORDER)))

    # 捨て札
    out.extend(_card_counts(obs.building_discard, len(CARD_ID_ORDER)))

    # 職場: 論理IDごとに「空き / 自分が占有 / 他家が占有(offset別) / 自分の所有 /
    # 実行時 usable_by 別の実体数」。usable_by は cards.json の静的定義ではなく
    # WorkplaceView の実行時値を使う(§13.1 / core-api §5.4)。
    stride = n + 4
    workplace = [0.0] * (len(WORKPLACE_LOGICAL_ID_ORDER) * stride)
    for w in obs.workplaces:
        base = _WORKPLACE_POS[w.logical_id] * stride
        if not w.occupant_offsets:
            workplace[base] += 1.0
        if 0 in w.occupant_offsets:
            workplace[base + 1] += 1.0
        for off in range(1, n):
            if off in w.occupant_offsets:
                workplace[base + 1 + off] += 1.0
        if w.owner_offset == 0:
            workplace[base + n + 1] += 1.0
        if w.usable_by == "owner_only":
            workplace[base + n + 2] += 1.0
        else:
            workplace[base + n + 3] += 1.0
    out.extend(x / nm["workplace"] for x in workplace)

    # 公開知識: card_id × holder_offset(+None 枠)。temporary_public は card_id 数。
    knowledge = [0.0] * (len(CARD_ID_ORDER) * (n + 1))
    for entry in obs.public_knowledge:
        card_pos = _CARD_POS[card_id_of(entry[0])]
        holder = entry[2]
        slot = n if holder is None else holder
        knowledge[card_pos * (n + 1) + slot] += 1.0
    out.extend(x / nm["knowledge"] for x in knowledge)
    out.extend(_card_counts(obs.temporary_public, len(CARD_ID_ORDER)))

    # 焼畑: 合計枚数と card_id ごとの積載実体数
    total = 0
    loads = [0.0] * len(CARD_ID_ORDER)
    for iid, count in obs.slash_and_burn_load:
        total += count
        if count > 0:
            loads[_CARD_POS[card_id_of(iid)]] += 1.0
    out.append(total / nm["count"])
    out.extend(x / nm["count"] for x in loads)

    # ドロー履歴(観測者相対)
    out.extend(x / nm["draws"] for x in obs.draws_since_reshuffle)

    # 決定文脈
    decision = [0.0] * (len(DECISION_KIND_ORDER) + 1)
    pending = obs.pending
    if pending is None:
        decision[-1] = 1.0
    else:
        decision[_DECISION_KIND_POS[pending.kind.value]] = 1.0
    out.extend(decision)
    if pending is None or pending.remaining is None:
        out.append(0.0)
        out.append(1.0)
    else:
        out.append(pending.remaining / nm["remaining"])
        out.append(0.0)
    out.append(0.0 if pending is None else pending.discount / nm["remaining"])
    if pending is None or pending.discarded_goods is None:
        out.append(0.0)
        out.append(1.0)
    else:
        out.append(pending.discarded_goods / nm["remaining"])
        out.append(0.0)
    if pending is None or pending.constraint_cost is None:
        out.append(0.0)
        out.append(1.0)
    else:
        out.append(pending.constraint_cost / nm["remaining"])
        out.append(0.0)
    out.append(
        0.0 if pending is None or pending.reserved_targets is None
        else len(pending.reserved_targets) / nm["lengths"]
    )
    out.append(
        0.0 if pending is None or pending.selected_discards is None
        else len(pending.selected_discards) / nm["lengths"]
    )
    out.append(1.0 if pending is not None and pending.actor_offset == 0 else 0.0)
    return out


def action_dim():
    """§5.3 のブロック構成を定数表から計算する。"""
    return (
        len(ACTION_KIND_ORDER)
        # 対象職場: card_id / 公共フラグ / usable_by / 占有モード / 存在フラグ(§13.1)
        + len(CARD_ID_ORDER) + 1 + 2 + 2 + 1
        + len(CARD_ID_ORDER) + 2 + 2 + 2 + len(TAG_ORDER)
        + 4
        + 1
        + 4
    )


def _workplace_view(workplaces, instance_id):
    """obs.workplaces から instance_id の WorkplaceView を引く(無ければ None)。"""
    for w in workplaces:
        if w.instance_id == instance_id:
            return w
    return None


def encode_action(obs, action, cards=None):
    """行動を float の list へ変換する(§5.3)。

    対象職場ブロックの公共職場フラグ・usable_by・占有モードは cards.json の静的定義
    ではなく `obs.workplaces` の実行時値(WorkplaceView)から作る(§13.1 /
    core-api §5.4)。`action.target_instance_id` が `obs.workplaces` に無い場合
    (山札・手札・建物捨て札内の職場カード等)は存在フラグ 0 とし、実行時値3種は全て
    0 にする。card_id one-hot だけは静的定義から立てる。
    """
    cards = cards_mod.load_cards() if cards is None else cards
    nm = _NORMALIZATION
    out = []

    # 種別
    kind = [0.0] * len(ACTION_KIND_ORDER)
    kind[_ACTION_KIND_POS[action.kind.value]] = 1.0
    out.extend(kind)

    # 対象職場: card_id は静的定義、公共職場フラグ・usable_by・占有モードは
    # obs.workplaces の実行時値(§13.1)。owner=None は公共職場の定義(core-api §1.2.1)。
    target = [0.0] * len(CARD_ID_ORDER)
    public_flag = 0.0
    usable_by = [0.0, 0.0]
    occupancy = [0.0, 0.0]
    workplace_present = 0.0
    if action.target_instance_id is not None:
        cd = cards.of_instance(action.target_instance_id)
        target[_CARD_POS[cd.card_id]] = 1.0
        wv = _workplace_view(obs.workplaces, action.target_instance_id)
        if wv is not None:
            workplace_present = 1.0
            if wv.owner_offset is None:
                public_flag = 1.0
            if wv.usable_by == "owner_only":
                usable_by[0] = 1.0
            else:
                usable_by[1] = 1.0
            if wv.occupancy_mode == "exclusive":
                occupancy[0] = 1.0
            else:
                occupancy[1] = 1.0
    out.extend(target)
    out.append(public_flag)
    out.extend(usable_by)
    out.extend(occupancy)
    out.append(workplace_present)

    # 対象カード(instance_id)。build_cost 等の None は存在フラグで区別する。
    inst = [0.0] * len(CARD_ID_ORDER)
    build_cost = 0.0
    build_cost_missing = 1.0
    printed_value = 0.0
    printed_value_missing = 1.0
    printed_points = 0.0
    printed_points_missing = 1.0
    tags = [0.0] * len(TAG_ORDER)
    effects = [0.0, 0.0, 0.0, 0.0]
    if action.instance_id is not None:
        cd = cards.of_instance(action.instance_id)
        inst[_CARD_POS[cd.card_id]] = 1.0
        if cd.build_cost is not None:
            build_cost = cd.build_cost / nm["cost"]
            build_cost_missing = 0.0
        if cd.printed_value is not None:
            printed_value = cd.printed_value / nm["value"]
            printed_value_missing = 0.0
        if _PRINTED_POINTS.get(cd.card_id) is not None:
            printed_points = _PRINTED_POINTS[cd.card_id] / nm["value"]
            printed_points_missing = 0.0
        for tag in cd.tags:
            tags[_TAG_POS[tag]] = 1.0
    out.extend(inst)
    out.extend([build_cost, build_cost_missing])
    out.extend([printed_value, printed_value_missing])
    out.extend([printed_points, printed_points_missing])
    out.extend(tags)

    # 効果量: 行動が指すカード(instance 優先、なければ職場)の on_place フック。
    effect_card_id = None
    if action.instance_id is not None:
        effect_card_id = card_id_of(action.instance_id)
    elif action.target_instance_id is not None:
        effect_card_id = card_id_of(action.target_instance_id)
    if effect_card_id is not None:
        eff_id = cards.on_place_effect.get(effect_card_id)
        if eff_id is not None:
            params = cards.effect_params[eff_id]
            effects[0] = params.get("draw", 0) / nm["effect"]
            effects[1] = params.get("amount", 0) / nm["effect"]
            effects[2] = params.get("discard", 0) / nm["effect"]
            effects[3] = params.get("points_per", 0) / nm["effect"]
    out.extend(effects)

    # 数量
    out.append(action.goods_count / nm["goods"])

    # 文脈との関係(観測から計算できる範囲)
    meets_remaining = 0.0
    remaining_after_pay = 0.0
    reserved_target_match = 0.0
    required_goods_match = 0.0
    pending = obs.pending
    if pending is not None and action.instance_id is not None:
        cd = cards.of_instance(action.instance_id)
        if pending.remaining is not None and cd.build_cost is not None:
            if cd.build_cost >= pending.remaining:
                meets_remaining = 1.0
            remaining_after_pay = (
                pending.remaining - cd.build_cost
            ) / nm["remaining"]
        if (pending.reserved_targets is not None
                and action.instance_id in pending.reserved_targets):
            reserved_target_match = 1.0
    if (pending is not None and pending.remaining is not None
            and pending.kind in (DecisionKind.DISCARD_N,
                                 DecisionKind.HAND_LIMIT_DISCARD)
            and action.goods_count == pending.remaining):
        required_goods_match = 1.0
    out.extend([meets_remaining, remaining_after_pay,
                reserved_target_match, required_goods_match])
    return out


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def _source_bytes():
    with open(os.path.abspath(__file__), "rb") as fh:
        return fh.read()


def encoder_sha256():
    """(obs_encoder_sha256, action_encoder_sha256) を返す(§3.2)。"""
    payload = (
        _source_bytes()
        + _canonical(_NORMALIZATION)
        + _canonical(CARD_ID_ORDER)
        + _canonical(TAG_ORDER)
        + _canonical(ACTION_KIND_ORDER)
        + _canonical(DECISION_KIND_ORDER)
        + _canonical(PHASE_ORDER)
        + _canonical(WORKPLACE_LOGICAL_ID_ORDER)
        + _canonical(_PRINTED_POINTS)
        + _canonical(CARDS_SHA256)
    )
    return (
        hashlib.sha256(payload + b"|obs").hexdigest(),
        hashlib.sha256(payload + b"|action").hexdigest(),
    )
