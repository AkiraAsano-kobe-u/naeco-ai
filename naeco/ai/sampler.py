"""観測だけから前進可能な完全情報局面を合成する。"""
import dataclasses
import math
import random

from naeco.engine import invariants
from naeco.engine.cards import (
    PHYSICAL_WORKERS_PER_COLOR, card_id_of, carpenter_count, load_cards,
)
from naeco.engine.observe import (
    OBS_VERSION, Observation, OpponentView, PendingView, WorkplaceView, observe,
)
from naeco.engine.state import (
    DecisionKind, GameState, KnowledgeEntry, KnowledgeKind, PendingDecision,
    PlayerState, WorkplaceState, derive_seed, encode_rng,
)


class SamplerError(Exception):
    """観測と両立する隠れ状態を構成できない。"""


def _count_for_sizes(count, sizes):
    result = math.factorial(count)
    for size in sizes:
        result //= math.factorial(size)
    return result


def _allocations_by_counts(s_count, other_count, deck_size, bin_sizes, caps):
    """有効な S の各ビン枚数と、その層の割当て数を辞書順で返す。"""
    if len(bin_sizes) != len(caps):
        raise ValueError("bin_sizes and caps length mismatch")
    if deck_size < 0 or any(x < 0 for x in bin_sizes):
        return []
    if s_count < deck_size:
        return []
    target = s_count - deck_size
    result = []

    def visit(index, remaining, prefix):
        if index == len(bin_sizes):
            if remaining != 0:
                return
            other_sizes = [bin_sizes[i] - prefix[i]
                           for i in range(len(bin_sizes))]
            if any(x < 0 for x in other_sizes) or sum(other_sizes) != other_count:
                return
            s_ways = _count_for_sizes(s_count, [deck_size] + prefix)
            other_ways = _count_for_sizes(other_count, other_sizes)
            result.append((tuple(prefix), s_ways * other_ways))
            return
        maximum = min(bin_sizes[index], caps[index], remaining)
        for value in range(maximum + 1):
            visit(index + 1, remaining - value, prefix + [value])

    visit(0, target, [])
    return result


def total_allocations(s_items, other_items, deck_size, bin_sizes, caps):
    """制約を満たす実体割当ての総数を整数で返す。"""
    return sum(weight for unused_counts, weight in _allocations_by_counts(
        len(s_items), len(other_items), deck_size, bin_sizes, caps))


def _unrank_combination(items, size, rank):
    """辞書順の ``rank`` 番目の組合せと残りを返す。"""
    if not (0 <= size <= len(items)):
        raise ValueError("bad combination size")
    total = math.comb(len(items), size)
    if not (0 <= rank < total):
        raise ValueError("combination rank out of range")
    selected = []
    start = 0
    remaining = size
    while remaining:
        for index in range(start, len(items)):
            below = math.comb(len(items) - index - 1, remaining - 1)
            if rank < below:
                selected.append(items[index])
                start = index + 1
                remaining -= 1
                break
            rank -= below
    selected_set = set(selected)
    return selected, [item for item in items if item not in selected_set]


def _unrank_partition(items, sizes, rank):
    """順序を持たない複数ビンへの辞書順の実体割当てを復元する。"""
    if sum(sizes) != len(items) or any(size < 0 for size in sizes):
        raise ValueError("partition sizes do not cover items")
    total = _count_for_sizes(len(items), sizes)
    if not (0 <= rank < total):
        raise ValueError("partition rank out of range")
    remaining = list(items)
    out = []
    for position, size in enumerate(sizes[:-1]):
        tail = _count_for_sizes(len(remaining) - size, sizes[position + 1:])
        choice_rank = rank // tail
        rank %= tail
        chosen, remaining = _unrank_combination(remaining, size, choice_rank)
        out.append(chosen)
    out.append(remaining)
    return out


def _unrank_allocation(s_items, other_items, deck_size, bin_sizes, caps, rank):
    """rank 番目の有効割当てを一意に返す。

    山札はソート済みタプルとして返し、各手札ビンのキーは ``bin_sizes`` の添字
    である。集合を返すと呼び出し側の反復順序が PYTHONHASHSEED に依存するため、
    決定順序をここで固定する。
    """
    s_items = tuple(sorted(s_items))
    other_items = tuple(sorted(other_items))
    layers = _allocations_by_counts(len(s_items), len(other_items), deck_size,
                                    bin_sizes, caps)
    total = sum(weight for unused_counts, weight in layers)
    if not (0 <= rank < total):
        raise ValueError("allocation rank out of range")
    for s_counts, weight in layers:
        if rank >= weight:
            rank -= weight
            continue
        s_ways = _count_for_sizes(len(s_items), [deck_size] + list(s_counts))
        other_rank = rank % _count_for_sizes(
            len(other_items), [bin_sizes[i] - s_counts[i]
                               for i in range(len(bin_sizes))])
        s_rank = rank // _count_for_sizes(
            len(other_items), [bin_sizes[i] - s_counts[i]
                               for i in range(len(bin_sizes))])
        s_bins = _unrank_partition(s_items, [deck_size] + list(s_counts), s_rank)
        other_bins = _unrank_partition(
            other_items,
            [bin_sizes[i] - s_counts[i] for i in range(len(bin_sizes))],
            other_rank)
        if s_ways <= 0:
            raise AssertionError("unreachable empty allocation layer")
        hands = {}
        for index in range(len(bin_sizes)):
            hands[index] = sorted(s_bins[index + 1] + other_bins[index])
        return tuple(sorted(s_bins[0])), hands
    raise AssertionError("allocation rank was not consumed")


def _absolute(pid, offset, count):
    return (pid + offset) % count


def _require_int(value, name):
    """外部入力の数値フィールドを int(bool は除外)に検証する。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SamplerError("%s must be an integer: %r" % (name, value))
    return value


def _require_worker_counts(value, name):
    """観測の worker 3要素列を型・長さ・値域まで構築前に検証する。"""
    if not isinstance(value, (tuple, list)):
        raise SamplerError("%s must be a tuple/list: %r" % (name, value))
    if len(value) != 3:
        raise SamplerError("%s must have exactly 3 elements: %r"
                           % (name, value))
    counts = tuple(_require_int(item, "%s entry" % name) for item in value)
    if any(count < 0 or count > PHYSICAL_WORKERS_PER_COLOR
           for count in counts):
        raise SamplerError(
            "%s entries must be in 0..%d: %r"
            % (name, PHYSICAL_WORKERS_PER_COLOR, value))
    return counts


def _require_building_sequence(value, name, cards):
    """pending の実体列を、tuple/list・建物実体ID・昇順一意まで検証する。

    `reserved_targets` / `selected_discards` はエンジンが昇順にしか作らない
    (game.py §3.5 R1b / R2)。不正な順序・重複・非建物IDを観測が運ぶと、
    復元後に legal_actions が実体と食い違うため、構築前に拒否する。
    """
    if not isinstance(value, (tuple, list)):
        raise SamplerError("%s must be a tuple/list: %r" % (name, value))
    previous = None
    for item in value:
        if not isinstance(item, str) or item not in cards.building_instances:
            raise SamplerError(
                "%s must contain building instance ids: %r" % (name, item))
        if previous is not None and item <= previous:
            raise SamplerError(
                "%s must be strictly ascending and unique: %r"
                % (name, value))
        previous = item
    return tuple(value)


def _require_sequence(value, name):
    """tuple 宣言フィールドの外側コンテナが tuple/list であることを要求する。"""
    if not isinstance(value, (tuple, list)):
        raise SamplerError("%s must be a tuple/list: %r" % (name, value))
    return value


def _validate_slash_and_burn_load(value, cards):
    """焼畑積載ペア列を型・長さ・既知実体・正値まで構築前に検証する。

    積載は `on_place__slash_and_burn` が自分の焼畑実体をキーに
    `params["amount"]`(cards.json では 5)を加算する経路だけが生成する
    (handlers.py §on_place__slash_and_burn / invariants INV-007)。
    """
    _require_sequence(value, "slash_and_burn_load")
    seen = set()
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise SamplerError(
                "slash_and_burn_load row must be exactly 2 elements: %r"
                % (row,))
        key, count = row
        if (not isinstance(key, str)
                or key not in cards.instances
                or card_id_of(key) != "base:slash-and-burn"):
            raise SamplerError(
                "slash_and_burn_load key must be a slash-and-burn instance: %r"
                % (key,))
        _require_int(count, "slash_and_burn_load value")
        if count <= 0:
            raise SamplerError(
                "slash_and_burn_load value must be positive: %r" % (count,))
        if key in seen:
            raise SamplerError("duplicate slash_and_burn_load key: %s" % (key,))
        seen.add(key)


_PENDING_SOURCE_IDS = frozenset(["RUL-ROUND-002", "RUL-INSOLV-001"])


def _known_pending_sources(cards):
    """エンジンが pending.source_effect_id に置く既知ID集合。"""
    return frozenset(cards.effect_hook) | _PENDING_SOURCE_IDS


def _known_knowledge_sources(cards):
    """public_knowledge source の生成可能集合。

    生成元は ops.reshuffle_deck の "TX-007" と、ops.knowledge_in_hand へ渡る
    pending.source_effect_id(game.py §_apply_select_keep)の2経路のみ。
    """
    return frozenset(cards.effect_hook) | frozenset(["TX-007"])


def _validate_pending_view(pv, cards, n):
    """actor の pending ビューが復元可能な構造を持つかを検査する。

    値域は handlers.py / game.py の生成経路に照合する:
    remaining は params["discard"] / 1 / 2 / 手札超過 / 賃金不足で負にならず、
    discount は params.get("discount", 0) 由来、discarded_goods は
    legal な捨て札選択から 0 以上、constraint_cost は build_cost 由来。
    """
    if pv is None:
        return
    if not isinstance(pv, PendingView):
        raise SamplerError("pending must be a PendingView: %r" % (pv,))
    if not isinstance(pv.kind, DecisionKind):
        raise SamplerError("pending kind must be a DecisionKind: %r"
                           % (pv.kind,))
    if pv.remaining is None or pv.reserved_targets is None:
        raise SamplerError("actor pending decision cannot be restored")
    _require_int(pv.actor_offset, "pending actor_offset")
    if not (0 <= pv.actor_offset < n):
        raise SamplerError("pending actor_offset is outside 0..%d: %r"
                           % (n - 1, pv.actor_offset))
    _require_int(pv.remaining, "pending remaining")
    if pv.remaining < 0:
        raise SamplerError("pending remaining must be non-negative: %r"
                           % (pv.remaining,))
    _require_int(pv.discount, "pending discount")
    if pv.discount < 0:
        raise SamplerError("pending discount must be non-negative: %r"
                           % (pv.discount,))
    if pv.discarded_goods is not None:
        _require_int(pv.discarded_goods, "pending discarded_goods")
        if pv.discarded_goods < 0:
            raise SamplerError(
                "pending discarded_goods must be non-negative: %r"
                % (pv.discarded_goods,))
    if pv.constraint_cost is not None:
        _require_int(pv.constraint_cost, "pending constraint_cost")
        if pv.constraint_cost < 0:
            raise SamplerError(
                "pending constraint_cost must be non-negative: %r"
                % (pv.constraint_cost,))
    if (not isinstance(pv.source_effect_id, str)
            or pv.source_effect_id not in _known_pending_sources(cards)):
        raise SamplerError(
            "pending source_effect_id is not a known effect or rule id: %r"
            % (pv.source_effect_id,))
    _require_building_sequence(pv.reserved_targets,
                               "pending reserved_targets", cards)
    _require_building_sequence(pv.selected_discards,
                               "pending selected_discards", cards)


def _validate_workplace_view(view, cards, n):
    """職場ビューをカード定義・実行時列挙契約に照合して検査する。"""
    if not isinstance(view.instance_id, str) or view.instance_id not in cards.instances:
        raise SamplerError("workplace instance_id must be a known instance: %r"
                           % (view.instance_id,))
    card_def = cards.of_instance(view.instance_id)
    if not card_def.is_workplace:
        raise SamplerError("workplace instance_id is not a workplace card: %r"
                           % (view.instance_id,))
    if view.logical_id != cards.logical_id_of(view.instance_id):
        raise SamplerError(
            "workplace logical_id %r does not match card %r"
            % (view.logical_id, cards.logical_id_of(view.instance_id)))
    if view.occupancy_mode != card_def.occupancy_mode:
        raise SamplerError(
            "workplace occupancy_mode %r does not match card %r"
            % (view.occupancy_mode, card_def.occupancy_mode))
    if view.usable_by not in ("any_player", "owner_only"):
        raise SamplerError("workplace usable_by must be any_player/owner_only: %r"
                           % (view.usable_by,))
    if (view.owner_offset is None) != (view.usable_by == "any_player"):
        raise SamplerError(
            "workplace owner_offset and usable_by disagree: %r / %r"
            % (view.owner_offset, view.usable_by))
    if view.owner_offset is not None:
        _require_int(view.owner_offset, "workplace owner_offset")
        if not (0 <= view.owner_offset < n):
            raise SamplerError(
                "workplace owner_offset is outside 0..%d: %r"
                % (n - 1, view.owner_offset))
    if not isinstance(view.can_place, bool):
        raise SamplerError("workplace can_place must be a bool: %r"
                           % (view.can_place,))
    if view.block_reason is not None and not isinstance(view.block_reason, str):
        raise SamplerError("workplace block_reason must be a string or None: %r"
                           % (view.block_reason,))
    for offset in view.occupant_offsets:
        _require_int(offset, "workplace occupant_offset")
        if not (0 <= offset < n):
            raise SamplerError(
                "workplace occupant_offset is outside 0..%d: %r"
                % (n - 1, offset))


def _normalize_observation(obs):
    """tuple 宣言フィールドの list 形入力を tuple へ整える。

    検証では tuple/list の両方を受け入れるが、復元後の観測は正準化された
    tuple になる。C1 round-trip を生の list 入力と比較すると必ず不一致に
    なるため、ここで一度だけ整えてから構築と比較の両方に使う。
    """
    others = tuple(dataclasses.replace(
        view, owned=tuple(view.owned), workers=tuple(view.workers))
                   for view in obs.others)
    workplaces = tuple(dataclasses.replace(
        view, occupant_offsets=tuple(view.occupant_offsets))
        for view in obs.workplaces)
    pending = obs.pending
    if pending is not None:
        pending = dataclasses.replace(
            pending,
            reserved_targets=tuple(pending.reserved_targets),
            selected_discards=tuple(pending.selected_discards))
    return dataclasses.replace(
        obs,
        my_hand_buildings=tuple(obs.my_hand_buildings),
        my_owned=tuple(obs.my_owned),
        my_workers=tuple(obs.my_workers),
        others=others,
        building_discard=tuple(obs.building_discard),
        workplaces=workplaces,
        slash_and_burn_load=tuple(tuple(row) for row in obs.slash_and_burn_load),
        temporary_public=tuple(obs.temporary_public),
        public_knowledge=tuple(tuple(row) for row in obs.public_knowledge),
        pending=pending,
        legal_actions=None if obs.legal_actions is None
        else tuple(obs.legal_actions),
        draws_since_reshuffle=tuple(obs.draws_since_reshuffle),
    )


def _validate_constructed_zones(state):
    """構成後の知識行が実在ゾーンと整合するかを production で検査する。

    `_validate_observation` は知識行の構造と値域しか見られないため、行が
    「公開ゾーンへ移動済みの実体」や「候補ゾーンに無い実体」を指すと、
    `invariants.check` は public_knowledge を見ずに通過しうる(sol SOL-P25G-001)。
    ここでは構成した状態の実在ゾーンと突き合わせ、失効済み知識を拒否する。
    """
    if not state.public_knowledge:
        return
    deckset = set(state.deck)
    eligible = tuple(j for j in range(state.num_players)
                     if state.draws_since_reshuffle[j] > 0)
    public = (set(state.building_discard) | set(state.temporary_public)
              | set(state.out_of_box) | set(state.round_deck)
              | set(state.workplaces))
    for player in state.players:
        public.update(player.owned_buildings)

    for entry in state.public_knowledge:
        if entry.kind == KnowledgeKind.IN_HAND:
            if entry.holder is None:
                raise SamplerError("IN_HAND knowledge lacks a holder")
            if entry.instance_id not in state.players[entry.holder].hand_buildings:
                raise SamplerError(
                    "IN_HAND %s is not in holder %d's hand"
                    % (entry.instance_id, entry.holder))
        elif entry.kind == KnowledgeKind.IN_DECK_SINCE:
            in_deck = entry.instance_id in deckset
            in_eligible = any(
                entry.instance_id in state.players[j].hand_buildings
                for j in eligible)
            if not (in_deck or in_eligible):
                raise SamplerError(
                    "IN_DECK_SINCE %s is neither in deck nor in a drawn hand"
                    % entry.instance_id)
        else:
            raise SamplerError("unknown public knowledge kind in constructed state")
        if entry.instance_id in public:
            raise SamplerError(
                "knowledge entry %s refers to a public or revoked zone"
                % entry.instance_id)


def _validate_observation(obs, cards):
    """入力 obs が履歴上あり得る形かを検査する(sol SOL-P25G-001)。

    `invariants.check` は `public_knowledge` を見ないので、壊れた知識集合を持つ
    観測は構成後の不変条件検査をすり抜ける。C4a/C4b/C4c が正しくても、その前提に
    なる知識集合が壊れていれば整合サンプルとはいえない。
    """
    if not isinstance(obs, Observation):
        raise SamplerError("obs must be an Observation: %r" % (obs,))
    if obs.obs_version != OBS_VERSION:
        raise SamplerError("obs_version mismatch: %r != %r"
                           % (obs.obs_version, OBS_VERSION))
    if (obs.cards_sha256 != cards.sha256
            or obs.cards_data_version != cards.data_version):
        raise SamplerError("card data does not match observation")
    n = _require_int(obs.num_players, "num_players")
    _require_int(obs.player_id, "player_id")
    if not (0 <= obs.player_id < n):
        raise SamplerError("player_id is outside 0..%d: %r"
                           % (n - 1, obs.player_id))
    _require_int(obs.round_number, "round_number")
    _require_int(obs.reshuffle_seq, "reshuffle_seq")
    _require_int(obs.deck_size, "deck_size")
    _require_int(obs.round_deck_size, "round_deck_size")
    _require_int(obs.end_phase_cursor, "end_phase_cursor")
    _require_int(obs.start_player_offset, "start_player_offset")
    _require_int(obs.current_player_offset, "current_player_offset")
    _require_int(obs.next_start_player_offset, "next_start_player_offset")
    _require_int(obs.household, "household")
    _require_int(obs.my_money, "my_money")
    _require_int(obs.my_hand_goods, "my_hand_goods")
    _require_int(obs.my_unpaid_wages, "my_unpaid_wages")
    for name in ("my_hand_buildings", "my_owned", "building_discard",
                 "temporary_public"):
        _require_sequence(getattr(obs, name), name)
    _require_worker_counts(obs.my_workers, "my_workers")
    if not (2 <= n <= 4):
        raise SamplerError("invalid player count")
    _require_sequence(obs.draws_since_reshuffle, "draws_since_reshuffle")
    if len(obs.draws_since_reshuffle) != n:
        raise SamplerError("draws_since_reshuffle length mismatch")
    for count in obs.draws_since_reshuffle:
        _require_int(count, "draws_since_reshuffle entry")
    if any(count < 0 for count in obs.draws_since_reshuffle):
        raise SamplerError("negative draw count")
    if obs.reshuffle_seq < 0:
        raise SamplerError("negative reshuffle_seq")
    _require_sequence(obs.public_knowledge, "public_knowledge")
    _require_sequence(obs.others, "others")
    for view in obs.others:
        if not isinstance(view, OpponentView):
            raise SamplerError("opponent view must be an OpponentView: %r"
                               % (view,))
        _require_int(view.seat_offset, "opponent seat_offset")
        _require_int(view.money, "opponent money")
        _require_int(view.hand_size, "opponent hand_size")
        _require_int(view.hand_goods, "opponent hand_goods")
        _require_int(view.unpaid_wages, "opponent unpaid_wages")
        _require_sequence(view.owned, "opponent owned")
        _require_worker_counts(view.workers, "opponent workers")
    _require_sequence(obs.workplaces, "workplaces")
    for view in obs.workplaces:
        if not isinstance(view, WorkplaceView):
            raise SamplerError("workplace view must be a WorkplaceView: %r"
                               % (view,))
        _require_sequence(view.occupant_offsets, "workplace occupant_offsets")
        _validate_workplace_view(view, cards, n)
    _validate_slash_and_burn_load(obs.slash_and_burn_load, cards)
    _validate_pending_view(obs.pending, cards, n)
    if obs.legal_actions is not None:
        _require_sequence(obs.legal_actions, "legal_actions")

    seen = set()
    for row in obs.public_knowledge:
        if not isinstance(row, (tuple, list)) or len(row) != 6:
            raise SamplerError(
                "public_knowledge row must be exactly 6 elements: %r" % (row,))
        instance_id, kind_value, holder_offset, sequence, round_number, source = row
        if (not isinstance(source, str)
                or source not in _known_knowledge_sources(cards)):
            raise SamplerError(
                "public_knowledge source is not a known source id: %r"
                % (source,))
        if (not isinstance(instance_id, str)
                or instance_id not in cards.building_instances):
            raise SamplerError("unknown or non-building instance id: %r"
                               % (instance_id,))
        if isinstance(round_number, bool) or not isinstance(round_number, int):
            raise SamplerError("knowledge round_number must be an int: %r"
                               % (round_number,))
        if not (1 <= round_number <= obs.round_number):
            raise SamplerError(
                "knowledge round_number %r is outside 1..%r"
                % (round_number, obs.round_number))
        if instance_id in seen:
            raise SamplerError("duplicate public_knowledge entry: %s" % instance_id)
        seen.add(instance_id)
        try:
            kind = KnowledgeKind(kind_value)
        except ValueError:
            raise SamplerError("unknown public knowledge kind: %r" % (kind_value,))
        if kind == KnowledgeKind.IN_HAND:
            if holder_offset is None:
                raise SamplerError("IN_HAND lacks a holder")
            _require_int(holder_offset, "knowledge holder_offset")
            if not (0 <= holder_offset < n):
                raise SamplerError("IN_HAND has a bad holder: %r" % (holder_offset,))
            if sequence is not None:
                raise SamplerError("IN_HAND must not carry a generation")
        else:   # IN_DECK_SINCE
            if holder_offset is not None:
                raise SamplerError("IN_DECK_SINCE must not have a holder")
            _require_int(sequence, "knowledge sequence")
            # 生成元は ops.reshuffle_deck だけで、そこで seq を 1 以上へ進める。
            if not (obs.reshuffle_seq > 0 and sequence == obs.reshuffle_seq):
                raise SamplerError(
                    "IN_DECK_SINCE generation %r does not match reshuffle_seq %r"
                    % (sequence, obs.reshuffle_seq))


def sample_hidden_state(obs, seed, cards=None):
    """obs と整合する完全情報 GameState を1つサンプルする。"""
    if not isinstance(obs, Observation):
        raise SamplerError("obs must be an Observation: %r" % (obs,))
    if obs.legal_actions is None:
        raise ValueError("sample_hidden_state requires an actor observation")
    cards = cards or load_cards()
    _validate_observation(obs, cards)
    obs = _normalize_observation(obs)

    pid = obs.player_id
    n = obs.num_players

    known = []
    fixed_hands = dict((seat, set()) for seat in range(n))
    since = set()
    for row in obs.public_knowledge:
        instance_id, kind_value, holder_offset, sequence, round_number, source = row
        try:
            kind = KnowledgeKind(kind_value)
        except ValueError:
            raise SamplerError("unknown public knowledge kind")
        holder = (None if holder_offset is None
                  else _absolute(pid, holder_offset, n))
        entry = KnowledgeEntry(instance_id, kind, holder, sequence,
                               round_number, source)
        known.append(entry)
        if kind == KnowledgeKind.IN_HAND:
            if holder is None:
                raise SamplerError("IN_HAND lacks a holder")
            fixed_hands[holder].add(instance_id)
        elif kind == KnowledgeKind.IN_DECK_SINCE:
            since.add(instance_id)

    if obs.pending is not None:
        pv = obs.pending
        pending = PendingDecision(
            pv.kind, _absolute(pid, pv.actor_offset, n), pv.source_effect_id,
            pv.remaining, pv.discount, tuple(pv.reserved_targets),
            tuple(pv.selected_discards), pv.discarded_goods,
            pv.constraint_cost,
        )
    else:
        pending = None

    fixed_hands[pid].update(obs.my_hand_buildings)
    if pending is not None:
        fixed_hands[pending.player_id].update(pending.reserved_targets)

    opponents = dict((o.seat_offset, o) for o in obs.others)
    if set(opponents) != set(range(1, n)):
        raise SamplerError("opponent offsets are not complete")
    players = []
    bins = []
    bin_seats = []
    for seat in range(n):
        if seat == pid:
            money = obs.my_money
            hand_goods = obs.my_hand_goods
            owned = tuple(sorted(obs.my_owned))
            workers = obs.my_workers
            unpaid = obs.my_unpaid_wages
        else:
            offset = (seat - pid) % n
            view = opponents[offset]
            money = view.money
            hand_goods = view.hand_goods
            owned = tuple(sorted(view.owned))
            workers = view.workers
            unpaid = view.unpaid_wages
            size = view.hand_size - view.hand_goods - len(fixed_hands[seat])
            if size < 0:
                raise SamplerError("known cards exceed an opponent hand")
            bins.append(size)
            bin_seats.append(seat)
        players.append(PlayerState(seat, money, tuple(sorted(fixed_hands[seat])),
                                   hand_goods, owned, workers[0], workers[1], unpaid))

    for seat in range(n):
        fixed_since = len(fixed_hands[seat] & since)
        draw_count = obs.draws_since_reshuffle[(seat - pid) % n]
        if fixed_since > draw_count:
            raise SamplerError("fixed cards exceed draw capacity")

    workplaces = {}
    for view in obs.workplaces:
        owner = (None if view.owner_offset is None
                 else _absolute(pid, view.owner_offset, n))
        occupants = tuple(_absolute(pid, offset, n)
                          for offset in view.occupant_offsets)
        workplaces[view.instance_id] = WorkplaceState(
            view.instance_id, view.logical_id, owner, view.usable_by,
            view.occupancy_mode, occupants)

    out_of_box = tuple(sorted(cards.instances_by_card["base:carpenter"][
        carpenter_count(n):]))
    round_deck = tuple(cards.round_instances[9 - obs.round_deck_size:])
    public_buildings = set(obs.building_discard) | set(obs.temporary_public)
    for player in players:
        public_buildings.update(player.owned_buildings)
    for instance_id in workplaces:
        if cards.of_instance(instance_id).card_type == "building":
            public_buildings.add(instance_id)
    fixed = set(players[pid].hand_buildings)
    for seat in range(n):
        fixed.update(fixed_hands[seat])
    unknown = set(cards.building_instances) - public_buildings - fixed
    if len(unknown) != obs.deck_size + sum(bins):
        raise SamplerError("unknown building count does not match slots")

    rng = random.Random(seed)
    unknown_sorted = tuple(sorted(unknown))
    if obs.reshuffle_seq == 0:
        if since:
            raise SamplerError("IN_DECK_SINCE exists before a reshuffle")
        shuffled = list(unknown_sorted)
        rng.shuffle(shuffled)
        deck_order = tuple(sorted(shuffled[:obs.deck_size]))
        assigned = {}
        cursor = obs.deck_size
        for index, size in enumerate(bins):
            assigned[index] = sorted(shuffled[cursor:cursor + size])
            cursor += size
    else:
        s_items = tuple(sorted(unknown & since))
        other_items = tuple(sorted(unknown - since))
        caps = []
        for seat, size in zip(bin_seats, bins):
            fixed_since = len(fixed_hands[seat] & since)
            caps.append(min(size, obs.draws_since_reshuffle[(seat - pid) % n]
                            - fixed_since))
        if any(cap < 0 for cap in caps):
            raise SamplerError("fixed cards exceed draw capacity")
        total = total_allocations(s_items, other_items, obs.deck_size, bins, caps)
        if total == 0:
            raise SamplerError("no valid hidden allocation")
        deck_order, assigned = _unrank_allocation(
            s_items, other_items, obs.deck_size, bins, caps, rng.randrange(total))

    for index, seat in enumerate(bin_seats):
        players[seat].hand_buildings = tuple(sorted(
            list(fixed_hands[seat]) + assigned[index]))
    deck = list(deck_order)
    rng.shuffle(deck)
    if len(deck) != obs.deck_size or any(
            cards.of_instance(item).card_type != "building" for item in deck):
        raise SamplerError("invalid deck allocation")

    # cards_mod の小さな定数だけを使うため、追加のカードデータ読込みはしない。
    state = GameState(
        num_players=n, round_number=obs.round_number, phase=obs.phase,
        start_player=_absolute(pid, obs.start_player_offset, n),
        next_start_player=_absolute(pid, obs.next_start_player_offset, n),
        current_player=_absolute(pid, obs.current_player_offset, n),
        end_phase_cursor=obs.end_phase_cursor, players=tuple(players),
        deck=tuple(deck), building_discard=tuple(sorted(obs.building_discard)),
        round_deck=round_deck, out_of_box=out_of_box, household=obs.household,
        workplaces=workplaces, slash_and_burn_load=dict(obs.slash_and_burn_load),
        temporary_public=tuple(sorted(obs.temporary_public)),
        public_knowledge=tuple(known), reshuffle_seq=obs.reshuffle_seq,
        draws_since_reshuffle=tuple(obs.draws_since_reshuffle[(seat - pid) % n]
                                    for seat in range(n)),
        pending=pending,
        rng_state=encode_rng(random.Random(derive_seed(seed, "det_env", 0))),
        history=(), cards_sha256=cards.sha256, cards_data_version=cards.data_version,
    )
    try:
        _validate_constructed_zones(state)
        invariants.check(state, cards, "always")
        if state.pending is None:
            invariants.check(state, cards, "after_resolution")
    except Exception as exc:
        raise SamplerError("sampled state violates invariants: {}".format(exc))
    # C1: 構成した状態の観測が入力と一致すること。ここまでの各制約が正しくても
    # 復元のどこか(席の戻し方・公開ゾーンの写し漏れ)を間違えれば観測がずれる。
    # Observation は frozen dataclass なので等値比較で足りる(直列化より安い)。
    restored = observe(state, pid, cards)
    if restored.player_id != obs.player_id or restored != obs:
        raise SamplerError("sampled state does not reproduce the observation")
    return state
