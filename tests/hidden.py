"""隠れ状態の整合性判定とスクランブル(設計書 §7.4 / §7.5)。

`tests/test_observe.py` から移設した共有ヘルパー(Phase 2.5)。
漏洩テスト(§7.4)と隠れ状態サンプラーの受入オラクル(§7.5 C1〜C6)の両方から使う。

**オラクルは条件ごとに独立した関数へ分解する**(sol P25-018/P25-022)。単一の
真偽値関数だと、あるテストが「どの条件で落ちたか」を区別できず、先に別の条件で
落ちているだけのテストを「制約に歯がある」と誤認する。

C4 の内訳(いずれも `ops.py` の実装から導かれる履歴上の必要条件):

- **C4a 候補ゾーン**: `IN_HAND` は指定保持者の手札、`IN_DECK_SINCE` は
  山札か「再構成以降に引いた家」の手札にしかない。
- **C4b 容量**: `#(IN_DECK_SINCE ∩ hand_j) <= draws_since_reshuffle[j]`。
  `ops.draw` は1枚引くごとにカウンタを加算する。
- **C4c 逆向き**: `reshuffle_seq > 0` なら `deck ⊆ IN_DECK_SINCE`。
  `ops.reshuffle_deck` は捨て札**全体**に印を付け、以後カードが山札へ入る経路は
  次の再構成だけである(`ops.draw` は知識を落とさない)。
  容量制約だけでは必要条件にしかならない。
"""
import random

import helpers as H
from naeco.engine import invariants
from naeco.engine.observe import observe, serialize_observation
from naeco.engine.state import KnowledgeKind, clone


def eligible_deck_since_players(s):
    """IN_DECK_SINCE 実体を保持しうるプレイヤー(§7.3): 再構成以降に引いた者。"""
    return set(j for j in range(s.num_players) if s.draws_since_reshuffle[j] > 0)


def deck_since_instances(s):
    return set(e.instance_id for e in s.public_knowledge
               if e.kind == KnowledgeKind.IN_DECK_SINCE)


# ------------------------------------------------------------ 条件ごとの判定

def check_invariants(s):
    """C2/C3: 全不変条件。"""
    try:
        invariants.check(s, H.CARDS, "always")
        if s.pending is None:
            invariants.check(s, H.CARDS, "after_resolution")
    except Exception:
        return False
    return True


def check_observation_matches(s2, s_orig, viewer):
    """C1: viewer の観測が一致する。"""
    return (serialize_observation(observe(s2, viewer, H.CARDS))
            == serialize_observation(observe(s_orig, viewer, H.CARDS)))


def check_candidate_zones(s):
    """C4a: public_knowledge の各実体が候補ゾーンのいずれかにある。"""
    elig = eligible_deck_since_players(s)
    deckset = set(s.deck)
    for e in s.public_knowledge:
        if e.kind == KnowledgeKind.IN_HAND:
            if e.holder is None:
                return False
            if e.instance_id not in s.players[e.holder].hand_buildings:
                return False
        elif e.kind == KnowledgeKind.IN_DECK_SINCE:
            in_deck = e.instance_id in deckset
            in_elig = any(e.instance_id in s.players[j].hand_buildings for j in elig)
            if not (in_deck or in_elig):
                return False
    return True


def check_draw_capacity(s):
    """C4b: 各家が持つ IN_DECK_SINCE 枚数はドロー枚数以下。"""
    since = deck_since_instances(s)
    for j in range(s.num_players):
        held = sum(1 for i in s.players[j].hand_buildings if i in since)
        if held > s.draws_since_reshuffle[j]:
            return False
    return True


def check_deck_is_reshuffled_subset(s):
    """C4c: 再構成後の山札は IN_DECK_SINCE 実体だけから成る。"""
    if s.reshuffle_seq <= 0:
        return True
    since = deck_since_instances(s)
    return all(i in since for i in s.deck)


def check_public_zones(s2, s_orig):
    """C5: 公開ゾーンの内容が一致する(観測一致とは独立に検査する)。"""
    if (sorted(s2.building_discard) != sorted(s_orig.building_discard)
            or sorted(s2.temporary_public) != sorted(s_orig.temporary_public)
            or sorted(s2.out_of_box) != sorted(s_orig.out_of_box)
            or list(s2.round_deck) != list(s_orig.round_deck)
            or s2.slash_and_burn_load != s_orig.slash_and_burn_load):
        return False
    if sorted(s2.workplaces) != sorted(s_orig.workplaces):
        return False
    for k, w in s2.workplaces.items():
        w0 = s_orig.workplaces[k]
        if (w.owner, w.usable_by, w.occupancy_mode, w.occupants) != (
                w0.owner, w0.usable_by, w0.occupancy_mode, w0.occupants):
            return False
    for p2, p0 in zip(s2.players, s_orig.players):
        if sorted(p2.owned_buildings) != sorted(p0.owned_buildings):
            return False
    return True


def check_hand_counts(s2, s_orig):
    """C6: 各家の手札合計と**消費財枚数**が一致する(いずれも公開値)。"""
    for p2, p0 in zip(s2.players, s_orig.players):
        if p2.hand_total != p0.hand_total or p2.hand_goods != p0.hand_goods:
            return False
    return True


def check_knowledge_shape(s):
    """知識エントリの形状(kind ごとの holder / reshuffle_seq と重複)。

    sol P25-027: `reshuffle_seq == 0` では `IN_DECK_SINCE` が1件も存在してはならない
    (`ops.reshuffle_deck` が唯一の生成元であり、そこで seq を 1 以上へ進めるため)。
    """
    seen = set()
    for e in s.public_knowledge:
        if e.instance_id in seen:
            return False                      # 同一実体の重複エントリ
        seen.add(e.instance_id)
        if e.kind == KnowledgeKind.IN_HAND:
            if e.holder is None or not (0 <= e.holder < s.num_players):
                return False
            if e.reshuffle_seq is not None:
                return False
        elif e.kind == KnowledgeKind.IN_DECK_SINCE:
            if e.holder is not None:
                return False
            # 世代は現在の再構成回数と一致し、かつ 1 以上である。
            if not (s.reshuffle_seq > 0 and e.reshuffle_seq == s.reshuffle_seq):
                return False
    return True


def check_pending_reserved(s):
    """pending の建設予約札は決定主体の手札に残っている。"""
    if s.pending is None or not s.pending.reserved_targets:
        return True
    hand = s.players[s.pending.player_id].hand_buildings
    return all(i in hand for i in s.pending.reserved_targets)


CONDITIONS = (
    ("C2/C3 invariants", lambda s2, s0, v: check_invariants(s2)),
    ("C1 observation", check_observation_matches),
    ("C4a candidate zones", lambda s2, s0, v: check_candidate_zones(s2)),
    ("C4b draw capacity", lambda s2, s0, v: check_draw_capacity(s2)),
    ("C4c deck subset", lambda s2, s0, v: check_deck_is_reshuffled_subset(s2)),
    ("C5 public zones", lambda s2, s0, v: check_public_zones(s2, s0)),
    ("C6 hand counts", lambda s2, s0, v: check_hand_counts(s2, s0)),
    ("knowledge shape", lambda s2, s0, v: check_knowledge_shape(s2)),
    ("pending reserved", lambda s2, s0, v: check_pending_reserved(s2)),
)


def failing_condition(s2, s_orig, viewer):
    """最初に破れた条件名を返す。すべて満たすなら None。

    どの条件で落ちたかを返すことで、「別の条件で先に落ちているだけ」のテストを
    見分けられる(sol P25-018)。
    """
    for name, fn in CONDITIONS:
        if not fn(s2, s_orig, viewer):
            return name
    return None


def is_consistent_hidden(s2, s_orig, viewer):
    """s2 が「viewer の観測を保ちつつ、履歴上あり得る」隠れ状態か(§7.5 C1〜C6)。

    sol Phase1 #1: 漏洩テストは**整合した**隠れ状態同士で比較しなければならない。
    非整合な状態(不可能な配札)を生成すると、観測一致が成り立っても意味がない。
    """
    return failing_condition(s2, s_orig, viewer) is None


# ------------------------------------------------------------ スクランブル

def _pinned_by_player(s, viewer):
    """各プレイヤーの手札に固定される実体(IN_HAND / pending予約)。"""
    pinned = dict((pid, set()) for pid in range(s.num_players))
    for e in s.public_knowledge:
        if e.kind == KnowledgeKind.IN_HAND and e.holder is not None:
            pinned[e.holder].add(e.instance_id)
    if s.pending is not None and s.pending.reserved_targets:
        for i in s.pending.reserved_targets:
            pinned[s.pending.player_id].add(i)
    return pinned


def _try_scramble(s, viewer, rng):
    """整合性を狙って隠れ状態を1つ生成する(採択は呼び出し側が検査する)。

    **消費財枚数は保存する**(derived-public。RULES.md OPEN-027 / core-api §7.4)。
    振り直すのは「他家の手札に入る**建物実体の同一性**」と「山札の内容・順序」だけである。
    枚数を振り直すと、正しい実装でも観測が変わって偽陽性になる。
    """
    s2 = clone(s)
    n = s.num_players
    D = len(s.deck)
    pinned = _pinned_by_player(s, viewer)

    since = deck_since_instances(s)
    elig = eligible_deck_since_players(s)
    reshuffled = s.reshuffle_seq > 0
    others = [pid for pid in range(n) if pid != viewer]

    # 動かせる建物プールと、各他家の**建物枠**(消費財枚数は動かさない)。
    movable = list(s.deck)
    building_slots = {}
    since_used = {}
    for pid in others:
        p = s.players[pid]
        for b in p.hand_buildings:
            if b not in pinned[pid]:
                movable.append(b)
        building_slots[pid] = len(p.hand_buildings) - len(pinned[pid])
        # 固定札のうち IN_DECK_SINCE のものは既に容量を消費している(f_j)。
        since_used[pid] = len(pinned[pid] & since)

    # C4c: 再構成後の山札は IN_DECK_SINCE 実体だけから成る。
    since_pool = [b for b in movable if b in since]
    other_pool = [b for b in movable if b not in since]
    rng.shuffle(since_pool)
    rng.shuffle(other_pool)
    if reshuffled:
        if len(since_pool) < D:
            return None                       # 山札を S だけで埋められない
        deck_new, to_hands = since_pool[:D], since_pool[D:] + other_pool
    else:
        pool = since_pool + other_pool        # reshuffle_seq==0 なら since_pool は空
        rng.shuffle(pool)
        deck_new, to_hands = pool[:D], pool[D:]
    rng.shuffle(to_hands)

    # 残りを建物枠へ配る。C4b の容量を超えないこと。
    hand_assign = dict((pid, []) for pid in others)
    order = list(others)
    for b in to_hands:
        rng.shuffle(order)                    # 配り先を偏らせない
        placed = False
        for pid in order:
            if len(hand_assign[pid]) >= building_slots[pid]:
                continue
            if b in since:
                if (pid not in elig
                        or since_used[pid] >= s.draws_since_reshuffle[pid]):
                    continue
                since_used[pid] += 1
            hand_assign[pid].append(b)
            placed = True
            break
        if not placed:
            return None                       # 制約下で収容できない -> 棄却

    for pid in others:
        if len(hand_assign[pid]) != building_slots[pid]:
            return None
        p2 = s2.players[pid]
        p2.hand_buildings = tuple(sorted(list(pinned[pid]) + hand_assign[pid]))
        p2.hand_goods = s.players[pid].hand_goods       # ★保存する(derived-public)
    s2.deck = tuple(deck_new)
    return s2


def _permute_deck_only(s, viewer, rng):
    """最小の整合スクランブル: 山札の順序だけ変える(常に整合)。"""
    s2 = clone(s)
    deck = list(s.deck)
    rng.shuffle(deck)
    s2.deck = tuple(deck)
    return s2


def scramble_hidden(s, viewer, seed):
    """viewer に隠れている部分だけを入れ替えた**整合する** s' を作る(§7.4)。

    採択棄却で整合状態のみ返す。制約が厳しく実体を入れ替えられない局面では、
    山札順のみを変える最小スクランブルにフォールバックする(それでも整合)。
    """
    for attempt in range(60):
        rng = random.Random((seed << 8) ^ attempt)
        cand = _try_scramble(s, viewer, rng)
        if cand is not None and is_consistent_hidden(cand, s, viewer):
            return cand
    fallback = _permute_deck_only(s, viewer, random.Random(seed ^ 0xDEAD))
    assert is_consistent_hidden(fallback, s, viewer), "fallback must be consistent"
    return fallback


def internals_differ(s, s2, viewer):
    """隠れ情報(他家手札の建物実体 or 山札)が何か変化したか。"""
    return hand_internals_differ(s, s2, viewer) or s.deck != s2.deck


def hand_internals_differ(s, s2, viewer):
    """★他家の手札の**建物実体の集合**が変化したか(山札順・消費財枚数は除外)。

    消費財枚数は derived-public になったので保存される(OPEN-027)。判定対象は
    建物実体の同一性だけである。「山札順だけの変化」を成功と数えないための厳密判定。
    """
    for pid in range(s.num_players):
        if pid == viewer:
            continue
        if (frozenset(s.players[pid].hand_buildings)
                != frozenset(s2.players[pid].hand_buildings)):
            return True
    return False
