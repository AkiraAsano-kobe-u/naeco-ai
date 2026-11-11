"""不変条件の検査(設計書 付録C)。

スコープ:
  always            … いつでも成立(pending の途中を含む)
  after_resolution  … pending is None になった時点
  after_round       … PHASE-END の当該プレイヤーの処理完了後

これを最初に用意しておくと、以降のデバッグでエンジンの誤りが即座に露見する。
"""
from . import cards as cards_mod
from .state import InvariantError, Phase, Zone

SCOPE_ALWAYS = "always"
SCOPE_AFTER_RESOLUTION = "after_resolution"
SCOPE_AFTER_ROUND = "after_round"


def _fail(inv_id, message):
    raise InvariantError("%s: %s" % (inv_id, message))


def check(state, cards=None, scope=SCOPE_ALWAYS):
    """scope に応じて不変条件を検査する。違反があれば InvariantError。

    scope="deserialize" は always 全件 + (pending is None なら) after_resolution。
    """
    cards = cards or cards_mod.load_cards()
    _check_always(state, cards)
    if scope == SCOPE_AFTER_RESOLUTION or (
        scope == "deserialize" and state.pending is None
    ):
        _check_after_resolution(state, cards)
    if scope == SCOPE_AFTER_ROUND:
        _check_after_resolution(state, cards)
        _check_after_round(state, cards)


# ------------------------------------------------------------------- always

def _check_always(state, cards):
    n = state.num_players
    if not (2 <= n <= 4):
        _fail("INV-000", "num_players=%d" % n)
    if len(state.players) != n:
        _fail("INV-000", "players length mismatch")
    if not (1 <= state.round_number <= 9):
        _fail("INV-000", "round_number=%d" % state.round_number)

    _check_canonical_order(state)
    _check_money(state)
    _check_building_conservation(state, cards)
    _check_public_workplaces(state, cards)
    _check_carpenters(state, cards)
    _check_round_cards(state, cards)
    _check_workers(state, cards)
    _check_occupancy(state, cards)
    _check_goods(state)
    _check_workplace_membership(state, cards)
    _check_pending_continuation(state, cards)


def _check_canonical_order(state):
    """set-like フィールドが正準順(昇順)で保持されていること(§2.3)。

    これが崩れると serialize と実状態が食い違い、deserialize(serialize(s)) == s
    が壊れる。安いので always で検査する。
    """
    for p in state.players:
        if list(p.hand_buildings) != sorted(p.hand_buildings):
            _fail("CANON", "player %d hand_buildings not sorted" % p.player_id)
        if list(p.owned_buildings) != sorted(p.owned_buildings):
            _fail("CANON", "player %d owned_buildings not sorted" % p.player_id)
    for name in ("building_discard", "out_of_box", "temporary_public"):
        v = getattr(state, name)
        if list(v) != sorted(v):
            _fail("CANON", "%s not sorted" % name)
    for k, v in state.slash_and_burn_load.items():
        if v <= 0:
            _fail("CANON", "slash_and_burn_load has non-positive value for %s" % k)


def _check_money(state):
    """INV-001 / INV-004: 金銭は非負。サプライは無限代用。"""
    if state.household < 0:
        _fail("INV-004", "household=%d" % state.household)
    for p in state.players:
        if p.money < 0:
            _fail("INV-004", "player %d money=%d" % (p.player_id, p.money))
        if p.unpaid_wages < 0:
            _fail("INV-004", "player %d unpaid_wages<0" % p.player_id)


def _building_locations(state, cards):
    """建物実体 -> ゾーンの一覧(重複検出のため list で返す)。"""
    loc = {}

    def put(instance_id, zone, owner=None):
        loc.setdefault(instance_id, []).append((zone, owner))

    for i in state.deck:
        put(i, Zone.DECK)
    for p in state.players:
        for i in p.hand_buildings:
            put(i, Zone.HAND, p.player_id)
        for i in p.owned_buildings:
            put(i, Zone.OWNED_BUILDING, p.player_id)
    for i in state.building_discard:
        put(i, Zone.BUILDING_DISCARD)
    for i in state.temporary_public:
        put(i, Zone.TEMPORARY_PUBLIC)
    # 売却された建物のみ public_workplace ゾーンに数える。
    # 所有建物のうち職場のものは owned_building 側で既に数えている。
    for w in state.workplaces.values():
        if cards.of_instance(w.instance_id).card_type != "building":
            continue
        if w.owner is None:
            put(w.instance_id, Zone.PUBLIC_WORKPLACE)
    return loc


def _check_building_conservation(state, cards):
    """INV-002a: 建物64実体がちょうど1ゾーンに1回ずつ存在する。"""
    loc = _building_locations(state, cards)
    for instance_id in cards.building_instances:
        places = loc.get(instance_id, [])
        if len(places) != 1:
            _fail("INV-002a", "%s appears in %d zones: %r"
                  % (instance_id, len(places), places))
    extra = set(loc) - set(cards.building_instances)
    if extra:
        _fail("INV-002a", "unknown building instances: %r" % sorted(extra))


def _check_public_workplaces(state, cards):
    """INV-002b: 初期公共職場(採石場・鉱山・学校)は常に存在し移動しない。"""
    for card_id in ("base:quarry", "base:mine", "base:school"):
        instance_id = cards.instances_by_card[card_id][0]
        w = state.workplaces.get(instance_id)
        if w is None:
            _fail("INV-002b", "%s missing from workplaces" % instance_id)
        if w.owner is not None or w.usable_by != "any_player":
            _fail("INV-002b", "%s is not a public workplace" % instance_id)


def _check_carpenters(state, cards):
    """INV-002c: 大工3実体は {public_workplace, out_of_box} のちょうど1つ。

    多重集合で数え、out_of_box の重複(同一実体が2回)も検出する(sol Phase1 #4)。
    """
    ids = cards.instances_by_card["base:carpenter"]
    idset = set(ids)
    placed = [i for i in ids if i in state.workplaces]
    boxed_all = [i for i in state.out_of_box]     # out_of_box 全体(大工以外の混入検出)
    boxed = [i for i in boxed_all if i in idset]
    # out_of_box には大工以外を入れない。
    if len(boxed_all) != len(boxed):
        _fail("INV-002c", "out_of_box contains non-carpenter: %r" % boxed_all)
    # 多重集合: どの実体も全ゾーン合わせてちょうど1回。
    if len(boxed) != len(set(boxed)):
        _fail("INV-002c", "carpenter duplicated in out_of_box: %r" % boxed_all)
    if len(placed) != cards_mod.carpenter_count(state.num_players):
        _fail("INV-002c", "carpenters in play=%d expected=%d"
              % (len(placed), cards_mod.carpenter_count(state.num_players)))
    if len(set(placed)) + len(set(boxed)) != len(ids):
        _fail("INV-002c", "carpenter instances lost: placed=%r boxed=%r"
              % (placed, boxed))
    if set(placed) & set(boxed):
        _fail("INV-002c", "carpenter in two zones")


def _check_round_cards(state, cards):
    """INV-002d: ラウンドカード9実体は {round_deck, public_workplace} のちょうど1つ。"""
    ids = set(cards.round_instances)
    in_deck = [i for i in state.round_deck if i in ids]
    if len(in_deck) != len(state.round_deck):
        _fail("INV-002d", "round_deck contains non-round instances")
    if len(set(in_deck)) != len(in_deck):
        _fail("INV-002d", "round_deck has duplicate round instances")
    revealed = [i for i in ids if i in state.workplaces]
    if len(set(in_deck)) + len(revealed) != 9:
        _fail("INV-002d", "round cards: deck=%d revealed=%d"
              % (len(in_deck), len(revealed)))
    if set(in_deck) & set(revealed):
        _fail("INV-002d", "round card in two zones")
    if len(state.round_deck) != 9 - len(revealed):
        _fail("INV-002d", "round_deck size mismatch")


def worker_cap(state, cards, pid):
    """雇用上限 min(5 + 所有社宅数, 7)(RUL-WORKER-002, OPEN-017)。"""
    housing = sum(
        1 for i in state.players[pid].owned_buildings
        if cards_mod.card_id_of(i) == "base:company-housing"
    )
    return min(cards_mod.BASE_WORKER_CAP + housing,
               cards_mod.PHYSICAL_WORKERS_PER_COLOR)


def hand_limit(state, cards, pid):
    """手札上限 5 + 4*所有倉庫数(RUL-ROUND-002, OPEN-019)。"""
    warehouses = sum(
        1 for i in state.players[pid].owned_buildings
        if cards_mod.card_id_of(i) == "base:warehouse"
    )
    return cards_mod.BASE_HAND_LIMIT + 4 * warehouses


def _check_workers(state, cards):
    """INV-003: ワーカー所在と雇用上限。"""
    for pid in range(state.num_players):
        p = state.players[pid]
        if p.workers_waiting < 0 or p.workers_training < 0:
            _fail("INV-003", "player %d negative worker count" % pid)
        employed = state.employed(pid)
        if employed > cards_mod.PHYSICAL_WORKERS_PER_COLOR:
            _fail("INV-003", "player %d employed=%d > 7" % (pid, employed))
        cap = worker_cap(state, cards, pid)
        if employed > cap:
            _fail("INV-003", "player %d employed=%d > cap=%d" % (pid, employed, cap))


def _check_occupancy(state, cards):
    """INV-005: 通常職場は各ラウンド1ワーカー。鉱山のみ無制限。

    無制限職場でも占有者のplayer IDは範囲内でなければならない(sol Phase1 #4:
    unlimited を即 continue すると範囲外IDを見逃す)。
    """
    for w in state.workplaces.values():
        for pid in w.occupants:
            if not (0 <= pid < state.num_players):
                _fail("INV-005", "%s occupied by unknown player %r"
                      % (w.instance_id, pid))
        if w.occupancy_mode == "unlimited":
            continue
        if len(w.occupants) > 1:
            _fail("INV-005", "%s has %d occupants" % (w.instance_id, len(w.occupants)))


def _check_goods(state):
    """INV-007: 消費財は枚数で保持され非負(OPEN-024)。"""
    for p in state.players:
        if p.hand_goods < 0:
            _fail("INV-007", "player %d hand_goods<0" % p.player_id)
    for k, v in state.slash_and_burn_load.items():
        if v < 0:
            _fail("INV-007", "slash_and_burn_load[%s]<0" % k)


def _check_workplace_membership(state, cards):
    """§1.2.1: workplaces は「今ワーカーを置ける職場」だけの疎辞書である。

    キー集合と各カードのゾーンが一致していなければならない。
    """
    for instance_id, w in state.workplaces.items():
        cd = cards.of_instance(instance_id)
        if cd.workplace is None:
            _fail("WP-001", "%s is not a workplace card" % instance_id)
        # 辞書キーと WorkplaceState.instance_id の一致(sol Phase1 #4)。
        if w.instance_id != instance_id:
            _fail("WP-001", "key %s != instance_id %s" % (instance_id, w.instance_id))
        if w.logical_id != cards.logical_id_of(instance_id):
            _fail("WP-001", "%s logical_id mismatch" % instance_id)
        # occupancy_mode がカード定義と一致していること。
        if w.occupancy_mode != cd.occupancy_mode:
            _fail("WP-001", "%s occupancy_mode %s != def %s"
                  % (instance_id, w.occupancy_mode, cd.occupancy_mode))
        if w.usable_by not in ("any_player", "owner_only"):
            _fail("WP-001", "%s bad usable_by %r" % (instance_id, w.usable_by))
        if w.owner is None:
            if w.usable_by != "any_player":
                _fail("WP-001", "%s public but owner_only" % instance_id)
        else:
            if not (0 <= w.owner < state.num_players):
                _fail("WP-001", "%s owner out of range: %r" % (instance_id, w.owner))
            if cd.card_type != "building":
                _fail("WP-001", "%s cannot be owned" % instance_id)
            if instance_id not in state.players[w.owner].owned_buildings:
                _fail("WP-001", "%s not in owner's owned_buildings" % instance_id)
            if w.usable_by != "owner_only":
                _fail("WP-001", "%s owned but not owner_only" % instance_id)

    # 所有建物のうち workplace タグを持つものは必ず workplaces にある。
    for p in state.players:
        for instance_id in p.owned_buildings:
            cd = cards.of_instance(instance_id)
            is_wp = cd.workplace is not None
            present = instance_id in state.workplaces
            if is_wp and not present:
                _fail("WP-001", "%s owned workplace missing from workplaces"
                      % instance_id)
            if (not is_wp) and present:
                _fail("WP-001", "%s facility must not be a workplace" % instance_id)

    # 焼畑の積載は所有中の焼畑実体に対してのみ。
    for instance_id in state.slash_and_burn_load:
        if cards_mod.card_id_of(instance_id) != "base:slash-and-burn":
            _fail("INV-007", "%s cannot carry goods" % instance_id)
        owned = any(instance_id in p.owned_buildings for p in state.players)
        if not owned:
            _fail("INV-007", "%s carries goods but is not owned" % instance_id)


def _check_pending_continuation(state, cards):
    """保証1b(§3.5, core-api.md 付録C): pending が非 None なら legal_actions は非空。

    設計書は本条件を `always` スコープに置く(sol Phase1 #3: ランダム対戦の確率的
    網羅ではなく不変条件として毎回検査する)。game を遅延 import して循環を避ける。
    """
    from .state import Phase
    if state.phase == Phase.GAME_OVER:
        return
    if state.pending is None:
        return
    from . import game
    if not game.legal_actions(state, cards):
        _fail("CONT-001", "pending %r has no legal actions" % (state.pending.kind,))


# --------------------------------------------------------- after_resolution

def _check_after_resolution(state, cards):
    """pending is None の境界でのみ成立する条件。"""
    if state.temporary_public:
        _fail("TEMP-001", "temporary_public must be empty after resolution: %r"
              % (state.temporary_public,))
    if state.pending is not None:
        _fail("TEMP-001", "pending must be None in after_resolution scope")


# --------------------------------------------------------------- after_round

def _check_after_round(state, cards):
    """INV-006: 手札上限はラウンド終了処理の調整後のみ成立する。"""
    for pid in range(state.num_players):
        p = state.players[pid]
        if p.hand_total > hand_limit(state, cards, pid):
            _fail("INV-006", "player %d hand_total=%d > limit=%d"
                  % (pid, p.hand_total, hand_limit(state, cards, pid)))
