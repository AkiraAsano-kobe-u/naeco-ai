"""cards.json ローダと物理カード実体(instance_id)の採番。

設計書 `docs/core-api.md` §1.1 / §1.2 / §1.2.1 に対応する。
本モジュールは**静的定義**だけを扱い、実行時状態(WorkplaceState 等)を持たない(§5.4)。
"""
import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "cards.json",
)

# ラウンドカード表面 → 公共職場の論理ID(§1.1 の転記ピン)。
# 実データからも導出でき、テストで一致を検証する。
ROUND_FACE_WORKPLACE = {
    "base:round-1": "base:stall",              # 露店
    "base:round-2": "base:market",             # 市場
    "base:round-3": "base:high-school",        # 高等学校
    "base:round-4": "base:supermarket",        # スーパーマーケット
    "base:round-5": "base:university",         # 大学
    "base:round-6": "base:department-store",   # 百貨店
    "base:round-7": "base:vocational-school",  # 専門学校
    "base:round-8": "base:expo",               # 万博
    "base:round-9": None,                      # 表面は使用しない [OPEN-005]
}

# RULES.md 第5章の workplace_id ↔ EFF-ID 対応(転記ピン)。
# ラウンドカードの effects[0].effect_id からこの表を引いて表面写像を導出する。
WORKPLACE_EFFECT_IDS = {
    "base:quarry": "EFF-quarry-01",
    "base:mine": "EFF-mine-01",
    "base:school": "EFF-school-01",
    "base:carpenter": "EFF-carpenter-01",
    "base:stall": "EFF-stall-01",
    "base:market": "EFF-market-01",
    "base:high-school": "EFF-high-school-01",
    "base:supermarket": "EFF-supermarket-01",
    "base:university": "EFF-university-01",
    "base:department-store": "EFF-department-store-01",
    "base:vocational-school": "EFF-vocational-school-01",
    "base:expo": "EFF-expo-01",
}

# 実体を採番するカード種別。消費財・未払い賃金は枚数で保持する(§1.4, OPEN-024)。
INSTANCED_CARD_TYPES = frozenset(["building", "public_workplace", "round"])

# 人数別の大工設置枚数(RULES.md 第2章, OPEN-001)。
CARPENTERS_BY_PLAYERS = {2: 1, 3: 2, 4: 3}

# 初期所持金は開始プレイヤーから $5, $6, $7, $8(RULES.md 第2章)。
INITIAL_MONEY_BASE = 5
INITIAL_WORKERS = 2
INITIAL_HAND = 3
PHYSICAL_WORKERS_PER_COLOR = 7
BASE_WORKER_CAP = 5
BASE_HAND_LIMIT = 5


@dataclasses.dataclass(frozen=True)
class CardDef:
    """cards.json 由来の静的定義。読み取り専用(§5.1)。"""

    card_id: str
    name_ja: str
    card_type: str
    copies: int
    build_cost: Optional[int]
    printed_value: Optional[int]
    tags: Tuple[str, ...]
    effects: Tuple[Dict[str, Any], ...]
    workplace: Optional[Dict[str, Any]]
    round_info: Optional[Dict[str, Any]]

    @property
    def is_workplace(self):
        return self.workplace is not None

    @property
    def occupancy_mode(self):
        if self.workplace is None:
            return None
        return self.workplace["occupancy"]["mode"]

    @property
    def initial_usable_by(self):
        if self.workplace is None:
            return None
        return self.workplace["usable_by"]

    def has_tag(self, tag):
        return tag in self.tags


class CardSet(object):
    """cards.json 一式と、そこから導出した実体・索引を保持する不変オブジェクト。"""

    def __init__(self, raw, source_bytes):
        self.schema_version = raw["schema_version"]
        self.data_version = raw["data_version"]
        # ★ 生バイト列をハッシュしてはならない(sol Phase1 #2)。改行変換(LF↔CRLF)や
        # 整形差だけで state_hash が環境間で変わる。**意味的正準形**(キー昇順・
        # 区切り最小のJSON)をハッシュし、EOL・空白・キー順に不変にする。
        # カード内容の変更は正準形が変わるので引き続き検出できる(data_version上げ忘れ検出)。
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True).encode("utf-8")
        self.sha256 = hashlib.sha256(canonical).hexdigest()

        defs = {}
        order = []
        for rec in raw["cards"]:
            cd = CardDef(
                card_id=rec["card_id"],
                name_ja=rec["name"]["ja"],
                card_type=rec["card_type"],
                copies=rec["copies"],
                build_cost=rec.get("build_cost"),
                printed_value=rec.get("printed_value"),
                tags=tuple(rec.get("tags") or ()),
                effects=tuple(rec.get("effects") or ()),
                workplace=rec.get("workplace"),
                round_info=rec.get("round"),
            )
            defs[cd.card_id] = cd
            order.append(cd.card_id)
        self.defs = defs
        self.order = tuple(order)

        # --- instance_id 採番(§1.2)。seed 非依存・cards.json 配列順。 ---
        instances = []
        by_card = {}
        for card_id in self.order:
            cd = defs[card_id]
            if cd.card_type not in INSTANCED_CARD_TYPES:
                by_card[card_id] = ()
                continue
            ids = tuple("%s#%02d" % (card_id, i) for i in range(cd.copies))
            by_card[card_id] = ids
            instances.extend(ids)
        self.instances = tuple(instances)
        self.instances_by_card = by_card

        self.building_instances = tuple(
            i for i in self.instances if defs[card_id_of(i)].card_type == "building"
        )
        self.round_instances = tuple(
            self.instances_by_card["base:round-%d" % n][0] for n in range(1, 10)
        )

        # --- 効果索引 ---
        self.effect_params = {}
        self.effect_hook = {}
        self.effect_timing = {}
        self.effect_owner_card = {}
        for cd in defs.values():
            for eff in cd.effects:
                self.effect_params[eff["effect_id"]] = dict(eff.get("params") or {})
                self.effect_hook[eff["effect_id"]] = eff["hook"]
                self.effect_timing[eff["effect_id"]] = eff["timing"]
                self.effect_owner_card[eff["effect_id"]] = cd.card_id

        # card_id -> on_place 効果(存在すれば1件)
        self.on_place_effect = {}
        for cd in defs.values():
            for eff in cd.effects:
                if eff["timing"] == "on_place":
                    self.on_place_effect[cd.card_id] = eff["effect_id"]

        # ラウンドカード表面写像をデータから導出する(§1.1)。
        eff_to_workplace = dict(
            (v, k) for (k, v) in WORKPLACE_EFFECT_IDS.items()
        )
        derived = {}
        for n in range(1, 10):
            card_id = "base:round-%d" % n
            effs = defs[card_id].effects
            if not effs:
                derived[card_id] = None
            else:
                derived[card_id] = eff_to_workplace.get(effs[0]["effect_id"])
        self.derived_round_face = derived

    # --- 便宜アクセサ ---
    def get(self, card_id):
        return self.defs[card_id]

    def of_instance(self, instance_id):
        return self.defs[card_id_of(instance_id)]

    def build_cost(self, instance_id):
        return self.defs[card_id_of(instance_id)].build_cost

    def printed_value(self, instance_id):
        return self.defs[card_id_of(instance_id)].printed_value

    def tags_of(self, instance_id):
        return self.defs[card_id_of(instance_id)].tags

    def logical_id_of(self, instance_id):
        """職場としての論理ID。ラウンドカードは表面の職場IDを返す(§1.2.1)。"""
        card_id = card_id_of(instance_id)
        if card_id in ROUND_FACE_WORKPLACE:
            return self.derived_round_face[card_id]
        return card_id

    def wage_rate(self, round_number):
        return self.defs["base:round-%d" % round_number].round_info["wage_rate"]

    def hooks_in_data(self):
        out = {}
        for cd in self.defs.values():
            for eff in cd.effects:
                out.setdefault(eff["hook"], []).append((cd.card_id, eff))
        return out


def card_id_of(instance_id):
    """"base:farm#03" -> "base:farm" """
    return instance_id.split("#", 1)[0]


_CACHE = {}


def load_cards(path=None):
    """cards.json を読み込む(プロセス内キャッシュ付き)。"""
    path = path or DATA_PATH
    key = os.path.abspath(path)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    with open(path, "rb") as fh:
        source = fh.read()
    raw = json.loads(source.decode("utf-8"))
    cs = CardSet(raw, source)
    _CACHE[key] = cs
    return cs


def carpenter_count(num_players):
    return CARPENTERS_BY_PLAYERS[num_players]


def initial_money(seat_offset_from_start):
    """開始プレイヤーから時計回りに $5, $6, $7, $8。"""
    return INITIAL_MONEY_BASE + seat_offset_from_start


def sorted_instances(items):
    """set-like フィールドの正準順(instance_id 昇順)。"""
    return tuple(sorted(items))


def all_hooks(cards):
    # type: (CardSet) -> List[str]
    return sorted(cards.hooks_in_data().keys())
