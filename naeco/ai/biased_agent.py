"""Phase 12b 偏り方策(設計 D4: 学校優先・雇用回避・鉱山優先)。stdlib のみ。

- 読むのは公開 Observation と合法手だけ。環境の内部状態・他家の手札に触れない。
- 対象職場は日本語名ではなく ``data/cards.json`` の既存 card_id で固定する
  (C3 で実在を確認、``check_card_ids`` が読込時に fail-closed で再検査する)。
- 該当手が無い局面・pending(原子選択)では既存 ``HeuristicAgent`` に委ねる。
  同値は合法手の安定順(``Action.sort_key()`` 昇順)で解消する。
- 乱数は使わない(``seed`` は HeuristicAgent への受け渡しだけ)。同じ観測列なら
  常に同じ行動を返す。
"""
import dataclasses

from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.engine.cards import card_id_of, load_cards
from naeco.engine.state import ActionKind

# 学校優先: 公開職場「学校」(EFF-school-01, on_place__hire_worker)。
SCHOOL_CARD_IDS = ("base:school",)
# 鉱山優先: 公開職場「鉱山」(EFF-mine-01, on_place__draw_cards amount=1)。
MINE_CARD_IDS = ("base:mine",)
# 雇用回避: cards.json で雇用 hook(on_place__hire_worker / _up_to / _ready)を持つ
# 全職場。学校・高等学校(R3)・大学(R5)・専門学校(R7)。
HIRING_CARD_IDS = ("base:school", "base:round-3", "base:round-5",
                   "base:round-7")
HIRING_HOOKS = ("on_place__hire_worker", "on_place__hire_worker_up_to",
                "on_place__hire_worker_ready")

BIASED_KINDS = ("school_first", "avoid_hiring", "mine_first")


def check_card_ids(cards=None):
    """固定 ID が cards.json に実在し、期待する hook を持つことを検査する。

    雇用回避の集合は「雇用 hook を持つ職場の全集合」と完全一致を要求する。
    カードデータが変わって集合がずれたら ValueError(黙って別集合を使わない)。
    """
    cards = cards or load_cards()
    for card_id, hook in (("base:school", "on_place__hire_worker"),
                          ("base:mine", "on_place__draw_cards")):
        card = cards.defs.get(card_id)
        if card is None:
            raise ValueError("biased agent card id missing: %s" % card_id)
        if hook not in [e["hook"] for e in card.effects]:
            raise ValueError("%s no longer has hook %s" % (card_id, hook))
    hiring = sorted(
        card_id for card_id, card in cards.defs.items()
        if any(e["hook"] in HIRING_HOOKS for e in card.effects))
    if tuple(hiring) != tuple(sorted(HIRING_CARD_IDS)):
        raise ValueError("hiring workplace set changed: %r" % (hiring,))
    return True


class BiasedAgent(object):
    """公開観測だけで特定の職場を優先/回避する固定方策。"""

    def __init__(self, kind, seed=0, cards=None):
        if kind not in BIASED_KINDS:
            raise ValueError("unknown biased kind %r (expected one of %s)"
                             % (kind, ", ".join(BIASED_KINDS)))
        self.kind = kind
        self.name = "biased:%s" % kind
        cards = cards or load_cards()
        check_card_ids(cards)
        self._heuristic = HeuristicAgent(seed)

    @staticmethod
    def _places(legal, card_ids):
        return [a for a in legal
                if a.kind == ActionKind.PLACE_WORKER
                and a.target_instance_id is not None
                and card_id_of(a.target_instance_id) in card_ids]

    def act(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        if observation.pending is not None:
            return self._heuristic.act(observation)
        if self.kind in ("school_first", "mine_first"):
            ids = SCHOOL_CARD_IDS if self.kind == "school_first" else MINE_CARD_IDS
            preferred = self._places(legal, ids)
            if preferred:
                return sorted(preferred, key=lambda a: a.sort_key())[0]
            return self._heuristic.act(observation)
        # avoid_hiring: 雇用職場を除いた合法手で heuristic。全部が雇用なら元のまま。
        hiring = set(self._places(legal, HIRING_CARD_IDS))
        rest = tuple(a for a in legal if a not in hiring)
        if not rest or len(rest) == len(legal):
            return self._heuristic.act(observation)
        return self._heuristic.act(
            dataclasses.replace(observation, legal_actions=rest))
