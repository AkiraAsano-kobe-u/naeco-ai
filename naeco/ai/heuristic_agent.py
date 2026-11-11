"""公開 Observation だけを使う、難易度「弱」のルールベース方策。"""
import random

from naeco.engine.cards import load_cards
from naeco.engine.state import ActionKind


class HeuristicAgent(object):
    """合法手を採点する決定論的なヒューリスティック CPU。

    カード定義は公開の静的データであり、局面について読むのは ``observation``
    だけである。環境の内部状態には一切触れない。乱数は将来の微小な揺らぎ用に自前で保持するが、現在の採点は
    同点も sort_key で解消するので、同じ観測列なら常に同じ行動を返す。
    """

    def __init__(self, seed=0):
        self._rng = random.Random(seed)
        self._cards = load_cards()

    def act(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        ranked = sorted(legal, key=lambda a: (-self._score(a, observation),
                                               a.sort_key()))
        return ranked[0]

    def _card_score(self, instance_id, observation):
        """建設・保持・廃棄の比較用の将来価値(公開カード定義だけを使用)。"""
        card = self._cards.of_instance(instance_id)
        value = float(card.printed_value or 0)
        cost = float(card.build_cost or 0)
        value -= 0.55 * cost
        owned_count = len(observation.my_owned)
        workers = sum(observation.my_workers)
        locked_count = sum(
            1 for item in observation.my_owned
            if "locked" in self._cards.tags_of(item)
        )
        for effect in card.effects:
            hook = effect["hook"]
            params = effect.get("params") or {}
            if hook == "on_game_end__score_per_building":
                value += params.get("points_per", 0) * (owned_count + 1)
            elif hook == "on_game_end__score_per_consumer_good":
                value += params.get("points_per", 0) * observation.my_hand_goods
            elif hook == "on_game_end__score_per_worker":
                value += params.get("points_per", 0) * workers
            elif hook == "on_game_end__score_per_locked_building":
                value += params.get("points_per", 0) * (locked_count + 1)
            elif hook == "on_game_end__score_per_factory":
                factories = sum(
                    1 for item in observation.my_owned
                    if "factory" in self._cards.tags_of(item)
                )
                value += params.get("points_per", 0) * factories
            elif hook == "on_game_end__forgive_unpaid_wages":
                value += min(params.get("amount", 0),
                             observation.my_unpaid_wages) * 3
            elif hook == "passive__hand_limit_bonus":
                value += 3.0
            elif hook == "passive__worker_cap_bonus":
                value += 2.0
        # 早い建設ほど再利用できる職場の価値が高い。
        if card.workplace is not None:
            value += max(0, 10 - observation.round_number) * 0.8
        return value

    def _best_hand_value(self, observation):
        if not observation.my_hand_buildings:
            return 0.0
        return max(self._card_score(item, observation)
                   for item in observation.my_hand_buildings)

    def _place_score(self, action, observation):
        workplace = self._cards.of_instance(action.target_instance_id)
        effects = [e for e in workplace.effects if e["timing"] == "on_place"]
        if not effects:
            return 0.0
        effect = effects[0]
        hook = effect["hook"]
        params = effect.get("params") or {}
        late = observation.round_number >= 7
        best_hand = self._best_hand_value(observation)
        # 効果の量と残りラウンドを意識した、意図的に粗い価値表。
        if hook in ("on_place__build", "on_place__build_then_draw",
                    "on_place__build_farm_free", "on_place__build_two_same_cost"):
            score = 42.0 + best_hand
            if hook == "on_place__build_then_draw":
                score += 8.0
            if hook == "on_place__build_two_same_cost":
                score += 10.0
            return score
        if hook == "on_place__gain_from_household":
            return 32.0 + params.get("amount", 0)
        if hook == "on_place__discard_gain_household":
            return 28.0 + params.get("amount", 0) - params.get("discard", 0) * 2
        if hook == "on_place__gain_consumer_goods":
            return 18.0 + params.get("amount", 0) * (1.0 if late else 1.8)
        if hook == "on_place__slash_and_burn":
            return 16.0
        if hook == "on_place__design_office":
            return 24.0
        if hook == "on_place__discard_then_draw":
            return 20.0 + params.get("draw", 0) * 3 - params.get("discard", 0)
        if hook == "on_place__draw_cards":
            return 17.0 + params.get("amount", 1) * 3
        if hook == "on_place__quarry":
            # 採石場は鉱山と同じ1枚ドローに加えて開始プレイヤーマーカーを得る
            # (EFF-quarry-01 / TX-023)。両者を同点にすると sort_key の昇順で
            # `base:mine` が必ず勝ち、採石場が一度も選ばれなくなる。
            # マーカーは次ラウンドから効くので R9 では価値がなく、すでに自分が
            # 開始プレイヤーなら「相手に渡さない」ぶんだけの価値になる。
            if observation.round_number >= 9:
                marker = 0.0
            elif observation.start_player_offset == 0:
                marker = 2.0
            else:
                marker = 4.0
            return 17.0 + params.get("draw", 1) * 3 + marker
        if hook == "on_place__orchard":
            return 18.0
        if hook in ("on_place__hire_worker", "on_place__hire_worker_up_to",
                    "on_place__hire_worker_ready"):
            # 資金の薄い序盤に雇い過ぎて賃金不能になることを避ける。
            employed = sum(observation.my_workers)
            safety = 0.0 if observation.my_money >= 2 * (employed + 1) else -16.0
            return (15.0 if not late else 5.0) + safety
        if hook == "on_place__chemical_plant":
            return 13.0
        return 8.0

    def _score(self, action, observation):
        if action.kind == ActionKind.PLACE_WORKER:
            return self._place_score(action, observation)
        if action.kind == ActionKind.SELECT_BUILD:
            return 100.0 + self._card_score(action.instance_id, observation)
        if action.kind == ActionKind.SELECT_KEEP:
            return 80.0 + self._card_score(action.instance_id, observation)
        if action.kind == ActionKind.SELL_BUILDING:
            # 強制売却では、将来価値が最も低い資産から現金化する。
            return -self._card_score(action.instance_id, observation)
        if action.kind == ActionKind.SELECT_DISCARD:
            if action.instance_id is None:
                # 消費財は農協以外では建設支払いにも使えるので最小限に残す。
                return -2.0 * action.goods_count
            return -self._card_score(action.instance_id, observation)
        return 0.0
