"""評価器プロトコル(Phase 5 設計書 D-5)。stdlib のみ。torch を import しない。

探索側は評価器へ `observe(determinized_state, actor, cards)` の結果
(Observation)だけを渡す。評価器は state に触れられない(D-6 の構造的な
情報境界)。torch アダプタは `naeco/train/search_evaluator.py` に置き、
本モジュールは torch なしで import できることを維持する。
"""


class Evaluator(object):
    """観測から priors と [0,1] 勝率を返す評価器プロトコル。

    `evaluate` は観測者視点(`observation.player_id`)の [0,1] スカラーを返す。
    2人ゼロ和のプレイヤーベクトル [p, 1-p](絶対席順)の組み立ては探索側
    (`naeco/ai/search/ismcts.py` の `evaluate_with_evaluator`)に置く(D-2)。
    効用空間 u∈[-1,1] から勝率 p への写像 p=(u+1)/2 はアダプタ側(本プロトコルの
    実装)にだけ置き、探索側・木には効用空間の値が入らない。
    """

    def evaluate(self, observation):
        """観測者視点の評価を返す。observation は公開情報のみ(構造的な情報境界)。

        戻り値:
          priors: list[(action, prob)]。observation.legal_actions と1対1で、
                  prob は合計 1.0(一様可)。
          value:  observation.player_id 視点の [0,1] スカラー(勝率)。
        """
        raise NotImplementedError

    def evaluate_value(self, observation):
        """観測者自席の [0,1] 価値だけを返す(Phase 10 D3、多人数の葉評価用)。

        合法手ゼロ・非手番席の観測でも動くこと。3/4 席の葉では探索側が
        決定化 state の各席 pid について `observe(state, pid)` を1つずつ渡し、
        戻り値を絶対席順の n ベクトルへ並べる(`ismcts.evaluate_with_evaluator`)。
        2 席の経路はこのメソッドを使わない([p, 1-p] のまま)。
        """
        raise NotImplementedError


class ActionPolicy(object):
    """Phase 12b 手法 C: observation -> 合法手確率の stdlib プロトコル(D2)。

    `action_probabilities(observation)` は observation.legal_actions と同じ集合を
    覆う [(action, prob), ...] を返す。prob は有限・非負・合計 1。探索側は
    決定化 state の手番席の観測だけを渡し、真の hidden state・実相手種別は渡さない。
    """

    def action_probabilities(self, observation):
        raise NotImplementedError


class EvaluatorPolicy(ActionPolicy):
    """Evaluator の priors(NetEvaluator なら softmax(policy logits))を方策にする。"""

    def __init__(self, evaluator):
        self._evaluator = evaluator

    def action_probabilities(self, observation):
        priors, unused_value = self._evaluator.evaluate(observation)
        return list(priors)


class DeterministicPolicy(ActionPolicy):
    """`act(observation)` を持つ固定方策(heuristic / biased)を確率 1 の方策にする。"""

    def __init__(self, agent):
        self._agent = agent

    def action_probabilities(self, observation):
        chosen = self._agent.act(observation)
        return [(action, 1.0 if action == chosen else 0.0)
                for action in observation.legal_actions]


def checked_probabilities(policy, observation, tolerance=1e-6):
    """方策の出力を検査し、legal_actions と同順の確率 list を返す。

    合法手集合の不一致・重複・負/非有限・合計が 1 から外れた分布は ValueError
    (壊れた probability を黙って正規化しない)。
    """
    import math
    legal = observation.legal_actions
    if not legal:
        raise ValueError("no legal actions for this observer")
    pairs = policy.action_probabilities(observation)
    probs = {}
    for action, prob in pairs:
        if action in probs:
            raise ValueError("policy returned a duplicate action %r" % (action,))
        prob = float(prob)
        if not math.isfinite(prob) or prob < 0.0:
            raise ValueError("policy returned an invalid probability %r" % (prob,))
        probs[action] = prob
    if set(probs) != set(legal):
        raise ValueError("policy actions do not match the legal actions")
    total = sum(probs.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError("policy probabilities sum to %r, not 1" % (total,))
    return [probs[action] for action in legal]


class PolicyMixture(object):
    """世代凍結した相手方策の混合(手法 C)。成分は (id, ActionPolicy, weight)。

    探索 1 回(1 決定)ごとに `sample_component(seed)` で成分を 1 つ選び、その
    探索中は固定する。id/重みは manifest に記録できる文字列/数値だけ。
    """

    def __init__(self, components):
        components = list(components)
        if not components:
            raise ValueError("policy mixture needs at least one component")
        ids = [c[0] for c in components]
        if len(set(ids)) != len(ids):
            raise ValueError("policy mixture ids must be unique")
        for cid, policy, weight in components:
            if not isinstance(cid, str) or not cid:
                raise ValueError("component id must be a non-empty string")
            if not isinstance(policy, ActionPolicy):
                raise ValueError("component %s is not an ActionPolicy" % cid)
            if not (float(weight) > 0.0):
                raise ValueError("component %s weight must be positive" % cid)
        self.components = tuple((c[0], c[1], float(c[2])) for c in components)

    def describe(self):
        return [{"id": cid, "weight": weight}
                for cid, unused_policy, weight in self.components]

    def sample_component(self, seed):
        import random
        rng = random.Random(seed)
        index = rng.choices(range(len(self.components)),
                            weights=[c[2] for c in self.components], k=1)[0]
        return self.components[index]


class UniformEvaluator(Evaluator):
    """テスト用の具象クラス: 合法手上の一様 priors と固定 value を返す。"""

    def __init__(self, value=0.5):
        self.value = float(value)

    def evaluate(self, observation):
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        prob = 1.0 / len(legal)
        return [(action, prob) for action in legal], self.value

    def evaluate_value(self, observation):
        return self.value
