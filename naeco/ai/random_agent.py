"""難易度「弱」以前の対戦相手・テスト用のランダム方策。

観測から合法手を選ぶ(エンジンには state から直接触らない)。方策RNGは環境RNGと
分離する(§2.4): エージェントが自前で持ち、GameState には触れない。
"""
import random


class RandomAgent(object):
    def __init__(self, seed=0):
        self._rng = random.Random(seed)

    def act(self, observation):
        """観測の legal_actions から一様ランダムに1つ選ぶ。"""
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        return self._rng.choice(legal)
