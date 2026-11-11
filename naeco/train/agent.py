"""推論用 PolicyAgent(Phase 3 設計書 §4.6)。

観測をエンコードし、``net.forward(obs_batch, action_batch, action_mask)`` の
logits から greedy / sample で行動を選ぶ。読むのは ``observation`` だけであり、
``naeco.engine.game`` を import しない(非公開情報に触れる経路を持たない)。

torch を触るのは ``naeco/train/net.py`` と ``naeco/train/ppo.py`` だけであり、
このモジュールは torch を一切 import しない。実 net は
``PolicyValueNet.infer()``(Python list 入力で tensor 変換と ``no_grad`` を
内包)経由で呼ぶ。``infer`` を持たない軽量フェイク net は従来どおり
``forward`` を Python の list のまま呼ぶ。sample は自前の
``random.Random(seed)`` を使うので、同じ seed と観測列なら完全に同じ
行動列を返す。
"""
import math
import random

from naeco.train import encoding_multi


class PolicyAgent(object):
    """公開 Observation だけを使うニューラル方策。

    ``mode="greedy"`` は logits 最大の合法手を返す。同点は
    ``Action.sort_key()`` の昇順で解消する(``HeuristicAgent`` と同じ規律)。
    ``mode="sample"`` は合法手集合上の softmax から自前 RNG で1つ抽選する。
    """

    def __init__(self, net, mode="greedy", seed=0):
        if mode not in ("greedy", "sample"):
            raise ValueError(
                "mode must be 'greedy' or 'sample', got %r" % (mode,))
        self._net = net
        self._mode = mode
        self._rng = random.Random(seed)

    def act(self, observation):
        """観測から合法手を1つ選んで返す。"""
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        # Phase 10 D2: 観測の人数で encoder 契約を選ぶ(n=2 は旧 encoding の関数
        # そのもの)。net の入力次元と合わない人数の観測は ValueError で止める。
        obs_vec, action_vecs = encoding_multi.encode_for_net(
            self._net, observation)
        logits, _values = self._forward(obs_vec, action_vecs)
        if len(logits) != len(legal):
            raise ValueError(
                "net returned %d logits for %d legal actions"
                % (len(logits), len(legal)))
        for logit in logits:
            if not math.isfinite(logit):
                raise ValueError("net returned non-finite logit %r" % (logit,))
        if self._mode == "greedy":
            best = min(
                range(len(legal)),
                key=lambda i: (-logits[i], legal[i].sort_key()))
            return legal[best]
        probs = self._softmax(logits)
        chosen = self._rng.choices(
            list(range(len(legal))), weights=probs, k=1)[0]
        return legal[chosen]

    def _forward(self, obs_vec, action_vecs):
        """obs/action の特徴 list を batch=1 の1行へ詰めて net を呼ぶ。

        PolicyValueNet の契約(設計書 §4.7)は ``forward(obs_batch,
        action_batch, action_mask)`` で、``obs_batch=(B, obs_dim)`` /
        ``action_batch=(B, A_max, action_dim)`` / ``action_mask=(B, A_max)``
        をとり ``logits=(B, A_max)`` を返す。推論では B=1 の1行にこの局面の
        合法手を全部並べ(mask は全 True)、``logits[0]``(長さ L)を取り出す。
        呼び出しは ``infer``(list 入力で tensor 変換と ``no_grad`` を内包)が
        あればそちら、無ければ ``forward`` を list のまま使う。
        """
        obs_batch = [list(obs_vec)]
        action_batch = [[list(vec) for vec in action_vecs]]
        mask = [[True] * len(action_vecs)]
        if hasattr(self._net, "infer"):
            # 実 PolicyValueNet: list 入力で tensor 変換 + no_grad を内包。
            logits, values = self._net.infer(obs_batch, action_batch, mask)
        else:
            # フェイク net など forward だけの互換実装は list のまま呼ぶ。
            logits, values = self._net.forward(obs_batch, action_batch, mask)
        return self._row_logits(logits, values)

    def _row_logits(self, logits, values):
        """batch=1 の ``(logits, values)`` から長さ L のスコア list を取り出す。

        net の返り値は (B, A_max) のレイアウトで、推論の B は常に1なので
        先頭行だけを使う。行数が1でなければ契約違反として ValueError にする。
        """
        if len(logits) != 1:
            raise ValueError(
                "net returned %d batch rows; expected 1" % len(logits))
        return self._as_float_list(logits[0]), self._as_float_list(values)

    @staticmethod
    def _as_float_list(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [float(x) for x in value]

    @staticmethod
    def _softmax(logits):
        peak = max(logits)
        exp = [math.exp(x - peak) for x in logits]
        total = sum(exp)
        return [x / total for x in exp]
