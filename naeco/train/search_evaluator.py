"""PolicyValueNet を Evaluator プロトコルへ適合させる torch フリーのアダプタ(D-5 / S2)。

コンストラクタで PolicyValueNet インスタンスを受け取る。evaluate() は
`encoding.encode_observation` / `encode_action` で観測と各合法手をエンコードし、
net を batch=1 の list 入力で呼ぶ。torch 境界(設計書 §2)に従い、このモジュール
は torch を一切 import しない。net が `infer` メソッドを持つ(PolicyValueNet)
場合は list 入力のまま `net.infer(obs_batch, action_batch, mask)` を呼び、
tensor への変換と torch.no_grad() は net.py 側が行う。`infer` を持たない
軽量フェイク net は従来どおり `net.forward(obs_batch, action_batch,
action_mask)` を list のまま呼ぶ。

S2: priors は policy ヘッドの logits(batch=1 の1行、長さ = legal_actions 数)を
数値安定 softmax した分布(PolicyAgent._softmax と同じ規律)。[(action, prob),
...] は observation.legal_actions と 1:1。UniformEvaluator(テスト用)は
`naeco/ai/search/evaluator.py` に残す。

D-2: value ヘッドの生線形出力 u(おおよそ [-1,1])は、このアダプタの
`utility_to_win_probability` の1箇所だけで p = clip((u+1)/2, 0, 1) へ写像する。
探索側・木には効用空間の値が入らない。
"""
import math

from naeco.ai.search.evaluator import Evaluator
from naeco.train import encoding_multi


def utility_to_win_probability(u):
    """効用空間 u を [0,1] 勝率へ写像する(D-2)。

    `p = clip((u + 1) / 2, 0, 1)`。u=+1(勝ち) → 1.0、u=-1(負け) → 0.0。
    未収束の尾部が [0,1] をわずかに外れても木の value_sum の範囲を壊さない
    ための clip はこの写像に含める。この写像はアダプタのここにだけ置く。
    """
    return max(0.0, min(1.0, (float(u) + 1.0) / 2.0))


class NetEvaluator(Evaluator):
    """PolicyValueNet を Evaluator プロトコルへ適合させるアダプタ。"""

    def __init__(self, net):
        self._net = net

    def evaluate(self, observation):
        """観測者視点の priors(softmax(policy logits))と [0,1] 勝率を返す。

        policy ヘッドの logits 行(長さ = legal_actions 数)を数値安定 softmax
        した分布を [(action, prob), ...] として legal_actions と 1:1 で返す。
        value の写像(D-2: u -> p=clip((u+1)/2, 0, 1))は変更しない。

        net が `infer` メソッドを持つ(PolicyValueNet)ときは list 入力のまま
        `net.infer(obs_batch, action_batch, mask)` へ委ね、tensor 変換と
        no_grad は net.py 側が行う。`infer` を持たない軽量フェイク net は
        従来どおり `net.forward(obs_batch, action_batch, mask)` を list で呼ぶ。
        """
        legal = observation.legal_actions
        if not legal:
            raise ValueError("no legal actions for this observer")
        # Phase 10 D2: 人数から encoder 契約を選ぶ(n=2 は旧 encoding そのもの)。
        obs_vec, action_vecs = encoding_multi.encode_for_net(
            self._net, observation)
        obs_batch = [list(obs_vec)]
        action_batch = [[list(vec) for vec in action_vecs]]
        mask = [[True] * len(action_vecs)]
        if hasattr(self._net, "infer"):
            logits, values = self._net.infer(obs_batch, action_batch, mask)
        else:
            logits, values = self._net.forward(obs_batch, action_batch, mask)
        u = self._value_scalar(values)
        logit_row = self._logits_row(logits, len(legal))
        probs = self._softmax(logit_row)
        return list(zip(legal, probs)), utility_to_win_probability(u)

    def evaluate_value(self, observation):
        """観測者(`observation.player_id`)自席の [0,1] 価値だけを返す(Phase 10 D3)。

        合法手ゼロ・非手番席の観測でも動く(value ヘッドは観測だけに依存する
        ので `net.infer_values` を使う)。多人数の葉評価で探索側が各席の観測を
        1 つずつ渡し、絶対席順のベクトルへ組み立てる。u→p の写像は
        `utility_to_win_probability`(D-2 と同じ 1 箇所)。
        """
        obs_vec, unused_actions = encoding_multi.encode_for_net(
            self._net, observation, with_actions=False)
        if not hasattr(self._net, "infer_values"):
            raise TypeError(
                "evaluate_value requires a net with infer_values(obs_list)")
        values = self._net.infer_values([list(obs_vec)])
        return utility_to_win_probability(self._value_scalar(values))

    @staticmethod
    def _logits_row(logits, expected):
        """batch=1 の logits 出力から長さ expected のスカラー list を取り出す。

        PolicyAgent._row_logits と同じ規律: 行数が1でなければ ValueError、
        合法手数と一致しなければ ValueError、非有限なら ValueError。
        """
        if hasattr(logits, "tolist"):
            logits = logits.tolist()
        if len(logits) != 1:
            raise ValueError(
                "net returned %d batch rows; expected 1" % len(logits))
        row = [float(x) for x in logits[0]]
        if len(row) != expected:
            raise ValueError(
                "net returned %d logits for %d legal actions"
                % (len(row), expected))
        for x in row:
            if not math.isfinite(x):
                raise ValueError("net returned non-finite logit %r" % (x,))
        return row

    @staticmethod
    def _softmax(logits):
        """数値安定 softmax(PolicyAgent._softmax と同じ規律)。

        max を引いて exp し正規化する。logits は空でないことが前提
        (evaluate は legal_actions 非空を確認済み)。
        """
        peak = max(logits)
        exp = [math.exp(x - peak) for x in logits]
        total = sum(exp)
        return [x / total for x in exp]

    @staticmethod
    def _value_scalar(values):
        """batch=1 の value ヘッド出力(長さ1)からスカラー float を取り出す。"""
        if hasattr(values, "tolist"):
            values = values.tolist()
        row = [float(x) for x in values]
        if len(row) != 1:
            raise ValueError(
                "net returned %d values; expected 1" % len(row))
        return row[0]
