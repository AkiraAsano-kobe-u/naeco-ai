"""合法手スコアリング型 policy/value ネット(Phase 3 設計書 §4.7 / core-api §8.1)。

状態特徴から共有表現 `h` を作り、各合法手の特徴 `f_i` を結合して
`logit_i = MLP([h ; f_i])` を出す。合法手集合のサイズはバッチ内で可変でもよく、
ネットワークの出力次元は行動数に依存しない。価値は共有表現から1値で出す。

torch 境界(設計書 §2)に従い、このモジュールだけが torch を import する。
import 時に4つのスレッド環境変数と `torch.set_num_threads(1)` を固定する(§8.4)。
"""
import os

import torch
import torch.nn as nn

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _fix_thread_env():
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


_fix_thread_env()
torch.set_num_threads(1)


def _mlp(in_dim, widths):
    """widths の各幅へ Linear + ReLU を積む。出力次元は widths[-1]。"""
    if not widths:
        raise ValueError("hidden widths must be non-empty")
    layers = []
    prev = in_dim
    for width in widths:
        layers.append(nn.Linear(prev, width))
        layers.append(nn.ReLU())
        prev = width
    return nn.Sequential(*layers)


class PolicyValueNet(torch.nn.Module):
    """可変長の合法手集合をスコアリングする policy/value ネット。

    `forward` は `(B, obs_dim)` の状態バッチと `(B, A_max, action_dim)` の
    合法手特徴バッチを受け、`(B, A_max)` の logits と `(B,)` の values を返す。
    `action_mask` が False の padding 位置の logit は `-inf` 相当にし、
    softmax に混ざらないようにする(設計書 §4.7 / 歯 #5)。
    """

    def __init__(self, obs_dim, action_dim, hidden=(256, 256),
                 action_hidden=(128,)):
        super().__init__()
        if not hidden:
            raise ValueError("hidden must be non-empty")
        if not action_hidden:
            raise ValueError("action_hidden must be non-empty")
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._hidden = tuple(int(x) for x in hidden)
        self._action_hidden = tuple(int(x) for x in action_hidden)
        self.obs_mlp = _mlp(self._obs_dim, self._hidden)
        self.action_mlp = _mlp(
            self._hidden[-1] + self._action_dim, self._action_hidden)
        self.action_head = nn.Linear(self._action_hidden[-1], 1)
        self.value_head = nn.Linear(self._hidden[-1], 1)

    def infer(self, obs_list, action_list, mask_list):
        """Python の list 入力で推論する(torch 変換 + no_grad を内包)。

        torch 境界外のモジュール(search_evaluator など)から呼ばれる入口。
        `obs_list` は list(list of float)、`action_list` は list(list of
        list of float)、`mask_list` は list(list of bool)。内部で
        `torch.tensor(...)` へ変換して `torch.no_grad()` で
        `self.forward(...)` を呼び、`(logits, values)` の tensor を返す。
        forward の契約は変えない。
        """
        with torch.no_grad():
            return self.forward(
                torch.tensor(obs_list, dtype=torch.float32),
                torch.tensor(action_list, dtype=torch.float32),
                torch.tensor(mask_list, dtype=torch.bool))

    def infer_values(self, obs_list):
        """観測ベクトルの list だけから value を推論する(Phase 7 段1 診断用)。

        value ヘッドは `obs_mlp` の出力にだけ依存し action 非依存なので
        (`docs/phase5-design.md` §2 D-2 の M4 後実測)、合法手を渡さずに
        両席の value を取れる。torch 境界外(scripts)から list 入力で呼ぶ
        入口で、戻り値は Python の float list。学習・対局経路では使わない。
        """
        with torch.no_grad():
            h = self.obs_mlp(torch.tensor(obs_list, dtype=torch.float32))
            return [float(v) for v in self.value_head(h).squeeze(-1).tolist()]

    def forward(self, obs_batch, action_batch, action_mask):
        """`(logits, values)` を返す。形状は §4.7 の契約どおり。"""
        if action_batch.dim() != 3 or action_mask.dim() != 2:
            raise ValueError(
                "action_batch must be (B, A_max, action_dim) and "
                "action_mask must be (B, A_max), got %s and %s"
                % (tuple(action_batch.shape), tuple(action_mask.shape)))
        if action_batch.shape[0] != obs_batch.shape[0]:
            raise ValueError(
                "obs_batch and action_batch must share batch size, "
                "got %d and %d" % (obs_batch.shape[0], action_batch.shape[0]))
        if action_batch.shape[1] != action_mask.shape[1]:
            raise ValueError(
                "action_batch and action_mask must share A_max, "
                "got %d and %d" % (action_batch.shape[1], action_mask.shape[1]))
        mask = torch.as_tensor(action_mask, dtype=torch.bool)
        if not mask.any(dim=-1).all():
            raise ValueError(
                "every batch row must have at least one legal action "
                "(all-False action_mask rows are not supported)")
        h = self.obs_mlp(obs_batch)
        h_expanded = h.unsqueeze(1).expand(-1, action_batch.shape[1], -1)
        fused = torch.cat((h_expanded, action_batch), dim=-1)
        logits = self.action_head(self.action_mlp(fused)).squeeze(-1)
        logits = torch.where(
            mask, logits, torch.full_like(logits, float("-inf")))
        values = self.value_head(h).squeeze(-1)
        return logits, values

    def arch_dict(self):
        """重みの形を一意に決める情報。checkpoint の model_arch へ入れる。"""
        return {
            "obs_dim": self._obs_dim,
            "action_dim": self._action_dim,
            "hidden": list(self._hidden),
            "action_hidden": list(self._action_hidden),
        }


def load_state_dict_from_path(weights_path):
    """checkpoint 重みファイルを CPU へロードして state_dict を返す。

    `torch.load(weights_path, map_location="cpu")` の薄い包みで、torch 境界外
    の呼び出し側(scripts など)が torch を import せずに重みを読むための入口。
    """
    return torch.load(weights_path, map_location="cpu")


# ---------------------------------------------------------------------------
# Phase 10 D2: 人数別モデルの生成・保存・読込
# ---------------------------------------------------------------------------

_META_SUFFIX = ".meta.json"
_WEIGHTS_SUFFIX = ".weights.pt"


def build_for_players(num_players, hidden=(256, 256), action_hidden=(128,),
                      seed=None):
    """人数の encoder 契約の入力次元で未学習 PolicyValueNet を作る。

    2 席 569 / 3 席 688 / 4 席 807、行動 109。seed を与えると
    `torch.manual_seed(seed)` してから初期化する(初期 checkpoint の再現用)。
    """
    from naeco.train import encoding_multi
    contract = encoding_multi.contract_for(num_players)
    if seed is not None:
        torch.manual_seed(int(seed))
    return PolicyValueNet(contract.obs_dim, contract.action_dim,
                          hidden=hidden, action_hidden=action_hidden)


def _weights_path_for(meta_path):
    if not meta_path.endswith(_META_SUFFIX):
        raise ValueError("meta path must end with %s: %r"
                         % (_META_SUFFIX, meta_path))
    return meta_path[:-len(_META_SUFFIX)] + _WEIGHTS_SUFFIX


def save_checkpoint(net, num_players, meta_path, extra=None):
    """重み(.weights.pt)とメタ(.meta.json)を人数の契約で保存する。

    メタは `checkpoint.current_meta(num_players, net.arch_dict(), extra)` で
    作るので、encoder/reward の版・hash は人数から内部で決まる(呼び出し側は
    指定できない)。net の入力次元が人数の契約と食い違えば ValueError。
    重みを先に書き、メタは原子的に置換する(メタがあれば重みもある)。
    戻り値は (meta_path, weights_path)。
    """
    from naeco.train import checkpoint as checkpoint_mod
    from naeco.train import encoding_multi
    contract = encoding_multi.contract_for(num_players)
    arch = net.arch_dict()
    if (arch["obs_dim"], arch["action_dim"]) != (contract.obs_dim,
                                                 contract.action_dim):
        raise ValueError(
            "net dims (%d, %d) do not match the %d-seat encoder (%d, %d)"
            % (arch["obs_dim"], arch["action_dim"], num_players,
               contract.obs_dim, contract.action_dim))
    record = {"torch_version": torch.__version__}
    if extra:
        record.update(extra)
    meta = checkpoint_mod.current_meta(num_players, arch, extra=record)
    weights_path = _weights_path_for(meta_path)
    parent = os.path.dirname(os.path.abspath(meta_path))
    os.makedirs(parent, exist_ok=True)
    torch.save(net.state_dict(), weights_path)
    checkpoint_mod.save(meta_path, meta)
    return meta_path, weights_path


def load_checkpoint(meta_path, num_players):
    """人数を指定して checkpoint を検証付きで読み、(net, meta) を返す。

    `checkpoint.load(meta, expected_meta(num_players, arch))` が A 表・人数別
    encoder hash・人数と入力次元の一致を検査した後でだけ重みを読む。重みの
    キー・形状が model_arch と食い違えば IncompatibleCheckpoint。
    """
    import json
    from naeco.train import checkpoint as checkpoint_mod
    with open(meta_path, "r", encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except ValueError as exc:
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: meta JSON is corrupt: %s" % exc)
    arch = raw.get("model_arch") if isinstance(raw, dict) else None
    if not isinstance(arch, dict) or set(arch) != {
            "obs_dim", "action_dim", "hidden", "action_hidden"}:
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: model_arch must have exactly "
            "obs_dim/action_dim/hidden/action_hidden, got %r" % (arch,))
    # 保存 arch を expected に渡すと型照合が自己比較になるため、先に検査する。
    # _strict_equal は比較器であり schema 検査ではない。int() で正規化しない。
    for key in ("obs_dim", "action_dim"):
        if type(arch[key]) is not int:
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: model_arch.%s must be an int "
                "(got %s)" % (key, type(arch[key]).__name__))
    for key in ("hidden", "action_hidden"):
        value = arch[key]
        if not isinstance(value, list) or \
                any(type(item) is not int for item in value):
            raise checkpoint_mod.IncompatibleCheckpoint(
                "checkpoint incompatible: model_arch.%s must be a list "
                "of ints (got %r)" % (key, value))
    meta = checkpoint_mod.load(
        meta_path, expected=checkpoint_mod.expected_meta(num_players, arch))
    net = PolicyValueNet(arch["obs_dim"], arch["action_dim"],
                         hidden=tuple(arch["hidden"]),
                         action_hidden=tuple(arch["action_hidden"]))
    weights_path = _weights_path_for(meta_path)
    if not os.path.isfile(weights_path):
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint %s has meta but weights %s is missing"
            % (meta_path, weights_path))
    state_dict = load_state_dict_from_path(weights_path)
    expected_shapes = dict((k, tuple(v.shape))
                           for k, v in net.state_dict().items())
    actual_shapes = dict((k, tuple(v.shape)) for k, v in state_dict.items())
    if expected_shapes != actual_shapes:
        raise checkpoint_mod.IncompatibleCheckpoint(
            "checkpoint incompatible: %s weights do not match model_arch"
            % weights_path)
    net.load_state_dict(state_dict)
    net.eval()
    return net, meta
