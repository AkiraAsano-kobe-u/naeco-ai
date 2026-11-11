"""B1 合法手スコアリング型 policy/value ネットのテスト(Phase 3 設計書 §4.7)。

「歯」の狙い: action_mask の False 位置を `0` 埋め(softmax に混ざる)に戻すと
マスクテストが落ちる。形状・パラメータ数・seed 再現・勾配有限性・arch_dict
一意性・スレッド固定も、実装から1つ外すと落ちる形で固定する。
torch が無い環境では `skipUnless` で skip し、import 失敗は隠さない。

次元定数はハードコードせず `naeco.train.encoding` の `obs_dim(2)` /
`action_dim()` から導出する。エンコーダと net の次元が食い違ったら
`EncoderDimensionConsistency` のテストが落ちる(設計書 §4.1 の次元 pin は
`tests/test_encoding.py` が担当)。
"""
import inspect
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

HAS_TORCH = torch is not None

from naeco.train import encoding as enc  # noqa: E402 (stdlib のみ)

if HAS_TORCH:
    from naeco.train import net as net_mod
else:
    net_mod = None

THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# エンコーダの実際の次元から導出する(ハードコードしない)。
# エンコーダ側の変更は tests/test_encoding.py の pin が検出し、
# ここでは net がその次元を受け入れることを常に検証する。
OBS_DIM = enc.obs_dim(2)
ACTION_DIM = enc.action_dim()
DEFAULT_HIDDEN = (256, 256)
DEFAULT_ACTION_HIDDEN = (128,)


def _count_params(net):
    return sum(p.numel() for p in net.parameters())


def _param_shapes(net):
    return [tuple(p.shape) for p in net.parameters()]


def _expected_params(obs_dim, action_dim, hidden, action_hidden):
    """arch_dict の情報だけで決まるパラメータ数を計算する。"""
    total = 0
    in_dim = obs_dim
    for width in hidden:
        total += in_dim * width + width
        in_dim = width
    total += in_dim + 1                      # value_head
    in_dim += action_dim                     # action_mlp の入力
    for width in action_hidden:
        total += in_dim * width + width
        in_dim = width
    total += in_dim + 1                      # action_head
    return total


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class VariableActionShapes(unittest.TestCase):
    """可変長の行動集合で出力形状が契約どおりになること。"""

    def test_variable_action_max_shapes(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(0)
        for action_max in (1, 3, 12):
            batch = 4
            obs = torch.rand(batch, OBS_DIM)
            actions = torch.rand(batch, action_max, ACTION_DIM)
            mask = torch.ones(batch, action_max, dtype=torch.bool)
            logits, values = net(obs, actions, mask)
            self.assertEqual(tuple(logits.shape), (batch, action_max))
            self.assertEqual(tuple(values.shape), (batch,))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class MaskBehaviour(unittest.TestCase):
    """歯 #5: padding の logit は `-inf` 相当でなければならない。"""

    def test_masked_positions_get_zero_softmax_probability(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(0)
        obs = torch.rand(2, OBS_DIM)
        actions = torch.rand(2, 5, ACTION_DIM)
        mask = torch.tensor([
            [True, True, False, False, True],
            [True, False, False, True, False],
        ])
        logits, _ = net(obs, actions, mask)
        probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue((probs[~mask] < 1e-12).all())
        self.assertTrue(torch.allclose(
            probs.sum(dim=-1), torch.ones(2, dtype=torch.float32)))

    def test_all_masked_row_raises_explicitly(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        obs = torch.rand(2, OBS_DIM)
        actions = torch.rand(2, 5, ACTION_DIM)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        with self.assertRaises(ValueError):
            net(obs, actions, mask)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class ParameterCount(unittest.TestCase):
    """出力次元が行動数に依存しないこと。"""

    def test_parameter_count_matches_arch_formula(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        self.assertEqual(
            _count_params(net),
            _expected_params(
                OBS_DIM, ACTION_DIM, DEFAULT_HIDDEN, DEFAULT_ACTION_HIDDEN))

    def test_parameter_count_is_independent_of_action_max(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        before = _count_params(net)
        torch.manual_seed(0)
        for action_max in (2, 9):
            obs = torch.rand(3, OBS_DIM)
            actions = torch.rand(3, action_max, ACTION_DIM)
            mask = torch.ones(3, action_max, dtype=torch.bool)
            net(obs, actions, mask)
        self.assertEqual(_count_params(net), before)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class SeedDeterminism(unittest.TestCase):
    """seed 固定で forward 出力が完全一致すること。"""

    def _inputs(self):
        torch.manual_seed(7)
        obs = torch.rand(3, OBS_DIM)
        actions = torch.rand(3, 6, ACTION_DIM)
        mask = torch.tensor([
            [True, True, True, False, True, True],
            [True, False, True, True, False, True],
            [True, True, False, True, True, False],
        ])
        return obs, actions, mask

    def test_same_seed_reproduces_forward_exactly(self):
        obs, actions, mask = self._inputs()
        torch.manual_seed(123)
        first = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(123)
        second = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        logits1, values1 = first(obs, actions, mask)
        logits2, values2 = second(obs, actions, mask)
        self.assertTrue(torch.equal(logits1, logits2))
        self.assertTrue(torch.equal(values1, values2))

    def test_different_seed_changes_output(self):
        obs, actions, mask = self._inputs()
        torch.manual_seed(123)
        first = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(124)
        second = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        logits1, values1 = first(obs, actions, mask)
        logits2, values2 = second(obs, actions, mask)
        self.assertFalse(torch.equal(logits1, logits2))
        self.assertFalse(torch.equal(values1, values2))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class GradientFiniteness(unittest.TestCase):
    """有効な合法手が1つ以上ある行では勾配が NaN / inf にならないこと。"""

    def test_gradients_are_finite(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(0)
        obs = torch.rand(4, OBS_DIM)
        actions = torch.rand(4, 7, ACTION_DIM)
        mask = torch.ones(4, 7, dtype=torch.bool)
        logits, values = net(obs, actions, mask)
        probs = torch.softmax(logits, dim=-1)
        loss = -(probs * mask.float()).sum() + values.sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class ArchDict(unittest.TestCase):
    """arch_dict が重みの形を一意に決めること。"""

    def test_arch_dict_is_exact_and_json_round_trippable(self):
        arch = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM).arch_dict()
        self.assertEqual(arch, {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "hidden": list(DEFAULT_HIDDEN),
            "action_hidden": list(DEFAULT_ACTION_HIDDEN),
        })
        self.assertEqual(json.loads(json.dumps(arch)), arch)

    def test_same_arch_gives_same_weight_shapes(self):
        arch = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM).arch_dict()
        twin = net_mod.PolicyValueNet(
            arch["obs_dim"], arch["action_dim"],
            hidden=arch["hidden"], action_hidden=arch["action_hidden"])
        self.assertEqual(_param_shapes(twin), _param_shapes(
            net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)))

    def test_different_hidden_gives_different_arch_and_shapes(self):
        base = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        other = net_mod.PolicyValueNet(
            OBS_DIM, ACTION_DIM, hidden=(64, 64), action_hidden=(32,))
        self.assertNotEqual(base.arch_dict(), other.arch_dict())
        self.assertNotEqual(_param_shapes(base), _param_shapes(other))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class EncoderDimensionConsistency(unittest.TestCase):
    """エンコーダと net の次元が食い違ったら落ちる(歯)。

    - net がエンコーダの次元(obs_dim(2) / action_dim())の入力をそのまま
      受理できることを固定する。
    - 次元が食い違う入力(obs 列数 ±1、行動特徴列数 ±1)は例外になり、
      静かに壊れないことを固定する。
    """

    def test_net_accepts_encoder_dimensions(self):
        net = net_mod.PolicyValueNet(enc.obs_dim(2), enc.action_dim())
        obs = torch.zeros(2, enc.obs_dim(2))
        actions = torch.zeros(2, 3, enc.action_dim())
        mask = torch.ones(2, 3, dtype=torch.bool)
        logits, values = net(obs, actions, mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))

    def test_mismatched_obs_dimension_raises(self):
        net = net_mod.PolicyValueNet(enc.obs_dim(2), enc.action_dim())
        actions = torch.zeros(2, 3, enc.action_dim())
        mask = torch.ones(2, 3, dtype=torch.bool)
        with self.assertRaises(Exception):
            net(torch.zeros(2, enc.obs_dim(2) + 1), actions, mask)
        with self.assertRaises(Exception):
            net(torch.zeros(2, enc.obs_dim(2) - 1), actions, mask)

    def test_mismatched_action_dimension_raises(self):
        net = net_mod.PolicyValueNet(enc.obs_dim(2), enc.action_dim())
        obs = torch.zeros(2, enc.obs_dim(2))
        mask = torch.ones(2, 3, dtype=torch.bool)
        with self.assertRaises(Exception):
            net(obs, torch.zeros(2, 3, enc.action_dim() + 1), mask)
        with self.assertRaises(Exception):
            net(obs, torch.zeros(2, 3, enc.action_dim() - 1), mask)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class InferContract(unittest.TestCase):
    """torch 境界の入口: infer / load_state_dict_from_path の契約。

    torch を触るのは net.py だけという境界の下で、search_evaluator 等の
    境界外モジュールが使う入口を固定する。
    """

    def test_infer_accepts_python_lists_and_returns_tensors(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        torch.manual_seed(0)
        obs_list = torch.rand(2, OBS_DIM).tolist()
        action_list = torch.rand(2, 3, ACTION_DIM).tolist()
        mask_list = [[True, True, False], [False, True, True]]
        logits, values = net.infer(obs_list, action_list, mask_list)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertIsInstance(values, torch.Tensor)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))
        # no_grad: 推論出力は勾配追跡しない。
        self.assertFalse(logits.requires_grad)
        self.assertFalse(values.requires_grad)
        # forward に tensor を渡した結果と完全一致する。
        expected_logits, expected_values = net(
            torch.tensor(obs_list, dtype=torch.float32),
            torch.tensor(action_list, dtype=torch.float32),
            torch.tensor(mask_list, dtype=torch.bool))
        self.assertTrue(torch.equal(logits, expected_logits))
        self.assertTrue(torch.equal(values, expected_values))

    def test_infer_exists_on_policy_value_net(self):
        """search_evaluator の dispatch 条件(hasattr infer)を固定する。"""
        self.assertTrue(hasattr(net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM),
                                "infer"))

    def test_load_state_dict_from_path_roundtrips(self):
        net = net_mod.PolicyValueNet(OBS_DIM, ACTION_DIM)
        state_dict = net.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
            torch.save(state_dict, fh.name)
            loaded = net_mod.load_state_dict_from_path(fh.name)
        self.assertEqual(set(loaded), set(state_dict))
        for name in state_dict:
            self.assertTrue(torch.equal(loaded[name], state_dict[name]))

    def test_load_state_dict_from_path_has_no_lazy_import(self):
        """呼出時に import を試みない(モジュール冒頭の torch だけに依存)。

        torch が無い環境では net モジュール自体が import できず、この関数に
        到達できない。関数本体に import 文が無いことをソースで固定する。
        """
        source = inspect.getsource(net_mod.load_state_dict_from_path)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                self.fail(
                    "lazy import inside load_state_dict_from_path: %r"
                    % line)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class ThreadControl(unittest.TestCase):
    """import 時にスレッド数と環境変数を1へ固定すること。"""

    def test_threads_are_pinned_to_one(self):
        self.assertEqual(torch.get_num_threads(), 1)
        for name in THREAD_ENV_NAMES:
            self.assertEqual(os.environ.get(name), "1")
