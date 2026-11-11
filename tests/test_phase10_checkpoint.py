"""Phase 10 B-C4: 人数別 encoder 契約と checkpoint の保存/読込/拒否(D2)。

- 既存 `naeco/train/encoding.py` は byte 単位で不変(gen1300 の hash を守る)。
- 3/4 席は `encoding_multi`(obs 2.0.0 / action 1.0.0 / reward 2.0.0)。
- 契約は人数から内部で選び、人数・版・次元・型の偽装を fail-closed で拒否する。
- gen1300 の SHA・obs/actions・logits/value・固定 seed 探索が golden と一致する。

golden は B-C4 着手前の HEAD(c245069)で生成し、変更後も Mac(arm64, torch
2.8.0)で bit 一致を確認した値。浮動小数は他プラットフォームの末尾ビット差だけを
許す(1e-6)。行動・hash・探索の計数は完全一致を要求する。
"""
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.engine import cards as cards_mod
from naeco.engine import game
from naeco.engine.observe import observe
from naeco.engine.state import Phase
from naeco.train import checkpoint
from naeco.train import encoding
from naeco.train import encoding_multi
from naeco.train import reward

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - CI without torch
    torch = None
HAS_TORCH = torch is not None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAMPION_DIR = os.path.join(ROOT, "docs", "eval-runs", "phase7-champion")
CARDS = cards_mod.load_cards()

# --- golden(B-C4 着手前の HEAD で生成) ---
ENCODING_PY_SHA256 = (
    "778a551cd9e337979d2c7755b6b5e10d5a303e65dd2447c8930a04d19d92151c")
LEGACY_OBS_SHA256 = (
    "ecc202bbb6e470764061a5138b6b721f13a2c86a14f1792935ea413917f782a0")
LEGACY_ACTION_SHA256 = (
    "7c389a0e1348d9e1611ebb2dad450e5681773d3075bec7a0c22b68af58b2660c")
GEN1300_WEIGHTS_SHA256 = (
    "a0ac138573e7cf68134ecfd86fd08e6e4c2e67823d6b62f5d577b8346ca2566b")
GEN1300_META_SHA256 = (
    "79d1e6b2673d5639930c0794c887859ba02268ca59e3e4f3759edb8aed2fdeaf")

# seed=7 の 2 人戦を gen1300 greedy で進めた先頭 10 決定。
GREEDY_ROWS = [
    {"obs_hash": "0b56c51918f5b53a459ecedc785c49b03aadec5869cae179b9044e905692a05e", "actions_hash": "f7d60f767854afb5bb16545b7b50d58627b846bd5ab071bca6736f4d5c3ed543", "chosen": ["place_worker", "base:school#00", "", 0], "logits": [3.2631590366363525, 2.3449347019195557, 4.259374618530273, 11.772302627563477], "value": 0.43587633967399597, "p": 0.717938169836998},  # noqa: E501
    {"obs_hash": "155a322b5aa65aa6d954be41f2f4b6aec1edb64c668aefa416199643bf1f7993", "actions_hash": "bda6533b647d709b3257c84ad307a9b52983f0713167ce44b9ca8a0ebbc1520f", "chosen": ["place_worker", "base:mine#00", "", 0], "logits": [-28.008502960205078, -21.49104881286621, -21.60127830505371], "value": 0.15063199400901794, "p": 0.575315997004509},  # noqa: E501
    {"obs_hash": "9b61d39bc9ea3cdf56c675001c38fdb9f529827cfcbbae63873b5d5dfe7214d9", "actions_hash": "bda6533b647d709b3257c84ad307a9b52983f0713167ce44b9ca8a0ebbc1520f", "chosen": ["place_worker", "base:quarry#00", "", 0], "logits": [-17.473567962646484, -17.827945709228516, -16.258527755737305], "value": 0.4926234781742096, "p": 0.7463117390871048},  # noqa: E501
    {"obs_hash": "6fe4179479e7ef6b7289aa49d52365a475c01ce622a0b38178a8281d2fbcfe57", "actions_hash": "df80f34ef51c3517be0a0f4a7dc31e7d289d24139c434df1316fa6dc67a6232d", "chosen": ["place_worker", "base:mine#00", "", 0], "logits": [-14.267690658569336, -10.750030517578125], "value": 0.12805694341659546, "p": 0.5640284717082977},  # noqa: E501
    {"obs_hash": "fabe8858f06521e89ab87a76c05c9cc8be1be93ae025c6d51c4c8e601c8bcc11", "actions_hash": "818d87facc56bd14578b752647557de9458746c515ec6c3c2ad5868d2d7a57be", "chosen": ["place_worker", "base:quarry#00", "", 0], "logits": [-16.532827377319336, -15.659374237060547, -12.14079475402832, -14.001773834228516, -16.17828941345215], "value": 0.5352413654327393, "p": 0.7676206827163696},  # noqa: E501
    {"obs_hash": "75d9b1dda6d8dd830a160a380a03366329aa72b46b155f1621dadca737802bec", "actions_hash": "7ee80f88079a7e53eff10c4aacdfbe59ca2e0f27e3c4991b6fcd625e9ccc18b3", "chosen": ["place_worker", "base:carpenter#00", "", 0], "logits": [6.2225494384765625, 1.3908312320709229, 4.731740951538086, 4.193239212036133], "value": 0.06888505816459656, "p": 0.5344425290822983},  # noqa: E501
    {"obs_hash": "7a7f7cb271d10dd57ed29b5d7e2ecfb552dc282eb4a1a912512bbb54a7744535", "actions_hash": "4fba93fc8c39e47b9208a77d0989ddbc38be6dc66181d10e163adc3c237208f0", "chosen": ["select_build", "", "base:steel-mill#00", 0], "logits": [-10.889719009399414, -16.990880966186523, -15.41773796081543, -3.6568944454193115, -17.99424171447754], "value": 0.02477404475212097, "p": 0.5123870223760605},  # noqa: E501
    {"obs_hash": "f2506a8c85ca5b7258acc752e3625b333d27a6c49c1c7a050429c69464268d32", "actions_hash": "b13cde359e9714701d630d2dc0d879f5be21ab01f6de0dfea940a657cd59b470", "chosen": ["select_discard", "", "", 0], "logits": [-3.993722677230835], "value": 0.005275160074234009, "p": 0.502637580037117},  # noqa: E501
    {"obs_hash": "ae1689dc9a76e5704c2a6b98058636ecd473d9b8703a6db7a9837e0c637785bd", "actions_hash": "5bdf09257fdd78b0b271bb3cf343093f54dc4b21b81c0450ccacfea7faf0b698", "chosen": ["select_discard", "", "base:factory#03", 0], "logits": [4.168351173400879], "value": 0.019890815019607544, "p": 0.5099454075098038},  # noqa: E501
    {"obs_hash": "1d6fcdbefc4a5335889b04f1e1b7cd630d752f90a69e46e4fc1ad17e5d291fa9", "actions_hash": "f25900d1bdafa65ad6a298ed863423da5d88e6b12e5a7abecb11a69fdcf16f20", "chosen": ["select_discard", "", "base:mansion#00", 0], "logits": [5.590417861938477], "value": 0.05716446042060852, "p": 0.5285822302103043},  # noqa: E501
]

# seed=11 の 2 人戦。席0 = IsmctsAgent(root_seed=1234, T=300, NetEvaluator(gen1300),
# policy_agent=greedy)、席1 = greedy。席0 の先頭 12 決定。
SEARCH_GOLDEN = {"chosen": [["place_worker", "base:quarry#00", "", 0], ["place_worker", "base:mine#00", "", 0], ["place_worker", "base:round-1#00", "", 0], ["select_discard", "", "", 0], ["select_discard", "", "base:factory#04", 0], ["place_worker", "base:carpenter#00", "", 0], ["select_build", "", "base:orchard#02", 0], ["select_discard", "", "", 0], ["select_discard", "", "base:factory#05", 0], ["select_discard", "", "base:warehouse#00", 0], ["place_worker", "base:orchard#02", "", 0], ["place_worker", "base:quarry#00", "", 0]], "transitions": 1800, "evaluations": 109, "nodes_created": 56, "determinizations": 53, "delegated_calls": 6, "decisions": 6}  # noqa: E501

# seed=5 の 2 人戦。席1 = IsmctsAgent(99, T=200, evaluator=None)、席0 = greedy。
STATIC_SEARCH_GOLDEN = {"chosen": [["place_worker", "base:carpenter#00", "", 0], ["select_build", "", "base:factory#04", 0], ["select_discard", "", "", 0], ["select_discard", "", "base:automobile-factory#02", 0], ["select_discard", "", "base:design-office#00", 0], ["place_worker", "base:mine#00", "", 0], ["place_worker", "base:mine#00", "", 0], ["place_worker", "base:factory#04", "", 0]], "evaluations": 17, "nodes_created": 20, "transitions": 600}  # noqa: E501

FLOAT_TOL = 1e-6


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _vec_hash(rows):
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def _rewrite(path, meta):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")


def _arch(n):
    return {"obs_dim": encoding_multi.contract_for(n).obs_dim,
            "action_dim": encoding_multi.action_dim(),
            "hidden": [8], "action_hidden": [4]}


class LegacyEncoderIsUntouched(unittest.TestCase):
    """既存 encoding.py の byte と hash が B-C4 前と同一(D2)。"""

    def test_encoding_py_bytes_and_hashes(self):
        self.assertEqual(_sha(os.path.join(ROOT, "naeco/train/encoding.py")),
                         ENCODING_PY_SHA256)
        self.assertEqual(encoding.encoder_sha256(),
                         (LEGACY_OBS_SHA256, LEGACY_ACTION_SHA256))
        self.assertEqual(encoding.OBS_ENCODER_VERSION, "1.0.0")
        self.assertEqual(encoding.ACTION_ENCODER_VERSION, "1.0.0")

    def test_two_seat_contract_is_the_legacy_module(self):
        contract = encoding_multi.contract_for(2)
        self.assertIs(contract.encode_observation, encoding.encode_observation)
        self.assertIs(contract.encode_action, encoding.encode_action)
        self.assertIs(contract.encoder_sha256, encoding.encoder_sha256)
        self.assertEqual(contract.obs_dim, 569)
        self.assertEqual(contract.obs_encoder_version, "1.0.0")

    def test_two_seat_meta_is_unchanged(self):
        meta = checkpoint.current_meta(2, {"obs_dim": 569})
        self.assertEqual(meta["obs_encoder_sha256"], LEGACY_OBS_SHA256)
        self.assertEqual(meta["action_encoder_sha256"], LEGACY_ACTION_SHA256)
        self.assertEqual(meta["obs_encoder_version"], "1.0.0")
        self.assertEqual(meta["action_encoder_version"], "1.0.0")
        self.assertEqual(meta["reward_version"], "1.0.0")


class MultiSeatEncoderContract(unittest.TestCase):

    def test_versions_dims_and_hashes(self):
        for n, dim in ((3, 688), (4, 807)):
            contract = encoding_multi.contract_for(n)
            self.assertEqual(contract.obs_dim, dim)
            self.assertEqual(contract.action_dim, 109)
            self.assertEqual(contract.obs_encoder_version, "2.0.0")
            self.assertEqual(contract.action_encoder_version, "1.0.0")
            obs_hash, action_hash = contract.encoder_sha256()
            self.assertEqual(action_hash, LEGACY_ACTION_SHA256)
            self.assertNotEqual(obs_hash, LEGACY_OBS_SHA256)

    def test_multi_obs_hash_covers_the_legacy_encoder(self):
        """委譲先(旧 encoder)の obs hash が変われば多人数 obs hash も変わる。"""
        before = encoding_multi.encoder_sha256()
        with mock.patch.object(encoding, "encoder_sha256",
                               return_value=("0" * 64, LEGACY_ACTION_SHA256)):
            changed = encoding_multi.encoder_sha256()
        self.assertNotEqual(before[0], changed[0])
        self.assertEqual(before[1], changed[1])

    def test_multi_obs_hash_covers_its_own_source(self):
        before = encoding_multi.encoder_sha256()
        with mock.patch.object(encoding_multi, "_source_bytes",
                               return_value=b"different source"):
            changed = encoding_multi.encoder_sha256()
        self.assertNotEqual(before[0], changed[0])

    def test_encoded_lengths_and_two_seat_rejection(self):
        for n in (3, 4):
            state = game.new_game(n, 3, CARDS)
            obs = observe(state, game.actor(state), CARDS)
            vec = encoding_multi.encode_observation(obs)
            self.assertEqual(len(vec), encoding_multi.obs_dim(n))
            self.assertEqual(vec, encoding.encode_observation(obs))
            self.assertEqual(len(encoding_multi.encode_action(
                obs, obs.legal_actions[0])), 109)
        two = game.new_game(2, 3, CARDS)
        obs2 = observe(two, game.actor(two), CARDS)
        with self.assertRaises(ValueError):
            encoding_multi.encode_observation(obs2)

    def test_contract_rejects_unsupported_counts(self):
        for bad in (1, 5, 0, True, 3.0, "3", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    encoding_multi.contract_for(bad)

    def test_encode_for_net_rejects_wrong_seat_model(self):
        class Fake(object):
            def __init__(self, dim):
                self.dim = dim

            def arch_dict(self):
                return {"obs_dim": self.dim, "action_dim": 109}

        state = game.new_game(3, 4, CARDS)
        obs = observe(state, game.actor(state), CARDS)
        encoding_multi.encode_for_net(Fake(688), obs)
        for dim in (569, 807):
            with self.assertRaises(ValueError):
                encoding_multi.encode_for_net(Fake(dim), obs)


class MultiSeatMeta(unittest.TestCase):

    def test_current_meta_selects_contract_from_player_count(self):
        for n in (3, 4):
            obs_hash, action_hash = encoding_multi.encoder_sha256()
            meta = checkpoint.current_meta(n, _arch(n))
            self.assertEqual(meta["num_players"], n)
            self.assertEqual(meta["obs_encoder_version"], "2.0.0")
            self.assertEqual(meta["action_encoder_version"], "1.0.0")
            self.assertEqual(meta["obs_encoder_sha256"], obs_hash)
            self.assertEqual(meta["action_encoder_sha256"], action_hash)
            self.assertEqual(meta["reward_version"],
                             reward.MULTI_REWARD_VERSION)

    def test_expected_meta_rejects_unsupported_counts(self):
        for bad in (0, 1, 5, True, 3.0, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    checkpoint.expected_meta(bad, _arch(3))

    def test_round_trip_meta(self):
        for n in (3, 4):
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "init.meta.json")
                meta = checkpoint.current_meta(n, _arch(n))
                checkpoint.save(path, meta)
                loaded = checkpoint.load(
                    path, expected=checkpoint.expected_meta(n, _arch(n)))
                self.assertEqual(loaded, meta)

    def _assert_rejected(self, saved, num_players, arch, forge):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.meta.json")
            changed = copy.deepcopy(saved)
            forge(changed)
            _rewrite(path, changed)
            with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(num_players, arch))

    def test_seat_count_and_version_forgeries_are_rejected(self):
        three = checkpoint.current_meta(3, _arch(3))
        four = checkpoint.current_meta(4, _arch(4))
        two = checkpoint.current_meta(2, {"obs_dim": 569, "action_dim": 109,
                                          "hidden": [8],
                                          "action_hidden": [4]})
        cases = [
            # 3 席モデルを 4 席として読む(meta の人数を書換え)。hash/版は同一
            # 契約なので次元照合が最後の防壁になる。
            ("3->4 num_players", three, 4, _arch(3),
             lambda m: m.__setitem__("num_players", 4)),
            ("3 read as 4", three, 4, _arch(3), lambda m: None),
            ("4 read as 3", four, 3, _arch(4), lambda m: None),
            ("3 read as 2", three, 2, _arch(3), lambda m: None),
            ("3 forged to 2", three, 2, _arch(3),
             lambda m: m.__setitem__("num_players", 2)),
            # 2 席 (旧版) メタを 3 席として読む。旧版を全人数に許す例外は無い。
            ("2 read as 3", two, 3, two["model_arch"], lambda m: None),
            ("2 forged to 3", two, 3, two["model_arch"],
             lambda m: m.__setitem__("num_players", 3)),
            ("obs version 1.0.0 on 3", three, 3, _arch(3),
             lambda m: m.__setitem__("obs_encoder_version", "1.0.0")),
            ("legacy obs hash on 3", three, 3, _arch(3),
             lambda m: m.__setitem__("obs_encoder_sha256", LEGACY_OBS_SHA256)),
            ("reward 1.0.0 on 4", four, 4, _arch(4),
             lambda m: m.__setitem__("reward_version", "1.0.0")),
            ("num_players float", three, 3, _arch(3),
             lambda m: m.__setitem__("num_players", 3.0)),
            ("num_players bool", three, 3, _arch(3),
             lambda m: m.__setitem__("num_players", True)),
            ("obs_dim float", three, 3, dict(_arch(3), obs_dim=688.0),
             lambda m: m["model_arch"].__setitem__("obs_dim", 688.0)),
            ("obs_dim of 4 on 3", three, 3, dict(_arch(3), obs_dim=807),
             lambda m: m["model_arch"].__setitem__("obs_dim", 807)),
        ]
        for name, saved, n, arch, forge in cases:
            with self.subTest(name=name):
                self._assert_rejected(saved, n, arch, forge)

    def test_two_seat_strict_check_is_preserved(self):
        """2 席の既存厳格検査は変えない: 旧版メタは 2 席としてだけ読め、
        多人数の版を名乗る 2 席メタは拒否される。"""
        arch = {"obs_dim": 16, "action_dim": 8}
        two = checkpoint.current_meta(2, arch)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "two.meta.json")
            checkpoint.save(path, two)
            checkpoint.load(path, expected=checkpoint.expected_meta(2, arch))
        for key, value in (("obs_encoder_version", "2.0.0"),
                           ("reward_version", "2.0.0"),
                           ("obs_encoder_sha256",
                            encoding_multi.encoder_sha256()[0])):
            with self.subTest(key=key):
                self._assert_rejected(
                    two, 2, arch, lambda m, k=key, v=value: m.__setitem__(k, v))

    def test_gen1300_meta_is_two_seat_only(self):
        meta_path = os.path.join(CHAMPION_DIR, "gen1300.meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            arch = json.load(fh)["model_arch"]
        checkpoint.load(meta_path, expected=checkpoint.expected_meta(2, arch))
        for n in (3, 4):
            with self.subTest(n=n):
                with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                    checkpoint.load(
                        meta_path, expected=checkpoint.expected_meta(n, arch))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class InitialCheckpointsWithTorch(unittest.TestCase):

    def test_save_load_initial_checkpoint_per_seat_count(self):
        from naeco.train import net as net_mod
        from naeco.train.agent import PolicyAgent
        for n in (3, 4):
            with tempfile.TemporaryDirectory() as td:
                net = net_mod.build_for_players(n, seed=100 + n)
                meta_path = os.path.join(td, "init%d.meta.json" % n)
                net_mod.save_checkpoint(net, n, meta_path,
                                        extra={"root_seed": 100 + n})
                self.assertTrue(os.path.isfile(meta_path[:-len(".meta.json")]
                                               + ".weights.pt"))
                loaded, meta = net_mod.load_checkpoint(meta_path, n)
                self.assertEqual(meta["num_players"], n)
                self.assertEqual(meta["obs_encoder_version"], "2.0.0")
                self.assertEqual(meta["reward_version"], "2.0.0")
                self.assertEqual(meta["root_seed"], 100 + n)
                state = game.new_game(n, 9, CARDS)
                obs = observe(state, game.actor(state), CARDS)
                self.assertEqual(PolicyAgent(net).act(obs),
                                 PolicyAgent(loaded).act(obs))
                other = 4 if n == 3 else 3
                with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                    net_mod.load_checkpoint(meta_path, other)
                with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                    net_mod.load_checkpoint(meta_path, 2)

    def test_initial_checkpoint_is_seed_deterministic(self):
        from naeco.train import net as net_mod
        a = net_mod.build_for_players(3, seed=5).state_dict()
        b = net_mod.build_for_players(3, seed=5).state_dict()
        for key in a:
            self.assertTrue(bool((a[key] == b[key]).all()), key)

    def test_save_rejects_net_of_other_seat_count(self):
        from naeco.train import net as net_mod
        net = net_mod.build_for_players(3, seed=1)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                net_mod.save_checkpoint(net, 4, os.path.join(td, "x.meta.json"))
            self.assertEqual(os.listdir(td), [])

    def test_weights_shape_forgery_is_rejected(self):
        from naeco.train import net as net_mod
        with tempfile.TemporaryDirectory() as td:
            meta_path = os.path.join(td, "a.meta.json")
            net_mod.save_checkpoint(net_mod.build_for_players(3, seed=1), 3,
                                    meta_path)
            other = net_mod.build_for_players(3, hidden=(16,), seed=1)
            torch.save(other.state_dict(),
                       meta_path[:-len(".meta.json")] + ".weights.pt")
            with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                net_mod.load_checkpoint(meta_path, 3)

    def test_policy_agent_rejects_observation_of_other_seat_count(self):
        from naeco.train import net as net_mod
        from naeco.train.agent import PolicyAgent
        from naeco.train.search_evaluator import NetEvaluator
        net3 = net_mod.build_for_players(3, seed=2)
        state = game.new_game(4, 2, CARDS)
        obs = observe(state, game.actor(state), CARDS)
        with self.assertRaises(ValueError):
            PolicyAgent(net3).act(obs)
        with self.assertRaises(ValueError):
            NetEvaluator(net3).evaluate(obs)
        with self.assertRaises(ValueError):
            NetEvaluator(net3).evaluate_value(obs)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class Gen1300Golden(unittest.TestCase):
    """gen1300 の SHA・obs/actions・logits/value・固定 seed 探索が golden と一致。"""

    @classmethod
    def setUpClass(cls):
        from naeco.train import net as net_mod
        meta_path = os.path.join(CHAMPION_DIR, "gen1300.meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            arch = json.load(fh)["model_arch"]
        checkpoint.load(meta_path, expected=checkpoint.expected_meta(2, arch))
        net = net_mod.PolicyValueNet(
            arch["obs_dim"], arch["action_dim"], hidden=tuple(arch["hidden"]),
            action_hidden=tuple(arch["action_hidden"]))
        net.load_state_dict(net_mod.load_state_dict_from_path(
            os.path.join(CHAMPION_DIR, "gen1300.weights.pt")))
        net.eval()
        cls.net = net

    def _close(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=FLOAT_TOL,
                                     abs_tol=FLOAT_TOL),
                        "%r != %r" % (actual, expected))

    def test_sha256(self):
        self.assertEqual(_sha(os.path.join(CHAMPION_DIR, "gen1300.weights.pt")),
                         GEN1300_WEIGHTS_SHA256)
        self.assertEqual(_sha(os.path.join(CHAMPION_DIR, "gen1300.meta.json")),
                         GEN1300_META_SHA256)

    def test_obs_actions_logits_value(self):
        from naeco.train.agent import PolicyAgent
        from naeco.train.search_evaluator import NetEvaluator
        agent = PolicyAgent(self.net, "greedy")
        evaluator = NetEvaluator(self.net)
        state = game.new_game(2, 7, CARDS)
        for row in GREEDY_ROWS:
            obs = observe(state, game.actor(state), CARDS)
            obs_vec = encoding.encode_observation(obs)
            act_vecs = [encoding.encode_action(obs, a)
                        for a in obs.legal_actions]
            self.assertEqual(_vec_hash(obs_vec), row["obs_hash"])
            self.assertEqual(_vec_hash(act_vecs), row["actions_hash"])
            logits, values = self.net.infer(
                [obs_vec], [act_vecs], [[True] * len(act_vecs)])
            got = [float(x) for x in logits[0].tolist()]
            self.assertEqual(len(got), len(row["logits"]))
            for a, b in zip(got, row["logits"]):
                self._close(a, b)
            self._close(float(values.tolist()[0]), row["value"])
            unused_priors, p = evaluator.evaluate(obs)
            self._close(p, row["p"])
            action = agent.act(obs)
            self.assertEqual(list(action.sort_key()), row["chosen"])
            state = game.step(state, action, CARDS).state

    def test_fixed_seed_search(self):
        from naeco.ai.search.ismcts import IsmctsAgent
        from naeco.train.agent import PolicyAgent
        from naeco.train.search_evaluator import NetEvaluator
        greedy = PolicyAgent(self.net, "greedy")
        search = IsmctsAgent(1234, 300, cards=CARDS,
                             evaluator=NetEvaluator(self.net),
                             policy_agent=PolicyAgent(self.net, "greedy"))
        state = game.new_game(2, 11, CARDS)
        chosen = []
        while state.phase != Phase.GAME_OVER and len(chosen) < 12:
            actor = game.actor(state)
            obs = observe(state, actor, CARDS)
            if actor == 0:
                action = search.act(obs)
                chosen.append(list(action.sort_key()))
            else:
                action = greedy.act(obs)
            state = game.step(state, action, CARDS).state
        s = search.stats
        self.assertEqual(chosen, SEARCH_GOLDEN["chosen"])
        for key in ("transitions", "evaluations", "nodes_created",
                    "determinizations", "delegated_calls", "decisions"):
            self.assertEqual(getattr(s, key), SEARCH_GOLDEN[key], key)

    def test_fixed_seed_static_search(self):
        from naeco.ai.search.ismcts import IsmctsAgent
        from naeco.train.agent import PolicyAgent
        greedy = PolicyAgent(self.net, "greedy")
        static = IsmctsAgent(99, 200, cards=CARDS)
        state = game.new_game(2, 5, CARDS)
        chosen = []
        while state.phase != Phase.GAME_OVER and len(chosen) < 8:
            actor = game.actor(state)
            obs = observe(state, actor, CARDS)
            if actor == 1:
                action = static.act(obs)
                chosen.append(list(action.sort_key()))
            else:
                action = greedy.act(obs)
            state = game.step(state, action, CARDS).state
        self.assertEqual(chosen, STATIC_SEARCH_GOLDEN["chosen"])
        for key in ("transitions", "evaluations", "nodes_created"):
            self.assertEqual(getattr(static.stats, key),
                             STATIC_SEARCH_GOLDEN[key], key)


@pytest.mark.skipif(not HAS_TORCH, reason="torch が無い環境では skip")
@pytest.mark.parametrize("num_players", (2, 3, 4))
@pytest.mark.parametrize("key,value", (
    ("hidden", [8.0]),
    ("hidden", ["8"]),
    ("action_hidden", [4.0]),
    ("hidden", [True]),
    ("action_hidden", [False]),
    ("hidden", [8, "8"]),
    ("action_hidden", [4, 4.0]),
    ("hidden", 8),
    ("action_hidden", {"width": 4}),
    ("hidden", None),
    ("action_hidden", "4"),
    ("obs_dim", True),
    ("obs_dim", 569.0),
    ("obs_dim", "569"),
    ("action_dim", False),
    ("action_dim", 109.0),
    ("action_dim", "109"),
))
def test_load_checkpoint_rejects_arch_types_before_expected(
        tmp_path, num_players, key, value):
    """B-R3-01: 保存 arch の自己比較や constructor の int() で偽装を通さない。"""
    from naeco.train import net as net_mod
    path = str(tmp_path / "forged.meta.json")
    net = net_mod.build_for_players(
        num_players, hidden=(8,), action_hidden=(4,), seed=1)
    net_mod.save_checkpoint(net, num_players, path)
    with open(path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["model_arch"][key] = value
    _rewrite(path, meta)
    with mock.patch.object(checkpoint, "expected_meta") as expected, \
            mock.patch.object(net_mod, "PolicyValueNet") as constructor, \
            mock.patch.object(net_mod, "load_state_dict_from_path") as weights:
        with pytest.raises(checkpoint.IncompatibleCheckpoint,
                           match="model_arch\\.%s" % key):
            net_mod.load_checkpoint(path, num_players)
        expected.assert_not_called()
        constructor.assert_not_called()
        weights.assert_not_called()


@pytest.mark.skipif(not HAS_TORCH, reason="torch が無い環境では skip")
@pytest.mark.parametrize("num_players", (2, 3, 4))
def test_load_checkpoint_preserves_valid_arch_and_inference(tmp_path, num_players):
    from naeco.train import net as net_mod
    path = str(tmp_path / "valid.meta.json")
    net = net_mod.build_for_players(
        num_players, hidden=(8, 8), action_hidden=(4, 4), seed=1)
    net_mod.save_checkpoint(net, num_players, path)
    loaded, meta = net_mod.load_checkpoint(path, num_players)
    assert loaded.arch_dict() == meta["model_arch"] == net.arch_dict()
    assert not loaded.training
    state = game.new_game(num_players, 7, CARDS)
    obs = observe(state, game.actor(state), CARDS)
    contract = encoding_multi.contract_for(num_players)
    assert obs.legal_actions is not None
    args = ([contract.encode_observation(obs)],
            [[contract.encode_action(obs, a) for a in obs.legal_actions]],
            [[True] * len(obs.legal_actions)])
    for actual, expected in zip(loaded.infer(*args), net.infer(*args)):
        assert actual.tolist() == expected.tolist()


if __name__ == "__main__":
    unittest.main()
