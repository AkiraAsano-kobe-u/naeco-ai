"""A4 チェックポイント互換キーと拒否のテスト(Phase 3 設計書 §4.4 / §14.2 / §15)。

「歯」の狙い: A 表の拒否検査・エンコーダ sha256 の再計算照合・force 回避の
不存在・expected 必須化(設計書 14.2)と、expected の専用不変型化・A 表の
load 側組み立て・extra の B 表限定・内部マーカー廃止(設計書 15.1)を、
実装から1つ外すと落ちる形で固定する。さらに設計書 16 の改訂 — A 表の
型まで一致を要求する照合(_strict_equal)と、等値になる異型値
(1 → true / 1 → 1.0 / 2 → 2.0 / model_arch 内の float 偽装)の拒否(16.1)、
expected_meta の model_arch 防御的コピー(16.2)も固定する。Phase 4 の
排他ロック(CheckpointLock / acquire_checkpoint_lock)も固定する。torch は
import しない。
"""
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from naeco.engine import cards as cards_mod
from naeco.engine import observe as observe_mod
from naeco.engine import state as state_mod
from naeco.train import checkpoint
from naeco.train import encoding
from naeco.train import reward

MODEL_ARCH = {"hidden": [64, 64], "value_head": True}

# 契約で定められた A 表 13 キー(設計書 §4.4 / core-api §8.3)。
EXPECTED_A_KEYS = (
    "checkpoint_schema_version",
    "rules_version",
    "cards_data_version",
    "cards_sha256",
    "api_version",
    "obs_version",
    "obs_encoder_version",
    "action_encoder_version",
    "obs_encoder_sha256",
    "action_encoder_sha256",
    "reward_version",
    "num_players",
    "model_arch",
)

# 契約で定められた B 表キー。train_ppo の再開状態(next_episode_index /
# start_index)を記録できるよう拡張した(設計書 §15.1 第三の「extra は B 表
# キーのみ許可」を満たすため。記録専用で拒否しない意味論のまま)。
EXPECTED_B_KEYS = (
    "git_commit",
    "python_version",
    "torch_version",
    "root_seed",
    "hyperparameters",
    "training_steps",
    "generation",
    "eval_results",
    "created_at",
    "next_episode_index",
    "start_index",
)


def _break_value(value):
    """A 表キーを確実に別の型・値へ壊す(設計書 §16.1 第三)。

    整数は「+1 した別値」ではなく「等値になる異型値(1 → 1.0 / 2 → 2.0)」
    へ変える。Python では 1.0 == 1 / 2.0 == 2 が成立するため、単純な等値
    比較の実装はこの偽装を検出できない。型まで一致を要求しない実装へ
    戻すと、13キーの照合テスト(test_a_key_mismatch_is_rejected)が
    checkpoint_schema_version / num_players / model_arch でここで落ちる。
    """
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return "changed-" + value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, dict):
        return {k: _break_type_only(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_break_type_only(v) for v in value]
    if value is None:
        return "changed"
    return "changed"


def _break_type_only(value):
    """値を保ったまま型だけ変える(設計書 §16.1 第三の型偽装)。

    1 → true / 1 → 1.0 / 2 → 2.0 / model_arch 内の整数 → float へ変える。
    偽装後の値は Python の等値比較では元の値と区別できない
    (True == 1 / 1.0 == 1 / 2.0 == 2)ため、型まで一致を要求しない
    実装はこれを素通りさせる。bool は int(True → 1)へ、int は float へ
    変え、dict / list は再帰する。
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, dict):
        return {k: _break_type_only(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_break_type_only(v) for v in value]
    return value


def _record_value(value):
    """B 表キーを記録値として別の値へ変える。"""
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return "changed-" + value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, dict):
        return {"changed": True}
    if isinstance(value, list):
        return ["changed"]
    if value is None:
        return "changed"
    return "changed"


def _rewrite(path, meta):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")


class CurrentMetaTests(unittest.TestCase):

    def test_contract_key_lists_are_pinned(self):
        """A/B 表から1つ除くとここで契約のピンが外れて落ちる。"""
        self.assertEqual(tuple(checkpoint.A_KEYS), EXPECTED_A_KEYS)
        self.assertEqual(tuple(checkpoint.B_KEYS), EXPECTED_B_KEYS)

    def test_current_meta_reflects_current_versions(self):
        """current_meta の A 表が現在の定数・再計算ハッシュと一致する。"""
        cards = cards_mod.load_cards()
        obs_hash, action_hash = encoding.encoder_sha256()
        # Phase 10 D2: 旧 encoding / REWARD_VERSION は 2 席の契約。3/4 席の契約は
        # tests/test_phase10_checkpoint.py で固定する。
        meta = checkpoint.current_meta(2, MODEL_ARCH)
        self.assertEqual(meta["checkpoint_schema_version"],
                         checkpoint.CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(meta["rules_version"], state_mod.RULES_VERSION)
        self.assertEqual(meta["cards_data_version"], cards.data_version)
        self.assertEqual(meta["cards_sha256"], cards.sha256)
        self.assertEqual(meta["api_version"], state_mod.API_VERSION)
        self.assertEqual(meta["obs_version"], observe_mod.OBS_VERSION)
        self.assertEqual(meta["obs_encoder_version"],
                         encoding.OBS_ENCODER_VERSION)
        self.assertEqual(meta["action_encoder_version"],
                         encoding.ACTION_ENCODER_VERSION)
        self.assertEqual(meta["obs_encoder_sha256"], obs_hash)
        self.assertEqual(meta["action_encoder_sha256"], action_hash)
        self.assertEqual(meta["reward_version"], reward.REWARD_VERSION)
        self.assertEqual(meta["num_players"], 2)
        self.assertEqual(meta["model_arch"], MODEL_ARCH)
        for key in checkpoint.B_KEYS:
            self.assertIn(key, meta)

    def test_current_meta_has_no_internal_keys(self):
        """内部マーカーは廃止された(設計書 §15.1 第四)。

        歯: マーカーキーを再導入して current_meta の戻り値や保存メタへ
        現れるようにすると落ちる。
        """
        self.assertFalse(hasattr(checkpoint, "_EXPECTED_MARKER_KEY"))
        meta = checkpoint.current_meta(2, MODEL_ARCH)
        for key in meta:
            self.assertFalse(key.startswith("_"), key)
        self.assertNotIn("_naeco_expected_marker", meta)


class ExtraKeyTests(unittest.TestCase):
    """設計書 15.1 第三: current_meta の extra は B 表キーのみ許可。"""

    def test_extra_rejects_saved_meta(self):
        """経路1の攻撃: current_meta(np, arch, extra=saved) は ValueError。

        保存メタ全体は A 表キーを含むため extra として渡すと ValueError に
        なり、expected を偽装して load を通す手段にならない(設計書 §15.3)。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            with self.assertRaises(ValueError):
                checkpoint.current_meta(
                    saved["num_players"], saved["model_arch"], extra=saved)

    def test_extra_rejects_each_a_key(self):
        """A 表13キーのどれを extra に入れても ValueError。"""
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        for key in checkpoint.A_KEYS:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    checkpoint.current_meta(
                        2, MODEL_ARCH, extra={key: saved[key]})

    def test_extra_rejects_unknown_and_internal_keys(self):
        """B 表に無い任意キー(内部キー風の名前も)は ValueError。"""
        for key in ("_naeco_expected_marker", "_internal", "foo",
                    "next_episode_index2"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    checkpoint.current_meta(
                        2, MODEL_ARCH, extra={key: 1})

    def test_extra_accepts_b_keys(self):
        """B 表キーの上書きは許可される(記録用)。"""
        meta = checkpoint.current_meta(
            2, MODEL_ARCH,
            extra={"torch_version": "2.8.0+cpu", "root_seed": 42,
                   "training_steps": 3, "generation": 1,
                   "next_episode_index": 2002, "start_index": 2000})
        self.assertEqual(meta["torch_version"], "2.8.0+cpu")
        self.assertEqual(meta["root_seed"], 42)
        self.assertEqual(meta["training_steps"], 3)
        self.assertEqual(meta["next_episode_index"], 2002)
        # A 表の値は extra では上書きできない(許可されない)。
        self.assertEqual(meta["num_players"], 2)
        self.assertEqual(meta["model_arch"], MODEL_ARCH)


class ExpectedMetaTests(unittest.TestCase):
    """設計書 15.1 第一・第二: expected は専用の不変型。dict からは作れない。"""

    def test_factory_returns_immutable_type(self):
        expected = checkpoint.expected_meta(2, MODEL_ARCH)
        self.assertEqual(expected.num_players, 2)
        self.assertEqual(expected.model_arch, MODEL_ARCH)
        # 不変: 属性の書換えは AttributeError。
        with self.assertRaises(AttributeError):
            expected.num_players = 3
        with self.assertRaises(AttributeError):
            expected.model_arch = {}
        # __slots__ のみで __dict__ を持たない(dict 化できない)。
        self.assertFalse(hasattr(expected, "__dict__"))

    def test_factory_validates_arguments(self):
        with self.assertRaises(ValueError):
            checkpoint.expected_meta(0, MODEL_ARCH)
        with self.assertRaises(ValueError):
            checkpoint.expected_meta(True, MODEL_ARCH)
        with self.assertRaises(ValueError):
            checkpoint.expected_meta("2", MODEL_ARCH)
        with self.assertRaises(ValueError):
            checkpoint.expected_meta(2, [])

    def test_expected_carries_only_num_players_and_model_arch(self):
        """A 表の値は expected に持てない(load 側が現在の実装から組み立てる)。

        歯: expected に A 表キー(rules_version 等)を保持できる型へ戻すと
        落ちる。呼び出し側が指定できるのは num_players と model_arch の2つ
        だけである(設計書 §15.1 第二)。
        """
        expected = checkpoint.expected_meta(2, MODEL_ARCH)
        for key in checkpoint.A_KEYS:
            if key in ("num_players", "model_arch"):
                continue
            self.assertFalse(hasattr(expected, key), key)
        with self.assertRaises(AttributeError):
            expected.rules_version = "forged"

    def test_model_arch_is_defensively_copied(self):
        """expected 生成時に model_arch の防御的コピーを取る(設計書 §16.2)。

        歯: expected_meta が model_arch をそのまま保持する実装へ戻すと、
        生成後に呼び出し側が元の dict を書き換えた内容が expected へ
        反映され、ここで落ちる。
        """
        arch = {"hidden": [64, 64], "value_head": True}
        expected = checkpoint.expected_meta(2, arch)
        self.assertIsNot(expected.model_arch, arch)
        # 生成後に呼び出し側が元の dict を書き換えても、expected が保持
        # する model_arch は影響を受けない(ネストした list も含む)。
        arch["hidden"][0] = 999
        arch["value_head"] = False
        arch["extra_key"] = 1
        self.assertEqual(
            expected.model_arch,
            {"hidden": [64, 64], "value_head": True})
        arch2 = {"hidden": [64, 64]}
        expected2 = checkpoint.expected_meta(2, arch2)
        arch2["hidden"].append(32)
        self.assertEqual(expected2.model_arch["hidden"], [64, 64])

    def test_load_rejects_dict_expected(self):
        """expected に dict(保存メタ)を渡すと TypeError(設計書 §15.1 第一)。

        歯: load の型検査を外して dict を受理する実装へ戻すと落ちる。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "r", encoding="utf-8") as fh:
                saved_meta = json.load(fh)
            with self.assertRaises(TypeError):
                checkpoint.load(path, expected=saved_meta)
            # current_meta の戻り値そのもの(dict)も同様に TypeError。
            with self.assertRaises(TypeError):
                checkpoint.load(path, expected=meta)
            # 経路2: マーカー(廃止済み)や任意キーを注入した dict も TypeError。
            forged = dict(saved_meta)
            forged["_naeco_expected_marker"] = "current-meta-v1"
            forged["_naeco_expected_marker2"] = "current-meta-v1"
            with self.assertRaises(TypeError):
                checkpoint.load(path, expected=forged)

    def test_load_rejects_non_expected_types(self):
        """専用型以外(None / list / tuple / str / 任意オブジェクト)は TypeError。"""
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            for bad in (None, [], (), "meta", object()):
                with self.assertRaises(TypeError):
                    checkpoint.load(path, expected=bad)


class RejectionTests(unittest.TestCase):

    def test_a_key_mismatch_is_rejected(self):
        """A 表の13キーを1つずつ壊すと IncompatibleCheckpoint(歯 #2)。

        A 表の照合は load 側が現在の実装から組み立てた値で毎回実行される
        (expected の有無・内容では緩まない)。ここでは正しい expected_meta
        を渡した状態で、13キーそれぞれの不一致が必ず拒否されることを固定
        する。
        """
        for key in checkpoint.A_KEYS:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, MODEL_ARCH)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    changed = dict(meta)
                    changed[key] = _break_value(changed[key])
                    _rewrite(path, changed)
                    with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                        checkpoint.load(
                            path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
                    self.assertIn(key, str(cm.exception))

    def test_forged_rules_version_rejected_via_any_route(self):
        """経路1(extra=saved)と経路2(マーカー注入)の両方で受理されない。

        rules_version を偽装した保存メタは:
        - 経路1: current_meta(np, arch, extra=saved) が ValueError を出し
          expected を構築できない。
        - 経路2: マーカーや任意キーを注入した dict は expected として
          TypeError(型で担保するため)。
        - 正規経路: 専用型を渡す load は IncompatibleCheckpoint。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            saved["rules_version"] = "forged-rules"
            saved["_naeco_expected_marker"] = "current-meta-v1"
            # 偽装した内容をファイルへ書き戻す。書き戻さないと正規経路の
            # load が「偽装されていない正しいファイル」を読んでしまい、
            # 検査が意図した攻撃を検出できない(テストの歯が抜ける)。
            _rewrite(path, saved)
            # 経路1: extra=saved で expected を偽装しようとする → ValueError。
            with self.assertRaises(ValueError):
                checkpoint.current_meta(
                    saved["num_players"], saved["model_arch"], extra=saved)
            # 経路2: マーカー注入済み dict を expected に渡す → TypeError。
            with self.assertRaises(TypeError):
                checkpoint.load(path, expected=saved)
            # 正規経路: 専用型を渡す load は、ファイル上に実在する偽装を
            # 拒否する(偽装 rules_version が A 表照合で落ちる)。
            with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
            self.assertIn("rules_version", str(cm.exception))

    def test_b_key_change_does_not_reject(self):
        """B 表のキーを変えても読み込める(記録専用)。"""
        for key in checkpoint.B_KEYS:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, MODEL_ARCH)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    changed = dict(meta)
                    changed[key] = _record_value(changed[key])
                    _rewrite(path, changed)
                    loaded = checkpoint.load(
                        path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
                    self.assertEqual(loaded, changed)

    def test_encoder_hashes_are_recomputed_from_implementation(self):
        """保存値だけ改竄した sha256 は expected を渡しても拒否される。

        エンコーダ hash の再計算照合は常時実行する(14.2 / 15.1 でも従来
        どおり)。A 表照合も同じ値を現在の実装から組み立てるため二重に防壁
        になる。
        """
        for key in ("obs_encoder_sha256", "action_encoder_sha256"):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, MODEL_ARCH)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    changed = dict(meta)
                    changed[key] = "0" * 64
                    _rewrite(path, changed)
                    with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                        checkpoint.load(
                            path,
                            expected=checkpoint.expected_meta(2, MODEL_ARCH))
                    self.assertIn(key, str(cm.exception))

    def test_missing_a_key_is_rejected(self):
        """A 表キーが欠落したメタも不一致として拒否する。"""
        for key in ("reward_version", "model_arch"):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, MODEL_ARCH)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    with open(path, "r", encoding="utf-8") as fh:
                        stripped = json.load(fh)
                    del stripped[key]
                    _rewrite(path, stripped)
                    with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                        checkpoint.load(
                            path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
                    self.assertIn(key, str(cm.exception))


class TypeForgeryTests(unittest.TestCase):
    """設計書 16.1 第三: 等値になる異型値の型偽装はすべて拒否する。

    Python では True == 1 / 1.0 == 1 / 2.0 == 2 が成立するため、単純な
    等値比較の実装は次の偽装を素通りさせる。型まで一致を要求しない
    実装へ戻すと、ここで落ちる。
    """

    def test_equivalent_type_forgeries_are_rejected(self):
        """1→true / 1→1.0 / 2→2.0 は、値が等しくても型が違えば拒否。"""
        cases = (
            ("checkpoint_schema_version", 1, True),   # 1 → true
            ("checkpoint_schema_version", 1, 1.0),    # 1 → 1.0
            ("num_players", 2, 2.0),                  # 2 → 2.0
        )
        for key, original, forged in cases:
            with self.subTest(key=key, forged=forged):
                # 偽装値は Python の等値比較では original と区別できない
                # (True == 1 / 1.0 == 1 / 2.0 == 2)。
                self.assertEqual(forged, original)
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, MODEL_ARCH)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    changed = dict(meta)
                    changed[key] = forged
                    _rewrite(path, changed)
                    with self.assertRaises(
                            checkpoint.IncompatibleCheckpoint) as cm:
                        checkpoint.load(
                            path,
                            expected=checkpoint.expected_meta(
                                2, MODEL_ARCH))
                    self.assertIn(key, str(cm.exception))

    def test_model_arch_nested_type_forgery_is_rejected(self):
        """model_arch 内の整数を同じ値の float へ偽装しても拒否される。

        Opus が実測で再現した経路(model_arch.obs_dim = 569.0)を固定する。
        入れ子の値でも型偽装を検出するには再帰的な型照合が必要で、
        浅い等値比較の実装はこれを素通りさせる。
        """
        arch = {"obs_dim": 569, "hidden": [64, 64]}
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, arch)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            # 値は 569 のまま、型だけ float へ偽装する(569.0 == 569)。
            saved["model_arch"]["obs_dim"] = 569.0
            self.assertEqual(saved["model_arch"]["obs_dim"], 569)
            _rewrite(path, saved)
            with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(2, arch))
            self.assertIn("model_arch", str(cm.exception))

    def test_model_arch_structure_mismatches_are_rejected(self):
        """model_arch はキー集合・リスト長・要素の型まで再帰的に照合する。

        設計書 §16.1 第二: 正規化(キャスト)せず、辞書のキー集合・各値の
        型・リストの長さと各要素の型まで一致を要求する。
        """
        arch = {"obs_dim": 569, "hidden": [64, 64], "value_head": True}
        forged_arches = [
            # キー集合の不一致(キー追加)
            dict(arch, extra_key=1),
            # キー集合の不一致(キー除去)
            {"obs_dim": 569, "hidden": [64, 64]},
            # リストの長さの不一致
            {"obs_dim": 569, "hidden": [64], "value_head": True},
            # リスト要素の型の不一致(64 → 64.0。64.0 == 64 が成立)
            {"obs_dim": 569, "hidden": [64.0, 64], "value_head": True},
        ]
        for bad_arch in forged_arches:
            with self.subTest(bad_arch=bad_arch):
                with tempfile.TemporaryDirectory() as td:
                    meta = checkpoint.current_meta(2, arch)
                    path = os.path.join(td, "meta.json")
                    checkpoint.save(path, meta)
                    changed = dict(meta)
                    changed["model_arch"] = bad_arch
                    _rewrite(path, changed)
                    with self.assertRaises(
                            checkpoint.IncompatibleCheckpoint) as cm:
                        checkpoint.load(
                            path, expected=checkpoint.expected_meta(
                                2, arch))
                    self.assertIn("model_arch", str(cm.exception))


class SaveLoadTests(unittest.TestCase):

    def test_round_trip_meta_is_identical(self):
        """save -> load でメタが完全一致する(マーカーは存在しない)。"""
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(
                4, MODEL_ARCH,
                extra={"root_seed": 42, "training_steps": 100,
                       "eval_results": {"win_rate": 0.75, "pairs": 800}},
            )
            path = os.path.join(td, "nested", "meta.json")
            checkpoint.save(path, meta)
            self.assertTrue(os.path.isfile(path))
            loaded = checkpoint.load(
                path, expected=checkpoint.expected_meta(4, MODEL_ARCH))
            self.assertEqual(loaded, dict(meta))

    def test_load_requires_expected(self):
        """expected を省略すると TypeError(既定値なし = 回避経路が無い)。

        歯: `load(meta_path, expected=None)` の既定値へ戻すと落ちる。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with self.assertRaises(TypeError):
                checkpoint.load(path)
            # None を明示しても型検査で TypeError(専用型以外は受理しない)。
            with self.assertRaises(TypeError):
                checkpoint.load(path, None)
        # シグネチャでも既定値が無いことを固定する。
        self.assertEqual(
            inspect.signature(checkpoint.load).parameters["expected"].default,
            inspect.Parameter.empty)

    def test_no_force_argument_in_public_signatures(self):
        """load に force 相当の引数が無い(キーワード迂回も無い)。"""
        self.assertEqual(
            tuple(inspect.signature(checkpoint.load).parameters),
            ("meta_path", "expected"),
        )
        self.assertNotIn("force", inspect.signature(checkpoint.load).parameters)
        self.assertNotIn("force", inspect.signature(checkpoint.save).parameters)
        self.assertNotIn(
            "force", inspect.signature(checkpoint.current_meta).parameters
        )
        self.assertNotIn(
            "force", inspect.signature(checkpoint.expected_meta).parameters
        )

    def test_environment_variable_cannot_bypass_rejection(self):
        """環境変数で拒否を迂回できない。"""
        env = {
            "NAECO_CHECKPOINT_FORCE": "1",
            "CHECKPOINT_FORCE": "1",
            "NAECO_FORCE": "1",
            "FORCE": "1",
        }
        with mock.patch.dict(os.environ, env):
            with tempfile.TemporaryDirectory() as td:
                meta = checkpoint.current_meta(2, MODEL_ARCH)
                path = os.path.join(td, "meta.json")
                checkpoint.save(path, meta)
                changed = dict(meta)
                changed["checkpoint_schema_version"] = 999
                _rewrite(path, changed)
                with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                    checkpoint.load(
                        path, expected=checkpoint.expected_meta(2, MODEL_ARCH))

    def test_json_is_sorted_and_lf(self):
        """JSON は sort_keys=True、ensure_ascii=True、LF 1行終端。

        内部マーカーは廃止されたため、保存内容は current_meta の戻り値と
        完全一致する。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = checkpoint.current_meta(2, MODEL_ARCH)
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, meta)
            with open(path, "rb") as fh:
                raw = fh.read()
            expected = (
                json.dumps(meta, sort_keys=True, ensure_ascii=True)
                + "\n"
            ).encode("utf-8")
            self.assertEqual(raw, expected)
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\r\n", raw)
            text = raw.decode("utf-8")
            self.assertEqual(
                text,
                json.dumps(meta, sort_keys=True, ensure_ascii=True) + "\n")
            self.assertEqual(text.encode("utf-8"), text.encode("ascii"))


class AtomicSaveTests(unittest.TestCase):
    """Phase 4: save() の原子的書込(一時ファイル + fsync + os.replace)。

    最終パスには常に「完全な旧内容」または「完全な新内容」だけが存在し、
    SIGKILL・ディスクフル・クラッシュでも途切れた/部分的な JSON が決して
    現れないことを固定する。最終パスへ直接 open() で書く実装へ戻すと落ちる。
    """

    def test_os_replace_used_with_temp_file_in_same_directory(self):
        """(a) os.replace が使われ、一時ファイルが同ディレクトリに作られる。

        歯: 最終パスへ直接 open() で書く実装へ戻すと os.replace が呼ばれず
        落ちる。置換直前の一時ファイルが meta_path と同一ディレクトリに実在
        することも固定する。
        """
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "nested")
            path = os.path.join(nested, "meta.json")
            seen = {}
            real_replace = os.replace

            def spy_replace(src, dst):
                seen["src"] = src
                seen["dst"] = dst
                # 置換直前: 一時ファイルが同一ディレクトリに実在する。
                self.assertEqual(os.path.dirname(src), nested)
                self.assertTrue(src.endswith(".tmp"))
                self.assertTrue(os.path.isfile(src))
                real_replace(src, dst)

            with mock.patch("os.replace", side_effect=spy_replace):
                checkpoint.save(
                    path, checkpoint.current_meta(2, MODEL_ARCH))
            self.assertEqual(seen["dst"], path)
            # 置換後の最終パスは完全な新内容(往復読込で検証)。
            loaded = checkpoint.load(
                path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
            self.assertEqual(loaded, dict(checkpoint.current_meta(2, MODEL_ARCH)))

    def test_no_temp_file_left_after_normal_save(self):
        """(c) 正常 save 後に一時ファイルが残らない。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, checkpoint.current_meta(2, MODEL_ARCH))
            self.assertEqual(os.listdir(td), ["meta.json"])

    def test_fsync_failure_keeps_old_content_and_cleans_temp(self):
        """(b) 置換前の失敗(ディスクフル: fsync 失敗)でも meta_path は汚れない。

        歯: 最終パスへ直接書く実装へ戻すと失敗時に旧内容が失われる。失敗時
        に一時ファイルを後始末しない実装へ戻すとディレクトリに残って落ちる。
        """
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, checkpoint.current_meta(2, MODEL_ARCH))
            with open(path, "rb") as fh:
                original = fh.read()

            def boom(*args, **kwargs):
                raise OSError(28, "No space left on device")

            with mock.patch("os.fsync", side_effect=boom):
                with self.assertRaises(OSError):
                    checkpoint.save(
                        path, checkpoint.current_meta(
                            2, MODEL_ARCH, extra={"generation": 1}))
            # meta_path は完全な旧内容のまま(部分書き・破損なし)。
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), original)
            # 一時ファイルは残らない。
            self.assertEqual(os.listdir(td), ["meta.json"])

    def test_json_dump_failure_keeps_old_content_and_cleans_temp(self):
        """json.dump の失敗(直列化できない値)でも meta_path は汚れない。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("old content\n")
            with self.assertRaises(TypeError):
                checkpoint.save(path, {"root_seed": object()})
            # meta_path は元の内容のまま。
            with open(path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "old content\n")
            # 一時ファイルは残らない。
            self.assertEqual(os.listdir(td), ["meta.json"])


class DefensiveLoadTests(unittest.TestCase):
    """Phase 4: load() の防御的変換(壊れた/途切れた JSON → IncompatibleCheckpoint)。

    再開時に json.JSONDecodeError / UnicodeDecodeError / ValueError を外へ
    漏らさない。json.load を包まない実装へ戻すと、以下で未捕捉例外が出て
    落ちる。正常な往復読込は既存の SaveLoadTests が固定する。
    """

    def test_truncated_json_load_raises_incompatible_checkpoint(self):
        """(d) 途切れた JSON を load すると IncompatibleCheckpoint。

        クラッシュ時の部分書き(完全な JSON の先頭半分だけが残る状態)を
        再現する。JSONDecodeError をそのまま漏らす実装へ戻すと落ちる
        (JSONDecodeError は ValueError の派生のため assertNotIsInstance も
        効く)。
        """
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, checkpoint.current_meta(2, MODEL_ARCH))
            with open(path, "rb") as fh:
                raw = fh.read()
            with open(path, "wb") as fh:
                fh.write(raw[: len(raw) // 2])
            with self.assertRaises(checkpoint.IncompatibleCheckpoint) as cm:
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
            self.assertNotIsInstance(cm.exception, ValueError)

    def test_garbage_json_load_raises_incompatible_checkpoint(self):
        """壊れた(JSON として読めない)内容も IncompatibleCheckpoint に変換。"""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write('{"checkpoint_schema_version": \n')
            with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(2, MODEL_ARCH))

    def test_invalid_utf8_load_raises_incompatible_checkpoint(self):
        """UTF-8 として読めないバイト列も IncompatibleCheckpoint に変換。

        UnicodeDecodeError をそのまま漏らす実装へ戻すと落ちる。
        """
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            with open(path, "wb") as fh:
                fh.write(b"\xff\xfe\x00\x01")
            with self.assertRaises(checkpoint.IncompatibleCheckpoint):
                checkpoint.load(
                    path, expected=checkpoint.expected_meta(2, MODEL_ARCH))


class CheckpointLockTests(unittest.TestCase):
    """Phase 4: チェックポイントディレクトリの排他ロック(fcntl.flock)。

    「歯」の狙い: 複数の train_ppo ジョブが同一 --checkpoint-dir を指して
    互いのチェックポイントを上書きし合う残留リスク(#4)を塞ぐロックを、
    実装から1つ外すと落ちる形で固定する。
    - (a) 同一プロセス内の二重取得(別 fd)が失敗する。
    - (b) 別プロセスが保持中に取得が失敗し、保持プロセスが死ぬ(SIGKILL)
      とカーネルが自動解放して再取得できる。
    - (c) release 後に再取得できる(with 文と acquire_checkpoint_lock も)。
    - (d) ロックファイルは世代として扱われない(_list_generations に出ず、
      _prune_checkpoints が孤児として削除しない)。
    - fcntl の無い非Unix環境ではロックが no-op になり、import と既存機能
      (save / load)は壊れない。
    torch は import しない。
    """

    def test_double_acquire_in_same_process_fails(self):
        """(a) 同一プロセス内の二重取得(別 fd)は CheckpointLockError。

        flock は open ごとの open file description に結び付くため、同じ
        プロセスでも別の open() からの取得は拒否される(flock(2))。ロック
        の有無をファイルの存在や内容で判定する実装へ戻すと落ちる。
        """
        if checkpoint.fcntl is None:
            self.skipTest("fcntl の無い環境ではロックは no-op")
        with tempfile.TemporaryDirectory() as td:
            lock1 = checkpoint.CheckpointLock(td)
            lock1.acquire()
            try:
                self.assertTrue(lock1.held)
                lock2 = checkpoint.CheckpointLock(td)
                with self.assertRaises(
                        checkpoint.CheckpointLockError) as cm:
                    lock2.acquire()
                self.assertFalse(lock2.held)
                self.assertIn("exclusive lock", str(cm.exception))
            finally:
                lock1.release()
            self.assertFalse(lock1.held)

    def test_other_process_holding_lock_blocks_acquire(self):
        """(b) 別プロセスが保持中は取得が失敗し、SIGKILL でカーネルが解放。

        サブプロセスに .lock を flock させ、親からの取得が CheckpointLock
        Error で失敗することを固定する。保持プロセスを SIGKILL した後は
        カーネルが自動解放するため、再取得できる(残置ロックで後続の再開
        ジョブを妨げないセマンティクス)。
        """
        if checkpoint.fcntl is None:
            self.skipTest("fcntl の無い環境ではロックは no-op")
        with tempfile.TemporaryDirectory() as td:
            child_code = (
                "import sys, time\n"
                "sys.path.insert(0, %r)\n"
                "from naeco.train import checkpoint\n"
                "lock = checkpoint.CheckpointLock(%r)\n"
                "lock.acquire()\n"
                "sys.stdout.write('LOCKED\\n')\n"
                "sys.stdout.flush()\n"
                "time.sleep(60)\n"
            ) % (ROOT, td)
            proc = subprocess.Popen(
                [sys.executable, "-c", child_code],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                # ロック取得の確認は stdout の LOCKED 行だけを使う。
                # stderr は子が sleep(60) 中は EOF にならず、assertEqual の
                # メッセージ引数で read() すると60秒ブロックし、その間に子が
                # sleep を終えて終了し flock が解放されてしまう(後の
                # assertRaises が不成立になる)。stderr は finally で子を
                # kill した後に回収する。
                self.assertEqual(proc.stdout.readline().strip(), "LOCKED")
                # 子が生存中(LOCKED 直後)は親からの取得が必ず失敗する。
                with self.assertRaises(checkpoint.CheckpointLockError):
                    checkpoint.CheckpointLock(td).acquire()
            finally:
                proc.kill()
                # 子の終了を待ち、stdout / stderr の残りを EOF まで回収する。
                # (readline で消費していない分もここで読み切る。)
                proc.communicate(timeout=30)
            # SIGKILL 後: flock はプロセス死亡時にカーネルが自動解放する。
            lock = checkpoint.CheckpointLock(td)
            lock.acquire()
            try:
                self.assertTrue(lock.held)
            finally:
                lock.release()

    def test_reacquire_after_release(self):
        """(c) release 後に再取得できる。with 文でも同じ意味論。

        flock は同一プロセスの別 fd からの取得も拒否するため、「保持中に
        別インスタンスの acquire が成功する」ことを検証してはならない
        (保持中に別 fd が失敗することは (a) が固定する)。ここでは
        release 後の再取得と with 文の acquire / release だけを固定する。
        """
        if checkpoint.fcntl is None:
            self.skipTest("fcntl の無い環境ではロックは no-op")
        with tempfile.TemporaryDirectory() as td:
            lock = checkpoint.CheckpointLock(td)
            # (1) acquire -> release で解放され、held が False になる。
            lock.acquire()
            self.assertTrue(lock.held)
            lock.release()
            self.assertFalse(lock.held)
            # (2) release 後に再取得でき、release で再び解放される。
            lock.acquire()
            self.assertTrue(lock.held)
            lock.release()
            self.assertFalse(lock.held)
            # (3) with 文で acquire / release でき、__exit__ 後に held が
            #     False になる。
            with checkpoint.CheckpointLock(td) as other:
                self.assertTrue(other.held)
            self.assertFalse(other.held)

    def test_acquire_checkpoint_lock_helper(self):
        """acquire_checkpoint_lock() は取得済みインスタンスを返す。"""
        if checkpoint.fcntl is None:
            self.skipTest("fcntl の無い環境ではロックは no-op")
        with tempfile.TemporaryDirectory() as td:
            lock = checkpoint.acquire_checkpoint_lock(td)
            try:
                self.assertTrue(lock.held)
                # ロックファイルはディレクトリ内の固定名(.lock)。
                self.assertEqual(
                    os.path.basename(lock.path), checkpoint.LOCK_FILENAME)
            finally:
                lock.release()

    def test_lock_file_not_listed_or_pruned_as_generation(self):
        """(d) ロックファイルは世代として扱われない。

        ロックファイル(.lock)は gen パターン(^gen(\\d+)\\.meta\\.json$ 等)
        に一致しないため、_list_generations に出ず、_prune_checkpoints の
        孤児掃除でも削除されない。ロックファイルを世代として数えたり孤児
        として削除する実装へ戻すと落ちる。
        """
        scripts_dir = os.path.join(ROOT, "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import train_ppo  # torch は module import しない(遅延 import)
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, checkpoint.LOCK_FILENAME)
            with open(lock_path, "w", encoding="utf-8") as fh:
                fh.write("")
            self.assertEqual(train_ppo._list_generations(td), [])
            # 世代ファイルと混在しても .lock は世代として数えない。
            for gen in (0, 1):
                with open(train_ppo._gen_meta_path(td, gen), "w",
                          encoding="utf-8") as fh:
                    fh.write("x")
            self.assertEqual(train_ppo._list_generations(td), [0, 1])
            # 孤児掃除(_prune_checkpoints)でも .lock は削除されない。
            train_ppo._prune_checkpoints(td, 1)
            self.assertTrue(
                os.path.exists(lock_path), ".lock が削除された")
            self.assertEqual(train_ppo._list_generations(td), [1])

    def test_noop_without_fcntl(self):
        """fcntl の無い非Unix環境ではロックは no-op で壊れない。

        歯: fcntl を必須 import する実装へ戻すと、この環境を再現した
        mock で import エラーが再現されず、また acquire() が失敗して落ち
        る。ロックが無効化されても save / load を含む既存機能は壊れない。
        """
        self.assertTrue(hasattr(checkpoint, "fcntl"))
        with mock.patch.object(checkpoint, "fcntl", None):
            with tempfile.TemporaryDirectory() as td:
                lock = checkpoint.CheckpointLock(td)
                lock.acquire()  # no-op: 失敗せず、保持もしない
                self.assertFalse(lock.held)
                lock.release()  # no-op: 何も解放しない
                lock.acquire()
                lock.release()
        # 既存機能(save / load)も壊れない。
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "meta.json")
            checkpoint.save(path, checkpoint.current_meta(2, MODEL_ARCH))
            loaded = checkpoint.load(
                path, expected=checkpoint.expected_meta(2, MODEL_ARCH))
            self.assertEqual(loaded["num_players"], 2)


if __name__ == "__main__":
    unittest.main()
