"""B5 学習CLI scripts/train_ppo.py のテスト(Phase 3 設計書 §2 / §4 / §8 / §13.2)。

受け入れ項目を「壊したときに落ちる」形で固定する:
- SIGTERM でチェックポイントを保存してから終了する(サブプロセス実測)。
- 世代ローテーションが保持数上限を守る(古い世代と孤立 weights / optimizer の整理)。
- 中断から再開でき、学習ステップ数・世代番号・次のエピソード index を引き継ぐ。
- Adam の状態(optimizer.state_dict())をチェックポイントへ保存し、再開時に
  読み戻す(設計書 14.3)。連続 N 世代と「途中で保存して別プロセスで再開した
  N 世代」で、重み・optimizer 状態・次 episode index が一致する。
- optimizer 状態の無い旧チェックポイントからの再開は警告を出したうえで
  新規作成を許す(既存の成果物を読めなくしない)。
- 実 PolicyValueNet の学習を workers=1 と workers=4 で走らせ、最終
  チェックポイントまで完全一致する(ロールアウト層に留まらない並列一致検査)。
- checkpoint.load へ必ず expected=expected_meta(...)(専用の不変型)を渡す
  (ソース走査 + 実行時。設計書 §15.1)。model_arch はこのプロセスの net から
  来るため保存メタの model_arch とは A 表照合で真に検証される(§15.2)。
- 4つのスレッド環境変数を 1 に固定し torch.set_num_threads(1) を行う。
- scripts/csi_job.sh から起動できる(カレントディレクトリ非依存)。
- 自己対戦 seed は train グループの 2000 以降のみ(dev / accept 不使用)。
- 破棄局を学習データに混ぜず、件数をカウンタとしてログに残す。
- 進捗ログに世代番号・学習ステップ数・損失統計・破棄局数・decision 毎秒。
- 待機上限は壁時計の絶対 deadline(--max-seconds の時刻判定)。
- 既定引数のまま _parse_args([]) が成功し、--generations 0(無制限)を
  受け付ける(負値のみ拒否)。
- Python 3.9 互換。torch が無い環境では torch 依存テストを skip する。
"""
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    torch = None

HAS_TORCH = torch is not None

import train_ppo  # noqa: E402
from naeco.train import encoding as enc  # noqa: E402

TRAIN_PPO_PATH = os.path.join(ROOT, "scripts", "train_ppo.py")
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_CAPTURE_SEQ = [0]


class _CaptureHandler(logging.Handler):
    """テスト用のログ回収ハンドラ。"""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def text(self):
        return "\n".join(self.format(record) for record in self.records)


def _capture_logger():
    """run() に渡すロガーと、記録を回収するハンドラを返す。"""
    _CAPTURE_SEQ[0] += 1
    logger = logging.getLogger("test_train_cli.%d" % _CAPTURE_SEQ[0])
    logger.propagate = False
    handler = _CaptureHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger, handler


def _base_argv(checkpoint_dir):
    """小さな net で速く走る学習設定の argv 列(テスト共通)。"""
    return [
        "--checkpoint-dir", checkpoint_dir,
        "--episodes", "2",
        "--generations", "1",
        "--workers", "1",
        "--seed", "7",
        "--hidden", "8", "8",
        "--action-hidden", "4",
        "--epochs", "1",
        "--minibatch-size", "16",
    ]


def _call_text(src, open_paren_index):
    """開き括弧位置から対応する閉じ括弧までのテキストを返す。"""
    depth = 0
    for i in range(open_paren_index, len(src)):
        char = src[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return src[open_paren_index:i + 1]
    return src[open_paren_index:]


class ThreadEnvTests(unittest.TestCase):
    """4つのスレッド環境変数と torch.set_num_threads(1)(設計書 §8.4)。"""

    def test_thread_env_vars_pinned_to_one(self):
        saved = dict((name, os.environ.get(name)) for name in THREAD_ENV_NAMES)
        try:
            train_ppo._fix_thread_env()
            for name in THREAD_ENV_NAMES:
                self.assertEqual(os.environ.get(name), "1", name)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_run_pins_torch_threads_via_ppo_wrapper(self):
        """run() は ppo_mod.set_num_threads(1) で torch スレッドを固定する(§8.4)。

        歯: train_ppo は torch を直接触らず、naeco/train/ppo.py の包み関数
        (set_num_threads)を呼ぶ。包みを外して torch.set_num_threads を直接
        呼ぶ実装へ戻すと落ちる。
        """
        from naeco.train import ppo as ppo_mod
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            with mock.patch.object(ppo_mod, "set_num_threads") as fake:
                logger, unused_handler = _capture_logger()
                rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            fake.assert_called_once_with(1)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_real_torch_threads_pinned_after_import(self):
        import naeco.train.net  # noqa: F401
        import naeco.train.ppo  # noqa: F401
        self.assertEqual(torch.get_num_threads(), 1)
        for name in THREAD_ENV_NAMES:
            self.assertEqual(os.environ.get(name), "1", name)


class SeedContractTests(unittest.TestCase):
    """自己対戦 seed は train グループの 2000 以降のみ(eval-seeds.md Phase 3)。"""

    def test_train_group_and_min_index_constants(self):
        self.assertEqual(train_ppo.TRAIN_GROUP, "train")
        self.assertEqual(train_ppo.MIN_TRAIN_INDEX, 2000)

    def test_validate_start_index_enforces_2000(self):
        self.assertEqual(train_ppo._validate_start_index(2000), 2000)
        self.assertEqual(train_ppo._validate_start_index(2500), 2500)
        for bad in (0, 1999):
            with self.assertRaises(ValueError):
                train_ppo._validate_start_index(bad)

    def test_parse_args_rejects_start_index_below_2000(self):
        with self.assertRaises(SystemExit) as ctx:
            train_ppo._parse_args(["--start-index", "1999"])
        self.assertEqual(ctx.exception.code, 2)
        args = train_ppo._parse_args(["--start-index", "2500"])
        self.assertEqual(args.start_index, 2500)

    def test_episode_root_is_train_group(self):
        from naeco.engine.state import eval_seed
        self.assertEqual(
            train_ppo._episode_root(2000), eval_seed("train", 2000))
        self.assertNotEqual(
            train_ppo._episode_root(2000), eval_seed("dev", 2000))
        self.assertNotEqual(
            train_ppo._episode_root(2000), eval_seed("accept", 2000))

    def test_specs_use_train_group_only(self):
        specs = train_ppo._make_specs(
            2000, 3, lambda index: ["agent%d" % index])
        self.assertEqual([spec[0] for spec in specs], ["train"] * 3)
        self.assertEqual([spec[1] for spec in specs], [2000, 2001, 2002])
        self.assertEqual([spec[2] for spec in specs], [0, 0, 0])

    def test_no_dev_or_accept_seed_path_in_source(self):
        """eval_seed の呼び出しはすべて TRAIN_GROUP 固定で、--group も無い。"""
        with open(TRAIN_PPO_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        matches = list(re.finditer(r"eval_seed\s*\(", src))
        self.assertTrue(matches, "eval_seed の呼び出しが無い")
        for match in matches:
            call = _call_text(src, match.end() - 1)
            self.assertIn(
                "TRAIN_GROUP", call,
                "eval_seed の group は TRAIN_GROUP 固定でなければならない: "
                + call)
        self.assertNotIn("--group", src)


class CheckpointLoadContractTests(unittest.TestCase):
    """§13.2 / §15.1: checkpoint.load へ必ず expected=expected_meta(...) を渡す。"""

    def test_every_load_call_passes_expected(self):
        with open(TRAIN_PPO_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        loads = list(re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\.load\s*\(", src))
        self.assertTrue(loads, "train_ppo.py に load 呼び出しが無い")
        for match in loads:
            receiver = match.group(1)
            if receiver in ("torch", "torch_module"):
                continue  # 重み読込。互換性照合の対象ではない
            call = _call_text(src, match.end() - 1)
            self.assertIn(
                "expected=", call,
                "checkpoint.load には expected= が必須: " + call)
            self.assertIn(
                "expected_meta(", call[call.index("expected="):],
                "expected= には expected_meta(...) を渡す: " + call)

    def test_source_uses_expected_factory_form(self):
        with open(TRAIN_PPO_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("checkpoint_mod.load(", src)
        self.assertIn("expected=checkpoint_mod.expected_meta(", src)
        # 旧形式(expected=current_meta(...) の dict)は存在してはならない(§15.1)。
        self.assertNotIn("expected=checkpoint_mod.current_meta(", src)
        self.assertNotIn("expected=checkpoint.current_meta(", src)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_resume_rejects_incompatible_checkpoint(self):
        """expected なしの load なら通ってしまう互換性拒否を実行時にも固定。"""
        from naeco.train import checkpoint as checkpoint_mod
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            os.makedirs(checkpoint_dir)
            meta = checkpoint_mod.current_meta(3, {"obs_dim": 1})
            checkpoint_mod.save(
                os.path.join(checkpoint_dir, "gen000.meta.json"), meta)
            torch.save(
                {"w": torch.zeros(1)},
                os.path.join(checkpoint_dir, "gen000.weights.pt"))
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            with self.assertRaises(checkpoint_mod.IncompatibleCheckpoint):
                train_ppo.run(args, logger=logger)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_resume_rejects_forged_model_arch(self):
        """保存メタの model_arch を偽装したチェックポイントは再開を拒否する。

        train_ppo の model_arch はこのプロセスの net から来る(保存メタから
        読まない)ため、偽装 arch は A 表照合で落ちる(設計書 §15.1 第二 /
        §15.2 の「検証を経ない読込経路を作らない」)。
        """
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            os.makedirs(checkpoint_dir)
            arch = PolicyValueNet(
                enc.obs_dim(2), enc.action_dim(),
                hidden=(8, 8), action_hidden=(4,)).arch_dict()
            checkpoint_mod.save(
                os.path.join(checkpoint_dir, "gen000.meta.json"),
                checkpoint_mod.current_meta(2, arch))
            torch.save(
                {"w": torch.zeros(1)},
                os.path.join(checkpoint_dir, "gen000.weights.pt"))
            meta_path = os.path.join(checkpoint_dir, "gen000.meta.json")
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            raw["model_arch"] = dict(raw["model_arch"], hidden=[16, 8])
            with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(raw, fh, sort_keys=True, ensure_ascii=True)
                fh.write("\n")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            with self.assertRaises(checkpoint_mod.IncompatibleCheckpoint):
                train_ppo.run(args, logger=logger)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_resume_missing_weights_raises(self):
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            os.makedirs(checkpoint_dir)
            arch = PolicyValueNet(
                enc.obs_dim(2), enc.action_dim(),
                hidden=(8, 8), action_hidden=(4,)).arch_dict()
            meta = checkpoint_mod.current_meta(2, arch)
            checkpoint_mod.save(
                os.path.join(checkpoint_dir, "gen000.meta.json"), meta)
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            with self.assertRaises(ValueError):
                train_ppo.run(args, logger=logger)


class RotationTests(unittest.TestCase):
    """世代ローテーションが保持数上限を守る(ディスクを埋めない)。"""

    def test_prune_keeps_newest_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            for gen in range(8):
                for path in (train_ppo._gen_meta_path(tmp, gen),
                             train_ppo._gen_weights_path(tmp, gen),
                             train_ppo._gen_optimizer_path(tmp, gen)):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("x")
            logger, unused_handler = _capture_logger()
            removed = train_ppo._prune_checkpoints(tmp, 3, logger)
            self.assertEqual(removed, [0, 1, 2, 3, 4])
            self.assertEqual(train_ppo._list_generations(tmp), [5, 6, 7])
            for gen in (5, 6, 7):
                for path in (train_ppo._gen_meta_path(tmp, gen),
                             train_ppo._gen_weights_path(tmp, gen),
                             train_ppo._gen_optimizer_path(tmp, gen)):
                    self.assertTrue(os.path.exists(path), path)
            for gen in (0, 1, 2, 3, 4):
                for path in (train_ppo._gen_meta_path(tmp, gen),
                             train_ppo._gen_weights_path(tmp, gen),
                             train_ppo._gen_optimizer_path(tmp, gen)):
                    self.assertFalse(os.path.exists(path), path)

    def test_prune_removes_orphan_weights_without_meta(self):
        """クラッシュで meta 書込前に中断した世代は保持枠内でも削除する。
        optimizer ファイルも weights と一緒に削除する。"""
        with tempfile.TemporaryDirectory() as tmp:
            for gen in (0, 1):
                for path in (train_ppo._gen_meta_path(tmp, gen),
                             train_ppo._gen_weights_path(tmp, gen),
                             train_ppo._gen_optimizer_path(tmp, gen)):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("x")
            for path in (train_ppo._gen_weights_path(tmp, 4),
                         train_ppo._gen_optimizer_path(tmp, 4)):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("x")
            train_ppo._prune_checkpoints(tmp, 1)
            self.assertEqual(train_ppo._list_generations(tmp), [1])
            self.assertFalse(
                os.path.exists(train_ppo._gen_meta_path(tmp, 0)))
            self.assertFalse(
                os.path.exists(train_ppo._gen_weights_path(tmp, 0)))
            self.assertFalse(
                os.path.exists(train_ppo._gen_optimizer_path(tmp, 0)))
            self.assertFalse(
                os.path.exists(train_ppo._gen_weights_path(tmp, 4)))
            self.assertFalse(
                os.path.exists(train_ppo._gen_optimizer_path(tmp, 4)))

    def test_prune_rejects_keep_latest_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                train_ppo._prune_checkpoints(tmp, 0)

    def test_prune_missing_dir_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope")
            self.assertEqual(train_ppo._prune_checkpoints(missing, 3), [])
            self.assertIsNone(train_ppo._find_latest_generation(missing))

    def test_find_latest_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(train_ppo._find_latest_generation(tmp))
            for gen in (0, 2, 5):
                with open(train_ppo._gen_meta_path(tmp, gen), "w",
                          encoding="utf-8") as fh:
                    fh.write("x")
            self.assertEqual(train_ppo._find_latest_generation(tmp), 5)

    def test_generation_path_layout(self):
        self.assertEqual(
            train_ppo._gen_meta_path("/tmp/ckpt", 7),
            os.path.join("/tmp/ckpt", "gen007.meta.json"))
        self.assertEqual(
            train_ppo._gen_weights_path("/tmp/ckpt", 7),
            os.path.join("/tmp/ckpt", "gen007.weights.pt"))
        self.assertEqual(
            train_ppo._gen_optimizer_path("/tmp/ckpt", 7),
            os.path.join("/tmp/ckpt", "gen007.optimizer.pt"))


class ArgValidationTests(unittest.TestCase):
    """CLI 引数の拒否と設計書 §4.8 の既定値。"""

    def test_invalid_values_rejected(self):
        cases = [
            ["--episodes", "0"],
            ["--workers", "0"],
            ["--keep-latest", "0"],
            ["--generations", "-1"],
            ["--max-seconds", "-1"],
            ["--heuristic-weight", "-0.5"],
            ["--hidden"],
            ["--hidden", "0"],
            ["--action-hidden", "0"],
        ]
        for argv in cases:
            with self.assertRaises(SystemExit) as ctx:
                train_ppo._parse_args(argv)
            self.assertEqual(ctx.exception.code, 2, argv)

    def test_generations_zero_accepted_as_unlimited(self):
        """--generations 0 は無制限を意味し、既定値でも検証を通過する。"""
        for argv in ([], ["--generations", "0"]):
            args = train_ppo._parse_args(argv)
            self.assertEqual(args.generations, 0, argv)
        with self.assertRaises(SystemExit) as ctx:
            train_ppo._parse_args(["--generations", "-1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_defaults_match_design_contract(self):
        args = train_ppo._parse_args([])
        self.assertEqual(args.start_index, 2000)
        self.assertEqual(args.episodes_per_generation, 64)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.keep_latest, 5)
        # 0 は「無制限」を意味する既定値。>= 1 検証に含めると既定起動が
        # 落ちるため、この固定が矛盾の再発を防ぐ。
        self.assertEqual(args.generations, 0)
        self.assertEqual(args.max_seconds, 0.0)
        self.assertEqual(args.hidden, [256, 256])
        self.assertEqual(args.action_hidden, [128])
        self.assertEqual(args.gamma, 0.997)
        self.assertEqual(args.gae_lambda, 0.95)
        self.assertEqual(args.clip_eps, 0.2)
        self.assertEqual(args.value_coef, 0.5)
        self.assertEqual(args.entropy_coef, 0.01)
        self.assertEqual(args.lr, 3e-4)
        self.assertEqual(args.epochs, 4)
        self.assertEqual(args.minibatch_size, 256)
        self.assertEqual(args.max_grad_norm, 0.5)


class CwdIndependenceTests(unittest.TestCase):
    """csi_job.sh の runner は JOB_DIR を作業ディレクトリにして exec するため、
    カレントディレクトリ非依存で起動できること(docs/csi-jobs.md)。"""

    def test_help_runs_from_foreign_cwd_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, TRAIN_PPO_PATH, "--help"],
                cwd=tmp, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"--checkpoint-dir", result.stdout)

    def test_default_paths_are_absolute(self):
        self.assertTrue(os.path.isabs(train_ppo.REPO_ROOT))
        self.assertTrue(os.path.isabs(train_ppo.DEFAULT_CHECKPOINT_DIR))
        self.assertTrue(
            train_ppo.DEFAULT_CHECKPOINT_DIR.startswith(
                train_ppo.REPO_ROOT + os.sep))

    def test_script_inserts_repo_root_into_sys_path(self):
        self.assertIn(train_ppo.REPO_ROOT, sys.path)


class DiscardedExclusionTests(unittest.TestCase):
    """破棄局を学習データに混ぜず、件数をカウンタとしてログに残す(§4.3)。"""

    @staticmethod
    def _fake_trajectory(index, discarded=False):
        from naeco.train.env import Decision, Trajectory
        obs_dim = enc.obs_dim(2)
        action_dim = enc.action_dim()
        obs_vec = tuple([0.0] * obs_dim)
        action_vecs = tuple(tuple([0.0] * action_dim) for _ in range(2))
        decisions = (
            Decision(0, obs_vec, action_vecs, 0, 0.0, True),
            Decision(1, obs_vec, action_vecs, 0, 0.0, True),
        )
        terminal = (1.0, -1.0) if index % 2 == 0 else (-1.0, 1.0)
        return Trajectory(
            group="train", index=index, seat_variant=0,
            seats=("learner", "opponent"), decisions=decisions,
            terminal_rewards=terminal, initial_state_hash="h%d" % index,
            discarded=discarded,
            discard_reason="LegalityError" if discarded else "")

    def test_select_usable_splits_discarded(self):
        trajs = [
            self._fake_trajectory(2000, discarded=False),
            self._fake_trajectory(2001, discarded=True),
            self._fake_trajectory(2002, discarded=False),
        ]
        usable, discarded = train_ppo._select_usable(trajs)
        self.assertEqual([t.index for t in usable], [2000, 2002])
        self.assertEqual([t.index for t in discarded], [2001])

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_discarded_never_reaches_update(self):
        from naeco.train import ppo as ppo_mod
        real_rollout = train_ppo.rollout
        real_update = ppo_mod.update
        captured = {}
        rc = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(
                _base_argv(checkpoint_dir) + ["--episodes", "3"])
            try:
                def fake_rollout(specs, workers=1, cards=None,
                                 reward_config=None, randomize_seats=False,
                                 learner_index=None):
                    return [
                        self._fake_trajectory(2000),
                        self._fake_trajectory(2001, discarded=True),
                        self._fake_trajectory(2002),
                    ]

                def fake_update(net, optimizer, batch, config, seed):
                    captured["batch"] = batch
                    return {
                        "policy_loss": 0.1, "value_loss": 0.2,
                        "entropy": 0.5, "total_loss": 0.3,
                        "clip_frac": 0.0, "approx_kl": 0.01,
                    }

                train_ppo.rollout = fake_rollout
                ppo_mod.update = fake_update
                logger, handler = _capture_logger()
                rc = train_ppo.run(args, logger=logger)
            finally:
                train_ppo.rollout = real_rollout
                ppo_mod.update = real_update
        self.assertEqual(rc, 0)
        self.assertIn("batch", captured)
        # 2 usable × 2 decisions = 4 行。破棄局(2001)の決定は含まれない。
        self.assertEqual(captured["batch"]["obs"].shape[0], 4)
        self.assertIn("discarded=1", handler.text())


class LearnerSeatContractTests(unittest.TestCase):
    """設計書 14.1: train_ppo が learner の決定だけを学習データへ入れる。

    歯の狙い:
    - rollout 呼び出しが learner_index を渡さなくなる(または正準順の先頭を
      learner としなくなる)と、実ロールアウトの Trajectory に learner_seat が
      記録されず、バッチ行数が learner の決定点数の合計と一致しなくなる。
    - `_run_rollout` が randomize_seats を外すと、席ランダム化の契約
      (設計書 §4.3)に反する(スパイで固定)。
    """

    def test_run_rollout_passes_learner_index_and_seat_randomization(self):
        """_run_rollout が rollout へ learner_index=LEARNER_INDEX と
        randomize_seats=True を渡すことをスパイで固定する。"""
        captured = {}
        real = train_ppo.rollout
        try:
            def spy_rollout(specs, workers=1, cards=None,
                            reward_config=None, randomize_seats=False,
                            learner_index=None):
                captured["workers"] = workers
                captured["randomize_seats"] = randomize_seats
                captured["learner_index"] = learner_index
                return []

            train_ppo.rollout = spy_rollout
            holder = {}
            train_ppo._run_rollout(
                [("train", 2000, 0, ["learner", "opponent"])], 3, holder)
        finally:
            train_ppo.rollout = real
        self.assertEqual(captured["workers"], 3)
        self.assertTrue(captured["randomize_seats"])
        self.assertEqual(captured["learner_index"], 0)
        self.assertEqual(holder["trajectories"], [])

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_real_training_batch_contains_only_learner_decisions(self):
        """実ロールアウト + 実バッチ構築で、学習バッチが learner の決定だけで
        構成されることを固定する(設計書 14.1 のエンドツーエンド歯)。

        rollout を包んで実軌跡を捕捉し、update を差し替えてバッチを捕捉する。
        バッチ行数は全非破棄軌跡の learner 決定点数の合計に一致しなければ
        ならない(相手の決定が混入していると落ちる)。
        """
        from naeco.train import ppo as ppo_mod
        real_rollout = train_ppo.rollout
        real_update = ppo_mod.update
        captured = {}
        rc = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            try:
                def wrapping_rollout(specs, workers=1, cards=None,
                                     reward_config=None,
                                     randomize_seats=False,
                                     learner_index=None):
                    trajectories = real_rollout(
                        specs, workers=workers, cards=cards,
                        reward_config=reward_config,
                        randomize_seats=randomize_seats,
                        learner_index=learner_index)
                    captured["trajectories"] = trajectories
                    return trajectories

                def fake_update(net, optimizer, batch, config, seed):
                    captured["batch"] = batch
                    return {
                        "policy_loss": 0.1, "value_loss": 0.2,
                        "entropy": 0.5, "total_loss": 0.3,
                        "clip_frac": 0.0, "approx_kl": 0.01,
                    }

                train_ppo.rollout = wrapping_rollout
                ppo_mod.update = fake_update
                logger, handler = _capture_logger()
                rc = train_ppo.run(args, logger=logger)
            finally:
                train_ppo.rollout = real_rollout
                ppo_mod.update = real_update
        self.assertEqual(rc, 0)
        self.assertIn("trajectories", captured)
        self.assertIn("batch", captured)
        self.assertTrue(captured["trajectories"], "実ロールアウトが空")
        # 全実軌跡が learner_seat を記録し、バッチが learner の決定だけを
        # 含む(行数 == learner 決定点数の合計)。
        total_learner = 0
        for traj in captured["trajectories"]:
            self.assertIsNotNone(
                traj.learner_seat,
                "Trajectory %s/%d に learner_seat が記録されていない"
                % (traj.group, traj.index))
            self.assertIn(traj.learner_seat, (0, 1))
            if not traj.discarded:
                total_learner += sum(
                    1 for d in traj.decisions
                    if d.player_id == traj.learner_seat)
        self.assertGreater(total_learner, 0)
        self.assertEqual(
            captured["batch"]["obs"].shape[0], total_learner,
            "学習バッチに learner 以外の決定(相手の手)が混入している")


class GitCommitMetadataTests(unittest.TestCase):
    """再現メタデータ: git_commit を必ず記録する(設計書 §18.1 / PLAN.md)。

    - リポジトリの HEAD を記録し、working tree が dirty なら <sha>-dirty。
    - git が使えない環境では "unavailable" と記録し、None のままにしない。
    """

    def test_git_commit_helper_never_returns_none(self):
        commit = train_ppo._git_commit()
        self.assertIsNotNone(commit)
        self.assertIsInstance(commit, str)
        self.assertTrue(
            re.match(r"^(unavailable|[0-9a-f]{40}(-dirty)?)$", commit),
            "git_commit=%r" % commit)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_checkpoint_meta_records_git_commit(self):
        """実学習ループで保存されたメタの git_commit が None でない(歯)。

        保存経路から git_commit 記録を外すと(あるいは None のまま保存すると)
        このテストが落ちる。"""
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            with open(os.path.join(checkpoint_dir, "gen000.meta.json"),
                      "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            git_commit = meta["git_commit"]
            self.assertIsNotNone(git_commit,
                                 "git_commit を None のままにしてはならない")
            self.assertIsInstance(git_commit, str)
            self.assertTrue(
                re.match(r"^(unavailable|[0-9a-f]{40}(-dirty)?)$",
                         git_commit),
                "git_commit=%r" % git_commit)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class TrainingLoopTests(unittest.TestCase):
    """実学習ループ: 進捗ログ・チェックポイント・再開・壁時計 deadline。"""

    def test_progress_log_and_checkpoint_meta(self):
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, handler = _capture_logger()
            rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            meta_path = os.path.join(checkpoint_dir, "gen000.meta.json")
            self.assertTrue(os.path.exists(meta_path))
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta["generation"], 0)
            self.assertEqual(meta["training_steps"], 1)
            self.assertEqual(meta["next_episode_index"], 2002)
            self.assertEqual(meta["checkpoint_schema_version"], 1)
            self.assertIn("obs_encoder_sha256", meta)
            self.assertIn("action_encoder_sha256", meta)
            self.assertEqual(meta["root_seed"], 7)
            self.assertTrue(os.path.exists(
                os.path.join(checkpoint_dir, "gen000.weights.pt")))
            # 14.3: optimizer 状態もチェックポイントへ保存される。
            self.assertTrue(os.path.exists(
                os.path.join(checkpoint_dir, "gen000.optimizer.pt")))
        text = handler.text()
        for field in ("generation=", "training_steps=", "discarded=",
                      "decisions_per_second=", "policy_loss=",
                      "value_loss=", "entropy=", "approx_kl="):
            self.assertIn(field, text, field)

    def test_resume_continues_steps_and_generation(self):
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(
                _base_argv(checkpoint_dir) + ["--generations", "2"])
            logger1, unused_handler1 = _capture_logger()
            rc1 = train_ppo.run(args, logger=logger1)
            self.assertEqual(rc1, 0)
            with open(os.path.join(checkpoint_dir, "gen001.meta.json"),
                      "r", encoding="utf-8") as fh:
                meta1 = json.load(fh)
            self.assertEqual(meta1["generation"], 1)
            self.assertEqual(meta1["training_steps"], 2)
            self.assertEqual(meta1["next_episode_index"], 2004)

            # 中断からの再開: 同じ checkpoint_dir で続きから走る。
            train_ppo._TERMINATE["flag"] = False
            logger2, handler2 = _capture_logger()
            rc2 = train_ppo.run(args, logger=logger2)
            self.assertEqual(rc2, 0)
            with open(os.path.join(checkpoint_dir, "gen003.meta.json"),
                      "r", encoding="utf-8") as fh:
                meta3 = json.load(fh)
            self.assertEqual(meta3["generation"], 3)
            self.assertEqual(meta3["training_steps"], 4)
            # 再開後も index は 2004 から続く(2000 に戻らない = seed 再利用なし)。
            self.assertEqual(meta3["next_episode_index"], 2008)
            self.assertIn("resumed from generation", handler2.text())

    def test_max_seconds_is_absolute_wall_clock_deadline(self):
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(
                _base_argv(checkpoint_dir)
                + ["--generations", "0", "--max-seconds", "1.0"])
            logger, handler = _capture_logger()
            started = time.monotonic()

            def fake_exit(code):
                raise SystemExit(code)

            # ロールアウト中に deadline へ達した経路は os._exit(0) で終わる
            # (Phase 4 残留リスク #5 対策)。in-process では os._exit がテス
            # トランナーごと落とすため、SystemExit へ差し替えて終了コード 0
            # として扱う。世代境界の経路は run() が 0 を返す。
            with mock.patch.object(os, "_exit", fake_exit):
                try:
                    rc = train_ppo.run(args, logger=logger)
                except SystemExit as exc:
                    self.assertEqual(exc.code, 0)
                    rc = 0
            elapsed = time.monotonic() - started
            self.assertEqual(rc, 0)
            self.assertLess(elapsed, 120.0)
            gens = train_ppo._list_generations(checkpoint_dir)
            self.assertTrue(gens, "at least one generation must be saved")
        self.assertIn("wall-clock deadline", handler.text())


class RolloutInterruptTests(unittest.TestCase):
    """_rollout_or_terminate の中断: SIGTERM フラグと壁時計 deadline(残留リスク #5)。

    ロールアウトを unittest.mock のダミー(開始イベントを立てて解放待ちする
    ブロックロールアウト)へ置き換え、(a) deadline 到達時に完了を待たず None
    を返すこと、(b) SIGTERM フラグで None を返すこと、(c) deadline=None の
    とき完了を待って結果を返すことを固定する。torch 不要。
    """

    def test_deadline_returns_none_without_waiting_for_rollout(self):
        """(a) deadline 到達時、ブロックする rollout の完了を待たず None を返す。"""
        started = threading.Event()
        release = threading.Event()
        completed = []

        def blocking_rollout(specs, workers=1, cards=None,
                             reward_config=None, randomize_seats=False,
                             learner_index=None):
            started.set()
            release.wait(timeout=60.0)
            completed.append(True)
            return [("trajectory", 0)]

        logger, unused_handler = _capture_logger()
        try:
            with mock.patch.object(train_ppo, "rollout", blocking_rollout):
                deadline = time.monotonic() + 0.2
                result = train_ppo._rollout_or_terminate([], 1, logger, deadline)
            # 戻った時点では rollout は解放待ちのまま(completed は空でなければ
            # ならない)。解放前に検証する。
            self.assertIsNone(result)
            self.assertFalse(completed, "rollout の完了を待ってはならない")
            self.assertTrue(started.is_set(), "rollout が開始されていない")
        finally:
            release.set()

    def test_sigterm_flag_returns_none(self):
        """(b) SIGTERM フラグで None を返す(既存挙動の不変)。"""
        started = threading.Event()
        release = threading.Event()

        def blocking_rollout(specs, workers=1, cards=None,
                             reward_config=None, randomize_seats=False,
                             learner_index=None):
            started.set()
            release.wait(timeout=60.0)
            return [("trajectory", 0)]

        train_ppo._TERMINATE["flag"] = True
        logger, unused_handler = _capture_logger()
        try:
            with mock.patch.object(train_ppo, "rollout", blocking_rollout):
                result = train_ppo._rollout_or_terminate([], 1, logger, None)
        finally:
            train_ppo._TERMINATE["flag"] = False
            release.set()
        self.assertIsNone(result)
        self.assertTrue(started.is_set(), "rollout が開始されていない")

    def test_no_deadline_waits_for_completion(self):
        """(c) deadline=None(無制限)は rollout 完了を待って結果を返す。"""
        completed = []

        def quick_rollout(specs, workers=1, cards=None,
                          reward_config=None, randomize_seats=False,
                          learner_index=None):
            completed.append(True)
            return [("trajectory", 0)]

        logger, unused_handler = _capture_logger()
        with mock.patch.object(train_ppo, "rollout", quick_rollout):
            result = train_ppo._rollout_or_terminate([], 1, logger, None)
        self.assertEqual(result, [("trajectory", 0)])
        self.assertEqual(completed, [True])

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_run_stops_at_deadline_during_rollout(self):
        """統合: --max-seconds を極小にすると、ロールアウト中に deadline へ
        達し、完了を待たずチェックポイントを保存して終了コード 0 で終わる。

        ブロックするダミーロールアウトへ置き換え、os._exit を SystemExit へ
        差し替えて in-process で走らせる(サブプロセスでは完了待ち経路との
        競合で「ロールアウト中」を決定的に再現できない)。os._exit は
        train_ppo と同一の os モジュールオブジェクトを参照するため、
        mock.patch.object(os, "_exit", ...) で置き換わる。
        """
        train_ppo._TERMINATE["flag"] = False
        started = threading.Event()
        release = threading.Event()

        def blocking_rollout(specs, workers=1, cards=None,
                             reward_config=None, randomize_seats=False,
                             learner_index=None):
            started.set()
            release.wait(timeout=60.0)
            return []

        rc = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(
                _base_argv(checkpoint_dir)
                + ["--generations", "0", "--max-seconds", "0.5"])
            logger, handler = _capture_logger()
            try:
                def fake_exit(code):
                    raise SystemExit(code)

                with mock.patch.object(train_ppo, "rollout", blocking_rollout), \
                        mock.patch.object(os, "_exit", fake_exit):
                    with self.assertRaises(SystemExit) as ctx:
                        train_ppo.run(args, logger=logger)
                    rc = ctx.exception.code
            finally:
                release.set()
                train_ppo._TERMINATE["flag"] = False
            # チェックポイントの存在検証は with スコープが生きている間に行う。
            # スコープ外(メソッド直下)で行うと TemporaryDirectory.__exit__ に
            # より tmp が削除済みのため os.path.exists が常に False になり、
            # torch が在る環境で必ず FAIL する(過去のテストバグ)。このテスト
            # は torch が在る環境での pass を想定しており、torch が無い環境で
            # はこのメソッドの skipUnless で skip される。
            self.assertTrue(
                os.path.exists(os.path.join(checkpoint_dir, "gen000.meta.json")),
                "deadline 中断時にチェックポイントが保存されていない")
        self.assertEqual(rc, 0)
        self.assertTrue(started.is_set(), "rollout が開始されていない")
        self.assertIn("wall-clock deadline during rollout", handler.text())


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class OptimizerResumeTests(unittest.TestCase):
    """設計書 14.3 / 14.4: optimizer 状態の保存・読戻しと並列一致(歯)。

    - 連続 N 世代と「途中で保存して別プロセスで再開した N 世代」で、
      重み・optimizer 状態・次 episode index が一致する。
    - optimizer 状態の無い旧チェックポイントからの再開は警告のうえ新規作成
      を許す(既存の成果物を読めなくしない)。
    - 実 PolicyValueNet の学習を workers=1 と workers=4 で走らせ、最終
      チェックポイントまで完全一致する(ロールアウト層に留まらない歯)。
    """

    @staticmethod
    def _run_cli(cwd, checkpoint_dir, generations, workers=1, timeout=600):
        """train_ppo.py をサブプロセスで走らせ、rc==0 と stdout を返す。

        _base_argv に含まれる --workers 1 / --generations 1 は argparse の
        「後勝ち」で末尾の指定に置き換わる。
        """
        argv = _base_argv(checkpoint_dir) + [
            "--generations", str(generations), "--workers", str(workers)]
        result = subprocess.run(
            [sys.executable, TRAIN_PPO_PATH] + argv,
            cwd=cwd, capture_output=True, timeout=timeout)
        if result.returncode != 0:
            raise AssertionError(
                "train_ppo.py exited with %d:\n%s\n%s"
                % (result.returncode,
                   result.stdout.decode("utf-8", errors="replace"),
                   result.stderr.decode("utf-8", errors="replace")))
        return result.stdout.decode("utf-8", errors="replace")

    def _assert_saved_state_equal(self, left, right, label):
        """torch.save した dict(重み / optimizer 状態)を再帰的に比較する。"""
        if isinstance(left, dict) and isinstance(right, dict):
            self.assertEqual(sorted(left), sorted(right), label)
            for key in left:
                # Adam の state_dict() は state 配下に整数のパラメータ index
                # をキーとして持つため、key を文字列化してから連結する。
                self._assert_saved_state_equal(left[key], right[key],
                                               label + "/" + str(key))
        elif isinstance(left, list) and isinstance(right, list):
            self.assertEqual(len(left), len(right), label)
            for k, (a, b) in enumerate(zip(left, right)):
                self._assert_saved_state_equal(a, b, "%s[%d]" % (label, k))
        elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            self.assertTrue(torch.equal(left, right), label)
        else:
            self.assertEqual(left, right, label)

    def test_checkpoint_saves_optimizer_state(self):
        """重みファイルは従来どおり裸の state_dict(eval_agent 互換)で、
        optimizer 状態は別ファイルに param_groups と state を持つ。"""
        train_ppo._TERMINATE["flag"] = False
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            weights = torch.load(
                os.path.join(checkpoint_dir, "gen000.weights.pt"),
                map_location="cpu")
            optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "gen000.optimizer.pt"),
                map_location="cpu")
            # 重みファイルは裸の state_dict のまま(eval_agent / promote 互換)。
            self.assertIn("obs_mlp.0.weight", weights)
            # optimizer 状態: param_groups と state(step / exp_avg / exp_avg_sq)。
            self.assertEqual(len(optimizer_state["param_groups"]), 1)
            self.assertTrue(optimizer_state["state"], "update 後は空でない")
            for idx, state in optimizer_state["state"].items():
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertIn(key, state, "param %s" % idx)
                    self.assertTrue(torch.isfinite(state[key]).all(),
                                    (idx, key))

    def test_resume_matches_continuous_run_exactly(self):
        """連続 N 世代と「途中で保存して別プロセスで再開した N 世代」で、
        重み・optimizer 状態・次 episode index が一致する(歯)。

        optimizer.state_dict() を保存・読戻ししない実装へ戻すと、再開側の
        Adam の momentum / 二次モーメントが新規になり exp_avg / exp_avg_sq /
        重みが連続実行と一致せず落ちる。
        """
        with tempfile.TemporaryDirectory() as tmp:
            cont_dir = os.path.join(tmp, "cont")
            res_dir = os.path.join(tmp, "res")
            self._run_cli(tmp, cont_dir, generations=2)   # 連続 2 世代
            self._run_cli(tmp, res_dir, generations=1)    # 途中で保存
            self._run_cli(tmp, res_dir, generations=1)    # 別プロセスで再開
            for suffix in ("gen001.weights.pt", "gen001.optimizer.pt"):
                left = torch.load(os.path.join(cont_dir, suffix),
                                  map_location="cpu")
                right = torch.load(os.path.join(res_dir, suffix),
                                   map_location="cpu")
                self._assert_saved_state_equal(left, right, suffix)
            with open(os.path.join(cont_dir, "gen001.meta.json"),
                      "r", encoding="utf-8") as fh:
                cont_meta = json.load(fh)
            with open(os.path.join(res_dir, "gen001.meta.json"),
                      "r", encoding="utf-8") as fh:
                res_meta = json.load(fh)
            for key in ("generation", "training_steps",
                        "next_episode_index"):
                self.assertEqual(cont_meta[key], res_meta[key], key)

    def test_training_matches_across_worker_counts(self):
        """実 PolicyValueNet の学習を workers=1 と workers=4 で走らせ、最終
        チェックポイント(重み・optimizer 状態・次 episode index)まで一致する。

        ロールアウト層の一致検査(naeco/train/env.py の workers 非依存)に
        留まらず、学習ループ全体(プール抽選・update・保存)が worker 数に
        依存しないことを固定する。
        """
        with tempfile.TemporaryDirectory() as tmp:
            dirs = {}
            for workers in (1, 4):
                dirs[workers] = os.path.join(tmp, "w%d" % workers)
                self._run_cli(tmp, dirs[workers], generations=2,
                              workers=workers)
            for suffix in ("gen001.weights.pt", "gen001.optimizer.pt"):
                left = torch.load(os.path.join(dirs[1], suffix),
                                  map_location="cpu")
                right = torch.load(os.path.join(dirs[4], suffix),
                                   map_location="cpu")
                self._assert_saved_state_equal(left, right, suffix)
            with open(os.path.join(dirs[1], "gen001.meta.json"),
                      "r", encoding="utf-8") as fh:
                meta1 = json.load(fh)
            with open(os.path.join(dirs[4], "gen001.meta.json"),
                      "r", encoding="utf-8") as fh:
                meta4 = json.load(fh)
            for key in ("generation", "training_steps",
                        "next_episode_index"):
                self.assertEqual(meta1[key], meta4[key], key)

    def test_resume_without_optimizer_state_warns_and_continues(self):
        """optimizer 状態の無い旧チェックポイントからの再開は警告を出したうえ
        で新規作成を許す(既存の成果物を読めなくしない)。"""
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            os.makedirs(checkpoint_dir)
            torch.manual_seed(3)
            net = PolicyValueNet(
                enc.obs_dim(2), enc.action_dim(),
                hidden=(8, 8), action_hidden=(4,))
            meta = checkpoint_mod.current_meta(
                2, net.arch_dict(),
                extra={"generation": 0, "training_steps": 0,
                       "next_episode_index": 2000, "start_index": 2000})
            checkpoint_mod.save(
                os.path.join(checkpoint_dir, "gen000.meta.json"), meta)
            # 旧形式: 重みは裸の state_dict だけで optimizer ファイルが無い。
            torch.save(net.state_dict(),
                       os.path.join(checkpoint_dir, "gen000.weights.pt"))
            self.assertFalse(os.path.exists(
                os.path.join(checkpoint_dir, "gen000.optimizer.pt")))
            train_ppo._TERMINATE["flag"] = False
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, handler = _capture_logger()
            rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            # 警告(levelno >= WARNING)として記録され、内容に optimizer state
            # が含まれる(_CaptureHandler の format は %(message)s のため、
            # レベル名ではなく record.levelno で判定する)。
            self.assertTrue(any(
                record.levelno >= logging.WARNING
                and "optimizer state" in record.getMessage()
                for record in handler.records),
                "旧チェックポイント再開の警告が記録されていない")
            # 新規作成で学習は続行され、次の世代は optimizer 状態付きで保存される。
            self.assertTrue(os.path.exists(
                os.path.join(checkpoint_dir, "gen001.weights.pt")))
            self.assertTrue(os.path.exists(
                os.path.join(checkpoint_dir, "gen001.optimizer.pt")))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class SigtermTests(unittest.TestCase):
    """SIGTERM でチェックポイントを保存してから終了する(受け入れ項目1)。"""

    def test_sigterm_saves_checkpoint_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            log_path = os.path.join(tmp, "train.log")
            argv = _base_argv(checkpoint_dir) + [
                "--generations", "0", "--max-seconds", "0",
            ]
            with open(log_path, "wb") as out:
                proc = subprocess.Popen(
                    [sys.executable, TRAIN_PPO_PATH] + argv,
                    cwd=tmp, stdout=out, stderr=subprocess.STDOUT)
                try:
                    # gen000 の保存を壁時計 deadline で待つ(反復回数で書かない)。
                    deadline = time.monotonic() + 180.0
                    first_meta = os.path.join(
                        checkpoint_dir, "gen000.meta.json")
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            break
                        if os.path.exists(first_meta):
                            break
                        time.sleep(0.5)
                    self.assertIsNone(
                        proc.poll(),
                        "training process exited before gen000 was saved")
                    os.kill(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=180.0)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=30.0)
            self.assertEqual(proc.returncode, 0)
            second_meta = os.path.join(checkpoint_dir, "gen001.meta.json")
            self.assertTrue(
                os.path.exists(second_meta),
                "SIGTERM でチェックポイント(gen001)が保存されていない")
            with open(second_meta, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertGreaterEqual(meta["generation"], 1)
            with open(log_path, "r", encoding="utf-8") as fh:
                log_text = fh.read()
            self.assertIn("SIGTERM", log_text)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class CheckpointLockTests(unittest.TestCase):
    """Phase 4: ロック取得失敗時に run() が非ゼロを返し、チェックポイントを
    書かない(受け入れ項目)。torch が在る環境で走る統合テスト。"""

    def test_run_fails_fast_when_lock_is_held(self):
        from naeco.train import checkpoint as checkpoint_mod
        if checkpoint_mod.fcntl is None:
            self.skipTest("fcntl の無い環境ではロックは no-op")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "ckpt")
            os.makedirs(checkpoint_dir)
            holder = checkpoint_mod.CheckpointLock(checkpoint_dir)
            holder.acquire()
            try:
                args = train_ppo._parse_args(_base_argv(checkpoint_dir))
                logger, handler = _capture_logger()
                rc = train_ppo.run(args, logger=logger)
                self.assertNotEqual(rc, 0)
                # チェックポイントは一切書き込まれない(ロックファイルのみ)。
                self.assertEqual(
                    train_ppo._list_generations(checkpoint_dir), [])
                self.assertEqual(
                    os.listdir(checkpoint_dir),
                    [checkpoint_mod.LOCK_FILENAME])
                # 明確なログが出る(ロックファイル名を含む)。
                text = handler.text()
                self.assertIn("exclusive lock", text)
                self.assertIn(checkpoint_mod.LOCK_FILENAME, text)
            finally:
                holder.release()
            # 解放後は同じ checkpoint_dir で run() が正常に走れる
            # (残置ロックが後続の再開ジョブを妨げない)。
            train_ppo._TERMINATE["flag"] = False
            args = train_ppo._parse_args(_base_argv(checkpoint_dir))
            logger, unused_handler = _capture_logger()
            rc = train_ppo.run(args, logger=logger)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(
                os.path.join(checkpoint_dir, "gen000.meta.json")))


if __name__ == "__main__":
    unittest.main()
