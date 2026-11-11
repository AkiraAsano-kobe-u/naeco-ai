"""B6 評価CLI scripts/eval_agent.py と昇格・退行判定 scripts/promote.py のテスト。

受け入れ項目を「壊したときに落ちる」形で固定する:
- harness の cluster_bootstrap / summarize_pairs / passes_threshold を再利用
  (統計処理の新規実装が無いこと: 配線を記録ラッパで実測)。
- eval_agent は random / heuristic / checkpoint の3種のエージェントを指定
  できる。
- seat_variant 0 / 1 の2局で1ペアとし、ペアの2局が同一配札になる。
- --execute を付けない限り実行しない。
- 1局1行の対応表 JSONL に seed / 席 / 初期配札 / 版を記録する。
- 合否は CI 下限による厳密不等号 >(>= にすると落ちる歯)。
- promote は昇格2条件(直接対戦 CI下限 > 0.50 かつ ベンチ −3pt 以上)と
  退行(5pt 超低下 → ロールバック)を §7 どおり実装する。
- 判定条件を実行前に読み込んで sha256 で固定し、実行中の変更を検出する。
- 判定は dev seed のみ(train / accept は拒否)。
- checkpoint.load へ必ず expected=expected_meta(...)(専用の不変型)を渡す
  (ソース走査 + 実行時。設計書 §15.1)。model_arch は読んだ arch で
  PolicyValueNet を構築し、重み state_dict のキー集合と各テンソル形状を
  照合してから使う(設計書 §15.2)。num_players は評価プロトコル側の値(2)を
  渡し保存メタから読まない。
- model_arch は _read_model_arch がキー集合の完全一致(余分も欠落も拒否)と
  型(obs_dim / action_dim は type(x) is int、hidden / action_hidden は
  list で全要素 int)を厳格に検証し(設計書 §17.1)、int(...) による正規化を
  しない。型偽装・余分なキー・キー欠落は評価経路で
  IncompatibleCheckpoint になる。
- worker 数を変えても評価結果が完全一致する。
- 4つのスレッド環境変数を 1 に固定する。
- torch を直接 import しない(ソース走査)。torch 依存テストは無い環境で skip。
- Python 3.9 互換。
"""
import copy
import json
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:
    torch = None

HAS_TORCH = torch is not None

import eval_agent  # noqa: E402
import promote  # noqa: E402
from naeco.engine.observe import OBS_VERSION  # noqa: E402
from naeco.engine.state import API_VERSION, RULES_VERSION  # noqa: E402
from naeco.train import encoding as enc  # noqa: E402
from naeco.train import reward as reward_mod  # noqa: E402

EVAL_AGENT_PATH = os.path.join(ROOT, "scripts", "eval_agent.py")
PROMOTE_PATH = os.path.join(ROOT, "scripts", "promote.py")
THREAD_ENV_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

CONDITIONS_DEFAULT = {
    "direct": {"group": "dev", "start_index": 7800, "pairs": 2},
    "benchmark": {"group": "dev", "start_index": 9000, "pairs": 2},
    "promote_threshold": 0.50,
    "promote_tolerance": 0.03,
    "regression_tolerance": 0.05,
    "benchmark_entries": [
        {"name": "heuristic", "kind": "heuristic"},
    ],
    "champion_benchmark": {"heuristic": 0.60},
}

# 正式条件(docs/eval-seeds.md Phase 3 予約 / 設計書 §18.2)。
FORMAL_CONDITIONS = {
    "direct": {"group": "dev", "start_index": 7800, "pairs": 800},
    "benchmark": {"group": "dev", "start_index": 9000, "pairs": 800},
    "promote_threshold": 0.50,
    "promote_tolerance": 0.03,
    "regression_tolerance": 0.05,
    "benchmark_entries": [
        {"name": "heuristic", "kind": "heuristic"},
    ],
    "champion_benchmark": {"heuristic": 0.60},
}


def _write_formal_conditions(path):
    """正式条件 JSON を書く(テスト用)。"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(FORMAL_CONDITIONS, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")


def _write_ledger(path, consumed):
    """テスト用の消費済み seed 台帳を書く。

    実物の docs/eval-seeds.md を読ませると、台帳が進むたびにテストが壊れる。
    テストは**台帳の解釈**を固定したいので、消費済み帯を注入する。
    """
    body = " ".join("%d-%d" % (lo, hi) for lo, hi in consumed)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# test ledger\n\n")
        fh.write("<!-- CONSUMED-DEV-BANDS:BEGIN -->\n")
        fh.write(body + "\n")
        fh.write("<!-- CONSUMED-DEV-BANDS:END -->\n")


def _write_formal_conditions_with(path, overrides):
    """正式条件を土台に、指定キーだけ差し替えて書く(テスト用)。

    `_write_conditions` は pairs=2 の**非正式**な土台を使うので、
    「1箇所だけ変えた正式条件」を作る用途には使えない。
    """
    payload = copy.deepcopy(FORMAL_CONDITIONS)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")


def _write_dummy_checkpoint(meta_path, weights_bytes=b"dummy-weights"):
    """凍結用のダミーのチェックポイント(meta + weights)を書く。

    評価はスタブ化されるため内容は使われないが、promote のモデル凍結
    (meta / weights の SHA-256)は実ファイルを読むため必要。
    """
    parent = os.path.dirname(os.path.abspath(meta_path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"dummy": True}, fh)
        fh.write("\n")
    weights_path = eval_agent.checkpoint_weights_path(meta_path)
    with open(weights_path, "wb") as fh:
        fh.write(weights_bytes)


def _write_conditions(path, overrides=None):
    """テスト用の判定条件 JSON を書く。overrides はキー単位で差し替える。

    非空の dict 値は既定値へマージし、空 dict はキーごと置き換える
    (マージだと既定値が残り、空 champion_benchmark が検証経路に届かないため)。
    """
    conditions = json.loads(json.dumps(CONDITIONS_DEFAULT))
    if overrides:
        for key, value in overrides.items():
            if value and isinstance(value, dict) and \
                    isinstance(conditions.get(key), dict):
                conditions[key].update(value)
            else:
                conditions[key] = value
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(conditions, fh, sort_keys=True, ensure_ascii=True)
        fh.write("\n")
    return conditions


def _fake_result(point, ci_low, pairs, group="dev", start_index=0,
                 left="left", right="right"):
    """テスト用の評価結果。統計経路は通さない(直接組み立てる)。"""
    games = pairs * 2
    wins = int(round(point * games))
    losses = games - wins
    return eval_agent.AgentEvaluationResult(
        left=left, right=right, group=group, start_index=start_index,
        pairs=pairs, games=games, wins=wins, draws=0, losses=losses,
        point_estimate=point, ci_low=ci_low, ci_high=ci_low,
        bootstrap_samples=10, bootstrap_seed=1, pair_results=(),
        initial_state_hashes=(), game_records=())


class EvalAgentCliTests(unittest.TestCase):
    """eval_agent: execute 規律・3種のエージェント指定・対応表の記録。"""

    def test_no_execute_does_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = os.path.join(tmp, "rec.jsonl")
            code = eval_agent.main([
                "heuristic", "random", "--group", "train",
                "--start-index", "100", "--pairs", "4",
                "--records", records])
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(records),
                             "--execute なしで実行してはならない")

    def test_execute_writes_one_row_per_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = os.path.join(tmp, "rec.jsonl")
            code = eval_agent.main([
                "heuristic", "random", "--group", "train",
                "--start-index", "100", "--pairs", "2",
                "--records", records, "--execute"])
            self.assertIn(code, (0, 1))
            with open(records, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            self.assertEqual(len(rows), 4)  # 2ペア × 2局
            for row in rows:
                # 席と seed の対応が保存されている。
                self.assertIn(row["seat_variant"], (0, 1))
                self.assertEqual(row["target_seat"], row["seat_variant"])
                self.assertEqual(row["opponent_seat"],
                                 1 - row["seat_variant"])
                self.assertIn(row["result"], (0.0, 0.5, 1.0))
                self.assertIsInstance(row["env_seed"], int)
                self.assertIsInstance(row["setup_seed"], int)
                self.assertIsInstance(row["policy_seed"], int)
                self.assertIsInstance(row["opponent_seed"], int)
                self.assertIsInstance(row["start_player"], int)
                # 初期配札が明示的に保存されている。
                self.assertTrue(row["initial_deck"])
                self.assertEqual(len(row["initial_hands"]), 2)
                self.assertEqual(len(row["initial_money"]), 2)
                self.assertEqual(len(row["initial_state_hash"]), 64)
                self.assertEqual(len(row["cards_sha256"]), 64)
                # 版が記録されている。
                self.assertEqual(row["rules_version"], RULES_VERSION)
                self.assertEqual(row["api_version"], API_VERSION)
                self.assertEqual(row["obs_version"], OBS_VERSION)
                self.assertEqual(row["obs_encoder_version"],
                                 enc.OBS_ENCODER_VERSION)
                self.assertEqual(row["action_encoder_version"],
                                 enc.ACTION_ENCODER_VERSION)
                self.assertEqual(row["reward_version"],
                                 reward_mod.REWARD_VERSION)
                self.assertEqual(row["left"], "heuristic")
                self.assertEqual(row["right"], "random")
            # ペアの2局は同一配札(初期山札・初期手札・env/setup seed 一致)。
            by_index = {}
            for row in rows:
                by_index.setdefault(row["index"], []).append(row)
            self.assertEqual(len(by_index), 2)
            for pair in by_index.values():
                self.assertEqual(len(pair), 2)
                self.assertEqual(pair[0]["initial_state_hash"],
                                 pair[1]["initial_state_hash"])
                self.assertEqual(pair[0]["initial_deck"],
                                 pair[1]["initial_deck"])
                self.assertEqual(pair[0]["initial_hands"],
                                 pair[1]["initial_hands"])
                self.assertEqual(pair[0]["env_seed"], pair[1]["env_seed"])
                self.assertEqual(pair[0]["setup_seed"], pair[1]["setup_seed"])
                self.assertNotEqual(pair[0]["policy_seed"],
                                    pair[1]["policy_seed"])

    def test_refuses_existing_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = os.path.join(tmp, "rec.jsonl")
            with open(records, "w", encoding="utf-8") as fh:
                fh.write("x\n")
            with self.assertRaises(SystemExit):
                eval_agent.main([
                    "heuristic", "random", "--group", "train",
                    "--start-index", "100", "--pairs", "1",
                    "--records", records, "--execute"])

    def test_unknown_agent_spec_rejected(self):
        with self.assertRaises(SystemExit):
            eval_agent.main(["mystery", "random", "--pairs", "1"])

    def test_checkpoint_spec_requires_meta_path(self):
        with self.assertRaises(SystemExit):
            eval_agent.main(["checkpoint:", "random", "--pairs", "1"])

    def test_three_agent_kinds_accepted(self):
        # 3種の指定が parse できる(execute なし → 実行しない)。
        for spec in ("random", "heuristic", "checkpoint:x.meta.json"):
            kind, path = eval_agent.parse_agent_spec(spec)
            self.assertEqual(kind, spec.partition(":")[0])
        with self.assertRaises(ValueError):
            eval_agent.parse_agent_spec("pimc")
        with self.assertRaises(ValueError):
            eval_agent.parse_agent_spec("heuristic:x")

    def test_checkpoint_weights_path_derivation(self):
        self.assertEqual(
            eval_agent.checkpoint_weights_path("a/gen000.meta.json"),
            "a/gen000.weights.pt")
        with self.assertRaises(ValueError):
            eval_agent.checkpoint_weights_path("a/gen000.json")


class HarnessReuseTests(unittest.TestCase):
    """統計処理は harness 経由で行い、新規実装しない(配線の実測)。"""

    def test_statistics_are_routed_through_harness(self):
        calls = {"summarize": 0, "bootstrap": 0}
        real_summarize = eval_agent.summarize_pairs
        real_bootstrap = eval_agent.cluster_bootstrap

        def recording_summarize(pair_games):
            calls["summarize"] += 1
            return real_summarize(pair_games)

        def recording_bootstrap(pair_results, samples, seed):
            calls["bootstrap"] += 1
            return real_bootstrap(pair_results, samples, seed)

        eval_agent.summarize_pairs = recording_summarize
        eval_agent.cluster_bootstrap = recording_bootstrap
        try:
            result = eval_agent.run_evaluation(
                "heuristic", "random", "train", 400, 1, workers=1)
        finally:
            eval_agent.summarize_pairs = real_summarize
            eval_agent.cluster_bootstrap = real_bootstrap
        self.assertEqual(calls["summarize"], 1)
        self.assertEqual(calls["bootstrap"], 1)
        self.assertEqual(result.pairs, 1)
        self.assertEqual(result.games, 2)
        self.assertEqual(len(result.game_records), 2)

    def test_verdict_uses_passes_threshold(self):
        called = []
        real = eval_agent.passes_threshold

        def recording(result, threshold):
            called.append((result, threshold))
            return real(result, threshold)

        eval_agent.passes_threshold = recording
        try:
            code = eval_agent.main([
                "heuristic", "random", "--group", "train",
                "--start-index", "500", "--pairs", "1",
                "--threshold", "0.50", "--execute"])
        finally:
            eval_agent.passes_threshold = real
        self.assertIn(code, (0, 1))
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0][1], 0.50)


class ParallelDeterminismTests(unittest.TestCase):
    """worker 数を変えても評価結果が完全一致する(設計書 §6.3)。"""

    def test_worker_count_does_not_change_results(self):
        kwargs = dict(left="heuristic", right="random", group="train",
                      start_index=600, pairs=2)
        one = eval_agent.run_evaluation(workers=1, **kwargs)
        many = eval_agent.run_evaluation(workers=2, **kwargs)
        self.assertEqual(one.game_records, many.game_records)
        self.assertEqual(one.pair_results, many.pair_results)
        self.assertEqual(one.initial_state_hashes,
                         many.initial_state_hashes)
        self.assertEqual(one.point_estimate, many.point_estimate)
        self.assertEqual((one.ci_low, one.ci_high),
                         (many.ci_low, many.ci_high))
        self.assertEqual((one.wins, one.draws, one.losses),
                         (many.wins, many.draws, many.losses))


class ThreadEnvTests(unittest.TestCase):
    """4つのスレッド環境変数を 1 に固定する(設計書 §8.4)。"""

    def test_thread_env_vars_pinned_to_one(self):
        saved = dict((name, os.environ.get(name)) for name in THREAD_ENV_NAMES)
        try:
            eval_agent._fix_thread_env()
            for name in THREAD_ENV_NAMES:
                self.assertEqual(os.environ.get(name), "1", name)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class CheckpointContractTests(unittest.TestCase):
    """§13.2 / §15.1: checkpoint.load へ必ず expected=expected_meta(...) を渡す。
    設計書 §15.2: model_arch は重み state_dict とのキー集合・形状照合で
    検証され、偽装は拒否される。"""

    @staticmethod
    def _call_text(src, open_paren_index):
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

    @staticmethod
    def _break(value):
        """A 表キーを確実に別の値へ壊す。"""
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

    def test_every_load_call_passes_expected(self):
        for path in (EVAL_AGENT_PATH, PROMOTE_PATH):
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            loads = list(re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\.load\s*\(",
                                     src))
            # promote.py はチェックポイント読込を eval_agent へ委譲するため
            # load 呼び出し自体が無くてもよい(空でないことを要求するのは
            # 読込経路を持つ eval_agent.py だけ)。
            if not loads and path == EVAL_AGENT_PATH:
                self.fail("%s に load 呼び出しが無い" % path)
            for match in loads:
                receiver = match.group(1)
                if receiver in ("json", "torch", "torch_module"):
                    continue  # メタ先読み / 重み読込。互換性照合の対象ではない
                call = self._call_text(src, match.end() - 1)
                self.assertIn(
                    "expected=", call,
                    "checkpoint.load には expected= が必須 (%s): %s"
                    % (path, call))
                self.assertIn(
                    "expected_meta(", call[call.index("expected="):],
                    "expected= には expected_meta(...) を渡す (%s): %s"
                    % (path, call))

    def test_no_load_call_uses_current_meta_as_expected(self):
        """旧形式(expected=current_meta(...) の dict)は存在しない(§15.1)。"""
        for path in (EVAL_AGENT_PATH, PROMOTE_PATH):
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            self.assertNotIn("expected=checkpoint_mod.current_meta(", src,
                             path)
            self.assertNotIn("expected=checkpoint.current_meta(", src, path)

    def test_eval_agent_num_players_is_protocol_constant(self):
        """評価経路の num_players は評価プロトコル側の値(2)を使い、保存メタ
        から読まない(設計書 §15.2)。"""
        self.assertEqual(eval_agent.NUM_PLAYERS, 2)
        with open(EVAL_AGENT_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        for needle in ('get("num_players")', '["num_players"]',
                       "['num_players']"):
            self.assertNotIn(needle, src)

    def test_no_direct_torch_import(self):
        for path in (EVAL_AGENT_PATH, PROMOTE_PATH):
            with open(path, "r", encoding="utf-8") as fh:
                src = fh.read()
            self.assertNotIn("import torch", src, path)
            self.assertNotIn("from torch", src, path)

    def test_checkpoint_incompatible_meta_rejected_without_torch(self):
        """A表のキーを壊すと、net 構築より前に IncompatibleCheckpoint。"""
        from naeco.train import checkpoint as checkpoint_mod
        arch = {"obs_dim": 16, "action_dim": 8,
                "hidden": [4], "action_hidden": [2]}
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            weights_path = os.path.join(tmp, "gen000.weights.pt")
            checkpoint_mod.save(meta_path, checkpoint_mod.current_meta(2, arch))
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            raw["rules_version"] = "0.0.0"
            with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(raw, fh, sort_keys=True, ensure_ascii=True)
                fh.write("\n")
            with open(weights_path, "wb") as fh:
                fh.write(b"dummy")
            with self.assertRaises(checkpoint_mod.IncompatibleCheckpoint):
                eval_agent._make_checkpoint_agent(meta_path, seed=0)

    def test_checkpoint_missing_weights_rejected_without_torch(self):
        from naeco.train import checkpoint as checkpoint_mod
        arch = {"obs_dim": 16, "action_dim": 8,
                "hidden": [4], "action_hidden": [2]}
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            checkpoint_mod.save(meta_path, checkpoint_mod.current_meta(2, arch))
            with self.assertRaises(ValueError):
                eval_agent._make_checkpoint_agent(meta_path, seed=0)

    def test_checkpoint_meta_path_must_end_with_meta_json(self):
        with self.assertRaises(ValueError):
            eval_agent._make_checkpoint_agent("/tmp/foo.json", seed=0)

    def test_each_forged_a_key_rejected_on_eval_path(self):
        """A 表13キー(model_arch 以外)を1つずつ偽装した保存メタは、評価経路
        (_make_checkpoint_agent)でも IncompatibleCheckpoint で拒否される。

        model_arch の偽装は重みとの形状照合で拒否されるため torch が必要
        (test_forged_model_arch_rejected_by_weight_shape_check で固定)。
        """
        from naeco.train import checkpoint as checkpoint_mod
        arch = {"obs_dim": 16, "action_dim": 8,
                "hidden": [4], "action_hidden": [2]}
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            weights_path = os.path.join(tmp, "gen000.weights.pt")
            checkpoint_mod.save(
                meta_path, checkpoint_mod.current_meta(2, arch))
            with open(weights_path, "wb") as fh:
                fh.write(b"dummy")
            for key in checkpoint_mod.A_KEYS:
                if key == "model_arch":
                    continue
                with self.subTest(key=key):
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                    raw[key] = self._break(raw[key])
                    with open(meta_path, "w", encoding="utf-8",
                              newline="\n") as fh:
                        json.dump(raw, fh, sort_keys=True, ensure_ascii=True)
                        fh.write("\n")
                    with self.assertRaises(
                            checkpoint_mod.IncompatibleCheckpoint):
                        eval_agent._make_checkpoint_agent(meta_path, seed=0)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_forged_model_arch_rejected_by_weight_shape_check(self):
        """設計書 §15.2: 保存メタの model_arch を偽装すると、重み state_dict
        とのキー集合・形状照合で拒否される(歯)。

        model_arch は expected_meta と自己比較されるため load の A 表照合は
        通る。偽装 arch で PolicyValueNet を構築して重みと照合する検証が
        無い実装へ戻すと(arch を信頼して load_state_dict だけで済ませると)、
        このテストは落ちる。promote の読込も eval_agent 経由のため同じ経路で
        拒否される。
        """
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        arch = PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(8, 8), action_hidden=(4,)).arch_dict()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            weights_path = os.path.join(tmp, "gen000.weights.pt")
            checkpoint_mod.save(
                meta_path, checkpoint_mod.current_meta(2, arch))
            net = PolicyValueNet(
                arch["obs_dim"], arch["action_dim"],
                hidden=tuple(arch["hidden"]),
                action_hidden=tuple(arch["action_hidden"]))
            torch.save(net.state_dict(), weights_path)
            # model_arch を偽装する(重みは本物のまま)。形状が矛盾する。
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            raw["model_arch"] = dict(raw["model_arch"], hidden=[16, 8])
            with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(raw, fh, sort_keys=True, ensure_ascii=True)
                fh.write("\n")
            with self.assertRaises(checkpoint_mod.IncompatibleCheckpoint) as cm:
                eval_agent._make_checkpoint_agent(meta_path, seed=0)
            self.assertIn("model_arch", str(cm.exception))

    def _write_forged_meta(self, tmp, mutate):
        """実評価経路用の保存メタを書き、model_arch を mutate で偽装する。"""
        from naeco.train import checkpoint as checkpoint_mod
        arch = {"obs_dim": 16, "action_dim": 8,
                "hidden": [4], "action_hidden": [2]}
        meta_path = os.path.join(tmp, "gen000.meta.json")
        checkpoint_mod.save(meta_path, checkpoint_mod.current_meta(2, arch))
        with open(meta_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        mutate(raw["model_arch"])
        with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(raw, fh, sort_keys=True, ensure_ascii=True)
            fh.write("\n")
        return meta_path

    def test_model_arch_type_and_key_forgery_rejected_on_eval_path(self):
        """設計書 §17.1: 評価経路(_make_checkpoint_agent)で model_arch の
        型偽装・余分なキー・キー欠落をすべて拒否する(歯)。

        checkpoint.load 単体の型厳密化(§16)は正しく効いているが、評価経路
        は保存メタの arch を先読みして**その同じ値**で expected を作る
        (自己比較)ため、等値な異型値(obs_dim: 569.0 / hidden: [16.0, 8])や
        余分なキーが int(...) 正規化と共に素通りしていた(Opus 実測)。
        _read_model_arch がキー集合の完全一致・type(x) is int・list 全要素
        int を要求しない実装へ戻すと、このテストは落ちる。
        """
        from naeco.train import checkpoint as checkpoint_mod
        cases = (
            ("obs_dim_float",
             lambda arch: arch.update(obs_dim=569.0)),
            ("action_dim_float",
             lambda arch: arch.update(action_dim=8.0)),
            ("hidden_element_float",
             lambda arch: arch.update(hidden=[16.0, 8])),
            ("action_hidden_element_float",
             lambda arch: arch.update(action_hidden=[2.0])),
            ("extra_key", lambda arch: arch.update(bogus=1)),
            ("missing_key", lambda arch: arch.pop("action_hidden")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, mutate in cases:
                with self.subTest(forgery=name):
                    meta_path = self._write_forged_meta(tmp, mutate)
                    with self.assertRaises(
                            checkpoint_mod.IncompatibleCheckpoint) as cm:
                        eval_agent._make_checkpoint_agent(meta_path, seed=0)
                    self.assertIn("model_arch", str(cm.exception))

    def test_no_int_normalization_on_eval_path(self):
        """設計書 §17.1 第二: 受理時の int(...) 正規化を廃止する(歯)。

        _read_model_arch の厳格検証を通った arch をそのまま PolicyValueNet
        へ渡す。int(arch["obs_dim"]) のような正規化(型偽装を黙って受け入れる
        経路)へ戻すとこのテストが落ちる。
        """
        with open(EVAL_AGENT_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("int(arch[", src)
        self.assertNotIn("int(x) for x in arch[", src)

    def test_promote_delegates_checkpoint_loading_to_eval_agent(self):
        """promote のチェックポイント読込は eval_agent へ委譲されている。

        promote.py に独自の model_arch 読込経路(checkpoint の import や
        _read_model_arch の再実装)が無いことを固定する。§17.1 の厳格検証を
        通らない別経路を promote へ追加するとこのテストが落ちる。
        """
        with open(PROMOTE_PATH, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("import checkpoint", src)
        self.assertNotIn("from naeco.train import checkpoint", src)
        self.assertNotIn("_read_model_arch", src)

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_checkpoint_agent_plays_games(self):
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        arch = PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(8, 8), action_hidden=(4,)).arch_dict()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            weights_path = os.path.join(tmp, "gen000.weights.pt")
            checkpoint_mod.save(
                meta_path, checkpoint_mod.current_meta(2, arch))
            net = PolicyValueNet(
                arch["obs_dim"], arch["action_dim"],
                hidden=tuple(arch["hidden"]),
                action_hidden=tuple(arch["action_hidden"]))
            torch.save(net.state_dict(), weights_path)
            spec = "checkpoint:" + meta_path
            result = eval_agent.run_evaluation(
                spec, "random", "train", 700, 1, workers=1)
            self.assertEqual(result.pairs, 1)
            self.assertEqual(len(result.game_records), 2)
            for rec in result.game_records:
                self.assertEqual(rec.left, spec)
                self.assertEqual(rec.right, "random")


class PromoteDecisionTests(unittest.TestCase):
    """§7 の判定ロジック(境界値で固定)。"""

    def test_decide_boundaries(self):
        champion = {"heuristic": 0.60}
        # ベンチが記録値 −3pt ちょうど → 昇格②は満たす。
        self.assertEqual(
            promote.decide(True, {"heuristic": 0.57}, champion, 0.03, 0.05),
            "promote")
        # −3pt を下回る → 昇格しない。
        self.assertEqual(
            promote.decide(True, {"heuristic": 0.5699}, champion, 0.03, 0.05),
            "reject")
        # 直接対戦が不合格(CI 下限 <= 0.50)でも昇格しない。
        self.assertEqual(
            promote.decide(False, {"heuristic": 0.99}, champion, 0.03, 0.05),
            "reject")
        # 5pt 超の低下 → ロールバック(直接合格でも最優先)。
        self.assertEqual(
            promote.decide(True, {"heuristic": 0.5499}, champion, 0.03, 0.05),
            "rollback")
        # ちょうど 5pt 低下は「超」ではない → ロールバックしない。
        self.assertEqual(
            promote.decide(True, {"heuristic": 0.55}, champion, 0.03, 0.05),
            "reject")

    def test_decide_requires_full_benchmark_coverage(self):
        with self.assertRaises(ValueError):
            promote.decide(True, {}, {"heuristic": 0.60}, 0.03, 0.05)


class PromoteConditionsTests(unittest.TestCase):
    """判定条件の固定(実行前読込・sha256・dev 分離)。"""

    def test_conditions_reject_non_dev_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            for group in ("train", "accept"):
                path = os.path.join(tmp, group + "-direct.json")
                _write_conditions(path, {"direct": {"group": group}})
                with self.assertRaises(ValueError):
                    promote._load_conditions(path)
                path = os.path.join(tmp, group + "-bench.json")
                _write_conditions(path, {"benchmark": {"group": group}})
                with self.assertRaises(ValueError):
                    promote._load_conditions(path)

    def test_conditions_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = (
                {"benchmark_entries": []},
                {"champion_benchmark": {"heuristic": 0.60,
                                        "other": 0.50}},
                {"regression_tolerance": 0.02},   # promote 幅以下は矛盾
                {"promote_threshold": 1.5},
                {"direct": {"pairs": 0}},
                {"direct": {"start_index": -1}},
                {"benchmark_entries": [
                    {"name": "heuristic", "kind": "checkpoint"}]},
            )
            for i, overrides in enumerate(cases):
                path = os.path.join(tmp, "conditions-%d.json" % i)
                _write_conditions(path, overrides)
                with self.assertRaises(ValueError):
                    promote._load_conditions(path)
            # 空 champion_benchmark は非空検査の歯: 集合一致検査ではなく
            # 「must be a non-empty object」検査が弾くことを固定する(空 dict
            # の override は置換なので、ここで書いたファイルは本当に空になる)。
            path = os.path.join(tmp, "conditions-empty-champion.json")
            _write_conditions(path, {"champion_benchmark": {}})
            with self.assertRaisesRegex(
                    ValueError,
                    "champion_benchmark must be a non-empty object"):
                promote._load_conditions(path)

    def test_conditions_sha256_freeze_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conditions.json")
            _write_conditions(path)
            # このテストの条件(pairs=2)は正式条件でないため、読込は非正式
            # モードで行う(正式条件の拘束は FormalConditionsTests が固定)。
            conditions, digest = promote._load_conditions(
                path, allow_informal=True)
            self.assertEqual(len(digest), 64)
            self.assertEqual(conditions.direct_pairs, 2)
            _write_conditions(path, {"direct": {"pairs": 3}})
            with self.assertRaises(RuntimeError):
                promote._verify_conditions_unchanged(path, digest)


class PromoteFlowTestBase(unittest.TestCase):
    """promote CLI の実行ヘルパーを共有する基底クラス。

    PromoteFlowTests と FormalConditionsTests の両方が _run_main を使う
    (判定フローの実行と、出力 JSON の formal フラグ検証)。
    """

    def _run_main(self, tmp, conditions_path, out_path, records=None,
                  execute=True, allow_informal=True, ledger=None):
        """promote.main を実行する。challenger / champion のダミーチェック
        ポイントを tmp 下に作り、評価はスタブで置き換える前提。

        allow_informal=True は非正式条件(pairs=2 等)を使うテスト用。
        正式条件の拘束は FormalConditionsTests と
        test_strict_flow_with_formal_conditions_writes_formal_decision が固定。
        """
        challenger = os.path.join(tmp, "ckpt", "challenger.meta.json")
        champion = os.path.join(tmp, "ckpt", "champion.meta.json")
        _write_dummy_checkpoint(challenger)
        _write_dummy_checkpoint(champion)
        argv = [
            "--conditions", conditions_path,
            "--challenger", challenger,
            "--champion", champion,
            "--out", out_path,
        ]
        if ledger:
            argv += ["--seed-ledger", ledger]
        if records:
            argv += ["--records", records]
        if allow_informal:
            argv.append("--allow-informal")
        if execute:
            argv.append("--execute")
        return promote.main(argv)


class PromoteFlowTests(PromoteFlowTestBase):
    """promote CLI の実行規律と判定フロー(評価はスタブで固定)。"""

    def test_no_execute_prepares_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            code = self._run_main(tmp, conditions_path, out_path,
                                  execute=False)
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(out_path),
                             "--execute なしで判定を出してはならない")

    def test_promote_flow_promotes_when_conditions_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                return _fake_result(0.90, 0.85, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                decision = json.load(fh)
            self.assertEqual(decision["decision"], "promote")
            self.assertEqual(
                decision["challenger"],
                "checkpoint:" + os.path.join(tmp, "ckpt",
                                             "challenger.meta.json"))
            self.assertEqual(
                decision["champion"],
                "checkpoint:" + os.path.join(tmp, "ckpt",
                                             "champion.meta.json"))
            self.assertEqual(decision["conditions_sha256"],
                             promote._conditions_sha256(
                                 promote._read_conditions_bytes(
                                     conditions_path)))
            # 判定条件の内容が出力に埋め込まれている(固定の証跡)。
            with open(conditions_path, "r", encoding="utf-8") as fh:
                self.assertEqual(decision["conditions"], json.load(fh))
            self.assertTrue(decision["direct"]["passes_threshold"])
            self.assertEqual(decision["new_champion_record"]["heuristic"],
                             0.90)
            self.assertEqual(
                decision["benchmark"]["champion_record"]["heuristic"], 0.60)

    def test_direct_ci_low_equal_to_threshold_does_not_promote(self):
        """歯 #6: 合否比較を >= にすると落ちる(CI 下限 == 閾値で不合格)。"""
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            for ci_low, expected in ((0.50, "reject"), (0.5000001, "promote")):
                out_path = os.path.join(tmp, "decision-%s.json" % ci_low)
                real = eval_agent.run_evaluation

                def fake(left, right, group, start_index, pairs, workers=1,
                         _ci_low=ci_low):
                    return _fake_result(0.90, _ci_low, pairs, group,
                                        start_index, left, right)

                eval_agent.run_evaluation = fake
                try:
                    code = self._run_main(tmp, conditions_path, out_path)
                finally:
                    eval_agent.run_evaluation = real
                self.assertEqual(code, 0)
                with open(out_path, "r", encoding="utf-8") as fh:
                    decision = json.load(fh)
                self.assertEqual(decision["decision"], expected)

    def test_regression_triggers_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path, {
                "champion_benchmark": {"heuristic": 0.90}})
            out_path = os.path.join(tmp, "decision.json")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                # ベンチ 0.80 はチャンピオン記録 0.90 より 10pt 低下。
                return _fake_result(0.80, 0.51, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                decision = json.load(fh)
            self.assertEqual(decision["decision"], "rollback")
            self.assertTrue(decision["benchmark"]["regression_detected"])

    def test_flow_writes_records_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            records = os.path.join(tmp, "phase3-promote")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                return _fake_result(0.90, 0.85, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path,
                                      records=records)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 0)
            # スタブ結果の game_records は空なので空ファイルができる(経路確認)。
            self.assertTrue(os.path.exists(records + "-direct.jsonl"))
            self.assertTrue(os.path.exists(records + "-bench.jsonl"))

    def test_flow_detects_conditions_change_mid_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                # 評価の途中で条件ファイルが外部変更された状況を再現する。
                _write_conditions(conditions_path, {"direct": {"pairs": 999}})
                return _fake_result(0.90, 0.85, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(out_path),
                             "条件変更を検出したら判定を出してはならない")

    def test_promote_rejects_non_dev_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path, {
                "direct": {"group": "train"}})
            out_path = os.path.join(tmp, "decision.json")
            with self.assertRaises(SystemExit):
                self._run_main(tmp, conditions_path, out_path, execute=False)

    def test_promote_requires_checkpoint_meta_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            with self.assertRaises(SystemExit):
                promote.main([
                    "--conditions", conditions_path,
                    "--challenger", "challenger.pt",
                    "--champion", "ckpt/champion.meta.json",
                    "--out", out_path])


class FormalConditionsTests(PromoteFlowTestBase):
    """正式条件の拘束(設計書 §18.2)。

    既定モードで 800ペア / 0.50 / 0.03 / 0.05 / 予約 seed 範囲
    (direct は dev 7800 または 10800 から 800 刻み、benchmark は dev 9000-9799)を
    要求し、逸脱は --allow-informal でのみ許される。非正式の実行は出力に
    「非正式」と明記される。
    """

    def test_strict_mode_accepts_formal_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conditions.json")
            _write_formal_conditions(path)
            # 7800 がまだ未消費だった時点の台帳(第1回昇格判定の状況)。
            ledger = os.path.join(tmp, "ledger.md")
            _write_ledger(ledger, [(0, 499), (1000, 1499)])
            conditions, digest = promote._load_conditions(
                path, ledger_path=ledger)
            self.assertTrue(
                promote._conditions_are_formal(conditions, ledger_path=ledger))
            self.assertEqual(len(digest), 64)
            self.assertEqual(conditions.direct_start_index, 7800)
            self.assertEqual(conditions.direct_pairs, 800)
            self.assertEqual(conditions.bench_start_index, 9000)
            self.assertEqual(conditions.bench_pairs, 800)
            self.assertEqual(conditions.promote_threshold, 0.50)
            self.assertEqual(conditions.promote_tolerance, 0.03)
            self.assertEqual(conditions.regression_tolerance, 0.05)

    def test_strict_mode_rejects_each_deviation(self):
        """正式条件からの各逸脱(1ペア・別閾値・別 seed 範囲)を既定モードで
        拒否する(歯)。逸脱を許すのは --allow-informal だけ。"""
        cases = (
            ("direct_pairs", {"direct": {"pairs": 1}}),
            ("benchmark_pairs", {"benchmark": {"pairs": 1}}),
            ("direct_start_index", {"direct": {"start_index": 7801}}),
            ("benchmark_start_index", {"benchmark": {"start_index": 9001}}),
            ("promote_threshold", {"promote_threshold": 0.51}),
            ("promote_tolerance", {"promote_tolerance": 0.04}),
            ("regression_tolerance", {"regression_tolerance": 0.06}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for name, overrides in cases:
                with self.subTest(deviation=name):
                    path = os.path.join(tmp, "conditions.json")
                    _write_conditions(path, overrides)
                    with self.assertRaises(ValueError) as cm:
                        promote._load_conditions(path)
                    self.assertIn("--allow-informal", str(cm.exception))

    def test_only_the_next_unconsumed_reserved_band_is_formal(self):
        """正式に使えるのは**台帳で次に空いている予約帯**ただ1つである。

        帯を 7800 に固定したままだと、台帳(docs/eval-seeds.md)が
        「次は dev 10800 以降」と定めている以上、2回目の正式昇格が実行できない。
        かといって予約列を全部認めると、飛ばし・再利用・事後選択を防げない
        (Sol ゲート4回目の指摘)。台帳を読んで**次の1つ**だけを認める。
        """
        cases = (
            # 消費済み                              次に使える帯
            ([(0, 499)],                            7800),
            ([(7800, 8599)],                        10800),
            ([(7800, 8599), (10000, 10799)],        10800),
            ([(7800, 8599), (10800, 11599)],        11600),
            ([(7800, 8599), (10800, 11599),
              (11600, 12399)],                      12400),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for consumed, expected in cases:
                with self.subTest(consumed=consumed):
                    ledger = os.path.join(tmp, "ledger.md")
                    _write_ledger(ledger, consumed)
                    self.assertEqual(
                        promote._next_reserved_direct_band(
                            promote._parse_consumed_dev_bands(
                                open(ledger, encoding="utf-8").read())),
                        expected)
                    path = os.path.join(tmp, "conditions.json")
                    _write_formal_conditions_with(
                        path, {"direct": {"group": "dev",
                                          "start_index": expected,
                                          "pairs": 800}})
                    conditions, unused = promote._load_conditions(
                        path, ledger_path=ledger)
                    self.assertTrue(promote._conditions_are_formal(
                        conditions, ledger_path=ledger))

    def test_skipping_or_reusing_a_band_is_rejected(self):
        """次を飛ばす帯も、消費済みへ戻る帯も拒否する(歯)。"""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.md")
            _write_ledger(ledger, [(7800, 8599), (10000, 10799)])
            # 次は 10800。7800 は消費済み、11600 は次を飛ばしている。
            for start in (7800, 11600, 12400, 10000):
                with self.subTest(start_index=start):
                    path = os.path.join(tmp, "conditions.json")
                    _write_formal_conditions_with(
                        path, {"direct": {"group": "dev",
                                          "start_index": start,
                                          "pairs": 800}})
                    with self.assertRaises(ValueError) as cm:
                        promote._load_conditions(path, ledger_path=ledger)
                    self.assertIn("--allow-informal", str(cm.exception))

    def test_missing_or_empty_ledger_block_refuses_to_run(self):
        """台帳が読めない・ブロックが無い・空なら実行を拒否する(fail-closed)。

        「読めなかったので予約列だけで判断する」に落ちると、消費済み帯を
        正式条件として通してしまう。
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conditions.json")
            _write_formal_conditions(path)
            cases = {
                "missing_file": os.path.join(tmp, "does-not-exist.md"),
                "no_block": os.path.join(tmp, "no-block.md"),
                "empty_block": os.path.join(tmp, "empty-block.md"),
            }
            with open(cases["no_block"], "w", encoding="utf-8") as fh:
                fh.write("# 台帳のつもりだが目印が無い\n")
            with open(cases["empty_block"], "w", encoding="utf-8") as fh:
                fh.write("<!-- CONSUMED-DEV-BANDS:BEGIN -->\n"
                         "<!-- CONSUMED-DEV-BANDS:END -->\n")
            for name, ledger in cases.items():
                with self.subTest(case=name):
                    with self.assertRaises(ValueError):
                        promote._load_conditions(path, ledger_path=ledger)

    def test_unreserved_bands_are_rejected_even_if_unused(self):
        """予約列に無い帯は、未使用であっても正式条件として認めない(歯)。"""
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.md")
            _write_ledger(ledger, [(7800, 8599)])
            for start in (10000, 11000, 20000, 10799):
                with self.subTest(start_index=start):
                    path = os.path.join(tmp, "conditions.json")
                    _write_formal_conditions_with(
                        path, {"direct": {"group": "dev",
                                          "start_index": start,
                                          "pairs": 800}})
                    with self.assertRaises(ValueError) as cm:
                        promote._load_conditions(path, ledger_path=ledger)
                    self.assertIn("--allow-informal", str(cm.exception))

    def test_real_ledger_parses_and_names_the_next_band(self):
        """実物の docs/eval-seeds.md が解析でき、次帯が 14800 であること。

        台帳のブロックが壊れたら(書式を変えた、消し忘れた)ここで落ちる。
        Phase 7 で 10800–11599(昇格)/ 11600–12399(G1)/ 12400–13199(G2)/
        13200–13999(G1 再)/ 14000–14799(G2 再)を消費したので、次帯は 14800。
        """
        consumed = promote._load_consumed_dev_bands()
        self.assertGreater(len(consumed), 10)
        self.assertIn((7800, 8599), consumed)
        self.assertIn((10800, 11599), consumed)
        self.assertIn((14000, 14799), consumed)
        self.assertEqual(promote._next_reserved_direct_band(consumed), 14800)

    def test_allow_informal_accepts_deviations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conditions.json")
            _write_conditions(path, {"direct": {"pairs": 1}})
            conditions, unused_digest = promote._load_conditions(
                path, allow_informal=True)
            self.assertFalse(promote._conditions_are_formal(conditions))
            self.assertEqual(conditions.direct_pairs, 1)

    def test_allow_informal_flow_marks_decision_informal(self):
        """非正式条件の実行は、出力 JSON に formal=false として明記される。"""
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)   # pairs=2 → 非正式
            out_path = os.path.join(tmp, "decision.json")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                return _fake_result(0.90, 0.85, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                decision = json.load(fh)
            self.assertIn("formal", decision,
                          "formal フラグが出力から落ちるとこのテストが落ちる")
            self.assertFalse(decision["formal"])

    def test_strict_flow_with_formal_conditions_writes_formal_decision(self):
        """正式条件の実行は、出力 JSON に formal=true として明記される。"""
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_formal_conditions(conditions_path)
            # 7800 が未消費だった時点の台帳(第1回昇格判定の状況)を使う。
            ledger = os.path.join(tmp, "ledger.md")
            _write_ledger(ledger, [(0, 499)])
            out_path = os.path.join(tmp, "decision.json")
            real = eval_agent.run_evaluation

            def fake(left, right, group, start_index, pairs, workers=1):
                return _fake_result(0.90, 0.85, pairs, group, start_index,
                                    left, right)

            eval_agent.run_evaluation = fake
            try:
                code = self._run_main(tmp, conditions_path, out_path,
                                      allow_informal=False, ledger=ledger)
            finally:
                eval_agent.run_evaluation = real
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                decision = json.load(fh)
            self.assertIn("formal", decision,
                          "formal フラグが出力から落ちるとこのテストが落ちる")
            self.assertTrue(decision["formal"])


class ModelFreezeTests(unittest.TestCase):
    """判定対象モデルの meta / weights の SHA-256 凍結(設計書 §18.2)。

    eval_agent は対局タスクごとにチェックポイントを再読込するため、評価中に
    同じパスの重みが置換されると複数モデルの結果が混在しても検出できない。
    判定開始時に全モデル(challenger / champion / ベンチの過去世代)を凍結し、
    判定終了時に再検査して、変わっていたら RuntimeError で失敗させる
    (fail-closed。判定を出さない)。
    """

    def _run_main_with_tamper(self, tmp, conditions_path, out_path, tamper):
        challenger = os.path.join(tmp, "ckpt", "challenger.meta.json")
        champion = os.path.join(tmp, "ckpt", "champion.meta.json")
        _write_dummy_checkpoint(challenger)
        _write_dummy_checkpoint(champion)
        real = eval_agent.run_evaluation
        tampered = []

        def fake(left, right, group, start_index, pairs, workers=1):
            if not tampered:
                tampered.append(True)
                tamper(challenger, champion)
            return _fake_result(0.90, 0.85, pairs, group, start_index,
                                left, right)

        eval_agent.run_evaluation = fake
        try:
            return promote.main([
                "--conditions", conditions_path,
                "--challenger", challenger,
                "--champion", champion,
                "--out", out_path,
                "--allow-informal",
                "--execute"])
        finally:
            eval_agent.run_evaluation = real

    def _assert_no_decision_on_tamper(self, tmp, out_path, code):
        self.assertEqual(code, 1)
        self.assertFalse(os.path.exists(out_path),
                         "モデル置換を検出したら判定を出してはならない")

    def test_tampered_challenger_meta_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")

            def tamper(challenger, champion):
                # 評価の途中で challenger のメタが置換された状況を再現する。
                with open(challenger, "w", encoding="utf-8",
                          newline="\n") as fh:
                    fh.write("{}\n")

            code = self._run_main_with_tamper(
                tmp, conditions_path, out_path, tamper)
            self._assert_no_decision_on_tamper(tmp, out_path, code)

    def test_tampered_champion_weights_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")

            def tamper(challenger, champion):
                # 評価の途中で champion の重みが置換された状況を再現する。
                with open(
                        eval_agent.checkpoint_weights_path(champion),
                        "wb") as fh:
                    fh.write(b"replaced-weights")

            code = self._run_main_with_tamper(
                tmp, conditions_path, out_path, tamper)
            self._assert_no_decision_on_tamper(tmp, out_path, code)

    def test_tampered_benchmark_past_generation_detected(self):
        """凍結対象は challenger / champion だけでなく、ベンチの過去世代
        (checkpoint エントリ)の meta / weights も含む(歯)。"""
        with tempfile.TemporaryDirectory() as tmp:
            past = os.path.join(tmp, "past.meta.json")
            _write_dummy_checkpoint(past)
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path, {
                "benchmark_entries": [
                    {"name": "heuristic", "kind": "heuristic"},
                    {"name": "past", "kind": "checkpoint", "path": past},
                ],
                "champion_benchmark": {"heuristic": 0.60, "past": 0.55},
            })
            out_path = os.path.join(tmp, "decision.json")

            def tamper(challenger, champion):
                with open(eval_agent.checkpoint_weights_path(past),
                          "wb") as fh:
                    fh.write(b"replaced-past")

            code = self._run_main_with_tamper(
                tmp, conditions_path, out_path, tamper)
            self._assert_no_decision_on_tamper(tmp, out_path, code)

    def test_unchanged_models_pass_and_hashes_in_decision(self):
        """凍結・再検査が通る正常系。出力に frozen_models の SHA-256 が残る。"""
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            code = self._run_main_with_tamper(
                tmp, conditions_path, out_path,
                lambda challenger, champion: None)
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as fh:
                decision = json.load(fh)
            frozen = decision["frozen_models"]
            self.assertEqual(sorted(frozen),
                             ["challenger", "champion"])
            for label, files in frozen.items():
                self.assertEqual(len(files["meta"]), 64)
                self.assertEqual(len(files["weights"]), 64)

    def test_missing_model_file_fails_closed(self):
        """凍結対象のファイルが欠けていれば失敗(fail-closed。判定を出さない)。"""
        with tempfile.TemporaryDirectory() as tmp:
            conditions_path = os.path.join(tmp, "conditions.json")
            _write_conditions(conditions_path)
            out_path = os.path.join(tmp, "decision.json")
            challenger = os.path.join(tmp, "challenger.meta.json")
            champion = os.path.join(tmp, "champion.meta.json")
            _write_dummy_checkpoint(challenger)
            # champion のファイルを作らない(欠落)。
            code = promote.main([
                "--conditions", conditions_path,
                "--challenger", challenger,
                "--champion", champion,
                "--out", out_path,
                "--allow-informal",
                "--execute"])
            self.assertEqual(code, 1)
            self.assertFalse(os.path.exists(out_path))


class EvalResultRecordingTests(unittest.TestCase):
    """チェックポイントのメタへの評価結果追記(設計書 §18.1 / PLAN.md)。

    - 評価対象がチェックポイントのとき、そのメタの eval_results へ結果
      (相手の指定・group・index 範囲・ペア数・勝分敗・点推定・CI・対応表の
      SHA-256・合否判定)を追記できる。
    - eval_results は B 表キーなので、追記しても A 表13キーは不変。
    - 追記先は評価対象の checkpoint エージェントのメタだけ(それ以外は
      ValueError)。
    """

    @staticmethod
    def _write_checkpoint(tmp, name="gen000"):
        from naeco.train import checkpoint as checkpoint_mod
        arch = {"obs_dim": 16, "action_dim": 8,
                "hidden": [4], "action_hidden": [2]}
        meta_path = os.path.join(tmp, name + ".meta.json")
        weights_path = os.path.join(tmp, name + ".weights.pt")
        checkpoint_mod.save(meta_path, checkpoint_mod.current_meta(2, arch))
        with open(weights_path, "wb") as fh:
            fh.write(b"dummy")
        return meta_path

    @staticmethod
    def _result(spec, opponent="random", point=0.5625, ci_low=0.54,
                ci_high=0.59, pairs=800, group="dev", start_index=7800):
        games = pairs * 2
        wins = int(round(point * games))
        return eval_agent.AgentEvaluationResult(
            left=spec, right=opponent, group=group, start_index=start_index,
            pairs=pairs, games=games, wins=wins, draws=0,
            losses=games - wins, point_estimate=point, ci_low=ci_low,
            ci_high=ci_high, bootstrap_samples=10, bootstrap_seed=1,
            pair_results=(), initial_state_hashes=(), game_records=())

    def test_append_records_required_fields_and_preserves_a_table(self):
        from naeco.train import checkpoint as checkpoint_mod
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            spec = "checkpoint:" + meta_path
            result = self._result(spec)
            with open(meta_path, "r", encoding="utf-8") as fh:
                before = json.load(fh)
            entry = eval_agent.append_eval_result(
                meta_path, result, threshold=0.50, records_path=None)
            with open(meta_path, "r", encoding="utf-8") as fh:
                after = json.load(fh)
            # A 表13キーは不変(eval_results は B 表キー。設計書 §18.1)。
            for key in checkpoint_mod.A_KEYS:
                self.assertEqual(before[key], after[key], key)
            self.assertEqual(after["eval_results"], [entry])
            self.assertEqual(entry["recorded_agent"], spec)
            self.assertEqual(entry["opponent"], "random")
            self.assertEqual(entry["group"], "dev")
            self.assertEqual(entry["start_index"], 7800)
            self.assertEqual(entry["end_index"], 8599)
            self.assertEqual(entry["pairs"], 800)
            self.assertEqual(entry["games"], 1600)
            self.assertEqual(entry["wins"], 900)
            self.assertEqual(entry["draws"], 0)
            self.assertEqual(entry["losses"], 700)
            self.assertEqual(entry["point_estimate"], 0.5625)
            self.assertEqual(entry["ci_low"], 0.54)
            self.assertEqual(entry["ci_high"], 0.59)
            self.assertEqual(entry["threshold"], 0.50)
            self.assertTrue(entry["passed"])
            self.assertIsNone(entry["records_sha256"])
            # 再度追記すると2件目として追加される(created_at は1秒境界で
            # 変わり得るため比較から除く)。
            eval_agent.append_eval_result(meta_path, result, 0.50)
            with open(meta_path, "r", encoding="utf-8") as fh:
                after2 = json.load(fh)
            self.assertEqual(len(after2["eval_results"]), 2)
            second = dict(after2["eval_results"][1])
            second.pop("created_at", None)
            first = dict(entry)
            first.pop("created_at", None)
            self.assertEqual(second, first)

    def test_append_records_records_sha256(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            records = os.path.join(tmp, "rec.jsonl")
            with open(records, "wb") as fh:
                fh.write(b"row\n")
            spec = "checkpoint:" + meta_path
            result = self._result(spec)
            entry = eval_agent.append_eval_result(
                meta_path, result, 0.50, records_path=records)
            self.assertEqual(
                entry["records_sha256"],
                hashlib.sha256(b"row\n").hexdigest())

    def test_append_records_fail_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            spec = "checkpoint:" + meta_path
            # CI 下限 <= 閾値 → 不合格。
            result = self._result(spec, point=0.51, ci_low=0.49)
            entry = eval_agent.append_eval_result(meta_path, result, 0.50)
            self.assertFalse(entry["passed"])

    def test_append_records_opponent_side(self):
        """評価対象が右席(opponent 側)のときも正しく記録される。"""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            spec = "checkpoint:" + meta_path
            result = self._result("heuristic", opponent=spec)
            entry = eval_agent.append_eval_result(meta_path, result, 0.50)
            self.assertEqual(entry["recorded_agent"], spec)
            self.assertEqual(entry["opponent"], "heuristic")

    def test_append_rejects_unrelated_meta_path(self):
        """評価対象でないチェックポイントへの追記は ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            left_path = self._write_checkpoint(tmp, name="left")
            right_path = self._write_checkpoint(tmp, name="right")
            result = self._result("checkpoint:" + left_path,
                                  opponent="checkpoint:" + right_path)
            unrelated = self._write_checkpoint(tmp, name="other")
            with self.assertRaises(ValueError):
                eval_agent.append_eval_result(unrelated, result, 0.50)

    def test_append_rejects_non_list_eval_results(self):
        """eval_results が list / null 以外のときは ValueError(fail-closed)。"""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            meta["eval_results"] = {"corrupted": True}
            with open(meta_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(meta, fh, sort_keys=True, ensure_ascii=True)
                fh.write("\n")
            spec = "checkpoint:" + meta_path
            result = self._result(spec)
            with self.assertRaises(ValueError):
                eval_agent.append_eval_result(meta_path, result, 0.50)

    def test_cli_record_eval_to_requires_matching_checkpoint_agent(self):
        """--record-eval-to は評価対象の checkpoint メタでなければ拒否される。"""
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            with self.assertRaises(SystemExit):
                eval_agent.main([
                    "heuristic", "random", "--group", "train",
                    "--start-index", "100", "--pairs", "1",
                    "--record-eval-to", meta_path, "--execute"])
            with self.assertRaises(SystemExit):
                eval_agent.main([
                    "checkpoint:" + meta_path, "random", "--group", "train",
                    "--start-index", "100", "--pairs", "1",
                    "--record-eval-to", meta_path + ".json", "--execute"])

    def test_cli_record_eval_to_without_execute_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = self._write_checkpoint(tmp)
            code = eval_agent.main([
                "checkpoint:" + meta_path, "random", "--group", "train",
                "--start-index", "100", "--pairs", "1",
                "--record-eval-to", meta_path])
            self.assertEqual(code, 0)
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertIsNone(meta["eval_results"],
                              "--execute なしでメタを書き換えてはならない")

    @unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
    def test_cli_records_eval_result_into_checkpoint_meta(self):
        """実評価(対局実行)の結果が --record-eval-to でメタへ追記される。"""
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train.net import PolicyValueNet
        arch = PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(8, 8), action_hidden=(4,)).arch_dict()
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "gen000.meta.json")
            weights_path = os.path.join(tmp, "gen000.weights.pt")
            checkpoint_mod.save(
                meta_path, checkpoint_mod.current_meta(2, arch))
            net = PolicyValueNet(
                arch["obs_dim"], arch["action_dim"],
                hidden=tuple(arch["hidden"]),
                action_hidden=tuple(arch["action_hidden"]))
            torch.save(net.state_dict(), weights_path)
            spec = "checkpoint:" + meta_path
            records = os.path.join(tmp, "rec.jsonl")
            code = eval_agent.main([
                spec, "random", "--group", "train",
                "--start-index", "800", "--pairs", "1",
                "--records", records,
                "--record-eval-to", meta_path,
                "--execute"])
            self.assertIn(code, (0, 1))
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            entries = meta["eval_results"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["recorded_agent"], spec)
            self.assertEqual(entries[0]["opponent"], "random")
            self.assertEqual(entries[0]["group"], "train")
            self.assertEqual(entries[0]["start_index"], 800)
            self.assertEqual(entries[0]["end_index"], 800)
            self.assertEqual(entries[0]["pairs"], 1)
            self.assertEqual(entries[0]["wins"] + entries[0]["draws"]
                             + entries[0]["losses"], 2)
            self.assertEqual(len(entries[0]["records_sha256"]), 64)
            self.assertIn(entries[0]["passed"], (True, False))


if __name__ == "__main__":
    unittest.main()
