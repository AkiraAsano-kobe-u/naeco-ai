"""Phase 5 M3/M4: search-on vs search-off 対局ハーネスのテスト。

torch の無い環境では net をロードしない軽量経路(フェイク net 注入)で、
**記録形式・ペア対局・CI 計算・決定論**を検証する(受入条件)。torch がある
環境だけ、checkpoint ロード経路(`_load_net`)を fabricated checkpoint で
検証する。gen532 の実重みはリポジトリに含まれないため、`NAECO_GEN532_
CHECKPOINT_DIR` が指定されたときだけ実 checkpoint ロードを実行する。

フェイク net は PolicyValueNet と同じ `(logits, values)` 契約
(`forward(obs_batch, action_batch, action_mask)` が list を返す)を持ち、
全 logits = 0 / value = 0 を返す。greedy は sort_key 最小の合法手を選ぶ
ため完全に決定論的で、torch を必要としない。
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for _entry in (_TESTS_DIR, _PROJECT_ROOT, _SCRIPTS_DIR):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import compare_search_net  # noqa: E402
import search_records  # noqa: E402
from search_records import (  # noqa: E402
    environment_info, find_deterministic_diffs, load_bundle,
    load_validated_bundles, regenerate_aggregates, SearchRecordError,
)
import helpers as H  # noqa: E402
from naeco.engine import game  # noqa: E402
from naeco.engine.observe import observe  # noqa: E402
from naeco.engine.state import (  # noqa: E402
    Action, ActionKind, derive_seed, eval_seed, state_hash,
)
from naeco.train.agent import PolicyAgent  # noqa: E402

SCRIPT = os.path.join(_SCRIPTS_DIR, "compare_search_net.py")


def _thread_env():
    return dict((name, "1") for name in search_records.THREAD_ENV_NAMES)


def _torch_available():
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# import 時点で torch が読み込まれていないことの固定(import 順に依存しない
# よう、torch を import し得る decorator 評価より前に撮る)。
_TORCH_IMPORTED_AT_IMPORT = "torch" in sys.modules


class _FakeValueNet(object):
    """決定論的フェイク net。PolicyValueNet と同じ (logits, values) 契約。

    全 logits = 0(greedy は sort_key 最小の合法手)、value = 0(勝率 0.5)。
    torch を必要としない。net 呼出を数えるカウンタを持つ。

    PolicyValueNet の値ヘッドは batch 行ごとに1値(`NetEvaluator._value_scalar`
    が長さ1を要求)なので、values は obs_batch の行数ぶんだけ返す
    (action 数ぶんではない)。
    """

    def __init__(self):
        self.calls = 0

    def forward(self, obs_batch, action_batch, action_mask):
        self.calls += 1
        logits = [[0.0] * len(actions) for actions in action_batch]
        values = [0.0] * len(obs_batch)
        return logits, values


def _fake_net_factory():
    """spawn 子プロセスでも名前解決できるようモジュール直下に置く。"""
    return _FakeValueNet()


def _cli_env(**overrides):
    env = dict(os.environ)
    for name in search_records.THREAD_ENV_NAMES:
        env[name] = "1"
    env.pop("PYTHONHASHSEED", None)
    env.pop(compare_search_net._HASH_SEED_BOOTSTRAP_MARKER, None)
    env.update(overrides)
    return env


def _run_cli(argv, env):
    return subprocess.run([sys.executable, SCRIPT] + argv, env=env,
                          capture_output=True, text=True, timeout=180,
                          cwd=_PROJECT_ROOT)


def _replay_count_off_decisions(record):
    """記録された行動列を seed から独立に再適用し、off 席の決定数を数える。

    (env_seed / setup_seed, 行動列) だけから game.new_game + game.step で
    再構成し、各 step 後の state_hash 列も返す(D-7 の独立 replay と同じ
    規律)。search_off の「net 呼出 = 決定数」を記録と独立に検証できる。
    引数名は `game` ではなく `record` にする(naeco.engine.game モジュールを
    隠蔽すると game.new_game 等が AttributeError になる)。
    """
    state = game.new_game(2, record["env_seed"], H.CARDS,
                          setup_seed=record["setup_seed"])
    off_seat = record["right_seat"]
    count = 0
    hashes = []
    for kind, target_instance_id, instance_id, goods_count \
            in record["replay"]["actions"]:
        actor = game.actor(state)
        if actor == off_seat:
            count += 1
        action = Action(kind=ActionKind(kind),
                        target_instance_id=target_instance_id,
                        instance_id=instance_id,
                        goods_count=goods_count)
        state = game.step(state, action, H.CARDS).state
        hashes.append(state_hash(state))
    return count, hashes


class FakeNetHarnessTests(unittest.TestCase):
    """フェイク net 注入の軽量経路で記録形式・ペア対局・CI・決定論を検証する。"""

    def _run(self, pairs, start_index, replay_audit=False):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"},
                             clear=False):
            return compare_search_net.run_comparison(
                "train", start_index, pairs, 300, 2, 30, 1.41,
                net_factory=_fake_net_factory, replay_audit=replay_audit)

    def test_run_comparison_records_bundle_format(self):
        row, games = self._run(2, 500000)
        self.assertEqual(row["left"], "search_on")
        self.assertEqual(row["right"], "search_off")
        self.assertEqual(row["budget"], 300)
        self.assertEqual(len(games), 4)
        self.assertEqual(len(row["pair_scores"]), 2)
        self.assertTrue(math.isfinite(row["point_estimate"]))
        self.assertTrue(math.isfinite(row["ci_low"]))
        self.assertTrue(math.isfinite(row["ci_high"]))
        # CI は search_records と同じ bootstrap(B=10000 / seed=2026072102)。
        self.assertEqual(
            (row["ci_low"], row["ci_high"]),
            search_records.recompute_ci(row["pair_scores"]))
        self.assertEqual(set(row["search_metrics"]),
                         {"search_on", "search_off"})
        for arm in ("search_on", "search_off"):
            self.assertEqual(row["search_metrics"][arm]["method"], arm)
        # 両腕の遷移数・net 呼出数を併記する(handoff §6)。search_on の net
        # 呼出総数は探索内の葉評価(evaluations)と委譲(delegated_calls)の合計。
        self.assertGreater(row["search_metrics"]["search_on"]["transitions"],
                           0)
        self.assertGreater(row["search_metrics"]["search_on"]["evaluations"],
                           0)
        self.assertGreater(row["search_metrics"]["search_off"]["evaluations"],
                           0)
        on_metrics = row["search_metrics"]["search_on"]
        self.assertEqual(
            on_metrics["total_net_calls"],
            on_metrics["evaluations"] + on_metrics["delegated_calls"])
        self.assertGreater(on_metrics["total_net_calls"], 0)
        # summary row の net 呼出総数も同じ不変式を満たす。
        self.assertEqual(row["total_net_calls"],
                         row["evaluations"] + row["delegated_calls"])
        for game in games:
            self.assertEqual(game["left"], "search_on")
            self.assertEqual(game["right"], "search_off")
            self.assertEqual(game["left_seat"], game["seat_variant"])
            self.assertEqual(game["right_seat"], 1 - game["seat_variant"])
            self.assertIn(game["result_left"], (0.0, 0.5, 1.0))
            self.assertEqual(
                game["env_seed"],
                derive_seed(eval_seed("train", game["index"]),
                            "env", game["index"]))
            for name in ("setup_seed", "policy_seed_seat0",
                         "policy_seed_seat1", "start_player",
                         "initial_money", "initial_deck", "initial_hands",
                         "initial_state_hash", "final_scores",
                         "cards_sha256", "cards_data_version", "api_version",
                         "rules_version", "obs_version", "horizon", "scale",
                         "c", "search_metrics"):
                self.assertIn(name, game)
            metrics = game["search_metrics"]
            self.assertEqual(set(metrics), {"search_on", "search_off"})
            self.assertEqual(
                set(metrics["search_on"]),
                {"transitions", "nodes_created", "determinizations",
                 "evaluations", "delegated_calls", "total_net_calls",
                 "elapsed_ms_mean", "elapsed_ms_p95"})
            on = metrics["search_on"]
            self.assertGreaterEqual(on["transitions"], 0)
            self.assertGreaterEqual(on["nodes_created"], 0)
            self.assertGreaterEqual(on["determinizations"], 0)
            self.assertGreaterEqual(on["delegated_calls"], 0)
            self.assertEqual(on["total_net_calls"],
                             on["evaluations"] + on["delegated_calls"])
            off = metrics["search_off"]
            # search_off は遷移ゼロ / net 呼出 = 決定数 / 思考時間は実測。
            self.assertEqual(off["transitions"], 0)
            self.assertEqual(off["nodes_created"], 0)
            self.assertEqual(off["determinizations"], 0)
            self.assertGreater(off["evaluations"], 0)
            self.assertIsNotNone(off["elapsed_ms_mean"])
            self.assertIsNotNone(off["elapsed_ms_p95"])

    def test_bundle_round_trip_and_validation(self):
        row, games = self._run(1, 500010)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            config = compare_search_net._config(
                "train", 500010, 1, "search_on", "search_off", 300, None,
                2, 30, 1.41)
            with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"},
                                 clear=False):
                env = environment_info(1, _thread_env())
                compare_search_net._finalize_bundle(
                    out, compare_search_net.LABEL, compare_search_net.LABEL,
                    row, games, config,
                    compare_search_net._versions(H.CARDS), env)
            # load_bundle が W/D/L・pair_scores・CI・ペア整合を完全再検証する。
            bundle = load_bundle(out, compare_search_net.LABEL)
            self.assertEqual(bundle["summary"]["label"],
                             compare_search_net.LABEL)
            self.assertEqual(len(bundle["games"]), 2)
            self.assertIn("pair_scores", bundle["summary"])
            self.assertIn("ci_low", bundle["summary"])
            self.assertIn("ci_high", bundle["summary"])
            self.assertIn("point_estimate", bundle["summary"])
            self.assertIn("search_metrics", bundle["summary"])
            self.assertIn("search_metrics", bundle["games"][0])
            bundles = load_validated_bundles(out)
            self.assertEqual(set(bundles), {compare_search_net.LABEL})
            regenerate_aggregates(out, bundles)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(out + ".manifest.jsonl"))

    def test_ci_matches_harness_bootstrap(self):
        row, unused_games = self._run(1, 500004)
        expected_low, expected_high = search_records.recompute_ci(
            row["pair_scores"])
        self.assertEqual(row["ci_low"], expected_low)
        self.assertEqual(row["ci_high"], expected_high)
        self.assertEqual(
            (row["ci_low"], row["ci_high"]),
            search_records.recompute_ci(row["pair_scores"]))

    def test_in_process_determinism(self):
        row_one, games_one = self._run(1, 500020)
        row_two, games_two = self._run(1, 500020)
        diffs = find_deterministic_diffs(
            row_one, row_two, games_one, games_two,
            excluded=compare_search_net.DETERMINISM_EXCLUDED_PATHS)
        self.assertEqual(diffs, [])

    def test_replay_and_off_arm_decision_counting(self):
        row, games = self._run(1, 500006, replay_audit=True)
        self.assertEqual(len(games), 2)
        off_total = 0
        for game in games:
            replay = game["replay"]
            self.assertGreater(len(replay["actions"]), 0)
            self.assertEqual(len(replay["actions"]),
                             len(replay["state_hashes"]))
            self.assertEqual(len(replay["actions"]),
                             len(replay["rng_states"]))
            self.assertEqual(len(replay["actions"][0]), 4)
            # 独立 replay で off 席の決定数を数え、search_off の
            # 「net 呼出 = 決定数」を記録と突き合わせる。state_hash 列も
            # 記録と一致することを確認する(D-7 の独立 replay)。
            count, hashes = _replay_count_off_decisions(game)
            self.assertEqual(
                game["search_metrics"]["search_off"]["evaluations"], count)
            self.assertEqual(hashes, replay["state_hashes"])
            off_total += count
        self.assertGreater(off_total, 0)

    def test_policy_agent_makes_one_net_call_per_decision(self):
        net = _FakeValueNet()
        agent = PolicyAgent(net, mode="greedy", seed=0)
        state = H.new(2, seed=44)
        obs = observe(state, game.actor(state), H.CARDS)
        self.assertTrue(obs.legal_actions)
        first = agent.act(obs)
        second = agent.act(obs)
        self.assertIn(first, obs.legal_actions)
        self.assertIn(second, obs.legal_actions)
        # 全決定で1 net 呼出(「search_off の net 呼出 = 決定数」の根拠)。
        self.assertEqual(net.calls, 2)


class DeterminismSpawnTests(unittest.TestCase):
    """worker=1 と worker=N の spawn 実行で決定論的 field が一致する。"""

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_check_determinism_matches_across_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det.json")
            code = compare_search_net.run_determinism_check(
                pairs=1, budget=300, horizon=2, scale=30, c=1.41,
                start_index=400, records=report, workers=2, group="train",
                net_factory=_fake_net_factory)
            self.assertEqual(code, 0)
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["all_labels_matched"])
            self.assertEqual(payload["label"], compare_search_net.LABEL)
            self.assertEqual(payload["schema_version"],
                             "phase5-determinism/v1")
            # elapsed_ms(壁時計由来)が明示 allowlist に載っている。
            self.assertIn("games.[].search_metrics.[].elapsed_ms_mean",
                          payload["excluded_fields"])
            self.assertIn("summary.search_metrics.[].elapsed_ms_p95",
                          payload["excluded_fields"])
            self.assertIsNone(payload["checkpoint_dir"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_check_determinism_replay_audit_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det-replay.json")
            code = compare_search_net.run_determinism_check(
                pairs=1, budget=300, horizon=2, scale=30, c=1.41,
                start_index=450, records=report, workers=2, group="train",
                replay_audit=True, net_factory=_fake_net_factory)
            self.assertEqual(code, 0)
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["replay_audit"])
            self.assertTrue(payload["all_labels_matched"])


class CliGuardrails(unittest.TestCase):
    """CLI の検証順序と拒否条件。実行本体はフェイク net 経路でなく CLI は
    --execute の入口だけを検証する(実 net は --checkpoint-dir が要求する)。"""

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_checkpoint_dir_is_required(self):
        with self.assertRaises(SystemExit):
            compare_search_net.main([])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_execute_requires_pairs_and_start_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--execute"])
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--execute"])
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "0", "--start-index", "0",
                                         "--execute"])
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--start-index", "0",
                                         "--budget", "0", "--execute"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_replay_audit_requires_check_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--start-index", "0",
                                         "--replay-audit", "--execute"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_check_determinism_requires_workers_records_pairs_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--check-determinism",
                                         "--workers", "1", "--execute"])
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--check-determinism",
                                         "--workers", "2", "--pairs", "1",
                                         "--execute"])
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--check-determinism",
                                         "--workers", "2", "--pairs", "1",
                                         "--start-index", "0", "--execute"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_resume_requires_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--start-index", "0",
                                         "--resume", "--execute"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_existing_records_are_rejected_without_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            with open(out, "w", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--start-index", "0",
                                         "--records", out, "--execute"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_checkpoint_dir_must_contain_gen532_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            with self.assertRaises(SystemExit) as ctx:
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--pairs", "1", "--start-index", "0",
                                         "--records", out, "--execute"])
            self.assertIn("gen532", str(ctx.exception))

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_prepared_mode_requires_no_checkpoint_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = compare_search_net.main(["--checkpoint-dir", tmp])
            self.assertEqual(code, 0)

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_regenerate_requires_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net.main(["--checkpoint-dir", tmp,
                                         "--regenerate"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_main_passes_checkpoint_dir_to_run_phase5_comparison(self):
        """回帰テスト: main() が run_phase5_comparison へ checkpoint_dir を渡す。

        workers=1(既定)では pool が None になり、run_comparison の
        in-process 経路(_prepare_net)が checkpoint_dir を要求する。渡さない
        と実チェックポイント実行で _load_net(None) の TypeError になる。
        """
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "gen532.meta.json"), "w",
                      encoding="utf-8") as handle:
                handle.write("{}\n")
            with open(os.path.join(tmp, "gen532.weights.pt"), "w",
                      encoding="utf-8") as handle:
                handle.write("dummy")
            original = compare_search_net.run_phase5_comparison
            calls = []

            def fake_run_phase5_comparison(
                    group, start_index, pairs, budget, horizon, scale, c,
                    out_path, pool=None, resume=False, environment=None,
                    replay_audit=False, checkpoint_dir=None, net_factory=None):
                calls.append({
                    "group": group, "start_index": start_index,
                    "pairs": pairs, "budget": budget, "horizon": horizon,
                    "scale": scale, "c": c, "out_path": out_path,
                    "pool": pool, "resume": resume,
                    "environment": environment,
                    "replay_audit": replay_audit,
                    "checkpoint_dir": checkpoint_dir,
                    "net_factory": net_factory,
                })
                return {"checkpoint_dir": checkpoint_dir}

            compare_search_net.run_phase5_comparison = fake_run_phase5_comparison
            try:
                code = compare_search_net.main([
                    "--checkpoint-dir", tmp,
                    "--pairs", "1", "--start-index", "0", "--execute",
                ])
            finally:
                compare_search_net.run_phase5_comparison = original
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            call = calls[0]
            # 実チェックポイント実行の in-process 経路(workers=1 / pool=None)
            # で checkpoint_dir が受け取れること。
            self.assertEqual(call["checkpoint_dir"], tmp)
            self.assertIsNone(call["pool"])
            self.assertIsNone(call["net_factory"])
            self.assertEqual(call["group"], "dev")
            self.assertEqual(call["start_index"], 0)
            self.assertEqual(call["pairs"], 1)
            self.assertEqual(call["budget"], compare_search_net.DEFAULT_BUDGET)
            self.assertEqual(call["resume"], False)
            self.assertEqual(call["replay_audit"], False)
            self.assertIsNone(call["out_path"])

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_regenerate_from_bundle(self):
        row, games = compare_search_net.run_comparison(
            "train", 500030, 1, 300, 2, 30, 1.41,
            net_factory=_fake_net_factory)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            config = compare_search_net._config(
                "train", 500030, 1, "search_on", "search_off", 300, None,
                2, 30, 1.41)
            env = environment_info(1, _thread_env())
            compare_search_net._finalize_bundle(
                out, compare_search_net.LABEL, compare_search_net.LABEL,
                row, games, config,
                compare_search_net._versions(H.CARDS), env)
            code = compare_search_net.main(
                ["--checkpoint-dir", tmp, "--regenerate", "--records", out])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(out + ".manifest.jsonl"))


class _Killed(BaseException):
    pass


class NetJournalResumeTests(unittest.TestCase):
    """途中停止 → --resume の最終 bundle が一度に通した結果と同一集計。"""

    def _go(self, out, resume=False, stop_after=None):
        real = compare_search_net._run_pair
        calls = {"n": 0}

        def flaky(task):
            if stop_after is not None and calls["n"] >= stop_after:
                raise _Killed()
            calls["n"] += 1
            return real(task)

        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False), \
                mock.patch.object(compare_search_net, "_run_pair", flaky):
            env = environment_info(1, _thread_env())
            compare_search_net.run_phase5_comparison(
                "train", 500040, 3, 300, 2, 30, 1.41, out, resume=resume,
                environment=env, net_factory=_fake_net_factory)

    def test_kill_then_resume_equals_single_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            whole = os.path.join(tmp, "w.jsonl")
            cut = os.path.join(tmp, "c.jsonl")
            self._go(whole)
            with self.assertRaises(_Killed):
                self._go(cut, stop_after=2)
            self.assertFalse(os.path.isdir(cut + "-bundle"))
            self._go(cut, resume=True)
            key = compare_search_net.LABEL
            a = load_validated_bundles(whole)[key]
            b = load_validated_bundles(cut)[key]
            self.assertEqual(find_deterministic_diffs(
                a["summary"], b["summary"], a["games"], b["games"],
                excluded=compare_search_net.DETERMINISM_EXCLUDED_PATHS), [])
            self.assertEqual(a["summary"]["pair_scores"],
                             b["summary"]["pair_scores"])
            self.assertFalse(os.path.exists(cut + "-journal"))


class _OtherNet(_FakeValueNet):
    def forward(self, obs_batch, action_batch, action_mask):
        self.calls += 1
        return ([[float(i) for i in range(len(a))] for a in action_batch],
                [0.0] * len(obs_batch))


class NetCheckpointFingerprintTests(unittest.TestCase):
    """checkpoint の中身が変われば、部分 journal の resume を拒否する。"""

    def _checkpoint(self, root, name, weights):
        path = os.path.join(root, name)
        os.makedirs(path)
        with open(os.path.join(path, compare_search_net.CHECKPOINT_META_NAME),
                  "w") as handle:
            handle.write("{}")
        with open(os.path.join(path,
                               compare_search_net.CHECKPOINT_WEIGHTS_NAME),
                  "wb") as handle:
            handle.write(weights)
        return path

    def _first_half(self, out, env, ckpt):
        real = compare_search_net._run_pair

        def stop_second(task):
            if task[1] == 500041:
                raise _Killed()
            return real(task)

        compare_search_net._NET = None
        with mock.patch.object(compare_search_net, "_load_net",
                               return_value=_FakeValueNet()), \
                mock.patch.object(compare_search_net, "_run_pair",
                                  stop_second):
            with self.assertRaises(_Killed):
                compare_search_net.run_phase5_comparison(
                    "train", 500040, 2, 60, 2, 30, 1.41, out,
                    environment=env, checkpoint_dir=ckpt)

    def _resume(self, out, env, ckpt, net):
        compare_search_net._NET = None
        with mock.patch.object(compare_search_net, "_load_net",
                               return_value=net) as load:
            try:
                return compare_search_net.run_phase5_comparison(
                    "train", 500040, 2, 60, 2, 30, 1.41, out,
                    environment=env, checkpoint_dir=ckpt, resume=True), load
            finally:
                compare_search_net._NET = None

    def test_changed_weights_are_rejected_before_running_pairs(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), \
                tempfile.TemporaryDirectory() as tmp:
            env = environment_info(1, _thread_env())
            out = os.path.join(tmp, "net.jsonl")
            a = self._checkpoint(tmp, "a", b"weights-A")
            b = self._checkpoint(tmp, "b", b"weights-B")
            self._first_half(out, env, a)
            with self.assertRaises(SearchRecordError):
                self._resume(out, env, b, _OtherNet())
            self.assertFalse(os.path.isdir(out + "-bundle"))

    def test_same_content_other_path_and_same_dir_resume_pass(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), \
                tempfile.TemporaryDirectory() as tmp:
            env = environment_info(1, _thread_env())
            out = os.path.join(tmp, "net.jsonl")
            a = self._checkpoint(tmp, "a", b"weights-A")
            same = self._checkpoint(tmp, "same", b"weights-A")
            self._first_half(out, env, a)
            row, unused = self._resume(out, env, same, _FakeValueNet())
            self.assertEqual(row["pairs"], 2)

    def test_modified_weights_in_same_dir_are_rejected(self):
        with mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}), \
                tempfile.TemporaryDirectory() as tmp:
            env = environment_info(1, _thread_env())
            out = os.path.join(tmp, "net.jsonl")
            a = self._checkpoint(tmp, "a", b"weights-A")
            self._first_half(out, env, a)
            with open(os.path.join(
                    a, compare_search_net.CHECKPOINT_WEIGHTS_NAME),
                    "wb") as handle:
                handle.write(b"weights-A-updated")
            with self.assertRaises(SearchRecordError):
                self._resume(out, env, a, _OtherNet())

def _wait_for_pair_line(path, timeout):
    """journal に pair 行が書かれるまで待つ(テスト内の完了順同期用)。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                if '"kind": "pair"' in handle.read():
                    return True
        except OSError:
            pass
        time.sleep(0.001)
    return False


class NetCompletionOrderTests(unittest.TestCase):

    def test_later_pair_is_journaled_before_earlier_pair_fails(self):
        import threading
        from multiprocessing.pool import ThreadPool
        with tempfile.TemporaryDirectory() as tmp:
            journal = search_records.PairJournal(
                os.path.join(tmp, "o.jsonl"), "k", "run", 0, 2)
            ready = threading.Event()
            finished = []

            def worker(task):
                if task[1] == 0:
                    self.assertTrue(ready.wait(10))
                    # ready は worker の return 前に立つため、後続ペアが親で
                    # journal に書かれたことを観測してから失敗させる。
                    self.assertTrue(_wait_for_pair_line(journal.path, 10))
                    raise RuntimeError("earlier pair failed")
                finished.append(task[1])
                ready.set()
                return ([1.0, 0.0], {}, [])

            pool = ThreadPool(2)
            try:
                with mock.patch.object(compare_search_net, "_run_pair",
                                       worker):
                    with self.assertRaises(RuntimeError):
                        compare_search_net.run_comparison(
                            "train", 0, 2, 60, 2, 30, 1.41, pool=pool,
                            journal=journal)
            finally:
                pool.terminate()
                pool.join()
            self.assertEqual(finished, [1])
            self.assertEqual(sorted(journal.open_resume()), [1])


class HashSeedCliContract(unittest.TestCase):
    """compare_search と同じ hash seed ブートストラップ契約。"""

    def test_unset_reexec_help(self):
        proc = _run_cli(["--help"], _cli_env())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage", proc.stdout)

    def test_explicit_zero_does_not_reexec(self):
        proc = _run_cli(["--help"], _cli_env(PYTHONHASHSEED="0"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage", proc.stdout)

    def test_explicit_nonzero_is_rejected(self):
        proc = _run_cli(["--help"], _cli_env(PYTHONHASHSEED="1"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PYTHONHASHSEED", proc.stderr)

    def test_bootstrap_marker_prevents_infinite_restart(self):
        proc = _run_cli(["--help"], _cli_env(
            **{compare_search_net._HASH_SEED_BOOTSTRAP_MARKER: "1"}))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("拒否", proc.stderr)


class ImportBoundaryTests(unittest.TestCase):

    def test_module_import_does_not_import_torch(self):
        """torch は遅延 import。torch の無い /usr/bin/python3 でも import できる。"""
        self.assertFalse(_TORCH_IMPORTED_AT_IMPORT)


class CheckpointDirValidation(unittest.TestCase):

    def test_validate_requires_gen532_meta_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                compare_search_net._validate_checkpoint_dir(tmp)
            with open(os.path.join(tmp, "gen532.meta.json"), "w",
                      encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaises(SystemExit):
                compare_search_net._validate_checkpoint_dir(tmp)
            with open(os.path.join(tmp, "gen532.weights.pt"), "w",
                      encoding="utf-8") as handle:
                handle.write("dummy")
            compare_search_net._validate_checkpoint_dir(tmp)


@unittest.skipUnless(_torch_available(), "torch not available")
class TorchCheckpointTests(unittest.TestCase):
    """gen532 相当の checkpoint ロード経路(_load_net)を torch 環境で検証する。"""

    @staticmethod
    def _make_checkpoint_dir(tmp, arch=None):
        from naeco.train import checkpoint as checkpoint_mod
        from naeco.train import encoding as enc
        if arch is None:
            arch = {
                "obs_dim": enc.obs_dim(2),
                "action_dim": enc.action_dim(),
                "hidden": [256, 256],
                "action_hidden": [128],
            }
        meta = checkpoint_mod.current_meta(2, arch)
        meta["generation"] = 532
        checkpoint_mod.save(os.path.join(tmp, "gen532.meta.json"), meta)
        return arch

    def test_load_net_from_fabricated_checkpoint(self):
        import torch
        from naeco.train import encoding as enc
        from naeco.train import net as net_mod
        with tempfile.TemporaryDirectory() as tmp:
            self._make_checkpoint_dir(tmp)
            net = net_mod.PolicyValueNet(
                enc.obs_dim(2), enc.action_dim(),
                hidden=(256, 256), action_hidden=(128,))
            torch.save(net.state_dict(),
                       os.path.join(tmp, "gen532.weights.pt"))
            loaded = compare_search_net._load_net(tmp)
            self.assertIsInstance(loaded, torch.nn.Module)
            self.assertFalse(loaded.training)

    def test_load_net_rejects_arch_mismatch(self):
        import torch
        from naeco.train import encoding as enc
        from naeco.train import net as net_mod
        with tempfile.TemporaryDirectory() as tmp:
            bad_arch = {
                "obs_dim": enc.obs_dim(2) + 1,
                "action_dim": enc.action_dim(),
                "hidden": [256, 256],
                "action_hidden": [128],
            }
            self._make_checkpoint_dir(tmp, arch=bad_arch)
            net = net_mod.PolicyValueNet(
                enc.obs_dim(2), enc.action_dim(),
                hidden=(256, 256), action_hidden=(128,))
            torch.save(net.state_dict(),
                       os.path.join(tmp, "gen532.weights.pt"))
            with self.assertRaises(ValueError):
                compare_search_net._load_net(tmp)

    def test_load_net_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_checkpoint_dir(tmp)
            with self.assertRaises(ValueError):
                compare_search_net._load_net(tmp)

    def test_load_net_rejects_corrupt_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "gen532.meta.json"), "w",
                      encoding="utf-8") as handle:
                handle.write("not json\n")
            with self.assertRaises(ValueError):
                compare_search_net._load_net(tmp)

    @unittest.skipUnless(os.environ.get("NAECO_GEN532_CHECKPOINT_DIR"),
                         "NAECO_GEN532_CHECKPOINT_DIR not set")
    def test_real_gen532_load(self):
        from naeco.train import encoding as enc
        net = compare_search_net._load_net(
            os.environ["NAECO_GEN532_CHECKPOINT_DIR"])
        self.assertEqual(net._obs_dim, enc.obs_dim(2))
        self.assertEqual(net._action_dim, enc.action_dim())


if __name__ == "__main__":
    unittest.main()
