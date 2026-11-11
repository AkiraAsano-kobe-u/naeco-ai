"""Phase 2.5 記録ハーネスの原子性・復元検証・決定論比較のテスト。"""
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import compare_search  # noqa: E402  (import 時に 4 thread env を "1" へ設定する)
import search_records  # noqa: E402
import decide_search  # noqa: E402
from search_records import (  # noqa: E402
    SearchRecordError, assert_thread_env, build_manifest, environment_info,
    find_deterministic_diffs, load_bundle, load_validated_bundles,
    regenerate_aggregates, validate_audit, validate_legacy_records,
    validate_resume_state, write_bundle,
)


def _thread_env():
    return dict((name, "1") for name in search_records.THREAD_ENV_NAMES)


_FIXTURE_CI_CACHE = {}


def _fixture_ci(pair_scores):
    """fixture ごとに実 bootstrap を繰り返さないための決定的キャッシュ。"""
    key = tuple(pair_scores)
    if key not in _FIXTURE_CI_CACHE:
        _FIXTURE_CI_CACHE[key] = search_records.recompute_ci(key)
    return _FIXTURE_CI_CACHE[key]


def _versions():
    from naeco.engine.cards import load_cards
    from naeco.engine.observe import OBS_VERSION
    from naeco.engine.state import API_VERSION, RULES_VERSION
    cards = load_cards()
    return {
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "api_version": API_VERSION,
        "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
    }


def make_bundle(group="dev", start_index=0, pairs=1, left="pimc",
                right="heuristic", budget=500, budget_seconds=None,
                horizon=2, scale=30, c=1.41, label=None, key=None):
    """bundle 検証を通過する合成記録を作る(エンジン対局はしない)。"""
    from naeco.engine.state import derive_seed, eval_seed
    label = label or "%s-vs-%s" % (left, right)
    key = key or label
    config = {
        "group": group, "start_index": start_index, "pairs": pairs,
        "left": left, "right": right, "budget": budget,
        "budget_seconds": budget_seconds, "horizon": horizon,
        "scale": scale, "c": c,
    }
    manifest = build_manifest(key, label, config, _versions(),
                              environment_info(1, _thread_env()))
    # 正式 protocol は Python 3.9 runtime を拘束するため、テスト bundle の
    # manifest は 3.9 系として扱う(実行インタプリタに依存させない)。
    manifest["python_version"] = "3.9.6"
    # 正式 protocol は python_hash_seed="0" を厳密要求する。
    manifest["python_hash_seed"] = "0"
    # 正式 csi protocol は /usr/bin/python3 を厳密要求するため fixture も同期。
    manifest["python_executable"] = "/usr/bin/python3"
    manifest["run_id"] = search_records.compute_run_id(manifest)
    games = []
    for offset in range(pairs):
        index = start_index + offset
        root = eval_seed(group, index)
        base = {
            "group": group, "index": index,
            "start_index": start_index, "pairs": pairs,
            "env_seed": derive_seed(root, "env", index),
            "setup_seed": derive_seed(root, "setup", index),
            "policy_seed_seat0": derive_seed(root, "policy", index, 0),
            "policy_seed_seat1": derive_seed(root, "policy", index, 1),
            "start_player": 0, "initial_money": [6, 5],
            "initial_deck": ["base:farm#01", "base:factory#01"],
            "initial_hands": [["base:realtor#01"], ["base:pioneer#01"]],
            "initial_state_hash": "fake-hash",
            "left": left, "right": right,
            "budget": budget, "budget_seconds": budget_seconds,
            "horizon": horizon, "scale": scale, "c": c,
            "key": key, "label": label, "run_id": manifest["run_id"],
            "manifest": manifest,
        }
        base.update(_versions())
        for variant in (0, 1):
            game = dict(base)
            game["seat_variant"] = variant
            game["left_seat"] = variant
            game["right_seat"] = 1 - variant
            game["result_left"] = 1.0 if variant == 0 else 0.5
            games.append(game)
    wins, draws, losses, pair_scores, point = \
        search_records.recompute_summary(games)
    expected_ci_low, expected_ci_high = _fixture_ci(pair_scores)
    row = dict(config)
    row.update(_versions())
    row.update({
        "wins": wins, "draws": draws, "losses": losses,
        "point_estimate": point, "ci_low": expected_ci_low,
        "ci_high": expected_ci_high,
        "pair_scores": pair_scores,
        "key": key, "label": label, "run_id": manifest["run_id"],
        "manifest": manifest,
    })
    return row, games, manifest, key


class BundleRoundTrip(unittest.TestCase):

    def test_bundle_is_complete_before_aggregate_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            # 原子確定前に summary だけ先に書く旧挙動が無いこと。
            self.assertFalse(os.path.exists(out))
            bundle = load_bundle(out, key)
            self.assertEqual(bundle["summary"]["run_id"], manifest["run_id"])
            self.assertEqual(len(bundle["games"]), 2)
            regenerate_aggregates(out, {key: bundle})
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(
                out.replace(".jsonl", "") + "-games.jsonl"))
            self.assertTrue(os.path.exists(out + ".manifest.jsonl"))

    def test_incomplete_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            os.remove(os.path.join(out + "-bundle", key + ".manifest.json"))
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)

    def test_summary_only_aggregate_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            with open(out, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": "ghost"}) + "\n")
            with self.assertRaises(SearchRecordError):
                validate_resume_state(out, ["ghost"])

    def test_manifest_mismatch_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            other, unused_g, unused_m, unused_k = make_bundle(budget=999)
            with self.assertRaises(SearchRecordError):
                validate_resume_state(out, [key], {key: other["manifest"]})

    def test_duplicate_detail_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            games_path = os.path.join(out + "-bundle", key + ".games.jsonl")
            with open(games_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(games[0]) + "\n")
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)

    def test_recompute_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            row["wins"] += 1
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)

    def test_paired_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            games[0]["env_seed"] += 1
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)

    def test_slash_key_roundtrips_without_filename_collision(self):
        """stage1 の `default/pimc/500` 形式 key を bundle で往復できる。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "stage1.jsonl")
            row, games, manifest, key = make_bundle(
                left="pimc", right="heuristic", label="default",
                key="default/pimc/500")
            write_bundle(out, key, row, games, manifest)
            bundle = load_bundle(out, key)
            self.assertEqual(bundle["summary"]["key"], key)
            bundles = load_validated_bundles(out)
            self.assertEqual(set(bundles), {key})
            regenerate_aggregates(out, bundles)
            self.assertEqual(set(load_validated_bundles(out)), {key})
            # `/` は可逆エンコードされ、衝突する別 key と区別できる。
            self.assertEqual(
                search_records.decode_key_filename(
                    search_records.encode_key_filename("a/b")), "a/b")
            self.assertNotEqual(
                search_records.encode_key_filename("a/b"),
                search_records.encode_key_filename("a%2Fb"))

    def test_invalid_result_left_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            games[0]["result_left"] = 0.7
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)
            for bad in (True, float("nan"), float("inf"), "1.0"):
                row2, games2, manifest2, key2 = make_bundle()
                games2[0]["result_left"] = bad
                with self.assertRaises(SearchRecordError):
                    search_records.recompute_summary(games2)

    def test_invalid_pair_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            row["pair_scores"] = [0.7]
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)
            row2, games2, manifest2, key2 = make_bundle()
            row2["pair_scores"] = [float("nan")]
            write_bundle(out, key2, row2, games2, manifest2)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key2)

    def test_ci_recompute_mismatch_is_rejected(self):
        """生成時と同じ bootstrap から外れた ci_low / ci_high は拒否する。"""
        for mutate in (lambda row: row.update({"ci_low": row["ci_low"] + 0.01}),
                       lambda row: row.update({"ci_high": row["ci_high"] - 0.01}),
                       lambda row: row.pop("ci_low", None),
                       lambda row: row.pop("ci_high", None),
                       lambda row: row.update({"ci_low": "0.5"}),
                       lambda row: row.update({"ci_high": float("nan")}),
                       lambda row: row.update({"ci_low": True})):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "run.jsonl")
                row, games, manifest, key = make_bundle()
                mutate(row)
                write_bundle(out, key, row, games, manifest)
                with self.assertRaises(SearchRecordError):
                    load_bundle(out, key)

    def test_ci_recompute_matches_generation_path(self):
        from naeco.ai.eval.harness import (
            BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap,
        )
        scores = [1.0, 1.0, 0.5, 0.0, 1.0, 1.0]
        self.assertEqual(
            search_records.recompute_ci(scores),
            cluster_bootstrap(scores, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED))

    def test_required_labels_match_canonical_matchup_table(self):
        self.assertEqual(
            search_records.REQUIRED_LABELS,
            tuple(search_records.REQUIRED_LABEL_MATCHUPS))

    def test_required_label_matchup_reversal_is_rejected(self):
        """pimc-vs-ismcts の left/right 反転は manifest 照合で拒否する。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle(
                left="ismcts", right="pimc", label="pimc-vs-ismcts",
                key="pimc-vs-ismcts")
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                load_bundle(out, key)
            with self.assertRaises(SearchRecordError):
                search_records.check_manifest(
                    manifest, expected_key=key,
                    expected_label="pimc-vs-ismcts")

    def test_required_label_one_side_tamper_is_rejected(self):
        """required label の left / right 片側改竄は拒否する。"""
        for label, left, right in (
                ("pimc-vs-heuristic", "ismcts", "heuristic"),
                ("ismcts-vs-heuristic", "ismcts", "pimc")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "run.jsonl")
                row, games, manifest, key = make_bundle(
                    left=left, right=right, label=label, key=label)
                write_bundle(out, key, row, games, manifest)
                with self.assertRaises(SearchRecordError):
                    load_bundle(out, key)
                with self.assertRaises(SearchRecordError):
                    search_records.check_manifest(
                        manifest, expected_key=key, expected_label=label)

    def test_resume_rejects_planned_extra_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle(key="pimc-vs-heuristic")
            write_bundle(out, key, row, games, manifest)
            row2, games2, manifest2, key2 = make_bundle(
                key="ismcts-vs-heuristic")
            write_bundle(out, key2, row2, games2, manifest2)
            with self.assertRaises(SearchRecordError):
                validate_resume_state(out, [key])

    def test_resume_rejects_aggregate_without_bundle_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            with open(out, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaises(SearchRecordError):
                validate_resume_state(out, ["pimc-vs-heuristic"])

    def test_invalid_seat_variant_raises_search_record_error(self):
        row, games, unused_m, unused_k = make_bundle()
        games[0]["seat_variant"] = "left"
        with self.assertRaises(SearchRecordError):
            search_records.recompute_summary(games)

    def test_manifest_requires_python_hash_seed(self):
        row, games, manifest, key = make_bundle()
        del manifest["python_hash_seed"]
        with self.assertRaises(SearchRecordError):
            search_records.check_manifest(manifest, expected_key=key)

    def test_manifest_requires_python_executable(self):
        row, games, manifest, key = make_bundle()
        del manifest["python_executable"]
        with self.assertRaises(SearchRecordError):
            search_records.check_manifest(manifest, expected_key=key)

    def test_protocol_rejects_wrong_python_executable(self):
        for executable in (None, "/usr/local/bin/python3"):
            rows = []
            for left, right in (("pimc", "heuristic"),
                                ("ismcts", "heuristic"),
                                ("pimc", "ismcts")):
                row, unused_games, manifest, unused_key = make_bundle(
                    group="train", start_index=1100, pairs=200, left=left,
                    right=right, budget=1000000, budget_seconds=2.0,
                    horizon=2, scale=30, c=1.41)
                if executable is None:
                    del manifest["python_executable"]
                else:
                    manifest["python_executable"] = executable
                manifest["run_id"] = search_records.compute_run_id(manifest)
                row["manifest"] = manifest
                row["run_id"] = manifest["run_id"]
                rows.append(row)
            # 正規 bundle を壊さず _check_protocol 単独で照合し、
            # 例外理由が protocol の executable 不一致であることを確認する。
            with self.assertRaises(SearchRecordError) as ctx:
                decide_search._check_protocol(
                    rows, decide_search.PROTOCOLS["phase2.5-wallclock"])
            self.assertIn("python_executable", str(ctx.exception))

    def test_environment_info_does_not_fake_zero_when_unset(self):
        old = os.environ.pop("PYTHONHASHSEED", None)
        try:
            info = environment_info(1, _thread_env())
            self.assertEqual(info["python_hash_seed"], "random")
        finally:
            if old is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = old

    def test_partial_bundle_without_manifest_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle(key="pimc-vs-heuristic")
            write_bundle(out, key, row, games, manifest)
            row2, games2, manifest2, key2 = make_bundle(
                key="ismcts-vs-heuristic")
            write_bundle(out, key2, row2, games2, manifest2)
            bundle_dir = out + "-bundle"
            summary_name, games_name, unused = \
                search_records.bundle_filenames(key2)
            os.remove(os.path.join(bundle_dir,
                                   search_records.bundle_filenames(key2)[2]))
            bundles = load_validated_bundles(out)
            self.assertEqual(set(bundles), {key})
            self.assertFalse(os.path.exists(os.path.join(bundle_dir,
                                                         summary_name)))
            self.assertFalse(os.path.exists(os.path.join(bundle_dir,
                                                         games_name)))
            # partial は再実行で原子再確定できる。
            write_bundle(out, key2, row2, games2, manifest2)
            self.assertEqual(set(load_validated_bundles(out)),
                             {key, key2})

    def test_partial_bundle_with_single_file_is_recoverable(self):
        """manifest なし・summary のみは正規クラッシュ形として cleanup できる。"""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            bundle_dir = out + "-bundle"
            summary_name, games_name, manifest_name = \
                search_records.bundle_filenames(key)
            os.remove(os.path.join(bundle_dir, manifest_name))
            os.remove(os.path.join(bundle_dir, games_name))
            self.assertEqual(load_validated_bundles(out), {})
            self.assertFalse(os.path.exists(os.path.join(bundle_dir,
                                                         summary_name)))
            # partial は再実行で原子再確定できる。
            write_bundle(out, key, row, games, manifest)
            self.assertEqual(set(load_validated_bundles(out)), {key})

    def test_corrupt_manifest_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            manifest_path = os.path.join(out + "-bundle",
                                         search_records.bundle_filenames(
                                             key)[2])
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write("broken json\n")
            with self.assertRaises(SearchRecordError):
                load_validated_bundles(out)


class WorkerEnvironment(unittest.TestCase):

    def test_all_four_thread_envs_must_be_one(self):
        effective = assert_thread_env(_thread_env())
        self.assertEqual(effective, _thread_env())
        bad = _thread_env()
        bad["OMP_NUM_THREADS"] = "2"
        with self.assertRaises(SearchRecordError):
            assert_thread_env(bad)


class DeterminismComparison(unittest.TestCase):

    def test_timing_and_process_fields_are_excluded(self):
        row_one, games_one, unused_m, key = make_bundle()
        row_two, games_two, unused_m2, unused_k = make_bundle()
        for row in (row_one, row_two):
            row.setdefault("wall_seconds", 1.0)
            row.setdefault("mean_think_ms", 10.0)
            row.setdefault("peak_rss_bytes", 1024)
        row_two["wall_seconds"] = row_one["wall_seconds"] + 12.5
        row_two["mean_think_ms"] = row_one["mean_think_ms"] + 1.0
        row_two["pid"] = 12345
        row_two["worker_thread_envs"] = [{"OMP_NUM_THREADS": "1"}]
        self.assertEqual(
            find_deterministic_diffs(row_one, row_two, games_one, games_two),
            [])

    def test_path_not_on_allowlist_is_compared(self):
        """部分一致時代なら見逃した合成 diff を allowlist 外 path として検出する。"""
        row_one, games_one, unused_m, key = make_bundle()
        row_two, games_two, unused_m2, unused_k = make_bundle()
        row_one["wall_seconds_planned"] = 1.0
        row_two["wall_seconds_planned"] = 2.0
        diffs = find_deterministic_diffs(row_one, row_two, games_one, games_two)
        self.assertTrue(any("wall_seconds_planned" in line for line in diffs))

    def test_allowlisted_nested_search_metrics_are_ignored(self):
        """search_metrics 内の方式別実行時計測は allowlist 済みで無視する。"""
        row_one, games_one, unused_m, key = make_bundle()
        row_two, games_two, unused_m2, unused_k = make_bundle()
        metrics = {
            "pimc": {
                "search_elapsed_ms": 1.0,
                "nodes_per_second": 10.0,
                "mean_think_ms": 1.0,
                "p95_think_ms": 2.0,
                "worker_thread_envs": [{"OMP_NUM_THREADS": "1"}],
            }
        }
        row_one["search_metrics"] = copy.deepcopy(metrics)
        row_two["search_metrics"] = copy.deepcopy(metrics)
        row_two["search_metrics"]["pimc"]["search_elapsed_ms"] = 999.0
        row_two["search_metrics"]["pimc"]["nodes_per_second"] = 999.0
        row_two["search_metrics"]["pimc"]["worker_thread_envs"] = [
            {"OMP_NUM_THREADS": "2"}]
        self.assertEqual(
            find_deterministic_diffs(row_one, row_two, games_one, games_two),
            [])

    @staticmethod
    def _replay_fixture():
        return {
            "actions": [
                ["place_worker", "base:farm#01", None, 0],
                ["select_build", "base:factory#01", "base:factory#01", 0],
            ],
            "state_hashes": ["hash-1", "hash-2"],
            "rng_states": ["rng-1", "rng-2"],
        }

    def test_replay_action_mismatch_is_detected(self):
        row_one, games_one, unused_m, unused_k = make_bundle()
        row_two, games_two, unused_m2, unused_k2 = make_bundle()
        for games in (games_one, games_two):
            for game in games:
                game["replay"] = copy.deepcopy(self._replay_fixture())
        games_two[0]["replay"]["actions"][0][0] = "sell_building"
        diffs = find_deterministic_diffs(row_one, row_two, games_one, games_two)
        self.assertTrue(
            any("replay" in line and "actions" in line for line in diffs),
            "expected replay.actions diff, got %r" % (diffs,))

    def test_replay_state_hash_mismatch_is_detected(self):
        row_one, games_one, unused_m, unused_k = make_bundle()
        row_two, games_two, unused_m2, unused_k2 = make_bundle()
        for games in (games_one, games_two):
            for game in games:
                game["replay"] = copy.deepcopy(self._replay_fixture())
        games_two[0]["replay"]["state_hashes"][1] = "mutated"
        diffs = find_deterministic_diffs(row_one, row_two, games_one, games_two)
        self.assertTrue(
            any("replay" in line and "state_hashes" in line for line in diffs),
            "expected replay.state_hashes diff, got %r" % (diffs,))

    def test_replay_rng_state_mismatch_is_detected(self):
        row_one, games_one, unused_m, unused_k = make_bundle()
        row_two, games_two, unused_m2, unused_k2 = make_bundle()
        for games in (games_one, games_two):
            for game in games:
                game["replay"] = copy.deepcopy(self._replay_fixture())
        games_two[0]["replay"]["rng_states"][0] = "mutated"
        diffs = find_deterministic_diffs(row_one, row_two, games_one, games_two)
        self.assertTrue(
            any("replay" in line and "rng_states" in line for line in diffs),
            "expected replay.rng_states diff, got %r" % (diffs,))

    def test_game_outcome_difference_is_detected(self):
        row_one, games_one, unused_m, key = make_bundle()
        row_two, games_two, unused_m2, unused_k = make_bundle()
        games_two[0]["result_left"] = 0.0
        diffs = find_deterministic_diffs(row_one, row_two, games_one, games_two)
        self.assertTrue(any("result_left" in line for line in diffs))

    def test_manifest_runtime_fields_are_not_excluded_from_diffs(self):
        """python_hash_seed / source_digest / version / run_id の差は diff になる。"""
        for field in ("python_hash_seed", "source_digest", "python_version",
                      "run_id", "python_executable"):
            row_one, games_one, manifest_one, unused_k = make_bundle()
            row_two, games_two, manifest_two, unused_k2 = make_bundle()
            manifest_two[field] = "different-" + field
            for row in (row_one, row_two):
                row["manifest"] = manifest_one if row is row_one \
                    else manifest_two
            diffs = find_deterministic_diffs(
                row_one, row_two, games_one, games_two)
            self.assertTrue(
                any(field in line for line in diffs),
                "expected diff for %s, got %r" % (field, diffs))


class Audit(unittest.TestCase):

    def test_aggregate_audit_sums_per_decision_fields(self):
        acc = {
            "samples_per_action": [{"a": 2, "b": 2}, {"a": 1, "b": 0}],
            "transitions_per_action": [{"a": 10, "b": 8}, {"a": 5, "b": 0}],
            "max_sample_gap": [0, 1],
            "zero_sample_actions": [0, 1],
            "decisions_with_zero_sample_actions": 1,
            "deadline_overrun_count": 3,
            "deadline_overrun_total_ms": 45.0,
            "deadline_overrun_max_ms": 20.0,
        }
        audit = search_records.aggregate_audit([acc])
        self.assertEqual(audit["max_sample_gap"], 1)
        self.assertEqual(audit["zero_sample_actions"], 1)
        self.assertEqual(audit["deadline_overrun_count"], 3)
        self.assertEqual(audit["action_transition_distribution"]["a"],
                         {"samples": 3, "transitions": 15})

    def test_validate_audit_refuses_pimc_gap_and_wall_zero(self):
        audit = {"max_sample_gap": 2, "zero_sample_actions": 0}
        with self.assertRaises(SearchRecordError):
            validate_audit("pimc", None, audit)
        audit = {"max_sample_gap": 0, "zero_sample_actions": 1}
        with self.assertRaises(SearchRecordError):
            validate_audit("ismcts", 5.0, audit)
        validate_audit("ismcts", 5.0,
                       {"max_sample_gap": 5, "zero_sample_actions": 0})
        validate_audit("pimc", None,
                       {"max_sample_gap": 1, "zero_sample_actions": 0})


class LegacyRecords(unittest.TestCase):

    def _write_legacy(self, tmp, rows, games):
        out = os.path.join(tmp, "legacy.jsonl")
        games_path = out.replace(".jsonl", "") + "-games.jsonl"
        with open(out, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with open(games_path, "w", encoding="utf-8") as handle:
            for game in games:
                handle.write(json.dumps(game) + "\n")
        return out

    def test_valid_legacy_records_pass(self):
        rows = []
        all_games = []
        for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                            ("pimc", "ismcts")):
            row, games, unused_m, unused_k = make_bundle(left=left, right=right)
            rows.append(row)
            all_games.extend(games)
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_legacy(tmp, rows, all_games)
            validate_legacy_records(out)

    def test_legacy_recompute_mismatch_is_rejected(self):
        row, games, unused_m, unused_k = make_bundle()
        row["point_estimate"] += 0.25
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_legacy(tmp, [row], games)
            with self.assertRaises(SearchRecordError):
                validate_legacy_records(out)

    def test_legacy_invalid_result_and_pair_scores_are_rejected(self):
        row, games, unused_m, unused_k = make_bundle()
        games[0]["result_left"] = 0.7
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_legacy(tmp, [row], games)
            with self.assertRaises(SearchRecordError):
                validate_legacy_records(out)
        row2, games2, unused_m2, unused_k2 = make_bundle()
        row2["pair_scores"] = [0.7]
        with tempfile.TemporaryDirectory() as tmp:
            out = self._write_legacy(tmp, [row2], games2)
            with self.assertRaises(SearchRecordError):
                validate_legacy_records(out)


class DecideSearch(unittest.TestCase):

    def test_duplicate_labels_are_rejected_before_dict_overwrite(self):
        """protocol 必須チェックより先に重複 label を _unique_required_labels で拒否する。"""
        rows = []
        for key in ("pimc-vs-heuristic", "dup"):
            row, unused_games, unused_m, unused_key = make_bundle(
                label="pimc-vs-heuristic", key=key)
            rows.append(row)
        with self.assertRaises(SearchRecordError) as ctx:
            decide_search._unique_required_labels(rows)
        self.assertIn("duplicate labels", str(ctx.exception))

    def test_three_labels_must_share_config_and_versions(self):
        """source_digest の label 間不一致は _check_shared_config が拒否する。"""
        rows = []
        for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                            ("pimc", "ismcts")):
            row, unused_games, manifest, unused_key = make_bundle(
                left=left, right=right)
            rows.append(row)
        rows[1]["manifest"]["source_digest"] = "different-digest"
        with self.assertRaises(SearchRecordError) as ctx:
            decide_search._check_shared_config(rows)
        self.assertIn("source_digest", str(ctx.exception))

    def test_new_format_requires_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(out)

    def test_protocol_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                                ("pimc", "ismcts")):
                row, games, manifest, key = make_bundle(
                    group="dev", start_index=0, pairs=1, left=left,
                    right=right)
                write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(out, protocol="phase2.5-dev")

    def test_new_format_rejects_label_manifest_version_mismatch(self):
        """cards_data_version の label 間不一致は _check_shared_config が拒否する。"""
        rows = []
        for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                            ("pimc", "ismcts")):
            row, unused_games, manifest, unused_key = make_bundle(
                left=left, right=right)
            rows.append(row)
        rows[1]["manifest"]["cards_data_version"] = "9.9.9"
        with self.assertRaises(SearchRecordError) as ctx:
            decide_search._check_shared_config(rows)
        self.assertIn("cards_data_version", str(ctx.exception))

    def test_protocol_rejects_non_39_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                                ("pimc", "ismcts")):
                row, games, manifest, key = make_bundle(
                    group="train", start_index=1100, pairs=200, left=left,
                    right=right, budget=1000000, budget_seconds=2.0,
                    horizon=2, scale=30, c=1.41)
                manifest["python_version"] = "3.12.7"
                manifest["run_id"] = search_records.compute_run_id(manifest)
                write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(out,
                                           protocol="phase2.5-wallclock")

    def test_protocol_rejects_hash_seed_not_zero(self):
        for seed in ("1", "random"):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "run.jsonl")
                for left, right in (("pimc", "heuristic"),
                                    ("ismcts", "heuristic"),
                                    ("pimc", "ismcts")):
                    row, games, manifest, key = make_bundle(
                        group="train", start_index=1100, pairs=200, left=left,
                        right=right, budget=1000000, budget_seconds=2.0,
                        horizon=2, scale=30, c=1.41)
                    manifest["python_hash_seed"] = seed
                    manifest["run_id"] = search_records.compute_run_id(
                        manifest)
                    write_bundle(out, key, row, games, manifest)
                with self.assertRaises(SearchRecordError):
                    decide_search.load_records(
                        out, protocol="phase2.5-wallclock")

    def test_old_artifact_range_does_not_pass_new_protocol(self):
        """旧 start3000/100・直前 start5000/700・a41 start4000/400 は拒否。"""
        for group, start_index, protocol in (
                ("dev", 3000, "phase2.5-dev"),
                ("train", 100, "phase2.5-wallclock"),
                ("dev", 4000, "phase2.5-dev"),
                ("train", 400, "phase2.5-wallclock"),
                ("dev", 5000, "phase2.5-dev"),
                ("train", 700, "phase2.5-wallclock")):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "run.jsonl")
                for left, right in (("pimc", "heuristic"),
                                    ("ismcts", "heuristic"),
                                    ("pimc", "ismcts")):
                    row, games, manifest, key = make_bundle(
                        group=group, start_index=start_index, pairs=200,
                        left=left, right=right, budget=1000000,
                        budget_seconds=2.0, horizon=2, scale=30, c=1.41)
                    write_bundle(out, key, row, games, manifest)
                with self.assertRaises(SearchRecordError):
                    decide_search.load_records(out, protocol=protocol)

    def test_three_labels_must_share_runtime_version(self):
        """python_version の label 間不一致は _check_shared_config が拒否する。"""
        rows = []
        for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                            ("pimc", "ismcts")):
            row, unused_games, manifest, unused_key = make_bundle(
                left=left, right=right)
            rows.append(row)
        rows[1]["manifest"]["python_version"] = "3.9.7"
        with self.assertRaises(SearchRecordError) as ctx:
            decide_search._check_shared_config(rows)
        self.assertIn("python_version", str(ctx.exception))

    def test_legacy_compares_top_level_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            games = []
            for left, right in (("pimc", "heuristic"), ("ismcts", "heuristic"),
                                ("pimc", "ismcts")):
                row, one_games, unused_m, unused_k = make_bundle(
                    left=left, right=right)
                rows.append(row)
                games.extend(one_games)
            rows[1]["cards_data_version"] = "9.9.9"
            out = os.path.join(tmp, "legacy.jsonl")
            games_path = out.replace(".jsonl", "") + "-games.jsonl"
            with open(out, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            with open(games_path, "w", encoding="utf-8") as handle:
                for game in games:
                    handle.write(json.dumps(game) + "\n")
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(out, legacy=True)

    def test_protocol_matching_records_pass(self):
        cases = (
            ("phase2.5-dev", "dev", 6000, 785, 4500, None),
            ("phase2.5-wallclock", "train", 1100, 200, 1000000, 2.0),
        )
        for protocol, group, start_index, pairs, budget, budget_seconds in cases:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "run.jsonl")
                for left, right in (("pimc", "heuristic"),
                                    ("ismcts", "heuristic"),
                                    ("pimc", "ismcts")):
                    row, games, manifest, key = make_bundle(
                        group=group, start_index=start_index, pairs=pairs,
                        left=left, right=right, budget=budget,
                        budget_seconds=budget_seconds, horizon=2, scale=30,
                        c=1.41)
                    write_bundle(out, key, row, games, manifest)
                rows = decide_search.load_records(out, protocol=protocol)
                self.assertEqual(len(rows), 3)

    def test_protocol_custom_is_not_a_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games, manifest, key = make_bundle()
            write_bundle(out, key, row, games, manifest)
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(out, protocol="custom")
            with self.assertRaises(SystemExit):
                decide_search.main([out, "--protocol", "custom"])

    def test_legacy_and_protocol_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaises(SearchRecordError):
                decide_search.load_records(path, legacy=True,
                                           protocol="phase2.5-dev")
            with self.assertRaises(SystemExit):
                decide_search.main([path, "--legacy",
                                    "--protocol", "phase2.5-dev"])

    def test_decide_secondary_at_point_estimate_half_goes_to_tiebreak(self):
        # primary はどちらも全勝で Holm 棄却される。secondary は点推定 0.5 の
        # データで、exact 条件付き符号検定では両側 p=0.125 の不有意となり方向を
        # 確定できない。タイブレーク(p95)が ismcts を採用する点まで検証し、
        # 分岐を壊して既定順で pimc を返す実装はここで落ちる。
        rows = []
        for left, right, p95 in (("pimc", "heuristic", 10.0),
                                 ("ismcts", "heuristic", 5.0)):
            rows.append({
                "label": "%s-vs-%s" % (left, right),
                "left": left, "right": right,
                "point_estimate": 1.0,
                "pair_scores": [1.0] * 10,
                "p95_think_ms": p95, "peak_rss_bytes": 10,
            })
        rows.append({
            "label": "pimc-vs-ismcts",
            "left": "pimc", "right": "ismcts",
            "point_estimate": 0.5,
            "pair_scores": [1.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5,
                            1.0, 0.5],
            "p95_think_ms": 5.0, "peak_rss_bytes": 10,
        })
        choice, trace = decide_search.decide(rows)
        self.assertEqual(choice, "ismcts")
        self.assertTrue(any("両側 p=0.12500" in line for line in trace))
        self.assertTrue(any("点推定=0.5000" in line for line in trace))
        self.assertTrue(any("タイブレークへ" in line for line in trace))


class RealTinyComparison(unittest.TestCase):

    def test_run_finalize_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run.jsonl")
            row, games = compare_search.run_comparison(
                "dev", 0, 1, "pimc", "heuristic", 500, 2, 30, 1.41)
            # 既定は replay 記録なし。既存の記録フォーマットを維持する。
            self.assertTrue(all("replay" not in game for game in games))
            key = "pimc-vs-heuristic"
            config = {
                "group": "dev", "start_index": 0, "pairs": 1,
                "left": "pimc", "right": "heuristic", "budget": 500,
                "budget_seconds": None, "horizon": 2, "scale": 30, "c": 1.41,
            }
            compare_search._finalize_bundle(
                out, key, key, row, games, config,
                compare_search._versions(
                    __import__("naeco.engine.cards", fromlist=["load_cards"])
                    .load_cards()),
                environment_info(1, _thread_env()))
            bundle = load_bundle(out, key)
            self.assertEqual(bundle["summary"]["wins"] + bundle["summary"]["losses"]
                             + bundle["summary"]["draws"], 2)
            self.assertIn("nodes_per_second", bundle["summary"])
            self.assertIn("action_transition_distribution", bundle["summary"])

    def test_replay_audit_records_actions_states_and_rng(self):
        row, games = compare_search.run_comparison(
            "dev", 0, 1, "pimc", "heuristic", 300, 2, 30, 1.41,
            replay_audit=True)
        self.assertEqual(len(games), 2)
        for game in games:
            replay = game["replay"]
            self.assertGreater(len(replay["actions"]), 0)
            self.assertEqual(len(replay["actions"]),
                             len(replay["state_hashes"]))
            self.assertEqual(len(replay["actions"]),
                             len(replay["rng_states"]))
            self.assertEqual(
                len(replay["actions"][0]), 4,
                "action は kind/target/instance/goods_count の4要素")

    def test_direct_matchup_reports_both_methods_nodes_per_second(self):
        row, unused_games = compare_search.run_comparison(
            "dev", 0, 1, "pimc", "ismcts", 300, 2, 30, 1.41)
        metrics = row["search_metrics"]
        self.assertIn("pimc", metrics)
        self.assertIn("ismcts", metrics)
        self.assertGreater(metrics["ismcts"]["total_nodes"], 0)
        self.assertIsNotNone(metrics["ismcts"]["nodes_per_second"])
        self.assertIn("nodes_per_second", metrics["pimc"])
        self.assertIn("action_transition_distribution", metrics["ismcts"])
        self.assertIn("deadline_overrun_count", metrics["pimc"])

    def test_stage_two_group_train_flows_to_config_details_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "train.jsonl")
            env = environment_info(1, _thread_env())
            compare_search.run_stage_two(
                1, 300, 2, 30, 1.41, out, pool=None, budget_seconds=None,
                group="train", start_index=400,
                only={"pimc-vs-heuristic"}, environment=env)
            bundle = load_bundle(out, "pimc-vs-heuristic")
            self.assertEqual(bundle["summary"]["group"], "train")
            self.assertEqual(bundle["summary"]["start_index"], 400)
            self.assertEqual(bundle["games"][0]["group"], "train")
            self.assertEqual(bundle["manifest"]["group"], "train")


class DeterminismReport(unittest.TestCase):

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_check_determinism_group_train_start_100_writes_report(self):
        # import 直呼び単体テストは正式 CLI 契約(PYTHONHASHSEED=0)を明示する。
        # process の hash seed を変える意図ではなく、compare の正式前提を表す。
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det.json")
            compare_search.run_determinism_check(
                pairs=1, budget=300, horizon=2, scale=30, c=1.41,
                budget_seconds=None, start_index=400,
                matchups={"pimc-vs-heuristic"}, records=report, workers=2,
                group="train")
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["all_labels_matched"])
            self.assertEqual(payload["group"], "train")
            self.assertEqual(payload["start_index"], 400)

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_check_determinism_replay_audit_matches_across_workers(self):
        """worker=1/2 で action 列・state_hash 列・RNG state まで完全一致する。"""
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det-replay.json")
            compare_search.run_determinism_check(
                pairs=1, budget=300, horizon=2, scale=30, c=1.41,
                budget_seconds=None, start_index=450,
                matchups={"pimc-vs-heuristic"}, records=report, workers=2,
                group="train", replay_audit=True)
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["replay_audit"])
            self.assertTrue(payload["all_labels_matched"])
            self.assertIn("summary.wall_seconds",
                          payload["excluded_fields"])


class CliGuardrails(unittest.TestCase):

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_wallclock_requires_transition_cap(self):
        # 同上: compare_search.main の import 直呼びに正式契約を明示する。
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--budget-seconds", "2.0",
                                 "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--budget-seconds", "2.0",
                                 "--budget", "999999", "--execute"])
        code = compare_search.main(["--stage", "2", "--budget-seconds", "2.0",
                                    "--budget", "1000000"])
        self.assertEqual(code, 0)

    def test_check_determinism_requires_pairs_and_known_matchup(self):
        with self.assertRaises(SystemExit):
            compare_search.main(["--check-determinism", "--workers", "2",
                                 "--records", "x.json", "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--check-determinism", "--pairs", "1",
                                 "--workers", "2", "--records", "x.json",
                                 "--matchups", "not-a-matchup", "--execute"])

    def test_check_determinism_requires_start_index_and_rejects_wallclock(self):
        with self.assertRaises(SystemExit):
            compare_search.main(["--check-determinism", "--pairs", "1",
                                 "--workers", "2", "--records", "x.json",
                                 "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--check-determinism", "--pairs", "1",
                                 "--start-index", "100", "--workers", "2",
                                 "--records", "x.json", "--budget-seconds",
                                 "2.0", "--execute"])

    def test_stage2_requires_explicit_start_index(self):
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--pairs", "1",
                                 "--budget", "4500", "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--pairs", "1",
                                 "--budget", "4500", "--start-index", "3000",
                                 "--matchups", "not-a-matchup", "--execute"])

    def test_stage1_rejects_wallclock_and_invalid_budget_values(self):
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "1", "--budget-seconds", "2.0",
                                 "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "1", "--budgets", "0",
                                 "--execute"])
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--pairs", "0",
                                 "--start-index", "3000", "--execute"])

    def test_resume_requires_records(self):
        with self.assertRaises(SystemExit):
            compare_search.main(["--stage", "2", "--pairs", "1",
                                 "--start-index", "3000", "--resume",
                                 "--execute"])

    def test_check_determinism_rejects_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det.json")
            with open(report, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaises(SystemExit):
                compare_search.run_determinism_check(
                    pairs=1, budget=300, horizon=2, scale=30, c=1.41,
                    budget_seconds=None, start_index=400,
                    matchups={"pimc-vs-heuristic"}, records=report, workers=2,
                    group="train")

    @mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False)
    def test_cli_check_determinism_group_train_start_100(self):
        # 同上: compare_search.main の import 直呼びに正式契約を明示する。
        with tempfile.TemporaryDirectory() as tmp:
            report = os.path.join(tmp, "det.json")
            code = compare_search.main([
                "--stage", "2", "--pairs", "1", "--budget", "300",
                "--check-determinism", "--group", "train",
                "--start-index", "400", "--matchups", "pimc-vs-heuristic",
                "--workers", "2", "--records", report, "--execute"])
            self.assertEqual(code, 0)
            with open(report, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["all_labels_matched"])
            self.assertEqual(payload["group"], "train")


if __name__ == "__main__":
    unittest.main()
