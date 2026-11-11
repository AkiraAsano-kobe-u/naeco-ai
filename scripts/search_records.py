"""Phase 2.5 探索比較記録の共通形式・原子書込・復元検証。

label 単位の bundle(summary 1行 + 全 game details + manifest 1件)を
一時ファイルへ完全書込 → fsync → os.replace で確定する。
集約 JSONL(summary / -games / .manifest)は bundle からの導出物であり、
`--resume` や `--regenerate` で bundle から再生成する。
"""
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naeco.engine.state import derive_seed, eval_seed  # noqa: E402

SCHEMA_VERSION = "phase2.5-record/v1"
THREAD_ENV_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# required label と (left, right) の正規対応表。新形式 bundle では
# manifest/config の意味対応をこの表へ厳格に照合する。
REQUIRED_LABEL_MATCHUPS = {
    "pimc-vs-heuristic": ("pimc", "heuristic"),
    "ismcts-vs-heuristic": ("ismcts", "heuristic"),
    "pimc-vs-ismcts": ("pimc", "ismcts"),
}
REQUIRED_LABELS = tuple(REQUIRED_LABEL_MATCHUPS)

# summary / detail / manifest のどれでも同じ意味を持つ設定キー。
CONFIG_KEYS = ("group", "start_index", "pairs", "left", "right", "budget",
               "budget_seconds", "horizon", "scale", "c")
VERSION_KEYS = ("cards_sha256", "cards_data_version", "api_version",
                "rules_version", "obs_version")
# run_id は worker 数・環境・実行時刻を含めない。worker=1/N 決定論検査や
# 同一設定の再実行で共通になるように、実験の同一性を決める鍵だけから導く。
RUN_ID_KEYS = ("schema_version",) + CONFIG_KEYS + VERSION_KEYS + (
    "source_digest", "python_hash_seed", "python_version",
    "python_executable")

# 決定論比較から外す path の明示 allowlist。部分一致ではなく、
# この一覧に完全一致する path だけを比較対象外にする(allowlist 外は全比較)。
# リスト要素の任意インデックスは `*` で表す。理由は各要素のコメントに残す。
DETERMINISM_EXCLUDED_PATHS = (
    # summary 直下の実行時計測: wall-clock 由来で再現しない。
    ("summary", "wall_seconds"),
    ("summary", "mean_think_ms"),
    ("summary", "p95_think_ms"),
    ("summary", "transitions_per_second"),
    ("summary", "nodes_per_second"),
    ("summary", "peak_rss_bytes"),
    # summary 直下のプロセス/並列環境情報。
    ("summary", "pid"),
    ("summary", "rss"),
    ("summary", "workers"),
    ("summary", "worker_thread_envs"),
    ("summary", "thread_env"),
    # summary.manifest の実行環境。worker 数・スレッド環境は実行ごとに変わり、
    # git 状態は記録だけして決定論比較から外す。
    ("summary", "manifest", "workers"),
    ("summary", "manifest", "thread_env"),
    ("summary", "manifest", "git_revision"),
    ("summary", "manifest", "git_dirty"),
    # search_metrics 内の方式別実行時計測(集計値のうち wall-clock 由来の項目)。
    ("summary", "search_metrics", "*", "search_elapsed_ms"),
    ("summary", "search_metrics", "*", "nodes_per_second"),
    ("summary", "search_metrics", "*", "mean_think_ms"),
    ("summary", "search_metrics", "*", "p95_think_ms"),
    ("summary", "search_metrics", "*", "worker_thread_envs"),
    # games 側にも同じ実行時依存項目が現れ得る場合に備えた明示 path。
    ("games", "*", "wall_seconds"),
    ("games", "*", "mean_think_ms"),
    ("games", "*", "p95_think_ms"),
    ("games", "*", "transitions_per_second"),
    ("games", "*", "nodes_per_second"),
    ("games", "*", "peak_rss_bytes"),
    ("games", "*", "pid"),
    ("games", "*", "rss"),
    ("games", "*", "workers"),
    ("games", "*", "worker_thread_envs"),
    ("games", "*", "thread_env"),
    ("games", "*", "manifest", "workers"),
    ("games", "*", "manifest", "thread_env"),
    ("games", "*", "manifest", "git_revision"),
    ("games", "*", "manifest", "git_dirty"),
)


def _dotted_path(path):
    """path タプルを report 用のドット記法へ変換する(`*` は `[]`)。"""
    return ".".join("[]" if part == "*" else str(part) for part in path)


def _parse_dotted_path(spec):
    """ドット記法(`games[].wall_seconds` 等)を path タプルへ戻す。"""
    parts = []
    for chunk in spec.split("."):
        if chunk.endswith("[]"):
            parts.append("*")
            chunk = chunk[:-2]
        parts.append(chunk)
    return tuple(parts)


def _normalize_excluded_paths(excluded):
    """excluded 引数を path タプルの set へ正規化する(文字列・タプル両対応)。"""
    paths = set()
    for spec in excluded:
        if isinstance(spec, tuple):
            paths.add(tuple(str(part) for part in spec))
        elif isinstance(spec, str):
            paths.add(_parse_dotted_path(spec))
        else:
            raise SearchRecordError(
                "excluded path must be tuple or dotted string, got %r"
                % (spec,))
    return paths


def _path_matches(candidate, allowed):
    """candidate path が allowlist path に一致するか。`*` は任意1要素に一致。"""
    if len(candidate) != len(allowed):
        return False
    return all(part == "*" or part == other
               for part, other in zip(allowed, candidate))


# report の excluded_fields 互換形式(ドット記法の文字列)。
DETERMINISM_EXCLUDED = tuple(_dotted_path(path)
                             for path in DETERMINISM_EXCLUDED_PATHS)


class SearchRecordError(Exception):
    """記録の形式・整合・原子性に関する検証失敗。"""


def canonical_json(value):
    """決定論比較・hash 用の正準 JSON バイト列。"""
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _source_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for sub in ("naeco", "scripts"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                name for name in dirnames if name != "__pycache__")
            for name in sorted(filenames):
                if name.endswith(".py"):
                    paths.append(os.path.relpath(
                        os.path.join(dirpath, name), root))
    for name in ("data/cards.json", "pyproject.toml"):
        if os.path.exists(os.path.join(root, name)):
            paths.append(name)
    return sorted(paths)


def source_digest():
    """実験結果に効き得るソースとカードデータの正準ダイジェスト。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    digest = hashlib.sha256()
    for rel in _source_files():
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        with open(os.path.join(root, rel), "rb") as handle:
            digest.update(handle.read())
        digest.update(b"\x00")
    return digest.hexdigest()


def git_metadata():
    """HEAD と dirty 判定。git が使えない環境では None にして source_digest に頼る。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    revision = None
    dirty = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return revision, dirty
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root,
            stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except Exception:
        dirty = None
    return revision, dirty


def environment_info(workers, thread_env):
    """manifest に載せる環境情報。"""
    # 探索コアの set/dict 反復は hash seed に依存するため、結果決定入力として
    # 実際の環境値を文字列で記録する。未設定を後付けで "0" へ偽装しない。
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        hash_seed = "random"
    revision, dirty = git_metadata()
    return {
        "workers": workers,
        "thread_env": dict((name, thread_env.get(name))
                           for name in THREAD_ENV_NAMES),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_hash_seed": hash_seed,
        "git_revision": revision,
        "git_dirty": dirty,
        "source_digest": source_digest(),
    }


def compute_run_id(manifest):
    """環境・worker 数に依存しない実験同一性の指紋。"""
    payload = dict((key, manifest[key]) for key in RUN_ID_KEYS
                   if key in manifest)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def build_manifest(key, label, config, versions, environment):
    """summary/detail 双方へ載せる共通 manifest を組み立てる。"""
    manifest = {"schema_version": SCHEMA_VERSION, "key": key, "label": label}
    for name in CONFIG_KEYS:
        manifest[name] = config[name]
    for name in VERSION_KEYS:
        manifest[name] = versions[name]
    for name in ("workers", "thread_env", "python_version",
                 "python_executable", "python_hash_seed", "git_revision",
                 "git_dirty", "source_digest"):
        manifest[name] = environment[name]
    manifest["run_id"] = compute_run_id(manifest)
    return manifest


def check_manifest(manifest, expected_key=None, expected_label=None):
    if not isinstance(manifest, dict):
        raise SearchRecordError("manifest is not an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SearchRecordError(
            "unsupported schema_version: %r" % manifest.get("schema_version"))
    required = CONFIG_KEYS + VERSION_KEYS + ("key", "label", "run_id",
                                             "workers", "thread_env",
                                             "python_version",
                                             "python_executable",
                                             "python_hash_seed",
                                             "source_digest")
    missing = [name for name in required if name not in manifest]
    if missing:
        raise SearchRecordError("manifest lacks %r" % (missing,))
    if not isinstance(manifest["python_hash_seed"], str) \
            or not manifest["python_hash_seed"]:
        raise SearchRecordError(
            "manifest python_hash_seed must be a non-empty string")
    if not isinstance(manifest["python_executable"], str) \
            or not manifest["python_executable"]:
        raise SearchRecordError(
            "manifest python_executable must be a non-empty string")
    if expected_key is not None and manifest["key"] != expected_key:
        raise SearchRecordError(
            "manifest key %r does not match bundle %r"
            % (manifest["key"], expected_key))
    if expected_label is not None and manifest["label"] != expected_label:
        raise SearchRecordError(
            "manifest label %r does not match expected %r"
            % (manifest["label"], expected_label))
    canonical = REQUIRED_LABEL_MATCHUPS.get(expected_label)
    if canonical is not None:
        left, right = canonical
        if manifest.get("left") != left or manifest.get("right") != right:
            raise SearchRecordError(
                "manifest label %r requires left/right %r/%r, got %r/%r"
                % (expected_label, left, right,
                   manifest.get("left"), manifest.get("right")))
    if compute_run_id(manifest) != manifest["run_id"]:
        raise SearchRecordError(
            "manifest run_id does not match its own contents")


def bundle_dir_for(out_path):
    return out_path + "-bundle"


_SAFE_FILENAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def encode_key_filename(key):
    """bundle ファイル名用の key エンコード。`/` 等も一意かつ可逆に扱う。"""
    out = []
    for byte in key.encode("utf-8"):
        char = chr(byte)
        if char in _SAFE_FILENAME_CHARS:
            out.append(char)
        else:
            out.append("%%%02X" % byte)
    return "".join(out)


def decode_key_filename(encoded):
    """`encode_key_filename` の逆変換。衝突しない可逆形式を保証する。"""
    parts = encoded.split("%")
    out = bytearray()
    for index, part in enumerate(parts):
        if index == 0:
            out.extend(part.encode("ascii"))
            continue
        if len(part) < 2:
            raise SearchRecordError("invalid encoded bundle key: %r" % encoded)
        try:
            out.append(int(part[:2], 16))
        except ValueError:
            raise SearchRecordError(
                "invalid encoded bundle key: %r" % encoded)
        out.extend(part[2:].encode("ascii"))
    return out.decode("utf-8")


def bundle_filenames(key):
    encoded = encode_key_filename(key)
    return (encoded + ".summary.jsonl", encoded + ".games.jsonl",
            encoded + ".manifest.json")


def _fsync_write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _json_line(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n"


def write_bundle(out_path, key, summary_row, game_records, manifest):
    """label 単位 bundle を一時書込→fsync→os.replace で原子的に確定する。"""
    if not out_path:
        return
    check_manifest(manifest, expected_key=key)
    bundle_dir = bundle_dir_for(out_path)
    os.makedirs(bundle_dir, exist_ok=True)
    summary_name, games_name, manifest_name = bundle_filenames(key)
    tmp_dir = tempfile.mkdtemp(prefix=".bundle-", dir=bundle_dir)
    try:
        tmp_summary = os.path.join(tmp_dir, "summary.jsonl")
        tmp_games = os.path.join(tmp_dir, "games.jsonl")
        tmp_manifest = os.path.join(tmp_dir, "manifest.json")
        _fsync_write(tmp_summary, _json_line(summary_row))
        _fsync_write(tmp_games, "".join(_json_line(rec)
                                        for rec in game_records))
        _fsync_write(tmp_manifest, _json_line(manifest))
        # manifest を最後に置くことで「確定のコミットマーカー」にする。
        os.replace(tmp_summary, os.path.join(
            bundle_dir, summary_name))
        os.replace(tmp_games, os.path.join(
            bundle_dir, games_name))
        os.replace(tmp_manifest, os.path.join(
            bundle_dir, manifest_name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


JOURNAL_SCHEMA = "phase2.5-pair-journal/v1"


class PairJournal(object):
    """label 単位の完了ペア逐次ジャーナル(JSONL 追記、1 行 1 ペア)。

    長い評価が途中で止まっても完了済みペアを失わないための再開用ログ。
    先頭行の header に run 条件の指紋(manifest.run_id・replay_audit・
    評価モデルの内容ハッシュ)を持ち、
    --resume で条件が一致しないジャーナルは拒否する。最終成果は従来どおり
    `write_bundle` が確定する。ジャーナルは確定後に `discard` で消す。
    """

    def __init__(self, out_path, key, run_id, start_index, pairs,
                 replay_audit=False, model_fingerprint=None):
        self.key = key
        # 評価モデル(重み+meta)の内容ハッシュ。run_id は重みを含まないため、
        # header 条件に入れて別モデルの対局が混ざる resume を拒否する。
        self.model_fingerprint = model_fingerprint
        self.run_id = run_id
        self.start_index = start_index
        self.pairs = pairs
        self.replay_audit = bool(replay_audit)
        self.directory = journal_dir_for(out_path)
        self.path = os.path.join(self.directory,
                                 encode_key_filename(key) + ".pairs.jsonl")

    def _header(self):
        return {"kind": "header", "schema": JOURNAL_SCHEMA, "key": self.key,
                "run_id": self.run_id, "start_index": self.start_index,
                "pairs": self.pairs, "replay_audit": self.replay_audit,
                "model_fingerprint": self.model_fingerprint}

    def _fsync_append(self, text):
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def open_new(self):
        """新規ジャーナルを作る。残骸があれば『1回だけ実行』保護のため拒否。"""
        if os.path.exists(self.path):
            raise SearchRecordError(
                "pair journal already exists: %s (--resume で続行するか、"
                "退避してから再実行する)" % self.path)
        os.makedirs(self.directory, exist_ok=True)
        self._fsync_append(_json_line(self._header()))

    def open_resume(self):
        """条件一致を検証して完了ペア {index: entry} を返す。無ければ新規。"""
        if not os.path.exists(self.path):
            self.open_new()
            return {}
        with open(self.path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        lines = raw.split("\n")
        torn = lines.pop()          # 末尾が改行で終わらない = 書込途中の torn 行
        entries = []
        for number, line in enumerate(lines, 1):
            try:
                entries.append(json.loads(line))
            except ValueError as exc:
                raise SearchRecordError(
                    "pair journal %s line %d is corrupt: %s"
                    % (self.path, number, exc))
        if torn.strip():
            # 強制停止で途中までしか書けなかった最終行。完了していないので捨てる。
            keep = raw[: len(raw) - len(torn)]
            with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(keep)
                handle.flush()
                os.fsync(handle.fileno())
        if not entries:
            # header すら書けていない: 何も完了していない。
            os.remove(self.path)
            self.open_new()
            return {}
        header, pair_entries = entries[0], entries[1:]
        expected = self._header()
        if not isinstance(header, dict) or header != expected:
            raise SearchRecordError(
                "pair journal %s does not match this run's conditions "
                "(journal header %r != expected %r)"
                % (self.path, header, expected))
        done = {}
        for entry in pair_entries:
            if not isinstance(entry, dict) or entry.get("kind") != "pair" \
                    or entry.get("run_id") != self.run_id:
                raise SearchRecordError(
                    "pair journal %s has a foreign entry" % self.path)
            index = entry.get("index")
            low = self.start_index
            if not isinstance(index, int) or isinstance(index, bool) \
                    or not low <= index < low + self.pairs:
                raise SearchRecordError(
                    "pair journal %s has out-of-range index %r"
                    % (self.path, index))
            if index in done:
                raise SearchRecordError(
                    "pair journal %s has duplicate index %d"
                    % (self.path, index))
            for name in ("games", "acc", "records"):
                if name not in entry:
                    raise SearchRecordError(
                        "pair journal %s index %d lacks %s"
                        % (self.path, index, name))
            done[index] = entry
        return done

    def append(self, index, games, acc, records):
        self._fsync_append(_json_line({
            "kind": "pair", "run_id": self.run_id, "index": index,
            "games": games, "acc": acc, "records": records}))

    def discard(self):
        """bundle 確定後にジャーナルを消す(残っても resume は bundle を優先)。"""
        try:
            os.remove(self.path)
            os.rmdir(self.directory)
        except OSError:
            pass


def journal_dir_for(out_path):
    return out_path + "-journal"


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recompute_summary(games):
    """詳細行だけから W/D/L・pair_scores・点推定を再計算する。"""
    for game in games:
        validate_result_left(game.get("result_left"))
    wins = sum(1 for game in games if game["result_left"] == 1.0)
    draws = sum(1 for game in games if game["result_left"] == 0.5)
    losses = sum(1 for game in games if game["result_left"] == 0.0)
    by_index = {}
    for game in games:
        seat_variant = game.get("seat_variant")
        if not isinstance(seat_variant, int) or isinstance(seat_variant, bool):
            raise SearchRecordError(
                "seat_variant must be an integer, got %r" % (seat_variant,))
        by_index.setdefault(game["index"], {})[seat_variant] = \
            game["result_left"]
    pair_scores = [(by_index[index][0] + by_index[index][1]) / 2.0
                   for index in sorted(by_index)]
    point = (sum(pair_scores) / len(pair_scores)) if pair_scores else 0.0
    return wins, draws, losses, pair_scores, point


def recompute_ci(pair_scores):
    """summary の CI を生成時と同じ bootstrap 経路で決定論的に再計算する。"""
    from naeco.ai.eval.harness import (
        BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, cluster_bootstrap,
    )
    return cluster_bootstrap(pair_scores, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)


def validate_result_left(value):
    """局結果が有限かつ厳密に {0.0, 0.5, 1.0} であることを要求する。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchRecordError("result_left must be 0.0/0.5/1.0, got %r"
                                % (value,))
    if not math.isfinite(value) or value not in (0.0, 0.5, 1.0):
        raise SearchRecordError("result_left must be 0.0/0.5/1.0, got %r"
                                % (value,))


def validate_pair_scores(pair_scores):
    """pair_scores が有限かつ [0,1] の数値列であることを要求する。"""
    if not isinstance(pair_scores, list):
        raise SearchRecordError("pair_scores must be a list")
    for value in pair_scores:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SearchRecordError("pair_scores must be finite [0,1] values")
        if not math.isfinite(value) or not (0.0 <= value <= 1.0):
            raise SearchRecordError("pair_scores must be finite [0,1] values")


def _config_matches(manifest, value, where):
    for name in CONFIG_KEYS:
        if value.get(name) != manifest[name]:
            raise SearchRecordError(
                "%s %s=%r does not match manifest %r"
                % (where, name, value.get(name), manifest[name]))


def validate_pair_consistency(games, group):
    """ペア内 seed・初期状態の一致と、seed 派生式の再計算を検証する。"""
    by_index = {}
    for game in games:
        by_index.setdefault(game["index"], []).append(game)
    for index, pair in sorted(by_index.items()):
        if len(pair) != 2:
            raise SearchRecordError(
                "index %s has %d games, expected 2" % (index, len(pair)))
        by_seat = dict((game["seat_variant"], game) for game in pair)
        if sorted(by_seat) != [0, 1]:
            raise SearchRecordError(
                "index %s seat variants %r, expected [0, 1]"
                % (index, sorted(by_seat)))
        seat0, seat1 = by_seat[0], by_seat[1]
        if seat0["left_seat"] != 0 or seat0["right_seat"] != 1:
            raise SearchRecordError(
                "index %s seat_variant 0 seat mapping is %r/%r"
                % (index, seat0["left_seat"], seat0["right_seat"]))
        if seat1["left_seat"] != 1 or seat1["right_seat"] != 0:
            raise SearchRecordError(
                "index %s seat_variant 1 seat mapping is %r/%r"
                % (index, seat1["left_seat"], seat1["right_seat"]))
        for name in ("env_seed", "setup_seed", "initial_deck",
                     "initial_hands", "initial_money", "initial_state_hash",
                     "start_player", "policy_seed_seat0", "policy_seed_seat1"):
            if seat0[name] != seat1[name]:
                raise SearchRecordError(
                    "index %s paired games disagree on %s" % (index, name))
        root = eval_seed(group, index)
        expected = {
            "env_seed": derive_seed(root, "env", index),
            "setup_seed": derive_seed(root, "setup", index),
            "policy_seed_seat0": derive_seed(root, "policy", index, 0),
            "policy_seed_seat1": derive_seed(root, "policy", index, 1),
        }
        for name, want in expected.items():
            if seat0[name] != want:
                raise SearchRecordError(
                    "index %s %s %r does not match derived %r"
                    % (index, name, seat0[name], want))


def assert_thread_env(env=None):
    """worker の 4 スレッド環境変数実効値を返し、すべて "1" を要求する。"""
    if env is None:
        env = os.environ
    effective = {}
    for name in THREAD_ENV_NAMES:
        value = env.get(name)
        if value != "1":
            raise SearchRecordError(
                "thread env %s=%r must be '1'" % (name, value))
        effective[name] = value
    return effective


def validate_label_bundle(key, row, games, manifest):
    """1 bundle の manifest・行数・重複・再計算・ペア整合を完全検証する。"""
    label = manifest["label"]
    check_manifest(manifest, expected_key=key, expected_label=label)
    canonical = REQUIRED_LABEL_MATCHUPS.get(label)
    if canonical is not None:
        for name, value, want in (("left", row.get("left"), canonical[0]),
                                  ("right", row.get("right"), canonical[1])):
            if value != want:
                raise SearchRecordError(
                    "key %s label %r requires %s=%r, got %r"
                    % (key, label, name, want, value))
    if row.get("key") != key:
        raise SearchRecordError("summary key %r != %r" % (row.get("key"), key))
    if row.get("label") != label:
        raise SearchRecordError("summary label %r != %r"
                                % (row.get("label"), label))
    if row.get("run_id") != manifest["run_id"]:
        raise SearchRecordError("summary run_id does not match manifest")
    if canonical_json(row.get("manifest")) != canonical_json(manifest):
        raise SearchRecordError("summary manifest does not match bundle")
    _config_matches(manifest, row, "summary")
    pair_scores = row.get("pair_scores")
    validate_pair_scores(pair_scores)
    if len(pair_scores) != manifest["pairs"]:
        raise SearchRecordError(
            "key %s summary has %d pair_scores, expected %d"
            % (key, len(row.get("pair_scores", [])), manifest["pairs"]))
    if len(games) != 2 * manifest["pairs"]:
        raise SearchRecordError(
            "key %s has %d details, expected %d"
            % (key, len(games), 2 * manifest["pairs"]))
    seen = set()
    for game in games:
        if game.get("key") != key:
            raise SearchRecordError("detail key %r != %r"
                                    % (game.get("key"), key))
        if game.get("label") != label:
            raise SearchRecordError("detail label %r != %r"
                                    % (game.get("label"), label))
        if game.get("run_id") != manifest["run_id"]:
            raise SearchRecordError("detail run_id does not match manifest")
        if canonical_json(game.get("manifest")) != canonical_json(manifest):
            raise SearchRecordError("detail manifest does not match bundle")
        _config_matches(manifest, game, "detail")
        pair_key = (game["index"], game["seat_variant"])
        if pair_key in seen:
            raise SearchRecordError("duplicate detail %r in key %s"
                                    % (pair_key, key))
        seen.add(pair_key)
    index_range = set(range(manifest["start_index"],
                            manifest["start_index"] + manifest["pairs"]))
    if set(game["index"] for game in games) != index_range:
        raise SearchRecordError(
            "key %s index set %r does not match range %s"
            % (key, sorted(set(game["index"] for game in games)),
               sorted(index_range)))
    wins, draws, losses, pair_scores, point = recompute_summary(games)
    if wins + draws + losses != 2 * manifest["pairs"]:
        raise SearchRecordError(
            "key %s W+D+L=%d does not equal 2*pairs=%d"
            % (key, wins + draws + losses, 2 * manifest["pairs"]))
    if (wins, draws, losses) != (row["wins"], row["draws"], row["losses"]):
        raise SearchRecordError(
            "key %s W/D/L recompute %s != %s"
            % (key, (wins, draws, losses),
               (row["wins"], row["draws"], row["losses"])))
    if pair_scores != row["pair_scores"]:
        raise SearchRecordError("key %s pair_scores recompute mismatch" % key)
    if point != row["point_estimate"]:
        raise SearchRecordError("key %s point_estimate recompute mismatch"
                                % key)
    expected_ci_low, expected_ci_high = recompute_ci(pair_scores)
    for name, value, expected in (("ci_low", row.get("ci_low"),
                                   expected_ci_low),
                                  ("ci_high", row.get("ci_high"),
                                   expected_ci_high)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SearchRecordError(
                "key %s %s must be a finite number, got %r"
                % (key, name, value))
        if not math.isfinite(value):
            raise SearchRecordError(
                "key %s %s must be finite, got %r" % (key, name, value))
        if value != expected:
            raise SearchRecordError(
                "key %s %s recompute %r != %r"
                % (key, name, value, expected))
    validate_pair_consistency(games, manifest["group"])


def load_bundle(out_path, key):
    """1 bundle を読み、完全性を検証して返す。"""
    bundle_dir = bundle_dir_for(out_path)
    summary_name, games_name, manifest_name = bundle_filenames(key)
    summary_path = os.path.join(bundle_dir, summary_name)
    games_path = os.path.join(bundle_dir, games_name)
    manifest_path = os.path.join(bundle_dir, manifest_name)
    missing = [os.path.basename(path)
               for path in (summary_path, games_path, manifest_path)
               if not os.path.exists(path)]
    if missing:
        raise SearchRecordError("incomplete bundle for %s: missing %s"
                                % (key, ", ".join(missing)))
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as exc:
        raise SearchRecordError(
            "bundle %s manifest is corrupt: %s" % (key, exc))
    rows = _read_jsonl(summary_path)
    if len(rows) != 1:
        raise SearchRecordError(
            "bundle %s summary has %d rows, expected 1" % (key, len(rows)))
    games = _read_jsonl(games_path)
    validate_label_bundle(key, rows[0], games, manifest)
    return {"summary": rows[0], "games": games, "manifest": manifest}


def load_validated_bundles(out_path):
    """bundle ディレクトリ全体を検証する。不一致・不完全・重複は拒否する。"""
    bundle_dir = bundle_dir_for(out_path)
    if not os.path.isdir(bundle_dir):
        raise SearchRecordError("no bundle directory: %s" % bundle_dir)
    suffixes = (".summary.jsonl", ".games.jsonl", ".manifest.json")
    keys = sorted(set(
        decode_key_filename(name[: -len(suffix)])
        for name in os.listdir(bundle_dir)
        for suffix in suffixes if name.endswith(suffix)))
    bundles = {}
    partial = []
    for key in keys:
        summary_name, games_name, manifest_name = bundle_filenames(key)
        summary_path = os.path.join(bundle_dir, summary_name)
        games_path = os.path.join(bundle_dir, games_name)
        manifest_path = os.path.join(bundle_dir, manifest_name)
        has_summary = os.path.exists(summary_path)
        has_games = os.path.exists(games_path)
        has_manifest = os.path.exists(manifest_path)
        if not has_manifest:
            # manifest は bundle の唯一のコミットマーカー。replace 順
            # summary→games→manifest の途中クラッシュでは manifest なしで
            # summary のみ・games のみ・両方のいずれも残り得る。これらは
            # すべて未確定 partial として cleanup し、resume を可能にする。
            # 未知ファイルや manifest あり不整合は従来どおり fail-closed。
            if has_summary or has_games:
                partial.append(key)
                continue
        bundles[key] = load_bundle(out_path, key)
    for key in partial:
        summary_name, games_name, unused_manifest = bundle_filenames(key)
        for name in (summary_name, games_name):
            path = os.path.join(bundle_dir, name)
            if os.path.exists(path):
                os.remove(path)
    return bundles


def _write_aggregate(path, lines):
    if not path:
        return
    tmp_path = os.path.join(
        os.path.dirname(os.path.abspath(path)),
        ".%s.tmp-%d" % (os.path.basename(path), os.getpid()))
    try:
        _fsync_write(tmp_path, "".join(lines))
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_report(path, payload):
    """検査 report を一時書込 → fsync → os.replace で原子的に保存する。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, ".%s.tmp-%d"
                            % (os.path.basename(path), os.getpid()))
    try:
        _fsync_write(tmp_path,
                     json.dumps(payload, ensure_ascii=True, sort_keys=True,
                                indent=2) + "\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _aggregate_paths(out_path):
    return (out_path,
            out_path.replace(".jsonl", "") + "-games.jsonl",
            out_path + ".manifest.jsonl")


def _check_aggregate_keys(out_path, bundle_keys):
    summary_path, games_path, manifest_path = _aggregate_paths(out_path)
    if not os.path.exists(summary_path):
        return
    stored_rows = _read_jsonl(summary_path)
    stored_keys = set(row.get("key") for row in stored_rows)
    if stored_keys != bundle_keys:
        raise SearchRecordError(
            "aggregate keys %r do not match bundle keys %r"
            % (sorted(stored_keys), sorted(bundle_keys)))
    all_keys = [row.get("key") for row in stored_rows]
    duplicates = sorted(set(key for key in all_keys
                            if all_keys.count(key) > 1))
    if duplicates:
        raise SearchRecordError(
            "aggregate summary has duplicate keys: %r" % duplicates)


def regenerate_aggregates(out_path, bundles):
    """bundle から既存の summary / -games / .manifest 形式を再生成する。"""
    if not out_path:
        return
    _check_aggregate_keys(out_path, set(bundles))
    summary_path, games_path, manifest_path = _aggregate_paths(out_path)
    summary_lines = []
    games_lines = []
    manifest_lines = []
    for key in sorted(bundles):
        bundle = bundles[key]
        summary_lines.append(_json_line(bundle["summary"]))
        manifest_lines.append(_json_line(bundle["manifest"]))
        for game in sorted(bundle["games"],
                           key=lambda item: (item["index"],
                                             item["seat_variant"])):
            games_lines.append(_json_line(game))
    _write_aggregate(summary_path, summary_lines)
    _write_aggregate(games_path, games_lines)
    _write_aggregate(manifest_path, manifest_lines)


def aggregate_audit(accs):
    """複数ペアの探索統計 acc から監査指標を集約する。"""
    gaps = []
    zero = 0
    transitions_by_action = {}
    counts_by_action = {}
    sample_totals_by_action = {}
    deadline_count = 0
    deadline_total_ms = 0.0
    deadline_max_ms = 0.0
    zero_decisions = 0
    for acc in accs:
        samples = acc.get("samples_per_action", [])
        transitions = acc.get("transitions_per_action", [])
        gap_values = acc.get("max_sample_gap", [])
        zero_values = acc.get("zero_sample_actions", [])
        for decision_samples, decision_transitions in zip(samples, transitions):
            for action, used in decision_transitions.items():
                counts_by_action[action] = counts_by_action.get(action, 0) + 1
                transitions_by_action[action] = \
                    transitions_by_action.get(action, 0) + used
            for action, count in decision_samples.items():
                sample_totals_by_action[action] = \
                    sample_totals_by_action.get(action, 0) + count
        # コアが per-decision 監査を追加していない統計でも動くように、
        # あれば新フィールドを、なければ per-action から導出する。
        if gap_values or zero_values:
            gaps.extend(gap_values)
            zero += sum(zero_values)
        else:
            for decision_samples in samples:
                values = list(decision_samples.values())
                if values:
                    gaps.append(max(values) - min(values))
                zero += sum(1 for value in values if value == 0)
        zero_decisions += int(
            acc.get("decisions_with_zero_sample_actions", 0) or 0)
        deadline_count += int(acc.get("deadline_overrun_count", 0) or 0)
        deadline_total_ms += float(
            acc.get("deadline_overrun_total_ms", 0.0) or 0.0)
        deadline_max_ms = max(
            deadline_max_ms, float(
                acc.get("deadline_overrun_max_ms", 0.0) or 0.0))
    distribution = dict(
        (action, {"samples": sample_totals_by_action.get(action, 0),
                  "transitions": transitions_by_action[action]})
        for action in sorted(counts_by_action))
    return {
        "max_sample_gap": max(gaps) if gaps else 0,
        "zero_sample_actions": zero,
        "decisions_with_zero_sample_actions": zero_decisions,
        "action_transition_distribution": distribution,
        "deadline_overrun_count": deadline_count,
        "deadline_overrun_total_ms": deadline_total_ms,
        "deadline_overrun_max_ms": deadline_max_ms,
    }


def validate_audit(left, budget_seconds, audit):
    """監査指標が許容範囲外なら結果の確定を拒否する。"""
    if budget_seconds is not None:
        if audit["zero_sample_actions"] > 0:
            raise SearchRecordError(
                "wall-clock experiment has actions with 0 samples "
                "(zero_sample_actions=%d); refusing to finalize"
                % audit["zero_sample_actions"])
        return
    if left == "pimc" and audit["zero_sample_actions"] > 0:
        raise SearchRecordError(
            "PIMC left decision actions with 0 samples "
            "(zero_sample_actions=%d); refusing to finalize"
            % audit["zero_sample_actions"])
    if left == "pimc" and audit["max_sample_gap"] > 1:
        raise SearchRecordError(
            "PIMC transition-budget sample gap %d > 1; refusing to finalize"
            % audit["max_sample_gap"])


def validate_resume_state(out_path, planned_keys, expected_manifests=None):
    """resume 前に既存成果物が bundle 由来かつ検証済みであることを確認する。"""
    bundle_dir = bundle_dir_for(out_path)
    if not os.path.isdir(bundle_dir):
        aggregate_exists = any(
            os.path.exists(path) for path in _aggregate_paths(out_path))
        if aggregate_exists:
            raise SearchRecordError(
                "bundle directory missing but aggregate files exist: %s"
                % out_path)
        return {}
    bundles = load_validated_bundles(out_path)
    planned = set(planned_keys)
    extra = sorted(set(bundles) - planned)
    if extra:
        raise SearchRecordError(
            "existing bundles outside planned keys: %r" % extra)
    summary_path = _aggregate_paths(out_path)[0]
    if os.path.exists(summary_path):
        stored_keys = set(row.get("key")
                          for row in _read_jsonl(summary_path))
        missing_bundles = sorted(stored_keys - set(bundles))
        if missing_bundles:
            raise SearchRecordError(
                "summary-only rows without valid bundles: %r"
                % missing_bundles)
    if expected_manifests:
        for key in planned:
            bundle = bundles.get(key)
            if bundle is None:
                continue
            expected = expected_manifests.get(key)
            if expected is None:
                continue
            if canonical_json(bundle["manifest"]) != canonical_json(expected):
                raise SearchRecordError(
                    "existing bundle %s manifest does not match this run"
                    % key)
    return bundles


def validate_legacy_records(out_path):
    """bundle なしの旧 JSONL 形式を、保存可能な範囲で厳格検証する。"""
    summary_path, games_path, unused_manifest = _aggregate_paths(out_path)
    if not os.path.exists(summary_path):
        raise SearchRecordError("no summary file: %s" % summary_path)
    if not os.path.exists(games_path):
        raise SearchRecordError("no details file: %s" % games_path)
    rows = _read_jsonl(summary_path)
    games = _read_jsonl(games_path)
    labels = [row.get("label") for row in rows]
    if any(label is None for label in labels):
        raise SearchRecordError("summary rows without label")
    duplicates = sorted(set(label for label in labels
                            if labels.count(label) > 1))
    if duplicates:
        raise SearchRecordError("duplicate summary labels: %r" % duplicates)
    by_label = dict((row["label"], row) for row in rows)
    if not set(REQUIRED_LABELS) <= set(by_label):
        raise SearchRecordError(
            "legacy records lack required labels: %r"
            % (sorted(set(REQUIRED_LABELS) - set(by_label)),))
    for label, row in by_label.items():
        label_games = [game for game in games
                       if game.get("label") == label]
        pairs = row.get("pairs")
        if not isinstance(pairs, int) or pairs <= 0:
            raise SearchRecordError("legacy summary %s lacks pairs" % label)
        validate_pair_scores(row.get("pair_scores"))
        if len(row.get("pair_scores", [])) != pairs:
            raise SearchRecordError(
                "legacy %s has %d pair_scores, expected %d"
                % (label, len(row.get("pair_scores", [])), pairs))
        if len(label_games) != 2 * pairs:
            raise SearchRecordError(
                "legacy %s has %d details, expected %d"
                % (label, len(label_games), 2 * pairs))
        seen = set()
        for game in label_games:
            pair_key = (game.get("index"), game.get("seat_variant"))
            if pair_key in seen:
                raise SearchRecordError(
                    "legacy duplicate detail %r in %s" % (pair_key, label))
            seen.add(pair_key)
        wins, draws, losses, pair_scores, point = recompute_summary(label_games)
        if wins + draws + losses != 2 * pairs:
            raise SearchRecordError(
                "legacy %s W+D+L=%d does not equal 2*pairs=%d"
                % (label, wins + draws + losses, 2 * pairs))
        if (wins, draws, losses) != (row.get("wins"), row.get("draws"),
                                     row.get("losses")):
            raise SearchRecordError(
                "legacy %s W/D/L recompute mismatch" % label)
        if pair_scores != row.get("pair_scores"):
            raise SearchRecordError(
                "legacy %s pair_scores recompute mismatch" % label)
        if point != row.get("point_estimate"):
            raise SearchRecordError(
                "legacy %s point_estimate recompute mismatch" % label)
        validate_pair_consistency(label_games, row.get("group"))


def _filter_deterministic(value, excluded, path=()):
    """allowlist に完全一致する path だけを取り除く。

    キー名の部分一致は行わない。リスト要素は任意インデックスを表す `*` を
    path に積み、`games[0].manifest.workers` 等は `("games", "*", ...)` で
    allowlist 照合する。
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child = path + (key,)
            if any(_path_matches(child, allowed)
                   for allowed in excluded):
                continue
            out[key] = _filter_deterministic(item, excluded, child)
        return out
    if isinstance(value, list):
        return [_filter_deterministic(item, excluded, path + ("*",))
                for item in value]
    return value


def _diff_paths(left, right, path=""):
    out = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                out.append("%s.%s: only in second" % (path, key))
            elif key not in right:
                out.append("%s.%s: only in first" % (path, key))
            else:
                out.extend(_diff_paths(left[key], right[key],
                                       "%s.%s" % (path, key)))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append("%s: length %d != %d" % (path, len(left), len(right)))
        else:
            for i, (one, two) in enumerate(zip(left, right)):
                out.extend(_diff_paths(one, two, "%s[%d]" % (path, i)))
    elif left != right:
        out.append("%s: %r != %r" % (path or "<root>", left, right))
    return out[:5]


def find_deterministic_diffs(row_one, row_two, games_one, games_two,
                             excluded=DETERMINISM_EXCLUDED_PATHS):
    """worker=1/N 比較。明示 allowlist に載った path だけ除外し、他は全比較する。"""
    excluded_paths = tuple(_normalize_excluded_paths(excluded))
    first = _filter_deterministic(
        {"summary": row_one, "games": games_one}, excluded_paths)
    second = _filter_deterministic(
        {"summary": row_two, "games": games_two}, excluded_paths)
    first["games"] = sorted(
        first["games"], key=lambda item: (item["index"], item["seat_variant"]))
    second["games"] = sorted(
        second["games"], key=lambda item: (item["index"], item["seat_variant"]))
    return _diff_paths(first, second)
