"""チェックポイント互換キーの保存と拒否(Phase 3 設計書 §4.4 / §14.2 / §15 / core-api §8.3)。

標準ライブラリのみを使い、torch は import しない。重みファイル(.pt)は
torch 境界(naeco/train/net.py / ppo.py)が別途管理し、本モジュールは
メタ JSON だけで互換性を検証できる構造にする。

設計書 §15.1(Phase 3A ゲート2回目の高重大度指摘に対する改訂)の4要点:

1. `expected` は dict をやめ、専用の不変型 `_ExpectedMeta` にする。公開
   ファクトリは `expected_meta(num_players, model_arch)` だけであり、
   保存メタ(dict)からは構築できない。`load` は専用型以外を受け取ったら
   TypeError にする。
2. A 表のキーは `load` 側が現在の実装から組み立てる(`_current_a_values`)。
   呼び出し側が値を渡せるのはファイルから知りようがない `num_players` と
   `model_arch` の2つだけ。`rules_version` / `cards_sha256` /
   `obs_encoder_version` などは呼び出し側が指定できない。
3. `current_meta` の `extra` は B 表キーのみ許可し、A 表キーや内部キーが
   含まれていたら ValueError にする。
4. 内部マーカー(14.2 の `_naeco_expected_marker`)は廃止する。型で担保する
   ので不要であり、保存メタへ漏れる余地も消える。

B 表は train_ppo の再開状態(next_episode_index / start_index)を記録できる
よう拡張した。「記録するが拒否しないキー」という B 表の意味論のままであり、
load は B 表を検査しないため旧チェックポイントの読込互換には影響しない。

設計書 §16(ゲート3回目の指摘)の改訂:

5. A 表の照合は値だけでなく型も一致を要求する(_strict_equal)。Python では
   True == 1 / 1.0 == 1 / 2.0 == 2 が成立するため、単純な等値比較では
   JSON 上の型偽装(checkpoint_schema_version: true 等)を素通りさせる。
   model_arch はキー集合・値の型・リストの長さと要素の型まで再帰的に
   照合し、正規化(キャスト)はしない。
6. `expected_meta` は model_arch を生成時に防御的コピーする(§16.2)。
   expected 作成後に呼び出し側が元の dict を書き換えても照合へ影響しない。

Phase 4 の改訂(原子的書込と防御的読込):

7. `save` は同一ディレクトリの一時ファイルへ JSON を書いて fsync し、
   os.replace() で原子的に置換する。SIGKILL・ディスクフル・クラッシュでも
   最終パスに「完全な旧内容」または「完全な新内容」だけが存在し、途切れた
   JSON は決して現れない。失敗時は一時ファイルを後始末し meta_path を汚さ
   ない。書込形式(sort_keys / ensure_ascii / 末尾改行 / LF)は不変。
8. `load` は json.load を包み、壊れた/途切れた JSON を
   IncompatibleCheckpoint へ変換する。json.JSONDecodeError /
   UnicodeDecodeError / ValueError を外へ漏らさない。

Phase 4 の改訂(チェックポイントディレクトリの排他ロック):

9. 複数の train_ppo ジョブが同一 --checkpoint-dir を指して互いの
   チェックポイントを上書きし合う残留リスク(#4)を塞ぐため、
   `CheckpointLock` を提供する。ディレクトリ内の固定名ロックファイル
   (.lock)を開いて fcntl.flock(LOCK_EX | LOCK_NB) で取得し、取得できな
   ければ CheckpointLockError を raise する。flock はプロセス死亡
   (SIGKILL / os._exit 含む)時にカーネルが自動解放する。fcntl は guarded
   import とし、無い環境(Windows 等)ではロックが no-op(無効化)になるだけ
   で、本モジュールの import と既存機能は壊れない。
"""

import copy
import json
import os
import platform
import tempfile

try:
    import fcntl
except ImportError:  # Windows 等、fcntl の無い環境。ロックは no-op になる
    fcntl = None

from naeco.engine import cards as cards_mod
from naeco.engine import observe as observe_mod
from naeco.engine import state as state_mod
from naeco.train import encoding
from naeco.train import encoding_multi
from naeco.train import reward

CHECKPOINT_SCHEMA_VERSION = 1

# A表: 不一致なら読込を拒否するキー(core-api §8.3 / 設計書 §4.4)。
A_KEYS = (
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

# B表: 記録するが拒否しないキー(再現・デバッグ用)。
# train_ppo の再開状態(next_episode_index / start_index)を記録できるよう
# 拡張した(設計書 §15.1 第三「extra は B 表キーのみ許可」を満たすため。
# 記録専用キーという意味論は不変)。
B_KEYS = (
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


class IncompatibleCheckpoint(Exception):
    """チェックポイントの互換キーが現在の実装と一致しない。"""


# チェックポイントディレクトリ内の固定名ロックファイル(設計書/Phase 4)。
# 世代チェックポイントの gen パターン(^gen(\d+)\.meta\.json$ 等)に一致
# しないため、_list_generations / _prune_checkpoints はこれを世代や孤児
# ファイルとして誤認識・削除しない(テストで固定)。
LOCK_FILENAME = ".lock"


class CheckpointLockError(Exception):
    """チェックポイントディレクトリの排他ロックを取得できなかった。

    他プロセス(または同一プロセスの別 fd)がロックを保持している。
    """


class CheckpointLock(object):
    """チェックポイントディレクトリの排他ロック(fcntl.flock、Phase 4)。

    複数の train_ppo ジョブが同一 --checkpoint-dir を指して互いの
    チェックポイントを上書きし合う残留リスク(#4)を塞ぐ。ロックは
    checkpoint ディレクトリ内の固定名ロックファイル(LOCK_FILENAME =
    ".lock")を開き、fcntl.flock(LOCK_EX | LOCK_NB) で取得する。

    - 他プロセス(または同一プロセスの別 fd)が保持中なら LOCK_NB が
      EWOULDBLOCK で失敗し、CheckpointLockError を raise する。
    - flock のロックはカーネルが開いた fd と結び付くため、プロセスが死ぬ
      (SIGKILL / os._exit(0) 含む)と自動解放される。残置ロックで後続の
      再開ジョブを妨げない。
    - fcntl の無い非Unix環境(Windows 等)では acquire() は no-op になり、
      ロックは無効化されるが import と既存機能は壊れない。
    - ロックファイルは世代チェックポイントではなく、_list_generations /
      _prune_checkpoints の gen パターン照合(^gen(\\d+)\\.meta\\.json$ 等)
      に一致しないため、世代や孤児ファイルとして誤認識・削除されない。

    使い方:

        lock = CheckpointLock(checkpoint_dir)
        lock.acquire()
        try:
            ...
        finally:
            lock.release()

    with 文でも使える(acquire は __enter__、release は __exit__)。
    """

    def __init__(self, checkpoint_dir):
        self._path = os.path.join(checkpoint_dir, LOCK_FILENAME)
        self._fh = None

    @property
    def path(self):
        """ロックファイルの絶対パス。"""
        return self._path

    @property
    def held(self):
        """この CheckpointLock がロックを保持しているか。"""
        return self._fh is not None

    def acquire(self):
        """排他ロックを取得する。取得できなければ CheckpointLockError。

        ロックファイルは開くだけで内容は書かない(flock は inode に対して
        効くため、空のまま使う)。この CheckpointLock が既に保持していれば
        何もせず self を返す(再入)。fcntl の無い環境では no-op(ロック
        無効化)になり、失敗しない。
        """
        if self._fh is not None:
            return self
        if fcntl is None:
            return self
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fh = open(self._path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise CheckpointLockError(
                "cannot acquire exclusive lock %s: %s; another train_ppo "
                "job may be holding the same --checkpoint-dir"
                % (self._path, exc))
        self._fh = fh
        return self

    def release(self):
        """ロックを解放する(LOCK_UN + close)。べき等。

        flock は fd が閉じられた時点でもカーネルが解放するが、明示的に
        LOCK_UN してから閉じる(解放を他の参照の終了に依存させない)。
        """
        if self._fh is None:
            return
        fh = self._fh
        self._fh = None
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass  # 解放失敗でも close でカーネルが解放する
        fh.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


def acquire_checkpoint_lock(checkpoint_dir):
    """CheckpointLock を生成して取得し、取得済みのインスタンスを返す。

    取得できなければ CheckpointLockError。戻り値は with 文で使い、終了時に
    release() する。
    """
    lock = CheckpointLock(checkpoint_dir)
    lock.acquire()
    return lock


class _ExpectedMeta(object):
    """load() の expected 用の専用の不変型(設計書 §15.1 第一)。

    `expected_meta(num_players, model_arch)` ファクトリからだけ作る。
    A 表の値は持たない — 保持するのはファイルから知りようがない
    `num_players` と `model_arch` の2つだけで、残りの A 表キーは load 側が
    現在の実装から組み立てる(§15.1 第二)。dict である保存メタからは構築
    できず、属性の書換えは AttributeError で拒否される(不変)。
    """

    __slots__ = ("num_players", "model_arch")

    def __init__(self, num_players, model_arch):
        object.__setattr__(self, "num_players", num_players)
        object.__setattr__(self, "model_arch", model_arch)

    def __setattr__(self, name, value):
        raise AttributeError("_ExpectedMeta is immutable")

    def __repr__(self):
        return "_ExpectedMeta(num_players=%r, model_arch=%r)" % (
            self.num_players, self.model_arch)


def expected_meta(num_players, model_arch):
    """load() の expected を作るファクトリ(設計書 §15.1 第一)。

    呼び出し側が値を渡せるのは num_players と model_arch の2つだけ。
    rules_version / cards_sha256 / obs_encoder_version などの A 表キーは
    load 側が現在の実装から組み立てるため、ここで指定できない(§15.1 第二)。
    保存メタ(dict)を expected として使う経路は存在しない。
    """
    # Phase 10 D2: 契約は人数から内部で選ぶ。対応外の人数(1, 5 以上)は
    # 契約が無いので受け付けない(旧契約を全人数へ流用する例外は作らない)。
    if not isinstance(num_players, int) or isinstance(num_players, bool) \
            or num_players not in encoding_multi.ALL_PLAYERS:
        raise ValueError(
            "num_players must be an integer in 2..4, got %r"
            % (num_players,))
    if not isinstance(model_arch, dict):
        raise ValueError(
            "model_arch must be a dict, got %s" % type(model_arch).__name__)
    # §16.2: 防御的コピーを取ってから保持する。生成後に呼び出し側が元の
    # dict を書き換えても、expected が照合に使う model_arch へ影響しない。
    return _ExpectedMeta(num_players, copy.deepcopy(model_arch))


def _current_a_values(num_players, model_arch):
    """現在の実装から A 表13キーの値を組み立てる(設計書 §15.1 第二)。

    load の照合はこの値で行う。呼び出し側が指定できるのは num_players と
    model_arch だけで、他はすべて現在の実装(定数・cards・encoder の
    再計算)から導出される。
    """
    cards = cards_mod.load_cards()
    # Phase 10 D2: encoder / reward の契約は人数から選ぶ。n=2 は旧 encoding /
    # REWARD_VERSION そのもの(値・計算は従来と同一)、n=3,4 は encoding_multi
    # (obs 2.0.0)と reward 2.0.0。呼び出し側は版を指定できない。
    contract = encoding_multi.contract_for(num_players)
    obs_encoder_sha256, action_encoder_sha256 = contract.encoder_sha256()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "rules_version": state_mod.RULES_VERSION,
        "cards_data_version": cards.data_version,
        "cards_sha256": cards.sha256,
        "api_version": state_mod.API_VERSION,
        "obs_version": observe_mod.OBS_VERSION,
        "obs_encoder_version": contract.obs_encoder_version,
        "action_encoder_version": contract.action_encoder_version,
        "obs_encoder_sha256": obs_encoder_sha256,
        "action_encoder_sha256": action_encoder_sha256,
        "reward_version": reward.reward_version(num_players),
        "num_players": num_players,
        "model_arch": model_arch,
    }


def current_meta(num_players, model_arch, extra=None):
    """現在の実装から拒否キーと記録キーを含むメタ dict を作る(保存用)。

    extra は B 表(記録)キーの上書き用(torch_version や git_commit など)。
    B 表に無いキー(A 表キーや任意キー)が含まれていたら ValueError
    (設計書 §15.1 第三)。torch は import しないため torch_version は
    空文字で始まり、呼び出し側が torch.__version__ を extra で渡す。

    戻り値は dict(保存メタ)。load() の expected には使わない — expected は
    `expected_meta(num_players, model_arch)` が返す専用の不変型を渡す
    (設計書 §15.1 第一)。
    """
    meta = dict(_current_a_values(num_players, model_arch))
    meta["git_commit"] = None
    meta["python_version"] = platform.python_version()
    meta["torch_version"] = ""
    meta["root_seed"] = None
    meta["hyperparameters"] = None
    meta["training_steps"] = 0
    meta["generation"] = 0
    meta["eval_results"] = None
    meta["created_at"] = None
    meta["next_episode_index"] = None
    meta["start_index"] = None
    if extra:
        _check_extra_keys(extra)
        meta.update(extra)
    return meta


def _check_extra_keys(extra):
    """extra が B 表キーだけを含むことを検証する(設計書 §15.1 第三)。

    保存メタ全体を extra へ渡すと A 表キーが混入し、ここで ValueError に
    なる(ゲート2回目で再現された経路1の攻撃 `current_meta(num_players,
    model_arch, extra=saved)` を塞ぐ)。内部マーカーは廃止されたため
    「内部キー」に該当するキーは存在しないが、B 表に無い任意キーも同様に
    拒否する(B 表のみ許可の契約)。
    """
    if not isinstance(extra, dict):
        raise ValueError(
            "extra must be a dict of B-table (recording) keys, got %s"
            % type(extra).__name__)
    for key in extra:
        if key not in B_KEYS:
            raise ValueError(
                "extra keys must be B-table (recording) keys only; "
                "A-table and internal keys are not allowed: %r" % key)


def save(meta_path, meta, weights_path=None):
    """メタを JSON で原子的に保存する(sort_keys=True / ensure_ascii=True / LF)。

    Phase 4: SIGKILL・ディスクフル・クラッシュで最終パス(genNNN.meta.json 等)
    に部分書きされた JSON が残らないよう、meta_path と同一ディレクトリの一時
    ファイルへ JSON を書いて fsync し、os.replace() で原子的に置換する。最終
    パスには常に「完全な旧内容」または「完全な新内容」だけが存在する。
    json.dump や fsync が失敗した場合(ディスクフル等)も一時ファイルを後始末
    し、meta_path を汚さない。

    書込形式は従来と不変: 有効な JSON、sort_keys=True、ensure_ascii=True、
    末尾改行、newline='\n' (LF)。正常系では一時ファイルは残らない。

    weights_path は呼び出し側の重み管理用の互換引数。本モジュールは torch を
    import しないため .pt の書き込みには関与しない(メタだけで検証できる構造)。

    内部マーカーは廃止したため(設計書 §15.1 第四)、永続化時に取り除く処理
    は存在しない。current_meta の戻り値(または B 表キーを追加した dict)を
    そのまま書き出す。
    """
    parent = os.path.dirname(os.path.abspath(meta_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = None
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix="." + os.path.basename(meta_path) + ".",
            suffix=".tmp",
            dir=parent,
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fd = None  # 所有権が fh に移った(fh.close() が閉じる)
            json.dump(meta, fh, sort_keys=True, ensure_ascii=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, meta_path)
        tmp_path = None  # 置換に成功したため、後始末対象から外す
    except BaseException:
        # json.dump / flush / fsync / os.replace の失敗(ディスクフル等)でも
        # 一時ファイルを後始末し、meta_path を汚さない。KeyboardInterrupt 等
        # の割込みでも一時ファイルを残さないよう BaseException で受ける。
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def load(meta_path, expected):
    """メタを読み、互換性を検証して返す。不一致は IncompatibleCheckpoint。

    expected は必須(既定値なし。省略すると TypeError)。専用の不変型
    `expected_meta(num_players, model_arch)` の戻り値だけを受け付け、
    dict(保存メタ)や他の型を渡すと TypeError にする(設計書 §15.1 第一)。
    保存メタ自身を expected として渡す「検査が自明に通る」経路は存在しない。

    A 表13キーの照合は、load 側が現在の実装から組み立てた値で毎回実行する
    (§15.1 第二 / §3.2 の最後の防壁)。エンコーダ実装から再計算した sha256
    との照合も常時実行する。

    Phase 4: json.load を包み、壊れた/途切れた JSON(クラッシュやディスク
    フルで残り得る部分書き、手動編集ミス等)を IncompatibleCheckpoint へ変換
    する。再開時に json.JSONDecodeError / UnicodeDecodeError / ValueError を
    外へ漏らさない。
    """
    if not isinstance(expected, _ExpectedMeta):
        raise TypeError(
            "expected must be the immutable value built by "
            "checkpoint.expected_meta(num_players, model_arch); passing the "
            "saved meta (a dict) or any other value is rejected, got %s"
            % type(expected).__name__)
    try:
        with open(meta_path, "r", encoding="utf-8", newline="") as fh:
            meta = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        # 壊れた/途切れた JSON を IncompatibleCheckpoint に変換する。
        # JSONDecodeError は ValueError の派生だが、契約(Phase 4)を明示する
        # ため3つを列挙する。OSError(ファイル不在等)は変換せずそのまま
        # 呼び出し側へ伝播させる。
        raise IncompatibleCheckpoint(
            "checkpoint incompatible: meta JSON is corrupt or truncated: "
            "%s: %s" % (type(exc).__name__, exc))
    if not isinstance(meta, dict):
        raise IncompatibleCheckpoint(
            "checkpoint incompatible: meta is not a JSON object")
    _check_encoder_hashes(meta, expected.num_players)
    _check_a_keys(meta, expected)
    _check_model_dims(meta, expected.num_players)
    return meta


def _check_encoder_hashes(meta, num_players):
    """人数の契約から再計算した encoder_sha256() と保存値を照合する(常時実行)。

    n=2 は旧 `encoding.encoder_sha256()` と同一の値・同一の計算(Phase 10 D2)。
    """
    obs_hash, action_hash = encoding_multi.contract_for(
        num_players).encoder_sha256()
    for key, current in (
        ("obs_encoder_sha256", obs_hash),
        ("action_encoder_sha256", action_hash),
    ):
        saved = meta.get(key)
        if saved != current:
            raise IncompatibleCheckpoint(
                "checkpoint incompatible: %s (current %s, got %r)"
                % (key, current, saved)
            )


def _strict_equal(saved, current):
    """値だけでなく型も一致するかを再帰的に判定する(設計書 §16.1)。

    Python では True == 1 / 1.0 == 1 / 2.0 == 2 が成立するため、単純な
    等値比較では JSON 上の型偽装(checkpoint_schema_version: true や
    num_players: 2.0、model_arch 内の obs_dim: 569.0)を素通りさせる。
    型が一致しなければ、値が等しくても不一致とする。

    - 整数を期待するキーで bool / float を拒否する。bool は int の派生な
      ので isinstance では通ってしまうため、type の完全一致
      (type(saved) is type(current))で判定する。
    - dict はキー集合と各値の型、list は長さと各要素の型まで一致を要求
      する。正規化(int へのキャスト等)は一切しない。
    """
    if type(saved) is not type(current):
        return False
    if isinstance(current, dict):
        if set(saved.keys()) != set(current.keys()):
            return False
        for key in current:
            if not _strict_equal(saved[key], current[key]):
                return False
        return True
    if isinstance(current, list):
        if len(saved) != len(current):
            return False
        for saved_item, current_item in zip(saved, current):
            if not _strict_equal(saved_item, current_item):
                return False
        return True
    return saved == current


def _check_a_keys(meta, expected):
    """A 表13キーを、現在の実装から組み立てた値と照合する(§15.1 第二)。

    §16.1: 値だけでなく型も一致を要求する(_strict_equal)。型偽装
    (true / 1.0 / 2.0 / model_arch 内の float)を素通りさせる単純な等値
    比較の実装へ戻すと、型偽装テストが落ちる。
    """
    current = _current_a_values(expected.num_players, expected.model_arch)
    for key in A_KEYS:
        saved = meta.get(key)
        if not _strict_equal(saved, current[key]):
            raise IncompatibleCheckpoint(
                "checkpoint incompatible: %s (expected %r, got %r)"
                % (key, current[key], saved)
            )


def _check_model_dims(meta, num_players):
    """model_arch の obs_dim / action_dim が人数の契約と一致するかを照合する。

    model_arch は呼び出し側が保存メタから読んで expected に渡すことが多く、A 表
    照合だけでは自己比較になる。人数と入力次元(2 席 569 / 3 席 688 / 4 席 807)
    の食い違い(別人数モデルへの num_players 偽装)をここで拒否する。次元キーを
    持たない arch(テスト用の最小 arch)は照合対象外。

    n=2 は既存の厳格検査をそのまま保つため、この追加照合を適用しない(2 席の
    受理/拒否集合を変えない。2 席の次元偽装は従来どおり重み形状照合で拒否)。
    """
    if num_players == 2:
        return
    arch = meta.get("model_arch")
    if not isinstance(arch, dict):
        return
    contract = encoding_multi.contract_for(num_players)
    for key, want in (("obs_dim", contract.obs_dim),
                      ("action_dim", contract.action_dim)):
        if key in arch and not _strict_equal(arch[key], want):
            raise IncompatibleCheckpoint(
                "checkpoint incompatible: model_arch.%s %r does not match "
                "the %d-seat encoder (%r)" % (key, arch[key], num_players,
                                              want))
