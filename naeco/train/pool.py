"""学習用相手プールと固定ベンチマーク群(Phase 3 設計書 §4.5 / §7)。

- 標準ライブラリのみを使う(numpy / torch は import しない、§2)。
- 学習用プール(OpponentPool)の抽選は `derive_seed(root_seed, "pool",
  episode_index)` 由来。素の random.Random() を直接作らず、派生 seed だけを
  使う(§6.3)。
- 学習用プールと固定ベンチマーク群は**型で分離**する(PLAN.md Phase 3)。
  `benchmark_suite` は `OpponentPool` を返さず、抽選メソッドを持たない
  tuple を返す。昇格・退行判定(§7)は固定ベンチマーク側だけで完結する。
"""
import dataclasses
import random

from naeco.engine.state import derive_seed

VALID_KINDS = ("checkpoint", "heuristic", "random")


@dataclasses.dataclass(frozen=True)
class PoolEntry:
    """相手プールの1エントリ。

    kind == "checkpoint" のとき path はメタ JSON のパス。heuristic / random
    は path を使わない(空文字列)。weight は抽選重み(正)。
    """

    name: str
    kind: str
    path: str
    weight: float

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(
                "unknown pool entry kind: {!r}".format(self.kind))
        if self.weight <= 0:
            raise ValueError(
                "pool entry weight must be positive: {!r}".format(
                    self.weight))
        if self.kind == "checkpoint" and not self.path:
            raise ValueError(
                "checkpoint entry requires a path: {!r}".format(self.name))


def _validate_entries(entries, label):
    """PoolEntry の並びを検証し、固定順の tuple で返す。"""
    if isinstance(entries, (str, bytes)):
        raise ValueError(
            "{} entries must be a sequence of PoolEntry".format(label))
    materialized = list(entries)
    if not materialized:
        raise ValueError(
            "{} must contain at least one entry".format(label))
    for entry in materialized:
        if not isinstance(entry, PoolEntry):
            raise ValueError(
                "{} entries must be PoolEntry, got {!r}".format(
                    label, type(entry).__name__))
    return tuple(materialized)


class OpponentPool(object):
    """学習用の抽選プール。

    sample は (root_seed, episode_index) と `"pool"` ストリームだけから
    決定論的に1エントリを選ぶ。席・worker・実行順を seed に混ぜない。
    """

    def __init__(self, entries):
        self._entries = _validate_entries(entries, "opponent pool")
        self._weights = tuple(entry.weight for entry in self._entries)

    def sample_multi(self, root_seed, episode_index, opponent_index):
        """D5: independent 50/25/25 category draw per opponent."""
        latest = [e for e in self._entries if e.name == "latest"]
        past = [e for e in self._entries
                if e.kind == "checkpoint" and e.name != "latest"][-5:]
        heuristic = [e for e in self._entries if e.kind == "heuristic"]
        if len(latest) != 1 or len(heuristic) != 1:
            raise ValueError("multi pool requires latest and heuristic")
        rng = random.Random(derive_seed(
            root_seed, "pool", episode_index, opponent_index))
        category = rng.random()
        if category < 0.5:
            return latest[0]
        if category < 0.75:
            return rng.choice(past) if past else latest[0]
        return heuristic[0]

    def sample(self, root_seed, episode_index):
        """root_seed と episode_index だけで決まる PoolEntry を返す。"""
        pool_seed = derive_seed(root_seed, "pool", episode_index)
        rng = random.Random(pool_seed)
        return rng.choices(self._entries, weights=self._weights, k=1)[0]


def training_pool(latest, past_generations, heuristic_weight=1.0,
                  random_weight=0.0):
    """最新チャンピオン + 指定過去世代 + heuristic / random の学習用プール。

    latest と past_generations は kind == "checkpoint" の PoolEntry。
    heuristic_weight / random_weight はその相手の抽選重みで、0 以下なら
    その相手はプールに入れない。既定は PLAN.md の「最新版 + 過去世代 +
    ヒューリスティック」混合に合わせ、random はオプトインにする。
    """
    if isinstance(past_generations, PoolEntry):
        raise ValueError(
            "past_generations must be a sequence of PoolEntry")
    entries = [latest]
    entries.extend(past_generations)
    for entry in entries:
        if entry.kind != "checkpoint":
            raise ValueError(
                "training_pool entries must be checkpoints, got kind {!r}"
                .format(entry.kind))
    if heuristic_weight > 0:
        entries.append(
            PoolEntry("heuristic", "heuristic", "", heuristic_weight))
    if random_weight > 0:
        entries.append(PoolEntry("random", "random", "", random_weight))
    return OpponentPool(entries)


# ---------------------------------------------------------------------------
# Phase 12b 方法 B: 卓単位の PFSP sampler(opt-in。旧 sample / sample_multi は不変)
# ---------------------------------------------------------------------------
POOL_SCHEMA = "phase12b-pool/v1"
TABLE_CATEGORIES = (("latest", 0.50), ("pfsp", 0.35), ("biased", 0.15))
PFSP_FLOOR = 0.05
PFSP_UNMEASURED_X = 0.5
# 卓の席を埋める記号。具体 entry 名か次の予約語。
#   @latest        現行 learner の net(sample)
#   @past          凍結した過去方策集合(window 開始時)から席ごとに一様抽選
#   heuristic / random / biased:<kind>   固定方策
TABLE_TOKENS = ("@latest", "@past")
TABLE_SEED_STREAM = "pfsp-table"


def pfsp_x(win_rate, num_players):
    """正規化勝率 x = clip(n * p, 0, 1)。p は None(未測定)なら 0.5。"""
    if win_rate is None:
        return PFSP_UNMEASURED_X
    p = float(win_rate)
    if not (0.0 <= p <= 1.0):
        raise ValueError("win rate must be in [0,1], got %r" % (win_rate,))
    return min(1.0, max(0.0, num_players * p))


def pfsp_weights(win_rates, num_players):
    """D3: w = max(0.05, (1 - x)^2) を正規化した重み(win_rates と同順)。"""
    if not win_rates:
        raise ValueError("pfsp needs at least one frozen table")
    raw = [max(PFSP_FLOOR, (1.0 - pfsp_x(p, num_players)) ** 2)
           for p in win_rates]
    total = sum(raw)
    return [w / total for w in raw]


def _check_table(table, num_players, entry_names):
    if not isinstance(table, dict) or set(table) != {"id", "opponents"}:
        raise ValueError("table must be {id, opponents}: %r" % (table,))
    opponents = table["opponents"]
    if not isinstance(opponents, list) or len(opponents) != num_players - 1:
        raise ValueError("table %s needs %d opponents" % (table["id"],
                                                         num_players - 1))
    for token in opponents:
        if token in TABLE_TOKENS or token in entry_names:
            continue
        if token in ("heuristic", "random") or token.startswith("biased:"):
            continue
        raise ValueError("table %s: unknown seat %r" % (table["id"], token))
    return tuple(opponents)


def validate_pool_manifest(manifest, num_players):
    """pool manifest(phase12b-pool/v1)を検査し正規化した dict を返す。

    欠損・余剰キー・人数不一致・重み/区分の改変・未知の席は ValueError。
    checkpoint entry の SHA 照合は呼び出し側(ファイルを読む学習 CLI)が行う。
    """
    required = {"schema", "num_players", "categories", "entries",
                "pfsp_tables", "biased_tables", "pfsp", "tie_break",
                "seed_domain"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("pool manifest keys must be exactly %s"
                         % sorted(required))
    if manifest["schema"] != POOL_SCHEMA:
        raise ValueError("pool manifest schema must be %s" % POOL_SCHEMA)
    if manifest["num_players"] != num_players or num_players not in (3, 4):
        raise ValueError("pool manifest is for %r players, run has %d"
                         % (manifest["num_players"], num_players))
    if manifest["categories"] != dict(TABLE_CATEGORIES):
        raise ValueError("table categories are fixed at 50/35/15 (D3)")
    names = set()
    for entry in manifest["entries"]:
        if set(entry) != {"name", "kind", "path", "sha256"}:
            raise ValueError("entry must be {name, kind, path, sha256}")
        if entry["kind"] != "checkpoint" or entry["name"] in names \
                or entry["name"].startswith("@"):
            raise ValueError("bad pool entry %r" % (entry["name"],))
        if set(entry["sha256"]) != {"meta", "weights"}:
            raise ValueError("entry %s needs meta/weights sha256" % entry["name"])
        names.add(entry["name"])
    pfsp_tables = [_check_table(t, num_players, names)
                   for t in manifest["pfsp_tables"]]
    biased = [_check_table(t, num_players, names)
              for t in manifest["biased_tables"]]
    if not pfsp_tables or not biased:
        raise ValueError("pfsp and biased tables must be non-empty")
    for table in manifest["biased_tables"]:
        if not all(s.startswith("biased:") for s in table["opponents"]):
            raise ValueError("biased table %s has a non-biased seat" % table["id"])
    ids = [t["id"] for t in manifest["pfsp_tables"] + manifest["biased_tables"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate table id")
    pfsp = manifest["pfsp"]
    if set(pfsp) != {"interval_generations", "clusters", "diag_start_index",
                     "diag_max_index", "floor", "unmeasured_x"}:
        raise ValueError("pfsp block keys mismatch")
    if pfsp["floor"] != PFSP_FLOOR or pfsp["unmeasured_x"] != PFSP_UNMEASURED_X:
        raise ValueError("pfsp floor/unmeasured x are fixed (D3)")
    return manifest


class TableSampler(object):
    """卓カテゴリ 50% latest / 35% PFSP 凍結卓 / 15% 固定偏り卓(D3)。

    抽選は (root_seed, episode_index) と "pfsp-table" ストリームだけから決まる。
    worker・完了順・席を seed に混ぜない。PFSP 重みは window 単位で凍結した
    値を外から渡す(dev/accept の結果を使わない)。
    """

    def __init__(self, num_players, pfsp_tables, pfsp_weights_, biased_tables,
                 past_names):
        if len(pfsp_tables) != len(pfsp_weights_):
            raise ValueError("pfsp weights must align with tables")
        if any(w <= 0 for w in pfsp_weights_):
            raise ValueError("pfsp weights must be positive")
        self.num_players = num_players
        self._pfsp = [(t["id"], tuple(t["opponents"])) for t in pfsp_tables]
        self._weights = tuple(float(w) for w in pfsp_weights_)
        self._biased = [(t["id"], tuple(t["opponents"])) for t in biased_tables]
        self._past = tuple(past_names)

    def sample(self, root_seed, episode_index):
        """(category, table_id, opponents) を返す。opponents は n-1 個の席記号。"""
        rng = random.Random(derive_seed(root_seed, TABLE_SEED_STREAM,
                                        episode_index))
        u = rng.random()
        if u < 0.50:
            return "latest", "latest", ("@latest",) * (self.num_players - 1)
        if u < 0.85:
            table_id, seats = rng.choices(self._pfsp, weights=self._weights,
                                          k=1)[0]
            category = "pfsp"
        else:
            table_id, seats = rng.choice(self._biased)
            category = "biased"
        resolved = []
        for token in seats:
            if token == "@past":
                if not self._past:
                    raise ValueError("@past seat needs a frozen past set")
                token = rng.choice(self._past)
            resolved.append(token)
        return category, table_id, tuple(resolved)


def benchmark_suite(generations):
    """固定ベンチマーク群を tuple で返す(抽選しない・順序固定)。

    判定用(§7)は学習用プールと分離する。呼び出し側が固定したい相手
    (HeuristicAgent + 指定過去世代など)を PoolEntry の並びで渡し、この
    tuple をそのまま昇格・退行判定に使う。weight は抽選に使われないが、
    PoolEntry の整合性検証は通常どおり行う。
    """
    return _validate_entries(generations, "benchmark suite")
