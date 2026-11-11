"""Phase 2 で固定する、2人戦の席交換・クラスタ bootstrap 評価。"""
import dataclasses
import hashlib
import json
import math
import os
import random

from naeco.ai.heuristic_agent import HeuristicAgent
from naeco.ai.random_agent import RandomAgent
from naeco.engine import game
from naeco.engine.cards import load_cards
from naeco.engine.observe import observe
from naeco.engine.state import (
    API_VERSION, GROUPS, RULES_VERSION, Phase, derive_seed, eval_seed,
    state_hash,
)

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 2026072102


@dataclasses.dataclass(frozen=True)
class GameRecord:
    """1局ぶんの対応表エントリ(PLAN.md §5「seed・席・初期山札の対応表」)。

    初期山札は `initial_deck`(順序付き)と `initial_hands`(席別)として**明示的に**保存する。
    `initial_state_hash` はその整合性検証用の指紋であり、`cards_sha256` /
    `cards_data_version` はどのカードデータ・ルール版で生成した配札かを固定する
    (将来カードデータや実装が変わっても、保存済みの配札そのものは失われない)。
    """
    group: str
    index: int
    seat_variant: int
    env_seed: int
    setup_seed: int          # 開始席抽選の seed(席非依存ストリーム "setup")
    policy_seed: int         # HeuristicAgent(評価対象)の方策 seed
    opponent_seed: int       # RandomAgent の方策 seed
    target_seat: int         # HeuristicAgent の席
    opponent_seat: int       # RandomAgent の席
    start_player: int        # R1 の開始プレイヤー席(RUL-SETUP-001 の抽選結果)
    initial_deck: tuple      # 初期局面の山札(順序付き instance_id)
    initial_hands: tuple     # 席別の初期手札(seat 昇順、各 instance_id タプル)
    initial_money: tuple     # 席別の初期所持金(開始プレイヤーが $5)
    initial_state_hash: str  # 上記の整合性検証用の正準指紋
    cards_sha256: str        # 配札を生成したカードデータの正準ハッシュ
    cards_data_version: str  # カードデータ版(ルール裁定の版に対応)
    api_version: str         # エンジンAPI版(状態表現・正準化規則)
    rules_version: str       # ルール裁定版
    result: float            # target から見た {1.0 勝 / 0.5 分 / 0.0 敗}


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
    group: str
    start_index: int
    pairs: int
    games: int
    wins: int
    draws: int
    losses: int
    point_estimate: float
    ci_low: float
    ci_high: float
    bootstrap_samples: int
    bootstrap_seed: int
    pair_results: tuple
    initial_state_hashes: tuple
    game_records: tuple


def _percentile(values, percentile):
    """NumPy 非依存の線形補間パーセンタイル。"""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    index = (len(ordered) - 1) * percentile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def cluster_bootstrap(pair_results, samples, seed):
    """ペア(クラスタ)単位の復元抽出で 95%CI を返す(§10.3)。

    リサンプルの単位は**ペア**であり、局ではない。同一配札の席交換2局は相関するため、
    局単位で復元抽出すると CI が不当に狭くなる。
    """
    rng = random.Random(seed)
    count = len(pair_results)
    if count == 0:
        raise ValueError("pairs must be positive")
    means = []
    for unused in range(samples):
        total = 0.0
        for unused_inner in range(count):
            total += pair_results[rng.randrange(count)]
        means.append(total / count)
    return _percentile(means, 0.025), _percentile(means, 0.975)


# 後方互換の別名(内部呼び出し・既存参照向け)。
_cluster_bootstrap = cluster_bootstrap


def summarize_pairs(pair_games):
    """(g0, g1) の並び(各 g は {0.0, 0.5, 1.0})から集計する。

    引き分け(0.5)は 0.5 勝として勝/分/敗と点推定・ペア勝率へ反映する(§9.3)。
    実対局では発生しない想定だが、集計経路はここで一元化してテスト可能にする。
    """
    wins = draws = losses = 0
    pair_results = []
    for g0, g1 in pair_games:
        for r in (g0, g1):
            if r == 1.0:
                wins += 1
            elif r == 0.5:
                draws += 1
            else:
                losses += 1
        pair_results.append((g0 + g1) / 2.0)
    point = sum(pair_results) / len(pair_results) if pair_results else 0.0
    return wins, draws, losses, tuple(pair_results), point


def _play_game(cards, group, index, seat_variant,
               environment_seed, target_seed, opponent_seed, target_seat,
               setup_seed):
    state = game.new_game(2, environment_seed, cards, setup_seed=setup_seed)
    initial_hash = state_hash(state)
    initial_deck = tuple(state.deck)
    initial_hands = tuple(tuple(p.hand_buildings) for p in state.players)
    initial_money = tuple(p.money for p in state.players)
    start_player = state.start_player
    opponent_seat = 1 - target_seat
    agents = [None, None]
    agents[target_seat] = HeuristicAgent(target_seed)
    agents[opponent_seat] = RandomAgent(opponent_seed)
    result = None
    while state.phase != Phase.GAME_OVER:
        who = game.actor(state)
        action = agents[who].act(observe(state, who, cards))
        result = game.step(state, action, cards)
        state = result.state
    utilities = result.utilities
    if utilities is not None:
        utility = utilities[target_seat]
        outcome = 1.0 if utility > 0 else 0.0 if utility < 0 else 0.5
    else:
        # 2人戦の utilities は通常必ずあるが、ranking による安全なフォールバック。
        outcome = 1.0 if result.ranking[target_seat] == 0 else 0.0
    record = GameRecord(
        group=group, index=index, seat_variant=seat_variant,
        env_seed=environment_seed, setup_seed=setup_seed,
        policy_seed=target_seed,
        opponent_seed=opponent_seed, target_seat=target_seat,
        opponent_seat=opponent_seat, start_player=start_player,
        initial_deck=initial_deck, initial_hands=initial_hands,
        initial_money=initial_money, initial_state_hash=initial_hash,
        cards_sha256=state.cards_sha256,
        cards_data_version=state.cards_data_version,
        api_version=API_VERSION, rules_version=RULES_VERSION,
        result=outcome,
    )
    return record


def evaluate(pairs=500, group="dev", start_index=0,
             bootstrap_samples=BOOTSTRAP_SAMPLES,
             bootstrap_seed=BOOTSTRAP_SEED):
    """HeuristicAgent vs RandomAgent を指定 seed 範囲で評価する。"""
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    cards = load_cards()
    pair_games = []
    hashes = []
    records = []
    for offset in range(pairs):
        index = start_index + offset
        root = eval_seed(group, index)
        environment_seed = derive_seed(root, "env", index)
        # "setup" は席非依存ストリーム(§2.5)。ペアの2局は同一配札かつ同一開始席。
        setup_seed = derive_seed(root, "setup", index)
        game_outcomes = []
        pair_hashes = []
        for seat_variant in (0, 1):
            policy_seed = derive_seed(root, "policy", index, seat_variant)
            # seat_variant=0 は評価対象を席0に、=1 は席1に座らせる。
            target_seat = seat_variant
            record = _play_game(
                cards, group, index, seat_variant, environment_seed,
                policy_seed, policy_seed, target_seat, setup_seed)
            game_outcomes.append(record.result)
            pair_hashes.append(record.initial_state_hash)
            records.append(record)
        if pair_hashes[0] != pair_hashes[1]:
            raise AssertionError("paired games did not share an initial deal")
        pair_games.append((game_outcomes[0], game_outcomes[1]))
        hashes.append(tuple(pair_hashes))
    wins, draws, losses, pair_results, point = summarize_pairs(pair_games)
    ci_low, ci_high = cluster_bootstrap(pair_results, bootstrap_samples,
                                        bootstrap_seed)
    return EvaluationResult(
        group=group, start_index=start_index, pairs=pairs, games=pairs * 2,
        wins=wins, draws=draws, losses=losses, point_estimate=point,
        ci_low=ci_low, ci_high=ci_high, bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed, pair_results=pair_results,
        initial_state_hashes=tuple(hashes), game_records=tuple(records),
    )


def write_game_records(result, path):
    """対応表(1局1行の JSONL)を書き出す。EOL は LF 固定(環境非依存)。"""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in result.game_records:
            fh.write(json.dumps(dataclasses.asdict(rec),
                                ensure_ascii=True, sort_keys=True))
            fh.write("\n")


def passes_threshold(result, threshold):
    """合否は仕様どおり厳密不等号。"""
    return result.ci_low > threshold


# ---------------------------------------------------------------------------
# Phase 10 D4: n 局巡回クラスタ(3・4 人戦)。
#
# n=2 は上の旧 runner / GameRecord / summarize_pairs / cluster_bootstrap を
# そのまま使い、ここの新形式では扱わない(D4「新形式に再保存しない」)。
# 以下は stdlib のみ。torch を import しない。
# ---------------------------------------------------------------------------

MULTI_PROTOCOL = "phase10-cyclic/v1"
MULTI_RECORD_SCHEMA = "multi-eval-record/v2"
MULTI_SUMMARY_SCHEMA = "multi-eval-summary/v2"
MULTI_PLAYER_COUNTS = (3, 4)
MULTI_AGENT_KINDS = ("heuristic", "random")
# 固定方策 biased:<kind>(BiasedAgent。kind は biased_agent.BIASED_KINDS のみ)。
# parse_multi_agent_spec は ("biased", <kind>) を返す(2 値目はモデル path ではない)。
MULTI_BIASED_PREFIX = "biased"

# D4 の固定値(core-api §10.6)。結果を見てから変えない。
D4_SIGMA = 0.5                 # クラスタ値 ∈ [0,1] の保守 σ
D4_Z_ALPHA = 1.96              # 95% 両側 CI 下限 → 片側 α=0.025
D4_Z_BETA = 0.842              # power 0.80
D4_DELTA = 0.05                # 対立 = 閾値 + 0.05
D4_CLUSTERS = 800              # ceil(786) を 100 単位で切上げ
D4_FAMILY_JUDGEMENTS = 2       # 3 席・4 席の最終 2 判定(Bonferroni)
D4_COMPLETION_THRESHOLDS = {3: 0.60, 4: 0.50}


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def required_clusters(delta=D4_DELTA, sigma=D4_SIGMA, z_alpha=D4_Z_ALPHA,
                      z_beta=D4_Z_BETA):
    """`ceil(((z_a+z_b)*σ/Δ)^2)`。D4 の値では 786。"""
    if not delta > 0:
        raise ValueError("delta must be positive")
    return int(math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2))


def approximate_power(clusters, delta=D4_DELTA, sigma=D4_SIGMA,
                      z_alpha=D4_Z_ALPHA):
    """正規近似の検出力 Φ(Δ√C/σ − z_α)(bootstrap の有限標本保証ではない)。"""
    if clusters < 1:
        raise ValueError("clusters must be positive")
    return _normal_cdf(delta * math.sqrt(clusters) / sigma - z_alpha)


def d4_power_record(threshold, clusters=D4_CLUSTERS):
    """実行前登録に保存する検出力の根拠(D4)。閾値なしの D5 ベンチは対象外。"""
    if threshold is None:
        return None
    alpha_one_sided = 1.0 - _normal_cdf(D4_Z_ALPHA)
    return {
        "threshold": threshold,
        "alternative": threshold + D4_DELTA,
        "delta": D4_DELTA,
        "sigma": D4_SIGMA,
        "z_alpha": D4_Z_ALPHA,
        "z_beta": D4_Z_BETA,
        "target_power": 0.80,
        "required_clusters": required_clusters(),
        "clusters": clusters,
        "approximate_power": approximate_power(clusters),
        "ci": "95% two-sided percentile cluster bootstrap; pass iff ci_low > threshold",
        "alpha_one_sided_per_judgement": round(alpha_one_sided, 6),
        "bonferroni_judgements": D4_FAMILY_JUDGEMENTS,
        "family_alpha_bound": round(alpha_one_sided * D4_FAMILY_JUDGEMENTS, 6),
        "note": "normal approximation; not a finite-sample bootstrap guarantee",
    }


def cyclic_seat_order(num_players, k):
    """`order[seat] = agent index`。env._seat_order(randomize_seats=False) と同じ写像。"""
    return [(seat + k) % num_players for seat in range(num_players)]


def target_seat_for(num_players, k):
    """正準 agent 0(評価対象)が座る席。"""
    return cyclic_seat_order(num_players, k).index(0)


# 学習済みモデルを座らせる指定(B-C6-fix t_4cf080a7)。PATH は人数別 checkpoint の
# メタ(`.meta.json`)。重みは同名の `.weights.pt`。
#   checkpoint:PATH  中 = 人数別 greedy PolicyAgent
#   strong:PATH      強 = 同モデル + ISMCTS(D5 の固定構成。引数では変えられない)
MULTI_MODEL_KINDS = ("checkpoint", "strong")
_META_SUFFIX = ".meta.json"
_WEIGHTS_SUFFIX = ".weights.pt"

# D5「強=同モデル+PUCT/ISMCTS(H=2、c_puct=1.5、T=4500)」。Web の強(Phase 5 認定
# 構成 S2+L1)と同じ組立て。評価・昇格・最終判定で変えない固定値なので
# 関数引数・CLI 引数から上書きする経路を作らない。
STRONG_SEARCH = {
    "algorithm": "ismcts",
    "root_policy": "puct",
    "horizon": 2,
    "c_puct": 1.5,
    "budget_transitions": 4500,
    "budget_seconds": None,
    "fpu": 0.5,
    "evaluator": "NetEvaluator(same model)",
    "pending_delegate": "PolicyAgent(same model, greedy, seed=0)",
    "root_seed": "policy_seed = derive_seed(root,'policy',index,seat)",
}
STRONG_SEED_RULE = ("IsmctsAgent root_seed = derive_seed(eval_seed(group,index),"
                    "'policy',index,seat); no worker index, no wall clock")


def parse_multi_agent_spec(spec):
    """n>=3 の agent 指定を (kind, path) に正規化する(path はモデル以外 '')。"""
    if not isinstance(spec, str) or not spec:
        raise ValueError("agent spec must be a non-empty string")
    if spec in MULTI_AGENT_KINDS:
        return spec, ""
    kind, sep, path = spec.partition(":")
    if sep and kind == MULTI_BIASED_PREFIX:
        from naeco.ai.biased_agent import BIASED_KINDS
        if path not in BIASED_KINDS:
            raise ValueError("unknown biased kind %r in %r (expected one of %s)"
                             % (path, spec, ", ".join(BIASED_KINDS)))
        return kind, path
    if sep and kind in MULTI_MODEL_KINDS:
        if not path.endswith(_META_SUFFIX) or len(path) == len(_META_SUFFIX):
            raise ValueError("%s spec needs a checkpoint meta path ending in "
                             "%s: %r" % (kind, _META_SUFFIX, spec))
        return kind, path
    raise ValueError(
        "multi-seat evaluation supports agent specs %s, biased:KIND, "
        "checkpoint:META and strong:META (got %r)"
        % (", ".join(MULTI_AGENT_KINDS), spec))


def parse_basic_agent_spec(spec):
    """後方互換: heuristic/random/checkpoint:/strong: を検証して spec を返す。"""
    parse_multi_agent_spec(spec)
    return spec


def weights_path_for(meta_path):
    if not meta_path.endswith(_META_SUFFIX):
        raise ValueError("checkpoint meta path must end with %s: %r"
                         % (_META_SUFFIX, meta_path))
    return meta_path[:-len(_META_SUFFIX)] + _WEIGHTS_SUFFIX


def _file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def model_file_shas(meta_path):
    """{"meta": sha, "weights": sha}。どちらかが無ければ ValueError。"""
    weights = weights_path_for(meta_path)
    for path in (meta_path, weights):
        if not os.path.isfile(path):
            raise ValueError("checkpoint file missing: %s" % path)
    return {"meta": _file_sha256(meta_path), "weights": _file_sha256(weights)}


def check_model_for_players(meta_path, num_players):
    """メタの人数・A 表を席数で検査する(torch を読まない)。不一致は ValueError。

    `checkpoint.load(meta, expected_meta(num_players, arch))` を通すので、人数・
    encoder hash・入力次元の偽装も拒否される。人数は評価側(席数)から渡し、
    メタから読んだ値を期待値にしない。
    """
    from naeco.train import checkpoint as checkpoint_mod
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read checkpoint meta %s: %s" % (meta_path, exc))
    saved_n = raw.get("num_players") if isinstance(raw, dict) else None
    if saved_n != num_players:
        raise ValueError(
            "checkpoint %s is a %r-seat model; this evaluation has %d seats "
            "(per-player-count models only, D2)" % (meta_path, saved_n,
                                                    num_players))
    arch = raw.get("model_arch")
    try:
        checkpoint_mod.load(
            meta_path, expected=checkpoint_mod.expected_meta(num_players, arch))
    except checkpoint_mod.IncompatibleCheckpoint as exc:
        raise ValueError(str(exc))
    except ValueError as exc:
        raise ValueError("checkpoint %s rejected: %s" % (meta_path, exc))


def model_record(spec, num_players):
    """モデル指定の照合記録(conditions["models"] の 1 値)。"""
    kind, path = parse_multi_agent_spec(spec)
    if kind not in MULTI_MODEL_KINDS:
        raise ValueError("%r has no model" % (spec,))
    check_model_for_players(path, num_players)
    record = {"kind": kind, "num_players": num_players,
              "sha256": model_file_shas(path)}
    record["mode"] = "greedy" if kind == "checkpoint" else "ismcts"
    return record


# プロセス内のモデル cache(spawn worker ごと)。key は (meta パス, 期待 SHA)。
_NET_CACHE = {}


def _load_net(meta_path, num_players, expected_shas):
    """SHA を照合してから人数別 checkpoint を読む(torch は net.py 経由の遅延 import)。"""
    key = (meta_path, num_players, expected_shas["meta"],
           expected_shas["weights"])
    net = _NET_CACHE.get(key)
    if net is not None:
        return net
    actual = model_file_shas(meta_path)
    if actual != expected_shas:
        raise RuntimeError(
            "model %s changed after the run conditions were fixed "
            "(expected %r, got %r); refusing to play" % (meta_path,
                                                          expected_shas, actual))
    from naeco.train import net as net_mod
    net, unused_meta = net_mod.load_checkpoint(meta_path, num_players)
    if model_file_shas(meta_path) != expected_shas:
        raise RuntimeError("model %s changed while loading" % meta_path)
    _NET_CACHE[key] = net
    return net


def _make_multi_agent(spec, seed, cards, num_players, models):
    """spec から 1 席ぶんの agent と (model_sha256, search) を作る。"""
    kind, path = parse_multi_agent_spec(spec)
    if kind == "heuristic":
        return HeuristicAgent(seed), None, None
    if kind == "random":
        return RandomAgent(seed), None, None
    if kind == MULTI_BIASED_PREFIX:
        from naeco.ai.biased_agent import BiasedAgent
        return BiasedAgent(path, seed, cards=cards), None, None
    info = (models or {}).get(spec)
    if info is None:
        raise ValueError("model %r is not fixed in the run conditions" % spec)
    shas = info["sha256"]
    net = _load_net(path, num_players, shas)
    from naeco.train.agent import PolicyAgent
    if kind == "checkpoint":
        return PolicyAgent(net, mode="greedy", seed=seed), dict(shas), None
    from naeco.ai.search.ismcts import IsmctsAgent
    from naeco.train.search_evaluator import NetEvaluator
    agent = IsmctsAgent(
        root_seed=seed,
        budget_transitions=STRONG_SEARCH["budget_transitions"],
        c_puct=STRONG_SEARCH["c_puct"],
        horizon=STRONG_SEARCH["horizon"],
        cards=cards,
        evaluator=NetEvaluator(net),
        policy_agent=PolicyAgent(net, mode="greedy", seed=0),
        root_policy=STRONG_SEARCH["root_policy"])
    search = dict(STRONG_SEARCH)
    search["root_seed"] = seed
    return agent, dict(shas), search


def _validate_multi(num_players, agents):
    if type(num_players) is not int or num_players not in MULTI_PLAYER_COUNTS:
        raise ValueError(
            "multi-seat evaluation needs num_players in %s (n=2 uses the "
            "legacy pair runner), got %r" % (MULTI_PLAYER_COUNTS, num_players))
    agents = tuple(agents)
    if len(agents) != num_players:
        raise ValueError("need exactly %d agent specs (target first), got %d"
                         % (num_players, len(agents)))
    for spec in agents:
        parse_multi_agent_spec(spec)
    return agents


def multi_models(num_players, agents):
    """agent リスト中のモデル指定 → 照合記録。heuristic/random だけなら {}。"""
    out = {}
    for spec in _validate_multi(num_players, agents):
        kind, unused_path = parse_multi_agent_spec(spec)
        if kind in MULTI_MODEL_KINDS and spec not in out:
            out[spec] = model_record(spec, num_players)
    return out


def verify_models_unchanged(conditions):
    """conditions に固定したモデルの SHA を再検査する(判定前の fail-closed)。"""
    for spec, info in sorted((conditions.get("models") or {}).items()):
        unused_kind, path = parse_multi_agent_spec(spec)
        actual = model_file_shas(path)
        if actual != info["sha256"]:
            raise RuntimeError(
                "model %s changed during the run (expected %r, got %r); no "
                "judgement is issued" % (path, info["sha256"], actual))


def multi_run_id(conditions):
    """条件だけから決まる run_id(worker 数・時刻を含めない)。"""
    payload = json.dumps(conditions, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def multi_conditions(num_players, agents, group, start_index, clusters,
                     purpose, threshold, cards=None):
    """実行前に固定する条件(run_id の材料)。"""
    from naeco.engine.observe import OBS_VERSION
    from naeco.train.reward import reward_version
    cards = cards or load_cards()
    agents = _validate_multi(num_players, agents)
    if group not in GROUPS:
        raise ValueError("unknown seed group: %r" % (group,))
    if type(start_index) is not int or start_index < 0:
        raise ValueError("start_index must be a non-negative int")
    if type(clusters) is not int or clusters < 1:
        raise ValueError("clusters must be a positive int")
    conditions = {
        "protocol": MULTI_PROTOCOL,
        "record_schema": MULTI_RECORD_SCHEMA,
        "num_players": num_players,
        "agents": list(agents),
        "group": group,
        "start_index": start_index,
        "end_index": start_index + clusters - 1,
        "clusters": clusters,
        "games": clusters * num_players,
        "purpose": purpose,
        "threshold": threshold,
        "win_metric": "ranking[target_seat] == 0",
        "seat_assignment": "cyclic order[seat]=(seat+k)%n, k=0..n-1",
        "policy_seed": "derive_seed(root,'policy',index,seat)",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "api_version": API_VERSION,
        "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
        "reward_version": reward_version(num_players),
    }
    # モデル席があるときだけ SHA・探索構成を条件(= run_id)に入れる。
    # heuristic/random だけの条件は従来と同じ dict のまま(既存 run_id 不変)。
    models = multi_models(num_players, agents)
    if models:
        conditions["models"] = models
        if any(m["kind"] == "strong" for m in models.values()):
            conditions["strong_search"] = dict(STRONG_SEARCH)
            conditions["strong_seed_rule"] = STRONG_SEED_RULE
    return conditions


def conditions_registration_view(conditions):
    """条件から台帳の予約行と照合する欄だけを取り出す(人数は評価条件由来)。"""
    return {
        "num_players": conditions["num_players"],
        "group": conditions["group"],
        "lo": conditions["start_index"],
        "hi": conditions["end_index"],
        "purpose": conditions["purpose"],
        "threshold": conditions["threshold"],
    }


def cyclic_tasks_for(conditions, run_id, skip=()):
    """保存済み条件から全局(または skip 以外)のタスク列を作る。"""
    return cyclic_tasks(run_id, conditions["group"], conditions["start_index"],
                        conditions["clusters"], conditions["agents"],
                        conditions["num_players"], skip=skip,
                        models=conditions.get("models"))


def play_cyclic_game(cards, run_id, group, index, k, agents, num_players,
                     models=None, protocol=MULTI_PROTOCOL):
    """クラスタ index の k 番目の局(1 局 1 行の record v2 を返す)。

    models は conditions["models"](モデル指定 → kind / SHA)。モデル席は
    読込前後に SHA を照合し、record の席欄へ model_sha256・探索構成・
    policy seed を残す。heuristic/random 席は従来どおり None。
    protocol は record の protocol 欄だけを決める(Phase 12b の opt-in。
    既定は従来の phase10-cyclic/v1 で、既定の出力は byte 不変)。
    """
    from naeco.engine.observe import OBS_VERSION
    from naeco.train.reward import reward_version
    agents = _validate_multi(num_players, agents)
    if not (0 <= k < num_players):
        raise ValueError("k must be in 0..%d" % (num_players - 1))
    root = eval_seed(group, index)
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game.new_game(num_players, env_seed, cards, setup_seed=setup_seed)
    initial_hash = state_hash(state)
    initial_deck = list(state.deck)
    initial_hands = [list(p.hand_buildings) for p in state.players]
    initial_money = [p.money for p in state.players]
    start_player = state.start_player
    seat_agents = cyclic_seat_order(num_players, k)      # seat -> agent index
    order = [seat_agents.index(j) for j in range(num_players)]  # agent -> seat
    target_seat = order[0]
    seats = []
    seated = []
    for seat in range(num_players):
        spec = agents[seat_agents[seat]]
        # 方策 seed はクラスタ座標(group, index, seat)だけから決まる。強の
        # IsmctsAgent の root_seed もこれ(worker・実行順・時刻を含めない)。
        policy_seed = derive_seed(root, "policy", index, seat)
        agent, model_sha, search = _make_multi_agent(
            spec, policy_seed, cards, num_players, models)
        seated.append(agent)
        seats.append({"seat": seat, "agent_index": seat_agents[seat],
                      "agent": spec, "model_sha256": model_sha,
                      "policy_seed": policy_seed, "search": search})
    result = None
    decisions = 0
    while state.phase != Phase.GAME_OVER:
        who = game.actor(state)
        action = seated[who].act(observe(state, who, cards))
        result = game.step(state, action, cards)
        state = result.state
        decisions += 1
    if result is None or result.ranking is None or result.utilities is None:
        raise AssertionError("game %s/%d k=%d ended without a terminal result"
                             % (group, index, k))
    ranking = list(result.ranking)
    return {
        "schema": MULTI_RECORD_SCHEMA,
        "protocol": protocol,
        "run_id": run_id,
        "group": group,
        "index": index,
        "n": num_players,
        "seat_variant": k,
        "agents": list(agents),
        "order": order,
        "seat_agents": seat_agents,
        "target_seat": target_seat,
        "seats": seats,
        "env_seed": env_seed,
        "setup_seed": setup_seed,
        "start_player": start_player,
        "initial_deck": initial_deck,
        "initial_hands": initial_hands,
        "initial_money": initial_money,
        "initial_state_hash": initial_hash,
        "cards_sha256": cards.sha256,
        "cards_data_version": cards.data_version,
        "api_version": API_VERSION,
        "rules_version": RULES_VERSION,
        "obs_version": OBS_VERSION,
        "reward_version": reward_version(num_players),
        # RUL-SCORE-002 の同点処理は終局時点の開始プレイヤーを使う
        # (局中に移る)。保存物だけから順位を再計算できるよう別に残す。
        "final_start_player": state.start_player,
        "scores": list(result.scores),
        "ranking": ranking,
        "utilities": list(result.utilities),
        "target_rank": ranking[target_seat],
        "target_score": result.scores[target_seat],
        "target_win": 1 if ranking[target_seat] == 0 else 0,
        "decisions": decisions,
    }


def _run_cyclic_task(task):
    run_id, group, index, k, agents, num_players = task[:6]
    models = task[6] if len(task) > 6 else None
    return play_cyclic_game(load_cards(), run_id, group, index, k, agents,
                            num_players, models=models)


def cyclic_tasks(run_id, group, start_index, clusters, agents, num_players,
                 skip=(), models=None):
    """論理座標だけのタスク列。モデル席があれば models(SHA 照合用)を 7 要素目に付ける。"""
    skip = set(skip)
    extra = (models,) if models else ()
    return [(run_id, group, start_index + c, k, tuple(agents), num_players)
            + extra
            for c in range(clusters) for k in range(num_players)
            if (start_index + c, k) not in skip]


def run_cyclic_games(tasks, workers=1):
    """局を実行して (index, k) 順の record を返す。worker 数は結果に影響しない。"""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1 or len(tasks) <= 1:
        records = [_run_cyclic_task(t) for t in tasks]
    else:
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=workers)
        try:
            records = list(pool.map(_run_cyclic_task, tasks, chunksize=1))
        finally:
            pool.close()
            pool.join()
    return sorted(records, key=lambda r: (r["index"], r["seat_variant"]))


def record_line(record):
    return json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"


def write_multi_records(records, path):
    """1 局 1 行 JSONL(LF 固定、(index, k) 順)。"""
    ordered = sorted(records, key=lambda r: (r["index"], r["seat_variant"]))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in ordered:
            fh.write(record_line(rec))


def read_multi_records(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("%s:%d is not JSON: %s" % (path, lineno, exc))
    return out


def _check_ranking(rec):
    """保存された scores / 終局時の開始プレイヤーから順位を再計算して照合する。"""
    n = rec["n"]
    scores = rec["scores"]
    start = rec["final_start_player"]
    order = sorted(range(n), key=lambda pid: (-scores[pid], (pid - start) % n))
    expected = [0] * n
    for place, pid in enumerate(order):
        expected[pid] = place
    if expected != rec["ranking"]:
        raise ValueError("ranking does not follow scores/tie-break: index %d k=%d"
                         % (rec["index"], rec["seat_variant"]))
    utilities = [1.0 - 2.0 * r / (n - 1) for r in expected]
    if any(abs(a - b) > 1e-12 for a, b in zip(utilities, rec["utilities"])):
        raise ValueError("utilities do not follow D1 rank-linear map: index %d k=%d"
                         % (rec["index"], rec["seat_variant"]))


def missing_games(records, start_index, clusters, num_players):
    """対応表に無い (index, k) の一覧(決定論的に再開する局)。"""
    have = set((r["index"], r["seat_variant"]) for r in records)
    return [(start_index + c, k) for c in range(clusters)
            for k in range(num_players) if (start_index + c, k) not in have]


def multi_cluster_results(records, run_id, start_index, clusters, num_players,
                          agents, protocol=MULTI_PROTOCOL, agents_for=None):
    """保存された record だけからクラスタ値を作る。欠け・不整合は拒否する。

    戻り値は `(cluster_results, per_cluster_rows)`。欠けたクラスタがあれば
    ValueError(部分集合で合否を出さない。D4)。
    protocol / agents_for は Phase 12b の opt-in: 別 protocol の record を
    混ぜない照合と、クラスタ index ごとの正準 agent 列(異種卓の他家順序の
    均等化)を与える。既定は従来どおり全クラスタ同じ `agents`。
    """
    agents = list(agents)
    by_key = {}
    for rec in records:
        if rec.get("schema") != MULTI_RECORD_SCHEMA or \
                rec.get("protocol") != protocol:
            raise ValueError("record is not %s/%s" % (MULTI_RECORD_SCHEMA,
                                                       protocol))
        if rec["run_id"] != run_id:
            raise ValueError("record from another run: %s" % rec["run_id"])
        key = (rec["index"], rec["seat_variant"])
        if key in by_key:
            raise ValueError("duplicate record index %d k=%d" % key)
        by_key[key] = rec
    missing = missing_games(records, start_index, clusters, num_players)
    if missing:
        raise ValueError("%d games missing (first %r); resume the same run "
                         "before judging" % (len(missing), missing[0]))
    extra = set(by_key) - set((start_index + c, k) for c in range(clusters)
                              for k in range(num_players))
    if extra:
        raise ValueError("records outside the registered band: %r"
                         % sorted(extra)[:3])
    cluster_values = []
    rows = []
    for c in range(clusters):
        index = start_index + c
        games = [by_key[(index, k)] for k in range(num_players)]
        hashes = set(g["initial_state_hash"] for g in games)
        if len(hashes) != 1:
            raise ValueError("cluster %d did not share an initial deal" % index)
        seat_cover = [[None] * num_players for _ in range(num_players)]
        expected_agents = agents if agents_for is None else \
            list(agents_for(index))
        for k, g in enumerate(games):
            if g["n"] != num_players or g["agents"] != expected_agents:
                raise ValueError("cluster %d has another agent list / n" % index)
            if g["seat_agents"] != cyclic_seat_order(num_players, k):
                raise ValueError("cluster %d k=%d is not the cyclic order"
                                 % (index, k))
            if [g["seat_agents"].index(j) for j in range(num_players)] != \
                    g["order"] or g["target_seat"] != g["order"][0]:
                raise ValueError("cluster %d k=%d order/target mismatch"
                                 % (index, k))
            _check_ranking(g)
            if g["target_win"] != (1 if g["ranking"][g["target_seat"]] == 0
                                   else 0):
                raise ValueError("cluster %d k=%d target_win mismatch"
                                 % (index, k))
            for seat, j in enumerate(g["seat_agents"]):
                if seat_cover[j][seat] is not None:
                    raise ValueError("cluster %d agent %d sat twice at seat %d"
                                     % (index, j, seat))
                seat_cover[j][seat] = k
        if any(v is None for row in seat_cover for v in row):
            raise ValueError("cluster %d is not seat-balanced" % index)
        wins = sum(g["target_win"] for g in games)
        cluster_values.append(wins / float(num_players))
        rows.append({
            "index": index,
            "initial_state_hash": games[0]["initial_state_hash"],
            "target_wins": wins,
            "cluster_value": wins / float(num_players),
            "target_ranks": [g["target_rank"] for g in games],
            "target_scores": [g["target_score"] for g in games],
        })
    return tuple(cluster_values), rows


def summarize_multi(records, conditions, run_id, threshold=None,
                    bootstrap_samples=BOOTSTRAP_SAMPLES,
                    bootstrap_seed=BOOTSTRAP_SEED):
    """保存物(records + 条件)だけから集計・CI・合否を再計算する。"""
    n = conditions["num_players"]
    start = conditions["start_index"]
    clusters = conditions["clusters"]
    cluster_values, rows = multi_cluster_results(
        records, run_id, start, clusters, n, conditions["agents"])
    _check_seat_provenance(records, conditions)
    lo, hi = cluster_bootstrap(cluster_values, bootstrap_samples, bootstrap_seed)
    games = clusters * n
    wins = sum(r["target_wins"] for r in rows)
    ranks = [rank for r in rows for rank in r["target_ranks"]]
    rank_hist = [ranks.count(place) for place in range(n)]
    point = sum(cluster_values) / len(cluster_values)
    seat_wins = [0] * n
    for rec in records:
        seat_wins[rec["target_seat"]] += rec["target_win"]
    summary = {
        "schema": MULTI_SUMMARY_SCHEMA,
        "protocol": MULTI_PROTOCOL,
        "run_id": run_id,
        "num_players": n,
        "agents": list(conditions["agents"]),
        "group": conditions["group"],
        "start_index": start,
        "end_index": start + clusters - 1,
        "clusters": clusters,
        "games": games,
        "wins": wins,
        "draws": 0,
        "losses": games - wins,
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "threshold": threshold,
        "passed": (lo > threshold) if threshold is not None else None,
        "baseline_1_over_n": 1.0 / n,
        "target_rank_histogram": rank_hist,
        "target_mean_rank": sum(ranks) / float(len(ranks)),
        "target_mean_score": sum(s for r in rows for s in r["target_scores"])
        / float(games),
        "target_wins_by_seat": seat_wins,
        "cluster_results": [
            {"index": r["index"], "value": r["cluster_value"],
             "target_ranks": r["target_ranks"],
             "initial_state_hash": r["initial_state_hash"]} for r in rows],
        "limitations": "cyclic (not all n!) seatings; adjacency-order bias "
                       "between heterogeneous opponents is not cancelled",
    }
    if conditions.get("models"):
        # モデル席の SHA・探索構成・policy seed の規則(席別の実値は records)。
        summary["models"] = conditions["models"]
        summary["strong_search"] = conditions.get("strong_search")
        summary["policy_seed_rule"] = conditions["policy_seed"]
    return summary


def _check_seat_provenance(records, conditions):
    """record の席欄(agent・model SHA・探索構成・policy seed)を条件と照合する。

    保存物だけから「どのモデルが・どの seed で・どの探索構成で座ったか」を
    検証できるようにする。heuristic/random 席は model/search が None。
    """
    models = conditions.get("models") or {}
    for rec in records:
        root = eval_seed(rec["group"], rec["index"])
        for seat_rec in rec["seats"]:
            spec = seat_rec["agent"]
            where = "index %d k=%d seat %d" % (rec["index"], rec["seat_variant"],
                                               seat_rec["seat"])
            if spec != rec["agents"][seat_rec["agent_index"]]:
                raise ValueError("%s: agent does not match the order" % where)
            seed = derive_seed(root, "policy", rec["index"], seat_rec["seat"])
            if seat_rec["policy_seed"] != seed:
                raise ValueError("%s: policy_seed is not derived from the "
                                 "cluster coordinates" % where)
            kind, unused_path = parse_multi_agent_spec(spec)
            if kind in MULTI_MODEL_KINDS:
                if spec not in models or \
                        seat_rec["model_sha256"] != models[spec]["sha256"]:
                    raise ValueError("%s: model_sha256 differs from the run "
                                     "conditions" % where)
                expected_search = None
                if kind == "strong":
                    expected_search = dict(STRONG_SEARCH, root_seed=seed)
                if seat_rec["search"] != expected_search:
                    raise ValueError("%s: search configuration differs from "
                                     "the fixed D5 strong setting" % where)
            elif seat_rec["model_sha256"] is not None or \
                    seat_rec["search"] is not None:
                raise ValueError("%s: %s seat must not carry a model" %
                                 (where, spec))
