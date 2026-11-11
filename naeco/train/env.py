"""自己対戦エピソード駆動と並列ロールアウト(Phase 3 設計書 §4.3)。

- 標準ライブラリのみを使う(numpy / torch を import しない、§2)。
- seed は `derive_seed` だけから派生する。worker 番号・完了順・PID を
  seed に混ぜない(worker 数を変えても結果が完全一致する、§6.3)。
- `agents` はエージェントの**正準順**(識別子順)で渡す。`play_episode` が
  既定では決定論的巡回 `order[seat] = (seat + seat_variant) % n` で席へ
  割り当て、`Trajectory.seats[seat]` に識別子を残す(設計書 §6.3 のペア評価)。
  学習時は `learner_index` で学習対象エージェント(正準順の位置)を指定し、
  席置換後の実際の席を `Trajectory.learner_seat` へ記録する(設計書 14.1)。
  学習時の席ランダム化は `randomize_seats=True` を明示したときだけ
  席ストリーム `"seat"` 由来の置換を使う(§4.3 の分離)。
- `LegalityError` / `InvariantError` はエンジンのバグとしてその局を破棄し、
  カウンタとログに残す(core-api §9.4)。破棄局を学習データへ混ぜてはならない。
"""
import dataclasses
import logging
import multiprocessing
import os
import random
import time
from typing import Optional

from naeco.engine import cards as cards_mod
from naeco.engine import game as game_mod
from naeco.engine import observe as observe_mod
from naeco.engine.state import (
    InvariantError, LegalityError, Phase, derive_seed, eval_seed, state_hash,
)
from naeco.train import encoding as enc
from naeco.train import reward as reward_mod

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# 壁時計の絶対 deadline。反復回数で書かない(設計書 §8.3)。
EPISODE_DEADLINE_SECONDS = 300.0

_logger = logging.getLogger("naeco.train.env")

# 破棄カウンタ。process 単位で増える。rollout は親側でも集約する。
_DISCARD_COUNTS = {"LegalityError": 0, "InvariantError": 0}


def _fix_thread_env():
    """spawn 前、および import 時に4つのスレッド環境変数を "1" へ固定する。"""
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


_fix_thread_env()


def discard_counts():
    """これまでに破棄した局の種類別カウンタ(コピー)を返す。"""
    return dict(_DISCARD_COUNTS)


def _count_discard(kind):
    _DISCARD_COUNTS[kind] = _DISCARD_COUNTS.get(kind, 0) + 1


@dataclasses.dataclass(frozen=True)
class Decision:
    player_id: int
    obs_vec: tuple
    action_vecs: tuple      # 合法手ごと。legal_actions と同順
    chosen_index: int
    reward: float           # その決定点に帰属する報酬(中間決定は 0.0)
    turn_advanced: bool


@dataclasses.dataclass(frozen=True)
class Trajectory:
    group: str
    index: int
    seat_variant: int
    seats: tuple            # seats[seat] = エージェント識別子(文字列)
    decisions: tuple
    terminal_rewards: tuple
    initial_state_hash: str
    discarded: bool
    discard_reason: str
    # 学習対象席(設計書 14.1)。play_episode / rollout が learner_index で
    # 学習対象エージェントを指定されたとき、席置換後の実際の席を記録する。
    # None は「学習対象を指定しない」(trajectories_to_batch は全決定を入れる
    # 後方互換の経路)。
    learner_seat: Optional[int] = None
    # Multi-seat diagnostic observations: (viewer, is_actor, pending, vector).
    value_probes: tuple = ()


def _agent_name(agent):
    """エージェント識別子。`name` 属性が文字列ならそれ、なければクラス名。"""
    name = getattr(agent, "name", None)
    if isinstance(name, str):
        return name
    return type(agent).__name__


def _seat_order(root, index, seat_variant, num_players, randomize_seats=False):
    """席割当を返す。order[seat] = エージェントの入力位置。

    既定は決定論的巡回 `order[seat] = (seat + seat_variant) % num_players`。
    N=2 では seat_variant=0 が恒等、=1 が席交換になる(設計書 §6.3 のペア評価)。
    学習時の席ランダム化は `randomize_seats=True` のときだけ "seat" ストリーム
    由来の置換を使う(設計書 §4.3・§6.3 の分離)。
    """
    if not randomize_seats:
        return [
            (seat + seat_variant) % num_players
            for seat in range(num_players)
        ]
    order = list(range(num_players))
    random.Random(derive_seed(root, "seat", index, seat_variant)).shuffle(order)
    return order


def _potentials(state, cards):
    """各席から観測した Φ のタプル(席 0..N-1 順)。"""
    return tuple(
        reward_mod.score_diff_potentials(
            observe_mod.observe(state, pid, cards), cards)[pid]
        for pid in range(state.num_players)
    )


def play_episode(group, index, seat_variant, agents, cards=None,
                 reward_config=None, record_vectors=True,
                 randomize_seats=False, learner_index=None):
    """1エピソードを終端まで駆動し、決定列と終端報酬を返す。

    `Decision.reward` はその決定点へ帰属する shaped step 報酬のみで、主報酬
    (終端効用)は `Trajectory.terminal_rewards` に分離する。既定 shaping では
    全決定の reward が 0.0 になる。`randomize_seats=False` が既定で、席割当は
    決定論的巡回。学習時のみ `randomize_seats=True` で "seat" ストリーム由来の
    ランダム化を使う。

    `learner_index` は agents(正準順)のうち学習対象エージェントの位置
    (既定 None = 指定なし)。指定すると、席置換(seat_variant / 席ランダム化)
    の**後**の実際の席を `Trajectory.learner_seat` へ記録する(設計書 14.1)。
    呼び出し側が席を直接渡すのではなく、学習対象エージェントを指定して
    席の解決は env に委ねることで、席を入れ替えても learner を正しく
    特定できる。
    """
    cards = cards or cards_mod.load_cards()
    if reward_config is None:
        reward_config = reward_mod.RewardConfig()
    agent_list = tuple(agents)
    n = len(agent_list)
    encoder = enc
    if n in (3, 4):
        from naeco.train import encoding_multi
        encoder = encoding_multi
    if n not in (2, 3, 4):
        raise ValueError("num players must be 2..4, got %d agents" % n)

    root = eval_seed(group, index)
    # "env" / "setup" は席非依存ストリーム。ペアの2局は同一配札・同一開始席。
    env_seed = derive_seed(root, "env", index)
    setup_seed = derive_seed(root, "setup", index)
    state = game_mod.new_game(n, env_seed, cards, setup_seed=setup_seed)
    initial_hash = state_hash(state)

    order = _seat_order(root, index, seat_variant, n,
                        randomize_seats=randomize_seats)
    seated = [agent_list[pid] for pid in order]
    seats = tuple(_agent_name(agent) for agent in seated)

    # 学習対象エージェント(正準順の位置)を、席置換後の実際の席へ解決する
    # (設計書 14.1)。席は env 内で決まるため、呼び出し側ではなくここで
    # 記録する。
    learner_seat = None
    if learner_index is not None:
        if not (0 <= learner_index < n):
            raise ValueError(
                "learner_index must be in 0..%d, got %r"
                % (n - 1, learner_index))
        learner_seat = order.index(learner_index)

    use_shaping = reward_config.shaping != "none"
    prev_potentials = _potentials(state, cards) if use_shaping else None
    decisions = []
    value_probes = []
    probe_keys = set()
    deadline = time.monotonic() + EPISODE_DEADLINE_SECONDS
    terminal_rewards = ()
    try:
        while state.phase != Phase.GAME_OVER:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "episode did not terminate within %.1fs (%s/%d v=%d)"
                    % (EPISODE_DEADLINE_SECONDS, group, index, seat_variant))
            who = game_mod.actor(state)
            obs = observe_mod.observe(state, who, cards)
            if n > 2 and record_vectors:
                for pid in range(n):
                    key = (pid, pid == who, state.pending is not None)
                    if key in probe_keys:
                        continue
                    probe_keys.add(key)
                    viewer_obs = obs if pid == who else observe_mod.observe(state, pid, cards)
                    value_probes.append(key + (tuple(encoder.encode_observation(viewer_obs, cards)),))
            legal = obs.legal_actions
            if legal is None:
                raise RuntimeError("actor %d has no legal actions" % who)
            action = seated[who].act(obs)
            try:
                chosen_index = legal.index(action)
            except ValueError:
                chosen_index = -1
            if record_vectors:
                obs_vec = tuple(encoder.encode_observation(obs, cards))
                action_vecs = tuple(
                    tuple(encoder.encode_action(obs, act, cards)) for act in legal)
            else:
                obs_vec = ()
                action_vecs = ()
            result = game_mod.step(state, action, cards)
            if prev_potentials is not None:
                step_rewards = reward_mod.step_rewards(
                    prev_potentials, result, reward_config, cards)
            else:
                step_rewards = (0.0,) * n
            decisions.append(Decision(
                player_id=who,
                obs_vec=obs_vec,
                action_vecs=action_vecs,
                chosen_index=chosen_index,
                reward=step_rewards[who],
                turn_advanced=result.turn_advanced,
            ))
            # turn_advanced=False の step の後は Φ の参照値を更新しない
            # (step_rewards の契約。telescope が壊れる)。
            if result.turn_advanced and prev_potentials is not None:
                prev_potentials = _potentials(result.state, cards)
            state = result.state
        terminal_rewards = reward_mod.terminal_rewards(state, cards)
    except (LegalityError, InvariantError) as exc:
        kind = type(exc).__name__
        _count_discard(kind)
        _logger.error(
            "discarded %s/%d seat_variant=%d (%s): %s",
            group, index, seat_variant, kind, exc, exc_info=True)
        return Trajectory(
            group=group, index=index, seat_variant=seat_variant,
            seats=seats, decisions=tuple(decisions),
            terminal_rewards=(), initial_state_hash=initial_hash,
            discarded=True, discard_reason=str(exc),
            learner_seat=learner_seat,
        )
    return Trajectory(
        group=group, index=index, seat_variant=seat_variant,
        seats=seats, decisions=tuple(decisions),
        terminal_rewards=terminal_rewards, initial_state_hash=initial_hash,
        discarded=False, discard_reason="",
        learner_seat=learner_seat, value_probes=tuple(value_probes),
    )


_WORKER_CARDS = None
_WORKER_REWARD_CONFIG = None
_WORKER_RANDOMIZE_SEATS = False
_WORKER_LEARNER_INDEX = None


def _worker_init(cards, reward_config, randomize_seats, learner_index):
    global _WORKER_CARDS, _WORKER_REWARD_CONFIG
    global _WORKER_RANDOMIZE_SEATS, _WORKER_LEARNER_INDEX
    _WORKER_CARDS = cards
    _WORKER_REWARD_CONFIG = reward_config
    _WORKER_RANDOMIZE_SEATS = randomize_seats
    _WORKER_LEARNER_INDEX = learner_index


def _run_spec(spec):
    """spawn ワーカー内で1spec を実行し、破棄種類も親へ返す。"""
    group, index, seat_variant, agents = spec
    before = dict(_DISCARD_COUNTS)
    trajectory = play_episode(
        group, index, seat_variant, agents,
        cards=_WORKER_CARDS, reward_config=_WORKER_REWARD_CONFIG,
        randomize_seats=_WORKER_RANDOMIZE_SEATS,
        learner_index=_WORKER_LEARNER_INDEX)
    kinds = tuple(sorted(
        kind for kind, count in _DISCARD_COUNTS.items()
        if count > before.get(kind, 0)))
    return trajectory, kinds


def _check_spec(spec):
    if not isinstance(spec, (tuple, list)):
        raise ValueError(
            "spec must be a (group, index, seat_variant, agents) tuple, got %r"
            % (spec,))
    spec = tuple(spec)
    if len(spec) != 4:
        raise ValueError(
            "spec must have 4 items (group, index, seat_variant, agents), "
            "got %d" % len(spec))
    return spec


def rollout(specs, workers=1, cards=None, reward_config=None,
            randomize_seats=False, learner_index=None):
    """specs を順序保持のまま並列実行し、Trajectory の list を返す。

    spec は `(group, index, seat_variant, agents)` の4タプル。`agents` は
    `play_episode` と同じ正準順の並び。worker 数を変えても各局の展開は
    完全一致する(seed は論理座標だけから派生し、worker index に依存しない)。
    `randomize_seats` は `play_episode` と同じ契約で、学習時のみ True にする。
    `learner_index` も `play_episode` と同じ契約で、全 spec の学習対象
    エージェント位置(正準順)。指定すると各 Trajectory の `learner_seat` へ
    席置換後の実際の席が記録される(設計書 14.1)。
    """
    _fix_thread_env()
    cards = cards or cards_mod.load_cards()
    if reward_config is None:
        reward_config = reward_mod.RewardConfig()
    spec_list = [_check_spec(spec) for spec in specs]
    if workers == 1:
        return [
            play_episode(group, index, seat_variant, agents,
                         cards=cards, reward_config=reward_config,
                         randomize_seats=randomize_seats,
                         learner_index=learner_index)
            for group, index, seat_variant, agents in spec_list
        ]
    if workers < 1:
        raise ValueError("workers must be >= 1, got %d" % workers)
    context = multiprocessing.get_context("spawn")
    with context.Pool(
            processes=workers, initializer=_worker_init,
            initargs=(cards, reward_config, randomize_seats,
                      learner_index)) as pool:
        raw = list(pool.map(_run_spec, spec_list))
    results = []
    for trajectory, kinds in raw:
        for kind in kinds:
            _count_discard(kind)
        results.append(trajectory)
    return results
