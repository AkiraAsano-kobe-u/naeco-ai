"""探索方式で共有する予算、統計、ロールアウト。"""
import dataclasses
import time

from naeco.ai.value import rollout_view, static_value
from naeco.engine import game
from naeco.engine.state import Phase


H = 2


class _WallDeadline(object):
    """固定壁時計の共有期限。親予算とサブ予算で同一インスタンスを共有する。

    `check` が「期限をまたいだ」境界を最初の1回だけ overrun として記録する。
    同じ超過を親とサブ予算で二重計上しないため、記録はこのオブジェクトに集める。
    """

    def __init__(self, deadline, clock):
        self.deadline = deadline
        self.clock = clock
        self.expired = False
        self.overrun_total_ms = 0.0
        self.overrun_max_ms = 0.0
        self.overrun_count = 0
        self.overruns_ms = []


@dataclasses.dataclass
class Budget:
    """探索が使ってよい計算量。すべての `game.step` はここを通す。

    主軸は**状態遷移数**(決定論的で再現する)。`seconds` を与えると
    **固定壁時計**での比較もできる(設計書 D-6 の交差確認。sol SOL-P25G-003)。
    時間軸は再現しないので、判定には使わず交差確認にだけ使う。
    `clock` は単調時計を注入するためのもので、テストから差し替えられる。
    """
    transitions: int
    seconds: object = None
    clock: object = time.monotonic
    # 実際に消費した遷移数。`remaining` は時間切れで 0 を返すので、消費量の計算に
    # 使うと未消費ぶんまで消費したことになる(時間予算のテストで検出した)。
    spent: int = dataclasses.field(default=0, init=False)
    _wall: object = dataclasses.field(default=None, init=False)
    _start: object = dataclasses.field(default=None, init=False)

    def start(self):
        """サブ予算の経過時間も決める1決定点の開始時刻を固定する。"""
        self._wall = None
        self._start = self.clock()
        self._deadline = (None if self.seconds is None
                          else self._start + self.seconds)
        if self._deadline is not None:
            self._wall = _WallDeadline(self._deadline, self.clock)
        return self

    @property
    def elapsed_ms(self):
        """論理経過時間(ms)。`time.perf_counter` に頼らず決定論的にする。"""
        if self._start is None:
            return 0.0
        return max(0.0, (self.clock() - self._start) * 1000.0)

    def cap(self, limit):
        """同じ期限を共有する上限付きサブ予算を作る。"""
        sub = Budget(limit, clock=self.clock)
        sub._wall = self._wall
        return sub

    @property
    def remaining(self):
        """残り遷移数。時間切れなら 0 を返す(呼び出し側の停止条件が共通になる)。"""
        if self._wall is not None:
            if self._wall.expired or self.clock() >= self._wall.deadline:
                return 0
        return self.transitions

    def check(self):
        """sample/action 境界の期限確認。残り遷移数を返し、超過を監査記録する。

        `remaining` は記録しない読み取り専用で、探索ループの停止判定は原則こちらを
        使う。実行中の `step` が期限をまたぐのは不可避なので、超過は次の境界で
        最初の1回だけ `deadline_overruns_ms` へ残す。
        """
        if self._wall is None:
            return self.transitions
        if self._wall.expired:
            return 0
        now = self.clock()
        if now >= self._wall.deadline:
            self._wall.expired = True
            overrun_ms = (now - self._wall.deadline) * 1000.0
            if overrun_ms > 0:
                self._wall.overrun_total_ms += overrun_ms
                self._wall.overrun_max_ms = max(self._wall.overrun_max_ms,
                                                overrun_ms)
                self._wall.overrun_count += 1
                self._wall.overruns_ms.append(overrun_ms)
            return 0
        return self.transitions

    def step(self, state, action, cards):
        # 硬い上限は**遷移数**だけ。時間は `remaining` を呼び出し側がポーリングして
        # 止める軟らかい期限である。step の実行中に期限が来るのは preempt できない
        # ので、ここで時間を理由に例外を投げると正常な探索が落ちる。
        if self.transitions <= 0:
            raise RuntimeError("search transition budget is exhausted")
        self.transitions -= 1
        self.spent += 1
        return game.step(state, action, cards).state, 1

    @property
    def deadline_overrun_total_ms(self):
        return self._wall.overrun_total_ms if self._wall is not None else 0.0

    @property
    def deadline_overrun_max_ms(self):
        return self._wall.overrun_max_ms if self._wall is not None else 0.0

    @property
    def deadline_overrun_count(self):
        return self._wall.overrun_count if self._wall is not None else 0

    @property
    def deadline_expired(self):
        return self._wall is not None and self._wall.expired

    @property
    def deadline_overruns_ms(self):
        return self._wall.overruns_ms if self._wall is not None else []


@dataclasses.dataclass
class SearchStats:
    """設計書 §6.7 で定めた決定点ごとの集計元。"""
    decisions: int = 0
    transitions: int = 0
    sampler_calls: int = 0
    nodes_created: int = 0
    total_nodes: int = 0                # nodes_created と同じ累計。nodes/s の分子
    determinizations: int = 0           # PIMC の K は予算から決まるので実測値を残す
    elapsed_ms: object = dataclasses.field(default_factory=list)
    search_elapsed_ms: float = 0.0      # elapsed_ms の合計。nodes/s の分母
    samples_per_action: object = dataclasses.field(default_factory=list)
    transitions_per_action: object = dataclasses.field(default_factory=list)
    max_sample_gap: object = dataclasses.field(default_factory=list)
    zero_sample_actions: object = dataclasses.field(default_factory=list)
    decisions_with_zero_sample_actions: int = 0
    truncated_samples: int = 0
    clip_saturated: int = 0
    evaluations: int = 0
    delegated_calls: int = 0      # policy_agent 委譲の決定数(委譲先 net 呼出数)
    deadline_overruns_ms: object = dataclasses.field(default_factory=list)
    deadline_overrun_total_ms: float = 0.0
    deadline_overrun_max_ms: float = 0.0
    deadline_overrun_count: int = 0


def merge_deadline_overruns(budget, stats):
    """Budget が記録した期限超過を SearchStats へ集約する。"""
    stats.deadline_overrun_total_ms += budget.deadline_overrun_total_ms
    stats.deadline_overrun_max_ms = max(stats.deadline_overrun_max_ms,
                                        budget.deadline_overrun_max_ms)
    stats.deadline_overrun_count += budget.deadline_overrun_count
    stats.deadline_overruns_ms.extend(budget.deadline_overruns_ms)


def choose_heuristic_action(state, cards, heuristic):
    """下位決定を既存ヒューリスティックと同じ採点で解く。"""
    actor = game.actor(state)
    legal = game.legal_actions(state, cards)
    if actor is None or not legal:
        return None
    view = rollout_view(state, cards, actor)
    return sorted(legal, key=lambda action: (-heuristic._score(action, view),
                                              action.sort_key()))[0]


def evaluate_static(state, cards, stats):
    """静的評価を記録し、非終端の clip 飽和も数える。"""
    value = static_value(state, cards)
    stats.evaluations += 1
    if state.phase != Phase.GAME_OVER:
        # 2 席は従来どおり value[0] だけで判定(1-p なので同値)。3/4 席は各席の
        # 成分が独立に clip されるので、いずれかの飽和を数える(Phase 10 D3)。
        probe = value[:1] if state.num_players == 2 else value
        if any(v == 0.0 or v == 1.0 for v in probe):
            stats.clip_saturated += 1
    return value


def rollout(state, budget, cards, heuristic, horizon, start_round, stats):
    """切詰め地平までヒューリスティック委譲で進め、現在局面を評価する。"""
    limit = start_round + horizon
    while state.phase != Phase.GAME_OVER and state.round_number <= limit:
        if budget.check() == 0:
            stats.truncated_samples += 1
            return evaluate_static(state, cards, stats)
        action = choose_heuristic_action(state, cards, heuristic)
        if action is None:
            break
        state, unused_used = budget.step(state, action, cards)
    return evaluate_static(state, cards, stats)


def rollout_from(state, action, budget, cards, heuristic, horizon, stats):
    """根行動を含めて一標本を実行し、消費遷移数も返す。"""
    before = budget.spent
    start_round = state.round_number
    state, unused_used = budget.step(state, action, cards)
    if budget.check() == 0:
        stats.truncated_samples += 1
        value = evaluate_static(state, cards, stats)
    else:
        value = rollout(state, budget, cards, heuristic, horizon, start_round, stats)
    return value, budget.spent - before
