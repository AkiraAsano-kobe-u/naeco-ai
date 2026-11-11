"""報酬帰属(Phase 3 設計書 §4.2 / core-api §9)。

主報酬は終端の `game.terminal_utilities` のみ。中間報酬は既定で 0。
`shaping="score_diff"` はポテンシャル型の得点差シェーピングで、終端の主報酬を
変えない。標準ライブラリのみを使う(§2)。
"""
import dataclasses

from naeco.engine import cards as cards_mod
from naeco.engine import game as game_mod
from naeco.engine import observe as observe_mod

# Keep the legacy constant for existing two-seat checkpoints. Multi-seat callers
# select the contract internally from num_players, never from supplied metadata.
REWARD_VERSION = "1.0.0"
MULTI_REWARD_VERSION = "2.0.0"


def reward_version(num_players):
    """Return the reward contract for a supported player count (Phase 10 D2)."""
    if type(num_players) is not int or num_players not in (2, 3, 4):
        raise ValueError("num_players must be an integer in 2..4")
    return REWARD_VERSION if num_players == 2 else MULTI_REWARD_VERSION

# ポテンシャル式の γ。RewardConfig に割引率フィールドが無く、reward.py は
# torch 層の PPOConfig.gamma を import できないため、本モジュールの定義で固定する。
# γ=1 のとき、報酬帰属された step の shaped 報酬合計は Φ(s_T) − Φ(s_0) へ厳密に
# telescope する。
SHAPING_GAMMA = 1.0

# Φ の正規化。得点偏差を 100 で割る(設計書 §5.2 の金額正規化と同じ桁合わせ)。
_SCORE_NORMALIZER = 100.0


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    shaping: str = "none"        # "none" | "score_diff"
    shaping_scale: float = 0.0   # shaping="none" のとき無視。既定 0.0


def terminal_rewards(state, cards=None):
    """N人ぶんの終端効用。`game.terminal_utilities` をそのまま返す。

    独自の勝敗判定や順位写像は持たない。
    """
    return game_mod.terminal_utilities(state, cards)


def score_diff_potentials(obs, cards=None):
    """観測から得点差ベースのポテンシャル Φ を計算する。

    戻り値は長さ N の float タプルで、**席 0..N-1 の絶対席順**に並ぶ
    (Observation.player_id を基準に逆算する。観測者相対ではない)。
    `step_rewards` の `prev_obs_potentials` にはこの関数の出力を渡す。

    Φ_i(s) = (estimated_score_i − mean(estimated_scores)) / 100.0
    estimated_score_i = 所持金 + 所有建物の printed_value 合計
                        − EFF-unpaid-wage-01.points_per × 未払い賃金

    on_game_end 効果の得点は Observation から見えないため含まない。
    終端の主報酬を置き換えるものではなく、学習の leading indicator として
    定義する(設計書 §4.2 の「Observation から計算できる得点差ベースの量」)。
    """
    cards = cards or cards_mod.load_cards()
    n = obs.num_players
    penalty = cards.effect_params["EFF-unpaid-wage-01"]["points_per"]
    estimates = [0.0] * n
    estimates[obs.player_id] = _estimated_score(
        obs.my_money, obs.my_owned, obs.my_unpaid_wages, penalty, cards)
    for view in obs.others:
        pid = (obs.player_id + view.seat_offset) % n
        estimates[pid] = _estimated_score(
            view.money, view.owned, view.unpaid_wages, penalty, cards)
    mean = sum(estimates) / n
    return tuple((value - mean) / _SCORE_NORMALIZER for value in estimates)


def _estimated_score(money, owned, unpaid_wages, penalty, cards):
    """score_breakdown の内訳のうち Observation から見える項だけを推定する。"""
    building_value = sum(cards.printed_value(iid) or 0 for iid in owned)
    return float(money + building_value - penalty * unpaid_wages)


def step_rewards(prev_obs_potentials, step_result, config, cards=None):
    """1 step の報酬タプル(長さ N、席 0..N-1 順)。

    - `shaping="none"`: 常に全 0。
    - `step_result.turn_advanced == False`: シェーピング有効時も全 0
      (中間決定・同一決定の継続には報酬を帰属しない)。
    - `shaping="score_diff"`:
      `r_i = shaping_scale × (SHAPING_GAMMA × Φ_i(s') − prev_obs_potentials[i])`。
      Φ(s') は step_result.state を各席から観測して `score_diff_potentials` で計算。
      `prev_obs_potentials` は「直前に報酬帰属された step の後の観測」に対する
      `score_diff_potentials` の出力(長さ N)を渡す。turn_advanced=False の step の
      後は**更新しない**こと(更新すると Φ の telescope が壊れる)。
    - N>=3 の shaping は "none" のみ(Phase 10 D1)。score_diff は拒否する。
    - 未知の `shaping` 値は `ValueError`。
    """
    if config.shaping not in ("none", "score_diff"):
        raise ValueError("unknown shaping: {!r}".format(config.shaping))
    n = step_result.state.num_players
    if n >= 3 and config.shaping != "none":
        raise ValueError("multi-player rewards require shaping='none'")
    if config.shaping == "none" or not step_result.turn_advanced:
        return (0.0,) * n
    if len(prev_obs_potentials) != n:
        raise ValueError(
            "prev_obs_potentials must have length %d, got %d"
            % (n, len(prev_obs_potentials)))
    cards = cards or cards_mod.load_cards()
    next_potentials = tuple(
        score_diff_potentials(
            observe_mod.observe(step_result.state, pid, cards), cards)[pid]
        for pid in range(n)
    )
    return tuple(
        config.shaping_scale * (SHAPING_GAMMA * nxt - prev)
        for nxt, prev in zip(next_potentials, prev_obs_potentials)
    )
