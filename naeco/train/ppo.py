"""PPO 学習ループ(Phase 3 設計書 §4.8 / core-api §9)。

torch を import してよいのは `naeco/train/net.py` と本モジュールだけ(設計書 §2)。
import 時に4つのスレッド環境変数と `torch.set_num_threads(1)` を固定する(§8.4)。

設計上の注意(3点)と本実装の扱い
--------------------------------
1. 主報酬は終端のみ
   `Trajectory.terminal_rewards` が主報酬(2人戦で ±1.0、席順のタプル)で、
   `Decision.reward` は shaped step 報酬のみ(既定 shaping="none" では全ゼロ)。
   本モジュールは `trajectories_to_batch` で、**最後の決定の報酬へ
   `terminal_rewards[actor]` を加算**し、それ以外の決定は `Decision.reward` を
   そのまま使う。

2. 価値の符号規約(2人ゼロ和・手番視点) ★
   各決定点の観測はその決定の手番プレイヤー(actor)の観測であり、`net` の
   value 出力は **actor 視点の値**である。`Decision.reward` も env.py が
   `step_rewards[who]` を採るため actor に帰属済み、終端報酬は最後の決定者の
   席の `terminal_rewards` を使う。ここまでがすべて「その決定の actor 視点」。
   GAE のブートストラップは「次決定の actor 視点」の値で行うため、2人ゼロ和の
   自己対戦では **手番が交代したとき視点が反転する**(V^p0 = -V^p1 が厳密に
   成り立つ)。本実装では軌跡ごとに「先頭決定者の視点」を基準視点として固定し、
   flip = +1(先頭決定者と同一 player_id) / -1(相手) を全報酬・全価値に掛けた
   **固定視点列**で GAE を計算する(`compute_gae` 自身は視点を関知せず、
   固定視点の系列をそのまま GAE する)。その後、policy / value 損失は
   actor 視点で計算する必要があるため、アドバンテージと returns へ flip を
   掛け直してバッチへ格納する。
   flip は隣接決定の player_id 比較ではなく「先頭決定者との一致」で決まる点に
   注意(中間決定は同一 player_id なので自動的に無 flip になる)。

3. 中間決定(turn_advanced=False)の扱い
   中間決定も Decision として記録される(同一プレイヤーの連続決定、報酬ゼロ)。
   本実装はこれを**報酬ゼロの通常ステップ**として GAE のブートストラップ対象に
   含める(特別扱いしない)。視点は同一プレイヤーなので flip は変化せず、通常の
   割引・λ 伝播がそのまま効く。整合性チェックとして、
   `turn_advanced=False` の直後の決定が別プレイヤーになる軌跡は ValueError で
   拒否する。逆方向(turn_advanced=True でも同一プレイヤーが続く、ラウンド境界
   等)は許可する — 符号反転は player_id の実値で判定するため正確。

4. 学習対象の絞り込み(設計書 14.1)★
   `Trajectory.learner_seat` が設定されている軌跡は、**その席の決定だけ**を
   バッチへ入れる。相手(過去世代 / HeuristicAgent / RandomAgent)の手は
   環境遷移の一部として扱い、GAE は learner の決定列の上で計算する。既定
   shaping では相手の手番に報酬は発生しないため取りこぼしは無く、終端効用は
   learner の**最後の決定**へ帰属させる。learner の決定だけが残るためバッチ内
   の視点は単一になり flip は恒等になるが、**flip の仕組みは残す**
   (`learner_seat=None` の後方互換経路と、将来の多人数対応のため)。
   `learner_seat=None`(既定)は従来どおり全決定を入れる(現行方策どうしの
   純粋な自己対戦を想定した後方互換の経路)。

学習データは `naeco/train/env.py` の `Trajectory` / `Decision` から
`trajectories_to_batch` で作る。`Trajectory.discarded == True` の軌跡は
学習データへ混ぜない(設計書 §4.3)。
"""
import dataclasses
import os
import random

import torch

from naeco.engine.state import derive_seed

_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _fix_thread_env():
    """4つのスレッド環境変数を "1" へ固定する(設計書 §8.4)。"""
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


_fix_thread_env()
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# torch 操作の包み関数(scripts から torch を直接触らせない)。
# torch を import してよいのは naeco/train/net.py と本モジュールだけなので、
# scripts/train_ppo.py 等の学習CLIはここから torch 操作を呼ぶ。
# ---------------------------------------------------------------------------


def manual_seed(seed):
    """torch.manual_seed の包み(学習 root seed の固定)。"""
    torch.manual_seed(seed)


def set_num_threads(n):
    """torch.set_num_threads の包み(設計書 §8.4 のスレッド固定)。"""
    torch.set_num_threads(n)


def new_adam_optimizer(params, lr):
    """torch.optim.Adam の包み(Adam の生成)。"""
    return torch.optim.Adam(params, lr=lr)


def save_state_dict(obj, path):
    """torch.save の包み(重み / optimizer 状態の保存)。"""
    torch.save(obj, path)


def load_state_dict(path):
    """torch.load(map_location="cpu") の包み(重み / optimizer 状態の読込)。"""
    return torch.load(path, map_location="cpu")


def torch_version():
    """torch.__version__ の包み(再現メタデータ用)。"""
    return torch.__version__


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """PPO ハイパーパラメータ(設計書 §4.8 の固定値。変更禁止)。"""

    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5


def compute_gae(rewards, values, dones, config):
    """一般化アドバンテージ推定(GAE)を計算し `(advantages, returns)` を返す。

    規約(★ 呼び出し側で守ること):
    - `rewards` / `values` は**同一視点**(軌跡内で固定した1プレイヤー視点)の
      系列で与えること。2人ゼロ和の自己対戦では手番交代ごとに視点が反転する
      ため、呼び出し側(`trajectories_to_batch`)が player_id の変化を検出して
      符号を掛けてから渡す。本関数は視点を関知しない。
    - `dones[i] == True` は「決定 i でその軌跡が終わる」ことを表す。その決定の
      reward には**終端報酬が含まれている**こと。done の境界を越えて
      ブートストラップもアドバンテージの伝播もしない。
    - 同一プレイヤーの連続決定(中間決定、turn_advanced=False)は報酬ゼロの
      通常ステップとして扱う(特別扱いしない)。

    `rewards` / `values` / `dones` はリストでも Tensor でもよい。戻り値は
    float32 の Tensor。長さ 0 のときは空の Tensor を返す。
    """
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).detach()
    values_t = torch.as_tensor(values, dtype=torch.float32).detach()
    dones_t = torch.as_tensor(dones, dtype=torch.bool).detach()
    if rewards_t.dim() != 1 or values_t.dim() != 1 or dones_t.dim() != 1:
        raise ValueError("rewards / values / dones must be 1-D sequences")
    if not (len(rewards_t) == len(values_t) == len(dones_t)):
        raise ValueError(
            "rewards / values / dones must have the same length, "
            "got %d / %d / %d"
            % (len(rewards_t), len(values_t), len(dones_t)))
    t = len(rewards_t)
    advantages = torch.zeros(t, dtype=torch.float32)
    returns = torch.zeros(t, dtype=torch.float32)
    last_advantage = torch.zeros((), dtype=torch.float32)
    gamma_lambda = config.gamma * config.gae_lambda
    for i in range(t - 1, -1, -1):
        done = dones_t[i]
        next_value = torch.zeros((), dtype=torch.float32) if done \
            else values_t[i + 1]
        delta = rewards_t[i] + config.gamma * next_value - values_t[i]
        last_advantage = delta + gamma_lambda * (
            0.0 if done else last_advantage)
        advantages[i] = last_advantage
        returns[i] = advantages[i] + values_t[i]
    return advantages, returns


def policy_entropy(logits, mask):
    """可変長の合法手集合上の平均エントロピー(設計書 §4.8)。

    padding(mask=False) の位置は含めない。mask=False の logit は `-inf` 相当
    なので softmax 確率は 0、log 確率は -inf になる。素朴な `p * log p` は
    `0 * -inf = NaN` になるため、掛け算の前に log_probs をゼロ埋めする。

    実装の理由と backward の安全性(2026-08-12 の高重大度欠陥修正):
    - 旧実装は `torch.where(mask, probs * log_probs, 0)` と「積の後にマスク」を
      掛けていた。torch.where の backward は**両方の枝**の勾配を計算するため、
      padding 位置でも `probs * log_probs` の枝が微分され、upstream(マスクで 0)
      と `log_probs = -inf` の積 `0 * -inf` が NaN になった。log_softmax の
      backward は行内全要素の勾配和(grad_logit_i = grad_lp_i - p_i * Σ_j
      grad_lp_j)を持つため、1行に1つでも padding があるとその行の全 logit
      勾配が NaN になり、共有の obs_mlp / action_mlp 経由で net の全パラメータ
      へ伝播し、最初の padding 含有ミニバッチで学習が死んだ(Opus 実測、torch
      2.8.0)。
    - 修正は方針「掛け算に入る前に両オペランドを有限にする」に従い、積を
      取る**前**に `log_probs` を `torch.where(mask, log_probs, 0)` でゼロ埋め
      する。この where の他方の枝は定数 0 なので、backward は `grad * mask`
      だけで NaN を生まない。padding 位置の積は `probs = exp(-inf) = 0` と
      `log_probs = 0` の `0 * 0 = 0` になり、積の backward の偏微分も
      ∂terms/∂log_probs = probs = 0、∂terms/∂probs = log_probs = 0 で両方
      有限。したがって計算グラフのどこにも `0 * -inf` が現れず、backward は
      常に有限の勾配を返す。
    - forward の性質は従来どおり: padding の項は積が 0 なので合計へ寄与せず、
      「padding を含まない合法手集合上の平均エントロピー」のまま。勾配も
      合法手の logit には通常のエントロピー勾配(行平均、一様分布で 0)が
      正確に流れ、padding の logit には where の mask と log_softmax の行内和
      (`0 - p_i * Σ = 0`)で厳密に 0 が流れる。

    (歯: 元の torch.where 版へ戻すと tests/test_ppo.py の
    test_backward_with_padding_produces_finite_grads /
    test_tensor_level_backward_is_finite_with_inf_logits /
    test_update_with_padded_batch_keeps_weights_finite が NaN 勾配で落ちる。)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # 積の前に log_probs をゼロ埋めする(0 * -inf をグラフから排除)。
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    terms = probs * log_probs
    return -terms.sum(dim=-1).mean()


def _check_trajectory(traj):
    """1軌跡の整合性を検査する。問題があれば ValueError。"""
    decisions = traj.decisions
    if not decisions:
        raise ValueError(
            "non-discarded trajectory %s/%d has no decisions"
            % (traj.group, traj.index))
    if len(traj.terminal_rewards) in (3, 4) and traj.learner_seat is None:
        raise ValueError("multi-player PPO requires learner_seat")
    if len(traj.terminal_rewards) not in (2, 3, 4):
        raise ValueError(
            "PPO batch builder requires 2..4 players "
            "(trajectory %s/%d has %d terminal rewards)"
            % (traj.group, traj.index, len(traj.terminal_rewards)))
    players = [d.player_id for d in decisions]
    if any(p < 0 or p >= len(traj.terminal_rewards) for p in players):
        raise ValueError(
            "player_id outside terminal_rewards range in %s/%d"
            % (traj.group, traj.index))
    if traj.learner_seat is not None and not (
            0 <= traj.learner_seat < len(traj.terminal_rewards)):
        raise ValueError(
            "learner_seat %r outside terminal_rewards range in %s/%d"
            % (traj.learner_seat, traj.group, traj.index))
    action_dim = None
    for d in decisions:
        if not d.obs_vec:
            raise ValueError(
                "trajectory %s/%d has empty obs_vec "
                "(record_vectors=False の軌跡は学習に使えない)"
                % (traj.group, traj.index))
        if not d.action_vecs:
            raise ValueError(
                "trajectory %s/%d has a decision with no legal actions"
                % (traj.group, traj.index))
        if not (0 <= d.chosen_index < len(d.action_vecs)):
            raise ValueError(
                "chosen_index %d out of range in %s/%d"
                % (d.chosen_index, traj.group, traj.index))
        if action_dim is None:
            action_dim = len(d.action_vecs[0])
        if any(len(v) != action_dim for v in d.action_vecs):
            raise ValueError(
                "inconsistent action vector length in %s/%d"
                % (traj.group, traj.index))
    if not all(len(d.obs_vec) == len(decisions[0].obs_vec)
               for d in decisions):
        raise ValueError(
            "inconsistent obs vector length in %s/%d"
            % (traj.group, traj.index))
    # 中間決定の整合性: turn_advanced=False なら次も同一プレイヤーでなければ
    # ならない(設計上の注意 3)。
    for k in range(len(decisions) - 1):
        if not decisions[k].turn_advanced \
                and decisions[k + 1].player_id != decisions[k].player_id:
            raise ValueError(
                "turn_advanced=False なのに手番プレイヤーが変わっている "
                "(%s/%d decision %d: player %d -> %d)"
                % (traj.group, traj.index, k, decisions[k].player_id,
                   decisions[k + 1].player_id))
    return decisions, players, action_dim


def trajectories_to_batch(trajectories, net, config):
    """`Trajectory` の list を PPO 学習バッチ(dict of Tensor)へ変換する。

    返す dict のキー(形状は設計書 §4.7 の net 契約に対応):
      obs            (B, obs_dim)            float32。各決定点の観測
      actions        (B, A_max, action_dim)  float32。合法手ごとの行動特徴
      action_mask    (B, A_max)              bool。False は padding
      chosen_index   (B,)                    int64。選ばれた合法手の index
      advantages     (B,)                    float32。**actor 視点**(flip 済み)
      returns        (B,)                    float32。**actor 視点**(flip 済み)
      old_log_probs  (B,)                    float32。構築時点の選択 log prob

    - `Trajectory.discarded == True` の軌跡は混ぜない(設計書 §4.3)。
    - n>=3 requires learner_seat: GAE uses only that seat's atomic choices.
    - Legacy two-player unfiltered self-play keeps its fixed-view sign flips.
    - 終端の主報酬は注意 1 のとおり最後の決定の報酬へ加算する。
    - 中間決定は注意 3 のとおり報酬ゼロの通常ステップとして扱う。
    - `Trajectory.learner_seat` が設定されている軌跡は、その席の決定だけを
      バッチへ入れる(注意 4 / 設計書 14.1)。相手の手は環境遷移の一部として
      扱い、GAE は learner の決定列の上で計算する。learner の視点は単一に
      なるため flip は恒等になるが、仕組み自体は残す。`None`(既定)は全決定を
      入れる後方互換の経路。
    """
    obs_rows = []
    action_rows = []
    mask_rows = []
    chosen_rows = []
    reward_rows = []      # 固定視点(先頭決定者視点)の報酬列
    flip_rows = []
    done_rows = []
    common_action_dim = None
    common_obs_dim = None
    for traj in trajectories:
        if traj.discarded:
            continue
        decisions, players, action_dim = _check_trajectory(traj)
        if common_action_dim is None:
            common_action_dim = action_dim
        elif action_dim != common_action_dim:
            raise ValueError(
                "trajectories have different action dimensions "
                "(%d vs %d)" % (action_dim, common_action_dim))
        if common_obs_dim is None:
            common_obs_dim = len(decisions[0].obs_vec)
        elif len(decisions[0].obs_vec) != common_obs_dim:
            raise ValueError(
                "trajectories have different obs dimensions")
        if traj.learner_seat is not None:
            # 設計書 14.1: 学習対象席の決定だけをバッチへ入れる。相手の手は
            # 環境遷移の一部として扱い、GAE は learner の決定列の上で計算
            # する。整合性検査(_check_trajectory)は絞り込み前の全決定列に
            # 対して行っている。
            selected = [
                k for k in range(len(decisions))
                if decisions[k].player_id == traj.learner_seat]
            if not selected:
                raise ValueError(
                    "learner_seat %d has no decisions in %s/%d"
                    % (traj.learner_seat, traj.group, traj.index))
            decisions = [decisions[k] for k in selected]
            players = [players[k] for k in selected]
        first = players[0]
        flips = [1.0 if p == first else -1.0 for p in players]
        for k, d in enumerate(decisions):
            reward = d.reward
            if k == len(decisions) - 1:
                # 主報酬は終端のみ(注意 1)。最後の決定者の視点で加算する。
                reward += traj.terminal_rewards[d.player_id]
            obs_rows.append(d.obs_vec)
            action_rows.append(d.action_vecs)
            mask_rows.append([True] * len(d.action_vecs))
            chosen_rows.append(d.chosen_index)
            reward_rows.append(reward * flips[k])   # 固定視点へ flip
            flip_rows.append(flips[k])
            done_rows.append(k == len(decisions) - 1)
    if not obs_rows:
        raise ValueError(
            "no usable decisions (all trajectories discarded?)")
    obs_t = torch.as_tensor(obs_rows, dtype=torch.float32)
    chosen_t = torch.as_tensor(chosen_rows, dtype=torch.long)
    a_max = max(len(row) for row in action_rows)
    actions_t = torch.zeros(
        (len(action_rows), a_max, common_action_dim), dtype=torch.float32)
    mask_t = torch.zeros((len(action_rows), a_max), dtype=torch.bool)
    for i, row in enumerate(action_rows):
        actions_t[i, :len(row)] = torch.as_tensor(
            row, dtype=torch.float32)
        mask_t[i, :len(row)] = True

    logits, values = net(obs_t, actions_t, mask_t)
    flips_t = torch.as_tensor(flip_rows, dtype=torch.float32)
    rewards_t = torch.as_tensor(reward_rows, dtype=torch.float32)
    dones_t = torch.as_tensor(done_rows, dtype=torch.bool)
    # 価値は actor 視点なので、固定視点へ flip してから GAE する(注意 2)。
    advantages_fixed, returns_fixed = compute_gae(
        rewards_t, values * flips_t, dones_t, config)
    # 損失は actor 視点で計算するため flip を戻す(注意 2)。
    advantages = advantages_fixed * flips_t
    returns = returns_fixed * flips_t
    old_log_probs = torch.log_softmax(
        logits, dim=-1).gather(1, chosen_t.unsqueeze(1)).squeeze(1).detach()
    return {
        "obs": obs_t,
        "actions": actions_t,
        "action_mask": mask_t,
        "chosen_index": chosen_t,
        "advantages": advantages,
        "returns": returns,
        "old_log_probs": old_log_probs,
    }


def value_diagnostics(trajectories, net):
    """Train-only diagnostics, never used as policy/value targets."""
    bins = {"actor=%d,pending=%d" % (a, p):
            {"count": 0, "squared_error": 0.0, "value_sum": 0.0,
             "utility_sum": 0.0} for a in (0, 1) for p in (0, 1)}
    with torch.no_grad():
        for traj in trajectories:
            if traj.discarded:
                continue
            for offset in range(0, len(traj.value_probes), 256):
                probes = traj.value_probes[offset:offset + 256]
                values = net.infer_values([row[3] for row in probes])
                for row, value in zip(probes, values):
                    pid, actor, pending, _ = row
                    target = traj.terminal_rewards[pid]
                    item = bins["actor=%d,pending=%d" % (actor, pending)]
                    item["count"] += 1
                    item["squared_error"] += (value - target) ** 2
                    item["value_sum"] += value
                    item["utility_sum"] += target
    for item in bins.values():
        count = item["count"]
        item["mse"] = item["squared_error"] / count if count else None
        item["mean_value"] = item["value_sum"] / count if count else None
        item["mean_terminal_utility"] = item["utility_sum"] / count if count else None
    return bins


def _normalize_advantages(advantages):
    """ミニバッチ内で advantage を正規化する(標準 PPO。決定論的)。

    要素数1ミニバッチの NaN 対策(2026-08-12 の高重大度欠陥修正):
    `torch.std()` の既定は不偏推定(correction=1)で、要素数1では自由度
    n−1 = 0 のため NaN を返す(Opus 実測: `torch.tensor([0.5]).std()` は
    nan)。`+ 1e-8` を足しても `nan + 1e-8 = nan` なのでガードにならない。
    この NaN は policy 損失 → total loss → backward 経由で net の全
    パラメータへ伝播して学習が死ぬ。実学習スモーク(scripts/train_ppo.py
    を `--episodes 8 --generations 3 --workers 4 --keep-latest 2` で実行)
    では世代0/1は正常に完了するが、世代2で std の UserWarning(degrees of
    freedom is <= 0)の直後に `FloatingPointError: non-finite PPO loss:
    nan` として再現した。総サンプル数を minibatch_size で割った余りが1の
    とき末尾に要素数1のミニバッチができ、実学習では普通に起こる。

    修正方針は「要素数2未満のミニバッチでは正規化をスキップし、生の
    advantage をそのまま返す」。選んだ理由:
    - correction=0(unbiased=False)案は n=1 で std=0 になり
      adv = 0 / (0+1e-8) = 0 へ収まるが、(a) 唯一の advantage をゼロ化して
      policy 勾配を黙って消し、(b) n ≥ 2 の**全**ミニバッチの std が
      √((n−1)/n) 倍に変わり従来の学習ダイナミクスを変えてしまう。
      スキップ案は n ≥ 2 の経路に一切触れないため、従来挙動とビット単位
      で同一のまま退化ケースだけが有限になる。
    - 分割調整案(余り1を直前のミニバッチへ吸収)も n ≥ 2 の経路の分割構造
      を変えるため、変更範囲が広い。
    - 1要素では `(x − x.mean())` が恒等的に 0 であり、正規化の「中心化」は
      情報を消す。正規化はバッチ内の分散縮小のための便宜なので、分散が
      定義されない n=1 では「正規化しない」が一貫しており、生値を返せば
      そのミニバッチの勾配信号も保たれる。

    なぜ NaN が出なくなるか: 要素数1のときは有限な生の advantage をその
    まま返すため計算グラフに NaN が現れない。n ≥ 2 では不偏 std の自由度
    ≥ 1 なので std は有限(advantage が有限なら)で、従来どおり正規化しても
    NaN は出ない。

    決定論: スキップ判定は要素数だけで決まり乱数に依存しないため、同一
    seed・同一バッチなら従来どおり完全に同一の重み更新になる(要素数1の
    ミニバッチを含む構成でも決定的)。

    (歯: このガードを外して常に
    `(advantages - advantages.mean()) / (advantages.std() + 1e-8)` を
    使う実装へ戻すと、tests/test_ppo.py の
    test_normalize_advantages_single_element_is_finite と
    test_update_with_single_element_minibatch_keeps_finite が NaN で落ちる。)
    """
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def update(net, optimizer, batch, config, seed):
    """PPO クリップ更新を `config.epochs` 回実行し、統計 dict を返す。

    ミニバッチ分割は `config.minibatch_size` ごと。シャッフルは
    `derive_seed(seed, "ppo", epoch)` 由来の乱数で行う(設計書 §6.3 の "ppo"
    ストリーム)。同一 seed・同一バッチ・同一初期重みなら**完全に同一の重み
    更新**になる。epoch は `derive_seed` の episode_index 引数として使う。

    バッチは `trajectories_to_batch` の出力を想定する。`old_log_probs` が
    無い場合は更新開始時に現在の net から一度だけ計算する(手組みバッチ用の
    後方互換。学習パイプラインは必ず `trajectories_to_batch` を使うこと)。

    返り値の統計(全ミニバッチ・全 epoch の平均、float):
      policy_loss / value_loss / entropy / total_loss / clip_frac / approx_kl
    """
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32)
    mask = torch.as_tensor(batch["action_mask"], dtype=torch.bool)
    chosen = torch.as_tensor(batch["chosen_index"], dtype=torch.long)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
    n = obs.shape[0]
    if n == 0:
        raise ValueError("update requires a non-empty batch")
    if actions.shape[0] != n or mask.shape[0] != n \
            or chosen.shape[0] != n:
        raise ValueError("batch tensors must share the batch size")
    if advantages.shape[0] != n or returns.shape[0] != n:
        raise ValueError("advantages / returns must match the batch size")
    if mask.shape[1] != actions.shape[1]:
        raise ValueError("action_mask and actions must share A_max")
    if not mask.gather(1, chosen.unsqueeze(1)).all():
        raise ValueError(
            "chosen_index must point to a legal (mask=True) action")
    if "old_log_probs" in batch:
        old_log_probs = torch.as_tensor(
            batch["old_log_probs"], dtype=torch.float32)
    else:
        with torch.no_grad():
            init_logits, _ = net(obs, actions, mask)
            old_log_probs = torch.log_softmax(
                init_logits, dim=-1).gather(
                    1, chosen.unsqueeze(1)).squeeze(1)

    indices = list(range(n))
    sums = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
        "clip_frac": 0.0,
        "approx_kl": 0.0,
    }
    count = 0
    for epoch in range(config.epochs):
        rng = random.Random(derive_seed(seed, "ppo", epoch))
        rng.shuffle(indices)
        for start in range(0, n, config.minibatch_size):
            mb = indices[start:start + config.minibatch_size]
            mb_obs = obs[mb]
            mb_actions = actions[mb]
            mb_mask = mask[mb]
            mb_chosen = chosen[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            mb_old = old_log_probs[mb]
            # ミニバッチ内で advantage を正規化する(標準 PPO。決定論的)。
            # 要素数2未満のミニバッチは正規化しない(不偏 std が自由度0で
            # NaN になるため。理由は `_normalize_advantages` の docstring)。
            adv = _normalize_advantages(mb_adv)

            logits, values = net(mb_obs, mb_actions, mb_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(
                1, mb_chosen.unsqueeze(1)).squeeze(1)
            ratio = (chosen_log_probs - mb_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - mb_ret) ** 2).mean()
            entropy = policy_entropy(logits, mb_mask)
            loss = (policy_loss + config.value_coef * value_loss
                    - config.entropy_coef * entropy)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite PPO loss: %s" % (loss.item(),))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                clip_frac = (
                    torch.abs(ratio - 1.0) > config.clip_eps
                ).float().mean().item()
                approx_kl = (
                    ratio - 1.0 - (chosen_log_probs - mb_old)
                ).mean().item()
            sums["policy_loss"] += policy_loss.item()
            sums["value_loss"] += value_loss.item()
            sums["entropy"] += entropy.item()
            sums["total_loss"] += loss.item()
            sums["clip_frac"] += clip_frac
            sums["approx_kl"] += approx_kl
            count += 1
    return dict((k, v / count) for k, v in sums.items())


# ===========================================================================
# Expert Iteration の蒸留学習(Phase 7 段2/4)。torch を触るので本モジュールに置く。
# 公開 I/F は `naeco/train/exit.py`(torch を import しない facade)から再輸出する。
# ===========================================================================
import gzip     # noqa: E402
import json     # noqa: E402
import math     # noqa: E402
import time     # noqa: E402

from naeco.train import checkpoint as checkpoint_mod   # noqa: E402
from naeco.train import net as net_mod                 # noqa: E402

TARGET_VERSION = "exit-1"


class ExitDataset(object):
    """gzip JSONL を読み、方策行と価値行に分けて保持する。"""

    def __init__(self, paths, obs_dim, action_dim, policy_target="improved",
                 holdout_fraction=0.05):
        if policy_target not in ("improved", "visits"):
            raise ValueError("policy_target must be 'improved' or 'visits'")
        self.policy_target = policy_target
        self.headers = []
        self.rows = []
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                first = json.loads(fh.readline())
                header = first.get("header")
                if header is None:
                    raise ValueError("%s: first line must be a header" % path)
                if header["obs_dim"] != obs_dim or header["action_dim"] != action_dim:
                    raise ValueError(
                        "%s: encoder dims %d/%d do not match net %d/%d"
                        % (path, header["obs_dim"], header["action_dim"],
                           obs_dim, action_dim))
                self.headers.append(header)
                for line in fh:
                    if line.strip():
                        self.rows.append(json.loads(line))
        if not self.rows:
            raise ValueError("no rows loaded")
        games = sorted(set(r["index"] for r in self.rows))
        n_hold = int(round(len(games) * holdout_fraction))
        holdout = set(games[len(games) - n_hold:]) if n_hold else set()
        self.train_rows = [r for r in self.rows if r["index"] not in holdout]
        self.holdout_rows = [r for r in self.rows if r["index"] in holdout]

    def policy_row(self, row):
        return "policy" in row and self.policy_target == "improved" or \
            ("visits" in row and self.policy_target == "visits" and
             row.get("full"))

    @staticmethod
    def stats(rows):
        return {
            "rows": len(rows),
            "policy_rows": sum(1 for r in rows if "policy" in r),
            "search_rows": sum(1 for r in rows if r["kind"] == "search"),
            "games": len(set(r["index"] for r in rows)),
        }


def _target_distribution(row, policy_target):
    if policy_target == "improved":
        return row["policy"]
    visits = row["visits"]
    total = float(sum(visits))
    if total <= 0:
        return None
    return [v / total for v in visits]


def _value_target(row, mode):
    z = float(row["z"])
    if mode == "z":
        return z
    p = row.get("root_value")
    if p is None:
        return z
    return 0.5 * (z + (2.0 * float(p) - 1.0))


def build_batch(rows, action_dim, policy_target, value_mode, surprise_weighting):
    """行の list をテンソルの dict へ。方策ターゲットの無い行は policy_mask=False。"""
    obs = []
    actions = []
    targets = []
    values = []
    has_policy = []
    weights = []
    kls = []
    a_max = 1
    for row in rows:
        acts = row.get("actions") or [[0.0] * action_dim]
        a_max = max(a_max, len(acts))
    for row in rows:
        obs.append(row["obs"])
        acts = row.get("actions")
        dist = None
        if acts is not None and row["kind"] == "search":
            if policy_target == "improved" and "policy" in row:
                dist = row["policy"]
            elif policy_target == "visits" and row.get("full"):
                dist = _target_distribution(row, "visits")
        if acts is None:
            acts = [[0.0] * action_dim]
        padded = [list(a) for a in acts] + \
            [[0.0] * action_dim] * (a_max - len(acts))
        actions.append(padded)
        if dist is not None:
            tgt = list(dist) + [0.0] * (a_max - len(dist))
            has_policy.append(True)
            prior = row.get("prior")
            if prior:
                kl = sum(p * math.log(max(p, 1e-12) / max(q, 1e-12))
                         for p, q in zip(dist, prior))
            else:
                kl = 0.0
        else:
            tgt = [0.0] * a_max
            has_policy.append(False)
            kl = 0.0
        targets.append(tgt)
        kls.append(kl)
        values.append(_value_target(row, value_mode))
    mask = torch.zeros((len(rows), a_max), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row.get("actions") or [[0.0]])
        mask[i, :n] = True
    if surprise_weighting and any(has_policy):
        mean_kl = sum(k for k, h in zip(kls, has_policy) if h) / max(
            1, sum(1 for h in has_policy if h))
        for k, h in zip(kls, has_policy):
            weights.append(0.5 + 0.5 * (k / mean_kl if mean_kl > 0 else 1.0)
                           if h else 1.0)
    else:
        weights = [1.0] * len(rows)
    return {
        "obs": torch.tensor(obs, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.float32),
        "action_mask": mask,
        "policy_target": torch.tensor(targets, dtype=torch.float32),
        "policy_mask": torch.tensor(has_policy, dtype=torch.bool),
        "value_target": torch.tensor(values, dtype=torch.float32),
        "weight": torch.tensor(weights, dtype=torch.float32),
    }


def losses(net, batch, value_coef):
    logits, values = net(batch["obs"], batch["actions"], batch["action_mask"])
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = torch.where(batch["action_mask"], log_probs,
                            torch.zeros_like(log_probs))
    ce = -(batch["policy_target"] * log_probs).sum(dim=-1)
    pm = batch["policy_mask"].float() * batch["weight"]
    n_policy = pm.sum()
    policy_loss = (ce * pm).sum() / n_policy if n_policy > 0 else ce.sum() * 0.0
    value_loss = ((values - batch["value_target"]) ** 2).mean()
    total = policy_loss + value_coef * value_loss
    return total, policy_loss, value_loss


def load_net(meta_path, num_players=2):
    """メタと重みから PolicyValueNet を作る(eval_agent と同じ検証経路)。"""
    with open(meta_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    arch = raw["model_arch"]
    checkpoint_mod.load(
        meta_path, expected=checkpoint_mod.expected_meta(num_players, arch))
    net = net_mod.PolicyValueNet(
        arch["obs_dim"], arch["action_dim"], hidden=tuple(arch["hidden"]),
        action_hidden=tuple(arch["action_hidden"]))
    weights_path = meta_path[:-len(".meta.json")] + ".weights.pt"
    net.load_state_dict(net_mod.load_state_dict_from_path(weights_path))
    return net


def train(dataset, net, out_dir, name, seed, epochs=4, minibatch_size=256,
          lr=1e-4, value_coef=0.5, max_grad_norm=0.5, value_mode="z",
          surprise_weighting=False, generation=0, extra_hyper=None, log=print):
    """蒸留学習を回してチェックポイントを書く。戻り値は統計 dict。"""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    rows = dataset.train_rows
    n = len(rows)
    stats_history = []
    started = time.monotonic()
    for epoch in range(epochs):
        rng = random.Random(derive_seed(seed, "exit", epoch))
        order = list(range(n))
        rng.shuffle(order)
        sums = {"total": 0.0, "policy": 0.0, "value": 0.0}
        count = 0
        net.train()
        for start in range(0, n, minibatch_size):
            mb = [rows[i] for i in order[start:start + minibatch_size]]
            batch = build_batch(mb, net.arch_dict()["action_dim"],
                                dataset.policy_target, value_mode,
                                surprise_weighting)
            total, policy_loss, value_loss = losses(net, batch, value_coef)
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite loss %s" % total.item())
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            sums["total"] += total.item()
            sums["policy"] += policy_loss.item()
            sums["value"] += value_loss.item()
            count += 1
        epoch_stats = dict((k, v / max(count, 1)) for k, v in sums.items())
        epoch_stats["epoch"] = epoch
        epoch_stats.update(evaluate(dataset.holdout_rows, net, dataset.policy_target,
                                    value_mode, value_coef, minibatch_size))
        stats_history.append(epoch_stats)
        log("epoch %d train total=%.4f policy=%.4f value=%.4f | holdout policy=%.4f "
            "value=%.4f acc=%.3f (%.0fs)" % (
                epoch, epoch_stats["total"], epoch_stats["policy"],
                epoch_stats["value"], epoch_stats.get("holdout_policy", float("nan")),
                epoch_stats.get("holdout_value", float("nan")),
                epoch_stats.get("holdout_top1", float("nan")),
                time.monotonic() - started))
    net.eval()
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "%s.meta.json" % name)
    weights_path = os.path.join(out_dir, "%s.weights.pt" % name)
    save_state_dict(net.state_dict(), weights_path)
    hyper = {
        "algorithm": "expert_iteration", "target_version": TARGET_VERSION,
        "policy_target": dataset.policy_target, "value_mode": value_mode,
        "epochs": epochs, "minibatch_size": minibatch_size, "lr": lr,
        "value_coef": value_coef, "max_grad_norm": max_grad_norm,
        "surprise_weighting": surprise_weighting,
        "train_rows": n, "holdout_rows": len(dataset.holdout_rows),
        "data_headers": dataset.headers,
        "hidden": net.arch_dict()["hidden"],
        "action_hidden": net.arch_dict()["action_hidden"],
        "history": stats_history,
    }
    if extra_hyper:
        hyper.update(extra_hyper)
    meta = checkpoint_mod.current_meta(2, net.arch_dict(), extra={
        "torch_version": torch.__version__,
        "root_seed": seed,
        "hyperparameters": hyper,
        "training_steps": sum(1 for _ in range(epochs)) * max(1, n // minibatch_size),
        "generation": generation,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    checkpoint_mod.save(meta_path, meta)
    return {"meta_path": meta_path, "history": stats_history}


def evaluate(rows, net, policy_target, value_mode, value_coef, minibatch_size=512):
    """hold-out 行の損失と、方策行での top-1 一致率(教師の argmax とネットの argmax)。"""
    if not rows:
        return {}
    net.eval()
    sums = {"policy": 0.0, "value": 0.0}
    n_policy = 0
    n_rows = 0
    hits = 0
    with torch.no_grad():
        for start in range(0, len(rows), minibatch_size):
            mb = rows[start:start + minibatch_size]
            batch = build_batch(mb, net.arch_dict()["action_dim"], policy_target,
                                value_mode, False)
            logits, values = net(batch["obs"], batch["actions"], batch["action_mask"])
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = torch.where(batch["action_mask"], log_probs,
                                    torch.zeros_like(log_probs))
            ce = -(batch["policy_target"] * log_probs).sum(dim=-1)
            pm = batch["policy_mask"]
            sums["policy"] += ce[pm].sum().item()
            n_policy += int(pm.sum().item())
            sums["value"] += ((values - batch["value_target"]) ** 2).sum().item()
            n_rows += len(mb)
            if pm.any():
                pred = logits[pm].argmax(dim=-1)
                tgt = batch["policy_target"][pm].argmax(dim=-1)
                hits += int((pred == tgt).sum().item())
    return {
        "holdout_policy": sums["policy"] / max(n_policy, 1),
        "holdout_value": sums["value"] / max(n_rows, 1),
        "holdout_top1": hits / max(n_policy, 1),
        "holdout_policy_rows": n_policy,
    }
