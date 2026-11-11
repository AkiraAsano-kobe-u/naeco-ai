"""B4 PPO 学習ループのテスト(Phase 3 設計書 §4.8)。

歯の狙い: 価値の符号規約(手番視点のゼロ和符号反転)・中間決定の GAE 扱い・
dones 境界・entropy の padding 除外・seed 決定論・discarded 除外・勾配有限性・
合成タスクでの収束(value 損失・エントロピー減少と正解行動への収束)は、
それぞれ実装から外すと落ちる形で固定する。

設計書 14.4(Phase 3A ゲート1回目の指摘)の追加: 実局面由来の可変長合法手
バッチを多数生成し、update → backward → optimizer.step の全体で損失と全
パラメータ勾配が有限であることをファズ的に検査する(RealGameBatchFuzzTests)。

- 符号反転を player_id でなく隣接 turn_advanced だけで判定すると落ちる。
- 終端報酬を最後の決定者の席でなく席0固定で加算すると落ちる。
- `dones` 境界を無視(ブートストラップを跨がせる)すると落ちる。
- entropy の mask 除外を外すと padding 不変性テストが落ちる。
- entropy の backward が padding 込みで NaN 勾配になる旧実装
  (`torch.where(mask, probs * log_probs, 0)`)へ戻すと、勾配有限性の
  新テスト 3 件が落ちる。
- advantage 正規化は要素数2未満のミニバッチでスキップする。不偏 std
  (correction=1)は要素数1で自由度0のため NaN になり loss が死ぬ
  (scripts/train_ppo.py の実学習スモークで世代2に再現)。ガードを外して
  元の `(adv - adv.mean()) / (adv.std() + 1e-8)` へ戻すと
  test_normalize_advantages_single_element_is_finite /
  test_update_with_single_element_minibatch_keeps_finite が落ちる。
- シャッフルを derive_seed("ppo", …) 以外にすると決定論テストが落ちる。
- ゼロ和の符号反転テスト(test_zero_sum_flip_negates_advantages)は
  value ヘッドをゼロ化(V ≡ 0)して GAE を割引報酬へ帰着させて検証する。
  一般の net では value 項が報酬と一緒に反転しないため厳密反転は成り立たず
  (残差 −2V 由来)、V ≡ 0 のときだけ厳密に成立する。value 側の flip は
  非ゼロ value の手計算テストが固定する。

torch が無い環境では skip し、import 失敗は隠さない。
"""
import dataclasses
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

HAS_TORCH = torch is not None

from naeco.ai.heuristic_agent import HeuristicAgent  # noqa: E402
from naeco.ai.random_agent import RandomAgent  # noqa: E402
from naeco.train import encoding as enc  # noqa: E402
from naeco.train.agent import PolicyAgent  # noqa: E402
from naeco.train.env import (  # noqa: E402
    Decision, Trajectory, play_episode,
)

if HAS_TORCH:
    from naeco.train import ppo  # noqa: E402
    from naeco.train.net import PolicyValueNet  # noqa: E402
else:
    ppo = None
    PolicyValueNet = None

THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _hand_gae(rewards, values, dones, gamma, gae_lambda):
    """テスト用の独立実装。compute_gae と同じ規約で素朴にループする。

    dones[i] == True は「決定 i で軌跡が終わる」= ブートストラップしない。
    """
    t = len(rewards)
    adv = [0.0] * t
    ret = [0.0] * t
    last = 0.0
    for i in range(t - 1, -1, -1):
        next_value = 0.0 if dones[i] else values[i + 1]
        delta = rewards[i] + gamma * next_value - values[i]
        last = delta + gamma * gae_lambda * (0.0 if dones[i] else last)
        adv[i] = last
        ret[i] = adv[i] + values[i]
    return adv, ret


def _trajectory(players, terminal, rewards=None, turn_advanced=None,
                chosen=None, num_actions=3, index=0, obs_dim=4,
                action_dim=2, discarded=False, discard_reason="",
                obs_vec=None):
    """手組みの Trajectory。obs は共通ベクトル、行動は num_actions 個。"""
    if obs_vec is None:
        obs_vec = tuple(float(i) / 10.0 for i in range(obs_dim))
    action_vecs = tuple(
        tuple(float((i + j) % 7) / 7.0 for j in range(action_dim))
        for i in range(num_actions))
    if rewards is None:
        rewards = [0.0] * len(players)
    if turn_advanced is None:
        turn_advanced = [True] * len(players)
    if chosen is None:
        chosen = [0] * len(players)
    decisions = tuple(
        Decision(
            player_id=players[k],
            obs_vec=obs_vec,
            action_vecs=action_vecs,
            chosen_index=chosen[k],
            reward=rewards[k],
            turn_advanced=turn_advanced[k],
        )
        for k in range(len(players))
    )
    return Trajectory(
        group="train", index=index, seat_variant=0, seats=("p0", "p1"),
        decisions=decisions, terminal_rewards=terminal,
        initial_state_hash="h%d" % index, discarded=discarded,
        discard_reason=discard_reason,
    )


def _synthetic_trajectories():
    """合成タスクの軌跡。obs_dim=4 / action_dim=2 / 合法手3。

    label 0 の軌跡は全決定で chosen=0 かつ終端 +1、label 1 は全決定で
    chosen=1 かつ終端 -1。正解は「行動 0 を選ぶ」こと。obs は label ごとに
    異なるベクトルを使い、価値ヘッドが returns を学習できるようにする。
    """
    trajs = []
    for k in range(24):
        label = k % 2
        chosen = 0 if label == 0 else 1
        terminal = (1.0, -1.0) if label == 0 else (-1.0, 1.0)
        obs_vec = (0.1, 0.2, 0.3, 0.4) if label == 0 \
            else (0.9, 0.8, 0.7, 0.6)
        trajs.append(_trajectory(
            players=[0, 1, 0], terminal=terminal,
            chosen=[chosen, chosen, chosen], index=k, obs_vec=obs_vec))
    return trajs


def _synthetic_batch(seed, config=None):
    """合成タスクのバッチ(軌跡は _synthetic_trajectories と同一)。"""
    torch.manual_seed(seed)
    if config is None:
        config = ppo.PPOConfig(epochs=2, minibatch_size=16)
    net = PolicyValueNet(4, 2, hidden=(16, 16), action_hidden=(8,))
    batch = ppo.trajectories_to_batch(_synthetic_trajectories(), net, config)
    return net, batch, config


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class PPOConfigTests(unittest.TestCase):
    """§4.8 の固定値が契約どおりであること。"""

    def test_defaults_match_contract(self):
        cfg = ppo.PPOConfig()
        self.assertEqual(cfg.gamma, 0.997)
        self.assertEqual(cfg.gae_lambda, 0.95)
        self.assertEqual(cfg.clip_eps, 0.2)
        self.assertEqual(cfg.value_coef, 0.5)
        self.assertEqual(cfg.entropy_coef, 0.01)
        self.assertEqual(cfg.lr, 3e-4)
        self.assertEqual(cfg.epochs, 4)
        self.assertEqual(cfg.minibatch_size, 256)
        self.assertEqual(cfg.max_grad_norm, 0.5)

    def test_frozen(self):
        cfg = ppo.PPOConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.gamma = 0.5


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class ComputeGaeTests(unittest.TestCase):
    """GAE が手計算値と一致し、dones 境界を越えないこと。"""

    def test_hand_computed_sequence(self):
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        rewards = [0.0, 0.0, 1.0]
        values = [0.5, 0.8, 0.7]
        dones = [False, False, True]
        advantages, returns = ppo.compute_gae(rewards, values, dones, cfg)
        # 手計算:
        #   i=2: done → δ = 1.0 - 0.7 = 0.3, A = 0.3, R = 1.0
        #   i=1: δ = 0.9*0.7 - 0.8 = -0.17,
        #         A = -0.17 + 0.855*0.3 = 0.0865, R = 0.8865
        #   i=0: δ = 0.9*0.8 - 0.5 = 0.22,
        #         A = 0.22 + 0.855*0.0865 = 0.2939575, R = 0.7939575
        expected_adv = [0.2939575, 0.0865, 0.3]
        expected_ret = [0.7939575, 0.8865, 1.0]
        for got, exp in zip(advantages.tolist(), expected_adv):
            self.assertAlmostEqual(got, exp, places=5)
        for got, exp in zip(returns.tolist(), expected_ret):
            self.assertAlmostEqual(got, exp, places=5)

    def test_matches_independent_implementation(self):
        cfg = ppo.PPOConfig(gamma=0.997, gae_lambda=0.95)
        torch.manual_seed(0)
        t = 9
        rewards = torch.rand(t).tolist()
        values = torch.rand(t).tolist()
        dones = [i in (2, 6, 8) for i in range(t)]
        advantages, returns = ppo.compute_gae(rewards, values, dones, cfg)
        exp_adv, exp_ret = _hand_gae(
            rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        self.assertTrue(torch.allclose(
            advantages, torch.tensor(exp_adv), atol=1e-6))
        self.assertTrue(torch.allclose(
            returns, torch.tensor(exp_ret), atol=1e-6))

    def test_single_terminal_decision(self):
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        advantages, returns = ppo.compute_gae(
            [2.0], [0.5], [True], cfg)
        self.assertAlmostEqual(advantages[0].item(), 1.5, places=6)
        self.assertAlmostEqual(returns[0].item(), 2.0, places=6)

    def test_dones_boundary_prevents_trajectory_leakage(self):
        """dones の境界を越えてアドバンテージが伝播しないこと(歯)。"""
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        values = [0.5, 0.8, 0.5, 0.8]
        dones = [False, True, False, True]
        # 軌跡A(rewards 前半)は同一のまま、軌跡B(後半)だけ変える。
        adv_a1, _ = ppo.compute_gae([0.0, 1.0, 0.0, -1.0], values, dones, cfg)
        adv_a2, _ = ppo.compute_gae([0.0, 1.0, 0.0, -5.0], values, dones, cfg)
        self.assertTrue(torch.allclose(adv_a1[:2], adv_a2[:2], atol=1e-7))
        self.assertFalse(torch.allclose(adv_a1, adv_a2))
        # 手計算でも固定する: A1 = [0.22 + 0.855*0.2, 0.2, ...]
        exp_adv, exp_ret = _hand_gae(
            [0.0, 1.0, 0.0, -1.0], values, dones, 0.9, 0.95)
        self.assertTrue(torch.allclose(
            adv_a1, torch.tensor(exp_adv), atol=1e-6))

    def test_empty_input(self):
        cfg = ppo.PPOConfig()
        advantages, returns = ppo.compute_gae([], [], [], cfg)
        self.assertEqual(tuple(advantages.shape), (0,))
        self.assertEqual(tuple(returns.shape), (0,))

    def test_invalid_shapes_raise(self):
        cfg = ppo.PPOConfig()
        with self.assertRaises(ValueError):
            ppo.compute_gae([0.0], [0.0, 1.0], [True], cfg)
        with self.assertRaises(ValueError):
            ppo.compute_gae([[0.0]], [0.0], [True], cfg)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class SignConventionTests(unittest.TestCase):
    """価値の符号規約(手番視点・ゼロ和の符号反転)を固定する。"""

    def _net(self):
        torch.manual_seed(0)
        return PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))

    def test_alternating_players_match_hand_computation(self):
        net = self._net()
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        # 席0 → 席1 → 席0。席0 が勝ち。全決定の shaped 報酬は 0。
        traj = _trajectory(players=[0, 1, 0], terminal=(1.0, -1.0))
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        logits, values = net(batch["obs"], batch["actions"],
                             batch["action_mask"])
        # 先頭決定者(席0)視点へ flip した列で GAE し、actor 視点へ戻す。
        flips = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
        rewards_fixed = torch.tensor([0.0, 0.0, 1.0])  # 終端は席0 の +1
        values_fixed = values * flips
        exp_adv, exp_ret = _hand_gae(
            rewards_fixed.tolist(), values_fixed.tolist(),
            [False, False, True], 0.9, 0.95)
        exp_adv = torch.tensor(exp_adv) * flips
        exp_ret = torch.tensor(exp_ret) * flips
        self.assertTrue(torch.allclose(
            batch["advantages"], exp_adv, atol=1e-6))
        self.assertTrue(torch.allclose(
            batch["returns"], exp_ret, atol=1e-6))

    def test_last_return_is_last_actors_terminal_reward(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        # 最後の決定者は席1。席1 は負けなので actor 視点の returns[-1] は -1。
        traj = _trajectory(players=[0, 0, 1], terminal=(1.0, -1.0),
                           turn_advanced=[False, True, True])
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        self.assertAlmostEqual(batch["returns"][-1].item(), -1.0, places=6)

    def test_zero_sum_flip_negates_advantages(self):
        """勝敗の入れ替えでアドバンテージが厳密に符号反転する(V ≡ 0 のとき)。

        なぜ value ヘッドをゼロ化するか: 一般の net ではこの主張は**成り立た
        ない**。GAE の δ_t = r_t + γV_{t+1} − V_t の value 項は報酬と一緒に
        反転しないため、同一 obs(win / lose で V は同一)のとき
        A_win + A_lose = Σ_k (γλ)^k · 2 · (γV_{t+k+1} − V_{t+k}) ≠ 0 になる
        (実測残差 ≈ −2V 由来)。そこで value ヘッドの重み・バイアスをゼロ化し
        V ≡ 0 にすると GAE は割引報酬そのものに帰着し、報酬の反転に対して
        厳密に符号反転する。この形でも、報酬の固定視点変換・終端の actor
        帰属・アドバンテージの flip 戻しのいずれかを壊すと、下の厳密値の
        主張が落ちる。players [0,1,1] は flip が [1,−1,−1] なので、報酬
        flip の除去も actor 視点の報酬列 [0,0,−1] との差で検出できる。
        value 側の flip は非ゼロ value の手計算テスト
        (test_alternating_players_match_hand_computation 等)が固定する。
        """
        net = self._net()
        # V ≡ 0: value ヘッドの重み・バイアスをゼロ化する(実装は変更しない)。
        net.value_head.weight.data.zero_()
        net.value_head.bias.data.zero_()
        cfg = ppo.PPOConfig()
        # 最後の決定者は席1。win(1,-1) の終端は席1 視点で -1。
        win = _trajectory(players=[0, 1, 1], terminal=(1.0, -1.0), index=0)
        lose = _trajectory(players=[0, 1, 1], terminal=(-1.0, 1.0), index=1)
        b_win = ppo.trajectories_to_batch([win], net, cfg)
        b_lose = ppo.trajectories_to_batch([lose], net, cfg)
        # V ≡ 0 では returns = advantages = 割引報酬(actor 視点)。
        # win: 固定視点の報酬列 [0, 0, +1] → GAE [γ²λ², γλ, +1]
        #   → flip 戻し [1,−1,−1] で [γ²λ², −γλ, −1]。
        gl = cfg.gamma * cfg.gae_lambda
        expected = torch.tensor([gl * gl, -gl, -1.0])
        self.assertTrue(torch.allclose(
            b_win["advantages"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(
            b_win["returns"], expected, atol=1e-6))
        # 勝敗入れ替え(報酬の符号反転)で厳密に符号反転する。
        self.assertTrue(torch.allclose(
            b_win["advantages"], -b_lose["advantages"], atol=1e-6))
        self.assertTrue(torch.allclose(
            b_win["returns"], -b_lose["returns"], atol=1e-6))

    def test_same_player_across_turn_advanced_is_allowed(self):
        """ラウンド境界等で turn_advanced=True でも同一プレイヤーが続く場合は
        反転せず、player_id の実値だけで flip が決まることを固定する。"""
        net = self._net()
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        traj = _trajectory(players=[0, 0, 1], terminal=(1.0, -1.0))
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        logits, values = net(batch["obs"], batch["actions"],
                             batch["action_mask"])
        flips = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)
        # 最後の決定者は席1。席1 の actor 視点の終端報酬は -1、固定視点では
        # (-1) * (-1) = +1 になる。
        rewards_fixed = torch.tensor([0.0, 0.0, 1.0])
        exp_adv, exp_ret = _hand_gae(
            rewards_fixed.tolist(), (values * flips).tolist(),
            [False, False, True], 0.9, 0.95)
        exp_adv = torch.tensor(exp_adv) * flips
        exp_ret = torch.tensor(exp_ret) * flips
        self.assertTrue(torch.allclose(
            batch["advantages"], exp_adv, atol=1e-6))
        self.assertTrue(torch.allclose(
            batch["returns"], exp_ret, atol=1e-6))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class IntermediateDecisionTests(unittest.TestCase):
    """中間決定(turn_advanced=False)の GAE 扱いを固定する。"""

    def _net(self):
        torch.manual_seed(0)
        return PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))

    def test_intermediate_decision_is_zero_reward_normal_step(self):
        net = self._net()
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        # players [0, 0, 1]: 決定0→1 が中間決定(turn_advanced=False)。
        traj = _trajectory(players=[0, 0, 1], terminal=(1.0, -1.0),
                           turn_advanced=[False, True, True])
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        logits, values = net(batch["obs"], batch["actions"],
                             batch["action_mask"])
        flips = torch.tensor([1.0, 1.0, -1.0], dtype=torch.float32)
        # 中間決定は報酬 0 の通常ステップ。終端は最後の決定者(席1)の -1。
        rewards_fixed = torch.tensor([0.0, 0.0, 1.0])
        exp_adv, exp_ret = _hand_gae(
            rewards_fixed.tolist(), (values * flips).tolist(),
            [False, False, True], 0.9, 0.95)
        exp_adv = torch.tensor(exp_adv) * flips
        exp_ret = torch.tensor(exp_ret) * flips
        self.assertTrue(torch.allclose(
            batch["advantages"], exp_adv, atol=1e-6))
        self.assertTrue(torch.allclose(
            batch["returns"], exp_ret, atol=1e-6))

    def test_inconsistent_turn_advanced_raises(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        # turn_advanced=False なのに次の決定が別プレイヤー → 整合性エラー。
        traj = _trajectory(players=[0, 1, 0], terminal=(1.0, -1.0),
                           turn_advanced=[False, True, True])
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([traj], net, cfg)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class EntropyTests(unittest.TestCase):
    """entropy が可変長の合法手集合上で計算されること(歯)。"""

    def test_entropy_is_invariant_to_padding(self):
        torch.manual_seed(0)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        obs = torch.rand(3, 4)
        actions = torch.rand(3, 2, 2)
        mask = torch.ones(3, 2, dtype=torch.bool)
        logits, _ = net(obs, actions, mask)
        e1 = ppo.policy_entropy(logits, mask).item()
        # 同じ内容を A_max=5 へ padding しても entropy は一致する。
        actions5 = torch.zeros(3, 5, 2)
        actions5[:, :2] = actions
        mask5 = torch.zeros(3, 5, dtype=torch.bool)
        mask5[:, :2] = True
        logits5, _ = net(obs, actions5, mask5)
        e2 = ppo.policy_entropy(logits5, mask5).item()
        self.assertAlmostEqual(e1, e2, places=5)

    def test_entropy_matches_legal_only_softmax(self):
        torch.manual_seed(1)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        obs = torch.rand(4, 4)
        actions = torch.rand(4, 4, 2)
        mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ])
        logits, _ = net(obs, actions, mask)
        got = ppo.policy_entropy(logits, mask).item()
        expected = 0.0
        for row, row_mask in zip(logits, mask):
            legal = row[row_mask]
            probs = torch.softmax(legal, dim=-1)
            expected += -(probs * torch.log(probs)).sum().item()
        expected /= len(logits)
        self.assertAlmostEqual(got, expected, places=5)

    def test_single_legal_action_has_zero_entropy(self):
        torch.manual_seed(2)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        obs = torch.rand(2, 4)
        actions = torch.rand(2, 3, 2)
        mask = torch.tensor([[True, False, False], [False, True, False]])
        logits, _ = net(obs, actions, mask)
        self.assertAlmostEqual(
            ppo.policy_entropy(logits, mask).item(), 0.0, places=5)

    def test_uniform_entropy_is_log_of_legal_count(self):
        torch.manual_seed(3)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        obs = torch.rand(1, 4)
        actions = torch.zeros(1, 3, 2)
        mask = torch.ones(1, 3, dtype=torch.bool)
        logits, _ = net(obs, actions, mask)
        # ロジットを全て 0 にした行は合法手集合上の一様分布。
        flat = logits.detach().clone()
        flat[:] = 0.0
        self.assertAlmostEqual(
            ppo.policy_entropy(flat, mask).item(),
            1.0986122886681098, places=4)   # log(3)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class EntropyBackwardTests(unittest.TestCase):
    """entropy の backward が padding を含んでも有限であること(歯)。

    独立監査で指摘された欠陥: 旧実装の `torch.where(mask, probs * log_probs,
    0)` は forward では正しい値を返すが、backward では padding 位置の
    `0 * -inf` の積を微分して NaN 勾配を生み、log_softmax の行内勾配和経由で
    net の全パラメータが NaN になった(Opus が torch 2.8.0 で実測)。以下の
    テストは、実装を元の torch.where 版へ戻すと落ちる。
    """

    def test_tensor_level_backward_is_finite_with_inf_logits(self):
        """-inf logit を含む行でも backward の勾配が有限であること(最小歯)。

        行1(全合法・ロジット 0)は一様分布でエントロピー log(3)、行2(合法手1つ)
        はエントロピー 0 なので、forward 値は log(3)/2 に一致しなければなら
        ない。勾配は全て有限でなければならない。旧実装では padding の
        `0 * -inf` 由来の NaN が行2の logit 勾配へ入り、このテストは落ちる。
        """
        logits = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, float("-inf"), float("-inf")]],
            dtype=torch.float32, requires_grad=True)
        mask = torch.tensor(
            [[True, True, True], [True, False, False]])
        entropy = ppo.policy_entropy(logits, mask)
        self.assertAlmostEqual(
            entropy.item(), math.log(3.0) / 2.0, places=5)
        entropy.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_backward_with_padding_produces_finite_grads(self):
        """padding を含むバッチで backward しても全パラメータ勾配が有限(歯)。

        監査の再現手順どおり: PolicyValueNet(4, 2, hidden=(8, 8),
        action_hidden=(4,)) を torch.manual_seed(0) で作り、obs=randn(2,4) /
        acts=randn(2,3,2) / mask=[[T,T,T],[T,F,F]] を forward すると logits の
        2行目は [有限, -inf, -inf] になる。このエントロピーで backward した
        とき、logits 経路の8パラメータ(obs_mlp / action_mlp / action_head)の
        勾配が全て有限であること。旧実装では全て NaN になった。
        """
        torch.manual_seed(0)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        obs = torch.randn(2, 4)
        acts = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        logits, _ = net(obs, acts, mask)
        # mask 適用の構造を確認(2行目の padding が -inf になっている)。
        self.assertTrue(torch.isinf(logits[1, 1:]).all())
        self.assertTrue(torch.isfinite(logits[1, 0]))
        entropy = ppo.policy_entropy(logits, mask)
        self.assertTrue(torch.isfinite(entropy))
        (-entropy).backward()
        finite = []
        for name, param in net.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                finite.append(name)
        # value_head は entropy に寄与しないため勾配を持たない(grad=None)。
        # logits 経路の8パラメータは必ず有限勾配を持つこと。
        for name in ("obs_mlp.0.weight", "obs_mlp.0.bias",
                     "obs_mlp.2.weight", "obs_mlp.2.bias",
                     "action_mlp.0.weight", "action_mlp.0.bias",
                     "action_head.weight", "action_head.bias"):
            self.assertIn(name, finite)

    def test_update_with_padded_batch_keeps_weights_finite(self):
        """合法手数が異なる軌跡(padding を含むバッチ)で update しても重みが
        有限のままであること(歯)。実自己対戦では決定点ごとに合法手数が異なる
        ため padding はほぼ必ず発生し、旧実装では最初の更新で NaN 重みが
        生まれて学習が死んだ(scripts/train_ppo.py の実学習テストが
        FloatingPointError になることを Opus が実測)。
        """
        torch.manual_seed(0)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        cfg = ppo.PPOConfig(epochs=1, minibatch_size=64)
        t1 = _trajectory(players=[0, 1, 0], terminal=(1.0, -1.0),
                         num_actions=3, index=0)
        t2 = _trajectory(players=[0, 1], terminal=(-1.0, 1.0),
                         num_actions=1, index=1)
        batch = ppo.trajectories_to_batch([t1, t2], net, cfg)
        self.assertFalse(batch["action_mask"].all())  # padding が含まれる
        stats = ppo.update(
            net, torch.optim.Adam(net.parameters(), lr=cfg.lr),
            batch, cfg, seed=1)
        for key in ("policy_loss", "value_loss", "entropy", "total_loss"):
            self.assertTrue(math.isfinite(stats[key]), key)
        for name, param in net.named_parameters():
            self.assertTrue(torch.isfinite(param).all(), name)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class BatchConstructionTests(unittest.TestCase):
    """trajectories_to_batch の形状・整合性・discarded 除外。"""

    def _net(self):
        torch.manual_seed(0)
        return PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))

    def test_batch_shapes_mask_and_chosen(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        t1 = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=0,
                         num_actions=3)
        t2 = _trajectory(players=[0, 1], terminal=(-1.0, 1.0), index=1,
                         num_actions=2)
        batch = ppo.trajectories_to_batch([t1, t2], net, cfg)
        self.assertEqual(tuple(batch["obs"].shape), (4, 4))
        self.assertEqual(tuple(batch["actions"].shape), (4, 3, 2))
        self.assertEqual(tuple(batch["action_mask"].shape), (4, 3))
        self.assertEqual(tuple(batch["chosen_index"].shape), (4,))
        self.assertEqual(tuple(batch["advantages"].shape), (4,))
        self.assertEqual(tuple(batch["returns"].shape), (4,))
        expected_mask = torch.tensor([
            [True, True, True],
            [True, True, True],
            [True, True, False],
            [True, True, False],
        ])
        self.assertTrue(torch.equal(batch["action_mask"], expected_mask))
        self.assertTrue((batch["chosen_index"] >= 0).all())
        self.assertTrue(batch["action_mask"].gather(
            1, batch["chosen_index"].unsqueeze(1)).all())

    def test_old_log_probs_match_net_at_build_time(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        traj = _trajectory(players=[0, 1, 0], terminal=(1.0, -1.0))
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        logits, _ = net(batch["obs"], batch["actions"], batch["action_mask"])
        expected = torch.log_softmax(logits, dim=-1).gather(
            1, batch["chosen_index"].unsqueeze(1)).squeeze(1)
        self.assertTrue(torch.allclose(
            batch["old_log_probs"], expected, atol=1e-6))

    def test_discarded_trajectories_are_excluded(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        good1 = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=0)
        bad = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=1,
                          discarded=True, discard_reason="LegalityError")
        good2 = _trajectory(players=[0, 1], terminal=(-1.0, 1.0), index=2)
        batch = ppo.trajectories_to_batch([good1, bad, good2], net, cfg)
        self.assertEqual(batch["obs"].shape[0], 4)   # 2軌跡 × 2決定

    def test_all_discarded_raises(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        bad = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=1,
                          discarded=True, discard_reason="LegalityError")
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([bad], net, cfg)

    def test_non_two_player_rejected(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        three = _trajectory(players=[0, 1], terminal=(1.0, -1.0, 1.0))
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([three], net, cfg)

    def test_record_vectors_false_rejected(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        empty_obs = _trajectory(players=[0, 1], terminal=(1.0, -1.0))
        empty_obs = dataclasses.replace(empty_obs, decisions=tuple(
            dataclasses.replace(d, obs_vec=()) for d in empty_obs.decisions))
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([empty_obs], net, cfg)

    def test_chosen_out_of_range_rejected(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        bad = _trajectory(players=[0, 1], terminal=(1.0, -1.0))
        bad = dataclasses.replace(bad, decisions=tuple(
            dataclasses.replace(d, chosen_index=7) for d in bad.decisions))
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([bad], net, cfg)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class LearnerSeatBatchTests(unittest.TestCase):
    """設計書 14.1: learner の決定だけをバッチへ入れる(歯)。

    歯の狙い(実装を元へ戻すと落ちる):
    - `trajectories_to_batch` が `Trajectory.learner_seat` を無視して全決定を
      入れる実装へ戻すと、実軌跡テストのバッチサイズが learner の決定点数と
      一致せず落ちる(相手の決定が混入する)。
    - env が learner_seat を席置換後に解決せず(例: 常に 0)、seat_variant=1
      の軌跡で learner_seat が誤ると、相手の決定がバッチへ入り obs 多重集合
      の一致検査が落ちる。
    - learner の決定列でなく全決定列で GAE し、終端効用を learner の最後の
      決定以外へ帰属させると、手計算テストが落ちる。
    - learner_seat の範囲検査 / 決定が無い場合の拒否を外すと落ちる。
    - learner_seat=None(既定)の後方互換経路(全決定)を壊すと落ちる。
    """

    @staticmethod
    def _net():
        torch.manual_seed(0)
        return PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))

    @staticmethod
    def _real_net():
        """実ゲーム用の次元を持つ net(obs_dim(2) / action_dim)。"""
        torch.manual_seed(0)
        return PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(8, 8), action_hidden=(4,))

    def _assert_batch_is_exactly_learner_decisions(self, traj, batch):
        """バッチの行が軌跡の learner 決定と完全一致することを検査する。

        (1) 行数 == learner の決定点数、(2) 行の obs 多重集合が learner の
        決定列の obs 多重集合と一致(相手の決定が1つも入らない)。
        """
        learner_seat = traj.learner_seat
        self.assertIsNotNone(learner_seat)
        learner_decisions = [
            d for d in traj.decisions if d.player_id == learner_seat]
        self.assertGreater(len(learner_decisions), 0)
        self.assertGreater(
            len(traj.decisions) - len(learner_decisions), 0,
            "相手の決定が無いテストは無意味(vacuous)")
        self.assertEqual(
            batch["obs"].shape[0], len(learner_decisions),
            "バッチの行数は learner の決定点数と一致しなければならない")
        remaining = [
            torch.as_tensor(d.obs_vec, dtype=torch.float32)
            for d in learner_decisions]
        for row in batch["obs"]:
            matched = None
            for k, expected in enumerate(remaining):
                if torch.allclose(row, expected, atol=1e-6):
                    matched = k
                    break
            self.assertIsNotNone(
                matched, "バッチに learner 以外の決定(相手の手)が混入している")
            del remaining[matched]
        self.assertEqual(remaining, [], "learner の決定がバッチから欠落している")

    def test_real_trajectory_learner_only_both_seat_variants(self):
        """PolicyAgent(learner)対 HeuristicAgent の実軌跡で、バッチが learner
        の決定だけを含むことを seat_variant 0/1 の両方で固定する。

        seat_variant=0 では learner は席0、=1 では席1(席交換)。learner_seat が
        席置換後の正しい席を指し、その席の決定だけがバッチへ入ること、つまり
        席を入れ替えても learner を正しく特定できることが要点。
        """
        cfg = ppo.PPOConfig()
        for variant in (0, 1):
            net = self._real_net()
            learner = PolicyAgent(net, mode="sample", seed=123)
            traj = play_episode(
                "train", 2400 + variant, variant,
                [learner, HeuristicAgent()], learner_index=0)
            self.assertFalse(traj.discarded, traj.discard_reason)
            # N=2 の決定論的巡回: variant 0 → 席0 / variant 1 → 席1。
            self.assertEqual(
                traj.learner_seat, variant,
                "席置換後に learner の席が正しく記録されていない")
            self.assertEqual(traj.seats[traj.learner_seat], "PolicyAgent")
            batch = ppo.trajectories_to_batch([traj], net, cfg)
            self._assert_batch_is_exactly_learner_decisions(traj, batch)

    def test_real_trajectory_learner_vs_random_and_past_generation(self):
        """相手が RandomAgent / 過去世代の PolicyAgent でも learner の決定だけを
        バッチへ入れることを、両 seat_variant で固定する。"""
        cfg = ppo.PPOConfig()
        learner_net = self._real_net()
        learner = PolicyAgent(learner_net, mode="sample", seed=11)
        torch.manual_seed(1)
        past_net = PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(8, 8), action_hidden=(4,))
        past = PolicyAgent(past_net, mode="sample", seed=22)
        opponents = [
            ("random", lambda variant: RandomAgent(seed=99)),
            ("past", lambda variant: past),
        ]
        for variant in (0, 1):
            for label, make_opponent in opponents:
                index = 2410 + variant * 2 + (0 if label == "random" else 1)
                traj = play_episode(
                    "train", index, variant,
                    [learner, make_opponent(variant)], learner_index=0)
                self.assertFalse(traj.discarded, traj.discard_reason)
                self.assertEqual(traj.learner_seat, variant)
                batch = ppo.trajectories_to_batch(
                    [traj], learner_net, cfg)
                self._assert_batch_is_exactly_learner_decisions(traj, batch)

    def test_mixed_real_trajectories_in_one_batch(self):
        """learner と相手が混ざった複数実軌跡でも、バッチ全体が learner の
        決定だけで構成されること。"""
        cfg = ppo.PPOConfig()
        net = self._real_net()
        learner = PolicyAgent(net, mode="sample", seed=5)
        trajs = []
        for variant in (0, 1):
            trajs.append(play_episode(
                "train", 2420 + variant, variant,
                [learner, HeuristicAgent()], learner_index=0))
        batch = ppo.trajectories_to_batch(trajs, net, cfg)
        all_learner_obs = [
            torch.as_tensor(d.obs_vec, dtype=torch.float32)
            for traj in trajs
            for d in traj.decisions if d.player_id == traj.learner_seat]
        self.assertGreater(len(all_learner_obs), 0)
        self.assertEqual(
            batch["obs"].shape[0], len(all_learner_obs),
            "複数軌跡バッチの行数は learner の決定点数の合計と一致しなければ"
            "ならない(相手の決定が混入している)")
        remaining = list(all_learner_obs)
        for row in batch["obs"]:
            matched = None
            for k, expected in enumerate(remaining):
                if torch.allclose(row, expected, atol=1e-6):
                    matched = k
                    break
            self.assertIsNotNone(
                matched, "バッチに learner 以外の決定(相手の手)が混入している")
            del remaining[matched]
        self.assertEqual(remaining, [], "learner の決定がバッチから欠落している")

    def test_gae_computed_over_learner_sequence_only(self):
        """GAE は learner の決定列の上で計算され、終端効用は learner の最後の
        決定へ帰属する。視点は単一なので flip は恒等(手計算で固定)。"""
        net = self._net()
        cfg = ppo.PPOConfig(gamma=0.9, gae_lambda=0.95)
        # 決定列 [0, 1, 0, 1]: learner(席0)は 0 番目と 2 番目。相手の手は
        # 環境遷移の一部として扱い、バッチには learner の2決定だけが残る。
        traj = dataclasses.replace(
            _trajectory(players=[0, 1, 0, 1], terminal=(1.0, -1.0)),
            learner_seat=0)
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        self.assertEqual(batch["obs"].shape[0], 2)
        logits, values = net(batch["obs"], batch["actions"],
                             batch["action_mask"])
        # learner の決定列の報酬 [0, 0] + 終端 +1(learner の最後の決定へ)。
        flips = torch.tensor([1.0, 1.0], dtype=torch.float32)
        rewards_fixed = torch.tensor([0.0, 1.0])
        exp_adv, exp_ret = _hand_gae(
            rewards_fixed.tolist(), (values * flips).tolist(),
            [False, True], 0.9, 0.95)
        exp_adv = torch.tensor(exp_adv) * flips
        exp_ret = torch.tensor(exp_ret) * flips
        self.assertTrue(torch.allclose(
            batch["advantages"], exp_adv, atol=1e-6))
        self.assertTrue(torch.allclose(
            batch["returns"], exp_ret, atol=1e-6))

    def test_learner_at_second_seat_gets_own_terminal_utility(self):
        """learner が席1のとき、終端効用は learner の最後の決定へ席1の値を
        帰属させる(actor 視点の returns は -1)。"""
        net = self._net()
        cfg = ppo.PPOConfig()
        traj = dataclasses.replace(
            _trajectory(players=[0, 1, 0], terminal=(1.0, -1.0)),
            learner_seat=1)
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        self.assertEqual(batch["obs"].shape[0], 1)
        self.assertAlmostEqual(batch["returns"][0].item(), -1.0, places=6)

    def test_learner_seat_none_keeps_all_decisions(self):
        """learner_seat=None(既定)は全決定を入れる後方互換の経路。"""
        net = self._net()
        cfg = ppo.PPOConfig()
        traj = _trajectory(players=[0, 1, 0, 1], terminal=(1.0, -1.0))
        self.assertIsNone(traj.learner_seat)
        batch = ppo.trajectories_to_batch([traj], net, cfg)
        self.assertEqual(batch["obs"].shape[0], 4)

    def test_learner_seat_out_of_range_rejected(self):
        net = self._net()
        cfg = ppo.PPOConfig()
        traj = dataclasses.replace(
            _trajectory(players=[0, 1], terminal=(1.0, -1.0)),
            learner_seat=5)
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([traj], net, cfg)

    def test_learner_seat_with_no_decisions_rejected(self):
        """learner_seat の決定が1つも無い軌跡は拒否する。"""
        net = self._net()
        cfg = ppo.PPOConfig()
        traj = dataclasses.replace(
            _trajectory(players=[0, 0, 0], terminal=(1.0, -1.0),
                        turn_advanced=[False, False, True]),
            learner_seat=1)
        with self.assertRaises(ValueError):
            ppo.trajectories_to_batch([traj], net, cfg)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class RealGameBatchFuzzTests(unittest.TestCase):
    """設計書 14.4: 実局面由来の可変長合法手バッチでの学習全体の有限性(歯)。

    実自己対戦の軌跡(決定点ごとに合法手数が異なる)を多数生成して
    `trajectories_to_batch` でバッチ化し、`update`(backward と
    optimizer.step を含む)を繰り返しても、損失と全パラメータの勾配・重みが
    有限であることをファズ的に検査する。合成バッチのテストだけでは拾えない
    経路(実 obs のスケール、合法手数の偏り、padding の偏在)を固定する。
    実装を壊すと(padding 行の logit が -inf でなくなる / entropy の
    backward が NaN を生む / 実 obs で発散する)落ちる。
    """

    def test_many_real_variable_length_batches_stay_finite(self):
        torch.manual_seed(0)
        net = PolicyValueNet(
            enc.obs_dim(2), enc.action_dim(),
            hidden=(16, 16), action_hidden=(8,))
        learner = PolicyAgent(net, mode="sample", seed=20260813)
        trajectories = []
        for variant in (0, 1):
            for k in range(12):
                index = 2600 + variant * 100 + k
                if k % 2 == 0:
                    opponent = HeuristicAgent()
                else:
                    opponent = RandomAgent(seed=index)
                trajectories.append(play_episode(
                    "train", index, variant, [learner, opponent],
                    learner_index=0))
        usable = [t for t in trajectories if not t.discarded]
        self.assertGreater(len(usable), 0, "実軌跡が全て破棄された")
        # 可変長の実測: 合法手数の種類が複数あること(単一なら fuzz にならない)。
        legal_counts = {
            len(d.action_vecs)
            for t in usable
            for d in t.decisions if d.player_id == t.learner_seat}
        self.assertGreater(
            len(legal_counts), 1,
            "実局面の合法手数が単一(この fuzz は可変長を検証できない)")

        cfg = ppo.PPOConfig(epochs=2, minibatch_size=64)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        for step in range(5):
            # 本番同様、バッチは更新ごとに(価値・old log probs 込みで)作り直す。
            batch = ppo.trajectories_to_batch(usable, net, cfg)
            # 可変長の歯: padding が含まれている(合法手数の偏りがある)。
            self.assertFalse(batch["action_mask"].all())
            stats = ppo.update(net, optimizer, batch, cfg, seed=1000 + step)
            for key in ("policy_loss", "value_loss", "entropy",
                        "total_loss", "clip_frac", "approx_kl"):
                self.assertTrue(math.isfinite(stats[key]), key)
            for name, param in net.named_parameters():
                self.assertTrue(torch.isfinite(param).all(), name)
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class UpdateDeterminismTests(unittest.TestCase):
    """同一 seed・同一バッチ → 同一重み更新(設計書 §6.3 "ppo" ストリーム)。"""

    def test_same_seed_same_batch_same_weights(self):
        net1, batch1, cfg1 = _synthetic_batch(seed=3)
        net2, batch2, cfg2 = _synthetic_batch(seed=3)
        opt1 = torch.optim.Adam(net1.parameters(), lr=cfg1.lr)
        opt2 = torch.optim.Adam(net2.parameters(), lr=cfg2.lr)
        for _ in range(3):
            stats1 = ppo.update(net1, opt1, batch1, cfg1, seed=7)
            stats2 = ppo.update(net2, opt2, batch2, cfg2, seed=7)
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2), "weights diverged")
        self.assertEqual(stats1, stats2)

    def test_different_seed_changes_weights(self):
        net1, batch1, cfg1 = _synthetic_batch(seed=3)
        net2, batch2, cfg2 = _synthetic_batch(seed=3)
        ppo.update(net1, torch.optim.Adam(net1.parameters(), lr=cfg1.lr),
                   batch1, cfg1, seed=7)
        ppo.update(net2, torch.optim.Adam(net2.parameters(), lr=cfg2.lr),
                   batch2, cfg2, seed=8)
        self.assertTrue(any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(net1.parameters(), net2.parameters())))

    def test_update_without_old_log_probs_matches(self):
        """old_log_probs 無しバッチは更新開始時に同値が計算される。"""
        net1, batch1, cfg1 = _synthetic_batch(seed=5)
        net2, batch2, cfg2 = _synthetic_batch(seed=5)
        stripped = dict((k, v) for k, v in batch2.items()
                        if k != "old_log_probs")
        ppo.update(net1, torch.optim.Adam(net1.parameters(), lr=cfg1.lr),
                   batch1, cfg1, seed=1)
        ppo.update(net2, torch.optim.Adam(net2.parameters(), lr=cfg2.lr),
                   stripped, cfg2, seed=1)
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_update_validation(self):
        net, batch, cfg = _synthetic_batch(seed=5)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        empty = dict((k, v[:0]) for k, v in batch.items())
        with self.assertRaises(ValueError):
            ppo.update(net, opt, empty, cfg, seed=1)
        bad_mask = dict(batch)
        bad_mask["chosen_index"] = torch.zeros(
            batch["obs"].shape[0], dtype=torch.long)
        # chosen=0 は全行で合法なので、全行を masked にした行を作る。
        bad_mask["action_mask"] = torch.zeros_like(batch["action_mask"])
        bad_mask["action_mask"][:, 0] = True
        bad_mask["chosen_index"][:] = 1
        with self.assertRaises(ValueError):
            ppo.update(net, opt, bad_mask, cfg, seed=1)
        missing = dict((k, v) for k, v in batch.items() if k != "actions")
        with self.assertRaises(KeyError):
            ppo.update(net, opt, missing, cfg, seed=1)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class GradientAndStatsTests(unittest.TestCase):
    """勾配が有限で NaN / inf が出ないこと。"""

    def test_update_produces_finite_weights_and_stats(self):
        net, batch, cfg = _synthetic_batch(seed=11)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        stats = ppo.update(net, opt, batch, cfg, seed=1)
        for key in ("policy_loss", "value_loss", "entropy", "total_loss",
                    "clip_frac", "approx_kl"):
            self.assertIn(key, stats)
            self.assertTrue(math.isfinite(stats[key]), key)
        for name, param in net.named_parameters():
            self.assertTrue(torch.isfinite(param).all(), name)
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class SingleElementMinibatchTests(unittest.TestCase):
    """要素数1ミニバッチの advantage 正規化が NaN を生まないこと(歯)。

    総サンプル数が minibatch_size の倍数プラス1のとき末尾に要素数1の
    ミニバッチができ、不偏 std(自由度0)が NaN を返し loss が NaN 化して
    学習が死ぬ(scripts/train_ppo.py の実学習スモークで世代2に
    FloatingPointError として再現)。以下のテストは実装を元の
    `(adv - adv.mean()) / (adv.std() + 1e-8)` へ戻すと落ちる。
    """

    def test_normalize_advantages_single_element_is_finite(self):
        """1要素の正規化ヘルパーが有限の生値を返すこと(最小歯)。

        Opus 実測: `torch.tensor([0.5]).std()` は nan(自由度0の不偏推定)。
        `nan + 1e-8` は nan なので、std に 1e-8 を足す方式では守れない。
        修正は要素数2未満を正規化せず生値を返す。
        """
        adv = torch.tensor([0.5])
        got = ppo._normalize_advantages(adv)
        self.assertTrue(torch.isfinite(got).all())
        self.assertEqual(got.item(), 0.5)
        # 複数要素は従来どおり中心化・標準化される(退化ケース以外は
        # 従来実装と同一)。
        multi = torch.tensor([1.0, 2.0, 3.0])
        expected = (multi - multi.mean()) / (multi.std() + 1e-8)
        self.assertTrue(torch.allclose(
            ppo._normalize_advantages(multi), expected, atol=1e-7))

    def test_update_with_single_element_minibatch_keeps_finite(self):
        """サンプル数 = minibatch_size の倍数 + 1 で update しても loss と
        全パラメータ勾配が有限であること(歯・更新経路)。

        minibatch_size=2・サンプル数5(= 2*2 + 1)のとき、各 epoch の分割
        [0:2], [2:4], [4:5] はシャッフルに関係なく常に末尾へ要素数1の
        ミニバッチを含む。修正前の実装ではこのミニバッチで std が NaN に
        なり loss が非有限で FloatingPointError が上がる(あるいは isfinite
        検査を外した実装なら NaN 勾配・NaN 重みで落ちる)。
        """
        torch.manual_seed(0)
        net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
        cfg = ppo.PPOConfig(epochs=3, minibatch_size=2)
        t1 = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=0)
        t2 = _trajectory(players=[0, 1], terminal=(-1.0, 1.0), index=1)
        t3 = _trajectory(players=[0], terminal=(1.0, -1.0), index=2)
        batch = ppo.trajectories_to_batch([t1, t2, t3], net, cfg)
        self.assertEqual(batch["obs"].shape[0], 5)   # 2 + 2 + 1 決定
        stats = ppo.update(
            net, torch.optim.Adam(net.parameters(), lr=cfg.lr),
            batch, cfg, seed=1)
        for key in ("policy_loss", "value_loss", "entropy", "total_loss",
                    "clip_frac", "approx_kl"):
            self.assertTrue(math.isfinite(stats[key]), key)
        for name, param in net.named_parameters():
            self.assertTrue(torch.isfinite(param).all(), name)
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_single_element_minibatch_config_is_deterministic(self):
        """要素数1ミニバッチを含む構成でも同一 seed・同一バッチなら同一の
        重み更新になること。スキップ判定は要素数だけで決まり乱数に依存
        しないため、決定論(設計書 §6.3)は壊れない。
        """
        def build():
            torch.manual_seed(0)
            net = PolicyValueNet(4, 2, hidden=(8, 8), action_hidden=(4,))
            cfg = ppo.PPOConfig(epochs=2, minibatch_size=2)
            t1 = _trajectory(players=[0, 1], terminal=(1.0, -1.0), index=0)
            t2 = _trajectory(players=[0, 1], terminal=(-1.0, 1.0), index=1)
            t3 = _trajectory(players=[0], terminal=(1.0, -1.0), index=2)
            batch = ppo.trajectories_to_batch([t1, t2, t3], net, cfg)
            return net, batch, cfg
        net1, batch1, cfg1 = build()
        net2, batch2, cfg2 = build()
        for _ in range(2):
            ppo.update(net1,
                       torch.optim.Adam(net1.parameters(), lr=cfg1.lr),
                       batch1, cfg1, seed=11)
            ppo.update(net2,
                       torch.optim.Adam(net2.parameters(), lr=cfg2.lr),
                       batch2, cfg2, seed=11)
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.equal(p1, p2))


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class SyntheticTaskTests(unittest.TestCase):
    """小規模な合成タスクで学習が収束すること。

    収束の指標は (1) value_loss の減少、(2) policy のエントロピー減少(分布の
    尖り)、(3) 正解行動(行動 0)の選択確率の上昇。**PPO の clipped surrogate
    (policy_loss) は単調減少する量ではなくゼロ付近を上下するため、収束の
    指標として使わない**(実測で末尾10回平均が初回を下回らない)。

    収束の主張は**更新予算に依存する**(下の range(400) が予算)。同一設定で
    予算 150 回のとき prob0 は 0.3344 → 0.5646 で 0.6 に届かないが、400 回で
    0.8601、1000 回で 0.9815 になる(Opus 実測)。lr / epochs /
    minibatch_size / net の規模を変えたときは、prob0 が 0.6 を超えるまでに
    必要な更新数が変わるため、**予算を増やすか閾値を再検討する**こと。
    閾値を下げて通すのは「学習が進んでいる」という歯を失うため不可。
    """

    def test_loss_decreases_and_policy_improves(self):
        torch.manual_seed(0)
        config = ppo.PPOConfig(epochs=2, minibatch_size=16, lr=3e-3)
        net = PolicyValueNet(4, 2, hidden=(16, 16), action_hidden=(8,))
        trajs = _synthetic_trajectories()
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        batch0 = ppo.trajectories_to_batch(trajs, net, config)
        logits0, _ = net(batch0["obs"], batch0["actions"],
                         batch0["action_mask"])
        prob0_initial = torch.softmax(logits0, dim=-1)[:, 0].mean().item()
        value_losses = []
        entropies = []
        # 更新予算 400 回。150 回では prob0 = 0.5646 で閾値 0.6 未達だった
        # (Opus 実測)。予算の見直し方法はクラス docstring を参照。
        for _ in range(400):
            # 本番同様に「軌跡は同じ、バッチ(価値・old log probs)は更新ごとに
            # 作り直す」。同一バッチを回し続けると clip が飽和して止まるため。
            batch = ppo.trajectories_to_batch(trajs, net, config)
            stats = ppo.update(net, optimizer, batch, config, seed=1)
            value_losses.append(stats["value_loss"])
            entropies.append(stats["entropy"])
        logits1, _ = net(batch0["obs"], batch0["actions"],
                         batch0["action_mask"])
        prob0_final = torch.softmax(logits1, dim=-1)[:, 0].mean().item()
        # value_loss の減少: 価値ヘッドが returns(±1 付近)へ近づく。
        self.assertLess(
            sum(value_losses[-10:]) / 10.0, value_losses[0])
        # entropy の減少: 方策が一様分布(≈log 3)から正解へ尖る。
        self.assertLess(
            sum(entropies[-10:]) / 10.0, entropies[0])
        # 正解行動(行動 0)の選択確率の上昇。
        self.assertGreater(prob0_final, 0.6)
        self.assertGreater(prob0_final, prob0_initial)
        for name, param in net.named_parameters():
            self.assertTrue(torch.isfinite(param).all(), name)


@unittest.skipUnless(HAS_TORCH, "torch が無い環境では skip")
class ThreadControlTests(unittest.TestCase):
    """ppo モジュール import 時にスレッド数と環境変数を1へ固定すること。"""

    def test_threads_are_pinned_to_one(self):
        self.assertEqual(torch.get_num_threads(), 1)
        for name in THREAD_ENV_NAMES:
            self.assertEqual(os.environ.get(name), "1")


if __name__ == "__main__":
    unittest.main()
