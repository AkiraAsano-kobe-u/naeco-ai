"""D5 own-seat GAE and independent pool draws."""
import dataclasses
import importlib.util

import pytest

from naeco.train import env, pool

# net / ppo import torch at module top; without torch (CI: pip install -e .[dev])
# only the tests that use them are skipped, the pool test still runs.
_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    from naeco.train import net as net_mod
    from naeco.train import ppo
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch が無い環境")


@requires_torch
@pytest.mark.parametrize("n", [3, 4])
def test_self_seat_gae(n):
    torch = net_mod.torch

    class Values:
        def __call__(self, obs, actions, mask):
            return torch.zeros(mask.shape), obs[:, 0]

    decisions = (
        env.Decision(0, (0.2,), ((1.0,),), 0, 0.0, False),
        env.Decision(0, (0.4,), ((1.0,),), 0, 0.0, True),
        env.Decision(1, (50.0,), ((1.0,),), 0, 7.0, True),
        env.Decision(0, (0.6,), ((1.0,),), 0, 0.0, True),
    )
    trajectory = env.Trajectory("train", 900000, 0, tuple(range(n)),
                                decisions, (1.0,) + (-1.0,) * (n - 1),
                                "hash", False, "", 0)
    config = ppo.PPOConfig(gamma=0.997, gae_lambda=0.95)
    batch = ppo.trajectories_to_batch([trajectory], Values(), config)
    a2 = 1.0 - 0.6
    a1 = 0.997 * 0.6 - 0.4 + 0.997 * 0.95 * a2
    a0 = 0.997 * 0.4 - 0.2 + 0.997 * 0.95 * a1
    assert batch["advantages"].tolist() == pytest.approx([a0, a1, a2])
    changed = dataclasses.replace(trajectory, decisions=decisions[:2] + (
        dataclasses.replace(decisions[2], obs_vec=(-999.0,), reward=-80.0),
    ) + decisions[3:])
    other = ppo.trajectories_to_batch([changed], Values(), config)
    assert torch.equal(batch["advantages"], other["advantages"])
    assert batch["returns"].tolist() == pytest.approx([a0 + .2, a1 + .4, 1.0])
    with pytest.raises(ValueError, match="requires learner_seat"):
        ppo.trajectories_to_batch([
            dataclasses.replace(trajectory, learner_seat=None)], Values(), config)


@requires_torch
@pytest.mark.parametrize("n", [2, 3, 4])
def test_meta_extra_is_b_table_only(n):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_ppo
    from naeco.train import checkpoint
    args = train_ppo._parse_args(["--num-players", str(n), "--start-index", "930000"])
    extra = train_ppo._meta_extra(args, ppo.PPOConfig(), 0, 0, 930064, 930000, "x")
    assert set(extra) <= set(checkpoint.B_KEYS)
    model = net_mod.build_for_players(n)
    meta = checkpoint.current_meta(n, model.arch_dict(), extra=extra)
    hyper = meta["hyperparameters"]
    if n == 2:
        assert "pool_manifest" not in hyper and "device" not in hyper
    else:
        assert hyper["device"] == "cpu"
        assert hyper["pool_manifest"]["num_players"] == n


def test_pool_categories_and_independent_coordinates():
    latest = pool.PoolEntry("latest", "checkpoint", "latest", 1.0)
    past = [pool.PoolEntry("gen%d" % i, "checkpoint", "past", 1.0)
            for i in range(5)]
    opponents = pool.training_pool(latest, past)
    draws = [opponents.sample_multi(7, i, 1).name for i in range(10000)]
    assert .47 < draws.count("latest") / len(draws) < .53
    assert .22 < draws.count("heuristic") / len(draws) < .28
    assert set(draws) == {"latest", "heuristic"} | {e.name for e in past}
    assert draws == [opponents.sample_multi(7, i, 1).name for i in range(10000)]
    assert draws != [opponents.sample_multi(7, i, 2).name for i in range(10000)]
    fresh = pool.training_pool(latest, [])
    names = [fresh.sample_multi(7, i, 1).name for i in range(10000)]
    assert .72 < names.count("latest") / len(names) < .78
