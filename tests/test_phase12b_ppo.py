"""Phase 12b C3: gen199 fork / 保存座標 resume / lr 段 / snapshot / PFSP / exploiter /
順位 GAE / worker・席置換一致 / SIGTERM→resume(学習 CLI の opt-in 経路)。

gen199 の実 Adam 状態は repo に無い(csi の Phase 10 run と gitignored の smoke コピーだけ)。
ここでは同じ形の合成 checkpoint(generation=199、Adam を 1 step 進めた状態、lr=2e-5、
next_episode_index=1204800)を作って契約を検査し、実 checkpoint での確認は
docs/eval-runs/phase12b/smoke/ の証跡に残す。
"""
import dataclasses
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch が無い環境")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

if _HAS_TORCH:
    import train_ppo  # noqa: E402
    from naeco.train import checkpoint as checkpoint_mod  # noqa: E402
    from naeco.train import env, pool  # noqa: E402
    from naeco.train import net as net_mod  # noqa: E402
    from naeco.train import ppo  # noqa: E402

# C3 smoke 帯(train 19,910,000–19,919,999。docs/eval-runs/phase12b/smoke/ で事前登録)。
SMOKE_LO = 19910000
EP = 4


def _sha(path):
    return train_ppo._sha256_file(str(path))


def make_fork(tmp, n, lr=2e-5, seed=7):
    """gen199 と同じ形の fork 入力(meta/weights/optimizer)を作り (meta, sha) を返す。"""
    d = Path(tmp) / ("src%d" % n)
    net = net_mod.build_for_players(n, seed=seed)
    opt = ppo.new_adam_optimizer(net.parameters(), lr)
    loss = sum((p * 0.001).sum() for p in net.parameters())
    loss.backward()
    opt.step()
    args = train_ppo._parse_args(["--num-players", str(n), "--start-index",
                                  "1000000", "--lr", str(lr)])
    extra = train_ppo._meta_extra(args, ppo.PPOConfig(lr=lr), 199, 200, 1204800,
                                  1000000, ppo.torch_version())
    train_ppo._save_checkpoint(str(d), 199, net, opt, extra, n)
    meta = d / "gen199.meta.json"
    sha = ":".join(_sha(d / name) for name in (
        "gen199.meta.json", "gen199.weights.pt", "gen199.optimizer.pt"))
    return str(meta), sha


def _argv(n, mode, ckpt, target_age, lr="2e-05", stage="start", workers=1,
          start=SMOKE_LO, extra=()):
    return ["--num-players", str(n), "--phase12b-mode", mode,
            "--checkpoint-dir", str(ckpt), "--start-index", str(start),
            "--max-index", str(start + 999), "--target-age", str(target_age),
            "--generations", "50", "--max-seconds", "600", "--episodes", str(EP),
            "--workers", str(workers), "--lr", lr, "--lr-stage", stage,
            "--snapshot-every", "2"] + list(extra)


def _run(argv):
    args = train_ppo._parse_args(argv)
    return train_ppo.run(args)


def _cli(argv, timeout=900):
    env_ = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "train_ppo.py")]
                          + argv, cwd=str(ROOT), env=env_, capture_output=True,
                          text=True, timeout=timeout)


def _meta(ckpt, gen):
    with open(Path(ckpt) / ("gen%03d.meta.json" % gen), encoding="utf-8") as fh:
        return json.load(fh)


def _log(ckpt):
    with open(Path(ckpt) / "phase12b-generations.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _opt(ckpt, gen):
    return ppo.load_state_dict(str(Path(ckpt) / ("gen%03d.optimizer.pt" % gen)))


# --------------------------------------------------------------------- fork


def test_fork_keeps_weights_and_adam_and_uses_block_head(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    ckpt = tmp_path / "a"
    assert _run(_argv(3, "A", ckpt, 1, extra=["--fork-from", src,
                                               "--fork-sha256", sha])) == 0
    meta = _meta(ckpt, 0)
    block = meta["hyperparameters"]["phase12b"]
    assert block["fork"]["sha256"] == dict(zip(("meta", "weights", "optimizer"),
                                               sha.split(":")))
    assert block["fork"]["source_next_episode_index"] == 1204800
    assert block["absolute_generation"] == 200 and block["age"] == 1
    assert _log(ckpt)[0]["index_lo"] == SMOKE_LO          # 新ブロック先頭
    assert meta["next_episode_index"] == SMOKE_LO + EP
    assert meta["training_steps"] == 201                    # 継承 + 1
    # Adam: fork 入力は step=1。継承していれば 1 回の更新後は step = 1 + 更新数 > 1。
    src_state = ppo.load_state_dict(src.replace(".meta.json", ".optimizer.pt"))
    src_steps = {k: float(v["step"]) for k, v in src_state["state"].items()}
    new_steps = {k: float(v["step"]) for k, v in _opt(ckpt, 0)["state"].items()}
    assert set(src_steps) == set(new_steps)
    updates = int(round(new_steps[0] - src_steps[0]))
    assert updates >= 1 and all(new_steps[k] - src_steps[k] == updates
                                for k in src_steps)
    # fork 入力の複製は読取専用で保持され SHA 一致
    for name, digest in zip(("gen199.meta.json", "gen199.weights.pt",
                             "gen199.optimizer.pt"), sha.split(":")):
        copy = ckpt / "fork" / name
        assert _sha(copy) == digest and not os.access(str(copy), os.W_OK)


def test_fork_weights_are_the_fork_input_before_update(tmp_path, monkeypatch):
    src, sha = make_fork(tmp_path, 4)
    seen = {}
    real = ppo.update

    def spy(net, optimizer, batch, config, seed):
        if "w" not in seen:
            seen["w"] = {k: v.clone() for k, v in net.state_dict().items()}
            seen["lr"] = [g["lr"] for g in optimizer.param_groups]
            seen["state"] = len(optimizer.state)
        return real(net, optimizer, batch, config, seed)

    monkeypatch.setattr(ppo, "update", spy)
    assert _run(_argv(4, "A", tmp_path / "a", 1, extra=[
        "--fork-from", src, "--fork-sha256", sha])) == 0
    ref = ppo.load_state_dict(src.replace(".meta.json", ".weights.pt"))
    assert all(net_mod.torch.equal(seen["w"][k], ref[k]) for k in ref)
    assert seen["lr"] == [2e-05] and seen["state"] > 0


def test_fork_rejects_sha_mismatch_lr_mismatch_and_old_block(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    bad = sha[:-1] + ("0" if sha[-1] != "0" else "1")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _run(_argv(3, "A", tmp_path / "x1", 1,
                   extra=["--fork-from", src, "--fork-sha256", bad]))
    with pytest.raises(ValueError, match="must equal the fork checkpoint lr"):
        _run(_argv(3, "A", tmp_path / "x2", 1, lr="3e-05",
                   extra=["--fork-from", src, "--fork-sha256", sha]))
    with pytest.raises(ValueError, match="new block"):
        _run(_argv(3, "A", tmp_path / "x3", 1, start=1100000,
                   extra=["--fork-from", src, "--fork-sha256", sha]))
    with pytest.raises(ValueError, match="fresh phase12b run must use"):
        _run(_argv(3, "A", tmp_path / "x4", 1, lr="1e-05", stage="extend",
                   extra=["--fork-from", src, "--fork-sha256", sha]))


@pytest.mark.parametrize("argv_fix,match", [
    (lambda a: [x for x in a if x != "--lr" and x != "2e-05"], "explicit --lr"),
    (lambda a: a[:a.index("--target-age")] + a[a.index("--target-age") + 2:],
     "target-age"),
    (lambda a: a[:a.index("--max-seconds")] + a[a.index("--max-seconds") + 2:],
     "bounded"),
])
def test_cli_requires_explicit_bounded_options(tmp_path, argv_fix, match, capsys):
    src, sha = make_fork(tmp_path, 3)
    argv = argv_fix(_argv(3, "A", tmp_path / "c", 1,
                          extra=["--fork-from", src, "--fork-sha256", sha]))
    with pytest.raises(SystemExit):
        train_ppo._parse_args(argv)
    assert match in capsys.readouterr().err


def test_phase12b_options_without_mode_are_rejected(capsys):
    with pytest.raises(SystemExit):
        train_ppo._parse_args(["--num-players", "3", "--max-index", "5"])
    with pytest.raises(SystemExit):
        train_ppo._parse_args(["--num-players", "2", "--phase12b-mode", "A"])


# ------------------------------------------------------ resume and lr stage


def test_resume_uses_saved_coordinates_and_lr_halves_once(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    ckpt = tmp_path / "a"
    fork = ["--fork-from", src, "--fork-sha256", sha]
    assert _run(_argv(3, "A", ckpt, 2, extra=fork)) == 0
    # 通常 resume: 保存座標(start_index ではない)から続く
    assert _run(_argv(3, "A", ckpt, 3, extra=fork)) == 0
    log = _log(ckpt)
    assert [r["index_lo"] for r in log] == [SMOKE_LO + EP * k for k in range(3)]
    assert [r["optimizer_lr"] for r in log] == [[2e-05]] * 3
    # 延長: 1 回だけ半減し、optimizer の実 lr と meta が一致
    assert _run(_argv(3, "A", ckpt, 4, lr="1e-05", stage="extend", extra=fork)) == 0
    hist = _meta(ckpt, 3)["hyperparameters"]["phase12b"]["lr_history"]
    assert hist == [{"from_age": 0, "lr": 2e-05, "stage": "start"},
                    {"from_age": 3, "lr": 1e-05, "stage": "extend"}]
    # 延長後 resume: 再半減しない(5e-6 は拒否、1e-5 は既適用として継続)
    with pytest.raises(ValueError, match="half of"):
        _run(_argv(3, "A", ckpt, 5, lr="5e-06", stage="extend", extra=fork))
    with pytest.raises(ValueError, match="no extension"):
        _run(_argv(3, "A", ckpt, 5, extra=fork))
    assert _run(_argv(3, "A", ckpt, 5, lr="1e-05", stage="extend", extra=fork)) == 0
    meta = _meta(ckpt, 4)
    assert meta["hyperparameters"]["phase12b"]["lr_history"] == hist
    assert meta["hyperparameters"]["phase12b"]["optimizer_lr"] == [1e-05]
    assert [g["lr"] for g in _opt(ckpt, 4)["param_groups"]] == [1e-05]
    assert [r["optimizer_lr"] for r in _log(ckpt)] == \
        [[2e-05]] * 3 + [[1e-05]] * 2
    assert [r["index_lo"] for r in _log(ckpt)] == \
        [SMOKE_LO + EP * k for k in range(5)]


def test_resume_rejects_changed_inputs(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    other, other_sha = make_fork(tmp_path / "o", 3, seed=8)
    assert other_sha != sha
    ckpt = tmp_path / "a"
    fork = ["--fork-from", src, "--fork-sha256", sha]
    assert _run(_argv(3, "A", ckpt, 1, extra=fork)) == 0
    with pytest.raises(ValueError, match="fork SHA differs"):
        _run(_argv(3, "A", ckpt, 2, extra=["--fork-from", other,
                                            "--fork-sha256", other_sha]))
    with pytest.raises(ValueError, match="resume block"):
        _run(_argv(3, "A", ckpt, 2, start=SMOKE_LO + 1000, extra=fork))
    with pytest.raises(ValueError, match="resume mode"):
        _run(_argv(3, "B", ckpt, 2, extra=fork + ["--pool-manifest",
                                                   _pool_manifest(tmp_path, 3)]))


def test_block_exhaustion_stops_before_crossing_max_index(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    argv = _argv(3, "A", tmp_path / "a", 5, extra=["--fork-from", src,
                                                    "--fork-sha256", sha])
    argv[argv.index("--max-index") + 1] = str(SMOKE_LO + 2 * EP - 1)
    assert _run(argv) == train_ppo.PHASE12B_EXIT_BLOCK_EXHAUSTED
    assert [r["index_hi"] for r in _log(tmp_path / "a")] == \
        [SMOKE_LO + EP - 1, SMOKE_LO + 2 * EP - 1]


def test_snapshots_are_kept_beyond_keep_latest(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    ckpt = tmp_path / "a"
    assert _run(_argv(3, "A", ckpt, 12, extra=["--fork-from", src,
                                                "--fork-sha256", sha])) == 0
    gens = train_ppo._list_generations(str(ckpt))
    assert gens == list(range(7, 12))                       # keep_latest=5
    with open(ckpt / "snapshots" / "index.json", encoding="utf-8") as fh:
        rows = json.load(fh)["snapshots"]
    assert [r["age"] for r in rows] == [2, 4, 6, 8, 10, 12]
    assert [r["absolute_generation"] for r in rows] == [201, 203, 205, 207, 209, 211]
    for r in rows:
        for key, suffix in (("meta", ".meta.json"), ("weights", ".weights.pt"),
                            ("optimizer", ".optimizer.pt")):
            path = ckpt / "snapshots" / ("age%04d%s" % (r["age"], suffix))
            assert _sha(path) == r["sha256"][key]
    # snapshot meta は G4 ラダー等の consumer が読める(人数・A 表検証を通る)
    meta = str(ckpt / "snapshots" / "age0002.meta.json")
    model = net_mod.load_checkpoint(meta, 3)[0]
    assert model.arch_dict()["obs_dim"] == 688


# --------------------------------------------------------------- PFSP (B)


def _pool_manifest(tmp, n, clusters=1, interval=2):
    init_meta = str(ROOT / ("docs/eval-runs/phase10/models/n%d/initial.meta.json" % n))
    m = {"schema": pool.POOL_SCHEMA, "num_players": n,
         "categories": {"latest": 0.5, "pfsp": 0.35, "biased": 0.15},
         "entries": [{"name": "initial", "kind": "checkpoint", "path": init_meta,
                      "sha256": {"meta": _sha(init_meta), "weights": _sha(
                          init_meta.replace(".meta.json", ".weights.pt"))}}],
         "pfsp_tables": [
             {"id": "past", "opponents": ["@past"] * (n - 1)},
             {"id": "past-heuristic",
              "opponents": ["@past"] + ["heuristic"] * (n - 2)},
             {"id": "initial", "opponents": ["initial"] * (n - 1)}],
         "biased_tables": [
             {"id": "school", "opponents": ["biased:school_first"] * (n - 1)},
             {"id": "avoid", "opponents": ["biased:avoid_hiring"] * (n - 1)},
             {"id": "mine", "opponents": ["biased:mine_first"] * (n - 1)}],
         "pfsp": {"interval_generations": interval, "clusters": clusters,
                  "diag_start_index": SMOKE_LO + 5000,
                  "diag_max_index": SMOKE_LO + 5999,
                  "floor": 0.05, "unmeasured_x": 0.5},
         "tie_break": "Action.sort_key ascending (biased/heuristic)",
         "seed_domain": "derive_seed(eval_seed(train,index),pfsp-table,index)"}
    path = Path(tmp) / ("pool%d.json" % n)
    path.write_text(json.dumps(m, indent=1, sort_keys=True))
    return str(path)


def test_pfsp_windows_measure_freeze_and_weight(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    manifest = _pool_manifest(tmp_path, 3)
    ckpt = tmp_path / "b"
    extra = ["--fork-from", src, "--fork-sha256", sha, "--pool-manifest", manifest]
    assert _run(_argv(3, "B", ckpt, 3, extra=extra)) == 0
    w0 = json.loads((ckpt / "pfsp" / "window000.json").read_text())
    w1 = json.loads((ckpt / "pfsp" / "window001.json").read_text())
    for w in (w0, w1):
        rates = [t["win_rate"] for t in w["tables"]]
        assert w["weights"] == pool.pfsp_weights(rates, 3)
        assert all(t["games"] == 3 for t in w["tables"])    # 1 cluster x n
    assert set(w0["past_set"]) == {"fork"}
    assert set(w1["past_set"]) == {"fork", "age0002"}      # window 境界で更新
    assert w1["index_lo"] == w0["index_hi"] + 1
    meta = _meta(ckpt, 2)
    assert meta["hyperparameters"]["pool_manifest"]["sha256"] == _sha(manifest)
    tables = {}
    for row in _log(ckpt):
        for k, v in row["tables"].items():
            tables[k] = tables.get(k, 0) + v
    assert sum(tables.values()) == 3 * EP
    # resume は凍結済み window を再計測せず読み直す
    assert _run(_argv(3, "B", ckpt, 4, extra=extra)) == 0
    assert json.loads((ckpt / "pfsp" / "window001.json").read_text()) == w1


# ---------------------------------------------------------------- exploiter


def test_exploiter_learner_one_seat_vs_frozen_strong(tmp_path, monkeypatch):
    from naeco.ai.search.ismcts import IsmctsAgent
    from naeco.train.search_evaluator import NetEvaluator
    from naeco.train.agent import PolicyAgent
    made = []

    def cheap(net, seed, cards):
        made.append(seed)
        return IsmctsAgent(root_seed=seed, budget_transitions=40, cards=cards,
                           evaluator=NetEvaluator(net),
                           policy_agent=PolicyAgent(net, mode="greedy", seed=0))

    monkeypatch.setattr(train_ppo, "_strong_agent", cheap)
    n = 3
    init = str(ROOT / "docs/eval-runs/phase10/models/n3/initial.meta.json")
    target = str(ROOT / "docs/eval-runs/phase10/models/n3/gen199.meta.json")
    isha = "%s:%s" % (_sha(init), _sha(init.replace(".meta.json", ".weights.pt")))
    tsha = "%s:%s" % (_sha(target), _sha(target.replace(".meta.json", ".weights.pt")))
    ckpt = tmp_path / "e"
    seen = []
    real = env.rollout

    def spy(specs, **kw):
        seen.extend(specs)
        return real(specs, **kw)

    monkeypatch.setattr(train_ppo, "rollout", spy)
    assert _run(_argv(n, "exploiter", ckpt, 1, extra=[
        "--init-from", init, "--init-sha256", isha,
        "--exploit-target", target, "--exploit-target-sha256", tsha])) == 0
    assert len(made) == EP * (n - 1)
    for unused_g, unused_i, unused_v, agents in seen:
        assert [a.name for a in agents] == ["learner"] + ["target-strong"] * (n - 1)
    block = _meta(ckpt, 0)["hyperparameters"]["phase12b"]
    assert block["target"]["sha256"]["weights"] == tsha.split(":")[1]
    assert block["init"]["sha256"]["meta"] == isha.split(":")[0]
    assert "fork" not in block
    with pytest.raises(SystemExit):
        train_ppo._parse_args(_argv(n, "exploiter", ckpt, 2, lr="1e-05",
                                    stage="extend", extra=[
            "--init-from", init, "--init-sha256", isha,
            "--exploit-target", target, "--exploit-target-sha256", tsha]))


# ------------------------------------------------------------ rank GAE


def test_rank_utility_gae_matches_hand_calculation_n4():
    torch = net_mod.torch

    class Values:
        def __call__(self, obs, actions, mask):
            return torch.zeros(mask.shape), obs[:, 0]

    # learner seat 2 finishes 3rd of 4: u = 1 - 2*2/3 = -1/3
    terminal = (1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0)
    d = env.Decision
    decisions = (
        d(2, (0.1,), ((1.0,),), 0, 0.0, True),
        d(0, (9.0,), ((1.0,),), 0, 0.0, True),
        d(2, (0.3,), ((1.0,),), 0, 0.0, False),
        d(2, (-0.2,), ((1.0,),), 0, 0.0, True),
        d(3, (5.0,), ((1.0,),), 0, 0.0, True),
    )
    traj = env.Trajectory("train", 1, 0, ("a", "b", "learner", "c"), decisions,
                          terminal, "h", False, "", 2)
    cfg = ppo.PPOConfig(gamma=0.997, gae_lambda=0.95)
    batch = ppo.trajectories_to_batch([traj], Values(), cfg)
    g, lam = 0.997, 0.95
    r = -1.0 / 3.0
    a2 = r - (-0.2)
    a1 = g * (-0.2) - 0.3 + g * lam * a2
    a0 = g * 0.3 - 0.1 + g * lam * a1
    assert batch["advantages"].tolist() == pytest.approx([a0, a1, a2], abs=1e-6)
    assert batch["returns"].tolist() == pytest.approx([a0 + 0.1, a1 + 0.3, r],
                                                      abs=1e-6)


# ------------------------------------- determinism: seats, tables, workers


def test_seat_and_table_assignment_depend_only_on_coordinates(tmp_path):
    """spec 順の置換で各局は不変、席は全席を巡る(席・卓置換の一致)。"""
    from naeco.ai.heuristic_agent import HeuristicAgent
    from naeco.ai.biased_agent import BiasedAgent
    from naeco.train.agent import PolicyAgent
    n = 4
    net = net_mod.build_for_players(n, seed=3)

    def agents(i):
        root = train_ppo._episode_root(i)
        a = [PolicyAgent(net, mode="sample",
                         seed=train_ppo.derive_seed(root, "seat", i, 1)),
             HeuristicAgent(0), BiasedAgent("school_first"), BiasedAgent("mine_first")]
        for k, name in enumerate(["learner", "h", "s", "m"]):
            a[k].name = name
        return a

    idx = list(range(SMOKE_LO + 9000, SMOKE_LO + 9012))
    fwd = env.rollout([("train", i, 0, agents(i)) for i in idx],
                      randomize_seats=True, learner_index=0)
    rev = env.rollout([("train", i, 0, agents(i)) for i in reversed(idx)],
                      randomize_seats=True, learner_index=0)[::-1]
    assert [train_ppo._trajectory_sha256([t]) for t in fwd] == \
        [train_ppo._trajectory_sha256([t]) for t in rev]
    assert {t.learner_seat for t in fwd} == set(range(n))


@pytest.mark.parametrize("mode,n", [("B", 4), ("A", 3)])
def test_worker_one_and_four_trajectories_and_weights_identical(tmp_path, mode, n):
    src, sha = make_fork(tmp_path, n)
    extra = ["--fork-from", src, "--fork-sha256", sha]
    if mode == "B":
        extra += ["--pool-manifest", _pool_manifest(tmp_path, n)]
    outs = []
    for workers in (1, 4):
        ckpt = tmp_path / ("w%d" % workers)
        proc = _cli(_argv(n, mode, ckpt, 2, workers=workers, extra=extra))
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        pf = sorted((ckpt / "pfsp").glob("*.json")) if mode == "B" else []
        outs.append(([(r["trajectory_sha256"], r["weights_sha256"], r["tables"])
                      for r in _log(ckpt)],
                     [json.loads(p.read_text())["weights"] for p in pf],
                     [[t["trajectory_sha256"] for t in json.loads(p.read_text())["tables"]]
                      for p in pf]))
    assert outs[0] == outs[1]


# ----------------------------------------------------------- SIGTERM -> resume


def test_sigterm_then_resume_matches_continuous_run(tmp_path):
    src, sha = make_fork(tmp_path, 3)
    extra = ["--fork-from", src, "--fork-sha256", sha]
    cont = tmp_path / "cont"
    proc = _cli(_argv(3, "A", cont, 4, extra=extra))
    assert proc.returncode == 0, proc.stdout[-2000:]
    cut = tmp_path / "cut"
    log = tmp_path / "cut.log"
    env_ = dict(os.environ, OMP_NUM_THREADS="1")
    with open(log, "w") as out:
        child = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "train_ppo.py")] +
            _argv(3, "A", cut, 4, extra=extra), cwd=str(ROOT), env=env_,
            stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if "generation 0 completed" in log.read_text():
                break
            time.sleep(0.05)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=120) == 0
    text = log.read_text()
    assert "SIGTERM" in text
    before = _log(cut)
    assert 1 <= len(before) < 4, text[-2000:]
    proc = _cli(_argv(3, "A", cut, 4, extra=extra))
    assert proc.returncode == 0, proc.stdout[-2000:]
    a, b = _log(cont), _log(cut)
    keys = ("age", "index_lo", "index_hi", "trajectory_sha256", "weights_sha256",
            "optimizer_lr")
    assert [[r[k] for k in keys] for r in a] == [[r[k] for k in keys] for r in b]
    assert _sha(cont / "gen003.optimizer.pt") == _sha(cut / "gen003.optimizer.pt")
