"""Phase 12b C3fix: runtime gate regressions (t_f23cfe7b).

B1 deadline/TERM で worker・resource_tracker を残さない(実 CLI と csi_job)
B2 不合法 discard が1局でもあれば更新・checkpoint なしで非0
B3 有限 watchdog / safety / RSS 監視のダミー job smoke(csi 実機は使わない)
B4 ladder.py のモデル席 consumer CLI
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch が無い環境")

if _HAS_TORCH:
    import test_phase12b_ppo as fixture  # noqa: E402
    import train_ppo  # noqa: E402

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def _group_members(pgid):
    out = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], stdout=subprocess.PIPE,
                         universal_newlines=True, check=True).stdout
    return [int(c[0]) for c in (line.split() for line in out.splitlines())
            if len(c) >= 3 and c[1] == str(pgid) and not c[2].startswith("Z")]


def _deadline_argv(n, ckpt, extra):
    argv = fixture._argv(n, "A", ckpt, 1, workers=2, extra=extra)
    argv[argv.index("--max-seconds") + 1] = "0.5"
    argv[argv.index("--episodes") + 1] = "160"
    return argv


# ---------------------------------------------------------------- B1


@needs_torch
@pytest.mark.parametrize("n", [3, 4])
def test_deadline_during_rollout_leaves_no_children(tmp_path, n):
    src, sha = fixture.make_fork(tmp_path, n)
    argv = _deadline_argv(n, tmp_path / "ck", ["--fork-from", src, "--fork-sha256", sha])
    log = tmp_path / "train.log"
    with open(log, "w") as out:
        child = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "train_ppo.py")] + argv,
            cwd=str(ROOT), env=ENV, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            code = child.wait(timeout=180)
            time.sleep(1.0)
            remaining = _group_members(child.pid)
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    text = log.read_text()
    assert "deadline during rollout" in text, text[-2000:]
    assert code == 0, text[-2000:]
    assert remaining == [], (remaining, text[-1500:])
    assert "child convergence" in text and "remaining=0" in text



@needs_torch
def test_one_invalid_discard_stops_without_update_or_checkpoint(tmp_path, monkeypatch):
    src, sha = fixture.make_fork(tmp_path, 3)
    real = train_ppo.rollout

    def one_invalid(specs, **kw):
        trajectories = real(specs, **kw)
        trajectories[0] = dataclasses.replace(
            trajectories[0], discarded=True, terminal_rewards=(),
            discard_reason="injected LegalityError")
        return trajectories

    def no_update(*a, **kw):
        raise AssertionError("update must not run after an invalid discard")

    monkeypatch.setattr(train_ppo, "rollout", one_invalid)
    monkeypatch.setattr(fixture.ppo, "update", no_update)
    ck = tmp_path / "ck"
    code = train_ppo.run(train_ppo._parse_args(
        fixture._argv(3, "A", ck, 1, extra=["--fork-from", src, "--fork-sha256", sha])))
    assert code == train_ppo.PHASE12B_EXIT_INVALID_DISCARD != 0
    assert not list(ck.glob("gen*.meta.json"))
    assert not list(ck.glob("gen*.weights.pt"))
    assert not (ck / "phase12b-generations.jsonl").exists()
    record = json.loads((ck / train_ppo.PHASE12B_FAILURE_FILE).read_text())
    assert record["kind"] == "invalid_discard"
    assert record["update_applied"] is False and record["checkpoint_saved"] is False
    assert record["detail"]["discards"][0]["reason"] == "injected LegalityError"
    assert record["detail"]["episodes"] == fixture.EP


# ---------------------------------------------------------------- B3




@needs_torch
def test_ladder_cli_accepts_model_seats(tmp_path):
    """モデル席(checkpoint:)を含む多人数 ladder CLI が exit 0 で出力する。"""
    src, unused = fixture.make_fork(tmp_path, 3)
    out = tmp_path / "ladder.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ladder.py"), "--num-players", "3",
         "--group", "train", "--start-index", "19928010", "--pairs", "1",
         "--workers", "1", "--out", str(out), "--execute",
         "checkpoint:" + src, "heuristic", "random"],
        cwd=str(ROOT), env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-1500:]
    data = json.loads(out.read_text())
    assert data["num_players"] == 3 and len(data["rows"]) == 3
    assert all(row["games"] == 3 for row in data["rows"])
