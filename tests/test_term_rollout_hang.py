"""W2-C-TERMHANG: rollout 中の SIGTERM で train_ppo が固まらないこと。

原因: グループ TERM で死んだ worker が Pool の inqueue 書込ロックを握ったままだと、
_converge_children の Pool.terminate() が永久にブロックした(stack で確認)。
"""
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch が無い環境")

ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")

_BLOCKED_TERMINATE_SCRIPT = """
import logging, multiprocessing, multiprocessing.pool, os, sys, threading, time
sys.path.insert(0, sys.argv[1])
if __name__ == "__main__":
    import train_ppo
    pool = multiprocessing.get_context("spawn").Pool(processes=2)
    release = threading.Event()
    # inqueue ロック待ちで Pool.terminate() が戻らない状況の再現
    def blocked_terminate(self):
        self._state = multiprocessing.pool.TERMINATE  # 実物も先に状態を変える
        release.wait(300)
    multiprocessing.pool.Pool.terminate = blocked_terminate
    started = time.monotonic()
    remaining = train_ppo._converge_children(logging.getLogger("t"), grace=10.0)
    print("RESULT remaining=%d elapsed=%.1f" % (remaining, time.monotonic() - started),
          flush=True)
    os._exit(0 if remaining == 0 else 7)  # 本番 CLI と同じく atexit を通さない
"""


@needs_torch
def test_converge_children_survives_blocked_pool_terminate():
    """Pool.terminate() が戻らなくても _converge_children は有限時間で戻り子を収束する。"""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_TERMINATE_SCRIPT, str(ROOT / "scripts")],
        env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=60)
    assert proc.returncode == 0, proc.stdout[-1500:]
    assert "RESULT remaining=0" in proc.stdout, proc.stdout[-1500:]


def _members(pgid):
    out = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="],
                         stdout=subprocess.PIPE, universal_newlines=True,
                         check=True).stdout
    return [c[0] for c in (line.split() for line in out.splitlines())
            if len(c) >= 3 and c[1] == str(pgid) and not c[2].startswith("Z")]


@needs_torch
def test_group_sigterm_during_rollout_always_exits():
    """rollout 中の各タイミングで PGID へ TERM を送り、有限時間で exit し子が残らない。"""
    for delay in [0.3, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        with tempfile.TemporaryDirectory() as tmp:
            logp = os.path.join(tmp, "log")
            argv = [sys.executable, str(ROOT / "scripts" / "train_ppo.py"),
                    "--checkpoint-dir", os.path.join(tmp, "ck"),
                    "--episodes", "256", "--generations", "3", "--workers", "6",
                    "--seed", "7", "--hidden", "8", "8", "--action-hidden", "4",
                    "--epochs", "1", "--minibatch-size", "16", "--max-seconds", "0"]
            code = None
            left = []
            with open(logp, "wb") as out:
                child = subprocess.Popen(argv, cwd=tmp, env=ENV, stdout=out,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    limit = time.monotonic() + 120
                    while time.monotonic() < limit:
                        with open(logp, "rb") as fh:
                            if b"episodes=" in fh.read():
                                break
                        time.sleep(0.05)
                    time.sleep(delay)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        with open(logp) as fh:
                            pytest.fail("train_ppo hung after group SIGTERM "
                                        "(delay=%s): %s" % (delay, fh.read()[-1200:]))
                    time.sleep(0.5)
                    left = _members(child.pid)
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            with open(logp) as fh:
                text = fh.read()
            assert left == [], (delay, left, text[-1000:])
            assert code == 0, (delay, code, text[-1000:])
