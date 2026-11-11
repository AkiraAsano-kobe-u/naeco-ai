"""Fit Phase 9 display calibration from existing gen1300 actor-value rows.

No self-play, seed consumption, engine imports, torch or third-party packages.
The fit/holdout split is fixed by Phase 9 D2, not selected by observed metrics.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from naeco.ai.calibration import ece, fit_platt

FIT = (502800, 503099)
HOLDOUT = (503100, 503199)
_CHAMPION_WEIGHTS = (_REPO_ROOT / "docs" / "eval-runs" / "phase7-champion" /
                     "gen1300.weights.pt")


def collect_pairs(raw):
    """Return actor-perspective, non-pending pairs in the two fixed train bands."""
    summary = raw["summary"]
    if summary["group"] != "train" or \
            Path(summary["meta"]).name != "gen1300.meta.json":
        raise ValueError("input must be the gen1300 train diagnostic")
    fit, holdout, seen = [], [], set()
    for game in raw["games"]:
        index = game["index"]
        if game["group"] != "train" or type(index) is not int or \
                not FIT[0] <= index <= HOLDOUT[1] or index in seen:
            raise ValueError("input contains duplicate or out-of-band train games")
        seen.add(index)
        utilities = game["utilities"]
        if len(utilities) != 2 or any(
                isinstance(u, bool) or not isinstance(u, (int, float)) or
                u not in (-1, 0, 1) for u in utilities):
            raise ValueError("input must contain two terminal utilities")
        target = fit if index <= FIT[1] else holdout
        for row in game["rows"]:
            if type(row["pending"]) is not bool:
                raise ValueError("pending must be a boolean")
            if row["pending"]:
                continue
            actor = row["actor"]
            if type(actor) is not int or actor not in (0, 1):
                raise ValueError("actor must be seat 0 or 1")
            u = row["u%d" % actor]
            if isinstance(u, bool) or not isinstance(u, (int, float)) or \
                    not math.isfinite(u):
                raise ValueError("actor utility prediction must be finite")
            y = (utilities[actor] + 1.0) / 2.0
            # Linear value-head tails can exceed [-1, 1]. Match the D-2
            # adapter and value_calibration.py before fitting in p-space.
            p = max(0.0, min(1.0, (u + 1.0) / 2.0))
            target.append((p, y))
    if seen != set(range(FIT[0], HOLDOUT[1] + 1)) or not fit or not holdout:
        raise ValueError("input must cover train 502800-503199 with both splits")
    return fit, holdout


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(input_path):
    input_bytes = Path(input_path).read_bytes()
    fit, holdout = collect_pairs(json.loads(input_bytes))
    calibrator = fit_platt(fit)
    report = {
        "schema": 1, "kind": "platt", "a": calibrator.a, "b": calibrator.b,
        "input": "actor p=(u+1)/2, pending=false",
        "source": {"file": os.fspath(input_path),
                   "file_sha256": hashlib.sha256(input_bytes).hexdigest(),
                   "group": "train", "fit": list(FIT), "holdout": list(HOLDOUT)},
        "champion_weights_sha256": _sha256_file(_CHAMPION_WEIGHTS),
        "n_fit": len(fit), "n_holdout": len(holdout), "ece": {}, "brier": {},
    }
    for name, pairs in (("fit", fit), ("holdout", holdout)):
        calibrated = [(calibrator.apply(p), y) for p, y in pairs]
        report["ece"][name] = {"before": ece(pairs), "after": ece(calibrated)}
        report["brier"][name] = {
            "before": sum((p - y) ** 2 for p, y in pairs) / len(pairs),
            "after": sum((p - y) ** 2 for p, y in calibrated) / len(calibrated),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit gen1300 display calibration")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if Path(args.input_path).resolve() == Path(args.out).resolve():
            raise ValueError("output must not overwrite the diagnostic input")
        report = build_report(args.input_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) +
                      "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
