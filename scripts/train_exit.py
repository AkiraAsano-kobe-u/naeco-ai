"""Expert Iteration 蒸留学習の CLI(torch は `naeco/train/exit.py` 経由でだけ触る)。

例:
  python scripts/train_exit.py --data data1.jsonl.gz --init gen532.meta.json \
      --out runs/phase7 --name exit001 --epochs 4 --lr 1e-4 --execute
"""
import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _entry in (_SCRIPT_DIR, _REPO_ROOT):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import argparse
import json
import time

from naeco.train import encoding as enc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Expert Iteration 蒸留学習")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--init", required=True, help="初期重みのメタ(checkpoint)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-target", choices=("improved", "visits"),
                        default="improved")
    parser.add_argument("--value-mode", choices=("z", "mix"), default="z")
    parser.add_argument("--surprise-weighting", action="store_true")
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("prepared: data=%s init=%s out=%s/%s" % (
            args.data, args.init, args.out, args.name))
        return 0
    from naeco.train import exit as exit_mod
    started = time.monotonic()
    net = exit_mod.load_net(args.init)
    dataset = exit_mod.ExitDataset(
        args.data, enc.obs_dim(2), enc.action_dim(),
        policy_target=args.policy_target, holdout_fraction=args.holdout)
    print("train:", json.dumps(dataset.stats(dataset.train_rows), sort_keys=True))
    print("holdout:", json.dumps(dataset.stats(dataset.holdout_rows), sort_keys=True))
    before = exit_mod.evaluate(dataset.holdout_rows, net, args.policy_target,
                               args.value_mode, args.value_coef)
    print("init holdout:", json.dumps(before, sort_keys=True))
    result = exit_mod.train(
        dataset, net, args.out, args.name, args.seed, epochs=args.epochs,
        minibatch_size=args.minibatch_size, lr=args.lr,
        value_coef=args.value_coef, max_grad_norm=args.max_grad_norm,
        value_mode=args.value_mode, surprise_weighting=args.surprise_weighting,
        generation=args.generation,
        extra_hyper={"init": args.init, "data": list(args.data),
                     "init_holdout": before})
    print("saved:", result["meta_path"], "(%.0fs)" % (time.monotonic() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
