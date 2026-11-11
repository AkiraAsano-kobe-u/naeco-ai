"""Phase 7: Expert Iteration 蒸留学習(naeco/train/exit.py)のテスト。torch 必須。

歯の狙い:
- build_batch: 可変長合法手を padding し、mask が False の位置は損失に入らない。
  方策ターゲットの無い行(pending / 低予算)は policy_mask=False。
- losses: 方策損失は policy_mask の行だけで計算され、価値損失は全行。
- train: 小さな合成データで方策損失が下がり、保存した checkpoint は
  `checkpoint.load(expected=...)` を通る(A 表キー・エンコーダ hash が整合)。
- 決定論: 同一データ・同一 seed で同一の重み。
"""
import gzip
import json
import os
import tempfile
import unittest

import helpers as H
from naeco.train import encoding as enc

try:
    import torch
except ImportError:      # pragma: no cover
    torch = None

if torch is not None:
    from naeco.train import checkpoint as checkpoint_mod
    from naeco.train import exit as exit_mod
    from naeco.train import net as net_mod


def _rows(n_games=4, seed=0):
    """合成データ: obs / actions は乱数、方策ターゲットは先頭の手に集中。"""
    import random
    rng = random.Random(seed)
    obs_dim, action_dim = enc.obs_dim(2), enc.action_dim()
    rows = []
    for g in range(n_games):
        for d in range(6):
            k = 2 + (d % 3)
            row = {
                "index": g, "decision": d, "seat": d % 2, "round": 1 + d // 2,
                "kind": "search" if d % 3 else "pending", "full": d % 2 == 0,
                "obs": [rng.random() for _ in range(obs_dim)],
                "z": 1.0 if g % 2 == 0 else -1.0,
                "score_self": 30, "score_opp": 25, "score_diff": 5,
            }
            if row["kind"] == "search":
                row["actions"] = [[rng.random() for _ in range(action_dim)]
                                  for _ in range(k)]
                row["visits"] = [5] + [1] * (k - 1)
                row["chosen"] = 0
                row["root_value"] = 0.6
                if row["full"]:
                    row["policy"] = [0.8] + [0.2 / (k - 1)] * (k - 1)
                    row["prior"] = [1.0 / k] * k
                    row["completed_q"] = [0.6] * k
            else:
                row["chosen"] = 0
            rows.append(row)
    return rows


def _write(path, rows):
    header = {"data_version": "exit-1", "obs_dim": enc.obs_dim(2),
              "action_dim": enc.action_dim(), "games": 4}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@unittest.skipUnless(torch is not None, "torch が無い環境では skip")
class ExitTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "d.jsonl.gz")
        _write(self.data, _rows())

    def _net(self):
        torch.manual_seed(1)
        return net_mod.PolicyValueNet(enc.obs_dim(2), enc.action_dim(),
                                      hidden=(32, 32), action_hidden=(16,))

    def test_build_batch_masks_and_targets(self):
        rows = _rows()
        batch = exit_mod.build_batch(rows, enc.action_dim(), "improved", "z", False)
        n = len(rows)
        self.assertEqual(batch["obs"].shape, (n, enc.obs_dim(2)))
        self.assertEqual(batch["actions"].shape[2], enc.action_dim())
        expected_policy = [r["kind"] == "search" and "policy" in r for r in rows]
        self.assertEqual(batch["policy_mask"].tolist(), expected_policy)
        for i, r in enumerate(rows):
            n_legal = len(r.get("actions") or [0])
            self.assertEqual(int(batch["action_mask"][i].sum()), n_legal)
            if expected_policy[i]:
                self.assertAlmostEqual(float(batch["policy_target"][i].sum()), 1.0,
                                       places=5)
                self.assertEqual(float(batch["policy_target"][i, n_legal:].sum()), 0.0)
            self.assertEqual(float(batch["value_target"][i]), r["z"])

    def test_policy_loss_ignores_value_only_rows(self):
        rows = _rows()
        net = self._net()
        only_policy = [r for r in rows if "policy" in r]
        b_all = exit_mod.build_batch(rows, enc.action_dim(), "improved", "z", False)
        b_pol = exit_mod.build_batch(only_policy, enc.action_dim(), "improved", "z",
                                     False)
        _, p_all, _ = exit_mod.losses(net, b_all, 0.5)
        _, p_pol, _ = exit_mod.losses(net, b_pol, 0.5)
        self.assertAlmostEqual(float(p_all), float(p_pol), places=4)

    def test_train_lowers_policy_loss_and_saves_loadable_checkpoint(self):
        dataset = exit_mod.ExitDataset([self.data], enc.obs_dim(2), enc.action_dim(),
                                       holdout_fraction=0.25)
        self.assertTrue(dataset.holdout_rows)
        net = self._net()
        result = exit_mod.train(dataset, net, self.tmp, "exitT", 7, epochs=6,
                                minibatch_size=8, lr=1e-3, log=lambda *a: None)
        hist = result["history"]
        self.assertLess(hist[-1]["policy"], hist[0]["policy"])
        meta = checkpoint_mod.load(
            result["meta_path"],
            expected=checkpoint_mod.expected_meta(2, net.arch_dict()))
        self.assertEqual(meta["hyperparameters"]["algorithm"], "expert_iteration")
        self.assertTrue(os.path.exists(result["meta_path"][:-10] + ".weights.pt"))

    def test_train_is_deterministic(self):
        dataset = exit_mod.ExitDataset([self.data], enc.obs_dim(2), enc.action_dim(),
                                       holdout_fraction=0.25)
        a = self._net()
        b = self._net()
        exit_mod.train(dataset, a, os.path.join(self.tmp, "a"), "x", 3, epochs=2,
                       minibatch_size=8, log=lambda *a: None)
        exit_mod.train(dataset, b, os.path.join(self.tmp, "b"), "x", 3, epochs=2,
                       minibatch_size=8, log=lambda *a: None)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))


if __name__ == "__main__":
    unittest.main()
