"""Phase 7: `scripts/ladder.py` の判定用来歴(Sol ゲート 2026-09-05 の指摘)のテスト。

歯の狙い:
- 既存の出力ファイルがあれば `--execute` の有無にかかわらず SystemExit で拒否する。
- `freeze_models` は checkpoint / search 指定のモデルを meta と weights の SHA-256 で凍結し、
  `verify_frozen` はどちらか一方でも変わると RuntimeError(fail-closed)。
- `provenance` は HEAD・台帳 SHA・条件・formal(group=="dev" のときだけ True)を持つ。
- `parse_spec` は search/gumbel の `@budget` を読む。
"""
import hashlib
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import ladder  # noqa: E402


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


class LadderProvenanceTests(unittest.TestCase):

    def test_parse_spec_budgets(self):
        self.assertEqual(ladder.parse_spec("search:x.meta.json@600"),
                         ("search", "x.meta.json", 600))
        self.assertEqual(ladder.parse_spec("gumbel:x.meta.json"),
                         ("gumbel", "x.meta.json", ladder.DEFAULT_BUDGET))
        self.assertEqual(ladder.parse_spec("checkpoint:x.meta.json"),
                         ("checkpoint", "x.meta.json", 0))
        with self.assertRaises(ValueError):
            ladder.parse_spec("heuristic:foo")

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.json")
            _write(out, b"{}")
            with self.assertRaises(SystemExit):
                ladder.main(["random", "heuristic", "--start-index", "0",
                             "--pairs", "1", "--out", out])

    def test_freeze_and_verify_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = os.path.join(tmp, "g.meta.json")
            weights = os.path.join(tmp, "g.weights.pt")
            _write(meta, b"{}")
            _write(weights, b"w1")
            spec = "search:%s@10" % meta
            frozen = ladder.freeze_models([spec, "random"])
            self.assertEqual(list(frozen), [spec])
            self.assertEqual(frozen[spec]["weights_sha256"],
                             hashlib.sha256(b"w1").hexdigest())
            ladder.verify_frozen(frozen)          # 変更なしなら通る
            _write(weights, b"w2")
            with self.assertRaises(RuntimeError):
                ladder.verify_frozen(frozen)
            _write(weights, b"w1")
            ladder.verify_frozen(frozen)
            _write(meta, b'{"x": 1}')
            with self.assertRaises(RuntimeError):
                ladder.verify_frozen(frozen)

    def test_provenance_fields_and_formal_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.md")
            _write(ledger, b"ledger")
            prov = ladder.provenance(["random", "heuristic"], "dev", 100, 5, 2,
                                     ledger_path=ledger)
            self.assertTrue(prov["formal"])
            self.assertEqual(prov["seed_ledger"]["sha256"],
                             hashlib.sha256(b"ledger").hexdigest())
            self.assertEqual(prov["conditions"]["end_index"], 104)
            self.assertIn("git_commit", prov)
            self.assertIn("started_at", prov)
            self.assertEqual(prov["frozen_models"], {})
            self.assertFalse(
                ladder.provenance(["random", "heuristic"], "train", 0, 1, 1,
                                  ledger_path=ledger)["formal"])
            self.assertTrue(
                ladder.provenance(["random", "heuristic"], "accept", 0, 1, 1,
                                  ledger_path=ledger)["formal"])


if __name__ == "__main__":
    unittest.main()
