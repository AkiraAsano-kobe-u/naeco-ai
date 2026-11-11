"""decide_search の p 値(exact 符号検定)と旧 bootstrap 欠陥の退行テスト。"""
import math
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import decide_search  # noqa: E402


def _binom_tail(wins, n):
    """旧実装の意味を保ったまま exact 符号検定の p を直接計算する。"""
    if n == 0:
        return 1.0
    return sum(math.comb(n, k)
               for k in range(wins, n + 1)) / float(1 << n)


def legacy_observed_bootstrap_p(pair_scores, samples=10000, seed=2026072102):
    """Sol が指摘した旧実装: 観測ペアをそのまま bootstrap し平均<=0.5 の割合。"""
    rng = random.Random(seed)
    count = len(pair_scores)
    if count == 0:
        raise ValueError("pairs must be positive")
    total = 0
    for unused in range(samples):
        mean = 0.0
        for unused_inner in range(count):
            mean += pair_scores[rng.randrange(count)]
        if mean / count <= decide_search.NULL:
            total += 1
    return (1 + total) / float(samples + 1)


class ExactSignTest(unittest.TestCase):

    def test_no_difference_input_has_near_nominal_rejection_rate(self):
        """真に差が無い入力で p がほぼ一様、α=0.05 の棄却率が 5% 近傍。"""
        rng = random.Random(42)
        trials = 500
        n = 30
        rejected = 0
        p_sum = 0.0
        for unused in range(trials):
            wins = sum(1 for unused_inner in range(n) if rng.random() < 0.5)
            scores = [1.0] * wins + [0.0] * (n - wins)
            p = decide_search.one_sided_p(scores)
            p_sum += p
            if p <= decide_search.ALPHA:
                rejected += 1
        rate = rejected / float(trials)
        # Binomial(n=500, p≈0.049) で期待 24.5 件。±3.3σ でも 8..41 件。
        self.assertGreaterEqual(rate, 0.02)
        self.assertLessEqual(rate, 0.09)
        mean_p = p_sum / float(trials)
        self.assertGreater(mean_p, 0.40)
        self.assertLess(mean_p, 0.60)

    def test_clear_difference_has_small_p(self):
        """明確な差がある入力で p が小さくなる(片側検定の下側裾)。"""
        scores = [1.0] * 30 + [0.0] * 5 + [0.5] * 5
        p = decide_search.one_sided_p(scores)
        self.assertEqual(p, _binom_tail(30, 35))
        self.assertLess(p, 0.001)

    def test_same_input_same_p_is_deterministic(self):
        """乱数を持たない exact 検定なので同じ入力から常に同じ p。"""
        scores = [1.0, 1.0, 0.5, 0.0, 1.0, 0.5]
        first = decide_search.one_sided_p(scores)
        for unused in range(5):
            self.assertEqual(decide_search.one_sided_p(scores), first)

    def test_draw_pairs_are_excluded_from_sign_denominator(self):
        """引き分け(0.5)ペアは非情報として n から除外する定義。"""
        self.assertEqual(decide_search.one_sided_p([1.0, 1.0]), 0.25)
        self.assertEqual(
            decide_search.one_sided_p([1.0, 1.0] + [0.5] * 100), 0.25)
        self.assertEqual(decide_search.one_sided_p([1.0, 0.5]), 0.5)
        self.assertEqual(decide_search.one_sided_p([0.0, 0.5]), 1.0)
        self.assertEqual(decide_search.one_sided_p([0.5] * 7), 1.0)

    def test_old_observed_bootstrap_is_not_the_null_p_value(self):
        """旧実装(観測 bootstrap)は exact な p と食い違い、強い差で 1e-4 に張り付く。"""
        strong = [1.0] * 90 + [0.0] * 10
        exact = decide_search.one_sided_p(strong)
        old = legacy_observed_bootstrap_p(strong)
        self.assertNotEqual(exact, old)
        # B=10,000 では観測 bootstrap 平均が 0.5 を下回ることは事実上なく、
        # +1/+1 補正により下限 1e-4 に張り付く(Sol 指摘の欠陥)。
        self.assertEqual(old, 1.0 / 10001.0)
        self.assertGreater(old, exact * 1e10)

        # 穏当な差では旧実装は exact p より小さくなりうる(第一種過誤方向)。
        moderate = [1.0, 1.0, 1.0, 1.0, 0.0]
        self.assertNotEqual(decide_search.one_sided_p(moderate),
                            legacy_observed_bootstrap_p(moderate))

    def test_fmt_p_marks_bootstrap_resolution_floor_as_lower_bound(self):
        """1e-4 未満は指数表記で表示され、bootstrap 下限と区別できる。"""
        self.assertEqual(decide_search._fmt_p(1e-5), "1.000e-05")
        self.assertEqual(decide_search._fmt_p(0.0005), "0.00050")
        self.assertEqual(decide_search._fmt_p(0.0), "0")


class SavedEvaluationRecalculation(unittest.TestCase):

    def test_recalculate_saved_phase25_runs(self):
        """保存済み2本の pair_scores で exact p と結論を pin する。"""
        cases = (
            ("phase2.5-stage2f-dev-6000.jsonl", "phase2.5-dev",
             "ismcts", {
                 "pimc-vs-heuristic": 3.297868e-229,
                 "ismcts-vs-heuristic": 7.503089e-217,
                 "direct-two-sided": 3.952003e-33,
             }),
            ("phase2.5-wallclock-train-1100-1299.jsonl",
             "phase2.5-wallclock", "ismcts", {
                 "pimc-vs-heuristic": 3.982730e-59,
                 "ismcts-vs-heuristic": 1.593092e-58,
                 "direct-two-sided": 5.775236e-20,
             }),
        )
        for filename, protocol, expected_choice, expected_p in cases:
            path = os.path.join(ROOT, "docs", "eval-runs", filename)
            if not os.path.exists(path):
                self.skipTest("saved run missing: %s" % filename)
            rows = decide_search.load_records(path, protocol=protocol)
            by_label = dict((row["label"], row) for row in rows)
            for label, expected in expected_p.items():
                if label == "direct-two-sided":
                    scores = by_label["pimc-vs-ismcts"]["pair_scores"]
                    forward = decide_search.one_sided_p(scores)
                    reverse = decide_search.one_sided_p(
                        [1.0 - s for s in scores])
                    got = min(1.0, 2.0 * min(forward, reverse))
                else:
                    got = decide_search.one_sided_p(
                        by_label[label]["pair_scores"])
                self.assertAlmostEqual(
                    math.log10(got), math.log10(expected), places=6)
                self.assertLess(got, 1e-4)
            choice, unused_trace = decide_search.decide(rows)
            self.assertEqual(choice, expected_choice)


if __name__ == "__main__":
    unittest.main()
