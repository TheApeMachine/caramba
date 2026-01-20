from __future__ import annotations

import unittest

from benchmark.stats import mcnemar_exact_pvalue, paired_bootstrap_delta_ci, wilson_ci


class TestStats(unittest.TestCase):
    def test_wilson_ci_bounds(self) -> None:
        ci = wilson_ci(50, 100)
        self.assertGreaterEqual(ci.low, 0.0)
        self.assertLessEqual(ci.high, 1.0)
        self.assertLess(ci.low, ci.high)

    def test_paired_bootstrap_delta_ci_zero(self) -> None:
        a = [1.0, 0.0, 1.0, 1.0]
        b = [1.0, 0.0, 1.0, 1.0]
        ci = paired_bootstrap_delta_ci(a, b, n_boot=200, seed=0)
        self.assertAlmostEqual(ci.delta, 0.0, places=12)
        self.assertAlmostEqual(ci.low, 0.0, places=12)
        self.assertAlmostEqual(ci.high, 0.0, places=12)

    def test_mcnemar_exact_pvalue_symmetry(self) -> None:
        p1 = mcnemar_exact_pvalue(10, 2)
        p2 = mcnemar_exact_pvalue(2, 10)
        self.assertAlmostEqual(p1, p2, places=12)


if __name__ == "__main__":
    unittest.main()

