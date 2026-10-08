"""Check orientation, absent classes and large counts in offline reports."""

import unittest

import numpy as np

from experiments.wavedsp.confusion_matrix import matrix_report


class ConfusionMatrixTests(unittest.TestCase):
    def test_asymmetric_counts_and_one_vs_rest(self) -> None:
        report = matrix_report({"confusion_matrix": [
            [5, 2, 0, 1], [3, 7, 4, 0], [0, 1, 6, 2], [4, 0, 3, 9],
        ], "count": 47})
        self.assertEqual(report["one_vs_rest"]["ghost"], {
            "tp": 9, "fn": 7, "fp": 3, "tn": 28,
            "confusion_matrix": [[28, 3], [7, 9]],
        })
        self.assertAlmostEqual(report["row_normalized"][3][3], 9 / 16)
        self.assertAlmostEqual(report["row_normalized"][0][3], 1 / 8)

    def test_large_counts_and_zero_support(self) -> None:
        matrix = [[43_199_125_215, 2, 0, 0], [0, 0, 0, 0],
                  [0, 0, 0, 0], [0, 0, 0, 1]]
        report = matrix_report({"confusion_matrix": matrix})
        self.assertEqual(report["count"], 43_199_125_218)
        self.assertEqual(report["one_vs_rest"]["noise"]["tp"], 43_199_125_215)
        self.assertEqual(report["row_normalized"][1], [0, 0, 0, 0])
        self.assertTrue(np.isfinite(report["row_normalized"]).all())
        empty = matrix_report({"confusion_matrix": np.zeros((4, 4), dtype=int).tolist()})
        self.assertEqual(empty["one_vs_rest"]["ghost"]["confusion_matrix"], [[0, 0], [0, 0]])

    def test_reject_inconsistent_source(self) -> None:
        matrix = np.eye(4, dtype=int).tolist()
        with self.assertRaisesRegex(ValueError, "count"):
            matrix_report({"confusion_matrix": matrix, "count": 99})
        with self.assertRaisesRegex(ValueError, "support"):
            matrix_report({"confusion_matrix": matrix,
                           "per_class": {"noise": {"support": 99}}})
        with self.assertRaisesRegex(ValueError, "integer"):
            matrix_report({"confusion_matrix": np.eye(4).tolist()})


if __name__ == "__main__":
    unittest.main()
