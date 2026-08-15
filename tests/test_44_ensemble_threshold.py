"""第 44 步 Ensemble 阈值分析单元测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "44_analyze_ensemble_threshold.py"


def load_step44_module():
    spec = importlib.util.spec_from_file_location("step44", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step44"] = module
    spec.loader.exec_module(module)
    return module


step44 = load_step44_module()


class TestEnsembleThresholdAnalysis(unittest.TestCase):
    def test_exact_max_f1_matches_brute_force(self) -> None:
        rng = np.random.default_rng(42)
        probabilities = rng.uniform(0.0, 1.0, size=200)
        clicks = (probabilities + rng.normal(0, 0.15, size=200) > 0.55).astype(np.int8)

        curve_df = step44.compute_exact_threshold_curve(probabilities, clicks)
        selected = step44.select_max_f1_threshold(curve_df)
        brute = step44.brute_force_max_f1_threshold(probabilities, clicks)

        self.assertAlmostEqual(selected, brute, places=12)

    def test_same_probability_samples_are_not_split(self) -> None:
        probabilities = np.array([0.8, 0.8, 0.8, 0.2, 0.2], dtype=np.float64)
        clicks = np.array([1, 0, 1, 0, 0], dtype=np.int8)

        curve_df = step44.compute_exact_threshold_curve(probabilities, clicks)
        row_08 = curve_df.loc[np.isclose(curve_df["threshold"], 0.8)].iloc[0]

        self.assertEqual(int(row_08["predicted_positive_rows"]), 3)
        self.assertEqual(int(row_08["tp"]), 2)
        self.assertEqual(int(row_08["fp"]), 1)

        intermediate_thresholds = curve_df["threshold"].to_numpy()
        self.assertNotIn(0.8, set(np.unique(probabilities)) - set(intermediate_thresholds))

    def test_threshold_0_5_confusion_matrix(self) -> None:
        probabilities = np.array([0.9, 0.6, 0.4, 0.1], dtype=np.float64)
        clicks = np.array([1, 1, 0, 0], dtype=np.int8)

        point = step44.compute_operating_point(
            probabilities,
            clicks,
            0.5,
            "reference_0_5",
            "validation",
        )

        self.assertEqual(point.tp, 2)
        self.assertEqual(point.fp, 0)
        self.assertEqual(point.tn, 2)
        self.assertEqual(point.fn, 0)
        self.assertAlmostEqual(point.precision, 1.0)
        self.assertAlmostEqual(point.recall, 1.0)
        self.assertAlmostEqual(point.f1, 1.0)

    def test_invalid_probabilities_are_rejected(self) -> None:
        clicks = np.array([0, 1, 0], dtype=np.int8)

        with self.assertRaises(ValueError):
            step44.validate_probabilities(np.array([0.1, np.nan, 0.3]), "validation")
        with self.assertRaises(ValueError):
            step44.validate_probabilities(np.array([0.1, np.inf, 0.3]), "validation")
        with self.assertRaises(ValueError):
            step44.compute_operating_point(
                np.array([0.1, 1.2, 0.3]),
                clicks,
                0.5,
                "reference_0_5",
                "validation",
            )

    def test_curve_contains_exact_optimal_threshold(self) -> None:
        probabilities = np.array([0.9, 0.7, 0.7, 0.3, 0.1], dtype=np.float64)
        clicks = np.array([1, 1, 0, 0, 0], dtype=np.int8)

        curve_df, thresholds = step44.select_thresholds_from_validation(probabilities, clicks)
        step44.ensure_curve_contains_thresholds(
            curve_df,
            [thresholds.development_max_f1, thresholds.development_max_youden_j],
        )

    def test_holdout_not_used_for_threshold_selection(self) -> None:
        rng = np.random.default_rng(7)
        valid_probs = rng.uniform(0.0, 1.0, size=50)
        valid_clicks = (valid_probs > 0.55).astype(np.int8)

        holdout_probs = np.ones(50, dtype=np.float64)
        holdout_clicks = np.zeros(50, dtype=np.int8)

        curve_df, thresholds = step44.select_thresholds_from_validation(valid_probs, valid_clicks)
        selection_without_holdout = thresholds.development_max_f1

        holdout_only_curve = step44.compute_exact_threshold_curve(holdout_probs, holdout_clicks)
        holdout_only_threshold = step44.select_max_f1_threshold(holdout_only_curve)

        self.assertNotEqual(selection_without_holdout, holdout_only_threshold)
        self.assertEqual(selection_without_holdout, step44.select_max_f1_threshold(curve_df))

    def test_weight_mismatch_raises(self) -> None:
        step42_meta = {
            "validation_passed": True,
            "holdout_used": False,
            "best_lightgbm_weight": 0.588,
            "best_deepfm_weight": 0.412,
        }
        step43_meta = {
            "one_shot_evaluation": True,
            "ensemble_lightgbm_weight": 0.500,
            "ensemble_deepfm_weight": 0.500,
            "holdout_used_for_training": False,
            "holdout_used_for_preprocessing_fit": False,
            "holdout_used_for_model_selection": False,
            "holdout_used_for_weight_selection": False,
        }

        with self.assertRaises(ValueError):
            step44.verify_upstream_metadata(step42_meta, step43_meta)


class TestThresholdSelectionTieBreak(unittest.TestCase):
    def test_f1_tie_break_prefers_precision_then_higher_threshold(self) -> None:
        probabilities = np.array([0.9, 0.8, 0.5, 0.5, 0.1], dtype=np.float64)
        clicks = np.array([1, 0, 1, 0, 0], dtype=np.int8)

        curve_df = step44.compute_exact_threshold_curve(probabilities, clicks)
        threshold = step44.select_max_f1_threshold(curve_df)

        self.assertAlmostEqual(threshold, 0.9)


if __name__ == "__main__":
    unittest.main()
