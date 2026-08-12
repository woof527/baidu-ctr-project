"""
百度 CTR 项目 — LightGBM + DeepFM Validation Ensemble（第 42 步）

功能：
    在 unified valid 500K 上搜索 LightGBM 与 DeepFM 的线性 blending 权重。
    主优化指标：LogLoss。禁止读取 holdout / test.csv。

输入：
    outputs/predictions/unified_lightgbm_valid_predictions.parquet
    outputs/predictions/deepfm_valid_predictions.parquet

用法：
    python scripts/42_ensemble_lightgbm_deepfm.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

LIGHTGBM_PREDICTIONS_PATH = Path("outputs/predictions/unified_lightgbm_valid_predictions.parquet")
DEEPFM_PREDICTIONS_PATH = Path("outputs/predictions/deepfm_valid_predictions.parquet")

WEIGHT_SEARCH_PATH = Path("outputs/ensemble_weight_search.csv")
ENSEMBLE_PREDICTIONS_PATH = Path("outputs/predictions/lightgbm_deepfm_ensemble_valid_predictions.parquet")
METRICS_PATH = Path("outputs/lightgbm_deepfm_ensemble_metrics.json")
METADATA_PATH = Path("outputs/lightgbm_deepfm_ensemble_metadata.json")

FORMAL_VALID_ROWS = 500_000
PROB_CLIP_EPS = 1e-15

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

COARSE_WEIGHT_STEP = 0.01
FINE_WEIGHT_STEP = 0.001
FINE_WEIGHT_WINDOW = 0.02

LIGHTGBM_PRED_CANDIDATES = (
    "lgbm_unified_pred",
    "lightgbm_pred",
    "lightgbm_unified_pred",
    "pred",
    "prediction",
)
DEEPFM_PRED_CANDIDATES = (
    "deepfm_pred",
    "pred",
    "prediction",
)
CLICK_CANDIDATES = ("click", "label", "y")


@dataclass
class ModelMetrics:
    auc: float
    logloss: float
    brier: float
    average_precision: float
    actual_ctr: float
    mean_predicted_ctr: float
    calibration_gap: float


@dataclass
class WeightSearchRow:
    lightgbm_weight: float
    deepfm_weight: float
    auc: float
    logloss: float
    brier: float
    average_precision: float
    mean_predicted_ctr: float
    calibration_gap: float
    search_phase: str


@dataclass
class WeightSearchAudit:
    coarse_grid: str
    coarse_grid_step: float
    coarse_grid_count: int
    coarse_best_alpha: float
    coarse_best_logloss: float
    fine_grid_range: list[float]
    fine_grid_step: float
    fine_grid_count: int
    fine_best_alpha: float
    fine_best_logloss: float
    final_selected_alpha: float
    final_selected_deepfm_alpha: float
    selection_metric: str
    weight_search_reproducible: bool


def assert_safe_path(path: Path) -> None:
    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def compute_click_checksum(clicks: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(clicks.astype(np.int8).tobytes())
    return hasher.hexdigest()


def detect_column(columns: list[str], candidates: tuple[str, ...], role: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    suffix_matches = [
        col
        for col in columns
        if any(col.endswith(suffix) for suffix in ("_pred", "_prediction", "_prob"))
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise ValueError(f"无法识别 {role} 列。可用列：{columns}")


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> ModelMetrics:
    clipped = clip_probabilities(probabilities)
    actual_ctr = float(y_true.mean())
    mean_predicted_ctr = float(probabilities.mean())
    return ModelMetrics(
        auc=float(roc_auc_score(y_true, probabilities)),
        logloss=float(log_loss(y_true, clipped, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        actual_ctr=actual_ctr,
        mean_predicted_ctr=mean_predicted_ctr,
        calibration_gap=abs(mean_predicted_ctr - actual_ctr),
    )


def metrics_to_dict(metrics: ModelMetrics) -> dict[str, float]:
    return {
        "auc": metrics.auc,
        "logloss": metrics.logloss,
        "brier": metrics.brier,
        "average_precision": metrics.average_precision,
        "actual_ctr": metrics.actual_ctr,
        "mean_predicted_ctr": metrics.mean_predicted_ctr,
        "calibration_gap": metrics.calibration_gap,
    }


def validate_probabilities(name: str, probabilities: np.ndarray) -> None:
    if np.isnan(probabilities).any():
        raise ValueError(f"{name} 存在 NaN。")
    if np.isinf(probabilities).any():
        raise ValueError(f"{name} 存在 inf。")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError(f"{name} 存在超出 [0, 1] 的概率。")


def validate_clicks(clicks: np.ndarray) -> None:
    if not np.isin(clicks, [0, 1]).all():
        raise ValueError("click 列存在非 0/1 值。")


def load_and_align_predictions() -> tuple[pd.DataFrame, str, str, bool]:
    assert_safe_path(LIGHTGBM_PREDICTIONS_PATH)
    assert_safe_path(DEEPFM_PREDICTIONS_PATH)

    lightgbm_df = pd.read_parquet(LIGHTGBM_PREDICTIONS_PATH)
    deepfm_df = pd.read_parquet(DEEPFM_PREDICTIONS_PATH)

    if len(lightgbm_df) != FORMAL_VALID_ROWS:
        raise ValueError(
            f"LightGBM 行数 {len(lightgbm_df):,} != {FORMAL_VALID_ROWS:,}"
        )
    if len(deepfm_df) != FORMAL_VALID_ROWS:
        raise ValueError(
            f"DeepFM 行数 {len(deepfm_df):,} != {FORMAL_VALID_ROWS:,}"
        )

    click_col_lgbm = detect_column(list(lightgbm_df.columns), CLICK_CANDIDATES, "LightGBM click")
    click_col_dfm = detect_column(list(deepfm_df.columns), CLICK_CANDIDATES, "DeepFM click")
    lgbm_pred_col = detect_column(list(lightgbm_df.columns), LIGHTGBM_PRED_CANDIDATES, "LightGBM prediction")
    deepfm_pred_col = detect_column(list(deepfm_df.columns), DEEPFM_PRED_CANDIDATES, "DeepFM prediction")

    clicks_lgbm = lightgbm_df[click_col_lgbm].to_numpy(dtype=np.int8)
    clicks_dfm = deepfm_df[click_col_dfm].to_numpy(dtype=np.int8)
    validate_clicks(clicks_lgbm)
    validate_clicks(clicks_dfm)

    checksum_lgbm = compute_click_checksum(clicks_lgbm)
    checksum_dfm = compute_click_checksum(clicks_dfm)
    click_sequence_match = checksum_lgbm == checksum_dfm
    if not click_sequence_match:
        raise RuntimeError(
            "LightGBM 与 DeepFM 的 click 序列不一致，禁止强行对齐。"
            f" lgbm={checksum_lgbm}, deepfm={checksum_dfm}"
        )

    id_sequence_match = True
    if "id" in lightgbm_df.columns and "id" in deepfm_df.columns:
        id_sequence_match = lightgbm_df["id"].astype(str).equals(deepfm_df["id"].astype(str))
        if not id_sequence_match:
            raise RuntimeError("LightGBM 与 DeepFM 的 id 序列不一致。")
    else:
        print(
            "DeepFM 预测文件无 id 列；依赖 Step 38/39/41 保持的 unified valid row order，"
            "已通过 click sequence exact match 确认对齐。"
        )

    lightgbm_pred = lightgbm_df[lgbm_pred_col].to_numpy(dtype=np.float64)
    deepfm_pred = deepfm_df[deepfm_pred_col].to_numpy(dtype=np.float64)
    validate_probabilities("LightGBM prediction", lightgbm_pred)
    validate_probabilities("DeepFM prediction", deepfm_pred)

    aligned = pd.DataFrame(
        {
            "click": clicks_lgbm.astype(np.int8),
            "lightgbm_pred": lightgbm_pred,
            "deepfm_pred": deepfm_pred,
        }
    )
    if "id" in lightgbm_df.columns:
        aligned.insert(0, "id", lightgbm_df["id"].astype(str).to_numpy())

    return aligned, lgbm_pred_col, deepfm_pred_col, click_sequence_match and id_sequence_match


def evaluate_weight(
    y_true: np.ndarray,
    lightgbm_pred: np.ndarray,
    deepfm_pred: np.ndarray,
    lightgbm_weight: float,
    search_phase: str,
) -> WeightSearchRow:
    lightgbm_weight = float(np.clip(lightgbm_weight, 0.0, 1.0))
    deepfm_weight = 1.0 - lightgbm_weight
    ensemble_pred = lightgbm_weight * lightgbm_pred + deepfm_weight * deepfm_pred
    validate_probabilities("ensemble prediction", ensemble_pred)
    metrics = compute_metrics(y_true, ensemble_pred)
    return WeightSearchRow(
        lightgbm_weight=lightgbm_weight,
        deepfm_weight=deepfm_weight,
        auc=metrics.auc,
        logloss=metrics.logloss,
        brier=metrics.brier,
        average_precision=metrics.average_precision,
        mean_predicted_ctr=metrics.mean_predicted_ctr,
        calibration_gap=metrics.calibration_gap,
        search_phase=search_phase,
    )


def is_better_candidate(candidate: WeightSearchRow, current: WeightSearchRow) -> bool:
    if candidate.logloss < current.logloss - 1e-12:
        return True
    if abs(candidate.logloss - current.logloss) <= 1e-12:
        if candidate.auc > current.auc + 1e-12:
            return True
        if abs(candidate.auc - current.auc) <= 1e-12 and candidate.calibration_gap < current.calibration_gap - 1e-12:
            return True
    return False


def pick_best_row(rows: list[WeightSearchRow]) -> WeightSearchRow:
    best_row = rows[0]
    for row in rows[1:]:
        if is_better_candidate(row, best_row):
            best_row = row
    return best_row


def build_coarse_weights() -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + COARSE_WEIGHT_STEP / 2, COARSE_WEIGHT_STEP), 2)


def build_fine_weights(coarse_best_alpha: float) -> np.ndarray:
    fine_start = max(0.0, coarse_best_alpha - FINE_WEIGHT_WINDOW)
    fine_end = min(1.0, coarse_best_alpha + FINE_WEIGHT_WINDOW)
    return np.arange(fine_start, fine_end + FINE_WEIGHT_STEP / 2, FINE_WEIGHT_STEP)


def search_weights(
    y_true: np.ndarray,
    lightgbm_pred: np.ndarray,
    deepfm_pred: np.ndarray,
) -> tuple[list[WeightSearchRow], WeightSearchRow, WeightSearchAudit]:
    coarse_weights = build_coarse_weights()
    coarse_rows = [
        evaluate_weight(y_true, lightgbm_pred, deepfm_pred, weight, "coarse")
        for weight in coarse_weights
    ]
    coarse_best = pick_best_row(coarse_rows)

    fine_weights = build_fine_weights(coarse_best.lightgbm_weight)
    fine_rows = [
        evaluate_weight(y_true, lightgbm_pred, deepfm_pred, weight, "fine")
        for weight in fine_weights
    ]
    fine_best = pick_best_row(fine_rows)

    all_rows = coarse_rows + fine_rows
    best_row = pick_best_row(all_rows)

    matched_rows = [
        row
        for row in all_rows
        if abs(row.lightgbm_weight - best_row.lightgbm_weight) <= 1e-12
    ]
    weight_search_reproducible = len(matched_rows) > 0

    audit = WeightSearchAudit(
        coarse_grid=f"0.00:{COARSE_WEIGHT_STEP:.2f}:1.00",
        coarse_grid_step=COARSE_WEIGHT_STEP,
        coarse_grid_count=len(coarse_weights),
        coarse_best_alpha=coarse_best.lightgbm_weight,
        coarse_best_logloss=coarse_best.logloss,
        fine_grid_range=[float(fine_weights.min()), float(fine_weights.max())],
        fine_grid_step=FINE_WEIGHT_STEP,
        fine_grid_count=len(fine_weights),
        fine_best_alpha=fine_best.lightgbm_weight,
        fine_best_logloss=fine_best.logloss,
        final_selected_alpha=best_row.lightgbm_weight,
        final_selected_deepfm_alpha=best_row.deepfm_weight,
        selection_metric="LogLoss",
        weight_search_reproducible=weight_search_reproducible,
    )
    return all_rows, best_row, audit


def save_weight_search(rows: list[WeightSearchRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "lightgbm_weight": row.lightgbm_weight,
                "deepfm_weight": row.deepfm_weight,
                "auc": row.auc,
                "logloss": row.logloss,
                "brier": row.brier,
                "average_precision": row.average_precision,
                "mean_predicted_ctr": row.mean_predicted_ctr,
                "calibration_gap": row.calibration_gap,
                "search_phase": row.search_phase,
            }
            for row in rows
        ]
    ).to_csv(path, index=False)


def audit_to_dict(audit: WeightSearchAudit) -> dict[str, Any]:
    return {
        "coarse_grid": audit.coarse_grid,
        "coarse_grid_step": audit.coarse_grid_step,
        "coarse_grid_count": audit.coarse_grid_count,
        "coarse_best_alpha": audit.coarse_best_alpha,
        "coarse_best_logloss": audit.coarse_best_logloss,
        "fine_grid_range": audit.fine_grid_range,
        "fine_grid_step": audit.fine_grid_step,
        "fine_grid_count": audit.fine_grid_count,
        "fine_best_alpha": audit.fine_best_alpha,
        "fine_best_logloss": audit.fine_best_logloss,
        "final_selected_alpha": audit.final_selected_alpha,
        "final_selected_deepfm_alpha": audit.final_selected_deepfm_alpha,
        "selection_metric": audit.selection_metric,
        "weight_search_reproducible": audit.weight_search_reproducible,
    }


def print_weight_search_audit(audit: WeightSearchAudit) -> None:
    print("\n--- Weight Search Audit ---")
    print(f"coarse_grid = {audit.coarse_grid}")
    print(f"coarse_grid_step = {audit.coarse_grid_step}")
    print(f"coarse_grid_count = {audit.coarse_grid_count}")
    print(f"coarse_best_alpha = {audit.coarse_best_alpha:.6f}")
    print(f"coarse_best_logloss = {audit.coarse_best_logloss:.12f}")
    print(f"fine_grid_range = [{audit.fine_grid_range[0]:.6f}, {audit.fine_grid_range[1]:.6f}]")
    print(f"fine_grid_step = {audit.fine_grid_step}")
    print(f"fine_grid_count = {audit.fine_grid_count}")
    print(f"fine_best_alpha = {audit.fine_best_alpha:.6f}")
    print(f"fine_best_logloss = {audit.fine_best_logloss:.12f}")
    print(f"final_selected_alpha = {audit.final_selected_alpha:.6f}")
    print(f"final_selected_deepfm_alpha = {audit.final_selected_deepfm_alpha:.6f}")
    print()
    print(f"WEIGHT_SEARCH_REPRODUCIBLE = {audit.weight_search_reproducible}")
    print(f"FINAL_LIGHTGBM_WEIGHT = {audit.final_selected_alpha:.3f}")
    print(f"FINAL_DEEPFM_WEIGHT = {audit.final_selected_deepfm_alpha:.3f}")
    print("HOLDOUT_USED = False")


def print_final_summary(
    valid_rows: int,
    lightgbm_metrics: ModelMetrics,
    deepfm_metrics: ModelMetrics,
    equal_metrics: ModelMetrics,
    best_row: WeightSearchRow,
    best_metrics: ModelMetrics,
    click_sequence_match: bool,
) -> None:
    print("\n" + "=" * 40)
    print("LIGHTGBM + DEEPFM ENSEMBLE SUMMARY")
    print("=" * 40)
    print(f"VALID_ROWS = {valid_rows}")
    print()
    print("LIGHTGBM")
    print(f"AUC = {lightgbm_metrics.auc:.6f}")
    print(f"LOGLOSS = {lightgbm_metrics.logloss:.6f}")
    print(f"BRIER = {lightgbm_metrics.brier:.6f}")
    print()
    print("DEEPFM")
    print(f"AUC = {deepfm_metrics.auc:.6f}")
    print(f"LOGLOSS = {deepfm_metrics.logloss:.6f}")
    print(f"BRIER = {deepfm_metrics.brier:.6f}")
    print()
    print("50_50_ENSEMBLE")
    print(f"AUC = {equal_metrics.auc:.6f}")
    print(f"LOGLOSS = {equal_metrics.logloss:.6f}")
    print(f"BRIER = {equal_metrics.brier:.6f}")
    print()
    print(f"BEST_LIGHTGBM_WEIGHT = {best_row.lightgbm_weight:.6f}")
    print(f"BEST_DEEPFM_WEIGHT = {best_row.deepfm_weight:.6f}")
    print()
    print(f"BEST_ENSEMBLE_AUC = {best_metrics.auc:.6f}")
    print(f"BEST_ENSEMBLE_LOGLOSS = {best_metrics.logloss:.6f}")
    print(f"BEST_ENSEMBLE_BRIER = {best_metrics.brier:.6f}")
    print(f"BEST_ENSEMBLE_AVERAGE_PRECISION = {best_metrics.average_precision:.6f}")
    print()
    print(f"ACTUAL_CTR = {best_metrics.actual_ctr:.6f}")
    print(f"MEAN_PREDICTED_CTR = {best_metrics.mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {best_metrics.calibration_gap:.6f}")
    print()
    print(f"AUC_DIFF_VS_LIGHTGBM = {best_metrics.auc - lightgbm_metrics.auc:+.6f}")
    print(f"LOGLOSS_DIFF_VS_LIGHTGBM = {best_metrics.logloss - lightgbm_metrics.logloss:+.6f}")
    print(f"BRIER_DIFF_VS_LIGHTGBM = {best_metrics.brier - lightgbm_metrics.brier:+.6f}")
    print()
    print(f"AUC_DIFF_VS_DEEPFM = {best_metrics.auc - deepfm_metrics.auc:+.6f}")
    print(f"LOGLOSS_DIFF_VS_DEEPFM = {best_metrics.logloss - deepfm_metrics.logloss:+.6f}")
    print(f"BRIER_DIFF_VS_DEEPFM = {best_metrics.brier - deepfm_metrics.brier:+.6f}")
    print()
    print(f"CLICK_SEQUENCE_MATCH = {click_sequence_match}")
    print("HOLDOUT_USED = False")
    print("VALIDATION_PASSED = True")
    print("=" * 40)
    print()
    print(
        "注意：该 ensemble 权重是在当前 development validation set 上选择的，"
        "属于 development model selection，不是最终泛化性能。"
        "最终模型冻结后，应使用 untouched holdout 做一次最终评估。"
    )


def main() -> None:
    print("=" * 72)
    print("百度 CTR 项目 — 第 42 步 LightGBM + DeepFM Ensemble")
    print(f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")

    aligned_df, lgbm_pred_col, deepfm_pred_col, click_sequence_match = load_and_align_predictions()
    print(f"\nLightGBM prediction column: {lgbm_pred_col}")
    print(f"DeepFM prediction column: {deepfm_pred_col}")
    print(f"VALID_ROWS = {len(aligned_df):,}")
    print(f"CLICK_SEQUENCE_MATCH = {click_sequence_match}")

    y_true = aligned_df["click"].to_numpy(dtype=np.int8)
    lightgbm_pred = aligned_df["lightgbm_pred"].to_numpy(dtype=np.float64)
    deepfm_pred = aligned_df["deepfm_pred"].to_numpy(dtype=np.float64)

    lightgbm_metrics = compute_metrics(y_true, lightgbm_pred)
    deepfm_metrics = compute_metrics(y_true, deepfm_pred)
    equal_weight_row = evaluate_weight(y_true, lightgbm_pred, deepfm_pred, 0.5, "reference")
    equal_metrics = ModelMetrics(
        auc=equal_weight_row.auc,
        logloss=equal_weight_row.logloss,
        brier=equal_weight_row.brier,
        average_precision=equal_weight_row.average_precision,
        actual_ctr=lightgbm_metrics.actual_ctr,
        mean_predicted_ctr=equal_weight_row.mean_predicted_ctr,
        calibration_gap=equal_weight_row.calibration_gap,
    )

    search_rows, best_row, search_audit = search_weights(y_true, lightgbm_pred, deepfm_pred)
    save_weight_search(search_rows, WEIGHT_SEARCH_PATH)
    print_weight_search_audit(search_audit)

    best_ensemble_pred = (
        best_row.lightgbm_weight * lightgbm_pred + best_row.deepfm_weight * deepfm_pred
    )
    best_metrics = compute_metrics(y_true, best_ensemble_pred)

    output_df = pd.DataFrame(
        {
            "click": y_true.astype(np.int8),
            "lightgbm_pred": lightgbm_pred.astype(np.float64),
            "deepfm_pred": deepfm_pred.astype(np.float64),
            "ensemble_pred": best_ensemble_pred.astype(np.float64),
        }
    )
    if "id" in aligned_df.columns:
        output_df.insert(0, "id", aligned_df["id"].astype(str).to_numpy())

    ENSEMBLE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(ENSEMBLE_PREDICTIONS_PATH, index=False)

    metrics_out = {
        "model": "LightGBM_DeepFM_Ensemble",
        "validation_rows": len(aligned_df),
        "selection_metric": "LogLoss",
        "best_lightgbm_weight": best_row.lightgbm_weight,
        "best_deepfm_weight": best_row.deepfm_weight,
        "weight_search_audit": audit_to_dict(search_audit),
        "best_ensemble": metrics_to_dict(best_metrics),
        "lightgbm_only": metrics_to_dict(lightgbm_metrics),
        "deepfm_only": metrics_to_dict(deepfm_metrics),
        "equal_weight_ensemble": metrics_to_dict(equal_metrics),
        "click_sequence_match": click_sequence_match,
        "holdout_used": False,
        "development_model_selection_note": (
            "Ensemble weight selected on unified valid only; "
            "not final generalization performance."
        ),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata_out = {
        "script_name": "scripts/42_ensemble_lightgbm_deepfm.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "LightGBM_DeepFM_Ensemble",
        "validation_rows": len(aligned_df),
        "lightgbm_source": str(LIGHTGBM_PREDICTIONS_PATH),
        "deepfm_source": str(DEEPFM_PREDICTIONS_PATH),
        "lightgbm_prediction_column": lgbm_pred_col,
        "deepfm_prediction_column": deepfm_pred_col,
        "best_lightgbm_weight": best_row.lightgbm_weight,
        "best_deepfm_weight": best_row.deepfm_weight,
        "weight_search_audit": audit_to_dict(search_audit),
        "selection_metric": "LogLoss",
        "best_auc": best_metrics.auc,
        "best_logloss": best_metrics.logloss,
        "best_brier": best_metrics.brier,
        "best_average_precision": best_metrics.average_precision,
        "actual_ctr": best_metrics.actual_ctr,
        "mean_predicted_ctr": best_metrics.mean_predicted_ctr,
        "calibration_gap": best_metrics.calibration_gap,
        "lightgbm_auc": lightgbm_metrics.auc,
        "lightgbm_logloss": lightgbm_metrics.logloss,
        "lightgbm_brier": lightgbm_metrics.brier,
        "deepfm_auc": deepfm_metrics.auc,
        "deepfm_logloss": deepfm_metrics.logloss,
        "deepfm_brier": deepfm_metrics.brier,
        "equal_weight_ensemble_auc": equal_metrics.auc,
        "equal_weight_ensemble_logloss": equal_metrics.logloss,
        "equal_weight_ensemble_brier": equal_metrics.brier,
        "click_sequence_match": click_sequence_match,
        "holdout_used": False,
        "validation_passed": True,
        "weight_search_path": str(WEIGHT_SEARCH_PATH),
        "ensemble_predictions_path": str(ENSEMBLE_PREDICTIONS_PATH),
        "metrics_path": str(METRICS_PATH),
        "development_model_selection_note": (
            "该 ensemble 权重是在当前 development validation set 上选择的，"
            "属于 development model selection，不是最终泛化性能。"
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nWeight search: {WEIGHT_SEARCH_PATH}")
    print(f"Ensemble predictions: {ENSEMBLE_PREDICTIONS_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print_final_summary(
        len(aligned_df),
        lightgbm_metrics,
        deepfm_metrics,
        equal_metrics,
        best_row,
        best_metrics,
        click_sequence_match,
    )


if __name__ == "__main__":
    main()
