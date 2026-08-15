"""
百度 CTR 项目 — 当前 Ensemble 最佳阈值选择与固定阈值 Holdout 评估（第 44 步）

功能：
    1. 仅在 development validation（500K，2014-10-29）上精确搜索最大 F1 阈值
    2. 冻结阈值后，原样应用到 final holdout（2014-10-30）做固定阈值评价
    3. 禁止在 holdout 上选择、微调或比较候选阈值

输入：
    outputs/predictions/lightgbm_deepfm_ensemble_valid_predictions.parquet
    outputs/predictions/final_holdout_predictions.parquet
    outputs/lightgbm_deepfm_ensemble_metadata.json
    outputs/final_holdout_metadata.json

用法：
    python scripts/44_analyze_ensemble_threshold.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径与正式口径
# ---------------------------------------------------------------------------

VALID_PREDICTIONS_PATH = Path(
    "outputs/predictions/lightgbm_deepfm_ensemble_valid_predictions.parquet"
)
HOLDOUT_PREDICTIONS_PATH = Path("outputs/predictions/final_holdout_predictions.parquet")
STEP42_METADATA_PATH = Path("outputs/lightgbm_deepfm_ensemble_metadata.json")
STEP43_METADATA_PATH = Path("outputs/final_holdout_metadata.json")

OUTPUT_DIR = Path("outputs/ensemble_threshold")
VALIDATION_CURVE_PATH = OUTPUT_DIR / "validation_threshold_curve.csv"
OPERATING_POINTS_PATH = OUTPUT_DIR / "fixed_threshold_operating_points.csv"
METRICS_JSON_PATH = OUTPUT_DIR / "ensemble_threshold_metrics.json"
METADATA_JSON_PATH = OUTPUT_DIR / "ensemble_threshold_metadata.json"
REPORT_PATH = OUTPUT_DIR / "ensemble_threshold_report.txt"
PLOT_PATH = OUTPUT_DIR / "plots" / "ensemble_threshold_analysis.png"

FORMAL_VALID_ROWS = 500_000
FORMAL_HOLDOUT_ROWS = 4_218_938
EXPECTED_LIGHTGBM_WEIGHT = 0.588
EXPECTED_DEEPFM_WEIGHT = 0.412
WEIGHT_TOLERANCE = 1e-9

REFERENCE_THRESHOLD = 0.5

LABEL_CANDIDATES = ("click", "label", "y")
PROBABILITY_CANDIDATES = (
    "ensemble_pred",
    "weighted_ensemble_pred",
    "prediction",
    "pred",
    "probability",
)

FORBIDDEN_PATH_KEYWORDS = ("test.csv",)


@dataclass(frozen=True)
class ThresholdSelection:
    development_max_f1: float
    development_max_youden_j: float
    reference_threshold: float = REFERENCE_THRESHOLD


@dataclass
class OperatingPointMetrics:
    operating_point: str
    dataset: str
    threshold: float
    predicted_positive_rows: int
    coverage: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    youden_j: float
    accuracy: float
    actual_ctr: float
    selected_ctr: float
    lift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "operating_point": self.operating_point,
            "dataset": self.dataset,
            "threshold": self.threshold,
            "predicted_positive_rows": self.predicted_positive_rows,
            "coverage": self.coverage,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "specificity": self.specificity,
            "youden_j": self.youden_j,
            "accuracy": self.accuracy,
            "actual_ctr": self.actual_ctr,
            "selected_ctr": self.selected_ctr,
            "lift": self.lift,
        }


def assert_safe_path(path: Path) -> None:
    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def detect_column(columns: list[str], candidates: tuple[str, ...], purpose: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"未找到{purpose}列，候选={candidates}，实际={columns}")


def compute_click_checksum(clicks: np.ndarray) -> str:
    return hashlib.sha256(clicks.astype(np.int8).tobytes()).hexdigest()


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def validate_labels(clicks: np.ndarray, context: str) -> None:
    if clicks.size == 0:
        raise ValueError(f"{context} 标签为空。")
    if np.isnan(clicks).any():
        raise ValueError(f"{context} 标签存在 NaN。")
    unique = set(np.unique(clicks).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{context} 标签只能是 0 或 1，实际={sorted(unique)}。")


def validate_probabilities(probabilities: np.ndarray, context: str) -> None:
    if probabilities.size == 0:
        raise ValueError(f"{context} 概率为空。")
    if np.isnan(probabilities).any():
        raise ValueError(f"{context} 概率存在 NaN。")
    if np.isinf(probabilities).any():
        raise ValueError(f"{context} 概率存在 inf。")
    if (probabilities < 0).any() or (probabilities > 1).any():
        min_prob = float(probabilities.min())
        max_prob = float(probabilities.max())
        raise ValueError(f"{context} 概率超出 [0,1]：min={min_prob}, max={max_prob}。")


def load_prediction_frame(path: Path) -> tuple[pd.DataFrame, str, str]:
    assert_safe_path(path)
    if not path.exists():
        raise FileNotFoundError(f"预测文件不存在：{path}")
    frame = pd.read_parquet(path)
    label_col = detect_column(list(frame.columns), LABEL_CANDIDATES, "标签")
    prob_col = detect_column(list(frame.columns), PROBABILITY_CANDIDATES, "概率")
    return frame, label_col, prob_col


def verify_upstream_metadata(step42_meta: dict[str, Any], step43_meta: dict[str, Any]) -> None:
    if step42_meta.get("validation_passed") is not True:
        raise ValueError("第 42 步 validation_passed 必须为 true。")
    if step42_meta.get("holdout_used") is not False:
        raise ValueError("第 42 步 holdout_used 必须为 false。")
    if step43_meta.get("one_shot_evaluation") is not True:
        raise ValueError("第 43 步 one_shot_evaluation 必须为 true。")

    holdout_flags = [
        "holdout_used_for_training",
        "holdout_used_for_preprocessing_fit",
        "holdout_used_for_model_selection",
        "holdout_used_for_weight_selection",
    ]
    for flag in holdout_flags:
        if step43_meta.get(flag) is not False:
            raise ValueError(f"第 43 步 {flag} 必须为 false。")

    lgbm42 = float(step42_meta["best_lightgbm_weight"])
    deepfm42 = float(step42_meta["best_deepfm_weight"])
    lgbm43 = float(step43_meta["ensemble_lightgbm_weight"])
    deepfm43 = float(step43_meta["ensemble_deepfm_weight"])

    if not np.isclose(lgbm42, EXPECTED_LIGHTGBM_WEIGHT, atol=WEIGHT_TOLERANCE):
        raise ValueError(f"第 42 步 LightGBM 权重异常：{lgbm42}")
    if not np.isclose(deepfm42, EXPECTED_DEEPFM_WEIGHT, atol=WEIGHT_TOLERANCE):
        raise ValueError(f"第 42 步 DeepFM 权重异常：{deepfm42}")
    if not np.isclose(lgbm43, EXPECTED_LIGHTGBM_WEIGHT, atol=WEIGHT_TOLERANCE):
        raise ValueError(f"第 43 步 LightGBM 权重异常：{lgbm43}")
    if not np.isclose(deepfm43, EXPECTED_DEEPFM_WEIGHT, atol=WEIGHT_TOLERANCE):
        raise ValueError(f"第 43 步 DeepFM 权重异常：{deepfm43}")
    if not np.isclose(lgbm42, lgbm43, atol=WEIGHT_TOLERANCE):
        raise ValueError(
            f"development 与 final Ensemble 权重不一致：step42={lgbm42}, step43={lgbm43}"
        )
    if not np.isclose(deepfm42, deepfm43, atol=WEIGHT_TOLERANCE):
        raise ValueError(
            f"development 与 final Ensemble 权重不一致：step42={deepfm42}, step43={deepfm43}"
        )


def compute_exact_threshold_curve(
    probabilities: np.ndarray,
    clicks: np.ndarray,
) -> pd.DataFrame:
    """
    基于全部可实现唯一预测概率的精确阈值曲线。
    相同概率样本作为整体处理，不在相同概率中间切断。
    """
    validate_labels(clicks, "validation")
    validate_probabilities(probabilities, "validation")

    order = np.argsort(-probabilities, kind="mergesort")
    probs_sorted = probabilities[order]
    clicks_sorted = clicks[order]

    total_rows = len(clicks)
    total_clicks = int(clicks.sum())
    overall_ctr = float(clicks.mean())

    rows: list[dict[str, Any]] = []
    index = 0
    cumulative_positives = 0
    cumulative_rows = 0

    while index < total_rows:
        threshold = float(probs_sorted[index])
        group_start = index
        while index < total_rows and probs_sorted[index] == threshold:
            index += 1
        group_clicks = int(clicks_sorted[group_start:index].sum())
        group_rows = index - group_start

        cumulative_positives += group_clicks
        cumulative_rows += group_rows

        tp = cumulative_positives
        fp = cumulative_rows - cumulative_positives
        fn = total_clicks - tp
        tn = (total_rows - total_clicks) - fp

        predicted_positive_rows = tp + fp
        coverage = safe_divide(predicted_positive_rows, total_rows)
        precision = safe_divide(tp, predicted_positive_rows)
        recall = safe_divide(tp, total_clicks)
        specificity = safe_divide(tn, tn + fp)
        accuracy = safe_divide(tp + tn, total_rows)
        f1 = (
            float("nan")
            if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0
            else 2 * precision * recall / (precision + recall)
        )
        youden_j = (
            float("nan")
            if np.isnan(recall) or np.isnan(specificity)
            else recall + specificity - 1.0
        )
        selected_ctr = safe_divide(tp, predicted_positive_rows)
        lift = float("nan") if np.isnan(selected_ctr) else safe_divide(selected_ctr, overall_ctr)

        rows.append(
            {
                "threshold": threshold,
                "predicted_positive_rows": predicted_positive_rows,
                "coverage": coverage,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "youden_j": youden_j,
                "accuracy": accuracy,
                "actual_ctr": overall_ctr,
                "selected_ctr": selected_ctr,
                "lift": lift,
            }
        )

    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        raise ValueError("阈值曲线为空。")
    return curve_df.sort_values("threshold", ascending=False).reset_index(drop=True)


def select_max_f1_threshold(curve_df: pd.DataFrame) -> float:
    working = curve_df.copy()
    working = working.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    best = working.iloc[0]
    return float(best["threshold"])


def select_max_youden_threshold(curve_df: pd.DataFrame) -> float:
    working = curve_df.copy()
    working = working.sort_values(
        ["youden_j", "threshold"],
        ascending=[False, False],
        kind="mergesort",
    )
    best = working.iloc[0]
    return float(best["threshold"])


def compute_operating_point(
    probabilities: np.ndarray,
    clicks: np.ndarray,
    threshold: float,
    operating_point: str,
    dataset: str,
) -> OperatingPointMetrics:
    validate_labels(clicks, dataset)
    validate_probabilities(probabilities, dataset)

    predicted_positive = probabilities >= threshold
    predicted_negative = ~predicted_positive

    tp = int(np.sum(predicted_positive & (clicks == 1)))
    fp = int(np.sum(predicted_positive & (clicks == 0)))
    tn = int(np.sum(predicted_negative & (clicks == 0)))
    fn = int(np.sum(predicted_negative & (clicks == 1)))

    total_rows = len(clicks)
    total_clicks = int(clicks.sum())
    overall_ctr = float(clicks.mean())
    predicted_positive_rows = tp + fp

    coverage = safe_divide(predicted_positive_rows, total_rows)
    precision = safe_divide(tp, predicted_positive_rows)
    recall = safe_divide(tp, total_clicks)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, total_rows)
    f1 = (
        float("nan")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0
        else 2 * precision * recall / (precision + recall)
    )
    youden_j = (
        float("nan")
        if np.isnan(recall) or np.isnan(specificity)
        else recall + specificity - 1.0
    )
    selected_ctr = safe_divide(tp, predicted_positive_rows)
    lift = float("nan") if np.isnan(selected_ctr) else safe_divide(selected_ctr, overall_ctr)

    return OperatingPointMetrics(
        operating_point=operating_point,
        dataset=dataset,
        threshold=float(threshold),
        predicted_positive_rows=predicted_positive_rows,
        coverage=coverage,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        youden_j=youden_j,
        accuracy=accuracy,
        actual_ctr=overall_ctr,
        selected_ctr=selected_ctr,
        lift=lift,
    )


def brute_force_max_f1_threshold(probabilities: np.ndarray, clicks: np.ndarray) -> float:
    """暴力枚举全部唯一概率阈值，用于测试对齐。"""
    unique_thresholds = np.unique(probabilities)
    best_threshold = float(unique_thresholds[0])
    best_key = (-1.0, -1.0, -1.0)

    for threshold in unique_thresholds:
        point = compute_operating_point(
            probabilities,
            clicks,
            float(threshold),
            operating_point="brute_force",
            dataset="validation",
        )
        key = (point.f1, point.precision, point.threshold)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def ensure_curve_contains_thresholds(
    curve_df: pd.DataFrame,
    thresholds: list[float],
) -> None:
    curve_values = curve_df["threshold"].to_numpy(dtype=np.float64)
    for threshold in thresholds:
        if not np.any(np.isclose(curve_values, threshold, rtol=0.0, atol=1e-12)):
            raise ValueError(f"阈值曲线缺少精确阈值点：{threshold}")


def mark_special_points(curve_df: pd.DataFrame, thresholds: ThresholdSelection) -> pd.DataFrame:
    marked = curve_df.copy()
    marked["is_development_max_f1"] = np.isclose(
        marked["threshold"], thresholds.development_max_f1, rtol=0.0, atol=1e-12
    )
    marked["is_development_max_youden_j"] = np.isclose(
        marked["threshold"], thresholds.development_max_youden_j, rtol=0.0, atol=1e-12
    )
    marked["is_reference_0_5"] = np.isclose(
        marked["threshold"], thresholds.reference_threshold, rtol=0.0, atol=1e-12
    )
    return marked


def plot_threshold_analysis(
    curve_df: pd.DataFrame,
    thresholds: ThresholdSelection,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = curve_df.sort_values("threshold", ascending=True)
    max_f1_row = curve_df.loc[
        np.isclose(curve_df["threshold"], thresholds.development_max_f1, rtol=0.0, atol=1e-12)
    ].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(plot_df["threshold"], plot_df["precision"], label="Precision")
    axes[0].plot(plot_df["threshold"], plot_df["recall"], label="Recall")
    axes[0].plot(plot_df["threshold"], plot_df["f1"], label="F1")
    axes[0].axvline(
        thresholds.development_max_f1,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Max F1={thresholds.development_max_f1:.6f}",
    )
    axes[0].scatter(
        [max_f1_row["threshold"]],
        [max_f1_row["f1"]],
        color="red",
        s=40,
        zorder=5,
    )
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold vs Precision / Recall / F1 (Validation Ensemble)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    axes[1].plot(plot_df["threshold"], plot_df["coverage"], color="tab:green", label="Coverage")
    axes[1].axvline(
        thresholds.development_max_f1,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Max F1 coverage={max_f1_row['coverage']:.4f}",
    )
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Coverage")
    axes[1].set_title("Threshold vs Coverage")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_report_text(
    thresholds: ThresholdSelection,
    valid_max_f1: OperatingPointMetrics,
    valid_max_youden: OperatingPointMetrics,
    valid_ref_05: OperatingPointMetrics,
    holdout_max_f1: OperatingPointMetrics,
    holdout_max_youden: OperatingPointMetrics,
    holdout_ref_05: OperatingPointMetrics,
) -> str:
    lines = [
        "百度 CTR 项目 — 第 44 步 Ensemble 阈值分析报告",
        "=" * 72,
        "",
        "一、阈值选择原则",
        "- 最佳阈值仅在 development validation（500,000 行，2014-10-29）上选择。",
        "- holdout 未参与阈值选择、微调或候选比较。",
        "- 主选择指标为最大 F1；F1 并列时优先 Precision 更高，再优先更高阈值。",
        "- 当前 Ensemble 权重：LightGBM=0.588，DeepFM=0.412。",
        "",
        "二、当前 Ensemble 最大 F1 阈值（development validation）",
        f"- 冻结阈值：{thresholds.development_max_f1:.12f}",
        f"- Precision：{valid_max_f1.precision:.6f}",
        f"- Recall：{valid_max_f1.recall:.6f}",
        f"- F1：{valid_max_f1.f1:.6f}",
        f"- Coverage：{valid_max_f1.coverage:.6f}",
        f"- Selected CTR：{valid_max_f1.selected_ctr:.6f}",
        f"- Lift：{valid_max_f1.lift:.6f}",
        "",
        "三、为什么 threshold=0.5 不合适",
        f"- 0.5 阈值下 validation F1={valid_ref_05.f1:.6f}，显著低于最大 F1={valid_max_f1.f1:.6f}。",
        f"- 0.5 阈值下 Coverage={valid_ref_05.coverage:.6f}，Recall={valid_ref_05.recall:.6f}，"
        "在低 CTR 场景下过于保守，会漏掉大量潜在点击。",
        f"- 验证集实际 CTR 仅约 {valid_max_f1.actual_ctr:.4f}，0.5 远高于多数样本预测概率分布。",
        "",
        "四、同一冻结阈值在 holdout 上的固定评价",
        f"- holdout 行数：4,218,938（2014-10-30）",
        f"- 使用冻结阈值 {thresholds.development_max_f1:.12f}：",
        f"  Precision={holdout_max_f1.precision:.6f}, Recall={holdout_max_f1.recall:.6f}, "
        f"F1={holdout_max_f1.f1:.6f}, Coverage={holdout_max_f1.coverage:.6f}, "
        f"Selected CTR={holdout_max_f1.selected_ctr:.6f}, Lift={holdout_max_f1.lift:.6f}",
        "",
        "五、参考阈值",
        f"- 最大 Youden J 阈值（仅参考）：{thresholds.development_max_youden_j:.12f}",
        f"  validation Youden J={valid_max_youden.youden_j:.6f}",
        f"- threshold=0.5 对照：validation F1={valid_ref_05.f1:.6f}，holdout F1={holdout_ref_05.f1:.6f}",
        "",
        "六、重要限制",
        "- 最大 F1 阈值只是 Precision 与 Recall 的统计折中，不是利润最优阈值。",
        "- 当前没有 CPC、点击价值、预算和库存数据，不能声称找到业务收益最优阈值。",
        "- 预算固定场景仍建议采用 Top-K 作为业务参考方案。",
        "- 不得根据 holdout 结果重新调整阈值。",
        "",
        "七、与旧版阈值分析的区别",
        "- 旧版 threshold=0.21 来自 Optuna LightGBM + Isotonic + 200,000 行 development evaluation。",
        "- 本报告阈值来自当前 LightGBM(0.588)+DeepFM(0.412) Ensemble 的 500,000 行 validation。",
        "",
    ]
    return "\n".join(lines) + "\n"


def select_thresholds_from_validation(
    probabilities: np.ndarray,
    clicks: np.ndarray,
) -> tuple[pd.DataFrame, ThresholdSelection]:
    """仅在 validation 上选择阈值；不得传入 holdout 数据。"""
    curve_df = compute_exact_threshold_curve(probabilities, clicks)
    selection = ThresholdSelection(
        development_max_f1=select_max_f1_threshold(curve_df),
        development_max_youden_j=select_max_youden_threshold(curve_df),
        reference_threshold=REFERENCE_THRESHOLD,
    )
    ensure_curve_contains_thresholds(
        curve_df,
        [selection.development_max_f1, selection.development_max_youden_j],
    )
    return curve_df, selection


def evaluate_fixed_thresholds_on_holdout(
    probabilities: np.ndarray,
    clicks: np.ndarray,
    thresholds: ThresholdSelection,
) -> tuple[OperatingPointMetrics, OperatingPointMetrics, OperatingPointMetrics]:
    return (
        compute_operating_point(
            probabilities,
            clicks,
            thresholds.development_max_f1,
            "fixed_development_max_f1",
            "holdout",
        ),
        compute_operating_point(
            probabilities,
            clicks,
            thresholds.development_max_youden_j,
            "fixed_development_max_youden_j",
            "holdout",
        ),
        compute_operating_point(
            probabilities,
            clicks,
            thresholds.reference_threshold,
            "reference_0_5",
            "holdout",
        ),
    )


def print_summary(
    thresholds: ThresholdSelection,
    valid_max_f1: OperatingPointMetrics,
    valid_ref_05: OperatingPointMetrics,
    holdout_max_f1: OperatingPointMetrics,
) -> None:
    print("\n" + "=" * 72)
    print("ENSEMBLE THRESHOLD ANALYSIS")
    print("=" * 72)
    print(f"DEVELOPMENT_MAX_F1_THRESHOLD = {thresholds.development_max_f1:.12f}")
    print(f"DEVELOPMENT_MAX_YOUDEN_J_THRESHOLD = {thresholds.development_max_youden_j:.12f}")
    print("VALIDATION (500,000 rows, frozen threshold selection)")
    print(
        f"  Precision={valid_max_f1.precision:.6f}, Recall={valid_max_f1.recall:.6f}, "
        f"F1={valid_max_f1.f1:.6f}, Coverage={valid_max_f1.coverage:.6f}, "
        f"Selected CTR={valid_max_f1.selected_ctr:.6f}, Lift={valid_max_f1.lift:.6f}"
    )
    print("HOLDOUT (4,218,938 rows, fixed threshold evaluation only)")
    print(
        f"  Precision={holdout_max_f1.precision:.6f}, Recall={holdout_max_f1.recall:.6f}, "
        f"F1={holdout_max_f1.f1:.6f}, Coverage={holdout_max_f1.coverage:.6f}, "
        f"Selected CTR={holdout_max_f1.selected_ctr:.6f}, Lift={holdout_max_f1.lift:.6f}"
    )
    print("VS threshold=0.5 on validation")
    print(
        f"  F1 improvement = {valid_max_f1.f1 - valid_ref_05.f1:.6f}, "
        f"Recall improvement = {valid_max_f1.recall - valid_ref_05.recall:.6f}, "
        f"Coverage change = {valid_max_f1.coverage - valid_ref_05.coverage:.6f}"
    )
    print("HOLDOUT_USED_FOR_THRESHOLD_SELECTION = False")
    print("HOLDOUT_USED_FOR_FIXED_THRESHOLD_EVALUATION = True")
    print("BUSINESS_OPTIMALITY_CLAIMED = False")
    print("=" * 72)


def main() -> None:
    print("=" * 72)
    print("百度 CTR 项目 — 第 44 步 Ensemble 阈值分析")
    print(f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")

    for path in (
        VALID_PREDICTIONS_PATH,
        HOLDOUT_PREDICTIONS_PATH,
        STEP42_METADATA_PATH,
        STEP43_METADATA_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}，请先补跑对应步骤。")

    step42_meta = json.loads(STEP42_METADATA_PATH.read_text(encoding="utf-8"))
    step43_meta = json.loads(STEP43_METADATA_PATH.read_text(encoding="utf-8"))
    verify_upstream_metadata(step42_meta, step43_meta)

    valid_frame, valid_label_col, valid_prob_col = load_prediction_frame(VALID_PREDICTIONS_PATH)
    if len(valid_frame) != FORMAL_VALID_ROWS:
        raise ValueError(
            f"validation 行数 {len(valid_frame):,} != 正式口径 {FORMAL_VALID_ROWS:,}"
        )

    valid_clicks = valid_frame[valid_label_col].to_numpy(dtype=np.int8)
    valid_probs = valid_frame[valid_prob_col].to_numpy(dtype=np.float64)
    validate_labels(valid_clicks, "validation")
    validate_probabilities(valid_probs, "validation")
    valid_checksum = compute_click_checksum(valid_clicks)

    curve_df, thresholds = select_thresholds_from_validation(valid_probs, valid_clicks)

    valid_max_f1 = compute_operating_point(
        valid_probs,
        valid_clicks,
        thresholds.development_max_f1,
        "development_max_f1",
        "validation",
    )
    valid_max_youden = compute_operating_point(
        valid_probs,
        valid_clicks,
        thresholds.development_max_youden_j,
        "development_max_youden_j",
        "validation",
    )
    valid_ref_05 = compute_operating_point(
        valid_probs,
        valid_clicks,
        thresholds.reference_threshold,
        "reference_0_5",
        "validation",
    )

    holdout_frame, holdout_label_col, holdout_prob_col = load_prediction_frame(
        HOLDOUT_PREDICTIONS_PATH
    )
    if len(holdout_frame) != FORMAL_HOLDOUT_ROWS:
        raise ValueError(
            f"holdout 行数 {len(holdout_frame):,} != 正式口径 {FORMAL_HOLDOUT_ROWS:,}"
        )

    holdout_clicks = holdout_frame[holdout_label_col].to_numpy(dtype=np.int8)
    holdout_probs = holdout_frame[holdout_prob_col].to_numpy(dtype=np.float64)
    validate_labels(holdout_clicks, "holdout")
    validate_probabilities(holdout_probs, "holdout")
    holdout_checksum = compute_click_checksum(holdout_clicks)

    holdout_max_f1, holdout_max_youden, holdout_ref_05 = evaluate_fixed_thresholds_on_holdout(
        holdout_probs,
        holdout_clicks,
        thresholds,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    marked_curve = mark_special_points(curve_df, thresholds)
    marked_curve.to_csv(VALIDATION_CURVE_PATH, index=False)

    operating_points = [
        valid_max_f1,
        valid_max_youden,
        valid_ref_05,
        holdout_max_f1,
        holdout_max_youden,
        holdout_ref_05,
    ]
    pd.DataFrame([point.to_dict() for point in operating_points]).to_csv(
        OPERATING_POINTS_PATH, index=False
    )

    metrics_payload = {
        "script_name": "scripts/44_analyze_ensemble_threshold.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ensemble_weights": {
            "lightgbm": EXPECTED_LIGHTGBM_WEIGHT,
            "deepfm": EXPECTED_DEEPFM_WEIGHT,
        },
        "development_max_f1_threshold": thresholds.development_max_f1,
        "development_max_youden_j_threshold": thresholds.development_max_youden_j,
        "reference_threshold": thresholds.reference_threshold,
        "validation_metrics": {
            "development_max_f1": valid_max_f1.to_dict(),
            "development_max_youden_j": valid_max_youden.to_dict(),
            "reference_0_5": valid_ref_05.to_dict(),
        },
        "holdout_fixed_threshold_metrics": {
            "fixed_development_max_f1": holdout_max_f1.to_dict(),
            "fixed_development_max_youden_j": holdout_max_youden.to_dict(),
            "reference_0_5": holdout_ref_05.to_dict(),
        },
        "holdout_used_for_threshold_selection": False,
        "holdout_used_for_fixed_threshold_evaluation": True,
        "business_optimality_claimed": False,
        "selection_metric": "max_f1",
        "tie_break_rules": ["higher_precision", "higher_threshold"],
        "note": "Maximum F1 threshold is a precision-recall statistical compromise, not profit-optimal.",
    }
    METRICS_JSON_PATH.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metadata_payload = {
        "script_name": "scripts/44_analyze_ensemble_threshold.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "validation_predictions": str(VALID_PREDICTIONS_PATH),
            "holdout_predictions": str(HOLDOUT_PREDICTIONS_PATH),
            "step42_metadata": str(STEP42_METADATA_PATH),
            "step43_metadata": str(STEP43_METADATA_PATH),
        },
        "detected_columns": {
            "validation_label": valid_label_col,
            "validation_probability": valid_prob_col,
            "holdout_label": holdout_label_col,
            "holdout_probability": holdout_prob_col,
        },
        "formal_row_counts": {
            "validation": FORMAL_VALID_ROWS,
            "holdout": FORMAL_HOLDOUT_ROWS,
        },
        "actual_row_counts": {
            "validation": int(len(valid_frame)),
            "holdout": int(len(holdout_frame)),
        },
        "click_checksums": {
            "validation": valid_checksum,
            "holdout": holdout_checksum,
        },
        "threshold_selection_dataset": "development_validation_2014-10-29",
        "threshold_selection_metric": "max_f1",
        "tie_break_rules": ["higher_precision", "higher_threshold"],
        "ensemble_weight_consistency": {
            "step42_lightgbm": step42_meta["best_lightgbm_weight"],
            "step42_deepfm": step42_meta["best_deepfm_weight"],
            "step43_lightgbm": step43_meta["ensemble_lightgbm_weight"],
            "step43_deepfm": step43_meta["ensemble_deepfm_weight"],
            "consistent": True,
        },
        "leakage_audit": {
            "holdout_used_for_threshold_selection": False,
            "holdout_used_for_fixed_threshold_evaluation": True,
            "step42_holdout_used": step42_meta.get("holdout_used"),
            "step42_validation_passed": step42_meta.get("validation_passed"),
            "step43_one_shot_evaluation": step43_meta.get("one_shot_evaluation"),
            "step43_holdout_used_for_training": step43_meta.get("holdout_used_for_training"),
            "step43_holdout_used_for_weight_selection": step43_meta.get(
                "holdout_used_for_weight_selection"
            ),
        },
        "unique_validation_threshold_count": int(curve_df.shape[0]),
        "outputs": {
            "validation_threshold_curve": str(VALIDATION_CURVE_PATH),
            "fixed_threshold_operating_points": str(OPERATING_POINTS_PATH),
            "metrics_json": str(METRICS_JSON_PATH),
            "report_txt": str(REPORT_PATH),
            "plot_png": str(PLOT_PATH),
        },
    }
    METADATA_JSON_PATH.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report_text = build_report_text(
        thresholds,
        valid_max_f1,
        valid_max_youden,
        valid_ref_05,
        holdout_max_f1,
        holdout_max_youden,
        holdout_ref_05,
    )
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    plot_threshold_analysis(curve_df, thresholds, PLOT_PATH)

    print(f"\nValidation curve: {VALIDATION_CURVE_PATH}")
    print(f"Operating points: {OPERATING_POINTS_PATH}")
    print(f"Metrics JSON: {METRICS_JSON_PATH}")
    print(f"Metadata JSON: {METADATA_JSON_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Plot: {PLOT_PATH}")

    print_summary(thresholds, valid_max_f1, valid_ref_05, holdout_max_f1)


if __name__ == "__main__":
    main()
