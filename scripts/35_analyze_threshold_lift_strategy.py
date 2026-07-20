"""
百度 CTR 项目 — 阈值、Lift 与投放覆盖率策略分析

功能：
    使用第 34 步独立 development evaluation 子集上的最佳校准概率，完成
    阈值扫描、Top-K 投放、Lift / Gain 与候选业务运行点分析。
    禁止读取 holdout，不重新训练模型或拟合校准器。

数据输入：
    outputs/predictions/probability_calibration_evaluation_predictions.parquet
    outputs/probability_calibration_metadata.json
    outputs/probability_calibration_metrics.csv

用法：
    python scripts/35_analyze_threshold_lift_strategy.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

RANDOM_STATE = 42
TEST_ROWS = 50_000
FORMAL_EVAL_ROWS = 200_000
PROBABILITY_EPSILON = 1e-7

DATA_SCOPE = "development_calibration_evaluation_split"

METHOD_TO_COLUMN = {
    "uncalibrated": "raw_probability",
    "sigmoid": "sigmoid_probability",
    "isotonic": "isotonic_probability",
}

TOPK_COVERAGES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

TOPK_PERCENTILE_THRESHOLDS = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

PROBABILITY_QUANTILES = [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]

EVALUATION_PREDICTIONS_PATH = Path(
    "outputs/predictions/probability_calibration_evaluation_predictions.parquet"
)
CALIBRATION_METADATA_PATH = Path("outputs/probability_calibration_metadata.json")
CALIBRATION_METRICS_PATH = Path("outputs/probability_calibration_metrics.csv")


@dataclass
class OutputPaths:
    """第 35 步输出路径。"""

    test_mode: bool
    strategy_dir: Path
    plots_dir: Path
    threshold_metrics_csv: Path
    topk_metrics_csv: Path
    decile_csv: Path
    operating_points_csv: Path
    ranked_predictions: Path
    summary_csv: Path
    report_txt: Path
    metadata_json: Path


@dataclass
class ProbabilitySummary:
    """selected probability 汇总统计。"""

    rows: int
    clicks: int
    actual_ctr: float
    mean_predicted_ctr: float
    roc_auc: float
    average_precision: float
    log_loss: float
    brier_score: float
    minimum_probability: float
    maximum_probability: float
    probability_quantiles: dict[str, float]


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    suffix = "_test" if test_mode else ""
    strategy_dir = Path("outputs/strategy")
    plots_dir = strategy_dir / ("plots_test" if test_mode else "plots")

    return OutputPaths(
        test_mode=test_mode,
        strategy_dir=strategy_dir,
        plots_dir=plots_dir,
        threshold_metrics_csv=strategy_dir / f"threshold_metrics{suffix}.csv",
        topk_metrics_csv=strategy_dir / f"topk_lift_metrics{suffix}.csv",
        decile_csv=strategy_dir / f"score_decile_analysis{suffix}.csv",
        operating_points_csv=strategy_dir / f"operating_points{suffix}.csv",
        ranked_predictions=Path(
            f"outputs/predictions/selected_probability_strategy_evaluation{suffix}.parquet"
        ),
        summary_csv=Path(f"outputs/threshold_strategy_summary{suffix}.csv"),
        report_txt=Path(f"outputs/threshold_strategy_report{suffix}.txt"),
        metadata_json=Path(f"outputs/threshold_strategy_metadata{suffix}.json"),
    )


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    if not path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_calibration_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取并校验概率校准元数据。"""

    metadata = load_json(metadata_path)

    if metadata.get("test_mode") is not False:
        raise ValueError("校准元数据 test_mode 必须为 false（正式结果）。")

    if int(metadata.get("total_valid_rows", -1)) != 500_000:
        raise ValueError("total_valid_rows 必须为 500000。")

    if int(metadata.get("calibration_rows", -1)) != 300_000:
        raise ValueError("calibration_rows 必须为 300000。")

    if int(metadata.get("evaluation_rows", -1)) != FORMAL_EVAL_ROWS:
        raise ValueError(f"evaluation_rows 必须为 {FORMAL_EVAL_ROWS}。")

    if metadata.get("holdout_used") is not False:
        raise ValueError("holdout_used 必须为 false。")

    if metadata.get("validation_passed") is not True:
        raise ValueError("validation_passed 必须为 true。")

    required_keys = ("best_development_method", "final_calibrator_method")
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"校准元数据缺少字段：{key}")

    best_method = str(metadata["best_development_method"])
    if best_method not in METHOD_TO_COLUMN:
        raise ValueError(f"未知 best_development_method：{best_method}")

    return metadata


def load_calibration_metrics(
    metrics_path: Path,
    expected_best_method: str,
) -> pd.DataFrame:
    """读取并校验校准指标表。"""

    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到校准指标文件：{metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    expected_methods = set(METHOD_TO_COLUMN.keys())
    actual_methods = set(metrics_df["method"].astype(str))
    if actual_methods != expected_methods:
        raise ValueError(
            f"metrics.csv 方法集合不正确：期望 {expected_methods}，实际 {actual_methods}"
        )

    selected_rows = metrics_df.loc[metrics_df["selected_as_best"] == True]  # noqa: E712
    if len(selected_rows) != 1:
        raise ValueError("metrics.csv 中 selected_as_best=true 必须恰好 1 行。")

    selected_method = str(selected_rows.iloc[0]["method"])
    if selected_method != expected_best_method:
        raise ValueError(
            f"selected_as_best 方法 {selected_method} 与 metadata "
            f"{expected_best_method} 不一致。"
        )

    return metrics_df


def resolve_probability_column(selected_method: str) -> str:
    """根据最佳校准方法返回概率列名。"""

    if selected_method not in METHOD_TO_COLUMN:
        raise ValueError(f"无法映射概率列：{selected_method}")
    return METHOD_TO_COLUMN[selected_method]


def load_evaluation_predictions(
    prediction_path: Path,
    probability_column: str,
) -> pd.DataFrame:
    """读取并校验 evaluation 预测文件。"""

    if not prediction_path.exists():
        raise FileNotFoundError(f"未找到 evaluation 预测文件：{prediction_path}")

    evaluation_df = pd.read_parquet(prediction_path)

    required_columns = [
        "row_position",
        "id",
        "click",
        "split_date",
        "raw_probability",
        "sigmoid_probability",
        "isotonic_probability",
        "calibration_split",
    ]
    missing = [column for column in required_columns if column not in evaluation_df.columns]
    if missing:
        raise ValueError(f"evaluation 预测文件缺少列：{missing}")

    if len(evaluation_df) != FORMAL_EVAL_ROWS:
        raise ValueError(
            f"evaluation 行数 {len(evaluation_df):,} 不等于 {FORMAL_EVAL_ROWS:,}。"
        )

    if not (evaluation_df["calibration_split"] == "evaluation").all():
        raise ValueError("calibration_split 必须全部为 evaluation。")

    if evaluation_df["row_position"].isna().any():
        raise ValueError("row_position 存在缺失。")

    clicks = evaluation_df["click"].to_numpy()
    if not np.isin(clicks, [0, 1]).all():
        raise ValueError("click 不是仅包含 0 和 1。")

    probabilities = evaluation_df[probability_column].to_numpy(dtype=np.float64)
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError(f"{probability_column} 存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{probability_column} 超出 [0, 1] 范围。")

    evaluation_df = evaluation_df.copy()
    evaluation_df["selected_probability"] = probabilities
    return evaluation_df


def select_analysis_subset(
    evaluation_df: pd.DataFrame,
    test_mode: bool,
) -> pd.DataFrame:
    """测试模式分层抽样，正式模式使用完整 evaluation。"""

    if not test_mode:
        return evaluation_df.copy().reset_index(drop=True)

    if TEST_ROWS > len(evaluation_df):
        raise ValueError(
            f"TEST_ROWS={TEST_ROWS:,} 超过 evaluation 行数 {len(evaluation_df):,}。"
        )

    sample_indices, _ = train_test_split(
        np.arange(len(evaluation_df)),
        train_size=TEST_ROWS,
        stratify=evaluation_df["click"].to_numpy(),
        random_state=RANDOM_STATE,
    )
    sample_indices = np.sort(sample_indices.astype(np.int64))
    return evaluation_df.iloc[sample_indices].copy().reset_index(drop=True)


def safe_divide(numerator: float, denominator: float) -> float:
    """安全除法，分母为 0 时返回 NaN。"""

    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_probability_summary(
    probabilities: np.ndarray,
    clicks: np.ndarray,
) -> ProbabilitySummary:
    """计算 selected probability 基础指标。"""

    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    quantile_values = np.quantile(probabilities, PROBABILITY_QUANTILES)
    quantile_map = {
        f"q{int(q * 100):02d}" if q < 1 else "q100": float(value)
        for q, value in zip(PROBABILITY_QUANTILES, quantile_values, strict=True)
    }

    return ProbabilitySummary(
        rows=len(clicks),
        clicks=int(clicks.sum()),
        actual_ctr=float(clicks.mean()),
        mean_predicted_ctr=float(probabilities.mean()),
        roc_auc=float(roc_auc_score(clicks, probabilities)),
        average_precision=float(average_precision_score(clicks, probabilities)),
        log_loss=float(log_loss(clicks, clipped, labels=[0, 1])),
        brier_score=float(brier_score_loss(clicks, probabilities)),
        minimum_probability=float(probabilities.min()),
        maximum_probability=float(probabilities.max()),
        probability_quantiles=quantile_map,
    )


def build_threshold_grid(probabilities: np.ndarray) -> np.ndarray:
    """构建候选阈值集合。"""

    fixed_thresholds = list(np.round(np.arange(0.01, 0.51, 0.01), 2))
    fixed_thresholds.extend(np.round(np.arange(0.55, 1.0, 0.05), 2))
    fixed_thresholds.append(0.5)

    percentile_thresholds: list[float] = []
    for top_fraction in TOPK_PERCENTILE_THRESHOLDS:
        quantile_level = 1.0 - top_fraction
        percentile_thresholds.append(float(np.quantile(probabilities, quantile_level)))

    boundary_thresholds = [
        float(np.min(probabilities)),
        float(np.max(probabilities)),
    ]

    all_thresholds = np.unique(
        np.concatenate(
            [
                np.asarray(fixed_thresholds, dtype=np.float64),
                np.asarray(percentile_thresholds, dtype=np.float64),
                np.asarray(boundary_thresholds, dtype=np.float64),
            ]
        )
    )
    return np.sort(all_thresholds)


def compute_threshold_metrics(
    probabilities: np.ndarray,
    clicks: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """计算阈值扫描指标。"""

    total_rows = len(clicks)
    total_clicks = int(clicks.sum())
    overall_ctr = float(clicks.mean())

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        predicted_positive = probabilities >= threshold
        predicted_negative = ~predicted_positive

        tp = int(np.sum(predicted_positive & (clicks == 1)))
        fp = int(np.sum(predicted_positive & (clicks == 0)))
        tn = int(np.sum(predicted_negative & (clicks == 0)))
        fn = int(np.sum(predicted_negative & (clicks == 1)))

        predicted_positive_rows = tp + fp
        predicted_negative_rows = tn + fn

        coverage = safe_divide(predicted_positive_rows, total_rows)
        precision = safe_divide(tp, predicted_positive_rows)
        recall = safe_divide(tp, total_clicks)
        specificity = safe_divide(tn, tn + fp)
        false_positive_rate = safe_divide(fp, fp + tn)
        false_negative_rate = safe_divide(fn, fn + tp)
        accuracy = safe_divide(tp + tn, total_rows)
        balanced_accuracy = (
            float("nan")
            if np.isnan(recall) or np.isnan(specificity)
            else (recall + specificity) / 2.0
        )
        f1 = (
            float("nan")
            if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0
            else 2 * precision * recall / (precision + recall)
        )
        negative_predictive_value = safe_divide(tn, tn + fn)
        selected_group_ctr = safe_divide(tp, predicted_positive_rows)
        unselected_group_ctr = safe_divide(
            int(clicks[predicted_negative].sum()),
            predicted_negative_rows,
        )
        lift = (
            float("nan")
            if np.isnan(selected_group_ctr)
            else safe_divide(selected_group_ctr, overall_ctr)
        )
        click_capture_rate = safe_divide(tp, total_clicks)
        missed_click_rate = safe_divide(fn, total_clicks)
        youden_j = (
            float("nan")
            if np.isnan(recall) or np.isnan(specificity)
            else recall + specificity - 1.0
        )

        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "predicted_positive_rows": predicted_positive_rows,
                "predicted_negative_rows": predicted_negative_rows,
                "coverage": coverage,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1": f1,
                "negative_predictive_value": negative_predictive_value,
                "selected_group_ctr": selected_group_ctr,
                "unselected_group_ctr": unselected_group_ctr,
                "overall_ctr": overall_ctr,
                "lift": lift,
                "click_capture_rate": click_capture_rate,
                "missed_click_rate": missed_click_rate,
                "youden_j": youden_j,
            }
        )

    return pd.DataFrame(rows)


def sort_for_ranking(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """按 selected_probability 降序、row_position 升序排序。"""

    return analysis_df.sort_values(
        ["selected_probability", "row_position"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def compute_topk_metrics(
    ranked_df: pd.DataFrame,
    overall_ctr: float,
) -> pd.DataFrame:
    """计算 Top-K 投放指标。"""

    total_rows = len(ranked_df)
    total_clicks = int(ranked_df["click"].sum())
    probabilities = ranked_df["selected_probability"].to_numpy(dtype=np.float64)
    clicks = ranked_df["click"].to_numpy(dtype=np.int8)

    rows: list[dict[str, Any]] = []
    cumulative_clicks = 0

    for target_coverage in TOPK_COVERAGES:
        selected_rows = max(1, int(np.ceil(total_rows * target_coverage))) if target_coverage < 1.0 else total_rows
        selected_rows = min(selected_rows, total_rows)

        selected_slice = ranked_df.iloc[:selected_rows]
        selected_clicks = int(selected_slice["click"].sum())
        cumulative_clicks = selected_clicks
        actual_coverage = selected_rows / total_rows
        probability_cutoff = float(selected_slice["selected_probability"].min())
        selected_ctr = safe_divide(selected_clicks, selected_rows)
        lift = safe_divide(selected_ctr, overall_ctr)
        cumulative_click_capture_rate = safe_divide(selected_clicks, total_clicks)
        cumulative_gain = cumulative_click_capture_rate
        random_expected_clicks = selected_rows * overall_ctr
        incremental_clicks_vs_random = (
            float("nan")
            if np.isnan(random_expected_clicks)
            else selected_clicks - random_expected_clicks
        )

        rows.append(
            {
                "target_coverage": target_coverage,
                "selected_rows": selected_rows,
                "actual_coverage": actual_coverage,
                "probability_cutoff": probability_cutoff,
                "selected_clicks": selected_clicks,
                "selected_ctr": selected_ctr,
                "overall_ctr": overall_ctr,
                "lift": lift,
                "cumulative_click_capture_rate": cumulative_click_capture_rate,
                "cumulative_gain": cumulative_gain,
                "random_expected_clicks": random_expected_clicks,
                "incremental_clicks_vs_random": incremental_clicks_vs_random,
            }
        )

    return pd.DataFrame(rows)


def compute_decile_analysis(
    ranked_df: pd.DataFrame,
    overall_ctr: float,
) -> pd.DataFrame:
    """计算十分位分析。"""

    total_rows = len(ranked_df)
    total_clicks = int(ranked_df["click"].sum())
    split_indices = np.array_split(np.arange(total_rows), 10)

    rows: list[dict[str, Any]] = []
    cumulative_rows = 0
    cumulative_clicks = 0

    for decile, index_array in enumerate(split_indices, start=1):
        if len(index_array) == 0:
            continue

        decile_df = ranked_df.iloc[index_array]
        decile_rows = len(decile_df)
        decile_clicks = int(decile_df["click"].sum())
        cumulative_rows += decile_rows
        cumulative_clicks += decile_clicks

        probabilities = decile_df["selected_probability"].to_numpy(dtype=np.float64)
        actual_ctr = safe_divide(decile_clicks, decile_rows)
        lift = safe_divide(actual_ctr, overall_ctr)

        rows.append(
            {
                "decile": decile,
                "rows": decile_rows,
                "clicks": decile_clicks,
                "probability_min": float(probabilities.min()),
                "probability_max": float(probabilities.max()),
                "mean_probability": float(probabilities.mean()),
                "actual_ctr": actual_ctr,
                "overall_ctr": overall_ctr,
                "lift": lift,
                "share_of_total_clicks": safe_divide(decile_clicks, total_clicks),
                "cumulative_rows": cumulative_rows,
                "cumulative_coverage": safe_divide(cumulative_rows, total_rows),
                "cumulative_clicks": cumulative_clicks,
                "cumulative_click_capture_rate": safe_divide(
                    cumulative_clicks,
                    total_clicks,
                ),
            }
        )

    return pd.DataFrame(rows)


def pick_threshold_row(
    threshold_df: pd.DataFrame,
    mask: pd.Series,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.Series | None:
    """从 threshold 表按条件选取一行。"""

    candidates = threshold_df.loc[mask].copy()
    if candidates.empty:
        return None

    candidates = candidates.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    )
    return candidates.iloc[0]


def build_operating_points(
    threshold_df: pd.DataFrame,
) -> pd.DataFrame:
    """构建候选业务运行点。"""

    definitions: list[dict[str, Any]] = [
        {
            "operating_point": "maximum_f1",
            "mask": threshold_df["f1"].notna(),
            "sort_columns": ["f1", "threshold"],
            "ascending": [False, False],
            "interpretation": "F1 最大的数学折中阈值；并列时选择更高阈值。",
        },
        {
            "operating_point": "maximum_youden_j",
            "mask": threshold_df["youden_j"].notna(),
            "sort_columns": ["youden_j", "threshold"],
            "ascending": [False, False],
            "interpretation": "Youden J 最大，兼顾 recall 与 specificity。",
        },
        {
            "operating_point": "recall_at_least_80_percent",
            "mask": threshold_df["recall"] >= 0.80,
            "sort_columns": ["precision", "threshold"],
            "ascending": [False, False],
            "interpretation": "Recall >= 80% 时 precision 最高的阈值。",
        },
        {
            "operating_point": "recall_at_least_90_percent",
            "mask": threshold_df["recall"] >= 0.90,
            "sort_columns": ["precision", "threshold"],
            "ascending": [False, False],
            "interpretation": "Recall >= 90% 时 precision 最高的阈值。",
        },
        {
            "operating_point": "precision_at_least_25_percent",
            "mask": threshold_df["precision"] >= 0.25,
            "sort_columns": ["recall", "threshold"],
            "ascending": [False, False],
            "interpretation": "Precision >= 25% 时 recall 最高的阈值。",
        },
        {
            "operating_point": "coverage_nearest_10_percent",
            "mask": threshold_df["coverage"].notna(),
            "sort_columns": ["coverage_distance", "threshold"],
            "ascending": [True, False],
            "interpretation": "coverage 最接近 10% 的阈值。",
        },
        {
            "operating_point": "coverage_nearest_20_percent",
            "mask": threshold_df["coverage"].notna(),
            "sort_columns": ["coverage_distance_20", "threshold"],
            "ascending": [True, False],
            "interpretation": "coverage 最接近 20% 的阈值。",
        },
    ]

    working_df = threshold_df.copy()
    working_df["coverage_distance"] = (working_df["coverage"] - 0.10).abs()
    working_df["coverage_distance_20"] = (working_df["coverage"] - 0.20).abs()

    rows: list[dict[str, Any]] = []
    metric_columns = [
        "threshold",
        "coverage",
        "precision",
        "recall",
        "f1",
        "lift",
        "selected_group_ctr",
        "click_capture_rate",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    for definition in definitions:
        row = pick_threshold_row(
            working_df,
            definition["mask"],
            definition["sort_columns"],
            definition["ascending"],
        )
        record: dict[str, Any] = {
            "operating_point": definition["operating_point"],
            "feasible": row is not None,
            "interpretation": definition["interpretation"],
        }

        if row is None:
            for column in metric_columns:
                record[column] = float("nan")
        else:
            for column in metric_columns:
                record[column] = float(row[column]) if column in row else float("nan")

        rows.append(record)

    return pd.DataFrame(rows)


def build_ranked_predictions(
    analysis_df: pd.DataFrame,
    selected_method: str,
) -> pd.DataFrame:
    """构建带排序与十分位的预测明细。"""

    ranked_df = sort_for_ranking(analysis_df)
    total_rows = len(ranked_df)

    ranked_df = ranked_df.copy()
    ranked_df["selected_method"] = selected_method
    ranked_df["probability_rank"] = np.arange(1, total_rows + 1, dtype=np.int64)
    ranked_df["probability_percentile"] = ranked_df["probability_rank"] / total_rows

    decile_map: dict[int, int] = {}
    start = 0
    split_indices = np.array_split(np.arange(total_rows), 10)
    for decile, index_array in enumerate(split_indices, start=1):
        for idx in index_array:
            decile_map[int(idx)] = decile

    ranked_df["score_decile"] = [
        decile_map[position] for position in range(total_rows)
    ]

    return ranked_df[
        [
            "row_position",
            "id",
            "click",
            "split_date",
            "selected_probability",
            "selected_method",
            "probability_rank",
            "probability_percentile",
            "score_decile",
        ]
    ]


def save_strategy_plots(
    threshold_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    overall_ctr: float,
    selected_method: str,
    plots_dir: Path,
) -> None:
    """保存策略分析图表。"""

    plots_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f"{selected_method}, development evaluation"

    prf_path = plots_dir / "threshold_precision_recall_f1.png"
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="f1")
    plt.xlabel("threshold")
    plt.ylabel("metric value")
    plt.title(f"Threshold vs Precision / Recall / F1 ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prf_path, dpi=160, bbox_inches="tight")
    plt.close()

    coverage_lift_path = plots_dir / "threshold_coverage_lift.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(threshold_df["threshold"], threshold_df["coverage"])
    axes[0].set_ylabel("coverage")
    axes[0].set_title(f"Threshold vs Coverage ({title_suffix})")
    axes[1].plot(threshold_df["threshold"], threshold_df["lift"])
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("lift")
    axes[1].set_title(f"Threshold vs Lift ({title_suffix})")
    plt.tight_layout()
    plt.savefig(coverage_lift_path, dpi=160, bbox_inches="tight")
    plt.close()

    topk_lift_path = plots_dir / "topk_lift_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(
        topk_df["target_coverage"],
        topk_df["lift"],
        marker="o",
    )
    plt.xlabel("Top-K coverage")
    plt.ylabel("lift")
    plt.title(f"Top-K Lift Curve ({title_suffix})")
    plt.tight_layout()
    plt.savefig(topk_lift_path, dpi=160, bbox_inches="tight")
    plt.close()

    gain_path = plots_dir / "cumulative_gain_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(
        topk_df["target_coverage"],
        topk_df["cumulative_click_capture_rate"],
        marker="o",
        label="model",
    )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="random baseline")
    plt.xlabel("coverage")
    plt.ylabel("cumulative click capture rate")
    plt.title(f"Cumulative Gain Curve ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(gain_path, dpi=160, bbox_inches="tight")
    plt.close()

    ctr_path = plots_dir / "topk_ctr_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(
        topk_df["target_coverage"],
        topk_df["selected_ctr"],
        marker="o",
        label="selected CTR",
    )
    plt.axhline(overall_ctr, color="gray", linestyle="--", label="overall CTR")
    plt.xlabel("coverage")
    plt.ylabel("selected group CTR")
    plt.title(f"Top-K CTR Curve ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ctr_path, dpi=160, bbox_inches="tight")
    plt.close()

    decile_ctr_path = plots_dir / "score_decile_ctr.png"
    plt.figure(figsize=(10, 6))
    plt.bar(decile_df["decile"].astype(str), decile_df["actual_ctr"])
    plt.axhline(overall_ctr, color="gray", linestyle="--", label="overall CTR")
    plt.xlabel("decile (1=highest score)")
    plt.ylabel("actual CTR")
    plt.title(f"Score Decile CTR ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(decile_ctr_path, dpi=160, bbox_inches="tight")
    plt.close()

    decile_lift_path = plots_dir / "score_decile_lift.png"
    plt.figure(figsize=(10, 6))
    plt.bar(decile_df["decile"].astype(str), decile_df["lift"])
    plt.axhline(1.0, color="gray", linestyle="--", label="lift=1")
    plt.xlabel("decile (1=highest score)")
    plt.ylabel("lift")
    plt.title(f"Score Decile Lift ({title_suffix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(decile_lift_path, dpi=160, bbox_inches="tight")
    plt.close()


def get_topk_row(topk_df: pd.DataFrame, target_coverage: float) -> pd.Series:
    """获取指定 coverage 的 Top-K 行。"""

    rows = topk_df.loc[np.isclose(topk_df["target_coverage"], target_coverage)]
    if rows.empty:
        raise ValueError(f"Top-K 表缺少 target_coverage={target_coverage}")
    return rows.iloc[0]


def build_summary_row(
    selected_method: str,
    evaluation_rows: int,
    prob_summary: ProbabilitySummary,
    threshold_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    operating_points_df: pd.DataFrame,
) -> dict[str, Any]:
    """构建 summary.csv 单行。"""

    max_f1_row = operating_points_df.loc[
        operating_points_df["operating_point"] == "maximum_f1"
    ].iloc[0]

    top1 = get_topk_row(topk_df, 0.01)
    top5 = get_topk_row(topk_df, 0.05)
    top10 = get_topk_row(topk_df, 0.10)
    top20 = get_topk_row(topk_df, 0.20)

    return {
        "selected_method": selected_method,
        "data_scope": DATA_SCOPE,
        "evaluation_rows": evaluation_rows,
        "overall_ctr": prob_summary.actual_ctr,
        "roc_auc": prob_summary.roc_auc,
        "average_precision": prob_summary.average_precision,
        "max_f1_threshold": max_f1_row["threshold"],
        "max_f1": max_f1_row["f1"],
        "max_f1_precision": max_f1_row["precision"],
        "max_f1_recall": max_f1_row["recall"],
        "top_1pct_ctr": top1["selected_ctr"],
        "top_1pct_lift": top1["lift"],
        "top_5pct_ctr": top5["selected_ctr"],
        "top_5pct_lift": top5["lift"],
        "top_10pct_ctr": top10["selected_ctr"],
        "top_10pct_lift": top10["lift"],
        "top_10pct_click_capture_rate": top10["cumulative_click_capture_rate"],
        "top_20pct_ctr": top20["selected_ctr"],
        "top_20pct_lift": top20["lift"],
        "top_20pct_click_capture_rate": top20["cumulative_click_capture_rate"],
        "holdout_used": False,
        "validation_passed": True,
    }


def write_text_report(
    report_path: Path,
    test_mode: bool,
    selected_method: str,
    prob_summary: ProbabilitySummary,
    operating_points_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> None:
    """写入中文策略报告。"""

    max_f1 = operating_points_df.loc[
        operating_points_df["operating_point"] == "maximum_f1"
    ].iloc[0]
    recall80 = operating_points_df.loc[
        operating_points_df["operating_point"] == "recall_at_least_80_percent"
    ].iloc[0]
    recall90 = operating_points_df.loc[
        operating_points_df["operating_point"] == "recall_at_least_90_percent"
    ].iloc[0]

    top1 = get_topk_row(topk_df, 0.01)
    top5 = get_topk_row(topk_df, 0.05)
    top10 = get_topk_row(topk_df, 0.10)
    top20 = get_topk_row(topk_df, 0.20)

    best_decile = decile_df.sort_values("actual_ctr", ascending=False).iloc[0]
    threshold_05 = threshold_df.loc[np.isclose(threshold_df["threshold"], 0.5)].iloc[0]

    lines = [
        "百度 CTR 项目 — 第 35 步 阈值与投放覆盖率策略分析报告",
        "=" * 72,
        "",
        f"【1. 使用的概率方法】{selected_method}",
        "",
        "【2. 为什么使用第 34 步 evaluation 预测】",
        "  最终校准器已在完整 valid 上重拟合，不能再在同一 valid 上评价。",
        "  本步骤仅使用第 34 步独立 development evaluation 子集上的预测概率。",
        "",
        f"【3. 分析样本】行数={prob_summary.rows:,}，actual CTR={prob_summary.actual_ctr:.6f}",
        "",
        f"【4. 区分能力】ROC-AUC={prob_summary.roc_auc:.6f}，"
        f"Average Precision={prob_summary.average_precision:.6f}",
        "",
        "【5. 阈值与投放覆盖率关系】",
        "  阈值越低，coverage 越高，能覆盖更多潜在点击，但目标人群 CTR 通常下降。",
        "",
        "【6. Precision 与 Recall 权衡】",
        "  提高阈值可提升 precision、降低 coverage；降低阈值则相反。",
        "",
        "【7. 最大 F1 阈值】",
        f"  threshold={max_f1['threshold']:.6f}, F1={max_f1['f1']:.6f}, "
        f"precision={max_f1['precision']:.6f}, recall={max_f1['recall']:.6f}",
        "",
        "【8. Recall >= 80% 候选点】",
        f"  feasible={recall80['feasible']}, threshold={recall80['threshold']}, "
        f"precision={recall80['precision']}, recall={recall80['recall']}",
        "",
        "【9. Recall >= 90% 候选点】",
        f"  feasible={recall90['feasible']}, threshold={recall90['threshold']}, "
        f"precision={recall90['precision']}, recall={recall90['recall']}",
        "",
        "【10. Top 1% / 5% / 10% / 20% 指标】",
        f"  Top 1%: CTR={top1['selected_ctr']:.6f}, lift={top1['lift']:.4f}, "
        f"capture={top1['cumulative_click_capture_rate']:.4f}",
        f"  Top 5%: CTR={top5['selected_ctr']:.6f}, lift={top5['lift']:.4f}, "
        f"capture={top5['cumulative_click_capture_rate']:.4f}",
        f"  Top 10%: CTR={top10['selected_ctr']:.6f}, lift={top10['lift']:.4f}, "
        f"capture={top10['cumulative_click_capture_rate']:.4f}",
        f"  Top 20%: CTR={top20['selected_ctr']:.6f}, lift={top20['lift']:.4f}, "
        f"capture={top20['cumulative_click_capture_rate']:.4f}",
        "",
        f"【11. CTR 最高的十分位】decile={int(best_decile['decile'])}, "
        f"CTR={best_decile['actual_ctr']:.6f}, lift={best_decile['lift']:.4f}",
        "",
        "【12. 高分区域是否集中点击】",
        f"  decile 1（最高分 10%）点击占比={decile_df.iloc[0]['share_of_total_clicks']:.4f}，"
        f"CTR={decile_df.iloc[0]['actual_ctr']:.6f}",
        "",
        "【13. 为什么 threshold=0.5 不适合当前 CTR 场景】",
        f"  在 0.5 阈值下 coverage={threshold_05['coverage']:.4f}，"
        f"recall={threshold_05['recall']:.4f}，precision={threshold_05['precision']:.4f}，"
        "对低 CTR 广告场景过于保守，会漏掉大量点击。",
        "",
        "【14. 为什么 Top-K 更适合预算控制】",
        "  Top-K 直接按预算/容量选择最高概率流量，比固定 0.5 更符合投放约束。",
        "",
        "【15. 不存在唯一最佳阈值】",
        "  追求更多点击覆盖应使用较低阈值；追求更高目标 CTR 应使用较高阈值。",
        "",
        "【16. 成本假设】当前分析默认每条曝光成本相同。",
        "",
        "【17. 缺少的数据】缺少广告收益、CPC、CPA、预算与库存约束。",
        "",
        "【18. Lift 不是利润】incremental_clicks_vs_random 仅表示相对随机投放的额外点击估计。",
        "",
        "【19. holdout 尚未使用】是",
        "",
        "【20. 下一步】更新 README、技术报告和本周周报。",
        "",
        f"当前模式：{'TEST_MODE=True' if test_mode else 'TEST_MODE=False'}",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_validation(
    paths: OutputPaths,
    analysis_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    prob_summary: ProbabilitySummary,
    selected_method: str,
    calibration_metadata: dict[str, Any],
    test_mode: bool,
) -> bool:
    """最终验收检查。"""

    expected_rows = TEST_ROWS if test_mode else FORMAL_EVAL_ROWS
    if len(analysis_df) != expected_rows:
        raise ValueError(
            f"分析行数 {len(analysis_df):,} 与期望 {expected_rows:,} 不一致。"
        )

    if calibration_metadata["best_development_method"] != selected_method:
        raise ValueError("selected method 与第 34 步不一致。")

    probabilities = analysis_df["selected_probability"].to_numpy(dtype=np.float64)
    clicks = analysis_df["click"].to_numpy(dtype=np.int8)

    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("selected probability 存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("selected probability 超出 [0, 1] 范围。")

    total_rows = len(clicks)
    total_clicks = int(clicks.sum())

    for _, row in threshold_df.iterrows():
        matrix_total = int(row["tp"] + row["fp"] + row["tn"] + row["fn"])
        if matrix_total != total_rows:
            raise ValueError("阈值混淆矩阵总数不正确。")

        if not np.isnan(row["coverage"]) and not (0.0 <= row["coverage"] <= 1.0):
            raise ValueError("coverage 超出 [0, 1]。")

        for metric_name in ("precision", "recall", "f1"):
            value = row[metric_name]
            if not np.isnan(value) and not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name} 超出 [0, 1] 或为非法值。")

        if not np.isnan(row["lift"]) and row["lift"] < 0:
            raise ValueError("lift 为负数。")

    if not topk_df["selected_rows"].is_monotonic_increasing:
        raise ValueError("Top-K selected_rows 未单调增加。")

    if not topk_df["cumulative_click_capture_rate"].is_monotonic_increasing:
        raise ValueError("Top-K click capture rate 未单调增加。")

    top100 = get_topk_row(topk_df, 1.0)
    if not np.isclose(top100["selected_ctr"], prob_summary.actual_ctr, rtol=1e-6, atol=1e-6):
        raise ValueError("100% coverage 的 CTR 不等于 overall CTR。")
    if not np.isclose(top100["lift"], 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("100% coverage 的 lift 不接近 1。")
    if not np.isclose(top100["cumulative_click_capture_rate"], 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("100% coverage 的点击捕获率不接近 1。")

    if int(decile_df["rows"].sum()) != total_rows:
        raise ValueError("decile 行数之和不等于分析行数。")

    if int(decile_df["clicks"].sum()) != total_clicks:
        raise ValueError("decile 点击数之和不等于总点击数。")

    required_outputs = [
        paths.threshold_metrics_csv,
        paths.topk_metrics_csv,
        paths.decile_csv,
        paths.operating_points_csv,
        paths.ranked_predictions,
        paths.summary_csv,
        paths.report_txt,
        paths.plots_dir / "cumulative_gain_curve.png",
        paths.plots_dir / "topk_lift_curve.png",
    ]
    for output_path in required_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"缺少输出文件：{output_path}")

    return True


def main() -> None:
    """主流程：读取 evaluation 预测 → 阈值 / Top-K / Lift 分析 → 保存结果。"""

    paths = get_output_paths(TEST_MODE)

    print("=" * 72)
    print("第 35 步：阈值、Lift 与投放覆盖率策略分析")
    print("=" * 72)
    print(f"TEST_MODE：{TEST_MODE}")

    calibration_metadata = load_calibration_metadata(CALIBRATION_METADATA_PATH)
    selected_method = str(calibration_metadata["best_development_method"])
    probability_column = resolve_probability_column(selected_method)

    _ = load_calibration_metrics(CALIBRATION_METRICS_PATH, selected_method)

    print(f"selected method：{selected_method}")
    print(f"probability column：{probability_column}")

    print("\n读取 evaluation 预测 ...")
    evaluation_df = load_evaluation_predictions(
        EVALUATION_PREDICTIONS_PATH,
        probability_column,
    )

    analysis_df = select_analysis_subset(evaluation_df, TEST_MODE)
    probabilities = analysis_df["selected_probability"].to_numpy(dtype=np.float64)
    clicks = analysis_df["click"].to_numpy(dtype=np.int8)

    print(f"分析行数：{len(analysis_df):,}")
    print(f"actual CTR：{clicks.mean():.6f}")

    print("\n计算基础指标 ...")
    prob_summary = compute_probability_summary(probabilities, clicks)

    print("\n构建阈值网格并扫描 ...")
    thresholds = build_threshold_grid(probabilities)
    threshold_df = compute_threshold_metrics(probabilities, clicks, thresholds)
    print(f"阈值数量：{len(threshold_df):,}")

    ranked_source = sort_for_ranking(analysis_df)

    print("\n计算 Top-K 投放指标 ...")
    topk_df = compute_topk_metrics(ranked_source, prob_summary.actual_ctr)

    print("计算十分位分析 ...")
    decile_df = compute_decile_analysis(ranked_source, prob_summary.actual_ctr)

    print("构建候选运行点 ...")
    operating_points_df = build_operating_points(threshold_df)

    ranked_predictions_df = build_ranked_predictions(
        analysis_df,
        selected_method,
    )

    paths.strategy_dir.mkdir(parents=True, exist_ok=True)
    paths.ranked_predictions.parent.mkdir(parents=True, exist_ok=True)

    threshold_df.to_csv(paths.threshold_metrics_csv, index=False, encoding="utf-8")
    topk_df.to_csv(paths.topk_metrics_csv, index=False, encoding="utf-8")
    decile_df.to_csv(paths.decile_csv, index=False, encoding="utf-8")
    operating_points_df.to_csv(paths.operating_points_csv, index=False, encoding="utf-8")
    ranked_predictions_df.to_parquet(paths.ranked_predictions, index=False)

    print("\n保存图表 ...")
    save_strategy_plots(
        threshold_df,
        topk_df,
        decile_df,
        prob_summary.actual_ctr,
        selected_method,
        paths.plots_dir,
    )

    summary_row = build_summary_row(
        selected_method,
        len(analysis_df),
        prob_summary,
        threshold_df,
        topk_df,
        operating_points_df,
    )
    pd.DataFrame([summary_row]).to_csv(
        paths.summary_csv,
        index=False,
        encoding="utf-8",
    )

    write_text_report(
        paths.report_txt,
        TEST_MODE,
        selected_method,
        prob_summary,
        operating_points_df,
        topk_df,
        decile_df,
        threshold_df,
    )

    metadata_payload = {
        "script_name": "scripts/35_analyze_threshold_lift_strategy.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "random_state": RANDOM_STATE,
        "selected_method": selected_method,
        "selected_probability_column": probability_column,
        "source_prediction_path": str(EVALUATION_PREDICTIONS_PATH),
        "source_calibration_metadata": str(CALIBRATION_METADATA_PATH),
        "data_scope": DATA_SCOPE,
        "evaluation_rows": len(analysis_df),
        "evaluation_ctr": prob_summary.actual_ctr,
        "probability_summary": {
            "rows": prob_summary.rows,
            "clicks": prob_summary.clicks,
            "actual_ctr": prob_summary.actual_ctr,
            "mean_predicted_ctr": prob_summary.mean_predicted_ctr,
            "roc_auc": prob_summary.roc_auc,
            "average_precision": prob_summary.average_precision,
            "log_loss": prob_summary.log_loss,
            "brier_score": prob_summary.brier_score,
            "minimum_probability": prob_summary.minimum_probability,
            "maximum_probability": prob_summary.maximum_probability,
            "probability_quantiles": prob_summary.probability_quantiles,
        },
        "threshold_count": int(len(threshold_df)),
        "threshold_range": {
            "min": float(threshold_df["threshold"].min()),
            "max": float(threshold_df["threshold"].max()),
        },
        "topk_coverages": TOPK_COVERAGES,
        "decile_count": 10,
        "operating_points": operating_points_df.to_dict(orient="records"),
        "output_paths": {
            "threshold_metrics_csv": str(paths.threshold_metrics_csv),
            "topk_metrics_csv": str(paths.topk_metrics_csv),
            "decile_csv": str(paths.decile_csv),
            "operating_points_csv": str(paths.operating_points_csv),
            "ranked_predictions": str(paths.ranked_predictions),
            "summary_csv": str(paths.summary_csv),
            "report_txt": str(paths.report_txt),
            "plots_dir": str(paths.plots_dir),
        },
        "limitations": [
            "当前结果属于 development evaluation",
            "LightGBM 参数曾使用完整 valid 选择",
            "最终校准器尚未在 holdout 上评价",
            "Top-K 默认每条曝光成本相同",
            "缺少真实收益与投放成本数据",
            "阈值不能仅根据 F1 确定",
        ],
        "holdout_used": False,
        "validation_passed": False,
    }

    validation_passed = final_validation(
        paths=paths,
        analysis_df=analysis_df,
        threshold_df=threshold_df,
        topk_df=topk_df,
        decile_df=decile_df,
        prob_summary=prob_summary,
        selected_method=selected_method,
        calibration_metadata=calibration_metadata,
        test_mode=TEST_MODE,
    )
    metadata_payload["validation_passed"] = validation_passed

    paths.metadata_json.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    max_f1 = operating_points_df.loc[
        operating_points_df["operating_point"] == "maximum_f1"
    ].iloc[0]
    recall80 = operating_points_df.loc[
        operating_points_df["operating_point"] == "recall_at_least_80_percent"
    ].iloc[0]
    top1 = get_topk_row(topk_df, 0.01)
    top5 = get_topk_row(topk_df, 0.05)
    top10 = get_topk_row(topk_df, 0.10)
    top20 = get_topk_row(topk_df, 0.20)

    print("\n" + "=" * 72)
    print("第 35 步完成摘要")
    print("=" * 72)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"selected method：{selected_method}")
    print(f"evaluation 行数：{len(analysis_df):,}")
    print(f"actual CTR：{prob_summary.actual_ctr:.6f}")
    print(f"ROC-AUC：{prob_summary.roc_auc:.6f}")
    print(f"Average Precision：{prob_summary.average_precision:.6f}")
    print(
        f"最大 F1：threshold={max_f1['threshold']:.6f}, "
        f"F1={max_f1['f1']:.6f}, precision={max_f1['precision']:.6f}, "
        f"recall={max_f1['recall']:.6f}"
    )
    print(
        f"Recall>=80% 候选：feasible={recall80['feasible']}, "
        f"threshold={recall80['threshold']}, precision={recall80['precision']}, "
        f"recall={recall80['recall']}"
    )
    print("\nTop-K 指标：")
    for label, row in [
        ("Top 1%", top1),
        ("Top 5%", top5),
        ("Top 10%", top10),
        ("Top 20%", top20),
    ]:
        print(
            f"  {label}: CTR={row['selected_ctr']:.6f}, lift={row['lift']:.4f}, "
            f"capture={row['cumulative_click_capture_rate']:.4f}"
        )

    print("\n最重要的业务结论：")
    print(
        "  模型可将高点击流量集中到高分区域，但不存在脱离业务目标的唯一最佳阈值；"
        "Top-K 投放比固定 0.5 阈值更适合预算控制。"
    )
    print("\n输出路径：")
    print(f"  threshold metrics：{paths.threshold_metrics_csv}")
    print(f"  topk metrics：{paths.topk_metrics_csv}")
    print(f"  decile analysis：{paths.decile_csv}")
    print(f"  operating points：{paths.operating_points_csv}")
    print(f"  ranked predictions：{paths.ranked_predictions}")
    print(f"  summary：{paths.summary_csv}")
    print(f"  report：{paths.report_txt}")
    print(f"  metadata：{paths.metadata_json}")
    print(f"  plots：{paths.plots_dir}")
    print(f"holdout_used：False")
    print(f"validation_passed：{validation_passed}")
    print("阈值与投放覆盖率分析完成，holdout 尚未使用。")


if __name__ == "__main__":
    main()
