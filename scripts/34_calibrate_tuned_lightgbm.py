"""
百度 CTR 项目 — 调优 LightGBM 概率校准

功能：
    对第 32 步 Optuna 调优后的 LightGBM 验证集预测概率进行校准，比较
    uncalibrated / sigmoid / isotonic 三种方法，选出开发阶段最佳方法并
    保存最终校准器。禁止读取 holdout，不重新训练 LightGBM。

数据输入：
    outputs/predictions/tuned_lightgbm_valid_predictions.parquet
    outputs/fixed_tuning_sample_metadata.json
    outputs/lightgbm_optuna_metadata.json
    models/tuned_lightgbm_optuna_model.joblib

用法：
    python scripts/34_calibrate_tuned_lightgbm.py
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

RANDOM_STATE = 42
CALIBRATION_RATIO = 0.60
N_CALIBRATION_BINS = 20
PROBABILITY_EPSILON = 1e-7
THRESHOLD = 0.5

TEST_TOTAL_ROWS = 100_000
FORMAL_TOTAL_ROWS = 500_000
FULL_VALID_ROWS = 500_000
EXPECTED_FEATURE_COUNT = 33

MEAN_PROB_TOLERANCE = 0.01
CTR_DIFF_TOLERANCE = 0.02
ECE_TOLERANCE = 1e-9

DATA_SCOPE = "development_calibration_evaluation_split"
METHODS = ("uncalibrated", "sigmoid", "isotonic")

PREDICTIONS_PATH = Path(
    "outputs/predictions/tuned_lightgbm_valid_predictions.parquet"
)
FIXED_SAMPLE_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")
OPTUNA_METADATA_PATH = Path("outputs/lightgbm_optuna_metadata.json")
MODEL_PATH = Path("models/tuned_lightgbm_optuna_model.joblib")

SELECTION_PRIORITY = [
    "log_loss",
    "brier_score",
    "expected_calibration_error",
    "calibration_gap",
]


@dataclass
class OutputPaths:
    """第 34 步输出路径。"""

    test_mode: bool
    calibration_dir: Path
    plots_dir: Path
    bins_csv: Path
    evaluation_predictions: Path
    metrics_csv: Path
    comparison_csv: Path
    report_txt: Path
    metadata_json: Path
    sigmoid_calibrator: Path
    isotonic_calibrator: Path
    selected_calibrator: Path


@dataclass
class MethodMetrics:
    """单校准方法 evaluation 指标。"""

    method: str
    data_scope: str
    calibration_rows: int
    evaluation_rows: int
    roc_auc: float
    log_loss: float
    brier_score: float
    actual_ctr: float
    mean_predicted_ctr: float
    calibration_gap: float
    expected_calibration_error: float
    maximum_calibration_error: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    predicted_click_ratio_at_threshold: float
    threshold: float
    selected_as_best: bool
    holdout_used: bool = False


@dataclass
class SplitInfo:
    """calibration / evaluation 拆分信息。"""

    total_rows: int
    calibration_rows: int
    evaluation_rows: int
    calibration_ctr: float
    evaluation_ctr: float
    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    calibration_indices_sha256: str
    evaluation_indices_sha256: str


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    suffix = "_test" if test_mode else ""
    calibration_dir = Path("outputs/calibration")
    plots_dir = calibration_dir / ("plots_test" if test_mode else "plots")
    model_dir = Path("models/test") if test_mode else Path("models")

    return OutputPaths(
        test_mode=test_mode,
        calibration_dir=calibration_dir,
        plots_dir=plots_dir,
        bins_csv=calibration_dir / f"probability_calibration_bins{suffix}.csv",
        evaluation_predictions=Path(
            f"outputs/predictions/probability_calibration_evaluation_predictions{suffix}.parquet"
        ),
        metrics_csv=Path(f"outputs/probability_calibration_metrics{suffix}.csv"),
        comparison_csv=Path(f"outputs/probability_calibration_comparison{suffix}.csv"),
        report_txt=Path(f"outputs/probability_calibration_report{suffix}.txt"),
        metadata_json=Path(f"outputs/probability_calibration_metadata{suffix}.json"),
        sigmoid_calibrator=model_dir / "tuned_lightgbm_sigmoid_calibrator.joblib",
        isotonic_calibrator=model_dir / "tuned_lightgbm_isotonic_calibrator.joblib",
        selected_calibrator=model_dir / "tuned_lightgbm_selected_calibrator.joblib",
    )


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    if not path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixed_sample_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取并校验固定样本元数据。"""

    metadata = load_json(metadata_path)

    if int(metadata.get("actual_valid_rows", -1)) != FULL_VALID_ROWS:
        raise ValueError(
            f"actual_valid_rows={metadata.get('actual_valid_rows')!r}，"
            f"必须为 {FULL_VALID_ROWS:,}。"
        )

    if int(metadata.get("feature_count", -1)) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature_count={metadata.get('feature_count')!r}，"
            f"必须为 {EXPECTED_FEATURE_COUNT}。"
        )

    if metadata.get("holdout_used") is not False:
        raise ValueError("固定样本 holdout_used 必须为 false。")

    if metadata.get("validation_passed") is not True:
        raise ValueError("固定样本 validation_passed 必须为 true。")

    valid_sha256 = str(metadata.get("valid_id_sha256", ""))
    if len(valid_sha256) != 64:
        raise ValueError("valid_id_sha256 长度必须为 64。")

    return metadata


def load_optuna_metadata(
    metadata_path: Path,
    fixed_metadata: dict[str, Any],
) -> dict[str, Any]:
    """读取并校验 Optuna 调优元数据。"""

    metadata = load_json(metadata_path)

    if metadata.get("test_mode") is not False:
        raise ValueError("Optuna 元数据 test_mode 必须为 false（正式调优结果）。")

    if metadata.get("holdout_used") is not False:
        raise ValueError("Optuna holdout_used 必须为 false。")

    if metadata.get("validation_passed") is not True:
        raise ValueError("Optuna validation_passed 必须为 true。")

    if metadata.get("valid_id_sha256") != fixed_metadata["valid_id_sha256"]:
        raise ValueError("Optuna valid_id_sha256 与固定样本元数据不一致。")

    if "tuned_metrics" not in metadata:
        raise KeyError("Optuna 元数据缺少 tuned_metrics。")

    model_path = Path(metadata["model_output_path"])
    prediction_path = Path(metadata["prediction_output_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"调优模型不存在：{model_path}")

    if not prediction_path.exists():
        raise FileNotFoundError(f"调优预测文件不存在：{prediction_path}")

    return metadata


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """裁剪概率到开区间 (0, 1)。"""

    return np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)


def probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    """将概率转换为 logit。"""

    clipped = clip_probabilities(probabilities)
    return np.log(clipped / (1.0 - clipped))


def calculate_indices_sha256(indices: np.ndarray) -> str:
    """计算行位置索引 SHA256。"""

    hasher = hashlib.sha256()
    for index_value in np.sort(indices.astype(np.int64)):
        hasher.update((str(int(index_value)) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def load_and_validate_predictions(
    prediction_path: Path,
    expected_mean_probability: float,
) -> pd.DataFrame:
    """读取并校验调优模型验证预测。"""

    if not prediction_path.exists():
        raise FileNotFoundError(f"未找到预测文件：{prediction_path}")

    predictions_df = pd.read_parquet(prediction_path)

    required_columns = ["id", "click", "split_date", "tuned_lightgbm_probability"]
    missing = [column for column in required_columns if column not in predictions_df.columns]
    if missing:
        raise ValueError(f"预测文件缺少列：{missing}")

    if len(predictions_df) != FULL_VALID_ROWS:
        raise ValueError(
            f"预测文件行数 {len(predictions_df):,} 不等于 {FULL_VALID_ROWS:,}。"
        )

    clicks = predictions_df["click"].to_numpy()
    if not np.isin(clicks, [0, 1]).all():
        raise ValueError("click 不是仅包含 0 和 1。")

    if predictions_df["id"].isna().any():
        raise ValueError("id 存在缺失。")

    if predictions_df["split_date"].isna().any():
        raise ValueError("split_date 存在缺失。")

    probabilities = predictions_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("tuned_lightgbm_probability 存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("tuned_lightgbm_probability 超出 [0, 1] 范围。")

    mean_probability = float(probabilities.mean())
    if abs(mean_probability - expected_mean_probability) > MEAN_PROB_TOLERANCE:
        raise ValueError(
            f"平均预测概率 {mean_probability:.6f} 与 Optuna 元数据 "
            f"{expected_mean_probability:.6f} 差异超过 {MEAN_PROB_TOLERANCE}。"
        )

    predictions_df = predictions_df.copy()
    predictions_df["row_position"] = np.arange(len(predictions_df), dtype=np.int64)
    return predictions_df


def select_working_subset(
    predictions_df: pd.DataFrame,
    test_mode: bool,
) -> pd.DataFrame:
    """测试模式分层抽样，正式模式使用完整 valid。"""

    if not test_mode:
        return predictions_df.copy()

    sample_indices, _ = train_test_split(
        predictions_df["row_position"].to_numpy(),
        train_size=TEST_TOTAL_ROWS,
        stratify=predictions_df["click"].to_numpy(),
        random_state=RANDOM_STATE,
    )
    sample_indices = np.sort(sample_indices.astype(np.int64))
    return predictions_df.loc[sample_indices].copy().reset_index(drop=True)


def split_calibration_evaluation(
    working_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """按 click 分层拆分 calibration / evaluation。"""

    row_positions = working_df["row_position"].to_numpy()
    clicks = working_df["click"].to_numpy()

    calibration_indices, evaluation_indices = train_test_split(
        row_positions,
        train_size=CALIBRATION_RATIO,
        stratify=clicks,
        random_state=RANDOM_STATE,
    )

    calibration_indices = np.sort(calibration_indices.astype(np.int64))
    evaluation_indices = np.sort(evaluation_indices.astype(np.int64))

    overlap = set(calibration_indices.tolist()) & set(evaluation_indices.tolist())
    if overlap:
        raise ValueError("calibration 与 evaluation 存在重叠行位置。")

    if len(calibration_indices) + len(evaluation_indices) != len(working_df):
        raise ValueError("calibration 与 evaluation 行数之和不等于输入行数。")

    calibration_df = working_df.loc[
        working_df["row_position"].isin(calibration_indices)
    ].copy()
    evaluation_df = working_df.loc[
        working_df["row_position"].isin(evaluation_indices)
    ].copy()

    calibration_df = calibration_df.sort_values("row_position").reset_index(drop=True)
    evaluation_df = evaluation_df.sort_values("row_position").reset_index(drop=True)

    split_info = SplitInfo(
        total_rows=len(working_df),
        calibration_rows=len(calibration_df),
        evaluation_rows=len(evaluation_df),
        calibration_ctr=float(calibration_df["click"].mean()),
        evaluation_ctr=float(evaluation_df["click"].mean()),
        calibration_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        calibration_indices_sha256=calculate_indices_sha256(calibration_indices),
        evaluation_indices_sha256=calculate_indices_sha256(evaluation_indices),
    )

    total_ctr = float(working_df["click"].mean())
    if abs(split_info.calibration_ctr - total_ctr) > CTR_DIFF_TOLERANCE:
        warnings.warn(
            f"calibration CTR {split_info.calibration_ctr:.6f} 与总 CTR "
            f"{total_ctr:.6f} 差异偏大。",
            stacklevel=2,
        )

    if abs(split_info.evaluation_ctr - total_ctr) > CTR_DIFF_TOLERANCE:
        warnings.warn(
            f"evaluation CTR {split_info.evaluation_ctr:.6f} 与总 CTR "
            f"{total_ctr:.6f} 差异偏大。",
            stacklevel=2,
        )

    return calibration_df, evaluation_df, split_info


def fit_sigmoid_calibrator(
    calibration_probabilities: np.ndarray,
    calibration_clicks: np.ndarray,
) -> LogisticRegression:
    """在 calibration 数据上拟合 Sigmoid 校准器。"""

    logit_scores = probability_to_logit(calibration_probabilities)
    model = LogisticRegression(
        solver="lbfgs",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    model.fit(logit_scores.reshape(-1, 1), calibration_clicks)
    return model


def apply_sigmoid_calibrator(
    model: LogisticRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    """应用 Sigmoid 校准器。"""

    logit_scores = probability_to_logit(probabilities)
    calibrated = model.predict_proba(logit_scores.reshape(-1, 1))[:, 1]
    return clip_probabilities(calibrated.astype(np.float64))


def fit_isotonic_calibrator(
    calibration_probabilities: np.ndarray,
    calibration_clicks: np.ndarray,
) -> IsotonicRegression:
    """在 calibration 数据上拟合 Isotonic 校准器。"""

    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        increasing=True,
        out_of_bounds="clip",
    )
    model.fit(calibration_probabilities, calibration_clicks)
    return model


def apply_isotonic_calibrator(
    model: IsotonicRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    """应用 Isotonic 校准器。"""

    calibrated = model.predict(probabilities)
    return clip_probabilities(calibrated.astype(np.float64))


def check_isotonic_monotonicity(
    model: IsotonicRegression,
    probabilities: np.ndarray,
) -> None:
    """检查 Isotonic 映射单调不下降。"""

    sorted_probs = np.sort(probabilities)
    transformed = model.predict(sorted_probs)
    if np.any(np.diff(transformed) < -1e-12):
        raise ValueError("Isotonic 校准输出未保持单调不下降。")


def compute_calibration_bins(
    method: str,
    probabilities: np.ndarray,
    clicks: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> pd.DataFrame:
    """计算等样本量分箱校准结果。"""

    if len(probabilities) != len(clicks):
        raise ValueError("概率与标签长度不一致。")

    order = np.argsort(probabilities)
    sorted_probs = probabilities[order]
    sorted_clicks = clicks[order]
    split_indices = np.array_split(np.arange(len(probabilities)), n_bins)

    rows: list[dict[str, Any]] = []
    cumulative_rows = 0

    for bin_number, bin_index_array in enumerate(split_indices, start=1):
        if len(bin_index_array) == 0:
            continue

        bin_probs = sorted_probs[bin_index_array]
        bin_clicks = sorted_clicks[bin_index_array]
        bin_rows = len(bin_index_array)
        cumulative_rows += bin_rows

        mean_predicted = float(bin_probs.mean())
        actual_rate = float(bin_clicks.mean())
        absolute_gap = abs(mean_predicted - actual_rate)
        weighted_gap = absolute_gap * bin_rows / len(probabilities)

        rows.append(
            {
                "method": method,
                "bin_number": bin_number,
                "bin_rows": bin_rows,
                "probability_min": float(bin_probs.min()),
                "probability_max": float(bin_probs.max()),
                "mean_predicted_probability": mean_predicted,
                "actual_click_rate": actual_rate,
                "absolute_gap": absolute_gap,
                "weighted_gap": weighted_gap,
                "cumulative_rows": cumulative_rows,
            }
        )

    return pd.DataFrame(rows)


def compute_ece_mce(bin_df: pd.DataFrame) -> tuple[float, float]:
    """由分箱结果计算 ECE 与 MCE。"""

    ece = float(bin_df["weighted_gap"].sum())
    mce = float(bin_df["absolute_gap"].max())
    return ece, mce


def compute_method_metrics(
    method: str,
    probabilities: np.ndarray,
    clicks: np.ndarray,
    calibration_rows: int,
    evaluation_rows: int,
    selected_as_best: bool,
) -> tuple[MethodMetrics, pd.DataFrame]:
    """计算单方法 evaluation 指标与分箱表。"""

    clipped = clip_probabilities(probabilities)
    predicted_labels = (probabilities >= THRESHOLD).astype(np.int8)

    bin_df = compute_calibration_bins(method, probabilities, clicks)
    ece, mce = compute_ece_mce(bin_df)

    actual_ctr = float(clicks.mean())
    mean_predicted_ctr = float(probabilities.mean())

    metrics = MethodMetrics(
        method=method,
        data_scope=DATA_SCOPE,
        calibration_rows=calibration_rows,
        evaluation_rows=evaluation_rows,
        roc_auc=float(roc_auc_score(clicks, probabilities)),
        log_loss=float(log_loss(clicks, clipped, labels=[0, 1])),
        brier_score=float(brier_score_loss(clicks, probabilities)),
        actual_ctr=actual_ctr,
        mean_predicted_ctr=mean_predicted_ctr,
        calibration_gap=abs(mean_predicted_ctr - actual_ctr),
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        accuracy=float(accuracy_score(clicks, predicted_labels)),
        precision=float(precision_score(clicks, predicted_labels, zero_division=0)),
        recall=float(recall_score(clicks, predicted_labels, zero_division=0)),
        f1=float(f1_score(clicks, predicted_labels, zero_division=0)),
        predicted_click_ratio_at_threshold=float(predicted_labels.mean()),
        threshold=THRESHOLD,
        selected_as_best=selected_as_best,
        holdout_used=False,
    )
    return metrics, bin_df


def method_metrics_to_row(metrics: MethodMetrics) -> dict[str, Any]:
    """MethodMetrics 转 CSV 行。"""

    return {
        "method": metrics.method,
        "data_scope": metrics.data_scope,
        "calibration_rows": metrics.calibration_rows,
        "evaluation_rows": metrics.evaluation_rows,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "brier_score": metrics.brier_score,
        "actual_ctr": metrics.actual_ctr,
        "mean_predicted_ctr": metrics.mean_predicted_ctr,
        "calibration_gap": metrics.calibration_gap,
        "expected_calibration_error": metrics.expected_calibration_error,
        "maximum_calibration_error": metrics.maximum_calibration_error,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "predicted_click_ratio_at_threshold": metrics.predicted_click_ratio_at_threshold,
        "threshold": metrics.threshold,
        "selected_as_best": metrics.selected_as_best,
        "holdout_used": metrics.holdout_used,
    }


def select_best_method(metrics_list: list[MethodMetrics]) -> tuple[str, str]:
    """按 evaluation 指标选择最佳校准方法。"""

    sorted_metrics = sorted(
        metrics_list,
        key=lambda item: (
            item.log_loss,
            item.brier_score,
            item.expected_calibration_error,
            item.calibration_gap,
        ),
    )
    best = sorted_metrics[0]

    raw_metrics = next(item for item in metrics_list if item.method == "uncalibrated")
    improvements: list[str] = []

    if best.method != "uncalibrated":
        if best.log_loss < raw_metrics.log_loss:
            improvements.append("LogLoss 改善")
        else:
            improvements.append("LogLoss 未改善")

        if best.brier_score < raw_metrics.brier_score:
            improvements.append("Brier Score 改善")
        else:
            improvements.append("Brier Score 未改善")

        if best.expected_calibration_error < raw_metrics.expected_calibration_error:
            improvements.append("ECE 改善")
        else:
            improvements.append("ECE 未改善")

        if best.calibration_gap < raw_metrics.calibration_gap:
            improvements.append("calibration_gap 改善")
        else:
            improvements.append("calibration_gap 未改善")
    else:
        improvements.append("未校准方法在 development evaluation 上最优")

    reason = (
        f"按优先级 {SELECTION_PRIORITY} 选择 {best.method}；"
        + "；".join(improvements)
    )
    return best.method, reason


def build_comparison_dataframe(
    metrics_list: list[MethodMetrics],
    best_method: str,
    selection_reason: str,
) -> pd.DataFrame:
    """构建方法对比表。"""

    rows = [method_metrics_to_row(item) for item in metrics_list]
    comparison_df = pd.DataFrame(rows)

    raw_row = comparison_df.loc[comparison_df["method"] == "uncalibrated"].iloc[0]
    best_row = comparison_df.loc[comparison_df["method"] == best_method].iloc[0]

    comparison_df["selected_as_best"] = comparison_df["method"] == best_method
    comparison_df["selection_reason"] = ""
    comparison_df.loc[comparison_df["method"] == best_method, "selection_reason"] = (
        selection_reason
    )

    comparison_df["raw_to_best_logloss_change"] = (
        best_row["log_loss"] - raw_row["log_loss"]
    )
    comparison_df["raw_to_best_brier_change"] = (
        best_row["brier_score"] - raw_row["brier_score"]
    )
    comparison_df["raw_to_best_ece_change"] = (
        best_row["expected_calibration_error"]
        - raw_row["expected_calibration_error"]
    )
    comparison_df["raw_to_best_calibration_gap_change"] = (
        best_row["calibration_gap"] - raw_row["calibration_gap"]
    )

    return comparison_df


def save_calibration_plots(
    bins_df: pd.DataFrame,
    evaluation_predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """保存校准相关图表。"""

    plots_dir.mkdir(parents=True, exist_ok=True)

    curve_path = plots_dir / "probability_calibration_curve.png"
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")

    for method in METHODS:
        method_bins = bins_df.loc[bins_df["method"] == method]
        plt.plot(
            method_bins["mean_predicted_probability"],
            method_bins["actual_click_rate"],
            marker="o",
            label=method,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual click rate")
    plt.title("Probability Calibration Curve (development evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160, bbox_inches="tight")
    plt.close()

    distribution_path = plots_dir / "probability_calibration_distribution.png"
    plt.figure(figsize=(10, 6))
    for method, column in (
        ("uncalibrated", "raw_probability"),
        ("sigmoid", "sigmoid_probability"),
        ("isotonic", "isotonic_probability"),
    ):
        plt.hist(
            evaluation_predictions[column].to_numpy(),
            bins=50,
            alpha=0.45,
            density=True,
            label=method,
        )
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Calibrated Probability Distribution (development evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=160, bbox_inches="tight")
    plt.close()

    metric_path = plots_dir / "probability_calibration_metric_comparison.png"
    metric_names = [
        "log_loss",
        "brier_score",
        "expected_calibration_error",
        "calibration_gap",
    ]
    x_positions = np.arange(len(metric_names))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    for offset, method in enumerate(METHODS):
        method_row = metrics_df.loc[metrics_df["method"] == method].iloc[0]
        values = [method_row[name] for name in metric_names]
        plt.bar(
            x_positions + offset * bar_width,
            values,
            width=bar_width,
            label=method,
        )

    plt.xticks(
        x_positions + bar_width,
        ["LogLoss", "Brier Score", "ECE", "Calibration Gap"],
    )
    plt.title("Calibration Method Metric Comparison (development evaluation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path, dpi=160, bbox_inches="tight")
    plt.close()


def fit_final_selected_calibrator(
    best_method: str,
    full_probabilities: np.ndarray,
    full_clicks: np.ndarray,
) -> Any:
    """在完整 working valid 上拟合最终校准器。"""

    if best_method == "uncalibrated":
        return {"method": "uncalibrated"}

    if best_method == "sigmoid":
        return fit_sigmoid_calibrator(full_probabilities, full_clicks)

    if best_method == "isotonic":
        return fit_isotonic_calibrator(full_probabilities, full_clicks)

    raise ValueError(f"未知 best_method：{best_method}")


def extract_sigmoid_parameters(model: LogisticRegression) -> dict[str, Any]:
    """提取 Sigmoid 校准器参数。"""

    return {
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "n_iter": int(model.n_iter_[0]) if model.n_iter_ is not None else None,
        "converged": bool(model.n_iter_[0] < 1000) if model.n_iter_ is not None else None,
    }


def extract_isotonic_parameters(model: IsotonicRegression) -> dict[str, Any]:
    """提取 Isotonic 校准器参数摘要。"""

    return {
        "X_min": float(model.X_min_) if model.X_min_ is not None else None,
        "X_max": float(model.X_max_) if model.X_max_ is not None else None,
        "n_knots": int(len(model.X_thresholds_)) if hasattr(model, "X_thresholds_") else None,
        "increasing": bool(model.increasing),
        "out_of_bounds": model.out_of_bounds,
    }


def write_text_report(
    report_path: Path,
    test_mode: bool,
    optuna_metadata: dict[str, Any],
    split_info: SplitInfo,
    metrics_list: list[MethodMetrics],
    best_method: str,
    selection_reason: str,
    final_calibrator_method: str,
    final_calibrator_path: Path,
) -> None:
    """写入中文校准报告。"""

    tuned_metrics = optuna_metadata["tuned_metrics"]
    raw_metrics = next(item for item in metrics_list if item.method == "uncalibrated")
    best_metrics = next(item for item in metrics_list if item.method == best_method)

    def fmt_metrics(metrics: MethodMetrics) -> list[str]:
        return [
            f"    LogLoss={metrics.log_loss:.6f}",
            f"    Brier={metrics.brier_score:.6f}",
            f"    ECE={metrics.expected_calibration_error:.6f}",
            f"    calibration_gap={metrics.calibration_gap:.6f}",
            f"    AUC={metrics.roc_auc:.6f}",
        ]

    lines = [
        "百度 CTR 项目 — 第 34 步 调优 LightGBM 概率校准报告",
        "=" * 72,
        "",
        "【1. 概率校准是什么】",
        "  概率校准调整模型输出，使预测概率更接近真实点击发生率。",
        "",
        "【2. 为什么调优模型需要校准】",
        "  树模型优化 LogLoss / AUC 时，概率绝对值可能偏离真实 CTR，"
        "影响阈值决策与投放覆盖率分析。",
        "",
        "【3. 原始调优模型 valid 指标】",
        f"  AUC={tuned_metrics['roc_auc']:.6f}",
        f"  LogLoss={tuned_metrics['log_loss']:.6f}",
        f"  calibration_gap={tuned_metrics['calibration_gap']:.6f}",
        "",
        "【4. 为什么不能在同一批数据上拟合并评价】",
        "  校准器若在 evaluation 标签上拟合会造成信息泄露；"
        "因此仅在 calibration 上拟合，在 evaluation 上比较方法。",
        "",
        "【5. valid 拆分方法】",
        f"  train_test_split(stratify=click, train_size={CALIBRATION_RATIO}, "
        f"random_state={RANDOM_STATE})",
        "",
        "【6. 拆分规模与 CTR】",
        f"  总 valid 行数：{split_info.total_rows:,}",
        f"  calibration 行数：{split_info.calibration_rows:,}，CTR={split_info.calibration_ctr:.6f}",
        f"  evaluation 行数：{split_info.evaluation_rows:,}，CTR={split_info.evaluation_ctr:.6f}",
        f"  随机种子：{RANDOM_STATE}",
        "",
        "【7. Sigmoid 方法】",
        "  将 clipped 概率转 logit，再用 LogisticRegression 在 calibration 上拟合。",
        "",
        "【8. Isotonic 方法】",
        "  使用 IsotonicRegression 在 calibration 原始概率与 click 上拟合单调映射。",
        "",
        "【9. 三种方法 evaluation 指标】",
    ]

    for metrics in metrics_list:
        lines.append(f"  {metrics.method}:")
        lines.extend(fmt_metrics(metrics))
        lines.append("")

    lines.extend(
        [
            f"【10. 最佳开发阶段方法】{best_method}",
            f"【11. 选择原因】{selection_reason}",
            "",
            f"【12. LogLoss 是否改善】"
            f"{'是' if best_metrics.log_loss < raw_metrics.log_loss else '否'}",
            f"【13. Brier Score 是否改善】"
            f"{'是' if best_metrics.brier_score < raw_metrics.brier_score else '否'}",
            f"【14. ECE 是否改善】"
            f"{'是' if best_metrics.expected_calibration_error < raw_metrics.expected_calibration_error else '否'}",
            f"【15. calibration_gap 是否改善】"
            f"{'是' if best_metrics.calibration_gap < raw_metrics.calibration_gap else '否'}",
            f"【16. AUC 是否基本保持】"
            f"best={best_metrics.roc_auc:.6f}, raw={raw_metrics.roc_auc:.6f}",
            "",
            "【17. 指标取舍说明】",
            "  若 LogLoss 改善但 AUC 略降，或 ECE 改善但 Brier 略升，"
            "本报告按预设优先级如实记录，不隐瞒取舍。",
            "",
            f"【18. 最终校准器】已使用完整 working valid ({split_info.total_rows:,} 行) 重新拟合",
            f"  方法：{final_calibrator_method}",
            f"  路径：{final_calibrator_path}",
            "",
            "【19. 最终校准器尚未在 holdout 上评价】是",
            "",
            "【20. 开发阶段限制】",
            "  LightGBM 超参数与 Optuna 选择已使用完整 valid；"
            "本次 evaluation 仅用于校准方法比较，不是最终独立测试。",
            "",
            "【21. holdout 尚未使用】是",
            "",
            "【22. 下一步建议】",
            "  建议继续开展阈值、Lift 和投放覆盖率分析，并在 holdout 上完成最终评估。",
            "",
            f"当前模式：{'TEST_MODE=True' if test_mode else 'TEST_MODE=False'}",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_validation(
    paths: OutputPaths,
    split_info: SplitInfo,
    evaluation_predictions: pd.DataFrame,
    bins_df: pd.DataFrame,
    metrics_list: list[MethodMetrics],
    best_method: str,
    sigmoid_eval_probs: np.ndarray,
    isotonic_eval_probs: np.ndarray,
    development_isotonic: IsotonicRegression,
    evaluation_raw_probs: np.ndarray,
    test_mode: bool,
) -> bool:
    """最终验收检查。"""

    if split_info.total_rows != (TEST_TOTAL_ROWS if test_mode else FORMAL_TOTAL_ROWS):
        raise ValueError("working valid 总行数与模式设定不一致。")

    if len(set(split_info.calibration_indices) & set(split_info.evaluation_indices)) > 0:
        raise ValueError("calibration 与 evaluation 存在重叠。")

    if (
        split_info.calibration_rows + split_info.evaluation_rows
        != split_info.total_rows
    ):
        raise ValueError("calibration + evaluation 行数之和不正确。")

    for probabilities in (
        evaluation_predictions["raw_probability"].to_numpy(),
        sigmoid_eval_probs,
        isotonic_eval_probs,
    ):
        if np.isnan(probabilities).any() or np.isinf(probabilities).any():
            raise ValueError("校准概率存在 NaN 或 inf。")
        if (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError("校准概率超出 [0, 1] 范围。")

    check_isotonic_monotonicity(development_isotonic, evaluation_raw_probs)

    for metrics in metrics_list:
        if metrics.expected_calibration_error < -ECE_TOLERANCE:
            raise ValueError(f"{metrics.method} ECE 为负值。")
        if metrics.expected_calibration_error > 1.0 + ECE_TOLERANCE:
            raise ValueError(f"{metrics.method} ECE 超出 [0, 1]。")

    for method in METHODS:
        method_bins = bins_df.loc[bins_df["method"] == method]
        if method_bins["bin_rows"].sum() != split_info.evaluation_rows:
            raise ValueError(f"{method} 分箱行数之和不等于 evaluation 行数。")

    if not paths.selected_calibrator.exists():
        raise FileNotFoundError(f"最终校准器未保存：{paths.selected_calibrator}")

    required_outputs = [
        paths.bins_csv,
        paths.evaluation_predictions,
        paths.metrics_csv,
        paths.comparison_csv,
        paths.report_txt,
        paths.sigmoid_calibrator,
        paths.isotonic_calibrator,
        paths.plots_dir / "probability_calibration_curve.png",
        paths.plots_dir / "probability_calibration_distribution.png",
        paths.plots_dir / "probability_calibration_metric_comparison.png",
    ]
    for output_path in required_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"缺少输出文件：{output_path}")

    if best_method not in METHODS:
        raise ValueError(f"best_method 无效：{best_method}")

    return True


def main() -> None:
    """主流程：读取预测 → 拆分 → 校准 → 评价 → 保存。"""

    import sklearn

    paths = get_output_paths(TEST_MODE)

    print("=" * 72)
    print("第 34 步：调优 LightGBM 概率校准")
    print("=" * 72)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"scikit-learn 版本：{sklearn.__version__}")

    fixed_metadata = load_fixed_sample_metadata(FIXED_SAMPLE_METADATA_PATH)
    optuna_metadata = load_optuna_metadata(OPTUNA_METADATA_PATH, fixed_metadata)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"调优模型不存在：{MODEL_PATH}")

    tuned_metrics = optuna_metadata["tuned_metrics"]
    expected_mean_probability = float(tuned_metrics["valid_mean_predicted_ctr"])

    print("\n读取并校验调优模型验证预测 ...")
    predictions_df = load_and_validate_predictions(
        PREDICTIONS_PATH,
        expected_mean_probability,
    )

    working_df = select_working_subset(predictions_df, TEST_MODE)
    total_valid_ctr = float(working_df["click"].mean())

    print(f"working valid 行数：{len(working_df):,}")
    print(f"working valid CTR：{total_valid_ctr:.6f}")

    print("\n拆分 calibration / evaluation ...")
    calibration_df, evaluation_df, split_info = split_calibration_evaluation(working_df)

    print(f"calibration 行数：{split_info.calibration_rows:,}，CTR={split_info.calibration_ctr:.6f}")
    print(f"evaluation 行数：{split_info.evaluation_rows:,}，CTR={split_info.evaluation_ctr:.6f}")

    calibration_raw = clip_probabilities(
        calibration_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    )
    calibration_clicks = calibration_df["click"].to_numpy(dtype=np.int8)

    evaluation_raw = clip_probabilities(
        evaluation_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    )
    evaluation_clicks = evaluation_df["click"].to_numpy(dtype=np.int8)

    print("\n拟合 Sigmoid 校准器（calibration）...")
    development_sigmoid = fit_sigmoid_calibrator(calibration_raw, calibration_clicks)
    sigmoid_eval_probs = apply_sigmoid_calibrator(development_sigmoid, evaluation_raw)

    print("拟合 Isotonic 校准器（calibration）...")
    development_isotonic = fit_isotonic_calibrator(calibration_raw, calibration_clicks)
    check_isotonic_monotonicity(development_isotonic, calibration_raw)
    isotonic_eval_probs = apply_isotonic_calibrator(development_isotonic, evaluation_raw)
    check_isotonic_monotonicity(development_isotonic, evaluation_raw)

    evaluation_predictions = evaluation_df[
        ["row_position", "id", "click", "split_date"]
    ].copy()
    evaluation_predictions["raw_probability"] = evaluation_raw
    evaluation_predictions["sigmoid_probability"] = sigmoid_eval_probs
    evaluation_predictions["isotonic_probability"] = isotonic_eval_probs
    evaluation_predictions["calibration_split"] = "evaluation"

    print("\n计算 evaluation 指标 ...")
    method_probabilities = {
        "uncalibrated": evaluation_raw,
        "sigmoid": sigmoid_eval_probs,
        "isotonic": isotonic_eval_probs,
    }

    metrics_list: list[MethodMetrics] = []
    bins_parts: list[pd.DataFrame] = []

    for method, probabilities in method_probabilities.items():
        metrics, bin_df = compute_method_metrics(
            method=method,
            probabilities=probabilities,
            clicks=evaluation_clicks,
            calibration_rows=split_info.calibration_rows,
            evaluation_rows=split_info.evaluation_rows,
            selected_as_best=False,
        )
        metrics_list.append(metrics)
        bins_parts.append(bin_df)
        print(
            f"  {method}: LogLoss={metrics.log_loss:.6f}, "
            f"Brier={metrics.brier_score:.6f}, ECE={metrics.expected_calibration_error:.6f}"
        )

    best_method, selection_reason = select_best_method(metrics_list)

    for metrics in metrics_list:
        metrics.selected_as_best = metrics.method == best_method

    metrics_df = pd.DataFrame([method_metrics_to_row(item) for item in metrics_list])
    comparison_df = build_comparison_dataframe(
        metrics_list,
        best_method,
        selection_reason,
    )
    bins_df = pd.concat(bins_parts, ignore_index=True)

    print(f"\n最佳开发阶段方法：{best_method}")
    print(f"选择原因：{selection_reason}")

    paths.calibration_dir.mkdir(parents=True, exist_ok=True)
    paths.evaluation_predictions.parent.mkdir(parents=True, exist_ok=True)
    paths.sigmoid_calibrator.parent.mkdir(parents=True, exist_ok=True)

    bins_df.to_csv(paths.bins_csv, index=False, encoding="utf-8")
    evaluation_predictions.to_parquet(paths.evaluation_predictions, index=False)
    metrics_df.to_csv(paths.metrics_csv, index=False, encoding="utf-8")
    comparison_df.to_csv(paths.comparison_csv, index=False, encoding="utf-8")

    joblib.dump(development_sigmoid, paths.sigmoid_calibrator)
    joblib.dump(development_isotonic, paths.isotonic_calibrator)

    print("\n在完整 working valid 上拟合最终校准器 ...")
    full_raw = clip_probabilities(
        working_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    )
    full_clicks = working_df["click"].to_numpy(dtype=np.int8)

    final_calibrator = fit_final_selected_calibrator(
        best_method,
        full_raw,
        full_clicks,
    )
    joblib.dump(final_calibrator, paths.selected_calibrator)

    sigmoid_params = extract_sigmoid_parameters(development_sigmoid)
    isotonic_params = extract_isotonic_parameters(development_isotonic)

    save_calibration_plots(
        bins_df,
        evaluation_predictions,
        metrics_df,
        paths.plots_dir,
    )

    write_text_report(
        paths.report_txt,
        TEST_MODE,
        optuna_metadata,
        split_info,
        metrics_list,
        best_method,
        selection_reason,
        best_method,
        paths.selected_calibrator,
    )

    limitations = [
        "LightGBM 超参数已经使用完整 valid 选择",
        "本次 evaluation 用于校准方法开发比较",
        "最终无偏指标必须等待 holdout",
        "最终全量 valid 校准器不能在同一 valid 上自我评价",
    ]

    metadata_payload = {
        "script_name": "scripts/34_calibrate_tuned_lightgbm.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "total_valid_rows": split_info.total_rows,
        "total_valid_ctr": total_valid_ctr,
        "calibration_ratio": CALIBRATION_RATIO,
        "calibration_rows": split_info.calibration_rows,
        "evaluation_rows": split_info.evaluation_rows,
        "calibration_ctr": split_info.calibration_ctr,
        "evaluation_ctr": split_info.evaluation_ctr,
        "calibration_indices_sha256": split_info.calibration_indices_sha256,
        "evaluation_indices_sha256": split_info.evaluation_indices_sha256,
        "methods": list(METHODS),
        "metrics": [method_metrics_to_row(item) for item in metrics_list],
        "best_development_method": best_method,
        "selection_priority": SELECTION_PRIORITY,
        "selection_reason": selection_reason,
        "sigmoid_parameters": sigmoid_params,
        "isotonic_parameters": isotonic_params,
        "final_calibrator_method": best_method,
        "final_calibrator_path": str(paths.selected_calibrator),
        "source_model_path": str(MODEL_PATH),
        "source_prediction_path": str(PREDICTIONS_PATH),
        "source_optuna_metadata": str(OPTUNA_METADATA_PATH),
        "valid_id_sha256": fixed_metadata["valid_id_sha256"],
        "holdout_used": False,
        "validation_passed": False,
        "limitations": limitations,
        "output_paths": {
            "bins_csv": str(paths.bins_csv),
            "evaluation_predictions": str(paths.evaluation_predictions),
            "metrics_csv": str(paths.metrics_csv),
            "comparison_csv": str(paths.comparison_csv),
            "report_txt": str(paths.report_txt),
            "plots_dir": str(paths.plots_dir),
            "sigmoid_calibrator": str(paths.sigmoid_calibrator),
            "isotonic_calibrator": str(paths.isotonic_calibrator),
            "selected_calibrator": str(paths.selected_calibrator),
        },
    }

    validation_passed = final_validation(
        paths=paths,
        split_info=split_info,
        evaluation_predictions=evaluation_predictions,
        bins_df=bins_df,
        metrics_list=metrics_list,
        best_method=best_method,
        sigmoid_eval_probs=sigmoid_eval_probs,
        isotonic_eval_probs=isotonic_eval_probs,
        development_isotonic=development_isotonic,
        evaluation_raw_probs=evaluation_raw,
        test_mode=TEST_MODE,
    )
    metadata_payload["validation_passed"] = validation_passed

    paths.metadata_json.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_metrics = next(item for item in metrics_list if item.method == "uncalibrated")
    sigmoid_metrics = next(item for item in metrics_list if item.method == "sigmoid")
    isotonic_metrics = next(item for item in metrics_list if item.method == "isotonic")

    print("\n" + "=" * 72)
    print("第 34 步完成摘要")
    print("=" * 72)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"总 valid 行数：{split_info.total_rows:,}")
    print(f"calibration 行数：{split_info.calibration_rows:,}")
    print(f"evaluation 行数：{split_info.evaluation_rows:,}")
    print(f"calibration CTR：{split_info.calibration_ctr:.6f}")
    print(f"evaluation CTR：{split_info.evaluation_ctr:.6f}")
    print("\n三种方法 LogLoss：")
    print(f"  uncalibrated={raw_metrics.log_loss:.6f}")
    print(f"  sigmoid={sigmoid_metrics.log_loss:.6f}")
    print(f"  isotonic={isotonic_metrics.log_loss:.6f}")
    print("\n三种方法 Brier Score：")
    print(f"  uncalibrated={raw_metrics.brier_score:.6f}")
    print(f"  sigmoid={sigmoid_metrics.brier_score:.6f}")
    print(f"  isotonic={isotonic_metrics.brier_score:.6f}")
    print("\n三种方法 ECE：")
    print(f"  uncalibrated={raw_metrics.expected_calibration_error:.6f}")
    print(f"  sigmoid={sigmoid_metrics.expected_calibration_error:.6f}")
    print(f"  isotonic={isotonic_metrics.expected_calibration_error:.6f}")
    print("\n三种方法 calibration_gap：")
    print(f"  uncalibrated={raw_metrics.calibration_gap:.6f}")
    print(f"  sigmoid={sigmoid_metrics.calibration_gap:.6f}")
    print(f"  isotonic={isotonic_metrics.calibration_gap:.6f}")
    print(f"\n最佳开发阶段方法：{best_method}")
    print(f"最终校准器路径：{paths.selected_calibrator}")
    print(f"holdout_used：False")
    print(f"validation_passed：{validation_passed}")
    print("调优 LightGBM 概率校准完成，holdout 尚未使用。")


if __name__ == "__main__":
    main()
