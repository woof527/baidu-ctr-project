"""
百度 CTR 项目 — 统一样本 LightGBM 基线（第 38 步）

功能：
    在 Step 37 统一建模样本上训练两个 LightGBM 模型：
    - LightGBM_Numerical33：仅 33 个工程化数值特征
    - LightGBM_Unified：33 数值 + recommended categorical（排除高基数）
    复用 Step 32 Optuna 最优超参，禁止读取 holdout。

数据输入：
    data/modeling/unified_train/
    data/modeling/unified_valid/
    outputs/unified_modeling_sample_metadata.json
    outputs/lightgbm_optuna_best_params.json

用法：
    python scripts/38_train_unified_lightgbm_baselines.py
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

TRAIN_INPUT_DIR = Path("data/modeling/unified_train")
VALID_INPUT_DIR = Path("data/modeling/unified_valid")

UNIFIED_METADATA_PATH = Path("outputs/unified_modeling_sample_metadata.json")
STEP32_BEST_PARAMS_PATH = Path("outputs/lightgbm_optuna_best_params.json")

MODEL_A_PATH = Path("models/unified_lightgbm_numerical33.txt")
MODEL_B_PATH = Path("models/unified_lightgbm_full.txt")
PREDICTIONS_PATH = Path("outputs/predictions/unified_lightgbm_valid_predictions.parquet")
METRICS_CSV_PATH = Path("outputs/unified_lightgbm_metrics.csv")
COMPARISON_TXT_PATH = Path("outputs/unified_lightgbm_comparison.txt")
METADATA_JSON_PATH = Path("outputs/unified_lightgbm_metadata.json")

FORMAL_TRAIN_ROWS = 2_000_000
FORMAL_VALID_ROWS = 500_000
TEST_TRAIN_ROWS = 100_000
TEST_VALID_ROWS = 50_000

RANDOM_STATE = 42
BATCH_SIZE = 200_000
THRESHOLD = 0.5
PROB_CLIP_EPS = 1e-7
EARLY_STOPPING_ROUNDS = 100
MAX_ESTIMATORS = 2000
EXPECTED_NUMERICAL_COUNT = 33
UNK_TOKEN = "__UNK__"

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")
FORBIDDEN_DATE = "2014-10-30"

LOG1P_SUFFIXES = ("_freq", "_hist_impressions", "_hist_clicks")

OLD_STEP32_REFERENCE_AUC = 0.737077
OLD_STEP32_REFERENCE_LOGLOSS = 0.384831
PARAMETER_SOURCE = "Step32_Optuna_best_trial_8"

MODEL_A_NAME = "LightGBM_Numerical33"
MODEL_B_NAME = "LightGBM_Unified"

STEP32_BEST_PARAMS_FALLBACK = {
    "learning_rate": 0.10342099303484481,
    "max_depth": 5,
    "num_leaves": 29,
    "min_child_samples": 143,
    "subsample": 0.9313811040057837,
    "colsample_bytree": 0.7222133955202271,
    "reg_alpha": 1.683416412018213e-05,
    "reg_lambda": 1.1036250149900698e-07,
    "min_split_gain": 0.17262068517511872,
}


@dataclass
class ModelMetrics:
    """单模型评估指标。"""

    model_name: str
    roc_auc: float
    log_loss_value: float
    brier_score: float
    valid_actual_ctr: float
    valid_mean_predicted_ctr: float
    calibration_gap: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    best_iteration: int
    training_seconds: float
    feature_count: int


def assert_safe_path(path: Path) -> None:
    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def get_row_limits(test_mode: bool) -> tuple[int, int]:
    if test_mode:
        return TEST_TRAIN_ROWS, TEST_VALID_ROWS
    return FORMAL_TRAIN_ROWS, FORMAL_VALID_ROWS


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    assert_safe_path(parquet_dir)
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{parquet_dir}")
    return files


def load_unified_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到统一样本元数据：{metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("validation_passed") is not True:
        raise ValueError("unified metadata validation_passed 必须为 true。")
    if metadata.get("holdout_used") is not False:
        raise ValueError("unified metadata holdout_used 必须为 false。")

    required = (
        "numerical_features",
        "recommended_embedding_features",
        "high_cardinality_features",
        "train_rows",
        "valid_rows",
    )
    for key in required:
        if key not in metadata:
            raise KeyError(f"unified metadata 缺少字段：{key}")

    numerical = metadata["numerical_features"]
    if len(numerical) != EXPECTED_NUMERICAL_COUNT:
        raise ValueError(
            f"numerical_features 数量 {len(numerical)} != {EXPECTED_NUMERICAL_COUNT}"
        )

    excluded = set(metadata["high_cardinality_features"])
    categorical = metadata["recommended_embedding_features"]
    overlap = excluded & set(categorical)
    if overlap:
        raise ValueError(f"recommended_embedding_features 包含高基数字段：{sorted(overlap)}")

    return metadata


def load_step32_params(params_path: Path) -> dict[str, Any]:
    if params_path.exists():
        payload = json.loads(params_path.read_text(encoding="utf-8"))
        params = dict(payload.get("best_params", STEP32_BEST_PARAMS_FALLBACK))
    else:
        params = dict(STEP32_BEST_PARAMS_FALLBACK)

    for key, value in STEP32_BEST_PARAMS_FALLBACK.items():
        params.setdefault(key, value)

    return params


def get_log1p_columns(numerical_features: list[str]) -> list[str]:
    return sorted(
        col
        for col in numerical_features
        if any(col.endswith(suffix) for suffix in LOG1P_SUFFIXES)
    )


def preprocess_numerical_matrix(
    raw_matrix: np.ndarray,
    numerical_features: list[str],
    log1p_columns: list[str],
) -> np.ndarray:
    """对 raw 数值矩阵应用 log1p + nan/inf 清洗（与 Step 27/30 一致）。"""

    matrix = pd.DataFrame(raw_matrix, columns=numerical_features).copy()
    log1p_index = [numerical_features.index(col) for col in log1p_columns]

    for col in log1p_columns:
        matrix[col] = np.log1p(
            pd.to_numeric(matrix[col], errors="coerce").astype(np.float64)
        )

    for col in numerical_features:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")

    feature_array = matrix[numerical_features].to_numpy(dtype=np.float64)
    feature_array[~np.isfinite(feature_array)] = np.nan
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_array.astype(np.float32)


def is_missing_categorical(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def value_to_category_text(value: Any) -> str:
    if is_missing_categorical(value):
        return UNK_TOKEN
    return str(value)


def build_categorical_vocabularies(
    train_files: list[Path],
    categorical_columns: list[str],
    max_rows: int | None,
) -> dict[str, list[str]]:
    """仅扫描 train，建立 categorical vocabulary（含 UNK）。"""

    vocab_sets: dict[str, set[str]] = {col: set() for col in categorical_columns}
    collected = 0

    for parquet_path in train_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=categorical_columns,
            batch_size=BATCH_SIZE,
        ):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            for column in categorical_columns:
                for value in batch_df[column]:
                    if not is_missing_categorical(value):
                        vocab_sets[column].add(str(value))

            collected += len(batch_df)
            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    vocabularies: dict[str, list[str]] = {}
    for column in categorical_columns:
        categories = sorted(vocab_sets[column])
        if UNK_TOKEN not in categories:
            categories.append(UNK_TOKEN)
        else:
            categories = [c for c in categories if c != UNK_TOKEN] + [UNK_TOKEN]
        vocabularies[column] = categories

    return vocabularies


def map_to_categorical_series(
    series: pd.Series,
    categories: list[str],
) -> pd.Series:
    category_set = set(categories)
    mapped = series.apply(value_to_category_text).apply(
        lambda text: text if text in category_set else UNK_TOKEN
    )
    return pd.Series(pd.Categorical(mapped, categories=categories), dtype="category")


def load_numerical_split(
    parquet_files: list[Path],
    numerical_features: list[str],
    log1p_columns: list[str],
    max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    read_columns = ["id", "click", "split_date", *numerical_features]
    collected = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=read_columns,
            batch_size=BATCH_SIZE,
        ):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            raw_matrix = batch_df[numerical_features].to_numpy(dtype=np.float64)
            processed = preprocess_numerical_matrix(
                raw_matrix,
                numerical_features,
                log1p_columns,
            )
            feature_parts.append(processed)
            label_parts.append(batch_df["click"].to_numpy(dtype=np.int8))
            meta_parts.append(batch_df[["id", "click", "split_date"]].copy())

            collected += len(batch_df)
            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    if collected == 0:
        raise ValueError("未读取到任何样本行。")

    x_matrix = np.vstack(feature_parts).astype(np.float32)
    y_vector = np.concatenate(label_parts).astype(np.int8)
    meta_df = pd.concat(meta_parts, ignore_index=True)

    del feature_parts, label_parts, meta_parts
    gc.collect()

    return x_matrix, y_vector, meta_df


def load_unified_split_dataframe(
    parquet_files: list[Path],
    numerical_features: list[str],
    log1p_columns: list[str],
    categorical_columns: list[str],
    vocabularies: dict[str, list[str]],
    max_rows: int | None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """加载 Model B 用 DataFrame：numerical float + categorical category dtype。"""

    numerical_parts: list[np.ndarray] = []
    categorical_parts: dict[str, list[pd.Series]] = {col: [] for col in categorical_columns}
    label_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    read_columns = list(
        dict.fromkeys(["id", "click", "split_date", *numerical_features, *categorical_columns])
    )
    collected = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=read_columns,
            batch_size=BATCH_SIZE,
        ):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            raw_matrix = batch_df[numerical_features].to_numpy(dtype=np.float64)
            processed = preprocess_numerical_matrix(
                raw_matrix,
                numerical_features,
                log1p_columns,
            )
            numerical_parts.append(processed)

            for column in categorical_columns:
                cat_series = map_to_categorical_series(
                    batch_df[column],
                    vocabularies[column],
                )
                categorical_parts[column].append(cat_series)

            label_parts.append(batch_df["click"].to_numpy(dtype=np.int8))
            meta_parts.append(batch_df[["id", "click", "split_date"]].copy())

            collected += len(batch_df)
            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    if collected == 0:
        raise ValueError("未读取到任何样本行。")

    x_numerical = np.vstack(numerical_parts).astype(np.float32)
    y_vector = np.concatenate(label_parts).astype(np.int8)
    meta_df = pd.concat(meta_parts, ignore_index=True)

    data: dict[str, Any] = {}
    for index, column in enumerate(numerical_features):
        data[column] = x_numerical[:, index]

    for column in categorical_columns:
        data[column] = pd.concat(categorical_parts[column], ignore_index=True)

    x_df = pd.DataFrame(data)

    del numerical_parts, categorical_parts, label_parts, meta_parts, x_numerical
    gc.collect()

    return x_df, y_vector, meta_df


def validate_loaded_meta(
    meta_df: pd.DataFrame,
    split_name: str,
    expected_dates: set[str] | None,
) -> None:
    if meta_df["id"].isna().any():
        raise ValueError(f"{split_name} id 存在缺失。")
    if not meta_df["click"].isin([0, 1]).all():
        raise ValueError(f"{split_name} click 非法。")

    dates = set(meta_df["split_date"].astype(str).unique())
    if FORBIDDEN_DATE in dates:
        raise ValueError(f"{split_name} 含禁止日期 {FORBIDDEN_DATE}。")
    if expected_dates is not None:
        unexpected = dates - expected_dates
        if unexpected:
            raise ValueError(f"{split_name} 含意外日期：{sorted(unexpected)}")


def build_lightgbm_classifier(params: dict[str, Any]) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=MAX_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def train_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    params: dict[str, Any],
    categorical_feature: str | list[str] = "auto",
) -> tuple[lgb.LGBMClassifier, np.ndarray, float]:
    model = build_lightgbm_classifier(params)

    fit_kwargs: dict[str, Any] = {}
    if categorical_feature != "auto" or any(
        pd.api.types.is_categorical_dtype(x_train[col]) for col in x_train.columns
    ):
        fit_kwargs["categorical_feature"] = categorical_feature

    start_time = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=25),
        ],
        **fit_kwargs,
    )
    training_seconds = time.perf_counter() - start_time

    if not hasattr(model, "best_iteration_") or model.best_iteration_ is None:
        raise ValueError("LightGBM 训练完成后 best_iteration_ 不存在。")

    probabilities = model.predict_proba(
        x_valid,
        num_iteration=model.best_iteration_,
    )[:, 1].astype(np.float64)

    return model, probabilities, training_seconds


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def compute_model_metrics(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    training_seconds: float,
    best_iteration: int,
    feature_count: int,
) -> ModelMetrics:
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError(f"{model_name} 预测概率存在 NaN 或 inf。")

    clipped = clip_probabilities(probabilities)
    predicted_labels = (probabilities >= THRESHOLD).astype(np.int8)
    valid_actual_ctr = float(y_true.mean())
    valid_mean_predicted_ctr = float(probabilities.mean())

    return ModelMetrics(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        log_loss_value=float(log_loss(y_true, clipped, labels=[0, 1])),
        brier_score=float(brier_score_loss(y_true, probabilities)),
        valid_actual_ctr=valid_actual_ctr,
        valid_mean_predicted_ctr=valid_mean_predicted_ctr,
        calibration_gap=abs(valid_mean_predicted_ctr - valid_actual_ctr),
        precision=float(precision_score(y_true, predicted_labels, zero_division=0)),
        recall=float(recall_score(y_true, predicted_labels, zero_division=0)),
        f1=float(f1_score(y_true, predicted_labels, zero_division=0)),
        accuracy=float(accuracy_score(y_true, predicted_labels)),
        best_iteration=best_iteration,
        training_seconds=training_seconds,
        feature_count=feature_count,
    )


def metrics_to_row(metrics: ModelMetrics, train_rows: int, valid_rows: int) -> dict[str, Any]:
    return {
        "model": metrics.model_name,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "feature_count": metrics.feature_count,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss_value,
        "brier_score": metrics.brier_score,
        "valid_actual_ctr": metrics.valid_actual_ctr,
        "valid_mean_predicted_ctr": metrics.valid_mean_predicted_ctr,
        "calibration_gap": metrics.calibration_gap,
        "precision_at_0.5": metrics.precision,
        "recall_at_0.5": metrics.recall,
        "f1_at_0.5": metrics.f1,
        "accuracy_at_0.5": metrics.accuracy,
        "best_iteration": metrics.best_iteration,
        "training_seconds": metrics.training_seconds,
        "holdout_used": False,
    }


def choose_better_model(metrics_a: ModelMetrics, metrics_b: ModelMetrics) -> str:
    if metrics_a.log_loss_value != metrics_b.log_loss_value:
        return (
            metrics_a.model_name
            if metrics_a.log_loss_value < metrics_b.log_loss_value
            else metrics_b.model_name
        )
    if metrics_a.roc_auc != metrics_b.roc_auc:
        return (
            metrics_a.model_name
            if metrics_a.roc_auc > metrics_b.roc_auc
            else metrics_b.model_name
        )
    if metrics_a.brier_score != metrics_b.brier_score:
        return (
            metrics_a.model_name
            if metrics_a.brier_score < metrics_b.brier_score
            else metrics_b.model_name
        )
    return metrics_b.model_name


def write_comparison_report(
    path: Path,
    metrics_a: ModelMetrics,
    metrics_b: ModelMetrics,
    categorical_features: list[str],
    excluded_high_cardinality: list[str],
    better_model: str,
) -> None:
    lines = [
        "百度 CTR 项目 — 统一样本 LightGBM 基线对比报告",
        "=" * 72,
        "",
        "【Model A】LightGBM_Numerical33",
        f"  AUC：{metrics_a.roc_auc:.6f}",
        f"  LogLoss：{metrics_a.log_loss_value:.6f}",
        f"  Brier：{metrics_a.brier_score:.6f}",
        f"  Calibration Gap：{metrics_a.calibration_gap:.6f}",
        f"  Best Iteration：{metrics_a.best_iteration}",
        "",
        "【Model B】LightGBM_Unified",
        f"  AUC：{metrics_b.roc_auc:.6f}",
        f"  LogLoss：{metrics_b.log_loss_value:.6f}",
        f"  Brier：{metrics_b.brier_score:.6f}",
        f"  Calibration Gap：{metrics_b.calibration_gap:.6f}",
        f"  Best Iteration：{metrics_b.best_iteration}",
        "",
        "【Model A vs Model B】",
        f"  AUC 变化：{metrics_b.roc_auc - metrics_a.roc_auc:+.6f}",
        f"  LogLoss 变化：{metrics_b.log_loss_value - metrics_a.log_loss_value:+.6f}",
        f"  Brier 变化：{metrics_b.brier_score - metrics_a.brier_score:+.6f}",
        f"  Calibration Gap 变化：{metrics_b.calibration_gap - metrics_a.calibration_gap:+.6f}",
        "",
        "【加入 raw categorical 后是否改善】",
        f"  AUC 提高：{'是' if metrics_b.roc_auc > metrics_a.roc_auc else '否'}",
        f"  LogLoss 降低：{'是' if metrics_b.log_loss_value < metrics_a.log_loss_value else '否'}",
        f"  Brier 改善：{'是' if metrics_b.brier_score < metrics_a.brier_score else '否'}",
        f"  Calibration 改善：{'是' if metrics_b.calibration_gap < metrics_a.calibration_gap else '否'}",
        "",
        f"【BETTER_MODEL】{better_model}",
        "",
        "【Model B categorical 使用】",
        f"  {categorical_features}",
        "",
        "【排除的高基数字段】",
        f"  {excluded_high_cardinality}",
        "",
        "【Step 32 历史参考（旧 Step 30 固定样本，仅供参考）】",
        f"  AUC = {OLD_STEP32_REFERENCE_AUC:.6f}",
        f"  LogLoss = {OLD_STEP32_REFERENCE_LOGLOSS:.6f}",
        "",
        "说明：旧 Step 32 与本次 unified sample 不是完全相同行集合，",
        "差异不应全部解释为模型性能变化。",
        "",
        "HOLDOUT_USED = False",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_final_summary(
    train_rows: int,
    valid_rows: int,
    metrics_a: ModelMetrics,
    metrics_b: ModelMetrics,
    categorical_features: list[str],
    excluded_high_cardinality: list[str],
    better_model: str,
) -> None:
    print("\n" + "=" * 40)
    print("UNIFIED LIGHTGBM BASELINE SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_rows}")
    print(f"VALID_ROWS = {valid_rows}")
    print()
    print(f"MODEL_A = {MODEL_A_NAME}")
    print(f"AUC = {metrics_a.roc_auc:.6f}")
    print(f"LOGLOSS = {metrics_a.log_loss_value:.6f}")
    print(f"BRIER = {metrics_a.brier_score:.6f}")
    print(f"ACTUAL_CTR = {metrics_a.valid_actual_ctr:.6f}")
    print(f"MEAN_PREDICTED_CTR = {metrics_a.valid_mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {metrics_a.calibration_gap:.6f}")
    print(f"BEST_ITERATION = {metrics_a.best_iteration}")
    print()
    print(f"MODEL_B = {MODEL_B_NAME}")
    print(f"AUC = {metrics_b.roc_auc:.6f}")
    print(f"LOGLOSS = {metrics_b.log_loss_value:.6f}")
    print(f"BRIER = {metrics_b.brier_score:.6f}")
    print(f"ACTUAL_CTR = {metrics_b.valid_actual_ctr:.6f}")
    print(f"MEAN_PREDICTED_CTR = {metrics_b.valid_mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {metrics_b.calibration_gap:.6f}")
    print(f"BEST_ITERATION = {metrics_b.best_iteration}")
    print()
    print(f"CATEGORICAL_FEATURES_USED = {categorical_features}")
    print(f"HIGH_CARDINALITY_FEATURES_EXCLUDED = {excluded_high_cardinality}")
    print(f"BETTER_MODEL = {better_model}")
    print(f"OLD_STEP32_REFERENCE_AUC = {OLD_STEP32_REFERENCE_AUC}")
    print(f"OLD_STEP32_REFERENCE_LOGLOSS = {OLD_STEP32_REFERENCE_LOGLOSS}")
    print("HOLDOUT_USED = False")
    print("VALIDATION_PASSED = True")
    print("=" * 40)


def main() -> None:
    train_limit, valid_limit = get_row_limits(TEST_MODE)

    print("=" * 72)
    print("统一样本 LightGBM 基线训练（第 38 步）")
    print("=" * 72)
    print(f"TEST_MODE = {TEST_MODE}")
    print(f"train 行数上限：{train_limit:,}")
    print(f"valid 行数上限：{valid_limit:,}")

    assert_safe_path(TRAIN_INPUT_DIR)
    assert_safe_path(VALID_INPUT_DIR)

    metadata = load_unified_metadata(UNIFIED_METADATA_PATH)
    numerical_features: list[str] = metadata["numerical_features"]
    categorical_features: list[str] = metadata["recommended_embedding_features"]
    excluded_high_cardinality: list[str] = metadata["high_cardinality_features"]
    log1p_columns = get_log1p_columns(numerical_features)
    step32_params = load_step32_params(STEP32_BEST_PARAMS_PATH)

    train_files = get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)

    train_expected_dates = set(
        pd.date_range("2014-10-21", "2014-10-28", freq="D").strftime("%Y-%m-%d")
    )
    valid_expected_dates = {"2014-10-29"}

    print(f"\n数值特征：{len(numerical_features)} 个")
    print(f"log1p 列：{len(log1p_columns)} 个")
    print(f"Model B categorical：{len(categorical_features)} 个")
    print(f"排除高基数：{excluded_high_cardinality}")
    print(f"参数来源：{PARAMETER_SOURCE}")

    # --- Model A: numerical only ---
    print("\n" + "-" * 72)
    print("加载 Model A 数据（numerical only）...")
    x_train_a, y_train, train_meta = load_numerical_split(
        train_files,
        numerical_features,
        log1p_columns,
        train_limit,
    )
    x_valid_a, y_valid, valid_meta = load_numerical_split(
        valid_files,
        numerical_features,
        log1p_columns,
        valid_limit,
    )

    validate_loaded_meta(train_meta, "train", train_expected_dates)
    validate_loaded_meta(valid_meta, "valid", valid_expected_dates)

    if np.isnan(x_train_a).any() or np.isnan(x_valid_a).any():
        raise ValueError("Model A 数值特征存在 NaN。")

    train_rows = len(y_train)
    valid_rows = len(y_valid)

    if not TEST_MODE:
        if train_rows != FORMAL_TRAIN_ROWS:
            raise ValueError(f"train 行数 {train_rows:,} != {FORMAL_TRAIN_ROWS:,}")
        if valid_rows != FORMAL_VALID_ROWS:
            raise ValueError(f"valid 行数 {valid_rows:,} != {FORMAL_VALID_ROWS:,}")

    x_train_a_df = pd.DataFrame(x_train_a, columns=numerical_features)
    x_valid_a_df = pd.DataFrame(x_valid_a, columns=numerical_features)

    print("\n训练 Model A: LightGBM_Numerical33 ...")
    model_a, pred_a, train_seconds_a = train_lightgbm_model(
        x_train_a_df,
        y_train,
        x_valid_a_df,
        y_valid,
        step32_params,
        categorical_feature=[],
    )
    metrics_a = compute_model_metrics(
        MODEL_A_NAME,
        y_valid,
        pred_a,
        train_seconds_a,
        int(model_a.best_iteration_),
        len(numerical_features),
    )

    MODEL_A_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_a.booster_.save_model(str(MODEL_A_PATH))
    print(f"Model A 已保存：{MODEL_A_PATH}")

    del x_train_a, x_valid_a, x_train_a_df, x_valid_a_df, model_a
    gc.collect()

    # --- Model B: numerical + categorical ---
    print("\n" + "-" * 72)
    print("建立 train-only categorical vocabulary ...")
    vocabularies = build_categorical_vocabularies(
        train_files,
        categorical_features,
        train_limit,
    )
    for column, categories in vocabularies.items():
        print(f"  {column}: {len(categories) - 1} train categories + UNK")

    print("\n加载 Model B 数据（numerical + categorical）...")
    x_train_b, y_train_b, _ = load_unified_split_dataframe(
        train_files,
        numerical_features,
        log1p_columns,
        categorical_features,
        vocabularies,
        train_limit,
    )
    x_valid_b, y_valid_b, _ = load_unified_split_dataframe(
        valid_files,
        numerical_features,
        log1p_columns,
        categorical_features,
        vocabularies,
        valid_limit,
    )

    if not np.array_equal(y_train, y_train_b) or not np.array_equal(y_valid, y_valid_b):
        raise ValueError("Model A 与 Model B 的标签不一致。")

    print("\n训练 Model B: LightGBM_Unified ...")
    model_b, pred_b, train_seconds_b = train_lightgbm_model(
        x_train_b,
        y_train_b,
        x_valid_b,
        y_valid_b,
        step32_params,
        categorical_feature="auto",
    )
    metrics_b = compute_model_metrics(
        MODEL_B_NAME,
        y_valid_b,
        pred_b,
        train_seconds_b,
        int(model_b.best_iteration_),
        len(numerical_features) + len(categorical_features),
    )

    model_b.booster_.save_model(str(MODEL_B_PATH))
    print(f"Model B 已保存：{MODEL_B_PATH}")

    # --- Predictions ---
    predictions_df = valid_meta.copy()
    predictions_df["lgbm_numerical33_pred"] = pred_a.astype(np.float32)
    predictions_df["lgbm_unified_pred"] = pred_b.astype(np.float32)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"预测已保存：{PREDICTIONS_PATH}")

    # --- Metrics & reports ---
    metrics_df = pd.DataFrame(
        [
            metrics_to_row(metrics_a, train_rows, valid_rows),
            metrics_to_row(metrics_b, train_rows, valid_rows),
        ]
    )
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False, encoding="utf-8")

    better_model = choose_better_model(metrics_a, metrics_b)
    write_comparison_report(
        COMPARISON_TXT_PATH,
        metrics_a,
        metrics_b,
        categorical_features,
        excluded_high_cardinality,
        better_model,
    )

    vocab_summary = {
        column: {
            "category_count": len(categories),
            "train_categories_excluding_unk": len(categories) - 1,
        }
        for column, categories in vocabularies.items()
    }

    metadata_out = {
        "script_name": "scripts/38_train_unified_lightgbm_baselines.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "train_date_range": metadata["train_date_range"],
        "valid_date_range": metadata["valid_date_range"],
        "numerical_features": numerical_features,
        "log1p_columns": log1p_columns,
        "categorical_features_used": categorical_features,
        "excluded_high_cardinality_features": excluded_high_cardinality,
        "parameter_source": PARAMETER_SOURCE,
        "step32_best_params": step32_params,
        "random_seed": RANDOM_STATE,
        "best_iteration_model_a": metrics_a.best_iteration,
        "best_iteration_model_b": metrics_b.best_iteration,
        "model_a_metrics": metrics_to_row(metrics_a, train_rows, valid_rows),
        "model_b_metrics": metrics_to_row(metrics_b, train_rows, valid_rows),
        "better_model": better_model,
        "old_step32_reference_auc": OLD_STEP32_REFERENCE_AUC,
        "old_step32_reference_logloss": OLD_STEP32_REFERENCE_LOGLOSS,
        "categorical_vocab_summary": vocab_summary,
        "output_paths": {
            "model_a": str(MODEL_A_PATH),
            "model_b": str(MODEL_B_PATH),
            "predictions": str(PREDICTIONS_PATH),
            "metrics_csv": str(METRICS_CSV_PATH),
            "comparison_txt": str(COMPARISON_TXT_PATH),
        },
        "holdout_used": False,
        "validation_passed": True,
    }
    METADATA_JSON_PATH.write_text(
        json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nMetrics CSV: {METRICS_CSV_PATH}")
    print(f"Comparison: {COMPARISON_TXT_PATH}")
    print(f"Metadata: {METADATA_JSON_PATH}")

    print_final_summary(
        train_rows,
        valid_rows,
        metrics_a,
        metrics_b,
        categorical_features,
        excluded_high_cardinality,
        better_model,
    )


if __name__ == "__main__":
    main()
