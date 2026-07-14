"""
百度 CTR 项目 — XGBoost 基线模型

功能：
    从 target_encoded 特征中按全文件均匀抽样，训练 XGBoost 二分类模型，
    在 valid 上评估并与逻辑回归、LightGBM 基线对比。禁止读取 holdout。

数据输入：
    data/features/target_encoded/train/*.parquet
    data/features/target_encoded/valid/*.parquet

用法：
    python scripts/28_train_xgboost_baseline.py
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

TEST_MODE = False

TEST_TRAIN_ROWS = 500_000
TEST_VALID_ROWS = 200_000

FULL_TRAIN_ROWS = 2_000_000
FULL_VALID_ROWS = 500_000

RANDOM_STATE = 42
BATCH_SIZE = 200_000
PREDICTION_THRESHOLD = 0.5
PREDICTION_SAMPLE_SIZE = 10_000
MAX_MEMORY_GB = 6.0

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

TRAIN_DIR = Path("data/features/target_encoded/train")
VALID_DIR = Path("data/features/target_encoded/valid")

LOGISTIC_METRICS_PATH = Path("outputs/logistic_baseline_valid_metrics.json")
LIGHTGBM_METRICS_PATH = Path("outputs/lightgbm_baseline_valid_metrics.json")

FEATURE_SUFFIXES = (
    "_freq",
    "_hist_impressions",
    "_hist_clicks",
    "_hist_ctr",
    "_exposure_percentile",
    "_te",
)

LOG1P_SUFFIXES = ("_freq", "_hist_impressions", "_hist_clicks")

BINARY_FEATURES = ("is_weekend",)

FORBIDDEN_FEATURES = {
    "id",
    "click",
    "hour",
    "date",
    "day_of_week",
    "banner_pos",
    "device_type",
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_model",
    "hour_of_day",
}

DYNAMIC_FEATURES = ("hour_sin", "hour_cos")


@dataclass
class FeatureConfig:
    """特征配置。"""

    raw_feature_columns: list[str]
    log1p_columns: list[str]
    dynamic_feature_columns: list[str]
    feature_columns: list[str]
    use_hour_cyclical: bool


@dataclass
class RunContext:
    """输出路径。"""

    test_mode: bool
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    report_path: Path
    importance_path: Path
    predictions_sample_path: Path
    sampling_summary_path: Path
    comparison_path: Path


@dataclass
class SamplingConfig:
    """抽样配置。"""

    target_train_rows: int
    target_valid_rows: int
    random_state: int
    batch_size: int


@dataclass
class MetricsResult:
    """验证集指标。"""

    roc_auc: float
    log_loss_value: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    valid_actual_ctr: float
    valid_mean_predicted_ctr: float
    predicted_click_ratio_at_threshold: float
    best_iteration: int


def get_run_context(test_mode: bool) -> RunContext:
    """根据运行模式返回输出路径。"""

    if test_mode:
        return RunContext(
            test_mode=True,
            model_path=Path("models/xgboost_baseline_test.json"),
            metadata_path=Path("models/xgboost_baseline_test_metadata.joblib"),
            metrics_path=Path("outputs/xgboost_baseline_test_metrics.json"),
            report_path=Path("outputs/xgboost_baseline_test_report.txt"),
            importance_path=Path("outputs/xgboost_baseline_test_feature_importance.csv"),
            predictions_sample_path=Path(
                "outputs/xgboost_baseline_test_predictions_sample.csv"
            ),
            sampling_summary_path=Path(
                "outputs/xgboost_baseline_test_sampling_summary.csv"
            ),
            comparison_path=Path("outputs/model_comparison_test.csv"),
        )

    return RunContext(
        test_mode=False,
        model_path=Path("models/xgboost_baseline.json"),
        metadata_path=Path("models/xgboost_baseline_metadata.joblib"),
        metrics_path=Path("outputs/xgboost_baseline_valid_metrics.json"),
        report_path=Path("outputs/xgboost_baseline_report.txt"),
        importance_path=Path("outputs/xgboost_baseline_feature_importance.csv"),
        predictions_sample_path=Path(
            "outputs/xgboost_baseline_valid_predictions_sample.csv"
        ),
        sampling_summary_path=Path("outputs/xgboost_baseline_sampling_summary.csv"),
        comparison_path=Path("outputs/model_comparison.csv"),
    )


def get_sampling_config(test_mode: bool) -> SamplingConfig:
    """返回当前模式的抽样目标。"""

    if test_mode:
        return SamplingConfig(
            target_train_rows=TEST_TRAIN_ROWS,
            target_valid_rows=TEST_VALID_ROWS,
            random_state=RANDOM_STATE,
            batch_size=BATCH_SIZE,
        )

    return SamplingConfig(
        target_train_rows=FULL_TRAIN_ROWS,
        target_valid_rows=FULL_VALID_ROWS,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
    )


def list_parquet_files(parquet_dir: Path) -> list[Path]:
    """列出全部 Parquet 分块。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到输入目录：{parquet_dir}\n"
            "请先运行：python scripts/24_build_target_encoding.py"
        )

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            "请先运行：python scripts/24_build_target_encoding.py"
        )

    return files


def get_file_row_counts(parquet_files: list[Path]) -> list[int]:
    """用 metadata 统计每个文件行数。"""

    return [pq.read_metadata(path).num_rows for path in parquet_files]


def allocate_sample_quotas(total_target: int, file_row_counts: list[int]) -> list[int]:
    """按文件行数比例分配抽样额度，并保证总和等于目标行数。"""

    total_rows = sum(file_row_counts)
    if total_target > total_rows:
        raise ValueError(
            f"目标抽样行数 {total_target:,} 超过可用总行数 {total_rows:,}"
        )

    raw_quotas = [total_target * count / total_rows for count in file_row_counts]
    quotas = [int(np.floor(value)) for value in raw_quotas]
    remaining = total_target - sum(quotas)

    # 将剩余额度分配给余数最大的文件
    remainders = [
        (raw_quotas[index] - quotas[index], index)
        for index in range(len(raw_quotas))
    ]
    for _, file_index in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        quotas[file_index] += 1
        remaining -= 1

    if sum(quotas) != total_target:
        raise ValueError(
            f"抽样额度分配失败：目标 {total_target:,}，实际 {sum(quotas):,}"
        )

    return quotas


def read_schema_columns(parquet_path: Path) -> list[str]:
    """读取 schema 列名。"""

    return pq.read_schema(parquet_path).names


def discover_feature_config(schema_columns: list[str]) -> FeatureConfig:
    """从 schema 自动确定特征列表（与 26 号脚本保持一致）。"""

    raw_features: list[str] = []

    for column_name in schema_columns:
        if column_name in FORBIDDEN_FEATURES:
            continue

        if column_name.startswith("C") and column_name[1:].isdigit():
            continue

        if column_name in BINARY_FEATURES:
            raw_features.append(column_name)
            continue

        if any(column_name.endswith(suffix) for suffix in FEATURE_SUFFIXES):
            raw_features.append(column_name)

    raw_features = sorted(set(raw_features))

    if "is_weekend" not in schema_columns:
        raise ValueError("期望存在的二元特征缺失：is_weekend")

    log1p_columns = sorted(
        column_name
        for column_name in raw_features
        if any(column_name.endswith(suffix) for suffix in LOG1P_SUFFIXES)
    )

    use_hour_cyclical = "hour_of_day" in schema_columns
    dynamic_feature_columns = list(DYNAMIC_FEATURES) if use_hour_cyclical else []
    feature_columns = raw_features + dynamic_feature_columns

    if not feature_columns:
        raise ValueError("未能从 schema 中识别任何可用特征列。")

    return FeatureConfig(
        raw_feature_columns=raw_features,
        log1p_columns=log1p_columns,
        dynamic_feature_columns=dynamic_feature_columns,
        feature_columns=feature_columns,
        use_hour_cyclical=use_hour_cyclical,
    )


def get_read_columns(
    feature_config: FeatureConfig,
    schema_columns: list[str],
) -> list[str]:
    """抽样 batch 需要读取的列。"""

    columns = ["click", *feature_config.raw_feature_columns]

    if feature_config.use_hour_cyclical:
        columns.append("hour_of_day")

    if "hour" in schema_columns:
        columns.append("hour")
    elif "date" in schema_columns:
        columns.append("date")

    if "id" in schema_columns:
        columns.append("id")

    return list(dict.fromkeys(columns))


def extract_event_date(dataframe: pd.DataFrame) -> pd.Series:
    """从 date 或 hour 提取归一化日期。"""

    if "date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["date"], errors="coerce").dt.normalize()
        if dates.notna().any():
            return dates

    if "hour" in dataframe.columns:
        hour_text = dataframe["hour"].astype(str).str.replace(r"\.0$", "", regex=True)
        hour_text = hour_text.str.zfill(8)
        date_text = (
            "20"
            + hour_text.str.slice(0, 2)
            + "-"
            + hour_text.str.slice(2, 4)
            + "-"
            + hour_text.str.slice(4, 6)
        )
        return pd.to_datetime(date_text, errors="coerce").dt.normalize()

    raise ValueError("无法从 date / hour 字段解析 event_date。")


def validate_click_values(labels: np.ndarray, context: str) -> None:
    """检查 click 是否仅包含 0 和 1。"""

    unique_values = set(np.unique(labels).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{context} 的 click 存在非法取值：{sorted(unique_values)}")


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_config: FeatureConfig,
) -> np.ndarray:
    """构造特征矩阵：log1p → hour sin/cos → 清洗 → float32。"""

    matrix = dataframe[feature_config.raw_feature_columns].copy()

    for column_name in feature_config.log1p_columns:
        matrix[column_name] = np.log1p(matrix[column_name].astype(np.float64))

    if feature_config.use_hour_cyclical:
        hour_values = dataframe["hour_of_day"].astype(np.float64).to_numpy()
        radians = 2.0 * np.pi * hour_values / 24.0
        matrix["hour_sin"] = np.sin(radians)
        matrix["hour_cos"] = np.cos(radians)

    feature_array = matrix[feature_config.feature_columns].to_numpy(dtype=np.float64)
    feature_array[~np.isfinite(feature_array)] = np.nan
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_array.astype(np.float32)


def iter_file_batches(
    parquet_path: Path,
    read_columns: list[str],
    batch_size: int,
):
    """逐 batch 读取单个 Parquet 文件。"""

    parquet_file = pq.ParquetFile(parquet_path)

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=read_columns,
    ):
        yield record_batch.to_pandas()


def sample_from_split(
    split_name: str,
    parquet_files: list[Path],
    file_quotas: list[int],
    read_columns: list[str],
    feature_config: FeatureConfig,
    sampling_config: SamplingConfig,
    has_id: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict]]:
    """
    从某个 split 的全部文件中按比例抽样。

    返回：X, y, id_sample_df(optional metadata), sampling_records
    """

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    sampling_records: list[dict] = []
    date_counter: dict[str, int] = {}

    total_collected = 0

    for file_index, (parquet_path, file_quota) in enumerate(
        zip(parquet_files, file_quotas)
    ):
        if file_quota <= 0:
            continue

        file_collected = 0
        file_dates: list[pd.Timestamp] = []

        print(
            f"[{split_name}] 文件 {file_index + 1}/{len(parquet_files)}: "
            f"{parquet_path.name}，目标抽样 {file_quota:,} 行"
        )

        for batch_index, batch_df in enumerate(
            iter_file_batches(
                parquet_path,
                read_columns,
                sampling_config.batch_size,
            ),
            start=1,
        ):
            if file_collected >= file_quota:
                break

            remaining = file_quota - file_collected
            if len(batch_df) <= remaining:
                sampled_df = batch_df
            else:
                sampled_df = batch_df.sample(
                    n=remaining,
                    random_state=sampling_config.random_state,
                )

            labels = sampled_df["click"].to_numpy(dtype=np.int8)
            validate_click_values(
                labels,
                context=f"{split_name}/{parquet_path.name} batch {batch_index}",
            )

            features = build_feature_matrix(sampled_df, feature_config)
            event_dates = extract_event_date(sampled_df)

            for event_date in event_dates.dropna().astype(str):
                date_counter[event_date] = date_counter.get(event_date, 0) + 1

            valid_dates = event_dates.dropna()
            if not valid_dates.empty:
                file_dates.extend(valid_dates.tolist())

            feature_parts.append(features)
            label_parts.append(labels)

            if has_id:
                id_parts.append(sampled_df["id"].to_numpy())

            file_collected += len(sampled_df)
            total_collected += len(sampled_df)

            print(
                f"  batch {batch_index}: 抽取 {len(sampled_df):,} 行，"
                f"文件累计 {file_collected:,}/{file_quota:,}，"
                f"split 累计 {total_collected:,}"
            )

            del batch_df, sampled_df, features, labels, event_dates
            gc.collect()

        date_min = min(file_dates).date().isoformat() if file_dates else None
        date_max = max(file_dates).date().isoformat() if file_dates else None

        sampling_records.append(
            {
                "split": split_name,
                "file_name": parquet_path.name,
                "file_rows_total": pq.read_metadata(parquet_path).num_rows,
                "allocated_quota": file_quota,
                "sampled_rows": file_collected,
                "date_min": date_min,
                "date_max": date_max,
            }
        )

    if total_collected == 0:
        raise ValueError(f"{split_name} 未抽到任何样本。")

    x_matrix = np.vstack(feature_parts).astype(np.float32)
    y_vector = np.concatenate(label_parts).astype(np.int8)

    del feature_parts, label_parts
    gc.collect()

    for event_date, row_count in sorted(date_counter.items()):
        sampling_records.append(
            {
                "split": split_name,
                "file_name": "__DATE_SUMMARY__",
                "file_rows_total": np.nan,
                "allocated_quota": np.nan,
                "sampled_rows": row_count,
                "date_min": event_date,
                "date_max": event_date,
            }
        )

    id_df = pd.DataFrame({"id": np.concatenate(id_parts)}) if has_id else pd.DataFrame()

    return x_matrix, y_vector, id_df, sampling_records


def estimate_memory_gb(*arrays: np.ndarray) -> float:
    """估算数组总内存（GB）。"""

    total_bytes = sum(array.nbytes for array in arrays)
    return total_bytes / (1024**3)


def check_memory_limit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    """若估算内存超过阈值则停止。"""

    memory_gb = estimate_memory_gb(x_train, y_train, x_valid, y_valid)

    print("\n内存估算：")
    print(f"  X_train shape={x_train.shape}, dtype={x_train.dtype}")
    print(f"  X_valid shape={x_valid.shape}, dtype={x_valid.dtype}")
    print(f"  估算总内存：{memory_gb:.3f} GB")

    if memory_gb > MAX_MEMORY_GB:
        raise MemoryError(
            f"估算内存 {memory_gb:.3f} GB 超过 {MAX_MEMORY_GB:.1f} GB 限制，已停止训练。"
        )


def load_baseline_metrics(metrics_path: Path, model_name: str) -> dict:
    """读取正式基线指标文件。"""

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"未找到 {model_name} 正式指标文件：{metrics_path}\n"
            f"请先运行对应的正式模式训练脚本。"
        )

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> XGBClassifier:
    """训练 XGBoost 并启用 early stopping。"""

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=-1,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=50,
        verbosity=1,
    )

    print("\n开始训练 XGBoost ...")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=25,
    )

    if not hasattr(model, "best_iteration") or model.best_iteration is None:
        raise ValueError("训练完成后 model.best_iteration 不存在，无法继续评估。")

    return model


def evaluate_model(
    model: XGBClassifier,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    id_df: pd.DataFrame,
    has_id: bool,
) -> tuple[MetricsResult, pd.DataFrame]:
    """在 valid 上评估模型。"""

    probabilities = model.predict_proba(
        x_valid,
        iteration_range=(0, model.best_iteration + 1),
    )[:, 1]

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("预测概率存在超出 [0, 1] 范围的值。")

    predicted_labels = (probabilities >= PREDICTION_THRESHOLD).astype(np.int8)

    metrics = MetricsResult(
        roc_auc=float(roc_auc_score(y_valid, probabilities)),
        log_loss_value=float(log_loss(y_valid, probabilities, labels=[0, 1])),
        accuracy=float(accuracy_score(y_valid, predicted_labels)),
        precision=float(
            precision_score(y_valid, predicted_labels, zero_division=0)
        ),
        recall=float(recall_score(y_valid, predicted_labels, zero_division=0)),
        f1=float(f1_score(y_valid, predicted_labels, zero_division=0)),
        valid_actual_ctr=float(y_valid.mean()),
        valid_mean_predicted_ctr=float(probabilities.mean()),
        predicted_click_ratio_at_threshold=float(predicted_labels.mean()),
        best_iteration=int(model.best_iteration),
    )

    sample_size = min(PREDICTION_SAMPLE_SIZE, len(y_valid))
    sample_records: list[dict] = []

    for index in range(sample_size):
        record = {
            "click": int(y_valid[index]),
            "prediction": float(probabilities[index]),
            "predicted_label": int(predicted_labels[index]),
        }
        if has_id and not id_df.empty:
            record["id"] = id_df.iloc[index]["id"]
        sample_records.append(record)

    sample_columns = ["id", "click", "prediction", "predicted_label"] if has_id else [
        "click",
        "prediction",
        "predicted_label",
    ]

    return metrics, pd.DataFrame(sample_records)[sample_columns]


def build_feature_importance_dataframe(
    model: XGBClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """构建 gain / weight 特征重要性表。"""

    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": [
                float(gain_scores.get(feature_name, 0.0))
                for feature_name in feature_columns
            ],
            "weight_importance": [
                float(weight_scores.get(feature_name, 0.0))
                for feature_name in feature_columns
            ],
        }
    )

    gain_sum = importance_df["gain_importance"].sum()
    if gain_sum > 0:
        importance_df["gain_importance_normalized"] = (
            importance_df["gain_importance"] / gain_sum
        )
    else:
        importance_df["gain_importance_normalized"] = 0.0

    importance_df = importance_df.sort_values(
        "gain_importance",
        ascending=False,
    ).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    return importance_df


def compare_with_baselines(
    metrics: MetricsResult,
    logistic_metrics: dict,
    lightgbm_metrics: dict,
) -> dict:
    """与逻辑回归、LightGBM 正式基线对比。"""

    logistic_auc = float(logistic_metrics["roc_auc"])
    logistic_logloss = float(logistic_metrics["log_loss"])
    logistic_predicted_ctr = float(logistic_metrics["valid_mean_predicted_ctr"])
    logistic_actual_ctr = float(logistic_metrics["valid_actual_ctr"])

    lightgbm_auc = float(lightgbm_metrics["roc_auc"])
    lightgbm_logloss = float(lightgbm_metrics["log_loss"])
    lightgbm_predicted_ctr = float(lightgbm_metrics["valid_mean_predicted_ctr"])
    lightgbm_actual_ctr = float(lightgbm_metrics["valid_actual_ctr"])

    return {
        "logistic_baseline": {
            "roc_auc": logistic_auc,
            "log_loss": logistic_logloss,
            "valid_actual_ctr": logistic_actual_ctr,
            "valid_mean_predicted_ctr": logistic_predicted_ctr,
        },
        "lightgbm_baseline": {
            "roc_auc": lightgbm_auc,
            "log_loss": lightgbm_logloss,
            "valid_actual_ctr": lightgbm_actual_ctr,
            "valid_mean_predicted_ctr": lightgbm_predicted_ctr,
        },
        "comparison_with_logistic": {
            "auc_diff": metrics.roc_auc - logistic_auc,
            "logloss_diff": metrics.log_loss_value - logistic_logloss,
            "predicted_ctr_gap": abs(
                metrics.valid_mean_predicted_ctr - metrics.valid_actual_ctr
            ),
            "logistic_predicted_ctr_gap": abs(
                logistic_predicted_ctr - logistic_actual_ctr
            ),
            "auc_higher_than_logistic": "是"
            if metrics.roc_auc > logistic_auc
            else "否",
            "logloss_lower_than_logistic": "是"
            if metrics.log_loss_value < logistic_logloss
            else "否",
        },
        "comparison_with_lightgbm": {
            "auc_diff": metrics.roc_auc - lightgbm_auc,
            "logloss_diff": metrics.log_loss_value - lightgbm_logloss,
            "predicted_ctr_gap": abs(
                metrics.valid_mean_predicted_ctr - metrics.valid_actual_ctr
            ),
            "lightgbm_predicted_ctr_gap": abs(
                lightgbm_predicted_ctr - lightgbm_actual_ctr
            ),
            "auc_higher_than_lightgbm": "是"
            if metrics.roc_auc > lightgbm_auc
            else "否",
            "logloss_lower_than_lightgbm": "是"
            if metrics.log_loss_value < lightgbm_logloss
            else "否",
        },
    }


def metrics_to_dict(metrics: MetricsResult, comparison: dict, test_mode: bool) -> dict:
    """指标与对比结果转 JSON。"""

    return {
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss_value,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "valid_actual_ctr": metrics.valid_actual_ctr,
        "valid_mean_predicted_ctr": metrics.valid_mean_predicted_ctr,
        "predicted_click_ratio_at_threshold": metrics.predicted_click_ratio_at_threshold,
        "best_iteration": metrics.best_iteration,
        "threshold": PREDICTION_THRESHOLD,
        "test_mode": test_mode,
        "result_note": (
            "测试模式结果仅用于链路检查，不能作为最终结论。"
            if test_mode
            else "正式模式结果，可与 holdout 评估结合使用。"
        ),
        "logistic_baseline": comparison["logistic_baseline"],
        "lightgbm_baseline": comparison["lightgbm_baseline"],
        "comparison_with_logistic": comparison["comparison_with_logistic"],
        "comparison_with_lightgbm": comparison["comparison_with_lightgbm"],
    }


def build_model_comparison_dataframe(
    xgboost_metrics: MetricsResult,
    logistic_metrics: dict,
    lightgbm_metrics: dict,
    test_mode: bool,
) -> pd.DataFrame:
    """生成三模型对比表。"""

    rows = [
        {
            "model": "logistic_regression",
            "mode": "formal",
            "roc_auc": float(logistic_metrics["roc_auc"]),
            "log_loss": float(logistic_metrics["log_loss"]),
            "accuracy": float(logistic_metrics["accuracy"]),
            "precision": float(logistic_metrics["precision"]),
            "recall": float(logistic_metrics["recall"]),
            "f1": float(logistic_metrics["f1"]),
            "valid_actual_ctr": float(logistic_metrics["valid_actual_ctr"]),
            "valid_mean_predicted_ctr": float(
                logistic_metrics["valid_mean_predicted_ctr"]
            ),
            "predicted_click_ratio_at_threshold": float(
                logistic_metrics["predicted_click_ratio_at_threshold"]
            ),
            "best_iteration": logistic_metrics.get("best_iteration"),
            "note": "正式模式基线",
        },
        {
            "model": "lightgbm",
            "mode": "formal",
            "roc_auc": float(lightgbm_metrics["roc_auc"]),
            "log_loss": float(lightgbm_metrics["log_loss"]),
            "accuracy": float(lightgbm_metrics["accuracy"]),
            "precision": float(lightgbm_metrics["precision"]),
            "recall": float(lightgbm_metrics["recall"]),
            "f1": float(lightgbm_metrics["f1"]),
            "valid_actual_ctr": float(lightgbm_metrics["valid_actual_ctr"]),
            "valid_mean_predicted_ctr": float(
                lightgbm_metrics["valid_mean_predicted_ctr"]
            ),
            "predicted_click_ratio_at_threshold": float(
                lightgbm_metrics["predicted_click_ratio_at_threshold"]
            ),
            "best_iteration": lightgbm_metrics.get("best_iteration"),
            "note": "正式模式基线",
        },
        {
            "model": "xgboost",
            "mode": "test" if test_mode else "formal",
            "roc_auc": xgboost_metrics.roc_auc,
            "log_loss": xgboost_metrics.log_loss_value,
            "accuracy": xgboost_metrics.accuracy,
            "precision": xgboost_metrics.precision,
            "recall": xgboost_metrics.recall,
            "f1": xgboost_metrics.f1,
            "valid_actual_ctr": xgboost_metrics.valid_actual_ctr,
            "valid_mean_predicted_ctr": xgboost_metrics.valid_mean_predicted_ctr,
            "predicted_click_ratio_at_threshold": (
                xgboost_metrics.predicted_click_ratio_at_threshold
            ),
            "best_iteration": xgboost_metrics.best_iteration,
            "note": (
                "测试模式链路检查，不能作为最终结论"
                if test_mode
                else "正式模式结果"
            ),
        },
    ]

    return pd.DataFrame(rows)


def write_report(
    context: RunContext,
    sampling_config: SamplingConfig,
    feature_config: FeatureConfig,
    train_row_count: int,
    valid_row_count: int,
    train_date_range: tuple[str | None, str | None],
    valid_date_range: tuple[str | None, str | None],
    metrics: MetricsResult,
    comparison: dict,
    importance_df: pd.DataFrame,
) -> None:
    """写入文本报告。"""

    mode_label = "TEST_MODE=True" if context.test_mode else "TEST_MODE=False"
    logistic_cmp = comparison["comparison_with_logistic"]
    lightgbm_cmp = comparison["comparison_with_lightgbm"]

    lines = [
        "百度 CTR 项目 — XGBoost 基线报告",
        "=" * 70,
        "",
        f"当前模式：{mode_label}",
        "",
        "【模型定义】",
        "  XGBClassifier(objective='binary:logistic', n_estimators=1500,",
        "                learning_rate=0.05, max_depth=6, min_child_weight=10,",
        "                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,",
        "                tree_method='hist', max_bin=256, random_state=42)",
        "",
        "【抽样配置】",
        f"  目标 train 行数：{sampling_config.target_train_rows:,}",
        f"  目标 valid 行数：{sampling_config.target_valid_rows:,}",
        f"  random_state：{sampling_config.random_state}",
        f"  batch_size：{sampling_config.batch_size:,}",
        "",
        "【样本规模】",
        f"  训练样本数：{train_row_count:,}",
        f"  验证样本数：{valid_row_count:,}",
        f"  train 日期覆盖：{train_date_range[0]} ~ {train_date_range[1]}",
        f"  valid 日期覆盖：{valid_date_range[0]} ~ {valid_date_range[1]}",
        "",
        "【特征配置】",
        f"  最终特征数量：{len(feature_config.feature_columns)}",
        f"  特征列表：{feature_config.feature_columns}",
        "",
        "【验证集指标】",
        f"  best_iteration：{metrics.best_iteration}",
        f"  ROC-AUC：{metrics.roc_auc:.6f}",
        f"  LogLoss：{metrics.log_loss_value:.6f}",
        f"  Accuracy：{metrics.accuracy:.6f}",
        f"  Precision：{metrics.precision:.6f}",
        f"  Recall：{metrics.recall:.6f}",
        f"  F1：{metrics.f1:.6f}",
        f"  valid 实际 CTR：{metrics.valid_actual_ctr:.6f}",
        f"  valid 平均预测 CTR：{metrics.valid_mean_predicted_ctr:.6f}",
        f"  0.5 阈值预测点击比例：{metrics.predicted_click_ratio_at_threshold:.6f}",
        "",
        "【与逻辑回归正式基线对比】",
        f"  逻辑回归 ROC-AUC：{comparison['logistic_baseline']['roc_auc']:.6f}",
        f"  XGBoost ROC-AUC：{metrics.roc_auc:.6f}（差异 {logistic_cmp['auc_diff']:+.6f}）",
        f"  AUC 是否更高：{logistic_cmp['auc_higher_than_logistic']}",
        "",
        f"  逻辑回归 LogLoss：{comparison['logistic_baseline']['log_loss']:.6f}",
        f"  XGBoost LogLoss：{metrics.log_loss_value:.6f}（差异 {logistic_cmp['logloss_diff']:+.6f}）",
        f"  LogLoss 是否更低：{logistic_cmp['logloss_lower_than_logistic']}",
        "",
        "【与 LightGBM 正式基线对比】",
        f"  LightGBM ROC-AUC：{comparison['lightgbm_baseline']['roc_auc']:.6f}",
        f"  XGBoost ROC-AUC：{metrics.roc_auc:.6f}（差异 {lightgbm_cmp['auc_diff']:+.6f}）",
        f"  AUC 是否更高：{lightgbm_cmp['auc_higher_than_lightgbm']}",
        "",
        f"  LightGBM LogLoss：{comparison['lightgbm_baseline']['log_loss']:.6f}",
        f"  XGBoost LogLoss：{metrics.log_loss_value:.6f}（差异 {lightgbm_cmp['logloss_diff']:+.6f}）",
        f"  LogLoss 是否更低：{lightgbm_cmp['logloss_lower_than_lightgbm']}",
        "",
        "【特征重要性 Top 15（gain）】",
        "  说明：重要性仅代表模型中的预测贡献，不能解释为因果关系。",
    ]

    for _, row in importance_df.head(15).iterrows():
        lines.append(
            f"  {int(row['rank'])}. {row['feature']}: "
            f"gain={row['gain_importance']:.4f}, "
            f"weight={int(row['weight_importance'])}"
        )

    lines.extend(
        [
            "",
            "【说明】",
            "  - holdout 尚未使用",
            "  - 测试模式结果仅用于链路检查，不能作为最终结论"
            if context.test_mode
            else "  - 当前为限流全量抽样训练，尚未使用 holdout",
        ]
    )

    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_date_range_from_summary(
    sampling_records: list[dict],
    split_name: str,
) -> tuple[str | None, str | None]:
    """从抽样记录中提取日期覆盖范围。"""

    file_records = [
        record
        for record in sampling_records
        if record["split"] == split_name and record["file_name"] != "__DATE_SUMMARY__"
    ]

    date_mins = [record["date_min"] for record in file_records if record["date_min"]]
    date_maxs = [record["date_max"] for record in file_records if record["date_max"]]

    if not date_mins or not date_maxs:
        return None, None

    return min(date_mins), max(date_maxs)


def print_feature_summary(feature_config: FeatureConfig) -> None:
    """打印特征选择结果。"""

    print("\n特征选择结果：")
    print(f"  基础输入特征：{len(feature_config.raw_feature_columns)} 个")
    print(f"  动态生成特征：{feature_config.dynamic_feature_columns}")
    print(f"  最终特征数量：{len(feature_config.feature_columns)}")


def main() -> None:
    """主流程：分配抽样 → 构建矩阵 → 训练 XGBoost → 评估与保存。"""

    context = get_run_context(TEST_MODE)
    sampling_config = get_sampling_config(TEST_MODE)

    print("=" * 70)
    print("XGBoost 基线模型")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"目标 train 样本：{sampling_config.target_train_rows:,}")
    print(f"目标 valid 样本：{sampling_config.target_valid_rows:,}")

    logistic_metrics = load_baseline_metrics(LOGISTIC_METRICS_PATH, "逻辑回归")
    lightgbm_metrics = load_baseline_metrics(LIGHTGBM_METRICS_PATH, "LightGBM")

    train_files = list_parquet_files(TRAIN_DIR)
    valid_files = list_parquet_files(VALID_DIR)

    schema_columns = read_schema_columns(train_files[0])
    feature_config = discover_feature_config(schema_columns)
    read_columns = get_read_columns(feature_config, schema_columns)
    has_id = "id" in schema_columns

    valid_schema = read_schema_columns(valid_files[0])
    missing_valid = [col for col in read_columns if col not in valid_schema]
    if missing_valid:
        raise ValueError(f"valid schema 缺少字段：{missing_valid}")

    print_feature_summary(feature_config)

    train_row_counts = get_file_row_counts(train_files)
    valid_row_counts = get_file_row_counts(valid_files)

    train_quotas = allocate_sample_quotas(
        sampling_config.target_train_rows,
        train_row_counts,
    )
    valid_quotas = allocate_sample_quotas(
        sampling_config.target_valid_rows,
        valid_row_counts,
    )

    print("\n开始 train 抽样 ...")
    x_train, y_train, _, train_sampling_records = sample_from_split(
        split_name="train",
        parquet_files=train_files,
        file_quotas=train_quotas,
        read_columns=read_columns,
        feature_config=feature_config,
        sampling_config=sampling_config,
        has_id=False,
    )

    print("\n开始 valid 抽样 ...")
    x_valid, y_valid, valid_id_df, valid_sampling_records = sample_from_split(
        split_name="valid",
        parquet_files=valid_files,
        file_quotas=valid_quotas,
        read_columns=read_columns,
        feature_config=feature_config,
        sampling_config=sampling_config,
        has_id=has_id,
    )

    check_memory_limit(x_train, y_train, x_valid, y_valid)

    x_train_df = pd.DataFrame(x_train, columns=feature_config.feature_columns)
    x_valid_df = pd.DataFrame(x_valid, columns=feature_config.feature_columns)

    model = train_xgboost_model(x_train_df, y_train, x_valid_df, y_valid)

    metrics, predictions_sample = evaluate_model(
        model,
        x_valid_df,
        y_valid,
        valid_id_df,
        has_id=has_id,
    )

    comparison = compare_with_baselines(metrics, logistic_metrics, lightgbm_metrics)
    importance_df = build_feature_importance_dataframe(
        model,
        feature_config.feature_columns,
    )

    sampling_summary_df = pd.DataFrame(train_sampling_records + valid_sampling_records)
    comparison_df = build_model_comparison_dataframe(
        metrics,
        logistic_metrics,
        lightgbm_metrics,
        test_mode=context.test_mode,
    )
    train_date_range = get_date_range_from_summary(train_sampling_records, "train")
    valid_date_range = get_date_range_from_summary(valid_sampling_records, "valid")

    context.model_path.parent.mkdir(parents=True, exist_ok=True)
    context.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(context.model_path)

    metadata = {
        "model_path": str(context.model_path),
        "feature_columns": feature_config.feature_columns,
        "raw_feature_columns": feature_config.raw_feature_columns,
        "log1p_columns": feature_config.log1p_columns,
        "test_mode": context.test_mode,
        "train_row_count": len(y_train),
        "valid_row_count": len(y_valid),
        "best_iteration": metrics.best_iteration,
        "sampling_config": {
            "target_train_rows": sampling_config.target_train_rows,
            "target_valid_rows": sampling_config.target_valid_rows,
            "random_state": sampling_config.random_state,
            "batch_size": sampling_config.batch_size,
        },
    }

    joblib.dump(metadata, context.metadata_path)
    context.metrics_path.write_text(
        json.dumps(
            metrics_to_dict(metrics, comparison, context.test_mode),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    importance_df.to_csv(context.importance_path, index=False)
    predictions_sample.to_csv(context.predictions_sample_path, index=False)
    sampling_summary_df.to_csv(context.sampling_summary_path, index=False)
    comparison_df.to_csv(context.comparison_path, index=False)

    write_report(
        context=context,
        sampling_config=sampling_config,
        feature_config=feature_config,
        train_row_count=len(y_train),
        valid_row_count=len(y_valid),
        train_date_range=train_date_range,
        valid_date_range=valid_date_range,
        metrics=metrics,
        comparison=comparison,
        importance_df=importance_df,
    )

    logistic_cmp = comparison["comparison_with_logistic"]
    lightgbm_cmp = comparison["comparison_with_lightgbm"]

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"训练样本数：{len(y_train):,}")
    print(f"验证样本数：{len(y_valid):,}")
    print(f"train 日期覆盖：{train_date_range[0]} ~ {train_date_range[1]}")
    print(f"valid 日期覆盖：{valid_date_range[0]} ~ {valid_date_range[1]}")
    print(f"特征数：    {len(feature_config.feature_columns)}")
    print(f"best_iteration：{metrics.best_iteration}")
    print(f"ROC-AUC：   {metrics.roc_auc:.6f}")
    print(f"LogLoss：   {metrics.log_loss_value:.6f}")
    print(f"实际 CTR：  {metrics.valid_actual_ctr:.6f}")
    print(f"平均预测 CTR：{metrics.valid_mean_predicted_ctr:.6f}")
    print(
        f"与逻辑回归 AUC 差异：{logistic_cmp['auc_diff']:+.6f}，"
        f"LogLoss 差异：{logistic_cmp['logloss_diff']:+.6f}"
    )
    print(
        f"与 LightGBM AUC 差异：{lightgbm_cmp['auc_diff']:+.6f}，"
        f"LogLoss 差异：{lightgbm_cmp['logloss_diff']:+.6f}"
    )
    print("输出文件：")
    print(f"  模型：         {context.model_path}")
    print(f"  元数据：       {context.metadata_path}")
    print(f"  指标：         {context.metrics_path}")
    print(f"  报告：         {context.report_path}")
    print(f"  特征重要性：   {context.importance_path}")
    print(f"  预测样本：     {context.predictions_sample_path}")
    print(f"  抽样摘要：     {context.sampling_summary_path}")
    print(f"  三模型对比：   {context.comparison_path}")
    print("XGBoost 基线测试完成，holdout 尚未使用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
