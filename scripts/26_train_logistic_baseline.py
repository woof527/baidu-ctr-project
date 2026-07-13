"""
百度 CTR 项目 — 增量式逻辑回归基线（SGDClassifier）

功能：
    使用 StandardScaler + SGDClassifier 对 target_encoded 特征做增量训练，
    在 valid 上评估 ROC-AUC、LogLoss 等指标。禁止读取 holdout。

数据输入：
    data/features/target_encoded/train/*.parquet
    data/features/target_encoded/valid/*.parquet

数据输出（测试模式）：
    models/logistic_baseline_test.joblib
    outputs/logistic_baseline_test_*.json/csv/txt

数据输出（正式模式）：
    models/logistic_baseline.joblib
    outputs/logistic_baseline_*.json/csv/txt

用法：
    python scripts/26_train_logistic_baseline.py
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

TEST_MODE = False

BATCH_SIZE = 200_000
TEST_TRAIN_FILE_LIMIT = 3
TEST_VALID_FILE_LIMIT = 2
TEST_BATCH_LIMIT_PER_FILE = 2

PREDICTION_THRESHOLD = 0.5
PREDICTION_SAMPLE_SIZE = 10_000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

TRAIN_DIR = Path("data/features/target_encoded/train")
VALID_DIR = Path("data/features/target_encoded/valid")

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
    "hour_of_day",  # 使用 sin/cos 代替
}

DYNAMIC_FEATURES = ("hour_sin", "hour_cos")


@dataclass
class FeatureConfig:
    """特征配置：原始列、log1p 列、最终模型输入列。"""

    raw_feature_columns: list[str]
    log1p_columns: list[str]
    dynamic_feature_columns: list[str]
    feature_columns: list[str]
    use_hour_cyclical: bool


@dataclass
class RunContext:
    """运行上下文与输出路径。"""

    test_mode: bool
    model_path: Path
    metrics_path: Path
    coefficients_path: Path
    report_path: Path
    predictions_sample_path: Path


@dataclass
class MetricsResult:
    """验证集评估指标。"""

    roc_auc: float | None
    log_loss_value: float | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    valid_actual_ctr: float
    valid_mean_predicted_ctr: float
    predicted_click_ratio_at_threshold: float
    roc_auc_note: str | None = None


def get_run_context(test_mode: bool) -> RunContext:
    """根据运行模式返回输出路径。"""

    if test_mode:
        return RunContext(
            test_mode=True,
            model_path=Path("models/logistic_baseline_test.joblib"),
            metrics_path=Path("outputs/logistic_baseline_test_metrics.json"),
            coefficients_path=Path("outputs/logistic_baseline_test_coefficients.csv"),
            report_path=Path("outputs/logistic_baseline_test_report.txt"),
            predictions_sample_path=Path(
                "outputs/logistic_baseline_test_predictions_sample.csv"
            ),
        )

    return RunContext(
        test_mode=False,
        model_path=Path("models/logistic_baseline.joblib"),
        metrics_path=Path("outputs/logistic_baseline_valid_metrics.json"),
        coefficients_path=Path("outputs/logistic_baseline_coefficients.csv"),
        report_path=Path("outputs/logistic_baseline_report.txt"),
        predictions_sample_path=Path(
            "outputs/logistic_baseline_valid_predictions_sample.csv"
        ),
    )


def list_parquet_files(parquet_dir: Path, limit: int | None = None) -> list[Path]:
    """列出 Parquet 分块；目录不存在或为空时抛出错误。"""

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

    if limit is not None:
        return files[:limit]

    return files


def read_schema_columns(parquet_path: Path) -> list[str]:
    """读取 Parquet schema 列名。"""

    return pq.read_schema(parquet_path).names


def discover_feature_config(schema_columns: list[str]) -> FeatureConfig:
    """
    从 schema 自动确定特征列表。

    优先选择工程化数值特征（频次 / 历史统计 / TE）及 is_weekend；
    hour_of_day 转为 hour_sin / hour_cos。
    """

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

    missing_binary = [name for name in BINARY_FEATURES if name not in schema_columns]
    if missing_binary:
        raise ValueError(f"期望存在的二元特征缺失：{missing_binary}")

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


def get_read_columns(feature_config: FeatureConfig, schema_columns: list[str]) -> list[str]:
    """每个 batch 需要读取的列。"""

    columns = ["click", *feature_config.raw_feature_columns]

    if feature_config.use_hour_cyclical:
        columns.append("hour_of_day")

    if "id" in schema_columns:
        columns.append("id")

    return list(dict.fromkeys(columns))


def validate_click_values(labels: np.ndarray, context: str) -> None:
    """检查 click 是否仅包含 0 和 1。"""

    unique_values = set(np.unique(labels).tolist())
    allowed = {0, 1}

    if not unique_values.issubset(allowed):
        raise ValueError(f"{context} 的 click 存在非法取值：{sorted(unique_values)}")


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_config: FeatureConfig,
) -> np.ndarray:
    """
    构造单个 batch 的特征矩阵。

    步骤：log1p → hour sin/cos → inf 转 NaN → NaN 填 0 → float32。
    """

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
    batch_limit: int | None,
):
    """使用 PyArrow iter_batches 逐 batch 读取单个 Parquet 文件。"""

    parquet_file = pq.ParquetFile(parquet_path)
    batch_count = 0

    for record_batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=read_columns,
    ):
        if batch_limit is not None and batch_count >= batch_limit:
            break

        dataframe = record_batch.to_pandas()
        batch_count += 1
        yield dataframe

        del dataframe
        gc.collect()


def partial_fit_scaler_on_files(
    parquet_files: list[Path],
    read_columns: list[str],
    feature_config: FeatureConfig,
    scaler: StandardScaler,
    batch_size: int,
    batch_limit: int | None,
    split_label: str,
) -> int:
    """第一遍遍历 train：StandardScaler.partial_fit。"""

    total_rows = 0

    for file_index, parquet_path in enumerate(parquet_files, start=1):
        print(f"[Scaler] {split_label} 文件 {file_index}/{len(parquet_files)}: {parquet_path.name}")

        for batch_index, batch_df in enumerate(
            iter_file_batches(parquet_path, read_columns, batch_size, batch_limit),
            start=1,
        ):
            features = build_feature_matrix(batch_df, feature_config)
            scaler.partial_fit(features)

            row_count = len(batch_df)
            total_rows += row_count
            print(
                f"  batch {batch_index}: {row_count:,} 行，"
                f"累计 {total_rows:,} 行"
            )

            del batch_df, features
            gc.collect()

    return total_rows


def train_model_on_files(
    parquet_files: list[Path],
    read_columns: list[str],
    feature_config: FeatureConfig,
    scaler: StandardScaler,
    model: SGDClassifier,
    batch_size: int,
    batch_limit: int | None,
    split_label: str,
) -> tuple[int, float]:
    """第二遍遍历 train：transform + partial_fit。"""

    total_rows = 0
    total_clicks = 0
    first_batch = True

    for file_index, parquet_path in enumerate(parquet_files, start=1):
        print(f"[Train] {split_label} 文件 {file_index}/{len(parquet_files)}: {parquet_path.name}")

        for batch_index, batch_df in enumerate(
            iter_file_batches(parquet_path, read_columns, batch_size, batch_limit),
            start=1,
        ):
            labels = batch_df["click"].to_numpy(dtype=np.int8)
            validate_click_values(labels, context=f"{parquet_path.name} batch {batch_index}")

            features = build_feature_matrix(batch_df, feature_config)
            scaled_features = scaler.transform(features)

            if first_batch:
                model.partial_fit(scaled_features, labels, classes=np.array([0, 1]))
                first_batch = False
            else:
                model.partial_fit(scaled_features, labels)

            row_count = len(batch_df)
            total_rows += row_count
            total_clicks += int(labels.sum())

            print(
                f"  batch {batch_index}: {row_count:,} 行，"
                f"累计 {total_rows:,} 行"
            )

            del batch_df, labels, features, scaled_features
            gc.collect()

    train_ctr = total_clicks / total_rows if total_rows > 0 else 0.0
    return total_rows, train_ctr


def evaluate_valid_files(
    parquet_files: list[Path],
    read_columns: list[str],
    feature_config: FeatureConfig,
    scaler: StandardScaler,
    model: SGDClassifier,
    batch_size: int,
    batch_limit: int | None,
    has_id: bool,
) -> tuple[MetricsResult, pd.DataFrame, int, float]:
    """分批评估 valid，累计标签与预测概率。"""

    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    sample_records: list[dict] = []
    total_rows = 0
    total_clicks = 0

    for file_index, parquet_path in enumerate(parquet_files, start=1):
        print(f"[Valid] 文件 {file_index}/{len(parquet_files)}: {parquet_path.name}")

        for batch_index, batch_df in enumerate(
            iter_file_batches(parquet_path, read_columns, batch_size, batch_limit),
            start=1,
        ):
            labels = batch_df["click"].to_numpy(dtype=np.int8)
            validate_click_values(labels, context=f"{parquet_path.name} batch {batch_index}")

            features = build_feature_matrix(batch_df, feature_config)
            scaled_features = scaler.transform(features)
            probabilities = model.predict_proba(scaled_features)[:, 1]

            if (probabilities < 0).any() or (probabilities > 1).any():
                raise ValueError(
                    f"{parquet_path.name} batch {batch_index} 存在超出 [0, 1] 的预测概率"
                )

            y_true_parts.append(labels)
            y_prob_parts.append(probabilities.astype(np.float64))

            if len(sample_records) < PREDICTION_SAMPLE_SIZE:
                remaining = PREDICTION_SAMPLE_SIZE - len(sample_records)
                take_n = min(remaining, len(batch_df))

                sample_df = batch_df.iloc[:take_n].copy()
                sample_probs = probabilities[:take_n]
                sample_labels = (sample_probs >= PREDICTION_THRESHOLD).astype(np.int8)

                for row_idx in range(take_n):
                    record = {
                        "click": int(sample_df.iloc[row_idx]["click"]),
                        "prediction": float(sample_probs[row_idx]),
                        "predicted_label": int(sample_labels[row_idx]),
                    }
                    if has_id:
                        record["id"] = sample_df.iloc[row_idx]["id"]
                    sample_records.append(record)

            row_count = len(batch_df)
            total_rows += row_count
            total_clicks += int(labels.sum())

            print(
                f"  batch {batch_index}: {row_count:,} 行，"
                f"累计 {total_rows:,} 行"
            )

            del batch_df, labels, features, scaled_features, probabilities
            gc.collect()

    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int8)
    y_prob = np.concatenate(y_prob_parts) if y_prob_parts else np.array([], dtype=np.float64)

    if total_rows == 0:
        raise ValueError("valid 没有读取到任何样本，无法评估。")

    valid_ctr = total_clicks / total_rows
    predicted_labels = (y_prob >= PREDICTION_THRESHOLD).astype(np.int8)

    metrics = MetricsResult(
        roc_auc=None,
        log_loss_value=None,
        accuracy=float(accuracy_score(y_true, predicted_labels)),
        precision=float(
            precision_score(y_true, predicted_labels, zero_division=0)
        ),
        recall=float(recall_score(y_true, predicted_labels, zero_division=0)),
        f1=float(f1_score(y_true, predicted_labels, zero_division=0)),
        valid_actual_ctr=float(valid_ctr),
        valid_mean_predicted_ctr=float(y_prob.mean()) if len(y_prob) > 0 else 0.0,
        predicted_click_ratio_at_threshold=float(predicted_labels.mean())
        if len(predicted_labels) > 0
        else 0.0,
    )

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        metrics.roc_auc_note = (
            f"valid 样本仅包含单一类别 {unique_classes.tolist()}，无法计算 ROC-AUC。"
        )
    else:
        metrics.roc_auc = float(roc_auc_score(y_true, y_prob))

    if len(unique_classes) >= 2:
        metrics.log_loss_value = float(log_loss(y_true, y_prob, labels=[0, 1]))
    else:
        metrics.log_loss_value = None

    sample_columns = ["id", "click", "prediction", "predicted_label"] if has_id else [
        "click",
        "prediction",
        "predicted_label",
    ]
    predictions_sample = pd.DataFrame(sample_records)[sample_columns]

    return metrics, predictions_sample, total_rows, valid_ctr


def build_coefficients_dataframe(
    model: SGDClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """构建系数表。"""

    coefficients = model.coef_.ravel()
    coefficient_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )
    coefficient_df["direction"] = np.where(
        coefficient_df["coefficient"] >= 0,
        "positive",
        "negative",
    )

    return coefficient_df.sort_values("abs_coefficient", ascending=False).reset_index(
        drop=True
    )


def metrics_to_dict(metrics: MetricsResult) -> dict:
    """指标转为 JSON 可序列化字典。"""

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
        "roc_auc_note": metrics.roc_auc_note,
        "threshold": PREDICTION_THRESHOLD,
    }


def write_report(
    context: RunContext,
    feature_config: FeatureConfig,
    train_files: list[Path],
    valid_files: list[Path],
    train_row_count: int,
    valid_row_count: int,
    train_ctr: float,
    valid_ctr: float,
    metrics: MetricsResult,
    coefficients_df: pd.DataFrame,
) -> None:
    """写入文本报告。"""

    mode_label = "TEST_MODE=True（测试模式）" if context.test_mode else "TEST_MODE=False（正式模式）"

    lines = [
        "百度 CTR 项目 — 增量式逻辑回归基线报告",
        "=" * 70,
        "",
        f"当前模式：{mode_label}",
        "",
        "【模型定义】",
        "  SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4,",
        "               learning_rate='optimal', random_state=42, average=True)",
        "",
        "【数据读取范围】",
        f"  train 目录：{TRAIN_DIR}",
        f"  valid 目录：{VALID_DIR}",
        f"  train 文件数：{len(train_files)}",
        f"  valid 文件数：{len(valid_files)}",
        f"  batch_size：{BATCH_SIZE:,}",
        "",
        "【样本规模】",
        f"  训练样本数：{train_row_count:,}",
        f"  验证样本数：{valid_row_count:,}",
        f"  训练集点击率：{train_ctr:.6f}",
        f"  验证集点击率：{valid_ctr:.6f}",
        "",
        "【特征配置】",
        f"  最终特征数量：{len(feature_config.feature_columns)}",
        f"  基础输入特征：{feature_config.raw_feature_columns}",
        f"  log1p 特征：{feature_config.log1p_columns}",
        f"  动态生成特征：{feature_config.dynamic_feature_columns}",
        f"  最终特征列表：{feature_config.feature_columns}",
        "",
        "【验证集指标】",
        f"  ROC-AUC：{metrics.roc_auc}",
        f"  LogLoss：{metrics.log_loss_value}",
        f"  Accuracy：{metrics.accuracy:.6f}",
        f"  Precision：{metrics.precision:.6f}",
        f"  Recall：{metrics.recall:.6f}",
        f"  F1：{metrics.f1:.6f}",
        f"  valid 实际点击率：{metrics.valid_actual_ctr:.6f}",
        f"  valid 平均预测点击率：{metrics.valid_mean_predicted_ctr:.6f}",
        f"  0.5 阈值预测为点击比例：{metrics.predicted_click_ratio_at_threshold:.6f}",
    ]

    if metrics.roc_auc_note:
        lines.extend(["", f"  说明：{metrics.roc_auc_note}"])

    lines.extend(
        [
            "",
            "【系数绝对值 Top 15】",
        ]
    )

    for _, row in coefficients_df.head(15).iterrows():
        lines.append(
            f"  {row['feature']}: coef={row['coefficient']:.6f}, "
            f"abs={row['abs_coefficient']:.6f}, direction={row['direction']}"
        )

    lines.extend(
        [
            "",
            "【当前局限性】",
            "  - 线性模型，无法自动捕捉复杂非线性交互",
            "  - 增量训练对特征缩放与 batch 顺序敏感",
            "  - 未使用 class_weight / 采样策略",
            "",
            "【重要说明】",
            "  - holdout 尚未使用",
        ]
    )

    if context.test_mode:
        lines.append("  - 当前为测试模式，结果不能作为正式模型结论")

    context.report_path.parent.mkdir(parents=True, exist_ok=True)
    context.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    context: RunContext,
    model: SGDClassifier,
    scaler: StandardScaler,
    feature_config: FeatureConfig,
    train_row_count: int,
    valid_row_count: int,
    metrics: MetricsResult,
    coefficients_df: pd.DataFrame,
    predictions_sample: pd.DataFrame,
) -> None:
    """保存模型、指标、系数与预测样本。"""

    context.model_path.parent.mkdir(parents=True, exist_ok=True)
    context.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_config.feature_columns,
        "raw_feature_columns": feature_config.raw_feature_columns,
        "log1p_columns": feature_config.log1p_columns,
        "test_mode": context.test_mode,
        "batch_size": BATCH_SIZE,
        "train_row_count": train_row_count,
        "valid_row_count": valid_row_count,
        "threshold": PREDICTION_THRESHOLD,
    }

    joblib.dump(bundle, context.model_path)
    context.metrics_path.write_text(
        json.dumps(metrics_to_dict(metrics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coefficients_df.to_csv(context.coefficients_path, index=False)
    predictions_sample.to_csv(context.predictions_sample_path, index=False)


def print_feature_summary(feature_config: FeatureConfig) -> None:
    """打印特征选择结果。"""

    print("\n特征选择结果：")
    print(f"  基础输入特征（{len(feature_config.raw_feature_columns)} 个）：")
    for feature_name in feature_config.raw_feature_columns:
        print(f"    - {feature_name}")

    print(f"  动态生成特征（{len(feature_config.dynamic_feature_columns)} 个）：")
    for feature_name in feature_config.dynamic_feature_columns:
        print(f"    - {feature_name}")

    print(f"  最终特征数量：{len(feature_config.feature_columns)}")


def main() -> None:
    """主流程：特征选择 → scaler 第一遍 → 模型第二遍 → valid 评估 → 保存输出。"""

    context = get_run_context(TEST_MODE)
    batch_limit = TEST_BATCH_LIMIT_PER_FILE if TEST_MODE else None
    train_file_limit = TEST_TRAIN_FILE_LIMIT if TEST_MODE else None
    valid_file_limit = TEST_VALID_FILE_LIMIT if TEST_MODE else None

    print("=" * 70)
    print("增量式逻辑回归基线（SGDClassifier）")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"BATCH_SIZE：{BATCH_SIZE:,}")

    train_files = list_parquet_files(TRAIN_DIR, limit=train_file_limit)
    valid_files = list_parquet_files(VALID_DIR, limit=valid_file_limit)

    schema_columns = read_schema_columns(train_files[0])
    feature_config = discover_feature_config(schema_columns)
    read_columns = get_read_columns(feature_config, schema_columns)
    has_id = "id" in schema_columns

    missing_in_train = [
        column_name
        for column_name in read_columns
        if column_name not in schema_columns
    ]
    if missing_in_train:
        raise ValueError(f"train schema 缺少必要字段：{missing_in_train}")

    valid_schema_columns = read_schema_columns(valid_files[0])
    missing_in_valid = [
        column_name
        for column_name in read_columns
        if column_name not in valid_schema_columns
    ]
    if missing_in_valid:
        raise ValueError(f"valid schema 缺少必要字段：{missing_in_valid}")

    print_feature_summary(feature_config)

    scaler = StandardScaler()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
        average=True,
    )

    print("\n第一遍：StandardScaler.partial_fit ...")
    train_row_count_pass1 = partial_fit_scaler_on_files(
        parquet_files=train_files,
        read_columns=read_columns,
        feature_config=feature_config,
        scaler=scaler,
        batch_size=BATCH_SIZE,
        batch_limit=batch_limit,
        split_label="train",
    )

    print("\n第二遍：SGDClassifier.partial_fit ...")
    train_row_count, train_ctr = train_model_on_files(
        parquet_files=train_files,
        read_columns=read_columns,
        feature_config=feature_config,
        scaler=scaler,
        model=model,
        batch_size=BATCH_SIZE,
        batch_limit=batch_limit,
        split_label="train",
    )

    if train_row_count != train_row_count_pass1:
        raise ValueError(
            f"train 两遍读取行数不一致：pass1={train_row_count_pass1:,}, "
            f"pass2={train_row_count:,}"
        )

    print("\n验证集评估 ...")
    metrics, predictions_sample, valid_row_count, valid_ctr = evaluate_valid_files(
        parquet_files=valid_files,
        read_columns=read_columns,
        feature_config=feature_config,
        scaler=scaler,
        model=model,
        batch_size=BATCH_SIZE,
        batch_limit=batch_limit,
        has_id=has_id,
    )

    coefficients_df = build_coefficients_dataframe(model, feature_config.feature_columns)

    save_outputs(
        context=context,
        model=model,
        scaler=scaler,
        feature_config=feature_config,
        train_row_count=train_row_count,
        valid_row_count=valid_row_count,
        metrics=metrics,
        coefficients_df=coefficients_df,
        predictions_sample=predictions_sample,
    )
    write_report(
        context=context,
        feature_config=feature_config,
        train_files=train_files,
        valid_files=valid_files,
        train_row_count=train_row_count,
        valid_row_count=valid_row_count,
        train_ctr=train_ctr,
        valid_ctr=valid_ctr,
        metrics=metrics,
        coefficients_df=coefficients_df,
    )

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"训练样本数：{train_row_count:,}")
    print(f"验证样本数：{valid_row_count:,}")
    print(f"特征数：    {len(feature_config.feature_columns)}")
    print(f"ROC-AUC：   {metrics.roc_auc}")
    print(f"LogLoss：   {metrics.log_loss_value}")
    print("输出文件：")
    print(f"  模型：     {context.model_path}")
    print(f"  指标：     {context.metrics_path}")
    print(f"  系数：     {context.coefficients_path}")
    print(f"  报告：     {context.report_path}")
    print(f"  预测样本： {context.predictions_sample_path}")
    print("逻辑回归测试链路运行完成，holdout 尚未使用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
