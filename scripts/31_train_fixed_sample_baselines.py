"""
百度 CTR 项目 — 固定共享样本三模型基线训练

功能：
    在第 30 步保存的固定 train / valid 样本上，复跑逻辑回归、LightGBM、
    XGBoost 基线参数并进行公平比较。禁止读取 holdout。

数据输入：
    data/tuning/lightgbm_train/*.parquet
    data/tuning/lightgbm_valid/*.parquet
    outputs/fixed_tuning_sample_metadata.json

用法：
    python scripts/31_train_fixed_sample_baselines.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
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
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

TRAIN_INPUT_DIR = Path("data/tuning/lightgbm_train")
VALID_INPUT_DIR = Path("data/tuning/lightgbm_valid")

FIXED_SAMPLE_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")

FULL_TRAIN_ROWS = 2_000_000
FULL_VALID_ROWS = 500_000
TEST_TRAIN_ROWS = 100_000
TEST_VALID_ROWS = 50_000

RANDOM_STATE = 42
BATCH_SIZE = 200_000
PREDICTION_THRESHOLD = 0.5
PROB_CLIP_EPS = 1e-7

ORIGINAL_BASELINE_METRICS = {
    "Logistic Regression": Path("outputs/logistic_baseline_valid_metrics.json"),
    "LightGBM": Path("outputs/lightgbm_baseline_valid_metrics.json"),
    "XGBoost": Path("outputs/xgboost_baseline_valid_metrics.json"),
}


@dataclass
class ModelMetrics:
    """单模型评估指标。"""

    model: str
    data_scope: str
    train_rows: int
    valid_rows: int
    feature_count: int
    roc_auc: float
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    valid_actual_ctr: float
    valid_mean_predicted_ctr: float
    calibration_gap: float
    predicted_click_ratio_at_threshold: float
    threshold: float
    best_iteration: int | None
    training_seconds: float
    model_size_mb: float
    random_state: int
    train_id_sha256: str
    valid_id_sha256: str
    holdout_used: bool = False


@dataclass
class OutputPaths:
    """输出路径。"""

    test_mode: bool
    logistic_model: Path
    logistic_scaler: Path
    lightgbm_model: Path
    xgboost_model: Path
    predictions: Path
    metrics_csv: Path
    comparison_csv: Path
    report_txt: Path
    metadata_json: Path


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    if test_mode:
        return OutputPaths(
            test_mode=True,
            logistic_model=Path("models/test/fixed_sample_logistic_model.joblib"),
            logistic_scaler=Path("models/test/fixed_sample_logistic_scaler.joblib"),
            lightgbm_model=Path("models/test/fixed_sample_lightgbm_model.joblib"),
            xgboost_model=Path("models/test/fixed_sample_xgboost_model.json"),
            predictions=Path(
                "outputs/predictions/fixed_sample_baseline_valid_predictions_test.parquet"
            ),
            metrics_csv=Path("outputs/fixed_sample_baseline_metrics_test.csv"),
            comparison_csv=Path("outputs/fixed_sample_baseline_comparison_test.csv"),
            report_txt=Path("outputs/fixed_sample_baseline_report_test.txt"),
            metadata_json=Path("outputs/fixed_sample_baseline_metadata_test.json"),
        )

    return OutputPaths(
        test_mode=False,
        logistic_model=Path("models/fixed_sample_logistic_model.joblib"),
        logistic_scaler=Path("models/fixed_sample_logistic_scaler.joblib"),
        lightgbm_model=Path("models/fixed_sample_lightgbm_model.joblib"),
        xgboost_model=Path("models/fixed_sample_xgboost_model.json"),
        predictions=Path(
            "outputs/predictions/fixed_sample_baseline_valid_predictions.parquet"
        ),
        metrics_csv=Path("outputs/fixed_sample_baseline_metrics.csv"),
        comparison_csv=Path("outputs/fixed_sample_baseline_comparison.csv"),
        report_txt=Path("outputs/fixed_sample_baseline_report.txt"),
        metadata_json=Path("outputs/fixed_sample_baseline_metadata.json"),
    )


def get_row_limits(test_mode: bool) -> tuple[int, int]:
    """返回 train / valid 行数上限。"""

    if test_mode:
        return TEST_TRAIN_ROWS, TEST_VALID_ROWS
    return FULL_TRAIN_ROWS, FULL_VALID_ROWS


def load_fixed_sample_metadata(metadata_path: Path) -> dict:
    """读取并校验固定样本元数据。"""

    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到固定样本元数据：{metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("validation_passed") is not True:
        raise ValueError(
            f"固定样本 validation_passed={metadata.get('validation_passed')!r}，禁止训练。"
        )

    if metadata.get("holdout_used") is not False:
        raise ValueError(
            f"固定样本 holdout_used={metadata.get('holdout_used')!r}，禁止训练。"
        )

    required_keys = (
        "feature_columns",
        "feature_count",
        "actual_train_rows",
        "actual_valid_rows",
        "train_id_sha256",
        "valid_id_sha256",
    )
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"固定样本元数据缺少字段：{key}")

    return metadata


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    """稳定排序 Parquet 文件。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(f"未找到固定样本目录：{parquet_dir}")

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{parquet_dir}")

    return files


def calculate_id_sha256(parquet_files: list[Path], batch_size: int = BATCH_SIZE) -> str:
    """按第 30 步规则计算 id SHA256。"""

    hasher = hashlib.sha256()

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(columns=["id"], batch_size=batch_size):
            for id_value in record_batch.to_pandas()["id"]:
                hasher.update((str(id_value) + "\n").encode("utf-8"))

    digest = hasher.hexdigest()
    if len(digest) != 64:
        raise ValueError(f"SHA256 长度异常：{len(digest)}")

    return digest


def verify_formal_fingerprints(
    train_files: list[Path],
    valid_files: list[Path],
    metadata: dict,
) -> tuple[str, str]:
    """正式模式验证 SHA256 指纹。"""

    train_sha256 = calculate_id_sha256(train_files)
    valid_sha256 = calculate_id_sha256(valid_files)

    expected_train = metadata["train_id_sha256"]
    expected_valid = metadata["valid_id_sha256"]

    if train_sha256 != expected_train:
        raise ValueError(
            f"train_id_sha256 不一致：当前 {train_sha256}，元数据 {expected_train}"
        )

    if valid_sha256 != expected_valid:
        raise ValueError(
            f"valid_id_sha256 不一致：当前 {valid_sha256}，元数据 {expected_valid}"
        )

    return train_sha256, valid_sha256


def load_fixed_split(
    parquet_files: list[Path],
    feature_columns: list[str],
    max_rows: int | None,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """加载固定 split 的特征矩阵与元数据。"""

    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    collected = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)

        for record_batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = record_batch.to_pandas()

            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            feature_parts.append(batch_df[feature_columns].to_numpy(dtype=np.float32))
            label_parts.append(batch_df["click"].to_numpy(dtype=np.int8))
            meta_parts.append(batch_df[["id", "click", "split_date"]].copy())

            collected += len(batch_df)

            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    if collected == 0:
        raise ValueError("未读取到任何固定样本行。")

    meta_df = pd.concat(meta_parts, ignore_index=True)
    x_matrix = np.vstack(feature_parts).astype(np.float32)
    y_vector = np.concatenate(label_parts).astype(np.int8)

    del feature_parts, label_parts, meta_parts
    gc.collect()

    return x_matrix, y_vector, meta_df


def partial_fit_scaler_on_train(
    train_files: list[Path],
    feature_columns: list[str],
    scaler: StandardScaler,
    max_rows: int | None,
) -> int:
    """第一遍：StandardScaler.partial_fit。"""

    total_rows = 0

    for file_index, parquet_path in enumerate(train_files, start=1):
        print(f"[Scaler] train 文件 {file_index}/{len(train_files)}: {parquet_path.name}")

        parquet_file = pq.ParquetFile(parquet_path)
        file_rows = 0

        for batch_index, record_batch in enumerate(
            parquet_file.iter_batches(batch_size=BATCH_SIZE),
            start=1,
        ):
            if max_rows is not None and total_rows >= max_rows:
                break

            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - total_rows
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            features = batch_df[feature_columns].to_numpy(dtype=np.float32)
            scaler.partial_fit(features)

            row_count = len(batch_df)
            total_rows += row_count
            file_rows += row_count

            print(
                f"  batch {batch_index}: {row_count:,} 行，"
                f"文件累计 {file_rows:,}，总累计 {total_rows:,}"
            )

            del batch_df, features
            gc.collect()

        if max_rows is not None and total_rows >= max_rows:
            break

    return total_rows


def train_logistic_model(
    train_files: list[Path],
    feature_columns: list[str],
    max_train_rows: int | None,
) -> tuple[SGDClassifier, StandardScaler, int, float]:
    """两遍增量训练逻辑回归。"""

    scaler = StandardScaler()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=RANDOM_STATE,
        average=True,
    )

    print("\n逻辑回归 — 第一遍：StandardScaler.partial_fit ...")
    pass1_rows = partial_fit_scaler_on_train(
        train_files,
        feature_columns,
        scaler,
        max_train_rows,
    )

    print("\n逻辑回归 — 第二遍：SGDClassifier.partial_fit ...")
    total_rows = 0
    total_clicks = 0
    first_batch = True

    for file_index, parquet_path in enumerate(train_files, start=1):
        print(f"[Train-LR] 文件 {file_index}/{len(train_files)}: {parquet_path.name}")

        parquet_file = pq.ParquetFile(parquet_path)

        for batch_index, record_batch in enumerate(
            parquet_file.iter_batches(batch_size=BATCH_SIZE),
            start=1,
        ):
            if max_train_rows is not None and total_rows >= max_train_rows:
                break

            batch_df = record_batch.to_pandas()
            if max_train_rows is not None:
                remaining = max_train_rows - total_rows
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            labels = batch_df["click"].to_numpy(dtype=np.int8)
            features = batch_df[feature_columns].to_numpy(dtype=np.float32)
            scaled_features = scaler.transform(features)

            if first_batch:
                model.partial_fit(scaled_features, labels, classes=np.array([0, 1]))
                first_batch = False
            else:
                model.partial_fit(scaled_features, labels)

            total_rows += len(batch_df)
            total_clicks += int(labels.sum())

            print(
                f"  batch {batch_index}: {len(batch_df):,} 行，累计 {total_rows:,} 行"
            )

            del batch_df, labels, features, scaled_features
            gc.collect()

        if max_train_rows is not None and total_rows >= max_train_rows:
            break

    if total_rows != pass1_rows:
        raise ValueError(
            f"逻辑回归两遍 train 行数不一致：pass1={pass1_rows:,}, pass2={total_rows:,}"
        )

    train_ctr = total_clicks / total_rows if total_rows > 0 else 0.0
    return model, scaler, total_rows, train_ctr


def predict_logistic_probabilities(
    valid_x: np.ndarray,
    scaler: StandardScaler,
    model: SGDClassifier,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """分批预测逻辑回归概率。"""

    probability_parts: list[np.ndarray] = []

    for start in range(0, len(valid_x), batch_size):
        end = min(start + batch_size, len(valid_x))
        scaled = scaler.transform(valid_x[start:end])
        probabilities = model.predict_proba(scaled)[:, 1]
        probability_parts.append(probabilities.astype(np.float64))

    return np.concatenate(probability_parts)


def train_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[lgb.LGBMClassifier, float]:
    """训练 LightGBM（参数与第 27 步一致）。"""

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=500,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    print("\n开始训练 LightGBM ...")
    start_time = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=25),
        ],
    )
    elapsed = time.perf_counter() - start_time

    return model, elapsed


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[XGBClassifier, float]:
    """训练 XGBoost（参数与第 28 步一致）。"""

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
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric=["logloss", "auc"],
        early_stopping_rounds=50,
        verbosity=1,
    )

    print("\n开始训练 XGBoost ...")
    start_time = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=25,
    )
    elapsed = time.perf_counter() - start_time

    if not hasattr(model, "best_iteration") or model.best_iteration is None:
        raise ValueError("XGBoost 训练完成后 best_iteration 不存在。")

    return model, elapsed


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """裁剪概率用于 LogLoss 计算。"""

    return np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    train_rows: int,
    valid_rows: int,
    feature_count: int,
    train_id_sha256: str,
    valid_id_sha256: str,
    best_iteration: int | None,
    training_seconds: float,
    model_size_mb: float,
) -> ModelMetrics:
    """计算统一评估指标。"""

    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError(f"{model_name} 预测概率存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{model_name} 预测概率超出 [0, 1] 范围。")

    clipped = clip_probabilities(probabilities)
    predicted_labels = (probabilities >= PREDICTION_THRESHOLD).astype(np.int8)

    valid_actual_ctr = float(y_true.mean())
    valid_mean_predicted_ctr = float(probabilities.mean())

    return ModelMetrics(
        model=model_name,
        data_scope="fixed_shared_sample",
        train_rows=train_rows,
        valid_rows=valid_rows,
        feature_count=feature_count,
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        log_loss=float(log_loss(y_true, clipped, labels=[0, 1])),
        accuracy=float(accuracy_score(y_true, predicted_labels)),
        precision=float(precision_score(y_true, predicted_labels, zero_division=0)),
        recall=float(recall_score(y_true, predicted_labels, zero_division=0)),
        f1=float(f1_score(y_true, predicted_labels, zero_division=0)),
        valid_actual_ctr=valid_actual_ctr,
        valid_mean_predicted_ctr=valid_mean_predicted_ctr,
        calibration_gap=abs(valid_mean_predicted_ctr - valid_actual_ctr),
        predicted_click_ratio_at_threshold=float(predicted_labels.mean()),
        threshold=PREDICTION_THRESHOLD,
        best_iteration=best_iteration,
        training_seconds=training_seconds,
        model_size_mb=model_size_mb,
        random_state=RANDOM_STATE,
        train_id_sha256=train_id_sha256,
        valid_id_sha256=valid_id_sha256,
        holdout_used=False,
    )


def get_file_size_mb(path: Path) -> float:
    """获取文件大小（MB）。"""

    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def metrics_to_row(metrics: ModelMetrics) -> dict:
    """ModelMetrics 转 CSV 行。"""

    return {
        "model": metrics.model,
        "data_scope": metrics.data_scope,
        "train_rows": metrics.train_rows,
        "valid_rows": metrics.valid_rows,
        "feature_count": metrics.feature_count,
        "roc_auc": metrics.roc_auc,
        "log_loss": metrics.log_loss,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "valid_actual_ctr": metrics.valid_actual_ctr,
        "valid_mean_predicted_ctr": metrics.valid_mean_predicted_ctr,
        "calibration_gap": metrics.calibration_gap,
        "predicted_click_ratio_at_threshold": metrics.predicted_click_ratio_at_threshold,
        "threshold": metrics.threshold,
        "best_iteration": metrics.best_iteration,
        "training_seconds": metrics.training_seconds,
        "model_size_mb": metrics.model_size_mb,
        "random_state": metrics.random_state,
        "train_id_sha256": metrics.train_id_sha256,
        "valid_id_sha256": metrics.valid_id_sha256,
        "holdout_used": metrics.holdout_used,
    }


def build_comparison_dataframe(metrics_list: list[ModelMetrics]) -> pd.DataFrame:
    """构建模型对比表。"""

    df = pd.DataFrame([metrics_to_row(item) for item in metrics_list])

    df["auc_rank"] = df["roc_auc"].rank(method="min", ascending=False).astype(int)
    df["logloss_rank"] = df["log_loss"].rank(method="min", ascending=True).astype(int)
    df["calibration_rank"] = df["calibration_gap"].rank(
        method="min",
        ascending=True,
    ).astype(int)

    logistic_row = df.loc[df["model"] == "Logistic Regression"].iloc[0]

    improvements: list[float] = []
    for _, row in df.iterrows():
        if row["model"] == "Logistic Regression":
            improvements.append(0.0)
        else:
            improvements.append(row["roc_auc"] - logistic_row["roc_auc"])
    df["auc_improvement_vs_logistic"] = improvements

    logloss_reductions: list[float] = []
    for _, row in df.iterrows():
        if row["model"] == "Logistic Regression":
            logloss_reductions.append(0.0)
        else:
            logloss_reductions.append(logistic_row["log_loss"] - row["log_loss"])
    df["logloss_reduction_vs_logistic"] = logloss_reductions

    best_row = df.sort_values(
        ["auc_rank", "logloss_rank", "calibration_rank"],
        ascending=[True, True, True],
    ).iloc[0]
    df["is_best_model"] = df["model"] == best_row["model"]

    return df


def determine_best_model(metrics_list: list[ModelMetrics]) -> str:
    """按 AUC → LogLoss → calibration_gap 确定最佳模型。"""

    return sorted(
        metrics_list,
        key=lambda item: (-item.roc_auc, item.log_loss, item.calibration_gap),
    )[0].model


def load_original_baseline_metrics() -> dict[str, dict]:
    """读取第 26—28 步原始基线指标。"""

    baselines: dict[str, dict] = {}
    for model_name, path in ORIGINAL_BASELINE_METRICS.items():
        if path.exists():
            baselines[model_name] = json.loads(path.read_text(encoding="utf-8"))
    return baselines


def write_text_report(
    report_path: Path,
    test_mode: bool,
    feature_columns: list[str],
    train_rows: int,
    valid_rows: int,
    train_sha256: str,
    valid_sha256: str,
    metrics_list: list[ModelMetrics],
    comparison_df: pd.DataFrame,
    best_model: str,
    original_baselines: dict[str, dict],
) -> None:
    """写入中文文本报告。"""

    mode_label = "TEST_MODE=True" if test_mode else "TEST_MODE=False"

    lines = [
        "百度 CTR 项目 — 固定共享样本三模型基线报告",
        "=" * 70,
        "",
        f"当前模式：{mode_label}",
        "",
        "【为什么重新训练】",
        "  第 26—28 步原始基线使用了不同训练口径（逻辑回归全量、树模型抽样）。",
        "  第 31 步在完全相同的固定样本上复跑三种模型，实现公平比较。",
        "",
        "【固定样本说明】",
        "  三种模型使用同一批固定 train / valid 样本、相同 click 标签、",
        "  相同 feature_columns 与相同 valid id 顺序。",
        f"  train 行数：{train_rows:,}",
        f"  valid 行数：{valid_rows:,}",
        f"  特征数量：{len(feature_columns)}",
        "",
        "【SHA256 指纹】",
        f"  train_id_sha256：{train_sha256}",
        f"  valid_id_sha256：{valid_sha256}",
        "",
        "【模型参数来源】",
        "  Logistic Regression：scripts/26_train_logistic_baseline.py",
        "  LightGBM：scripts/27_train_lightgbm_baseline.py",
        "  XGBoost：scripts/28_train_xgboost_baseline.py",
        "",
        "【本次固定样本指标】",
    ]

    for metrics in metrics_list:
        lines.extend(
            [
                f"  {metrics.model}:",
                f"    ROC-AUC：{metrics.roc_auc:.6f}",
                f"    LogLoss：{metrics.log_loss:.6f}",
                f"    calibration_gap：{metrics.calibration_gap:.6f}",
                f"    best_iteration：{metrics.best_iteration}",
                f"    training_seconds：{metrics.training_seconds:.2f}",
                "",
            ]
        )

    lines.extend(
        [
            f"【当前表现最好模型】{best_model}",
            "",
            "【与第 26—29 步原始基线区别】",
            "  原始逻辑回归使用全量 train（约 3237 万行）与全量 valid（约 383 万行）。",
            "  原始 LightGBM / XGBoost 使用 200 万 / 50 万随机抽样，且与本次固定 id 顺序不同。",
            "  因此原始基线结果不能与本次固定样本结果直接视为同一口径。",
            "",
            "【原始基线 valid 指标参考】",
        ]
    )

    for model_name, baseline in original_baselines.items():
        lines.append(
            f"  {model_name}: AUC={baseline.get('roc_auc', 'N/A')}, "
            f"LogLoss={baseline.get('log_loss', 'N/A')}"
        )

    lines.extend(
        [
            "",
            "【说明】",
            "  - 本次属于固定样本公平比较",
            "  - holdout 尚未使用",
            "  - 下一步建议使用固定样本 LightGBM 作为 Optuna 调优基线",
        ]
    )

    if test_mode:
        lines.append("  - 当前为测试模式，结果不能作为正式模型结论")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_validation(
    metrics_list: list[ModelMetrics],
    valid_meta: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_columns: list[str],
    expected_valid_rows: int,
    train_sha256: str,
    valid_sha256: str,
    formal_fingerprint_checked: bool,
) -> None:
    """最终验收。"""

    train_rows_set = {item.train_rows for item in metrics_list}
    valid_rows_set = {item.valid_rows for item in metrics_list}

    if len(train_rows_set) != 1:
        raise ValueError(f"三个模型 train 行数不一致：{train_rows_set}")

    if len(valid_rows_set) != 1:
        raise ValueError(f"三个模型 valid 行数不一致：{valid_rows_set}")

    if list(predictions_df.columns[:3]) != ["id", "click", "split_date"]:
        raise ValueError("预测文件元数据列顺序不正确。")

    prob_columns = [
        "logistic_probability",
        "lightgbm_probability",
        "xgboost_probability",
    ]
    for column in prob_columns:
        if column not in predictions_df.columns:
            raise ValueError(f"预测文件缺少列：{column}")

    if len(predictions_df) != expected_valid_rows:
        raise ValueError(
            f"预测行数 {len(predictions_df):,} 与期望 {expected_valid_rows:,} 不一致。"
        )

    if not predictions_df["id"].equals(valid_meta["id"].reset_index(drop=True)):
        raise ValueError("预测文件 id 顺序与 valid 元数据不一致。")

    for column in prob_columns:
        values = predictions_df[column].to_numpy(dtype=np.float64)
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError(f"{column} 存在 NaN 或 inf。")
        if (values < 0).any() or (values > 1).any():
            raise ValueError(f"{column} 超出 [0, 1] 范围。")

    if len({tuple(feature_columns)}) != 1:
        raise ValueError("特征列不一致。")

    for metrics in metrics_list:
        if metrics.train_id_sha256 != train_sha256:
            raise ValueError(f"{metrics.model} train_id_sha256 不一致。")
        if metrics.valid_id_sha256 != valid_sha256:
            raise ValueError(f"{metrics.model} valid_id_sha256 不一致。")
        if metrics.holdout_used is not False:
            raise ValueError(f"{metrics.model} holdout_used 必须为 false。")

    if formal_fingerprint_checked:
        if len(train_sha256) != 64 or len(valid_sha256) != 64:
            raise ValueError("SHA256 长度必须为 64。")


def main() -> None:
    """主流程：读取固定样本 → 训练三模型 → 评估与保存。"""

    paths = get_output_paths(TEST_MODE)
    train_limit, valid_limit = get_row_limits(TEST_MODE)

    print("=" * 70)
    print("固定共享样本三模型基线训练")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"train 行数上限：{train_limit:,}")
    print(f"valid 行数上限：{valid_limit:,}")

    metadata = load_fixed_sample_metadata(FIXED_SAMPLE_METADATA_PATH)
    feature_columns: list[str] = metadata["feature_columns"]
    feature_count = int(metadata["feature_count"])

    if len(feature_columns) != feature_count:
        raise ValueError(
            f"feature_columns 长度 {len(feature_columns)} 与 feature_count {feature_count} 不一致。"
        )

    forbidden = {"id", "click", "split_date"} & set(feature_columns)
    if forbidden:
        raise ValueError(f"feature_columns 包含禁止字段：{sorted(forbidden)}")

    train_files = get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)

    formal_fingerprint_checked = False
    if TEST_MODE:
        print("\n测试模式：跳过与完整正式指纹的一致性校验。")
        train_sha256 = metadata["train_id_sha256"]
        valid_sha256 = metadata["valid_id_sha256"]
    else:
        train_sha256, valid_sha256 = verify_formal_fingerprints(
            train_files,
            valid_files,
            metadata,
        )
        formal_fingerprint_checked = True
        print("\n正式模式：SHA256 指纹校验通过。")

    print("\n加载 valid 固定样本 ...")
    x_valid, y_valid, valid_meta = load_fixed_split(
        valid_files,
        feature_columns,
        max_rows=valid_limit,
    )

    print("\n加载 train 固定样本（供树模型使用）...")
    x_train, y_train, _ = load_fixed_split(
        train_files,
        feature_columns,
        max_rows=train_limit,
    )

    actual_train_rows = len(y_train)
    actual_valid_rows = len(y_valid)

    if not TEST_MODE:
        if actual_train_rows != FULL_TRAIN_ROWS:
            raise ValueError(
                f"正式模式 train 行数 {actual_train_rows:,} 不等于 {FULL_TRAIN_ROWS:,}"
            )
        if actual_valid_rows != FULL_VALID_ROWS:
            raise ValueError(
                f"正式模式 valid 行数 {actual_valid_rows:,} 不等于 {FULL_VALID_ROWS:,}"
            )

    x_train_df = pd.DataFrame(x_train, columns=feature_columns)
    x_valid_df = pd.DataFrame(x_valid, columns=feature_columns)

    metrics_list: list[ModelMetrics] = []
    logistic_probabilities: np.ndarray | None = None
    lightgbm_probabilities: np.ndarray | None = None
    xgboost_probabilities: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 逻辑回归
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("训练 Logistic Regression")
    print("=" * 70)

    lr_start = time.perf_counter()
    logistic_model, logistic_scaler, lr_train_rows, _ = train_logistic_model(
        train_files,
        feature_columns,
        max_train_rows=train_limit,
    )
    logistic_probabilities = predict_logistic_probabilities(
        x_valid,
        logistic_scaler,
        logistic_model,
    )
    lr_elapsed = time.perf_counter() - lr_start

    paths.logistic_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_model, paths.logistic_model)
    joblib.dump(logistic_scaler, paths.logistic_scaler)

    logistic_size = get_file_size_mb(paths.logistic_model) + get_file_size_mb(
        paths.logistic_scaler
    )

    metrics_list.append(
        compute_metrics(
            model_name="Logistic Regression",
            y_true=y_valid,
            probabilities=logistic_probabilities,
            train_rows=lr_train_rows,
            valid_rows=actual_valid_rows,
            feature_count=feature_count,
            train_id_sha256=train_sha256,
            valid_id_sha256=valid_sha256,
            best_iteration=None,
            training_seconds=lr_elapsed,
            model_size_mb=logistic_size,
        )
    )

    del logistic_model, logistic_scaler
    gc.collect()

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("训练 LightGBM")
    print("=" * 70)

    lightgbm_model, lgb_elapsed = train_lightgbm_model(
        x_train_df,
        y_train,
        x_valid_df,
        y_valid,
    )
    lightgbm_probabilities = lightgbm_model.predict_proba(
        x_valid_df,
        num_iteration=lightgbm_model.best_iteration_,
    )[:, 1].astype(np.float64)

    joblib.dump(lightgbm_model, paths.lightgbm_model)
    lgb_size = get_file_size_mb(paths.lightgbm_model)

    metrics_list.append(
        compute_metrics(
            model_name="LightGBM",
            y_true=y_valid,
            probabilities=lightgbm_probabilities,
            train_rows=actual_train_rows,
            valid_rows=actual_valid_rows,
            feature_count=feature_count,
            train_id_sha256=train_sha256,
            valid_id_sha256=valid_sha256,
            best_iteration=int(lightgbm_model.best_iteration_),
            training_seconds=lgb_elapsed,
            model_size_mb=lgb_size,
        )
    )

    del lightgbm_model
    gc.collect()

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("训练 XGBoost")
    print("=" * 70)

    xgboost_model, xgb_elapsed = train_xgboost_model(
        x_train_df,
        y_train,
        x_valid_df,
        y_valid,
    )
    xgboost_probabilities = xgboost_model.predict_proba(
        x_valid_df,
        iteration_range=(0, xgboost_model.best_iteration + 1),
    )[:, 1].astype(np.float64)

    xgboost_model.save_model(paths.xgboost_model)
    xgb_size = get_file_size_mb(paths.xgboost_model)

    metrics_list.append(
        compute_metrics(
            model_name="XGBoost",
            y_true=y_valid,
            probabilities=xgboost_probabilities,
            train_rows=actual_train_rows,
            valid_rows=actual_valid_rows,
            feature_count=feature_count,
            train_id_sha256=train_sha256,
            valid_id_sha256=valid_sha256,
            best_iteration=int(xgboost_model.best_iteration),
            training_seconds=xgb_elapsed,
            model_size_mb=xgb_size,
        )
    )

    del xgboost_model, x_train_df, x_valid_df, x_train, x_valid
    gc.collect()

    assert logistic_probabilities is not None
    assert lightgbm_probabilities is not None
    assert xgboost_probabilities is not None

    if not (
        len(logistic_probabilities)
        == len(lightgbm_probabilities)
        == len(xgboost_probabilities)
    ):
        raise ValueError("三个模型预测长度不一致。")

    predictions_df = valid_meta.copy()
    predictions_df["logistic_probability"] = logistic_probabilities
    predictions_df["lightgbm_probability"] = lightgbm_probabilities
    predictions_df["xgboost_probability"] = xgboost_probabilities

    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(paths.predictions, index=False)

    comparison_df = build_comparison_dataframe(metrics_list)
    best_model = determine_best_model(metrics_list)
    original_baselines = load_original_baseline_metrics()

    paths.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_to_row(item) for item in metrics_list]).to_csv(
        paths.metrics_csv,
        index=False,
        encoding="utf-8",
    )
    comparison_df.to_csv(paths.comparison_csv, index=False, encoding="utf-8")

    write_text_report(
        paths.report_txt,
        TEST_MODE,
        feature_columns,
        actual_train_rows,
        actual_valid_rows,
        train_sha256,
        valid_sha256,
        metrics_list,
        comparison_df,
        best_model,
        original_baselines,
    )

    model_parameters = {
        "Logistic Regression": {
            "class": "SGDClassifier",
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "learning_rate": "optimal",
            "random_state": RANDOM_STATE,
            "average": True,
            "scaler": "StandardScaler",
        },
        "LightGBM": {
            "objective": "binary",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 500,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": RANDOM_STATE,
            "early_stopping_rounds": 50,
        },
        "XGBoost": {
            "objective": "binary:logistic",
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 10,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "max_bin": 256,
            "random_state": RANDOM_STATE,
            "early_stopping_rounds": 50,
        },
    }

    output_metadata = {
        "script_name": "scripts/31_train_fixed_sample_baselines.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "train_rows": actual_train_rows,
        "valid_rows": actual_valid_rows,
        "feature_columns": feature_columns,
        "feature_count": feature_count,
        "train_id_sha256": train_sha256,
        "valid_id_sha256": valid_sha256,
        "source_fixed_sample_metadata": str(FIXED_SAMPLE_METADATA_PATH),
        "model_names": [item.model for item in metrics_list],
        "model_parameters": model_parameters,
        "model_output_paths": {
            "logistic_model": str(paths.logistic_model),
            "logistic_scaler": str(paths.logistic_scaler),
            "lightgbm_model": str(paths.lightgbm_model),
            "xgboost_model": str(paths.xgboost_model),
        },
        "prediction_output_path": str(paths.predictions),
        "best_model": best_model,
        "holdout_used": False,
        "validation_passed": True,
    }

    paths.metadata_json.write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_validation(
        metrics_list=metrics_list,
        valid_meta=valid_meta,
        predictions_df=predictions_df,
        feature_columns=feature_columns,
        expected_valid_rows=actual_valid_rows,
        train_sha256=train_sha256,
        valid_sha256=valid_sha256,
        formal_fingerprint_checked=formal_fingerprint_checked,
    )

    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"train 行数：{actual_train_rows:,}")
    print(f"valid 行数：{actual_valid_rows:,}")
    print(f"特征数量：{feature_count}")
    print(f"train_id_sha256：{train_sha256}")
    print(f"valid_id_sha256：{valid_sha256}")

    for metrics in metrics_list:
        print(
            f"{metrics.model} — AUC: {metrics.roc_auc:.6f}, "
            f"LogLoss: {metrics.log_loss:.6f}, "
            f"calibration_gap: {metrics.calibration_gap:.6f}"
        )

    print(f"当前最佳模型：{best_model}")
    print("输出路径：")
    print(f"  逻辑回归模型：   {paths.logistic_model}")
    print(f"  逻辑回归 scaler：{paths.logistic_scaler}")
    print(f"  LightGBM 模型：  {paths.lightgbm_model}")
    print(f"  XGBoost 模型：   {paths.xgboost_model}")
    print(f"  验证预测：       {paths.predictions}")
    print(f"  指标 CSV：       {paths.metrics_csv}")
    print(f"  对比 CSV：       {paths.comparison_csv}")
    print(f"  文本报告：       {paths.report_txt}")
    print(f"  元数据 JSON：    {paths.metadata_json}")
    print("validation_passed：True")
    print("三模型固定共享样本基线训练完成，holdout 尚未使用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
