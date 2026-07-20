"""
百度 CTR 项目 — 固定共享样本 LightGBM Optuna 超参数调优

功能：
    在第 30 步保存的固定 train / valid 样本上，使用 Optuna 对 LightGBM 进行
    超参数调优，并以第 31 步固定样本 LightGBM 作为调优前基线。禁止读取 holdout。

数据输入：
    data/tuning/lightgbm_train/*.parquet
    data/tuning/lightgbm_valid/*.parquet
    outputs/fixed_tuning_sample_metadata.json
    outputs/fixed_sample_baseline_metrics.csv

用法：
    python scripts/32_tune_lightgbm_optuna.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from optuna.samplers import TPESampler
from sklearn.metrics import (
    accuracy_score,
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

RANDOM_STATE = 42
THRESHOLD = 0.5

TEST_TRAIN_ROWS = 100_000
TEST_VALID_ROWS = 50_000
TEST_N_TRIALS = 3

FORMAL_TRAIN_ROWS = 2_000_000
FORMAL_VALID_ROWS = 500_000
FORMAL_N_TRIALS = 20

EARLY_STOPPING_ROUNDS = 100
MAX_ESTIMATORS = 2000

BATCH_SIZE = 200_000
PROB_CLIP_EPS = 1e-7
EXPECTED_FEATURE_COUNT = 33

TRAIN_INPUT_DIR = Path("data/tuning/lightgbm_train")
VALID_INPUT_DIR = Path("data/tuning/lightgbm_valid")
FIXED_SAMPLE_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")
BASELINE_METRICS_PATH = Path("outputs/fixed_sample_baseline_metrics.csv")

STUDY_NAME_FORMAL = "baidu_ctr_lightgbm_fixed_sample_v1"
STUDY_NAME_TEST = "baidu_ctr_lightgbm_fixed_sample_test_v1"
STORAGE_FORMAL = "sqlite:///outputs/optuna/lightgbm_fixed_sample_study.db"
STORAGE_TEST = "sqlite:///outputs/optuna/lightgbm_fixed_sample_test_study.db"

BASELINE_MODEL_KEY = "fixed_sample_lightgbm_baseline"
TUNED_MODEL_KEY = "optuna_tuned_lightgbm"
BASELINE_MODEL_NAME = "LightGBM"


@dataclass
class OutputPaths:
    """第 32 步输出路径。"""

    test_mode: bool
    optuna_dir: Path
    trials_csv: Path
    best_params_json: Path
    metrics_csv: Path
    comparison_csv: Path
    report_txt: Path
    metadata_json: Path
    tuned_model: Path
    predictions: Path


@dataclass
class BaselineMetrics:
    """第 31 步固定样本 LightGBM 基线指标。"""

    model: str
    data_scope: str
    train_rows: int
    valid_rows: int
    feature_count: int
    roc_auc: float
    log_loss: float
    calibration_gap: float
    valid_actual_ctr: float
    valid_mean_predicted_ctr: float
    best_iteration: float | None
    training_seconds: float
    train_id_sha256: str
    valid_id_sha256: str
    holdout_used: bool


@dataclass
class TunedMetrics:
    """调优后最佳模型指标。"""

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
    best_iteration: int
    training_seconds: float


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    optuna_dir = Path("outputs/optuna")
    suffix = "_test" if test_mode else ""

    return OutputPaths(
        test_mode=test_mode,
        optuna_dir=optuna_dir,
        trials_csv=optuna_dir / f"lightgbm_optuna_trials{suffix}.csv",
        best_params_json=Path(f"outputs/lightgbm_optuna_best_params{suffix}.json"),
        metrics_csv=Path(f"outputs/lightgbm_optuna_metrics{suffix}.csv"),
        comparison_csv=Path(f"outputs/lightgbm_optuna_comparison{suffix}.csv"),
        report_txt=Path(f"outputs/lightgbm_optuna_report{suffix}.txt"),
        metadata_json=Path(f"outputs/lightgbm_optuna_metadata{suffix}.json"),
        tuned_model=Path(
            f"models/test/tuned_lightgbm_optuna_model_test.joblib"
            if test_mode
            else "models/tuned_lightgbm_optuna_model.joblib"
        ),
        predictions=Path(
            f"outputs/predictions/tuned_lightgbm_valid_predictions_test.parquet"
            if test_mode
            else "outputs/predictions/tuned_lightgbm_valid_predictions.parquet"
        ),
    )


def get_row_limits(test_mode: bool) -> tuple[int, int]:
    """返回 train / valid 行数上限。"""

    if test_mode:
        return TEST_TRAIN_ROWS, TEST_VALID_ROWS
    return FORMAL_TRAIN_ROWS, FORMAL_VALID_ROWS


def get_requested_trials(test_mode: bool) -> int:
    """返回本次计划新增 trial 数。"""

    return TEST_N_TRIALS if test_mode else FORMAL_N_TRIALS


def get_study_config(test_mode: bool) -> tuple[str, str]:
    """返回 study 名称与 storage。"""

    if test_mode:
        return STUDY_NAME_TEST, STORAGE_TEST
    return STUDY_NAME_FORMAL, STORAGE_FORMAL


def load_fixed_sample_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取并校验固定样本元数据。"""

    if not metadata_path.exists():
        raise FileNotFoundError(f"未找到固定样本元数据：{metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if metadata.get("validation_passed") is not True:
        raise ValueError(
            f"validation_passed={metadata.get('validation_passed')!r}，禁止调优。"
        )

    if metadata.get("holdout_used") is not False:
        raise ValueError(
            f"holdout_used={metadata.get('holdout_used')!r}，禁止调优。"
        )

    if int(metadata.get("feature_count", -1)) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature_count={metadata.get('feature_count')!r}，"
            f"必须为 {EXPECTED_FEATURE_COUNT}。"
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

    feature_columns = metadata["feature_columns"]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature_columns 长度 {len(feature_columns)} 不等于 {EXPECTED_FEATURE_COUNT}。"
        )

    forbidden = {"id", "click", "split_date"} & set(feature_columns)
    if forbidden:
        raise ValueError(f"feature_columns 包含禁止字段：{sorted(forbidden)}")

    return metadata


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    """按 part 文件名稳定排序 Parquet 文件。"""

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
    metadata: dict[str, Any],
) -> tuple[str, str]:
    """正式模式重新验证 SHA256 指纹。"""

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


def validate_loaded_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_meta: pd.DataFrame,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_meta: pd.DataFrame,
    feature_columns: list[str],
    expected_feature_count: int,
) -> None:
    """校验加载后的固定样本数据。"""

    if x_train.shape[1] != expected_feature_count:
        raise ValueError(
            f"train 特征数 {x_train.shape[1]} 不等于 {expected_feature_count}。"
        )

    if x_valid.shape[1] != expected_feature_count:
        raise ValueError(
            f"valid 特征数 {x_valid.shape[1]} 不等于 {expected_feature_count}。"
        )

    if x_train.dtype != np.float32 or x_valid.dtype != np.float32:
        raise ValueError("X_train / X_valid 必须为 np.float32。")

    for split_name, x_matrix in ("train", x_train), ("valid", x_valid):
        if np.isnan(x_matrix).any():
            raise ValueError(f"{split_name} 特征存在 NaN。")
        if np.isinf(x_matrix).any():
            raise ValueError(f"{split_name} 特征存在 inf。")

    for split_name, y_vector in ("train", y_train), ("valid", y_valid):
        if not np.isin(y_vector, [0, 1]).all():
            raise ValueError(f"{split_name} click 不是仅包含 0 和 1。")

    for split_name, meta_df in ("train", train_meta), ("valid", valid_meta):
        if meta_df["id"].isna().any():
            raise ValueError(f"{split_name} id 存在缺失。")
        if meta_df["split_date"].isna().any():
            raise ValueError(f"{split_name} split_date 存在缺失。")

    train_dates = pd.to_datetime(train_meta["split_date"], errors="coerce")
    valid_dates = pd.to_datetime(valid_meta["split_date"], errors="coerce")

    if train_dates.isna().any() or valid_dates.isna().any():
        raise ValueError("split_date 无法解析为日期。")

    if train_dates.max() >= valid_dates.min():
        raise ValueError(
            f"train 日期必须早于 valid：train_max={train_dates.max()}, "
            f"valid_min={valid_dates.min()}"
        )

    if len(feature_columns) != expected_feature_count:
        raise ValueError("feature_columns 长度与期望特征数不一致。")


def load_baseline_metrics(
    baseline_path: Path,
    metadata: dict[str, Any],
) -> BaselineMetrics:
    """读取并校验第 31 步 LightGBM 基线指标。"""

    if not baseline_path.exists():
        raise FileNotFoundError(f"未找到基线指标文件：{baseline_path}")

    baseline_df = pd.read_csv(baseline_path)
    lightgbm_rows = baseline_df.loc[baseline_df["model"] == BASELINE_MODEL_NAME]

    if lightgbm_rows.empty:
        raise ValueError(f"基线文件中未找到 model={BASELINE_MODEL_NAME!r} 的行。")

    if len(lightgbm_rows) > 1:
        raise ValueError("基线文件中 LightGBM 行数超过 1。")

    row = lightgbm_rows.iloc[0]

    data_scope = str(row["data_scope"])
    if data_scope != "fixed_shared_sample":
        raise ValueError(f"基线 data_scope={data_scope!r}，必须为 fixed_shared_sample。")

    holdout_used = row["holdout_used"]
    if isinstance(holdout_used, str):
        holdout_used = holdout_used.lower() in {"true", "1", "yes"}
    if bool(holdout_used) is not False:
        raise ValueError(f"基线 holdout_used={row['holdout_used']!r}，必须为 false。")

    train_sha256 = str(row["train_id_sha256"])
    valid_sha256 = str(row["valid_id_sha256"])

    if train_sha256 != metadata["train_id_sha256"]:
        raise ValueError("基线 train_id_sha256 与第 30 步元数据不一致。")

    if valid_sha256 != metadata["valid_id_sha256"]:
        raise ValueError("基线 valid_id_sha256 与第 30 步元数据不一致。")

    best_iteration_value = row["best_iteration"]
    best_iteration: float | None
    if pd.isna(best_iteration_value):
        best_iteration = None
    else:
        best_iteration = float(best_iteration_value)

    return BaselineMetrics(
        model=BASELINE_MODEL_NAME,
        data_scope=data_scope,
        train_rows=int(row["train_rows"]),
        valid_rows=int(row["valid_rows"]),
        feature_count=int(row["feature_count"]),
        roc_auc=float(row["roc_auc"]),
        log_loss=float(row["log_loss"]),
        calibration_gap=float(row["calibration_gap"]),
        valid_actual_ctr=float(row["valid_actual_ctr"]),
        valid_mean_predicted_ctr=float(row["valid_mean_predicted_ctr"]),
        best_iteration=best_iteration,
        training_seconds=float(row["training_seconds"]),
        train_id_sha256=train_sha256,
        valid_id_sha256=valid_sha256,
        holdout_used=False,
    )


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """裁剪概率用于 LogLoss 计算。"""

    return np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def suggest_lightgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """定义 Optuna 参数搜索空间。"""

    max_depth = trial.suggest_int("max_depth", 5, 12)
    num_leaves_upper = min(128, 2**max_depth)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 20, num_leaves_upper),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            20,
            300,
            log=True,
        ),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
    }

    return params


def build_lightgbm_classifier(params: dict[str, Any]) -> lgb.LGBMClassifier:
    """根据 trial 参数构建 LightGBM 分类器。"""

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=MAX_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        subsample_freq=1,
        **params,
    )


def create_objective(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> Any:
    """创建 Optuna 目标函数。"""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lightgbm_params(trial)
        model: lgb.LGBMClassifier | None = None

        try:
            model = build_lightgbm_classifier(params)

            print(
                f"\n[Trial {trial.number}] 开始训练，参数："
                f"learning_rate={params['learning_rate']:.6f}, "
                f"max_depth={params['max_depth']}, "
                f"num_leaves={params['num_leaves']}"
            )

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
                    lgb.log_evaluation(period=0),
                ],
            )
            training_seconds = time.perf_counter() - start_time

            if not hasattr(model, "best_iteration_") or model.best_iteration_ is None:
                raise ValueError("LightGBM 训练完成后 best_iteration_ 不存在。")

            probabilities = model.predict_proba(
                x_valid,
                num_iteration=model.best_iteration_,
            )[:, 1].astype(np.float64)

            if np.isnan(probabilities).any() or np.isinf(probabilities).any():
                raise ValueError("预测概率存在 NaN 或 inf。")

            clipped = clip_probabilities(probabilities)
            valid_log_loss = float(log_loss(y_valid, clipped, labels=[0, 1]))
            valid_auc = float(roc_auc_score(y_valid, probabilities))
            valid_actual_ctr = float(y_valid.mean())
            valid_mean_predicted_ctr = float(probabilities.mean())
            calibration_gap = abs(valid_mean_predicted_ctr - valid_actual_ctr)
            best_iteration = int(model.best_iteration_)

            trial.set_user_attr("roc_auc", valid_auc)
            trial.set_user_attr("calibration_gap", calibration_gap)
            trial.set_user_attr("valid_mean_predicted_ctr", valid_mean_predicted_ctr)
            trial.set_user_attr("valid_actual_ctr", valid_actual_ctr)
            trial.set_user_attr("best_iteration", best_iteration)
            trial.set_user_attr("training_seconds", training_seconds)

            print(
                f"[Trial {trial.number}] 完成：LogLoss={valid_log_loss:.6f}, "
                f"AUC={valid_auc:.6f}, best_iteration={best_iteration}, "
                f"耗时={training_seconds:.2f}s"
            )

            return valid_log_loss

        except Exception as exc:
            print(f"[Trial {trial.number}] 失败：{exc}")
            print(f"[Trial {trial.number}] 参数：{params}")
            raise

        finally:
            del model
            gc.collect()

    return objective


def count_trials_by_state(study: optuna.Study) -> dict[str, int]:
    """统计各状态 trial 数量。"""

    counts = {"COMPLETE": 0, "FAIL": 0, "PRUNED": 0, "RUNNING": 0, "WAITING": 0}
    for trial in study.trials:
        state_name = trial.state.name
        if state_name in counts:
            counts[state_name] += 1
        else:
            counts[state_name] = counts.get(state_name, 0) + 1
    return counts


def save_trials_dataframe(study: optuna.Study, output_path: Path) -> pd.DataFrame:
    """保存 study.trials_dataframe() 结果。"""

    trials_df = study.trials_dataframe()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(output_path, index=False, encoding="utf-8")
    return trials_df


def train_best_model(
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[lgb.LGBMClassifier, np.ndarray, float]:
    """使用最佳参数重新训练并返回模型与 valid 概率。"""

    model = build_lightgbm_classifier(best_params)

    print("\n使用最佳参数重新训练 LightGBM ...")
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
    )
    training_seconds = time.perf_counter() - start_time

    if not hasattr(model, "best_iteration_") or model.best_iteration_ is None:
        raise ValueError("最佳模型训练完成后 best_iteration_ 不存在。")

    probabilities = model.predict_proba(
        x_valid,
        num_iteration=model.best_iteration_,
    )[:, 1].astype(np.float64)

    return model, probabilities, training_seconds


def compute_tuned_metrics(
    y_valid: np.ndarray,
    probabilities: np.ndarray,
    training_seconds: float,
    best_iteration: int,
) -> TunedMetrics:
    """计算调优后最佳模型指标。"""

    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("调优模型预测概率存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("调优模型预测概率超出 [0, 1] 范围。")

    clipped = clip_probabilities(probabilities)
    predicted_labels = (probabilities >= THRESHOLD).astype(np.int8)
    valid_actual_ctr = float(y_valid.mean())
    valid_mean_predicted_ctr = float(probabilities.mean())

    return TunedMetrics(
        roc_auc=float(roc_auc_score(y_valid, probabilities)),
        log_loss=float(log_loss(y_valid, clipped, labels=[0, 1])),
        accuracy=float(accuracy_score(y_valid, predicted_labels)),
        precision=float(precision_score(y_valid, predicted_labels, zero_division=0)),
        recall=float(recall_score(y_valid, predicted_labels, zero_division=0)),
        f1=float(f1_score(y_valid, predicted_labels, zero_division=0)),
        valid_actual_ctr=valid_actual_ctr,
        valid_mean_predicted_ctr=valid_mean_predicted_ctr,
        calibration_gap=abs(valid_mean_predicted_ctr - valid_actual_ctr),
        predicted_click_ratio_at_threshold=float(predicted_labels.mean()),
        threshold=THRESHOLD,
        best_iteration=best_iteration,
        training_seconds=training_seconds,
    )


def build_comparison_dataframe(
    baseline: BaselineMetrics,
    tuned: TunedMetrics,
    train_rows: int,
    valid_rows: int,
    feature_count: int,
    train_id_sha256: str,
    valid_id_sha256: str,
) -> pd.DataFrame:
    """构建基线与调优模型对比表。"""

    rows = [
        {
            "model": BASELINE_MODEL_KEY,
            "roc_auc": baseline.roc_auc,
            "log_loss": baseline.log_loss,
            "calibration_gap": baseline.calibration_gap,
            "valid_actual_ctr": baseline.valid_actual_ctr,
            "valid_mean_predicted_ctr": baseline.valid_mean_predicted_ctr,
            "best_iteration": baseline.best_iteration,
            "training_seconds": baseline.training_seconds,
            "train_rows": baseline.train_rows,
            "valid_rows": baseline.valid_rows,
            "feature_count": baseline.feature_count,
            "train_id_sha256": baseline.train_id_sha256,
            "valid_id_sha256": baseline.valid_id_sha256,
            "holdout_used": False,
        },
        {
            "model": TUNED_MODEL_KEY,
            "roc_auc": tuned.roc_auc,
            "log_loss": tuned.log_loss,
            "calibration_gap": tuned.calibration_gap,
            "valid_actual_ctr": tuned.valid_actual_ctr,
            "valid_mean_predicted_ctr": tuned.valid_mean_predicted_ctr,
            "best_iteration": tuned.best_iteration,
            "training_seconds": tuned.training_seconds,
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "feature_count": feature_count,
            "train_id_sha256": train_id_sha256,
            "valid_id_sha256": valid_id_sha256,
            "holdout_used": False,
        },
    ]

    comparison_df = pd.DataFrame(rows)

    baseline_auc = baseline.roc_auc
    tuned_auc = tuned.roc_auc
    baseline_logloss = baseline.log_loss
    tuned_logloss = tuned.log_loss
    baseline_calibration = baseline.calibration_gap
    tuned_calibration = tuned.calibration_gap

    comparison_df["auc_absolute_improvement"] = [
        0.0,
        tuned_auc - baseline_auc,
    ]
    comparison_df["auc_relative_improvement_percent"] = [
        0.0,
        ((tuned_auc - baseline_auc) / baseline_auc * 100.0) if baseline_auc != 0 else 0.0,
    ]
    comparison_df["logloss_absolute_reduction"] = [
        0.0,
        baseline_logloss - tuned_logloss,
    ]
    comparison_df["logloss_relative_reduction_percent"] = [
        0.0,
        ((baseline_logloss - tuned_logloss) / baseline_logloss * 100.0)
        if baseline_logloss != 0
        else 0.0,
    ]
    comparison_df["calibration_gap_change"] = [
        0.0,
        tuned_calibration - baseline_calibration,
    ]

    return comparison_df


def write_text_report(
    report_path: Path,
    test_mode: bool,
    train_rows: int,
    valid_rows: int,
    feature_count: int,
    feature_columns: list[str],
    train_sha256: str,
    valid_sha256: str,
    study_name: str,
    storage: str,
    requested_new_trials: int,
    total_trials: int,
    complete_trials: int,
    failed_trials: int,
    pruned_trials: int,
    best_trial_number: int,
    best_params: dict[str, Any],
    best_trial_user_attrs: dict[str, Any],
    baseline: BaselineMetrics,
    tuned: TunedMetrics,
    comparison_df: pd.DataFrame,
) -> None:
    """写入中文调优报告。"""

    tuned_row = comparison_df.loc[comparison_df["model"] == TUNED_MODEL_KEY].iloc[0]

    logloss_improved = tuned.log_loss < baseline.log_loss
    auc_improved = tuned.roc_auc > baseline.roc_auc
    calibration_improved = tuned.calibration_gap < baseline.calibration_gap

    mode_label = "TEST_MODE=True（测试结果不能作为正式结论）" if test_mode else "TEST_MODE=False"

    lines = [
        "百度 CTR 项目 — 第 32 步 LightGBM Optuna 调优报告",
        "=" * 72,
        "",
        "【1. 第 32 步目的】",
        "  在固定共享 train / valid 样本上使用 Optuna 对 LightGBM 超参数进行调优，",
        "  并以第 31 步固定样本 LightGBM 作为调优前基线。",
        "",
        "【2. 为什么使用固定共享样本】",
        "  保证每次 trial 与基线复训使用完全相同的样本、标签、特征列与 valid id 顺序，",
        "  使调优比较公平且可复现。",
        "",
        "【3. 为什么主要优化 LogLoss】",
        "  CTR 预测更关注概率校准与排序稳定性，LogLoss 直接衡量概率质量；",
        "  Accuracy 受阈值影响大，不适合作为本次调优主目标。",
        "",
        f"【4. train / valid 行数】train={train_rows:,}，valid={valid_rows:,}",
        f"【5. 特征数量】{feature_count}",
        "",
        "【6. SHA256 指纹】",
        f"  train_id_sha256：{train_sha256}",
        f"  valid_id_sha256：{valid_sha256}",
        "",
        "【7. Optuna 采样器与随机种子】",
        f"  sampler=TPESampler，seed={RANDOM_STATE}",
        "",
        "【8. Study 名称与数据库位置】",
        f"  study_name={study_name}",
        f"  storage={storage}",
        "",
        f"【9. 本次新增 trial 数】{requested_new_trials}",
        f"【10. 累计 trial 数】{total_trials}",
        "",
        "【11. 参数搜索空间】",
        "  learning_rate: [0.02, 0.15], log=True",
        "  max_depth: [5, 12]",
        "  num_leaves: [20, min(128, 2**max_depth)]",
        "  min_child_samples: [20, 300], log=True",
        "  subsample: [0.70, 1.00]",
        "  colsample_bytree: [0.70, 1.00]",
        "  reg_alpha: [1e-8, 10.0], log=True",
        "  reg_lambda: [1e-8, 10.0], log=True",
        "  min_split_gain: [0.0, 0.2]",
        "  固定参数：objective=binary, n_estimators=2000, subsample_freq=1, random_state=42",
        "",
        f"【12. 最佳 trial 编号】{best_trial_number}",
        "【13. 最佳参数】",
    ]

    for key, value in best_params.items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "【14. 最佳模型指标】",
            f"  ROC-AUC：{tuned.roc_auc:.6f}",
            f"  LogLoss：{tuned.log_loss:.6f}",
            f"  calibration_gap：{tuned.calibration_gap:.6f}",
            f"  best_iteration：{tuned.best_iteration}",
            f"  training_seconds：{tuned.training_seconds:.2f}",
            "",
            "【15. 固定样本基线指标（第 31 步 LightGBM）】",
            f"  ROC-AUC：{baseline.roc_auc:.6f}",
            f"  LogLoss：{baseline.log_loss:.6f}",
            f"  calibration_gap：{baseline.calibration_gap:.6f}",
            f"  best_iteration：{baseline.best_iteration}",
            f"  training_seconds：{baseline.training_seconds:.2f}",
            "",
            "【16. 调优前后差异】",
            f"  AUC 绝对提升：{tuned_row['auc_absolute_improvement']:.6f}",
            f"  AUC 相对提升(%)：{tuned_row['auc_relative_improvement_percent']:.4f}",
            f"  LogLoss 绝对下降：{tuned_row['logloss_absolute_reduction']:.6f}",
            f"  LogLoss 相对下降(%)：{tuned_row['logloss_relative_reduction_percent']:.4f}",
            f"  calibration_gap 变化：{tuned_row['calibration_gap_change']:.6f}",
            "",
            f"【17. 是否真正改善 LogLoss】{'是' if logloss_improved else '否'}",
            f"【18. AUC 是否提升】{'是' if auc_improved else '否'}",
            f"【19. calibration_gap 是否改善】{'是' if calibration_improved else '否'}",
            "",
            f"【20. 失败 trial 数量】{failed_trials}",
            "【21. holdout 尚未使用】是",
            "【22. 当前结论只基于 valid，不是最终 holdout 结论】是",
            "",
            f"当前模式：{mode_label}",
            f"特征列数量：{len(feature_columns)}",
            f"最佳 trial user_attrs：{json.dumps(best_trial_user_attrs, ensure_ascii=False)}",
            f"COMPLETE trial 数：{complete_trials}",
            f"PRUNED trial 数：{pruned_trials}",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_validation(
    test_mode: bool,
    study: optuna.Study,
    tuned_metrics: TunedMetrics,
    probabilities: np.ndarray,
    predictions_df: pd.DataFrame,
    valid_meta: pd.DataFrame,
    feature_columns: list[str],
    expected_valid_rows: int,
    train_rows: int,
    valid_rows: int,
    train_sha256: str,
    valid_sha256: str,
    formal_fingerprint_checked: bool,
    paths: OutputPaths,
) -> bool:
    """最终验收检查。"""

    state_counts = count_trials_by_state(study)

    if state_counts["COMPLETE"] < 1:
        raise ValueError("至少需要一个 COMPLETE trial。")

    if study.best_trial.state != optuna.trial.TrialState.COMPLETE:
        raise ValueError("best trial 必须为 COMPLETE 状态。")

    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("最佳模型概率存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("最佳模型概率超出 [0, 1] 范围。")

    if len(predictions_df) != expected_valid_rows:
        raise ValueError(
            f"预测行数 {len(predictions_df):,} 与期望 {expected_valid_rows:,} 不一致。"
        )

    if not predictions_df["id"].equals(valid_meta["id"].reset_index(drop=True)):
        raise ValueError("预测文件 id 顺序与 valid 元数据不一致。")

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"特征数量必须为 {EXPECTED_FEATURE_COUNT}。")

    if not test_mode:
        if train_rows != FORMAL_TRAIN_ROWS:
            raise ValueError(
                f"正式模式 train 行数 {train_rows:,} 不等于 {FORMAL_TRAIN_ROWS:,}。"
            )
        if valid_rows != FORMAL_VALID_ROWS:
            raise ValueError(
                f"正式模式 valid 行数 {valid_rows:,} 不等于 {FORMAL_VALID_ROWS:,}。"
            )
        if len(predictions_df) != FORMAL_VALID_ROWS:
            raise ValueError(
                f"正式模式验证预测必须为 {FORMAL_VALID_ROWS:,} 行。"
            )

    if formal_fingerprint_checked:
        if len(train_sha256) != 64 or len(valid_sha256) != 64:
            raise ValueError("SHA256 长度必须为 64。")

    required_outputs = [
        paths.trials_csv,
        paths.best_params_json,
        paths.metrics_csv,
        paths.comparison_csv,
        paths.report_txt,
        paths.metadata_json,
        paths.tuned_model,
        paths.predictions,
    ]
    for output_path in required_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"缺少输出文件：{output_path}")

    return True


def main() -> None:
    """主流程：读取固定样本 → Optuna 调优 → 复训最佳模型 → 保存结果。"""

    paths = get_output_paths(TEST_MODE)
    train_limit, valid_limit = get_row_limits(TEST_MODE)
    requested_new_trials = get_requested_trials(TEST_MODE)
    study_name, storage = get_study_config(TEST_MODE)

    print("=" * 72)
    print("第 32 步：固定共享样本 LightGBM Optuna 调优")
    print("=" * 72)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"Optuna 版本：{optuna.__version__}")
    print(f"LightGBM 版本：{lgb.__version__}")
    print(f"train 行数上限：{train_limit:,}")
    print(f"valid 行数上限：{valid_limit:,}")
    print(f"study_name：{study_name}")
    print(f"storage：{storage}")

    metadata = load_fixed_sample_metadata(FIXED_SAMPLE_METADATA_PATH)
    feature_columns: list[str] = metadata["feature_columns"]
    feature_count = int(metadata["feature_count"])

    baseline = load_baseline_metrics(BASELINE_METRICS_PATH, metadata)

    train_files = get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)

    formal_fingerprint_checked = False
    if TEST_MODE:
        print("\n测试模式：跳过与完整正式指纹的一致性校验。")
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

    print("加载 train 固定样本 ...")
    x_train, y_train, train_meta = load_fixed_split(
        train_files,
        feature_columns,
        max_rows=train_limit,
    )

    actual_train_rows = len(y_train)
    actual_valid_rows = len(y_valid)

    validate_loaded_data(
        x_train,
        y_train,
        train_meta,
        x_valid,
        y_valid,
        valid_meta,
        feature_columns,
        feature_count,
    )

    if TEST_MODE:
        # 测试模式：按第 30 步规则记录实际加载子集的 id 指纹
        train_hasher = hashlib.sha256()
        for id_value in train_meta["id"]:
            train_hasher.update((str(id_value) + "\n").encode("utf-8"))
        train_sha256 = train_hasher.hexdigest()

        valid_hasher = hashlib.sha256()
        for id_value in valid_meta["id"]:
            valid_hasher.update((str(id_value) + "\n").encode("utf-8"))
        valid_sha256 = valid_hasher.hexdigest()

        print("\n测试模式：已记录子集 SHA256 指纹。")
        print(f"  train 子集 SHA256：{train_sha256}")
        print(f"  valid 子集 SHA256：{valid_sha256}")
    else:
        if actual_train_rows != FORMAL_TRAIN_ROWS:
            raise ValueError(
                f"正式模式 train 行数 {actual_train_rows:,} 不等于 {FORMAL_TRAIN_ROWS:,}。"
            )
        if actual_valid_rows != FORMAL_VALID_ROWS:
            raise ValueError(
                f"正式模式 valid 行数 {actual_valid_rows:,} 不等于 {FORMAL_VALID_ROWS:,}。"
            )

    x_train_df = pd.DataFrame(x_train, columns=feature_columns)
    x_valid_df = pd.DataFrame(x_valid, columns=feature_columns)

    paths.optuna_dir.mkdir(parents=True, exist_ok=True)

    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    print(f"已有 trial 数量：{existing_trials}")
    print(f"本次计划新增 trial 数量：{requested_new_trials}")

    objective = create_objective(x_train_df, y_train, x_valid_df, y_valid)
    study.optimize(
        objective,
        n_trials=requested_new_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    total_trials = len(study.trials)
    state_counts = count_trials_by_state(study)
    complete_trials = state_counts["COMPLETE"]
    failed_trials = state_counts["FAIL"]
    pruned_trials = state_counts["PRUNED"]

    if complete_trials < 1:
        raise RuntimeError("没有任何 COMPLETE trial，无法选择最佳参数。")

    if study.best_trial.state != optuna.trial.TrialState.COMPLETE:
        raise RuntimeError("best trial 不是 COMPLETE 状态。")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_trial_user_attrs = dict(best_trial.user_attrs)

    save_trials_dataframe(study, paths.trials_csv)

    print("\n" + "=" * 72)
    print("Optuna 调优完成，开始复训最佳模型")
    print("=" * 72)
    print(f"最佳 trial 编号：{best_trial.number}")
    print(f"最佳 validation LogLoss：{best_trial.value:.6f}")
    print(f"最佳参数：{best_params}")

    best_model, tuned_probabilities, retrain_seconds = train_best_model(
        best_params,
        x_train_df,
        y_train,
        x_valid_df,
        y_valid,
    )

    tuned_metrics = compute_tuned_metrics(
        y_valid,
        tuned_probabilities,
        retrain_seconds,
        int(best_model.best_iteration_),
    )

    comparison_df = build_comparison_dataframe(
        baseline=baseline,
        tuned=tuned_metrics,
        train_rows=actual_train_rows,
        valid_rows=actual_valid_rows,
        feature_count=feature_count,
        train_id_sha256=train_sha256,
        valid_id_sha256=valid_sha256,
    )

    predictions_df = valid_meta.copy()
    predictions_df["tuned_lightgbm_probability"] = tuned_probabilities

    paths.tuned_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, paths.tuned_model)

    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(paths.predictions, index=False)

    tuned_metrics_row = {
        "model": TUNED_MODEL_KEY,
        "roc_auc": tuned_metrics.roc_auc,
        "log_loss": tuned_metrics.log_loss,
        "accuracy": tuned_metrics.accuracy,
        "precision": tuned_metrics.precision,
        "recall": tuned_metrics.recall,
        "f1": tuned_metrics.f1,
        "valid_actual_ctr": tuned_metrics.valid_actual_ctr,
        "valid_mean_predicted_ctr": tuned_metrics.valid_mean_predicted_ctr,
        "calibration_gap": tuned_metrics.calibration_gap,
        "predicted_click_ratio_at_threshold": tuned_metrics.predicted_click_ratio_at_threshold,
        "threshold": tuned_metrics.threshold,
        "best_iteration": tuned_metrics.best_iteration,
        "training_seconds": tuned_metrics.training_seconds,
        "train_rows": actual_train_rows,
        "valid_rows": actual_valid_rows,
        "feature_count": feature_count,
        "train_id_sha256": train_sha256,
        "valid_id_sha256": valid_sha256,
        "holdout_used": False,
    }

    paths.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([tuned_metrics_row]).to_csv(
        paths.metrics_csv,
        index=False,
        encoding="utf-8",
    )
    comparison_df.to_csv(paths.comparison_csv, index=False, encoding="utf-8")

    best_params_payload = {
        "study_name": study_name,
        "best_trial_number": best_trial.number,
        "objective_name": "validation_log_loss",
        "best_validation_log_loss": best_trial.value,
        "best_params": best_params,
        "best_trial_user_attributes": best_trial_user_attrs,
        "total_trials": total_trials,
        "complete_trials": complete_trials,
        "failed_trials": failed_trials,
        "pruned_trials": pruned_trials,
        "train_id_sha256": train_sha256,
        "valid_id_sha256": valid_sha256,
        "holdout_used": False,
    }
    paths.best_params_json.write_text(
        json.dumps(best_params_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_text_report(
        paths.report_txt,
        TEST_MODE,
        actual_train_rows,
        actual_valid_rows,
        feature_count,
        feature_columns,
        train_sha256,
        valid_sha256,
        study_name,
        storage,
        requested_new_trials,
        total_trials,
        complete_trials,
        failed_trials,
        pruned_trials,
        best_trial.number,
        best_params,
        best_trial_user_attrs,
        baseline,
        tuned_metrics,
        comparison_df,
    )

    metadata_payload = {
        "script_name": "scripts/32_tune_lightgbm_optuna.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "optuna_version": optuna.__version__,
        "lightgbm_version": lgb.__version__,
        "study_name": study_name,
        "storage": storage,
        "sampler": "TPESampler",
        "sampler_seed": RANDOM_STATE,
        "direction": "minimize",
        "objective_metric": "validation_log_loss",
        "requested_new_trials": requested_new_trials,
        "total_trials": total_trials,
        "complete_trials": complete_trials,
        "failed_trials": failed_trials,
        "pruned_trials": pruned_trials,
        "train_rows": actual_train_rows,
        "valid_rows": actual_valid_rows,
        "feature_columns": feature_columns,
        "feature_count": feature_count,
        "train_id_sha256": train_sha256,
        "valid_id_sha256": valid_sha256,
        "baseline_metrics": {
            "model": baseline.model,
            "roc_auc": baseline.roc_auc,
            "log_loss": baseline.log_loss,
            "calibration_gap": baseline.calibration_gap,
            "valid_actual_ctr": baseline.valid_actual_ctr,
            "valid_mean_predicted_ctr": baseline.valid_mean_predicted_ctr,
            "best_iteration": baseline.best_iteration,
            "training_seconds": baseline.training_seconds,
            "train_rows": baseline.train_rows,
            "valid_rows": baseline.valid_rows,
            "train_id_sha256": baseline.train_id_sha256,
            "valid_id_sha256": baseline.valid_id_sha256,
        },
        "best_trial_number": best_trial.number,
        "best_params": best_params,
        "tuned_metrics": tuned_metrics_row,
        "model_output_path": str(paths.tuned_model),
        "prediction_output_path": str(paths.predictions),
        "holdout_used": False,
        "validation_passed": True,
    }
    paths.metadata_json.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation_passed = final_validation(
        TEST_MODE,
        study,
        tuned_metrics,
        tuned_probabilities,
        predictions_df,
        valid_meta,
        feature_columns,
        expected_valid_rows=valid_limit,
        train_rows=actual_train_rows,
        valid_rows=actual_valid_rows,
        train_sha256=train_sha256,
        valid_sha256=valid_sha256,
        formal_fingerprint_checked=formal_fingerprint_checked,
        paths=paths,
    )

    tuned_row = comparison_df.loc[comparison_df["model"] == TUNED_MODEL_KEY].iloc[0]
    logloss_improved = tuned_metrics.log_loss < baseline.log_loss
    auc_improved = tuned_metrics.roc_auc > baseline.roc_auc

    print("\n" + "=" * 72)
    print("第 32 步完成摘要")
    print("=" * 72)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"study 名称：{study_name}")
    print(f"本次新增 trial 数：{requested_new_trials}")
    print(f"累计 trial 数：{total_trials}")
    print(
        f"COMPLETE / FAIL / PRUNED："
        f"{complete_trials} / {failed_trials} / {pruned_trials}"
    )
    print(f"train 行数：{actual_train_rows:,}")
    print(f"valid 行数：{actual_valid_rows:,}")
    print(f"特征数量：{feature_count}")
    print(f"train_id_sha256：{train_sha256}")
    print(f"valid_id_sha256：{valid_sha256}")
    print("\n固定样本 LightGBM 基线指标：")
    print(f"  ROC-AUC：{baseline.roc_auc:.6f}")
    print(f"  LogLoss：{baseline.log_loss:.6f}")
    print(f"  calibration_gap：{baseline.calibration_gap:.6f}")
    print(f"\n最佳 trial 编号：{best_trial.number}")
    print(f"最佳参数：{best_params}")
    print("\n调优模型指标：")
    print(f"  ROC-AUC：{tuned_metrics.roc_auc:.6f}")
    print(f"  LogLoss：{tuned_metrics.log_loss:.6f}")
    print(f"  calibration_gap：{tuned_metrics.calibration_gap:.6f}")
    print("\n调优前后差异：")
    print(f"  AUC 绝对提升：{tuned_row['auc_absolute_improvement']:.6f}")
    print(f"  LogLoss 绝对下降：{tuned_row['logloss_absolute_reduction']:.6f}")
    print(f"  LogLoss 是否改善：{'是' if logloss_improved else '否'}")
    print(f"  AUC 是否提升：{'是' if auc_improved else '否'}")
    print("\n输出路径：")
    print(f"  trials CSV：{paths.trials_csv}")
    print(f"  best params：{paths.best_params_json}")
    print(f"  metrics：{paths.metrics_csv}")
    print(f"  comparison：{paths.comparison_csv}")
    print(f"  report：{paths.report_txt}")
    print(f"  metadata：{paths.metadata_json}")
    print(f"  model：{paths.tuned_model}")
    print(f"  predictions：{paths.predictions}")
    print(f"validation_passed：{validation_passed}")
    print("LightGBM Optuna 调优完成，holdout 尚未使用。")


if __name__ == "__main__":
    main()
