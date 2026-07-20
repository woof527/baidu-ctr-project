"""
百度 CTR 项目 — 调优 LightGBM SHAP 特征解释

功能：
    使用 SHAP 对第 32 步 Optuna 调优后的 LightGBM 进行模型解释，分析全局特征
    重要性、特征组贡献及典型预测案例。禁止读取 holdout，不重新训练模型。

数据输入：
    models/tuned_lightgbm_optuna_model.joblib
    data/tuning/lightgbm_valid/part-*.parquet
    outputs/fixed_tuning_sample_metadata.json
    outputs/lightgbm_optuna_metadata.json
    outputs/predictions/tuned_lightgbm_valid_predictions.parquet

用法：
    python scripts/33_explain_tuned_lightgbm_shap.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

RANDOM_STATE = 42
TEST_SHAP_ROWS = 2_000
FORMAL_SHAP_ROWS = 20_000
TOP_FEATURES = 20
LOCAL_MAX_DISPLAY = 15
THRESHOLD = 0.5

BATCH_SIZE = 200_000
EXPECTED_FEATURE_COUNT = 33
FULL_VALID_ROWS = 500_000
ADDITIVITY_WARN_THRESHOLD = 1e-3
CTR_DIFF_WARN_THRESHOLD = 0.01
IMPORTANCE_SUM_TOLERANCE = 0.01
NEUTRAL_SHAP_EPS = 1e-12

VALID_INPUT_DIR = Path("data/tuning/lightgbm_valid")
FIXED_SAMPLE_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")
OPTUNA_METADATA_PATH = Path("outputs/lightgbm_optuna_metadata.json")
OPTUNA_BEST_PARAMS_PATH = Path("outputs/lightgbm_optuna_best_params.json")

LOCAL_CASE_TYPES = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)

TIME_FEATURES = frozenset({"hour_sin", "hour_cos", "is_weekend"})


@dataclass
class OutputPaths:
    """第 33 步输出路径。"""

    test_mode: bool
    shap_dir: Path
    plots_dir: Path
    global_importance_csv: Path
    family_importance_csv: Path
    entity_importance_csv: Path
    local_contributions_csv: Path
    sample_rows_parquet: Path
    shap_values_npz: Path
    report_txt: Path
    metadata_json: Path


@dataclass
class LocalCase:
    """典型局部解释案例。"""

    case_type: str
    row_position: int
    id_value: Any
    click: int
    split_date: Any
    predicted_probability: float
    prediction_class: int
    actual_class: int
    base_value_raw: float
    model_raw_score: float
    shap_values: np.ndarray
    feature_values: np.ndarray


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    suffix = "_test" if test_mode else ""
    shap_dir = Path("outputs/shap")
    plots_dir = shap_dir / ("plots_test" if test_mode else "plots")

    return OutputPaths(
        test_mode=test_mode,
        shap_dir=shap_dir,
        plots_dir=plots_dir,
        global_importance_csv=shap_dir / f"tuned_lightgbm_shap_global_importance{suffix}.csv",
        family_importance_csv=shap_dir / f"tuned_lightgbm_shap_family_importance{suffix}.csv",
        entity_importance_csv=shap_dir / f"tuned_lightgbm_shap_entity_importance{suffix}.csv",
        local_contributions_csv=shap_dir / f"tuned_lightgbm_shap_local_contributions{suffix}.csv",
        sample_rows_parquet=shap_dir / f"tuned_lightgbm_shap_sample_rows{suffix}.parquet",
        shap_values_npz=shap_dir / f"tuned_lightgbm_shap_values{suffix}.npz",
        report_txt=Path(f"outputs/tuned_lightgbm_shap_report{suffix}.txt"),
        metadata_json=Path(f"outputs/tuned_lightgbm_shap_metadata{suffix}.json"),
    )


def get_shap_sample_rows(test_mode: bool) -> int:
    """返回 SHAP 抽样行数。"""

    return TEST_SHAP_ROWS if test_mode else FORMAL_SHAP_ROWS


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。"""

    if not path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixed_sample_metadata(metadata_path: Path) -> dict[str, Any]:
    """读取并校验固定样本元数据。"""

    metadata = load_json(metadata_path)

    if metadata.get("validation_passed") is not True:
        raise ValueError(
            f"固定样本 validation_passed={metadata.get('validation_passed')!r}，禁止解释。"
        )

    if metadata.get("holdout_used") is not False:
        raise ValueError(
            f"固定样本 holdout_used={metadata.get('holdout_used')!r}，禁止解释。"
        )

    if int(metadata.get("feature_count", -1)) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"feature_count={metadata.get('feature_count')!r}，"
            f"必须为 {EXPECTED_FEATURE_COUNT}。"
        )

    if int(metadata.get("actual_valid_rows", -1)) != FULL_VALID_ROWS:
        raise ValueError(
            f"actual_valid_rows={metadata.get('actual_valid_rows')!r}，"
            f"必须为 {FULL_VALID_ROWS:,}。"
        )

    required_keys = (
        "feature_columns",
        "feature_count",
        "actual_valid_rows",
        "valid_id_sha256",
    )
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"固定样本元数据缺少字段：{key}")

    return metadata


def load_optuna_metadata(metadata_path: Path, fixed_metadata: dict[str, Any]) -> dict[str, Any]:
    """读取并校验 Optuna 调优元数据。"""

    metadata = load_json(metadata_path)

    if metadata.get("test_mode") is not False:
        raise ValueError(
            f"Optuna 元数据 test_mode={metadata.get('test_mode')!r}，"
            "必须使用正式模式调优结果。"
        )

    if metadata.get("validation_passed") is not True:
        raise ValueError(
            f"Optuna validation_passed={metadata.get('validation_passed')!r}，禁止解释。"
        )

    if metadata.get("holdout_used") is not False:
        raise ValueError(
            f"Optuna holdout_used={metadata.get('holdout_used')!r}，禁止解释。"
        )

    if metadata.get("valid_id_sha256") != fixed_metadata["valid_id_sha256"]:
        raise ValueError("Optuna valid_id_sha256 与固定样本元数据不一致。")

    model_path = Path(metadata["model_output_path"])
    prediction_path = Path(metadata["prediction_output_path"])

    if not model_path.exists():
        raise FileNotFoundError(f"调优模型不存在：{model_path}")

    if not prediction_path.exists():
        raise FileNotFoundError(f"调优预测文件不存在：{prediction_path}")

    required_keys = (
        "best_trial_number",
        "best_params",
        "tuned_metrics",
        "model_output_path",
        "prediction_output_path",
        "train_id_sha256",
        "valid_id_sha256",
    )
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"Optuna 元数据缺少字段：{key}")

    return metadata


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    """按 part 文件名稳定排序 Parquet 文件。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(f"未找到固定 valid 目录：{parquet_dir}")

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


def calculate_sample_indices_sha256(sample_indices: np.ndarray) -> str:
    """计算 SHAP 抽样索引的 SHA256。"""

    hasher = hashlib.sha256()
    for index_value in sample_indices:
        hasher.update((str(int(index_value)) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def load_fixed_valid_data(
    parquet_files: list[Path],
    feature_columns: list[str],
    batch_size: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """加载完整固定 valid 数据。"""

    meta_parts: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    collected = 0

    read_columns = ["id", "click", "split_date", *feature_columns]

    for file_index, parquet_path in enumerate(parquet_files, start=1):
        print(f"读取 valid 文件 {file_index}/{len(parquet_files)}: {parquet_path.name}")
        parquet_file = pq.ParquetFile(parquet_path)

        for record_batch in parquet_file.iter_batches(columns=read_columns, batch_size=batch_size):
            batch_df = record_batch.to_pandas()
            meta_parts.append(batch_df[["id", "click", "split_date"]].copy())
            feature_parts.append(batch_df[feature_columns].to_numpy(dtype=np.float32))
            collected += len(batch_df)

    if collected != FULL_VALID_ROWS:
        raise ValueError(
            f"固定 valid 行数 {collected:,} 不等于期望 {FULL_VALID_ROWS:,}。"
        )

    meta_df = pd.concat(meta_parts, ignore_index=True)
    feature_matrix = np.vstack(feature_parts).astype(np.float32)

    del meta_parts, feature_parts
    gc.collect()

    return meta_df, feature_matrix


def validate_valid_data(
    meta_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_columns: list[str],
) -> None:
    """校验固定 valid 数据质量。"""

    if list(feature_matrix.shape) != [FULL_VALID_ROWS, EXPECTED_FEATURE_COUNT]:
        raise ValueError(
            f"特征矩阵形状 {feature_matrix.shape} 不符合 "
            f"({FULL_VALID_ROWS}, {EXPECTED_FEATURE_COUNT})。"
        )

    if feature_matrix.dtype != np.float32:
        raise ValueError("特征矩阵必须为 np.float32。")

    if np.isnan(feature_matrix).any():
        raise ValueError("固定 valid 特征存在 NaN。")

    if np.isinf(feature_matrix).any():
        raise ValueError("固定 valid 特征存在 inf。")

    if not np.isfinite(feature_matrix).all():
        raise ValueError("固定 valid 特征存在非有限值。")

    clicks = meta_df["click"].to_numpy()
    if not np.isin(clicks, [0, 1]).all():
        raise ValueError("click 不是仅包含 0 和 1。")

    if meta_df["id"].isna().any():
        raise ValueError("id 存在缺失。")

    if meta_df["split_date"].isna().any():
        raise ValueError("split_date 存在缺失。")

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"feature_columns 长度必须为 {EXPECTED_FEATURE_COUNT}。")


def load_and_validate_predictions(
    prediction_path: Path,
    valid_meta: pd.DataFrame,
) -> pd.DataFrame:
    """读取并校验调优模型验证预测。"""

    predictions_df = pd.read_parquet(prediction_path)

    required_columns = ["id", "click", "split_date", "tuned_lightgbm_probability"]
    missing_columns = [column for column in required_columns if column not in predictions_df.columns]
    if missing_columns:
        raise ValueError(f"预测文件缺少列：{missing_columns}")

    if len(predictions_df) != FULL_VALID_ROWS:
        raise ValueError(
            f"预测文件行数 {len(predictions_df):,} 不等于 {FULL_VALID_ROWS:,}。"
        )

    if not predictions_df["id"].equals(valid_meta["id"].reset_index(drop=True)):
        raise ValueError("预测文件 id 顺序与固定 valid 不一致。")

    if not predictions_df["click"].equals(valid_meta["click"].reset_index(drop=True)):
        raise ValueError("预测文件 click 顺序与固定 valid 不一致。")

    if not predictions_df["split_date"].equals(valid_meta["split_date"].reset_index(drop=True)):
        raise ValueError("预测文件 split_date 顺序与固定 valid 不一致。")

    probabilities = predictions_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError("预测概率存在 NaN 或 inf。")

    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("预测概率超出 [0, 1] 范围。")

    return predictions_df


def draw_shap_sample_indices(valid_rows: int, shap_sample_rows: int) -> np.ndarray:
    """固定随机种子无放回抽样 SHAP 样本索引。"""

    if shap_sample_rows > valid_rows:
        raise ValueError(
            f"SHAP 抽样行数 {shap_sample_rows:,} 超过 valid 总行数 {valid_rows:,}。"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(
        rng.choice(valid_rows, size=shap_sample_rows, replace=False)
    )
    return sample_indices.astype(np.int64)


def load_tuned_model(model_path: Path) -> lgb.LGBMClassifier:
    """加载调优 LightGBM 模型。"""

    model = joblib.load(model_path)
    if not isinstance(model, lgb.LGBMClassifier):
        raise TypeError(f"模型类型错误：{type(model)!r}，期望 lightgbm.LGBMClassifier。")
    return model


def validate_model_features(model: lgb.LGBMClassifier, feature_columns: list[str]) -> int:
    """校验模型特征名及顺序。"""

    model_features = list(model.feature_name_)
    if model_features != feature_columns:
        raise ValueError(
            "模型 feature_name_ 与 metadata feature_columns 不一致。\n"
            f"模型：{model_features[:5]} ...\n"
            f"元数据：{feature_columns[:5]} ..."
        )

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration is not None and int(best_iteration) <= 0:
        raise ValueError(f"模型 best_iteration_ 无效：{best_iteration}")

    return int(best_iteration) if best_iteration is not None else -1


def build_shap_booster(model: lgb.LGBMClassifier, best_iteration: int) -> lgb.Booster:
    """从模型 booster 构建用于 SHAP 解释的固定 Booster。"""

    if best_iteration > 0:
        model_string = model.booster_.model_to_string(num_iteration=best_iteration)
    else:
        model_string = model.booster_.model_to_string()

    return lgb.Booster(model_str=model_string)


def normalize_shap_explanation(
    raw_explanation: shap.Explanation,
    feature_names: list[str],
    expected_rows: int,
    expected_features: int,
) -> tuple[shap.Explanation, tuple[int, ...], tuple[int, ...], int]:
    """统一 SHAP 输出为点击正类 Explanation。"""

    values = np.asarray(raw_explanation.values)
    original_shape = tuple(values.shape)
    explained_class_index = 1

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, explained_class_index]
        else:
            raise ValueError(
                f"无法识别三维 SHAP values 最后一维：{values.shape[-1]}"
            )
    elif values.ndim == 2:
        explained_class_index = 1
    elif isinstance(raw_explanation.values, list):
        if len(raw_explanation.values) != 2:
            raise ValueError(
                f"SHAP list 输出长度 {len(raw_explanation.values)}，期望二分类 2 个输出。"
            )
        values = np.asarray(raw_explanation.values[explained_class_index])
        if values.ndim != 2:
            raise ValueError(f"正类 SHAP values 维度异常：{values.shape}")
    else:
        raise ValueError(f"无法处理的 SHAP values 形状：{original_shape}")

    base_values = np.asarray(raw_explanation.base_values)
    if base_values.ndim == 2:
        if base_values.shape[-1] == 2:
            base_values = base_values[:, explained_class_index]
        else:
            raise ValueError(f"无法识别的 base_values 形状：{base_values.shape}")
    elif base_values.ndim == 0:
        base_values = np.full(expected_rows, float(base_values), dtype=np.float32)
    elif base_values.ndim == 1 and len(base_values) != expected_rows:
        raise ValueError(
            f"base_values 长度 {len(base_values)} 与样本数 {expected_rows} 不一致。"
        )

    data = np.asarray(raw_explanation.data)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if values.shape != (expected_rows, expected_features):
        raise ValueError(
            f"规范化后 SHAP values 形状 {values.shape} 不符合 "
            f"({expected_rows}, {expected_features})。"
        )

    if data.shape != (expected_rows, expected_features):
        raise ValueError(
            f"SHAP data 形状 {data.shape} 不符合 ({expected_rows}, {expected_features})。"
        )

    if len(feature_names) != expected_features:
        raise ValueError("feature_names 长度与期望特征数不一致。")

    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("SHAP values 存在 NaN 或 inf。")

    if np.isnan(base_values).any() or np.isinf(base_values).any():
        raise ValueError("SHAP base_values 存在 NaN 或 inf。")

    normalized = shap.Explanation(
        values=values.astype(np.float32),
        base_values=base_values.astype(np.float32),
        data=data.astype(np.float32),
        feature_names=feature_names,
    )

    normalized_shape = tuple(normalized.values.shape)
    return normalized, original_shape, normalized_shape, explained_class_index


def check_additivity(
    normalized_explanation: shap.Explanation,
    shap_booster: lgb.Booster,
    x_shap: pd.DataFrame,
) -> tuple[float, float]:
    """检查 SHAP 加和与 LightGBM raw score 的一致性。"""

    reconstructed_raw = (
        np.asarray(normalized_explanation.base_values, dtype=np.float64)
        + normalized_explanation.values.sum(axis=1)
    )
    model_raw = shap_booster.predict(x_shap, raw_score=True).astype(np.float64)

    abs_errors = np.abs(reconstructed_raw - model_raw)
    mean_error = float(abs_errors.mean())
    max_error = float(abs_errors.max())

    if max_error > ADDITIVITY_WARN_THRESHOLD:
        warnings.warn(
            "SHAP 加和一致性误差超过阈值："
            f"mean={mean_error:.6e}, max={max_error:.6e}, "
            f"shap={shap.__version__}, lightgbm={lgb.__version__}, "
            f"values_shape={normalized_explanation.values.shape}",
            stacklevel=2,
        )

    if not np.isfinite(mean_error) or not np.isfinite(max_error):
        raise ValueError("SHAP 加和一致性误差为非有限值，无法完成合理加和检查。")

    return mean_error, max_error


def build_global_importance_dataframe(
    normalized_explanation: shap.Explanation,
    feature_columns: list[str],
) -> pd.DataFrame:
    """构建全局特征重要性表。"""

    values = normalized_explanation.values
    mean_abs_shap = np.abs(values).mean(axis=0)
    mean_signed_shap = values.mean(axis=0)
    median_abs_shap = np.median(np.abs(values), axis=0)
    std_shap = values.std(axis=0)
    positive_shap_ratio = (values > 0).mean(axis=0)
    negative_shap_ratio = (values < 0).mean(axis=0)

    total_mean_abs = float(mean_abs_shap.sum())
    if total_mean_abs <= 0:
        raise ValueError("全局 mean_abs_shap 总和为 0，无法计算 importance_percent。")

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": mean_abs_shap.astype(np.float64),
            "mean_signed_shap": mean_signed_shap.astype(np.float64),
            "median_abs_shap": median_abs_shap.astype(np.float64),
            "std_shap": std_shap.astype(np.float64),
            "positive_shap_ratio": positive_shap_ratio.astype(np.float64),
            "negative_shap_ratio": negative_shap_ratio.astype(np.float64),
        }
    )
    importance_df["importance_percent"] = (
        importance_df["mean_abs_shap"] / total_mean_abs
    )
    importance_df = importance_df.sort_values(
        "mean_abs_shap",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1)

    return importance_df


def classify_feature_family(feature: str) -> str:
    """按特征名划分特征家族。"""

    if feature.endswith("_te"):
        return "target_encoding"
    if feature.endswith("_hist_ctr"):
        return "historical_ctr"
    if feature.endswith("_hist_impressions"):
        return "historical_impressions"
    if feature.endswith("_hist_clicks"):
        return "historical_clicks"
    if feature.endswith("_exposure_percentile"):
        return "exposure_percentile"
    if feature.endswith("_freq"):
        return "frequency"
    if feature in TIME_FEATURES:
        return "time"
    return "other"


def classify_feature_entity(feature: str) -> str:
    """按特征名划分业务实体。"""

    entity_prefixes = (
        ("site_id_", "site_id"),
        ("site_category_", "site_category"),
        ("app_id_", "app_id"),
        ("app_category_", "app_category"),
        ("device_model_", "device_model"),
    )
    for prefix, entity in entity_prefixes:
        if feature.startswith(prefix):
            return entity
    if feature in TIME_FEATURES:
        return "time"
    return "other"


def build_group_importance_dataframe(
    global_importance_df: pd.DataFrame,
    group_column: str,
    group_name: str,
    percent_column: str,
) -> pd.DataFrame:
    """构建家族或实体分组重要性表。"""

    grouped = global_importance_df.groupby(group_column, sort=False)
    rows: list[dict[str, Any]] = []

    for group_value, group_df in grouped:
        total_mean_abs = float(group_df["mean_abs_shap"].sum())
        top_row = group_df.sort_values("mean_abs_shap", ascending=False).iloc[0]
        rows.append(
            {
                group_name: group_value,
                "feature_count": int(len(group_df)),
                "total_mean_abs_shap": total_mean_abs,
                "top_feature": top_row["feature"],
                "top_feature_mean_abs_shap": float(top_row["mean_abs_shap"]),
            }
        )

    result_df = pd.DataFrame(rows)
    total = float(result_df["total_mean_abs_shap"].sum())
    if total <= 0:
        raise ValueError(f"{group_name} total_mean_abs_shap 总和为 0。")

    result_df[percent_column] = result_df["total_mean_abs_shap"] / total
    result_df = result_df.sort_values(
        "total_mean_abs_shap",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    return result_df


def save_global_shap_plots(
    normalized_explanation: shap.Explanation,
    plots_dir: Path,
    family_df: pd.DataFrame,
    entity_df: pd.DataFrame,
) -> None:
    """保存 SHAP 全局图表。"""

    plots_dir.mkdir(parents=True, exist_ok=True)

    bar_path = plots_dir / "tuned_lightgbm_shap_bar_top20.png"
    plt.figure(figsize=(10, 8))
    shap.plots.bar(
        normalized_explanation,
        max_display=TOP_FEATURES,
        show=False,
    )
    plt.title(f"调优 LightGBM 全局 SHAP 重要性（Top {TOP_FEATURES}）")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close()

    beeswarm_path = plots_dir / "tuned_lightgbm_shap_beeswarm_top20.png"
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(
        normalized_explanation,
        max_display=TOP_FEATURES,
        show=False,
    )
    plt.title(f"调优 LightGBM SHAP Beeswarm（Top {TOP_FEATURES}）")
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    family_path = plots_dir / "tuned_lightgbm_shap_family_importance.png"
    plt.figure(figsize=(10, 6))
    plt.barh(
        family_df["feature_family"][::-1],
        family_df["group_importance_percent"][::-1],
    )
    plt.xlabel("group_importance_percent")
    plt.title("特征家族 SHAP 重要性占比")
    plt.tight_layout()
    plt.savefig(family_path, dpi=160, bbox_inches="tight")
    plt.close()

    entity_path = plots_dir / "tuned_lightgbm_shap_entity_importance.png"
    plt.figure(figsize=(10, 6))
    plt.barh(
        entity_df["entity"][::-1],
        entity_df["entity_importance_percent"][::-1],
    )
    plt.xlabel("entity_importance_percent")
    plt.title("业务实体 SHAP 重要性占比")
    plt.tight_layout()
    plt.savefig(entity_path, dpi=160, bbox_inches="tight")
    plt.close()


def find_local_case_row(
    predictions_df: pd.DataFrame,
    case_type: str,
    threshold: float,
) -> int | None:
    """在完整 valid 预测中查找典型案例 row_position。"""

    probabilities = predictions_df["tuned_lightgbm_probability"].to_numpy(dtype=np.float64)
    clicks = predictions_df["click"].to_numpy(dtype=np.int8)
    row_positions = np.arange(len(predictions_df), dtype=np.int64)

    if case_type == "true_positive":
        mask = (clicks == 1) & (probabilities >= threshold)
        if not mask.any():
            return None
        candidates = row_positions[mask]
        return int(candidates[np.argmax(probabilities[mask])])

    if case_type == "true_negative":
        mask = (clicks == 0) & (probabilities < threshold)
        if not mask.any():
            return None
        candidates = row_positions[mask]
        return int(candidates[np.argmin(probabilities[mask])])

    if case_type == "false_positive":
        mask = (clicks == 0) & (probabilities >= threshold)
        if not mask.any():
            return None
        candidates = row_positions[mask]
        return int(candidates[np.argmax(probabilities[mask])])

    if case_type == "false_negative":
        mask = (clicks == 1) & (probabilities < threshold)
        if not mask.any():
            return None
        candidates = row_positions[mask]
        return int(candidates[np.argmin(probabilities[mask])])

    raise ValueError(f"未知 case_type：{case_type}")


def explain_local_case(
    case_type: str,
    row_position: int,
    valid_meta: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_columns: list[str],
    predictions_df: pd.DataFrame,
    explainer: shap.TreeExplainer,
    shap_booster: lgb.Booster,
    plots_dir: Path,
) -> LocalCase:
    """计算并保存单个局部案例 SHAP 解释。"""

    x_row = pd.DataFrame(
        feature_matrix[row_position : row_position + 1],
        columns=feature_columns,
    )
    raw_local = explainer(x_row, check_additivity=False)
    local_explanation, _, _, _ = normalize_shap_explanation(
        raw_local,
        feature_columns,
        expected_rows=1,
        expected_features=len(feature_columns),
    )

    shap_values = local_explanation.values[0].astype(np.float32)
    feature_values = local_explanation.data[0].astype(np.float32)
    base_value_raw = float(local_explanation.base_values[0])
    model_raw_score = float(shap_booster.predict(x_row, raw_score=True)[0])

    meta_row = valid_meta.iloc[row_position]
    predicted_probability = float(
        predictions_df.iloc[row_position]["tuned_lightgbm_probability"]
    )
    click = int(meta_row["click"])
    prediction_class = int(predicted_probability >= THRESHOLD)

    waterfall_path = plots_dir / f"tuned_lightgbm_shap_waterfall_{case_type}.png"
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        local_explanation[0],
        max_display=LOCAL_MAX_DISPLAY,
        show=False,
    )
    plt.title(f"调优 LightGBM 局部 SHAP Waterfall — {case_type}")
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    return LocalCase(
        case_type=case_type,
        row_position=row_position,
        id_value=meta_row["id"],
        click=click,
        split_date=meta_row["split_date"],
        predicted_probability=predicted_probability,
        prediction_class=prediction_class,
        actual_class=click,
        base_value_raw=base_value_raw,
        model_raw_score=model_raw_score,
        shap_values=shap_values,
        feature_values=feature_values,
    )


def build_local_contributions_dataframe_fixed(
    cases: list[LocalCase],
    feature_columns: list[str],
) -> pd.DataFrame:
    """构建局部贡献明细表（使用特征名）。"""

    rows: list[dict[str, Any]] = []

    for case in cases:
        abs_values = np.abs(case.shap_values)
        rank_order = np.argsort(-abs_values)

        for rank, feature_index in enumerate(rank_order, start=1):
            feature_name = feature_columns[feature_index]
            shap_value = float(case.shap_values[feature_index])
            if shap_value > NEUTRAL_SHAP_EPS:
                direction = "increase_prediction"
            elif shap_value < -NEUTRAL_SHAP_EPS:
                direction = "decrease_prediction"
            else:
                direction = "neutral"

            rows.append(
                {
                    "case_type": case.case_type,
                    "row_position": case.row_position,
                    "id": case.id_value,
                    "click": case.click,
                    "split_date": case.split_date,
                    "predicted_probability": case.predicted_probability,
                    "threshold": THRESHOLD,
                    "prediction_class": case.prediction_class,
                    "actual_class": case.actual_class,
                    "base_value_raw": case.base_value_raw,
                    "model_raw_score": case.model_raw_score,
                    "feature": feature_name,
                    "feature_value": float(case.feature_values[feature_index]),
                    "shap_value_raw": shap_value,
                    "absolute_shap_value": float(abs_values[feature_index]),
                    "contribution_rank": rank,
                    "direction": direction,
                }
            )

    return pd.DataFrame(rows)


def summarize_local_case_contributions(
    case: LocalCase,
    feature_columns: list[str],
) -> dict[str, Any]:
    """汇总单个案例最重要的正向和负向贡献。"""

    positive_index = int(np.argmax(case.shap_values))
    negative_index = int(np.argmin(case.shap_values))

    return {
        "case_type": case.case_type,
        "row_position": case.row_position,
        "predicted_probability": case.predicted_probability,
        "top_positive_feature": feature_columns[positive_index],
        "top_positive_shap": float(case.shap_values[positive_index]),
        "top_negative_feature": feature_columns[negative_index],
        "top_negative_shap": float(case.shap_values[negative_index]),
    }


def write_text_report(
    report_path: Path,
    test_mode: bool,
    optuna_metadata: dict[str, Any],
    fixed_metadata: dict[str, Any],
    full_valid_ctr: float,
    shap_sample_rows: int,
    shap_sample_ctr: float,
    best_iteration: int,
    original_shap_shape: tuple[int, ...],
    normalized_shap_shape: tuple[int, ...],
    mean_additivity_error: float,
    max_additivity_error: float,
    global_importance_df: pd.DataFrame,
    family_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    local_case_summaries: list[dict[str, Any]],
    missing_case_types: list[str],
) -> None:
    """写入中文 SHAP 报告。"""

    tuned_metrics = optuna_metadata["tuned_metrics"]
    best_params = optuna_metadata["best_params"]
    mode_label = "TEST_MODE=True（流程验证，非正式结论）" if test_mode else "TEST_MODE=False"

    top_features = global_importance_df.head(TOP_FEATURES)
    top_feature_lines = [
        f"  {row.feature}: mean_abs_shap={row.mean_abs_shap:.6f}, "
        f"mean_signed_shap={row.mean_signed_shap:.6f}"
        for row in top_features.itertuples(index=False)
    ]

    family_lines = [
        f"  {row.feature_family}: {row.group_importance_percent:.4f} "
        f"(top={row.top_feature})"
        for row in family_df.itertuples(index=False)
    ]

    entity_lines = [
        f"  {row.entity}: {row.entity_importance_percent:.4f} "
        f"(top={row.top_feature})"
        for row in entity_df.itertuples(index=False)
    ]

    local_lines: list[str] = []
    for summary in local_case_summaries:
        local_lines.extend(
            [
                f"  {summary['case_type']}: row={summary['row_position']}, "
                f"prob={summary['predicted_probability']:.6f}",
                f"    最重要正向：{summary['top_positive_feature']} "
                f"({summary['top_positive_shap']:.6f})",
                f"    最重要负向：{summary['top_negative_feature']} "
                f"({summary['top_negative_shap']:.6f})",
            ]
        )

    missing_lines = [
        f"  {case_type}: 不存在" for case_type in missing_case_types
    ] or ["  四类案例均存在"]

    lines = [
        "百度 CTR 项目 — 第 33 步 调优 LightGBM SHAP 特征解释报告",
        "=" * 72,
        "",
        "【1. SHAP 的作用】",
        "  SHAP 基于 Shapley 值，量化每个特征对单个预测的贡献，"
        "帮助理解模型依赖哪些信息做出点击概率判断。",
        "",
        "【2. 本次解释的模型】",
        f"  {optuna_metadata['model_output_path']}",
        "",
        f"【3. 最佳 Optuna trial 编号】{optuna_metadata['best_trial_number']}",
        "",
        "【4. 最佳参数】",
        *[f"  {key}: {value}" for key, value in best_params.items()],
        "",
        "【5. 调优模型 valid 指标】",
        f"  ROC-AUC：{tuned_metrics['roc_auc']:.6f}",
        f"  LogLoss：{tuned_metrics['log_loss']:.6f}",
        f"  calibration_gap：{tuned_metrics['calibration_gap']:.6f}",
        "",
        "【6. 数据范围】",
        "  使用固定 valid 样本，不是 holdout，也不是原始 test.csv。",
        "",
        f"【7. 完整 valid 行数】{FULL_VALID_ROWS:,}",
        f"【8. SHAP 抽样行数】{shap_sample_rows:,}",
        f"【9. SHAP 抽样 CTR / 完整 valid CTR】"
        f"{shap_sample_ctr:.6f} / {full_valid_ctr:.6f} "
        f"(差异 {abs(shap_sample_ctr - full_valid_ctr):.6f})",
        f"【10. 随机种子】{RANDOM_STATE}",
        f"【11. SHAP / LightGBM 版本】{shap.__version__} / {lgb.__version__}",
        f"【12. 模型 best_iteration】{best_iteration if best_iteration > 0 else '全部迭代'}",
        "",
        "【13. SHAP 输出尺度】",
        "  本次解释 LightGBM raw score（log-odds 尺度），"
        "SHAP 正值表示推动点击预测升高，负值表示推动降低。",
        "  不能将 raw SHAP 值直接解释为概率百分点变化。",
        "",
        "【14. 加和一致性误差】",
        f"  mean_absolute_additivity_error={mean_additivity_error:.6e}",
        f"  max_absolute_additivity_error={max_additivity_error:.6e}",
        "",
        f"【15. 全局前 {TOP_FEATURES} 个重要特征】",
        *top_feature_lines,
        "",
        "【16. 特征家族排名】",
        *family_lines,
        "",
        "【17. 业务实体排名】",
        *entity_lines,
        "",
        "【18. 局部案例是否存在】",
        *missing_lines,
        "",
        "【19. 局部案例主要贡献】",
        *(local_lines or ["  无可用局部案例"]),
        "",
        "【20. 因果说明】",
        "  SHAP 解释代表模型依赖关系，不代表特征与点击之间存在因果关系。",
        "",
        "【21. 相关特征分摊】",
        "  高相关特征可能分摊 SHAP 重要性，单个特征贡献不应被过度解读。",
        "",
        "【22. TE 与历史 CTR】",
        "  Target Encoding 与 historical_ctr 等相关特征应结合特征家族/实体组解释。",
        "",
        "【23. holdout 尚未使用】是",
        "",
        "【24. 下一步建议】",
        "  建议继续开展概率校准与阈值分析，并在 holdout 上完成最终评估。",
        "",
        f"当前模式：{mode_label}",
        f"valid_id_sha256：{fixed_metadata['valid_id_sha256']}",
        f"原始 SHAP 形状：{original_shap_shape}",
        f"规范化 SHAP 形状：{normalized_shap_shape}",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def final_validation(
    paths: OutputPaths,
    model: lgb.LGBMClassifier,
    feature_columns: list[str],
    valid_id_sha256: str,
    expected_valid_sha256: str,
    sample_indices: np.ndarray,
    shap_sample_rows: int,
    normalized_explanation: shap.Explanation,
    global_importance_df: pd.DataFrame,
    family_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    local_cases: list[LocalCase],
    shap_booster: lgb.Booster,
    x_shap: pd.DataFrame,
) -> bool:
    """最终验收检查。"""

    if not isinstance(model, lgb.LGBMClassifier):
        raise TypeError("模型类型必须为 LGBMClassifier。")

    if list(model.feature_name_) != feature_columns:
        raise ValueError("模型特征名及顺序不一致。")

    if valid_id_sha256 != expected_valid_sha256:
        raise ValueError("valid 指纹与第 30 步不一致。")

    if len(sample_indices) != shap_sample_rows:
        raise ValueError("SHAP 抽样索引数量不正确。")

    if normalized_explanation.values.shape != (
        shap_sample_rows,
        EXPECTED_FEATURE_COUNT,
    ):
        raise ValueError("SHAP values 形状不正确。")

    if np.isnan(normalized_explanation.values).any():
        raise ValueError("SHAP values 存在 NaN。")

    if np.isinf(normalized_explanation.values).any():
        raise ValueError("SHAP values 存在 inf。")

    base_values = np.asarray(normalized_explanation.base_values)
    if np.isnan(base_values).any() or np.isinf(base_values).any():
        raise ValueError("base_values 存在 NaN 或 inf。")

    importance_sum = float(global_importance_df["importance_percent"].sum())
    if abs(importance_sum - 1.0) > IMPORTANCE_SUM_TOLERANCE:
        raise ValueError(
            f"全局 importance_percent 总和 {importance_sum:.6f} 不接近 1。"
        )

    family_sum = float(family_df["group_importance_percent"].sum())
    if abs(family_sum - 1.0) > IMPORTANCE_SUM_TOLERANCE:
        raise ValueError(
            f"family importance 总和 {family_sum:.6f} 不接近 1。"
        )

    entity_sum = float(entity_df["entity_importance_percent"].sum())
    if abs(entity_sum - 1.0) > IMPORTANCE_SUM_TOLERANCE:
        raise ValueError(
            f"entity importance 总和 {entity_sum:.6f} 不接近 1。"
        )

    for case in local_cases:
        reconstructed = case.base_value_raw + float(case.shap_values.sum())
        if abs(reconstructed - case.model_raw_score) > ADDITIVITY_WARN_THRESHOLD:
            warnings.warn(
                f"局部案例 {case.case_type} raw score 还原误差偏大："
                f"{abs(reconstructed - case.model_raw_score):.6e}",
                stacklevel=2,
            )

    required_outputs = [
        paths.global_importance_csv,
        paths.family_importance_csv,
        paths.entity_importance_csv,
        paths.local_contributions_csv,
        paths.sample_rows_parquet,
        paths.shap_values_npz,
        paths.report_txt,
        paths.plots_dir / "tuned_lightgbm_shap_bar_top20.png",
        paths.plots_dir / "tuned_lightgbm_shap_beeswarm_top20.png",
        paths.plots_dir / "tuned_lightgbm_shap_family_importance.png",
        paths.plots_dir / "tuned_lightgbm_shap_entity_importance.png",
    ]
    for output_path in required_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"缺少输出文件：{output_path}")

    for case in local_cases:
        waterfall_path = paths.plots_dir / f"tuned_lightgbm_shap_waterfall_{case.case_type}.png"
        if not waterfall_path.exists():
            raise FileNotFoundError(f"缺少局部 waterfall 图：{waterfall_path}")

    # 再次确认 raw score 可还原
    reconstructed_raw = (
        np.asarray(normalized_explanation.base_values, dtype=np.float64)
        + normalized_explanation.values.sum(axis=1)
    )
    model_raw = shap_booster.predict(x_shap, raw_score=True)
    if not np.allclose(reconstructed_raw, model_raw, atol=ADDITIVITY_WARN_THRESHOLD * 10):
        warnings.warn(
            "最终验收：全局 SHAP 加和与 raw score 存在可见差异。",
            stacklevel=2,
        )

    return True


def main() -> None:
    """主流程：加载模型与 fixed valid → 计算 SHAP → 保存解释结果。"""

    paths = get_output_paths(TEST_MODE)
    shap_sample_rows = get_shap_sample_rows(TEST_MODE)

    print("=" * 72)
    print("第 33 步：调优 LightGBM SHAP 特征解释")
    print("=" * 72)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"SHAP 版本：{shap.__version__}")
    print(f"LightGBM 版本：{lgb.__version__}")
    print(f"SHAP 抽样行数：{shap_sample_rows:,}")

    fixed_metadata = load_fixed_sample_metadata(FIXED_SAMPLE_METADATA_PATH)
    optuna_metadata = load_optuna_metadata(OPTUNA_METADATA_PATH, fixed_metadata)
    _ = load_json(OPTUNA_BEST_PARAMS_PATH)

    feature_columns: list[str] = fixed_metadata["feature_columns"]
    feature_count = int(fixed_metadata["feature_count"])

    if len(feature_columns) != feature_count:
        raise ValueError("feature_columns 与 feature_count 不一致。")

    model_path = Path(optuna_metadata["model_output_path"])
    prediction_path = Path(optuna_metadata["prediction_output_path"])

    print(f"\n模型路径：{model_path}")
    print(f"预测路径：{prediction_path}")

    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)
    valid_id_sha256 = calculate_id_sha256(valid_files)

    if valid_id_sha256 != fixed_metadata["valid_id_sha256"]:
        raise ValueError(
            f"valid_id_sha256 不一致：当前 {valid_id_sha256}，"
            f"元数据 {fixed_metadata['valid_id_sha256']}"
        )

    print("\n加载完整 fixed valid ...")
    valid_meta, feature_matrix = load_fixed_valid_data(valid_files, feature_columns)
    validate_valid_data(valid_meta, feature_matrix, feature_columns)

    print("\n校验调优模型验证预测顺序 ...")
    predictions_df = load_and_validate_predictions(prediction_path, valid_meta)

    full_valid_ctr = float(valid_meta["click"].mean())
    sample_indices = draw_shap_sample_indices(FULL_VALID_ROWS, shap_sample_rows)
    sample_indices_sha256 = calculate_sample_indices_sha256(sample_indices)
    shap_sample_ctr = float(valid_meta.iloc[sample_indices]["click"].mean())
    ctr_diff = abs(shap_sample_ctr - full_valid_ctr)

    print(f"完整 valid CTR：{full_valid_ctr:.6f}")
    print(f"SHAP 样本 CTR：{shap_sample_ctr:.6f}")
    print(f"CTR 差异：{ctr_diff:.6f}")

    if ctr_diff > CTR_DIFF_WARN_THRESHOLD:
        warnings.warn(
            f"SHAP 抽样 CTR 与完整 valid CTR 相差 {ctr_diff:.6f}，超过 "
            f"{CTR_DIFF_WARN_THRESHOLD}，仅输出警告，不重新抽样。",
            stacklevel=2,
        )

    print("\n加载调优模型 ...")
    model = load_tuned_model(model_path)
    best_iteration = validate_model_features(model, feature_columns)
    shap_booster = build_shap_booster(model, best_iteration)

    x_shap = pd.DataFrame(
        feature_matrix[sample_indices],
        columns=feature_columns,
    )

    print("\n创建 SHAP TreeExplainer 并计算全局 SHAP ...")
    explainer = shap.TreeExplainer(
        shap_booster,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )
    raw_explanation = explainer(x_shap, check_additivity=False)

    normalized_explanation, original_shape, normalized_shape, explained_class_index = (
        normalize_shap_explanation(
            raw_explanation,
            feature_columns,
            expected_rows=shap_sample_rows,
            expected_features=feature_count,
        )
    )

    mean_additivity_error, max_additivity_error = check_additivity(
        normalized_explanation,
        shap_booster,
        x_shap,
    )

    print(f"原始 SHAP 形状：{original_shape}")
    print(f"规范化 SHAP 形状：{normalized_shape}")
    print(f"解释类别索引：{explained_class_index}")
    print(
        f"加和一致性误差：mean={mean_additivity_error:.6e}, "
        f"max={max_additivity_error:.6e}"
    )

    global_importance_df = build_global_importance_dataframe(
        normalized_explanation,
        feature_columns,
    )
    global_importance_df["feature_family"] = global_importance_df["feature"].map(
        classify_feature_family
    )
    global_importance_df["entity"] = global_importance_df["feature"].map(
        classify_feature_entity
    )

    family_df = build_group_importance_dataframe(
        global_importance_df,
        group_column="feature_family",
        group_name="feature_family",
        percent_column="group_importance_percent",
    )
    entity_df = build_group_importance_dataframe(
        global_importance_df,
        group_column="entity",
        group_name="entity",
        percent_column="entity_importance_percent",
    )

    paths.shap_dir.mkdir(parents=True, exist_ok=True)
    global_importance_df.drop(columns=["feature_family", "entity"]).to_csv(
        paths.global_importance_csv,
        index=False,
        encoding="utf-8",
    )
    family_df.to_csv(paths.family_importance_csv, index=False, encoding="utf-8")
    entity_df.to_csv(paths.entity_importance_csv, index=False, encoding="utf-8")

    print("\n保存全局 SHAP 图表 ...")
    save_global_shap_plots(
        normalized_explanation,
        paths.plots_dir,
        family_df,
        entity_df,
    )

    print("\n查找并解释典型局部案例 ...")
    local_cases: list[LocalCase] = []
    missing_case_types: list[str] = []

    for case_type in LOCAL_CASE_TYPES:
        row_position = find_local_case_row(predictions_df, case_type, THRESHOLD)
        if row_position is None:
            missing_case_types.append(case_type)
            print(f"  {case_type}: 不存在")
            continue

        print(f"  {case_type}: row_position={row_position}")
        local_case = explain_local_case(
            case_type=case_type,
            row_position=row_position,
            valid_meta=valid_meta,
            feature_matrix=feature_matrix,
            feature_columns=feature_columns,
            predictions_df=predictions_df,
            explainer=explainer,
            shap_booster=shap_booster,
            plots_dir=paths.plots_dir,
        )
        local_cases.append(local_case)

    local_contributions_df = build_local_contributions_dataframe_fixed(
        local_cases,
        feature_columns,
    )
    local_contributions_df.to_csv(
        paths.local_contributions_csv,
        index=False,
        encoding="utf-8",
    )

    sample_rows_df = valid_meta.iloc[sample_indices].copy().reset_index(drop=True)
    sample_rows_df.insert(0, "row_position", sample_indices)
    sample_rows_df["tuned_lightgbm_probability"] = predictions_df.iloc[sample_indices][
        "tuned_lightgbm_probability"
    ].to_numpy()
    for feature_name in feature_columns:
        sample_rows_df[feature_name] = feature_matrix[sample_indices, feature_columns.index(feature_name)]
    sample_rows_df.to_parquet(paths.sample_rows_parquet, index=False)

    np.savez_compressed(
        paths.shap_values_npz,
        values=normalized_explanation.values.astype(np.float32),
        base_values=np.asarray(normalized_explanation.base_values, dtype=np.float32),
        data=normalized_explanation.data.astype(np.float32),
        feature_names=np.array(feature_columns, dtype=object),
        sample_indices=sample_indices.astype(np.int64),
    )

    local_case_summaries = [
        summarize_local_case_contributions(case, feature_columns) for case in local_cases
    ]

    write_text_report(
        paths.report_txt,
        TEST_MODE,
        optuna_metadata,
        fixed_metadata,
        full_valid_ctr,
        shap_sample_rows,
        shap_sample_ctr,
        best_iteration,
        original_shape,
        normalized_shape,
        mean_additivity_error,
        max_additivity_error,
        global_importance_df,
        family_df,
        entity_df,
        local_case_summaries,
        missing_case_types,
    )

    top_features = global_importance_df.head(10)[
        ["feature", "mean_abs_shap", "mean_signed_shap", "importance_percent"]
    ].to_dict(orient="records")
    feature_family_ranking = family_df[
        ["feature_family", "group_importance_percent", "top_feature"]
    ].to_dict(orient="records")
    entity_ranking = entity_df[
        ["entity", "entity_importance_percent", "top_feature"]
    ].to_dict(orient="records")

    metadata_payload = {
        "script_name": "scripts/33_explain_tuned_lightgbm_shap.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "shap_version": shap.__version__,
        "lightgbm_version": lgb.__version__,
        "model_path": str(model_path),
        "source_optuna_metadata": str(OPTUNA_METADATA_PATH),
        "best_trial_number": optuna_metadata["best_trial_number"],
        "best_iteration": best_iteration if best_iteration > 0 else None,
        "feature_columns": feature_columns,
        "feature_count": feature_count,
        "full_valid_rows": FULL_VALID_ROWS,
        "full_valid_ctr": full_valid_ctr,
        "shap_sample_rows": shap_sample_rows,
        "shap_sample_ctr": shap_sample_ctr,
        "shap_sample_random_state": RANDOM_STATE,
        "sample_indices_sha256": sample_indices_sha256,
        "original_shap_shape": list(original_shape),
        "normalized_shap_shape": list(normalized_shape),
        "explained_class_index": explained_class_index,
        "model_output_scale": "raw",
        "mean_absolute_additivity_error": mean_additivity_error,
        "max_absolute_additivity_error": max_additivity_error,
        "top_features": top_features,
        "feature_family_ranking": feature_family_ranking,
        "entity_ranking": entity_ranking,
        "local_case_types": {
            "found": [case.case_type for case in local_cases],
            "missing": missing_case_types,
        },
        "output_paths": {
            "global_importance_csv": str(paths.global_importance_csv),
            "family_importance_csv": str(paths.family_importance_csv),
            "entity_importance_csv": str(paths.entity_importance_csv),
            "local_contributions_csv": str(paths.local_contributions_csv),
            "sample_rows_parquet": str(paths.sample_rows_parquet),
            "shap_values_npz": str(paths.shap_values_npz),
            "report_txt": str(paths.report_txt),
            "plots_dir": str(paths.plots_dir),
        },
        "valid_id_sha256": valid_id_sha256,
        "holdout_used": False,
        "validation_passed": False,
    }

    validation_passed = final_validation(
        paths=paths,
        model=model,
        feature_columns=feature_columns,
        valid_id_sha256=valid_id_sha256,
        expected_valid_sha256=fixed_metadata["valid_id_sha256"],
        sample_indices=sample_indices,
        shap_sample_rows=shap_sample_rows,
        normalized_explanation=normalized_explanation,
        global_importance_df=global_importance_df,
        family_df=family_df,
        entity_df=entity_df,
        local_cases=local_cases,
        shap_booster=shap_booster,
        x_shap=x_shap,
    )
    metadata_payload["validation_passed"] = validation_passed

    paths.metadata_json.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not paths.metadata_json.exists():
        raise FileNotFoundError(f"缺少输出文件：{paths.metadata_json}")

    top10 = global_importance_df.head(10)
    top_family = family_df.iloc[0]
    top_entity = entity_df.iloc[0]

    print("\n" + "=" * 72)
    print("第 33 步完成摘要")
    print("=" * 72)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"SHAP 版本：{shap.__version__}")
    print(f"LightGBM 版本：{lgb.__version__}")
    print(f"模型路径：{model_path}")
    print(f"最佳 trial 编号：{optuna_metadata['best_trial_number']}")
    print(f"best_iteration：{best_iteration if best_iteration > 0 else '全部迭代'}")
    print(f"完整 valid 行数：{FULL_VALID_ROWS:,}")
    print(f"SHAP 抽样行数：{shap_sample_rows:,}")
    print(f"SHAP 样本 CTR：{shap_sample_ctr:.6f}")
    print(f"特征数量：{feature_count}")
    print(f"原始 SHAP 形状：{original_shape}")
    print(f"规范化 SHAP 形状：{normalized_shape}")
    print(
        f"加和一致性误差：mean={mean_additivity_error:.6e}, "
        f"max={max_additivity_error:.6e}"
    )
    print("\n前 10 个重要特征：")
    for row in top10.itertuples(index=False):
        print(
            f"  {row.feature}: mean_abs_shap={row.mean_abs_shap:.6f}, "
            f"importance_percent={row.importance_percent:.4f}"
        )
    print(
        f"\n最重要特征家族：{top_family.feature_family} "
        f"({top_family.group_importance_percent:.4f})"
    )
    print(
        f"最重要业务实体：{top_entity.entity} "
        f"({top_entity.entity_importance_percent:.4f})"
    )
    print(
        f"已生成的局部案例：{[case.case_type for case in local_cases]}"
    )
    if missing_case_types:
        print(f"缺失的局部案例：{missing_case_types}")
    print("\n输出路径：")
    print(f"  全局重要性：{paths.global_importance_csv}")
    print(f"  家族重要性：{paths.family_importance_csv}")
    print(f"  实体重要性：{paths.entity_importance_csv}")
    print(f"  局部贡献：{paths.local_contributions_csv}")
    print(f"  SHAP 样本：{paths.sample_rows_parquet}")
    print(f"  SHAP 数值：{paths.shap_values_npz}")
    print(f"  报告：{paths.report_txt}")
    print(f"  元数据：{paths.metadata_json}")
    print(f"  图表目录：{paths.plots_dir}")
    print(f"holdout_used：False")
    print(f"validation_passed：{validation_passed}")
    print("调优 LightGBM SHAP 特征解释完成，holdout 尚未使用。")


if __name__ == "__main__":
    main()
