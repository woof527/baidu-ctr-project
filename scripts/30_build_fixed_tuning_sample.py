"""
百度 CTR 项目 — 固定 LightGBM 调优样本构建

功能：
    从 target_encoded 特征中按与第 27 步完全一致的规则抽样，
    构建固定、可复现的 train / valid Parquet 数据集，供后续 Optuna 调优使用。
    禁止读取 holdout。

数据输入：
    data/features/target_encoded/train/*.parquet
    data/features/target_encoded/valid/*.parquet

用法：
    python scripts/30_build_fixed_tuning_sample.py
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 配置（与 scripts/27_train_lightgbm_baseline.py 正式模式对齐）
# ---------------------------------------------------------------------------

TRAIN_INPUT_DIR = Path("data/features/target_encoded/train")
VALID_INPUT_DIR = Path("data/features/target_encoded/valid")

TRAIN_OUTPUT_DIR = Path("data/tuning/lightgbm_train")
VALID_OUTPUT_DIR = Path("data/tuning/lightgbm_valid")

TEST_TRAIN_OUTPUT_DIR = Path("data/tuning_test/lightgbm_train")
TEST_VALID_OUTPUT_DIR = Path("data/tuning_test/lightgbm_valid")

TRAIN_TARGET_ROWS = 2_000_000
VALID_TARGET_ROWS = 500_000

TEST_TRAIN_TARGET_ROWS = 100_000
TEST_VALID_TARGET_ROWS = 50_000

RANDOM_STATE = 42
READ_BATCH_SIZE = 200_000
OUTPUT_PART_ROWS = 250_000

TEST_MODE = False

SOURCE_SCRIPT_REFERENCE = "scripts/27_train_lightgbm_baseline.py"
EXPECTED_FEATURE_COUNT = 33

# 与第 27 步完全一致的特征定义
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
    """特征配置（与第 27 步一致）。"""

    raw_feature_columns: list[str]
    log1p_columns: list[str]
    dynamic_feature_columns: list[str]
    feature_columns: list[str]
    use_hour_cyclical: bool


@dataclass
class OutputPaths:
    """输出路径配置。"""

    test_mode: bool
    train_output_dir: Path
    valid_output_dir: Path
    summary_csv: Path
    daily_distribution_csv: Path
    source_distribution_csv: Path
    report_txt: Path
    metadata_json: Path


@dataclass
class SplitSampleResult:
    """单个 split 的抽样与写出结果。"""

    split_name: str
    target_rows: int
    actual_rows: int
    input_file_count: int
    output_file_count: int
    date_min: str | None
    date_max: str | None
    negative_rows: int
    positive_rows: int
    ctr: float
    id_sha256: str
    nan_count: int
    inf_count: int
    id_missing_count: int
    click_missing_count: int
    source_records: list[dict] = field(default_factory=list)
    output_part_records: list[dict] = field(default_factory=list)
    daily_records: list[dict] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)


def get_output_paths(test_mode: bool) -> OutputPaths:
    """根据运行模式返回输出路径。"""

    if test_mode:
        return OutputPaths(
            test_mode=True,
            train_output_dir=TEST_TRAIN_OUTPUT_DIR,
            valid_output_dir=TEST_VALID_OUTPUT_DIR,
            summary_csv=Path("outputs/fixed_tuning_sample_test_summary.csv"),
            daily_distribution_csv=Path(
                "outputs/fixed_tuning_sample_test_daily_distribution.csv"
            ),
            source_distribution_csv=Path(
                "outputs/fixed_tuning_sample_test_source_distribution.csv"
            ),
            report_txt=Path("outputs/fixed_tuning_sample_test_report.txt"),
            metadata_json=Path("outputs/fixed_tuning_sample_test_metadata.json"),
        )

    return OutputPaths(
        test_mode=False,
        train_output_dir=TRAIN_OUTPUT_DIR,
        valid_output_dir=VALID_OUTPUT_DIR,
        summary_csv=Path("outputs/fixed_tuning_sample_summary.csv"),
        daily_distribution_csv=Path("outputs/fixed_tuning_sample_daily_distribution.csv"),
        source_distribution_csv=Path("outputs/fixed_tuning_sample_source_distribution.csv"),
        report_txt=Path("outputs/fixed_tuning_sample_report.txt"),
        metadata_json=Path("outputs/fixed_tuning_sample_metadata.json"),
    )


def get_target_rows(test_mode: bool) -> tuple[int, int]:
    """返回 train / valid 目标行数。"""

    if test_mode:
        return TEST_TRAIN_TARGET_ROWS, TEST_VALID_TARGET_ROWS
    return TRAIN_TARGET_ROWS, VALID_TARGET_ROWS


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    """列出并稳定排序 Parquet 分块。"""

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


def inspect_parquet_rows(parquet_files: list[Path]) -> list[int]:
    """用 metadata 统计每个文件行数。"""

    return [pq.read_metadata(path).num_rows for path in parquet_files]


def read_schema_columns(parquet_path: Path) -> list[str]:
    """读取 schema 列名。"""

    return pq.read_schema(parquet_path).names


def discover_feature_config(schema_columns: list[str]) -> FeatureConfig:
    """从 schema 自动确定特征列表（与第 27 步完全一致）。"""

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

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"特征数量 {len(feature_columns)} 与第 27 步预期 "
            f"{EXPECTED_FEATURE_COUNT} 不一致。"
        )

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
    """抽样 batch 需要读取的列（含 id 与 hour_dt）。"""

    columns = ["click", *feature_config.raw_feature_columns]

    if feature_config.use_hour_cyclical:
        columns.append("hour_of_day")

    if "hour_dt" in schema_columns:
        columns.append("hour_dt")
    elif "hour" in schema_columns:
        columns.append("hour")
    elif "date" in schema_columns:
        columns.append("date")

    if "id" in schema_columns:
        columns.append("id")

    return list(dict.fromkeys(columns))


def extract_event_date(dataframe: pd.DataFrame) -> pd.Series:
    """从 hour_dt、date 或 hour 提取归一化日期（与第 27 步逻辑兼容）。"""

    if "hour_dt" in dataframe.columns:
        hour_dt = pd.to_datetime(dataframe["hour_dt"], errors="coerce")
        if hour_dt.notna().any():
            return hour_dt.dt.normalize()

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

    raise ValueError("无法从 hour_dt / date / hour 字段解析 split_date。")


def extract_split_date_strings(dataframe: pd.DataFrame) -> pd.Series:
    """提取 YYYY-MM-DD 格式的 split_date（不含 NaT）。"""

    event_dates = extract_event_date(dataframe)
    return event_dates.dt.strftime("%Y-%m-%d")


def validate_click_batch(
    dataframe: pd.DataFrame,
    split_name: str,
    source_file: str,
    batch_index: int,
) -> pd.Series:
    """标准化并严格验证 click，返回无缺失的 int8 Series。"""

    total_rows = len(dataframe)
    click_numeric = pd.to_numeric(dataframe["click"], errors="coerce")
    missing_mask = click_numeric.isna()
    missing_count = int(missing_mask.sum())

    valid_mask = ~missing_mask
    invalid_mask = valid_mask & ~click_numeric.isin([0, 1])
    invalid_count = int(invalid_mask.sum())

    if missing_count > 0 or invalid_count > 0:
        anomaly_mask = missing_mask | invalid_mask
        anomaly_examples = [
            {"id": str(record_id), "click": click_raw}
            for record_id, click_raw in zip(
                dataframe.loc[anomaly_mask, "id"].head(10),
                dataframe.loc[anomaly_mask, "click"].head(10),
            )
        ]
        raise ValueError(
            "click 数据异常："
            f"split={split_name}, source_file={source_file}, batch_index={batch_index}, "
            f"total_rows={total_rows}, missing_click_count={missing_count}, "
            f"invalid_click_count={invalid_count}, examples={anomaly_examples}"
        )

    return click_numeric.astype(np.int8)


def validate_split_date_batch(
    dataframe: pd.DataFrame,
    split_name: str,
    source_file: str,
    batch_index: int,
) -> pd.Series:
    """提取并严格验证 split_date，缺失日期立即报错。"""

    event_dates = extract_event_date(dataframe)
    missing_mask = event_dates.isna()
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        anomaly_ids = dataframe.loc[missing_mask, "id"].head(10).astype(str).tolist()
        raise ValueError(
            "split_date 无法解析："
            f"split={split_name}, source_file={source_file}, batch_index={batch_index}, "
            f"missing_date_count={missing_count}, example_ids={anomaly_ids}"
        )

    return event_dates.dt.strftime("%Y-%m-%d")


def validate_prepared_batch(
    prepared_df: pd.DataFrame,
    expected_rows: int,
    context: str,
    feature_config: FeatureConfig,
) -> None:
    """验证预处理 batch 的元数据与特征行对齐。"""

    if len(prepared_df) != expected_rows:
        raise ValueError(
            f"{context} 预处理后行数 {len(prepared_df):,} 与期望 {expected_rows:,} 不一致。"
        )

    if prepared_df["id"].isna().any():
        raise ValueError(f"{context} 存在缺失 id。")

    if prepared_df["click"].isna().any():
        raise ValueError(f"{context} 存在缺失 click（可能由索引错位导致）。")

    unique_clicks = set(prepared_df["click"].unique().tolist())
    if not unique_clicks.issubset({0, 1}):
        raise ValueError(f"{context} 的 click 存在非法取值：{sorted(unique_clicks)}")

    if prepared_df["split_date"].isna().any():
        raise ValueError(f"{context} 存在缺失 split_date。")

    if prepared_df["split_date"].astype(str).eq("NaT").any():
        raise ValueError(f"{context} 存在无法解析的 split_date（NaT）。")

    if list(prepared_df.columns[:3]) != ["id", "click", "split_date"]:
        raise ValueError(f"{context} 元数据列顺序不正确。")

    if list(prepared_df.columns[3:]) != feature_config.feature_columns:
        raise ValueError(f"{context} 特征列名称或顺序不正确。")


def validate_click_values(labels: np.ndarray, context: str) -> None:
    """检查 click 是否仅包含 0 和 1（兼容 numpy 数组）。"""

    if np.issubdtype(labels.dtype, np.floating) and np.isnan(labels).any():
        raise ValueError(f"{context} 的 click 存在 NaN。")

    unique_values = set(np.unique(labels).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{context} 的 click 存在非法取值：{sorted(unique_values)}")


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_config: FeatureConfig,
) -> np.ndarray:
    """构造特征矩阵：log1p → hour sin/cos → 清洗 → float32（与第 27 步一致）。"""

    matrix = dataframe[feature_config.raw_feature_columns].copy()

    for column_name in feature_config.log1p_columns:
        numeric_values = pd.to_numeric(matrix[column_name], errors="coerce")
        matrix[column_name] = np.log1p(numeric_values.astype(np.float64))

    if feature_config.use_hour_cyclical:
        hour_values = pd.to_numeric(
            dataframe["hour_of_day"],
            errors="coerce",
        ).astype(np.float64).to_numpy()
        radians = 2.0 * np.pi * hour_values / 24.0
        matrix["hour_sin"] = np.sin(radians)
        matrix["hour_cos"] = np.cos(radians)

    for column_name in feature_config.feature_columns:
        matrix[column_name] = pd.to_numeric(matrix[column_name], errors="coerce")

    feature_array = matrix[feature_config.feature_columns].to_numpy(dtype=np.float64)
    feature_array[~np.isfinite(feature_array)] = np.nan
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_array.astype(np.float32)


def prepare_feature_batch(
    dataframe: pd.DataFrame,
    feature_config: FeatureConfig,
    split_name: str,
    source_file: str,
    batch_index: int,
) -> pd.DataFrame:
    """将 batch 转为固定输出列顺序的 DataFrame（保证行对齐）。"""

    if "id" not in dataframe.columns:
        raise ValueError("输入 batch 缺少 id 字段，无法构建固定调优样本。")

    dataframe = dataframe.reset_index(drop=True)
    expected_rows = len(dataframe)

    click_series = validate_click_batch(
        dataframe,
        split_name=split_name,
        source_file=source_file,
        batch_index=batch_index,
    )
    split_date_series = validate_split_date_batch(
        dataframe,
        split_name=split_name,
        source_file=source_file,
        batch_index=batch_index,
    )

    feature_array = build_feature_matrix(dataframe, feature_config)
    feature_df = pd.DataFrame(
        feature_array,
        columns=feature_config.feature_columns,
    ).astype(np.float32).reset_index(drop=True)

    output_df = pd.DataFrame(
        {
            "id": dataframe["id"].astype(str).reset_index(drop=True),
            "click": click_series.reset_index(drop=True),
            "split_date": split_date_series.reset_index(drop=True),
        }
    ).reset_index(drop=True)

    if len(output_df) != expected_rows or len(feature_df) != expected_rows:
        raise ValueError(
            f"{split_name}/{source_file} batch {batch_index} 合并前行数不一致："
            f"input={expected_rows}, metadata={len(output_df)}, features={len(feature_df)}"
        )

    prepared_df = pd.concat([output_df, feature_df], axis=1)
    prepared_df = prepared_df.reset_index(drop=True)

    context = f"{split_name}/{source_file} batch {batch_index}"
    validate_prepared_batch(
        prepared_df,
        expected_rows=expected_rows,
        context=context,
        feature_config=feature_config,
    )

    return prepared_df


def update_batch_statistics(
    prepared_df: pd.DataFrame,
    date_counter: dict[str, int],
    date_click_counter: dict[str, int],
    click_counter: dict[int, int],
) -> None:
    """使用 groupby 更新日期与 click 统计（要求 prepared_df 已通过严格验证）。"""

    daily_stats = prepared_df.groupby("split_date", dropna=False)["click"].agg(
        rows="size",
        clicks="sum",
    )

    for split_date, stats in daily_stats.iterrows():
        date_key = str(split_date)
        date_counter[date_key] = date_counter.get(date_key, 0) + int(stats["rows"])
        date_click_counter[date_key] = date_click_counter.get(date_key, 0) + int(
            stats["clicks"]
        )

    for click_value, row_count in prepared_df["click"].value_counts().items():
        click_counter[int(click_value)] = click_counter.get(int(click_value), 0) + int(
            row_count
        )


def calculate_file_quotas(total_target: int, file_row_counts: list[int]) -> list[int]:
    """按文件行数比例分配抽样额度（与第 27 步一致）。"""

    total_rows = sum(file_row_counts)
    if total_target > total_rows:
        raise ValueError(
            f"目标抽样行数 {total_target:,} 超过可用总行数 {total_rows:,}"
        )

    raw_quotas = [total_target * count / total_rows for count in file_row_counts]
    quotas = [int(np.floor(value)) for value in raw_quotas]
    remaining = total_target - sum(quotas)

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


def clean_output_parquet_files(output_dir: Path) -> None:
    """仅清理目标目录中的旧 Parquet 文件。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in output_dir.glob("*.parquet"):
        parquet_path.unlink()


def write_parquet_parts(
    dataframe: pd.DataFrame,
    output_dir: Path,
    part_start_index: int,
    feature_config: FeatureConfig,
) -> tuple[list[Path], list[dict], int]:
    """将 DataFrame 按 OUTPUT_PART_ROWS 分块写出。"""

    column_order = ["id", "click", "split_date", *feature_config.feature_columns]
    dataframe = dataframe[column_order]

    output_files: list[Path] = []
    part_records: list[dict] = []
    part_index = part_start_index

    for start in range(0, len(dataframe), OUTPUT_PART_ROWS):
        part_df = dataframe.iloc[start : start + OUTPUT_PART_ROWS].copy()
        part_path = output_dir / f"part-{part_index:04d}.parquet"
        part_df.to_parquet(part_path, index=False)
        output_files.append(part_path)
        part_records.append({"file_name": part_path.name, "rows": len(part_df)})
        part_index += 1

    return output_files, part_records, part_index


def sample_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    target_rows: int,
    feature_config: FeatureConfig,
    read_columns: list[str],
) -> SplitSampleResult:
    """从某个 split 抽样、预处理并写出固定 Parquet。"""

    parquet_files = get_sorted_parquet_files(input_dir)
    file_row_counts = inspect_parquet_rows(parquet_files)
    file_quotas = calculate_file_quotas(target_rows, file_row_counts)

    clean_output_parquet_files(output_dir)

    buffer_frames: list[pd.DataFrame] = []
    buffer_row_count = 0
    part_start_index = 0
    all_output_files: list[Path] = []
    all_part_records: list[dict] = []
    source_records: list[dict] = []

    total_collected = 0
    date_counter: dict[str, int] = {}
    date_click_counter: dict[str, int] = {}
    click_counter = {0: 0, 1: 0}

    print(f"\n开始 {split_name} 抽样，目标 {target_rows:,} 行 ...")

    for file_index, (parquet_path, file_quota) in enumerate(
        zip(parquet_files, file_quotas)
    ):
        if file_quota <= 0:
            continue

        file_collected = 0

        print(
            f"[{split_name}] 文件 {file_index + 1}/{len(parquet_files)}: "
            f"{parquet_path.name}，目标抽样 {file_quota:,} 行"
        )

        for batch_index, batch_df in enumerate(
            iter_file_batches(parquet_path, read_columns, READ_BATCH_SIZE),
            start=1,
        ):
            if file_collected >= file_quota:
                break

            batch_df = batch_df.reset_index(drop=True)
            validate_click_batch(
                batch_df,
                split_name=split_name,
                source_file=parquet_path.name,
                batch_index=batch_index,
            )

            remaining = file_quota - file_collected
            if len(batch_df) <= remaining:
                sampled_df = batch_df
            else:
                sampled_df = batch_df.sample(
                    n=remaining,
                    random_state=RANDOM_STATE,
                )

            sampled_df = sampled_df.reset_index(drop=True)
            validate_click_batch(
                sampled_df,
                split_name=split_name,
                source_file=parquet_path.name,
                batch_index=batch_index,
            )

            prepared_df = prepare_feature_batch(
                sampled_df,
                feature_config,
                split_name=split_name,
                source_file=parquet_path.name,
                batch_index=batch_index,
            )
            buffer_frames.append(prepared_df)
            buffer_row_count += len(prepared_df)

            update_batch_statistics(
                prepared_df,
                date_counter,
                date_click_counter,
                click_counter,
            )

            file_collected += len(sampled_df)
            total_collected += len(sampled_df)

            print(
                f"  batch {batch_index}: 抽取 {len(sampled_df):,} 行，"
                f"文件累计 {file_collected:,}/{file_quota:,}，"
                f"split 累计 {total_collected:,}"
            )

            if buffer_row_count >= OUTPUT_PART_ROWS:
                buffer_df = pd.concat(buffer_frames, ignore_index=True)
                write_df = buffer_df.iloc[:OUTPUT_PART_ROWS]
                remainder_df = buffer_df.iloc[OUTPUT_PART_ROWS:]

                part_files, part_records, part_start_index = write_parquet_parts(
                    write_df,
                    output_dir,
                    part_start_index,
                    feature_config,
                )
                all_output_files.extend(part_files)
                all_part_records.extend(part_records)

                buffer_frames = [remainder_df] if len(remainder_df) > 0 else []
                buffer_row_count = len(remainder_df)

            del batch_df, sampled_df, prepared_df
            gc.collect()

        source_records.append(
            {
                "split": split_name,
                "source_file": parquet_path.name,
                "source_rows": pq.read_metadata(parquet_path).num_rows,
                "sampled_rows": file_collected,
                "sampling_ratio": file_collected / pq.read_metadata(parquet_path).num_rows,
            }
        )

    if total_collected == 0:
        raise ValueError(f"{split_name} 未抽到任何样本。")

    if buffer_row_count > 0:
        buffer_df = pd.concat(buffer_frames, ignore_index=True)
        part_files, part_records, part_start_index = write_parquet_parts(
            buffer_df,
            output_dir,
            part_start_index,
            feature_config,
        )
        all_output_files.extend(part_files)
        all_part_records.extend(part_records)

    del buffer_frames
    gc.collect()

    if total_collected != target_rows:
        raise ValueError(
            f"{split_name} 实际输出行数 {total_collected:,} 与目标 {target_rows:,} 不一致。"
        )

    id_sha256 = calculate_id_sha256(all_output_files)
    validation_stats = validate_output_split(
        split_name=split_name,
        output_files=all_output_files,
        feature_config=feature_config,
        expected_rows=target_rows,
        id_sha256=id_sha256,
    )

    unique_dates = sorted(date_counter.keys())
    date_min = unique_dates[0] if unique_dates else None
    date_max = unique_dates[-1] if unique_dates else None

    daily_records = build_daily_distribution(
        split_name=split_name,
        date_counter=date_counter,
        date_click_counter=date_click_counter,
        total_rows=total_collected,
    )

    return SplitSampleResult(
        split_name=split_name,
        target_rows=target_rows,
        actual_rows=total_collected,
        input_file_count=len(parquet_files),
        output_file_count=len(all_output_files),
        date_min=date_min,
        date_max=date_max,
        negative_rows=click_counter.get(0, 0),
        positive_rows=click_counter.get(1, 0),
        ctr=click_counter.get(1, 0) / total_collected,
        id_sha256=id_sha256,
        nan_count=validation_stats["nan_count"],
        inf_count=validation_stats["inf_count"],
        id_missing_count=validation_stats["id_missing_count"],
        click_missing_count=validation_stats["click_missing_count"],
        source_records=source_records,
        output_part_records=all_part_records,
        daily_records=daily_records,
        output_files=all_output_files,
    )


def calculate_id_sha256(output_files: list[Path]) -> str:
    """按输出文件顺序流式计算 id 的 SHA256 指纹。"""

    hasher = hashlib.sha256()

    for parquet_path in output_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(columns=["id"], batch_size=READ_BATCH_SIZE):
            id_series = record_batch.to_pandas()["id"]
            for id_value in id_series:
                hasher.update((str(id_value) + "\n").encode("utf-8"))

    digest = hasher.hexdigest()
    if len(digest) != 64:
        raise ValueError(f"SHA256 长度异常：{len(digest)}")

    return digest


def validate_output_split(
    split_name: str,
    output_files: list[Path],
    feature_config: FeatureConfig,
    expected_rows: int,
    id_sha256: str,
) -> dict[str, int]:
    """验收单个 split 的输出文件。"""

    if not output_files:
        raise ValueError(f"{split_name} 未生成任何输出文件。")

    expected_names = [f"part-{index:04d}.parquet" for index in range(len(output_files))]
    actual_names = [path.name for path in output_files]
    if actual_names != expected_names:
        raise ValueError(
            f"{split_name} 输出文件名不连续：期望 {expected_names}，实际 {actual_names}"
        )

    total_rows = 0
    nan_count = 0
    inf_count = 0
    id_missing_count = 0
    click_missing_count = 0

    forbidden_in_features = {"id", "click"} & set(feature_config.feature_columns)
    if forbidden_in_features:
        raise ValueError(
            f"feature_columns 包含禁止字段：{sorted(forbidden_in_features)}"
        )

    for parquet_path in output_files:
        table = pq.read_table(parquet_path)
        dataframe = table.to_pandas()

        total_rows += len(dataframe)

        if list(dataframe.columns[:3]) != ["id", "click", "split_date"]:
            raise ValueError(f"{parquet_path} 列顺序不符合 id / click / split_date 要求。")

        if list(dataframe.columns[3:]) != feature_config.feature_columns:
            raise ValueError(f"{parquet_path} 特征列名称或顺序与第 27 步不一致。")

        if dataframe["id"].isna().any():
            id_missing_count += int(dataframe["id"].isna().sum())

        if dataframe["click"].isna().any():
            click_missing_count += int(dataframe["click"].isna().sum())

        unique_clicks = set(dataframe["click"].dropna().unique().tolist())
        if not unique_clicks.issubset({0, 1}):
            raise ValueError(f"{parquet_path} 的 click 存在非法取值：{sorted(unique_clicks)}")

        feature_block = dataframe[feature_config.feature_columns]
        if not all(pd.api.types.is_numeric_dtype(feature_block[column]) for column in feature_block.columns):
            raise ValueError(f"{parquet_path} 存在非数值型模型特征。")

        if not all(feature_block[column].dtype == np.float32 for column in feature_block.columns):
            raise ValueError(f"{parquet_path} 模型特征 dtype 不是 float32。")

        nan_count += int(feature_block.isna().sum().sum())
        inf_count += int(np.isinf(feature_block.to_numpy()).sum())

    if total_rows != expected_rows:
        raise ValueError(
            f"{split_name} 重新读取行数 {total_rows:,} 与期望 {expected_rows:,} 不一致。"
        )

    if len(id_sha256) != 64:
        raise ValueError(f"{split_name} id_sha256 长度不是 64。")

    if nan_count > 0:
        raise ValueError(f"{split_name} 模型特征存在 NaN：{nan_count}")

    if inf_count > 0:
        raise ValueError(f"{split_name} 模型特征存在 inf：{inf_count}")

    if id_missing_count > 0:
        raise ValueError(f"{split_name} 存在缺失 id：{id_missing_count}")

    if click_missing_count > 0:
        raise ValueError(f"{split_name} 存在缺失 click：{click_missing_count}")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "id_missing_count": id_missing_count,
        "click_missing_count": click_missing_count,
    }


def build_daily_distribution(
    split_name: str,
    date_counter: dict[str, int],
    date_click_counter: dict[str, int],
    total_rows: int,
) -> list[dict]:
    """构建按日分布表。"""

    records: list[dict] = []

    for date_value in sorted(date_counter.keys()):
        rows = date_counter[date_value]
        clicks = date_click_counter.get(date_value, 0)
        ctr = clicks / rows if rows > 0 else 0.0
        records.append(
            {
                "split": split_name,
                "date": date_value,
                "rows": rows,
                "clicks": clicks,
                "ctr": ctr,
                "percentage_in_split": rows / total_rows if total_rows else 0.0,
            }
        )

    return records


def build_source_distribution(source_records: list[dict]) -> pd.DataFrame:
    """构建源文件抽样分布表。"""

    return pd.DataFrame(source_records)


def build_summary_dataframe(
    train_result: SplitSampleResult,
    valid_result: SplitSampleResult,
) -> pd.DataFrame:
    """构建 split 级汇总表。"""

    rows = []
    for result in (train_result, valid_result):
        rows.append(
            {
                "split": result.split_name,
                "target_rows": result.target_rows,
                "actual_rows": result.actual_rows,
                "input_file_count": result.input_file_count,
                "output_file_count": result.output_file_count,
                "date_min": result.date_min,
                "date_max": result.date_max,
                "negative_rows": result.negative_rows,
                "positive_rows": result.positive_rows,
                "ctr": result.ctr,
                "feature_count": EXPECTED_FEATURE_COUNT,
                "id_sha256": result.id_sha256,
                "nan_count": result.nan_count,
                "inf_count": result.inf_count,
                "validation_status": "passed",
            }
        )

    return pd.DataFrame(rows)


def validate_cross_split(
    train_result: SplitSampleResult,
    valid_result: SplitSampleResult,
    feature_config: FeatureConfig,
) -> None:
    """验收 train / valid 之间的约束。"""

    if train_result.date_max is None or valid_result.date_min is None:
        raise ValueError("无法确定 train / valid 日期范围。")

    if train_result.date_max >= valid_result.date_min:
        raise ValueError(
            f"train 最大日期 {train_result.date_max} 必须早于 "
            f"valid 最小日期 {valid_result.date_min}。"
        )

    train_dates = {record["date"] for record in train_result.daily_records}
    valid_dates = {record["date"] for record in valid_result.daily_records}
    overlap = train_dates & valid_dates
    if overlap:
        raise ValueError(f"train 与 valid 存在日期重叠：{sorted(overlap)}")

    if len(feature_config.feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"特征数量 {len(feature_config.feature_columns)} 与第 27 步不一致。"
        )


def write_text_report(
    report_path: Path,
    paths: OutputPaths,
    feature_config: FeatureConfig,
    train_result: SplitSampleResult,
    valid_result: SplitSampleResult,
    train_target: int,
    valid_target: int,
) -> None:
    """写入中文文本报告。"""

    mode_label = "TEST_MODE=True" if paths.test_mode else "TEST_MODE=False"
    lines = [
        "百度 CTR 项目 — 固定 LightGBM 调优样本报告",
        "=" * 70,
        "",
        f"当前模式：{mode_label}",
        "",
        "【脚本目的】",
        "  构建固定、可复现的 LightGBM 调优 train / valid 样本，",
        "  供基线复跑、Optuna 每个 trial 和调优模型比较统一使用。",
        "",
        "【数据来源】",
        f"  train：{TRAIN_INPUT_DIR}",
        f"  valid：{VALID_INPUT_DIR}",
        "",
        "【固定样本规模】",
        f"  train 目标行数：{train_target:,}，实际：{train_result.actual_rows:,}",
        f"  valid 目标行数：{valid_target:,}，实际：{valid_result.actual_rows:,}",
        "",
        "【为何保存固定样本】",
        "  避免不同实验因重新随机抽样导致结果不可比。",
        "",
        "【随机种子】",
        f"  RANDOM_STATE = {RANDOM_STATE}",
        "",
        "【特征数量】",
        f"  {len(feature_config.feature_columns)} 个（与 {SOURCE_SCRIPT_REFERENCE} 一致）",
        "",
        "【特征处理规则】",
        "  - 对 _freq / _hist_impressions / _hist_clicks 执行 log1p",
        "  - 由 hour_of_day 生成 hour_sin / hour_cos",
        "  - NaN / inf 按第 27 步规则置 0",
        "  - 模型特征保存为 float32",
        "",
        "【日期范围】",
        f"  train：{train_result.date_min} ~ {train_result.date_max}",
        f"  valid：{valid_result.date_min} ~ {valid_result.date_max}",
        "",
        "【CTR 与标签分布】",
        f"  train CTR：{train_result.ctr:.6f} "
        f"(0={train_result.negative_rows:,}, 1={train_result.positive_rows:,})",
        f"  valid CTR：{valid_result.ctr:.6f} "
        f"(0={valid_result.negative_rows:,}, 1={valid_result.positive_rows:,})",
        "",
        "【SHA256 指纹】",
        f"  train_id_sha256：{train_result.id_sha256}",
        f"  valid_id_sha256：{valid_result.id_sha256}",
        "",
        "【验收结果】",
        "  - train / valid 行数符合目标",
        "  - click 仅含 0 / 1，且无缺失",
        "  - id 无缺失",
        "  - 模型特征无 NaN / inf，dtype=float32",
        "  - 特征名称与顺序与第 27 步一致",
        "  - train 最大日期早于 valid 最小日期，无日期重叠",
        "  - 输出文件 part-0000 起连续编号",
        "  - SHA256 长度均为 64",
        "",
        "【说明】",
        "  - 未读取 holdout",
        "  - 后续所有 Optuna trial 必须使用本次固定样本",
        "  - 旧 LightGBM 基线与新固定样本调优结果不能直接视为完全相同抽样口径，",
        "    因此下一步要在固定样本上重新跑一次基线参数",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_reports(
    paths: OutputPaths,
    feature_config: FeatureConfig,
    train_result: SplitSampleResult,
    valid_result: SplitSampleResult,
    train_target: int,
    valid_target: int,
) -> None:
    """保存 CSV、JSON 与文本报告。"""

    summary_df = build_summary_dataframe(train_result, valid_result)
    daily_df = pd.DataFrame(train_result.daily_records + valid_result.daily_records)
    source_df = build_source_distribution(
        train_result.source_records + valid_result.source_records
    )

    paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(paths.summary_csv, index=False, encoding="utf-8")
    daily_df.to_csv(paths.daily_distribution_csv, index=False, encoding="utf-8")
    source_df.to_csv(paths.source_distribution_csv, index=False, encoding="utf-8")

    metadata = {
        "script_name": "scripts/30_build_fixed_tuning_sample.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": paths.test_mode,
        "random_state": RANDOM_STATE,
        "read_batch_size": READ_BATCH_SIZE,
        "output_part_rows": OUTPUT_PART_ROWS,
        "train_target_rows": train_target,
        "valid_target_rows": valid_target,
        "actual_train_rows": train_result.actual_rows,
        "actual_valid_rows": valid_result.actual_rows,
        "train_id_sha256": train_result.id_sha256,
        "valid_id_sha256": valid_result.id_sha256,
        "feature_columns": feature_config.feature_columns,
        "feature_count": len(feature_config.feature_columns),
        "log1p_columns": feature_config.log1p_columns,
        "train_date_min": train_result.date_min,
        "train_date_max": train_result.date_max,
        "valid_date_min": valid_result.date_min,
        "valid_date_max": valid_result.date_max,
        "train_ctr": train_result.ctr,
        "valid_ctr": valid_result.ctr,
        "train_input_directory": str(TRAIN_INPUT_DIR),
        "valid_input_directory": str(VALID_INPUT_DIR),
        "train_output_directory": str(paths.train_output_dir),
        "valid_output_directory": str(paths.valid_output_dir),
        "source_script_reference": SOURCE_SCRIPT_REFERENCE,
        "holdout_used": False,
        "validation_passed": True,
    }

    paths.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_text_report(
        paths.report_txt,
        paths,
        feature_config,
        train_result,
        valid_result,
        train_target,
        valid_target,
    )


def main() -> None:
    """主流程：抽样 → 预处理 → 分块写出 → 验收 → 报告。"""

    paths = get_output_paths(TEST_MODE)
    train_target, valid_target = get_target_rows(TEST_MODE)

    print("=" * 70)
    print("固定 LightGBM 调优样本构建")
    print("=" * 70)
    print(f"TEST_MODE：{TEST_MODE}")
    print(f"train 目标行数：{train_target:,}")
    print(f"valid 目标行数：{valid_target:,}")

    train_files = get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)

    schema_columns = read_schema_columns(train_files[0])
    feature_config = discover_feature_config(schema_columns)
    read_columns = get_read_columns(feature_config, schema_columns)

    if "id" not in schema_columns:
        raise ValueError("train schema 缺少 id 字段，无法构建固定调优样本。")

    valid_schema = read_schema_columns(valid_files[0])
    missing_valid = [column for column in read_columns if column not in valid_schema]
    if missing_valid:
        raise ValueError(f"valid schema 缺少字段：{missing_valid}")

    print("\n特征配置：")
    print(f"  特征数量：{len(feature_config.feature_columns)}")
    print(f"  log1p 列数：{len(feature_config.log1p_columns)}")

    train_result = sample_split(
        split_name="train",
        input_dir=TRAIN_INPUT_DIR,
        output_dir=paths.train_output_dir,
        target_rows=train_target,
        feature_config=feature_config,
        read_columns=read_columns,
    )

    valid_result = sample_split(
        split_name="valid",
        input_dir=VALID_INPUT_DIR,
        output_dir=paths.valid_output_dir,
        target_rows=valid_target,
        feature_config=feature_config,
        read_columns=read_columns,
    )

    validate_cross_split(train_result, valid_result, feature_config)
    save_reports(
        paths,
        feature_config,
        train_result,
        valid_result,
        train_target,
        valid_target,
    )

    print("\n" + "=" * 70)
    print("构建完成")
    print("=" * 70)
    print(f"当前模式：{'TEST_MODE=True' if TEST_MODE else 'TEST_MODE=False'}")
    print(f"train 输入文件数量：{train_result.input_file_count}")
    print(f"valid 输入文件数量：{valid_result.input_file_count}")
    print(f"train 实际输出行数：{train_result.actual_rows:,}")
    print(f"valid 实际输出行数：{valid_result.actual_rows:,}")
    print(f"train 日期范围：{train_result.date_min} ~ {train_result.date_max}")
    print(f"valid 日期范围：{valid_result.date_min} ~ {valid_result.date_max}")
    print(f"train CTR：{train_result.ctr:.6f}")
    print(f"valid CTR：{valid_result.ctr:.6f}")
    print(f"特征数量：{len(feature_config.feature_columns)}")
    print(f"train_id_sha256：{train_result.id_sha256}")
    print(f"valid_id_sha256：{valid_result.id_sha256}")
    print("输出路径：")
    print(f"  train 样本：     {paths.train_output_dir}")
    print(f"  valid 样本：     {paths.valid_output_dir}")
    print(f"  汇总 CSV：       {paths.summary_csv}")
    print(f"  日分布 CSV：     {paths.daily_distribution_csv}")
    print(f"  源分布 CSV：     {paths.source_distribution_csv}")
    print(f"  文本报告：       {paths.report_txt}")
    print(f"  元数据 JSON：    {paths.metadata_json}")
    print("所有验收：通过")
    print("固定调优样本构建完成，holdout 尚未使用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
