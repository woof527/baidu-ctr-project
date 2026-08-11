"""
百度 CTR 项目 — 统一建模样本构建（第 37 步）

功能：
    从完整上游 target_encoded 数据（Step 24 输出）独立构建统一固定样本，
    同时保留原始 categorical 与已验证的工程化 numerical 特征，
    供后续 LightGBM / Wide & Deep / DeepFM 共用。
    不要求与旧 Step 30 固定样本行一致。

数据来源：
    data/features/target_encoded/train/  （2014-10-21 ~ 2014-10-28）
    data/features/target_encoded/valid/ （2014-10-29）

输出：
    data/modeling/unified_train/
    data/modeling/unified_valid/
    outputs/unified_modeling_sample_metadata.json
    outputs/unified_modeling_sample_audit.txt

严禁读取 holdout（2014-10-30）或 test.csv。

用法：
    python scripts/37_build_unified_modeling_sample.py
"""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

TRAIN_INPUT_DIR = Path("data/features/target_encoded/train")
VALID_INPUT_DIR = Path("data/features/target_encoded/valid")

TRAIN_OUTPUT_DIR = Path("data/modeling/unified_train")
VALID_OUTPUT_DIR = Path("data/modeling/unified_valid")

METADATA_PATH = Path("outputs/unified_modeling_sample_metadata.json")
AUDIT_PATH = Path("outputs/unified_modeling_sample_audit.txt")

TRAIN_TARGET_ROWS = 2_000_000
VALID_TARGET_ROWS = 500_000
TEST_TRAIN_TARGET_ROWS = 50_000
TEST_VALID_TARGET_ROWS = 10_000

RANDOM_SEED = 42
READ_BATCH_SIZE = 200_000
OUTPUT_PART_ROWS = 250_000

TRAIN_DATE_START = "2014-10-21"
TRAIN_DATE_END = "2014-10-28"
VALID_DATE = "2014-10-29"
FORBIDDEN_DATE = "2014-10-30"

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

HIGH_CARDINALITY_THRESHOLD = 10_000

REQUESTED_CATEGORICALS = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

OPTIONAL_CATEGORICALS = [
    "banner_device_cross",
    "hour_banner_cross",
    "site_device_cross",
]

AUXILIARY_COLUMNS = ["hour_of_day", "day_of_week"]

NUMERICAL_SUFFIXES = (
    "_freq",
    "_hist_impressions",
    "_hist_clicks",
    "_hist_ctr",
    "_exposure_percentile",
    "_te",
)
NUMERICAL_EXACT = {"is_weekend"}
COMPUTED_NUMERICAL = {"hour_sin", "hour_cos"}

# 上游 QA 列，不进入建模样本
EXCLUDE_COLUMNS = {
    "is_dup_id_within_chunk",
    "is_invalid_click",
}


@dataclass
class ColumnConfig:
    """列分组配置。"""

    auxiliary_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]
    output_column_order: list[str]
    read_columns: list[str]


@dataclass
class SplitBuildResult:
    """单个 split 构建结果。"""

    split_name: str
    target_rows: int
    actual_rows: int
    output_files: list[Path]
    date_min: str
    date_max: str
    ctr: float
    negative_rows: int
    positive_rows: int
    id_sha256: str
    checksum: str


@dataclass
class BuildState:
    """构建过程状态。"""

    lines: list[str] = field(default_factory=list)


def log(state: BuildState, message: str = "") -> None:
    state.lines.append(message)
    print(message)


def assert_safe_path(path: Path) -> None:
    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def get_target_rows(test_mode: bool) -> tuple[int, int]:
    if test_mode:
        return TEST_TRAIN_TARGET_ROWS, TEST_VALID_TARGET_ROWS
    return TRAIN_TARGET_ROWS, VALID_TARGET_ROWS


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    assert_safe_path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"目录不存在：{parquet_dir}")
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{parquet_dir}")
    return files


def read_schema_columns(parquet_path: Path) -> list[str]:
    return pq.read_schema(parquet_path).names


def discover_column_config(schema_columns: list[str]) -> ColumnConfig:
    """从上游 schema 确定 auxiliary / categorical / numerical 列。"""

    missing = [col for col in ("id", "click", "hour_dt", "hour_of_day") if col not in schema_columns]
    if missing:
        raise ValueError(f"上游 schema 缺少必需列：{missing}")

    auxiliary = ["hour_dt"]
    for column in AUXILIARY_COLUMNS:
        if column in schema_columns:
            auxiliary.append(column)

    categoricals: list[str] = []
    missing_cats: list[str] = []
    for column in REQUESTED_CATEGORICALS:
        if column in schema_columns:
            categoricals.append(column)
        else:
            missing_cats.append(column)
    if missing_cats:
        raise ValueError(f"上游缺少必需的 categorical 列：{missing_cats}")

    for column in OPTIONAL_CATEGORICALS:
        if column in schema_columns:
            categoricals.append(column)

    numerical: list[str] = []
    for column in schema_columns:
        if column in EXCLUDE_COLUMNS:
            continue
        if column in {"id", "click", "hour", "hour_dt", "day", "date"}:
            continue
        if column in categoricals or column in auxiliary:
            continue
        if column in NUMERICAL_EXACT:
            numerical.append(column)
            continue
        if any(column.endswith(suffix) for suffix in NUMERICAL_SUFFIXES):
            numerical.append(column)

    numerical = sorted(set(numerical))
    for computed in COMPUTED_NUMERICAL:
        if computed not in numerical:
            numerical.append(computed)
    numerical = sorted(numerical)

    output_order = [
        "id",
        "split_date",
        *auxiliary,
        *categoricals,
        *numerical,
        "click",
    ]
    read_columns = list(
        dict.fromkeys(
            [
                "id",
                "click",
                "hour_dt",
                *auxiliary,
                *categoricals,
                *[col for col in numerical if col not in COMPUTED_NUMERICAL],
            ]
        )
    )

    return ColumnConfig(
        auxiliary_columns=auxiliary,
        categorical_columns=categoricals,
        numerical_columns=numerical,
        output_column_order=output_order,
        read_columns=read_columns,
    )


def extract_split_date(dataframe: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(dataframe["hour_dt"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("存在无法解析的 hour_dt。")
    return dates.dt.strftime("%Y-%m-%d")


def validate_click(dataframe: pd.DataFrame, context: str) -> pd.Series:
    click_numeric = pd.to_numeric(dataframe["click"], errors="coerce")
    if click_numeric.isna().any():
        raise ValueError(f"{context} 存在缺失 click。")
    if not click_numeric.isin([0, 1]).all():
        raise ValueError(f"{context} click 存在非法取值。")
    return click_numeric.astype(np.int8)


def validate_dates(
    split_date: pd.Series,
    split_name: str,
    expected_dates: set[str] | None = None,
) -> None:
    unique_dates = set(split_date.astype(str).unique())
    if FORBIDDEN_DATE in unique_dates:
        raise ValueError(f"{split_name} 样本含禁止日期 {FORBIDDEN_DATE}（holdout）。")

    if expected_dates is not None:
        unexpected = unique_dates - expected_dates
        if unexpected:
            raise ValueError(f"{split_name} 含意外日期：{sorted(unexpected)}")


def compute_hour_cyclical(hour_of_day: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    hour_values = pd.to_numeric(hour_of_day, errors="coerce").astype(np.float64).to_numpy()
    radians = 2.0 * np.pi * hour_values / 24.0
    return np.sin(radians).astype(np.float32), np.cos(radians).astype(np.float32)


def prepare_unified_batch(
    dataframe: pd.DataFrame,
    column_config: ColumnConfig,
    split_name: str,
    expected_dates: set[str] | None,
) -> pd.DataFrame:
    dataframe = dataframe.reset_index(drop=True)
    context = f"{split_name} batch"

    click = validate_click(dataframe, context)
    split_date = extract_split_date(dataframe)
    validate_dates(split_date, split_name, expected_dates)

    output = pd.DataFrame({"id": dataframe["id"].astype(str)})
    output["split_date"] = split_date.reset_index(drop=True)

    for column in column_config.auxiliary_columns:
        output[column] = dataframe[column].reset_index(drop=True)

    for column in column_config.categorical_columns:
        output[column] = dataframe[column].reset_index(drop=True)

    for column in column_config.numerical_columns:
        if column in COMPUTED_NUMERICAL:
            continue
        output[column] = pd.to_numeric(dataframe[column], errors="coerce").reset_index(drop=True)

    hour_sin, hour_cos = compute_hour_cyclical(dataframe["hour_of_day"])
    output["hour_sin"] = hour_sin
    output["hour_cos"] = hour_cos

    output["click"] = click.reset_index(drop=True)
    return output[column_config.output_column_order]


def calculate_file_quotas(total_target: int, file_row_counts: list[int]) -> list[int]:
    total_rows = sum(file_row_counts)
    if total_target > total_rows:
        raise ValueError(f"目标 {total_target:,} 超过可用行数 {total_rows:,}")

    raw_quotas = [total_target * count / total_rows for count in file_row_counts]
    quotas = [int(np.floor(value)) for value in raw_quotas]
    remaining = total_target - sum(quotas)
    remainders = [(raw_quotas[i] - quotas[i], i) for i in range(len(raw_quotas))]
    for _, file_index in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        quotas[file_index] += 1
        remaining -= 1
    if sum(quotas) != total_target:
        raise ValueError("抽样额度分配失败。")
    return quotas


def iter_file_batches(parquet_path: Path, columns: list[str], batch_size: int):
    parquet_file = pq.ParquetFile(parquet_path)
    for record_batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        yield record_batch.to_pandas()


def write_parquet_parts(
    dataframe: pd.DataFrame,
    output_dir: Path,
    part_start_index: int,
    column_order: list[str],
) -> tuple[list[Path], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    part_index = part_start_index

    for start in range(0, len(dataframe), OUTPUT_PART_ROWS):
        part_df = dataframe.iloc[start : start + OUTPUT_PART_ROWS][column_order]
        part_path = output_dir / f"part-{part_index:04d}.parquet"
        part_df.to_parquet(part_path, index=False)
        output_files.append(part_path)
        part_index += 1

    return output_files, part_index


def calculate_id_sha256(output_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for parquet_path in output_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["id"], batch_size=READ_BATCH_SIZE
        ):
            for id_value in record_batch.to_pandas()["id"]:
                hasher.update((str(id_value) + "\n").encode("utf-8"))
    return hasher.hexdigest()


def calculate_checksum(output_files: list[Path], columns: list[str]) -> str:
    hasher = hashlib.sha256()
    for parquet_path in output_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=columns, batch_size=READ_BATCH_SIZE
        ):
            batch_df = record_batch.to_pandas()
            for column in columns:
                hasher.update(column.encode("utf-8"))
                hasher.update(batch_df[column].to_numpy().tobytes())
    return hasher.hexdigest()


def sample_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    target_rows: int,
    column_config: ColumnConfig,
    expected_dates: set[str] | None,
) -> SplitBuildResult:
    parquet_files = get_sorted_parquet_files(input_dir)
    file_row_counts = [pq.read_metadata(path).num_rows for path in parquet_files]
    file_quotas = calculate_file_quotas(target_rows, file_row_counts)

    if output_dir.exists():
        for parquet_path in output_dir.glob("*.parquet"):
            parquet_path.unlink()

    buffer_frames: list[pd.DataFrame] = []
    buffer_row_count = 0
    part_start_index = 0
    all_output_files: list[Path] = []
    total_collected = 0
    click_counter = {0: 0, 1: 0}
    date_counter: dict[str, int] = {}

    print(f"\n开始 {split_name} 统一样本构建，目标 {target_rows:,} 行 ...")

    for file_index, (parquet_path, file_quota) in enumerate(zip(parquet_files, file_quotas)):
        if file_quota <= 0:
            continue

        file_collected = 0
        print(
            f"[{split_name}] 文件 {file_index + 1}/{len(parquet_files)}: "
            f"{parquet_path.name}，目标 {file_quota:,} 行"
        )

        for batch_index, batch_df in enumerate(
            iter_file_batches(parquet_path, column_config.read_columns, READ_BATCH_SIZE),
            start=1,
        ):
            if file_collected >= file_quota:
                break

            remaining = file_quota - file_collected
            sampled_df = batch_df if len(batch_df) <= remaining else batch_df.sample(
                n=remaining, random_state=RANDOM_SEED
            )
            sampled_df = sampled_df.reset_index(drop=True)

            prepared_df = prepare_unified_batch(
                sampled_df,
                column_config,
                split_name,
                expected_dates,
            )
            buffer_frames.append(prepared_df)
            buffer_row_count += len(prepared_df)
            file_collected += len(sampled_df)
            total_collected += len(sampled_df)

            for click_value, count in prepared_df["click"].value_counts().items():
                click_counter[int(click_value)] += int(count)
            for split_date, count in prepared_df["split_date"].value_counts().items():
                date_counter[str(split_date)] = date_counter.get(str(split_date), 0) + int(count)

            if buffer_row_count >= OUTPUT_PART_ROWS:
                buffer_df = pd.concat(buffer_frames, ignore_index=True)
                write_df = buffer_df.iloc[:OUTPUT_PART_ROWS]
                remainder_df = buffer_df.iloc[OUTPUT_PART_ROWS:]
                part_files, part_start_index = write_parquet_parts(
                    write_df, output_dir, part_start_index, column_config.output_column_order
                )
                all_output_files.extend(part_files)
                buffer_frames = [remainder_df] if len(remainder_df) > 0 else []
                buffer_row_count = len(remainder_df)

            del batch_df, sampled_df, prepared_df
            gc.collect()

    if buffer_row_count > 0:
        buffer_df = pd.concat(buffer_frames, ignore_index=True)
        part_files, _ = write_parquet_parts(
            buffer_df, output_dir, part_start_index, column_config.output_column_order
        )
        all_output_files.extend(part_files)

    if total_collected != target_rows:
        raise ValueError(f"{split_name} 实际行数 {total_collected:,} != 目标 {target_rows:,}")

    unique_dates = sorted(date_counter.keys())
    checksum_columns = column_config.numerical_columns + ["click"]
    return SplitBuildResult(
        split_name=split_name,
        target_rows=target_rows,
        actual_rows=total_collected,
        output_files=all_output_files,
        date_min=unique_dates[0],
        date_max=unique_dates[-1],
        ctr=click_counter.get(1, 0) / total_collected,
        negative_rows=click_counter.get(0, 0),
        positive_rows=click_counter.get(1, 0),
        id_sha256=calculate_id_sha256(all_output_files),
        checksum=calculate_checksum(all_output_files, checksum_columns),
    )


def is_missing_categorical(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def audit_categorical_features(
    train_files: list[Path],
    valid_files: list[Path],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in categorical_columns:
        train_values: set[str] = set()
        train_rows = 0
        train_missing = 0
        for parquet_path in train_files:
            for batch in pq.ParquetFile(parquet_path).iter_batches(columns=[column], batch_size=READ_BATCH_SIZE):
                for value in batch.to_pandas()[column]:
                    train_rows += 1
                    if is_missing_categorical(value):
                        train_missing += 1
                    else:
                        train_values.add(str(value))

        valid_values: set[str] = set()
        valid_rows = 0
        valid_missing = 0
        oov_row_count = 0
        for parquet_path in valid_files:
            for batch in pq.ParquetFile(parquet_path).iter_batches(columns=[column], batch_size=READ_BATCH_SIZE):
                for value in batch.to_pandas()[column]:
                    valid_rows += 1
                    if is_missing_categorical(value):
                        valid_missing += 1
                    else:
                        text_value = str(value)
                        valid_values.add(text_value)
                        if text_value not in train_values:
                            oov_row_count += 1

        valid_non_missing = valid_rows - valid_missing
        records.append(
            {
                "column": column,
                "dtype": str(pq.read_schema(train_files[0]).field(column).type),
                "train_unique": len(train_values),
                "valid_unique": len(valid_values),
                "train_missing_rate": train_missing / train_rows if train_rows else 0.0,
                "valid_missing_rate": valid_missing / valid_rows if valid_rows else 0.0,
                "valid_oov_category_count": len(valid_values - train_values),
                "valid_oov_row_count": oov_row_count,
                "valid_oov_rate": oov_row_count / valid_non_missing if valid_non_missing else 0.0,
            }
        )
    return records


def classify_embedding_groups(
    categorical_audit: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    recommended: list[str] = []
    high_cardinality: list[str] = []
    for record in categorical_audit:
        train_unique = record["train_unique"]
        if train_unique > HIGH_CARDINALITY_THRESHOLD:
            high_cardinality.append(record["column"])
        else:
            recommended.append(record["column"])
    return recommended, high_cardinality


def validate_output_split(
    split_name: str,
    output_files: list[Path],
    column_config: ColumnConfig,
    expected_rows: int,
    expected_dates: set[str],
) -> dict[str, Any]:
    total_rows = 0
    nan_count = 0
    inf_count = 0
    all_dates: set[str] = set()
    duplicate_columns = len(column_config.output_column_order) != len(
        set(column_config.output_column_order)
    )

    for parquet_path in output_files:
        df = pq.read_table(parquet_path).to_pandas()
        total_rows += len(df)
        if list(df.columns) != column_config.output_column_order:
            raise ValueError(f"{parquet_path} 列顺序不符合预期。")
        all_dates.update(df["split_date"].astype(str).unique().tolist())

        num_cols = [c for c in column_config.numerical_columns if c in df.columns]
        num_block = df[num_cols]
        nan_count += int(num_block.isna().sum().sum())
        inf_count += int(np.isinf(num_block.to_numpy(dtype=np.float64, na_value=np.nan)).sum())

        if not df["click"].isin([0, 1]).all():
            raise ValueError(f"{parquet_path} click 非法。")

    if duplicate_columns:
        raise ValueError(f"{split_name} 输出列存在重复。")
    if total_rows != expected_rows:
        raise ValueError(f"{split_name} 行数 {total_rows:,} != {expected_rows:,}")
    if FORBIDDEN_DATE in all_dates:
        raise ValueError(f"{split_name} 含 holdout 日期 {FORBIDDEN_DATE}")
    if not expected_dates.issuperset(all_dates):
        raise ValueError(f"{split_name} 日期超出预期：{sorted(all_dates - expected_dates)}")
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"{split_name} numerical 含 NaN={nan_count} inf={inf_count}")

    return {"nan_count": nan_count, "inf_count": inf_count, "dates": sorted(all_dates)}


def print_final_summary(
    train_result: SplitBuildResult,
    valid_result: SplitBuildResult,
    categorical_count: int,
    numerical_count: int,
    recommended: list[str],
    high_cardinality: list[str],
) -> None:
    print("\n" + "=" * 40)
    print("UNIFIED MODELING SAMPLE SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_result.actual_rows}")
    print(f"VALID_ROWS = {valid_result.actual_rows}")
    print(f"TRAIN_CTR = {train_result.ctr:.6f}")
    print(f"VALID_CTR = {valid_result.ctr:.6f}")
    print(f"CATEGORICAL_FEATURE_COUNT = {categorical_count}")
    print(f"NUMERICAL_FEATURE_COUNT = {numerical_count}")
    print(f"RECOMMENDED_EMBEDDING_FEATURES = {recommended}")
    print(f"HIGH_CARDINALITY_FEATURES = {high_cardinality}")
    print(f"TRAIN_DATE_RANGE = {TRAIN_DATE_START} ~ {TRAIN_DATE_END}")
    print(f"VALID_DATE = {VALID_DATE}")
    print(f"RANDOM_SEED = {RANDOM_SEED}")
    print("OLD_STEP30_SAMPLE_MODIFIED = False")
    print("HOLDOUT_USED = False")
    print("VALIDATION_PASSED = True")
    print("=" * 40)


def main() -> None:
    state = BuildState()
    train_target, valid_target = get_target_rows(TEST_MODE)

    log(state, "=" * 72)
    log(state, "百度 CTR 项目 — 第 37 步 统一建模样本构建")
    log(state, f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    log(state, f"TEST_MODE = {TEST_MODE}")
    log(state, "说明：独立于旧 Step 30，不要求行一致；旧 Step 30 数据不修改。")

    assert_safe_path(TRAIN_INPUT_DIR)
    assert_safe_path(VALID_INPUT_DIR)

    train_files = get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = get_sorted_parquet_files(VALID_INPUT_DIR)
    schema_columns = read_schema_columns(train_files[0])
    column_config = discover_column_config(schema_columns)

    valid_schema = read_schema_columns(valid_files[0])
    missing_valid = [c for c in column_config.read_columns if c not in valid_schema]
    if missing_valid:
        raise ValueError(f"valid schema 缺少列：{missing_valid}")

    train_expected_dates = set(
        pd.date_range(TRAIN_DATE_START, TRAIN_DATE_END, freq="D").strftime("%Y-%m-%d")
    )
    valid_expected_dates = {VALID_DATE}

    log(state, f"\n数据来源：{TRAIN_INPUT_DIR} / {VALID_INPUT_DIR}")
    log(state, f"categorical ({len(column_config.categorical_columns)})：{column_config.categorical_columns}")
    log(state, f"numerical ({len(column_config.numerical_columns)})：{column_config.numerical_columns}")

    train_result = sample_split(
        "train", TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, train_target, column_config, train_expected_dates
    )
    valid_result = sample_split(
        "valid", VALID_INPUT_DIR, VALID_OUTPUT_DIR, valid_target, column_config, valid_expected_dates
    )

    validate_output_split("train", train_result.output_files, column_config, train_target, train_expected_dates)
    validate_output_split("valid", valid_result.output_files, column_config, valid_target, valid_expected_dates)

    cat_audit = audit_categorical_features(
        train_result.output_files, valid_result.output_files, column_config.categorical_columns
    )
    recommended, high_cardinality = classify_embedding_groups(cat_audit)

    log(state, "\nCategorical audit:")
    for record in cat_audit:
        log(
            state,
            f"  {record['column']}: train_u={record['train_unique']:,}, "
            f"valid_u={record['valid_unique']:,}, "
            f"tr_miss={100*record['train_missing_rate']:.4f}%, "
            f"va_miss={100*record['valid_missing_rate']:.4f}%, "
            f"OOV={100*record['valid_oov_rate']:.4f}%",
        )

    metadata = {
        "script_name": "scripts/37_build_unified_modeling_sample.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "train_rows": train_result.actual_rows,
        "valid_rows": valid_result.actual_rows,
        "train_date_range": [train_result.date_min, train_result.date_max],
        "valid_date_range": [valid_result.date_min, valid_result.date_max],
        "train_ctr": train_result.ctr,
        "valid_ctr": valid_result.ctr,
        "auxiliary_columns": column_config.auxiliary_columns,
        "categorical_features": column_config.categorical_columns,
        "numerical_features": column_config.numerical_columns,
        "categorical_feature_count": len(column_config.categorical_columns),
        "numerical_feature_count": len(column_config.numerical_columns),
        "recommended_embedding_features": recommended,
        "high_cardinality_features": high_cardinality,
        "high_cardinality_threshold": HIGH_CARDINALITY_THRESHOLD,
        "categorical_audit": cat_audit,
        "random_seed": RANDOM_SEED,
        "read_batch_size": READ_BATCH_SIZE,
        "output_part_rows": OUTPUT_PART_ROWS,
        "source_paths": {
            "train_input": str(TRAIN_INPUT_DIR),
            "valid_input": str(VALID_INPUT_DIR),
        },
        "output_paths": {
            "train": str(TRAIN_OUTPUT_DIR),
            "valid": str(VALID_OUTPUT_DIR),
        },
        "train_checksum": train_result.checksum,
        "valid_checksum": valid_result.checksum,
        "train_id_sha256": train_result.id_sha256,
        "valid_id_sha256": valid_result.id_sha256,
        "old_step30_sample_modified": False,
        "holdout_used": False,
        "validation_passed": True,
        "legacy_step37_deprecated": True,
        "note": "Independent from Step 30 fixed sample; future models use this unified sample.",
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(state, f"\nMetadata: {METADATA_PATH}")
    AUDIT_PATH.write_text("\n".join(state.lines) + "\n", encoding="utf-8")
    log(state, f"Audit: {AUDIT_PATH}")

    print_final_summary(
        train_result,
        valid_result,
        len(column_config.categorical_columns),
        len(column_config.numerical_columns),
        recommended,
        high_cardinality,
    )


if __name__ == "__main__":
    main()
