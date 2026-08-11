"""
DEPRECATED — DO NOT RUN

本脚本已废弃（2026-08-09）。
原因：取消「恢复 Step 30 相同行 + 补 categorical」方案。
替代：scripts/37_build_unified_modeling_sample.py

---

百度 CTR 项目 — 固定深度学习样本构建（旧 Step 37，已废弃）
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

TEST_MODE = False  # True: 验证通过但不写正式目录；False: 验证通过后写入正式目录

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

TRAIN_INPUT_DIR = Path("data/features/target_encoded/train")
VALID_INPUT_DIR = Path("data/features/target_encoded/valid")

LGBM_TRAIN_DIR = Path("data/tuning/lightgbm_train")
LGBM_VALID_DIR = Path("data/tuning/lightgbm_valid")

TRAIN_OUTPUT_DIR = Path("data/tuning/deep_learning_train")
VALID_OUTPUT_DIR = Path("data/tuning/deep_learning_valid")

TRAIN_TEMP_DIR = Path("data/tuning/_deep_learning_train_tmp")
VALID_TEMP_DIR = Path("data/tuning/_deep_learning_valid_tmp")

STEP30_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")
METADATA_OUTPUT_PATH = Path("outputs/fixed_deep_learning_sample_metadata.json")

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

TRAIN_TARGET_ROWS = 2_000_000
VALID_TARGET_ROWS = 500_000

FLOAT_TOLERANCE = 1e-7
BATCH_SIZE = 200_000

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

# 可选 embedding 交叉特征（上游若存在则保留）
OPTIONAL_CATEGORICALS = [
    "banner_device_cross",
    "hour_banner_cross",
    "site_device_cross",
]

# 时间辅助字段（原始，非 sin/cos 工程特征）
TIME_AUXILIARY_COLUMNS = ["hour", "hour_of_day", "hour_dt", "date", "day_of_week"]


@dataclass
class SplitBuildResult:
    """单个 split 构建结果。"""

    split_name: str
    target_rows: int
    actual_rows: int
    output_files: list[Path]
    date_min: str | None
    date_max: str | None
    id_sha256: str
    checksum: str


@dataclass
class RowMatchResult:
    """逐行一致性验证结果。"""

    split_name: str
    rows_compared: int
    matched: bool
    checksum: str
    mismatch_summary: str | None = None


def load_step30_module():
    """动态加载 Step 30 模块以复用抽样逻辑。"""

    import sys

    script_path = Path(__file__).resolve().parent / "30_build_fixed_tuning_sample.py"
    module_name = "step30_sampling"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Step 30 模块：{script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_safe_path(path: Path) -> None:
    """禁止访问 holdout / test.csv。"""

    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def get_target_rows(_test_mode: bool) -> tuple[int, int]:
    """始终使用 Step 30 正式规模，保证抽样结果可比对。"""

    return TRAIN_TARGET_ROWS, VALID_TARGET_ROWS


def discover_available_columns(schema_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    从上游 schema 确定 auxiliary / categorical 列。

    返回：(auxiliary_columns, categorical_columns, missing_requested)
    """

    auxiliary: list[str] = []
    for column in TIME_AUXILIARY_COLUMNS:
        if column in schema_columns:
            auxiliary.append(column)

    categoricals: list[str] = []
    missing_requested: list[str] = []
    for column in REQUESTED_CATEGORICALS:
        if column in schema_columns:
            categoricals.append(column)
        else:
            missing_requested.append(column)

    for column in OPTIONAL_CATEGORICALS:
        if column in schema_columns:
            categoricals.append(column)

    return auxiliary, categoricals, missing_requested


def get_deep_learning_read_columns(
    step30,
    feature_config,
    schema_columns: list[str],
    auxiliary_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    """Step 30 read_columns + auxiliary + categorical（稳定去重顺序）。"""

    base_columns = step30.get_read_columns(feature_config, schema_columns)
    extra_columns = [col for col in auxiliary_columns + categorical_columns if col not in base_columns]
    return list(dict.fromkeys([*base_columns, *extra_columns]))


def prepare_deep_learning_batch(
    step30,
    dataframe: pd.DataFrame,
    feature_config,
    auxiliary_columns: list[str],
    categorical_columns: list[str],
    split_name: str,
    source_file: str,
    batch_index: int,
) -> pd.DataFrame:
    """
    基于 Step 30 的 prepare_feature_batch 构造深度学习 batch。

    数值特征与 Step 30 完全一致；额外附加原始 categorical 与时间辅助列。
    """

    lgbm_df = step30.prepare_feature_batch(
        dataframe,
        feature_config,
        split_name=split_name,
        source_file=source_file,
        batch_index=batch_index,
    )

    expected_rows = len(lgbm_df)
    dataframe = dataframe.reset_index(drop=True)

    auxiliary_data: dict[str, pd.Series] = {}
    for column in auxiliary_columns:
        if column not in dataframe.columns:
            raise ValueError(f"batch 缺少辅助列 {column}")
        auxiliary_data[column] = dataframe[column].reset_index(drop=True)

    categorical_data: dict[str, pd.Series] = {}
    for column in categorical_columns:
        if column not in dataframe.columns:
            raise ValueError(f"batch 缺少 categorical 列 {column}")
        categorical_data[column] = dataframe[column].reset_index(drop=True)

    output = pd.DataFrame({"id": lgbm_df["id"].reset_index(drop=True)})
    for column in auxiliary_columns:
        if column == "split_date":
            continue
        output[column] = auxiliary_data[column]

    output["split_date"] = lgbm_df["split_date"].reset_index(drop=True)

    for column in categorical_columns:
        output[column] = categorical_data[column]

    for column in feature_config.feature_columns:
        output[column] = lgbm_df[column].reset_index(drop=True)

    output["click"] = lgbm_df["click"].reset_index(drop=True)

    if len(output) != expected_rows:
        raise ValueError(
            f"{split_name}/{source_file} batch {batch_index} 深度学习 batch 行数不一致："
            f"{len(output)} vs {expected_rows}"
        )

    return output


def get_output_column_order(
    auxiliary_columns: list[str],
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> list[str]:
    """列顺序：[identifier/aux] [categorical] [numerical] [click]"""

    identifier_cols = ["id"]
    for column in auxiliary_columns:
        if column not in identifier_cols and column != "split_date":
            identifier_cols.append(column)
    if "split_date" not in identifier_cols:
        identifier_cols.append("split_date")

    return [*identifier_cols, *categorical_columns, *numerical_columns, "click"]


def write_deep_learning_parts(
    dataframe: pd.DataFrame,
    output_dir: Path,
    part_start_index: int,
    column_order: list[str],
    output_part_rows: int,
) -> tuple[list[Path], list[dict], int]:
    """按固定列顺序分块写出 Parquet。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = dataframe[column_order]

    output_files: list[Path] = []
    part_records: list[dict] = []
    part_index = part_start_index

    for start in range(0, len(dataframe), output_part_rows):
        part_df = dataframe.iloc[start : start + output_part_rows].copy()
        part_path = output_dir / f"part-{part_index:04d}.parquet"
        part_df.to_parquet(part_path, index=False)
        output_files.append(part_path)
        part_records.append({"file_name": part_path.name, "rows": len(part_df)})
        part_index += 1

    return output_files, part_records, part_index


def sample_split_deep_learning(
    step30,
    split_name: str,
    input_dir: Path,
    output_dir: Path,
    target_rows: int,
    feature_config,
    read_columns: list[str],
    auxiliary_columns: list[str],
    categorical_columns: list[str],
    column_order: list[str],
) -> SplitBuildResult:
    """复用 Step 30 sample_split 算法，写出含 categorical 的深度学习样本。"""

    assert_safe_path(input_dir)
    assert_safe_path(output_dir)

    parquet_files = step30.get_sorted_parquet_files(input_dir)
    file_row_counts = step30.inspect_parquet_rows(parquet_files)
    file_quotas = step30.calculate_file_quotas(target_rows, file_row_counts)

    if output_dir.exists():
        for parquet_path in output_dir.glob("*.parquet"):
            parquet_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_frames: list[pd.DataFrame] = []
    buffer_row_count = 0
    part_start_index = 0
    all_output_files: list[Path] = []

    total_collected = 0
    date_counter: dict[str, int] = {}

    print(f"\n开始 {split_name} 深度学习样本构建，目标 {target_rows:,} 行 ...")

    for file_index, (parquet_path, file_quota) in enumerate(zip(parquet_files, file_quotas)):
        if file_quota <= 0:
            continue

        file_collected = 0
        print(
            f"[{split_name}] 文件 {file_index + 1}/{len(parquet_files)}: "
            f"{parquet_path.name}，目标抽样 {file_quota:,} 行"
        )

        for batch_index, batch_df in enumerate(
            step30.iter_file_batches(parquet_path, read_columns, step30.READ_BATCH_SIZE),
            start=1,
        ):
            if file_collected >= file_quota:
                break

            batch_df = batch_df.reset_index(drop=True)
            step30.validate_click_batch(
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
                    random_state=step30.RANDOM_STATE,
                )

            sampled_df = sampled_df.reset_index(drop=True)
            prepared_df = prepare_deep_learning_batch(
                step30,
                sampled_df,
                feature_config,
                auxiliary_columns,
                categorical_columns,
                split_name=split_name,
                source_file=parquet_path.name,
                batch_index=batch_index,
            )
            buffer_frames.append(prepared_df)
            buffer_row_count += len(prepared_df)

            for split_date, row_count in prepared_df["split_date"].value_counts().items():
                date_counter[str(split_date)] = date_counter.get(str(split_date), 0) + int(row_count)

            file_collected += len(sampled_df)
            total_collected += len(sampled_df)

            if buffer_row_count >= step30.OUTPUT_PART_ROWS:
                buffer_df = pd.concat(buffer_frames, ignore_index=True)
                write_df = buffer_df.iloc[: step30.OUTPUT_PART_ROWS]
                remainder_df = buffer_df.iloc[step30.OUTPUT_PART_ROWS :]

                part_files, _, part_start_index = write_deep_learning_parts(
                    write_df,
                    output_dir,
                    part_start_index,
                    column_order,
                    step30.OUTPUT_PART_ROWS,
                )
                all_output_files.extend(part_files)

                buffer_frames = [remainder_df] if len(remainder_df) > 0 else []
                buffer_row_count = len(remainder_df)

            del batch_df, sampled_df, prepared_df
            gc.collect()

    if buffer_row_count > 0:
        buffer_df = pd.concat(buffer_frames, ignore_index=True)
        part_files, _, part_start_index = write_deep_learning_parts(
            buffer_df,
            output_dir,
            part_start_index,
            column_order,
            step30.OUTPUT_PART_ROWS,
        )
        all_output_files.extend(part_files)

    del buffer_frames
    gc.collect()

    if total_collected != target_rows:
        raise ValueError(
            f"{split_name} 实际输出行数 {total_collected:,} 与目标 {target_rows:,} 不一致。"
        )

    id_sha256 = step30.calculate_id_sha256(all_output_files)
    checksum = calculate_shared_feature_checksum(all_output_files, feature_config.feature_columns)

    unique_dates = sorted(date_counter.keys())
    return SplitBuildResult(
        split_name=split_name,
        target_rows=target_rows,
        actual_rows=total_collected,
        output_files=all_output_files,
        date_min=unique_dates[0] if unique_dates else None,
        date_max=unique_dates[-1] if unique_dates else None,
        id_sha256=id_sha256,
        checksum=checksum,
    )


def calculate_shared_feature_checksum(
    output_files: list[Path],
    feature_columns: list[str],
    max_rows: int | None = None,
) -> str:
    """对 click + 33 数值特征计算 SHA256 checksum（用于 metadata）。"""

    hasher = hashlib.sha256()
    rows_written = 0
    compare_columns = ["click", *feature_columns]

    for parquet_path in output_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - rows_written
                if remaining <= 0:
                    break
                batch_df = batch_df.iloc[:remaining]

            for column in compare_columns:
                values = batch_df[column].to_numpy()
                hasher.update(column.encode("utf-8"))
                hasher.update(values.tobytes())

            rows_written += len(batch_df)
            if max_rows is not None and rows_written >= max_rows:
                break

        if max_rows is not None and rows_written >= max_rows:
            break

    return hasher.hexdigest()


def compare_feature_values(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    column_name: str,
    float_tolerance: float,
) -> str | None:
    """比较单列数值，不一致时返回错误描述。"""

    if reference_values.shape != candidate_values.shape:
        return (
            f"{column_name}: shape 不一致 "
            f"{reference_values.shape} vs {candidate_values.shape}"
        )

    if column_name == "click":
        if not np.array_equal(reference_values, candidate_values):
            mismatch_count = int(np.sum(reference_values != candidate_values))
            return f"click: {mismatch_count} 行不一致"
        return None

    ref_is_nan = pd.isna(reference_values)
    cand_is_nan = pd.isna(candidate_values)
    if not np.array_equal(ref_is_nan, cand_is_nan):
        return f"{column_name}: NaN 位置不一致"

    ref_finite = reference_values[~ref_is_nan].astype(np.float64)
    cand_finite = candidate_values[~cand_is_nan].astype(np.float64)
    if ref_finite.size == 0:
        return None

    if not np.allclose(ref_finite, cand_finite, rtol=float_tolerance, atol=float_tolerance):
        diff = np.abs(ref_finite - cand_finite)
        max_diff = float(np.max(diff))
        mismatch_count = int(np.sum(diff > float_tolerance))
        return f"{column_name}: {mismatch_count} 行超出容差 {float_tolerance}，max_diff={max_diff:.3e}"

    return None


def validate_against_step30(
    split_name: str,
    reference_dir: Path,
    candidate_dir: Path,
    feature_columns: list[str],
    expected_rows: int | None = None,
) -> RowMatchResult:
    """
    逐 batch 比较 candidate 与 Step 30 lightgbm 固定样本（click + 33 numerical）。

    按全局行顺序对齐，支持 TEST_MODE 下 candidate 行数少于 reference。
    """

    assert_safe_path(reference_dir)
    assert_safe_path(candidate_dir)

    reference_files = sorted(reference_dir.glob("part-*.parquet"))
    candidate_files = sorted(candidate_dir.glob("part-*.parquet"))

    if not reference_files:
        raise FileNotFoundError(f"Step 30 参考样本不存在：{reference_dir}")
    if not candidate_files:
        raise FileNotFoundError(f"候选深度学习样本不存在：{candidate_dir}")

    compare_columns = ["click", *feature_columns]
    rows_compared = 0
    first_mismatch: str | None = None

    ref_file_index = 0
    ref_df: pd.DataFrame | None = None
    ref_pos = 0

    for cand_path in candidate_files:
        cand_df = pq.read_table(cand_path).to_pandas()
        cand_pos = 0

        while cand_pos < len(cand_df):
            if expected_rows is not None and rows_compared >= expected_rows:
                break

            if ref_df is None or ref_pos >= len(ref_df):
                if ref_file_index >= len(reference_files):
                    first_mismatch = (
                        f"参考样本文件不足：global_row={rows_compared}, "
                        f"candidate 仍有未比较行"
                    )
                    break
                ref_df = pq.read_table(reference_files[ref_file_index]).to_pandas()
                ref_pos = 0
                ref_file_index += 1

            remaining = expected_rows - rows_compared if expected_rows is not None else None
            take = min(len(cand_df) - cand_pos, len(ref_df) - ref_pos)
            if remaining is not None:
                take = min(take, remaining)

            ref_slice = ref_df.iloc[ref_pos : ref_pos + take]
            cand_slice = cand_df.iloc[cand_pos : cand_pos + take]

            for column in compare_columns:
                error = compare_feature_values(
                    ref_slice[column].to_numpy(),
                    cand_slice[column].to_numpy(),
                    column_name=column,
                    float_tolerance=FLOAT_TOLERANCE,
                )
                if error is not None:
                    first_mismatch = f"global_row={rows_compared}: {error}"
                    break

            if first_mismatch is not None:
                break

            ref_pos += take
            cand_pos += take
            rows_compared += take

        if first_mismatch is not None:
            break
        if expected_rows is not None and rows_compared >= expected_rows:
            break

    if first_mismatch is None and expected_rows is not None and rows_compared < expected_rows:
        first_mismatch = f"比较行数不足：期望 {expected_rows:,}，实际 {rows_compared:,}"

    checksum = calculate_shared_feature_checksum(
        candidate_files,
        feature_columns,
        max_rows=expected_rows,
    )

    return RowMatchResult(
        split_name=split_name,
        rows_compared=rows_compared,
        matched=first_mismatch is None,
        checksum=checksum,
        mismatch_summary=first_mismatch,
    )


def is_missing_categorical(value: Any) -> bool:
    """判断类别值是否缺失。"""

    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def audit_categorical_features(
    train_files: list[Path],
    valid_files: list[Path],
    categorical_columns: list[str],
) -> list[dict[str, Any]]:
    """对最终 categorical 列做 unique / missing / OOV 审计。"""

    records: list[dict[str, Any]] = []

    for column in categorical_columns:
        train_values: set[str] = set()
        train_rows = 0
        train_missing = 0

        for parquet_path in train_files:
            for record_batch in pq.ParquetFile(parquet_path).iter_batches(
                columns=[column],
                batch_size=BATCH_SIZE,
            ):
                series = record_batch.to_pandas()[column]
                train_rows += len(series)
                for value in series:
                    if is_missing_categorical(value):
                        train_missing += 1
                    else:
                        train_values.add(str(value))

        valid_values: set[str] = set()
        valid_rows = 0
        valid_missing = 0
        oov_row_count = 0

        for parquet_path in valid_files:
            for record_batch in pq.ParquetFile(parquet_path).iter_batches(
                columns=[column],
                batch_size=BATCH_SIZE,
            ):
                series = record_batch.to_pandas()[column]
                valid_rows += len(series)
                for value in series:
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


def promote_temp_to_final(temp_dir: Path, final_dir: Path) -> None:
    """验证通过后将临时目录提升为正式输出。"""

    if final_dir.exists():
        for parquet_path in final_dir.glob("*.parquet"):
            parquet_path.unlink()
    else:
        final_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in sorted(temp_dir.glob("part-*.parquet")):
        shutil.move(str(parquet_path), str(final_dir / parquet_path.name))

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def cleanup_temp_dirs() -> None:
    """清理临时目录。"""

    for temp_dir in (TRAIN_TEMP_DIR, VALID_TEMP_DIR):
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def print_categorical_audit(records: list[dict[str, Any]]) -> None:
    """打印 categorical 审计表。"""

    print("\nCategorical feature audit:")
    print(
        f"{'column':<22} {'dtype':<14} {'tr_u':>8} {'va_u':>8} "
        f"{'tr_miss%':>9} {'va_miss%':>9} {'OOV_row':>9} {'OOV_rate':>9}"
    )
    print("-" * 95)
    for record in records:
        print(
            f"{record['column']:<22} {record['dtype']:<14} "
            f"{record['train_unique']:>8,} {record['valid_unique']:>8,} "
            f"{100 * record['train_missing_rate']:>8.4f}% "
            f"{100 * record['valid_missing_rate']:>8.4f}% "
            f"{record['valid_oov_row_count']:>9,} "
            f"{100 * record['valid_oov_rate']:>8.4f}%"
        )


def print_final_summary(
    train_result: SplitBuildResult,
    valid_result: SplitBuildResult,
    train_match: RowMatchResult,
    valid_match: RowMatchResult,
    categorical_columns: list[str],
    numerical_columns: list[str],
    validation_passed: bool,
) -> None:
    """打印终端 Summary。"""

    exact_same = train_match.matched and valid_match.matched

    print("\n" + "=" * 40)
    print("FIXED DEEP LEARNING SAMPLE SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_result.actual_rows}")
    print(f"VALID_ROWS = {valid_result.actual_rows}")
    print(f"CATEGORICAL_FEATURE_COUNT = {len(categorical_columns)}")
    print(f"NUMERICAL_FEATURE_COUNT = {len(numerical_columns)}")
    print(f"TRAIN_ROW_MATCH_STEP30 = {train_match.matched}")
    print(f"VALID_ROW_MATCH_STEP30 = {valid_match.matched}")
    print(f"EXACT_SAME_ROWS_AS_STEP30 = {exact_same}")
    print("DEEP_LEARNING_TRAIN_PATH =")
    print("data/tuning/deep_learning_train/")
    print("DEEP_LEARNING_VALID_PATH =")
    print("data/tuning/deep_learning_valid/")
    print("HOLDOUT_USED = False")
    print(f"VALIDATION_PASSED = {validation_passed}")
    print("=" * 40)


def main() -> None:
    """主流程：复用 Step 30 抽样 → 验证 → 写出正式样本。"""

    step30 = load_step30_module()
    train_target, valid_target = get_target_rows(TEST_MODE)

    assert_safe_path(TRAIN_INPUT_DIR)
    assert_safe_path(VALID_INPUT_DIR)
    assert_safe_path(LGBM_TRAIN_DIR)
    assert_safe_path(LGBM_VALID_DIR)

    step30_metadata = json.loads(STEP30_METADATA_PATH.read_text(encoding="utf-8"))
    if step30_metadata.get("holdout_used") is not False:
        raise ValueError("Step 30 元数据 holdout_used 必须为 false。")

    feature_columns: list[str] = step30_metadata["feature_columns"]

    print("=" * 70)
    print("固定深度学习样本构建（第 37 步）")
    print("=" * 70)
    print(f"TEST_MODE = {TEST_MODE}（True=仅验证不写正式目录；False=验证后写入正式目录）")
    print(f"train 目标行数：{train_target:,}")
    print(f"valid 目标行数：{valid_target:,}")
    print(f"抽样方法：复用 scripts/30_build_fixed_tuning_sample.py")
    print(f"random_seed = {step30.RANDOM_STATE}")

    train_files = step30.get_sorted_parquet_files(TRAIN_INPUT_DIR)
    valid_files = step30.get_sorted_parquet_files(VALID_INPUT_DIR)
    schema_columns = step30.read_schema_columns(train_files[0])
    feature_config = step30.discover_feature_config(schema_columns)

    if feature_config.feature_columns != feature_columns:
        raise ValueError("Step 30 元数据 feature_columns 与当前 schema 推导结果不一致。")

    auxiliary_columns, categorical_columns, missing_requested = discover_available_columns(
        schema_columns
    )
    if missing_requested:
        raise ValueError(f"上游缺少必需的 categorical 列：{missing_requested}")

    read_columns = get_deep_learning_read_columns(
        step30,
        feature_config,
        schema_columns,
        auxiliary_columns,
        categorical_columns,
    )

    valid_schema = step30.read_schema_columns(valid_files[0])
    missing_valid = [column for column in read_columns if column not in valid_schema]
    if missing_valid:
        raise ValueError(f"valid schema 缺少字段：{missing_valid}")

    column_order = get_output_column_order(
        auxiliary_columns,
        categorical_columns,
        feature_columns,
    )

    print(f"\nauxiliary 列 ({len(auxiliary_columns)})：{auxiliary_columns}")
    print(f"categorical 列 ({len(categorical_columns)})：{categorical_columns}")
    print(f"numerical 列 ({len(feature_columns)})：与 Step 30 完全一致")

    cleanup_temp_dirs()

    try:
        train_result = sample_split_deep_learning(
            step30,
            split_name="train",
            input_dir=TRAIN_INPUT_DIR,
            output_dir=TRAIN_TEMP_DIR,
            target_rows=train_target,
            feature_config=feature_config,
            read_columns=read_columns,
            auxiliary_columns=auxiliary_columns,
            categorical_columns=categorical_columns,
            column_order=column_order,
        )

        valid_result = sample_split_deep_learning(
            step30,
            split_name="valid",
            input_dir=VALID_INPUT_DIR,
            output_dir=VALID_TEMP_DIR,
            target_rows=valid_target,
            feature_config=feature_config,
            read_columns=read_columns,
            auxiliary_columns=auxiliary_columns,
            categorical_columns=categorical_columns,
            column_order=column_order,
        )

        print("\n" + "=" * 70)
        print("Step 30 逐行一致性验证（click + 33 numerical features）")
        print("=" * 70)

        train_match = validate_against_step30(
            split_name="train",
            reference_dir=LGBM_TRAIN_DIR,
            candidate_dir=TRAIN_TEMP_DIR,
            feature_columns=feature_columns,
            expected_rows=train_target,
        )
        valid_match = validate_against_step30(
            split_name="valid",
            reference_dir=LGBM_VALID_DIR,
            candidate_dir=VALID_TEMP_DIR,
            feature_columns=feature_columns,
            expected_rows=valid_target,
        )

        print(f"TRAIN_ROW_MATCH = {train_match.matched}")
        print(f"  rows_compared = {train_match.rows_compared:,}")
        print(f"  checksum = {train_match.checksum}")
        if train_match.mismatch_summary:
            print(f"  mismatch = {train_match.mismatch_summary}")

        print(f"VALID_ROW_MATCH = {valid_match.matched}")
        print(f"  rows_compared = {valid_match.rows_compared:,}")
        print(f"  checksum = {valid_match.checksum}")
        if valid_match.mismatch_summary:
            print(f"  mismatch = {valid_match.mismatch_summary}")

        if not train_match.matched or not valid_match.matched:
            cleanup_temp_dirs()
            print_final_summary(
                train_result,
                valid_result,
                train_match,
                valid_match,
                categorical_columns,
                feature_columns,
                validation_passed=False,
            )
            raise RuntimeError(
                "Step 30 逐行一致性验证失败，未写出正式 deep-learning 样本。"
                f" train={train_match.mismatch_summary}; valid={valid_match.mismatch_summary}"
            )

        if not TEST_MODE:
            if train_result.id_sha256 != step30_metadata["train_id_sha256"]:
                cleanup_temp_dirs()
                raise RuntimeError(
                    "train id_sha256 与 Step 30 元数据不一致，未写出正式样本。"
                )
            if valid_result.id_sha256 != step30_metadata["valid_id_sha256"]:
                cleanup_temp_dirs()
                raise RuntimeError(
                    "valid id_sha256 与 Step 30 元数据不一致，未写出正式样本。"
                )
        else:
            print("\nTEST_MODE：跳过 id_sha256 指纹写入检查（正式模式会校验）。")
            print(f"  train id_sha256 = {train_result.id_sha256}")
            print(f"  valid id_sha256 = {valid_result.id_sha256}")
            print(f"  step30 train    = {step30_metadata['train_id_sha256']}")
            print(f"  step30 valid    = {step30_metadata['valid_id_sha256']}")
            if (
                train_result.id_sha256 != step30_metadata["train_id_sha256"]
                or valid_result.id_sha256 != step30_metadata["valid_id_sha256"]
            ):
                cleanup_temp_dirs()
                raise RuntimeError("id_sha256 与 Step 30 元数据不一致。")

        if TEST_MODE:
            print("\nTEST_MODE：样本写入临时验证路径，不覆盖正式 deep_learning 目录。")
            final_train_files = sorted(TRAIN_TEMP_DIR.glob("part-*.parquet"))
            final_valid_files = sorted(VALID_TEMP_DIR.glob("part-*.parquet"))
        else:
            promote_temp_to_final(TRAIN_TEMP_DIR, TRAIN_OUTPUT_DIR)
            promote_temp_to_final(VALID_TEMP_DIR, VALID_OUTPUT_DIR)
            final_train_files = sorted(TRAIN_OUTPUT_DIR.glob("part-*.parquet"))
            final_valid_files = sorted(VALID_OUTPUT_DIR.glob("part-*.parquet"))

        categorical_audit = audit_categorical_features(
            final_train_files,
            final_valid_files,
            categorical_columns,
        )
        print_categorical_audit(categorical_audit)

        metadata = {
            "script_name": "scripts/37_build_fixed_deep_learning_sample.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test_mode": TEST_MODE,
            "train_rows": train_result.actual_rows,
            "valid_rows": valid_result.actual_rows,
            "train_date_range": [train_result.date_min, train_result.date_max],
            "valid_date_range": [valid_result.date_min, valid_result.date_max],
            "auxiliary_columns": auxiliary_columns,
            "categorical_features": categorical_columns,
            "numerical_features": feature_columns,
            "categorical_feature_count": len(categorical_columns),
            "numerical_feature_count": len(feature_columns),
            "column_order": column_order,
            "sampling_method": "reuse scripts/30_build_fixed_tuning_sample.py sample_split algorithm",
            "random_seed": step30.RANDOM_STATE,
            "read_batch_size": step30.READ_BATCH_SIZE,
            "output_part_rows": step30.OUTPUT_PART_ROWS,
            "source_paths": {
                "train_input": str(TRAIN_INPUT_DIR),
                "valid_input": str(VALID_INPUT_DIR),
                "lightgbm_train_reference": str(LGBM_TRAIN_DIR),
                "lightgbm_valid_reference": str(LGBM_VALID_DIR),
            },
            "output_paths": {
                "train": str(TRAIN_OUTPUT_DIR),
                "valid": str(VALID_OUTPUT_DIR),
            },
            "train_match_step30": train_match.matched,
            "valid_match_step30": valid_match.matched,
            "exact_same_rows_as_step30": train_match.matched and valid_match.matched,
            "train_checksum": train_match.checksum,
            "valid_checksum": valid_match.checksum,
            "train_id_sha256": train_result.id_sha256,
            "valid_id_sha256": valid_result.id_sha256,
            "step30_train_id_sha256": step30_metadata["train_id_sha256"],
            "step30_valid_id_sha256": step30_metadata["valid_id_sha256"],
            "float_tolerance": FLOAT_TOLERANCE,
            "categorical_audit": categorical_audit,
            "holdout_used": False,
            "validation_passed": True,
        }

        METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_OUTPUT_PATH.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print_final_summary(
            train_result,
            valid_result,
            train_match,
            valid_match,
            categorical_columns,
            feature_columns,
            validation_passed=True,
        )
        print(f"\nMetadata: {METADATA_OUTPUT_PATH}")

    except Exception:
        cleanup_temp_dirs()
        raise


if __name__ == "__main__":
    main()
