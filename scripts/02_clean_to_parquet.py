"""
百度 CTR 数据清洗脚本（正式全量版）

功能：
    分块读取 train.csv / test.csv，做保守清洗后保存为 Parquet。
    本版本不删除任何行，不修改 data/raw 中的原始文件。

重要说明（质量检查字段）：
    is_invalid_click 和 is_dup_id_within_chunk 是数据质量检查字段，
    后续建模时不得直接作为模型训练特征。

正式全量配置：
    CHUNK_SIZE = 200_000（每块 20 万行）
    MAX_CHUNKS = None（处理到 CSV 文件结尾；train 约 200 块，test 约 23 块）
    输出目录：data/processed/train/、data/processed/test/、data/processed/cleaning_report.json

运行前清理：
    仅清理旧的 data/processed/train/、data/processed/test/、
    data/processed/cleaning_report.json，不会触碰 data/raw/ 和 test_run/。

用法：
    python scripts/02_clean_to_parquet.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 路径与全量参数
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

CHUNK_SIZE = 200_000
MAX_CHUNKS = None  # None 表示处理到 CSV 结尾；试跑时可改为 2

# 正式输出目录
OUTPUT_BASE = Path("data/processed")
OUT_TRAIN_DIR = OUTPUT_BASE / "train"
OUT_TEST_DIR = OUTPUT_BASE / "test"
REPORT_PATH = OUTPUT_BASE / "cleaning_report.json"
INVALID_CLICK_DIR = OUTPUT_BASE / "pending_review" / "invalid_click"

# 试跑目录（本脚本正式模式不会清理或写入，仅作文档说明）
TEST_RUN_DIR = Path("data/processed/test_run")


# ---------------------------------------------------------------------------
# 字段定义
# ---------------------------------------------------------------------------

# 读取与清洗阶段均按字符串处理的列（含 id、hour 及站点/应用/设备标识）
STRING_COLUMNS = [
    "id",
    "hour",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
]

# clean_chunk 中转为可空整型的数值列
INT8_COLUMNS = ["click"]  # 仅训练集存在
INT16_COLUMNS = ["banner_pos", "device_type", "device_conn_type"]
INT32_COLUMNS = [
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

# 各可空整型的最小/最大值（超出范围会先设为 NA，再 astype，避免溢出失败）
INT_RANGE: dict[str, tuple[int, int]] = {
    "Int8": (-128, 127),
    "Int16": (-32_768, 32_767),
    "Int32": (-2_147_483_648, 2_147_483_647),
}

# hour 字段的原始格式，例如 "14102100" 表示 2014-10-21 00:00
HOUR_FORMAT = "%y%m%d%H"


def get_all_columns(is_train: bool) -> list[str]:
    """返回 train 或 test 应包含的全部列名（顺序与 CSV 一致）。"""

    columns = [
        "id",
        "hour",
        "C1",
        "banner_pos",
        "site_id",
        "site_domain",
        "site_category",
        "app_id",
        "app_domain",
        "app_category",
        "device_id",
        "device_ip",
        "device_model",
        "device_type",
        "device_conn_type",
        "C14",
        "C15",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "C21",
    ]

    if is_train:
        return ["id", "click", *columns[1:]]

    return columns


def get_read_dtypes(is_train: bool) -> dict[str, str]:
    """
    返回 read_csv 用的 dtype 字典。

    所有列先读成 string，避免全量数据中存在脏数值导致读取直接失败。
    数值转换在 clean_chunk 中通过 pd.to_numeric(errors="coerce") 安全完成。
    """

    return {column: "string" for column in get_all_columns(is_train)}


def prepare_formal_output_directories() -> None:
    """
    正式运行前清理旧的正式输出，避免与上一次全量结果混淆。

    仅清理：
        - data/processed/train/
        - data/processed/test/
        - data/processed/cleaning_report.json

    不会触碰：
        - data/raw/
        - data/processed/test_run/（试跑结果保留供对比）
    """

    if OUT_TRAIN_DIR.exists():
        shutil.rmtree(OUT_TRAIN_DIR)

    if OUT_TEST_DIR.exists():
        shutil.rmtree(OUT_TEST_DIR)

    if REPORT_PATH.exists():
        REPORT_PATH.unlink()

    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)


def _empty_stats(is_train: bool) -> dict[str, Any]:
    """初始化单个数据集（train 或 test）的统计容器。"""

    stats: dict[str, Any] = {
        "chunks_processed": 0,
        "rows_read": 0,
        "rows_written": 0,
        "invalid_hour_count": 0,
        "dup_id_within_chunk_count": 0,
        "missing_value_counts": {},
        "empty_string_counts": {column: 0 for column in STRING_COLUMNS},
        "numeric_coerce_failed_counts": {},
        "numeric_out_of_range_counts": {},
    }

    if is_train:
        stats["click_distribution"] = {"0": 0, "1": 0, "<NA>": 0}
        stats["invalid_click_count"] = 0
        stats["invalid_click_part_files"] = []

    return stats


def _merge_count_dict(target: dict[str, int], source: dict[str, int]) -> None:
    """把 source 中的计数累加到 target。"""

    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _strip_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """对 STRING_COLUMNS 去除首尾空白，便于后续统计空串。"""

    result = dataframe.copy()

    for column in STRING_COLUMNS:
        if column in result.columns:
            result[column] = result[column].str.strip()

    return result


def _count_empty_strings(dataframe: pd.DataFrame) -> dict[str, int]:
    """统计各字符串列中的空串数量（strip 之后）。"""

    counts: dict[str, int] = {}

    for column in STRING_COLUMNS:
        if column in dataframe.columns:
            counts[column] = int((dataframe[column] == "").sum())

    return counts


def _convert_numeric_column(
    series: pd.Series,
    nullable_dtype: str,
) -> tuple[pd.Series, int, int]:
    """
    将字符串列安全转为可空整型。

    步骤：
        1. pd.to_numeric(errors="coerce") 尝试转换
        2. 统计转换失败（原值非空但无法转为数字）
        3. 检查是否超出目标整型范围，超出则设为 NA 并统计
        4. astype 为可空整型

    返回：
        converted           — 转换后的 Series
        coerce_failed_count — 无法 coerce 的数量
        out_of_range_count  — 超出整型范围的数量
    """

    min_val, max_val = INT_RANGE[nullable_dtype]

    # 原值非空：既不是 NA，strip 后也不是空串
    original_not_empty = series.notna() & series.str.strip().ne("")

    numeric = pd.to_numeric(series, errors="coerce")
    coerce_failed_count = int((original_not_empty & numeric.isna()).sum())

    # 已成功转为数字，但超出目标整型可表示范围
    out_of_range_mask = numeric.notna() & ((numeric < min_val) | (numeric > max_val))
    out_of_range_count = int(out_of_range_mask.sum())

    # 超出范围的值先设为 NA，再 astype，避免溢出导致转换失败
    numeric = numeric.mask(out_of_range_mask, other=pd.NA)

    return numeric.astype(nullable_dtype), coerce_failed_count, out_of_range_count


def clean_chunk(
    dataframe: pd.DataFrame,
    is_train: bool,
    chunk_index: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series | None]:
    """
    对单个分块执行清洗（不删行）。

    参数：
        dataframe   — 从 CSV 读入的一个分块（全 string）
        is_train    — 是否为训练集
        chunk_index — 当前分块序号（从 0 开始）

    返回：
        cleaned_df  — 清洗后的 DataFrame（行数与输入相同）
        chunk_stats — 本分块的统计信息
        click_raw   — 训练集 click 的原始字符串（仅 train；否则 None）
    """

    _ = chunk_index
    rows_in = len(dataframe)

    # 训练集：在处理 click 之前保留原始字符串，供 pending_review 使用
    click_raw: pd.Series | None = None
    if is_train and "click" in dataframe.columns:
        click_raw = dataframe["click"].copy()

    result = _strip_string_columns(dataframe)

    chunk_stats: dict[str, Any] = {
        "rows_read": rows_in,
        "rows_written": rows_in,
        "invalid_hour_count": 0,
        "dup_id_within_chunk_count": 0,
        "missing_value_counts": {},
        "empty_string_counts": _count_empty_strings(result),
        "numeric_coerce_failed_counts": {},
        "numeric_out_of_range_counts": {},
    }

    # --- 数值列：安全转换（含范围检查）---
    dtype_map = {
        **{column: "Int8" for column in INT8_COLUMNS},
        **{column: "Int16" for column in INT16_COLUMNS},
        **{column: "Int32" for column in INT32_COLUMNS},
    }

    for column, nullable_dtype in dtype_map.items():
        if column not in result.columns:
            continue

        converted, coerce_failed, out_of_range = _convert_numeric_column(
            result[column],
            nullable_dtype,
        )
        result[column] = converted
        chunk_stats["numeric_coerce_failed_counts"][column] = coerce_failed
        chunk_stats["numeric_out_of_range_counts"][column] = out_of_range

    # --- 解析 hour ---
    result["hour_dt"] = pd.to_datetime(
        result["hour"],
        format=HOUR_FORMAT,
        errors="coerce",
    )
    chunk_stats["invalid_hour_count"] = int(result["hour_dt"].isna().sum())

    # --- 块内重复 id（不使用全局 seen_ids）---
    dup_mask = result.duplicated(subset=["id"], keep=False)
    result["is_dup_id_within_chunk"] = dup_mask
    chunk_stats["dup_id_within_chunk_count"] = int(dup_mask.sum())

    # --- 训练集：click 合法性检查 ---
    if is_train and "click" in result.columns:
        click_series = result["click"]

        # 原值非空但无法转为数字
        if click_raw is not None:
            original_not_empty = click_raw.notna() & click_raw.str.strip().ne("")
            coerce_failed_mask = original_not_empty & click_series.isna()
        else:
            coerce_failed_mask = pd.Series(False, index=result.index)

        # 转换成功但不在 {0, 1} 范围内
        out_of_range_mask = click_series.notna() & ~click_series.isin([0, 1])

        invalid_click_mask = coerce_failed_mask | out_of_range_mask
        result["is_invalid_click"] = invalid_click_mask
        chunk_stats["invalid_click_count"] = int(invalid_click_mask.sum())

        valid_clicks = click_series.dropna()
        chunk_stats["click_distribution"] = {
            "0": int((valid_clicks == 0).sum()),
            "1": int((valid_clicks == 1).sum()),
            "<NA>": int(click_series.isna().sum()),
        }
    elif is_train:
        result["is_invalid_click"] = False
        chunk_stats["invalid_click_count"] = 0
        chunk_stats["click_distribution"] = {"0": 0, "1": 0, "<NA>": 0}

    # --- 缺失值统计（清洗后）---
    missing = result.isna().sum()
    chunk_stats["missing_value_counts"] = {
        column: int(count) for column, count in missing.items() if count > 0
    }

    chunk_stats["rows_written"] = len(result)
    return result, chunk_stats, click_raw


def _merge_chunk_stats(total: dict[str, Any], chunk: dict[str, Any], is_train: bool) -> None:
    """将一个分块的统计累加到总统计中。"""

    total["chunks_processed"] += 1
    total["rows_read"] += chunk["rows_read"]
    total["rows_written"] += chunk["rows_written"]
    total["invalid_hour_count"] += chunk["invalid_hour_count"]
    total["dup_id_within_chunk_count"] += chunk["dup_id_within_chunk_count"]

    _merge_count_dict(total["missing_value_counts"], chunk["missing_value_counts"])
    _merge_count_dict(total["empty_string_counts"], chunk["empty_string_counts"])
    _merge_count_dict(
        total["numeric_coerce_failed_counts"],
        chunk["numeric_coerce_failed_counts"],
    )
    _merge_count_dict(
        total["numeric_out_of_range_counts"],
        chunk["numeric_out_of_range_counts"],
    )

    if is_train:
        total["invalid_click_count"] += chunk["invalid_click_count"]
        dist = chunk["click_distribution"]
        for key in ("0", "1", "<NA>"):
            total["click_distribution"][key] += dist[key]


def process_file(
    input_path: Path,
    output_dir: Path,
    is_train: bool,
    max_chunks: int | None,
) -> dict[str, Any]:
    """
    分块读取 CSV，清洗后写入 Parquet。

    参数：
        input_path  — 原始 CSV 路径
        output_dir  — Parquet 输出目录
        is_train    — 是否为训练集
        max_chunks  — 最多处理几块；None 表示处理全部
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    total_stats = _empty_stats(is_train)

    reader = pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
        dtype=get_read_dtypes(is_train),
    )

    for chunk_index, raw_chunk in enumerate(reader):
        if max_chunks is not None and chunk_index >= max_chunks:
            break

        cleaned, chunk_stats, click_raw = clean_chunk(
            dataframe=raw_chunk,
            is_train=is_train,
            chunk_index=chunk_index,
        )

        part_path = output_dir / f"part-{chunk_index:04d}.parquet"
        cleaned.to_parquet(part_path, index=False)

        # 训练集：若本分块存在非法 click，按分块保存待检查文件（含 click_raw）
        if is_train and chunk_stats["invalid_click_count"] > 0 and click_raw is not None:
            INVALID_CLICK_DIR.mkdir(parents=True, exist_ok=True)
            invalid_mask = cleaned["is_invalid_click"]
            invalid_export = cleaned.loc[invalid_mask].copy()
            invalid_export["click_raw"] = click_raw.loc[invalid_mask].values

            invalid_part_path = INVALID_CLICK_DIR / f"part-{chunk_index:04d}.parquet"
            invalid_export.to_parquet(invalid_part_path, index=False)
            total_stats["invalid_click_part_files"].append(str(invalid_part_path))

        _merge_chunk_stats(total_stats, chunk_stats, is_train)

        print(
            f"  [{input_path.name}] 分块 {chunk_index + 1}: "
            f"读入 {chunk_stats['rows_read']:,} 行, "
            f"写出 {chunk_stats['rows_written']:,} 行"
        )

    return total_stats


def write_report(report: dict[str, Any]) -> None:
    """将清洗报告写入 JSON 文件。"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\n清洗报告已保存：{REPORT_PATH}")


def main() -> None:
    """主流程：清理旧正式输出，依次全量处理 train 和 test，生成 Parquet 与报告。"""

    print("=" * 70)
    print("百度 CTR 数据清洗（正式全量模式）")
    print("=" * 70)
    print(f"分块大小：{CHUNK_SIZE:,} 行")
    print(f"处理块数：{'全部（直到 CSV 结尾）' if MAX_CHUNKS is None else MAX_CHUNKS}")
    print(f"训练集输出：{OUT_TRAIN_DIR}")
    print(f"测试集输出：{OUT_TEST_DIR}")
    print(f"报告输出：{REPORT_PATH}")
    print(f"不会修改：{RAW_DIR}/")
    print(f"不会清理：{TEST_RUN_DIR}/")
    print()

    prepare_formal_output_directories()
    print("已清理旧正式输出目录，准备开始全量处理\n")

    train_stats = process_file(
        input_path=TRAIN_PATH,
        output_dir=OUT_TRAIN_DIR,
        is_train=True,
        max_chunks=MAX_CHUNKS,
    )

    test_stats = process_file(
        input_path=TEST_PATH,
        output_dir=OUT_TEST_DIR,
        is_train=False,
        max_chunks=MAX_CHUNKS,
    )

    report = {
        "config": {
            "chunk_size": CHUNK_SIZE,
            "max_chunks": MAX_CHUNKS,
            "output_base": str(OUTPUT_BASE),
            "hour_format": HOUR_FORMAT,
        },
        "train": train_stats,
        "test": test_stats,
        "notes": [
            "正式全量：零删行；不检查跨分块重复 id（无全局 seen_ids）。",
            "数值列先 string 读取，clean_chunk 内 to_numeric 安全转换。",
            "is_invalid_click / is_dup_id_within_chunk 为质量检查字段，不得直接用于建模。",
            "非法 click 按分块保存至 pending_review/invalid_click/part-XXXX.parquet。",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    write_report(report)

    print("\n" + "=" * 70)
    print("行数守恒检查")
    print("=" * 70)
    for name, stats in [("train", train_stats), ("test", test_stats)]:
        ok = stats["rows_read"] == stats["rows_written"]
        status = "通过" if ok else "失败"
        print(
            f"  {name}: 读入 {stats['rows_read']:,} / 写出 {stats['rows_written']:,} → {status}"
        )


if __name__ == "__main__":
    main()
