"""
百度 CTR 项目 — 训练集时间划分脚本（建模阶段）

功能：
    将有 click 的训练特征数据按 hour_dt 日期划分为 train / valid / holdout，
    用于后续模型训练、调参与最终留出评估。

数据输入：
    data/features/frequency/train/*.parquet

数据输出：
    data/model_input/train/
    data/model_input/valid/
    data/model_input/holdout/
    outputs/model_split_summary.csv

划分规则（按 hour_dt 日期）：
    - train：   2014-10-21 ~ 2014-10-28
    - valid：   2014-10-29
    - holdout： 2014-10-30

说明：
    - 仅在含 click 的 train 特征数据内部划分，不处理 test 集
    - 逐 Parquet 分块读取，不一次性载入全量数据
    - 超出 2014-10-21 ~ 2014-10-30 范围的记录单独统计并警告

用法：
    python scripts/22_time_split.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

INPUT_DIR = Path("data/features/frequency/train")

OUTPUT_TRAIN_DIR = Path("data/model_input/train")
OUTPUT_VALID_DIR = Path("data/model_input/valid")
OUTPUT_HOLDOUT_DIR = Path("data/model_input/holdout")

SUMMARY_CSV = Path("outputs/model_split_summary.csv")

# ---------------------------------------------------------------------------
# 时间划分边界（按 hour_dt 的日期部分）
# ---------------------------------------------------------------------------

TRAIN_START_DATE = pd.Timestamp("2014-10-21")
TRAIN_END_DATE = pd.Timestamp("2014-10-28")
VALID_DATE = pd.Timestamp("2014-10-29")
HOLDOUT_DATE = pd.Timestamp("2014-10-30")

# 期望数据覆盖的整体日期范围（用于检测异常记录）
EXPECTED_RANGE_START = TRAIN_START_DATE
EXPECTED_RANGE_END = HOLDOUT_DATE


def list_parquet_files(parquet_dir: Path) -> list[Path]:
    """列出输入 Parquet 分块路径。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到输入目录：{parquet_dir}\n"
            "请先运行：python scripts/20_build_frequency_features.py"
        )

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            "请先运行：python scripts/20_build_frequency_features.py"
        )

    return files


def get_total_input_rows(parquet_files: list[Path]) -> int:
    """读取各分块元数据，统计输入总行数（不加载全表）。"""

    return sum(pq.read_metadata(path).num_rows for path in parquet_files)


def split_chunk_by_date(dataframe: pd.DataFrame) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
    int,
]:
    """
    将单个分块按 hour_dt 日期划分为 train / valid / holdout。

    返回：
        train_df, valid_df, holdout_df, out_of_range_count, invalid_hour_dt_count
    """

    if "hour_dt" not in dataframe.columns:
        raise ValueError("输入数据缺少 hour_dt 字段，无法按日期划分。")

    result = dataframe.copy()
    hour_dt = pd.to_datetime(result["hour_dt"], errors="coerce")
    event_date = hour_dt.dt.normalize()

    # 无效 hour_dt（无法解析为时间）
    invalid_hour_dt_count = int(hour_dt.isna().sum())

    train_mask = (event_date >= TRAIN_START_DATE) & (event_date <= TRAIN_END_DATE)
    valid_mask = event_date == VALID_DATE
    holdout_mask = event_date == HOLDOUT_DATE

    in_expected_range = (event_date >= EXPECTED_RANGE_START) & (
        event_date <= EXPECTED_RANGE_END
    )
    # 日期可解析但不在期望范围内
    out_of_range_count = int((~in_expected_range & event_date.notna()).sum())

    train_df = result.loc[train_mask].copy()
    valid_df = result.loc[valid_mask].copy()
    holdout_df = result.loc[holdout_mask].copy()

    return train_df, valid_df, holdout_df, out_of_range_count, invalid_hour_dt_count


def save_chunk_if_not_empty(dataframe: pd.DataFrame, output_path: Path) -> None:
    """若分块非空则保存为 Parquet；空分块不写文件。"""

    if len(dataframe) == 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False, engine="pyarrow")


def process_single_file(
    input_path: Path,
    train_dir: Path,
    valid_dir: Path,
    holdout_dir: Path,
) -> dict[str, int]:
    """
    读取并划分单个 Parquet 分块，分别写出。

    返回：
        本文件的 train / valid / holdout / out_of_range / invalid_hour_dt 行数
    """

    chunk = pd.read_parquet(input_path)

    train_df, valid_df, holdout_df, out_of_range, invalid_hour_dt = split_chunk_by_date(
        chunk
    )

    output_name = input_path.name
    save_chunk_if_not_empty(train_df, train_dir / output_name)
    save_chunk_if_not_empty(valid_df, valid_dir / output_name)
    save_chunk_if_not_empty(holdout_df, holdout_dir / output_name)

    return {
        "input_rows": len(chunk),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "holdout_rows": len(holdout_df),
        "out_of_range_rows": out_of_range,
        "invalid_hour_dt_rows": invalid_hour_dt,
    }


def build_split_summary(
    train_rows: int,
    valid_rows: int,
    holdout_rows: int,
    total_input_rows: int,
) -> pd.DataFrame:
    """生成分划分摘要表。"""

    split_rows = {
        "train": train_rows,
        "valid": valid_rows,
        "holdout": holdout_rows,
    }

    split_dates = {
        "train": (TRAIN_START_DATE.date(), TRAIN_END_DATE.date()),
        "valid": (VALID_DATE.date(), VALID_DATE.date()),
        "holdout": (HOLDOUT_DATE.date(), HOLDOUT_DATE.date()),
    }

    rows_list: list[dict] = []

    for split_name, row_count in split_rows.items():
        start_date, end_date = split_dates[split_name]
        percentage = row_count / total_input_rows if total_input_rows > 0 else 0.0

        rows_list.append(
            {
                "split": split_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "rows": row_count,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(rows_list)


def save_summary(summary_df: pd.DataFrame) -> None:
    """保存划分摘要 CSV。"""

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n划分摘要已保存：{SUMMARY_CSV}")


def main() -> None:
    """主流程：逐文件划分 → 统计行数 → 一致性检查 → 保存摘要。"""

    print("=" * 70)
    print("训练集时间划分（train / valid / holdout）")
    print("=" * 70)

    parquet_files = list_parquet_files(INPUT_DIR)
    total_input_rows = get_total_input_rows(parquet_files)

    print(f"输入目录：     {INPUT_DIR}")
    print(f"输入文件数量： {len(parquet_files)}")
    print(f"输入总行数：   {total_input_rows:,}")
    print(f"划分规则：")
    print(f"  train：   {TRAIN_START_DATE.date()} ~ {TRAIN_END_DATE.date()}")
    print(f"  valid：   {VALID_DATE.date()}")
    print(f"  holdout： {HOLDOUT_DATE.date()}")
    print()

    totals = {
        "train_rows": 0,
        "valid_rows": 0,
        "holdout_rows": 0,
        "out_of_range_rows": 0,
        "invalid_hour_dt_rows": 0,
    }

    for input_path in parquet_files:
        print(f"正在处理：{input_path.name} ...")

        stats = process_single_file(
            input_path,
            OUTPUT_TRAIN_DIR,
            OUTPUT_VALID_DIR,
            OUTPUT_HOLDOUT_DIR,
        )

        for key in totals:
            if key in stats:
                totals[key] += stats[key]

        print(
            f"  输入 {stats['input_rows']:,} 行 → "
            f"train {stats['train_rows']:,}, "
            f"valid {stats['valid_rows']:,}, "
            f"holdout {stats['holdout_rows']:,}"
        )

        if stats["out_of_range_rows"] > 0:
            print(
                f"  WARNING: 本文件有 {stats['out_of_range_rows']:,} 行 "
                f"不在 {EXPECTED_RANGE_START.date()} ~ {EXPECTED_RANGE_END.date()} 范围内"
            )

        if stats["invalid_hour_dt_rows"] > 0:
            print(
                f"  WARNING: 本文件有 {stats['invalid_hour_dt_rows']:,} 行 "
                f"hour_dt 无法解析"
            )

    train_rows = totals["train_rows"]
    valid_rows = totals["valid_rows"]
    holdout_rows = totals["holdout_rows"]
    split_total = train_rows + valid_rows + holdout_rows

    print("\n" + "=" * 70)
    print("划分结果汇总")
    print("=" * 70)
    print(f"train 总行数：   {train_rows:,}")
    print(f"valid 总行数：   {valid_rows:,}")
    print(f"holdout 总行数： {holdout_rows:,}")
    print(f"三部分合计：     {split_total:,}")
    print(f"输入总行数：     {total_input_rows:,}")

    if totals["out_of_range_rows"] > 0:
        print(
            f"\nWARNING: 累计 {totals['out_of_range_rows']:,} 行日期不在期望范围内，"
            "未写入 train/valid/holdout。"
        )

    if totals["invalid_hour_dt_rows"] > 0:
        print(
            f"WARNING: 累计 {totals['invalid_hour_dt_rows']:,} 行 hour_dt 无效，"
            "未写入 train/valid/holdout。"
        )

    if split_total == total_input_rows:
        print("\n行数守恒检查：通过（train + valid + holdout = 输入总行数）")
    else:
        diff = total_input_rows - split_total
        print(
            f"\n行数守恒检查：未通过（差值 {diff:,} 行）。"
            "请检查是否存在超出日期范围或 hour_dt 无效的记录。"
        )

    summary_df = build_split_summary(
        train_rows=train_rows,
        valid_rows=valid_rows,
        holdout_rows=holdout_rows,
        total_input_rows=total_input_rows,
    )

    print("\n划分比例：")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['split']:>7}: {int(row['rows']):>12,} 行 "
            f"({row['percentage']:.2%})  "
            f"[{row['start_date']} ~ {row['end_date']}]"
        )

    save_summary(summary_df)

    print("\n输出目录：")
    print(f"  {OUTPUT_TRAIN_DIR}")
    print(f"  {OUTPUT_VALID_DIR}")
    print(f"  {OUTPUT_HOLDOUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
