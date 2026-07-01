"""
百度 CTR 项目 — 按小时 EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计 0—23 点每个小时的曝光量、点击量、CTR，
    并与整体 CTR 对比后保存汇总表。

数据输入：
    data/processed/train/*.parquet

数据输出：
    outputs/eda_tables/hourly_summary.csv

说明：
    仅读取 click 与 hour_dt 两列，不使用 id 或质量检查字段。

用法：
    python scripts/04_eda_hourly.py
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_PARQUET_GLOB = "data/processed/train/*.parquet"
OUTPUT_DIR = Path("outputs/eda_tables")
OUTPUT_CSV = OUTPUT_DIR / "hourly_summary.csv"

# 一天 24 个小时（0 点 ~ 23 点）
HOURS_OF_DAY = list(range(24))


def load_train_hourly_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click 与 hour_dt。

    hour_dt 是清洗脚本从 hour 字符串解析出的时间戳；
    本分析从中提取“几点钟”（0—23）用于分组统计。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "hour_dt"],
    )

    return dataframe


def count_invalid_hour_dt(dataframe: dd.DataFrame) -> int:
    """
    统计 hour_dt 缺失或无效（NaT）的记录数。

    这些记录在按小时分组时无法归类，需要单独计数并在终端提示。
    """

    invalid_count = int(dataframe["hour_dt"].isna().sum().compute())
    return invalid_count


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """计算 CTR， impressions 为 0 时返回 0.0，避免除零错误。"""

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def compute_hourly_summary(dataframe: dd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    按 hour_of_day 分组，计算每小时指标及整体 CTR。

    步骤：
        1. 过滤 hour_dt 有效的记录
        2. 提取 hour_of_day（0—23）
        3. 分组求 impressions（行数）与 clicks（求和）
        4. 补全 0—23 全部小时（无数据的 hour 记为 0）
        5. 计算每小时 ctr、整体 overall_ctr、与整体的差值 ctr_vs_overall

    返回：
        hourly_df   — 24 行汇总表（按 hour_of_day 升序）
        overall_ctr — 整体点击率（用于写入每行及终端展示）
    """

    # 只保留 hour_dt 有效的记录；无效记录已在 count_invalid_hour_dt 中单独统计
    valid_dataframe = dataframe[~dataframe["hour_dt"].isna()].copy()

    # 从时间戳中提取“小时”（0—23）
    valid_dataframe["hour_of_day"] = valid_dataframe["hour_dt"].dt.hour

    # 按小时分组：impressions = 行数，clicks = click 求和
    grouped = (
        valid_dataframe.groupby("hour_of_day")
        .agg(
            impressions=("hour_dt", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    hourly_pdf = grouped.compute()

    # 补全 0—23 全部小时，缺失小时 impressions/clicks 填 0
    hourly_pdf = (
        hourly_pdf.set_index("hour_of_day")
        .reindex(HOURS_OF_DAY, fill_value=0)
        .reset_index()
        .rename(columns={"index": "hour_of_day"})
    )

    # 若 reindex 后列名不是 hour_of_day，统一修正
    if "hour_of_day" not in hourly_pdf.columns:
        hourly_pdf = hourly_pdf.rename(columns={hourly_pdf.columns[0]: "hour_of_day"})

    hourly_pdf["impressions"] = hourly_pdf["impressions"].astype(int)
    hourly_pdf["clicks"] = hourly_pdf["clicks"].astype(int)

    # 每小时 CTR
    hourly_pdf["ctr"] = hourly_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    # 整体 CTR（有效 hour_dt 记录上的汇总）
    total_impressions = int(hourly_pdf["impressions"].sum())
    total_clicks = int(hourly_pdf["clicks"].sum())
    overall_ctr = safe_ctr(total_clicks, total_impressions)

    # 写入 overall_ctr，并计算与整体的差值
    hourly_pdf["overall_ctr"] = overall_ctr
    hourly_pdf["ctr_vs_overall"] = hourly_pdf["ctr"] - overall_ctr

    # 按 hour_of_day 升序排列（reindex 后通常已有序，此处再确保一次）
    hourly_pdf = hourly_pdf.sort_values("hour_of_day").reset_index(drop=True)

    return hourly_pdf, overall_ctr


def print_invalid_hour_warning(invalid_count: int, total_rows: int) -> None:
    """在终端提示 hour_dt 无效的记录数量，避免静默忽略。"""

    print("-" * 60)
    print("hour_dt 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：           {total_rows:,}")
    print(f"hour_dt 无效/缺失行数：  {invalid_count:,}")

    if invalid_count > 0:
        ratio = invalid_count / total_rows if total_rows > 0 else 0.0
        print(f"无效占比：               {ratio:.4%}")
        print("说明：以上记录未纳入按小时分组统计。")
    else:
        print("说明：未发现 hour_dt 无效或缺失的记录。")

    print("-" * 60)
    print()


def print_hourly_table(hourly_df: pd.DataFrame, overall_ctr: float) -> None:
    """在终端打印完整的 24 小时统计表。"""

    print("=" * 80)
    print("训练集按小时统计（0—23 点）")
    print("=" * 80)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.6%}")
    print("-" * 80)
    print(
        f"{'hour':>4}  {'impressions':>14}  {'clicks':>10}  "
        f"{'ctr':>10}  {'ctr_vs_overall':>14}"
    )
    print("-" * 80)

    for _, row in hourly_df.iterrows():
        print(
            f"{int(row['hour_of_day']):>4}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.6%}  "
            f"{row['ctr_vs_overall']:>+14.6%}"
        )

    print("=" * 80)


def save_hourly_summary(hourly_df: pd.DataFrame) -> None:
    """
    保存按小时汇总表。

    输出列：
        hour_of_day, impressions, clicks, ctr, overall_ctr, ctr_vs_overall
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "hour_of_day",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
    ]

    hourly_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"\n按小时汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：Dask 读取 → 无效 hour 计数 → 按小时统计 → 打印 → 保存。"""

    print("正在用 Dask 读取训练集 Parquet（click、hour_dt 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_hourly_dask()

    # 总行数与无效 hour_dt 计数（单独提示，不静默忽略）
    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    invalid_count = count_invalid_hour_dt(train_ddf)
    print_invalid_hour_warning(invalid_count, total_rows)

    hourly_df, overall_ctr = compute_hourly_summary(train_ddf)

    print_hourly_table(hourly_df, overall_ctr)
    save_hourly_summary(hourly_df)


if __name__ == "__main__":
    main()
