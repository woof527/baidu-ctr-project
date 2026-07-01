"""
百度 CTR 项目 — 按日期 EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计每个日期的曝光量、点击量、CTR，
    并与整体 CTR 对比，观察不同日期的点击率差异。

数据输入：
    data/processed/train/*.parquet

数据输出：
    outputs/eda_tables/daily_summary.csv

说明：
    仅读取 click 与 hour_dt 两列；
    不使用 id、is_invalid_click、is_dup_id_within_chunk 等字段。

用法：
    python scripts/05_eda_daily.py
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
OUTPUT_CSV = OUTPUT_DIR / "daily_summary.csv"


def load_train_daily_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click 与 hour_dt。

    为什么用 Dask：
        训练集约 4000 万行，pandas 一次性读入会占用大量内存；
        Dask 按分块惰性计算，适合扫全量 Parquet。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "hour_dt"],
    )

    return dataframe


def count_invalid_hour_dt(dataframe: dd.DataFrame) -> int:
    """
    统计 hour_dt 缺失或无效（NaT）的记录数。

    这些记录无法提取日期，需单独计数并在终端提示。
    """

    invalid_count = int(dataframe["hour_dt"].isna().sum().compute())
    return invalid_count


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """计算 CTR；impressions 为 0 时返回 0.0，避免除零错误。"""

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def compute_daily_summary(dataframe: dd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    按 date 分组，计算每日指标及整体 CTR。

    步骤：
        1. 过滤 hour_dt 有效的记录
        2. 从 hour_dt 提取 date（YYYY-MM-DD 字符串）
        3. 分组求 impressions（行数）与 clicks（求和）
        4. 计算每日 ctr、整体 overall_ctr、与整体的差值 ctr_vs_overall
        5. 按日期从早到晚排序

    返回：
        daily_df    — 按日期汇总的 DataFrame
        overall_ctr — 整体点击率
    """

    # 只保留 hour_dt 有效的记录；无效记录在 count_invalid_hour_dt 中单独统计
    valid_dataframe = dataframe[~dataframe["hour_dt"].isna()].copy()

    # 从时间戳中提取日期（格式：2024-10-21，便于排序与阅读）
    valid_dataframe["date"] = valid_dataframe["hour_dt"].dt.strftime("%Y-%m-%d")

    # 按日期分组：impressions = 当日曝光行数，clicks = 当日 click 求和
    grouped = (
        valid_dataframe.groupby("date")
        .agg(
            impressions=("hour_dt", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    daily_pdf = grouped.compute()

    daily_pdf["impressions"] = daily_pdf["impressions"].astype(int)
    daily_pdf["clicks"] = daily_pdf["clicks"].astype(int)

    # 每日 CTR（含除零保护）
    daily_pdf["ctr"] = daily_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    # 整体 CTR（有效 hour_dt 记录上的汇总）
    total_impressions = int(daily_pdf["impressions"].sum())
    total_clicks = int(daily_pdf["clicks"].sum())
    overall_ctr = safe_ctr(total_clicks, total_impressions)

    # 写入 overall_ctr，并计算当日 CTR 与整体的差值
    daily_pdf["overall_ctr"] = overall_ctr
    daily_pdf["ctr_vs_overall"] = daily_pdf["ctr"] - overall_ctr

    # 按日期从早到晚排序（YYYY-MM-DD 字符串可直接字典序排序）
    daily_pdf = daily_pdf.sort_values("date").reset_index(drop=True)

    return daily_pdf, overall_ctr


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
        print("说明：以上记录未纳入按日期分组统计。")
    else:
        print("说明：未发现 hour_dt 无效或缺失的记录。")

    print("-" * 60)
    print()


def print_daily_table(daily_df: pd.DataFrame, overall_ctr: float) -> None:
    """在终端打印完整的按日期统计结果。"""

    print("=" * 85)
    print("训练集按日期统计")
    print("=" * 85)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.6%}")
    print(f"统计日期数：           {len(daily_df)}")
    print("-" * 85)
    print(
        f"{'date':>12}  {'impressions':>14}  {'clicks':>10}  "
        f"{'ctr':>10}  {'ctr_vs_overall':>14}"
    )
    print("-" * 85)

    for _, row in daily_df.iterrows():
        print(
            f"{row['date']:>12}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.6%}  "
            f"{row['ctr_vs_overall']:>+14.6%}"
        )

    print("=" * 85)


def save_daily_summary(daily_df: pd.DataFrame) -> None:
    """
    保存按日期汇总表。

    输出列：
        date, impressions, clicks, ctr, overall_ctr, ctr_vs_overall
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "date",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
    ]

    daily_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"\n按日期汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：Dask 读取 → 无效 hour 计数 → 按日期统计 → 打印 → 保存。"""

    print("正在用 Dask 读取训练集 Parquet（click、hour_dt 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_daily_dask()

    # 总行数与无效 hour_dt 计数（单独提示，不静默忽略）
    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    invalid_count = count_invalid_hour_dt(train_ddf)
    print_invalid_hour_warning(invalid_count, total_rows)

    daily_df, overall_ctr = compute_daily_summary(train_ddf)

    print_daily_table(daily_df, overall_ctr)
    save_daily_summary(daily_df)


if __name__ == "__main__":
    main()
