"""
百度 CTR 项目 — 按 app_category EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计各 app_category 的曝光量、点击量、CTR 及曝光占比，
    识别主要应用类别及其点击表现。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv（读取整体 CTR 与总曝光量）

数据输出：
    outputs/eda_tables/app_category_summary.csv（完整结果）

说明：
    - 仅读取 click 与 app_category 两列
    - app_category 是数据中的匿名类别编码，本脚本不解释其具体应用行业或名称
    - 过滤缺失值时使用 notnull()（兼容当前 Dask 版本）

用法：
    python scripts/09_eda_app_category.py
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
OVERALL_SUMMARY_CSV = OUTPUT_DIR / "overall_summary.csv"
OUTPUT_CSV = OUTPUT_DIR / "app_category_summary.csv"

# 终端只打印曝光量最大的前 N 个类别（完整结果仍写入 CSV）
TOP_N_PRINT = 15


def load_overall_summary() -> tuple[float, int]:
    """
    从 overall_summary.csv 读取整体 CTR 与总曝光量。

    用于 ctr_vs_overall 对比及 impression_share 的分母。
    """

    if not OVERALL_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"未找到整体汇总文件：{OVERALL_SUMMARY_CSV}\n"
            "请先运行：python scripts/03_eda_overall.py"
        )

    overall_df = pd.read_csv(OVERALL_SUMMARY_CSV)
    overall_ctr = float(overall_df.loc[0, "ctr"])
    total_impressions = int(overall_df.loc[0, "impressions"])

    return overall_ctr, total_impressions


def load_train_app_category_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click 与 app_category。

    app_category 是应用类别的匿名编码字符串，本脚本仅做统计汇总。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "app_category"],
    )

    return dataframe


def count_app_category_issues(dataframe: dd.DataFrame) -> tuple[int, int, int]:
    """
    分别统计 app_category 的缺失值（null）与空字符串数量。

    使用 notnull() 判断非空，避免部分 Dask 版本对 notna() 的兼容问题。

    返回：
        null_count    — app_category 为 null 的行数
        empty_count   — app_category 为空字符串 "" 的行数
        invalid_total — 以上两者之和（未纳入分组统计的行数）
    """

    null_count = int((~dataframe["app_category"].notnull()).sum().compute())

    empty_count = int(
        (dataframe["app_category"].notnull() & (dataframe["app_category"] == "")).sum().compute()
    )

    invalid_total = null_count + empty_count
    return null_count, empty_count, invalid_total


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """计算 CTR；impressions 为 0 时返回 0.0，避免除零错误。"""

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def safe_share(part: float | int, total: float | int) -> float:
    """计算曝光占比；total 为 0 时返回 0.0，避免除零错误。"""

    if total > 0:
        return float(part) / float(total)

    return 0.0


def compute_app_category_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
    total_impressions: int,
) -> pd.DataFrame:
    """
    按 app_category 分组，计算各类别指标。

    步骤：
        1. 过滤 app_category 非 null 且非空字符串的记录（使用 notnull()）
        2. 分组求 impressions（行数）与 clicks（求和）
        3. 计算 ctr、overall_ctr、ctr_vs_overall、impression_share
        4. 计算 exposure_rank（曝光量排名，1 = 最高）
        5. 按 impressions 从大到小排序

    返回：
        完整汇总 DataFrame（写入 CSV；终端仅展示前 TOP_N_PRINT 行）
    """

    valid_mask = dataframe["app_category"].notnull() & (dataframe["app_category"] != "")
    valid_dataframe = dataframe[valid_mask].copy()

    grouped = (
        valid_dataframe.groupby("app_category")
        .agg(
            impressions=("app_category", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    summary_pdf = grouped.compute()

    summary_pdf["impressions"] = summary_pdf["impressions"].astype(int)
    summary_pdf["clicks"] = summary_pdf["clicks"].astype(int)

    summary_pdf["ctr"] = summary_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    summary_pdf["overall_ctr"] = overall_ctr
    summary_pdf["ctr_vs_overall"] = summary_pdf["ctr"] - overall_ctr

    summary_pdf["impression_share"] = summary_pdf["impressions"].apply(
        lambda value: safe_share(value, total_impressions)
    )

    # 曝光量排名：impressions 越大，exposure_rank 越小（1 表示曝光最高）
    summary_pdf["exposure_rank"] = (
        summary_pdf["impressions"].rank(method="min", ascending=False).astype(int)
    )

    summary_pdf = summary_pdf.sort_values("impressions", ascending=False).reset_index(drop=True)

    return summary_pdf


def print_missing_app_category_warning(
    null_count: int,
    empty_count: int,
    total_rows: int,
) -> None:
    """在终端分别提示 app_category 缺失值与空字符串的数量。"""

    invalid_total = null_count + empty_count

    print("-" * 60)
    print("app_category 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：                 {total_rows:,}")
    print(f"app_category 缺失(null)行数：  {null_count:,}")
    print(f"app_category 空字符串行数：    {empty_count:,}")
    print(f"未纳入分组统计合计：           {invalid_total:,}")

    if invalid_total > 0:
        ratio = invalid_total / total_rows if total_rows > 0 else 0.0
        print(f"无效占比：                     {ratio:.4%}")
        print("说明：以上记录未纳入按 app_category 分组统计。")
    else:
        print("说明：未发现 app_category 缺失或空字符串记录。")

    print("-" * 60)
    print()


def print_top_app_categories(
    summary_df: pd.DataFrame,
    overall_ctr: float,
    top_n: int = TOP_N_PRINT,
) -> None:
    """
    在终端打印曝光量最大的前 top_n 个 app_category。

    CTR 与 impression_share 均以百分比形式展示。
    """

    display_df = summary_df.head(top_n)

    print("=" * 110)
    print(f"训练集 app_category 统计 — 曝光量 Top {top_n}（匿名编码，不做应用行业/名称解释）")
    print("=" * 110)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.4%}")
    print(f"类别总数：             {len(summary_df):,}（完整结果见 CSV）")
    print("-" * 110)
    print(
        f"{'exposure_rank':>13}  {'app_category':>20}  {'impressions':>14}  "
        f"{'clicks':>10}  {'ctr':>10}  {'impression_share':>16}  {'ctr_vs_overall':>14}"
    )
    print("-" * 110)

    for _, row in display_df.iterrows():
        print(
            f"{int(row['exposure_rank']):>13}  "
            f"{str(row['app_category']):>20}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.4%}  "
            f"{row['impression_share']:>16.4%}  "
            f"{row['ctr_vs_overall']:>+14.4%}"
        )

    print("=" * 110)


def save_app_category_summary(summary_df: pd.DataFrame) -> None:
    """
    保存完整的 app_category 汇总表。

    输出列：
        app_category, impressions, clicks, ctr, overall_ctr,
        ctr_vs_overall, impression_share, exposure_rank
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "app_category",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
        "impression_share",
        "exposure_rank",
    ]

    summary_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"\n完整 app_category 汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：读取整体汇总 → Dask 加载 → 缺失计数 → 分组统计 → 打印 Top15 → 保存完整 CSV。"""

    print("正在读取整体 CTR 与总曝光量...")
    overall_ctr, total_impressions = load_overall_summary()
    print(f"整体 CTR：     {overall_ctr:.6%}")
    print(f"总曝光量：     {total_impressions:,}\n")

    print("正在用 Dask 读取训练集 Parquet（click、app_category 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_app_category_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    null_count, empty_count, _invalid_total = count_app_category_issues(train_ddf)
    print_missing_app_category_warning(null_count, empty_count, total_rows)

    summary_df = compute_app_category_summary(
        train_ddf,
        overall_ctr=overall_ctr,
        total_impressions=total_impressions,
    )

    print_top_app_categories(summary_df, overall_ctr, top_n=TOP_N_PRINT)
    save_app_category_summary(summary_df)


if __name__ == "__main__":
    main()
