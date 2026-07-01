"""
百度 CTR 项目 — 按广告位 banner_pos EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计各 banner_pos 广告位置的曝光量、点击量、CTR，
    并与整体 CTR 对比。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv（读取整体 CTR）

数据输出：
    outputs/eda_tables/banner_pos_summary.csv

说明：
    仅读取 click 与 banner_pos 两列，不使用 id 或质量检查字段。

用法：
    python scripts/06_eda_banner_pos.py
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
OUTPUT_CSV = OUTPUT_DIR / "banner_pos_summary.csv"


def load_overall_ctr() -> float:
    """
    从 overall_summary.csv 读取整体 CTR。

    该文件由 scripts/03_eda_overall.py 生成，避免重复扫描全量数据。
    """

    if not OVERALL_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"未找到整体 CTR 文件：{OVERALL_SUMMARY_CSV}\n"
            "请先运行：python scripts/03_eda_overall.py"
        )

    overall_df = pd.read_csv(OVERALL_SUMMARY_CSV)
    return float(overall_df.loc[0, "ctr"])


def load_train_banner_pos_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click 与 banner_pos。

    banner_pos 表示广告在页面中的展示位置编号。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "banner_pos"],
    )

    return dataframe


def count_missing_banner_pos(dataframe: dd.DataFrame) -> int:
    """
    统计 banner_pos 缺失（NA）的记录数。

    缺失记录无法归入具体广告位，需单独计数并在终端提示。
    """

    missing_count = int(dataframe["banner_pos"].isna().sum().compute())
    return missing_count


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """计算 CTR；impressions 为 0 时返回 0.0，避免除零错误。"""

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def compute_banner_pos_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
) -> pd.DataFrame:
    """
    按 banner_pos 分组，计算各广告位指标。

    步骤：
        1. 过滤 banner_pos 非空的记录
        2. 分组求 impressions（行数）与 clicks（求和）
        3. 计算 ctr、overall_ctr、ctr_vs_overall
        4. 按 banner_pos 升序排序
        5. 计算曝光量排名 impressions_rank（1 = 曝光最高）

    返回：
        按 banner_pos 排序的汇总 DataFrame
    """

    # 只保留 banner_pos 有效的记录；缺失值已在 count_missing_banner_pos 中统计
    valid_dataframe = dataframe[~dataframe["banner_pos"].isna()].copy()

    grouped = (
        valid_dataframe.groupby("banner_pos")
        .agg(
            impressions=("banner_pos", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    summary_pdf = grouped.compute()

    summary_pdf["banner_pos"] = summary_pdf["banner_pos"].astype(int)
    summary_pdf["impressions"] = summary_pdf["impressions"].astype(int)
    summary_pdf["clicks"] = summary_pdf["clicks"].astype(int)

    # 各广告位 CTR（含除零保护）
    summary_pdf["ctr"] = summary_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    summary_pdf["overall_ctr"] = overall_ctr
    summary_pdf["ctr_vs_overall"] = summary_pdf["ctr"] - overall_ctr

    # 曝光量排名：impressions 越大，排名越靠前（1 表示曝光最高）
    summary_pdf["impressions_rank"] = (
        summary_pdf["impressions"].rank(method="min", ascending=False).astype(int)
    )

    # 按 banner_pos 从小到大排序（同时保留 impressions_rank 列）
    summary_pdf = summary_pdf.sort_values("banner_pos").reset_index(drop=True)

    return summary_pdf


def print_missing_banner_pos_warning(missing_count: int, total_rows: int) -> None:
    """在终端提示 banner_pos 缺失的记录数量。"""

    print("-" * 60)
    print("banner_pos 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：              {total_rows:,}")
    print(f"banner_pos 缺失行数：       {missing_count:,}")

    if missing_count > 0:
        ratio = missing_count / total_rows if total_rows > 0 else 0.0
        print(f"缺失占比：                  {ratio:.4%}")
        print("说明：以上记录未纳入按 banner_pos 分组统计。")
    else:
        print("说明：未发现 banner_pos 缺失的记录。")

    print("-" * 60)
    print()


def print_banner_pos_table(summary_df: pd.DataFrame, overall_ctr: float) -> None:
    """在终端打印完整的 banner_pos 统计结果。"""

    print("=" * 95)
    print("训练集按广告位 banner_pos 统计")
    print("=" * 95)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.6%}")
    print(f"广告位数量：           {len(summary_df)}")
    print("-" * 95)
    print(
        f"{'banner_pos':>10}  {'impressions':>14}  {'impressions_rank':>16}  "
        f"{'clicks':>10}  {'ctr':>10}  {'ctr_vs_overall':>14}"
    )
    print("-" * 95)

    for _, row in summary_df.iterrows():
        print(
            f"{int(row['banner_pos']):>10}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['impressions_rank']):>16}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.6%}  "
            f"{row['ctr_vs_overall']:>+14.6%}"
        )

    print("=" * 95)


def save_banner_pos_summary(summary_df: pd.DataFrame) -> None:
    """
    保存按 banner_pos 汇总表。

    输出列：
        banner_pos, impressions, clicks, ctr, overall_ctr,
        ctr_vs_overall, impressions_rank
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "banner_pos",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
        "impressions_rank",
    ]

    summary_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"\n按广告位汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：读取整体 CTR → Dask 加载 → 缺失计数 → 分组统计 → 打印 → 保存。"""

    print("正在读取整体 CTR...")
    overall_ctr = load_overall_ctr()
    print(f"整体 CTR：{overall_ctr:.6%}\n")

    print("正在用 Dask 读取训练集 Parquet（click、banner_pos 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_banner_pos_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    missing_count = count_missing_banner_pos(train_ddf)
    print_missing_banner_pos_warning(missing_count, total_rows)

    summary_df = compute_banner_pos_summary(train_ddf, overall_ctr)

    print_banner_pos_table(summary_df, overall_ctr)
    save_banner_pos_summary(summary_df)


if __name__ == "__main__":
    main()
