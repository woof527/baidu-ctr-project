"""
百度 CTR 项目 — 训练集整体 EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计总曝光量、总点击量与整体 CTR，并保存汇总表。

数据输入：
    data/processed/train/*.parquet

数据输出：
    outputs/eda_tables/overall_summary.csv

说明：
    本脚本不使用 id、is_invalid_click、is_dup_id_within_chunk 等质量检查字段，
    仅读取 click 列用于汇总统计。

用法：
    python scripts/03_eda_overall.py
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

# 清洗后训练集 Parquet 所在目录（支持多个 part-XXXX.parquet 分块）
TRAIN_PARQUET_GLOB = "data/processed/train/*.parquet"

# 汇总结果输出路径
OUTPUT_DIR = Path("outputs/eda_tables")
OUTPUT_CSV = OUTPUT_DIR / "overall_summary.csv"


def load_train_click_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载全部训练集 Parquet，只读取 click 列。

    为什么只用 Dask、只读 click：
        - 训练集约 4000 万行，pandas 一次性读入会占用大量内存
        - Dask 按分块并行/惰性计算，适合扫全量 Parquet
        - 整体 CTR 只需要 click 列，无需加载 id 或质量检查字段
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click"],
    )

    return dataframe


def compute_overall_metrics(dataframe: dd.DataFrame) -> dict[str, float | int]:
    """
    计算整体曝光、点击与 CTR。

    指标定义：
        impressions — 训练集总行数（每条记录代表一次广告曝光）
        clicks      — click 列求和（点击次数）
        ctr         — clicks / impressions（整体点击率）

    返回：
        包含 impressions、clicks、ctr 三个键的字典
    """

    # 总曝光量：对各分块行数求和（等价于全表行数）
    impressions_lazy = dataframe.map_partitions(len).sum()

    # 总点击量：click 列求和（NA 会被 pandas/dask 默认跳过）
    clicks_lazy = dataframe["click"].sum()

    # 触发 Dask 计算（一次调度，避免重复扫描）
    impressions, clicks = dd.compute(impressions_lazy, clicks_lazy)

    impressions = int(impressions)
    clicks = int(clicks)

    # 除数为 0 保护：若无曝光记录，CTR 记为 0.0
    if impressions > 0:
        ctr = clicks / impressions
    else:
        ctr = 0.0

    return {
        "impressions": impressions,
        "clicks": clicks,
        "ctr": ctr,
    }


def print_summary(metrics: dict[str, float | int]) -> None:
    """在终端打印整体统计结果，便于快速查看。"""

    print("=" * 60)
    print("训练集整体数据概况（清洗后 Parquet）")
    print("=" * 60)
    print(f"总曝光量 (impressions)：{metrics['impressions']:,}")
    print(f"总点击量 (clicks)：     {metrics['clicks']:,}")
    print(f"整体 CTR (ctr)：        {metrics['ctr']:.6%}")
    print("=" * 60)


def save_summary(metrics: dict[str, float | int]) -> None:
    """
    将汇总结果保存为 CSV。

    输出列：impressions, clicks, ctr（单行汇总表）
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(
        [
            {
                "impressions": metrics["impressions"],
                "clicks": metrics["clicks"],
                "ctr": metrics["ctr"],
            }
        ]
    )

    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：Dask 读取 → 计算指标 → 打印 → 保存 CSV。"""

    print("正在用 Dask 读取训练集 Parquet（仅 click 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_click_dask()
    metrics = compute_overall_metrics(train_ddf)

    print_summary(metrics)
    save_summary(metrics)


if __name__ == "__main__":
    main()
