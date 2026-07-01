"""
百度 CTR 项目 — 按 device_type EDA 脚本

功能：
    使用 Dask 读取清洗后的全部训练集 Parquet 分块，
    统计各 device_type 的曝光量、点击量、CTR 及曝光占比，
    并与整体 CTR 对比，观察不同设备类型编码的点击表现差异。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv（读取整体 CTR 与总曝光量）

数据输出：
    outputs/eda_tables/device_type_summary.csv

说明：
    - 仅读取 click 与 device_type 两列
    - device_type 是数据中的数值编码，本脚本不对其做“手机/平板/电脑”等业务含义解释

用法：
    python scripts/07_eda_device_type.py
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
OUTPUT_CSV = OUTPUT_DIR / "device_type_summary.csv"


def load_overall_summary() -> tuple[float, int]:
    """
    从 overall_summary.csv 读取整体 CTR 与总曝光量。

    整体 CTR 用于 ctr_vs_overall 对比；
    总曝光量用于计算各 device_type 的 impression_share。
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


def load_train_device_type_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click 与 device_type。

    device_type 是数据集自带的设备类型编码（整型），
    本脚本仅做统计，不赋予额外业务标签。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "device_type"],
    )

    return dataframe


def count_missing_device_type(dataframe: dd.DataFrame) -> int:
    """
    统计 device_type 缺失（NA）的记录数。

    缺失记录无法归入具体设备类型编码，需单独计数并在终端提示。
    """

    missing_count = int(dataframe["device_type"].isna().sum().compute())
    return missing_count


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


def compute_device_type_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
    total_impressions: int,
) -> pd.DataFrame:
    """
    按 device_type 分组，计算各设备类型编码的指标。

    步骤：
        1. 过滤 device_type 非空的记录
        2. 分组求 impressions（行数）与 clicks（求和）
        3. 计算 ctr、overall_ctr、ctr_vs_overall、impression_share
        4. 按 impressions 从大到小排序

    返回：
        按曝光量降序排列的汇总 DataFrame
    """

    valid_dataframe = dataframe[~dataframe["device_type"].isna()].copy()

    grouped = (
        valid_dataframe.groupby("device_type")
        .agg(
            impressions=("device_type", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    summary_pdf = grouped.compute()

    summary_pdf["device_type"] = summary_pdf["device_type"].astype(int)
    summary_pdf["impressions"] = summary_pdf["impressions"].astype(int)
    summary_pdf["clicks"] = summary_pdf["clicks"].astype(int)

    summary_pdf["ctr"] = summary_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    summary_pdf["overall_ctr"] = overall_ctr
    summary_pdf["ctr_vs_overall"] = summary_pdf["ctr"] - overall_ctr

    # 曝光占比：该 device_type 曝光量 / 训练集总曝光量（来自 overall_summary）
    summary_pdf["impression_share"] = summary_pdf["impressions"].apply(
        lambda value: safe_share(value, total_impressions)
    )

    # 按曝光量从大到小排序，便于观察主要流量来源
    summary_pdf = summary_pdf.sort_values("impressions", ascending=False).reset_index(drop=True)

    return summary_pdf


def print_missing_device_type_warning(missing_count: int, total_rows: int) -> None:
    """在终端提示 device_type 缺失的记录数量。"""

    print("-" * 60)
    print("device_type 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：               {total_rows:,}")
    print(f"device_type 缺失行数：        {missing_count:,}")

    if missing_count > 0:
        ratio = missing_count / total_rows if total_rows > 0 else 0.0
        print(f"缺失占比：                    {ratio:.4%}")
        print("说明：以上记录未纳入按 device_type 分组统计。")
    else:
        print("说明：未发现 device_type 缺失的记录。")

    print("-" * 60)
    print()


def print_device_type_table(summary_df: pd.DataFrame, overall_ctr: float) -> None:
    """
    在终端打印完整的 device_type 统计结果。

    CTR 与 impression_share 均以百分比形式展示，便于阅读。
    """

    print("=" * 100)
    print("训练集按 device_type 统计（数值编码，不做业务含义解释）")
    print("=" * 100)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.4%}")
    print(f"device_type 编码数量： {len(summary_df)}")
    print("-" * 100)
    print(
        f"{'device_type':>11}  {'impressions':>14}  {'clicks':>10}  "
        f"{'ctr':>10}  {'impression_share':>16}  {'ctr_vs_overall':>14}"
    )
    print("-" * 100)

    for _, row in summary_df.iterrows():
        print(
            f"{int(row['device_type']):>11}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.4%}  "
            f"{row['impression_share']:>16.4%}  "
            f"{row['ctr_vs_overall']:>+14.4%}"
        )

    print("=" * 100)


def save_device_type_summary(summary_df: pd.DataFrame) -> None:
    """
    保存按 device_type 汇总表。

    输出列：
        device_type, impressions, clicks, ctr, overall_ctr,
        ctr_vs_overall, impression_share
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "device_type",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
        "impression_share",
    ]

    summary_df[output_columns].to_csv(OUTPUT_CSV, index=False)
    print(f"\n按 device_type 汇总表已保存：{OUTPUT_CSV}")


def main() -> None:
    """主流程：读取整体汇总 → Dask 加载 → 缺失计数 → 分组统计 → 打印 → 保存。"""

    print("正在读取整体 CTR 与总曝光量...")
    overall_ctr, total_impressions = load_overall_summary()
    print(f"整体 CTR：     {overall_ctr:.6%}")
    print(f"总曝光量：     {total_impressions:,}\n")

    print("正在用 Dask 读取训练集 Parquet（click、device_type 列）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_device_type_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    missing_count = count_missing_device_type(train_ddf)
    print_missing_device_type_warning(missing_count, total_rows)

    summary_df = compute_device_type_summary(
        train_ddf,
        overall_ctr=overall_ctr,
        total_impressions=total_impressions,
    )

    print_device_type_table(summary_df, overall_ctr)
    save_device_type_summary(summary_df)


if __name__ == "__main__":
    main()
