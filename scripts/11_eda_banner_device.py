"""
百度 CTR 项目 — banner_pos × device_type 交叉 EDA 脚本

功能：
    使用 Dask 读取清洗后的训练集 Parquet，
    按 banner_pos 与 device_type 交叉分组统计曝光、点击与 CTR，
    并生成 CTR 透视表，用于观察广告位 CTR 差异是否受设备类型构成影响。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv

数据输出：
    outputs/eda_tables/banner_device_summary.csv
    outputs/eda_tables/banner_device_ctr_pivot.csv

说明：
    - banner_pos、device_type 均为数据中的数值编码，不做具体页面位置或设备名称解释
    - 缺失判断使用 notnull() / isnull()，兼容当前 Dask 版本

用法：
    python scripts/11_eda_banner_device.py
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
SUMMARY_CSV = OUTPUT_DIR / "banner_device_summary.csv"
PIVOT_CSV = OUTPUT_DIR / "banner_device_ctr_pivot.csv"

# 终端打印曝光量最大的前 N 个组合
TOP_N_PRINT = 20

# 低流量组合阈值：impressions 低于此值标记 is_low_volume=True（不删行）
LOW_VOLUME_THRESHOLD = 100_000


def load_overall_summary() -> tuple[float, int]:
    """
    从 overall_summary.csv 读取整体 CTR 与总曝光量。

    用于 overall_ctr、ctr_vs_overall 及 impression_share 的分母。
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


def load_train_banner_device_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click、banner_pos、device_type。

    三列均为交叉分析所需的最小字段集，避免加载无关列占用内存。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "banner_pos", "device_type"],
    )

    return dataframe


def count_missing_values(dataframe: dd.DataFrame) -> tuple[int, int]:
    """
    分别统计 banner_pos 与 device_type 的缺失（null）数量。

    使用 isnull() 判断缺失，避免部分 Dask 版本对 notna() 的兼容问题。

    返回：
        (banner_pos 缺失数, device_type 缺失数)
    """

    banner_pos_missing = int(dataframe["banner_pos"].isnull().sum().compute())
    device_type_missing = int(dataframe["device_type"].isnull().sum().compute())

    return banner_pos_missing, device_type_missing


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


def compute_banner_device_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
    total_impressions: int,
) -> pd.DataFrame:
    """
    按 banner_pos × device_type 交叉分组，计算各组合指标。

    步骤：
        1. 过滤 banner_pos、device_type 均非 null 的记录
        2. 分组求 impressions、clicks
        3. 计算 ctr、overall_ctr、ctr_vs_overall、impression_share
        4. 标记 is_low_volume（impressions < 100000）
        5. 按 impressions 从大到小排序

    返回：
        pandas DataFrame（Dask 汇总完成后的小型结果）
    """

    valid_mask = dataframe["banner_pos"].notnull() & dataframe["device_type"].notnull()
    valid_dataframe = dataframe[valid_mask].copy()

    grouped = (
        valid_dataframe.groupby(["banner_pos", "device_type"])
        .agg(
            impressions=("banner_pos", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    summary_pdf = grouped.compute()

    summary_pdf["banner_pos"] = summary_pdf["banner_pos"].astype(int)
    summary_pdf["device_type"] = summary_pdf["device_type"].astype(int)
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

    # 低流量标记：仅作参考，不删除任何组合
    summary_pdf["is_low_volume"] = summary_pdf["impressions"] < LOW_VOLUME_THRESHOLD

    summary_pdf = summary_pdf.sort_values("impressions", ascending=False).reset_index(drop=True)

    return summary_pdf


def build_ctr_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    将交叉汇总结果转为 CTR 透视表。

    行：device_type
    列：banner_pos
    值：ctr

    若某 device_type × banner_pos 组合不存在，对应单元格为 NaN。
    """

    pivot_df = summary_df.pivot(
        index="device_type",
        columns="banner_pos",
        values="ctr",
    )

    # 索引与列名保持清晰（device_type 为行索引，banner_pos 为列名）
    pivot_df.index.name = "device_type"
    pivot_df.columns.name = "banner_pos"

    return pivot_df


def print_missing_warning(
    banner_pos_missing: int,
    device_type_missing: int,
    total_rows: int,
) -> None:
    """在终端打印 banner_pos 与 device_type 的缺失数量。"""

    print("-" * 60)
    print("banner_pos / device_type 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：            {total_rows:,}")
    print(f"banner_pos 缺失(null)数： {banner_pos_missing:,}")
    print(f"device_type 缺失(null)数：{device_type_missing:,}")

    either_missing = banner_pos_missing + device_type_missing
    if either_missing > 0:
        print("说明：任一侧为 null 的记录未纳入交叉分组统计。")
    else:
        print("说明：未发现 banner_pos 或 device_type 缺失记录。")

    print("-" * 60)
    print()


def print_top_combinations(
    summary_df: pd.DataFrame,
    overall_ctr: float,
    top_n: int = TOP_N_PRINT,
) -> None:
    """
    在终端打印曝光量最大的前 top_n 个 banner_pos × device_type 组合。

    CTR 与 impression_share 以百分比形式展示。
    """

    display_df = summary_df.head(top_n)

    print("=" * 120)
    print(
        f"banner_pos × device_type 交叉统计 — 曝光量 Top {top_n} "
        "（编码不做业务含义解释）"
    )
    print("=" * 120)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.4%}")
    print(f"组合总数：             {len(summary_df):,}（完整结果见 CSV）")
    print(f"低流量阈值：           impressions < {LOW_VOLUME_THRESHOLD:,} → is_low_volume=True")
    print("-" * 120)
    print(
        f"{'banner_pos':>10}  {'device_type':>11}  {'impressions':>14}  "
        f"{'clicks':>10}  {'ctr':>10}  {'impression_share':>16}  "
        f"{'ctr_vs_overall':>14}  {'is_low_volume':>13}"
    )
    print("-" * 120)

    for _, row in display_df.iterrows():
        print(
            f"{int(row['banner_pos']):>10}  "
            f"{int(row['device_type']):>11}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.4%}  "
            f"{row['impression_share']:>16.4%}  "
            f"{row['ctr_vs_overall']:>+14.4%}  "
            f"{str(row['is_low_volume']):>13}"
        )

    print("=" * 120)


def save_results(summary_df: pd.DataFrame, pivot_df: pd.DataFrame) -> None:
    """保存交叉汇总表与 CTR 透视表。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "banner_pos",
        "device_type",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
        "impression_share",
        "is_low_volume",
    ]

    summary_df[output_columns].to_csv(SUMMARY_CSV, index=False)
    print(f"\n完整交叉汇总表已保存：{SUMMARY_CSV}")

    pivot_df.to_csv(PIVOT_CSV)
    print(f"CTR 透视表已保存：     {PIVOT_CSV}")


def main() -> None:
    """主流程：读取整体汇总 → Dask 交叉统计 → 透视表 → 打印 Top20 → 保存 CSV。"""

    print("正在读取整体 CTR 与总曝光量...")
    overall_ctr, total_impressions = load_overall_summary()
    print(f"整体 CTR：     {overall_ctr:.6%}")
    print(f"总曝光量：     {total_impressions:,}\n")

    print("正在用 Dask 读取训练集 Parquet（click、banner_pos、device_type）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_banner_device_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    banner_pos_missing, device_type_missing = count_missing_values(train_ddf)
    print_missing_warning(banner_pos_missing, device_type_missing, total_rows)

    summary_df = compute_banner_device_summary(
        train_ddf,
        overall_ctr=overall_ctr,
        total_impressions=total_impressions,
    )

    pivot_df = build_ctr_pivot(summary_df)

    print_top_combinations(summary_df, overall_ctr, top_n=TOP_N_PRINT)
    save_results(summary_df, pivot_df)


if __name__ == "__main__":
    main()
