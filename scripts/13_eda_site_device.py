"""
百度 CTR 项目 — site_category × device_type 交叉 EDA 脚本

功能：
    使用 Dask 分块读取清洗后的训练集 Parquet，
    先按曝光量选出 Top 10 的 site_category（主要网站类别编码），
    再与 device_type（设备类型编码）交叉分组统计曝光、点击与 CTR，
    并生成 CTR / 曝光量透视表，
    用于分析不同网站类别中各设备类型的点击表现是否存在差异。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv（整体 CTR 与总曝光量）

数据输出：
    outputs/eda_tables/site_device_summary.csv
    outputs/eda_tables/site_device_ctr_pivot.csv
    outputs/eda_tables/site_device_impressions_pivot.csv

说明：
    - 使用 Dask 懒加载，不会一次性把全量训练集读入内存
    - site_category、device_type 均为数据中的匿名编码，不做具体行业或设备名称解释
    - Top 10 site_category 必须按曝光量选取，不能按 CTR 选取
    - 缺失判断使用 isnull() / notnull()，不使用 notna()

用法：
    python scripts/13_eda_site_device.py
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
SUMMARY_CSV = OUTPUT_DIR / "site_device_summary.csv"
CTR_PIVOT_CSV = OUTPUT_DIR / "site_device_ctr_pivot.csv"
IMPRESSIONS_PIVOT_CSV = OUTPUT_DIR / "site_device_impressions_pivot.csv"

# 按曝光量保留曝光最高的前 N 个 site_category（不能按 CTR 选）
TOP_N_SITES = 10

# 低流量组合阈值：impressions 低于此值标记 is_low_volume=True（不删行）
LOW_VOLUME_THRESHOLD = 100_000


def load_overall_summary() -> tuple[float, int]:
    """
    从 overall_summary.csv 读取整体 CTR 与总曝光量。

    该文件由 scripts/03_eda_overall.py 生成，避免重复扫描全量 Parquet。
    overall_ctr 用于 ctr_vs_overall 对比；
    total_impressions 用于计算各组合的 impression_share（占全训练集比例）。
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


def load_train_site_device_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click、site_category、device_type。

    Dask 会按分块（partition）逐块处理数据，适合大数据量场景；
    这里只选交叉分析需要的三列，减少内存占用。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "site_category", "device_type"],
    )

    return dataframe


def count_missing_values(dataframe: dd.DataFrame) -> tuple[int, int]:
    """
    分别统计 site_category 与 device_type 的缺失/无效数量。

    site_category：统计 null 以及空字符串（清洗后可能出现的无效类别）。
    device_type：统计 null。
    使用 isnull() 判断缺失，不使用 notna()。

    返回：
        (site_category 缺失/无效数, device_type 缺失数)
    """

    site_category_missing = int(
        (
            dataframe["site_category"].isnull()
            | (dataframe["site_category"] == "")
        ).sum().compute()
    )
    device_type_missing = int(dataframe["device_type"].isnull().sum().compute())

    return site_category_missing, device_type_missing


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """
    计算 CTR（点击率 = 点击量 / 曝光量）。

    当 impressions 为 0 时返回 0.0，避免除零错误。
    """

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def safe_share(part: float | int, total: float | int) -> float:
    """
    计算占比（分子 / 分母）。

    当 total 为 0 时返回 0.0，避免除零错误。
    用于 impression_share 与 within_site_share。
    """

    if total > 0:
        return float(part) / float(total)

    return 0.0


def select_top_site_categories(valid_dataframe: dd.DataFrame, top_n: int) -> list:
    """
    按 site_category 汇总曝光量，选出曝光最高的前 top_n 个类别编码。

    必须按 impressions（曝光量）排序选取，不能按 CTR 选取，
    以保证分析聚焦在流量最大的主要类别上。

    返回：
        Top N 的 site_category 编码列表
    """

    site_grouped = (
        valid_dataframe.groupby("site_category")
        .agg(impressions=("site_category", "count"))
        .reset_index()
    )

    site_pdf = site_grouped.compute()

    top_sites = (
        site_pdf.sort_values("impressions", ascending=False)
        .head(top_n)["site_category"]
        .tolist()
    )

    return top_sites


def compute_site_device_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
    total_impressions: int,
    top_n_sites: int = TOP_N_SITES,
) -> tuple[pd.DataFrame, list]:
    """
    先选 Top N site_category，再按 site_category × device_type 交叉分组。

    步骤：
        1. 过滤 site_category、device_type 均有效的记录（不修改原始 Parquet）
        2. 按 site_category 曝光量选 Top N
        3. 只保留 Top N 类别，再与 device_type 交叉分组
        4. 计算 ctr、overall_ctr、ctr_vs_overall、impression_share、within_site_share
        5. 标记 is_low_volume（impressions < 100000，仅标记不删行）
        6. 按 site_category 总曝光降序，同类别内按 impressions 降序排序

    返回：
        (交叉汇总 pandas DataFrame, Top N site_category 列表)
    """

    # site_category 非 null 且非空字符串；device_type 非 null
    valid_mask = (
        dataframe["site_category"].notnull()
        & (dataframe["site_category"] != "")
        & dataframe["device_type"].notnull()
    )
    valid_dataframe = dataframe[valid_mask].copy()

    top_sites = select_top_site_categories(valid_dataframe, top_n=top_n_sites)

    # 只分析曝光量最大的前 top_n_sites 个 site_category
    filtered_dataframe = valid_dataframe[
        valid_dataframe["site_category"].isin(top_sites)
    ]

    grouped = (
        filtered_dataframe.groupby(["site_category", "device_type"])
        .agg(
            impressions=("site_category", "count"),
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

    # 占全训练集曝光比例（分母来自 overall_summary.csv）
    summary_pdf["impression_share"] = summary_pdf["impressions"].apply(
        lambda value: safe_share(value, total_impressions)
    )

    # 占该 site_category 内部曝光比例（观察设备构成）
    site_totals = summary_pdf.groupby("site_category")["impressions"].sum()
    summary_pdf["within_site_share"] = summary_pdf.apply(
        lambda row: safe_share(row["impressions"], site_totals[row["site_category"]]),
        axis=1,
    )

    # 低流量标记：样本量过小时 CTR 波动大，仅作参考，不删除任何组合
    summary_pdf["is_low_volume"] = summary_pdf["impressions"] < LOW_VOLUME_THRESHOLD

    # 按 site_category 总曝光降序，同类别内按组合曝光降序
    site_total_impressions = summary_pdf.groupby("site_category")["impressions"].transform("sum")
    summary_pdf = summary_pdf.assign(_site_total=site_total_impressions)
    summary_pdf = summary_pdf.sort_values(
        ["_site_total", "impressions"],
        ascending=[False, False],
    ).drop(columns=["_site_total"]).reset_index(drop=True)

    return summary_pdf, top_sites


def _site_category_order(summary_df: pd.DataFrame) -> list:
    """
    按 site_category 总曝光量从大到小排列类别顺序。

    用于透视表行索引排序，与汇总表排序逻辑一致。
    """

    return (
        summary_df.groupby("site_category")["impressions"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )


def build_ctr_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    将交叉汇总结果转为 CTR 透视表（在 Dask compute 之后的 pandas 上操作）。

    行：site_category（Top 10 主要类别编码）
    列：device_type（设备类型编码）
    值：ctr

    若某 site_category × device_type 组合不存在，对应单元格为 NaN。
    """

    pivot_df = summary_df.pivot(
        index="site_category",
        columns="device_type",
        values="ctr",
    )

    pivot_df = pivot_df.reindex(_site_category_order(summary_df))
    pivot_df.index.name = "site_category"
    pivot_df.columns.name = "device_type"

    return pivot_df


def build_impressions_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    将交叉汇总结果转为曝光量透视表。

    行：site_category
    列：device_type
    值：impressions
    """

    pivot_df = summary_df.pivot(
        index="site_category",
        columns="device_type",
        values="impressions",
    )

    pivot_df = pivot_df.reindex(_site_category_order(summary_df))
    pivot_df.index.name = "site_category"
    pivot_df.columns.name = "device_type"

    return pivot_df


def print_missing_warning(
    site_category_missing: int,
    device_type_missing: int,
    total_rows: int,
) -> None:
    """在终端打印 site_category 与 device_type 的缺失/无效数量。"""

    print("-" * 60)
    print("site_category / device_type 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：                  {total_rows:,}")
    print(f"site_category 缺失/空值(null)数：{site_category_missing:,}")
    print(f"device_type 缺失(null)数：        {device_type_missing:,}")

    either_missing = site_category_missing + device_type_missing
    if either_missing > 0:
        print("说明：任一侧无效的记录未纳入 site_category × device_type 交叉统计。")
    else:
        print("说明：未发现 site_category 或 device_type 缺失记录。")

    print("-" * 60)
    print()


def print_full_summary(summary_df: pd.DataFrame, overall_ctr: float, top_sites: list) -> None:
    """
    在终端打印完整交叉汇总结果（Top 10 site_category 内的全部组合）。

    CTR、impression_share、within_site_share 以百分比形式展示。
    """

    print("=" * 130)
    print(
        f"site_category × device_type 交叉统计 — Top {len(top_sites)} site_category "
        "（按曝光量选取；编码不做行业/设备名称解释）"
    )
    print("=" * 130)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.4%}")
    print(f"纳入分析的 site_category：{top_sites}")
    print(f"组合总数：             {len(summary_df):,}")
    print(f"低流量阈值：           impressions < {LOW_VOLUME_THRESHOLD:,} → is_low_volume=True")
    print("-" * 130)
    print(
        f"{'site_category':>20}  {'device_type':>11}  {'impressions':>14}  "
        f"{'clicks':>10}  {'ctr':>10}  {'impression_share':>16}  "
        f"{'within_site_share':>17}  {'ctr_vs_overall':>14}  {'is_low_volume':>13}"
    )
    print("-" * 130)

    for _, row in summary_df.iterrows():
        print(
            f"{str(row['site_category']):>20}  "
            f"{int(row['device_type']):>11}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.4%}  "
            f"{row['impression_share']:>16.4%}  "
            f"{row['within_site_share']:>17.4%}  "
            f"{row['ctr_vs_overall']:>+14.4%}  "
            f"{str(row['is_low_volume']):>13}"
        )

    print("=" * 130)


def save_results(
    summary_df: pd.DataFrame,
    ctr_pivot_df: pd.DataFrame,
    impressions_pivot_df: pd.DataFrame,
) -> None:
    """保存交叉汇总表、CTR 透视表与曝光量透视表。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "site_category",
        "device_type",
        "impressions",
        "clicks",
        "ctr",
        "overall_ctr",
        "ctr_vs_overall",
        "impression_share",
        "within_site_share",
        "is_low_volume",
    ]

    summary_df[output_columns].to_csv(SUMMARY_CSV, index=False)
    print(f"\n完整交叉汇总表已保存：{SUMMARY_CSV}")

    ctr_pivot_df.to_csv(CTR_PIVOT_CSV)
    print(f"CTR 透视表已保存：       {CTR_PIVOT_CSV}")

    impressions_pivot_df.to_csv(IMPRESSIONS_PIVOT_CSV)
    print(f"曝光量透视表已保存：     {IMPRESSIONS_PIVOT_CSV}")


def main() -> None:
    """主流程：读取整体汇总 → Dask 选 Top10 类别 → 交叉统计 → 透视表 → 打印 → 保存。"""

    print("正在读取整体 CTR 与总曝光量...")
    overall_ctr, total_impressions = load_overall_summary()
    print(f"整体 CTR：     {overall_ctr:.6%}")
    print(f"总曝光量：     {total_impressions:,}\n")

    print("正在用 Dask 读取训练集 Parquet（click、site_category、device_type）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_site_device_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    site_category_missing, device_type_missing = count_missing_values(train_ddf)
    print_missing_warning(site_category_missing, device_type_missing, total_rows)

    summary_df, top_sites = compute_site_device_summary(
        train_ddf,
        overall_ctr=overall_ctr,
        total_impressions=total_impressions,
        top_n_sites=TOP_N_SITES,
    )

    ctr_pivot_df = build_ctr_pivot(summary_df)
    impressions_pivot_df = build_impressions_pivot(summary_df)

    print_full_summary(summary_df, overall_ctr, top_sites)
    save_results(summary_df, ctr_pivot_df, impressions_pivot_df)


if __name__ == "__main__":
    main()
