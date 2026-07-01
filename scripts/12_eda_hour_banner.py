"""
百度 CTR 项目 — hour_of_day × banner_pos 交叉 EDA 脚本

功能：
    使用 Dask 分块读取清洗后的训练集 Parquet，
    按“一天中的第几小时”（0—23 点）与广告位编码 banner_pos 交叉分组，
    统计各组合的曝光量、点击量、CTR，并生成 CTR 透视表，
    用于观察不同时间段内各广告位置的点击表现差异。

数据输入：
    data/processed/train/*.parquet
    outputs/eda_tables/overall_summary.csv（整体 CTR 与总曝光量）

数据输出：
    outputs/eda_tables/hour_banner_summary.csv
    outputs/eda_tables/hour_banner_ctr_pivot.csv

说明：
    - 使用 Dask 懒加载，不会一次性把全量训练集读入内存
    - hour_dt 是清洗脚本解析出的时间戳；本脚本从中提取 hour_of_day（0—23）
    - banner_pos 是数据中的广告位数值编码，本脚本不做具体页面位置解释
    - 缺失判断使用 isnull() / notnull()，不使用 notna()

用法：
    python scripts/12_eda_hour_banner.py
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
SUMMARY_CSV = OUTPUT_DIR / "hour_banner_summary.csv"
PIVOT_CSV = OUTPUT_DIR / "hour_banner_ctr_pivot.csv"

# 终端打印曝光量最大的前 N 个 hour_of_day × banner_pos 组合
TOP_N_PRINT = 30

# 低流量组合阈值：impressions 低于此值标记 is_low_volume=True（不删行）
LOW_VOLUME_THRESHOLD = 100_000


def load_overall_summary() -> tuple[float, int]:
    """
    从 overall_summary.csv 读取整体 CTR 与总曝光量。

    该文件由 scripts/03_eda_overall.py 生成，避免重复扫描全量 Parquet。
    overall_ctr 用于 ctr_vs_overall 对比；
    total_impressions 用于计算各组合的 impression_share（曝光占比）。
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


def load_train_hour_banner_dask() -> dd.DataFrame:
    """
    用 Dask 懒加载训练集 Parquet，只读取 click、hour_dt、banner_pos。

    Dask 会按分块（partition）逐块处理数据，适合大数据量场景；
    这里只选交叉分析需要的三列，减少内存占用。
    """

    dataframe = dd.read_parquet(
        TRAIN_PARQUET_GLOB,
        columns=["click", "hour_dt", "banner_pos"],
    )

    return dataframe


def count_missing_values(dataframe: dd.DataFrame) -> tuple[int, int]:
    """
    分别统计 hour_dt 与 banner_pos 的缺失（null / NaT）数量。

    使用 isnull() 判断缺失，兼容当前 Dask 版本，不使用 notna()。

    返回：
        (hour_dt 缺失数, banner_pos 缺失数)
    """

    hour_dt_missing = int(dataframe["hour_dt"].isnull().sum().compute())
    banner_pos_missing = int(dataframe["banner_pos"].isnull().sum().compute())

    return hour_dt_missing, banner_pos_missing


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
    计算曝光占比（某组合曝光量 / 训练集总曝光量）。

    当 total 为 0 时返回 0.0，避免除零错误。
    """

    if total > 0:
        return float(part) / float(total)

    return 0.0


def compute_hour_banner_summary(
    dataframe: dd.DataFrame,
    overall_ctr: float,
    total_impressions: int,
) -> pd.DataFrame:
    """
    按 hour_of_day × banner_pos 交叉分组，计算各组合指标。

    步骤：
        1. 过滤 hour_dt、banner_pos 均非 null 的记录（不修改原始 Parquet）
        2. 从 hour_dt 提取 hour_of_day（0—23）
        3. 分组求 impressions（行数）、clicks（click 求和）
        4. 计算 ctr、overall_ctr、ctr_vs_overall、impression_share
        5. 标记 is_low_volume（impressions < 100000，仅标记不删行）
        6. 按 hour_of_day、banner_pos 从小到大排序

    返回：
        pandas DataFrame（Dask 汇总完成后的小型结果，可安全放入内存）
    """

    # 两侧字段都有效才纳入交叉统计
    valid_mask = dataframe["hour_dt"].notnull() & dataframe["banner_pos"].notnull()
    valid_dataframe = dataframe[valid_mask].copy()

    # 从时间戳中提取“几点钟”，取值 0—23（例如 14 表示下午 2 点）
    valid_dataframe["hour_of_day"] = valid_dataframe["hour_dt"].dt.hour

    grouped = (
        valid_dataframe.groupby(["hour_of_day", "banner_pos"])
        .agg(
            impressions=("hour_dt", "count"),
            clicks=("click", "sum"),
        )
        .reset_index()
    )

    # compute() 触发 Dask 实际计算，得到小型 pandas 结果表
    summary_pdf = grouped.compute()

    summary_pdf["hour_of_day"] = summary_pdf["hour_of_day"].astype(int)
    summary_pdf["banner_pos"] = summary_pdf["banner_pos"].astype(int)
    summary_pdf["impressions"] = summary_pdf["impressions"].astype(int)
    summary_pdf["clicks"] = summary_pdf["clicks"].astype(int)

    summary_pdf["ctr"] = summary_pdf.apply(
        lambda row: safe_ctr(row["clicks"], row["impressions"]),
        axis=1,
    )

    summary_pdf["overall_ctr"] = overall_ctr
    summary_pdf["ctr_vs_overall"] = summary_pdf["ctr"] - overall_ctr

    # 曝光占比：该组合曝光量 / 训练集总曝光量（分母来自 overall_summary.csv）
    summary_pdf["impression_share"] = summary_pdf["impressions"].apply(
        lambda value: safe_share(value, total_impressions)
    )

    # 低流量标记：样本量过小时 CTR 波动大，仅作参考，不删除任何组合
    summary_pdf["is_low_volume"] = summary_pdf["impressions"] < LOW_VOLUME_THRESHOLD

    # 保存 CSV 时按 hour_of_day、banner_pos 升序排列
    summary_pdf = summary_pdf.sort_values(
        ["hour_of_day", "banner_pos"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary_pdf


def build_ctr_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    将交叉汇总结果转为 CTR 透视表（需在 Dask compute 之后，在 pandas 上操作）。

    行：hour_of_day（0—23 点）
    列：banner_pos（广告位编码）
    值：ctr

    若某 hour_of_day × banner_pos 组合在数据中不存在，对应单元格为 NaN。
    """

    pivot_df = summary_df.pivot(
        index="hour_of_day",
        columns="banner_pos",
        values="ctr",
    )

    pivot_df.index.name = "hour_of_day"
    pivot_df.columns.name = "banner_pos"

    return pivot_df


def print_missing_warning(
    hour_dt_missing: int,
    banner_pos_missing: int,
    total_rows: int,
) -> None:
    """在终端打印 hour_dt 与 banner_pos 的缺失数量，避免静默忽略。"""

    print("-" * 60)
    print("hour_dt / banner_pos 数据质量提示")
    print("-" * 60)
    print(f"训练集总行数：          {total_rows:,}")
    print(f"hour_dt 缺失(null)数：  {hour_dt_missing:,}")
    print(f"banner_pos 缺失(null)数：{banner_pos_missing:,}")

    either_missing = hour_dt_missing + banner_pos_missing
    if either_missing > 0:
        print("说明：任一侧为 null 的记录未纳入 hour_of_day × banner_pos 交叉统计。")
    else:
        print("说明：未发现 hour_dt 或 banner_pos 缺失记录。")

    print("-" * 60)
    print()


def print_top_combinations(
    summary_df: pd.DataFrame,
    overall_ctr: float,
    top_n: int = TOP_N_PRINT,
) -> None:
    """
    在终端打印曝光量最大的前 top_n 个 hour_of_day × banner_pos 组合。

    打印顺序按 impressions 降序；完整 CSV 仍按 hour_of_day、banner_pos 升序保存。
    CTR 与 impression_share 以百分比形式展示，便于阅读。
    """

    # 按曝光量从大到小取 Top N，不影响 summary_df 本身的排序
    display_df = summary_df.sort_values("impressions", ascending=False).head(top_n)

    print("=" * 120)
    print(
        f"hour_of_day × banner_pos 交叉统计 — 曝光量 Top {top_n} "
        "（banner_pos 为数值编码，不做页面位置解释）"
    )
    print("=" * 120)
    print(f"整体 CTR (overall_ctr)：{overall_ctr:.4%}")
    print(f"组合总数：             {len(summary_df):,}（完整结果见 CSV）")
    print(f"低流量阈值：           impressions < {LOW_VOLUME_THRESHOLD:,} → is_low_volume=True")
    print("-" * 120)
    print(
        f"{'hour_of_day':>11}  {'banner_pos':>10}  {'impressions':>14}  "
        f"{'clicks':>10}  {'ctr':>10}  {'impression_share':>16}  "
        f"{'ctr_vs_overall':>14}  {'is_low_volume':>13}"
    )
    print("-" * 120)

    for _, row in display_df.iterrows():
        print(
            f"{int(row['hour_of_day']):>11}  "
            f"{int(row['banner_pos']):>10}  "
            f"{int(row['impressions']):>14,}  "
            f"{int(row['clicks']):>10,}  "
            f"{row['ctr']:>10.4%}  "
            f"{row['impression_share']:>16.4%}  "
            f"{row['ctr_vs_overall']:>+14.4%}  "
            f"{str(row['is_low_volume']):>13}"
        )

    print("=" * 120)


def save_results(summary_df: pd.DataFrame, pivot_df: pd.DataFrame) -> None:
    """保存交叉汇总表与 CTR 透视表到 outputs/eda_tables/。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "hour_of_day",
        "banner_pos",
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
    """主流程：读取整体汇总 → Dask 交叉统计 → 透视表 → 打印 Top30 → 保存 CSV。"""

    print("正在读取整体 CTR 与总曝光量...")
    overall_ctr, total_impressions = load_overall_summary()
    print(f"整体 CTR：     {overall_ctr:.6%}")
    print(f"总曝光量：     {total_impressions:,}\n")

    print("正在用 Dask 读取训练集 Parquet（click、hour_dt、banner_pos）...")
    print(f"数据路径：{TRAIN_PARQUET_GLOB}\n")

    train_ddf = load_train_hour_banner_dask()

    total_rows = int(train_ddf.map_partitions(len).sum().compute())
    hour_dt_missing, banner_pos_missing = count_missing_values(train_ddf)
    print_missing_warning(hour_dt_missing, banner_pos_missing, total_rows)

    summary_df = compute_hour_banner_summary(
        train_ddf,
        overall_ctr=overall_ctr,
        total_impressions=total_impressions,
    )

    pivot_df = build_ctr_pivot(summary_df)

    print_top_combinations(summary_df, overall_ctr, top_n=TOP_N_PRINT)
    save_results(summary_df, pivot_df)


if __name__ == "__main__":
    main()
