"""
百度 CTR 项目 — hour_of_day × banner_pos CTR 热力图脚本

功能：
    读取 scripts/12 生成的 hour_banner_summary.csv，
    按各 banner_pos 总曝光量自动选出 Top 2 主要广告位编码，
    绘制 banner_pos（行）× hour_of_day（列，0—23）的 CTR 热力图。

数据输入：
    outputs/eda_tables/hour_banner_summary.csv

数据输出：
    outputs/eda_figures/hour_banner_ctr_heatmap.png

说明：
    - Top 2 banner_pos 由 impressions 汇总动态选取，不写死为 0 和 1
    - 仅读取汇总 CSV，不修改原文件
    - 不使用 seaborn，仅使用 pandas + matplotlib
    - 缺失组合显示为 NaN，不会误显示为 CTR=0

用法：
    python scripts/14b_hour_banner_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

INPUT_CSV = Path("outputs/eda_tables/hour_banner_summary.csv")
OUTPUT_DIR = Path("outputs/eda_figures")
OUTPUT_PNG = OUTPUT_DIR / "hour_banner_ctr_heatmap.png"

FIGURE_DPI = 150

# 自动选取曝光量最大的前 N 个 banner_pos
TOP_N_BANNERS = 2

# 一天 24 个小时（0 点 ~ 23 点），作为热力图列的完整顺序
HOURS_OF_DAY = list(range(24))

# 绘图所需的 CSV 列名
REQUIRED_COLUMNS = [
    "hour_of_day",
    "banner_pos",
    "impressions",
    "ctr",
    "is_low_volume",
]

# 低流量 / 缺失格子使用的浅灰色（不显示 CTR 数值）
MASKED_CELL_COLOR = "#E0E0E0"


def validate_input() -> None:
    """
    检查输入 CSV 是否存在，以及是否包含绘图所需的全部字段。

    若文件或字段缺失，抛出清晰的错误信息，提示用户先运行对应 EDA 脚本。
    """

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"未找到输入文件：{INPUT_CSV}\n"
            "请先运行：python scripts/12_eda_hour_banner.py"
        )

    header_df = pd.read_csv(INPUT_CSV, nrows=0)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in header_df.columns]

    if missing_columns:
        missing_list = "、".join(missing_columns)
        raise ValueError(
            f"输入文件缺少必要字段：{missing_list}\n"
            f"请检查 {INPUT_CSV} 是否由 scripts/12_eda_hour_banner.py 正确生成。"
        )


def load_summary() -> pd.DataFrame:
    """读取 hour_banner 交叉汇总表。"""

    validate_input()
    return pd.read_csv(INPUT_CSV)


def select_top_banner_positions(summary_df: pd.DataFrame, top_n: int) -> list:
    """
    按 banner_pos 汇总 impressions，选出曝光量最大的前 top_n 个编码。

    必须依据曝光量选取，不能写死具体数值，也不能按 CTR 选取。

    返回：
        按总曝光量从大到小排列的 banner_pos 列表
    """

    banner_totals = (
        summary_df.groupby("banner_pos")["impressions"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(banner_totals) < top_n:
        raise ValueError(
            f"汇总表中仅有 {len(banner_totals)} 个 banner_pos，"
            f"无法选取 Top {top_n}。请检查 {INPUT_CSV} 是否完整。"
        )

    return banner_totals.head(top_n).index.tolist()


def build_pivot_matrices(
    summary_df: pd.DataFrame,
    selected_banners: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    筛选 Top banner_pos 后，将长表转为 CTR 与 is_low_volume 透视矩阵。

    行：banner_pos（按总曝光量从大到小）
    列：hour_of_day（0—23 完整顺序）
    数据中不存在的组合在透视后为 NaN，不会填充为 0。

    返回：
        ctr_pct_df    — CTR 百分比矩阵（ctr × 100）
        low_volume_df — 低流量标记矩阵（布尔值）
        banner_rows   — 行顺序（选中的 banner_pos）
        hours         — 列顺序（0—23）
    """

    filtered_df = summary_df[summary_df["banner_pos"].isin(selected_banners)].copy()

    ctr_pivot = filtered_df.pivot(
        index="banner_pos",
        columns="hour_of_day",
        values="ctr",
    )
    low_volume_pivot = filtered_df.pivot(
        index="banner_pos",
        columns="hour_of_day",
        values="is_low_volume",
    )

    # 行按总曝光量降序（主要广告位在上）；列固定为 0—23
    ctr_pivot = ctr_pivot.reindex(index=selected_banners, columns=HOURS_OF_DAY)
    low_volume_pivot = low_volume_pivot.reindex(index=selected_banners, columns=HOURS_OF_DAY)

    ctr_pct_df = ctr_pivot * 100

    return ctr_pct_df, low_volume_pivot, selected_banners, HOURS_OF_DAY


def plot_ctr_heatmap(
    ctr_pct_df: pd.DataFrame,
    low_volume_df: pd.DataFrame,
    banner_rows: list,
    hours: list,
) -> plt.Figure:
    """
    绘制 CTR 热力图。

    规则：
        - 有数据且非低流量：按 CTR 着色，格内显示百分比（两位小数）
        - is_low_volume=True：保留格子，浅灰色，不显示数值
        - 缺失组合：浅灰色，不显示数值，不当作 CTR=0
    """

    is_missing = ctr_pct_df.isna()
    is_low_volume = low_volume_df.fillna(False).astype(bool)
    mask_for_color = is_missing | is_low_volume

    data_array = ctr_pct_df.to_numpy(dtype=float)
    masked_data = np.ma.masked_where(mask_for_color.to_numpy(), data_array)

    valid_values = ctr_pct_df.where(~mask_for_color)
    vmin = float(valid_values.min().min())
    vmax = float(valid_values.max().max())

    if np.isnan(vmin) or np.isnan(vmax):
        vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(14, 4))

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color=MASKED_CELL_COLOR)

    heatmap = ax.imshow(
        masked_data,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("CTR")

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([str(hour) for hour in hours])
    ax.set_yticks(range(len(banner_rows)))
    ax.set_yticklabels([str(banner) for banner in banner_rows])

    ax.set_xlabel("hour_of_day")
    ax.set_ylabel("banner_pos")
    ax.set_title("Hourly CTR by Major Banner Position")

    for row_idx, banner_pos in enumerate(banner_rows):
        for col_idx, hour in enumerate(hours):
            if is_missing.loc[banner_pos, hour]:
                continue
            if is_low_volume.loc[banner_pos, hour]:
                continue

            ctr_text = f"{ctr_pct_df.loc[banner_pos, hour]:.2f}%"
            ax.text(
                col_idx,
                row_idx,
                ctr_text,
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    return fig


def save_figure(fig: plt.Figure) -> Path:
    """保存图片：tight_layout → savefig(dpi=150) → plt.close()。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI)
    plt.close(fig)

    print(f"热力图已保存：{OUTPUT_PNG}")
    return OUTPUT_PNG


def main() -> None:
    """主流程：校验输入 → 选 Top2 banner_pos → 构建矩阵 → 绘图 → 保存。"""

    print("=" * 60)
    print("hour_of_day × banner_pos CTR 热力图")
    print("=" * 60)
    print(f"输入文件：{INPUT_CSV}\n")

    summary_df = load_summary()

    selected_banners = select_top_banner_positions(summary_df, top_n=TOP_N_BANNERS)
    print(f"按总曝光量选中的 Top {TOP_N_BANNERS} 个 banner_pos：{selected_banners}\n")

    ctr_pct_df, low_volume_df, banner_rows, hours = build_pivot_matrices(
        summary_df,
        selected_banners,
    )

    fig = plot_ctr_heatmap(
        ctr_pct_df,
        low_volume_df,
        banner_rows,
        hours,
    )

    save_figure(fig)
    print("\n热力图生成完毕。")


if __name__ == "__main__":
    main()
