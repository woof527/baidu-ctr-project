"""
百度 CTR 项目 — site_category × device_type CTR 热力图脚本

功能：
    读取 scripts/13 生成的 site_device_summary.csv，
    若 site_category 超过 10 个则按总曝光量保留 Top 10，
    绘制 site_category（行）× device_type（列）的 CTR 热力图。

数据输入：
    outputs/eda_tables/site_device_summary.csv

数据输出：
    outputs/eda_figures/site_device_ctr_heatmap.png

说明：
    - site_category、device_type 均为匿名编码，不做具体行业或设备名称解释
    - 仅读取汇总 CSV，不修改原文件
    - 不使用 seaborn，仅使用 pandas + matplotlib
    - 缺失组合显示为 NaN，不会误显示为 CTR=0

用法：
    python scripts/14c_site_device_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

INPUT_CSV = Path("outputs/eda_tables/site_device_summary.csv")
OUTPUT_DIR = Path("outputs/eda_figures")
OUTPUT_PNG = OUTPUT_DIR / "site_device_ctr_heatmap.png"

FIGURE_DPI = 150

# 热力图最多展示的 site_category 数量（按总曝光量选取）
MAX_SITE_CATEGORIES = 10

# 绘图所需的 CSV 列名
REQUIRED_COLUMNS = [
    "site_category",
    "device_type",
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
            "请先运行：python scripts/13_eda_site_device.py"
        )

    header_df = pd.read_csv(INPUT_CSV, nrows=0)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in header_df.columns]

    if missing_columns:
        missing_list = "、".join(missing_columns)
        raise ValueError(
            f"输入文件缺少必要字段：{missing_list}\n"
            f"请检查 {INPUT_CSV} 是否由 scripts/13_eda_site_device.py 正确生成。"
        )


def load_summary() -> pd.DataFrame:
    """读取 site_device 交叉汇总表。"""

    validate_input()
    return pd.read_csv(INPUT_CSV)


def select_site_categories(summary_df: pd.DataFrame, max_categories: int) -> list:
    """
    按 site_category 汇总 impressions，确定热力图行顺序。

    若 site_category 数量超过 max_categories，只保留总曝光量最大的前 max_categories 个；
    否则使用文件中全部 site_category。
    行顺序始终为总曝光量从大到小。

    返回：
        按总曝光量降序排列的 site_category 列表
    """

    site_totals = (
        summary_df.groupby("site_category")["impressions"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(site_totals) > max_categories:
        return site_totals.head(max_categories).index.tolist()

    return site_totals.index.tolist()


def build_pivot_matrices(
    summary_df: pd.DataFrame,
    selected_sites: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    将长表转为 CTR 与 is_low_volume 透视矩阵。

    行：site_category（按总曝光量从大到小）
    列：device_type（按数值升序）
    数据中不存在的组合在透视后为 NaN，不会填充为 0。

    返回：
        ctr_pct_df    — CTR 百分比矩阵（ctr × 100）
        low_volume_df — 低流量标记矩阵（布尔值）
        site_rows     — 行顺序（选中的 site_category）
        device_cols   — 列顺序（device_type 数值升序）
    """

    filtered_df = summary_df[summary_df["site_category"].isin(selected_sites)].copy()

    device_cols = sorted(filtered_df["device_type"].unique())

    ctr_pivot = filtered_df.pivot(
        index="site_category",
        columns="device_type",
        values="ctr",
    )
    low_volume_pivot = filtered_df.pivot(
        index="site_category",
        columns="device_type",
        values="is_low_volume",
    )

    ctr_pivot = ctr_pivot.reindex(index=selected_sites, columns=device_cols)
    low_volume_pivot = low_volume_pivot.reindex(index=selected_sites, columns=device_cols)

    ctr_pct_df = ctr_pivot * 100

    return ctr_pct_df, low_volume_pivot, selected_sites, device_cols


def plot_ctr_heatmap(
    ctr_pct_df: pd.DataFrame,
    low_volume_df: pd.DataFrame,
    site_rows: list,
    device_cols: list,
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

    fig, ax = plt.subplots(figsize=(10, max(6, len(site_rows) * 0.5)))

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

    ax.set_xticks(range(len(device_cols)))
    ax.set_xticklabels([str(value) for value in device_cols])
    ax.set_yticks(range(len(site_rows)))
    ax.set_yticklabels([str(value) for value in site_rows])

    ax.set_xlabel("device_type")
    ax.set_ylabel("site_category")
    ax.set_title("CTR by Site Category and Device Type")

    for row_idx, site_category in enumerate(site_rows):
        for col_idx, device_type in enumerate(device_cols):
            if is_missing.loc[site_category, device_type]:
                continue
            if is_low_volume.loc[site_category, device_type]:
                continue

            ctr_text = f"{ctr_pct_df.loc[site_category, device_type]:.2f}%"
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
    """主流程：校验输入 → 选 site_category → 构建矩阵 → 绘图 → 保存。"""

    print("=" * 60)
    print("site_category × device_type CTR 热力图")
    print("=" * 60)
    print(f"输入文件：{INPUT_CSV}\n")

    summary_df = load_summary()

    selected_sites = select_site_categories(summary_df, max_categories=MAX_SITE_CATEGORIES)
    print(
        f"纳入热力图的 site_category 数量：{len(selected_sites)}"
        f"（最多 {MAX_SITE_CATEGORIES} 个，按总曝光量选取）"
    )
    print(f"site_category 行顺序（曝光量降序）：{selected_sites}\n")

    ctr_pct_df, low_volume_df, site_rows, device_cols = build_pivot_matrices(
        summary_df,
        selected_sites,
    )

    fig = plot_ctr_heatmap(
        ctr_pct_df,
        low_volume_df,
        site_rows,
        device_cols,
    )

    save_figure(fig)
    print("\n热力图生成完毕。")


if __name__ == "__main__":
    main()
