"""
百度 CTR 项目 — banner_pos × device_type CTR 热力图脚本

功能：
    读取 scripts/11 生成的 banner_device_summary.csv，
    使用 matplotlib 绘制 device_type（行）× banner_pos（列）的 CTR 热力图，
    低流量组合保留格子位置但不显示数值，并以浅灰色展示。

数据输入：
    outputs/eda_tables/banner_device_summary.csv

数据输出：
    outputs/eda_figures/banner_device_ctr_heatmap.png

说明：
    - 仅读取汇总 CSV，不修改原文件
    - 不使用 seaborn，仅使用 pandas + matplotlib
    - 缺失的 device_type × banner_pos 组合显示为 NaN，不会误显示为 CTR=0

用法：
    python scripts/14a_banner_device_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

INPUT_CSV = Path("outputs/eda_tables/banner_device_summary.csv")
OUTPUT_DIR = Path("outputs/eda_figures")
OUTPUT_PNG = OUTPUT_DIR / "banner_device_ctr_heatmap.png"

FIGURE_DPI = 150

# 绘图所需的 CSV 列名
REQUIRED_COLUMNS = [
    "banner_pos",
    "device_type",
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
            "请先运行：python scripts/11_eda_banner_device.py"
        )

    header_df = pd.read_csv(INPUT_CSV, nrows=0)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in header_df.columns]

    if missing_columns:
        missing_list = "、".join(missing_columns)
        raise ValueError(
            f"输入文件缺少必要字段：{missing_list}\n"
            f"请检查 {INPUT_CSV} 是否由 scripts/11_eda_banner_device.py 正确生成。"
        )


def load_summary() -> pd.DataFrame:
    """读取 banner_device 交叉汇总表。"""

    validate_input()
    return pd.read_csv(INPUT_CSV)


def build_pivot_matrices(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    将长表转为 CTR 与 is_low_volume 透视矩阵。

    行：device_type（按数值升序）
    列：banner_pos（按数值升序）
    数据中不存在的组合在透视后为 NaN，不会填充为 0。

    返回：
        ctr_pct_df       — CTR 百分比矩阵（ctr × 100）
        low_volume_df    — 低流量标记矩阵（布尔值）
        device_types     — 排序后的 device_type 列表
        banner_positions — 排序后的 banner_pos 列表
    """

    # 按数值顺序排列类别编码（图中作为类别标签显示，非连续数值轴）
    device_types = sorted(summary_df["device_type"].unique())
    banner_positions = sorted(summary_df["banner_pos"].unique())

    ctr_pivot = summary_df.pivot(
        index="device_type",
        columns="banner_pos",
        values="ctr",
    )
    low_volume_pivot = summary_df.pivot(
        index="device_type",
        columns="banner_pos",
        values="is_low_volume",
    )

    # reindex 补齐完整网格；缺失组合保持 NaN
    ctr_pivot = ctr_pivot.reindex(index=device_types, columns=banner_positions)
    low_volume_pivot = low_volume_pivot.reindex(index=device_types, columns=banner_positions)

    # CTR 转为百分比，便于色条与格子标注统一
    ctr_pct_df = ctr_pivot * 100

    return ctr_pct_df, low_volume_pivot, device_types, banner_positions


def plot_ctr_heatmap(
    ctr_pct_df: pd.DataFrame,
    low_volume_df: pd.DataFrame,
    device_types: list,
    banner_positions: list,
) -> plt.Figure:
    """
    绘制 CTR 热力图。

    规则：
        - 有数据且非低流量：按 CTR 着色，格内显示百分比（两位小数）
        - is_low_volume=True：保留格子，浅灰色，不显示数值
        - 缺失组合：浅灰色，不显示数值，不当作 CTR=0
    """

    # 缺失组合：透视表中 ctr 为 NaN
    is_missing = ctr_pct_df.isna()

    # 低流量标记；缺失格子的 is_low_volume 视为 False（已由 is_missing 单独处理）
    is_low_volume = low_volume_df.fillna(False).astype(bool)

    # 不参与着色的格子：缺失 或 低流量
    mask_for_color = is_missing | is_low_volume

    data_array = ctr_pct_df.to_numpy(dtype=float)
    masked_data = np.ma.masked_where(mask_for_color.to_numpy(), data_array)

    # 色条范围：仅根据“有数据且非低流量”的格子计算，避免极值干扰
    valid_values = ctr_pct_df.where(~mask_for_color)
    vmin = float(valid_values.min().min())
    vmax = float(valid_values.max().max())

    # 若全部格子都被掩蔽（极端情况），给一个默认范围避免报错
    if np.isnan(vmin) or np.isnan(vmax):
        vmin, vmax = 0.0, 100.0

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.YlOrRd.copy()
    # 被 mask 的格子（缺失 / 低流量）显示为浅灰色
    cmap.set_bad(color=MASKED_CELL_COLOR)

    heatmap = ax.imshow(
        masked_data,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )

    # 添加 colorbar，名称 CTR（值为百分比）
    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("CTR")

    # 轴刻度：按数值顺序的类别标签（字符串），不作为连续数值轴
    ax.set_xticks(range(len(banner_positions)))
    ax.set_xticklabels([str(value) for value in banner_positions])
    ax.set_yticks(range(len(device_types)))
    ax.set_yticklabels([str(value) for value in device_types])

    ax.set_xlabel("banner_pos")
    ax.set_ylabel("device_type")
    ax.set_title("CTR by Device Type and Banner Position")

    # 在有效且非低流量的格子上标注 CTR 百分比（两位小数）
    for row_idx, device_type in enumerate(device_types):
        for col_idx, banner_pos in enumerate(banner_positions):
            if is_missing.loc[device_type, banner_pos]:
                continue
            if is_low_volume.loc[device_type, banner_pos]:
                continue

            ctr_text = f"{ctr_pct_df.loc[device_type, banner_pos]:.2f}%"
            ax.text(
                col_idx,
                row_idx,
                ctr_text,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
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
    """主流程：校验输入 → 构建透视矩阵 → 绘制热力图 → 保存 PNG。"""

    print("=" * 60)
    print("banner_pos × device_type CTR 热力图")
    print("=" * 60)
    print(f"输入文件：{INPUT_CSV}\n")

    summary_df = load_summary()

    ctr_pct_df, low_volume_df, device_types, banner_positions = build_pivot_matrices(summary_df)

    fig = plot_ctr_heatmap(
        ctr_pct_df,
        low_volume_df,
        device_types,
        banner_positions,
    )

    save_figure(fig)
    print("\n热力图生成完毕。")


if __name__ == "__main__":
    main()
