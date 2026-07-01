"""
百度 CTR 项目 — EDA 汇总表可视化脚本

功能：
    读取 outputs/eda_tables/ 下已生成的小型汇总 CSV，
    使用 pandas + matplotlib 绘制 CTR 分析图，
    保存到 outputs/eda_figures/。

说明：
    - 汇总表体积小，直接用 pandas 读取即可，无需 Dask
    - 不修改任何原有 CSV 文件

用法：
    python scripts/10_eda_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

EDA_TABLES_DIR = Path("outputs/eda_tables")
FIGURES_DIR = Path("outputs/eda_figures")

# 所有需要读取的输入文件（键名便于错误提示）
INPUT_FILES: dict[str, Path] = {
    "overall_summary": EDA_TABLES_DIR / "overall_summary.csv",
    "hourly_summary": EDA_TABLES_DIR / "hourly_summary.csv",
    "daily_summary": EDA_TABLES_DIR / "daily_summary.csv",
    "banner_pos_summary": EDA_TABLES_DIR / "banner_pos_summary.csv",
    "device_type_summary": EDA_TABLES_DIR / "device_type_summary.csv",
    "site_category_summary": EDA_TABLES_DIR / "site_category_summary.csv",
    "app_category_summary": EDA_TABLES_DIR / "app_category_summary.csv",
}

# 图表保存参数
FIGURE_DPI = 150
TOP_N_CATEGORY = 15


def check_input_files() -> None:
    """
    检查全部输入 CSV 是否存在。

    若有缺失，抛出 FileNotFoundError 并列出缺失文件路径，
    便于用户先运行对应的 EDA 脚本。
    """

    missing = [str(path) for path in INPUT_FILES.values() if not path.exists()]

    if missing:
        missing_list = "\n  - ".join(missing)
        raise FileNotFoundError(
            "以下 EDA 汇总文件不存在，请先运行对应的 EDA 脚本：\n  - "
            f"{missing_list}"
        )


def load_overall_ctr() -> float:
    """从 overall_summary.csv 读取整体 CTR（0~1 的小数）。"""

    overall_df = pd.read_csv(INPUT_FILES["overall_summary"])
    return float(overall_df.loc[0, "ctr"])


def ensure_figures_dir() -> None:
    """若输出目录不存在则自动创建。"""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """
    保存单张图片并关闭 Figure，避免多张图互相覆盖。

    参数：
        fig      — matplotlib Figure 对象
        filename — 文件名，例如 hourly_ctr.png

    返回：
        保存路径
    """

    output_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"已保存：{output_path}")
    return output_path


def add_overall_ctr_line(ax: plt.Axes, overall_ctr: float) -> None:
    """
    在 CTR 图上添加整体 CTR 水平参考线。

    overall_ctr 为小数（如 0.17），图中 Y 轴使用百分比，故乘以 100。
    """

    ax.axhline(
        overall_ctr * 100,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Overall CTR ({overall_ctr:.2%})",
    )
    ax.legend(loc="best")


def plot_hourly_ctr(overall_ctr: float) -> None:
    """绘制按小时 CTR 折线图：hourly_ctr.png"""

    df = pd.read_csv(INPUT_FILES["hourly_summary"])
    df = df.sort_values("hour_of_day")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        df["hour_of_day"],
        df["ctr"] * 100,
        marker="o",
        linewidth=1.5,
        color="steelblue",
    )

    ax.set_title("CTR by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("CTR (%)")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)

    add_overall_ctr_line(ax, overall_ctr)
    save_figure(fig, "hourly_ctr.png")


def plot_daily_ctr(overall_ctr: float) -> None:
    """绘制按日期 CTR 折线图：daily_ctr.png"""

    df = pd.read_csv(INPUT_FILES["daily_summary"])
    df = df.sort_values("date")

    # 日期转为字符串，避免被当作连续数值
    date_labels = df["date"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(len(df)),
        df["ctr"] * 100,
        marker="o",
        linewidth=1.5,
        color="steelblue",
    )

    ax.set_title("CTR by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("CTR (%)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    add_overall_ctr_line(ax, overall_ctr)
    save_figure(fig, "daily_ctr.png")


def plot_banner_pos_ctr(overall_ctr: float) -> None:
    """绘制 banner_pos CTR 柱状图：banner_pos_ctr.png"""

    df = pd.read_csv(INPUT_FILES["banner_pos_summary"])
    df = df.sort_values("banner_pos")

    # 分类编码转字符串，避免 matplotlib 当作连续数值
    x_labels = df["banner_pos"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_labels, df["ctr"] * 100, color="steelblue")

    ax.set_title("CTR by Banner Position")
    ax.set_xlabel("banner_pos")
    ax.set_ylabel("CTR (%)")
    ax.grid(True, axis="y", alpha=0.3)

    add_overall_ctr_line(ax, overall_ctr)
    save_figure(fig, "banner_pos_ctr.png")


def plot_device_type_ctr(overall_ctr: float) -> None:
    """绘制 device_type CTR 柱状图：device_type_ctr.png"""

    df = pd.read_csv(INPUT_FILES["device_type_summary"])
    # 汇总表已按 impressions 降序，保持该顺序便于观察主要设备类型
    x_labels = df["device_type"].astype(str)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_labels, df["ctr"] * 100, color="steelblue")

    ax.set_title("CTR by Device Type")
    ax.set_xlabel("device_type")
    ax.set_ylabel("CTR (%)")
    ax.grid(True, axis="y", alpha=0.3)

    add_overall_ctr_line(ax, overall_ctr)
    save_figure(fig, "device_type_ctr.png")


def plot_category_top15_ctr(
    summary_key: str,
    category_column: str,
    output_filename: str,
    title: str,
    overall_ctr: float,
) -> None:
    """
    绘制曝光量 Top15 类别的 CTR 柱状图（site_category / app_category 共用逻辑）。

    参数：
        summary_key     — INPUT_FILES 中的键名
        category_column — 类别列名
        output_filename — 输出 PNG 文件名
        title           — 图表英文标题
    """

    df = pd.read_csv(INPUT_FILES[summary_key])
    df = df.sort_values("impressions", ascending=False).head(TOP_N_CATEGORY)

    x_labels = df[category_column].astype(str)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_labels, df["ctr"] * 100, color="steelblue")

    ax.set_title(title)
    ax.set_xlabel(category_column)
    ax.set_ylabel("CTR (%)")
    ax.grid(True, axis="y", alpha=0.3)

    # 类别编码较长时旋转标签，避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    add_overall_ctr_line(ax, overall_ctr)
    save_figure(fig, output_filename)


def main() -> None:
    """主流程：检查输入 → 读取整体 CTR → 依次绘制并保存全部图表。"""

    print("=" * 60)
    print("EDA 汇总表可视化")
    print("=" * 60)

    check_input_files()
    ensure_figures_dir()

    overall_ctr = load_overall_ctr()
    print(f"整体 CTR：{overall_ctr:.4%}\n")

    plot_hourly_ctr(overall_ctr)
    plot_daily_ctr(overall_ctr)
    plot_banner_pos_ctr(overall_ctr)
    plot_device_type_ctr(overall_ctr)

    plot_category_top15_ctr(
        summary_key="site_category_summary",
        category_column="site_category",
        output_filename="site_category_top15_ctr.png",
        title="CTR by Site Category (Top 15 by Impressions)",
        overall_ctr=overall_ctr,
    )

    plot_category_top15_ctr(
        summary_key="app_category_summary",
        category_column="app_category",
        output_filename="app_category_top15_ctr.png",
        title="CTR by App Category (Top 15 by Impressions)",
        overall_ctr=overall_ctr,
    )

    print("\n全部图表已生成完毕。")


if __name__ == "__main__":
    main()
