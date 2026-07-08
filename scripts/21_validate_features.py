"""
百度 CTR 项目 — 特征初步有效性检查脚本（第二阶段）

功能：
    从 data/features/frequency/train/ 的多个 Parquet 分块中随机抽样，
    对基础特征与频次特征做质量检查、类别 CTR 区分度分析与数值相关性分析。

数据输入：
    data/features/frequency/train/*.parquet（仅 train，不使用 test）

数据输出：
    outputs/feature_validation/*.csv
    outputs/21_feature_validation_report.txt

说明：
    - 全量 train 超过 4000 万行，本脚本只抽样分析，不一次性读入全量
    - 不做模型训练、不做目标编码、不修改原始数据
    - 相关系数接近 0 不代表特征无用（非线性模型可能仍有效）

用法：
    python scripts/21_validate_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 抽样配置
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_INPUT_DIR = Path("data/features/frequency/train")
OUTPUT_DIR = Path("outputs/feature_validation")
REPORT_PATH = Path("outputs/21_feature_validation_report.txt")

QUALITY_CSV = OUTPUT_DIR / "feature_quality_summary.csv"
CATEGORICAL_CTR_CSV = OUTPUT_DIR / "categorical_ctr_summary.csv"
CATEGORICAL_SPREAD_CSV = OUTPUT_DIR / "categorical_feature_spread.csv"
NUMERIC_CORR_CSV = OUTPUT_DIR / "numeric_feature_correlation.csv"

# ---------------------------------------------------------------------------
# 特征分组定义
# ---------------------------------------------------------------------------

TIME_FEATURES = ["hour_of_day", "day", "day_of_week", "is_weekend"]
CROSS_FEATURES = ["banner_device_cross", "hour_banner_cross", "site_device_cross"]
FREQ_FEATURES = [
    "site_id_freq",
    "site_category_freq",
    "app_id_freq",
    "app_category_freq",
    "device_model_freq",
]
TARGET_FEATURE = "click"

ALL_INPUT_FEATURES = TIME_FEATURES + CROSS_FEATURES + FREQ_FEATURES

# 类别特征 CTR 区分度检查（含可视为类别的字段）
CATEGORICAL_CTR_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "banner_device_cross",
    "hour_banner_cross",
    "site_device_cross",
]

# 数值特征与 click 的 Pearson 相关
NUMERIC_CORRELATION_FEATURES = [
    "hour_of_day",
    "day",
    "day_of_week",
    "is_weekend",
    "site_id_freq",
    "site_category_freq",
    "app_id_freq",
    "app_category_freq",
    "device_model_freq",
]

# 高基数交叉特征：CTR 明细表与 spread 只保留曝光 Top N 类别
HIGH_CARDINALITY_FEATURES = set(CROSS_FEATURES)
TOP_CATEGORY_REPORT = 20

FEATURE_GROUP_MAP: dict[str, str] = {
    **{name: "时间特征" for name in TIME_FEATURES},
    **{name: "交叉特征" for name in CROSS_FEATURES},
    **{name: "频次特征" for name in FREQ_FEATURES},
    TARGET_FEATURE: "目标变量",
}


def list_parquet_files(parquet_dir: Path) -> list[Path]:
    """列出 train 特征 Parquet 分块路径。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到输入目录：{parquet_dir}\n"
            "请先运行：python scripts/20_build_frequency_features.py"
        )

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            "请先运行：python scripts/20_build_frequency_features.py"
        )

    return files


def get_parquet_row_counts(parquet_files: list[Path]) -> list[int]:
    """读取各 Parquet 分块行数（仅读元数据，不加载全表）。"""

    return [pq.read_metadata(path).num_rows for path in parquet_files]


def sample_from_multiple_parquets(
    parquet_files: list[Path],
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """
    从多个 Parquet 文件中按比例随机抽样，减少单一分块时间段偏差。

    步骤：
        1. 统计每个分块行数
        2. 按行数比例分配各分块抽样数量
        3. 逐分块 sample 后合并
    """

    row_counts = get_parquet_row_counts(parquet_files)
    total_rows = sum(row_counts)

    if total_rows == 0:
        raise ValueError("train 特征文件总行数为 0，无法抽样。")

    actual_sample_size = min(sample_size, total_rows)

    needed_columns = ALL_INPUT_FEATURES + [TARGET_FEATURE]
    sample_parts: list[pd.DataFrame] = []
    allocated = 0

    print(f"从 {len(parquet_files)} 个 Parquet 分块中抽样...")
    print(f"  全量行数：{total_rows:,}")
    print(f"  目标样本量：{actual_sample_size:,}")

    for index, (parquet_path, row_count) in enumerate(zip(parquet_files, row_counts)):
        # 最后一个分块承接剩余样本量，避免四舍五入导致总数不足
        if index == len(parquet_files) - 1:
            chunk_sample_size = actual_sample_size - allocated
        else:
            chunk_sample_size = int(round(actual_sample_size * row_count / total_rows))

        if chunk_sample_size <= 0:
            continue

        chunk_df = pd.read_parquet(parquet_path, columns=needed_columns)

        if chunk_sample_size >= len(chunk_df):
            sampled = chunk_df
        else:
            sampled = chunk_df.sample(
                n=chunk_sample_size,
                random_state=random_state,
            )

        sample_parts.append(sampled)
        allocated += len(sampled)

        print(
            f"  {parquet_path.name}: 读取 {len(chunk_df):,} 行, "
            f"抽样 {len(sampled):,} 行"
        )

    sample_df = pd.concat(sample_parts, ignore_index=True)

    # 若因分块边界导致略多于目标样本量，再随机下采样一次
    if len(sample_df) > actual_sample_size:
        sample_df = sample_df.sample(
            n=actual_sample_size,
            random_state=random_state,
        ).reset_index(drop=True)

    print(f"  最终样本量：{len(sample_df):,}\n")
    return sample_df


def is_numeric_feature(series: pd.Series) -> bool:
    """判断特征是否按数值型统计 min/max/mean 等。"""

    return pd.api.types.is_numeric_dtype(series)


def build_feature_quality_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    特征质量检查：缺失、唯一值、常数列；数值型额外统计描述量。
    """

    rows: list[dict] = []

    for feature_name in ALL_INPUT_FEATURES + [TARGET_FEATURE]:
        series = sample_df[feature_name]
        missing_count = int(series.isna().sum())
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())
        is_constant = unique_count <= 1

        row: dict = {
            "feature_name": feature_name,
            "feature_group": FEATURE_GROUP_MAP[feature_name],
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_rate": missing_count / len(sample_df) if len(sample_df) > 0 else 0.0,
            "unique_count": unique_count,
            "is_constant": is_constant,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

        if is_numeric_feature(series) and len(non_missing) > 0:
            row["min"] = float(non_missing.min())
            row["max"] = float(non_missing.max())
            row["mean"] = float(non_missing.mean())
            row["median"] = float(non_missing.median())
            row["std"] = float(non_missing.std())

        rows.append(row)

    return pd.DataFrame(rows)


def compute_category_ctr_table(
    sample_df: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """统计某一类别特征每个取值的曝光、点击与 CTR。"""

    grouped = (
        sample_df.groupby(feature_name, dropna=False)
        .agg(
            impressions=(TARGET_FEATURE, "count"),
            clicks=(TARGET_FEATURE, "sum"),
        )
        .reset_index()
    )

    grouped["ctr"] = grouped["clicks"] / grouped["impressions"]
    grouped.insert(0, "feature_name", feature_name)
    grouped = grouped.rename(columns={feature_name: "feature_value"})

    # 统一 feature_value 为字符串，便于 CSV 输出
    grouped["feature_value"] = grouped["feature_value"].astype(str)

    return grouped.sort_values("impressions", ascending=False).reset_index(drop=True)


def build_categorical_ctr_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    类别特征 CTR 区分度明细。

    高基数交叉特征只保留曝光 Top 20 类别写入汇总表。
    """

    summary_parts: list[pd.DataFrame] = []

    for feature_name in CATEGORICAL_CTR_FEATURES:
        feature_table = compute_category_ctr_table(sample_df, feature_name)

        if feature_name in HIGH_CARDINALITY_FEATURES:
            feature_table = feature_table.head(TOP_CATEGORY_REPORT)

        summary_parts.append(feature_table)

    return pd.concat(summary_parts, ignore_index=True)


def build_categorical_feature_spread(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    每个类别特征的 CTR spread 摘要。

    ctr_spread = maximum_ctr - minimum_ctr
    高基数交叉特征基于曝光 Top 20 类别计算。
    """

    rows: list[dict] = []

    for feature_name in CATEGORICAL_CTR_FEATURES:
        feature_table = compute_category_ctr_table(sample_df, feature_name)

        if feature_name in HIGH_CARDINALITY_FEATURES:
            used_table = feature_table.head(TOP_CATEGORY_REPORT)
        else:
            used_table = feature_table

        if used_table.empty:
            rows.append(
                {
                    "feature_name": feature_name,
                    "minimum_ctr": np.nan,
                    "maximum_ctr": np.nan,
                    "ctr_spread": np.nan,
                    "top_category_count": 0,
                }
            )
            continue

        minimum_ctr = float(used_table["ctr"].min())
        maximum_ctr = float(used_table["ctr"].max())

        rows.append(
            {
                "feature_name": feature_name,
                "minimum_ctr": minimum_ctr,
                "maximum_ctr": maximum_ctr,
                "ctr_spread": maximum_ctr - minimum_ctr,
                "top_category_count": len(used_table),
            }
        )

    return pd.DataFrame(rows)


def build_numeric_correlation(sample_df: pd.DataFrame) -> pd.DataFrame:
    """计算数值特征与 click 的 Pearson 相关系数。"""

    rows: list[dict] = []

    for feature_name in NUMERIC_CORRELATION_FEATURES:
        valid_mask = sample_df[[feature_name, TARGET_FEATURE]].notna().all(axis=1)
        valid_df = sample_df.loc[valid_mask, [feature_name, TARGET_FEATURE]]

        if len(valid_df) < 2:
            correlation = np.nan
        else:
            correlation = float(valid_df[feature_name].corr(valid_df[TARGET_FEATURE]))

        rows.append(
            {
                "feature_name": feature_name,
                "pearson_correlation_with_click": correlation,
            }
        )

    return pd.DataFrame(rows)


def format_rate(value: float) -> str:
    """格式化比例/CTR 为百分比字符串。"""

    return f"{value:.4%}"


def build_text_report(
    sample_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    categorical_spread_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> str:
    """生成特征有效性检查文字报告。"""

    lines: list[str] = [
        "=" * 70,
        "百度 CTR 项目 — 特征初步有效性检查报告",
        "=" * 70,
        "",
        "【1. 样本与检查范围】",
        f"样本量：{len(sample_df):,}",
        f"随机种子：{RANDOM_STATE}",
        f"抽样来源：data/features/frequency/train/ 多个 Parquet 分块",
        f"检查输入特征数量：{len(ALL_INPUT_FEATURES)}",
        f"目标变量：{TARGET_FEATURE}",
        "",
        "【2. 特征质量检查摘要】",
    ]

    high_missing = quality_df[quality_df["missing_rate"] > 0.01]
    constant_features = quality_df[quality_df["is_constant"]]

    if high_missing.empty:
        lines.append("未发现缺失率超过 1% 的特征。")
    else:
        lines.append("以下特征缺失率超过 1%：")
        for _, row in high_missing.iterrows():
            lines.append(
                f"  - {row['feature_name']} ({row['feature_group']}): "
                f"缺失率 {format_rate(row['missing_rate'])}"
            )

    lines.append("")
    if constant_features.empty:
        lines.append("未发现常数列。")
    else:
        lines.append("以下特征在样本中为常数列：")
        for _, row in constant_features.iterrows():
            lines.append(f"  - {row['feature_name']} ({row['feature_group']})")

    lines.extend(["", "【3. 类别特征 CTR 差异摘要】"])
    spread_sorted = categorical_spread_df.sort_values("ctr_spread", ascending=False)

    for _, row in spread_sorted.iterrows():
        lines.append(
            f"  - {row['feature_name']}: "
            f"ctr_spread={format_rate(row['ctr_spread'])}, "
            f"min={format_rate(row['minimum_ctr'])}, "
            f"max={format_rate(row['maximum_ctr'])}, "
            f"统计类别数={int(row['top_category_count'])}"
        )

    lines.extend(
        [
            "",
            "说明：高基数交叉特征（banner_device_cross、hour_banner_cross、"
            f"site_device_cross）的 CTR 差异基于曝光 Top {TOP_CATEGORY_REPORT} 类别计算。",
            "",
            "【4. 数值特征相关性摘要（Pearson）】",
        ]
    )

    corr_sorted = correlation_df.sort_values(
        "pearson_correlation_with_click",
        key=lambda s: s.abs(),
        ascending=False,
    )

    for _, row in corr_sorted.iterrows():
        corr = row["pearson_correlation_with_click"]
        if pd.isna(corr):
            corr_text = "NaN"
        else:
            corr_text = f"{corr:+.6f}"
        lines.append(f"  - {row['feature_name']}: {corr_text}")

    lines.extend(
        [
            "",
            "注意：Pearson 相关系数接近 0 不代表特征一定无用。",
            "树模型（LightGBM、XGBoost 等）可能学习到非线性关系，",
            "最终有效性仍需结合验证集 AUC、LogLoss 和模型特征重要性判断。",
            "",
            "【5. 当前阶段结论】",
            "- 本报告仅为 train 样本上的初步特征检查，不能替代正式建模评估。",
            "- 时间特征、交叉特征、频次特征整体已完成质量检查，可进入后续模型验证流程。",
            "- 类别 CTR spread 较大的特征，通常表示在不同取值之间「有一定区分度」。",
            "- 是否「值得进入后续模型验证」，需结合本报告与 EDA / SQL 分析综合判断。",
            "- 最终特征有效性必须结合验证集 AUC、LogLoss 和模型重要性结果确认。",
            "",
            "【6. 注意事项】",
            "- 未使用 test 数据进行特征有效性判断，避免信息泄漏。",
            "- 未做目标编码，未做模型训练。",
            "- 全量 train 超过 4000 万行，本报告基于随机样本，结论存在抽样误差。",
            "- 数据库与原始 Parquet 数据未被修改。",
            "",
            "【输出文件】",
            f"  - {QUALITY_CSV}",
            f"  - {CATEGORICAL_CTR_CSV}",
            f"  - {CATEGORICAL_SPREAD_CSV}",
            f"  - {NUMERIC_CORR_CSV}",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def save_outputs(
    quality_df: pd.DataFrame,
    categorical_ctr_df: pd.DataFrame,
    categorical_spread_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    report_text: str,
) -> None:
    """保存 CSV 与文字报告。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    quality_df.to_csv(QUALITY_CSV, index=False)
    categorical_ctr_df.to_csv(CATEGORICAL_CTR_CSV, index=False)
    categorical_spread_df.to_csv(CATEGORICAL_SPREAD_CSV, index=False)
    correlation_df.to_csv(NUMERIC_CORR_CSV, index=False)

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("结果已保存：")
    print(f"  {QUALITY_CSV}")
    print(f"  {CATEGORICAL_CTR_CSV}")
    print(f"  {CATEGORICAL_SPREAD_CSV}")
    print(f"  {NUMERIC_CORR_CSV}")
    print(f"  {REPORT_PATH}")


def main() -> None:
    """主流程：多文件抽样 → 质量检查 → CTR 区分度 → 相关性 → 报告。"""

    print("=" * 70)
    print("特征初步有效性检查")
    print("=" * 70)

    parquet_files = list_parquet_files(TRAIN_INPUT_DIR)
    sample_df = sample_from_multiple_parquets(
        parquet_files,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    missing_features = [
        name for name in ALL_INPUT_FEATURES + [TARGET_FEATURE]
        if name not in sample_df.columns
    ]
    if missing_features:
        raise ValueError(f"样本数据缺少以下字段：{missing_features}")

    print("正在生成特征质量摘要...")
    quality_df = build_feature_quality_summary(sample_df)

    print("正在生成类别特征 CTR 摘要...")
    categorical_ctr_df = build_categorical_ctr_summary(sample_df)
    categorical_spread_df = build_categorical_feature_spread(sample_df)

    print("正在计算数值特征 Pearson 相关性...")
    correlation_df = build_numeric_correlation(sample_df)

    report_text = build_text_report(
        sample_df,
        quality_df,
        categorical_spread_df,
        correlation_df,
    )

    save_outputs(
        quality_df,
        categorical_ctr_df,
        categorical_spread_df,
        correlation_df,
        report_text,
    )

    print("\n特征初步有效性检查完成。")


if __name__ == "__main__":
    main()
