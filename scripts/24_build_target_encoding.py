"""
百度 CTR 项目 — 平滑 Target Encoding 特征生成脚本

功能：
    为类别字段生成平滑 Target Encoding（TE）特征，严格避免数据泄漏。

数据输入：
    data/features/historical/train/*.parquet
    data/features/historical/valid/*.parquet
    data/features/historical/holdout/*.parquet

数据输出（正式模式）：
    data/features/target_encoded/train/
    data/features/target_encoded/valid/
    data/features/target_encoded/holdout/
    outputs/24_target_encoding_report.txt

数据输出（测试模式）：
    data/features/target_encoded_test/train/
    data/features/target_encoded_test/valid/
    data/features/target_encoded_test/holdout/
    outputs/24_target_encoding_test_report.txt

平滑公式：
    TE = (category_clicks + SMOOTHING_STRENGTH * prior_ctr)
         / (category_count + SMOOTHING_STRENGTH)

用法：
    python scripts/24_build_target_encoding.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

# True：少量文件测试逻辑；False：全量处理
TEST_MODE = False

TEST_TRAIN_MAX_FILES = 22
TEST_VALID_MAX_FILES = 1
TEST_HOLDOUT_MAX_FILES = 1


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

SMOOTHING_STRENGTH = 20
DEFAULT_PRIOR = 0.17

TE_FIELDS = [
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_model",
]

TE_FIELD_MAP: dict[str, str] = {
    "site_id": "site_id_te",
    "site_category": "site_category_te",
    "app_id": "app_id_te",
    "app_category": "app_category_te",
    "device_model": "device_model_te",
}

INPUT_BASE_DIR = Path("data/features/historical")

REQUIRED_COLUMNS = ["hour_dt", "click", *TE_FIELDS]

SPLIT_DATE_RANGES = {
    "train": (pd.Timestamp("2014-10-21"), pd.Timestamp("2014-10-28")),
    "valid": (pd.Timestamp("2014-10-29"), pd.Timestamp("2014-10-29")),
    "holdout": (pd.Timestamp("2014-10-30"), pd.Timestamp("2014-10-30")),
}


def get_output_dirs(test_mode: bool) -> dict[str, Path]:
    """根据运行模式返回输出目录。"""

    base = Path("data/features/target_encoded_test" if test_mode else "data/features/target_encoded")
    return {
        "train": base / "train",
        "valid": base / "valid",
        "holdout": base / "holdout",
    }


def get_report_path(test_mode: bool) -> Path:
    """根据运行模式返回报告路径。"""

    if test_mode:
        return Path("outputs/24_target_encoding_test_report.txt")
    return Path("outputs/24_target_encoding_report.txt")


def get_feature_names() -> list[str]:
    """返回 5 个 TE 特征名。"""

    return list(TE_FIELD_MAP.values())


def list_parquet_files(
    parquet_dir: Path,
    upstream_script: str,
    max_files: int | None = None,
) -> list[Path]:
    """列出 Parquet 分块；目录不存在或为空时抛出明确错误。"""

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到输入目录：{parquet_dir}\n"
            f"请先运行：python scripts/{upstream_script}"
        )

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            f"请先运行：python scripts/{upstream_script}"
        )

    if max_files is not None:
        return files[:max_files]

    return files


def warn_if_output_exists(output_dirs: dict[str, Path]) -> None:
    """若输出目录已有旧 Parquet，提示用户避免混用结果。"""

    for split_name, output_dir in output_dirs.items():
        if not output_dir.exists():
            continue

        existing = sorted(output_dir.glob("part-*.parquet"))
        if existing:
            print(
                f"WARNING: {output_dir} 已存在 {len(existing)} 个 Parquet 文件，"
                "继续运行可能覆盖或与旧结果混在一起。"
            )


def ensure_dirs(output_dirs: dict[str, Path], report_path: Path) -> None:
    """运行前自动创建输出目录。"""

    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)


def validate_columns(dataframe: pd.DataFrame, context: str) -> None:
    """检查必需字段是否存在。"""

    missing = [col for col in REQUIRED_COLUMNS if col not in dataframe.columns]
    if missing:
        raise ValueError(f"{context} 缺少必需字段：{missing}")


def extract_event_date(dataframe: pd.DataFrame) -> pd.Series:
    """从 hour_dt 提取归一化日期。"""

    if "hour_dt" not in dataframe.columns:
        raise ValueError("输入数据缺少 hour_dt 字段，无法提取日期。")

    hour_dt = pd.to_datetime(dataframe["hour_dt"], errors="coerce")
    if hour_dt.isna().all():
        raise ValueError("hour_dt 全部无法解析为有效时间。")

    return hour_dt.dt.normalize()


def compute_smoothed_te(
    category_clicks: int | float,
    category_count: int | float,
    prior_ctr: float,
) -> float:
    """
    计算平滑 Target Encoding，并限制在 [0, 1]。

    TE = (category_clicks + SMOOTHING_STRENGTH * prior_ctr)
         / (category_count + SMOOTHING_STRENGTH)
    """

    te_value = (
        category_clicks + SMOOTHING_STRENGTH * prior_ctr
    ) / (category_count + SMOOTHING_STRENGTH)

    return float(np.clip(te_value, 0.0, 1.0))


def fallback_te(prior_ctr: float, has_history_window: bool) -> float:
    """未见类别或无历史时的 TE 回退值。"""

    if has_history_window:
        return float(np.clip(prior_ctr, 0.0, 1.0))

    return float(np.clip(DEFAULT_PRIOR, 0.0, 1.0))


def build_te_mapping_from_stats(
    stats_df: pd.DataFrame,
    te_field: str,
    prior_ctr: float,
) -> pd.DataFrame:
    """
    由聚合统计（category_count, category_clicks）构建 TE 映射表。

    返回列：[te_field, category_count, category_clicks, te_value]
    """

    if stats_df.empty:
        return pd.DataFrame(
            columns=[te_field, "category_count", "category_clicks", "te_value"]
        )

    result = stats_df.copy()
    result["te_value"] = result.apply(
        lambda row: compute_smoothed_te(
            row["category_clicks"],
            row["category_count"],
            prior_ctr,
        ),
        axis=1,
    )

    result["category_count"] = result["category_count"].astype("int64")
    result["category_clicks"] = result["category_clicks"].astype("int64")
    result["te_value"] = result["te_value"].astype("float64")

    return result[[te_field, "category_count", "category_clicks", "te_value"]]


@dataclass
class TargetEncodingState:
    """train 增量 TE 状态：仅包含已处理日期（严格早于当前日）的累计统计。"""

    category_stats: dict[str, dict[object, dict[str, int]]] = field(
        default_factory=lambda: {field_name: {} for field_name in TE_FIELDS}
    )
    total_count: int = 0
    total_clicks: int = 0
    processed_dates: list[pd.Timestamp] = field(default_factory=list)

    def has_history_window(self) -> bool:
        """是否已有历史窗口。"""

        return self.total_count > 0

    def prior_ctr(self) -> float:
        """当前历史窗口的全局 CTR。"""

        if self.total_count <= 0:
            return DEFAULT_PRIOR

        return self.total_clicks / self.total_count

    def history_window_label(self, current_date: pd.Timestamp) -> str:
        """用于日志：当前日可用的历史日期范围。"""

        if not self.processed_dates:
            return "无（冷启动）"

        start_date = min(self.processed_dates).date().isoformat()
        end_date = max(self.processed_dates).date().isoformat()
        return f"{start_date} ~ {end_date}（严格早于 {current_date.date().isoformat()}）"

    def build_mapping(self, te_field: str) -> pd.DataFrame:
        """由当前累计状态构建单个字段的 TE 映射表。"""

        prior_ctr = self.prior_ctr()
        rows: list[dict] = []

        for category, stats in self.category_stats[te_field].items():
            rows.append(
                {
                    te_field: category,
                    "category_count": stats["count"],
                    "category_clicks": stats["clicks"],
                }
            )

        stats_df = pd.DataFrame(rows)
        return build_te_mapping_from_stats(stats_df, te_field, prior_ctr)

    def build_all_mappings(self) -> dict[str, pd.DataFrame]:
        """构建 5 个字段的 TE 映射表。"""

        return {te_field: self.build_mapping(te_field) for te_field in TE_FIELDS}

    def update_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        event_date: pd.Timestamp,
    ) -> None:
        """将当日数据累计进历史状态（必须在当日 TE 映射完成后调用）。"""

        validate_columns(dataframe, context=f"日期 {event_date.date()}")

        self.total_count += len(dataframe)
        self.total_clicks += int(dataframe["click"].sum())

        for te_field in TE_FIELDS:
            grouped = (
                dataframe.groupby(te_field, dropna=False)["click"]
                .agg(clicks="sum", count="count")
                .reset_index()
            )

            field_stats = self.category_stats[te_field]
            for _, row in grouped.iterrows():
                category = row[te_field]
                if category not in field_stats:
                    field_stats[category] = {"count": 0, "clicks": 0}

                field_stats[category]["count"] += int(row["count"])
                field_stats[category]["clicks"] += int(row["clicks"])

        if event_date not in self.processed_dates:
            self.processed_dates.append(event_date)


def apply_target_encoding(
    dataframe: pd.DataFrame,
    mappings_by_field: dict[str, pd.DataFrame],
    prior_ctr: float,
    has_history_window: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    将 TE 特征合并到 DataFrame。

    未见类别：使用 prior_ctr（有历史窗口）或 DEFAULT_PRIOR（冷启动）。
    返回：(带 TE 的 DataFrame, 各字段未见类别行数)
    """

    validate_columns(dataframe, context="TE 特征映射")
    result = dataframe.copy()
    fallback = fallback_te(prior_ctr, has_history_window)
    unseen_counts: dict[str, int] = {}

    for te_field, te_column in TE_FIELD_MAP.items():
        mapping = mappings_by_field[te_field]
        mapping_to_merge = mapping[[te_field, "te_value"]].rename(
            columns={"te_value": te_column}
        )

        result = result.merge(mapping_to_merge, on=te_field, how="left")
        unseen_counts[te_field] = int(result[te_column].isna().sum())
        result[te_column] = result[te_column].fillna(fallback).astype("float64")

        # 确保 TE 在 [0, 1] 范围内
        if (result[te_column] < 0).any() or (result[te_column] > 1).any():
            raise ValueError(f"{te_column} 存在超出 [0, 1] 范围的值。")

    return result, unseen_counts


def build_date_file_index(parquet_files: list[Path]) -> dict[pd.Timestamp, list[Path]]:
    """扫描 train 文件，建立 日期 -> 文件列表 索引。"""

    date_to_files: dict[pd.Timestamp, list[Path]] = defaultdict(list)

    for input_path in parquet_files:
        chunk = pd.read_parquet(input_path, columns=["hour_dt"])
        if chunk.empty:
            raise ValueError(f"输入文件为空：{input_path}")

        event_dates = extract_event_date(chunk).dropna().unique()
        if len(event_dates) == 0:
            raise ValueError(f"文件 {input_path} 无有效 hour_dt，无法确定日期。")

        for event_date in event_dates:
            date_to_files[pd.Timestamp(event_date)].append(input_path)

    return dict(sorted(date_to_files.items()))


def build_te_mapping_dask(
    parquet_files: list[Path],
    split_label: str,
) -> tuple[dict[str, pd.DataFrame], float, bool]:
    """
    使用 Dask 聚合 TE 映射（用于 valid / holdout 的先验历史）。

    只读取 click 与类别字段，避免全列加载。
    """

    if not parquet_files:
        raise ValueError(f"{split_label} 聚合时没有可用的 Parquet 文件。")

    paths = [str(path) for path in parquet_files]
    columns = ["click", *TE_FIELDS]

    print(f"\n使用 Dask 聚合 {split_label} TE 映射...")
    print(f"  参与文件数：{len(parquet_files)}")

    ddf = dd.read_parquet(paths, columns=columns)

    total_count = int(ddf.shape[0].compute())
    total_clicks = int(ddf["click"].sum().compute())
    has_history_window = total_count > 0
    prior_ctr = total_clicks / total_count if has_history_window else DEFAULT_PRIOR

    print(f"  历史总样本：{total_count:,}")
    print(f"  历史总点击：{total_clicks:,}")
    print(f"  prior_ctr：  {prior_ctr:.6f}")

    mappings: dict[str, pd.DataFrame] = {}

    for te_field in TE_FIELDS:
        print(f"  当前处理字段：{te_field}")

        aggregated = (
            ddf.groupby(te_field, dropna=False)
            .agg(
                category_clicks=("click", "sum"),
                category_count=("click", "count"),
            )
            .compute()
            .reset_index()
        )

        mappings[te_field] = build_te_mapping_from_stats(
            aggregated,
            te_field,
            prior_ctr,
        )
        print(f"    唯一类别数：{len(mappings[te_field]):,}")

    return mappings, prior_ctr, has_history_window


@dataclass
class TeFeatureStats:
    """TE 特征统计：min / max / mean（跨所有已处理行累计）。"""

    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    mins: dict[str, float | None] = field(default_factory=dict)
    maxs: dict[str, float | None] = field(default_factory=dict)

    def update(self, dataframe: pd.DataFrame) -> None:
        """累计各 TE 特征的 min / max / sum / count。"""

        for te_column in get_feature_names():
            if te_column not in dataframe.columns:
                continue

            series = dataframe[te_column]
            current_min = float(series.min())
            current_max = float(series.max())
            current_sum = float(series.sum())
            current_count = len(series)

            self.sums[te_column] = self.sums.get(te_column, 0.0) + current_sum
            self.counts[te_column] = self.counts.get(te_column, 0) + current_count

            self.mins[te_column] = (
                current_min
                if te_column not in self.mins or self.mins[te_column] is None
                else min(self.mins[te_column], current_min)
            )
            self.maxs[te_column] = (
                current_max
                if te_column not in self.maxs or self.maxs[te_column] is None
                else max(self.maxs[te_column], current_max)
            )

    def mean(self, te_column: str) -> float | None:
        """返回某 TE 特征的均值。"""

        count = self.counts.get(te_column, 0)
        if count == 0:
            return None

        return self.sums[te_column] / count


@dataclass
class RunStats:
    """全流程统计信息，用于报告。"""

    train_rows: int = 0
    valid_rows: int = 0
    holdout_rows: int = 0
    train_date_start: str | None = None
    train_date_end: str | None = None
    unseen_ratios_valid: dict[str, float] = field(default_factory=dict)
    unseen_ratios_holdout: dict[str, float] = field(default_factory=dict)
    te_stats: TeFeatureStats = field(default_factory=TeFeatureStats)


def process_train(
    train_files: list[Path],
    output_dir: Path,
    stats: RunStats,
) -> None:
    """
    train 按日期升序处理：
    1. 用严格早于当前日的历史状态建 TE 映射
    2. 映射并写出当前日所有分块
    3. 再将当前日 click 累计进状态
    """

    date_to_files = build_date_file_index(train_files)
    total_files = sum(len(files) for files in date_to_files.values())
    completed_files = 0

    if date_to_files:
        stats.train_date_start = min(date_to_files).date().isoformat()
        stats.train_date_end = max(date_to_files).date().isoformat()

    state = TargetEncodingState()

    print("\n" + "=" * 70)
    print("处理 split: train")
    print("=" * 70)

    for event_date, files_for_date in date_to_files.items():
        history_label = state.history_window_label(event_date)
        mappings = state.build_all_mappings()
        prior_ctr = state.prior_ctr()
        has_history_window = state.has_history_window()

        print(f"\n当前处理日期：{event_date.date().isoformat()}")
        print(f"当前历史窗口：{history_label}")
        print(f"prior_ctr：{prior_ctr:.6f}")

        pending_updates: list[pd.DataFrame] = []

        for input_path in sorted(files_for_date):
            for te_field in TE_FIELDS:
                print(f"  当前处理字段：{te_field}")

            output_path = output_dir / input_path.name
            print("  当前 split：train")
            print(f"  当前输入文件：{input_path}")
            print(f"  当前输出文件：{output_path}")

            chunk = pd.read_parquet(input_path)
            if chunk.empty:
                raise ValueError(f"输入文件为空：{input_path}")

            validate_columns(chunk, context=str(input_path))
            featured_chunk, _ = apply_target_encoding(
                chunk,
                mappings,
                prior_ctr,
                has_history_window,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

            row_count = len(featured_chunk)
            stats.train_rows += row_count
            stats.te_stats.update(featured_chunk)
            pending_updates.append(chunk)

            completed_files += 1
            progress = completed_files / total_files * 100
            print(f"  当前文件行数：{row_count:,}")
            print(
                f"  当前累计进度：{completed_files}/{total_files} 文件 "
                f"({progress:.1f}%)，train 累计行数 {stats.train_rows:,}"
            )

        # 当日全部文件映射完成后，才将当日 click 并入历史状态
        for chunk in pending_updates:
            state.update_from_dataframe(chunk, event_date)


def process_split_with_fixed_mapping(
    split_name: str,
    parquet_files: list[Path],
    output_dir: Path,
    mappings_by_field: dict[str, pd.DataFrame],
    prior_ctr: float,
    has_history_window: bool,
    stats: RunStats,
    track_unseen: bool,
) -> None:
    """valid / holdout：使用固定先验 TE 映射，逐文件读写。"""

    total_files = len(parquet_files)
    unseen_counts: dict[str, int] = {field_name: 0 for field_name in TE_FIELDS}
    total_rows = 0

    history_label = {
        "valid": f"{SPLIT_DATE_RANGES['train'][0].date()} ~ {SPLIT_DATE_RANGES['train'][1].date()}（完整 train）",
        "holdout": (
            f"{SPLIT_DATE_RANGES['train'][0].date()} ~ {SPLIT_DATE_RANGES['valid'][1].date()}"
            "（train + valid）"
        ),
    }[split_name]

    print("\n" + "=" * 70)
    print(f"处理 split: {split_name}")
    print("=" * 70)
    print(f"当前历史窗口：{history_label}")
    print(f"prior_ctr：{prior_ctr:.6f}")

    for file_index, input_path in enumerate(parquet_files, start=1):
        for te_field in TE_FIELDS:
            print(f"  当前处理字段：{te_field}")

        output_path = output_dir / input_path.name
        print(f"  当前 split：{split_name}")
        print(f"  当前输入文件：{input_path}")
        print(f"  当前输出文件：{output_path}")

        chunk = pd.read_parquet(input_path)
        if chunk.empty:
            raise ValueError(f"输入文件为空：{input_path}")

        validate_columns(chunk, context=str(input_path))
        featured_chunk, file_unseen = apply_target_encoding(
            chunk,
            mappings_by_field,
            prior_ctr,
            has_history_window,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

        row_count = len(featured_chunk)
        total_rows += row_count
        stats.te_stats.update(featured_chunk)

        if track_unseen:
            for te_field in TE_FIELDS:
                unseen_counts[te_field] += file_unseen[te_field]

        progress = file_index / total_files * 100
        print(f"  当前文件行数：{row_count:,}")
        print(
            f"  当前累计进度：{file_index}/{total_files} 文件 "
            f"({progress:.1f}%)，{split_name} 累计行数 {total_rows:,}"
        )

    if split_name == "valid":
        stats.valid_rows = total_rows
        if track_unseen and total_rows > 0:
            stats.unseen_ratios_valid = {
                field_name: unseen_counts[field_name] / total_rows
                for field_name in TE_FIELDS
            }

    if split_name == "holdout":
        stats.holdout_rows = total_rows
        if track_unseen and total_rows > 0:
            stats.unseen_ratios_holdout = {
                field_name: unseen_counts[field_name] / total_rows
                for field_name in TE_FIELDS
            }


def write_report(
    stats: RunStats,
    report_path: Path,
    output_dirs: dict[str, Path],
    test_mode: bool,
) -> None:
    """生成 TE 特征报告。"""

    feature_names = get_feature_names()
    mode_label = "测试" if test_mode else "正式"

    lines: list[str] = []
    lines.append(f"百度 CTR 项目 — 平滑 Target Encoding 特征报告（{mode_label}模式）")
    lines.append("=" * 70)
    lines.append("")
    lines.append("一、处理字段")
    for te_field in TE_FIELDS:
        lines.append(f"  - {te_field}")
    lines.append("")
    lines.append("二、新增特征")
    for feature_name in feature_names:
        lines.append(f"  - {feature_name}")
    lines.append("")
    lines.append("三、平滑参数")
    lines.append(f"  SMOOTHING_STRENGTH = {SMOOTHING_STRENGTH}")
    lines.append(f"  DEFAULT_PRIOR = {DEFAULT_PRIOR}")
    lines.append("")
    lines.append("四、prior 使用规则")
    lines.append("  - prior_ctr = 当前历史窗口全局 CTR（总点击 / 总样本）")
    lines.append("  - 有历史类别的 TE 使用平滑公式：")
    lines.append(
        "    TE = (category_clicks + SMOOTHING_STRENGTH * prior_ctr)"
        " / (category_count + SMOOTHING_STRENGTH)"
    )
    lines.append("  - 未见类别：使用 prior_ctr；无历史窗口时使用 DEFAULT_PRIOR")
    lines.append("  - 所有 TE 值限制在 [0, 1]")
    lines.append("")
    lines.append("五、各 split 总行数")
    lines.append(f"  train：   {stats.train_rows:,}")
    lines.append(f"  valid：   {stats.valid_rows:,}")
    lines.append(f"  holdout： {stats.holdout_rows:,}")
    lines.append("")
    lines.append("六、valid / holdout 未见类别比例")
    lines.append("  valid：")
    for te_field in TE_FIELDS:
        ratio = stats.unseen_ratios_valid.get(te_field, 0.0)
        lines.append(f"    {te_field}: {ratio:.4%}")
    lines.append("  holdout：")
    for te_field in TE_FIELDS:
        ratio = stats.unseen_ratios_holdout.get(te_field, 0.0)
        lines.append(f"    {te_field}: {ratio:.4%}")
    lines.append("")
    lines.append("七、各 TE 特征统计（min / max / mean）")
    for te_column in feature_names:
        lines.append(f"  {te_column}:")
        lines.append(f"    最小值：{stats.te_stats.mins.get(te_column)}")
        lines.append(f"    最大值：{stats.te_stats.maxs.get(te_column)}")
        lines.append(f"    均值：  {stats.te_stats.mean(te_column)}")
    lines.append("")
    lines.append("八、数据泄漏防范方式")
    lines.append("  - train 按日期升序处理；日期 d 的 TE 仅使用严格早于 d 的历史累计")
    lines.append("  - 当日全部样本映射完成后，才将当日 click 并入历史状态")
    lines.append("  - valid TE 映射仅由完整 train 聚合，不使用 valid 自身 click")
    lines.append("  - holdout TE 映射仅由 train + valid 聚合，不使用 holdout 自身 click")
    lines.append("  - 禁止使用当前行 click 生成本行 TE 特征")
    lines.append("")
    lines.append("九、输出目录")
    for split_name, output_dir in output_dirs.items():
        lines.append(f"  {split_name}: {output_dir}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{report_path}")


def main() -> None:
    """主流程：train 增量 TE → valid 映射 → holdout 映射 → 报告。"""

    test_mode = TEST_MODE
    output_dirs = get_output_dirs(test_mode)
    report_path = get_report_path(test_mode)
    mode_label = "TEST" if test_mode else "FULL"

    print("=" * 70)
    print("平滑 Target Encoding 特征生成")
    print("=" * 70)
    print(f"当前模式：           {mode_label}")
    print(f"SMOOTHING_STRENGTH： {SMOOTHING_STRENGTH}")
    print(f"DEFAULT_PRIOR：      {DEFAULT_PRIOR}")
    print(f"处理字段：           {TE_FIELDS}")
    print(f"新增特征：           {get_feature_names()}")

    if test_mode:
        train_files = list_parquet_files(
            INPUT_BASE_DIR / "train",
            upstream_script="23_build_historical_features.py",
            max_files=TEST_TRAIN_MAX_FILES,
        )
        valid_files = list_parquet_files(
            INPUT_BASE_DIR / "valid",
            upstream_script="23_build_historical_features.py",
            max_files=TEST_VALID_MAX_FILES,
        )
        holdout_files = list_parquet_files(
            INPUT_BASE_DIR / "holdout",
            upstream_script="23_build_historical_features.py",
            max_files=TEST_HOLDOUT_MAX_FILES,
        )
        # valid/holdout 的 TE 映射仍基于完整 train / train+valid
        train_files_for_mapping = list_parquet_files(
            INPUT_BASE_DIR / "train",
            upstream_script="23_build_historical_features.py",
            max_files=None,
        )
        valid_files_for_mapping = list_parquet_files(
            INPUT_BASE_DIR / "valid",
            upstream_script="23_build_historical_features.py",
            max_files=None,
        )
    else:
        train_files = list_parquet_files(
            INPUT_BASE_DIR / "train",
            upstream_script="23_build_historical_features.py",
        )
        valid_files = list_parquet_files(
            INPUT_BASE_DIR / "valid",
            upstream_script="23_build_historical_features.py",
        )
        holdout_files = list_parquet_files(
            INPUT_BASE_DIR / "holdout",
            upstream_script="23_build_historical_features.py",
        )
        train_files_for_mapping = train_files
        valid_files_for_mapping = valid_files

    print(f"待处理 train 文件数： {len(train_files)}")
    print(f"待处理 valid 文件数： {len(valid_files)}")
    print(f"待处理 holdout 文件数：{len(holdout_files)}")

    warn_if_output_exists(output_dirs)
    ensure_dirs(output_dirs, report_path)

    stats = RunStats()

    # 1. train：按日期递增、先映射后更新
    process_train(train_files, output_dirs["train"], stats)

    # 2. valid：Dask 聚合完整 train TE 映射
    valid_mappings, valid_prior_ctr, valid_has_history = build_te_mapping_dask(
        train_files_for_mapping,
        split_label="train（供 valid 使用）",
    )
    process_split_with_fixed_mapping(
        split_name="valid",
        parquet_files=valid_files,
        output_dir=output_dirs["valid"],
        mappings_by_field=valid_mappings,
        prior_ctr=valid_prior_ctr,
        has_history_window=valid_has_history,
        stats=stats,
        track_unseen=True,
    )

    # 3. holdout：Dask 聚合 train + valid TE 映射
    holdout_source_files = train_files_for_mapping + valid_files_for_mapping
    holdout_mappings, holdout_prior_ctr, holdout_has_history = build_te_mapping_dask(
        holdout_source_files,
        split_label="train + valid（供 holdout 使用）",
    )
    process_split_with_fixed_mapping(
        split_name="holdout",
        parquet_files=holdout_files,
        output_dir=output_dirs["holdout"],
        mappings_by_field=holdout_mappings,
        prior_ctr=holdout_prior_ctr,
        has_history_window=holdout_has_history,
        stats=stats,
        track_unseen=True,
    )

    write_report(stats, report_path, output_dirs, test_mode)

    print("\n" + "=" * 70)
    print("平滑 Target Encoding 特征生成完成")
    print("=" * 70)
    print(f"train 行数：   {stats.train_rows:,}")
    print(f"valid 行数：   {stats.valid_rows:,}")
    print(f"holdout 行数： {stats.holdout_rows:,}")
    print("输出目录：")
    for split_name, output_dir in output_dirs.items():
        file_count = len(list(output_dir.glob("part-*.parquet")))
        print(f"  {split_name}: {output_dir} ({file_count} 个文件)")
    print(f"报告：       {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
