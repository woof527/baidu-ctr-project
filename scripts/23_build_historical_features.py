"""
百度 CTR 项目 — 历史统计特征工程脚本

功能：
    根据历史时间窗口，为类别字段生成历史曝光量、点击量、CTR 和曝光排名分位特征，
    严格避免数据泄漏。

数据输入：
    data/model_input/train/*.parquet
    data/model_input/valid/*.parquet
    data/model_input/holdout/*.parquet

数据输出：
    data/features/historical/train/
    data/features/historical/valid/
    data/features/historical/holdout/
    outputs/feature_tables/historical/
    outputs/23_historical_feature_report.txt

划分规则（与 22_time_split 一致）：
    - train：   2014-10-21 ~ 2014-10-28
    - valid：   2014-10-29
    - holdout： 2014-10-30

用法：
    python scripts/23_build_historical_features.py
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# 保留常量；train 冷启动与未见类别现统一使用 0，不再用 prior 填充 hist_ctr
DEFAULT_PRIOR = 0.17

HIST_FIELDS = [
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_model",
]

INPUT_DIRS = {
    "train": Path("data/model_input/train"),
    "valid": Path("data/model_input/valid"),
    "holdout": Path("data/model_input/holdout"),
}

OUTPUT_DIRS = {
    split_name: Path(f"data/features/historical/{split_name}")
    for split_name in INPUT_DIRS
}

MAPPING_DIR = Path("outputs/feature_tables/historical")
REPORT_PATH = Path("outputs/23_historical_feature_report.txt")

REQUIRED_COLUMNS = ["click", *HIST_FIELDS]

# 与 22_time_split 一致的日期边界
SPLIT_DATE_RANGES = {
    "train": (pd.Timestamp("2014-10-21"), pd.Timestamp("2014-10-28")),
    "valid": (pd.Timestamp("2014-10-29"), pd.Timestamp("2014-10-29")),
    "holdout": (pd.Timestamp("2014-10-30"), pd.Timestamp("2014-10-30")),
}

COLD_START_HIST_FEATURES = [
    name
    for hist_field in HIST_FIELDS
    for name in (
        f"{hist_field}_hist_impressions",
        f"{hist_field}_hist_clicks",
        f"{hist_field}_hist_ctr",
        f"{hist_field}_exposure_percentile",
    )
]


def get_feature_names() -> list[str]:
    """返回 20 个历史统计特征名。"""

    names: list[str] = []
    for hist_field in HIST_FIELDS:
        names.extend(
            [
                f"{hist_field}_hist_impressions",
                f"{hist_field}_hist_clicks",
                f"{hist_field}_hist_ctr",
                f"{hist_field}_exposure_percentile",
            ]
        )
    return names


def list_parquet_files(parquet_dir: Path, upstream_script: str) -> list[Path]:
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

    return files


def count_rows_from_metadata(parquet_files: list[Path]) -> int:
    """使用 Parquet metadata 统计行数。"""

    return sum(pq.read_metadata(path).num_rows for path in parquet_files)


def clean_historical_outputs() -> None:
    """运行前清理 historical 三个 split 目录中的旧 Parquet 输出。"""

    print("\n清理旧的历史特征输出文件...")

    for split_name, output_dir in OUTPUT_DIRS.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        removed = 0

        for parquet_path in sorted(output_dir.glob("part-*.parquet")):
            parquet_path.unlink()
            removed += 1

        print(f"  {output_dir}: 已删除 {removed} 个旧 Parquet 文件")


def ensure_dirs() -> None:
    """运行前自动创建输出目录。"""

    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def resolve_date_source_columns(columns: list[str]) -> list[str]:
    """确定用于提取 event_date 的输入列（优先 date，其次 hour）。"""

    if "date" in columns:
        return ["date"]

    if "hour" in columns:
        return ["hour"]

    raise ValueError("输入数据缺少 date / hour 字段，无法提取 event_date。")


def extract_event_date(dataframe: pd.DataFrame) -> pd.Series:
    """
    提取归一化 event_date。

    优先 date 列；若无则解析 hour 列（格式 YYMMDDHH）。
    """

    if "date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["date"], errors="coerce").dt.normalize()
        if dates.notna().any():
            return dates

    if "hour" in dataframe.columns:
        hour_text = dataframe["hour"].astype(str).str.replace(r"\.0$", "", regex=True)
        hour_text = hour_text.str.zfill(8)
        date_text = (
            "20"
            + hour_text.str.slice(0, 2)
            + "-"
            + hour_text.str.slice(2, 4)
            + "-"
            + hour_text.str.slice(4, 6)
        )
        dates = pd.to_datetime(date_text, errors="coerce").dt.normalize()
        if dates.notna().any():
            return dates

    raise ValueError("无法从 date / hour 字段解析有效 event_date。")


def validate_columns(dataframe: pd.DataFrame, context: str) -> None:
    """检查必需字段是否存在。"""

    missing = [col for col in REQUIRED_COLUMNS if col not in dataframe.columns]
    if missing:
        raise ValueError(f"{context} 缺少必需字段：{missing}")


def feature_column_names(hist_field: str) -> dict[str, str]:
    """单个原始字段对应的 4 个特征列名。"""

    return {
        "hist_impressions": f"{hist_field}_hist_impressions",
        "hist_clicks": f"{hist_field}_hist_clicks",
        "hist_ctr": f"{hist_field}_hist_ctr",
        "exposure_percentile": f"{hist_field}_exposure_percentile",
    }


def finalize_mapping(
    mapping_df: pd.DataFrame,
    hist_field: str,
) -> pd.DataFrame:
    """
    为映射表补全 hist_ctr 与 exposure_percentile。

    - 有历史曝光：hist_ctr = hist_clicks / hist_impressions
    - 无历史曝光：hist_ctr = 0
    - exposure_percentile：对 hist_impressions > 0 的类别 rank(pct=True)，其余为 0
    """

    if mapping_df.empty:
        return pd.DataFrame(
            columns=[
                hist_field,
                "hist_impressions",
                "hist_clicks",
                "hist_ctr",
                "exposure_percentile",
            ]
        )

    result = mapping_df.copy()

    result["hist_ctr"] = np.where(
        result["hist_impressions"] > 0,
        result["hist_clicks"] / result["hist_impressions"],
        0.0,
    )

    result["exposure_percentile"] = 0.0
    positive_mask = result["hist_impressions"] > 0
    if positive_mask.any():
        result.loc[positive_mask, "exposure_percentile"] = (
            result.loc[positive_mask, "hist_impressions"].rank(pct=True)
        )

    result["hist_impressions"] = result["hist_impressions"].astype("int64")
    result["hist_clicks"] = result["hist_clicks"].astype("int64")
    result["hist_ctr"] = result["hist_ctr"].astype("float64")
    result["exposure_percentile"] = result["exposure_percentile"].astype("float64")

    return result


@dataclass
class HistoryState:
    """累计历史状态：仅包含严格早于当前日的统计。"""

    category_stats: dict[str, dict[object, dict[str, int]]] = field(
        default_factory=lambda: {field_name: {} for field_name in HIST_FIELDS}
    )
    total_impressions: int = 0
    total_clicks: int = 0
    processed_dates: list[pd.Timestamp] = field(default_factory=list)

    def build_mapping(self, hist_field: str) -> pd.DataFrame:
        """由当前累计状态构建单个字段的历史映射表。"""

        rows: list[dict] = []
        for category, stats in self.category_stats[hist_field].items():
            rows.append(
                {
                    hist_field: category,
                    "hist_impressions": stats["impressions"],
                    "hist_clicks": stats["clicks"],
                }
            )

        mapping_df = pd.DataFrame(rows)
        return finalize_mapping(mapping_df, hist_field)

    def build_all_mappings(self) -> dict[str, pd.DataFrame]:
        """构建 5 个字段的历史映射表。"""

        return {hist_field: self.build_mapping(hist_field) for hist_field in HIST_FIELDS}

    def update_from_daily_stats(
        self,
        daily_field_stats: dict[str, dict[object, dict[str, int]]],
        event_date: pd.Timestamp,
    ) -> None:
        """将某一天的类别统计并入累计历史。"""

        day_impressions = 0
        day_clicks = 0

        for hist_field in HIST_FIELDS:
            field_stats = self.category_stats[hist_field]

            for category, stats in daily_field_stats.get(hist_field, {}).items():
                if category not in field_stats:
                    field_stats[category] = {"impressions": 0, "clicks": 0}

                field_stats[category]["impressions"] += int(stats["impressions"])
                field_stats[category]["clicks"] += int(stats["clicks"])
                day_impressions += int(stats["impressions"])
                day_clicks += int(stats["clicks"])

        self.total_impressions += day_impressions
        self.total_clicks += day_clicks

        if event_date not in self.processed_dates:
            self.processed_dates.append(event_date)


DailyStats = dict[pd.Timestamp, dict[str, dict[object, dict[str, int]]]]


def aggregate_train_daily_stats(train_files: list[Path]) -> DailyStats:
    """
    第一阶段：逐文件汇总 train 的真实 event_date × 类别 日统计。

    不依赖文件级日期，同一文件内不同 event_date 会分别累计。
    """

    daily_stats: DailyStats = defaultdict(
        lambda: {field_name: {} for field_name in HIST_FIELDS}
    )

    print("\n[train 阶段 1/3] 按 event_date 汇总每日 impressions / clicks ...")

    for file_index, input_path in enumerate(train_files, start=1):
        schema_columns = pq.read_schema(input_path).names
        date_columns = resolve_date_source_columns(schema_columns)
        read_columns = list(dict.fromkeys(["click", *date_columns, *HIST_FIELDS]))

        chunk = pd.read_parquet(input_path, columns=read_columns)
        if chunk.empty:
            raise ValueError(f"输入文件为空：{input_path}")

        validate_columns(chunk, context=str(input_path))
        event_dates = extract_event_date(chunk)

        if event_dates.isna().any():
            invalid_count = int(event_dates.isna().sum())
            raise ValueError(f"{input_path} 存在 {invalid_count} 行无法解析 event_date")

        chunk = chunk.assign(event_date=event_dates)

        for event_date, day_chunk in chunk.groupby("event_date", sort=False):
            normalized_date = pd.Timestamp(event_date)
            day_field_stats = daily_stats[normalized_date]

            for hist_field in HIST_FIELDS:
                grouped = (
                    day_chunk.groupby(hist_field, dropna=False)["click"]
                    .agg(clicks="sum", impressions="count")
                    .reset_index()
                )

                field_stats = day_field_stats[hist_field]
                for _, row in grouped.iterrows():
                    category = row[hist_field]
                    if category not in field_stats:
                        field_stats[category] = {"impressions": 0, "clicks": 0}

                    field_stats[category]["impressions"] += int(row["impressions"])
                    field_stats[category]["clicks"] += int(row["clicks"])

        if file_index % 20 == 0 or file_index == len(train_files):
            print(f"  已汇总 {file_index}/{len(train_files)} 个文件")

        del chunk
        gc.collect()

    return dict(daily_stats)


def build_train_mappings_by_date(
    daily_stats: DailyStats,
) -> tuple[dict[pd.Timestamp, dict[str, pd.DataFrame]], list[pd.Timestamp]]:
    """
    第二阶段：按全局日期升序，构建“截至前一天”的累计历史映射。

    对 current_date：
        1. 先保存严格早于 current_date 的映射
        2. 再把 current_date 的日统计并入累计状态
    """

    sorted_dates = sorted(daily_stats.keys())
    cumulative_state = HistoryState()
    mappings_by_date: dict[pd.Timestamp, dict[str, pd.DataFrame]] = {}

    print("\n[train 阶段 2/3] 按全局日期升序构建截至前一天的历史映射 ...")

    for current_date in sorted_dates:
        mappings_by_date[current_date] = cumulative_state.build_all_mappings()
        cumulative_state.update_from_daily_stats(
            daily_stats[current_date],
            current_date,
        )
        print(f"  日期 {current_date.date()} 映射已就绪（严格早于该日的历史）")

    return mappings_by_date, sorted_dates


def apply_historical_features(
    dataframe: pd.DataFrame,
    mappings_by_field: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    将历史映射特征合并到 DataFrame。

    未见类别或冷启动：hist_impressions/clicks/ctr/exposure_percentile 均为 0。
    """

    validate_columns(dataframe, context="特征映射")
    result = dataframe.copy()

    for hist_field in HIST_FIELDS:
        col_names = feature_column_names(hist_field)
        mapping = mappings_by_field[hist_field]

        merge_cols = [
            hist_field,
            "hist_impressions",
            "hist_clicks",
            "hist_ctr",
            "exposure_percentile",
        ]
        mapping_to_merge = mapping[merge_cols].rename(
            columns={
                "hist_impressions": col_names["hist_impressions"],
                "hist_clicks": col_names["hist_clicks"],
                "hist_ctr": col_names["hist_ctr"],
                "exposure_percentile": col_names["exposure_percentile"],
            }
        )

        result = result.merge(mapping_to_merge, on=hist_field, how="left")

        result[col_names["hist_impressions"]] = (
            result[col_names["hist_impressions"]].fillna(0).astype("int64")
        )
        result[col_names["hist_clicks"]] = (
            result[col_names["hist_clicks"]].fillna(0).astype("int64")
        )
        result[col_names["hist_ctr"]] = (
            result[col_names["hist_ctr"]].fillna(0.0).astype("float64")
        )
        result[col_names["exposure_percentile"]] = (
            result[col_names["exposure_percentile"]].fillna(0.0).astype("float64")
        )

    return result


def apply_historical_features_by_row_date(
    dataframe: pd.DataFrame,
    event_dates: pd.Series,
    mappings_by_date: dict[pd.Timestamp, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """
    按行 event_date 匹配严格早于当天的历史映射。

    同一 Parquet 文件内若含多个日期，也会分别映射，且保持原始行顺序不变。
    """

    result = dataframe.copy()

    for feature_name in get_feature_names():
        if feature_name.endswith("_hist_impressions") or feature_name.endswith("_hist_clicks"):
            result[feature_name] = 0
        else:
            result[feature_name] = 0.0

    for event_date in pd.Series(event_dates.dropna().unique()).sort_values():
        normalized_date = pd.Timestamp(event_date)
        row_mask = event_dates == normalized_date

        if not row_mask.any():
            continue

        date_mappings = mappings_by_date.get(normalized_date)
        if date_mappings is None:
            continue

        featured_subset = apply_historical_features(
            dataframe.loc[row_mask],
            date_mappings,
        )

        for feature_name in get_feature_names():
            result.loc[row_mask, feature_name] = featured_subset[feature_name].to_numpy()

    return result


def update_feature_bounds(
    dataframe: pd.DataFrame,
    bounds: dict[str, float | None],
) -> None:
    """累计 hist_ctr 与 exposure_percentile 的最小/最大值。"""

    for hist_field in HIST_FIELDS:
        col_names = feature_column_names(hist_field)

        for metric_key, column_name in [
            ("hist_ctr", col_names["hist_ctr"]),
            ("exposure_percentile", col_names["exposure_percentile"]),
        ]:
            series = dataframe[column_name]
            current_min = float(series.min())
            current_max = float(series.max())

            min_key = f"{metric_key}_min"
            max_key = f"{metric_key}_max"

            bounds[min_key] = (
                current_min
                if bounds[min_key] is None
                else min(bounds[min_key], current_min)
            )
            bounds[max_key] = (
                current_max
                if bounds[max_key] is None
                else max(bounds[max_key], current_max)
            )


def count_unseen_rows(dataframe: pd.DataFrame, hist_field: str) -> int:
    """统计未见类别行数（hist_impressions == 0）。"""

    col_name = feature_column_names(hist_field)["hist_impressions"]
    return int((dataframe[col_name] == 0).sum())


def build_history_mapping_dask(
    parquet_files: list[Path],
    split_label: str,
) -> dict[str, pd.DataFrame]:
    """
    使用 Dask 聚合历史映射（用于 valid / holdout 的先验历史）。

    只读取 click 与类别字段，避免全列加载。
    """

    if not parquet_files:
        raise ValueError(f"{split_label} 聚合时没有可用的 Parquet 文件。")

    paths = [str(path) for path in parquet_files]
    columns = ["click", *HIST_FIELDS]

    print(f"\n使用 Dask 聚合 {split_label} 历史映射...")
    print(f"  参与文件数：{len(parquet_files)}")

    ddf = dd.read_parquet(paths, columns=columns)

    total_impressions = int(ddf.shape[0].compute())
    total_clicks = int(ddf["click"].sum().compute())

    print(f"  历史总曝光：{total_impressions:,}")
    print(f"  历史总点击：{total_clicks:,}")
    if total_impressions > 0:
        print(f"  prior_ctr：  {total_clicks / total_impressions:.6f}")

    mappings: dict[str, pd.DataFrame] = {}

    for hist_field in HIST_FIELDS:
        print(f"  当前处理字段：{hist_field}")

        aggregated = (
            ddf.groupby(hist_field, dropna=False)
            .agg(
                hist_clicks=("click", "sum"),
                hist_impressions=("click", "count"),
            )
            .compute()
            .reset_index()
        )

        mappings[hist_field] = finalize_mapping(aggregated, hist_field)
        print(f"    唯一类别数：{len(mappings[hist_field]):,}")

    return mappings


def save_mapping_tables(
    mappings_by_field: dict[str, pd.DataFrame],
    suffix: str,
) -> None:
    """保存历史映射表到 outputs/feature_tables/historical/。"""

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n保存历史映射表（{suffix}）...")

    for hist_field, mapping_df in mappings_by_field.items():
        output_path = MAPPING_DIR / f"{hist_field}_{suffix}_history_mapping.parquet"
        mapping_df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"  已保存：{output_path}")


def process_parquet_with_mapping(
    input_path: Path,
    output_path: Path,
    mappings_by_field: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """读取单个分块，映射历史特征并写出。"""

    chunk = pd.read_parquet(input_path)
    if chunk.empty:
        raise ValueError(f"输入文件为空：{input_path}")

    validate_columns(chunk, context=str(input_path))
    featured_chunk = apply_historical_features(chunk, mappings_by_field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

    return featured_chunk


@dataclass
class RunStats:
    """全流程统计信息，用于报告。"""

    train_rows: int = 0
    valid_rows: int = 0
    holdout_rows: int = 0
    train_input_rows: int = 0
    valid_input_rows: int = 0
    holdout_input_rows: int = 0
    train_date_start: str | None = None
    train_date_end: str | None = None
    train_earliest_date: str | None = None
    valid_date_start: str | None = None
    valid_date_end: str | None = None
    holdout_date_start: str | None = None
    holdout_date_end: str | None = None
    unseen_ratios_valid: dict[str, float] = field(default_factory=dict)
    unseen_ratios_holdout: dict[str, float] = field(default_factory=dict)
    feature_bounds: dict[str, float | None] = field(
        default_factory=lambda: {
            "hist_ctr_min": None,
            "hist_ctr_max": None,
            "exposure_percentile_min": None,
            "exposure_percentile_max": None,
        }
    )


def process_train(
    train_files: list[Path],
    stats: RunStats,
) -> tuple[pd.Timestamp, dict[str, int]]:
    """
    train 三阶段处理：
    1. 汇总每日统计
    2. 构建截至前一天的历史映射
    3. 逐文件、逐行 event_date 映射并写出
    """

    output_dir = OUTPUT_DIRS["train"]

    daily_stats = aggregate_train_daily_stats(train_files)
    mappings_by_date, sorted_dates = build_train_mappings_by_date(daily_stats)

    if sorted_dates:
        stats.train_date_start = sorted_dates[0].date().isoformat()
        stats.train_date_end = sorted_dates[-1].date().isoformat()
        stats.train_earliest_date = stats.train_date_start

    print("\n[train 阶段 3/3] 逐文件按行 event_date 映射历史特征 ...")
    print("\n" + "=" * 70)
    print("处理 split: train")
    print("=" * 70)

    earliest_date = sorted_dates[0] if sorted_dates else None
    earliest_violations = {feature_name: 0 for feature_name in COLD_START_HIST_FEATURES}

    for file_index, input_path in enumerate(train_files, start=1):
        schema_columns = pq.read_schema(input_path).names
        date_columns = resolve_date_source_columns(schema_columns)

        chunk = pd.read_parquet(input_path)
        if chunk.empty:
            raise ValueError(f"输入文件为空：{input_path}")

        validate_columns(chunk, context=str(input_path))
        event_dates = extract_event_date(chunk)

        if event_dates.isna().any():
            invalid_count = int(event_dates.isna().sum())
            raise ValueError(f"{input_path} 存在 {invalid_count} 行无法解析 event_date")

        featured_chunk = apply_historical_features_by_row_date(
            chunk,
            event_dates,
            mappings_by_date,
        )

        output_path = output_dir / input_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

        row_count = len(featured_chunk)
        stats.train_rows += row_count
        update_feature_bounds(featured_chunk, stats.feature_bounds)

        if earliest_date is not None:
            earliest_mask = event_dates == earliest_date
            if earliest_mask.any():
                earliest_subset = featured_chunk.loc[earliest_mask]
                for feature_name in COLD_START_HIST_FEATURES:
                    values = pd.to_numeric(earliest_subset[feature_name], errors="coerce").fillna(-1)
                    earliest_violations[feature_name] += int((values != 0).sum())

        print("  当前 split：train")
        print(f"  当前输入文件：{input_path}")
        print(f"  当前输出文件：{output_path}")
        print(f"  当前文件行数：{row_count:,}")
        print(
            f"  当前累计进度：{file_index}/{len(train_files)} 文件 "
            f"({file_index / len(train_files) * 100:.1f}%)，train 累计行数 {stats.train_rows:,}"
        )

        del chunk, featured_chunk, event_dates
        gc.collect()

    return earliest_date, earliest_violations


def process_split_with_fixed_mapping(
    split_name: str,
    parquet_files: list[Path],
    mappings_by_field: dict[str, pd.DataFrame],
    stats: RunStats,
    track_unseen: bool,
) -> None:
    """valid / holdout：使用固定先验历史映射，逐文件读写。"""

    output_dir = OUTPUT_DIRS[split_name]
    total_files = len(parquet_files)
    unseen_counts: dict[str, int] = {field_name: 0 for field_name in HIST_FIELDS}
    total_rows = 0

    start_date, end_date = SPLIT_DATE_RANGES[split_name]
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

    for file_index, input_path in enumerate(parquet_files, start=1):
        for hist_field in HIST_FIELDS:
            print(f"  当前处理字段：{hist_field}")

        output_path = output_dir / input_path.name
        print(f"  当前 split：{split_name}")
        print(f"  当前输入文件：{input_path}")
        print(f"  当前输出文件：{output_path}")

        featured_chunk = process_parquet_with_mapping(
            input_path,
            output_path,
            mappings_by_field,
        )

        row_count = len(featured_chunk)
        total_rows += row_count
        update_feature_bounds(featured_chunk, stats.feature_bounds)

        if track_unseen:
            for hist_field in HIST_FIELDS:
                unseen_counts[hist_field] += count_unseen_rows(featured_chunk, hist_field)

        progress = file_index / total_files * 100
        print(f"  当前文件行数：{row_count:,}")
        print(
            f"  当前累计进度：{file_index}/{total_files} 文件 "
            f"({progress:.1f}%)，{split_name} 累计行数 {total_rows:,}"
        )

        del featured_chunk
        gc.collect()

    if split_name == "valid":
        stats.valid_rows = total_rows
        stats.valid_date_start = start_date.date().isoformat()
        stats.valid_date_end = end_date.date().isoformat()
        if track_unseen and total_rows > 0:
            stats.unseen_ratios_valid = {
                field_name: unseen_counts[field_name] / total_rows
                for field_name in HIST_FIELDS
            }

    if split_name == "holdout":
        stats.holdout_rows = total_rows
        stats.holdout_date_start = start_date.date().isoformat()
        stats.holdout_date_end = end_date.date().isoformat()
        if track_unseen and total_rows > 0:
            stats.unseen_ratios_holdout = {
                field_name: unseen_counts[field_name] / total_rows
                for field_name in HIST_FIELDS
            }


def write_report(stats: RunStats) -> None:
    """生成 outputs/23_historical_feature_report.txt。"""

    feature_names = get_feature_names()
    lines: list[str] = []

    lines.append("百度 CTR 项目 — 历史统计特征工程报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append("一、输入数据范围")
    lines.append(f"  train 输入目录：   {INPUT_DIRS['train']}")
    lines.append(f"  valid 输入目录：   {INPUT_DIRS['valid']}")
    lines.append(f"  holdout 输入目录： {INPUT_DIRS['holdout']}")
    lines.append("")
    lines.append("二、各 split 日期范围")
    lines.append(
        f"  train：   {stats.train_date_start} ~ {stats.train_date_end}"
    )
    lines.append(
        f"  valid：   {stats.valid_date_start} ~ {stats.valid_date_end}"
    )
    lines.append(
        f"  holdout： {stats.holdout_date_start} ~ {stats.holdout_date_end}"
    )
    lines.append("")
    lines.append("三、处理字段")
    for hist_field in HIST_FIELDS:
        lines.append(f"  - {hist_field}")
    lines.append("")
    lines.append("四、新增特征（共 20 个）")
    for feature_name in feature_names:
        lines.append(f"  - {feature_name}")
    lines.append("")
    lines.append("五、各 split 总行数")
    lines.append(f"  train 输入：   {stats.train_input_rows:,}")
    lines.append(f"  train 输出：   {stats.train_rows:,}")
    lines.append(f"  valid 输入：   {stats.valid_input_rows:,}")
    lines.append(f"  valid 输出：   {stats.valid_rows:,}")
    lines.append(f"  holdout 输入： {stats.holdout_input_rows:,}")
    lines.append(f"  holdout 输出： {stats.holdout_rows:,}")
    lines.append("")
    lines.append("六、valid / holdout 未见类别比例（hist_impressions == 0）")
    lines.append("  valid：")
    for hist_field in HIST_FIELDS:
        ratio = stats.unseen_ratios_valid.get(hist_field, 0.0)
        lines.append(f"    {hist_field}: {ratio:.4%}")
    lines.append("  holdout：")
    for hist_field in HIST_FIELDS:
        ratio = stats.unseen_ratios_holdout.get(hist_field, 0.0)
        lines.append(f"    {hist_field}: {ratio:.4%}")
    lines.append("")
    lines.append("七、特征取值范围")
    lines.append(f"  hist_ctr 最小值：           {stats.feature_bounds['hist_ctr_min']}")
    lines.append(f"  hist_ctr 最大值：           {stats.feature_bounds['hist_ctr_max']}")
    lines.append(
        f"  exposure_percentile 最小值：{stats.feature_bounds['exposure_percentile_min']}"
    )
    lines.append(
        f"  exposure_percentile 最大值：{stats.feature_bounds['exposure_percentile_max']}"
    )
    lines.append("")
    lines.append("八、数据泄漏防范方式")
    lines.append("  - train 先汇总真实 event_date 日统计，再按全局日期升序构建截至前一天的映射")
    lines.append("  - 写特征时按行 event_date 匹配映射，同一 Parquet 多日期也分别处理")
    lines.append("  - current_date 全部样本映射完成后，才将 current_date 并入累计历史")
    lines.append("  - valid 映射仅由完整 train 聚合，不使用 valid 自身 click")
    lines.append("  - holdout 映射仅由 train + valid 聚合，不使用 holdout 自身 click")
    lines.append("  - 禁止使用当前行 click 生成本行历史特征")
    lines.append("")
    lines.append("九、冷启动处理")
    lines.append("  - train 最早日期与未见类别：hist_impressions=0, hist_clicks=0")
    lines.append("  - hist_ctr=0, exposure_percentile=0")
    lines.append("")
    lines.append("十、输出目录")
    lines.append(f"  特征：{OUTPUT_DIRS['train'].parent}/")
    lines.append(f"  映射：{MAPPING_DIR}/")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{REPORT_PATH}")


def print_run_summary(
    stats: RunStats,
    earliest_date: pd.Timestamp | None,
    earliest_violations: dict[str, int],
) -> None:
    """运行结束后打印汇总，并在最早日期存在非零历史计数时报错。"""

    print("\n" + "=" * 70)
    print("历史统计特征工程完成")
    print("=" * 70)
    print("各 split 输入 / 输出行数：")
    print(f"  train：   输入 {stats.train_input_rows:,} → 输出 {stats.train_rows:,}")
    print(f"  valid：   输入 {stats.valid_input_rows:,} → 输出 {stats.valid_rows:,}")
    print(f"  holdout： 输入 {stats.holdout_input_rows:,} → 输出 {stats.holdout_rows:,}")
    print(f"train 日期范围：{stats.train_date_start} ~ {stats.train_date_end}")
    print(f"train 最早日期：{stats.train_earliest_date}")

    if earliest_date is not None:
        print(f"\ntrain 最早日期 {earliest_date.date()} 冷启动异常统计：")
        total_violations = 0
        for feature_name in COLD_START_HIST_FEATURES:
            violation_count = earliest_violations.get(feature_name, 0)
            total_violations += violation_count
            print(f"  {feature_name}: 非零 {violation_count:,}")

        if total_violations > 0:
            raise ValueError(
                f"train 最早日期 {earliest_date.date()} 存在 {total_violations:,} 个非零历史特征值，"
                "冷启动检查失败。"
            )

        print("  最早日期冷启动检查：通过")

    print("\n输出目录：")
    for split_name, output_dir in OUTPUT_DIRS.items():
        file_count = len(list(output_dir.glob("part-*.parquet")))
        print(f"  {split_name}: {output_dir} ({file_count} 个文件)")
    print(f"映射表目录： {MAPPING_DIR}")
    print(f"报告：       {REPORT_PATH}")
    print("=" * 70)


def main() -> None:
    """主流程：清理旧输出 → train 三阶段 → valid/holdout 映射 → 报告。"""

    print("=" * 70)
    print("历史统计特征工程")
    print("=" * 70)
    print(f"处理字段：     {HIST_FIELDS}")
    print(f"新增特征数：   {len(get_feature_names())}")

    train_files = list_parquet_files(INPUT_DIRS["train"], upstream_script="22_time_split.py")
    valid_files = list_parquet_files(INPUT_DIRS["valid"], upstream_script="22_time_split.py")
    holdout_files = list_parquet_files(
        INPUT_DIRS["holdout"], upstream_script="22_time_split.py"
    )

    clean_historical_outputs()
    ensure_dirs()

    stats = RunStats()
    stats.train_input_rows = count_rows_from_metadata(train_files)
    stats.valid_input_rows = count_rows_from_metadata(valid_files)
    stats.holdout_input_rows = count_rows_from_metadata(holdout_files)

    # 1. train：三阶段按行 event_date 处理
    earliest_date, earliest_violations = process_train(train_files, stats)

    if stats.train_rows != stats.train_input_rows:
        raise ValueError(
            f"train 输出行数 {stats.train_rows:,} 与输入行数 {stats.train_input_rows:,} 不一致"
        )

    # 2. valid：Dask 聚合 train 历史
    valid_mappings = build_history_mapping_dask(
        train_files,
        split_label="train（供 valid 使用）",
    )
    save_mapping_tables(valid_mappings, suffix="train")
    process_split_with_fixed_mapping(
        split_name="valid",
        parquet_files=valid_files,
        mappings_by_field=valid_mappings,
        stats=stats,
        track_unseen=True,
    )

    if stats.valid_rows != stats.valid_input_rows:
        raise ValueError(
            f"valid 输出行数 {stats.valid_rows:,} 与输入行数 {stats.valid_input_rows:,} 不一致"
        )

    # 3. holdout：Dask 聚合 train + valid 历史
    holdout_source_files = train_files + valid_files
    holdout_mappings = build_history_mapping_dask(
        holdout_source_files,
        split_label="train + valid（供 holdout 使用）",
    )
    save_mapping_tables(holdout_mappings, suffix="train_valid")
    process_split_with_fixed_mapping(
        split_name="holdout",
        parquet_files=holdout_files,
        mappings_by_field=holdout_mappings,
        stats=stats,
        track_unseen=True,
    )

    if stats.holdout_rows != stats.holdout_input_rows:
        raise ValueError(
            f"holdout 输出行数 {stats.holdout_rows:,} 与输入行数 "
            f"{stats.holdout_input_rows:,} 不一致"
        )

    write_report(stats)
    print_run_summary(stats, earliest_date, earliest_violations)


if __name__ == "__main__":
    main()
