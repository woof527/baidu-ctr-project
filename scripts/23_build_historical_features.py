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

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------

# 冷启动先验 CTR：无历史窗口时使用，不从当前或未来数据计算
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

REQUIRED_COLUMNS = ["hour_dt", "click", *HIST_FIELDS]

# 与 22_time_split 一致的日期边界
SPLIT_DATE_RANGES = {
    "train": (pd.Timestamp("2014-10-21"), pd.Timestamp("2014-10-28")),
    "valid": (pd.Timestamp("2014-10-29"), pd.Timestamp("2014-10-29")),
    "holdout": (pd.Timestamp("2014-10-30"), pd.Timestamp("2014-10-30")),
}


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


def warn_if_output_exists() -> None:
    """若输出目录已有旧 Parquet，提示用户避免混用结果。"""

    for split_name, output_dir in OUTPUT_DIRS.items():
        if not output_dir.exists():
            continue

        existing = sorted(output_dir.glob("part-*.parquet"))
        if existing:
            print(
                f"WARNING: {output_dir} 已存在 {len(existing)} 个 Parquet 文件，"
                "继续运行可能覆盖或与旧结果混在一起。"
            )


def ensure_dirs() -> None:
    """运行前自动创建输出目录。"""

    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


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
    prior_ctr: float,
    has_history_window: bool,
) -> pd.DataFrame:
    """
    为映射表补全 hist_ctr 与 exposure_percentile。

    - 有历史曝光：hist_ctr = hist_clicks / hist_impressions
    - 无历史曝光：hist_ctr = prior_ctr（有历史窗口）或 DEFAULT_PRIOR（冷启动）
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
    fallback_ctr = prior_ctr if has_history_window else DEFAULT_PRIOR

    result["hist_ctr"] = np.where(
        result["hist_impressions"] > 0,
        result["hist_clicks"] / result["hist_impressions"],
        fallback_ctr,
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
    """train 增量历史状态：仅包含已处理日期（严格早于当前日）的累计统计。"""

    category_stats: dict[str, dict[object, dict[str, int]]] = field(
        default_factory=lambda: {field_name: {} for field_name in HIST_FIELDS}
    )
    total_impressions: int = 0
    total_clicks: int = 0
    processed_dates: list[pd.Timestamp] = field(default_factory=list)

    def has_history_window(self) -> bool:
        """是否已有历史窗口（即是否处理过至少一个日期）。"""

        return self.total_impressions > 0

    def prior_ctr(self) -> float:
        """当前历史窗口的全局 CTR。"""

        if self.total_impressions <= 0:
            return DEFAULT_PRIOR

        return self.total_clicks / self.total_impressions

    def history_window_label(self, current_date: pd.Timestamp) -> str:
        """用于日志：当前日可用的历史日期范围。"""

        if not self.processed_dates:
            return "无（冷启动）"

        start_date = min(self.processed_dates).date().isoformat()
        end_date = max(self.processed_dates).date().isoformat()
        return f"{start_date} ~ {end_date}（严格早于 {current_date.date().isoformat()}）"

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
        if mapping_df.empty:
            return finalize_mapping(
                mapping_df,
                hist_field,
                self.prior_ctr(),
                self.has_history_window(),
            )

        return finalize_mapping(
            mapping_df,
            hist_field,
            self.prior_ctr(),
            self.has_history_window(),
        )

    def build_all_mappings(self) -> dict[str, pd.DataFrame]:
        """构建 5 个字段的历史映射表。"""

        return {hist_field: self.build_mapping(hist_field) for hist_field in HIST_FIELDS}

    def update_from_dataframe(self, dataframe: pd.DataFrame, event_date: pd.Timestamp) -> None:
        """将当日数据累计进历史状态（必须在当日特征映射完成后调用）。"""

        validate_columns(dataframe, context=f"日期 {event_date.date()}")

        chunk_impressions = len(dataframe)
        chunk_clicks = int(dataframe["click"].sum())

        self.total_impressions += chunk_impressions
        self.total_clicks += chunk_clicks

        for hist_field in HIST_FIELDS:
            grouped = (
                dataframe.groupby(hist_field, dropna=False)["click"]
                .agg(clicks="sum", impressions="count")
                .reset_index()
            )

            field_stats = self.category_stats[hist_field]
            for _, row in grouped.iterrows():
                category = row[hist_field]
                if category not in field_stats:
                    field_stats[category] = {"impressions": 0, "clicks": 0}

                field_stats[category]["impressions"] += int(row["impressions"])
                field_stats[category]["clicks"] += int(row["clicks"])

        if event_date not in self.processed_dates:
            self.processed_dates.append(event_date)


def apply_historical_features(
    dataframe: pd.DataFrame,
    mappings_by_field: dict[str, pd.DataFrame],
    prior_ctr: float,
    has_history_window: bool,
) -> pd.DataFrame:
    """
    将历史映射特征合并到 DataFrame。

    未见类别：hist_impressions/clicks=0，hist_ctr=prior_ctr 或 DEFAULT_PRIOR，
    exposure_percentile=0。
    """

    validate_columns(dataframe, context="特征映射")
    result = dataframe.copy()
    fallback_ctr = prior_ctr if has_history_window else DEFAULT_PRIOR

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
        result[col_names["hist_ctr"]] = result[col_names["hist_ctr"]].fillna(fallback_ctr)
        result[col_names["exposure_percentile"]] = (
            result[col_names["exposure_percentile"]].fillna(0.0).astype("float64")
        )

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


def build_history_mapping_dask(
    parquet_files: list[Path],
    split_label: str,
) -> tuple[dict[str, pd.DataFrame], float, bool]:
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
    has_history_window = total_impressions > 0
    prior_ctr = (
        total_clicks / total_impressions if has_history_window else DEFAULT_PRIOR
    )

    print(f"  历史总曝光：{total_impressions:,}")
    print(f"  历史总点击：{total_clicks:,}")
    print(f"  prior_ctr：  {prior_ctr:.6f}")

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

        mappings[hist_field] = finalize_mapping(
            aggregated,
            hist_field,
            prior_ctr,
            has_history_window,
        )
        print(f"    唯一类别数：{len(mappings[hist_field]):,}")

    return mappings, prior_ctr, has_history_window


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
    prior_ctr: float,
    has_history_window: bool,
) -> pd.DataFrame:
    """读取单个分块，映射历史特征并写出。"""

    chunk = pd.read_parquet(input_path)
    if chunk.empty:
        raise ValueError(f"输入文件为空：{input_path}")

    validate_columns(chunk, context=str(input_path))
    featured_chunk = apply_historical_features(
        chunk,
        mappings_by_field,
        prior_ctr,
        has_history_window,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

    return featured_chunk


@dataclass
class RunStats:
    """全流程统计信息，用于报告。"""

    train_rows: int = 0
    valid_rows: int = 0
    holdout_rows: int = 0
    train_date_start: str | None = None
    train_date_end: str | None = None
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
) -> None:
    """
    train 按日期升序处理：
    1. 用严格早于当前日的历史状态建映射
    2. 映射并写出当前日所有分块
    3. 再将当前日 click/曝光累计进状态
    """

    date_to_files = build_date_file_index(train_files)
    total_files = sum(len(files) for files in date_to_files.values())
    completed_files = 0

    if date_to_files:
        stats.train_date_start = min(date_to_files).date().isoformat()
        stats.train_date_end = max(date_to_files).date().isoformat()

    state = HistoryState()
    output_dir = OUTPUT_DIRS["train"]

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
            for hist_field in HIST_FIELDS:
                print(f"  当前处理字段：{hist_field}")

            output_path = output_dir / input_path.name
            print("  当前 split：train")
            print(f"  当前输入文件：{input_path}")
            print(f"  当前输出文件：{output_path}")

            chunk = pd.read_parquet(input_path)
            if chunk.empty:
                raise ValueError(f"输入文件为空：{input_path}")

            validate_columns(chunk, context=str(input_path))
            featured_chunk = apply_historical_features(
                chunk,
                mappings,
                prior_ctr,
                has_history_window,
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

            row_count = len(featured_chunk)
            stats.train_rows += row_count
            update_feature_bounds(featured_chunk, stats.feature_bounds)
            pending_updates.append(chunk)

            completed_files += 1
            progress = completed_files / total_files * 100
            print(f"  当前文件行数：{row_count:,}")
            print(
                f"  当前累计进度：{completed_files}/{total_files} 文件 "
                f"({progress:.1f}%)，train 累计行数 {stats.train_rows:,}"
            )

        # 当日全部文件映射完成后，才将当日 click/曝光并入历史状态
        for chunk in pending_updates:
            state.update_from_dataframe(chunk, event_date)


def process_split_with_fixed_mapping(
    split_name: str,
    parquet_files: list[Path],
    mappings_by_field: dict[str, pd.DataFrame],
    prior_ctr: float,
    has_history_window: bool,
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
    print(f"prior_ctr：{prior_ctr:.6f}")

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
            prior_ctr,
            has_history_window,
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
    lines.append(f"  train：   {stats.train_rows:,}")
    lines.append(f"  valid：   {stats.valid_rows:,}")
    lines.append(f"  holdout： {stats.holdout_rows:,}")
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
    lines.append("  - train 按日期升序处理；日期 d 的特征仅使用严格早于 d 的历史累计")
    lines.append("  - 当日全部样本映射完成后，才将当日 click/曝光并入历史状态")
    lines.append("  - valid 映射仅由完整 train 聚合，不使用 valid 自身 click")
    lines.append("  - holdout 映射仅由 train + valid 聚合，不使用 holdout 自身 click")
    lines.append("  - 禁止使用当前行 click 生成本行历史特征")
    lines.append("")
    lines.append("九、第一日冷启动处理")
    lines.append(f"  - DEFAULT_PRIOR = {DEFAULT_PRIOR}")
    lines.append("  - 第一日无历史窗口：hist_impressions=0, hist_clicks=0")
    lines.append(f"  - hist_ctr 使用 DEFAULT_PRIOR（{DEFAULT_PRIOR}）")
    lines.append("  - exposure_percentile = 0")
    lines.append("")
    lines.append("十、未见类别处理")
    lines.append("  - hist_impressions = 0, hist_clicks = 0")
    lines.append("  - hist_ctr 使用当前历史窗口 prior_ctr（全局 CTR）")
    lines.append(f"  - 若无历史窗口则使用 DEFAULT_PRIOR（{DEFAULT_PRIOR}）")
    lines.append("  - exposure_percentile = 0")
    lines.append("")
    lines.append("十一、输出目录")
    lines.append(f"  特征：{OUTPUT_DIRS['train'].parent}/")
    lines.append(f"  映射：{MAPPING_DIR}/")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{REPORT_PATH}")


def main() -> None:
    """主流程：train 增量 → valid 映射 → holdout 映射 → 报告。"""

    print("=" * 70)
    print("历史统计特征工程")
    print("=" * 70)
    print(f"DEFAULT_PRIOR：{DEFAULT_PRIOR}")
    print(f"处理字段：     {HIST_FIELDS}")
    print(f"新增特征数：   {len(get_feature_names())}")

    train_files = list_parquet_files(INPUT_DIRS["train"], upstream_script="22_time_split.py")
    valid_files = list_parquet_files(INPUT_DIRS["valid"], upstream_script="22_time_split.py")
    holdout_files = list_parquet_files(
        INPUT_DIRS["holdout"], upstream_script="22_time_split.py"
    )

    warn_if_output_exists()
    ensure_dirs()

    stats = RunStats()

    # 1. train：按日期递增、先映射后更新
    process_train(train_files, stats)

    # 2. valid：Dask 聚合 train 历史
    valid_mappings, valid_prior_ctr, valid_has_history = build_history_mapping_dask(
        train_files,
        split_label="train（供 valid 使用）",
    )
    save_mapping_tables(valid_mappings, suffix="train")
    process_split_with_fixed_mapping(
        split_name="valid",
        parquet_files=valid_files,
        mappings_by_field=valid_mappings,
        prior_ctr=valid_prior_ctr,
        has_history_window=valid_has_history,
        stats=stats,
        track_unseen=True,
    )

    # 3. holdout：Dask 聚合 train + valid 历史
    holdout_source_files = train_files + valid_files
    holdout_mappings, holdout_prior_ctr, holdout_has_history = build_history_mapping_dask(
        holdout_source_files,
        split_label="train + valid（供 holdout 使用）",
    )
    save_mapping_tables(holdout_mappings, suffix="train_valid")
    process_split_with_fixed_mapping(
        split_name="holdout",
        parquet_files=holdout_files,
        mappings_by_field=holdout_mappings,
        prior_ctr=holdout_prior_ctr,
        has_history_window=holdout_has_history,
        stats=stats,
        track_unseen=True,
    )

    write_report(stats)

    print("\n" + "=" * 70)
    print("历史统计特征工程完成")
    print("=" * 70)
    print(f"train 行数：   {stats.train_rows:,}")
    print(f"valid 行数：   {stats.valid_rows:,}")
    print(f"holdout 行数： {stats.holdout_rows:,}")
    print("输出目录：")
    for split_name, output_dir in OUTPUT_DIRS.items():
        file_count = len(list(output_dir.glob("part-*.parquet")))
        print(f"  {split_name}: {output_dir} ({file_count} 个文件)")
    print(f"映射表目录： {MAPPING_DIR}")
    print(f"报告：       {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
