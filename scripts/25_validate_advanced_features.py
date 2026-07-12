"""
百度 CTR 项目 — 高级特征统一验收脚本

功能：
    对时间划分、历史统计特征、Target Encoding 三阶段产物进行统一验收，
    检查行数一致性、字段完整性、缺失值、取值范围、时间划分与数据泄漏风险。

验收流程：
    22_time_split.py
    → 23_build_historical_features.py
    → 24_build_target_encoding.py
    → 25_validate_advanced_features.py

数据输入：
    data/model_input/{train,valid,holdout}/*.parquet
    data/features/historical/{train,valid,holdout}/*.parquet
    data/features/target_encoded/{train,valid,holdout}/*.parquet

数据输出：
    outputs/advanced_feature_validation_summary.csv
    outputs/advanced_feature_column_stats.csv
    outputs/advanced_feature_validation_report.txt

用法：
    python scripts/25_validate_advanced_features.py
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 路径与特征定义
# ---------------------------------------------------------------------------

SPLITS = ("train", "valid", "holdout")

STAGE_DIRS: dict[str, Path] = {
    "model_input": Path("data/model_input"),
    "historical": Path("data/features/historical"),
    "target_encoded": Path("data/features/target_encoded"),
}

OUTPUT_SUMMARY_CSV = Path("outputs/advanced_feature_validation_summary.csv")
OUTPUT_COLUMN_STATS_CSV = Path("outputs/advanced_feature_column_stats.csv")
OUTPUT_REPORT_PATH = Path("outputs/advanced_feature_validation_report.txt")

CATEGORY_FIELDS = [
    "site_id",
    "site_category",
    "app_id",
    "app_category",
    "device_model",
]

HIST_FEATURE_SUFFIXES = [
    "hist_impressions",
    "hist_clicks",
    "hist_ctr",
    "exposure_percentile",
]

TE_FEATURES = [
    "site_id_te",
    "site_category_te",
    "app_id_te",
    "app_category_te",
    "device_model_te",
]

REQUIRED_BASE_COLUMNS = ["click", "id", "hour_dt"]

# ID 抽样检查：每个 split 抽取的文件数（首/中/尾）
ID_CHECK_FILE_INDICES = (0, -1)

# 历史统计冷启动与映射一致性检查最多处理的文件数（0 表示全部）
MAX_FILES_FOR_LEAKAGE_CHECK = 0


def get_hist_feature_names() -> list[str]:
    """返回 20 个历史统计特征名。"""

    names: list[str] = []
    for category_field in CATEGORY_FIELDS:
        for suffix in HIST_FEATURE_SUFFIXES:
            names.append(f"{category_field}_{suffix}")
    return names


def get_advanced_feature_names() -> list[str]:
    """返回 25 个高级特征（历史 + TE）。"""

    return get_hist_feature_names() + TE_FEATURES


@dataclass
class ValidationState:
    """收集验收过程中的 PASS / WARNING / ERROR。"""

    passes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary_rows: list[dict] = field(default_factory=list)

    def add_pass(self, message: str) -> None:
        self.passes.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def has_error(self) -> bool:
        return len(self.errors) > 0


@dataclass
class ColumnStatsAccumulator:
    """按 split + feature 累计列级统计。"""

    split: str
    feature: str
    missing_count: int = 0
    inf_count: int = 0
    invalid_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0
    count_value: int = 0

    def update(self, series: pd.Series, invalid_mask: pd.Series | None = None) -> None:
        """累计单列统计。"""

        numeric = pd.to_numeric(series, errors="coerce")

        self.missing_count += int(numeric.isna().sum())
        self.inf_count += int(np.isinf(numeric).sum())

        finite = numeric[np.isfinite(numeric)]
        if len(finite) > 0:
            current_min = float(finite.min())
            current_max = float(finite.max())
            self.min_value = (
                current_min if self.min_value is None else min(self.min_value, current_min)
            )
            self.max_value = (
                current_max if self.max_value is None else max(self.max_value, current_max)
            )
            self.sum_value += float(finite.sum())
            self.count_value += len(finite)

        if invalid_mask is not None:
            self.invalid_count += int(invalid_mask.sum())

    def to_row(self) -> dict:
        """转为 column_stats.csv 行。"""

        mean_value = self.sum_value / self.count_value if self.count_value > 0 else np.nan
        status = "ERROR" if self.invalid_count > 0 or self.inf_count > 0 else "PASS"

        if self.missing_count > 0 and status == "PASS":
            status = "WARNING"

        return {
            "split": self.split,
            "feature": self.feature,
            "missing_count": self.missing_count,
            "inf_count": self.inf_count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean_value,
            "invalid_count": self.invalid_count,
            "status": status,
        }


def list_parquet_files(parquet_dir: Path) -> list[Path]:
    """按文件名排序列出 Parquet 分块。"""

    return sorted(parquet_dir.glob("part-*.parquet"))


def count_rows_from_metadata(parquet_files: list[Path]) -> int:
    """使用 Parquet metadata 统计总行数。"""

    return sum(pq.read_metadata(path).num_rows for path in parquet_files)


def read_parquet_columns(parquet_path: Path) -> list[str]:
    """读取 Parquet schema 列名，不加载数据。"""

    return pq.read_schema(parquet_path).names


def resolve_date_source_column(columns: list[str]) -> tuple[str | None, str | None]:
    """
    确定日期来源列。

    返回：(主日期列, 回退日期列)
    优先 date，其次 hour；若均不存在则回退 hour_dt。
    """

    if "date" in columns:
        return "date", None

    if "hour" in columns:
        return "hour", "hour_dt" if "hour_dt" in columns else None

    if "hour_dt" in columns:
        return "hour_dt", None

    return None, None


def extract_event_dates(dataframe: pd.DataFrame, date_column: str) -> pd.Series:
    """从指定列提取归一化日期。"""

    if date_column == "date":
        return pd.to_datetime(dataframe["date"], errors="coerce").dt.normalize()

    if date_column == "hour_dt":
        return pd.to_datetime(dataframe["hour_dt"], errors="coerce").dt.normalize()

    if date_column == "hour":
        hour_text = dataframe["hour"].astype(str).str.replace(r"\.0$", "", regex=True)
        hour_text = hour_text.str.zfill(8)
        date_text = "20" + hour_text.str.slice(0, 2) + "-" + hour_text.str.slice(2, 4) + "-" + hour_text.str.slice(4, 6)
        return pd.to_datetime(date_text, errors="coerce").dt.normalize()

    raise ValueError(f"不支持的日期列：{date_column}")


def build_invalid_mask(dataframe: pd.DataFrame, feature_name: str) -> pd.Series:
    """构建单列取值范围违规掩码。"""

    series = pd.to_numeric(dataframe[feature_name], errors="coerce")
    invalid = pd.Series(False, index=dataframe.index)

    if feature_name.endswith("_hist_impressions"):
        invalid |= series < 0

    elif feature_name.endswith("_hist_clicks"):
        impressions_col = feature_name.replace("_hist_clicks", "_hist_impressions")
        if impressions_col in dataframe.columns:
            impressions = pd.to_numeric(dataframe[impressions_col], errors="coerce")
            invalid |= (series < 0) | (series > impressions)
        else:
            invalid |= series < 0

    elif feature_name.endswith("_hist_ctr") or feature_name.endswith("_te"):
        invalid |= (series < 0) | (series > 1)

    elif feature_name.endswith("_exposure_percentile"):
        invalid |= (series < 0) | (series > 1)

    return invalid.fillna(False)


def check_stage_directories(state: ValidationState) -> dict[str, dict[str, list[Path]]]:
    """
    检查三个阶段的 train/valid/holdout 目录与文件。

    返回：{stage: {split: [files]}}
    """

    stage_files: dict[str, dict[str, list[Path]]] = {}
    summary_rows: list[dict] = []

    for stage_name, stage_base in STAGE_DIRS.items():
        stage_files[stage_name] = {}

        for split_name in SPLITS:
            split_dir = stage_base / split_name
            status = "PASS"

            if not split_dir.exists():
                state.add_error(f"[文件检查] 目录不存在：{split_dir}")
                status = "ERROR"
                files: list[Path] = []
            else:
                files = list_parquet_files(split_dir)
                if not files:
                    state.add_error(f"[文件检查] 目录无 Parquet 文件：{split_dir}")
                    status = "ERROR"
                else:
                    state.add_pass(
                        f"[文件检查] {stage_name}/{split_name} 存在 {len(files)} 个 Parquet 文件"
                    )

            stage_files[stage_name][split_name] = files

            column_count = np.nan
            if files:
                column_count = len(read_parquet_columns(files[0]))

            summary_rows.append(
                {
                    "split": split_name,
                    "stage": stage_name,
                    "file_count": len(files),
                    "row_count": count_rows_from_metadata(files) if files else 0,
                    "column_count": column_count,
                    "status": status,
                }
            )

    state.summary_rows = summary_rows
    return stage_files


def check_row_count_consistency(
    state: ValidationState,
    stage_files: dict[str, dict[str, list[Path]]],
) -> None:
    """检查每个 split 三阶段行数是否一致。"""

    for split_name in SPLITS:
        counts = {
            stage_name: count_rows_from_metadata(stage_files[stage_name][split_name])
            for stage_name in STAGE_DIRS
        }

        model_rows = counts["model_input"]
        hist_rows = counts["historical"]
        te_rows = counts["target_encoded"]

        if model_rows == hist_rows == te_rows and model_rows > 0:
            state.add_pass(
                f"[行数一致性] {split_name}: model_input={model_rows:,}, "
                f"historical={hist_rows:,}, target_encoded={te_rows:,}"
            )
        else:
            state.add_error(
                f"[行数一致性] {split_name} 行数不一致："
                f"model_input={model_rows:,}, historical={hist_rows:,}, "
                f"target_encoded={te_rows:,}"
            )

        if model_rows != hist_rows or model_rows != te_rows:
            for row in state.summary_rows:
                if row["split"] == split_name:
                    row["status"] = "ERROR"


def check_target_encoded_columns(
    state: ValidationState,
    stage_files: dict[str, dict[str, list[Path]]],
    base_columns: list[str],
) -> None:
    """检查 target_encoded 是否保留原始字段并包含全部高级特征。"""

    required_features = get_advanced_feature_names()

    for split_name in SPLITS:
        files = stage_files["target_encoded"][split_name]
        if not files:
            continue

        columns = read_parquet_columns(files[0])

        missing_base = [col for col in base_columns if col not in columns]
        if missing_base:
            state.add_error(
                f"[字段检查] target_encoded/{split_name} 缺少原始字段：{missing_base}"
            )
        else:
            state.add_pass(
                f"[字段检查] target_encoded/{split_name} 保留全部原始字段（{len(base_columns)} 列基准）"
            )

        if "click" not in columns:
            state.add_error(f"[字段检查] target_encoded/{split_name} 缺少 click 字段")
        else:
            state.add_pass(f"[字段检查] target_encoded/{split_name} 包含 click 字段")

        missing_hist = [name for name in get_hist_feature_names() if name not in columns]
        if missing_hist:
            state.add_error(
                f"[字段检查] target_encoded/{split_name} 缺少历史统计特征：{missing_hist}"
            )
        else:
            state.add_pass(
                f"[字段检查] target_encoded/{split_name} 包含全部 20 个历史统计特征"
            )

        missing_te = [name for name in TE_FEATURES if name not in columns]
        if missing_te:
            state.add_error(
                f"[字段检查] target_encoded/{split_name} 缺少 TE 特征：{missing_te}"
            )
        else:
            state.add_pass(
                f"[字段检查] target_encoded/{split_name} 包含全部 5 个 TE 特征"
            )

        unexpected_missing = [
            feature_name
            for feature_name in required_features
            if feature_name not in columns
        ]
        if unexpected_missing:
            state.add_error(
                f"[字段检查] target_encoded/{split_name} 高级特征不完整："
                f"{unexpected_missing[:5]}{'...' if len(unexpected_missing) > 5 else ''}"
            )


def process_target_encoded_file_stats(
    parquet_path: Path,
    split_name: str,
    accumulators: dict[tuple[str, str], ColumnStatsAccumulator],
) -> None:
    """逐文件检查 target_encoded 高级特征的缺失值、无穷值与取值范围。"""

    feature_names = get_advanced_feature_names()
    category_columns = CATEGORY_FIELDS

    read_columns = list(dict.fromkeys(feature_names + category_columns))
    dataframe = pd.read_parquet(parquet_path, columns=read_columns)

    try:
        for feature_name in feature_names:
            key = (split_name, feature_name)
            if key not in accumulators:
                accumulators[key] = ColumnStatsAccumulator(
                    split=split_name,
                    feature=feature_name,
                )

            invalid_mask = build_invalid_mask(dataframe, feature_name)
            accumulators[key].update(dataframe[feature_name], invalid_mask=invalid_mask)
    finally:
        del dataframe
        gc.collect()


def check_feature_values(
    state: ValidationState,
    stage_files: dict[str, dict[str, list[Path]]],
) -> dict[tuple[str, str], ColumnStatsAccumulator]:
    """逐 Parquet 文件检查高级特征质量。"""

    accumulators: dict[tuple[str, str], ColumnStatsAccumulator] = {}

    for split_name in SPLITS:
        files = stage_files["target_encoded"][split_name]
        if not files:
            continue

        print(f"  检查 target_encoded/{split_name} 特征值（{len(files)} 个文件）...")

        for file_index, parquet_path in enumerate(files, start=1):
            process_target_encoded_file_stats(parquet_path, split_name, accumulators)

            if file_index % 20 == 0 or file_index == len(files):
                print(f"    已完成 {file_index}/{len(files)} 个文件")

    for (split_name, feature_name), accumulator in accumulators.items():
        if accumulator.inf_count > 0:
            state.add_error(
                f"[取值检查] {split_name}/{feature_name} 存在 inf："
                f"{accumulator.inf_count:,} 个"
            )
        if accumulator.invalid_count > 0:
            state.add_error(
                f"[取值检查] {split_name}/{feature_name} 存在违规取值："
                f"{accumulator.invalid_count:,} 行"
            )
        if accumulator.missing_count > 0:
            state.add_warning(
                f"[取值检查] {split_name}/{feature_name} 存在缺失值："
                f"{accumulator.missing_count:,} 个"
            )

    if not state.has_error:
        state.add_pass("[取值检查] 全部高级特征未发现 inf 或违规取值")

    return accumulators


def collect_split_date_range(
    parquet_files: list[Path],
    date_column: str,
    fallback_column: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """逐文件收集 split 的最小/最大日期。"""

    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    used_column = date_column

    for parquet_path in parquet_files:
        columns_to_read = [date_column]
        if fallback_column:
            columns_to_read.append(fallback_column)

        chunk = pd.read_parquet(parquet_path, columns=columns_to_read)
        try:
            dates = extract_event_dates(chunk, date_column)
            if dates.isna().all() and fallback_column:
                used_column = fallback_column
                dates = extract_event_dates(chunk, fallback_column)

            valid_dates = dates.dropna()
            if valid_dates.empty:
                continue

            file_min = pd.Timestamp(valid_dates.min())
            file_max = pd.Timestamp(valid_dates.max())

            min_date = file_min if min_date is None else min(min_date, file_min)
            max_date = file_max if max_date is None else max(max_date, file_max)
        finally:
            del chunk
            gc.collect()

    return min_date, max_date, used_column


def check_time_split(
    state: ValidationState,
    stage_files: dict[str, dict[str, list[Path]]],
) -> None:
    """检查 train / valid / holdout 日期范围是否无重叠且顺序正确。"""

    reference_files = stage_files["model_input"]
    split_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for split_name in SPLITS:
        files = reference_files[split_name]
        if not files:
            state.add_error(f"[时间划分] {split_name} 无文件，无法检查日期")
            continue

        columns = read_parquet_columns(files[0])
        date_column, fallback_column = resolve_date_source_column(columns)

        if date_column is None:
            state.add_error(f"[时间划分] {split_name} 缺少 date / hour / hour_dt 字段")
            continue

        if date_column == "hour_dt":
            state.add_warning(
                f"[时间划分] {split_name} 未找到 date/hour，回退使用 hour_dt 解析日期"
            )

        min_date, max_date, used_column = collect_split_date_range(
            files,
            date_column,
            fallback_column,
        )

        if min_date is None or max_date is None:
            state.add_error(f"[时间划分] {split_name} 无法解析有效日期")
            continue

        split_ranges[split_name] = (min_date, max_date)
        state.add_pass(
            f"[时间划分] {split_name}: {min_date.date()} ~ {max_date.date()} "
            f"(来源列: {used_column})"
        )

    if len(split_ranges) == 3:
        train_max = split_ranges["train"][1]
        valid_min = split_ranges["valid"][0]
        valid_max = split_ranges["valid"][1]
        holdout_min = split_ranges["holdout"][0]

        if train_max < valid_min:
            state.add_pass(
                f"[时间划分] train 最大日期 {train_max.date()} 早于 valid 最小日期 {valid_min.date()}"
            )
        else:
            state.add_error(
                f"[时间划分] train 最大日期 {train_max.date()} 未早于 valid 最小日期 {valid_min.date()}"
            )

        if valid_max < holdout_min:
            state.add_pass(
                f"[时间划分] valid 最大日期 {valid_max.date()} 早于 holdout 最小日期 {holdout_min.date()}"
            )
        else:
            state.add_error(
                f"[时间划分] valid 最大日期 {valid_max.date()} 未早于 holdout 最小日期 {holdout_min.date()}"
            )

        # 检查 split 之间无日期重叠
        train_range = pd.date_range(split_ranges["train"][0], split_ranges["train"][1], freq="D")
        valid_range = pd.date_range(split_ranges["valid"][0], split_ranges["valid"][1], freq="D")
        holdout_range = pd.date_range(split_ranges["holdout"][0], split_ranges["holdout"][1], freq="D")

        overlap_tv = set(train_range.date) & set(valid_range.date)
        overlap_vh = set(valid_range.date) & set(holdout_range.date)
        overlap_th = set(train_range.date) & set(holdout_range.date)

        if not overlap_tv and not overlap_vh and not overlap_th:
            state.add_pass("[时间划分] train / valid / holdout 之间无日期重叠")
        else:
            state.add_error(
                f"[时间划分] 存在日期重叠：train-valid={overlap_tv}, "
                f"valid-holdout={overlap_vh}, train-holdout={overlap_th}"
            )


def find_train_earliest_date(
    historical_train_files: list[Path],
) -> pd.Timestamp | None:
    """找出 train 历史特征文件的最早日期。"""

    earliest: pd.Timestamp | None = None

    for parquet_path in historical_train_files:
        columns = read_parquet_columns(parquet_path)
        date_column, fallback_column = resolve_date_source_column(columns)
        if date_column is None:
            continue

        read_columns = [date_column]
        if fallback_column:
            read_columns.append(fallback_column)

        chunk = pd.read_parquet(parquet_path, columns=read_columns)
        try:
            dates = extract_event_dates(chunk, date_column)
            if dates.isna().all() and fallback_column:
                dates = extract_event_dates(chunk, fallback_column)

            valid_dates = dates.dropna()
            if valid_dates.empty:
                continue

            file_min = pd.Timestamp(valid_dates.min())
            earliest = file_min if earliest is None else min(earliest, file_min)
        finally:
            del chunk
            gc.collect()

    return earliest


def check_train_cold_start(
    state: ValidationState,
    historical_train_files: list[Path],
) -> None:
    """
    检查 train 最早日期的冷启动特征：
    hist_impressions / hist_clicks / exposure_percentile 应全部为 0。
    """

    if not historical_train_files:
        state.add_error("[泄漏检查] train 历史特征文件为空，无法检查冷启动")
        return

    earliest_date = find_train_earliest_date(historical_train_files)
    if earliest_date is None:
        state.add_error("[泄漏检查] 无法确定 train 最早日期")
        return

    cold_start_features = [
        name
        for name in get_hist_feature_names()
        if name.endswith("_hist_impressions")
        or name.endswith("_hist_clicks")
        or name.endswith("_exposure_percentile")
    ]

    checked_rows = 0
    violation_rows = 0

    files_to_check = historical_train_files
    if MAX_FILES_FOR_LEAKAGE_CHECK > 0:
        files_to_check = historical_train_files[:MAX_FILES_FOR_LEAKAGE_CHECK]

    for parquet_path in files_to_check:
        columns = read_parquet_columns(parquet_path)
        date_column, fallback_column = resolve_date_source_column(columns)
        if date_column is None:
            continue

        read_columns = list(dict.fromkeys([date_column, *cold_start_features]))
        if fallback_column:
            read_columns.append(fallback_column)

        chunk = pd.read_parquet(parquet_path, columns=read_columns)
        try:
            dates = extract_event_dates(chunk, date_column)
            if dates.isna().all() and fallback_column:
                dates = extract_event_dates(chunk, fallback_column)

            earliest_mask = dates == earliest_date
            if not earliest_mask.any():
                continue

            earliest_chunk = chunk.loc[earliest_mask]
            checked_rows += len(earliest_chunk)

            for feature_name in cold_start_features:
                values = pd.to_numeric(earliest_chunk[feature_name], errors="coerce").fillna(-1)
                violation_rows += int((values != 0).sum())
        finally:
            del chunk
            gc.collect()

    if checked_rows == 0:
        state.add_error(
            f"[泄漏检查] train 最早日期 {earliest_date.date()} 未找到对应数据行"
        )
        return

    if violation_rows == 0:
        state.add_pass(
            f"[泄漏检查] train 最早日期 {earliest_date.date()} 冷启动正确："
            f"检查 {checked_rows:,} 行，hist_impressions/hist_clicks/exposure_percentile 均为 0"
        )
    else:
        state.add_error(
            f"[泄漏检查] train 最早日期 {earliest_date.date()} 冷启动违规："
            f"{violation_rows:,} 个非零值（共检查 {checked_rows:,} 行）"
        )


def check_mapping_consistency(
    state: ValidationState,
    split_name: str,
    target_encoded_files: list[Path],
) -> None:
    """
    检查 valid / holdout 中同一类别值的历史特征与 TE 是否一致。

    对每个文件：groupby(category)[feature].nunique() 应全部为 1。
    """

    if not target_encoded_files:
        state.add_error(f"[泄漏检查] {split_name} 无 target_encoded 文件，无法检查映射一致性")
        return

    features_to_check = get_advanced_feature_names()
    files_to_check = target_encoded_files
    if MAX_FILES_FOR_LEAKAGE_CHECK > 0:
        files_to_check = target_encoded_files[:MAX_FILES_FOR_LEAKAGE_CHECK]

    inconsistent_count = 0

    for parquet_path in files_to_check:
        read_columns = list(dict.fromkeys(CATEGORY_FIELDS + features_to_check))
        chunk = pd.read_parquet(parquet_path, columns=read_columns)

        try:
            for category_field in CATEGORY_FIELDS:
                related_features = [
                    feature_name
                    for feature_name in features_to_check
                    if feature_name.startswith(f"{category_field}_")
                ]

                grouped = chunk.groupby(category_field, dropna=False)
                for feature_name in related_features:
                    nunique_series = grouped[feature_name].nunique(dropna=False)
                    if (nunique_series > 1).any():
                        inconsistent_count += int((nunique_series > 1).sum())
        finally:
            del chunk
            gc.collect()

    if inconsistent_count == 0:
        state.add_pass(
            f"[泄漏检查] {split_name} 类别映射一致："
            f"同一类别值的历史特征与 TE 在 {len(files_to_check)} 个文件中保持一致"
        )
    else:
        state.add_error(
            f"[泄漏检查] {split_name} 存在 {inconsistent_count} 个类别值的映射不一致"
        )


def select_files_for_id_check(parquet_files: list[Path]) -> list[Path]:
    """选取首/尾等少量文件用于 ID 抽样检查。"""

    if not parquet_files:
        return []

    indices = sorted({idx if idx >= 0 else len(parquet_files) + idx for idx in ID_CHECK_FILE_INDICES})
    selected: list[Path] = []

    for index in indices:
        if 0 <= index < len(parquet_files):
            path = parquet_files[index]
            if path not in selected:
                selected.append(path)

    return selected


def check_id_consistency(
    state: ValidationState,
    stage_files: dict[str, dict[str, list[Path]]],
) -> None:
    """抽样检查三阶段同文件名分块的 id 列是否完全一致。"""

    for split_name in SPLITS:
        model_files = stage_files["model_input"][split_name]
        hist_files = stage_files["historical"][split_name]
        te_files = stage_files["target_encoded"][split_name]

        if not model_files or not hist_files or not te_files:
            state.add_warning(f"[ID 检查] {split_name} 某阶段缺少文件，跳过 ID 抽样")
            continue

        sample_files = select_files_for_id_check(model_files)

        for sample_path in sample_files:
            file_name = sample_path.name
            hist_path = STAGE_DIRS["historical"] / split_name / file_name
            te_path = STAGE_DIRS["target_encoded"] / split_name / file_name

            if not hist_path.exists() or not te_path.exists():
                state.add_error(
                    f"[ID 检查] {split_name}/{file_name} 在 historical 或 target_encoded 中不存在"
                )
                continue

            ids_model = pd.read_parquet(sample_path, columns=["id"])["id"]
            ids_hist = pd.read_parquet(hist_path, columns=["id"])["id"]
            ids_te = pd.read_parquet(te_path, columns=["id"])["id"]

            if len(ids_model) != len(ids_hist) or len(ids_model) != len(ids_te):
                state.add_error(
                    f"[ID 检查] {split_name}/{file_name} 行数不一致："
                    f"model_input={len(ids_model):,}, historical={len(ids_hist):,}, "
                    f"target_encoded={len(ids_te):,}"
                )
            elif not ids_model.equals(ids_hist) or not ids_model.equals(ids_te):
                mismatch_model_hist = int((ids_model != ids_hist).sum())
                mismatch_model_te = int((ids_model != ids_te).sum())
                state.add_error(
                    f"[ID 检查] {split_name}/{file_name} id 不一致："
                    f"model-hist 差异 {mismatch_model_hist:,} 行，"
                    f"model-te 差异 {mismatch_model_te:,} 行"
                )
            else:
                state.add_pass(
                    f"[ID 检查] {split_name}/{file_name} id 完全一致（{len(ids_model):,} 行）"
                )

            del ids_model, ids_hist, ids_te
            gc.collect()


def build_summary_dataframe(state: ValidationState) -> pd.DataFrame:
    """构建 summary.csv。"""

    if not state.summary_rows:
        return pd.DataFrame(
            columns=["split", "stage", "file_count", "row_count", "column_count", "status"]
        )

    return pd.DataFrame(state.summary_rows)


def build_column_stats_dataframe(
    accumulators: dict[tuple[str, str], ColumnStatsAccumulator],
) -> pd.DataFrame:
    """构建 column_stats.csv。"""

    rows = [accumulator.to_row() for accumulator in accumulators.values()]
    if not rows:
        return pd.DataFrame(
            columns=[
                "split",
                "feature",
                "missing_count",
                "inf_count",
                "min",
                "max",
                "mean",
                "invalid_count",
                "status",
            ]
        )

    return pd.DataFrame(rows).sort_values(["split", "feature"]).reset_index(drop=True)


def write_text_report(
    state: ValidationState,
    summary_df: pd.DataFrame,
    column_stats_df: pd.DataFrame,
) -> None:
    """生成文本验收报告。"""

    lines: list[str] = []
    lines.append("百度 CTR 项目 — 高级特征统一验收报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append("【通过项目】")
    if state.passes:
        for message in state.passes:
            lines.append(f"  ✓ {message}")
    else:
        lines.append("  （无）")
    lines.append("")
    lines.append("【警告项目】")
    if state.warnings:
        for message in state.warnings:
            lines.append(f"  ! {message}")
    else:
        lines.append("  （无）")
    lines.append("")
    lines.append("【错误项目】")
    if state.errors:
        for message in state.errors:
            lines.append(f"  ✗ {message}")
    else:
        lines.append("  （无）")
    lines.append("")
    lines.append("【阶段汇总】")
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            lines.append(
                f"  {row['stage']}/{row['split']}: files={int(row['file_count'])}, "
                f"rows={int(row['row_count']):,}, cols={row['column_count']}, "
                f"status={row['status']}"
            )
    lines.append("")
    lines.append("【高级特征列统计摘要】")
    if column_stats_df.empty:
        lines.append("  （无数据）")
    else:
        error_features = column_stats_df[column_stats_df["status"] == "ERROR"]
        warning_features = column_stats_df[column_stats_df["status"] == "WARNING"]
        lines.append(f"  检查特征数：{len(column_stats_df)}")
        lines.append(f"  ERROR 特征数：{len(error_features)}")
        lines.append(f"  WARNING 特征数：{len(warning_features)}")
    lines.append("")
    lines.append("【最终结论】")
    if state.has_error:
        lines.append("  高级特征验收失败，暂时不能进入模型训练。")
    elif state.warnings:
        lines.append("  高级特征验收通过（存在非关键 WARNING），可以进入模型训练。")
    else:
        lines.append("  高级特征验收通过，可以进入模型训练。")

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(state: ValidationState) -> None:
    """在终端打印简洁汇总。"""

    print("\n" + "=" * 70)
    print("高级特征验收汇总")
    print("=" * 70)
    print(f"通过：{len(state.passes)} 项")
    print(f"警告：{len(state.warnings)} 项")
    print(f"错误：{len(state.errors)} 项")

    if state.has_error:
        print("\n高级特征验收失败，暂时不能进入模型训练。")
        print("\n主要错误：")
        for message in state.errors[:10]:
            print(f"  - {message}")
        if len(state.errors) > 10:
            print(f"  ... 另有 {len(state.errors) - 10} 项错误，详见报告")
    elif state.warnings:
        print("\n高级特征验收通过（存在非关键 WARNING），可以进入模型训练。")
    else:
        print("\n高级特征验收通过，可以进入模型训练。")

    print("=" * 70)


def main() -> None:
    """主流程：分阶段执行全部验收检查并输出报告。"""

    print("=" * 70)
    print("高级特征统一验收")
    print("=" * 70)

    state = ValidationState()

    # 1. 文件检查
    print("\n[1/8] 检查目录与 Parquet 文件...")
    stage_files = check_stage_directories(state)

    # 2. 行数一致性
    print("\n[2/8] 检查三阶段行数一致性...")
    check_row_count_consistency(state, stage_files)

    # 3. 字段检查
    print("\n[3/8] 检查 target_encoded 字段完整性...")
    base_columns: list[str] = []
    if stage_files["model_input"]["train"]:
        base_columns = read_parquet_columns(stage_files["model_input"]["train"][0])
    check_target_encoded_columns(state, stage_files, base_columns)

    # 4. 缺失值 / 无穷值 / 取值范围
    print("\n[4/8] 检查高级特征缺失值、无穷值与取值范围...")
    accumulators = check_feature_values(state, stage_files)

    # 5. 时间划分
    print("\n[5/8] 检查时间划分...")
    check_time_split(state, stage_files)

    # 6. 数据泄漏基础检查
    print("\n[6/8] 检查冷启动与映射一致性...")
    check_train_cold_start(state, stage_files["historical"]["train"])
    check_mapping_consistency(state, "valid", stage_files["target_encoded"]["valid"])
    check_mapping_consistency(state, "holdout", stage_files["target_encoded"]["holdout"])

    # 7. ID 抽样检查
    print("\n[7/8] 检查 ID 抽样一致性...")
    check_id_consistency(state, stage_files)

    # 8. 输出报告
    print("\n[8/8] 生成验收报告...")
    summary_df = build_summary_dataframe(state)
    column_stats_df = build_column_stats_dataframe(accumulators)

    OUTPUT_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    column_stats_df.to_csv(OUTPUT_COLUMN_STATS_CSV, index=False)
    write_text_report(state, summary_df, column_stats_df)

    print(f"\n已保存：{OUTPUT_SUMMARY_CSV}")
    print(f"已保存：{OUTPUT_COLUMN_STATS_CSV}")
    print(f"已保存：{OUTPUT_REPORT_PATH}")

    print_terminal_summary(state)


if __name__ == "__main__":
    main()
