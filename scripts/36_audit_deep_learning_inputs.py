"""
百度 CTR 项目 — 深度学习输入体系审计（第 36 步）

功能：
    只读审计 Step 30 固定调优样本（data/tuning/lightgbm_{train,valid}/），
    检查是否保留适合神经网络 Embedding 的原始类别字段，
    并汇总数值工程特征、标签与行标识符。
    不重新抽样、不修改数据、不训练模型、不读取 holdout、不关联上游数据。

用法：
    python scripts/36_audit_deep_learning_inputs.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TRAIN_FIXED_DIR = Path("data/tuning/lightgbm_train")
VALID_FIXED_DIR = Path("data/tuning/lightgbm_valid")
FIXED_SAMPLE_METADATA_PATH = Path("outputs/fixed_tuning_sample_metadata.json")
OUTPUT_REPORT_PATH = Path("outputs/deep_learning_input_audit.txt")

EXPECTED_TRAIN_ROWS = 2_000_000
EXPECTED_VALID_ROWS = 500_000

BATCH_SIZE = 200_000

# 禁止读取的路径关键字（安全护栏）
FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

# 用户指定的原始类别字段（embedding 候选）
REQUESTED_RAW_CATEGORICALS = [
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "banner_pos",
    "device_type",
    "device_conn_type",
    "C1",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

# 其他常见 embedding 候选（交叉特征等）
OTHER_EMBEDDING_CANDIDATES = [
    "banner_device_cross",
    "hour_banner_cross",
    "site_device_cross",
]

# 判断「是否可直接做 embedding」的关键原始类别字段
KEY_RAW_CATEGORICALS = ("site_id", "app_id", "device_model")

# 标签与行标识符候选
LABEL_COLUMN = "click"
ROW_IDENTIFIER_CANDIDATES = [
    "id",
    "row_id",
    "index",
    "original_index",
    "hour",
    "date",
    "split_date",
]

META_COLUMNS = {"id", "click", "split_date", "hour", "date", "row_id", "index", "original_index"}

NUMERICAL_SUFFIXES = (
    "_freq",
    "_hist_impressions",
    "_hist_clicks",
    "_hist_ctr",
    "_exposure_percentile",
    "_te",
)
NUMERICAL_EXACT = {"hour_sin", "hour_cos", "is_weekend"}


@dataclass
class CategoricalStats:
    """固定样本中单个类别字段的统计。"""

    column: str
    dtype: str
    train_unique: int
    valid_unique: int
    train_missing_count: int
    train_missing_rate: float
    valid_missing_count: int
    valid_missing_rate: float
    valid_oov_category_count: int
    valid_oov_row_count: int
    valid_oov_rate: float


@dataclass
class AuditState:
    """审计过程状态。"""

    lines: list[str] = field(default_factory=list)
    train_rows: int = 0
    valid_rows: int = 0
    categorical_features: list[str] = field(default_factory=list)
    numerical_features: list[str] = field(default_factory=list)
    row_identifiers: list[str] = field(default_factory=list)
    fixed_has_raw_categoricals: bool = False
    can_build_embeddings_directly: bool = False


def log(state: AuditState, message: str = "") -> None:
    """追加报告行并打印。"""

    state.lines.append(message)
    print(message)


def assert_safe_path(path: Path) -> None:
    """确保路径不涉及 holdout / test.csv。"""

    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止读取路径（含 {keyword}）：{path}")


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    """稳定排序 Parquet 文件。"""

    assert_safe_path(parquet_dir)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"目录不存在：{parquet_dir}")

    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{parquet_dir}")
    return files


def inspect_parquet_schema(parquet_path: Path) -> list[tuple[str, str]]:
    """读取 Parquet schema，返回 (列名, dtype) 列表。"""

    assert_safe_path(parquet_path)
    schema = pq.read_schema(parquet_path)
    return [(name, str(schema.field(name).type)) for name in schema.names]


def count_parquet_rows(parquet_files: list[Path]) -> int:
    """统计 Parquet 总行数。"""

    total = 0
    for parquet_path in parquet_files:
        total += pq.ParquetFile(parquet_path).metadata.num_rows
    return total


def is_missing_value(value: Any) -> bool:
    """判断类别值是否视为缺失。"""

    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def is_numerical_feature(column_name: str, metadata_features: set[str]) -> bool:
    """判断列是否为工程化数值特征（不因整数 dtype 误判为 numerical）。"""

    if column_name in metadata_features:
        return True
    if column_name in NUMERICAL_EXACT:
        return True
    return any(column_name.endswith(suffix) for suffix in NUMERICAL_SUFFIXES)


def is_categorical_candidate(column_name: str, metadata_features: set[str]) -> bool:
    """
    判断列是否应视为 categorical embedding 候选。

    规则：在显式类别列表中，或既非 meta 也非已知数值工程特征的其他原始字段。
    """

    if column_name in META_COLUMNS:
        return False
    if column_name in REQUESTED_RAW_CATEGORICALS:
        return True
    if column_name in OTHER_EMBEDDING_CANDIDATES:
        return True
    if is_numerical_feature(column_name, metadata_features):
        return False
    # 未在 metadata 数值特征中的其他列，语义上可能是原始类别
    return True


def discover_categorical_features(
    column_names: list[str],
    metadata_features: set[str],
) -> list[str]:
    """从固定样本列名中发现所有 categorical 候选。"""

    categoricals = [
        col for col in column_names if is_categorical_candidate(col, metadata_features)
    ]
    return sorted(categoricals)


def discover_numerical_features(
    column_names: list[str],
    metadata_features: set[str],
) -> list[str]:
    """从固定样本列名中发现所有数值工程特征。"""

    numericals = [
        col
        for col in column_names
        if is_numerical_feature(col, metadata_features) and col not in META_COLUMNS
    ]
    return sorted(numericals)


def scan_categorical_column(
    parquet_files: list[Path],
    column: str,
) -> tuple[set[str], int, int]:
    """
    扫描单个 split 的类别列。

    返回：(unique_values, total_rows, missing_count)
    """

    unique_values: set[str] = set()
    total_rows = 0
    missing_count = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        if column not in parquet_file.schema.names:
            raise ValueError(f"{parquet_path} 缺少列：{column}")

        for record_batch in parquet_file.iter_batches(
            columns=[column],
            batch_size=BATCH_SIZE,
        ):
            series = record_batch.to_pandas()[column]
            total_rows += len(series)
            for value in series:
                if is_missing_value(value):
                    missing_count += 1
                else:
                    unique_values.add(str(value))

    return unique_values, total_rows, missing_count


def count_oov_rows(
    parquet_files: list[Path],
    column: str,
    train_vocab: set[str],
) -> tuple[int, int]:
    """
    统计 valid 中 OOV 类别数与 OOV 行数。

    返回：(oov_category_count, oov_row_count)
    """

    valid_values: set[str] = set()
    oov_row_count = 0
    total_non_missing = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=[column],
            batch_size=BATCH_SIZE,
        ):
            for value in record_batch.to_pandas()[column]:
                if is_missing_value(value):
                    continue
                total_non_missing += 1
                text_value = str(value)
                valid_values.add(text_value)
                if text_value not in train_vocab:
                    oov_row_count += 1

    oov_categories = valid_values - train_vocab
    return len(oov_categories), oov_row_count


def audit_schema_consistency(
    state: AuditState,
    train_files: list[Path],
    valid_files: list[Path],
) -> tuple[list[tuple[str, str]], list[str]]:
    """审计 train / valid schema 一致性。"""

    train_schema = inspect_parquet_schema(train_files[0])
    train_names = [name for name, _ in train_schema]

    for parquet_path in train_files[1:]:
        names = [name for name, _ in inspect_parquet_schema(parquet_path)]
        if names != train_names:
            raise ValueError(f"train schema 不一致：{train_files[0].name} vs {parquet_path.name}")

    valid_schema = inspect_parquet_schema(valid_files[0])
    valid_names = [name for name, _ in valid_schema]
    if valid_names != train_names:
        raise ValueError("train 与 valid 固定样本 schema 不一致。")

    log(state, "\n" + "=" * 72)
    log(state, "固定样本完整 schema（train / valid 一致）")
    log(state, f"{'列名':<42} {'dtype'}")
    log(state, "-" * 72)
    for name, dtype in train_schema:
        log(state, f"{name:<42} {dtype}")

    return train_schema, train_names


def audit_row_counts(state: AuditState, train_files: list[Path], valid_files: list[Path]) -> None:
    """验证行数并在不符时打印 WARNING。"""

    state.train_rows = count_parquet_rows(train_files)
    state.valid_rows = count_parquet_rows(valid_files)

    log(state, "\n" + "=" * 72)
    log(state, "行数验证")
    log(state, f"train rows = {state.train_rows:,}（期望 {EXPECTED_TRAIN_ROWS:,}）")
    log(state, f"valid rows = {state.valid_rows:,}（期望 {EXPECTED_VALID_ROWS:,}）")

    if state.train_rows != EXPECTED_TRAIN_ROWS:
        log(
            state,
            f"WARNING: train 行数 {state.train_rows:,} != 期望 {EXPECTED_TRAIN_ROWS:,}，"
            "未修改任何数据。",
        )
    if state.valid_rows != EXPECTED_VALID_ROWS:
        log(
            state,
            f"WARNING: valid 行数 {state.valid_rows:,} != 期望 {EXPECTED_VALID_ROWS:,}，"
            "未修改任何数据。",
        )


def audit_requested_categorical_presence(
    state: AuditState,
    column_names: list[str],
) -> None:
    """检查用户指定的原始类别字段是否存在。"""

    log(state, "\n" + "=" * 72)
    log(state, "指定原始类别字段存在性检查（固定样本）")
    log(state, "说明：整数 dtype 的 banner_pos / device_type / C14 等仍视为 categorical。")

    all_requested = REQUESTED_RAW_CATEGORICALS + OTHER_EMBEDDING_CANDIDATES
    for column in all_requested:
        present = column in column_names
        log(state, f"  {column}: {'存在' if present else '不存在'}")

    log(state, "\n关键原始类别字段（site_id / app_id / device_model）：")
    for column in KEY_RAW_CATEGORICALS:
        present = column in column_names
        log(state, f"  {column}: {'存在' if present else '不存在'}")


def audit_categorical_statistics(
    state: AuditState,
    train_files: list[Path],
    valid_files: list[Path],
    schema: list[tuple[str, str]],
    categorical_columns: list[str],
) -> list[CategoricalStats]:
    """对固定样本中存在的类别字段做完整统计。"""

    log(state, "\n" + "=" * 72)
    log(state, "固定样本 categorical 字段统计")

    if not categorical_columns:
        log(state, "固定样本中未发现任何 categorical 字段，跳过 OOV 统计。")
        return []

    dtype_map = dict(schema)
    results: list[CategoricalStats] = []

    log(
        state,
        f"\n{'列名':<22} {'dtype':<14} {'tr_u':>8} {'va_u':>8} "
        f"{'tr_miss%':>9} {'va_miss%':>9} {'OOV_cat':>8} {'OOV_row':>9} {'OOV_rate':>9}",
    )
    log(state, "-" * 105)

    for column in categorical_columns:
        train_vocab, train_rows, train_missing = scan_categorical_column(train_files, column)
        valid_vocab, valid_rows, valid_missing = scan_categorical_column(valid_files, column)
        oov_cat_count, oov_row_count = count_oov_rows(valid_files, column, train_vocab)

        train_missing_rate = train_missing / train_rows if train_rows else 0.0
        valid_missing_rate = valid_missing / valid_rows if valid_rows else 0.0
        valid_non_missing = valid_rows - valid_missing
        oov_rate = oov_row_count / valid_non_missing if valid_non_missing else 0.0

        stats = CategoricalStats(
            column=column,
            dtype=dtype_map[column],
            train_unique=len(train_vocab),
            valid_unique=len(valid_vocab),
            train_missing_count=train_missing,
            train_missing_rate=train_missing_rate,
            valid_missing_count=valid_missing,
            valid_missing_rate=valid_missing_rate,
            valid_oov_category_count=oov_cat_count,
            valid_oov_row_count=oov_row_count,
            valid_oov_rate=oov_rate,
        )
        results.append(stats)

        log(
            state,
            f"{column:<22} {stats.dtype:<14} {stats.train_unique:>8,} {stats.valid_unique:>8,} "
            f"{100 * stats.train_missing_rate:>8.4f}% {100 * stats.valid_missing_rate:>8.4f}% "
            f"{stats.valid_oov_category_count:>8,} {stats.valid_oov_row_count:>9,} "
            f"{100 * stats.valid_oov_rate:>8.4f}%",
        )

        log(state, f"\n  [{column}] 明细：")
        log(state, f"    dtype: {stats.dtype}")
        log(state, f"    train unique: {stats.train_unique:,}")
        log(state, f"    valid unique: {stats.valid_unique:,}")
        log(state, f"    train missing: {stats.train_missing_count:,} "
            f"({100 * stats.train_missing_rate:.4f}%)")
        log(state, f"    valid missing: {stats.valid_missing_count:,} "
            f"({100 * stats.valid_missing_rate:.4f}%)")
        log(state, f"    valid OOV category count: {stats.valid_oov_category_count:,}")
        log(state, f"    valid OOV row count: {stats.valid_oov_row_count:,}")
        log(state, f"    valid OOV rate: {100 * stats.valid_oov_rate:.4f}%")

    return results


def audit_numerical_features(
    state: AuditState,
    numerical_features: list[str],
) -> None:
    """汇总固定样本中的数值工程特征。"""

    log(state, "\n" + "=" * 72)
    log(state, "固定样本 numerical / engineered 特征")

    groups: dict[str, list[str]] = {
        "frequency (*_freq)": [],
        "historical impressions (*_hist_impressions)": [],
        "historical clicks (*_hist_clicks)": [],
        "historical CTR (*_hist_ctr)": [],
        "exposure percentile (*_exposure_percentile)": [],
        "target encoding (*_te)": [],
        "time cyclical (hour_sin / hour_cos)": [],
        "time binary (is_weekend)": [],
        "other numerical": [],
    }

    for feature in numerical_features:
        if feature.endswith("_freq"):
            groups["frequency (*_freq)"].append(feature)
        elif feature.endswith("_hist_impressions"):
            groups["historical impressions (*_hist_impressions)"].append(feature)
        elif feature.endswith("_hist_clicks"):
            groups["historical clicks (*_hist_clicks)"].append(feature)
        elif feature.endswith("_hist_ctr"):
            groups["historical CTR (*_hist_ctr)"].append(feature)
        elif feature.endswith("_exposure_percentile"):
            groups["exposure percentile (*_exposure_percentile)"].append(feature)
        elif feature.endswith("_te"):
            groups["target encoding (*_te)"].append(feature)
        elif feature in ("hour_sin", "hour_cos"):
            groups["time cyclical (hour_sin / hour_cos)"].append(feature)
        elif feature == "is_weekend":
            groups["time binary (is_weekend)"].append(feature)
        else:
            groups["other numerical"].append(feature)

    for group_name, features in groups.items():
        if not features:
            continue
        log(state, f"\n[{group_name}] ({len(features)} 个)")
        for feature in features:
            log(state, f"  - {feature}")

    log(state, f"\nNUMERICAL_FEATURES_FOUND = {numerical_features}")
    log(state, f"NUMERICAL_FEATURE_COUNT = {len(numerical_features)}")


def audit_label_and_identifiers(
    state: AuditState,
    column_names: list[str],
    train_files: list[Path],
) -> None:
    """检查标签列与行标识符。"""

    log(state, "\n" + "=" * 72)
    log(state, "标签及辅助字段检查")

    has_click = LABEL_COLUMN in column_names
    log(state, f"click 列存在: {has_click}")
    if not has_click:
        log(state, "WARNING: 固定样本缺少 click 标签列。")

    present_identifiers = [col for col in ROW_IDENTIFIER_CANDIDATES if col in column_names]
    state.row_identifiers = present_identifiers

    log(state, "\n行标识符 / 辅助字段：")
    for candidate in ROW_IDENTIFIER_CANDIDATES:
        log(state, f"  {candidate}: {'存在' if candidate in column_names else '不存在'}")

    if "id" in column_names:
        unique_ids: set[str] = set()
        total_rows = 0
        missing_ids = 0
        for parquet_path in train_files:
            parquet_file = pq.ParquetFile(parquet_path)
            for record_batch in parquet_file.iter_batches(columns=["id"], batch_size=BATCH_SIZE):
                series = record_batch.to_pandas()["id"]
                total_rows += len(series)
                for id_value in series:
                    if pd.isna(id_value):
                        missing_ids += 1
                    else:
                        unique_ids.add(str(id_value))

        duplicate_rows = total_rows - len(unique_ids) - missing_ids
        log(state, "\n稳定 row identifier: id")
        log(state, f"  train 中 id 总行数: {total_rows:,}")
        log(state, f"  train 中 unique id: {len(unique_ids):,}")
        log(state, f"  train 中 id 缺失: {missing_ids:,}")
        log(state, f"  train 中重复 id 行数: {duplicate_rows:,}")
        if missing_ids == 0 and duplicate_rows == 0:
            log(state, "  id 可作为稳定行标识符（本脚本不关联上游）。")
        else:
            log(state, "  WARNING: id 存在缺失或重复，需进一步确认唯一性。")
    else:
        log(state, "\n未发现稳定 row identifier（无 id 列）。")


def write_followup_note(state: AuditState) -> None:
    """当固定样本无原始类别列时，说明后续补回方案（本脚本不执行）。"""

    if state.fixed_has_raw_categoricals:
        return

    log(state, "\n" + "=" * 72)
    log(state, "后续 Wide & Deep 输入说明（本脚本不执行补回）")
    log(state, "固定样本仅含 33 个工程化数值特征，不含原始 categorical ID。")
    log(state, "Step 30 在抽样时已通过 FORBIDDEN_FEATURES 排除 site_id / app_id 等原始列。")
    log(state, "若要公平比较 Wide & Deep 与 LightGBM，后续需基于 Step 30 相同行集合")
    log(state, "（以 id 为键）补回原始类别字段；不得重新抽样。")


def print_summary(state: AuditState) -> None:
    """终端简洁 Summary。"""

    print("\n" + "=" * 72)
    print("SUMMARY — 第 36 步 深度学习输入审计")
    print("=" * 72)
    print(f"FIXED_TRAIN_ROWS = {state.train_rows:,}")
    print(f"FIXED_VALID_ROWS = {state.valid_rows:,}")
    print(f"RAW_CATEGORICAL_COUNT = {len(state.categorical_features)}")
    print(f"NUMERICAL_FEATURE_COUNT = {len(state.numerical_features)}")
    print(f"CATEGORICAL_FEATURES_FOUND = {state.categorical_features or '[]'}")
    print(f"NUMERICAL_FEATURES_FOUND = {len(state.numerical_features)} features "
          f"(see report for full list)")
    print(f"ROW_IDENTIFIERS = {state.row_identifiers}")
    print(f"FIXED_SAMPLE_HAS_RAW_CATEGORICALS = {state.fixed_has_raw_categoricals}")
    print(f"CAN_BUILD_EMBEDDINGS_DIRECTLY = {state.can_build_embeddings_directly}")
    print("HOLDOUT_USED = False")
    print(f"Report: {OUTPUT_REPORT_PATH}")
    print("=" * 72)


def main() -> None:
    """主流程：只读审计固定样本。"""

    state = AuditState()

    log(state, "=" * 72)
    log(state, "百度 CTR 项目 — 第 36 步 深度学习输入体系审计")
    log(state, f"审计时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    log(state, "模式：只读审计；不抽样、不修改数据、不训练、不读取 holdout、不关联上游")

    metadata = json.loads(FIXED_SAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("holdout_used") is not False:
        raise ValueError("固定样本元数据 holdout_used 必须为 false。")

    metadata_features = set(metadata["feature_columns"])

    train_files = get_sorted_parquet_files(TRAIN_FIXED_DIR)
    valid_files = get_sorted_parquet_files(VALID_FIXED_DIR)

    audit_row_counts(state, train_files, valid_files)
    schema, column_names = audit_schema_consistency(state, train_files, valid_files)

    state.numerical_features = discover_numerical_features(column_names, metadata_features)
    state.categorical_features = discover_categorical_features(column_names, metadata_features)

    audit_requested_categorical_presence(state, column_names)
    audit_categorical_statistics(state, train_files, valid_files, schema, state.categorical_features)
    audit_numerical_features(state, state.numerical_features)
    audit_label_and_identifiers(state, column_names, train_files)

    # 判断规则：关键原始类别字段 site_id / app_id / device_model 均存在
    key_present = all(col in column_names for col in KEY_RAW_CATEGORICALS)
    state.fixed_has_raw_categoricals = len(state.categorical_features) > 0 and key_present
    state.can_build_embeddings_directly = state.fixed_has_raw_categoricals

    write_followup_note(state)

    log(state, "\n" + "=" * 72)
    log(state, "审计结论")
    log(state, f"FIXED_TRAIN_ROWS = {state.train_rows:,}")
    log(state, f"FIXED_VALID_ROWS = {state.valid_rows:,}")
    log(state, f"RAW_CATEGORICAL_COUNT = {len(state.categorical_features)}")
    log(state, f"NUMERICAL_FEATURE_COUNT = {len(state.numerical_features)}")
    log(state, f"CATEGORICAL_FEATURES_FOUND = {state.categorical_features}")
    log(state, f"NUMERICAL_FEATURES_FOUND = {state.numerical_features}")
    log(state, f"FIXED_SAMPLE_HAS_RAW_CATEGORICALS = {state.fixed_has_raw_categoricals}")
    log(state, f"CAN_BUILD_EMBEDDINGS_DIRECTLY = {state.can_build_embeddings_directly}")
    log(state, "HOLDOUT_USED = False")
    log(state, "\n本次仅完成审计，未生成新训练数据，未训练神经网络，未补回原始类别字段。")

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_PATH.write_text("\n".join(state.lines) + "\n", encoding="utf-8")
    log(state, f"\n报告已保存：{OUTPUT_REPORT_PATH}")

    print_summary(state)


if __name__ == "__main__":
    main()
