"""
百度 CTR 项目 — PyTorch 深度学习输入准备（第 39 步）

功能：
    基于 Step 37 统一样本，为 Wide & Deep / DeepFM 构建编码后的 PyTorch 输入：
    - categorical：train-only vocabulary，0=OOV, 1=MISSING, 2+=known
    - numerical：train-only median fill + StandardScaler
    保持与 unified 样本完全相同的行顺序，不训练模型，禁止读取 holdout。

数据输入：
    data/modeling/unified_train/
    data/modeling/unified_valid/
    outputs/unified_modeling_sample_metadata.json

用法：
    python scripts/39_prepare_pytorch_inputs.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

UNIFIED_TRAIN_DIR = Path("data/modeling/unified_train")
UNIFIED_VALID_DIR = Path("data/modeling/unified_valid")

PYTORCH_TRAIN_DIR = Path("data/modeling/pytorch_train")
PYTORCH_VALID_DIR = Path("data/modeling/pytorch_valid")
PYTORCH_TRAIN_TMP = Path("data/modeling/_pytorch_train_tmp")
PYTORCH_VALID_TMP = Path("data/modeling/_pytorch_valid_tmp")
VOCAB_DIR = Path("data/modeling/pytorch_artifacts/vocabs")

UNIFIED_METADATA_PATH = Path("outputs/unified_modeling_sample_metadata.json")
PYTORCH_METADATA_PATH = Path("outputs/pytorch_input_metadata.json")

SCALER_PATH = Path("models/pytorch_numerical_scaler.joblib")
FILL_VALUES_PATH = Path("data/modeling/pytorch_artifacts/numerical_fill_values.json")

FORMAL_TRAIN_ROWS = 2_000_000
FORMAL_VALID_ROWS = 500_000
TEST_TRAIN_ROWS = 100_000
TEST_VALID_ROWS = 50_000

RANDOM_SEED = 42
BATCH_SIZE = 200_000
EXPECTED_NUMERICAL_COUNT = 33

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

UNKNOWN_INDEX = 0
MISSING_INDEX = 1
KNOWN_INDEX_START = 2


@dataclass
class CategoricalVocab:
    """单个 categorical 特征的 vocabulary。"""

    feature_name: str
    category_to_index: dict[str, int]
    index_to_category: list[str]
    vocab_size: int
    train_unique_count: int


@dataclass
class CategoricalAudit:
    """编码后 categorical 审计。"""

    feature: str
    vocab_size: int
    train_min: int
    train_max: int
    valid_oov_count: int
    valid_oov_rate: float
    valid_missing_rate: float


@dataclass
class BuildState:
    lines: list[str] = field(default_factory=list)


def log(state: BuildState, message: str = "") -> None:
    state.lines.append(message)
    print(message)


def assert_safe_path(path: Path) -> None:
    normalized = str(path).lower()
    for keyword in FORBIDDEN_PATH_KEYWORDS:
        if keyword in normalized:
            raise ValueError(f"禁止访问路径（含 {keyword}）：{path}")


def get_row_limits(test_mode: bool) -> tuple[int, int]:
    if test_mode:
        return TEST_TRAIN_ROWS, TEST_VALID_ROWS
    return FORMAL_TRAIN_ROWS, FORMAL_VALID_ROWS


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    assert_safe_path(parquet_dir)
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有 Parquet 文件：{parquet_dir}")
    return files


def load_unified_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("validation_passed") is not True:
        raise ValueError("unified metadata validation_passed 必须为 true。")
    if metadata.get("holdout_used") is not False:
        raise ValueError("unified metadata holdout_used 必须为 false。")

    numerical = metadata["numerical_features"]
    categorical = metadata["recommended_embedding_features"]
    excluded = set(metadata["high_cardinality_features"])

    if len(numerical) != EXPECTED_NUMERICAL_COUNT:
        raise ValueError(f"numerical_features 数量 {len(numerical)} != {EXPECTED_NUMERICAL_COUNT}")

    overlap = excluded & set(categorical)
    if overlap:
        raise ValueError(f"recommended_embedding_features 含高基数字段：{sorted(overlap)}")

    return metadata


def is_missing_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def value_to_text(value: Any) -> str:
    return str(value).strip()


def build_categorical_vocabularies(
    train_files: list[Path],
    categorical_features: list[str],
    max_rows: int | None,
) -> dict[str, CategoricalVocab]:
    """仅扫描 train，建立 categorical vocabulary。"""

    known_sets: dict[str, set[str]] = {col: set() for col in categorical_features}
    collected = 0

    for parquet_path in train_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=categorical_features,
            batch_size=BATCH_SIZE,
        ):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            for column in categorical_features:
                for value in batch_df[column]:
                    if not is_missing_value(value):
                        known_sets[column].add(value_to_text(value))

            collected += len(batch_df)
            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    vocabularies: dict[str, CategoricalVocab] = {}
    for column in categorical_features:
        known_sorted = sorted(known_sets[column])
        category_to_index: dict[str, int] = {}
        index_to_category = ["__UNKNOWN__", "__MISSING__"]
        category_to_index["__UNKNOWN__"] = UNKNOWN_INDEX
        category_to_index["__MISSING__"] = MISSING_INDEX

        for index_offset, category in enumerate(known_sorted, start=KNOWN_INDEX_START):
            category_to_index[category] = index_offset
            index_to_category.append(category)

        vocabularies[column] = CategoricalVocab(
            feature_name=column,
            category_to_index=category_to_index,
            index_to_category=index_to_category,
            vocab_size=len(index_to_category),
            train_unique_count=len(known_sorted),
        )

    return vocabularies


def save_vocabularies(vocabularies: dict[str, CategoricalVocab], vocab_dir: Path) -> None:
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for column, vocab in vocabularies.items():
        payload = {
            "feature_name": vocab.feature_name,
            "vocab_size": vocab.vocab_size,
            "train_unique_count": vocab.train_unique_count,
            "unknown_index": UNKNOWN_INDEX,
            "missing_index": MISSING_INDEX,
            "known_index_start": KNOWN_INDEX_START,
            "index_to_category": vocab.index_to_category,
            "category_to_index": vocab.category_to_index,
        }
        vocab_path = vocab_dir / f"{column}.pkl"
        with vocab_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def encode_categorical_value(value: Any, vocab: CategoricalVocab) -> int:
    if is_missing_value(value):
        return MISSING_INDEX
    text = value_to_text(value)
    encoded = vocab.category_to_index.get(text)
    if encoded is None:
        return UNKNOWN_INDEX
    return encoded


def encode_categorical_series(series: pd.Series, vocab: CategoricalVocab) -> np.ndarray:
    return series.map(lambda value: encode_categorical_value(value, vocab)).to_numpy(
        dtype=np.int64
    )


def compute_click_checksum(parquet_files: list[Path], max_rows: int | None) -> str:
    hasher = hashlib.sha256()
    collected = 0
    for parquet_path in parquet_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["click"],
            batch_size=BATCH_SIZE,
        ):
            clicks = record_batch.to_pandas()["click"].to_numpy()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                clicks = clicks[:remaining]
            hasher.update(clicks.tobytes())
            collected += len(clicks)
            if max_rows is not None and collected >= max_rows:
                break
        if max_rows is not None and collected >= max_rows:
            break
    return hasher.hexdigest()


def load_numerical_matrix(
    parquet_files: list[Path],
    numerical_features: list[str],
    max_rows: int | None,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    collected = 0

    for parquet_path in parquet_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(
            columns=numerical_features,
            batch_size=BATCH_SIZE,
        ):
            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - collected
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            matrix = batch_df[numerical_features].to_numpy(dtype=np.float64)
            if np.isinf(matrix).any():
                inf_cols = [
                    col
                    for col in numerical_features
                    if np.isinf(batch_df[col].to_numpy(dtype=np.float64)).any()
                ]
                raise ValueError(f"numerical 特征存在 inf：{inf_cols}（文件 {parquet_path.name}）")

            parts.append(matrix)
            collected += len(batch_df)
            if max_rows is not None and collected >= max_rows:
                break

        if max_rows is not None and collected >= max_rows:
            break

    if collected == 0:
        raise ValueError("未读取到 numerical 数据。")

    return np.vstack(parts)


def fit_numerical_preprocessing(
    train_matrix: np.ndarray,
    numerical_features: list[str],
) -> tuple[dict[str, float], StandardScaler, dict[str, Any]]:
    """基于 train 计算 median fill + StandardScaler。"""

    fill_values: dict[str, float] = {}
    constant_features: list[str] = []

    filled_matrix = train_matrix.copy()
    for col_index, column in enumerate(numerical_features):
        column_values = train_matrix[:, col_index]
        if np.isnan(column_values).any():
            median_value = float(np.nanmedian(column_values))
            fill_values[column] = median_value
            filled_matrix[:, col_index] = np.where(
                np.isnan(column_values),
                median_value,
                column_values,
            )
        else:
            fill_values[column] = float(np.nanmedian(column_values))

        if np.nanstd(filled_matrix[:, col_index]) == 0.0:
            constant_features.append(column)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(filled_matrix).astype(np.float32)

    if np.isnan(scaled_matrix).any() or np.isinf(scaled_matrix).any():
        raise ValueError("标准化后的 train numerical 存在 NaN 或 inf。")

    stats = {
        "train_mean_after_scale": {
            column: float(scaled_matrix[:, idx].mean())
            for idx, column in enumerate(numerical_features)
        },
        "train_std_after_scale": {
            column: float(scaled_matrix[:, idx].std())
            for idx, column in enumerate(numerical_features)
        },
        "constant_features": constant_features,
        "scaler_mean": {
            column: float(scaler.mean_[idx])
            for idx, column in enumerate(numerical_features)
        },
        "scaler_scale": {
            column: float(scaler.scale_[idx])
            for idx, column in enumerate(numerical_features)
        },
    }

    return fill_values, scaler, stats


def apply_numerical_preprocessing(
    raw_matrix: np.ndarray,
    numerical_features: list[str],
    fill_values: dict[str, float],
    scaler: StandardScaler,
) -> np.ndarray:
    if np.isinf(raw_matrix).any():
        raise ValueError("numerical 特征存在 inf，拒绝静默处理。")

    filled = raw_matrix.copy()
    for col_index, column in enumerate(numerical_features):
        column_values = raw_matrix[:, col_index]
        fill_value = fill_values[column]
        filled[:, col_index] = np.where(np.isnan(column_values), fill_value, column_values)

    scaled = scaler.transform(filled).astype(np.float32)
    if np.isnan(scaled).any() or np.isinf(scaled).any():
        raise ValueError("标准化后 numerical 存在 NaN 或 inf。")
    return scaled


def get_output_column_order(
    categorical_features: list[str],
    numerical_features: list[str],
    keep_id: bool = True,
) -> list[str]:
    cat_cols = [f"{col}_idx" for col in categorical_features]
    num_cols = [f"{col}_scaled" for col in numerical_features]
    prefix = ["id"] if keep_id else []
    return [*prefix, *cat_cols, *num_cols, "click"]


def process_and_write_split(
    split_name: str,
    input_files: list[Path],
    output_dir: Path,
    categorical_features: list[str],
    numerical_features: list[str],
    vocabularies: dict[str, CategoricalVocab],
    fill_values: dict[str, float],
    scaler: StandardScaler,
    max_rows: int | None,
    cat_audit_accum: dict[str, dict[str, float | int]],
    is_train: bool,
) -> tuple[int, np.ndarray]:
    """逐文件处理 unified split，写出 pytorch parquet，保持行顺序。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.parquet"):
        old_file.unlink()

    output_columns = get_output_column_order(categorical_features, numerical_features)
    total_rows = 0
    click_parts: list[np.ndarray] = []

    for parquet_path in input_files:
        parquet_file = pq.ParquetFile(parquet_path)
        read_columns = ["id", "click", *categorical_features, *numerical_features]
        out_path = output_dir / parquet_path.name

        writer: pq.ParquetWriter | None = None
        file_rows = 0

        for record_batch in parquet_file.iter_batches(
            columns=read_columns,
            batch_size=BATCH_SIZE,
        ):
            if max_rows is not None and total_rows >= max_rows:
                break

            batch_df = record_batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - total_rows
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]

            batch_len = len(batch_df)
            output_data: dict[str, Any] = {"id": batch_df["id"].astype(str)}

            for column in categorical_features:
                encoded = encode_categorical_series(batch_df[column], vocabularies[column])
                output_data[f"{column}_idx"] = encoded.astype(np.int64)

                if column not in cat_audit_accum:
                    cat_audit_accum[column] = {
                        "min": np.iinfo(np.int64).max,
                        "max": 0,
                        "oov_count": 0,
                        "missing_count": 0,
                        "total_non_missing": 0,
                        "total_rows": 0,
                    }

                stats = cat_audit_accum[column]
                stats["min"] = min(int(stats["min"]), int(encoded.min()))
                stats["max"] = max(int(stats["max"]), int(encoded.max()))
                stats["total_rows"] += batch_len

                missing_mask = batch_df[column].map(is_missing_value).to_numpy()
                stats["missing_count"] += int(missing_mask.sum())

                if not is_train:
                    oov_mask = (encoded == UNKNOWN_INDEX) & (~missing_mask)
                    stats["oov_count"] += int(oov_mask.sum())
                    stats["total_non_missing"] += int((~missing_mask).sum())

            raw_numerical = batch_df[numerical_features].to_numpy(dtype=np.float64)
            if np.isinf(raw_numerical).any():
                raise ValueError(f"{split_name} numerical 存在 inf：{parquet_path.name}")

            scaled = apply_numerical_preprocessing(
                raw_numerical,
                numerical_features,
                fill_values,
                scaler,
            )
            for col_index, column in enumerate(numerical_features):
                output_data[f"{column}_scaled"] = scaled[:, col_index]

            output_data["click"] = batch_df["click"].to_numpy(dtype=np.int8)
            click_parts.append(output_data["click"].copy())

            out_df = pd.DataFrame(output_data)[output_columns]
            table = pa.Table.from_pandas(out_df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)

            total_rows += batch_len
            file_rows += batch_len

            if max_rows is not None and total_rows >= max_rows:
                break

        if writer is not None:
            writer.close()
            print(f"[{split_name}] 写出 {out_path.name}：{file_rows:,} 行")
        elif file_rows == 0 and max_rows is not None and total_rows >= max_rows:
            pass

        if max_rows is not None and total_rows >= max_rows:
            break

    clicks = np.concatenate(click_parts).astype(np.int8) if click_parts else np.array([], dtype=np.int8)
    return total_rows, clicks


def finalize_categorical_audits(
    vocabularies: dict[str, CategoricalVocab],
    train_audit: dict[str, dict[str, float | int]],
    valid_audit: dict[str, dict[str, float | int]],
) -> tuple[list[CategoricalAudit], float]:
    audits: list[CategoricalAudit] = []
    max_valid_oov_rate = 0.0

    for column in vocabularies:
        vocab = vocabularies[column]
        train_stats = train_audit.get(column, {})
        valid_stats = valid_audit.get(column, {})

        valid_non_missing = int(valid_stats.get("total_non_missing", 0))
        valid_oov_count = int(valid_stats.get("oov_count", 0))
        valid_missing_count = int(valid_stats.get("missing_count", 0))
        valid_total = int(valid_stats.get("total_rows", 0))

        valid_oov_rate = valid_oov_count / valid_non_missing if valid_non_missing else 0.0
        valid_missing_rate = valid_missing_count / valid_total if valid_total else 0.0
        max_valid_oov_rate = max(max_valid_oov_rate, valid_oov_rate)

        train_min = int(train_stats.get("min", 0))
        train_max = int(train_stats.get("max", 0))

        if train_min < 0 or train_max >= vocab.vocab_size:
            raise ValueError(
                f"{column} train 编码越界：min={train_min}, max={train_max}, vocab_size={vocab.vocab_size}"
            )

        valid_max = int(valid_stats.get("max", 0))
        if valid_max >= vocab.vocab_size:
            raise ValueError(
                f"{column} valid 编码越界：max={valid_max}, vocab_size={vocab.vocab_size}"
            )

        audits.append(
            CategoricalAudit(
                feature=column,
                vocab_size=vocab.vocab_size,
                train_min=train_min,
                train_max=train_max,
                valid_oov_count=valid_oov_count,
                valid_oov_rate=valid_oov_rate,
                valid_missing_rate=valid_missing_rate,
            )
        )

    return audits, max_valid_oov_rate


def promote_temp_output(temp_dir: Path, final_dir: Path) -> None:
    """验证通过后将临时目录提升为正式输出。"""

    if final_dir.exists():
        for parquet_path in final_dir.glob("*.parquet"):
            parquet_path.unlink()
    else:
        final_dir.mkdir(parents=True, exist_ok=True)

    for parquet_path in sorted(temp_dir.glob("*.parquet")):
        target = final_dir / parquet_path.name
        parquet_path.replace(target)

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()


def cleanup_temp_dirs() -> None:
    for temp_dir in (PYTORCH_TRAIN_TMP, PYTORCH_VALID_TMP):
        if temp_dir.exists():
            for parquet_path in temp_dir.glob("*.parquet"):
                parquet_path.unlink()
            if not any(temp_dir.iterdir()):
                temp_dir.rmdir()


def print_final_summary(
    train_rows: int,
    valid_rows: int,
    categorical_features: list[str],
    vocab_sizes: dict[str, int],
    max_valid_oov_rate: float,
    train_click_match: bool,
    valid_click_match: bool,
) -> None:
    print("\n" + "=" * 40)
    print("PYTORCH INPUT PREPARATION SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_rows}")
    print(f"VALID_ROWS = {valid_rows}")
    print(f"CATEGORICAL_FEATURE_COUNT = {len(categorical_features)}")
    print(f"NUMERICAL_FEATURE_COUNT = {EXPECTED_NUMERICAL_COUNT}")
    print(f"CATEGORICAL_FEATURES = {categorical_features}")
    print(f"VOCAB_SIZES = {vocab_sizes}")
    print(f"MAX_VALID_OOV_RATE = {max_valid_oov_rate:.6f}")
    print(f"NUMERICAL_SCALER = {SCALER_PATH}")
    print(f"TRAIN_CLICK_MATCH = {train_click_match}")
    print(f"VALID_CLICK_MATCH = {valid_click_match}")
    print("PYTORCH_TRAIN_PATH =")
    print("data/modeling/pytorch_train/")
    print("PYTORCH_VALID_PATH =")
    print("data/modeling/pytorch_valid/")
    print("HOLDOUT_USED = False")
    print("VALIDATION_PASSED = True")
    print("=" * 40)


def main() -> None:
    state = BuildState()
    train_limit, valid_limit = get_row_limits(TEST_MODE)

    log(state, "=" * 72)
    log(state, "百度 CTR 项目 — 第 39 步 PyTorch 输入准备")
    log(state, f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    log(state, f"TEST_MODE = {TEST_MODE}")

    assert_safe_path(UNIFIED_TRAIN_DIR)
    assert_safe_path(UNIFIED_VALID_DIR)

    metadata = load_unified_metadata(UNIFIED_METADATA_PATH)
    numerical_features: list[str] = metadata["numerical_features"]
    categorical_features: list[str] = metadata["recommended_embedding_features"]

    train_files = get_sorted_parquet_files(UNIFIED_TRAIN_DIR)
    valid_files = get_sorted_parquet_files(UNIFIED_VALID_DIR)

    unified_train_checksum = compute_click_checksum(train_files, train_limit)
    unified_valid_checksum = compute_click_checksum(valid_files, valid_limit)

    log(state, f"\ncategorical 特征（{len(categorical_features)}）：{categorical_features}")
    log(state, f"numerical 特征（{len(numerical_features)}）：{len(numerical_features)} 个")

    log(state, "\nStep 1：基于 train 建立 categorical vocabulary ...")
    vocabularies = build_categorical_vocabularies(
        train_files,
        categorical_features,
        train_limit,
    )
    save_vocabularies(vocabularies, VOCAB_DIR)
    for column, vocab in vocabularies.items():
        log(state, f"  {column}: vocab_size={vocab.vocab_size}, train_unique={vocab.train_unique_count}")

    log(state, "\nStep 2：基于 train 计算 numerical fill + StandardScaler ...")
    train_numerical_raw = load_numerical_matrix(train_files, numerical_features, train_limit)
    fill_values, scaler, num_stats = fit_numerical_preprocessing(
        train_numerical_raw,
        numerical_features,
    )
    del train_numerical_raw
    gc.collect()

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    FILL_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FILL_VALUES_PATH.write_text(
        json.dumps(fill_values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if num_stats["constant_features"]:
        log(state, f"  常数特征（scale=0，StandardScaler 已处理）：{num_stats['constant_features']}")

    log(state, "\nStep 3：编码并写出 pytorch train / valid（保持行顺序）...")
    cleanup_temp_dirs()
    train_cat_audit: dict[str, dict[str, float | int]] = {}
    valid_cat_audit: dict[str, dict[str, float | int]] = {}

    try:
        train_rows, _ = process_and_write_split(
            "train",
            train_files,
            PYTORCH_TRAIN_TMP,
            categorical_features,
            numerical_features,
            vocabularies,
            fill_values,
            scaler,
            train_limit,
            train_cat_audit,
            is_train=True,
        )
        valid_rows, _ = process_and_write_split(
            "valid",
            valid_files,
            PYTORCH_VALID_TMP,
            categorical_features,
            numerical_features,
            vocabularies,
            fill_values,
            scaler,
            valid_limit,
            valid_cat_audit,
            is_train=False,
        )

        if not TEST_MODE:
            if train_rows != FORMAL_TRAIN_ROWS:
                raise ValueError(f"train 行数 {train_rows:,} != {FORMAL_TRAIN_ROWS:,}")
            if valid_rows != FORMAL_VALID_ROWS:
                raise ValueError(f"valid 行数 {valid_rows:,} != {FORMAL_VALID_ROWS:,}")

        pytorch_train_checksum = compute_click_checksum(
            get_sorted_parquet_files(PYTORCH_TRAIN_TMP),
            train_limit,
        )
        pytorch_valid_checksum = compute_click_checksum(
            get_sorted_parquet_files(PYTORCH_VALID_TMP),
            valid_limit,
        )

        train_click_match = pytorch_train_checksum == unified_train_checksum
        valid_click_match = pytorch_valid_checksum == unified_valid_checksum

        log(state, f"\nPYTORCH_TRAIN_ROWS = {train_rows:,}")
        log(state, f"PYTORCH_VALID_ROWS = {valid_rows:,}")
        log(state, f"TRAIN_CLICK_MATCH = {train_click_match}")
        log(state, f"VALID_CLICK_MATCH = {valid_click_match}")

        if not train_click_match or not valid_click_match:
            cleanup_temp_dirs()
            raise RuntimeError(
                "click 序列与 unified 样本不一致，未保存正式输出。"
                f" train_match={train_click_match}, valid_match={valid_click_match}"
            )

        promote_temp_output(PYTORCH_TRAIN_TMP, PYTORCH_TRAIN_DIR)
        promote_temp_output(PYTORCH_VALID_TMP, PYTORCH_VALID_DIR)
    except Exception:
        cleanup_temp_dirs()
        raise

    cat_audits, max_valid_oov_rate = finalize_categorical_audits(
        vocabularies,
        train_cat_audit,
        valid_cat_audit,
    )

    log(state, "\nCategorical 编码审计：")
    log(
        state,
        f"{'feature':<22} {'vocab':>6} {'tr_min':>6} {'tr_max':>6} "
        f"{'OOV_cnt':>8} {'OOV_rate':>9} {'miss_rate':>9}",
    )
    log(state, "-" * 75)
    for audit in cat_audits:
        log(
            state,
            f"{audit.feature:<22} {audit.vocab_size:>6} {audit.train_min:>6} {audit.train_max:>6} "
            f"{audit.valid_oov_count:>8} {100 * audit.valid_oov_rate:>8.4f}% "
            f"{100 * audit.valid_missing_rate:>8.4f}%",
        )

    log(state, "\nNumerical 标准化后 train 统计（抽样摘要）：")
    for column in numerical_features[:5]:
        log(
            state,
            f"  {column}: mean={num_stats['train_mean_after_scale'][column]:.4f}, "
            f"std={num_stats['train_std_after_scale'][column]:.4f}",
        )
    log(state, "  ...")

    vocab_sizes = {col: vocabularies[col].vocab_size for col in categorical_features}
    oov_rates = {audit.feature: audit.valid_oov_rate for audit in cat_audits}

    metadata_out = {
        "script_name": "scripts/39_prepare_pytorch_inputs.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_mode": TEST_MODE,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "categorical_feature_count": len(categorical_features),
        "numerical_feature_count": len(numerical_features),
        "vocab_sizes": vocab_sizes,
        "vocab_details": [
            {
                "feature_name": col,
                "vocab_size": vocabularies[col].vocab_size,
                "train_unique_count": vocabularies[col].train_unique_count,
                "valid_oov_rate": oov_rates[col],
                "vocab_path": str(VOCAB_DIR / f"{col}.pkl"),
            }
            for col in categorical_features
        ],
        "oov_rates": oov_rates,
        "numerical_fill_values": fill_values,
        "numerical_stats": num_stats,
        "scaler_path": str(SCALER_PATH),
        "fill_values_path": str(FILL_VALUES_PATH),
        "vocab_dir": str(VOCAB_DIR),
        "train_click_checksum": pytorch_train_checksum,
        "valid_click_checksum": pytorch_valid_checksum,
        "source_train_click_checksum": unified_train_checksum,
        "source_valid_click_checksum": unified_valid_checksum,
        "train_click_match": train_click_match,
        "valid_click_match": valid_click_match,
        "source_train_path": str(UNIFIED_TRAIN_DIR),
        "source_valid_path": str(UNIFIED_VALID_DIR),
        "pytorch_train_path": str(PYTORCH_TRAIN_DIR),
        "pytorch_valid_path": str(PYTORCH_VALID_DIR),
        "output_column_order": get_output_column_order(categorical_features, numerical_features),
        "encoding_scheme": {
            "unknown_index": UNKNOWN_INDEX,
            "missing_index": MISSING_INDEX,
            "known_index_start": KNOWN_INDEX_START,
        },
        "random_seed": RANDOM_SEED,
        "holdout_used": False,
        "validation_passed": True,
        "max_valid_oov_rate": max_valid_oov_rate,
    }

    PYTORCH_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PYTORCH_METADATA_PATH.write_text(
        json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log(state, f"\nMetadata: {PYTORCH_METADATA_PATH}")

    print_final_summary(
        train_rows,
        valid_rows,
        categorical_features,
        vocab_sizes,
        max_valid_oov_rate,
        train_click_match,
        valid_click_match,
    )


if __name__ == "__main__":
    main()
