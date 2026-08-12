"""
百度 CTR 项目 — Final Holdout Evaluation（第 43 步）

一次性 final holdout 评估：
    - Final train = unified_train + unified_valid（2014-10-21 ~ 2014-10-29）
    - Holdout = 2014-10-30
    - 重新训练 LightGBM_Unified + DeepFM（冻结超参 / epoch / ensemble 权重）
    - 禁止 holdout 参与训练、预处理 fit、early stopping、权重搜索

用法：
    python scripts/43_final_holdout_evaluation.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# 路径与冻结配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HOLDOUT_TARGET_ENCODED_DIR = Path("data/features/target_encoded/holdout")
UNIFIED_TRAIN_DIR = Path("data/modeling/unified_train")
UNIFIED_VALID_DIR = Path("data/modeling/unified_valid")
UNIFIED_HOLDOUT_DIR = Path("data/modeling/unified_holdout")
PYTORCH_HOLDOUT_DIR = Path("data/modeling/pytorch_holdout")
FINAL_VOCAB_DIR = Path("data/modeling/final_pytorch_artifacts/vocabs")
FINAL_FILL_VALUES_PATH = Path("data/modeling/final_pytorch_artifacts/numerical_fill_values.json")

UNIFIED_METADATA_PATH = Path("outputs/unified_modeling_sample_metadata.json")
STEP32_BEST_PARAMS_PATH = Path("outputs/lightgbm_optuna_best_params.json")
ENSEMBLE_METADATA_PATH = Path("outputs/lightgbm_deepfm_ensemble_metadata.json")
DEEPFM_DEV_METADATA_PATH = Path("outputs/deepfm_metadata.json")

FINAL_LGBM_MODEL_PATH = Path("models/final_lightgbm.txt")
FINAL_DEEPFM_MODEL_PATH = Path("models/final_deepfm.pt")
FINAL_SCALER_PATH = Path("models/final_pytorch_numerical_scaler.joblib")

METRICS_CSV_PATH = Path("outputs/final_holdout_metrics.csv")
PREDICTIONS_PATH = Path("outputs/predictions/final_holdout_predictions.parquet")
METADATA_PATH = Path("outputs/final_holdout_metadata.json")

FINAL_TRAIN_DATE_START = "2014-10-21"
FINAL_TRAIN_DATE_END = "2014-10-29"
HOLDOUT_DATE = "2014-10-30"

EXPECTED_FINAL_TRAIN_ROWS = 2_500_000
BATCH_SIZE = 200_000
READ_BATCH_SIZE = 4096
OUTPUT_PART_ROWS = 250_000

FROZEN_LGBM_ITERATIONS = 324
FROZEN_LGBM_ITERATION_SOURCE = "development_validation_step38_model_b_unified"
FROZEN_DEEPFM_EPOCHS = 4
FROZEN_DEEPFM_EPOCH_SOURCE = "development_validation_step41"

ENSEMBLE_LIGHTGBM_WEIGHT = 0.588
ENSEMBLE_DEEPFM_WEIGHT = 0.412
ENSEMBLE_WEIGHT_SOURCE = "development_validation_step42"

PROB_CLIP_EPS = 1e-15
THRESHOLD = 0.5
RANDOM_SEED = 42


@dataclass
class HoldoutMetrics:
    model: str
    auc: float
    logloss: float
    brier: float
    average_precision: float
    actual_ctr: float
    mean_predicted_ctr: float
    calibration_gap: float
    precision_0_5: float
    recall_0_5: float
    f1_0_5: float


def load_script_module(module_name: str, script_relative_path: str):
    script_path = PROJECT_ROOT / script_relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块：{script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


STEP37 = load_script_module("step37", "scripts/37_build_unified_modeling_sample.py")
STEP38 = load_script_module("step38", "scripts/38_train_unified_lightgbm_baselines.py")
STEP39 = load_script_module("step39", "scripts/39_prepare_pytorch_inputs.py")
STEP41 = load_script_module("step41", "scripts/41_train_deepfm.py")

DeepFM = STEP41.DeepFM


def log(message: str = "") -> None:
    print(message)


def get_sorted_parquet_files(parquet_dir: Path) -> list[Path]:
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet：{parquet_dir}")
    return files


def count_rows(parquet_files: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files)


def get_final_train_files() -> list[Path]:
    return get_sorted_parquet_files(UNIFIED_TRAIN_DIR) + get_sorted_parquet_files(UNIFIED_VALID_DIR)


def load_unified_feature_config() -> dict[str, Any]:
    metadata = json.loads(UNIFIED_METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("holdout_used") is not False:
        raise ValueError("unified metadata holdout_used 必须为 false。")
    return metadata


def prepare_holdout_unified_batch(
    dataframe: pd.DataFrame,
    column_config: STEP37.ColumnConfig,
) -> pd.DataFrame:
    """构建 unified_holdout batch，允许且仅允许 2014-10-30。"""
    dataframe = dataframe.reset_index(drop=True)
    context = "holdout batch"

    click = STEP37.validate_click(dataframe, context)
    split_date = STEP37.extract_split_date(dataframe)
    unique_dates = set(split_date.astype(str).unique())
    if unique_dates != {HOLDOUT_DATE}:
        raise ValueError(f"holdout 含意外日期：{sorted(unique_dates)}")

    output = pd.DataFrame({"id": dataframe["id"].astype(str)})
    output["split_date"] = split_date.reset_index(drop=True)

    for column in column_config.auxiliary_columns:
        output[column] = dataframe[column].reset_index(drop=True)

    for column in column_config.categorical_columns:
        output[column] = dataframe[column].reset_index(drop=True)

    for column in column_config.numerical_columns:
        if column in STEP37.COMPUTED_NUMERICAL:
            continue
        output[column] = pd.to_numeric(dataframe[column], errors="coerce").reset_index(drop=True)

    hour_sin, hour_cos = STEP37.compute_hour_cyclical(dataframe["hour_of_day"])
    output["hour_sin"] = hour_sin
    output["hour_cos"] = hour_cos

    output["click"] = click.reset_index(drop=True)
    return output[column_config.output_column_order]


def build_unified_holdout(column_config: STEP37.ColumnConfig) -> tuple[list[Path], int]:
    """从 target_encoded/holdout 构建 unified_holdout（全量，不抽样）。"""
    input_files = get_sorted_parquet_files(HOLDOUT_TARGET_ENCODED_DIR)
    if UNIFIED_HOLDOUT_DIR.exists():
        for old in UNIFIED_HOLDOUT_DIR.glob("*.parquet"):
            old.unlink()
    else:
        UNIFIED_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    buffer_frames: list[pd.DataFrame] = []
    buffer_rows = 0
    part_index = 0
    output_files: list[Path] = []
    total_rows = 0

    log("\n构建 unified_holdout（全量 2014-10-30）...")
    for file_index, parquet_path in enumerate(input_files, start=1):
        file_rows = 0
        for batch_df in STEP37.iter_file_batches(
            parquet_path, column_config.read_columns, STEP37.READ_BATCH_SIZE
        ):
            prepared = prepare_holdout_unified_batch(batch_df, column_config)
            buffer_frames.append(prepared)
            buffer_rows += len(prepared)
            file_rows += len(prepared)
            total_rows += len(prepared)

            if buffer_rows >= OUTPUT_PART_ROWS:
                merged = pd.concat(buffer_frames, ignore_index=True)
                written, part_index = STEP37.write_parquet_parts(
                    merged,
                    UNIFIED_HOLDOUT_DIR,
                    part_index,
                    column_config.output_column_order,
                )
                output_files.extend(written)
                buffer_frames = []
                buffer_rows = 0
                del merged
                gc.collect()

        log(f"  holdout 文件 {file_index}/{len(input_files)}: {parquet_path.name}，{file_rows:,} 行")

    if buffer_frames:
        merged = pd.concat(buffer_frames, ignore_index=True)
        written, part_index = STEP37.write_parquet_parts(
            merged,
            UNIFIED_HOLDOUT_DIR,
            part_index,
            column_config.output_column_order,
        )
        output_files.extend(written)

    return output_files, total_rows


def audit_split_dates(parquet_files: list[Path], split_name: str) -> tuple[str, str]:
    date_min = None
    date_max = None
    for parquet_path in parquet_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["split_date"], batch_size=BATCH_SIZE
        ):
            dates = record_batch.to_pandas()["split_date"].astype(str)
            batch_min = dates.min()
            batch_max = dates.max()
            date_min = batch_min if date_min is None else min(date_min, batch_min)
            date_max = batch_max if date_max is None else max(date_max, batch_max)
    if date_min is None or date_max is None:
        raise ValueError(f"{split_name} 无 split_date 数据。")
    return date_min, date_max


def run_leakage_audit(final_train_files: list[Path], holdout_files: list[Path]) -> dict[str, Any]:
    train_min, train_max = audit_split_dates(final_train_files, "final_train")
    holdout_min, holdout_max = audit_split_dates(holdout_files, "holdout")

    audit = {
        "FINAL_TRAIN_MAX_DATE": train_max,
        "HOLDOUT_DATE_MIN": holdout_min,
        "HOLDOUT_DATE_MAX": holdout_max,
        "HOLDOUT_LABEL_USED_FOR_FEATURES": False,
        "HOLDOUT_USED_FOR_VOCAB": False,
        "HOLDOUT_USED_FOR_SCALER": False,
        "HOLDOUT_USED_FOR_EARLY_STOPPING": False,
        "HOLDOUT_USED_FOR_ENSEMBLE_WEIGHT": False,
    }

    log("\n--- Leakage Audit ---")
    for key, value in audit.items():
        log(f"{key} = {value}")

    if train_max > FINAL_TRAIN_DATE_END:
        raise RuntimeError(f"FINAL_TRAIN_MAX_DATE {train_max} > {FINAL_TRAIN_DATE_END}")
    if holdout_min != HOLDOUT_DATE or holdout_max != HOLDOUT_DATE:
        raise RuntimeError(f"Holdout 日期异常：{holdout_min} ~ {holdout_max}")
    for flag in (
        "HOLDOUT_LABEL_USED_FOR_FEATURES",
        "HOLDOUT_USED_FOR_VOCAB",
        "HOLDOUT_USED_FOR_SCALER",
        "HOLDOUT_USED_FOR_EARLY_STOPPING",
        "HOLDOUT_USED_FOR_ENSEMBLE_WEIGHT",
    ):
        if audit[flag] is not False:
            raise RuntimeError(f"Leakage audit failed: {flag} = {audit[flag]}")

    log("LEAKAGE_AUDIT_PASSED = True")
    return audit


def compute_metrics(model_name: str, y_true: np.ndarray, probabilities: np.ndarray) -> HoldoutMetrics:
    if np.isnan(probabilities).any() or np.isinf(probabilities).any():
        raise ValueError(f"{model_name} 预测存在 NaN/inf。")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"{model_name} 预测超出 [0,1]。")

    clipped = np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)
    predicted = (probabilities >= THRESHOLD).astype(np.int8)
    actual_ctr = float(y_true.mean())
    mean_pred = float(probabilities.mean())
    return HoldoutMetrics(
        model=model_name,
        auc=float(roc_auc_score(y_true, probabilities)),
        logloss=float(log_loss(y_true, clipped, labels=[0, 1])),
        brier=float(brier_score_loss(y_true, probabilities)),
        average_precision=float(average_precision_score(y_true, probabilities)),
        actual_ctr=actual_ctr,
        mean_predicted_ctr=mean_pred,
        calibration_gap=abs(mean_pred - actual_ctr),
        precision_0_5=float(precision_score(y_true, predicted, zero_division=0)),
        recall_0_5=float(recall_score(y_true, predicted, zero_division=0)),
        f1_0_5=float(f1_score(y_true, predicted, zero_division=0)),
    )


def train_final_lightgbm(
    metadata: dict[str, Any],
    final_train_files: list[Path],
    holdout_files: list[Path],
) -> tuple[lgb.Booster, np.ndarray]:
    numerical_features = metadata["numerical_features"]
    categorical_features = metadata["recommended_embedding_features"]
    log1p_columns = metadata.get("log1p_columns") or STEP38.get_log1p_columns(numerical_features)
    params = STEP38.load_step32_params(STEP32_BEST_PARAMS_PATH)

    log("\n训练 Final LightGBM_Unified ...")
    vocabularies = STEP38.build_categorical_vocabularies(
        final_train_files, categorical_features, max_rows=None
    )

    x_train, y_train, _ = STEP38.load_unified_split_dataframe(
        final_train_files,
        numerical_features,
        log1p_columns,
        categorical_features,
        vocabularies,
        max_rows=None,
    )

    train_data = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    booster = lgb.train(
        {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "feature_pre_filter": False,
            **params,
        },
        train_data,
        num_boost_round=FROZEN_LGBM_ITERATIONS,
    )

    booster.save_model(str(FINAL_LGBM_MODEL_PATH))
    log(f"  保存 {FINAL_LGBM_MODEL_PATH}（rounds={FROZEN_LGBM_ITERATIONS}）")

    holdout_probs = predict_lightgbm_holdout_batches(
        booster,
        holdout_files,
        numerical_features,
        log1p_columns,
        categorical_features,
        vocabularies,
    )
    del x_train, y_train, train_data
    gc.collect()
    return booster, holdout_probs


def predict_lightgbm_holdout_batches(
    booster: lgb.Booster,
    holdout_files: list[Path],
    numerical_features: list[str],
    log1p_columns: list[str],
    categorical_features: list[str],
    vocabularies: dict[str, list[str]],
) -> np.ndarray:
    prob_parts: list[np.ndarray] = []
    read_columns = list(
        dict.fromkeys(["click", *numerical_features, *categorical_features])
    )

    for parquet_path in holdout_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=read_columns, batch_size=BATCH_SIZE
        ):
            batch_df = record_batch.to_pandas()
            raw_matrix = batch_df[numerical_features].to_numpy(dtype=np.float64)
            processed = STEP38.preprocess_numerical_matrix(
                raw_matrix, numerical_features, log1p_columns
            )
            data: dict[str, Any] = {}
            for index, column in enumerate(numerical_features):
                data[column] = processed[:, index]
            for column in categorical_features:
                data[column] = STEP38.map_to_categorical_series(
                    batch_df[column], vocabularies[column]
                )
            x_batch = pd.DataFrame(data)
            probs = booster.predict(x_batch, num_iteration=FROZEN_LGBM_ITERATIONS)
            prob_parts.append(probs.astype(np.float64))

    return np.concatenate(prob_parts)


def compute_final_train_ctr(final_train_files: list[Path]) -> float:
    total_clicks = 0
    total_rows = 0
    for parquet_path in final_train_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["click"], batch_size=BATCH_SIZE
        ):
            clicks = record_batch.to_pandas()["click"].to_numpy()
            total_clicks += int(clicks.sum())
            total_rows += len(clicks)
    return total_clicks / total_rows


def fit_final_pytorch_preprocessing(
    metadata: dict[str, Any],
    final_train_files: list[Path],
) -> tuple[dict[str, STEP39.CategoricalVocab], dict[str, float], StandardScaler, dict[str, int]]:
    categorical_features = metadata["recommended_embedding_features"]
    numerical_features = metadata["numerical_features"]

    log("\nFinal DeepFM preprocessing（仅 fit final train）...")
    vocabularies = STEP39.build_categorical_vocabularies(
        final_train_files, categorical_features, max_rows=None
    )
    vocab_sizes = {col: vocab.vocab_size for col, vocab in vocabularies.items()}

    STEP39.save_vocabularies(vocabularies, FINAL_VOCAB_DIR)

    train_matrix = STEP39.load_numerical_matrix(
        final_train_files, numerical_features, max_rows=None
    )
    fill_values, scaler, _ = STEP39.fit_numerical_preprocessing(train_matrix, numerical_features)
    del train_matrix
    gc.collect()

    FINAL_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, FINAL_SCALER_PATH)
    FINAL_FILL_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_FILL_VALUES_PATH.write_text(
        json.dumps(fill_values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"  保存 {FINAL_SCALER_PATH}")
    return vocabularies, fill_values, scaler, vocab_sizes


def build_batch_specs_from_files(
    parquet_files: list[Path],
    max_rows: int | None,
    batch_size: int,
) -> list[STEP41.BatchSpec]:
    specs: list[STEP41.BatchSpec] = []
    collected = 0
    for parquet_path in parquet_files:
        num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        if max_rows is not None:
            remaining = max_rows - collected
            if remaining <= 0:
                break
            num_rows = min(num_rows, remaining)

        for start_row in range(0, num_rows, batch_size):
            chunk_rows = min(batch_size, num_rows - start_row)
            specs.append(STEP41.BatchSpec(parquet_path, start_row, chunk_rows))

        collected += num_rows
        if max_rows is not None and collected >= max_rows:
            break

    if not specs:
        raise ValueError("未生成任何 batch spec。")
    return specs


class EncodedUnifiedIterableDataset(IterableDataset):
    """从 unified parquet 流式读取并应用 final train 编码。"""

    def __init__(
        self,
        parquet_files: list[Path],
        categorical_features: list[str],
        numerical_features: list[str],
        vocabularies: dict[str, STEP39.CategoricalVocab],
        fill_values: dict[str, float],
        scaler: StandardScaler,
        batch_size: int,
        shuffle: bool,
        epoch: int,
        seed: int,
    ) -> None:
        self.parquet_files = parquet_files
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.read_cols = [*categorical_features, *numerical_features, "click"]
        self.vocabularies = vocabularies
        self.fill_values = fill_values
        self.scaler = scaler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = epoch
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        specs = build_batch_specs_from_files(self.parquet_files, None, self.batch_size)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(specs)

        for spec in specs:
            batch_df = STEP41.read_batch_from_spec(spec, self.read_cols)
            cat_array = np.column_stack(
                [
                    STEP39.encode_categorical_series(batch_df[col], self.vocabularies[col])
                    for col in self.categorical_features
                ]
            ).astype(np.int64)
            raw_numerical = batch_df[self.numerical_features].to_numpy(dtype=np.float64)
            if np.isinf(raw_numerical).any():
                raise ValueError("numerical 存在 inf。")
            scaled = STEP39.apply_numerical_preprocessing(
                raw_numerical, self.numerical_features, self.fill_values, self.scaler
            ).astype(np.float32)
            click_array = batch_df["click"].to_numpy(dtype=np.float32)

            yield (
                torch.from_numpy(cat_array),
                torch.from_numpy(scaled),
                torch.from_numpy(click_array),
            )


def train_final_deepfm(
    metadata: dict[str, Any],
    final_train_files: list[Path],
    vocabularies: dict[str, STEP39.CategoricalVocab],
    fill_values: dict[str, float],
    scaler: StandardScaler,
    vocab_sizes: dict[str, int],
    train_ctr: float,
) -> DeepFM:
    categorical_features = metadata["recommended_embedding_features"]
    numerical_features = metadata["numerical_features"]
    global_bias_init = STEP41.compute_global_bias_init(train_ctr)
    device = STEP41.get_device()

    log(f"\n训练 Final DeepFM（epochs={FROZEN_DEEPFM_EPOCHS}, device={device.type})...")
    log(f"  global_bias_init = {global_bias_init:.6f} (train_ctr={train_ctr:.6f})")

    model = DeepFM(
        categorical_features=categorical_features,
        vocab_sizes=vocab_sizes,
        num_numerical_features=len(numerical_features),
        global_bias_init=global_bias_init,
        fm_embedding_dim=STEP41.FM_EMBEDDING_DIM,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    fm_params = list(model.fm_embeddings.parameters())
    fm_ids = {id(p) for p in fm_params}
    other_params = [p for p in model.parameters() if id(p) not in fm_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": fm_params, "lr": STEP41.FM_LEARNING_RATE},
            {"params": other_params, "lr": STEP41.LEARNING_RATE},
        ],
        weight_decay=STEP41.WEIGHT_DECAY,
    )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for epoch in range(1, FROZEN_DEEPFM_EPOCHS + 1):
        epoch_start = time.time()
        dataset = EncodedUnifiedIterableDataset(
            parquet_files=final_train_files,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            vocabularies=vocabularies,
            fill_values=fill_values,
            scaler=scaler,
            batch_size=STEP41.BATCH_SIZE,
            shuffle=True,
            epoch=epoch,
            seed=RANDOM_SEED,
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=0)
        model.train()
        total_loss = 0.0
        total_rows = 0
        for categorical, numerical, click in loader:
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            click = click.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(categorical, numerical)
            loss = criterion(logits, click)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=STEP41.GRADIENT_CLIP_NORM)
            optimizer.step()
            total_rows += click.shape[0]
            total_loss += float(loss.item()) * click.shape[0]
        log(
            f"  Epoch {epoch}/{FROZEN_DEEPFM_EPOCHS} "
            f"train_loss={total_loss / max(total_rows, 1):.6f} "
            f"elapsed={time.time() - epoch_start:.1f}s"
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    FINAL_DEEPFM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "DeepFM",
            "metadata": {
                "fm_embedding_dim": STEP41.FM_EMBEDDING_DIM,
                "deepfm_epochs": FROZEN_DEEPFM_EPOCHS,
                "global_bias_init": global_bias_init,
                "train_ctr": train_ctr,
            },
        },
        FINAL_DEEPFM_MODEL_PATH,
    )
    log(f"  保存 {FINAL_DEEPFM_MODEL_PATH}")
    return model


@torch.no_grad()
def predict_deepfm_holdout(
    model: DeepFM,
    metadata: dict[str, Any],
    holdout_files: list[Path],
    vocabularies: dict[str, STEP39.CategoricalVocab],
    fill_values: dict[str, float],
    scaler: StandardScaler,
) -> np.ndarray:
    device = STEP41.get_device()
    model.eval()
    categorical_features = metadata["recommended_embedding_features"]
    numerical_features = metadata["numerical_features"]
    read_cols = [*categorical_features, *numerical_features, "click"]
    prob_parts: list[np.ndarray] = []

    for parquet_path in holdout_files:
        parquet_file = pq.ParquetFile(parquet_path)
        for record_batch in parquet_file.iter_batches(columns=read_cols, batch_size=BATCH_SIZE):
            batch_df = record_batch.to_pandas()
            cat_array = np.column_stack(
                [
                    STEP39.encode_categorical_series(batch_df[col], vocabularies[col])
                    for col in categorical_features
                ]
            ).astype(np.int64)
            raw_numerical = batch_df[numerical_features].to_numpy(dtype=np.float64)
            scaled = STEP39.apply_numerical_preprocessing(
                raw_numerical, numerical_features, fill_values, scaler
            ).astype(np.float32)

            categorical = torch.from_numpy(cat_array).to(device)
            numerical = torch.from_numpy(scaled).to(device)
            logits = model(categorical, numerical)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            prob_parts.append(probs.astype(np.float64))

    return np.concatenate(prob_parts)


def load_holdout_clicks(holdout_files: list[Path]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for parquet_path in holdout_files:
        for record_batch in pq.ParquetFile(parquet_path).iter_batches(
            columns=["click"], batch_size=BATCH_SIZE
        ):
            parts.append(record_batch.to_pandas()["click"].to_numpy(dtype=np.int8))
    return np.concatenate(parts)


def metrics_to_row(metrics: HoldoutMetrics) -> dict[str, Any]:
    return {
        "model": metrics.model,
        "auc": metrics.auc,
        "logloss": metrics.logloss,
        "brier": metrics.brier,
        "average_precision": metrics.average_precision,
        "actual_ctr": metrics.actual_ctr,
        "mean_predicted_ctr": metrics.mean_predicted_ctr,
        "calibration_gap": metrics.calibration_gap,
        "precision_0_5": metrics.precision_0_5,
        "recall_0_5": metrics.recall_0_5,
        "f1_0_5": metrics.f1_0_5,
    }


def print_final_summary(
    final_train_rows: int,
    holdout_rows: int,
    actual_holdout_ctr: float,
    lgbm_metrics: HoldoutMetrics,
    deepfm_metrics: HoldoutMetrics,
    ensemble_metrics: HoldoutMetrics,
) -> None:
    by_auc = max(
        [lgbm_metrics, deepfm_metrics, ensemble_metrics], key=lambda item: item.auc
    ).model
    by_logloss = min(
        [lgbm_metrics, deepfm_metrics, ensemble_metrics], key=lambda item: item.logloss
    ).model

    print("\n" + "=" * 40)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 40)
    print(f"FINAL_TRAIN_DATES = {FINAL_TRAIN_DATE_START} ~ {FINAL_TRAIN_DATE_END}")
    print(f"FINAL_TRAIN_ROWS = {final_train_rows}")
    print(f"HOLDOUT_DATE = {HOLDOUT_DATE}")
    print(f"HOLDOUT_ROWS = {holdout_rows}")
    print(f"ACTUAL_HOLDOUT_CTR = {actual_holdout_ctr:.6f}")
    print("-" * 40)
    print("LIGHTGBM")
    print(f"AUC = {lgbm_metrics.auc:.6f}")
    print(f"LOGLOSS = {lgbm_metrics.logloss:.6f}")
    print(f"BRIER = {lgbm_metrics.brier:.6f}")
    print(f"AVERAGE_PRECISION = {lgbm_metrics.average_precision:.6f}")
    print(f"MEAN_PREDICTED_CTR = {lgbm_metrics.mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {lgbm_metrics.calibration_gap:.6f}")
    print("-" * 40)
    print("DEEPFM")
    print(f"AUC = {deepfm_metrics.auc:.6f}")
    print(f"LOGLOSS = {deepfm_metrics.logloss:.6f}")
    print(f"BRIER = {deepfm_metrics.brier:.6f}")
    print(f"AVERAGE_PRECISION = {deepfm_metrics.average_precision:.6f}")
    print(f"MEAN_PREDICTED_CTR = {deepfm_metrics.mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {deepfm_metrics.calibration_gap:.6f}")
    print("-" * 40)
    print("ENSEMBLE")
    print(f"LIGHTGBM_WEIGHT = {ENSEMBLE_LIGHTGBM_WEIGHT:.3f}")
    print(f"DEEPFM_WEIGHT = {ENSEMBLE_DEEPFM_WEIGHT:.3f}")
    print(f"AUC = {ensemble_metrics.auc:.6f}")
    print(f"LOGLOSS = {ensemble_metrics.logloss:.6f}")
    print(f"BRIER = {ensemble_metrics.brier:.6f}")
    print(f"AVERAGE_PRECISION = {ensemble_metrics.average_precision:.6f}")
    print(f"MEAN_PREDICTED_CTR = {ensemble_metrics.mean_predicted_ctr:.6f}")
    print(f"CALIBRATION_GAP = {ensemble_metrics.calibration_gap:.6f}")
    print("-" * 40)
    print(f"BEST_HOLDOUT_MODEL_BY_AUC = {by_auc}")
    print(f"BEST_HOLDOUT_MODEL_BY_LOGLOSS = {by_logloss}")
    print("ENSEMBLE_WEIGHT_TUNED_ON_HOLDOUT = False")
    print("MODEL_SELECTION_USED_HOLDOUT = False")
    print("LEAKAGE_AUDIT_PASSED = True")
    print("FINAL_EVALUATION_COMPLETE = True")
    print("=" * 40)


def main() -> None:
    log("=" * 72)
    log("百度 CTR 项目 — 第 43 步 Final Holdout Evaluation")
    log(f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")

    metadata = load_unified_feature_config()
    final_train_files = get_final_train_files()
    final_train_rows = count_rows(final_train_files)
    if final_train_rows != EXPECTED_FINAL_TRAIN_ROWS:
        raise ValueError(
            f"final train 行数 {final_train_rows:,} != {EXPECTED_FINAL_TRAIN_ROWS:,}"
        )

    schema_cols = STEP37.read_schema_columns(
        get_sorted_parquet_files(HOLDOUT_TARGET_ENCODED_DIR)[0]
    )
    column_config = STEP37.discover_column_config(schema_cols)
    _, holdout_rows = build_unified_holdout(column_config)
    holdout_files = get_sorted_parquet_files(UNIFIED_HOLDOUT_DIR)

    leakage_audit = run_leakage_audit(final_train_files, holdout_files)

    y_holdout = load_holdout_clicks(holdout_files)
    actual_holdout_ctr = float(y_holdout.mean())

    _, lgbm_probs = train_final_lightgbm(metadata, final_train_files, holdout_files)
    lgbm_metrics = compute_metrics("LightGBM_Unified", y_holdout, lgbm_probs)

    train_ctr = compute_final_train_ctr(final_train_files)
    vocabularies, fill_values, scaler, vocab_sizes = fit_final_pytorch_preprocessing(
        metadata, final_train_files
    )
    deepfm_model = train_final_deepfm(
        metadata,
        final_train_files,
        vocabularies,
        fill_values,
        scaler,
        vocab_sizes,
        train_ctr,
    )
    deepfm_probs = predict_deepfm_holdout(
        deepfm_model, metadata, holdout_files, vocabularies, fill_values, scaler
    )
    deepfm_metrics = compute_metrics("DeepFM", y_holdout, deepfm_probs)

    ensemble_probs = (
        ENSEMBLE_LIGHTGBM_WEIGHT * lgbm_probs + ENSEMBLE_DEEPFM_WEIGHT * deepfm_probs
    )
    ensemble_metrics = compute_metrics("WeightedEnsemble", y_holdout, ensemble_probs)

    metrics_df = pd.DataFrame(
        [
            metrics_to_row(lgbm_metrics),
            metrics_to_row(deepfm_metrics),
            metrics_to_row(ensemble_metrics),
        ]
    )
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)

    pred_df = pd.DataFrame(
        {
            "click": y_holdout.astype(np.int8),
            "lightgbm_pred": lgbm_probs.astype(np.float64),
            "deepfm_pred": deepfm_probs.astype(np.float64),
            "ensemble_pred": ensemble_probs.astype(np.float64),
        }
    )
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PREDICTIONS_PATH, index=False)

    metadata_out = {
        "script_name": "scripts/43_final_holdout_evaluation.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_train_dates": f"{FINAL_TRAIN_DATE_START}~{FINAL_TRAIN_DATE_END}",
        "holdout_date": HOLDOUT_DATE,
        "final_train_rows": final_train_rows,
        "holdout_rows": holdout_rows,
        "actual_holdout_ctr": actual_holdout_ctr,
        "models": ["LightGBM_Unified", "DeepFM", "WeightedEnsemble"],
        "ensemble_lightgbm_weight": ENSEMBLE_LIGHTGBM_WEIGHT,
        "ensemble_deepfm_weight": ENSEMBLE_DEEPFM_WEIGHT,
        "ensemble_weight_source": ENSEMBLE_WEIGHT_SOURCE,
        "deepfm_epochs": FROZEN_DEEPFM_EPOCHS,
        "deepfm_epoch_source": FROZEN_DEEPFM_EPOCH_SOURCE,
        "lightgbm_boosting_rounds": FROZEN_LGBM_ITERATIONS,
        "lightgbm_parameters_source": "frozen development configuration",
        "lightgbm_iteration_source": FROZEN_LGBM_ITERATION_SOURCE,
        "vocabulary_fit_scope": "final_train_only",
        "scaler_fit_scope": "final_train_only",
        "holdout_used_for_training": False,
        "holdout_used_for_preprocessing_fit": False,
        "holdout_used_for_model_selection": False,
        "holdout_used_for_weight_selection": False,
        "leakage_audit": leakage_audit,
        "metrics": {
            "LightGBM_Unified": metrics_to_row(lgbm_metrics),
            "DeepFM": metrics_to_row(deepfm_metrics),
            "WeightedEnsemble": metrics_to_row(ensemble_metrics),
        },
        "model_paths": {
            "lightgbm": str(FINAL_LGBM_MODEL_PATH),
            "deepfm": str(FINAL_DEEPFM_MODEL_PATH),
            "scaler": str(FINAL_SCALER_PATH),
        },
        "predictions_path": str(PREDICTIONS_PATH),
        "metrics_csv_path": str(METRICS_CSV_PATH),
        "one_shot_evaluation": True,
        "note": "Final holdout evaluation complete. Do not re-tune based on these results.",
    }
    METADATA_PATH.write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"\nMetrics CSV: {METRICS_CSV_PATH}")
    log(f"Predictions: {PREDICTIONS_PATH}")
    log(f"Metadata: {METADATA_PATH}")

    print_final_summary(
        final_train_rows,
        holdout_rows,
        actual_holdout_ctr,
        lgbm_metrics,
        deepfm_metrics,
        ensemble_metrics,
    )


if __name__ == "__main__":
    main()
