"""
百度 CTR 项目 — Wide & Deep 模型训练（第 40 步）

功能：
    使用 Step 39 已编码的 PyTorch 输入，训练第一版 Wide & Deep CTR 模型。
    支持 TEST_MODE smoke test 与正式 2M/500K 训练。

数据输入：
    data/modeling/pytorch_train/
    data/modeling/pytorch_valid/
    outputs/pytorch_input_metadata.json

用法：
    python scripts/40_train_wide_deep.py
"""

from __future__ import annotations

import csv
import gc
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

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
from torch.utils.data import DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

TEST_MODE = False

PYTORCH_METADATA_PATH = Path("outputs/pytorch_input_metadata.json")
PYTORCH_TRAIN_DIR = Path("data/modeling/pytorch_train")
PYTORCH_VALID_DIR = Path("data/modeling/pytorch_valid")

MODEL_PATH = Path("models/wide_deep_best.pt")
PREDICTIONS_PATH = Path("outputs/predictions/wide_deep_valid_predictions.parquet")
HISTORY_PATH = Path("outputs/wide_deep_training_history.csv")
METRICS_PATH = Path("outputs/wide_deep_metrics.json")
METADATA_PATH = Path("outputs/wide_deep_metadata.json")

FORMAL_TRAIN_ROWS = 2_000_000
FORMAL_VALID_ROWS = 500_000
TEST_TRAIN_ROWS = 200_000
TEST_VALID_ROWS = 100_000

RANDOM_SEED = 42
BATCH_SIZE = 4096
BATCH_SIZE_FALLBACKS = [2048, 1024]

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
FORMAL_MAX_EPOCHS = 10
TEST_MAX_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 2
PROB_CLIP_EPS = 1e-15
THRESHOLD = 0.5

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

LIGHTGBM_UNIFIED_AUC = 0.744509
LIGHTGBM_UNIFIED_LOGLOSS = 0.381338
LIGHTGBM_UNIFIED_BRIER = 0.117889

HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATES = [0.2, 0.2, 0.1]

READ_BATCH_SIZE = 4096


@dataclass
class BatchSpec:
    file_path: Path
    start_row: int
    num_rows: int


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    valid_auc: float
    valid_logloss: float
    valid_brier: float
    mean_predicted_ctr: float
    actual_ctr: float
    calibration_gap: float
    elapsed_seconds: float


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_row_limits(test_mode: bool) -> tuple[int, int]:
    if test_mode:
        return TEST_TRAIN_ROWS, TEST_VALID_ROWS
    return FORMAL_TRAIN_ROWS, FORMAL_VALID_ROWS


def get_max_epochs(test_mode: bool) -> int:
    return TEST_MAX_EPOCHS if test_mode else FORMAL_MAX_EPOCHS


def load_pytorch_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("validation_passed") is not True:
        raise ValueError("pytorch_input_metadata validation_passed 必须为 true。")
    if metadata.get("holdout_used") is not False:
        raise ValueError("pytorch_input_metadata holdout_used 必须为 false。")
    if metadata.get("train_click_match") is not True or metadata.get("valid_click_match") is not True:
        raise ValueError("Step 39 click 校验未通过，禁止训练。")

    excluded = {"device_id", "device_ip"}
    overlap = excluded & set(metadata["categorical_features"])
    if overlap:
        raise ValueError(f"categorical_features 含禁止字段：{sorted(overlap)}")

    return metadata


def get_feature_columns(metadata: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    categorical_features = metadata["categorical_features"]
    numerical_features = metadata["numerical_features"]
    cat_cols = [f"{col}_idx" for col in categorical_features]
    num_cols = [f"{col}_scaled" for col in numerical_features]
    read_cols = [*cat_cols, *num_cols, "click"]
    return categorical_features, numerical_features, cat_cols, num_cols, read_cols


def get_embedding_dim(vocab_size: int) -> int:
    if vocab_size <= 10:
        return 4
    if vocab_size <= 100:
        return 8
    if vocab_size <= 1000:
        return 12
    if vocab_size <= 10000:
        return 16
    return min(24, 32)


def build_batch_specs(parquet_dir: Path, max_rows: int | None, batch_size: int) -> list[BatchSpec]:
    assert_safe_path(parquet_dir)
    files = sorted(parquet_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件：{parquet_dir}")

    specs: list[BatchSpec] = []
    collected = 0
    for parquet_path in files:
        num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        if max_rows is not None:
            remaining = max_rows - collected
            if remaining <= 0:
                break
            num_rows = min(num_rows, remaining)

        for start_row in range(0, num_rows, batch_size):
            chunk_rows = min(batch_size, num_rows - start_row)
            specs.append(BatchSpec(parquet_path, start_row, chunk_rows))

        collected += num_rows
        if max_rows is not None and collected >= max_rows:
            break

    if not specs:
        raise ValueError(f"未生成任何 batch spec：{parquet_dir}")
    return specs


def read_batch_from_spec(
    spec: BatchSpec,
    read_cols: list[str],
) -> pd.DataFrame:
    table = pq.read_table(spec.file_path, columns=read_cols)
    return table.slice(spec.start_row, spec.num_rows).to_pandas()


def validate_batch(
    batch_df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    categorical_features: list[str],
    vocab_sizes: dict[str, int],
) -> None:
    clicks = batch_df["click"].to_numpy()
    if not np.isin(clicks, [0, 1]).all():
        raise ValueError("click 列存在非 0/1 值。")

    numerical = batch_df[num_cols].to_numpy(dtype=np.float64)
    if np.isnan(numerical).any():
        raise ValueError("numerical 特征存在 NaN。")
    if np.isinf(numerical).any():
        raise ValueError("numerical 特征存在 inf。")

    for col, feature in zip(cat_cols, categorical_features):
        values = batch_df[col].to_numpy(dtype=np.int64)
        vocab_size = vocab_sizes[feature]
        if (values < 0).any():
            raise ValueError(f"{feature} 存在负 index。")
        if (values >= vocab_size).any():
            bad = values[values >= vocab_size]
            raise ValueError(
                f"{feature} Embedding index 越界：max={bad.max()}, vocab_size={vocab_size}"
            )


def dataframe_to_tensors(
    batch_df: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical = torch.from_numpy(
        batch_df[cat_cols].to_numpy(dtype=np.int64, copy=True)
    )
    numerical = torch.from_numpy(
        batch_df[num_cols].to_numpy(dtype=np.float32, copy=True)
    )
    click = torch.from_numpy(
        batch_df["click"].to_numpy(dtype=np.float32, copy=True)
    )
    return categorical, numerical, click


def validate_dataset(
    parquet_dir: Path,
    read_cols: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    categorical_features: list[str],
    vocab_sizes: dict[str, int],
    max_rows: int | None,
    split_name: str,
) -> int:
    specs = build_batch_specs(parquet_dir, max_rows, READ_BATCH_SIZE)
    total_rows = 0
    for spec in specs:
        batch_df = read_batch_from_spec(spec, read_cols)
        validate_batch(batch_df, cat_cols, num_cols, categorical_features, vocab_sizes)
        total_rows += len(batch_df)
    print(f"[validate] {split_name}: {total_rows:,} 行通过安全检查。")
    return total_rows


class ParquetCTRIterableDataset(IterableDataset):
    """流式读取 PyTorch parquet，按 batch 产出张量。"""

    def __init__(
        self,
        parquet_dir: Path,
        read_cols: list[str],
        cat_cols: list[str],
        num_cols: list[str],
        categorical_features: list[str],
        vocab_sizes: dict[str, int],
        max_rows: int | None,
        batch_size: int,
        shuffle: bool,
        epoch: int,
        seed: int,
    ) -> None:
        self.parquet_dir = parquet_dir
        self.read_cols = read_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.categorical_features = categorical_features
        self.vocab_sizes = vocab_sizes
        self.max_rows = max_rows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = epoch
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        specs = build_batch_specs(self.parquet_dir, self.max_rows, self.batch_size)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(specs)

        for spec in specs:
            batch_df = read_batch_from_spec(spec, self.read_cols)
            validate_batch(
                batch_df,
                self.cat_cols,
                self.num_cols,
                self.categorical_features,
                self.vocab_sizes,
            )
            yield dataframe_to_tensors(batch_df, self.cat_cols, self.num_cols)


class WideAndDeep(nn.Module):
    """Wide & Deep CTR 模型。"""

    def __init__(
        self,
        categorical_features: list[str],
        vocab_sizes: dict[str, int],
        num_numerical_features: int,
        embedding_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.categorical_features = categorical_features
        self.num_numerical_features = num_numerical_features

        self.numerical_wide = nn.Linear(num_numerical_features, 1)

        self.wide_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_sizes[feature], 1)
                for feature in categorical_features
            }
        )
        self.deep_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_sizes[feature], embedding_dims[feature])
                for feature in categorical_features
            }
        )

        deep_input_dim = num_numerical_features + sum(
            embedding_dims[feature] for feature in categorical_features
        )
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_input_dim, HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[0]),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[1]),
            nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[2]),
            nn.Linear(HIDDEN_LAYERS[2], 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
    ) -> torch.Tensor:
        wide_logit = self.numerical_wide(numerical).squeeze(-1)

        for index, feature in enumerate(self.categorical_features):
            cat_index = categorical[:, index]
            wide_logit = wide_logit + self.wide_embeddings[feature](cat_index).squeeze(-1)

        deep_parts = [numerical]
        for index, feature in enumerate(self.categorical_features):
            cat_index = categorical[:, index]
            deep_parts.append(self.deep_embeddings[feature](cat_index))

        deep_input = torch.cat(deep_parts, dim=1)
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)
        return wide_logit + deep_logit


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, PROB_CLIP_EPS, 1.0 - PROB_CLIP_EPS)


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    clipped = clip_probabilities(probabilities)
    predicted_labels = (probabilities >= THRESHOLD).astype(np.int8)
    actual_ctr = float(y_true.mean())
    mean_predicted_ctr = float(probabilities.mean())

    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "logloss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "actual_ctr": actual_ctr,
        "mean_predicted_ctr": mean_predicted_ctr,
        "calibration_gap": abs(mean_predicted_ctr - actual_ctr),
        "precision_0.5": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall_0.5": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1_0.5": float(f1_score(y_true, predicted_labels, zero_division=0)),
    }


def train_one_epoch(
    model: WideAndDeep,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
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
        optimizer.step()

        batch_rows = click.shape[0]
        total_loss += float(loss.item()) * batch_rows
        total_rows += batch_rows

    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate_split(
    model: WideAndDeep,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_clicks: list[np.ndarray] = []

    for categorical, numerical, click in loader:
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        logits = model(categorical, numerical)
        all_logits.append(logits.detach().cpu().numpy())
        all_clicks.append(click.numpy())

    logits_array = np.concatenate(all_logits)
    clicks_array = np.concatenate(all_clicks).astype(np.int8)
    metrics = compute_metrics(clicks_array, logits_array)
    return metrics, clicks_array, logits_array


def create_optimizer(
    model: nn.Module,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name = "AdamW"
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer, optimizer_name
    except Exception as exc:
        if device.type != "mps":
            raise
        print(f"AdamW 初始化失败，回退 Adam：{exc}")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer, "Adam"


def save_training_history(history: list[EpochMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "valid_auc",
                "valid_logloss",
                "valid_brier",
                "mean_predicted_ctr",
                "actual_ctr",
                "calibration_gap",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "valid_auc": row.valid_auc,
                    "valid_logloss": row.valid_logloss,
                    "valid_brier": row.valid_brier,
                    "mean_predicted_ctr": row.mean_predicted_ctr,
                    "actual_ctr": row.actual_ctr,
                    "calibration_gap": row.calibration_gap,
                    "elapsed_seconds": row.elapsed_seconds,
                }
            )


def save_predictions(clicks: np.ndarray, logits: np.ndarray, path: Path) -> None:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "click": clicks.astype(np.int8),
            "wide_deep_pred": probabilities.astype(np.float64),
        }
    ).to_parquet(path, index=False)


def save_model_checkpoint(
    path: Path,
    model: WideAndDeep,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "WideAndDeep",
            "metadata": metadata,
        },
        path,
    )


def print_epoch_metrics(epoch_metrics: EpochMetrics) -> None:
    print(f"Epoch {epoch_metrics.epoch}")
    print(f"Train Loss = {epoch_metrics.train_loss:.6f}")
    print(f"Valid AUC = {epoch_metrics.valid_auc:.6f}")
    print(f"Valid LogLoss = {epoch_metrics.valid_logloss:.6f}")
    print(f"Valid Brier = {epoch_metrics.valid_brier:.6f}")
    print(f"Mean predicted CTR = {epoch_metrics.mean_predicted_ctr:.6f}")
    print(f"Actual CTR = {epoch_metrics.actual_ctr:.6f}")
    print(f"Calibration Gap = {epoch_metrics.calibration_gap:.6f}")
    print(f"Elapsed Time = {epoch_metrics.elapsed_seconds:.1f}s")


def print_final_summary(
    train_rows: int,
    valid_rows: int,
    device_name: str,
    best_epoch: int,
    metrics: dict[str, float],
    test_mode: bool,
) -> None:
    print("\n" + "=" * 40)
    print("WIDE & DEEP TRAINING SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_rows}")
    print(f"VALID_ROWS = {valid_rows}")
    print(f"DEVICE = {device_name}")
    print(f"BEST_EPOCH = {best_epoch}")
    print(f"AUC = {metrics['auc']:.6f}")
    print(f"LOGLOSS = {metrics['logloss']:.6f}")
    print(f"BRIER = {metrics['brier']:.6f}")
    print(f"AVERAGE_PRECISION = {metrics['average_precision']:.6f}")
    print(f"ACTUAL_CTR = {metrics['actual_ctr']:.6f}")
    print(f"MEAN_PREDICTED_CTR = {metrics['mean_predicted_ctr']:.6f}")
    print(f"CALIBRATION_GAP = {metrics['calibration_gap']:.6f}")
    print(f"PRECISION_0.5 = {metrics['precision_0.5']:.6f}")
    print(f"RECALL_0.5 = {metrics['recall_0.5']:.6f}")
    print(f"F1_0.5 = {metrics['f1_0.5']:.6f}")
    print(f"LIGHTGBM_UNIFIED_AUC = {LIGHTGBM_UNIFIED_AUC:.6f}")
    print(f"LIGHTGBM_UNIFIED_LOGLOSS = {LIGHTGBM_UNIFIED_LOGLOSS:.6f}")
    print(f"LIGHTGBM_UNIFIED_BRIER = {LIGHTGBM_UNIFIED_BRIER:.6f}")
    print(f"AUC_DIFF_VS_LIGHTGBM = {metrics['auc'] - LIGHTGBM_UNIFIED_AUC:+.6f}")
    print(f"LOGLOSS_DIFF_VS_LIGHTGBM = {metrics['logloss'] - LIGHTGBM_UNIFIED_LOGLOSS:+.6f}")
    print(f"BRIER_DIFF_VS_LIGHTGBM = {metrics['brier'] - LIGHTGBM_UNIFIED_BRIER:+.6f}")
    print("HOLDOUT_USED = False")
    print(f"VALIDATION_PASSED = True")
    if test_mode:
        print("TEST_MODE = True (smoke test，非正式结果)")
    print("=" * 40)


def run_training_with_batch_size(
    batch_size: int,
    device: torch.device,
    metadata: dict[str, Any],
    categorical_features: list[str],
    numerical_features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    read_cols: list[str],
    vocab_sizes: dict[str, int],
    embedding_dims: dict[str, int],
    train_rows: int,
    valid_rows: int,
    max_epochs: int,
    test_mode: bool,
) -> tuple[
    WideAndDeep,
    list[EpochMetrics],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    int,
    str,
]:
    model = WideAndDeep(
        categorical_features=categorical_features,
        vocab_sizes=vocab_sizes,
        num_numerical_features=len(numerical_features),
        embedding_dims=embedding_dims,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer, optimizer_name = create_optimizer(model, device)

    history: list[EpochMetrics] = []
    best_logloss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    best_metrics: dict[str, float] = {}
    best_clicks = np.array([], dtype=np.int8)
    best_logits = np.array([], dtype=np.float64)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        train_dataset = ParquetCTRIterableDataset(
            parquet_dir=PYTORCH_TRAIN_DIR,
            read_cols=read_cols,
            cat_cols=cat_cols,
            num_cols=num_cols,
            categorical_features=categorical_features,
            vocab_sizes=vocab_sizes,
            max_rows=train_rows,
            batch_size=batch_size,
            shuffle=True,
            epoch=epoch,
            seed=RANDOM_SEED,
        )
        train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)

        valid_dataset = ParquetCTRIterableDataset(
            parquet_dir=PYTORCH_VALID_DIR,
            read_cols=read_cols,
            cat_cols=cat_cols,
            num_cols=num_cols,
            categorical_features=categorical_features,
            vocab_sizes=vocab_sizes,
            max_rows=valid_rows,
            batch_size=batch_size,
            shuffle=False,
            epoch=epoch,
            seed=RANDOM_SEED,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=None, num_workers=0)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_metrics, valid_clicks, valid_logits = evaluate_split(model, valid_loader, device)

        elapsed = time.time() - epoch_start
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            valid_auc=valid_metrics["auc"],
            valid_logloss=valid_metrics["logloss"],
            valid_brier=valid_metrics["brier"],
            mean_predicted_ctr=valid_metrics["mean_predicted_ctr"],
            actual_ctr=valid_metrics["actual_ctr"],
            calibration_gap=valid_metrics["calibration_gap"],
            elapsed_seconds=elapsed,
        )
        history.append(epoch_metrics)
        print_epoch_metrics(epoch_metrics)

        if valid_metrics["logloss"] < best_logloss:
            best_logloss = valid_metrics["logloss"]
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = valid_metrics
            best_clicks = valid_clicks
            best_logits = valid_logits
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if best_state_dict is None:
        raise RuntimeError("训练未产生有效 checkpoint。")

    model.load_state_dict(best_state_dict)
    return (
        model,
        history,
        best_metrics,
        best_clicks,
        best_logits,
        best_epoch,
        optimizer_name,
    )


def main() -> None:
    state = BuildState()
    train_limit, valid_limit = get_row_limits(TEST_MODE)
    max_epochs = get_max_epochs(TEST_MODE)

    log(state, "=" * 72)
    log(state, "百度 CTR 项目 — 第 40 步 Wide & Deep 训练")
    log(state, f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    log(state, f"TEST_MODE = {TEST_MODE}")

    assert_safe_path(PYTORCH_TRAIN_DIR)
    assert_safe_path(PYTORCH_VALID_DIR)

    metadata = load_pytorch_metadata(PYTORCH_METADATA_PATH)
    categorical_features, numerical_features, cat_cols, num_cols, read_cols = get_feature_columns(
        metadata
    )
    vocab_sizes: dict[str, int] = metadata["vocab_sizes"]
    embedding_dims = {feature: get_embedding_dim(vocab_sizes[feature]) for feature in categorical_features}

    device = get_device()
    log(state, f"DEVICE = {device.type}")

    set_global_seed(RANDOM_SEED)

    log(state, "\nStep 1：训练前数据安全检查 ...")
    validated_train_rows = validate_dataset(
        PYTORCH_TRAIN_DIR,
        read_cols,
        cat_cols,
        num_cols,
        categorical_features,
        vocab_sizes,
        train_limit,
        "train",
    )
    validated_valid_rows = validate_dataset(
        PYTORCH_VALID_DIR,
        read_cols,
        cat_cols,
        num_cols,
        categorical_features,
        vocab_sizes,
        valid_limit,
        "valid",
    )
    log(state, f"TRAIN_ROWS = {validated_train_rows:,}")
    log(state, f"VALID_ROWS = {validated_valid_rows:,}")

    log(state, f"\nStep 2：构建 Wide & Deep（categorical={len(categorical_features)}, numerical={len(numerical_features)})")
    log(state, f"EMBEDDING_DIMS = {embedding_dims}")

    batch_size = BATCH_SIZE
    training_error: Exception | None = None
    model = None
    history: list[EpochMetrics] = []
    best_metrics: dict[str, float] = {}
    best_clicks = np.array([], dtype=np.int8)
    best_logits = np.array([], dtype=np.float64)
    best_epoch = 0
    optimizer_name = "AdamW"

    for candidate_batch_size in [batch_size, *BATCH_SIZE_FALLBACKS]:
        if candidate_batch_size > batch_size:
            continue
        try:
            log(state, f"\nStep 3：开始训练（batch_size={candidate_batch_size}, max_epochs={max_epochs}) ...")
            (
                model,
                history,
                best_metrics,
                best_clicks,
                best_logits,
                best_epoch,
                optimizer_name,
            ) = run_training_with_batch_size(
                batch_size=candidate_batch_size,
                device=device,
                metadata=metadata,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                cat_cols=cat_cols,
                num_cols=num_cols,
                read_cols=read_cols,
                vocab_sizes=vocab_sizes,
                embedding_dims=embedding_dims,
                train_rows=train_limit,
                valid_rows=valid_limit,
                max_epochs=max_epochs,
                test_mode=TEST_MODE,
            )
            batch_size = candidate_batch_size
            training_error = None
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            if device.type == "mps" and "out of memory" in message and candidate_batch_size != BATCH_SIZE_FALLBACKS[-1]:
                print(f"MPS OOM at batch_size={candidate_batch_size}，尝试更小 batch ...")
                training_error = exc
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                continue
            raise

    if training_error is not None or model is None:
        raise training_error or RuntimeError("训练失败。")

    checkpoint_metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "vocab_sizes": vocab_sizes,
        "embedding_dims": embedding_dims,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT_RATES,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "random_seed": RANDOM_SEED,
        "test_mode": TEST_MODE,
    }
    save_model_checkpoint(MODEL_PATH, model, checkpoint_metadata)
    save_predictions(best_clicks, best_logits, PREDICTIONS_PATH)
    save_training_history(history, HISTORY_PATH)

    metrics_out = {
        **best_metrics,
        "best_epoch": best_epoch,
        "train_rows": validated_train_rows,
        "valid_rows": validated_valid_rows,
        "test_mode": TEST_MODE,
        "holdout_used": False,
        "lightgbm_unified_auc": LIGHTGBM_UNIFIED_AUC,
        "lightgbm_unified_logloss": LIGHTGBM_UNIFIED_LOGLOSS,
        "lightgbm_unified_brier": LIGHTGBM_UNIFIED_BRIER,
        "auc_diff_vs_lightgbm": best_metrics["auc"] - LIGHTGBM_UNIFIED_AUC,
        "logloss_diff_vs_lightgbm": best_metrics["logloss"] - LIGHTGBM_UNIFIED_LOGLOSS,
        "brier_diff_vs_lightgbm": best_metrics["brier"] - LIGHTGBM_UNIFIED_BRIER,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata_out = {
        "script_name": "scripts/40_train_wide_deep.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "WideAndDeep",
        "train_rows": validated_train_rows,
        "valid_rows": validated_valid_rows,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "vocab_sizes": vocab_sizes,
        "embedding_dims": embedding_dims,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout": DROPOUT_RATES,
        "optimizer": optimizer_name,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "device": device.type,
        "best_valid_auc": best_metrics["auc"],
        "best_valid_logloss": best_metrics["logloss"],
        "best_valid_brier": best_metrics["brier"],
        "average_precision": best_metrics["average_precision"],
        "actual_ctr": best_metrics["actual_ctr"],
        "mean_predicted_ctr": best_metrics["mean_predicted_ctr"],
        "calibration_gap": best_metrics["calibration_gap"],
        "precision_0.5": best_metrics["precision_0.5"],
        "recall_0.5": best_metrics["recall_0.5"],
        "f1_0.5": best_metrics["f1_0.5"],
        "random_seed": RANDOM_SEED,
        "test_mode": TEST_MODE,
        "holdout_used": False,
        "validation_passed": True,
        "model_path": str(MODEL_PATH),
        "predictions_path": str(PREDICTIONS_PATH),
        "history_path": str(HISTORY_PATH),
        "metrics_path": str(METRICS_PATH),
        "pytorch_input_metadata_path": str(PYTORCH_METADATA_PATH),
        "lightgbm_unified_reference": {
            "auc": LIGHTGBM_UNIFIED_AUC,
            "logloss": LIGHTGBM_UNIFIED_LOGLOSS,
            "brier": LIGHTGBM_UNIFIED_BRIER,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(state, f"\n模型：{MODEL_PATH}")
    log(state, f"预测：{PREDICTIONS_PATH}")
    log(state, f"Metadata：{METADATA_PATH}")

    print_final_summary(
        validated_train_rows,
        validated_valid_rows,
        device.type,
        best_epoch,
        best_metrics,
        TEST_MODE,
    )


if __name__ == "__main__":
    main()
