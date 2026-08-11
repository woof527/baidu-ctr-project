"""
百度 CTR 项目 — DeepFM 模型训练（第 41 步）

功能：
    使用 Step 39 已编码的 PyTorch 输入，训练 DeepFM CTR 模型。
    复用 Step 40 的数据读取、MPS device、metrics、early stopping 等稳定逻辑。

数据输入：
    data/modeling/pytorch_train/
    data/modeling/pytorch_valid/
    outputs/pytorch_input_metadata.json

用法：
    python scripts/41_train_deepfm.py
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

MODEL_PATH = Path("models/deepfm_best.pt")
PREDICTIONS_PATH = Path("outputs/predictions/deepfm_valid_predictions.parquet")
HISTORY_PATH = Path("outputs/deepfm_training_history.csv")
METRICS_PATH = Path("outputs/deepfm_metrics.json")
METADATA_PATH = Path("outputs/deepfm_metadata.json")

FORMAL_TRAIN_ROWS = 2_000_000
FORMAL_VALID_ROWS = 500_000
TEST_TRAIN_ROWS = 200_000
TEST_VALID_ROWS = 100_000

RANDOM_SEED = 42
BATCH_SIZE = 4096
BATCH_SIZE_FALLBACKS = [2048, 1024]

LEARNING_RATE = 1e-3
FM_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
FM_EMBEDDING_INIT_STD = 0.001
GRADIENT_CLIP_NORM = 5.0
FORMAL_MAX_EPOCHS = 10
TEST_MAX_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 2
PROB_CLIP_EPS = 1e-15
THRESHOLD = 0.5

FM_EMBEDDING_DIM = 16
FM_INTERACTION_SCOPE = "categorical_features_only"

FORBIDDEN_PATH_KEYWORDS = ("holdout", "test.csv")

LIGHTGBM_AUC = 0.744509
LIGHTGBM_LOGLOSS = 0.381338
LIGHTGBM_BRIER = 0.117889

WIDE_DEEP_AUC = 0.742750
WIDE_DEEP_LOGLOSS = 0.383845
WIDE_DEEP_BRIER = 0.118432

DEEP_HIDDEN_LAYERS = [256, 128, 64]
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
    average_precision: float
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


def read_batch_from_spec(spec: BatchSpec, read_cols: list[str]) -> pd.DataFrame:
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


class DeepFM(nn.Module):
    """DeepFM CTR 模型：First-order + FM Second-order + Deep MLP。"""

    def __init__(
        self,
        categorical_features: list[str],
        vocab_sizes: dict[str, int],
        num_numerical_features: int,
        global_bias_init: float,
        fm_embedding_dim: int = FM_EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.categorical_features = categorical_features
        self.num_categorical_fields = len(categorical_features)
        self.num_numerical_features = num_numerical_features
        self.fm_embedding_dim = fm_embedding_dim
        self.global_bias_init = global_bias_init

        self.global_bias = nn.Parameter(torch.zeros(1))
        self.numerical_first_order = nn.Linear(num_numerical_features, 1, bias=False)

        self.first_order_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_sizes[feature], 1)
                for feature in categorical_features
            }
        )
        self.fm_embeddings = nn.ModuleDict(
            {
                feature: nn.Embedding(vocab_sizes[feature], fm_embedding_dim)
                for feature in categorical_features
            }
        )

        deep_input_dim = self.num_categorical_fields * fm_embedding_dim + num_numerical_features
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_input_dim, DEEP_HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[0]),
            nn.Linear(DEEP_HIDDEN_LAYERS[0], DEEP_HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[1]),
            nn.Linear(DEEP_HIDDEN_LAYERS[1], DEEP_HIDDEN_LAYERS[2]),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATES[2]),
            nn.Linear(DEEP_HIDDEN_LAYERS[2], 1),
        )

        self._init_weights(global_bias_init)

    def _init_weights(self, global_bias_init: float) -> None:
        nn.init.constant_(self.global_bias, global_bias_init)
        nn.init.zeros_(self.numerical_first_order.weight)

        for embedding in self.first_order_embeddings.values():
            nn.init.zeros_(embedding.weight)

        for embedding in self.fm_embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=FM_EMBEDDING_INIT_STD)

        linear_layers = [module for module in self.deep_mlp if isinstance(module, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        output_layer = linear_layers[-1]
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    def _first_order_categorical(self, categorical: torch.Tensor) -> torch.Tensor:
        logit = torch.zeros(categorical.shape[0], device=categorical.device, dtype=torch.float32)
        for index, feature in enumerate(self.categorical_features):
            logit = logit + self.first_order_embeddings[feature](categorical[:, index]).squeeze(-1)
        return logit

    def _get_fm_embeddings(self, categorical: torch.Tensor) -> torch.Tensor:
        embed_list = [
            self.fm_embeddings[feature](categorical[:, index])
            for index, feature in enumerate(self.categorical_features)
        ]
        return torch.stack(embed_list, dim=1)

    @staticmethod
    def _fm_second_order(embeddings: torch.Tensor) -> torch.Tensor:
        """向量化 FM 二阶项：0.5 * sum((sum E)^2 - sum(E^2), dim=embed_dim)。"""
        sum_embeddings = torch.sum(embeddings, dim=1)
        sum_square = sum_embeddings.pow(2)
        square_sum = torch.sum(embeddings.pow(2), dim=1)
        return 0.5 * torch.sum(sum_square - square_sum, dim=1)

    def forward(self, categorical: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        first_order_numerical = self.numerical_first_order(numerical).squeeze(-1)
        first_order_categorical = self._first_order_categorical(categorical)
        first_order_logit = first_order_categorical + first_order_numerical

        fm_embed = self._get_fm_embeddings(categorical)
        expected_shape = (categorical.shape[0], self.num_categorical_fields, self.fm_embedding_dim)
        if fm_embed.shape != expected_shape:
            raise ValueError(
                f"FM tensor shape 异常：got {tuple(fm_embed.shape)}, expected {expected_shape}"
            )

        fm_second_order_logit = self._fm_second_order(fm_embed)

        deep_input = torch.cat([fm_embed.flatten(start_dim=1), numerical], dim=1)
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)

        return self.global_bias.view(()) + first_order_logit + fm_second_order_logit + deep_logit

    @torch.no_grad()
    def forward_components(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """诊断用：分解输出各 logit 分量（first_order 不含 global_bias）。"""
        first_order_numerical = self.numerical_first_order(numerical).squeeze(-1)
        first_order_categorical = self._first_order_categorical(categorical)
        first_order_logit = first_order_categorical + first_order_numerical

        fm_embed = self._get_fm_embeddings(categorical)
        fm_second_order_logit = self._fm_second_order(fm_embed)

        deep_input = torch.cat([fm_embed.flatten(start_dim=1), numerical], dim=1)
        deep_logit = self.deep_mlp(deep_input).squeeze(-1)

        final_logit = self.global_bias.view(()) + first_order_logit + fm_second_order_logit + deep_logit
        return {
            "global_bias_logit": torch.full_like(final_logit, self.global_bias.item()),
            "first_order_logit": first_order_logit,
            "fm_second_order_logit": fm_second_order_logit,
            "deep_logit": deep_logit,
            "final_logit": final_logit,
        }


def compute_array_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def compute_train_ctr(parquet_dir: Path, max_rows: int | None) -> float:
    """仅根据 train 数据计算 CTR，禁止读取 valid / holdout。"""
    assert_safe_path(parquet_dir)
    specs = build_batch_specs(parquet_dir, max_rows, READ_BATCH_SIZE)
    total_clicks = 0
    total_rows = 0
    for spec in specs:
        batch_df = read_batch_from_spec(spec, ["click"])
        clicks = batch_df["click"].to_numpy()
        total_clicks += int(clicks.sum())
        total_rows += len(clicks)
    if total_rows == 0:
        raise ValueError("train CTR 计算失败：无有效行。")
    return total_clicks / total_rows


def compute_global_bias_init(train_ctr: float) -> float:
    clipped_ctr = min(max(train_ctr, PROB_CLIP_EPS), 1.0 - PROB_CLIP_EPS)
    return float(np.log(clipped_ctr / (1.0 - clipped_ctr)))


def format_stats_line(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name} mean={stats['mean']:.6f} std={stats['std']:.6f} "
        f"min={stats['min']:.6f} max={stats['max']:.6f}"
    )


def print_component_diagnostics(stats_map: dict[str, dict[str, float]], header: str) -> None:
    print(f"\n{header}")
    for key in (
        "first_order_logit",
        "fm_second_order_logit",
        "deep_logit",
        "final_logit",
        "pred_prob",
    ):
        print(format_stats_line(key.upper(), stats_map[key]))


def print_initial_diagnostics(stats_map: dict[str, dict[str, float]], train_ctr: float) -> None:
    print("\n--- INITIAL Diagnostics (before training) ---")
    print(f"TRAIN_CTR = {train_ctr:.6f}")
    for key in (
        "first_order_logit",
        "fm_second_order_logit",
        "deep_logit",
        "final_logit",
        "pred_prob",
    ):
        label = f"INITIAL_{key.upper()}"
        print(format_stats_line(label, stats_map[key]))


def diagnose_initialization(model: DeepFM) -> None:
    """检查初始化：FM embedding std、global bias、first-order / deep output layer。"""
    print("\n--- Initialization Diagnostics ---")

    fm_stds: list[float] = []
    for feature, embedding in model.fm_embeddings.items():
        weight_std = float(embedding.weight.std().item())
        fm_stds.append(weight_std)
        print(f"  fm_embeddings[{feature}].weight std = {weight_std:.6f}")

    print(
        f"FM embedding init std summary: "
        f"mean={np.mean(fm_stds):.6f}, min={np.min(fm_stds):.6f}, max={np.max(fm_stds):.6f}"
    )
    print(
        f"Expected init std ≈ {FM_EMBEDDING_INIT_STD:.6f} "
        f"(normal_, mean=0, std={FM_EMBEDDING_INIT_STD})"
    )

    global_bias = float(model.global_bias.item())
    print(f"global_bias init value = {global_bias:.6f} (train logit prior)")

    first_order_cat_max = max(
        float(embedding.weight.abs().max().item())
        for embedding in model.first_order_embeddings.values()
    )
    print(f"first_order_embeddings max abs weight = {first_order_cat_max:.6f} (expected 0.0)")

    num_weight = model.numerical_first_order.weight.detach().cpu().numpy().reshape(-1)
    print(
        f"numerical_first_order.weight abs_max = {np.abs(num_weight).max():.6f} (expected 0.0)"
    )

    linear_layers = [module for module in model.deep_mlp if isinstance(module, nn.Linear)]
    output_layer = linear_layers[-1]
    out_weight_max = float(output_layer.weight.abs().max().item())
    out_bias = float(output_layer.bias.item()) if output_layer.bias is not None else 0.0
    print(f"deep_mlp output layer weight abs_max = {out_weight_max:.6f} (expected 0.0)")
    print(f"deep_mlp output layer bias = {out_bias:.6f} (expected 0.0)")


def verify_fm_formula(model: DeepFM, device: torch.device) -> None:
    """验证 FM 公式、输出 shape、无重复累加。"""
    print("\n--- FM Formula Verification ---")

    batch_size = 64
    num_fields = model.num_categorical_fields
    embed_dim = model.fm_embedding_dim

    torch.manual_seed(RANDOM_SEED)
    embeddings = torch.randn(batch_size, num_fields, embed_dim, device=device) * 0.01

    vectorized = DeepFM._fm_second_order(embeddings)
    if vectorized.shape != (batch_size,):
        raise ValueError(f"FM 输出 shape 异常：{tuple(vectorized.shape)}，期望 ({batch_size},)")

    explicit = torch.zeros(batch_size, device=device)
    for dim in range(embed_dim):
        field_sum = embeddings[:, :, dim].sum(dim=1)
        field_square_sum = (embeddings[:, :, dim] ** 2).sum(dim=1)
        explicit = explicit + 0.5 * (field_sum.pow(2) - field_square_sum)

    max_diff = float((vectorized - explicit).abs().max().item())
    print(f"Vectorized vs explicit pairwise FM max abs diff = {max_diff:.2e}")
    if max_diff > 1e-5:
        raise ValueError("FM 向量化公式与显式 pairwise 公式不一致。")
    print("FM formula OK: 0.5 * sum_dim( (sum_fields E)^2 - sum_fields(E^2) ) -> scalar per sample")

    wrong_extra_dim = 0.5 * torch.sum(
        torch.sum(embeddings, dim=1).pow(2) - torch.sum(embeddings.pow(2), dim=1),
        dim=1,
    ) * embed_dim
    extra_scale_diff = float((vectorized - wrong_extra_dim).abs().mean().item())
    print(f"Mean abs diff vs wrongly multiplied by embedding_dim = {extra_scale_diff:.6f}")
    print("No extra embedding_dim multiplier detected in FM term.")

    categorical = torch.randint(0, 2, (batch_size, num_fields), device=device)
    numerical = torch.randn(batch_size, model.num_numerical_features, device=device)

    model.eval()
    components = model.forward_components(categorical, numerical)
    reconstructed = (
        components["global_bias_logit"]
        + components["first_order_logit"]
        + components["fm_second_order_logit"]
        + components["deep_logit"]
    )
    forward_out = model(categorical, numerical)

    recon_diff = float((reconstructed - components["final_logit"]).abs().max().item())
    forward_diff = float((components["final_logit"] - forward_out).abs().max().item())
    print(f"Component sum vs final_logit max abs diff = {recon_diff:.2e}")
    print(f"forward_components vs forward max abs diff = {forward_diff:.2e}")
    if recon_diff > 1e-5 or forward_diff > 1e-5:
        raise ValueError("检测到 FM / final logit 重复累加或 forward 不一致。")
    print(
        "No duplicate accumulation: final = global_bias + first_order + fm_second + deep (once each)."
    )


@torch.no_grad()
def collect_component_diagnostics(
    model: DeepFM,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """在 validation forward 时收集各 logit 分量统计。"""
    model.eval()

    buckets: dict[str, list[np.ndarray]] = {
        "first_order_logit": [],
        "fm_second_order_logit": [],
        "deep_logit": [],
        "final_logit": [],
        "pred_prob": [],
    }

    for categorical, numerical, _click in loader:
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        components = model.forward_components(categorical, numerical)

        for key in ("first_order_logit", "fm_second_order_logit", "deep_logit", "final_logit"):
            buckets[key].append(components[key].detach().cpu().numpy())

        final_logit = components["final_logit"]
        pred_prob = torch.sigmoid(final_logit)
        buckets["pred_prob"].append(pred_prob.detach().cpu().numpy())

    return {
        key: compute_array_stats(np.concatenate(values))
        for key, values in buckets.items()
    }


def validate_model_forward(
    model: DeepFM,
    device: torch.device,
    categorical_features: list[str],
) -> None:
    """Smoke test：验证 forward / FM tensor shape / backward。"""
    batch_size = 32
    num_cat = len(categorical_features)
    num_num = model.num_numerical_features

    categorical = torch.zeros(batch_size, num_cat, dtype=torch.long, device=device)
    numerical = torch.zeros(batch_size, num_num, dtype=torch.float32, device=device)

    model.train()
    logits = model(categorical, numerical)
    if logits.shape != (batch_size,):
        raise ValueError(f"forward logits shape 异常：{tuple(logits.shape)}")

    loss = logits.sum()
    loss.backward()
    print(
        f"[smoke] FM tensor shape OK: [batch={batch_size}, "
        f"fields={num_cat}, dim={FM_EMBEDDING_DIM}]"
    )


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
    model: DeepFM,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        batch_rows = click.shape[0]
        total_loss += float(loss.item()) * batch_rows
        total_rows += batch_rows

    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate_split(
    model: DeepFM,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, dict[str, dict[str, float]] | None]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_clicks: list[np.ndarray] = []

    component_buckets: dict[str, list[np.ndarray]] = {
        "first_order_logit": [],
        "fm_second_order_logit": [],
        "deep_logit": [],
        "final_logit": [],
        "pred_prob": [],
    }

    for categorical, numerical, click in loader:
        categorical = categorical.to(device)
        numerical = numerical.to(device)

        components = model.forward_components(categorical, numerical)
        logits = components["final_logit"]

        for key in ("first_order_logit", "fm_second_order_logit", "deep_logit", "final_logit"):
            component_buckets[key].append(components[key].detach().cpu().numpy())
        component_buckets["pred_prob"].append(torch.sigmoid(logits).detach().cpu().numpy())

        all_logits.append(logits.detach().cpu().numpy())
        all_clicks.append(click.numpy())

    logits_array = np.concatenate(all_logits)
    clicks_array = np.concatenate(all_clicks).astype(np.int8)
    metrics = compute_metrics(clicks_array, logits_array)

    component_stats = {
        key: compute_array_stats(np.concatenate(values))
        for key, values in component_buckets.items()
    }

    if epoch is not None:
        print_component_diagnostics(
            component_stats,
            header=f"--- Validation Component Diagnostics (Epoch {epoch}) ---",
        )
        print(format_stats_line("SIGMOID(FINAL_LOGIT)", component_stats["pred_prob"]))

    return metrics, clicks_array, logits_array, component_stats


def create_optimizer(model: DeepFM, device: torch.device) -> tuple[torch.optim.Optimizer, str]:
    """AdamW：FM embeddings 使用较低学习率，其余参数使用默认学习率。"""
    fm_params = list(model.fm_embeddings.parameters())
    fm_param_ids = {id(param) for param in fm_params}
    other_params = [param for param in model.parameters() if id(param) not in fm_param_ids]

    if len(fm_params) + len(other_params) != sum(1 for _ in model.parameters()):
        raise ValueError("FM embedding 参数与其他参数存在重复注册。")

    optimizer_name = "AdamW"
    param_groups = [
        {"params": fm_params, "lr": FM_LEARNING_RATE},
        {"params": other_params, "lr": LEARNING_RATE},
    ]
    try:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        return optimizer, optimizer_name
    except Exception as exc:
        if device.type != "mps":
            raise
        print(f"AdamW 初始化失败，回退 Adam：{exc}")
        optimizer = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
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
                "average_precision",
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
                    "average_precision": row.average_precision,
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
            "deepfm_pred": probabilities.astype(np.float64),
        }
    ).to_parquet(path, index=False)


def save_model_checkpoint(path: Path, model: DeepFM, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "DeepFM",
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
    print(f"Average Precision = {epoch_metrics.average_precision:.6f}")
    print(f"Actual CTR = {epoch_metrics.actual_ctr:.6f}")
    print(f"Mean Predicted CTR = {epoch_metrics.mean_predicted_ctr:.6f}")
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
    print("DEEPFM TRAINING SUMMARY")
    print("=" * 40)
    print(f"TRAIN_ROWS = {train_rows}")
    print(f"VALID_ROWS = {valid_rows}")
    print(f"DEVICE = {device_name}")
    print(f"FM_EMBEDDING_DIM = {FM_EMBEDDING_DIM}")
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
    print(f"LIGHTGBM_AUC = {LIGHTGBM_AUC:.6f}")
    print(f"WIDE_DEEP_AUC = {WIDE_DEEP_AUC:.6f}")
    print(f"AUC_DIFF_VS_LIGHTGBM = {metrics['auc'] - LIGHTGBM_AUC:+.6f}")
    print(f"AUC_DIFF_VS_WIDE_DEEP = {metrics['auc'] - WIDE_DEEP_AUC:+.6f}")
    print(f"LOGLOSS_DIFF_VS_LIGHTGBM = {metrics['logloss'] - LIGHTGBM_LOGLOSS:+.6f}")
    print(f"LOGLOSS_DIFF_VS_WIDE_DEEP = {metrics['logloss'] - WIDE_DEEP_LOGLOSS:+.6f}")
    print(f"BRIER_DIFF_VS_LIGHTGBM = {metrics['brier'] - LIGHTGBM_BRIER:+.6f}")
    print(f"BRIER_DIFF_VS_WIDE_DEEP = {metrics['brier'] - WIDE_DEEP_BRIER:+.6f}")
    print("HOLDOUT_USED = False")
    print("VALIDATION_PASSED = True")
    print(f"TEST_MODE = {test_mode}")
    if test_mode:
        print("(smoke test，非正式结果)")
    print("=" * 40)


def run_training_with_batch_size(
    batch_size: int,
    device: torch.device,
    categorical_features: list[str],
    numerical_features: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    read_cols: list[str],
    vocab_sizes: dict[str, int],
    train_rows: int,
    valid_rows: int,
    max_epochs: int,
    train_ctr: float,
    global_bias_init: float,
) -> tuple[
    DeepFM,
    list[EpochMetrics],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    int,
    str,
    dict[str, dict[str, float]] | None,
]:
    model = DeepFM(
        categorical_features=categorical_features,
        vocab_sizes=vocab_sizes,
        num_numerical_features=len(numerical_features),
        global_bias_init=global_bias_init,
        fm_embedding_dim=FM_EMBEDDING_DIM,
    ).to(device)

    print(f"TRAIN_CTR = {train_ctr:.6f}")
    print(f"INITIAL_GLOBAL_BIAS = {global_bias_init:.6f}")

    validate_model_forward(model, device, categorical_features)
    diagnose_initialization(model)
    verify_fm_formula(model, device)

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
        epoch=0,
        seed=RANDOM_SEED,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=None, num_workers=0)
    initial_component_stats = collect_component_diagnostics(model, valid_loader, device)
    print_initial_diagnostics(initial_component_stats, train_ctr)

    criterion = nn.BCEWithLogitsLoss()
    optimizer, optimizer_name = create_optimizer(model, device)

    history: list[EpochMetrics] = []
    best_valid_logloss = float("inf")
    best_epoch: int | None = None
    best_state_dict = None
    patience_counter = 0

    best_metrics: dict[str, float] = {}
    best_clicks = np.array([], dtype=np.int8)
    best_logits = np.array([], dtype=np.float64)
    best_component_stats: dict[str, dict[str, float]] | None = None

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
        valid_metrics, valid_clicks, valid_logits, component_stats = evaluate_split(
            model, valid_loader, device, epoch=epoch
        )

        elapsed = time.time() - epoch_start
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            valid_auc=valid_metrics["auc"],
            valid_logloss=valid_metrics["logloss"],
            valid_brier=valid_metrics["brier"],
            average_precision=valid_metrics["average_precision"],
            mean_predicted_ctr=valid_metrics["mean_predicted_ctr"],
            actual_ctr=valid_metrics["actual_ctr"],
            calibration_gap=valid_metrics["calibration_gap"],
            elapsed_seconds=elapsed,
        )
        history.append(epoch_metrics)
        print_epoch_metrics(epoch_metrics)
        fm_stats = component_stats["fm_second_order_logit"]
        print(
            f"FM_SECOND_ORDER_LOGIT mean={fm_stats['mean']:.6f} "
            f"std={fm_stats['std']:.6f}"
        )

        if valid_metrics["logloss"] < best_valid_logloss:
            best_valid_logloss = valid_metrics["logloss"]
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = valid_metrics
            best_clicks = valid_clicks
            best_logits = valid_logits
            best_component_stats = component_stats
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    if best_state_dict is None or best_epoch is None:
        raise RuntimeError("训练未产生有效 checkpoint（best_epoch is None）。")

    model.load_state_dict(best_state_dict)
    return (
        model,
        history,
        best_metrics,
        best_clicks,
        best_logits,
        best_epoch,
        optimizer_name,
        best_component_stats,
    )


def main() -> None:
    state = BuildState()
    train_limit, valid_limit = get_row_limits(TEST_MODE)
    max_epochs = get_max_epochs(TEST_MODE)

    log(state, "=" * 72)
    log(state, "百度 CTR 项目 — 第 41 步 DeepFM 训练")
    log(state, f"时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    log(state, f"TEST_MODE = {TEST_MODE}")

    assert_safe_path(PYTORCH_TRAIN_DIR)
    assert_safe_path(PYTORCH_VALID_DIR)

    metadata = load_pytorch_metadata(PYTORCH_METADATA_PATH)
    categorical_features, numerical_features, cat_cols, num_cols, read_cols = get_feature_columns(metadata)
    vocab_sizes: dict[str, int] = metadata["vocab_sizes"]

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

    log(state, f"\nStep 2：构建 DeepFM（categorical={len(categorical_features)}, numerical={len(numerical_features)})")
    log(state, f"FM_EMBEDDING_DIM = {FM_EMBEDDING_DIM}")
    log(state, f"FM_INTERACTION_SCOPE = {FM_INTERACTION_SCOPE}")

    log(state, "\nStep 2b：根据 train 计算 global bias 先验 ...")
    train_ctr = compute_train_ctr(PYTORCH_TRAIN_DIR, train_limit)
    global_bias_init = compute_global_bias_init(train_ctr)
    log(state, f"TRAIN_CTR = {train_ctr:.6f}")
    log(state, f"INITIAL_GLOBAL_BIAS = {global_bias_init:.6f}")

    batch_size = BATCH_SIZE
    training_error: Exception | None = None
    model = None
    history: list[EpochMetrics] = []
    best_metrics: dict[str, float] = {}
    best_clicks = np.array([], dtype=np.int8)
    best_logits = np.array([], dtype=np.float64)
    best_epoch = 0
    optimizer_name = "AdamW"
    best_component_stats: dict[str, dict[str, float]] | None = None

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
                best_component_stats,
            ) = run_training_with_batch_size(
                batch_size=candidate_batch_size,
                device=device,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                cat_cols=cat_cols,
                num_cols=num_cols,
                read_cols=read_cols,
                vocab_sizes=vocab_sizes,
                train_rows=train_limit,
                valid_rows=valid_limit,
                max_epochs=max_epochs,
                train_ctr=train_ctr,
                global_bias_init=global_bias_init,
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
        "fm_embedding_dim": FM_EMBEDDING_DIM,
        "fm_interaction_scope": FM_INTERACTION_SCOPE,
        "train_ctr": train_ctr,
        "global_bias_init": global_bias_init,
        "deep_hidden_layers": DEEP_HIDDEN_LAYERS,
        "dropout": DROPOUT_RATES,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": LEARNING_RATE,
        "fm_learning_rate": FM_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "fm_embedding_init_std": FM_EMBEDDING_INIT_STD,
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
        "lightgbm_auc": LIGHTGBM_AUC,
        "lightgbm_logloss": LIGHTGBM_LOGLOSS,
        "lightgbm_brier": LIGHTGBM_BRIER,
        "wide_deep_auc": WIDE_DEEP_AUC,
        "wide_deep_logloss": WIDE_DEEP_LOGLOSS,
        "wide_deep_brier": WIDE_DEEP_BRIER,
        "auc_diff_vs_lightgbm": best_metrics["auc"] - LIGHTGBM_AUC,
        "logloss_diff_vs_lightgbm": best_metrics["logloss"] - LIGHTGBM_LOGLOSS,
        "brier_diff_vs_lightgbm": best_metrics["brier"] - LIGHTGBM_BRIER,
        "auc_diff_vs_wide_deep": best_metrics["auc"] - WIDE_DEEP_AUC,
        "logloss_diff_vs_wide_deep": best_metrics["logloss"] - WIDE_DEEP_LOGLOSS,
        "brier_diff_vs_wide_deep": best_metrics["brier"] - WIDE_DEEP_BRIER,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metadata_out = {
        "script_name": "scripts/41_train_deepfm.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "DeepFM",
        "train_rows": validated_train_rows,
        "valid_rows": validated_valid_rows,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "vocab_sizes": vocab_sizes,
        "fm_embedding_dim": FM_EMBEDDING_DIM,
        "fm_interaction_scope": FM_INTERACTION_SCOPE,
        "train_ctr": train_ctr,
        "global_bias_init": global_bias_init,
        "deep_hidden_layers": DEEP_HIDDEN_LAYERS,
        "dropout": DROPOUT_RATES,
        "optimizer": optimizer_name,
        "learning_rate": LEARNING_RATE,
        "fm_learning_rate": FM_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "fm_embedding_init_std": FM_EMBEDDING_INIT_STD,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "device": device.type,
        "best_valid_auc": best_metrics["auc"],
        "best_valid_logloss": best_metrics["logloss"],
        "best_valid_brier": best_metrics["brier"],
        "best_average_precision": best_metrics["average_precision"],
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
        "reference_models": {
            "lightgbm_unified": {
                "auc": LIGHTGBM_AUC,
                "logloss": LIGHTGBM_LOGLOSS,
                "brier": LIGHTGBM_BRIER,
            },
            "wide_deep": {
                "auc": WIDE_DEEP_AUC,
                "logloss": WIDE_DEEP_LOGLOSS,
                "brier": WIDE_DEEP_BRIER,
            },
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

    if best_component_stats is not None:
        print_component_diagnostics(best_component_stats, header="DEEPFM COMPONENT DIAGNOSTICS")


if __name__ == "__main__":
    main()
