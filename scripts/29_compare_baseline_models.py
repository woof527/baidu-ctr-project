"""
百度 CTR 项目 — 三模型基线统一对比

功能：
    读取逻辑回归、LightGBM、XGBoost 的正式 valid 指标，
    生成统一对比表、相对提升、图表与报告。禁止读取 _test 结果。

用法：
    python scripts/29_compare_baseline_models.py
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

LOGISTIC_METRICS_PATH = Path("outputs/logistic_baseline_valid_metrics.json")
LIGHTGBM_METRICS_PATH = Path("outputs/lightgbm_baseline_valid_metrics.json")
XGBOOST_METRICS_PATH = Path("outputs/xgboost_baseline_valid_metrics.json")

OUTPUT_COMPARISON_CSV = Path("outputs/model_comparison.csv")
OUTPUT_RELATIVE_IMPROVEMENTS_CSV = Path("outputs/model_relative_improvements.csv")
OUTPUT_AUC_PNG = Path("outputs/model_comparison_auc.png")
OUTPUT_LOGLOSS_PNG = Path("outputs/model_comparison_logloss.png")
OUTPUT_CALIBRATION_PNG = Path("outputs/model_comparison_calibration_gap.png")
OUTPUT_REPORT_TXT = Path("outputs/model_comparison_report.txt")
OUTPUT_WEEK4_MD = Path("reports/week4_model_baseline_summary.md")

MODEL_SPECS: list[dict[str, str | Path]] = [
    {
        "key": "logistic_regression",
        "display_name": "Logistic Regression",
        "metrics_path": LOGISTIC_METRICS_PATH,
        "metadata_path": Path("models/logistic_baseline.joblib"),
        "report_path": Path("outputs/logistic_baseline_report.txt"),
    },
    {
        "key": "lightgbm",
        "display_name": "LightGBM",
        "metrics_path": LIGHTGBM_METRICS_PATH,
        "metadata_path": Path("models/lightgbm_baseline.joblib"),
        "report_path": Path("outputs/lightgbm_baseline_report.txt"),
    },
    {
        "key": "xgboost",
        "display_name": "XGBoost",
        "metrics_path": XGBOOST_METRICS_PATH,
        "metadata_path": Path("models/xgboost_baseline_metadata.joblib"),
        "report_path": Path("outputs/xgboost_baseline_report.txt"),
    },
]

REQUIRED_METRIC_KEYS = (
    "roc_auc",
    "log_loss",
    "valid_actual_ctr",
    "valid_mean_predicted_ctr",
)

OPTIONAL_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "predicted_click_ratio_at_threshold",
    "threshold",
    "best_iteration",
)

CHART_DPI = 150


@dataclass
class BestModelSummary:
    """各维度最优模型。"""

    best_auc_model: str
    best_logloss_model: str
    best_calibration_model: str
    best_overall_model: str


def load_json_metrics(metrics_path: Path) -> dict:
    """读取并解析指标 JSON 文件。"""

    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到正式指标文件：{metrics_path}")

    if "_test" in metrics_path.stem:
        raise ValueError(f"禁止读取测试模式指标文件：{metrics_path}")

    return json.loads(metrics_path.read_text(encoding="utf-8"))


def validate_metrics(metrics: dict, metrics_path: Path) -> None:
    """校验指标文件中的关键字段。"""

    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise KeyError(f"{metrics_path} 缺少关键指标字段：{key}")

    for key in ("roc_auc", "log_loss"):
        value = metrics[key]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"{metrics_path} 的 {key} 不是有限数值：{value!r}"
            )

    for key in ("valid_actual_ctr", "valid_mean_predicted_ctr"):
        value = metrics[key]
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{metrics_path} 的 {key} 不是数值类型：{value!r}"
            )

    if "test_mode" in metrics and metrics["test_mode"] is not False:
        raise ValueError(
            f"{metrics_path} 的 test_mode={metrics['test_mode']!r}，"
            "正式对比仅允许 test_mode=false。"
        )


def parse_report_counts(report_path: Path) -> dict[str, int]:
    """从训练报告解析样本规模与特征数量。"""

    if not report_path.exists():
        raise FileNotFoundError(f"未找到训练报告：{report_path}")

    text = report_path.read_text(encoding="utf-8")

    train_match = re.search(r"训练样本数[：:]\s*([\d,]+)", text)
    valid_match = re.search(r"验证样本数[：:]\s*([\d,]+)", text)
    feature_match = re.search(r"最终特征数量[：:]\s*(\d+)", text)

    if train_match is None or valid_match is None or feature_match is None:
        raise ValueError(f"无法从报告解析样本/特征信息：{report_path}")

    return {
        "train_row_count": int(train_match.group(1).replace(",", "")),
        "valid_row_count": int(valid_match.group(1).replace(",", "")),
        "feature_count": int(feature_match.group(1)),
    }


def load_sample_metadata(metadata_path: Path, report_path: Path) -> dict[str, int]:
    """优先从模型元数据读取样本规模，失败时回退到报告解析。"""

    if metadata_path.exists():
        try:
            bundle = joblib.load(metadata_path)
            train_row_count = bundle.get("train_row_count")
            valid_row_count = bundle.get("valid_row_count")
            feature_columns = bundle.get("feature_columns")

            if train_row_count is None or valid_row_count is None:
                raise KeyError("元数据缺少 train_row_count 或 valid_row_count")

            feature_count = (
                len(feature_columns)
                if feature_columns is not None
                else parse_report_counts(report_path)["feature_count"]
            )

            if bundle.get("test_mode") is True:
                raise ValueError(f"{metadata_path} 为测试模式元数据，禁止用于正式对比。")

            return {
                "train_row_count": int(train_row_count),
                "valid_row_count": int(valid_row_count),
                "feature_count": int(feature_count),
            }
        except Exception:
            pass

    return parse_report_counts(report_path)


def load_all_model_records() -> list[dict]:
    """读取并整理三个模型的正式记录。"""

    records: list[dict] = []

    for spec in MODEL_SPECS:
        metrics_path = Path(spec["metrics_path"])
        metrics = load_json_metrics(metrics_path)
        validate_metrics(metrics, metrics_path)

        metadata = load_sample_metadata(
            Path(spec["metadata_path"]),
            Path(spec["report_path"]),
        )

        valid_actual_ctr = float(metrics["valid_actual_ctr"])
        valid_mean_predicted_ctr = float(metrics["valid_mean_predicted_ctr"])
        ctr_calibration_gap = abs(valid_mean_predicted_ctr - valid_actual_ctr)

        best_iteration = metrics.get("best_iteration")
        if best_iteration is not None:
            best_iteration = int(best_iteration)

        record = {
            "model": spec["display_name"],
            "model_key": spec["key"],
            "mode": "formal",
            "train_row_count": metadata["train_row_count"],
            "valid_row_count": metadata["valid_row_count"],
            "feature_count": metadata["feature_count"],
            "roc_auc": float(metrics["roc_auc"]),
            "log_loss": float(metrics["log_loss"]),
            "accuracy": float(metrics.get("accuracy", np.nan)),
            "precision": float(metrics.get("precision", np.nan)),
            "recall": float(metrics.get("recall", np.nan)),
            "f1": float(metrics.get("f1", np.nan)),
            "valid_actual_ctr": valid_actual_ctr,
            "valid_mean_predicted_ctr": valid_mean_predicted_ctr,
            "ctr_calibration_gap": ctr_calibration_gap,
            "predicted_click_ratio_at_threshold": float(
                metrics.get("predicted_click_ratio_at_threshold", np.nan)
            ),
            "threshold": float(metrics.get("threshold", 0.5)),
            "best_iteration": best_iteration,
        }
        records.append(record)

    return records


def build_comparison_dataframe(records: list[dict]) -> pd.DataFrame:
    """构建统一指标表并计算排名。"""

    df = pd.DataFrame(records)

    df["auc_rank"] = df["roc_auc"].rank(method="min", ascending=False).astype(int)
    df["logloss_rank"] = df["log_loss"].rank(method="min", ascending=True).astype(int)
    df["calibration_rank"] = df["ctr_calibration_gap"].rank(
        method="min",
        ascending=True,
    ).astype(int)

    df["overall_rank_score"] = (
        df["auc_rank"] + df["logloss_rank"] + df["calibration_rank"]
    ) / 3.0
    df["overall_rank"] = df["overall_rank_score"].rank(
        method="min",
        ascending=True,
    ).astype(int)

    column_order = [
        "model",
        "mode",
        "train_row_count",
        "valid_row_count",
        "feature_count",
        "roc_auc",
        "log_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "valid_actual_ctr",
        "valid_mean_predicted_ctr",
        "ctr_calibration_gap",
        "predicted_click_ratio_at_threshold",
        "threshold",
        "best_iteration",
        "auc_rank",
        "logloss_rank",
        "calibration_rank",
        "overall_rank_score",
        "overall_rank",
    ]

    return df[column_order].sort_values("overall_rank").reset_index(drop=True)


def build_relative_improvements_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """以逻辑回归为基线，计算树模型相对提升。"""

    logistic_row = df.loc[df["model"] == "Logistic Regression"].iloc[0]
    tree_models = df[df["model"].isin(["LightGBM", "XGBoost"])].copy()

    rows: list[dict] = []
    for _, row in tree_models.iterrows():
        auc_absolute_improvement = row["roc_auc"] - logistic_row["roc_auc"]
        logloss_absolute_reduction = logistic_row["log_loss"] - row["log_loss"]
        calibration_gap_reduction = (
            logistic_row["ctr_calibration_gap"] - row["ctr_calibration_gap"]
        )

        rows.append(
            {
                "baseline_model": "Logistic Regression",
                "tree_model": row["model"],
                "logistic_roc_auc": logistic_row["roc_auc"],
                "tree_roc_auc": row["roc_auc"],
                "auc_absolute_improvement": auc_absolute_improvement,
                "auc_relative_improvement_percent": (
                    auc_absolute_improvement / logistic_row["roc_auc"] * 100.0
                ),
                "logistic_log_loss": logistic_row["log_loss"],
                "tree_log_loss": row["log_loss"],
                "logloss_absolute_reduction": logloss_absolute_reduction,
                "logloss_relative_reduction_percent": (
                    logloss_absolute_reduction / logistic_row["log_loss"] * 100.0
                ),
                "logistic_calibration_gap": logistic_row["ctr_calibration_gap"],
                "tree_calibration_gap": row["ctr_calibration_gap"],
                "calibration_gap_reduction": calibration_gap_reduction,
            }
        )

    return pd.DataFrame(rows)


def summarize_best_models(df: pd.DataFrame) -> BestModelSummary:
    """根据正式指标确定各维度最优模型。"""

    best_auc_row = df.loc[df["roc_auc"].idxmax()]
    best_logloss_row = df.loc[df["log_loss"].idxmin()]
    best_calibration_row = df.loc[df["ctr_calibration_gap"].idxmin()]
    best_overall_row = df.loc[df["overall_rank"].idxmin()]

    return BestModelSummary(
        best_auc_model=str(best_auc_row["model"]),
        best_logloss_model=str(best_logloss_row["model"]),
        best_calibration_model=str(best_calibration_row["model"]),
        best_overall_model=str(best_overall_row["model"]),
    )


def format_metric_table(df: pd.DataFrame) -> str:
    """将核心指标格式化为文本表。"""

    display_df = df[
        [
            "model",
            "train_row_count",
            "valid_row_count",
            "feature_count",
            "roc_auc",
            "log_loss",
            "ctr_calibration_gap",
            "valid_actual_ctr",
            "valid_mean_predicted_ctr",
            "overall_rank",
        ]
    ].copy()

    lines = ["模型正式指标表", "-" * 90]
    for _, row in display_df.iterrows():
        lines.append(
            f"{row['model']:<22} "
            f"train={int(row['train_row_count']):>10,} "
            f"valid={int(row['valid_row_count']):>8,} "
            f"feat={int(row['feature_count']):>2} "
            f"AUC={row['roc_auc']:.6f} "
            f"LogLoss={row['log_loss']:.6f} "
            f"CalibGap={row['ctr_calibration_gap']:.6f} "
            f"OverallRank={int(row['overall_rank'])}"
        )

    return "\n".join(lines)


def compare_tree_models(df: pd.DataFrame) -> str:
    """生成 LightGBM 与 XGBoost 的直接对比文本。"""

    lightgbm = df.loc[df["model"] == "LightGBM"].iloc[0]
    xgboost = df.loc[df["model"] == "XGBoost"].iloc[0]

    lines = [
        "LightGBM 与 XGBoost 直接对比",
        "-" * 90,
        "二者使用相同训练样本规模、相同验证样本规模、相同特征体系、",
        "相同随机种子与相同抽样摘要，因此比较相对公平。",
        "",
        f"LightGBM  AUC={lightgbm['roc_auc']:.6f}, "
        f"LogLoss={lightgbm['log_loss']:.6f}, "
        f"CalibGap={lightgbm['ctr_calibration_gap']:.6f}, "
        f"best_iteration={lightgbm['best_iteration']}",
        f"XGBoost   AUC={xgboost['roc_auc']:.6f}, "
        f"LogLoss={xgboost['log_loss']:.6f}, "
        f"CalibGap={xgboost['ctr_calibration_gap']:.6f}, "
        f"best_iteration={xgboost['best_iteration']}",
        "",
        f"AUC 差异（LightGBM - XGBoost）："
        f"{lightgbm['roc_auc'] - xgboost['roc_auc']:+.6f}",
        f"LogLoss 差异（LightGBM - XGBoost）："
        f"{lightgbm['log_loss'] - xgboost['log_loss']:+.6f}",
        f"校准差距差异（LightGBM - XGBoost）："
        f"{lightgbm['ctr_calibration_gap'] - xgboost['ctr_calibration_gap']:+.6f}",
    ]

    return "\n".join(lines)


def plot_bar_metric(
    df: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    lower_is_better: bool = False,
    as_percent: bool = False,
) -> None:
    """绘制单指标柱状图。"""

    plot_df = df.sort_values(metric_column, ascending=lower_is_better)
    values = plot_df[metric_column].to_numpy(dtype=float)
    models = plot_df["model"].tolist()

    if as_percent:
        display_values = values * 100.0
        value_suffix = "%"
    else:
        display_values = values
        value_suffix = ""

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(models, display_values)

    value_range = max(display_values) - min(display_values)
    padding = value_range * 0.15 if value_range > 0 else max(display_values) * 0.05
    axis.set_ylim(min(display_values) - padding, max(display_values) + padding)
    axis.set_title(title)
    axis.set_xlabel("Model")
    axis.set_ylabel(ylabel)

    for bar, raw_value, display_value in zip(bars, values, display_values):
        if as_percent:
            label = f"{display_value:.4f}%"
        else:
            label = f"{raw_value:.4f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=CHART_DPI)
    plt.close(figure)


def write_text_report(
    df: pd.DataFrame,
    relative_df: pd.DataFrame,
    summary: BestModelSummary,
    output_path: Path,
) -> None:
    """写入文本对比报告。"""

    lines = [
        "百度 CTR 项目 — 三模型基线统一对比报告",
        "=" * 90,
        "",
        format_metric_table(df),
        "",
        "核心比较结论",
        "-" * 90,
        f"1. AUC 最高模型：{summary.best_auc_model}",
        f"2. LogLoss 最低模型：{summary.best_logloss_model}",
        f"3. 平均预测 CTR 最接近实际 CTR 的模型：{summary.best_calibration_model}",
        f"4. 当前综合表现最好的模型：{summary.best_overall_model}",
        "",
        "说明：overall_rank 采用 AUC 排名、LogLoss 排名、校准差距排名的平均值，",
        "只是项目内部的简化综合排序，不能替代具体业务指标。",
        "",
        compare_tree_models(df),
        "",
        "树模型相对逻辑回归的提升",
        "-" * 90,
    ]

    for _, row in relative_df.iterrows():
        lines.extend(
            [
                f"{row['tree_model']}:",
                f"  AUC 绝对提升：{row['auc_absolute_improvement']:+.6f} "
                f"({row['auc_relative_improvement_percent']:+.4f}%)",
                f"  LogLoss 绝对下降：{row['logloss_absolute_reduction']:+.6f} "
                f"({row['logloss_relative_reduction_percent']:+.4f}%)",
                f"  校准差距下降：{row['calibration_gap_reduction']:+.6f}",
                "",
            ]
        )

    lines.extend(
        [
            "辅助指标说明",
            "-" * 90,
            "Accuracy、Precision、Recall、F1 依赖固定阈值下的分类结果。",
            "在 CTR 预测中，我们更关注排序能力（AUC）与概率质量（LogLoss），",
            "因此这些分类指标仅作为辅助参考，不应作为模型选择主依据。",
            "",
            "0.5 阈值说明",
            "-" * 90,
            "当前验证集实际 CTR 约为 15%-16%，而 0.5 阈值会显著低估点击比例。",
            "这意味着 Accuracy / Precision / Recall / F1 在默认阈值下并不反映真实业务场景，",
            "后续应结合业务目标 CTR、成本与收益做阈值或校准分析。",
            "",
            "训练样本口径说明",
            "-" * 90,
            "逻辑回归使用全量训练数据，而 LightGBM 和 XGBoost 使用 200 万条抽样训练数据，",
            "因此逻辑回归与树模型不是完全相同的训练样本口径。",
            "",
            "当前最佳模型结论",
            "-" * 90,
            f"综合 AUC、LogLoss 与概率校准表现，当前最佳传统模型基线为："
            f"{summary.best_overall_model}。",
            "",
            "实验局限",
            "-" * 90,
            "- 逻辑回归与树模型训练样本口径不同",
            "- LightGBM 和 XGBoost 虽然抽样摘要一致，但没有单独保存全部抽样 ID 进行逐条校验",
            "- 当前参数主要为基线参数，尚未进行 Optuna 调优",
            "- 尚未使用 holdout",
            "- 当前只使用工程化数值特征，未充分处理所有原始类别特征",
            "- 尚未进行 SHAP、概率校准和业务阈值分析",
            "",
            "holdout 尚未使用。",
            "",
            "下一阶段建议",
            "-" * 90,
            "- 超参数调优",
            "- SHAP 可解释性分析",
            "- 概率校准",
            "- 最终模型选择",
            "- holdout 一次性评估",
            "- 业务落地与 A/B 测试设计",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """将 DataFrame 转为 Markdown 表格。"""

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]

    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if column in {"roc_auc", "log_loss", "ctr_calibration_gap"}:
                    cells.append(f"{value:.6f}")
                elif column.endswith("_ctr"):
                    cells.append(f"{value:.6f}")
                else:
                    cells.append(f"{value:.4f}")
            elif pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def write_week4_summary(
    df: pd.DataFrame,
    relative_df: pd.DataFrame,
    summary: BestModelSummary,
    output_path: Path,
) -> None:
    """生成 Week 4 模型基线 Markdown 总结。"""

    core_df = df[
        [
            "model",
            "roc_auc",
            "log_loss",
            "valid_actual_ctr",
            "valid_mean_predicted_ctr",
            "ctr_calibration_gap",
            "overall_rank",
        ]
    ].copy()

    lines = [
        "# 本周工作概述",
        "",
        "- 完成时间划分",
        "- 完成历史统计特征",
        "- 完成平滑目标编码",
        "- 修复按 Parquet 文件日期处理导致的时间泄漏问题",
        "- 通过 40 项高级特征验收",
        "- 完成逻辑回归、LightGBM、XGBoost 三个传统模型基线",
        "",
        "# 模型原理简述",
        "",
        "## 逻辑回归",
        "",
        "逻辑回归是线性概率模型，通过 sigmoid 函数将特征线性组合映射到 [0, 1] 点击概率。",
        "它训练快、可解释性强，但难以自动表达复杂非线性关系。",
        "",
        "## LightGBM",
        "",
        "LightGBM 是基于梯度提升的 Leaf-wise 决策树集成模型。",
        "它每次优先分裂增益最大的叶子，通常在相同迭代预算下收敛更快。",
        "",
        "## XGBoost",
        "",
        "XGBoost 是默认更偏向 Depth-wise 的梯度提升树模型。",
        "它通过层级分裂与正则化控制树复杂度，在结构化表格数据上表现稳定。",
        "",
        "# 正式指标对比",
        "",
        dataframe_to_markdown_table(
            core_df,
            [
                "model",
                "roc_auc",
                "log_loss",
                "valid_actual_ctr",
                "valid_mean_predicted_ctr",
                "ctr_calibration_gap",
                "overall_rank",
            ],
        ),
        "",
        "# 当前结论",
        "",
        f"- **AUC 最高模型**：{summary.best_auc_model}",
        f"- **LogLoss 最低模型**：{summary.best_logloss_model}",
        f"- **平均预测 CTR 最接近真实 CTR 的模型**：{summary.best_calibration_model}",
        f"- **当前最佳传统模型基线**：{summary.best_overall_model}",
        "",
        "LightGBM 与 XGBoost 在相同抽样配置、相同特征体系与相同随机种子下对比，",
        "因此二者的差异更具参考价值。",
        "",
        "# 当前局限",
        "",
        "- 逻辑回归与树模型训练样本口径不同",
        "- LightGBM 和 XGBoost 虽然抽样摘要一致，但没有单独保存全部抽样 ID 进行逐条校验",
        "- 当前参数主要为基线参数，尚未进行 Optuna 调优",
        "- 尚未使用 holdout",
        "- 当前只使用工程化数值特征，未充分处理所有原始类别特征",
        "- 尚未进行 SHAP、概率校准和业务阈值分析",
        "",
        "# 下一阶段",
        "",
        "- 超参数调优",
        "- SHAP 可解释性分析",
        "- 概率校准",
        "- 最终模型选择",
        "- holdout 一次性评估",
        "- 业务落地与 A/B 测试设计",
        "",
        "# 相对逻辑回归的提升",
        "",
        dataframe_to_markdown_table(
            relative_df,
            [
                "tree_model",
                "auc_absolute_improvement",
                "auc_relative_improvement_percent",
                "logloss_absolute_reduction",
                "logloss_relative_reduction_percent",
                "calibration_gap_reduction",
            ],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(
    comparison_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    summary: BestModelSummary,
) -> None:
    """保存 CSV、图表与报告。"""

    comparison_df.to_csv(OUTPUT_COMPARISON_CSV, index=False, encoding="utf-8")
    relative_df.to_csv(
        OUTPUT_RELATIVE_IMPROVEMENTS_CSV,
        index=False,
        encoding="utf-8",
    )

    plot_bar_metric(
        comparison_df,
        metric_column="roc_auc",
        ylabel="ROC-AUC",
        title="Baseline Models ROC-AUC Comparison",
        output_path=OUTPUT_AUC_PNG,
        lower_is_better=False,
    )
    plot_bar_metric(
        comparison_df,
        metric_column="log_loss",
        ylabel="LogLoss",
        title="Baseline Models LogLoss Comparison (Lower is Better)",
        output_path=OUTPUT_LOGLOSS_PNG,
        lower_is_better=True,
    )
    plot_bar_metric(
        comparison_df,
        metric_column="ctr_calibration_gap",
        ylabel="CTR Calibration Gap",
        title="Baseline Models CTR Calibration Gap Comparison (Lower is Better)",
        output_path=OUTPUT_CALIBRATION_PNG,
        lower_is_better=True,
        as_percent=True,
    )

    write_text_report(comparison_df, relative_df, summary, OUTPUT_REPORT_TXT)
    write_week4_summary(comparison_df, relative_df, summary, OUTPUT_WEEK4_MD)


def main() -> None:
    """主流程：读取正式指标 → 排名 → 输出对比结果。"""

    print("=" * 70)
    print("三模型基线统一对比")
    print("=" * 70)

    records = load_all_model_records()
    comparison_df = build_comparison_dataframe(records)
    relative_df = build_relative_improvements_dataframe(comparison_df)
    summary = summarize_best_models(comparison_df)

    save_outputs(comparison_df, relative_df, summary)

    print("\n核心结论：")
    print(f"  AUC 最优模型：       {summary.best_auc_model}")
    print(f"  LogLoss 最优模型：   {summary.best_logloss_model}")
    print(f"  概率校准最优模型：   {summary.best_calibration_model}")
    print(f"  当前综合最佳模型：   {summary.best_overall_model}")
    print("\n输出文件：")
    print(f"  统一指标表：         {OUTPUT_COMPARISON_CSV}")
    print(f"  相对提升表：         {OUTPUT_RELATIVE_IMPROVEMENTS_CSV}")
    print(f"  AUC 图：             {OUTPUT_AUC_PNG}")
    print(f"  LogLoss 图：         {OUTPUT_LOGLOSS_PNG}")
    print(f"  校准差距图：         {OUTPUT_CALIBRATION_PNG}")
    print(f"  文本报告：           {OUTPUT_REPORT_TXT}")
    print(f"  周总结 Markdown：    {OUTPUT_WEEK4_MD}")
    print("三模型基线对比完成，holdout 尚未使用。")
    print("=" * 70)


if __name__ == "__main__":
    main()
