"""
百度 CTR 项目 — 清洗后 Parquet 数据质量验收脚本

功能：
    使用 Dask 分块读取 data/processed/train 与 test 下的 Parquet，
    对清洗结果做系统性质量检查，并生成文本报告与 CSV 汇总。

数据输入：
    data/processed/train/*.parquet
    data/processed/test/*.parquet
    data/raw/sampleSubmission.csv

数据输出：
    outputs/15_processed_validation_report.txt
    outputs/eda_tables/processed_validation_summary.csv

说明：
    - 训练集体量大，统计类检查均通过 Dask 惰性计算完成，不一次性读入 pandas
    - test 与 sampleSubmission 的 id 顺序对比按 Parquet 分块逐块进行，避免全量加载
    - 字段一致性比较时排除 click、is_invalid_click（后者为 train 专用质量检查字段）

用法：
    python scripts/15_validate_processed_data.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import dask.dataframe as dd
import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_PARQUET_GLOB = "data/processed/train/*.parquet"
TEST_PARQUET_GLOB = "data/processed/test/*.parquet"
TRAIN_PARQUET_DIR = Path("data/processed/train")
TEST_PARQUET_DIR = Path("data/processed/test")
SAMPLE_SUBMISSION_PATH = Path("data/raw/sampleSubmission.csv")

REPORT_PATH = Path("outputs/15_processed_validation_report.txt")
SUMMARY_CSV_PATH = Path("outputs/eda_tables/processed_validation_summary.csv")

# 字段一致性检查时，允许仅出现在 train 中的列（test 无 click，故也无相关质量检查字段）
TRAIN_ONLY_COLUMNS = frozenset({"click", "is_invalid_click"})


@dataclass
class ValidationState:
    """
    验收过程的状态容器。

    记录各项检查结果、WARNING 列表，以及写入 CSV 的关键指标。
    """

    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    report_lines: list[str] = field(default_factory=list)

    def add_line(self, line: str = "") -> None:
        """追加一行到报告正文。"""

        self.report_lines.append(line)

    def add_warning(self, message: str) -> None:
        """记录 WARNING，并写入报告。"""

        self.warnings.append(message)
        self.add_line(f"WARNING: {message}")

    def add_pass(self, message: str) -> None:
        """记录通过的检查项。"""

        self.add_line(f"PASS: {message}")

    @property
    def all_passed(self) -> bool:
        """是否全部检查通过（无 WARNING）。"""

        return len(self.warnings) == 0


def safe_ctr(clicks: float | int, impressions: float | int) -> float:
    """计算 CTR；impressions 为 0 时返回 0.0，避免除零错误。"""

    if impressions > 0:
        return float(clicks) / float(impressions)

    return 0.0


def count_rows_dask(parquet_glob: str) -> int:
    """
    用 Dask 统计 Parquet 总行数。

    通过 map_partitions(len).sum() 分块计数，不将全量数据载入内存。
    """

    dataframe = dd.read_parquet(parquet_glob)
    return int(dataframe.map_partitions(len).sum().compute())


def get_parquet_columns(parquet_glob: str) -> list[str]:
    """
    读取 Parquet 的列名列表（仅元数据，不加载全表）。

    Dask 读取时会解析 schema，columns 属性即可用。
    """

    dataframe = dd.read_parquet(parquet_glob)
    return list(dataframe.columns)


def compute_train_click_stats(train_glob: str) -> dict[str, int | float]:
    """
    统计训练集 click 相关指标。

    返回：
        impressions      — 总行数（曝光量）
        clicks           — click 求和
        ctr              — 整体点击率
        click_0          — click=0 的行数
        click_1          — click=1 的行数
        click_na         — click 缺失行数
        click_invalid    — click 非 0/1 且非缺失的行数
    """

    train_ddf = dd.read_parquet(train_glob, columns=["click"])

    impressions_lazy = train_ddf.map_partitions(len).sum()
    clicks_lazy = train_ddf["click"].sum()

    click_0_lazy = (train_ddf["click"] == 0).sum()
    click_1_lazy = (train_ddf["click"] == 1).sum()
    click_na_lazy = train_ddf["click"].isnull().sum()

    # 非法 click：有值但不在 {0, 1} 中
    click_invalid_lazy = (
        train_ddf["click"].notnull() & ~train_ddf["click"].isin([0, 1])
    ).sum()

    (
        impressions,
        clicks,
        click_0,
        click_1,
        click_na,
        click_invalid,
    ) = dd.compute(
        impressions_lazy,
        clicks_lazy,
        click_0_lazy,
        click_1_lazy,
        click_na_lazy,
        click_invalid_lazy,
    )

    impressions = int(impressions)
    clicks = int(clicks)

    return {
        "impressions": impressions,
        "clicks": clicks,
        "ctr": safe_ctr(clicks, impressions),
        "click_0": int(click_0),
        "click_1": int(click_1),
        "click_na": int(click_na),
        "click_invalid": int(click_invalid),
    }


def count_hour_dt_missing(parquet_glob: str) -> int:
    """统计 hour_dt 缺失（null / NaT）行数。"""

    dataframe = dd.read_parquet(parquet_glob, columns=["hour_dt"])
    return int(dataframe["hour_dt"].isnull().sum().compute())


def check_click_column_presence(state: ValidationState) -> None:
    """检查 train 含 click、test 不含 click。"""

    state.add_line("【检查 1】click 字段存在性")
    state.add_line("-" * 60)

    train_columns = set(get_parquet_columns(TRAIN_PARQUET_GLOB))
    test_columns = set(get_parquet_columns(TEST_PARQUET_GLOB))

    train_has_click = "click" in train_columns
    test_has_click = "click" in test_columns

    state.metrics["train_has_click"] = train_has_click
    state.metrics["test_has_click"] = test_has_click

    if train_has_click:
        state.add_pass("train 包含 click 字段")
    else:
        state.add_warning("train 缺少 click 字段")

    if not test_has_click:
        state.add_pass("test 不包含 click 字段")
    else:
        state.add_warning("test 不应包含 click 字段，但检测到 click 列")

    state.add_line("")


def check_columns_consistency(state: ValidationState) -> None:
    """
    检查 train 与 test 特征字段是否一致。

    比较时排除 TRAIN_ONLY_COLUMNS（click、is_invalid_click）：
    click 为训练标签；is_invalid_click 为基于 click 生成的质量检查字段，test 不需要。
    """

    excluded_label = "、".join(sorted(TRAIN_ONLY_COLUMNS))
    state.add_line(f"【检查 2】train / test 字段一致性（排除 {excluded_label}）")
    state.add_line("-" * 60)

    train_columns = set(get_parquet_columns(TRAIN_PARQUET_GLOB))
    test_columns = set(get_parquet_columns(TEST_PARQUET_GLOB))

    train_features = train_columns - TRAIN_ONLY_COLUMNS
    test_features = test_columns

    only_in_train = sorted(train_features - test_features)
    only_in_test = sorted(test_features - train_features)

    columns_match = train_features == test_features
    state.metrics["columns_match_except_click"] = columns_match

    if columns_match:
        state.add_pass(
            f"train（排除 {excluded_label}）与 test 字段完全一致，共 {len(train_features)} 列"
        )
    else:
        if only_in_train:
            state.add_warning(f"仅 train 有的字段：{only_in_train}")
        if only_in_test:
            state.add_warning(f"仅 test 有的字段：{only_in_test}")

    state.add_line("")


def check_hour_fields(state: ValidationState) -> None:
    """检查 hour 与 hour_dt 字段是否存在。"""

    state.add_line("【检查 3】hour / hour_dt 字段存在性")
    state.add_line("-" * 60)

    required_hour_fields = ["hour", "hour_dt"]

    for dataset_name, parquet_glob in [
        ("train", TRAIN_PARQUET_GLOB),
        ("test", TEST_PARQUET_GLOB),
    ]:
        columns = set(get_parquet_columns(parquet_glob))
        missing_fields = [field_name for field_name in required_hour_fields if field_name not in columns]
        state.metrics[f"{dataset_name}_hour_fields_complete"] = len(missing_fields) == 0

        if missing_fields:
            state.add_warning(f"{dataset_name} 缺少字段：{missing_fields}")
        else:
            state.add_pass(f"{dataset_name} 包含 hour 与 hour_dt")

    state.add_line("")


def check_hour_dt_missing(state: ValidationState) -> None:
    """检查 train / test 中 hour_dt 是否有缺失。"""

    state.add_line("【检查 4】hour_dt 缺失情况")
    state.add_line("-" * 60)

    for dataset_name, parquet_glob in [
        ("train", TRAIN_PARQUET_GLOB),
        ("test", TEST_PARQUET_GLOB),
    ]:
        missing_count = count_hour_dt_missing(parquet_glob)
        state.metrics[f"{dataset_name}_hour_dt_missing"] = missing_count

        if missing_count == 0:
            state.add_pass(f"{dataset_name} 的 hour_dt 无缺失")
        else:
            state.add_warning(f"{dataset_name} 的 hour_dt 缺失 {missing_count:,} 行")

    state.add_line("")


def check_click_values(state: ValidationState) -> None:
    """检查 train 中 click 是否只包含 0 和 1（允许缺失，但不允许其他值）。"""

    state.add_line("【检查 5】train click 取值合法性")
    state.add_line("-" * 60)

    click_invalid = int(state.metrics.get("click_invalid", 0))
    click_only_01 = click_invalid == 0
    state.metrics["click_only_0_or_1"] = click_only_01

    if click_only_01:
        state.add_pass("train 的 click 仅包含 0、1 或缺失，未发现非法取值")
    else:
        state.add_warning(
            f"train 中存在 {click_invalid:,} 行 click 不在 {{0, 1}} 且非缺失"
        )

    state.add_line("")


def load_submission_ids() -> pd.Series:
    """
    读取 sampleSubmission.csv 的 id 列。

    提交模板通常远小于全量特征表，仅读 id 列可接受。
    """

    if not SAMPLE_SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"未找到提交模板：{SAMPLE_SUBMISSION_PATH}")

    submission_df = pd.read_csv(
        SAMPLE_SUBMISSION_PATH,
        usecols=["id"],
        dtype={"id": "string"},
    )

    return submission_df["id"]


def get_sorted_parquet_part_paths(parquet_dir: Path) -> list[Path]:
    """按文件名排序返回 Parquet 分块路径，保证与清洗写出顺序一致。"""

    if not parquet_dir.exists():
        return []

    return sorted(parquet_dir.glob("part-*.parquet"))


def compare_test_id_order_with_submission(state: ValidationState) -> None:
    """
    检查 test 行数及 id 顺序是否与 sampleSubmission.csv 一致。

    按 Parquet 分块逐块对比 id，避免一次性加载全部 test id。
    """

    state.add_line("【检查 6】test 行数与 id 顺序（对比 sampleSubmission.csv）")
    state.add_line("-" * 60)

    if not SAMPLE_SUBMISSION_PATH.exists():
        state.add_warning(f"未找到 sampleSubmission.csv：{SAMPLE_SUBMISSION_PATH}")
        state.metrics["test_row_count_match_submission"] = False
        state.metrics["test_id_order_match_submission"] = False
        state.add_line("")
        return

    submission_ids = load_submission_ids()
    submission_rows = len(submission_ids)
    state.metrics["sample_submission_rows"] = submission_rows

    test_rows = int(state.metrics.get("test_rows", 0))
    row_count_match = test_rows == submission_rows
    state.metrics["test_row_count_match_submission"] = row_count_match

    if row_count_match:
        state.add_pass(
            f"test 行数与 sampleSubmission 一致：{test_rows:,} 行"
        )
    else:
        state.add_warning(
            f"test 行数 ({test_rows:,}) 与 sampleSubmission ({submission_rows:,}) 不一致"
        )
        state.metrics["test_id_order_match_submission"] = False
        state.add_line("")
        return

    part_paths = get_sorted_parquet_part_paths(TEST_PARQUET_DIR)
    if not part_paths:
        state.add_warning(f"test Parquet 目录为空或不存在：{TEST_PARQUET_DIR}")
        state.metrics["test_id_order_match_submission"] = False
        state.add_line("")
        return

    offset = 0
    order_match = True

    for part_path in part_paths:
        part_ids = pd.read_parquet(part_path, columns=["id"])["id"].astype("string")
        part_len = len(part_ids)

        submission_slice = submission_ids.iloc[offset : offset + part_len].reset_index(drop=True)
        part_ids_reset = part_ids.reset_index(drop=True)

        if not part_ids_reset.equals(submission_slice):
            # 定位第一个不一致位置，便于排查
            mismatch_mask = part_ids_reset != submission_slice
            first_bad = int(mismatch_mask.idxmax()) if mismatch_mask.any() else -1
            state.add_warning(
                f"id 顺序不一致：分块 {part_path.name} 内第 {first_bad} 行与 "
                f"sampleSubmission 对应位置不匹配"
            )
            order_match = False
            break

        offset += part_len

    if offset != submission_rows:
        state.add_warning(
            f"test 分块累计行数 ({offset:,}) 与 sampleSubmission ({submission_rows:,}) 不一致"
        )
        order_match = False

    state.metrics["test_id_order_match_submission"] = order_match

    if order_match:
        state.add_pass("test 的 id 顺序与 sampleSubmission.csv 完全一致")

    state.add_line("")


def collect_basic_stats(state: ValidationState) -> None:
    """收集 train / test 行数及 train click 统计，写入 metrics。"""

    state.add_line("【基础统计】")
    state.add_line("-" * 60)

    train_rows = count_rows_dask(TRAIN_PARQUET_GLOB)
    test_rows = count_rows_dask(TEST_PARQUET_GLOB)

    state.metrics["train_rows"] = train_rows
    state.metrics["test_rows"] = test_rows

    state.add_line(f"train 总行数：{train_rows:,}")
    state.add_line(f"test 总行数： {test_rows:,}")

    click_stats = compute_train_click_stats(TRAIN_PARQUET_GLOB)
    state.metrics.update(click_stats)

    state.add_line(f"train 曝光量 (impressions)：{click_stats['impressions']:,}")
    state.add_line(f"train 点击量 (clicks)：     {click_stats['clicks']:,}")
    state.add_line(f"train 整体 CTR：            {click_stats['ctr']:.6%}")
    state.add_line("train click 分布：")
    state.add_line(f"  click = 0：  {click_stats['click_0']:,}")
    state.add_line(f"  click = 1：  {click_stats['click_1']:,}")
    state.add_line(f"  click 缺失： {click_stats['click_na']:,}")
    state.add_line(f"  click 非法： {click_stats['click_invalid']:,}")
    state.add_line("")


def ensure_input_paths(state: ValidationState) -> bool:
    """
    检查必要输入路径是否存在。

    若 train / test Parquet 目录缺失，记录 WARNING 并返回 False。
    """

    inputs_ok = True

    if not TRAIN_PARQUET_DIR.exists() or not list(TRAIN_PARQUET_DIR.glob("part-*.parquet")):
        state.add_warning(
            f"train Parquet 不存在或为空：{TRAIN_PARQUET_DIR}\n"
            "请先运行：python scripts/02_clean_to_parquet.py"
        )
        inputs_ok = False

    if not TEST_PARQUET_DIR.exists() or not list(TEST_PARQUET_DIR.glob("part-*.parquet")):
        state.add_warning(
            f"test Parquet 不存在或为空：{TEST_PARQUET_DIR}\n"
            "请先运行：python scripts/02_clean_to_parquet.py"
        )
        inputs_ok = False

    return inputs_ok


def build_summary_dataframe(state: ValidationState) -> pd.DataFrame:
    """将关键验收指标整理为单行 CSV。"""

    validation_status = "PASSED" if state.all_passed else "FAILED"

    summary_row = {
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": state.metrics.get("train_rows"),
        "test_rows": state.metrics.get("test_rows"),
        "sample_submission_rows": state.metrics.get("sample_submission_rows"),
        "train_impressions": state.metrics.get("impressions"),
        "train_clicks": state.metrics.get("clicks"),
        "train_ctr": state.metrics.get("ctr"),
        "click_0": state.metrics.get("click_0"),
        "click_1": state.metrics.get("click_1"),
        "click_na": state.metrics.get("click_na"),
        "click_invalid": state.metrics.get("click_invalid"),
        "train_hour_dt_missing": state.metrics.get("train_hour_dt_missing"),
        "test_hour_dt_missing": state.metrics.get("test_hour_dt_missing"),
        "train_has_click": state.metrics.get("train_has_click"),
        "test_has_click": state.metrics.get("test_has_click"),
        "columns_match_except_click": state.metrics.get("columns_match_except_click"),
        "train_hour_fields_complete": state.metrics.get("train_hour_fields_complete"),
        "test_hour_fields_complete": state.metrics.get("test_hour_fields_complete"),
        "click_only_0_or_1": state.metrics.get("click_only_0_or_1"),
        "test_row_count_match_submission": state.metrics.get("test_row_count_match_submission"),
        "test_id_order_match_submission": state.metrics.get("test_id_order_match_submission"),
        "warning_count": len(state.warnings),
        "validation_status": validation_status,
    }

    return pd.DataFrame([summary_row])


def write_outputs(state: ValidationState) -> None:
    """写入文本报告与 CSV 汇总。"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "=" * 70,
        "百度 CTR 项目 — 清洗后 Parquet 数据质量验收报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    footer_lines = ["", "=" * 70]
    if state.all_passed:
        footer_lines.append("VALIDATION PASSED")
    else:
        footer_lines.append(f"VALIDATION FAILED — 共 {len(state.warnings)} 项 WARNING")
        footer_lines.append("")
        footer_lines.append("WARNING 汇总：")
        for index, warning in enumerate(state.warnings, start=1):
            footer_lines.append(f"  {index}. {warning}")
    footer_lines.append("=" * 70)

    full_report = "\n".join(header_lines + state.report_lines + footer_lines)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"验收报告已保存：{REPORT_PATH}")

    summary_df = build_summary_dataframe(state)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"验收汇总已保存：{SUMMARY_CSV_PATH}")


def main() -> None:
    """主流程：基础统计 → 逐项检查 → 写报告与 CSV。"""

    print("=" * 70)
    print("清洗后 Parquet 数据质量验收")
    print("=" * 70)

    state = ValidationState()

    if not ensure_input_paths(state):
        write_outputs(state)
        print("\n输入数据不完整，验收终止。")
        return

    collect_basic_stats(state)

    check_click_column_presence(state)
    check_columns_consistency(state)
    check_hour_fields(state)
    check_hour_dt_missing(state)
    check_click_values(state)
    compare_test_id_order_with_submission(state)

    write_outputs(state)

    if state.all_passed:
        print("\nVALIDATION PASSED")
    else:
        print(f"\nVALIDATION FAILED — 共 {len(state.warnings)} 项 WARNING")
        for warning in state.warnings:
            print(f"  WARNING: {warning}")


if __name__ == "__main__":
    main()
