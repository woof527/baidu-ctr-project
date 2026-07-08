"""
百度 CTR 项目 — 第一版基础特征工程脚本

功能：
    逐个读取清洗后的 train/test Parquet 分块，
    在保留全部原始字段的基础上，新增时间特征与简单交叉特征，
    并写出到 data/features/basic/ 目录。

数据输入：
    data/processed/train/*.parquet
    data/processed/test/*.parquet

数据输出：
    data/features/basic/train/
    data/features/basic/test/

说明：
    - 不使用 click 生成任何输入特征
    - 每个 Parquet 分块单独读取、处理、保存，避免一次性载入全量数据
    - train 保留 click；test 不新增 click

用法：
    python scripts/19_build_basic_features.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 运行模式（改这里即可切换样本 / 全量）
# ---------------------------------------------------------------------------

# True：只处理少量 Parquet 分块，便于本地快速试跑
# False：处理全部 Parquet 分块
TEST_MODE = False

TEST_TRAIN_MAX_FILES = 2
TEST_TEST_MAX_FILES = 1


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_INPUT_DIR = Path("data/processed/train")
TEST_INPUT_DIR = Path("data/processed/test")

TRAIN_OUTPUT_DIR = Path("data/features/basic/train")
TEST_OUTPUT_DIR = Path("data/features/basic/test")

# 本脚本新增的特征名称（用于日志输出）
NEW_FEATURE_NAMES = [
    # 时间特征
    "hour_of_day",
    "day",
    "day_of_week",
    "is_weekend",
    # 简单交叉特征
    "banner_device_cross",
    "hour_banner_cross",
    "site_device_cross",
]


def list_parquet_files(parquet_dir: Path, max_files: int | None) -> list[Path]:
    """
    按文件名排序列出 Parquet 分块路径。

    参数：
        parquet_dir — 输入目录
        max_files   — 最多读取几个文件；None 表示读取全部

    返回：
        排序后的 Parquet 文件路径列表
    """

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到 Parquet 目录：{parquet_dir}\n"
            "请先运行：python scripts/02_clean_to_parquet.py"
        )

    all_files = sorted(parquet_dir.glob("part-*.parquet"))

    if not all_files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            "请先运行：python scripts/02_clean_to_parquet.py"
        )

    if max_files is not None:
        return all_files[:max_files]

    return all_files


def build_basic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    在保留全部原始列的基础上，新增时间特征与简单交叉特征。

    时间特征来源：hour_dt
    交叉特征：banner_pos × device_type、hour_of_day × banner_pos、
              site_category × device_type

    注意：不使用 click 生成任何输入特征。
    """

    result = dataframe.copy()

    if "hour_dt" not in result.columns:
        raise ValueError("输入数据缺少 hour_dt 字段，无法生成时间特征。")

    # 若 hour_dt 不是 datetime，先安全转换
    hour_dt = pd.to_datetime(result["hour_dt"], errors="coerce")

    # ------------------------------------------------------------------
    # 一、时间特征
    # ------------------------------------------------------------------

    # 1. hour_of_day：一天中的第几小时（0—23）
    result["hour_of_day"] = hour_dt.dt.hour.astype("Int64")

    # 2. day：日期中的「日」（1—31）
    result["day"] = hour_dt.dt.day.astype("Int64")

    # 3. day_of_week：星期几（0=周一，6=周日）
    result["day_of_week"] = hour_dt.dt.dayofweek.astype("Int64")

    # 4. is_weekend：周六(5)、周日(6) 为 1，其余为 0
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype("Int8")

    # ------------------------------------------------------------------
    # 二、简单交叉特征（字符串拼接，便于类别型树模型或后续编码）
    # ------------------------------------------------------------------

    # 1. banner_device_cross，例如 "1_1"
    result["banner_device_cross"] = (
        result["banner_pos"].astype("string")
        + "_"
        + result["device_type"].astype("string")
    )

    # 2. hour_banner_cross，例如 "21_1"
    result["hour_banner_cross"] = (
        result["hour_of_day"].astype("string")
        + "_"
        + result["banner_pos"].astype("string")
    )

    # 3. site_device_cross，例如 "50e219e0_1"
    result["site_device_cross"] = (
        result["site_category"].astype("string")
        + "_"
        + result["device_type"].astype("string")
    )

    return result


def process_parquet_file(input_path: Path, output_path: Path) -> int:
    """
    读取单个 Parquet 分块，构建特征并写出。

    返回：
        本文件处理行数
    """

    chunk = pd.read_parquet(input_path)
    featured_chunk = build_basic_features(chunk)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

    return len(featured_chunk)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    parquet_files: list[Path],
    dataset_name: str,
) -> int:
    """
    逐个处理某一数据集（train 或 test）的全部分块。

    返回：
        累计处理行数
    """

    total_rows = 0

    print(f"\n开始处理 {dataset_name} ...")
    print(f"  输入目录：{input_dir}")
    print(f"  输出目录：{output_dir}")
    print(f"  待处理文件数：{len(parquet_files)}")

    for input_path in parquet_files:
        output_path = output_dir / input_path.name
        print(f"  正在处理：{input_path.name} ...")

        rows = process_parquet_file(input_path, output_path)
        total_rows += rows

        print(f"    完成，行数：{rows:,}")

    return total_rows


def print_run_header(mode_label: str, train_count: int, test_count: int) -> None:
    """打印运行模式与输入文件数量。"""

    print("=" * 70)
    print("第一版基础特征工程")
    print("=" * 70)
    print(f"当前模式：     {mode_label}")
    print(f"train 输入文件：{train_count}")
    print(f"test 输入文件： {test_count}")
    print(f"新增特征：     {NEW_FEATURE_NAMES}")


def main() -> None:
    """主流程：列出分块 → 逐文件构建特征 → 输出日志。"""

    if TEST_MODE:
        mode_label = "TEST"
        train_files = list_parquet_files(TRAIN_INPUT_DIR, TEST_TRAIN_MAX_FILES)
        test_files = list_parquet_files(TEST_INPUT_DIR, TEST_TEST_MAX_FILES)
    else:
        mode_label = "FULL"
        train_files = list_parquet_files(TRAIN_INPUT_DIR, max_files=None)
        test_files = list_parquet_files(TEST_INPUT_DIR, max_files=None)

    print_run_header(mode_label, len(train_files), len(test_files))

    train_rows = process_dataset(
        input_dir=TRAIN_INPUT_DIR,
        output_dir=TRAIN_OUTPUT_DIR,
        parquet_files=train_files,
        dataset_name="train",
    )

    test_rows = process_dataset(
        input_dir=TEST_INPUT_DIR,
        output_dir=TEST_OUTPUT_DIR,
        parquet_files=test_files,
        dataset_name="test",
    )

    train_output_count = len(list(TRAIN_OUTPUT_DIR.glob("part-*.parquet")))
    test_output_count = len(list(TEST_OUTPUT_DIR.glob("part-*.parquet")))

    print("\n" + "=" * 70)
    print("特征工程完成")
    print("=" * 70)
    print(f"train 累计处理行数：{train_rows:,}")
    print(f"test 累计处理行数： {test_rows:,}")
    print(f"train 输出文件数：  {train_output_count}")
    print(f"test 输出文件数：   {test_output_count}")
    print(f"输出目录：")
    print(f"  {TRAIN_OUTPUT_DIR}")
    print(f"  {TEST_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
