"""
百度 CTR 项目 — 类别频次特征生成脚本

功能：
    1. 使用 Dask 在训练集上统计类别字段出现次数，生成频次映射表
    2. 逐 Parquet 分块读取 basic 特征数据，映射频次特征并写出
    3. 将映射表保存到 outputs/feature_tables/

数据输入：
    data/features/basic/train/*.parquet
    data/features/basic/test/*.parquet

数据输出：
    data/features/frequency/train/
    data/features/frequency/test/
    outputs/feature_tables/*_frequency.csv

说明：
    - 频次统计仅基于 train，不使用 click
    - test 中训练集未见过的类别，频次填 0
    - 不做目标编码，不删除低频类别

用法：
    python scripts/20_build_frequency_features.py
"""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import pandas as pd


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

# True：用少量 train 文件建立映射，并只处理少量 train/test 分块
# False：用全部 train 建立映射，并处理全部分块
TEST_MODE =False

TEST_TRAIN_MAX_FILES = 2
TEST_TEST_MAX_FILES = 1


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_INPUT_DIR = Path("data/features/basic/train")
TEST_INPUT_DIR = Path("data/features/basic/test")

TRAIN_OUTPUT_DIR = Path("data/features/frequency/train")
TEST_OUTPUT_DIR = Path("data/features/frequency/test")

FREQUENCY_TABLE_DIR = Path("outputs/feature_tables")

# 原始字段 → 频次特征名
FREQUENCY_FIELD_MAP: dict[str, str] = {
    "site_id": "site_id_freq",
    "site_category": "site_category_freq",
    "app_id": "app_id_freq",
    "app_category": "app_category_freq",
    "device_model": "device_model_freq",
}


def list_parquet_files(parquet_dir: Path, max_files: int | None) -> list[Path]:
    """
    按文件名排序列出 Parquet 分块路径。

    参数：
        parquet_dir — 输入目录
        max_files   — 最多返回几个文件；None 表示全部
    """

    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"未找到输入目录：{parquet_dir}\n"
            "请先运行：python scripts/19_build_basic_features.py"
        )

    all_files = sorted(parquet_dir.glob("part-*.parquet"))

    if not all_files:
        raise FileNotFoundError(
            f"目录中没有 Parquet 文件：{parquet_dir}\n"
            "请先运行：python scripts/19_build_basic_features.py"
        )

    if max_files is not None:
        return all_files[:max_files]

    return all_files


def build_frequency_maps_with_dask(train_files: list[Path]) -> dict[str, dict]:
    """
    使用 Dask 读取 train Parquet，统计各字段 value_counts。

    只读取频次统计需要的列，避免加载无关字段。
    统计完成后在 pandas 中转为 {原始值: 出现次数} 字典。

    返回：
        {原始字段名: {类别值: 频次}}
    """

    source_columns = list(FREQUENCY_FIELD_MAP.keys())
    train_paths = [str(path) for path in train_files]

    print("\n使用 Dask 统计训练集类别频次...")
    print(f"  参与统计的 train 文件数：{len(train_files)}")
    print(f"  统计字段：{source_columns}")

    train_ddf = dd.read_parquet(train_paths, columns=source_columns)

    frequency_maps: dict[str, dict] = {}

    for source_column in source_columns:
        print(f"  正在统计：{source_column} ...")

        # value_counts 返回各类别值及其出现次数
        count_series = train_ddf[source_column].value_counts().compute()
        frequency_maps[source_column] = count_series.to_dict()

        unique_count = len(frequency_maps[source_column])
        print(f"    唯一值数量：{unique_count:,}")

    return frequency_maps


def save_frequency_tables(frequency_maps: dict[str, dict]) -> None:
    """
    将频次映射保存为 CSV 文件。

    文件名示例：site_id_frequency.csv
    列：原始字段名、frequency
    """

    FREQUENCY_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n保存频次映射表...")

    for source_column, freq_dict in frequency_maps.items():
        table_df = pd.DataFrame(
            {
                source_column: list(freq_dict.keys()),
                "frequency": list(freq_dict.values()),
            }
        )

        output_path = FREQUENCY_TABLE_DIR / f"{source_column}_frequency.csv"
        table_df.to_csv(output_path, index=False)
        print(f"  已保存：{output_path}")


def add_frequency_features(
    dataframe: pd.DataFrame,
    frequency_maps: dict[str, dict],
) -> pd.DataFrame:
    """
    根据 train 映射表，为 DataFrame 添加频次特征列。

    训练集未见过的类别（含 test 新类别）填 0。
    保留全部已有列，不删除任何字段。
    """

    result = dataframe.copy()

    for source_column, freq_column in FREQUENCY_FIELD_MAP.items():
        if source_column not in result.columns:
            raise ValueError(f"输入数据缺少字段：{source_column}")

        result[freq_column] = (
            result[source_column]
            .map(frequency_maps[source_column])
            .fillna(0)
            .astype("int64")
        )

    return result


def process_parquet_file(
    input_path: Path,
    output_path: Path,
    frequency_maps: dict[str, dict],
) -> int:
    """读取单个分块，添加频次特征并保存。"""

    chunk = pd.read_parquet(input_path)
    featured_chunk = add_frequency_features(chunk, frequency_maps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_chunk.to_parquet(output_path, index=False, engine="pyarrow")

    return len(featured_chunk)


def process_dataset(
    parquet_files: list[Path],
    output_dir: Path,
    frequency_maps: dict[str, dict],
    dataset_name: str,
) -> int:
    """逐文件处理 train 或 test，返回累计行数。"""

    total_rows = 0

    print(f"\n开始写入 {dataset_name} 频次特征...")
    print(f"  输出目录：{output_dir}")
    print(f"  待处理文件数：{len(parquet_files)}")

    for input_path in parquet_files:
        output_path = output_dir / input_path.name
        print(f"  正在处理：{input_path.name} ...")

        rows = process_parquet_file(input_path, output_path, frequency_maps)
        total_rows += rows

        print(f"    完成，行数：{rows:,}")

    return total_rows


def main() -> None:
    """主流程：Dask 统计 train 频次 → 保存映射表 → 逐文件写入 train/test。"""

    if TEST_MODE:
        mode_label = "TEST"
        train_files_for_mapping = list_parquet_files(
            TRAIN_INPUT_DIR, TEST_TRAIN_MAX_FILES
        )
        train_files_to_process = train_files_for_mapping
        test_files_to_process = list_parquet_files(
            TEST_INPUT_DIR, TEST_TEST_MAX_FILES
        )
    else:
        mode_label = "FULL"
        train_files_for_mapping = list_parquet_files(TRAIN_INPUT_DIR, max_files=None)
        train_files_to_process = train_files_for_mapping
        test_files_to_process = list_parquet_files(TEST_INPUT_DIR, max_files=None)

    print("=" * 70)
    print("类别频次特征生成")
    print("=" * 70)
    print(f"当前模式：           {mode_label}")
    print(f"新增频次特征：       {list(FREQUENCY_FIELD_MAP.values())}")
    print(f"映射用 train 文件数：{len(train_files_for_mapping)}")
    print(f"待处理 train 文件数： {len(train_files_to_process)}")
    print(f"待处理 test 文件数：  {len(test_files_to_process)}")

    # 1. 仅用 train 建立频次映射（Dask，不一次性载入 pandas）
    frequency_maps = build_frequency_maps_with_dask(train_files_for_mapping)

    # 2. 保存映射表
    save_frequency_tables(frequency_maps)

    # 3. 逐文件应用到 train / test
    train_rows = process_dataset(
        parquet_files=train_files_to_process,
        output_dir=TRAIN_OUTPUT_DIR,
        frequency_maps=frequency_maps,
        dataset_name="train",
    )

    test_rows = process_dataset(
        parquet_files=test_files_to_process,
        output_dir=TEST_OUTPUT_DIR,
        frequency_maps=frequency_maps,
        dataset_name="test",
    )

    train_output_count = len(list(TRAIN_OUTPUT_DIR.glob("part-*.parquet")))
    test_output_count = len(list(TEST_OUTPUT_DIR.glob("part-*.parquet")))

    print("\n" + "=" * 70)
    print("频次特征生成完成")
    print("=" * 70)
    print(f"train 累计处理行数：{train_rows:,}")
    print(f"test 累计处理行数： {test_rows:,}")
    print(f"train 输出文件数：  {train_output_count}")
    print(f"test 输出文件数：   {test_output_count}")
    print(f"特征输出目录：")
    print(f"  {TRAIN_OUTPUT_DIR}")
    print(f"  {TEST_OUTPUT_DIR}")
    print(f"映射表目录：")
    print(f"  {FREQUENCY_TABLE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
