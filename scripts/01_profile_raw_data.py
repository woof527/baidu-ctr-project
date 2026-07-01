from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

SAMPLE_ROWS = 100_000


def inspect_dataframe(
    dataframe: pd.DataFrame,
    name: str,
    has_target: bool,
) -> None:
    """检查一个数据样本的基本质量。"""

    print("\n" + "=" * 70)
    print(f"正在检查：{name}")
    print("=" * 70)

    print(f"样本行数：{len(dataframe):,}")
    print(f"字段数量：{len(dataframe.columns)}")

    print("\n字段名称：")
    print(dataframe.columns.tolist())

    print("\n数据类型：")
    print(dataframe.dtypes)

    print("\n缺失值数量：")
    missing = dataframe.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    if missing.sum() == 0:
        print("没有发现缺失值")

    if "id" in dataframe.columns:
        duplicate_ids = dataframe["id"].duplicated().sum()
        print(f"\n样本中的重复 id 数量：{duplicate_ids:,}")

    if "hour" in dataframe.columns:
        parsed_hour = pd.to_datetime(
            dataframe["hour"],
            format="%y%m%d%H",
            errors="coerce",
        )

        invalid_hour_count = parsed_hour.isna().sum()

        print(f"\n无效 hour 数量：{invalid_hour_count:,}")

        if invalid_hour_count == 0:
            print(f"最早时间：{parsed_hour.min()}")
            print(f"最晚时间：{parsed_hour.max()}")

    if has_target:
        if "click" not in dataframe.columns:
            print("\n错误：训练集中没有 click 字段")
        else:
            print("\nclick 取值：")
            print(sorted(dataframe["click"].dropna().unique().tolist()))

            print("\nclick 数量分布：")
            print(dataframe["click"].value_counts(dropna=False))

            invalid_click = ~dataframe["click"].isin([0, 1])
            print(f"\n非法 click 数量：{invalid_click.sum():,}")

            print(f"样本点击率：{dataframe['click'].mean():.4%}")


def main() -> None:
    print("开始读取少量样本，不会读取完整的 5.9GB 文件。")

    train_sample = pd.read_csv(
        TRAIN_PATH,
        nrows=SAMPLE_ROWS,
        dtype={
            "id": "string",
            "hour": "string",
        },
    )

    test_sample = pd.read_csv(
        TEST_PATH,
        nrows=SAMPLE_ROWS,
        dtype={
            "id": "string",
            "hour": "string",
        },
    )

    inspect_dataframe(
        dataframe=train_sample,
        name="train.csv",
        has_target=True,
    )

    inspect_dataframe(
        dataframe=test_sample,
        name="test.csv",
        has_target=False,
    )

    train_features = set(train_sample.columns) - {"click"}
    test_features = set(test_sample.columns)

    print("\n" + "=" * 70)
    print("训练集和测试集字段对比")
    print("=" * 70)

    print("训练集独有字段：", sorted(train_features - test_features))
    print("测试集独有字段：", sorted(test_features - train_features))

    if train_features == test_features:
        print("字段检查通过：训练集和测试集特征一致")
    else:
        print("字段检查未通过：训练集和测试集的特征不完全一致")


if __name__ == "__main__":
    main()