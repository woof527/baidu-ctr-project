"""
百度 CTR 项目 — SQLite 数据库搭建脚本（第二阶段）

功能：
    将清洗后的 train/test Parquet 分块逐个导入 SQLite 数据库，
    便于后续使用 SQL 做汇总分析与查询验证。

数据输入：
    data/processed/train/*.parquet  → 表 train_events
    data/processed/test/*.parquet   → 表 test_events

数据输出：
    TEST_MODE=True  → data/interim/baidu_ctr_sample.db（样本库，便于本地试跑）
    TEST_MODE=False → data/interim/baidu_ctr.db（全量库）

说明：
    - 每次只读取一个 Parquet 文件，处理后再 append 写入 SQLite，避免内存溢出
    - hour_dt 转为字符串；布尔质量标记字段转为 0/1
    - 不删除原始列，尽量保持与 Parquet 一致

用法：
    python scripts/16_create_sqlite_db.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 运行模式（改这里即可切换样本 / 全量）
# ---------------------------------------------------------------------------

# True：只导入少量 Parquet 分块，快速验证 SQL 流程
# False：导入全部 Parquet 分块，生成完整数据库
TEST_MODE = False

# 测试模式下读取的分块数量
TEST_TRAIN_MAX_FILES = 2
TEST_TEST_MAX_FILES = 1


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

TRAIN_PARQUET_DIR = Path("data/processed/train")
TEST_PARQUET_DIR = Path("data/processed/test")
INTERIM_DIR = Path("data/interim")

DB_PATH_SAMPLE = INTERIM_DIR / "baidu_ctr_sample.db"
DB_PATH_FULL = INTERIM_DIR / "baidu_ctr.db"

TRAIN_TABLE = "train_events"
TEST_TABLE = "test_events"

# 导入完成后，为 train_events 尝试创建索引的字段（不存在则跳过）
TRAIN_INDEX_COLUMNS = [
    "click",
    "hour",
    "banner_pos",
    "device_type",
    "site_category",
    "app_category",
]

# 写入 SQLite 前，若存在则转为 0/1 的布尔 / 质量标记字段
BOOLEAN_LIKE_COLUMNS = [
    "is_invalid_click",
    "is_dup_id_within_chunk",
    "is_low_volume",
]


def get_db_path() -> Path:
    """根据 TEST_MODE 返回目标数据库路径。"""

    if TEST_MODE:
        return DB_PATH_SAMPLE

    return DB_PATH_FULL


def list_parquet_files(parquet_dir: Path, max_files: int | None) -> list[Path]:
    """
    按文件名排序列出 Parquet 分块路径。

    参数：
        parquet_dir — train 或 test 的 Parquet 目录
        max_files   — 最多读取几个文件；None 表示读取全部

    返回：
        排序后的 Parquet 文件列表
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


def prepare_chunk_for_sqlite(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    对单个 Parquet 分块做 SQLite 写入前的基础字段处理。

    处理规则：
        1. hour_dt → 字符串（避免 SQLite 与时间戳类型兼容问题）
        2. is_invalid_click 等布尔字段 → 0/1 整数
        3. 其余字段保留，不随意删列
    """

    result = dataframe.copy()

    # hour_dt 转为字符串，便于 SQL 查询与展示
    if "hour_dt" in result.columns:
        hour_dt_series = pd.to_datetime(result["hour_dt"], errors="coerce")
        result["hour_dt"] = hour_dt_series.dt.strftime("%Y-%m-%d %H:%M:%S")
        # 解析失败的行记为 NULL（SQLite 中显示为空）
        result.loc[hour_dt_series.isna(), "hour_dt"] = None

    # 指定的布尔 / 质量标记字段统一转为 0/1
    for column in BOOLEAN_LIKE_COLUMNS:
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna(False)
                .astype(bool)
                .astype(int)
            )

    # 兜底：其他 bool 类型列也转为 0/1
    for column in result.columns:
        if column in BOOLEAN_LIKE_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(result[column]):
            result[column] = result[column].fillna(False).astype(int)

    return result


def import_parquet_files_to_table(
    conn: sqlite3.Connection,
    parquet_files: list[Path],
    table_name: str,
) -> int:
    """
    逐个读取 Parquet 文件并 append 写入 SQLite 表。

    第一个文件使用 if_exists='replace' 创建表；
    后续文件使用 if_exists='append' 追加数据。

    返回：
        本次导入的总行数
    """

    total_rows = 0

    for file_index, parquet_path in enumerate(parquet_files):
        chunk = pd.read_parquet(parquet_path)
        chunk = prepare_chunk_for_sqlite(chunk)

        if_exists = "replace" if file_index == 0 else "append"
        chunk.to_sql(table_name, conn, if_exists=if_exists, index=False)

        rows = len(chunk)
        total_rows += rows

        print(
            f"  [{table_name}] {parquet_path.name} → "
            f"写入 {rows:,} 行（累计 {total_rows:,} 行）"
        )

    return total_rows


def get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """查询指定表的行数。"""

    cursor = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    return int(cursor.fetchone()[0])


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    """读取指定表的字段名列表（按 SQLite 内部顺序）。"""

    cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
    return [row[1] for row in cursor.fetchall()]


def create_train_indexes(conn: sqlite3.Connection) -> None:
    """
    为 train_events 常用分析字段创建索引。

    若字段不存在则跳过，不中断脚本。
    """

    existing_columns = set(get_table_columns(conn, TRAIN_TABLE))

    print("\n正在为 train_events 创建索引...")

    for column in TRAIN_INDEX_COLUMNS:
        if column not in existing_columns:
            print(f"  跳过：字段不存在 → {TRAIN_TABLE}.{column}")
            continue

        index_name = f"idx_{TRAIN_TABLE}_{column}"
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{index_name}" '
            f'ON "{TRAIN_TABLE}" ("{column}")'
        )
        print(f"  已创建索引：{index_name}")


def print_database_summary(conn: sqlite3.Connection, db_path: Path) -> None:
    """在终端输出数据库基本信息。"""

    train_rows = get_table_row_count(conn, TRAIN_TABLE)
    test_rows = get_table_row_count(conn, TEST_TABLE)
    train_columns = get_table_columns(conn, TRAIN_TABLE)
    test_columns = get_table_columns(conn, TEST_TABLE)

    print("\n" + "=" * 70)
    print("SQLite 数据库导入完成")
    print("=" * 70)
    print(f"数据库路径：     {db_path.resolve()}")
    print(f"运行模式：       {'TEST_MODE（样本）' if TEST_MODE else '全量模式'}")
    print(f"{TRAIN_TABLE} 行数： {train_rows:,}")
    print(f"{TEST_TABLE} 行数：  {test_rows:,}")
    print(f"\n{TRAIN_TABLE} 字段（{len(train_columns)} 列）：")
    print(f"  {train_columns}")
    print(f"\n{TEST_TABLE} 字段（{len(test_columns)} 列）：")
    print(f"  {test_columns}")
    print("=" * 70)


def remove_existing_database(db_path: Path) -> None:
    """
    若目标数据库已存在则删除，避免 append 到旧表造成重复数据。

    仅删除本次将要写入的 db 文件，不影响其他目录。
    """

    if db_path.exists():
        db_path.unlink()
        print(f"已删除旧数据库：{db_path}")


def main() -> None:
    """主流程：确定模式 → 逐个导入 Parquet → 建索引 → 输出摘要。"""

    db_path = get_db_path()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        train_files = list_parquet_files(TRAIN_PARQUET_DIR, TEST_TRAIN_MAX_FILES)
        test_files = list_parquet_files(TEST_PARQUET_DIR, TEST_TEST_MAX_FILES)
        print("=" * 70)
        print("SQLite 导入 — 测试模式（TEST_MODE=True）")
        print("=" * 70)
        print(f"train 分块数：{len(train_files)}（最多 {TEST_TRAIN_MAX_FILES}）")
        print(f"test 分块数： {len(test_files)}（最多 {TEST_TEST_MAX_FILES}）")
    else:
        train_files = list_parquet_files(TRAIN_PARQUET_DIR, max_files=None)
        test_files = list_parquet_files(TEST_PARQUET_DIR, max_files=None)
        print("=" * 70)
        print("SQLite 导入 — 全量模式（TEST_MODE=False）")
        print("=" * 70)
        print(f"train 分块数：{len(train_files)}")
        print(f"test 分块数： {len(test_files)}")

    print(f"输出数据库：{db_path}\n")

    remove_existing_database(db_path)

    conn = sqlite3.connect(db_path)

    try:
        print(f"开始导入 {TRAIN_TABLE}...")
        import_parquet_files_to_table(conn, train_files, TRAIN_TABLE)

        print(f"\n开始导入 {TEST_TABLE}...")
        import_parquet_files_to_table(conn, test_files, TEST_TABLE)

        conn.commit()

        create_train_indexes(conn)
        conn.commit()

        print_database_summary(conn, db_path)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
