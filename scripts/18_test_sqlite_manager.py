"""
百度 CTR 项目 — SQLiteManager 功能测试脚本

功能：
    测试 src/database/sqlite_manager.py 中的 SQLiteManager
    是否能正常连接数据库并执行查询。

数据库：
    data/interim/baidu_ctr.db

用法（在项目根目录运行）：
    python scripts/18_test_sqlite_manager.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 Python 搜索路径，以便导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.sqlite_manager import SQLiteManager


# 数据库路径
DB_PATH = Path("data/interim/baidu_ctr.db")

# 测试用 SQL：按 banner_pos 统计曝光量与 CTR
SQL_BANNER_POS = """
SELECT
    banner_pos,
    COUNT(*) AS impressions,
    AVG(click) AS ctr
FROM train_events
GROUP BY banner_pos
ORDER BY impressions DESC
"""


def main() -> None:
    """主流程：连接 → 查看表结构 → 执行查询 → 关闭连接。"""

    print("=" * 60)
    print("SQLiteManager 功能测试")
    print("=" * 60)
    print(f"数据库路径：{DB_PATH}\n")

    # 1. 创建 SQLiteManager 对象
    manager = SQLiteManager(db_path=DB_PATH)

    try:
        # 2. 连接数据库
        print("【1】连接数据库...")
        manager.connect()
        print("  连接成功\n")

        # 3. 输出所有表名
        print("【2】数据库中的表：")
        table_names = manager.get_table_names()
        for name in table_names:
            print(f"  - {name}")
        print()

        # 4. 输出 train_events 字段信息
        print("【3】train_events 字段信息：")
        columns_df = manager.get_table_columns("train_events")
        print(columns_df[["name", "type"]].to_string(index=False))
        print()

        # 5. 输出 train_events 总行数
        print("【4】train_events 总行数：")
        row_count = manager.get_row_count("train_events")
        print(f"  {row_count:,}\n")

        # 6 & 7. 执行 SQL 并打印结果
        print("【5】按 banner_pos 查询 CTR：")
        result_df = manager.execute_query(SQL_BANNER_POS)
        print(result_df.to_string(index=False))
        print()

        print("测试完成：SQLiteManager 工作正常。")

    finally:
        # 8. 关闭数据库连接
        manager.close()
        print("\n数据库连接已关闭。")


if __name__ == "__main__":
    main()
