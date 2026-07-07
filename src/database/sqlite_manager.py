"""
百度 CTR 项目 — SQLite 数据库交互工具

提供 SQLiteManager 类，封装常用的数据库连接、查询与结果导出操作，
供 scripts/ 下的分析脚本或 notebooks 复用。

默认数据库：data/interim/baidu_ctr.db

用法示例：
    from src.database.sqlite_manager import SQLiteManager

    manager = SQLiteManager()
    manager.connect()
    df = manager.execute_query("SELECT COUNT(*) AS n FROM train_events")
    manager.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


# 项目默认 SQLite 数据库路径
DEFAULT_DB_PATH = Path("data/interim/baidu_ctr.db")


class SQLiteManager:
    """
    SQLite 数据库管理类。

    负责连接 data/interim/baidu_ctr.db，执行查询，
    并将结果转为 pandas DataFrame 或 CSV 文件。

    说明：
        - 本类提供的查询方法仅用于读取分析，不会修改 train_events 等原始表
        - 使用前需先调用 connect()；用完后建议调用 close() 释放连接
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """
        初始化管理器，保存数据库路径。

        参数：
            db_path — SQLite 数据库文件路径，默认为 data/interim/baidu_ctr.db
        """

        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """
        建立 sqlite3 数据库连接。

        若数据库文件不存在，抛出 FileNotFoundError。

        返回：
            已建立的 sqlite3.Connection 对象
        """

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"未找到 SQLite 数据库：{self.db_path}\n"
                "请先运行：python scripts/16_create_sqlite_db.py"
            )

        self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self) -> None:
        """关闭数据库连接；若尚未连接则不做任何操作。"""

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """
        确保已有可用连接。

        若尚未 connect()，则自动尝试连接（便于初学者少写一步）。
        """

        if self._conn is None:
            return self.connect()

        return self._conn

    def execute_query(
        self,
        sql: str,
        params: tuple | dict | None = None,
    ) -> pd.DataFrame:
        """
        执行 SELECT 查询，并返回 pandas DataFrame。

        参数：
            sql    — SQL 查询语句（建议仅使用 SELECT，避免修改原始表）
            params — 可选参数，用于占位符绑定（防止 SQL 注入）

        返回：
            查询结果的 DataFrame
        """

        conn = self._ensure_connected()
        return pd.read_sql_query(sql, conn, params=params)

    def execute_script(self, sql_file_path: str | Path) -> None:
        """
        读取并执行 .sql 文件中的 SQL 脚本。

        一个 SQL 文件可包含多条语句（以分号分隔），
        使用 sqlite3 的 executescript 依次执行。

        说明：
            - 适用于 sql/01_basic_analysis.sql 这类多段查询脚本
            - SELECT 语句的执行结果不会返回给 Python，仅完成执行
            - 请勿在此方法中运行会修改原始表的 DML/DDL 语句

        参数：
            sql_file_path — .sql 文件路径
        """

        path = Path(sql_file_path)

        if not path.exists():
            raise FileNotFoundError(f"未找到 SQL 文件：{path}")

        sql_text = path.read_text(encoding="utf-8")
        conn = self._ensure_connected()
        conn.executescript(sql_text)

    def get_table_names(self) -> list[str]:
        """
        返回数据库中所有用户表名（不含 sqlite_ 内部系统表）。

        返回：
            表名列表，例如 ['train_events', 'test_events']
        """

        sql = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """

        table_df = self.execute_query(sql)
        return table_df["name"].tolist()

    def get_table_columns(self, table_name: str) -> pd.DataFrame:
        """
        返回指定表的字段名称与字段类型。

        通过 PRAGMA table_info 读取元数据，返回列包括：
            cid, name, type, notnull, dflt_value, pk

        参数：
            table_name — 表名，例如 'train_events'
        """

        conn = self._ensure_connected()

        # 使用参数化查询表名，PRAGMA 不支持 ? 占位符，故先校验表是否存在
        existing_tables = self.get_table_names()
        if table_name not in existing_tables:
            raise ValueError(
                f"表不存在：{table_name}\n"
                f"当前数据库中的表：{existing_tables}"
            )

        return pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)

    def get_row_count(self, table_name: str) -> int:
        """
        返回指定表的总行数。

        参数：
            table_name — 表名

        返回：
            行数（整数）
        """

        existing_tables = self.get_table_names()
        if table_name not in existing_tables:
            raise ValueError(
                f"表不存在：{table_name}\n"
                f"当前数据库中的表：{existing_tables}"
            )

        sql = f'SELECT COUNT(*) AS row_count FROM "{table_name}"'
        result_df = self.execute_query(sql)
        return int(result_df.loc[0, "row_count"])

    def query_to_csv(
        self,
        sql: str,
        output_path: str | Path,
        params: tuple | dict | None = None,
    ) -> Path:
        """
        执行查询并将结果保存为 CSV 文件。

        若输出目录不存在，会自动创建。

        参数：
            sql         — SELECT 查询语句
            output_path — CSV 输出路径
            params      — 可选 SQL 参数

        返回：
            保存后的 CSV 文件路径
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_df = self.execute_query(sql, params=params)
        result_df.to_csv(output_path, index=False)

        return output_path
