"""
百度 CTR 项目 — 基础 SQL 分析脚本（第二阶段）

功能：
    连接 SQLite 数据库，对 train_events 表运行基础 CTR 汇总查询，
    将结果保存为 CSV，并生成可阅读的文本报告。

数据输入：
    data/interim/baidu_ctr.db（表 train_events）

数据输出：
    outputs/sql_tables/sql_*.csv
    outputs/17_basic_sql_analysis_report.txt

说明：
    - CTR 使用 AVG(click) 计算（click 为 0/1）
    - CSV 中 ctr 保留小数形式（如 0.1698）；TXT 报告中转为百分比便于阅读

用法：
    python scripts/17_basic_sql_analysis.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

DB_PATH = Path("data/interim/baidu_ctr.db")
TRAIN_TABLE = "train_events"

OUTPUT_SQL_DIR = Path("outputs/sql_tables")
REPORT_PATH = Path("outputs/17_basic_sql_analysis_report.txt")

# 各类别分析取 Top N
TOP_N_CATEGORY = 15

# CSV 输出文件
OUTPUT_FILES = {
    "overall": OUTPUT_SQL_DIR / "sql_overall_ctr.csv",
    "banner_pos": OUTPUT_SQL_DIR / "sql_banner_pos_ctr.csv",
    "device_type": OUTPUT_SQL_DIR / "sql_device_type_ctr.csv",
    "hour": OUTPUT_SQL_DIR / "sql_hour_ctr.csv",
    "site_category": OUTPUT_SQL_DIR / "sql_site_category_top15_ctr.csv",
    "app_category": OUTPUT_SQL_DIR / "sql_app_category_top15_ctr.csv",
}


# ---------------------------------------------------------------------------
# SQL 查询定义
# ---------------------------------------------------------------------------

# 一、整体 CTR
SQL_OVERALL_CTR = f"""
SELECT
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
"""

# 二、按 banner_pos
SQL_BANNER_POS_CTR = f"""
SELECT
    banner_pos,
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
GROUP BY banner_pos
ORDER BY impressions DESC
"""

# 三、按 device_type
SQL_DEVICE_TYPE_CTR = f"""
SELECT
    device_type,
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
GROUP BY device_type
ORDER BY impressions DESC
"""

# 四、按小时（从 hour 字符串 YYMMDDHH 中提取最后两位）
SQL_HOUR_CTR = f"""
SELECT
    substr(hour, 7, 2) AS hour_of_day,
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
GROUP BY hour_of_day
ORDER BY hour_of_day ASC
"""

# 五、按 site_category，曝光 Top 15
SQL_SITE_CATEGORY_TOP15 = f"""
SELECT
    site_category,
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
GROUP BY site_category
ORDER BY impressions DESC
LIMIT {TOP_N_CATEGORY}
"""

# 六、按 app_category，曝光 Top 15
SQL_APP_CATEGORY_TOP15 = f"""
SELECT
    app_category,
    COUNT(*) AS impressions,
    SUM(click) AS clicks,
    AVG(click) AS ctr
FROM {TRAIN_TABLE}
GROUP BY app_category
ORDER BY impressions DESC
LIMIT {TOP_N_CATEGORY}
"""


def validate_database() -> None:
    """
    检查 SQLite 数据库文件是否存在。

    若不存在，抛出 FileNotFoundError 并提示先运行导入脚本。
    """

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"未找到 SQLite 数据库：{DB_PATH}\n"
            "请先运行：python scripts/16_create_sqlite_db.py"
        )


def connect_database() -> sqlite3.Connection:
    """建立 SQLite 连接。"""

    validate_database()
    return sqlite3.connect(DB_PATH)


def run_query(conn: sqlite3.Connection, sql: str, step_name: str) -> pd.DataFrame:
    """
    执行 SQL 并用 pandas 读取结果为 DataFrame。

    参数：
        conn      — SQLite 连接
        sql       — 查询语句
        step_name — 步骤名称，用于终端打印

    返回：
        查询结果 DataFrame
    """

    print(f"正在执行：{step_name} ...")
    result_df = pd.read_sql_query(sql, conn)
    print(f"  完成：{step_name}（{len(result_df)} 行）")
    return result_df


def save_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    """
    保存查询结果为 CSV。

    ctr 列保持原始小数，不转为百分号字符串。
    """

    OUTPUT_SQL_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"  已保存：{output_path}")


def format_ctr_percent(ctr_value: float | int) -> str:
    """将 CTR 小数转为百分比字符串，用于 TXT 报告展示。"""

    return f"{float(ctr_value):.4%}"


def build_report(results: dict[str, pd.DataFrame]) -> str:
    """
    根据各 SQL 查询结果生成文本报告。

    TXT 报告中 CTR 以百分比形式展示，便于阅读。
    """

    lines: list[str] = [
        "=" * 70,
        "百度 CTR 项目 — 基础 SQL 分析报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据库：  {DB_PATH}",
        f"数据表：  {TRAIN_TABLE}",
        "=" * 70,
        "",
    ]

    # 一、整体 CTR
    overall_df = results["overall"]
    row = overall_df.iloc[0]
    lines.extend([
        "【一、整体 CTR】",
        "-" * 40,
        f"总曝光量 (impressions)：{int(row['impressions']):,}",
        f"总点击量 (clicks)：     {int(row['clicks']):,}",
        f"整体 CTR：              {format_ctr_percent(row['ctr'])}",
        "",
    ])

    # 二、banner_pos
    lines.extend([
        "【二、按 banner_pos 统计】（按 impressions 降序）",
        "-" * 40,
    ])
    for _, r in results["banner_pos"].iterrows():
        lines.append(
            f"  banner_pos={r['banner_pos']:>3}  "
            f"impressions={int(r['impressions']):>12,}  "
            f"clicks={int(r['clicks']):>10,}  "
            f"ctr={format_ctr_percent(r['ctr'])}"
        )
    lines.append("")

    # 三、device_type
    lines.extend([
        "【三、按 device_type 统计】（按 impressions 降序）",
        "-" * 40,
    ])
    for _, r in results["device_type"].iterrows():
        lines.append(
            f"  device_type={int(r['device_type']):>3}  "
            f"impressions={int(r['impressions']):>12,}  "
            f"clicks={int(r['clicks']):>10,}  "
            f"ctr={format_ctr_percent(r['ctr'])}"
        )
    lines.append("")

    # 四、hour_of_day
    lines.extend([
        "【四、按 hour_of_day 统计】（从 hour 字段提取，按小时升序）",
        "-" * 40,
    ])
    for _, r in results["hour"].iterrows():
        lines.append(
            f"  hour={r['hour_of_day']:>2}  "
            f"impressions={int(r['impressions']):>12,}  "
            f"clicks={int(r['clicks']):>10,}  "
            f"ctr={format_ctr_percent(r['ctr'])}"
        )
    lines.append("")

    # 五、site_category Top 15
    lines.extend([
        f"【五、按 site_category 统计】（曝光 Top {TOP_N_CATEGORY}）",
        "-" * 40,
    ])
    for _, r in results["site_category"].iterrows():
        lines.append(
            f"  site_category={str(r['site_category']):>20}  "
            f"impressions={int(r['impressions']):>12,}  "
            f"ctr={format_ctr_percent(r['ctr'])}"
        )
    lines.append("")

    # 六、app_category Top 15
    lines.extend([
        f"【六、按 app_category 统计】（曝光 Top {TOP_N_CATEGORY}）",
        "-" * 40,
    ])
    for _, r in results["app_category"].iterrows():
        lines.append(
            f"  app_category={str(r['app_category']):>20}  "
            f"impressions={int(r['impressions']):>12,}  "
            f"ctr={format_ctr_percent(r['ctr'])}"
        )
    lines.append("")

    lines.extend([
        "=" * 70,
        "CSV 结果目录：outputs/sql_tables/",
        "=" * 70,
    ])

    return "\n".join(lines)


def save_report(report_text: str) -> None:
    """保存文本报告。"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\n文本报告已保存：{REPORT_PATH}")


def main() -> None:
    """主流程：连接数据库 → 执行 6 组 SQL → 保存 CSV → 生成 TXT 报告。"""

    print("=" * 70)
    print("基础 SQL 分析")
    print("=" * 70)
    print(f"数据库：{DB_PATH}\n")

    conn = connect_database()

    try:
        results: dict[str, pd.DataFrame] = {}

        results["overall"] = run_query(conn, SQL_OVERALL_CTR, "整体 CTR")
        save_csv(results["overall"], OUTPUT_FILES["overall"])

        results["banner_pos"] = run_query(conn, SQL_BANNER_POS_CTR, "按 banner_pos")
        save_csv(results["banner_pos"], OUTPUT_FILES["banner_pos"])

        results["device_type"] = run_query(conn, SQL_DEVICE_TYPE_CTR, "按 device_type")
        save_csv(results["device_type"], OUTPUT_FILES["device_type"])

        results["hour"] = run_query(conn, SQL_HOUR_CTR, "按 hour_of_day")
        save_csv(results["hour"], OUTPUT_FILES["hour"])

        results["site_category"] = run_query(
            conn, SQL_SITE_CATEGORY_TOP15, "按 site_category Top 15"
        )
        save_csv(results["site_category"], OUTPUT_FILES["site_category"])

        results["app_category"] = run_query(
            conn, SQL_APP_CATEGORY_TOP15, "按 app_category Top 15"
        )
        save_csv(results["app_category"], OUTPUT_FILES["app_category"])

        report_text = build_report(results)
        save_report(report_text)

        print("\n全部 SQL 分析已完成。")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
