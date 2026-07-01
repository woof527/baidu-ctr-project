# 百度广告点击率预测项目 — 第一阶段数据处理报告

## 1. 项目背景

本项目基于广告曝光与点击日志，做点击率（CTR）预测和投放优化分析。第一阶段不急着建模，先把数据这条线走通：能读、能查、能洗、能验收，后面 EDA 和特征工程才有稳定输入。

当前阶段主要工作：

- 原始数据加载与抽样检查
- 数据质量排查
- 分块清洗与格式转换（CSV → Parquet）
- 清洗结果自动验收

---

## 2. 原始数据说明

原始数据放在 `data/raw/` 目录下，主要包括三个文件：

| 文件 | 说明 |
|------|------|
| `train.csv` | 训练集，含 `click` 标签（0/1） |
| `test.csv` | 测试集，不含 `click` 标签 |
| `sampleSubmission.csv` | 提交结果模板，含 `id` 和占位 `click` 列 |

数据量比较大。脚本 `01_profile_raw_data.py` 运行时提示 train 文件约 **5.9GB**，不适合用 Excel 或普通表格软件直接打开处理，需要用 Python 分块读取。

**字段概况（以 `outputs/01_raw_profile.txt` 为准）：**

- 训练集 24 列：比测试集多一列 `click`
- 测试集 23 列
- 除 `click` 外，训练集与测试集特征字段一致
- 主要字段包括：`id`、`hour`、广告位 `banner_pos`、站点/应用/设备相关字段，以及匿名特征 `C1`、`C14`–`C21` 等

---

## 3. 项目目录结构

| 目录 | 用途 |
|------|------|
| `data/raw/` | 原始 CSV，只读不改，保证可复现 |
| `data/processed/` | 清洗后的 Parquet、清洗报告等 |
| `scripts/` | 数据处理、EDA、验收等 Python 脚本 |
| `outputs/` | 脚本运行输出（日志、汇总表、验收报告等） |
| `reports/` | 项目阶段报告（本文档所在目录） |
| `models/` | 预留给后续模型文件（当前阶段尚未使用） |

---

## 4. 环境与工具

| 工具 | 本阶段作用 |
|------|-----------|
| **Python** | 数据处理和分析的主语言 |
| **Conda 环境 `baidu_ctr`** | 隔离项目依赖，避免和系统 Python 冲突 |
| **pandas** | 读取 CSV 样本、分块清洗、小表汇总 |
| **dask** | 对清洗后的大体量 Parquet 做惰性读取和聚合（验收、后续 EDA 会用到） |
| **pyarrow** | Parquet 读写引擎，配合 pandas / dask 使用 |
| **Git + GitHub** | 代码版本管理，脚本和报告可追踪、可协作 |

其他依赖（如 scikit-learn、matplotlib 等）已在 `requirements.txt` 中列出，主要供后续建模和可视化使用，第一阶段核心依赖是 pandas、dask、pyarrow。

---

## 5. 原始数据检查

**脚本：** `scripts/01_profile_raw_data.py`  
**输出：** `outputs/01_raw_profile.txt`

该脚本不会读完整训练文件，而是各抽取 **10 万行** 样本做快速摸底，主要检查：

- 字段名称与类型
- 缺失值情况
- 样本内是否有重复 `id`
- `hour` 能否按 `%y%m%d%H` 格式解析为有效时间
- 训练集 `click` 取值与分布（是否为 0/1）

**样本检查结论（详见 `outputs/01_raw_profile.txt`）：**

- 训练集、测试集样本中均未发现缺失值
- 样本内重复 `id` 为 0
- 样本内无效 `hour` 为 0
- 训练集样本中 `click` 仅出现 0 和 1，无非法值
- 训练集与测试集特征字段一致（训练集多 `click`）

说明：以上是**抽样结果**，全量数据以清洗和验收脚本的输出为准。

---

## 6. 数据清洗与转换

**脚本：** `scripts/02_clean_to_parquet.py`  
**输出：**

- `data/processed/train/*.parquet`
- `data/processed/test/*.parquet`
- `data/processed/cleaning_report.json`

### 6.1 基本策略

- **分块读取**：每次约 20 万行（`CHUNK_SIZE = 200_000`），避免内存撑爆
- **不修改原始 CSV**：只读 `data/raw/`，写出到 `data/processed/`
- **保守清洗**：本阶段不删行，读多少写多少
- **输出 Parquet**：列类型更稳定，后续 dask 读取更方便

### 6.2 主要处理内容

| 处理项 | 说明 |
|--------|------|
| 时间字段 | 从 `hour` 字符串解析出 `hour_dt`（格式 `%y%m%d%H`） |
| 数值字段 | 先按字符串读入，再 `to_numeric` 安全转换，超出整型范围记为缺失 |
| 质量标记 | 增加 `is_invalid_click`、`is_dup_id_within_chunk` 等检查字段 |
| 非法 click | 若有异常 click，按分块保存到 `pending_review/invalid_click/` 供人工查看 |

### 6.3 全量清洗结果（以 `data/processed/cleaning_report.json` 为准）

| 数据集 | 分块数 | 读入行数 | 写出中行数 |
|--------|--------|----------|------------|
| train | 203 | 40,428,967 | 40,428,967 |
| test | 23 | 4,577,464 | 4,577,464 |

行数守恒：读入与写出一致，没有丢行。

清洗报告中，`invalid_hour_count`、`invalid_click_count`、数值转换失败计数等均为 0（详见 JSON 文件）。

---

## 7. 清洗后数据验收

**脚本：** `scripts/15_validate_processed_data.py`  
**输出：**

- `outputs/15_processed_validation_report.txt`（文本报告）
- `outputs/eda_tables/processed_validation_summary.csv`（关键指标汇总）

验收使用 Dask 读取 Parquet，不会对 train 全量载入 pandas。

### 7.1 检查项

1. train / test 总行数
2. train 曝光量、点击量、整体 CTR、click 分布
3. train 含 `click`，test 不含 `click`
4. train 与 test 字段一致性（排除 `click`、`is_invalid_click`）
5. `hour`、`hour_dt` 是否存在
6. `hour_dt` 是否有缺失
7. train 的 `click` 是否仅为 0、1
8. test 行数是否与 `sampleSubmission.csv` 一致
9. test 的 `id` 顺序是否与提交模板一致

### 7.2 验收结论（以 `outputs/15_processed_validation_report.txt` 为准）

| 指标 | 结果 |
|------|------|
| train 行数 | 40,428,967 |
| test 行数 | 4,577,464 |
| train 整体 CTR | 16.98% |
| click 非法行数 | 0 |
| hour_dt 缺失 | train / test 均为 0 |
| 字段一致性 | 通过（排除 click、is_invalid_click） |
| test 与 sampleSubmission 行数 | 一致 |
| test 与 sampleSubmission id 顺序 | 一致 |

**最终状态：`VALIDATION PASSED`**

---

## 8. 阶段产出文件

| 类型 | 路径 |
|------|------|
| 原始数据检查脚本 | `scripts/01_profile_raw_data.py` |
| 清洗脚本 | `scripts/02_clean_to_parquet.py` |
| 验收脚本 | `scripts/15_validate_processed_data.py` |
| 清洗后训练集 | `data/processed/train/*.parquet` |
| 清洗后测试集 | `data/processed/test/*.parquet` |
| 原始数据检查输出 | `outputs/01_raw_profile.txt` |
| 清洗验收报告 | `outputs/15_processed_validation_report.txt` |
| 清洗过程统计 | `data/processed/cleaning_report.json` |

---

## 9. 当前结论

1. **原始数据可以正常读取。** 抽样检查未发现明显结构问题；全量文件体积较大，需用分块方式处理。
2. **清洗后的 Parquet 已通过验收。** 行数、字段、click 合法性、test 与提交模板对齐等检查均通过，可作为后续 EDA、特征工程和建模的标准输入。
3. **原始 CSV 未被修改。** 所有变换都在 `data/processed/` 下完成，流程可复现、可回滚。

---

## 10. 后续计划

第一阶段数据处理基本完成，后面还打算补充：

- **SQLite 数据库 + SQL 分析**：把部分汇总结果落到数据库，方便用 SQL 做查询和汇报
- **特征工程**：基于清洗后的 Parquet 构造训练特征
- **模型训练与评估**：CTR 预测模型搭建、调参和效果对比

以上工作会在后续脚本和报告中继续补充。
