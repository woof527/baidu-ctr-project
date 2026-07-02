# 数据处理报告

**项目名称：** 百度广告点击率预测与投放优化分析  
**报告阶段：** 第一阶段 — 数据工程  
**主要依据：** `scripts/01_profile_raw_data.py`、`scripts/02_clean_to_parquet.py`、`scripts/15_validate_processed_data.py` 及对应输出文件

---

## 1. 报告目的

本报告用于向导师汇报项目第一阶段的数据工程工作，重点说明数据处理的原因、方法与结果，而非展示具体代码实现。

本阶段已完成以下工作：

- 原始数据质量初步检查
- 大规模 CSV 的分块清洗与 Parquet 转换
- 清洗后数据的自动化质量验收

本报告旨在回答四个问题：**为什么要做这些处理、具体做了哪些处理、处理后数据质量是否合格、处理后数据能否支撑后续 EDA、SQL 分析和建模。**

---

## 2. 数据文件说明

本项目使用的原始数据位于 `data/raw/`，主要包括三个文件：

| 文件 | 说明 |
|------|------|
| `data/raw/train.csv` | 训练集，包含广告曝光记录及点击标签 `click` |
| `data/raw/test.csv` | 测试集，结构与训练集基本一致，但**不包含** `click` 字段 |
| `data/raw/sampleSubmission.csv` | 提交结果模板，用于对照 test 集行数与 `id` 顺序 |

**字段差异说明：**

- `train.csv` 共 **24** 个字段，比 `test.csv` 多一列 `click`（取值为 0 或 1，表示未点击/点击）。
- `test.csv` 共 **23** 个字段，除缺少 `click` 外，其余特征字段与训练集一致。

主要字段包括：`id`、`hour`、广告位 `banner_pos`、站点/应用/设备相关字段（如 `site_id`、`app_id`、`device_id` 等），以及匿名数值特征 `C1`、`C14`–`C21`。

原始 `train.csv` 体积较大。运行 `scripts/01_profile_raw_data.py` 时提示文件约 **5.9GB**，不适合用 Excel 或一次性全量读入内存的方式处理。

---

## 3. 原始数据初步检查

**使用脚本：** `scripts/01_profile_raw_data.py`  
**输出文件：** `outputs/01_raw_profile.txt`

该脚本对 `train.csv` 和 `test.csv` 各读取 **10 万行** 样本进行检查，不会读取完整大文件。以下为样本检查结果（全量结论以清洗与验收脚本输出为准）。

### 3.1 字段数量与一致性

| 数据集 | 字段数量 | 说明 |
|--------|----------|------|
| train | 24 | 含 `click` |
| test | 23 | 不含 `click` |

样本对比结果显示：训练集与测试集**除 `click` 外字段一致**，未发现测试集独有字段或训练集独有特征字段（`click` 除外）。

### 3.2 click 标签检查（train 样本）

- `click` 取值仅出现 **0** 和 **1**
- 样本中非法 `click` 数量为 **0**
- 样本点击率约为 **17.49%**（10 万行样本统计）

### 3.3 hour 字段检查（样本）

- 使用格式 `%y%m%d%H` 解析 `hour` 字段
- train、test 样本中无效 `hour` 数量均为 **0**
- 说明原始时间字符串在样本范围内可以正常解析

### 3.4 缺失值与重复 id（样本）

- train、test 样本中**未发现缺失值**
- 样本内重复 `id` 数量均为 **0**

**小结：** 原始数据在样本层面结构清晰、字段关系合理，具备进一步全量清洗的基础。但样本检查不能替代全量验收，后续仍以 Parquet 验收结果为准。

---

## 4. 数据清洗目标

原始 CSV 虽然可以直接打开查看结构，但尚不适合作为后续分析的标准输入，主要原因如下：

1. **文件体积大**：`train.csv` 约 5.9GB，无法一次性读入内存做稳定处理。
2. **字段类型不统一**：原始读取时部分数值列以字符串形式存在，存在脏值导致转换失败的风险。
3. **时间字段未结构化**：`hour` 为字符串，无法直接用于按小时、按日期等维度分析。
4. **后续环节需要标准化输入**：EDA、SQLite 汇总分析、特征工程和 CTR 建模都需要类型稳定、格式统一、可分批读取的数据。

因此，本阶段清洗的目标是：在**不修改原始 CSV** 的前提下，将 train/test 转换为类型规范、可复用、可验收的 Parquet 数据集。

---

## 5. 分块读取与处理策略

**使用脚本：** `scripts/02_clean_to_parquet.py`

### 5.1 为什么采用分块读取

- `train.csv` 数据量较大，全量读入 pandas 容易导致内存不足。
- 分块读取（`CHUNK_SIZE = 200_000`，即每块约 20 万行）可以控制单次内存占用。
- 每个 chunk 独立清洗并写出为 Parquet 分块，适合大规模数据的工程化处理。

### 5.2 全量处理结果

根据 `data/processed/cleaning_report.json`：

| 数据集 | 处理分块数 | 读入行数 | 写出中行数 |
|--------|------------|----------|------------|
| train | 203 | 40,428,967 | 40,428,967 |
| test | 23 | 4,577,464 | 4,577,464 |

**读入行数与写出中行数完全一致**，说明清洗阶段未删除任何记录，保证了样本总量不变。

---

## 6. 字段类型转换策略

清洗脚本在读取 CSV 时先将各列按字符串读入，再在分块内进行安全类型转换，主要策略如下：

| 字段类型 | 代表字段 | 转换方式 | 目的 |
|----------|----------|----------|------|
| 标签字段 | `click` | 可空 Int8 | 减小内存占用，明确其为 0/1 分类标签 |
| 小整数特征 | `banner_pos`、`device_type`、`device_conn_type` | 可空 Int16 | 统一数值类型，便于统计与建模 |
| 匿名数值特征 | `C1`、`C14`–`C21` | 可空 Int32 | 统一数值类型，便于后续特征处理 |
| 标识与文本字段 | `id`、`site_id`、`app_id`、`device_id`、`site_category` 等 | 保持字符串 | 避免将高基数 ID 误当作连续数值变量 |

**转换原则：**

- 使用 `pd.to_numeric(errors="coerce")` 做安全转换，无法转换的值记为缺失。
- 超出目标整型可表示范围的值也会记为缺失，并在清洗报告中统计。
- 全量清洗结果显示：各数值字段的转换失败计数和超范围计数均为 **0**（见 `cleaning_report.json`）。

---

## 7. 时间字段处理

### 7.1 原始格式

原始 `hour` 字段为字符串，格式类似 `YYMMDDHH`（例如 `14102100` 表示 2014-10-21 00:00）。清洗脚本使用格式 `%y%m%d%H` 进行解析。

### 7.2 新增字段

清洗过程中新增 **`hour_dt`** 字段，将 `hour` 解析为标准时间戳，便于后续：

- 提取日期、小时等时间特征
- 按小时、按天做 CTR 汇总分析
- 在 SQLite 或 pandas 中进行时间维度统计

### 7.3 解析质量

全量清洗报告（`cleaning_report.json`）显示：

- train 的 `invalid_hour_count` = **0**
- test 的 `invalid_hour_count` = **0**

处理后数据验收（第 10 节）也确认 train/test 的 `hour_dt` 均无缺失。

---

## 8. 重复值与异常值处理说明

本阶段清洗遵循**保守处理、尽量保留样本**的原则，没有随意删除数据。

| 情况 | 处理方式 |
|------|----------|
| 块内重复 `id` | 增加标记字段 `is_dup_id_within_chunk`，不删行 |
| 非法 `click` | 增加标记字段 `is_invalid_click`；若有异常，按分块导出至 `pending_review/invalid_click/` 供人工查看 |
| 跨分块重复 `id` | 本阶段**未做全局去重检查**（清洗报告 notes 中已说明），需后续专门分析 |
| 删行策略 | **零删行**：读多少写多少 |

全量清洗结果：

- train/test 的 `dup_id_within_chunk_count` 均为 **0**
- train 的 `invalid_click_count` 为 **0**

需要说明的是：`is_invalid_click` 和 `is_dup_id_within_chunk` 属于**数据质量检查字段**，后续建模时不应直接作为训练特征使用。

---

## 9. Parquet 转换结果

清洗完成后，数据输出至以下目录：

| 输出路径 | 内容 |
|----------|------|
| `data/processed/train/` | 清洗后的训练集 Parquet 分块（`part-XXXX.parquet`） |
| `data/processed/test/` | 清洗后的测试集 Parquet 分块 |
| `data/processed/cleaning_report.json` | 清洗过程统计与配置记录 |

**选择 Parquet 格式的原因：**

- 读取速度通常优于 CSV，适合反复做 EDA 和建模实验
- 列类型在文件中固化，避免每次读入重复推断类型
- 支持按列读取，配合 Dask 等工具可高效处理大规模数据
- 文件体积相对 CSV 更紧凑，便于本地管理与后续分析

---

## 10. 处理后数据质量验收

**使用脚本：** `scripts/15_validate_processed_data.py`  
**输出文件：**

- `outputs/15_processed_validation_report.txt`
- `outputs/eda_tables/processed_validation_summary.csv`

验收脚本使用 **Dask** 读取 Parquet，不会将完整 train 一次性载入 pandas，适合大体量数据的自动化检查。

### 10.1 验收项目与结果

| 检查项 | 结果 | 说明 |
|--------|------|------|
| train 行数 | 40,428,967 | 与清洗报告一致 |
| test 行数 | 4,577,464 | 与清洗报告一致 |
| train 是否含 `click` | 通过 | train 包含 click 字段 |
| test 是否不含 `click` | 通过 | test 不包含 click 字段 |
| click 取值合法性 | 通过 | 仅含 0、1；非法值 0 行；缺失 0 行 |
| `hour` / `hour_dt` 存在性 | 通过 | train、test 均包含两字段 |
| `hour_dt` 解析失败 | 通过 | train、test 缺失均为 0 |
| 字段一致性 | 通过 | 排除 `click`、`is_invalid_click` 后，train 与 test 均为 25 列且一致 |
| test 行数 vs sampleSubmission | 通过 | 均为 4,577,464 行 |
| test id 顺序 vs sampleSubmission | 通过 | 按 Parquet 分块逐块比对，完全一致 |

### 10.2 训练集核心统计（验收输出）

| 指标 | 数值 |
|------|------|
| 曝光量（impressions） | 40,428,967 |
| 点击量（clicks） | 6,865,066 |
| 整体 CTR | 16.98% |
| click = 0 | 33,563,901 |
| click = 1 | 6,865,066 |

### 10.3 验收结论

`outputs/15_processed_validation_report.txt` 最终结论为：

**`VALIDATION PASSED`**

`processed_validation_summary.csv` 中 `validation_status` 为 **PASSED**，`warning_count` 为 **0**。

**结论：** 处理后的 Parquet 数据已通过基本质量验收，字段结构、标签合法性、时间解析、test 与提交模板对齐等关键项目均符合预期，**可以作为后续 EDA、SQLite 分析、特征工程和模型训练的数据基础。**

---

## 11. 当前阶段仍需注意的问题

尽管第一阶段数据处理已通过验收，后续分析仍需注意以下几点：

1. **内存与计算资源**  
   训练集约 4,042 万行，后续全量特征处理或模型训练仍需采用分块、Dask 或采样策略，不宜默认一次性读入内存。

2. **高基数字段的处理**  
   `site_id`、`app_id`、`device_id` 等字段基数高，后续建模不宜直接简单 one-hot，需要结合频次截断、目标编码、哈希编码等方法。

3. **全局重复 id 尚未检查**  
   当前仅标记了分块内重复 id，尚未做跨分块或全量重复 id 检查，后续如需严格样本唯一性，应补充专门分析。

4. **本阶段尚未做特征工程**  
   当前清洗属于工程标准化处理，主要解决“能读、能洗、能验”的问题；距离可直接建模的特征矩阵仍有后续工作。

5. **原始检查基于样本**  
   `01_raw_profile.txt` 为 10 万行样本结果，全量层面的结论以清洗报告和验收报告为准。

---

## 12. 第一阶段小结

本阶段围绕“把原始大文件变成可分析、可验收的标准数据”这一目标，已完成以下工作：

| 工作项 | 完成情况 |
|--------|----------|
| Python 数据处理环境搭建 | 已完成 |
| 原始数据质量检查 | 已完成（`01_profile_raw_data.py`） |
| 大规模 CSV 分块读取 | 已完成（每块约 20 万行） |
| 字段类型转换 | 已完成（数值/字符串分类处理） |
| 时间字段解析 | 已完成（新增 `hour_dt`） |
| Parquet 标准化转换 | 已完成（train/test 共 226 个分块） |
| 处理后数据质量验收 | 已完成（`VALIDATION PASSED`） |

**阶段成果：**

- 原始 CSV 保持不变，处理流程可复现。
- 清洗后的 Parquet 数据质量合格，已通过自动化验收。
- 数据已具备进入第二阶段的条件，包括：多维 EDA、SQLite 汇总分析、特征字典整理、模型输入构建，以及逻辑回归、LightGBM、XGBoost 等 CTR 预测模型的训练与对比。

---

**相关文件索引**

| 类型 | 路径 |
|------|------|
| 原始数据检查脚本 | `scripts/01_profile_raw_data.py` |
| 原始数据检查输出 | `outputs/01_raw_profile.txt` |
| 数据清洗脚本 | `scripts/02_clean_to_parquet.py` |
| 清洗过程报告 | `data/processed/cleaning_report.json` |
| 数据验收脚本 | `scripts/15_validate_processed_data.py` |
| 数据验收报告 | `outputs/15_processed_validation_report.txt` |
| 验收指标汇总 | `outputs/eda_tables/processed_validation_summary.csv` |
