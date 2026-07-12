# 特征字典

**项目名称：** 百度广告点击率预测与投放优化分析  
**文档阶段：** 第二阶段 — 特征工程（第三版）  
**数据基础：** 清洗后 Parquet / SQLite；特征文件 `data/features/basic/`、`data/features/frequency/`、`data/features/historical/`、`data/features/target_encoded/`

> 说明：本文档记录原始字段、已实现衍生特征及初步验证状态。  
> 特征有效性结论以 `outputs/21_feature_validation_report.txt` 为准；文中不将任何特征表述为「一定有效」。

---

## 1. 目标变量

| 字段名 | 数据类型 | 含义 | 用途 | 是否作为模型输入 |
|--------|----------|------|------|------------------|
| `click` | 整型（0/1） | 用户是否点击广告。0 表示未点击，1 表示点击 | 训练集标签，用于监督学习；CTR 分析的核心因变量 | **否**（仅作标签，不能泄漏到特征中） |

**补充说明：**

- `click` 仅存在于训练集（`train.csv` / `train_events`）。
- 测试集（`test.csv` / `test_events`）**不包含** `click` 字段。
- 清洗后 `click` 转为可空 Int8，全量验收结果显示取值仅为 0 或 1（详见 `outputs/15_processed_validation_report.txt`）。

---

## 2. 原始特征

以下字段来自原始 CSV，经 `scripts/02_clean_to_parquet.py` 清洗后写入 Parquet / SQLite。  
表中「是否直接用于模型」指**是否建议不经处理直接入模**；多数类别字段和高基数 ID 需后续编码或衍生后再使用。

### 2.1 标识字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `id` | 标识 | 字符串 | 样本唯一标识，用于关联与提交对齐 | 否（一般不作为预测特征） |

### 2.2 时间字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `hour` | 时间 | 字符串 | 原始时间编码，格式为 `YYMMDDHH`（如 `14102100` 表示 2014-10-21 00:00） | 否（建议使用时间衍生特征） |
| `hour_dt` | 时间 | 时间戳（清洗新增） | 由 `hour` 解析得到的标准时间字段 | 否（建议使用 `hour_of_day` 等衍生字段） |

### 2.3 广告位字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `banner_pos` | 广告位 | 整型（Int16） | 广告位置数值编码，不解释为具体页面位置 | 视建模方案而定 |

### 2.4 网站字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `site_id` | 网站 | 字符串 | 网站匿名 ID，高基数类别字段 | 否（可配合频次特征使用） |
| `site_domain` | 网站 | 字符串 | 网站域名匿名编码 | 否 |
| `site_category` | 网站 | 字符串 | 网站类别匿名编码，不解释为具体行业或网站名称 | 视编码方式而定 |

### 2.5 应用字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `app_id` | 应用 | 字符串 | 应用匿名 ID，高基数类别字段 | 否（可配合频次特征使用） |
| `app_domain` | 应用 | 字符串 | 应用域名匿名编码 | 否 |
| `app_category` | 应用 | 字符串 | 应用类别匿名编码，不解释为具体应用名称 | 视编码方式而定 |

### 2.6 设备字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `device_id` | 设备 | 字符串 | 设备匿名 ID，高基数类别字段 | 否 |
| `device_ip` | 设备 | 字符串 | 设备 IP 匿名编码 | 否 |
| `device_model` | 设备 | 字符串 | 设备型号匿名编码 | 否（可配合频次特征使用） |
| `device_type` | 设备 | 整型（Int16） | 设备类型数值编码，不解释为具体设备名称 | 视建模方案而定 |
| `device_conn_type` | 设备 | 整型（Int16） | 设备网络连接类型编码 | 视建模方案而定 |

### 2.7 匿名字段

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `C1` | 匿名特征 | 整型（Int32） | 匿名数值或类别字段，具体业务含义未知 | 视建模方案而定 |
| `C14`–`C21` | 匿名特征 | 整型（Int32） | 同上 | 视建模方案而定 |

### 2.8 数据质量标记字段（清洗新增，非建模特征）

| 原始字段名 | 字段类别 | 数据类型 | 说明 | 是否直接用于模型 |
|------------|----------|----------|------|------------------|
| `is_invalid_click` | 质量检查 | 布尔 / 0-1 | 标记 `click` 是否非法，仅 train 存在 | **否** |
| `is_dup_id_within_chunk` | 质量检查 | 布尔 / 0-1 | 标记分块内是否重复 `id` | **否** |

---

## 3. 时间衍生特征（已实现）

**生成脚本：** `scripts/19_build_basic_features.py`  
**输出路径：** `data/features/basic/train/`、`data/features/basic/test/`

| 特征名 | 生成来源 | 含义 | 数据类型 | 状态 |
|--------|----------|------|----------|------|
| `hour_of_day` | `hour_dt` | 一天中的第几小时，取值 0—23 | 整型 | **已实现** |
| `day` | `hour_dt` | 日期中的「日」（1—31） | 整型 | **已实现** |
| `day_of_week` | `hour_dt` | 星期几（0=周一，6=周日） | 整型 | **已实现** |
| `is_weekend` | `day_of_week` | 周六、周日为 1，其余为 0 | 整型 | **已实现** |

**设计目的：** 描述曝光发生的时间规律，支撑 CTR 时间维度分析与建模。

**初步验证：** 已完成初步特征质量检查；部分时间特征在样本上表现出一定 CTR 差异，**值得进入后续模型验证**，最终有效性待模型阶段确认（详见第 7 节）。

---

## 4. 类别频次特征（已实现）

**生成脚本：** `scripts/20_build_frequency_features.py`  
**输出路径：** `data/features/frequency/train/`、`data/features/frequency/test/`  
**映射表：** `outputs/feature_tables/*_frequency.csv`

| 特征名 | 来源字段 | 含义 | 数据类型 | 状态 |
|--------|----------|------|----------|------|
| `site_id_freq` | `site_id` | 该 `site_id` 在训练集中出现的次数 | 整型 | **已实现** |
| `site_category_freq` | `site_category` | 该网站类别在训练集中出现的次数 | 整型 | **已实现** |
| `app_id_freq` | `app_id` | 该 `app_id` 在训练集中出现的次数 | 整型 | **已实现** |
| `app_category_freq` | `app_category` | 该应用类别在训练集中出现的次数 | 整型 | **已实现** |
| `device_model_freq` | `device_model` | 该设备型号在训练集中出现的次数 | 整型 | **已实现** |

### 4.1 生成方法说明

| 项目 | 说明 |
|------|------|
| 统计范围 | **仅基于训练集**（`data/features/basic/train/`）统计各类别出现次数 |
| 统计方式 | 使用 Dask 对 train 做 `value_counts`，生成映射表后再应用到 train / test |
| 映射规则 | 例如 `site_id=A` 在 train 中出现 100,000 次，则所有 `site_id=A` 样本的 `site_id_freq=100000` |
| test 未见类别 | 训练集中未出现的类别，频次填 **0** |
| 是否使用 click | **否**。频次统计不使用 `click`，不属于目标泄漏特征 |
| 低频处理 | 不删除低频类别，保留完整映射信息 |

**设计目的：** 对高基数类别字段做数值化压缩，避免直接 one-hot 维度过高，同时保留类别「热度」信息。

**初步验证：** 部分频次特征与 `click` 的 Pearson 相关绝对值相对较高，**有一定区分潜力**，但仍需结合验证集 AUC、LogLoss 和模型重要性进一步确认（详见第 7 节）。

---

## 5. 交叉特征（已实现）

**生成脚本：** `scripts/19_build_basic_features.py`  
**输出路径：** `data/features/basic/` → 经 `scripts/20_build_frequency_features.py` 传递至 `data/features/frequency/`

| 特征名 | 组合来源 | 实现方式 | 数据类型 | 状态 |
|--------|----------|----------|----------|------|
| `banner_device_cross` | `banner_pos` × `device_type` | 字符串拼接，如 `"1_1"` | 字符串 | **已实现** |
| `hour_banner_cross` | `hour_of_day` × `banner_pos` | 字符串拼接，如 `"21_1"` | 字符串 | **已实现** |
| `site_device_cross` | `site_category` × `device_type` | 字符串拼接，如 `"50e219e0_1"` | 字符串 | **已实现** |

**设计目的：** 刻画两个维度之间的联合效应，与 EDA / SQL 交叉分析方向一致（`11`/`12`/`13` 号脚本、`sql/01_basic_analysis.sql`）。

**初步验证：** 交叉特征在样本 CTR 区分度分析中普遍有一定 spread，**值得进入后续模型验证**；高基数组合在验证报告中按曝光 Top 20 类别展示，最终是否入模待模型阶段确认。

---

## 6. 特征使用注意事项

1. **高基数字段不宜直接 one-hot**  
   `site_id`、`app_id`、`device_id`、`device_model` 等唯一值数量大，直接独热编码易导致维度过高与过拟合。

2. **匿名编码字段不做业务含义解释**  
   `site_category`、`app_category`、`banner_pos`、`device_type` 及 `C1`、`C14`–`C21` 等均为匿名编码，不应随意解释为具体行业、页面位置或设备品牌。

3. **`click` 不能作为输入特征**  
   `click` 是预测目标。频次特征已明确不使用 `click`；若后续做目标编码，必须在交叉验证框架内完成。

4. **test 集没有 `click`**  
   频次映射等统计量仅用 train 拟合，再一致应用到 test。

5. **频次特征不是目标编码**  
   当前 `*_freq` 为出现次数，不含 click 信息；与基于标签的目标编码不同，泄漏风险较低，但仍需在建模阶段复核。

6. **质量检查字段不得入模**  
   `is_invalid_click`、`is_dup_id_within_chunk` 仅用于数据质量分析。

7. **Pearson 相关接近 0 不代表特征无用**  
   线性相关弱时，树模型仍可能学到非线性关系；不能据此排除特征。

8. **不要把初步验证结论当作最终结论**  
   当前仅完成 train 样本上的初步检查，最终是否保留特征需结合验证集 AUC、LogLoss 和模型重要性判断。

---

## 7. 特征验证状态

**验证脚本：** `scripts/21_validate_features.py`  
**验证数据：** `data/features/frequency/train/`（train 随机样本，未使用 test）  
**报告与明细：** `outputs/21_feature_validation_report.txt`、`outputs/feature_validation/`

### 7.1 已完成的检查项

| 检查类型 | 说明 | 输出文件 |
|----------|------|----------|
| 特征质量检查 | 缺失率、唯一值、常数列；数值型 min/max/mean 等 | `feature_quality_summary.csv` |
| 类别 CTR 区分度 | 对时间特征与交叉特征按类别统计 impressions、clicks、ctr | `categorical_ctr_summary.csv`、`categorical_feature_spread.csv` |
| 数值相关性 | 数值特征与 `click` 的 Pearson 相关（初步参考） | `numeric_feature_correlation.csv` |

### 7.2 初步结论（保守表述）

- **已完成初步特征质量检查。** 在当前 50 万行 train 样本上，未发现缺失率超过 1% 的特征，未发现常数列（详见验证报告）。
- **已完成类别特征 CTR 区分度分析。** 部分特征在不同取值之间表现出一定 CTR 差异（`ctr_spread` 指标），可认为**有一定区分度**，但仍属样本级初步观察。
- **已完成数值特征与 click 的初步相关性检查。** 部分频次特征线性相关绝对值相对较高，时间特征线性相关较弱；**不能据此认定特征一定有效或无效**。
- **当前结果属于初步分析。** 上述特征整体**值得进入后续模型验证**，但是否最终保留，需结合验证集 **AUC、LogLoss** 和**模型特征重要性**确认。

### 7.3 验证范围与限制

- 全量 train 超过 4000 万行，验证基于多 Parquet 分块随机样本（`SAMPLE_SIZE=500000`，`RANDOM_STATE=42`），存在抽样误差。
- 未使用 test 数据做有效性判断，避免信息泄漏。
- 未做目标编码，未做模型训练。

---

## 8. 特征数据流概览

```
data/processed/train|test/*.parquet
        ↓  scripts/19_build_basic_features.py
data/features/basic/train|test/
        ↓  scripts/20_build_frequency_features.py
data/features/frequency/train|test/
        ↓  scripts/21_validate_features.py
outputs/feature_validation/ 、 outputs/21_feature_validation_report.txt

data/features/frequency/train/
        ↓  scripts/22_time_split.py
data/model_input/train|valid|holdout/
        ↓  scripts/23_build_historical_features.py
data/features/historical/train|valid|holdout/
        ↓  scripts/24_build_target_encoding.py
data/features/target_encoded/train|valid|holdout/
        ↓  scripts/25_validate_advanced_features.py
outputs/advanced_feature_validation_summary.csv
outputs/advanced_feature_column_stats.csv
outputs/advanced_feature_validation_report.txt
```

---

## 9. 数据时间划分说明

**划分脚本：** `scripts/22_time_split.py`  
**输出路径：** `data/model_input/train/`、`data/model_input/valid/`、`data/model_input/holdout/`

模型数据按 `hour_dt`（或 `hour`）的**真实日期**做时间顺序划分，**不能随机划分**。若采用随机划分，历史统计特征与平滑目标编码会引入未来信息，造成数据泄漏。

| 划分 | 日期范围 | 用途 | 可用历史信息 |
|------|----------|------|--------------|
| **train** | 2014-10-21 ~ 2014-10-28 | 模型训练；构建按日递增的历史映射 | 仅严格早于当前样本日期的 train 数据 |
| **valid** | 2014-10-29 | 模型选择、调参 | 仅完整 **train** 历史（不含 valid 自身 click） |
| **holdout** | 2014-10-30 | 最终独立评估 | 完整 **train + valid** 历史（不含 holdout 自身 click） |

**补充说明：**

- train 内样本必须按全局日期升序构建历史；同一 Parquet 分块可能含多个日期，需按**行级日期**处理。
- valid / holdout 各 split 内部，相同类别值应映射到相同的历史特征与 TE 值。

---

## 10. 历史统计特征（已实现）

**生成脚本：** `scripts/23_build_historical_features.py`  
**输出路径：** `data/features/historical/train/`、`data/features/historical/valid/`、`data/features/historical/holdout/`  
**映射表：** `outputs/feature_tables/historical/*_history_mapping.parquet`

**类别字段（5 个）：** `site_id`、`site_category`、`app_id`、`app_category`、`device_model`

每个类别字段生成以下 4 类特征，**共 20 个历史统计特征**：

### 10.1 特征定义

| 特征名模式 | 含义 | 计算范围 | 冷启动 / 未见类别 |
|------------|------|----------|-------------------|
| `{field}_hist_impressions` | 当前日期之前，该类别累计曝光次数 | 只使用严格早于当前样本日期的数据 | **0** |
| `{field}_hist_clicks` | 当前日期之前，该类别累计点击次数 | 只使用严格早于当前样本日期的数据 | **0** |
| `{field}_hist_ctr` | 该类别历史点击率 | `hist_clicks / hist_impressions`；历史曝光为 0 时取 **0** | **0** |
| `{field}_exposure_percentile` | 该类别历史曝光量在当前历史映射中的百分位排名；表示相对流量规模 | 对 `hist_impressions > 0` 的类别做 `rank(pct=True)` | **0** |

**取值范围：**

- `hist_impressions`、`hist_clicks`：非负整数
- `hist_ctr`：**[0, 1]**
- `exposure_percentile`：**[0, 1]**

### 10.2 完整特征列表

| 特征名 | 来源字段 | 数据类型 | 状态 |
|--------|----------|----------|------|
| `site_id_hist_impressions` | `site_id` | 整型 | **已实现** |
| `site_id_hist_clicks` | `site_id` | 整型 | **已实现** |
| `site_id_hist_ctr` | `site_id` | 浮点 | **已实现** |
| `site_id_exposure_percentile` | `site_id` | 浮点 | **已实现** |
| `site_category_hist_impressions` | `site_category` | 整型 | **已实现** |
| `site_category_hist_clicks` | `site_category` | 整型 | **已实现** |
| `site_category_hist_ctr` | `site_category` | 浮点 | **已实现** |
| `site_category_exposure_percentile` | `site_category` | 浮点 | **已实现** |
| `app_id_hist_impressions` | `app_id` | 整型 | **已实现** |
| `app_id_hist_clicks` | `app_id` | 整型 | **已实现** |
| `app_id_hist_ctr` | `app_id` | 浮点 | **已实现** |
| `app_id_exposure_percentile` | `app_id` | 浮点 | **已实现** |
| `app_category_hist_impressions` | `app_category` | 整型 | **已实现** |
| `app_category_hist_clicks` | `app_category` | 整型 | **已实现** |
| `app_category_hist_ctr` | `app_category` | 浮点 | **已实现** |
| `app_category_exposure_percentile` | `app_category` | 浮点 | **已实现** |
| `device_model_hist_impressions` | `device_model` | 整型 | **已实现** |
| `device_model_hist_clicks` | `device_model` | 整型 | **已实现** |
| `device_model_hist_ctr` | `device_model` | 浮点 | **已实现** |
| `device_model_exposure_percentile` | `device_model` | 浮点 | **已实现** |

### 10.3 生成方法说明

| 项目 | 说明 |
|------|------|
| train 处理方式 | 三阶段：① 按行 `event_date` 汇总日统计；② 全局日期升序构建「截至前一天」映射；③ 逐文件、逐行匹配映射并写出 |
| valid 历史来源 | 完整 train 聚合（Dask），不使用 valid 自身 click |
| holdout 历史来源 | train + valid 聚合（Dask），不使用 holdout 自身 click |
| 日期字段 | 优先 `date`，否则从 `hour`（`YYMMDDHH`）解析为 `event_date` |
| train 最早日期冷启动 | 2014-10-21 全部 20 个历史特征应为 0 |

**设计目的：** 在避免目标泄漏的前提下，刻画类别在**过去**的曝光规模、点击表现与相对热度，为 CTR 建模提供时序安全的统计特征。

---

## 11. 平滑目标编码特征（已实现）

**生成脚本：** `scripts/24_build_target_encoding.py`  
**输出路径：** `data/features/target_encoded/train/`、`data/features/target_encoded/valid/`、`data/features/target_encoded/holdout/`

平滑目标编码（Target Encoding, TE）结合类别历史 CTR、历史曝光量与整体 CTR，降低低频类别原始 CTR 过度波动的问题。

| 特征名 | 来源字段 | 数据类型 | 状态 |
|--------|----------|----------|------|
| `site_id_te` | `site_id` | 浮点 | **已实现** |
| `site_category_te` | `site_category` | 浮点 | **已实现** |
| `app_id_te` | `app_id` | 浮点 | **已实现** |
| `app_category_te` | `app_category` | 浮点 | **已实现** |
| `device_model_te` | `device_model` | 浮点 | **已实现** |

### 11.1 计算公式

一般形式：

```
TE = (类别历史点击数 + 平滑系数 × 整体 CTR)
     / (类别历史曝光数 + 平滑系数)
```

当前项目参数：

| 参数 | 取值 | 说明 |
|------|------|------|
| 平滑系数（`SMOOTHING_STRENGTH`） | 20 | 曝光越少，越向整体 CTR 收缩 |
| 冷启动先验（`DEFAULT_PRIOR`） | 0.17 | 无历史窗口时使用 |
| 整体 CTR（`prior_ctr`） | 历史窗口内 `总点击 / 总曝光` | 作为平滑项中的整体 CTR |

### 11.2 映射与回退规则

| 项目 | 说明 |
|------|------|
| train | 必须遵守时间顺序；日期 *d* 的 TE 仅使用严格早于 *d* 的历史 |
| valid | 使用完整 **train** 映射 |
| holdout | 使用完整 **train + valid** 映射 |
| 未出现类别 | 回退到当前历史窗口的**整体 CTR**（`prior_ctr`）；无历史窗口时使用 `DEFAULT_PRIOR` |
| 取值范围 | **[0, 1]** |

**设计目的：** 在控制泄漏的同时，将高基数类别映射为单一数值特征，兼顾类别历史点击倾向与低频稳定性。

---

## 12. 数据泄漏控制（高级特征）

以下原则适用于历史统计特征与平滑目标编码的生成与使用：

1. **Parquet 只是物理分块**  
   一个文件可能包含多个日期（例如 `part-0020.parquet` 同时含 2014-10-21 与 2014-10-22）。历史特征必须依据**每行真实日期**计算，**不能**按文件级日期或文件顺序推断历史范围。

2. **先映射、后累计**  
   当前日期全部样本的历史特征生成完成后，才允许把当前日期的曝光量与点击量加入累计历史映射。禁止边处理当前日行、边把当前日行 click 写入历史。

3. **train 最早日期冷启动**  
   train 最早日期（2014-10-21）的全部历史统计特征必须为 **0**（含 `hist_impressions`、`hist_clicks`、`hist_ctr`、`exposure_percentile`）。

4. **valid / holdout 禁用自身标签**  
   valid 映射仅由 train 构建；holdout 映射仅由 train + valid 构建。两个 split **内部不得**使用自身 `click` 参与映射统计。

5. **禁止随机时间划分**  
   训练/验证/留出集必须按时间切分，否则历史特征与 TE 会引用未来 click，导致评估虚高。

6. **`click` 仍不得作为普通输入特征**  
   历史特征与 TE 虽使用历史 click 聚合，但均严格限定在样本日期之前；当前行 `click` 不得参与当前行特征计算。

---

## 13. 高级特征验收结果

**验收脚本：** `scripts/25_validate_advanced_features.py`  
**报告与明细：** `outputs/advanced_feature_validation_report.txt`、`outputs/advanced_feature_validation_summary.csv`、`outputs/advanced_feature_column_stats.csv`

### 13.1 验收范围

| 阶段 | 路径 | 列数（约） |
|------|------|------------|
| 时间划分结果 | `data/model_input/{train,valid,holdout}/` | 39 |
| 历史统计特征 | `data/features/historical/{train,valid,holdout}/` | 59 |
| 目标编码结果 | `data/features/target_encoded/{train,valid,holdout}/` | 64 |

验收项包括：文件与行数一致性、字段完整性、缺失值/无穷值、取值范围、时间划分、冷启动与映射一致性、ID 抽样对齐等。

### 13.2 验收结论

| 项目 | 结果 |
|------|------|
| 通过项目 | **40** |
| 警告项目 | **0** |
| 错误项目 | **0** |
| **验收结论** | **高级特征验收通过，可以进入模型训练** |

---

**相关文件索引**

| 类型 | 路径 |
|------|------|
| 基础特征工程 | `scripts/19_build_basic_features.py` |
| 频次特征工程 | `scripts/20_build_frequency_features.py` |
| 特征初步验证 | `scripts/21_validate_features.py` |
| 时间划分 | `scripts/22_time_split.py` |
| 历史统计特征工程 | `scripts/23_build_historical_features.py` |
| 平滑目标编码 | `scripts/24_build_target_encoding.py` |
| 高级特征验收 | `scripts/25_validate_advanced_features.py` |
| 特征字典（本文档） | `docs/feature_dictionary.md` |
| 频次映射表 | `outputs/feature_tables/*_frequency.csv` |
| 历史映射表 | `outputs/feature_tables/historical/` |
| 初步验证报告 | `outputs/21_feature_validation_report.txt` |
| 验证明细 CSV | `outputs/feature_validation/` |
| 高级特征验收报告 | `outputs/advanced_feature_validation_report.txt` |
| 高级特征验收明细 | `outputs/advanced_feature_validation_summary.csv`、`outputs/advanced_feature_column_stats.csv` |
| 特征数据目录 | `data/features/basic/`、`data/features/frequency/`、`data/features/historical/`、`data/features/target_encoded/` |
| 建模输入目录 | `data/model_input/` |
