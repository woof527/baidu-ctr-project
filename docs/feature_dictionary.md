# 特征字典

**项目名称：** 百度广告点击率预测与投放优化分析  
**文档阶段：** 第二阶段 — 特征工程（第二版）  
**数据基础：** 清洗后 Parquet / SQLite；特征文件 `data/features/basic/`、`data/features/frequency/`

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
```

---

**相关文件索引**

| 类型 | 路径 |
|------|------|
| 基础特征工程 | `scripts/19_build_basic_features.py` |
| 频次特征工程 | `scripts/20_build_frequency_features.py` |
| 特征初步验证 | `scripts/21_validate_features.py` |
| 特征字典（本文档） | `docs/feature_dictionary.md` |
| 频次映射表 | `outputs/feature_tables/*_frequency.csv` |
| 验证报告 | `outputs/21_feature_validation_report.txt` |
| 验证明细 CSV | `outputs/feature_validation/` |
| 特征数据目录 | `data/features/basic/`、`data/features/frequency/` |
