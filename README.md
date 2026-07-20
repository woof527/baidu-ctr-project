# 百度广告点击率预测与投放优化分析

基于百度广告曝光、点击日志，完成数据处理、探索性分析（EDA）、特征工程、CTR 预测建模，并输出可落地的投放优化建议。

---

## 项目目标

- 把大体量原始 CSV 清洗成可复用的 Parquet 数据
- 从多个维度分析 CTR 差异（时间、广告位、设备、媒体类别等）
- 构建可复现的特征工程与模型训练流程
- 形成 CTR 预测基线，并为后续调优、解释与业务落地提供依据

---

## 项目当前进度

### 已完成

- [x] 数据工程与 Parquet 转换
- [x] 数据质量检查
- [x] 探索性数据分析（EDA）
- [x] SQLite 数据库与 SQL 分析
- [x] 基础特征工程
- [x] 频次特征
- [x] 时间顺序划分
- [x] 历史统计特征
- [x] 平滑目标编码
- [x] 高级特征统一验收（40 项通过）
- [x] 逻辑回归 / LightGBM / XGBoost 基线（第 26—28 步）
- [x] 三模型统一对比（第 29 步）
- [x] 固定共享调参样本（第 30 步）
- [x] 固定样本三模型公平比较（第 31 步）
- [x] LightGBM Optuna 超参数调优（第 32 步）
- [x] 调优 LightGBM SHAP 解释（第 33 步）
- [x] 概率校准（第 34 步）
- [x] 阈值与 Top-K 投放策略分析（第 35 步）

### 尚未完成

- [ ] 一次性 holdout 评估
- [ ] 分日期 / 分实体 / 冷启动误差分析
- [ ] 最终项目报告与答辩材料
- [ ] 深度学习 CTR 模型
- [ ] 模型融合
- [ ] A/B 测试及业务落地（需真实收益与成本数据）

> **当前结论：** 已形成**当前最佳开发阶段方案**（Optuna 调优 LightGBM + Isotonic 校准 + Top-K 策略分析），但**不是最终上线模型**，**holdout 尚未使用**。

### 当前最佳开发流程

```
固定时间切分
→ 无泄漏历史特征与 Target Encoding
→ 固定共享样本（2,000,000 train / 500,000 valid）
→ LightGBM 基线
→ Optuna 调优
→ SHAP 解释
→ Isotonic 概率校准
→ Top-K 投放策略分析
→ （待完成）一次性 holdout 评估
```

---

## 目录结构

```
.
├── data/               # 数据目录（大文件不上传 GitHub，见 data/README.md）
├── docs/               # 特征字典等文档
├── scripts/            # 数据处理、特征工程、建模与对比脚本
├── outputs/            # 脚本输出（日志、指标、图表、验收报告等）
├── reports/            # 项目阶段报告
├── models/             # 模型文件（后续训练结果存放位置）
├── notebooks/          # Jupyter 笔记本（探索性分析、实验记录）
├── sql/                # SQL 脚本
├── requirements.txt    # Python 依赖
└── README.md
```

### 各目录用途

| 目录 | 用途 |
|------|------|
| `data/` | 原始 CSV、清洗后 Parquet、模型输入与特征工程结果 |
| `docs/` | 特征字典、字段说明 |
| `scripts/` | 可重复运行的 Python 脚本 |
| `outputs/` | 运行产物，如 EDA 汇总表、验收报告、模型指标、对比图表 |
| `reports/` | 阶段性 Markdown 报告 |
| `models/` | 训练好的模型文件（joblib、JSON 等） |
| `notebooks/` | 交互式分析与实验 |
| `sql/` | 数据库建表、查询 SQL |

---

## 数据说明

`data/` 中的原始 CSV 和处理后的 Parquet **体积较大，未上传到 GitHub**。仓库里保留了目录结构和 [`data/README.md`](data/README.md) 说明。

本地复现步骤：

1. 将 `train.csv`、`test.csv`、`sampleSubmission.csv` 放入 `data/raw/`
2. 运行 `python scripts/02_clean_to_parquet.py` 生成 Parquet
3. 运行 `python scripts/15_validate_processed_data.py` 做质量验收

更详细的数据处理说明见 [`reports/data_processing_report.md`](reports/data_processing_report.md)。

---

## 环境与依赖

推荐使用 Conda 创建独立环境：

```bash
conda create -n baidu_ctr python=3.11
conda activate baidu_ctr
pip install -r requirements.txt
```

主要依赖：pandas、dask、pyarrow、scikit-learn、lightgbm、xgboost、matplotlib 等。

---

## 早期工作概览

| 模块 | 说明 |
|------|------|
| 项目环境搭建 | Conda 环境 `baidu_ctr`，依赖见 `requirements.txt` |
| 原始数据检查 | `scripts/01_profile_raw_data.py` |
| 大文件分块读取 | 每次约 20 万行，避免内存溢出 |
| 数据清洗与 Parquet 转换 | `scripts/02_clean_to_parquet.py` |
| 清洗后数据质量验收 | `scripts/15_validate_processed_data.py`（已通过验收） |
| 整体与各维度 EDA | 整体 CTR、小时、日期、广告位、设备、站点/应用类别等 |
| 交叉分析与热力图 | banner×device、hour×banner、site×device 及对应热力图 |
| SQLite 数据库 | `scripts/16_create_sqlite_db.py`、`scripts/17_basic_sql_analysis.py` |
| 基础特征工程 | `scripts/19_build_basic_features.py` |
| 频次特征 | `scripts/20_build_frequency_features.py` |
| 基础特征验收 | `scripts/21_validate_features.py` |
| 版本管理 | Git + GitHub 管理代码与报告 |

---

## 特征字典与特征体系

[`docs/feature_dictionary.md`](docs/feature_dictionary.md) 是本项目的特征工程总说明与索引文档。[查看完整特征字典](docs/feature_dictionary.md)

### 特征字典的作用

特征字典用于统一记录每个模型输入特征的：

- 特征名称
- 来源字段
- 数据类型
- 业务含义
- 计算公式或生成逻辑
- 时间信息使用范围
- 缺失值或冷启动处理方式
- 模型预处理方式
- 潜在的数据泄漏风险
- 当前使用状态

它能够保证：

- 特征定义一致
- 训练集、验证集和 holdout 使用相同处理规则
- 后续模型训练可以复现
- 便于检查数据泄漏
- 便于项目维护和技术汇报

### 当前特征体系概览

#### 1. 基础时间特征

代表特征：

- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `hour_sin`
- `hour_cos`

`hour_sin` 和 `hour_cos` 用于表达小时的周期性，使 23 点与 0 点在特征空间中保持接近。完整列表见特征字典。

#### 2. 频次特征

代表特征：

- `site_id_freq`
- `site_category_freq`
- `app_id_freq`
- `app_category_freq`
- `device_model_freq`

频次特征表示某个类别在**训练集统计范围**内的出现规模或流量热度。具体统计口径以 [`scripts/20_build_frequency_features.py`](scripts/20_build_frequency_features.py) 和特征字典为准，**不等同于**按行级日期递推的历史统计特征。

#### 3. 交叉特征

概括广告位置、时间、设备、媒体类别之间的组合关系，用于表示单个字段无法直接表达的场景组合。

代表示例：`banner_device_cross`、`hour_banner_cross`、`site_device_cross`。完整列表见特征字典。

#### 4. 历史统计特征

针对 `site_id`、`site_category`、`app_id`、`app_category`、`device_model`，每个字段构建：

- `hist_impressions`
- `hist_clicks`
- `hist_ctr`
- `exposure_percentile`

合计 **20 个**历史统计特征：

- **hist_impressions**：严格早于当前日期的累计曝光量
- **hist_clicks**：严格早于当前日期的累计点击量
- **hist_ctr**：历史点击率
- **exposure_percentile**：历史流量规模的百分位排名

#### 5. 平滑目标编码特征

- `site_id_te`
- `site_category_te`
- `app_id_te`
- `app_category_te`
- `device_model_te`

目标编码利用类别的历史点击表现，将高维字符串类别转换为数值特征。采用平滑处理的原因是：

- 低频类别的原始 CTR 容易出现 0 或 1 等极端值
- 平滑后会向整体 CTR 回退
- 未见类别使用整体 CTR 作为默认值

#### 6. 当前模型输入特征

当前逻辑回归、LightGBM 和 XGBoost 基线主要使用 **33 个工程化数值特征**，包括：

- 频次特征
- 历史曝光量和历史点击量
- 历史 CTR
- 曝光百分位
- 平滑目标编码
- 时间周期特征（`hour_sin`、`hour_cos`）
- 部分二元时间特征（`is_weekend`）

原始高维字符串类别**尚未直接输入**当前基线模型，将在后续类别特征处理和深度学习模型阶段继续扩展。

### 代表性特征示例

下表仅展示部分代表性特征，**不能替代**完整特征字典：

| 特征示例 | 特征类别 | 含义 | 时间约束 | 预处理 |
|----------|----------|------|----------|--------|
| `hour_sin` | 时间特征 | 小时的周期正弦表示 | 当前样本时间 | 无 |
| `site_id_freq` | 频次特征 | 站点 ID 在训练集内的出现频次 | 按特征脚本定义（train 统计） | log1p |
| `site_id_hist_impressions` | 历史统计 | 当前日期前站点累计曝光 | 仅使用过去数据 | log1p |
| `site_id_hist_ctr` | 历史统计 | 当前日期前站点点击率 | 仅使用过去数据 | 保持比例值 |
| `site_id_te` | 目标编码 | 平滑后的站点历史点击倾向 | 仅使用允许的历史映射 | 保持比例值 |

> 特征重要性仅代表模型中的预测贡献，**不能解释为因果关系**。

### 数据泄漏控制

- Parquet 文件只是物理分块，一个文件可能包含多个日期
- 历史特征必须按照**每行真实日期**计算
- 当前日期只能使用**严格早于**该日期的数据
- valid 只能使用 train 建立的历史映射
- holdout 只能使用 train + valid 建立的历史映射
- 目标变量 `click` 不能通过未来数据进入模型特征
- 第 25 步验收脚本（`scripts/25_validate_advanced_features.py`）已对相关规则进行检查

### 维护规则

后续每新增或修改一个模型特征，都应同步更新 [`docs/feature_dictionary.md`](docs/feature_dictionary.md)。

README 只维护特征体系概览和代表性示例，**详细定义始终以特征字典为准**。

---

## 特征工程

本节介绍从基础特征到高级特征的具体构建步骤。建议先阅读上文「特征字典与特征体系」，再了解各项实现细节。

---

### 时间顺序划分

**脚本：** `scripts/22_time_split.py`

数据按时间顺序划分为三个 split，**不使用随机划分**，目的是防止历史特征使用未来信息：

| Split | 用途 |
|-------|------|
| `train` | 训练模型、建立历史映射 |
| `valid` | 模型比较与调参 |
| `holdout` | 最终一次性评估 |

**输出目录：**

- `data/model_input/train/`
- `data/model_input/valid/`
- `data/model_input/holdout/`

时间范围（来源列 `hour`）：

- train：2014-10-21 ~ 2014-10-28
- valid：2014-10-29
- holdout：2014-10-30

---

### 历史统计特征

**脚本：** `scripts/23_build_historical_features.py`

针对以下 5 个类别字段，分别构建 4 类历史统计特征，合计 **20 个**历史特征：

- `site_id`
- `site_category`
- `app_id`
- `app_category`
- `device_model`

每个字段生成：

- `*_hist_impressions` — 历史曝光次数
- `*_hist_clicks` — 历史点击次数
- `*_hist_ctr` — 历史点击率
- `*_exposure_percentile` — 历史曝光百分位

#### 已修复的时间泄漏问题

Parquet 文件只是物理分块，**一个文件中可能同时包含多个日期**。旧代码曾按文件日期处理历史映射，导致 train 最早日期部分记录错误使用了当天信息。

修复后按每一行真实日期计算历史特征：

- 当前日期样本只能使用**严格早于**当前日期的数据
- 当前日期全部特征生成完成后，才能把当天数据加入历史映射
- train 最早日期的历史统计特征全部为 0

该问题属于时间信息泄漏风险，已修复并重新生成数据。

---

### 平滑 Target Encoding

**脚本：** `scripts/24_build_target_encoding.py`

生成 5 个平滑目标编码特征：

- `site_id_te`
- `site_category_te`
- `app_id_te`
- `app_category_te`
- `device_model_te`

平滑目标编码通过类别历史点击表现和整体 CTR 进行加权，降低低频类别 CTR 波动。

**数据使用原则：**

| Split | 映射来源 |
|-------|----------|
| train | 只使用此前日期的信息 |
| valid | 只使用完整 train 映射 |
| holdout | 只使用完整 train + valid 映射 |

验收确认：未出现类别回退到整体 CTR 的情况。

---

### 高级特征统一验收

**脚本：** `scripts/25_validate_advanced_features.py`

**正式验收结果：**

| 结果 | 数量 |
|------|------|
| 通过 | 40 项 |
| 警告 | 0 项 |
| 错误 | 0 项 |

**验收内容包括：**

- 三阶段（model_input / historical / target_encoded）行数一致
- 字段完整
- 缺失值与无穷值检查
- 特征范围检查
- 时间划分检查
- 冷启动检查
- valid / holdout 映射一致性
- ID 一致性

**结论：** 高级特征验收通过，可以进入模型训练。

---

## 本周工作：传统机器学习基线

三个基线模型均使用 **33 个工程化数值特征**（详见上文「特征字典与特征体系」）。

### 26. 逻辑回归基线

**脚本：** `scripts/26_train_logistic_baseline.py`

由于训练数据规模较大（全量约 3237 万行），使用 `SGDClassifier(loss="log_loss")` 通过 `partial_fit` 实现**增量式逻辑回归训练**，而不是一次加载完整数据。

**正式验证结果（valid 全量 3,832,608 行）：**

| 指标 | 数值 |
|------|-----:|
| ROC-AUC | 0.731079 |
| LogLoss | 0.412632 |
| Accuracy | 0.843478 |
| Precision | 0.500601 |
| Recall | 0.106873 |
| F1 | 0.176141 |
| 验证集实际 CTR | 0.156562 |
| 平均预测 CTR | 0.243979 |

逻辑回归具有有效的排序能力，但明显高估点击概率，**概率校准较差**。

---

### 27. LightGBM 基线

**脚本：** `scripts/27_train_lightgbm_baseline.py`

**训练设置：**

- 训练样本：2,000,000
- 验证样本：500,000
- 随机种子：42
- 使用早停（best_iteration = 376）
- holdout 未使用

**正式结果：**

| 指标 | 数值 |
|------|-----:|
| ROC-AUC | 0.735406 |
| LogLoss | 0.385237 |
| Accuracy | 0.845780 |
| Precision | 0.590705 |
| Recall | 0.057005 |
| F1 | 0.103976 |
| 验证集实际 CTR | 0.156968 |
| 平均预测 CTR | 0.157023 |
| 最佳迭代轮数 | 376 |

LightGBM 使用 **Leaf-wise** 梯度提升策略，能够学习非线性关系和特征交互。

---

### 28. XGBoost 基线

**脚本：** `scripts/28_train_xgboost_baseline.py`

**训练设置：**

- 训练样本：2,000,000
- 验证样本：500,000
- 随机种子：42
- `tree_method="hist"`
- 使用早停（best_iteration = 125）
- holdout 未使用

**正式结果：**

| 指标 | 数值 |
|------|-----:|
| ROC-AUC | 0.733319 |
| LogLoss | 0.386072 |
| Accuracy | 0.845470 |
| Precision | 0.589043 |
| Recall | 0.051374 |
| F1 | 0.094505 |
| 验证集实际 CTR | 0.156968 |
| 平均预测 CTR | 0.156334 |
| 最佳迭代轮数 | 125 |

XGBoost 默认更偏向 **Depth-wise** 的树生长方式，并通过正则化控制模型复杂度。

---

### 29. 三模型统一对比

**脚本：** `scripts/29_compare_baseline_models.py`

**正式对比表：**

| 模型 | ROC-AUC | LogLoss | 实际 CTR | 平均预测 CTR | CTR 校准差距 |
|------|--------:|--------:|---------:|-------------:|-------------:|
| Logistic Regression | 0.731079 | 0.412632 | 0.156562 | 0.243979 | 0.087417 |
| LightGBM | 0.735406 | 0.385237 | 0.156968 | 0.157023 | 0.000055 |
| XGBoost | 0.733319 | 0.386072 | 0.156968 | 0.156334 | 0.000634 |

**根据正式指标：**

| 维度 | 最优模型 |
|------|----------|
| AUC 最优 | LightGBM |
| LogLoss 最优 | LightGBM |
| 概率校准最优 | LightGBM |
| 当前综合最佳传统模型 | LightGBM |

**对比说明：**

- LightGBM 与 XGBoost 使用相同训练样本规模、验证样本规模、随机种子、特征体系和一致的抽样摘要，因此**二者比较相对公平**。
- 但没有保存并逐条核验全部抽样 ID，因此**不能宣称**具体样本已经过逐行完全一致性验证。
- 逻辑回归使用**全量训练数据**（约 3237 万行），而两个树模型使用 **200 万抽样训练数据**，因此逻辑回归与树模型**不是完全相同的数据口径**。

**相对逻辑回归的提升（树模型）：**

| 树模型 | AUC 绝对提升 | AUC 相对提升 | LogLoss 绝对下降 | LogLoss 相对下降 | 校准差距下降 |
|--------|------------:|------------:|----------------:|----------------:|------------:|
| LightGBM | +0.004327 | +0.59% | 0.027394 | 6.64% | 0.087362 |
| XGBoost | +0.002240 | +0.31% | 0.026559 | 6.44% | 0.086783 |

> 以上为第 26—29 步**原始基线**对比（训练口径不完全相同）。第五周在**固定共享样本**上完成了更公平的比较与后续调优，见下文。

---

## 数据与时间切分

**脚本：** `scripts/22_time_split.py`

数据按时间顺序划分为三个 split，**不使用随机划分**：

| Split | 日期范围 | 用途 |
|-------|----------|------|
| train | 2014-10-21 ~ 2014-10-28 | 训练模型、建立历史映射 |
| valid | 2014-10-29 | 模型开发、调参、校准与策略分析 |
| holdout | 2014-10-30 | **最终一次性评估（尚未使用）** |

**输出目录：**

- `data/model_input/{train,valid,holdout}/`
- `data/features/target_encoded/{train,valid,holdout}/`

---

## 固定共享调参样本

**脚本：** `scripts/30_build_fixed_tuning_sample.py`

为保证调参、解释与校准使用**完全相同**的样本与特征，从 target-encoded 数据中抽取固定样本：

| 项目 | 数值 |
|------|-----:|
| train 行数 | 2,000,000 |
| valid 行数 | 500,000 |
| 特征数量 | 33 |
| holdout_used | false |

**输出：**

- `data/tuning/lightgbm_train/`、`data/tuning/lightgbm_valid/`
- `outputs/fixed_tuning_sample_metadata.json`

---

## 三模型公平比较

**脚本：** `scripts/31_train_fixed_sample_baselines.py`

在固定共享样本上复跑 Logistic Regression、LightGBM、XGBoost，实现**相同 train / valid / 特征 / id 顺序**的公平比较：

| 模型 | ROC-AUC | LogLoss | calibration_gap | 是否最佳 |
|------|--------:|--------:|----------------:|:--------:|
| Logistic Regression | 0.702093 | 0.409753 | 0.028553 | |
| **LightGBM** | **0.735406** | **0.385237** | **0.000055** | ✓ |
| XGBoost | 0.733319 | 0.386072 | 0.000634 | |

**结论：** 在固定样本上，LightGBM 为当前最佳传统机器学习基线（AUC 与 LogLoss 均最优）。

**输出：** `outputs/fixed_sample_baseline_metrics.csv`、`outputs/fixed_sample_baseline_comparison.csv`

---

## LightGBM Optuna 调优

**脚本：** `scripts/32_tune_lightgbm_optuna.py`

在固定样本上使用 Optuna（TPE，20 trials）最小化 valid LogLoss：

| 指标 | 固定样本 LightGBM 基线 | Optuna 调优后 | 变化 |
|------|----------------------:|-------------:|-----:|
| ROC-AUC | 0.735406 | **0.737077** | +0.001671 (+0.23%) |
| LogLoss | 0.385237 | **0.384831** | −0.000406 (−0.11%) |
| calibration_gap | 0.000055 | 0.003799 | 扩大 |
| best_iteration | 376 | 208 | — |

**最佳 trial：** 8

**最佳参数（摘要）：** max_depth=5，learning_rate≈0.103，num_leaves=29，min_child_samples=143，subsample≈0.931，colsample_bytree≈0.722

**说明：**

- 提升幅度不大，但 AUC 与 LogLoss 方向一致改善
- 调优后平均预测 CTR 略高于实际 CTR，calibration_gap 扩大，因此后续进行了概率校准
- 以上为**完整 500,000 行 fixed valid** 结果，**不是 holdout 结论**

**输出：** `models/tuned_lightgbm_optuna_model.joblib`、`outputs/lightgbm_optuna_*`

---

## SHAP 模型解释

**脚本：** `scripts/33_explain_tuned_lightgbm_shap.py`

对 Optuna 调优 LightGBM 在 fixed valid 上抽样 20,000 行做 SHAP 分析（raw score 尺度）：

**全局 Top 10 特征（importance_percent）：**

| 排名 | 特征 | 占比 |
|:----:|------|-----:|
| 1 | site_id_te | 25.54% |
| 2 | app_id_te | 18.80% |
| 3 | site_id_hist_ctr | 9.11% |
| 4 | device_model_te | 8.18% |
| 5 | site_id_freq | 4.17% |
| 6 | app_id_hist_ctr | 3.75% |
| 7 | app_id_freq | 2.71% |
| 8 | app_id_exposure_percentile | 2.61% |
| 9 | app_category_freq | 2.49% |
| 10 | hour_cos | 2.17% |

**特征家族：** Target Encoding 合计约 54.2%；**业务实体：** site_id 约 42.6%，app_id 约 29.6%

**要点：**

- Target Encoding 是模型最重要的信息来源
- site_id、app_id、device_model 的历史点击表现贡献突出
- 历史 CTR 比单纯曝光规模更重要；时间特征有贡献但非主因
- SHAP 表示模型依赖关系，**不是因果分析**；相关特征可能分摊重要性

**输出：** `outputs/shap/tuned_lightgbm_shap_*`、`outputs/tuned_lightgbm_shap_report.txt`

---

## 概率校准

**脚本：** `scripts/34_calibrate_tuned_lightgbm.py`

将 fixed valid（500,000 行）按 click 分层拆分为 calibration（300,000）与 evaluation（200,000），比较三种方法：

| 方法 | ROC-AUC | LogLoss | Brier | calibration_gap | ECE | 是否最佳 |
|------|--------:|--------:|------:|----------------:|----:|:--------:|
| uncalibrated | 0.735183 | 0.385643 | 0.119355 | 0.003744 | 0.006244 | |
| sigmoid | 0.735183 | 0.385587 | 0.119332 | 0.000079 | 0.004884 | |
| **isotonic** | **0.735297** | **0.385278** | **0.119254** | **0.000063** | **0.004339** | ✓ |

**最终选择：** Isotonic Regression

**说明：**

- Isotonic 在 LogLoss、Brier、ECE 与 calibration_gap 上均为最优；AUC 基本保持
- 相对未校准，calibration_gap 从 0.003744 降至 0.000063，约下降 **98.3%**
- 最终 Isotonic 校准器已在**完整 500,000 行 valid** 上重拟合（`models/tuned_lightgbm_selected_calibrator.joblib`）
- **不能在完整 valid 上再次自我评价**；**holdout 尚未使用**
- 上表 AUC≈0.735 来自 **200,000 行 development evaluation**，与 Optuna 全量 valid AUC 0.737 **数据范围不同**，不能写成“性能下降”

---

## 阈值与 Top-K 投放策略

**脚本：** `scripts/35_analyze_threshold_lift_strategy.py`

基于第 34 步 **isotonic** 概率，在 **200,000 行 development evaluation** 上分析：

| 指标 | 数值 |
|------|-----:|
| overall CTR | 0.156970 |
| ROC-AUC | 0.735297 |
| Average Precision | 0.331623 |

**最大 F1 候选点（数学折中，非业务最优）：** threshold=0.21，F1=0.3906，precision=0.3012，recall=0.5553

**Top-K 投放（isotonic 概率，每条曝光成本相同假设）：**

| 覆盖比例 | 行数 | CTR | Lift | 点击捕获率 | 相对随机额外点击 |
|:--------:|-----:|----:|-----:|-----------:|----------------:|
| Top 1% | 2,000 | 0.6025 | 3.84 | 3.84% | +891 |
| Top 5% | 10,000 | 0.4551 | 2.90 | 14.50% | +2,981 |
| Top 10% | 20,000 | 0.3835 | 2.44 | 24.43% | +4,531 |
| Top 20% | 40,000 | 0.3322 | 2.12 | 42.32% | +7,007 |

**要点：**

- 最高分 10% 流量捕获约 **24.43%** 的点击；Top 20% 约 **42.32%**
- 模型具有明显的高点击流量筛选能力
- Top-K 比固定 threshold=0.5 更适合预算控制（0.5 在低 CTR 场景过于保守）
- **Lift 不是利润提升**；缺少 CPC、CPA、点击价值与预算约束
- **不存在脱离业务目标的唯一最佳阈值**

**输出：** `outputs/strategy/`、`outputs/threshold_strategy_*`

---

## 当前最佳开发阶段方案

| 组件 | 选择 |
|------|------|
| 基础模型 | Optuna 调优 LightGBM（trial 8，best_iteration=208） |
| 概率输出 | Isotonic 校准 |
| 预算固定场景 | Top-K 按概率排序投放 |
| 分类折中参考 | threshold=0.21（最大 F1，仅作参考） |
| 最终无偏评价 | **等待一次性 holdout** |

**holdout_used：** false（全部步骤）

详细结果见 [`docs/model_results_summary.md`](docs/model_results_summary.md)、[`docs/week5_report.md`](docs/week5_report.md)。

---

## 模型评价指标说明

| 指标 | 含义 | 方向 |
|------|------|------|
| **ROC-AUC** | 衡量点击样本排序能力 | 越高越好 |
| **LogLoss** | 衡量概率预测质量 | 越低越好 |
| **CTR calibration gap** | \|平均预测 CTR − 真实 CTR\|，反映概率校准程度 | 越小越好 |
| Accuracy / Precision / Recall / F1 | 依赖分类阈值，只作为辅助指标 | — |

**注意：** 0.5 阈值通常不适合低概率 CTR 场景（验证集实际 CTR 约 15%—16%），因此 Accuracy、Precision、Recall、F1 不应作为模型选择的主要依据。

**当前模型比较主要依据：**

1. ROC-AUC
2. LogLoss
3. 概率校准差距

---

## 复现流程

### 基础流程（第 22—29 步）

```bash
python scripts/22_time_split.py
python scripts/23_build_historical_features.py
python scripts/24_build_target_encoding.py
python scripts/25_validate_advanced_features.py
python scripts/26_train_logistic_baseline.py
python scripts/27_train_lightgbm_baseline.py
python scripts/28_train_xgboost_baseline.py
python scripts/29_compare_baseline_models.py
```

### 第五周流程（第 30—35 步）

```bash
python scripts/30_build_fixed_tuning_sample.py
python scripts/31_train_fixed_sample_baselines.py
python scripts/32_tune_lightgbm_optuna.py
python scripts/33_explain_tuned_lightgbm_shap.py
python scripts/34_calibrate_tuned_lightgbm.py
python scripts/35_analyze_threshold_lift_strategy.py
```

**运行说明：**

- 各脚本需确认 `TEST_MODE=False` 后再跑正式结果
- 运行模型前必须先通过第 25 步高级特征验收
- **holdout 在当前阶段禁止使用**

---

## 正式输出成果

### 高级特征验收

- `outputs/advanced_feature_validation_summary.csv`
- `outputs/advanced_feature_column_stats.csv`
- `outputs/advanced_feature_validation_report.txt`

### 模型指标与报告

- `outputs/logistic_baseline_valid_metrics.json`
- `outputs/logistic_baseline_report.txt`
- `outputs/lightgbm_baseline_valid_metrics.json`
- `outputs/lightgbm_baseline_report.txt`
- `outputs/xgboost_baseline_valid_metrics.json`
- `outputs/xgboost_baseline_report.txt`

### 模型对比

- `outputs/model_comparison.csv`
- `outputs/model_relative_improvements.csv`
- `outputs/model_comparison_report.txt`
- `outputs/model_comparison_auc.png`
- `outputs/model_comparison_logloss.png`
- `outputs/model_comparison_calibration_gap.png`

### 第五周成果

- `outputs/fixed_tuning_sample_metadata.json`
- `outputs/fixed_sample_baseline_metrics.csv`
- `outputs/lightgbm_optuna_metrics.csv`
- `outputs/shap/tuned_lightgbm_shap_global_importance.csv`
- `outputs/probability_calibration_metrics.csv`
- `outputs/strategy/topk_lift_metrics.csv`
- `outputs/threshold_strategy_summary.csv`
- `docs/week5_report.md`
- `docs/model_results_summary.md`

### 阶段总结

- `reports/week4_model_baseline_summary.md`

---

## 结果限制

- 数据时间跨度约 **10 天**（2014-10-21 ~ 2014-10-30），泛化能力有限
- LightGBM 超参数已使用**完整 fixed valid** 选择，存在开发集信息复用
- 概率校准与 Top-K 策略基于 **200,000 行 development evaluation**，不是 holdout
- 最终 Isotonic 校准器在完整 valid 上重拟合，**不能在同一 valid 上自我评价**
- 逻辑回归与早期树模型基线**训练口径不同**（全量 vs 200 万抽样）
- SHAP 为模型解释，**不代表因果关系**；高基数历史特征可能随时间漂移
- Top-K / Lift 分析默认**每条曝光成本相同**，缺少 CPC、CPA、点击价值与预算约束
- **Lift 不能描述为利润提升**
- **holdout 尚未使用**；当前结果**不能代表**最终上线效果

---

## 下一阶段计划

1. 冻结当前数据处理、特征与模型方案
2. 编写一次性 holdout 评估脚本
3. 对 Optuna LightGBM + Isotonic 方案进行 **holdout 验证**
4. 比较 holdout 上的 AUC、LogLoss、Brier、ECE、Lift
5. 分日期、分实体及冷启动误差分析
6. 整理最终项目报告与答辩材料

---

## 主要脚本索引

| 脚本 | 功能 |
|------|------|
| `01_profile_raw_data.py` | 原始数据抽样检查 |
| `02_clean_to_parquet.py` | 分块清洗，输出 Parquet |
| `03_eda_overall.py` ~ `13_eda_site_device.py` | 各维度 EDA 与交叉分析 |
| `14a` ~ `14c` | CTR 热力图 |
| `15_validate_processed_data.py` | 清洗后数据验收 |
| `16_create_sqlite_db.py` | 创建 SQLite 数据库 |
| `17_basic_sql_analysis.py` | SQL 分析 |
| `19_build_basic_features.py` | 基础时间特征 |
| `20_build_frequency_features.py` | 频次特征 |
| `21_validate_features.py` | 基础特征验收 |
| `22_time_split.py` | 时间顺序划分 |
| `23_build_historical_features.py` | 历史统计特征 |
| `24_build_target_encoding.py` | 平滑目标编码 |
| `25_validate_advanced_features.py` | 高级特征统一验收 |
| `26_train_logistic_baseline.py` | 逻辑回归基线 |
| `27_train_lightgbm_baseline.py` | LightGBM 基线 |
| `28_train_xgboost_baseline.py` | XGBoost 基线 |
| `29_compare_baseline_models.py` | 三模型统一对比 |
| `30_build_fixed_tuning_sample.py` | 固定共享调参样本 |
| `31_train_fixed_sample_baselines.py` | 固定样本三模型公平比较 |
| `32_tune_lightgbm_optuna.py` | LightGBM Optuna 调优 |
| `33_explain_tuned_lightgbm_shap.py` | 调优 LightGBM SHAP 解释 |
| `34_calibrate_tuned_lightgbm.py` | 概率校准 |
| `35_analyze_threshold_lift_strategy.py` | 阈值与 Top-K 策略分析 |

---

## 许可证与说明

本项目为学习与分析用途。原始数据版权归数据提供方所有，请勿将大体量数据文件提交到公开仓库。
