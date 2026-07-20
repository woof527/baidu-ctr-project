# 模型结果汇总

百度广告点击率预测与投放优化分析 — 技术结果汇总（截至第 35 步）

> 本文档汇总开发阶段正式输出指标。**holdout 尚未使用。** 所有 `holdout_used=false`。

---

## 1. 数据范围与评价口径

本项目存在三类评价口径，**不可混用**：

| 口径 | 数据范围 | 用途 | 代表指标来源 |
|------|----------|------|--------------|
| **固定 valid 全量** | 500,000 行 fixed valid | Optuna 调优、固定样本基线 | `lightgbm_optuna_metrics.csv` |
| **development evaluation** | 200,000 行（valid 的 40%） | 校准方法比较、Top-K 策略 | `probability_calibration_metrics.csv`、`threshold_strategy_summary.csv` |
| **holdout** | 2014-10-30 | 最终一次性评估 | **尚未使用** |

**关键说明：**

- Optuna AUC **0.737077** 来自完整 500,000 行 fixed valid
- 校准 / 策略 AUC **≈0.735** 来自独立拆出的 200,000 行 evaluation
- 两者数据范围不同，**不能写成模型从 0.737 “降到” 0.735**

---

## 2. 固定样本基线

**脚本：** `scripts/30_build_fixed_tuning_sample.py`、`scripts/31_train_fixed_sample_baselines.py`

| 项目 | 数值 |
|------|-----:|
| train 行数 | 2,000,000 |
| valid 行数 | 500,000 |
| 特征数量 | 33 |
| valid CTR | 0.156968 |

### 三模型公平比较（相同固定样本）

| 模型 | ROC-AUC | LogLoss | calibration_gap | best_iteration |
|------|--------:|--------:|----------------:|---------------:|
| Logistic Regression | 0.702093 | 0.409753 | 0.028553 | — |
| **LightGBM** | **0.735406** | **0.385237** | **0.000055** | 376 |
| XGBoost | 0.733319 | 0.386072 | 0.000634 | 125 |

**结论：** LightGBM 为固定样本上最佳传统机器学习基线。

**指纹（valid_id_sha256）：** `46208ac9fdda8a91e5a8e5ca926db27c779c607f6e302255a3017e7bcf866a2f`

---

## 3. Optuna 调优结果

**脚本：** `scripts/32_tune_lightgbm_optuna.py`  
**Study：** `baidu_ctr_lightgbm_fixed_sample_v1`，20 trials，best trial **8**

### 调优前后对比（500,000 行 fixed valid）

| 指标 | 固定样本 LightGBM 基线 | Optuna 调优后 | 绝对变化 | 相对变化 |
|------|----------------------:|-------------:|---------:|---------:|
| ROC-AUC | 0.735406 | 0.737077 | +0.001671 | +0.23% |
| LogLoss | 0.385237 | 0.384831 | −0.000406 | −0.11% |
| calibration_gap | 0.000055 | 0.003799 | +0.003744 | 扩大 |
| best_iteration | 376 | 208 | — | — |

### 最佳超参数

| 参数 | 值 |
|------|-----:|
| max_depth | 5 |
| learning_rate | 0.103421 |
| num_leaves | 29 |
| min_child_samples | 143 |
| subsample | 0.931381 |
| colsample_bytree | 0.722213 |
| reg_alpha | 1.683e-05 |
| reg_lambda | 1.104e-07 |
| min_split_gain | 0.172621 |

**模型文件：** `models/tuned_lightgbm_optuna_model.joblib`

**说明：** 调优改善幅度不大但方向一致；调优后平均预测 CTR（0.160767）高于实际 CTR（0.156968），故后续进行概率校准。

---

## 4. SHAP 解释

**脚本：** `scripts/33_explain_tuned_lightgbm_shap.py`  
**样本：** fixed valid 抽样 20,000 行；**尺度：** LightGBM raw score（log-odds）

### 全局 Top 10 特征

| 排名 | 特征 | mean_abs_shap | importance_percent |
|:----:|------|-------------:|-------------------:|
| 1 | site_id_te | 0.382899 | 25.54% |
| 2 | app_id_te | 0.281873 | 18.80% |
| 3 | site_id_hist_ctr | 0.136616 | 9.11% |
| 4 | device_model_te | 0.122605 | 8.18% |
| 5 | site_id_freq | 0.062484 | 4.17% |
| 6 | app_id_hist_ctr | 0.056167 | 3.75% |
| 7 | app_id_freq | 0.040658 | 2.71% |
| 8 | app_id_exposure_percentile | 0.039122 | 2.61% |
| 9 | app_category_freq | 0.037336 | 2.49% |
| 10 | hour_cos | 0.032553 | 2.17% |

### 特征家族（group_importance_percent）

| 家族 | 占比 | top_feature |
|------|-----:|-------------|
| target_encoding | 54.22% | site_id_te |
| historical_ctr | 15.91% | site_id_hist_ctr |
| frequency | 12.54% | site_id_freq |
| exposure_percentile | 5.98% | app_id_exposure_percentile |
| historical_impressions | 4.54% | site_category_hist_impressions |
| time | 3.51% | hour_cos |
| historical_clicks | 3.30% | app_id_hist_clicks |

### 业务实体（entity_importance_percent）

| 实体 | 占比 | top_feature |
|------|-----:|-------------|
| site_id | 42.64% | site_id_te |
| app_id | 29.61% | app_id_te |
| device_model | 13.92% | device_model_te |
| app_category | 5.88% | app_category_freq |
| site_category | 4.44% | site_category_hist_impressions |
| time | 3.51% | hour_cos |

**注意：** SHAP 表示模型依赖，非因果；相关特征可能分摊重要性。

---

## 5. 概率校准

**脚本：** `scripts/34_calibrate_tuned_lightgbm.py`

### 数据拆分

| 子集 | 行数 | CTR |
|------|-----:|----:|
| calibration | 300,000 | 0.156967 |
| evaluation | 200,000 | 0.156970 |
| 合计（fixed valid） | 500,000 | 0.156968 |

### development evaluation 指标

| 方法 | ROC-AUC | LogLoss | Brier | mean_predicted_ctr | calibration_gap | ECE | 最佳 |
|------|--------:|--------:|------:|-------------------:|----------------:|----:|:----:|
| uncalibrated | 0.735183 | 0.385643 | 0.119355 | 0.160714 | 0.003744 | 0.006244 | |
| sigmoid | 0.735183 | 0.385587 | 0.119332 | 0.156891 | 0.000079 | 0.004884 | |
| **isotonic** | **0.735297** | **0.385278** | **0.119254** | **0.156907** | **0.000063** | **0.004339** | ✓ |

**最终选择：** Isotonic Regression  
**最终校准器：** `models/tuned_lightgbm_selected_calibrator.joblib`（完整 500,000 行 valid 重拟合）

**calibration_gap 改善：** 未校准 0.003744 → Isotonic 0.000063，约下降 **98.3%**

**限制：** 最终校准器不能在完整 valid 上再次自我评价；holdout 尚未使用。

---

## 6. 阈值与 Top-K 策略

**脚本：** `scripts/35_analyze_threshold_lift_strategy.py`  
**概率列：** isotonic_probability（selected_method=isotonic）  
**分析样本：** 200,000 行 development evaluation

### 基础指标

| 指标 | 数值 |
|------|-----:|
| overall CTR | 0.156970 |
| ROC-AUC | 0.735297 |
| Average Precision | 0.331623 |

### 候选运行点（摘要）

| operating_point | threshold | coverage | precision | recall | F1 | feasible |
|-----------------|----------:|---------:|----------:|-------:|---:|:--------:|
| maximum_f1 | 0.21 | 0.289 | 0.301 | 0.555 | 0.391 | true |
| maximum_youden_j | 0.16 | 0.419 | 0.266 | 0.710 | 0.387 | true |
| recall_at_least_80_percent | 0.12 | 0.528 | 0.241 | 0.809 | 0.371 | true |
| recall_at_least_90_percent | 0.09 | 0.667 | 0.214 | 0.909 | 0.346 | true |
| coverage_nearest_10_percent | 0.31 | 0.100 | 0.384 | 0.243 | 0.298 | true |
| coverage_nearest_20_percent | 0.26 | 0.189 | 0.336 | 0.404 | 0.367 | true |

> threshold=0.21 为 F1 最大候选点，是 Precision 与 Recall 的数学折中，**不是业务利润最优阈值**。threshold=0.5 在低 CTR 场景过于保守。

### Top-K 投放

| target_coverage | selected_rows | selected_ctr | lift | click_capture_rate | incremental_clicks_vs_random |
|----------------:|--------------:|-------------:|-----:|-------------------:|-----------------------------:|
| 1% | 2,000 | 0.6025 | 3.838 | 3.84% | +891.06 |
| 5% | 10,000 | 0.4551 | 2.899 | 14.50% | +2,981.30 |
| 10% | 20,000 | 0.3835 | 2.443 | 24.43% | +4,530.60 |
| 20% | 40,000 | 0.3322 | 2.116 | 42.32% | +7,007.20 |
| 100% | 200,000 | 0.1570 | 1.000 | 100.00% | 0.00 |

### 十分位（decile 1 = 最高分 10%）

| decile | rows | actual_ctr | lift | share_of_total_clicks |
|:------:|-----:|-----------:|-----:|----------------------:|
| 1 | 20,000 | 0.3835 | 2.443 | 24.43% |
| 2 | 20,000 | 0.2808 | 1.789 | 17.89% |
| 10 | 20,000 | 0.0189 | 0.120 | 1.20% |

**结论：** 模型能将高点击流量显著集中在高分区域；Top-K 适合预算固定场景。

---

## 7. 当前推荐方案

| 组件 | 推荐 |
|------|------|
| 基础模型 | Optuna 调优 LightGBM（trial 8，best_iteration=208） |
| 概率输出 | Isotonic 校准 |
| 预算固定场景 | Top-K 按 isotonic 概率降序投放 |
| 分类折中参考 | threshold=0.21（最大 F1，仅作开发参考） |
| 最终评价 | **等待一次性 holdout** |

这不是“最终上线方案”，而是**当前最佳开发阶段方案**。

---

## 8. 风险与限制

1. **development evaluation 非 holdout**：校准与策略结论均来自 valid 内部拆分，存在乐观偏差风险。
2. **超参数已用完整 valid 选择**：LightGBM Optuna 使用 500,000 行 fixed valid，信息复用不可忽略。
3. **最终校准器未在 holdout 评价**：Isotonic 在完整 valid 重拟合后，无独立 holdout 指标。
4. **时间跨度短**：约 10 天数据，季节性与长期漂移未知。
5. **缺少商业参数**：无 CPC、CPA、点击价值、预算、库存；Top-K 假设每条曝光成本相同。
6. **Lift ≠ 利润**：incremental_clicks_vs_random 仅为相对随机的额外点击估计。
7. **SHAP ≠ 因果**：高相关特征分摊重要性；冷启动实体未单独评估。
8. **holdout_used=false**：全文适用。

---

## 附录：主要输出文件索引

| 步骤 | 关键输出 |
|------|----------|
| 30 | `outputs/fixed_tuning_sample_metadata.json` |
| 31 | `outputs/fixed_sample_baseline_metrics.csv` |
| 32 | `outputs/lightgbm_optuna_metrics.csv`、`models/tuned_lightgbm_optuna_model.joblib` |
| 33 | `outputs/shap/tuned_lightgbm_shap_global_importance.csv` |
| 34 | `outputs/probability_calibration_metrics.csv`、`models/tuned_lightgbm_selected_calibrator.joblib` |
| 35 | `outputs/threshold_strategy_summary.csv`、`outputs/strategy/topk_lift_metrics.csv` |
