# 第五周工作报告

百度广告点击率预测与投放优化分析 — 固定样本调优、解释、校准与投放策略

---

## 一、本周目标

1. 构建固定共享 train / valid 样本，保证后续调参、解释与校准口径一致
2. 在相同样本上公平比较 Logistic Regression、LightGBM、XGBoost
3. 使用 Optuna 对 LightGBM 进行超参数调优
4. 使用 SHAP 解释调优模型的特征依赖
5. 对调优模型预测概率进行校准，选出开发阶段最佳校准方法
6. 在独立 evaluation 子集上完成阈值扫描与 Top-K 投放策略分析
7. **不读取 holdout**，不将当前结果描述为最终验证

---

## 二、本周完成内容

### 1. 固定共享样本

- **脚本：** `scripts/30_build_fixed_tuning_sample.py`
- train：2,000,000 行；valid：500,000 行；特征：33 个
- 特征已完成 log1p、缺失值处理、float32 转换等，后续步骤禁止重复处理
- `holdout_used=false`，`validation_passed=true`

### 2. 三模型公平比较

- **脚本：** `scripts/31_train_fixed_sample_baselines.py`
- 三种模型使用完全相同的固定样本、标签、特征列与 valid id 顺序
- 固定样本上 **LightGBM 为最佳传统机器学习基线**（AUC 与 LogLoss 均最优）

### 3. Optuna 调优

- **脚本：** `scripts/32_tune_lightgbm_optuna.py`
- Optuna TPE 采样，20 个 trial，目标最小化 valid LogLoss
- 最佳 trial：8；best_iteration：208
- 相对固定样本 LightGBM 基线，AUC 与 LogLoss 均有小幅改善，但 calibration_gap 有所扩大

### 4. SHAP 特征解释

- **脚本：** `scripts/33_explain_tuned_lightgbm_shap.py`
- 在 fixed valid 上抽样 20,000 行做 SHAP 分析
- Target Encoding 与 site_id / app_id 相关特征贡献最大
- 输出全局重要性、特征家族、业务实体、局部案例与图表

### 5. 概率校准

- **脚本：** `scripts/34_calibrate_tuned_lightgbm.py`
- fixed valid 分层拆分：calibration 300,000 行 / evaluation 200,000 行
- 比较 uncalibrated、sigmoid、isotonic 三种方法
- **最终选择 Isotonic**；完整 valid 上重拟合最终校准器

### 6. 阈值与投放策略分析

- **脚本：** `scripts/35_analyze_threshold_lift_strategy.py`
- 使用 isotonic 校准概率，在 200,000 行 development evaluation 上分析
- 完成阈值扫描、Top-K 投放、十分位分析与候选运行点输出
- 明确 Top-K 适合预算控制，Lift 不等于利润

---

## 三、核心结果

### 基线与 Optuna 调优（完整 fixed valid，500,000 行）

| 模型 | ROC-AUC | LogLoss | calibration_gap | best_iteration |
|------|--------:|--------:|----------------:|---------------:|
| 固定样本 LightGBM 基线 | 0.735406 | 0.385237 | 0.000055 | 376 |
| Optuna 调优 LightGBM | **0.737077** | **0.384831** | 0.003799 | 208 |
| 变化 | +0.001671 (+0.23%) | −0.000406 (−0.11%) | 扩大 | — |

### 固定样本三模型公平比较（500,000 行 valid）

| 模型 | ROC-AUC | LogLoss | calibration_gap |
|------|--------:|--------:|----------------:|
| Logistic Regression | 0.702093 | 0.409753 | 0.028553 |
| **LightGBM** | **0.735406** | **0.385237** | **0.000055** |
| XGBoost | 0.733319 | 0.386072 | 0.000634 |

### 概率校准（development evaluation，200,000 行）

| 方法 | ROC-AUC | LogLoss | Brier | calibration_gap | ECE | 最佳 |
|------|--------:|--------:|------:|----------------:|----:|:----:|
| uncalibrated | 0.735183 | 0.385643 | 0.119355 | 0.003744 | 0.006244 | |
| sigmoid | 0.735183 | 0.385587 | 0.119332 | 0.000079 | 0.004884 | |
| **isotonic** | **0.735297** | **0.385278** | **0.119254** | **0.000063** | **0.004339** | ✓ |

> 注意：上表 AUC 来自 200,000 行 evaluation 子集；Optuna 的 AUC 0.737077 来自完整 500,000 行 valid，**二者数据范围不同，不能横向比较为性能下降**。

### Top-K 投放（isotonic，200,000 行 evaluation）

| 覆盖比例 | 行数 | CTR | Lift | 点击捕获率 | 相对随机额外点击 |
|:--------:|-----:|----:|-----:|-----------:|----------------:|
| Top 1% | 2,000 | 0.6025 | 3.84 | 3.84% | +891 |
| Top 5% | 10,000 | 0.4551 | 2.90 | 14.50% | +2,981 |
| Top 10% | 20,000 | 0.3835 | 2.44 | 24.43% | +4,531 |
| Top 20% | 40,000 | 0.3322 | 2.12 | 42.32% | +7,007 |

### SHAP 前十特征（20,000 行抽样，importance_percent）

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

---

## 四、关键发现

1. **LightGBM 为当前最佳模型**：在固定样本公平比较中 AUC 与 LogLoss 均最优，Optuna 调优后进一步小幅改善。
2. **调优带来小幅但稳定改善**：AUC +0.001671，LogLoss −0.000406；幅度不大，方向一致。
3. **模型高度依赖站点与应用历史点击信息**：site_id_te、app_id_te 及 hist_ctr 类特征 SHAP 贡献最高；Target Encoding 家族合计约 54.2%。
4. **Isotonic 显著改善概率可靠性**：在 development evaluation 上 LogLoss、Brier、ECE 与 calibration_gap 均为最优；calibration_gap 相对未校准下降约 98.3%。
5. **高分流量具有较高 CTR 与 Lift**：Top 1% CTR 达 0.6025（Lift 3.84）；Top 10% 捕获约 24.43% 的点击。
6. **Top-K 方案具有实际投放筛选价值**：比固定 threshold=0.5 更适合低 CTR 场景下的预算控制，但仍需 holdout 与业务成本数据验证。

---

## 五、问题与限制

- 数据只有约 **十天**（2014-10-21 ~ 2014-10-30），时间跨度较短，长期泛化未知
- LightGBM **超参数已使用完整 fixed valid 选择**，存在开发集信息复用
- 概率校准方法比较属于 **development evaluation**（200,000 行），不是 holdout
- **holdout 尚未使用**；当前所有结论均为开发阶段结论
- 缺少真实广告收益、CPC、CPA、预算与库存约束；Top-K 默认每条曝光成本相同
- **SHAP 不代表因果关系**；Target Encoding 与 hist_ctr 等相关特征可能分摊重要性
- 高基数历史特征可能随时间漂移；冷启动实体仍需单独分析
- Optuna 调优后 calibration_gap 大于固定样本基线 LightGBM，需依赖后续 Isotonic 校准
- **不能将 Lift 或 incremental_clicks_vs_random 描述为利润提升**

---

## 六、下周计划

1. 冻结当前数据处理、特征和模型方案
2. 编写一次性 holdout 评估脚本
3. 对最终 **Optuna LightGBM + Isotonic** 方案进行一次 holdout 验证
4. 比较 holdout 上的 AUC、LogLoss、Brier、ECE、Lift
5. 进行分日期、分实体及冷启动误差分析
6. 整理最终项目报告和答辩材料

---

**holdout_used：** false  
**文档依据：** `outputs/` 下第 30—35 步正式输出文件
