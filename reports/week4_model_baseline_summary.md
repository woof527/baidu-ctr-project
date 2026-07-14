# 本周工作概述

- 完成时间划分
- 完成历史统计特征
- 完成平滑目标编码
- 修复按 Parquet 文件日期处理导致的时间泄漏问题
- 通过 40 项高级特征验收
- 完成逻辑回归、LightGBM、XGBoost 三个传统模型基线

# 模型原理简述

## 逻辑回归

逻辑回归是线性概率模型，通过 sigmoid 函数将特征线性组合映射到 [0, 1] 点击概率。
它训练快、可解释性强，但难以自动表达复杂非线性关系。

## LightGBM

LightGBM 是基于梯度提升的 Leaf-wise 决策树集成模型。
它每次优先分裂增益最大的叶子，通常在相同迭代预算下收敛更快。

## XGBoost

XGBoost 是默认更偏向 Depth-wise 的梯度提升树模型。
它通过层级分裂与正则化控制树复杂度，在结构化表格数据上表现稳定。

# 正式指标对比

| model | roc_auc | log_loss | valid_actual_ctr | valid_mean_predicted_ctr | ctr_calibration_gap | overall_rank |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.735406 | 0.385237 | 0.156968 | 0.157023 | 0.000055 | 1 |
| XGBoost | 0.733319 | 0.386072 | 0.156968 | 0.156334 | 0.000634 | 2 |
| Logistic Regression | 0.731079 | 0.412632 | 0.156562 | 0.243979 | 0.087417 | 3 |

# 当前结论

- **AUC 最高模型**：LightGBM
- **LogLoss 最低模型**：LightGBM
- **平均预测 CTR 最接近真实 CTR 的模型**：LightGBM
- **当前最佳传统模型基线**：LightGBM

LightGBM 与 XGBoost 在相同抽样配置、相同特征体系与相同随机种子下对比，
因此二者的差异更具参考价值。

# 当前局限

- 逻辑回归与树模型训练样本口径不同
- LightGBM 和 XGBoost 虽然抽样摘要一致，但没有单独保存全部抽样 ID 进行逐条校验
- 当前参数主要为基线参数，尚未进行 Optuna 调优
- 尚未使用 holdout
- 当前只使用工程化数值特征，未充分处理所有原始类别特征
- 尚未进行 SHAP、概率校准和业务阈值分析

# 下一阶段

- 超参数调优
- SHAP 可解释性分析
- 概率校准
- 最终模型选择
- holdout 一次性评估
- 业务落地与 A/B 测试设计

# 相对逻辑回归的提升

| tree_model | auc_absolute_improvement | auc_relative_improvement_percent | logloss_absolute_reduction | logloss_relative_reduction_percent | calibration_gap_reduction |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 0.0043 | 0.5918 | 0.0274 | 6.6389 | 0.0874 |
| XGBoost | 0.0022 | 0.3064 | 0.0266 | 6.4366 | 0.0868 |

