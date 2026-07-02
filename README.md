# 百度广告点击率预测与投放优化分析

基于百度广告曝光、点击日志，完成数据处理、探索性分析（EDA）、特征工程、CTR 预测建模，并输出可落地的投放优化建议。

---

## 项目目标

- 把大体量原始 CSV 清洗成可复用的 Parquet 数据
- 从多个维度分析 CTR 差异（时间、广告位、设备、媒体类别等）
- 为后续特征工程和模型训练准备稳定输入
- 最终形成 CTR 预测结果和业务侧投放优化建议

---

## 目前已完成

| 模块 | 说明 |
|------|------|
| 项目环境搭建 | Conda 环境 `baidu_ctr`，依赖见 `requirements.txt` |
| 原始数据检查 | `scripts/01_profile_raw_data.py` |
| 大文件分块读取 | 每次约 20 万行，避免内存溢出 |
| 数据清洗与 Parquet 转换 | `scripts/02_clean_to_parquet.py` |
| 清洗后数据质量验收 | `scripts/15_validate_processed_data.py`（已通过验收） |
| 整体与各维度 EDA | 整体 CTR、小时、日期、广告位、设备、站点/应用类别等 |
| 交叉分析与热力图 | banner×device、hour×banner、site×device 及对应热力图 |
| 版本管理 | Git + GitHub 管理代码与报告 |

更详细的数据处理说明见 [`reports/data_processing_report.md`](reports/data_processing_report.md)。

---

## 目录结构

```
.
├── data/               # 数据目录（大文件不上传 GitHub，见 data/README.md）
├── scripts/            # 数据处理、EDA、验收、可视化等脚本
├── outputs/            # 脚本输出（日志、汇总表、验收报告、图表等）
├── reports/            # 项目阶段报告
├── models/             # 模型文件（后续训练结果存放位置）
├── notebooks/          # Jupyter 笔记本（探索性分析、实验记录）
├── sql/                # SQL 脚本（后续 SQLite 分析）
├── requirements.txt    # Python 依赖
└── README.md
```

### 各目录用途

| 目录 | 用途 |
|------|------|
| `data/` | 原始 CSV、清洗后 Parquet、特征工程结果 |
| `scripts/` | 可重复运行的 Python 脚本 |
| `outputs/` | 运行产物，如 `01_raw_profile.txt`、EDA 汇总表、热力图 PNG |
| `reports/` | 阶段性 Markdown 报告 |
| `models/` | 训练好的模型文件（pkl 等） |
| `notebooks/` | 交互式分析与实验 |
| `sql/` | 数据库建表、查询 SQL |

---

## 数据说明

`data/` 中的原始 CSV 和处理后的 Parquet **体积较大，未上传到 GitHub**。仓库里保留了目录结构和 [`data/README.md`](data/README.md) 说明。

本地复现步骤：

1. 将 `train.csv`、`test.csv`、`sampleSubmission.csv` 放入 `data/raw/`
2. 运行 `python scripts/02_clean_to_parquet.py` 生成 Parquet
3. 运行 `python scripts/15_validate_processed_data.py` 做质量验收

---

## 主要脚本（节选）

| 脚本 | 功能 |
|------|------|
| `01_profile_raw_data.py` | 原始数据抽样检查 |
| `02_clean_to_parquet.py` | 分块清洗，输出 Parquet |
| `03_eda_overall.py` ~ `13_eda_site_device.py` | 各维度 EDA 与交叉分析 |
| `14a` ~ `14c` | CTR 热力图 |
| `15_validate_processed_data.py` | 清洗后数据验收 |

---

## 环境与依赖

推荐使用 Conda 创建独立环境：

```bash
conda create -n baidu_ctr python=3.11
conda activate baidu_ctr
pip install -r requirements.txt
```

本阶段主要用到：pandas、dask、pyarrow、matplotlib 等。

---

## 后续计划

- [ ] 补充 SQLite 数据库和 SQL 分析
- [ ] 整理特征字典
- [ ] 构建模型输入特征
- [ ] 训练逻辑回归、LightGBM、XGBoost 等模型
- [ ] 输出业务投放优化建议

---

## 许可证与说明

本项目为学习与分析用途。原始数据版权归数据提供方所有，请勿将大体量数据文件提交到公开仓库。
