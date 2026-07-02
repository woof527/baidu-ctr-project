# data 目录说明

本文件夹用于存放项目相关的全部数据文件。

---

## 子目录

| 路径 | 用途 |
|------|------|
| `data/raw/` | 原始 CSV 数据 |
| `data/interim/` | 中间过程数据（预留） |
| `data/processed/` | 清洗后的 Parquet 数据 |
| `data/features/` | 特征工程结果（后续使用） |

### data/raw/

存放比赛或业务提供的原始文件，通常包括：

- `train.csv` — 训练集，含 `click` 标签
- `test.csv` — 测试集，不含 `click` 标签
- `sampleSubmission.csv` — 提交结果模板

### data/processed/

由 `scripts/02_clean_to_parquet.py` 生成，主要包括：

- `train/*.parquet` — 清洗后的训练集分块
- `test/*.parquet` — 清洗后的测试集分块
- `cleaning_report.json` — 清洗过程统计（本地生成，默认不上传 GitHub）

### data/features/

预留给特征工程输出，例如训练/验证集特征表、编码映射表等。

---

## 关于 GitHub 上传策略

**由于数据文件体积较大，当前 GitHub 仓库不上传原始 CSV 和处理后的 Parquet 文件。**

仓库中只保留：

- 目录结构（通过 `.gitkeep` 占位）
- 本说明文档

---

## 本地复现

如需在本机复现实验，请按以下步骤操作：

1. 手动将原始数据放入 `data/raw/`：
   - `train.csv`
   - `test.csv`
   - `sampleSubmission.csv`

2. 运行清洗脚本：

   ```bash
   python scripts/02_clean_to_parquet.py
   ```

3. （可选）运行验收脚本，确认数据质量：

   ```bash
   python scripts/15_validate_processed_data.py
   ```

清洗完成后，`data/processed/train/` 和 `data/processed/test/` 下会出现 Parquet 分块文件，供后续 EDA 和建模脚本使用。
