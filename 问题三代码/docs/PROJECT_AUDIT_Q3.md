# 问题三项目审计报告（PROJECT_AUDIT_Q3）

生成日期：2026-09-26  
项目路径：`E:\2026\code\code\san2`  
审计意图：`AUDIT_ANALYSIS / RESEARCH`  
证据原则：代码与落盘结果优先于文件名和文字描述；没有真实结果的项目统一标记为“未完成”。

## 1. 审计结论摘要

1. 项目已有一条可复现的基础可解释模型链：
   `configs/aligned_hafusion.yaml → src.train → src.model.build_model → HAFusionNet`。
   该模型原生输出分类、回归、模态门控、位置级证据、模态对冲突与可加贡献。
2. 当前正式最优部署不是单个 C³-HAFusion checkpoint，而是 **EXP183 七成员异构共识 + Neutral-authenticity 仲裁器**。论文中不得把 EXP183 简化写成单一 C³-HAFusion。
3. EXP183 在预冻结后一次性 test 上得到 Accuracy `0.7331499312`、Macro-F1 `0.6903469466`、MAE `0.5308763824`、Pearson `0.7831292794`。Accuracy 超过 0.70，Macro-F1 未达到 0.70。
4. `aligned_50.pkl` 的三划分为 train `3395`、valid `728`、test `727`。样本 ID、视频组和完整特征哈希均无跨划分重叠；分类标签与连续标签符号冲突为 0。
5. 数据存在结构性现象：视觉全零样本分别为 train `110`、valid `15`、test `28`；文本 padding 区保留了非零预训练表示，但 `FeatureNormalizer.transform` 会按 mask 清零。模型前向还会把全零模态识别为不可用。
6. 基础模型具备解释输出代码；EXP183 不共享统一内部注意力，因此已用最终七成员管线的真实模态遮挡和局部删除补齐解释，产物位于 `paper_results/q3/interpretability/`、`paper_results/q3/cases/` 和 `paper_results/q3/attachment4/`，没有拿单成员注意力冒充最终集成解释。
7. 附件4的对齐特征与 20 个原视频均已存在，但审计时尚无正式附件4预测结果。
8. 项目中尚无 `.tex` 文件；任务书所述错误雷达图标题和对 74/35 维特征的过度物理解释，在当前可检索文本中均未发现。若这些内容位于项目外主论文，需要把主论文纳入项目后再做逐行修订。

## 2. 项目范围与保护边界

- 扫描文件总数：`2185`
- Python：`254`
- YAML/YML：`142`
- JSON：`632`
- CSV：`548`
- Markdown：`49`
- PKL：`42`
- 项目不是 Git 仓库，无法依赖 commit 追踪修改；后续所有关键产物必须附带配置、时间、来源路径和哈希。
- 永久只读保护：`EXP183` 相关源结果与 `delivery/SAN2_EXP183_complete`。
- 后续问题三新产物统一写入 `paper_results/q3/`，最终另建新的 Q3 全量交付目录，不覆盖 EXP183 冻结复现包。

## 3. 主要目录审计

| 路径 | 实际作用 | 状态 |
|---|---|---|
| `src/` | 数据、模型、训练、推理、解释、评估和集成 | 已扫描 |
| `configs/` | 基础模型与历史实验配置 | 历史分支很多；论文只保留主链 |
| `data/` | 附件2、标签、附件4对齐/未对齐特征与视频 | 数据存在 |
| `runs/` | EXP000–EXP208 左右的实验、OOF、校准和最终测试证据 | 已扫描关键主线 |
| `tests/` | 模型模块、梯度、CUDA smoke、仲裁器等测试 | 已存在 |
| `scripts/` | 本次新增的论文数据审计脚本 | 已建立 |
| `paper_results/q3/` | 论文统一结果目录 | 已建立，正在补全 |
| `results/`、`checkpoints/`、`figures/` | 任务书建议目录 | 根目录不存在；结果实际分散在 `runs/`，后续汇总到 `paper_results/q3/` |

## 4. 主配置范围

不把 142 个历史配置全部写入正文或最终操作入口。主链只保留以下三组：

1. **基础可解释模型**：`configs/aligned_hafusion.yaml`，对应 `HAFusionNet`。
2. **最终 EXP183 七成员配置**：
   `delivery/SAN2_EXP183_complete/models/configs/01_...07_*.json`。七份配置是最终模型的必要组成，不是可删除的历史调参文件。
3. **最终仲裁与冻结配置**：
   `runs/calibration/exp183_neutral_authenticity_valid_threshold_locked/final_metrics.json` 与 `DEPLOYMENT_LOCK.json`。

附件4将使用单独的推理清单记录七个 checkpoint、20 个对齐 PKL、视频目录、仲裁阈值和输出哈希。

## 5. 数据结构审计

### 5.1 原始字段

`aligned_50.pkl` 顶层为 `train/valid/test`。三划分原始字段一致：

| 字段 | 实际形状/类型 | 说明 |
|---|---|---|
| `id` | 长度 N 的 list | `video_id$_$clip_id` |
| `raw_text` | `[N]` Unicode ndarray | 原始文本 |
| `text` | `[N,50,768]`, float32 | 赛题提供的文本时序表示 |
| `text_bert` | `[N,3,50]`, int64 | token id、attention mask 等；代码取第 2 个通道为 mask |
| `audio` | `[N,50,74]`, float64 | 赛题提供的 74 维时序声学表示 |
| `vision` | `[N,50,35]`, float64 | 赛题提供的 35 维时序视觉表示 |
| `classification_labels` | `[N]`, float64 | 数值编码 0/1/2 |
| `regression_labels` | `[N]`, float64 | `[-3,3]` 连续标签 |

原始 PKL 不含 `annotations`、`length`、`mask` 字段；长度和三模态 mask 由 `text_bert` attention-mask 通道推导。不得在论文中声称这些字段由数据集直接提供。

完整字段清单：`paper_results/q3/dataset/field_inventory.csv`。

### 5.2 划分与类别分布

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 967 | 758 | 1670 | 3395 |
| Validation | 206 | 184 | 338 | 728 |
| Test | 207 | 158 | 362 | 727 |

证据：`paper_results/q3/dataset/class_distribution.csv`。

### 5.3 连续标签统计

| Split | Mean | Std | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0.1658 | 1.1151 | -3.0000 | -0.3333 | 0.0000 | 0.6667 | 3.0000 |
| Validation | 0.1609 | 1.0436 | -3.0000 | -0.3333 | 0.0000 | 0.6667 | 3.0000 |
| Test | 0.1412 | 1.1581 | -3.0000 | -0.3333 | 0.0000 | 1.0000 | 3.0000 |

Std 为样本标准差（`ddof=1`）。证据：`paper_results/q3/dataset/dataset_statistics.csv`。

### 5.4 完整性检查

| 检查 | Train | Valid | Test | 结论 |
|---|---:|---:|---:|---|
| NaN / Inf（全部三模态） | 0 | 0 | 0 | 通过 |
| 重复样本 ID | 0 | 0 | 0 | 通过 |
| 标签符号冲突 | 0 | 0 | 0 | 通过 |
| label.xlsx 连续标签冲突 | 0 | 0 | 0 | 通过 |
| label.xlsx 类别冲突 | 0 | 0 | 0 | 通过 |
| mask 非前缀结构 | 0 | 0 | 0 | 通过 |
| 零有效长度 | 0 | 0 | 0 | 通过 |
| 完整特征重复行 | 0 | 0 | 0 | 通过 |

跨划分样本 ID、视频 ID、特征哈希重叠均为 0。存在 2 条 train-valid 文本字面重复（`Alright`、`Okay`），但视频、样本 ID 和完整特征均不同，属于短文本内容重复，不构成样本泄漏。

### 5.5 mask、padding 与缺失模态

- mask 值严格为 `0/1`，有效长度 train/valid/test 最小为 `3/3/4`，中位数为 `22/23.5/23`，最大均为 `50`。
- audio 与 vision 的 padding 区为全零。
- text padding 区多数样本含非零预训练向量；训练前 `FeatureNormalizer.transform` 在 `src/data.py` 中执行 `x *= mask[...,None]`，因此进入模型的 padding 为零。
- vision 全零样本为 `110/15/28`。`HAFusionNet.forward` 会检查每个位置是否为非零，把整模态全零样本标为不可用；这应在论文的数据局限性中说明。

机器可读证据：`paper_results/q3/dataset/dataset_audit.json` 与 `integrity_checks.csv`。

## 6. 真实模型链

### 6.1 基础 C³-HAFusion

训练入口：`src/train.py`。  
模型工厂：`src/model.py::build_model`。  
真实模型类：`src/model.py::HAFusionNet`。  
基础运行：`runs/c3_hafusion_single/seed_20260924`。

该模型包含多尺度时序编码、对齐位置的一致/冲突关系、六路证据组件、可靠性调制组件交互、自适应门控、分类—回归耦合和严格可加的局部贡献。详细映射见 `MODEL_CODE_MAPPING_Q3.md`。

### 6.2 C³ 与 HAFusion 命名结论

- `HAFusionNet` 类注释明确写为：`conflict-aware, conservative and counterfactually calibrated`。因此当前最有直接代码证据的 C³ 展开是：
  **Conflict-aware–Conservative–Counterfactually Calibrated**。
- 代码确实还实现了 consensus/conflict/contribution，但 `Consistency–Conflict–Contribution` 不是当前代码中明确给出的正式缩写解释，不能未经说明直接替换。
- HAFusion 可以由两级结构支持为 `Hierarchical Adaptive Fusion`：先在序列级学习局部证据，再在模态/模态对组件级自适应融合。但这一英文全称尚未在源代码常量或正式模型卡中锁定，论文定稿时需在模型卡同步定义。

### 6.3 最终 EXP183

EXP183 的真实决策图包含：

1. EXP027 DeBERTa 动态路由；
2. EXP029 Macro-F1 continuation；
3. EXP020 DeBERTa 三模态原型；
4. EXP019 因果上下文模型；
5. EXP022 Twitter-RoBERTa 多视图模型；
6. EXP001 text+vision HAFusion；
7. EXP063 NLI Neutral-hurdle 专家；
8. 均值后验与多数投票分布各占 0.5 的父共识；
9. 只允许拒绝父级 Neutral 的 55 维真实性逻辑回归仲裁器。

仲裁器权重只在视频组隔离的 train OOF 上学习；最终阈值 `0.28616624178694333` 在 validation 上冻结。test 不参与成员、结构或阈值选择。

## 7. 当前真实性能

### 7.1 基础 C³-HAFusion 单模型

来源：`runs/c3_hafusion_single/seed_20260924/final_metrics.json`。

| Split | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| Train | 0.7225 | 0.6866 | 0.5291 | 0.7748 |
| Validation | 0.6374 | 0.6104 | 0.5895 | 0.6597 |
| Test | 0.6657 | 0.6093 | 0.6175 | 0.6988 |

### 7.2 正式最优 EXP183

| Scope | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc. | MAE | RMSE | Bias | Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train grouped OOF | 0.6919 | 0.6638 | 0.6951 | 0.6682 | 0.5292 | 0.7015 | -0.0170 | 0.7809 |
| Validation | 0.7115 | 0.6908 | 0.7102 | 0.6876 | 0.4989 | 0.6784 | -0.0111 | 0.7596 |
| Test（一次性冻结评估） | 0.7331 | 0.6903 | 0.7273 | 0.6890 | 0.5309 | 0.7198 | 0.0073 | 0.7831 |

test 上事后观察到父共识为 `0.7345/0.6970`，但该信息出现于 EXP183 冻结之后，不能据此改选模型。

## 8. 当前解释能力

### 已由代码直接支持

- 样本级 `modality_gates`，非负且归一化；
- 联合分类/回归的 `predicted_modality_importance`；
- text/audio/vision 的位置级可加贡献；
- 三个模态对的冲突强度、可靠性与局部贡献；
- 分类和回归贡献守恒检查；
- 模态全遮挡反事实；
- top-k 证据删除、随机删除和只保留关键证据；
- 文本片段、时间区间、视频关键帧映射代码。

### 尚未形成正式证据

- `src/audit_explanations.py` 存在，但 `runs/` 中未发现正式 explanation audit 输出；
- 现有 `valid_modality_ablation.json` 是对单 checkpoint 的输入遮挡，不等同于公平重训的单模态/组合模型；
- EXP183 未输出统一三模态贡献与局部证据；
- 附件4尚无预测、XLSX、解释卡和关键帧交付。

## 9. 论文与代码不一致风险

| 风险 | 审计结果 | 处理要求 |
|---|---|---|
| 把 EXP183 写成单一 C³-HAFusion | 不一致 | 区分基础解释模型与最终部署集成 |
| 把 74 维声学表示解释成具体音高/能量/语速 | 无代码依据 | 统一写“赛题提供的 74 维时序声学表示” |
| 把 35 维视觉表示解释成具体 AU | 无代码依据 | 统一写“赛题提供的 35 维时序视觉表示” |
| 把 attention 当因果贡献 | 不成立 | 使用“内部作用权重”，并以遮挡/删除验证 |
| C³ 直接写 Consistency–Conflict–Contribution | 与类注释不一致 | 优先沿用代码支持的三 C，或同步重命名模型卡 |
| 论文出现雷达干扰图标题 | 当前项目文本未找到 | 主论文导入后继续检查 |
| 图表数字来自历史最优挑选 | 高风险 | 每张表绑定唯一 CSV/JSON 来源 |

## 10. 关键实验缺口与优先级

1. **P0：附件4真实推理**。先完成七成员输出、冻结仲裁、CSV/XLSX、视频映射和哈希。
2. **P0：EXP183 解释忠实性（已完成）**。冻结 test 三模态遮挡已覆盖 727 个样本；局部证据删除覆盖 6 个典型 test 样本和附件4全部 20 个样本，且保留 top/random/bottom 联合删除对照。
3. **P1：公平单模态/组合实验**。当前只有输入遮挡结果，需固定训练方案独立训练或明确标为 post-hoc ablation。
4. **P1：融合基线与核心消融整理**。从既有真实实验筛选同划分、同训练预算的可比项；缺失项再运行。
5. **P1：稳定性**。当前多种 seed 结果并非 EXP183 完整决策图的同构重复，需要建立可比较的稳定性定义。
6. **P1：解释案例与群体统计**。在忠实性通过后生成典型正确、边界 Neutral、冲突和失败样本卡。
7. **P2：敏感性、图表与 LaTeX**。只引用已落盘结果；未完成项进入 `TODO_Q3_EXPERIMENTS.md`。

## 11. 可复现命令

重新生成本审计数据：

```powershell
Set-Location 'E:\2026\code\code\san2'
& 'E:\anaconda\envs\san2\python.exe' .\scripts\analyze_q3_dataset.py `
  --data .\data\aligned_50.pkl `
  --labels .\data\label.xlsx `
  --output .\paper_results\q3\dataset `
  --mask-strategy text_shared
```

## 12. 审计置信度

- 数据形状、标签、性能、代码链：高置信度（直接代码和落盘文件）。
- C³ 的三个 C：高置信度（类注释直接证据）。
- HAFusion 英文全称：中等置信度（结构支持，但代码未正式声明）。
- 项目外主论文中的图题/措辞：低置信度（当前项目无 `.tex` 和对应文本）。

