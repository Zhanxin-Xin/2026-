# 问题三最终一致性审计

审计日期：2026-09-26  
正式模型：EXP183  
正式测试结果：Accuracy 0.7331，Macro-F1 0.6903，MAE 0.5309，Pearson 0.7831。

## 1. 二十项核查

| # | 核查项 | 结论 | 证据 |
|---:|---|---|---|
| 1 | 公式与代码一致 | 通过 | `MODEL_CODE_MAPPING_Q3.md` |
| 2 | 模块名称与代码一致 | 通过 | 区分基础 HAFusion 与 EXP183 部署图 |
| 3 | C³ 名称有代码依据 | 通过 | 类注释为 conflict-aware、conservative、counterfactually calibrated |
| 4 | HAFusion 有层次自适应融合依据 | 通过 | 时序层、组件交互、可靠性门控 |
| 5 | 数字均来自真实产物 | 通过 | `Q3_RESULTS_MANIFEST.json` 与各 CSV/JSON |
| 6 | Train/valid/test 严格区分 | 通过 | main 图表使用 test；valid 仅作选择审计 |
| 7 | 数据泄漏检查 | 通过 | ID、视频组、特征哈希无跨划分重叠 |
| 8 | 未使用 test 调参 | 通过 | EXP183 在 test 前冻结；后续仅描述性遮挡/删除 |
| 9 | 图来自真实数据 | 通过 | `scripts/build_q3_paper_assets.py` 可重建 |
| 10 | 表格与 CSV 一致 | 通过 | 15 张表由同一 CSV/JSON 自动生成 |
| 11 | 正文与主指标一致 | 通过 | 0.7331/0.6903/0.5309/0.7831 |
| 12 | 图均有正文用途 | 通过 | 主文引用框架、架构、test 性能和解释图 |
| 13 | 表均有正文用途 | 通过 | LaTeX 使用 `tab:q3_*` 统一标签 |
| 14 | 公式变量有定义 | 通过 | `question3_complete.tex` |
| 15 | 缩写首次解释 | 通过 | OOF、CRCR 等在正文解释 |
| 16 | 创新声明有代码/实验支持 | 通过 | 只保留真实共识、仲裁、冲突、解释链 |
| 17 | 附件4全量输出 | 通过 | 20 行 CSV/XLSX，无缺失样本 |
| 18 | 极性、强度、全局/局部解释、视频回映射 | 通过 | 附件4完整表与 44 个以上关键帧 |
| 19 | Modality Occlusion | 通过 | test 727 样本、四种条件 |
| 20 | Evidence Deletion | 通过 | 6 个 test 案例 + 附件4 20 个样本，top/random/bottom |

## 2. 主结果口径审计

- `paper_results/q3/main/main_test_results.csv` 的三行均标注 `evaluation_split=test`。
- 混淆矩阵、分类别指标、回归散点、残差、错误归因和全体模态遮挡均来自正式 test。
- 训练曲线中的 validation 只标作“selection diagnostics”，不作为最终性能图；主论文可放附录。
- 父共识 test 0.7345/0.6970 只作冻结后描述性参照，不替换正式 EXP183。
- 附件4没有真值，清单明确 `labels_used=false`、`metrics_computed=false`。

## 3. 附件4完整性

- 样本数：20；唯一 ID：20。
- 视频匹配：20/20。
- 每行三类概率和为 1（浮点容差内）。
- 三模态作用程度和为 1（浮点容差内）。
- 分类分布：Negative 6、Neutral 3、Positive 11。
- 主要模态：Text 18、Vision 2、Audio 0。
- 包含文本关键片段、音频/视觉比例时间区间、视觉关键帧路径。
- 时间区间明确标为比例映射，未声称是官方帧级时间戳。

## 4. 已知边界

1. 正式 Macro-F1 为 0.6903，满足“超过 0.69”的最终交付口径，但未超过 0.70。
2. test 局部删除为六个分层典型案例；全体 727 个 test 样本已完成模态级遮挡。
3. 历史融合路线并非严格等容量，正文已说明，不作夸大公平性结论。
4. EXP183 是异构集成，不得简化成单一 C³-HAFusion 网络。
5. 语音 74 维和视觉 35 维不解释为具体手工物理量。

## 5. 审计结论

问题三已形成“数据审计—模型映射—冻结预测—测试集评价—管线级解释—忠实性检查—附件4推理—论文图表—LaTeX正文—可复现命令”的闭环。剩余严格研究增强项已单列到 `TODO_Q3_EXPERIMENTS.md`，没有虚构结果。
