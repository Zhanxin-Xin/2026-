# 问题三完整交付入口

这是问题三全量交付，不替代也不覆盖旧的 `delivery/SAN2_EXP183_complete` 冻结复现包。

## 最重要的结论

正式 EXP183 冻结测试集：Accuracy **0.7331**，Macro-F1 **0.6903**，MAE **0.5309**，Pearson **0.7831**。主论文表格和图片以 test 为准；validation 只用于模型与阈值选择。

## 文件导航

- 完整结果说明：`EXPERIMENT_RESULTS_Q3.md`
- 项目审计：`PROJECT_AUDIT_Q3.md`
- 模型—代码映射：`MODEL_CODE_MAPPING_Q3.md`
- 实验方案：`EXPERIMENT_PLAN_Q3.md`
- 演化记录：`MODEL_ITERATION_LOG_Q3.md`
- 最终审计：`FINAL_Q3_AUDIT.md`
- 操作手册：`问题三完整操作手册.md`
- LaTeX 正文：`paper_results/q3/latex/question3_complete.tex`
- 图：`paper_results/q3/figures/`
- 表：`paper_results/q3/tables/`
- 附件4：`paper_results/q3/attachment4/attachment4_predictions.xlsx`

## 模型边界

基础 C³-HAFusion 是单一可解释网络；正式 EXP183 是七成员异构共识加 Neutral 真实性仲裁器。论文已经分别描述，不能把 EXP183 写成一个 C³-HAFusion checkpoint。

## 附件4

20 个样本全部完成分类、概率、强度、三模态作用程度、主要参考模态、局部关键位置、关键文本、比例时间区间和视觉关键帧。附件4无标签，未计算监督指标。
