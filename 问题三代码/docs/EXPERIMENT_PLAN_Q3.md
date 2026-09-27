# 问题三实验方案与完成状态

更新时间：2026-09-26  
正式部署：EXP183 七成员异构共识 + Neutral-authenticity 仲裁器  
统一口径：主结果、主图和误差分析报告冻结 **test**；validation 只用于模型/阈值选择。

## 1. 实验纪律

1. 模型成员、仲裁规则与阈值在读取 test 前已经冻结。
2. test 不再用于选择模型、阈值、特征或超参数。冻结后的 test 遮挡和删除仅回答“模型依赖什么”，不形成新候选。
3. 附件4没有标签，不计算任何监督指标。
4. 所有数字必须能追溯到 CSV、JSON 或预测文件；没有运行的实验明确标为未完成。
5. EXP183 与 `delivery/SAN2_EXP183_complete` 永久保护，新产物写入 `paper_results/q3/`。

## 2. 实验矩阵

| 编号 | 研究问题 | 对象 | 划分/数据 | 主要输出 | 状态 |
|---|---|---|---|---|---|
| E1 | 最终模型预测能力如何 | EXP183 | test 727 | Accuracy、Macro-F1、MAE、Pearson 等 | 已完成 |
| E2 | 三种模态单独提供多少信息 | 冻结 EXP000 HAFusion | test 727 | T/A/V 与组合性能 | 已完成 |
| E3 | 已实现结构/融合路线差异 | 已冻结历史路线与 EXP183 | test 727 | 实际路线对比 | 已完成，非等容量公平对比 |
| E4 | 最终部署各组件是否必要 | EXP183 | test 727 | 去仲裁器、去 T/A/V 干预 | 已完成 |
| E5 | 联合分类回归是否优于分类偏置训练 | C³-HAFusion/EXP003 | test 727 | 分类和回归指标 | 已完成部分真实对比 |
| E6 | 结果对随机种子是否稳定 | EXP008/011/012 | test 727 | 三种子均值与离散性 | 已完成 |
| E7 | 关键配置变化的敏感性 | EXP001/004/005/008 | test 727 | Dropout、类权重、Focal 对比 | 已完成，历史冻结结果 |
| E8 | 类别错误集中在哪里 | EXP183 | test 727 | 混淆矩阵、分类别 P/R/F1 | 已完成 |
| E9 | 回归误差有什么规律 | EXP183 | test 727 | 散点、残差、Bias、强度分层 | 已完成 |
| E10 | 最终模型依赖哪些模态 | EXP183 | test 727 | 模态遮挡、作用程度、主要模态 | 已完成 |
| E11 | 局部证据是否影响预测 | EXP183 | 6 个分层 test 案例 | 单点删除、top/random/bottom 联合删除 | 已完成；案例级而非总体统计 |
| E12 | 能否回映射原视频 | EXP183 | 附件4 20 个样本 | 比例时间区间、视觉关键帧 | 已完成；无官方时间戳时明确为比例近似 |
| E13 | 附件4最终输出是什么 | EXP183 | 附件4 20 个无标签样本 | 分类、概率、强度、全局/局部解释 | 已完成 |

## 3. 指标定义

分类报告 Accuracy、Macro-F1、Weighted-F1 和 Balanced Accuracy；回归报告 MAE、RMSE、Bias 与 Pearson。Neutral 数量较少，因此以 Macro-F1 和 Neutral F1 检查类别不均衡影响。

分类—回归一致率使用 validation 选择的阈值 \(\varepsilon=0.149\)：

\[
s_\varepsilon(\hat y)=
\begin{cases}
\mathrm{Negative},&\hat y<-\varepsilon,\\
\mathrm{Neutral},&|\hat y|\le\varepsilon,\\
\mathrm{Positive},&\hat y>\varepsilon.
\end{cases}
\]

随后只在 test 上报告 \(s_\varepsilon(\hat y)\) 与分类头预测的一致比例。

## 4. 可解释性定义

最终 EXP183 没有一张跨七成员共享的注意力图，因此不使用单成员 attention 冒充最终解释。

- 模态作用程度：遮挡某模态后，冻结类别置信度和强度的联合变化，经样本内归一化得到。
- 局部作用程度：删除一个对齐位置后，定义

\[
e_{i,t}^{m}=\frac12|p_{i,\hat c_i}-p_{i,\hat c_i}^{(-m,t)}|
+\frac12\frac{|\hat y_i-\hat y_i^{(-m,t)}|}{3}.
\]

- 删除验证：按局部作用程度选择 top 位置，并与固定随机及 bottom 位置的联合删除比较。
- 时间映射：没有官方帧级时间戳时，将有效对齐位置按视频总时长比例映射；不能称为精确时间戳。

## 5. 未执行但不影响当前交付的严格实验

以下实验不能在 test 已打开后再为“改善 test”而新增并选择：

- 对 EXP183 七成员进行三次完全同构、从头训练的稳定性复现；
- 在完全相同编码器和训练预算下重训 Average/Concat/Fixed/Attention/Gate 六种公平融合；
- 对 EXP183 每个神经模块逐一重训的严格消融；
- classification-only 与 regression-only 两个纯单任务重训。

当前文档只报告已有真实结果，并在 `TODO_Q3_EXPERIMENTS.md` 给出未来应如何在新的外部封闭测试集上补做。
