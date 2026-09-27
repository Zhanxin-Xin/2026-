# 问题三模型—代码映射（MODEL_CODE_MAPPING_Q3）

生成日期：2026-09-26  
映射范围：基础可解释 C³-HAFusion、训练目标、解释链、正式 EXP183 部署图。  
说明：`B` 为 batch，`L=50`，`d=192`，类别数 `C=3`，基础模型证据组件数为 6（三个单模态 + 三个模态对）。

## 1. 入口映射

| 论文/部署层 | 代码类或函数 | 文件 | 真实作用 |
|---|---|---|---|
| 基础训练入口 | `main` | `src/train.py` | 加载配置和数据、训练、valid 选 checkpoint、导出各划分结果 |
| 数据加载 | `load_pickle`、`parse_split` | `src/data.py` | 检查三模态、生成 mask、解析分类/回归标签 |
| 模型工厂 | `build_model` | `src/model.py` | `architecture=hafusion` 时实例化 `HAFusionNet` |
| 基础解释推理 | `main`、`ensemble_forward` | `src/infer.py` | 同构 HAFusion checkpoint 推理、遮挡、证据与关键帧导出 |
| 解释忠实性 | `main` | `src/audit_explanations.py` | 模态遮挡、top 删除、随机删除、充分性和相关性 |
| EXP183 仲裁训练 | `main` | `src/train_neutral_authenticity_arbitrator.py` | OOF 训练 Neutral 真实性模型，valid 冻结阈值 |
| EXP183 最终评估 | `main` | `src/evaluate_neutral_authenticity_deployment.py` | 读取冻结仲裁器并一次性评估 test |

## 2. 基础 C³-HAFusion 完整数据流

| 论文模块 | 代码类/函数 | 文件 | 输入 | 输出 | 数学/代码操作 | 作用 |
|---|---|---|---|---|---|---|
| 三模态输入 | `parse_split` | `src/data.py` | text `[B,50,768]`；audio `[B,50,74]`；vision `[B,50,35]` | float32 三模态与 bool mask | `_as_feature_array`、`text_bert` attention mask | 统一批结构，不赋予 74/35 维未经证明的物理语义 |
| 有效位置掩码 | `_text_bert_attention_mask` | `src/data.py` | `text_bert [B,3,50]` | `[B,50]` | 读取第 2 通道并转 bool；`text_shared` 复制给三模态 | 排除 padding |
| 训练集归一化 | `FeatureNormalizer` | `src/data.py` | 三模态 | 同形状 | train 有效位置统计；audio/vision 标准化并 clip；所有模态乘 mask | 防止跨划分拟合统计量和 padding 污染 |
| 输入投影与位置编码 | `MultiScaleTemporalEncoder` | `src/model.py` | `[B,50,d_m]` | `[B,50,192]` | LayerNorm → Linear → 正弦位置编码 → Dropout | 映射异构模态到公共维度 |
| 多尺度局部时序 | `MultiScaleTemporalEncoder.local_convs` | `src/model.py` | `[B,50,192]` | `[B,50,192]` | 深度可分离 Conv1d，核 3/5，GELU，拼接后 Linear，残差缩放 | 建模局部邻域 |
| 长程模态内建模 | `MultiScaleTemporalEncoder.transformer` | `src/model.py` | `[B,50,192]` | `[B,50,192]` | 2 层、6 头、pre-norm Transformer，padding mask | 建模长程时序依赖 |
| 模态对一致性 | `ConflictAwarePairwiseContext.forward` | `src/model.py` | 两个对齐流 | 三个 pair 流 `[B,50,192]` | 公共投影后 cosine；`agreement=(sim+1)/2`；一致部分取加权均值 | 显式表示一致证据 |
| 模态对冲突 | `ConflictAwarePairwiseContext.forward` | `src/model.py` | 公共表示差 | pair 冲突表示和 `[B,3,50]` 冲突分数 | `1-agreement` 乘绝对差与有符号差的 MLP | 显式刻画冲突方向和强度 |
| pair 可靠性 | `ConflictAwarePairwiseContext.reliability` | `src/model.py` | 左/右流、pair、similarity | `[B,3,50]` | MLP + sigmoid + pair mask | 降低不可信交互影响 |
| 局部证据注意 | `EvidenceHead` | `src/model.py` | 单模态或 pair 流 `[B,50,192]` | 稀疏权重 `[B,50]` | 内容 MLP + 局部结构卷积 + saliency，经 sparsemax/softmax | 定位局部证据，padding 权重为 0 |
| 位置级预测证据 | `EvidenceHead` | `src/model.py` | `[B,50,192]` | local regression `[B,50]`；local logits `[B,50,3]` | 回归用 `3*tanh(raw/3)`；分类用 MLP | 为可加局部解释提供原子证据 |
| 时序聚合 | `EvidenceHead` | `src/model.py` | 局部证据与权重 | pooled `[B,192]`、模态预测 | 加权和 | 形成三个单模态和三个 pair 共六个组件 |
| 预测置信/不确定性 | `EvidenceHead` | `src/model.py` | pooled 分类、局部分类/回归 | confidence/uncertainty `[B]` | 归一化熵、局部分类和回归方差 | 路由的标签无关可靠性证据 |
| 共享—私有分解 | `SharedExclusiveDecomposer` | `src/model.py` | `[B,6,192]` | `[B,6,192]` + orthogonality | 单模态共享投影、私有投影、逐样本正交化；pair 注入共享共识 | 区分共性情感与模态特有残差 |
| 组件级交互 | `ComponentInteractionMixer`、`ReliabilityModulatedAttentionLayer` | `src/model.py` | 六组件 + reliability | `[B,6,192]` | 可靠性缩放 key 的 set self-attention + FFN + 残差 | 让六个证据组件交换上下文 |
| 自适应组件门控 | `ReliabilityAwareComponentGate` | `src/model.py` | 六组件、全局均值、差、乘积、可靠性 | `component_gates [B,6]` | 共享 MLP、可学习类型偏置、softmax、gate floor | 样本级分配六路作用权重 |
| 三模态作用权重 | `HAFusionNet.forward` | `src/model.py` | unary gates 与 pair gates | `modality_gates [B,3]` | 每个 pair gate 以 0.5/0.5 分配给所属模态 | 非负、和为 1 的内部动态模态作用权重 |
| 全局上下文修正 | `ContextConditionedDecision` | `src/model.py` | 组件、门控、全局上下文 | 每组件回归 delta 与分类 delta | `[component, global, |diff|, product]` → MLP，零附近残差尺度 | 在可控幅度内引入组件间决策上下文 |
| 分类—回归耦合 | `HAFusionNet.forward` | `src/model.py` | 未耦合分类/回归 | 耦合组件预测 | 类别正负 logit 差修正回归；回归符号基修正分类 | 利用标签方向与连续强度的一致关系 |
| 回归头 | `HAFusionNet.forward` | `src/model.py` | 三模态可加贡献 | `regression [B]` | `3*tanh((bias+Σ contribution)/3)` | 严格限制到 `[-3,3]` |
| 分类头 | `HAFusionNet.forward` | `src/model.py` | 三模态可加贡献 | logits/probability `[B,3]` | `class_bias + Σ classification_contributions`，softmax | 三分类预测 |
| 局部贡献守恒 | `HAFusionNet.forward` | `src/model.py` | 局部 unary/pair 证据 | `[B,3,50]` 与 `[B,3,50,3]` | gate × temporal weight × local evidence；pair 按 0.5 分回模态 | 局部贡献求和与全局贡献对应 |

## 3. 基础模型关键数学对应

### 3.1 输入与投影

对 `m∈{T,A,V}`：

```text
X_i^m ∈ R^(50×d_m),  d_T=768, d_A=74, d_V=35
H_i^m = TemporalEncoder_m(X_i^m, M_i) ∈ R^(50×192)
```

实际顺序为输入 LayerNorm、Linear、正弦位置编码、多尺度深度卷积残差、TransformerEncoder、输出 LayerNorm；不能在论文中写成 GRU/BiGRU。

### 3.2 一致与冲突

对模态对 `(m,n)` 的公共投影 `u_t^m,u_t^n`：

```text
s_t^(mn) = cosine(u_t^m,u_t^n)
a_t^(mn) = clip((s_t^(mn)+1)/2,0,1)
d_t^(mn) = 1-a_t^(mn)
consensus_t = a_t^(mn)(u_t^m+u_t^n)/2
conflict_t = d_t^(mn) MLP([|u_t^m-u_t^n|,u_t^m-u_t^n])
```

pair 表示还乘以 sigmoid 可靠性并严格应用 pair mask。

### 3.3 动态贡献

六组件门控 `g_i∈R^6` 由 masked softmax 得到，`g_ij≥0` 且可用组件上求和为 1。三模态权重为：

```text
I_i = g_i,unary + g_i,pair A
```

其中 `A` 是固定 pair-to-modality 分配矩阵，每个模态对向两个成员各分配 0.5。因此 `I_i^T+I_i^A+I_i^V=1`。

### 3.4 可加预测

```text
r_i,raw = b_r + Σ_m R_i^m
y_hat_i = 3 tanh(r_i,raw/3)
o_i = b_c + Σ_m C_i^m
p_i = softmax(o_i)
```

`R_i^m`、`C_i^m` 已包含 unary 以及分配后的 pair 贡献。

## 4. 训练目标映射

| 论文损失 | 代码项 | 实现 | 基础配置权重 |
|---|---|---|---:|
| 分类 | `classification` | 加权 CrossEntropy + label smoothing 0.03 | 1.0 |
| 回归 | `regression` | SmoothL1，beta=0.5 | 1.0 |
| Pearson | `pearson` | `1-corr`，常数 batch 安全处理 | 0.20 |
| 分类—回归一致 | `consistency` | SmoothL1(`p_pos-p_neg`, `y_hat/3`) | 0.12 |
| 注意力熵 | `attention_entropy` | 六路时序权重熵 | 0.008 |
| 注意力总变差 | `attention_total_variation` | 相邻权重绝对差 | 0.008 |
| 门控平衡 | `gate_balance` | 平均三模态门的 KL 型项 | 0.003 |
| 解释忠实性 | `faithfulness` | 预测作用权重与模态遮挡作用的 MSE | 0.08 |
| EMA 蒸馏 | `distillation` | 分类 KL + 回归 SmoothL1 | 0.15 |
| pair 稀疏 | `interaction_l1` | pair 回归贡献 L1 | 0.003 |
| 单模态深监督 | `unimodal_auxiliary` | 各可用模态分类/回归平均 | 0.15 |

优化器为 AdamW，基础运行学习率 `1.5e-4`、weight decay `2e-4`、cosine warmup、梯度裁剪 1.0、AMP、EMA 0.995。checkpoint 由 validation 复合分数选择：Accuracy 0.2、Macro-F1 0.3、归一化 MAE 0.2、Pearson 0.3。

## 5. 解释代码映射

| 解释输出 | 代码 | 数学含义 | 限制 |
|---|---|---|---|
| `modality_gates` | `HAFusionNet.forward` | 路由内部权重 | 不是因果贡献 |
| `predicted_modality_importance` | `src/losses.py` | 分类 margin 贡献与回归绝对贡献各 0.5 后归一化 | 需要遮挡验证 |
| `local_importance` | `src/explain.py` | 分类 top1-runner margin 与回归局部贡献各 0.5 | 是模型内部证据，不自动等于人类解释 |
| `ablation_modality_importance` | `src/losses.py` | 遮挡模态后预测置信与回归变化 | 可作为反事实忠实性目标 |
| top 删除 | `src/audit_explanations.py` | 删除最高局部重要度位置 | 与随机删除对比 |
| sufficient evidence | `src/audit_explanations.py` | 只保留 top 位置 | 检查证据充分性 |
| 文本片段 | `map_segment_to_words` | 将 50 个位置近似映射到词区间 | 不是 tokenizer offset 的严格字节映射 |
| 时间区间 | `segment_time_from_timestamps` / `proportional_time` | 时间戳优先，否则按视频时长比例映射 | 无时间戳时必须标注为比例近似 |
| 关键帧 | `extract_vision_keyframes` | 按关键视觉区间抽帧 | 依赖原视频可用 |

## 6. EXP183 最终部署映射

### 6.1 七成员

| 成员 | 配置证据 | 核心代码 | 输入/输出 |
|---|---|---|---|
| EXP027 | `01_exp027_resolved_config.json` | `PretrainedTextFusionNet`、动态融合路由 | raw text + audio + vision → 3 类概率/强度 |
| EXP029 | `02_exp029_resolved_config.json` | 同 EXP027，Macro checkpoint continuation | 同上 |
| EXP020 | `03_exp020_resolved_config.json` | `PretrainedTextFusionNet` + `ClassPrototypeRouter` | 三模态 → 原型辅助分类/强度 |
| EXP019 | `04_exp019_resolved_config.json` | `PretrainedTextFusionNet` + 上下文编码 | 当前文本 + 前序上下文 + 三模态 |
| EXP022 | `05_exp022_resolved_config.json` | Twitter-RoBERTa 预训练分类器 + 多模态适配 | raw text + legacy text/audio/vision |
| EXP001 | `06_exp001_resolved_config.json` | `HAFusionNet` | text+vision；audio 禁用 |
| EXP063 | `07_exp063_resolved_config.json` | `NLILabelSemanticExpert` | raw text 与三条标签假设 → NLI 概率/强度 |

### 6.2 父共识与仲裁

`parent_and_features` 将七成员概率堆为 `[B,7,3]`，计算：

```text
p_mean = mean_j p_j
p_vote = mean_j one_hot(argmax p_j)
p_parent = 0.5 p_mean + 0.5 p_vote
```

Neutral 真实性特征为 55 维，由七成员 log posterior、均值、标准差、最小/最大概率、投票、父概率、成员熵、七个强度、强度均值和标准差拼接。标准化逻辑回归只在父预测 Neutral 的 grouped train OOF 样本上学习真实性。

仲裁规则：

```text
if argmax(p_parent)=Neutral and authenticity<threshold:
    set Neutral probability to 0
    renormalize Negative/Positive
else:
    keep parent unchanged
```

该规则保证父级非 Neutral 决策不变，Neutral 被拒绝时 Negative/Positive odds 不变。

## 7. 基础模型与最终部署不可混淆项

| 项目 | 基础 C³-HAFusion | EXP183 |
|---|---|---|
| 是否单一神经网络 | 是 | 否，七成员 + 逻辑回归仲裁 |
| 原生三模态门控 | 是 | 否，成员结构异构 |
| 原生位置级可加贡献 | 是 | 否，仅 EXP001 等部分成员支持 |
| 正式 test Accuracy/Macro-F1 | 0.6657/0.6093 | 0.7331/0.6903 |
| 可直接使用 `src/infer.py` | 是，同构 checkpoint | 否，需要异构附件4推理器 |
| 论文可解释性证据 | 代码存在，实验待正式运行 | 必须额外做最终管线级遮挡/删除 |

## 8. 映射审计结论

1. 基础模型确实支持“冲突感知、保守残差、反事实校准”的 C³ 解释，并具备层次自适应融合结构。
2. `Consistency–Conflict–Contribution` 可以描述部分计算功能，但不是当前类注释定义的 C³ 全称；若论文采用该展开，必须同步修改模型卡并解释与代码注释的关系。
3. EXP183 是当前性能最好的正式部署图，不能用单一 C³-HAFusion 公式覆盖其全部决策。
4. 问题三最终章节应分别给出：基础可解释网络公式、EXP183 共识/仲裁公式、最终管线级解释验证方法。
5. 附件4异构推理已由 `scripts/predict_attachment4_exp183.py` 完成；最终管线局部删除解释由 `scripts/run_q3_exp183_local_evidence.py` 完成，20 个无标签样本均已输出预测、模态作用程度、关键位置、比例时间区间与关键帧。

## 9. 正式报告口径

- 主性能表、混淆矩阵、分类别指标、回归散点、残差、错误归因和最终管线模态遮挡均以**冻结 test**为对象。
- validation 只用于 checkpoint、仲裁阈值与分类—回归一致性阈值选择，单独存放在选择审计表中，不作为最终性能结论。
- test 在 EXP183 冻结后才被读取；后续 test 遮挡和删除仅用于描述性解释，未反向改变成员、阈值、特征或结构。
- 附件4没有标签，只输出预测与解释，不计算 Accuracy、F1、MAE 或 Pearson。
