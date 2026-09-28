# TASP-MSA 模型说明

## 1. 建模目标

TASP-MSA面向文本、语音或视觉局部连续时段不可用的场景，在模态不完整时同时输出
三分类情感极性与连续情感强度。模型不直接重建原始高维输入，而是在64维任务相关
共享语义空间中估计缺失信息，以降低小样本条件下高维重建的不稳定性。

## 2. 核心流程

1. 将文本、语音、视觉分别投影到96维，并加入位置、模态和缺失状态编码；
2. 三种模态使用独立的单层Transformer提取时序表示；
3. 每个模态分解为64维共享情感语义与32维模态私有表示，并进行注意力汇聚；
4. 高斯情感语义代理根据三模态共享表示与缺失比例预测均值和对数方差；
5. 依据缺失比例在观测共享表示与代理均值之间完成语义补全；
6. 以补全后的文本共享语义为锚点，将语音、视觉共享—私有残差作为辅助信息；
7. 根据缺失比例、代理不确定性和跨模态余弦一致性生成样本级可靠性权重；
8. 融合表示分别送入层次分类头和回归头。

最终模型包含560177个可训练参数。

## 3. 联合训练目标

同一batch的Full、Single Missing和Double Missing视图共享同一网络参数。总目标为：

```text
L = L_cls
  + 0.70 L_reg
  + 0.10 L_proxy
  + 0.08 L_consistency
  + 0.05 L_representation
  + 0.02 L_private_modality
  + 0.01 L_orthogonality
  + 0.01 L_shared_alignment
```

其中分类损失为带0.02标签平滑的交叉熵，回归损失为SmoothL1；代理损失只监督
任务共享语义，目标来自Full视图且停止梯度；一致性项约束缺失视图与Full视图的
分类、回归及融合表示；其余结构项用于维持共享—私有分解。

## 4. 关键训练参数

```text
hidden_dim = 96
shared_dim = 64
private_dim = 32
fusion_dim = 96
nhead = 4
num_layers = 1
dim_feedforward = 192
batch_size = 64
optimizer = AdamW
learning_rate = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
early_stopping = 20
seed = 42
EMA decay = 0.98
validation missing ratio = 0.30
validation missing seed = 3042
```

## 5. 最终checkpoint

默认文件位置：`checkpoints/best_model.pth`

```text
selected epoch: 15
selection split: 附件2 valid
best selection score: 0.6395375579595566
selection protocol: 0.5*full_accuracy + 0.5*mean_six_missing_accuracy
SHA-256: 46906d94d4dfe5b87784fd2cfdf3e7f81873e288050639ebd94b604a42f1fddf
```

该文件保留EMA模型参数、配置、最佳epoch、验证指标和选择协议，去除了训练优化器
状态以控制附件体积；加载后可直接执行附件2评价和附件3推理。

## 6. 冻结测试结果

附件2 test完整输入结果：

```text
Accuracy = 0.6713
Macro-F1 = 0.6105
MAE = 0.6371
Pearson = 0.6684
```

30%混合连续缺失、5个随机种子的平均结果：

```text
Accuracy = 0.6677 ± 0.0048
Macro-F1 = 0.6022 ± 0.0085
MAE = 0.6451 ± 0.0069
Pearson = 0.6639 ± 0.0076
```

以上数值来自最终冻结模型的实验记录，测试集未参与参数更新、超参数选择或阈值调整。
