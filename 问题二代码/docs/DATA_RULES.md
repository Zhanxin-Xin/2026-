# 数据处理与模态缺失规则

## 1. 附件2实际结构

模型使用附件2的 `aligned_50.pkl`，最外层为 `train`、`valid`、`test`。三种模态
张量形状分别为：

```text
text   : [N, 50, 768]
audio  : [N, 50, 74]
vision : [N, 50, 35]
text_bert : [N, 3, 50]
```

`text_bert`三通道依次为 `input_ids`、`attention_mask`、`token_type_ids`。
分类标签为一维整数数组，类别编号0、1、2依次对应Negative、Neutral、Positive；
回归标签为一维浮点数组，取值范围为 `[-3,3]`。

## 2. 有效位置掩码

`valid_mask`只表示真实内容位置与padding，不由三模态特征是否为零推断。其计算为：

```text
valid = attention_mask == 1
valid = valid AND input_ids != 101 ([CLS])
valid = valid AND input_ids != 102 ([SEP])
```

因此，视觉或语音中原本存在的全零特征不会被直接判定为padding。三个模态共享由
文本BERT token有效范围确定的时间轴。

## 3. 人工连续缺失

`missing_mask`与`valid_mask`严格分离：

- `valid_mask=1`：该位置属于样本的真实有效时间范围；
- `missing_mask=1`：该位置当前可观测；
- `missing_mask=0 且 valid_mask=1`：人为制造的局部连续缺失；
- padding永远不参与人工缺失，也不参与补全损失和评价统计。

训练时，同一原始样本构造三种共享参数视图：

1. Full视图：全部有效位置均可观测；
2. Single视图：随机选择一种模态并遮蔽一个连续区间；
3. Double视图：随机选择两种模态，各自使用相同或不同的连续区间。

缺失比例从 `Uniform(0.1,0.6)` 采样，区间长度根据样本真实有效长度计算，随后在
合法有效区间内选择起点。验证和测试鲁棒性实验使用样本ID、实验seed和设置共同
生成确定性连续区间，以保证重复运行得到相同结果。

## 4. 附件3检查与缺失检测

附件3实际采用对齐版本，共30个独立pkl文件。每个文件结构为：

```text
{"test": {"text_bert": [1,3,50], "audio": [1,50,74], "vision": [1,50,35]}}
```

推理时先依据 `text_bert` 的attention mask并排除 `[CLS]`、`[SEP]`得到有效范围，
再仅在有效范围内部检测语音、视觉的整行全零位置。有效范围之外的全零行为padding，
不计为缺失。附件3实际检查未发现可等价判定为文本局部缺失的文本零行，因此文本
位置按token有效掩码处理。

附件3中的文本字段是BERT token信息而不是768维特征，故使用标准
`bert-base-uncased`恢复与附件2一致的768维上下文特征。该预训练模型仅作特征
恢复，不参与情感数据训练或附件3调参。

## 5. 数值与异常处理

- 输入在Dataset中统一转换为 `float32`；
- 分类标签转换为 `torch.long`，回归标签转换为 `float32`；
- 模型不对原始数据自行reshape或插值；
- padding通过Transformer的key padding mask屏蔽；
- 回归头使用 `tanh × 3`，保证输出位于 `[-3,3]`；
- 数据检查脚本会报告各字段shape、dtype、NaN、Inf、全零行和标签分布。

