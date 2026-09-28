# 运行环境说明

## 1. 原始实验环境

```text
Operating system : Linux x86_64, glibc 2.39
Python           : 3.14.7
PyTorch          : 2.11.0+cu128
CUDA runtime     : 12.8
cuDNN            : 9.19.0
NumPy            : 2.5.3
PyYAML           : 6.0.3
Matplotlib       : 3.11.2
Transformers     : 5.17.0
```

正式训练日志记录的计算设备为CUDA。代码也支持CPU，但正式训练和完整鲁棒性实验
建议使用GPU。

## 2. Python依赖

精确版本记录在 `requirements.txt`。CPU环境可直接执行：

```bash
python -m pip install -r requirements.txt
```

CUDA 12.8环境建议先安装对应PyTorch轮子，再安装其他依赖：

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.5.3 PyYAML==6.0.3 matplotlib==3.11.2 transformers==5.17.0
```

如复现环境暂不提供Python 3.14，可优先使用Python 3.11或3.12并安装兼容的PyTorch
版本；模型代码只依赖标准PyTorch接口。

## 3. BERT文本特征恢复

附件3只提供BERT token信息，推理需要公开预训练模型：

```text
model id: google-bert/bert-base-uncased
local directory: .hf_model/bert-base-uncased
hidden size: 768
layers: 12
attention heads: 12
model.safetensors SHA-256:
68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
```

可使用Hugging Face CLI下载：

```bash
hf download google-bert/bert-base-uncased \
  --local-dir .hf_model/bert-base-uncased
```

由于该公开权重约423MB，超过竞赛全部附件50MB上限，因此未放入本材料包。该模型
仅用于把附件3 token还原为与附件2相同维度的文本特征，不进行情感任务微调。

## 4. 快速环境自检

```bash
python -c "import torch,numpy,yaml,matplotlib,transformers; \
print('torch=',torch.__version__); \
print('cuda=',torch.cuda.is_available()); \
print('numpy=',numpy.__version__); \
print('transformers=',transformers.__version__)"
```

随后执行模型最小测试：

```bash
python tests/run_tests.py
```
