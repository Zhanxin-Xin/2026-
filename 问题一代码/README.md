# 问题 1：可追溯三模态特征流水线

## 已完成状态（2026-09-24）

已在 Conda `myenv`（Python 3.8.20）中完成附件 1 全部 100 条的实际提取和 QC：

- 100/100 样本状态为 `success`，QC `passed: true`；
- 特征维度统一为文本 768、语音 50、视觉 465，均填充至 70 窗；
- 原始视频时长范围 2.648--34.567 秒；
- 1,932 个对齐词中 1,920 个有有效时间，12 个未对齐词保留且不伪造时间；
- 1,570 个采样帧中 1,411 个通过人脸检测与置信度阈值；
- 全量词和帧时间戳越界数为 0。

实际工具版本包括 FFmpeg 4.4.2、OpenFace 2.2.0（源码固定标签构建）、WhisperX
3.2.0、openSMILE 2.5.0、PyTorch 2.4.1、Transformers 4.39.3。OpenFace 在
Ubuntu 22.04/OpenCV 4.5 下使用 dlib 19.13，并对 dlib 的旧 OpenCV 转换调用做了
`cvIplImage` 兼容补丁；不改变检测算法。

本工程只读取 `dataset/` 中的原始 XLSX 和 MP4。样本主键严格为
`video_id + clip_id`，统一使用原视频的 0.5 秒时间窗并零填充至 70 窗；
`padding_mask` 区分真实时间轴和填充，三个 `*_mask` 分别表示该模态在窗口中
是否有可信观测。标签只原样读取、核对、写入元数据，不参与特征计算。

## 当前数据与环境

- 标签文件：`dataset/label-100.xlsx`；sheet `label`；字段
  `video_id, clip_id, text, label, annotation`，共 100 行。
- 视频规则：`dataset/<video_id>/<clip_id>.mp4`，当前发现 100 个文件。
- 初次检查环境为 Python 3.8.17，已有 torch 2.4.1、transformers 4.46.3；
  初始时缺少 ffmpeg/ffprobe、OpenFace、WhisperX、openSMILE、pandas/openpyxl、soundfile；
  目前均已安装。`scan` 仍使用内置只读 OOXML 解析器，不依赖 pandas。

建议新建 Python 3.10 或 3.11 的隔离环境。`requirements.txt` 记录了本工程验证目标；
PyTorch/CUDA 应按机器驱动从官方渠道单独安装，避免盲目覆盖现有版本。系统工具必须另行安装：

- FFmpeg（包含 `ffmpeg` 和 `ffprobe`）；
- OpenFace 2.2.0 的 `FeatureExtraction`；
- Hugging Face `FacebookAI/roberta-base`（配置名为 `roberta-base`）；
- WhisperX 3.2.0 及其英语 wav2vec2 对齐模型。`myenv` 为 Python 3.8，因此使用兼容的
  3.2.0；SpeechBrain 固定为 0.5.16，避免新版本使用 Python 3.9 的类型语法。

默认 `text.local_files_only: true`，因此流水线不会静默下载 RoBERTa。WhisperX 首次加载
对齐模型也可能联网；应预先显式下载/缓存并记录来源。网络受限环境下不要把此项改为假，
除非你明确同意下载。工具不全时程序报错并逐样本记录，不会用其他语义不同的特征替代。

## 命令

```bash
cd /root/HuaWeiCup
python run_pipeline.py scan
python run_pipeline.py preflight

# 工具与模型准备好后先跑一个样本
python run_pipeline.py run --limit 1
python run_pipeline.py qc
python plot_timeline.py 'outputs/samples/<video_id>__<clip_id>'

# 试跑通过后处理全部 100 条；成功且签名未变化的样本自动跳过
python run_pipeline.py run
python run_pipeline.py qc
```

`qc` 在尚未完成全部 100 条时会返回非零状态；报告会分别给出
`structural_checks_passed` 与 `extraction_complete`，避免把“没有可检查的成功样本”误报为完成。

外部程序不在 PATH 时，在 `config.yaml` 中填写绝对路径。更改窗口、模型、阈值等关键
配置会改变断点签名并触发重跑；`--overwrite` 可强制重跑。单条失败会写状态后继续下一条。

## 特征定义

- 文本：赛题转写是唯一正式文本。WhisperX 仅做强制对齐，不调用 ASR；未匹配词保留且
  无时间戳。固定 RoBERTa-base 最后一层子词向量先在词内取均值；词跨窗时按词与窗口的
  时间重叠长度加权。不能对齐的词不伪造时间。
- 语音：FFmpeg 解码为 16 kHz 单声道 PCM，但元数据保留原音轨采样率和处理方式。
  openSMILE eGeMAPSv02 LowLevelDescriptors 在每窗按各短时帧计算均值和标准差。低于 RMS
  阈值的静音窗掩码为假。
- 视觉：FFmpeg 按 2 fps 采样，OpenFace 2.2 `FeatureExtraction` 输出 AU、姿态、视线及
  landmarks。`success != 1` 或 confidence 低于 0.8 的窗掩码为假，不解释为无情绪。

## 输出

`outputs/` 不包含模型权重或原视频：

- `manifest.csv` / `manifest_report.json`：源行号、二元键、原始路径及匹配审计；
- `environment.json` / `pipeline.log`：工具版本、参数执行日志；
- `summary.csv`：按样本组织的宽表，始终覆盖标签表全部 100 行；包含原始时长、
  0.5 秒对齐粒度、真实窗数、三模态维度、有效窗数、状态与异常；
- `modality_summary.csv`：按“样本 × 模态”组织的长表，共 100×3=300 行；逐行给出
  `video_id + clip_id`、模态、原始时长、特征维度、对齐粒度、有效长度、有效窗数、
  mask 字段、特征文件与原视频路径，便于直接作为论文中的全量结果表；
- `samples/<key>/features.npz`：`text/audio/visual [70,D]`、三个模态 mask、
  `padding_mask [70]`、窗起止时间、真实窗数、原始 ID 与源行号；
- `metadata.json`：媒体、维度、特征名、标签原值、模型与聚合规则；
- `word_alignment.csv`：词、源文本词序号、起止时间、置信分；
- `timeline.json`：词、音频静音窗、抽帧时刻与人脸置信度的统一时间线；
- `status.json`：参数签名、各阶段状态、告警、错误和版本；
- `qc_report.json`：形状、有限值、掩码、填充、时间范围和追溯文件检查。

典型样本 `-3g5yACwYnA__13` 还包含：

- `typical_alignment/alignment_windows.csv`：逐 0.5 秒窗列出文本片段、语音区间、
  实际视频帧时刻、三模态有效标志、人脸置信度及 NPZ 行号；
- `typical_alignment/alignment_with_frames.png`：真实抽样帧、对应文本及三模态 mask 图；
- `typical_alignment/frames/decoded_*.jpg`：图中使用的原视频帧；
- `typical_alignment/README.json`：该示例的源视频、特征文件、维度和抽帧规则。

部分 MP4 的视频流 duration 元数据短于容器 duration，但实际帧数、音轨和词级时间支持
容器时长。因此媒体检查以容器时长建立统一时间轴，同时在 `metadata.json` 中保留
`reported_stream_end_seconds` 和 `duration_rule` 供审计；超出实际可解码视觉帧的窗口保持
`visual_mask=false`，不会补造帧或解释为“没有情绪”。

填充值固定为 0，但不能单看数值判断有效性，必须同时读取相应 mask。

## 与题目四项说明的对应关系

1. **整体方案**：本 README 的“特征定义”、`config.yaml`、各样本 `metadata.json`、
   `timeline.json` 和 `features.npz` 共同说明从原素材到统一 0.5 秒时序特征的全过程。
2. **文件规范与全量汇总**：`summary.csv` 覆盖全部 100 个原始样本，
   `modality_summary.csv` 覆盖全部 300 个样本模态组合；实际特征及 mask 位于各样本
   `features.npz`，读取字段见上文。
3. **典型样本验证**：上述 `typical_alignment/` 四类文件给出文本片段、语音时段、
   真实视频帧和特征行的可视化及表格对应关系。
4. **可复现性**：`environment.json` 记录工具版本，`config.yaml` 集中记录核心参数，
   `pipeline.log`/`status.json` 记录运行过程，本 README“命令”章节给出完整运行顺序。
