# 典型样本时序对齐可视化

`visualize_typical_alignment.py` 只读取原视频和问题 1 已生成的文件，不修改原始数据或特征。
它会交叉核对命令行样本编号、`status.json`、`timeline.json`、NPZ 内的 `video_id/clip_id`
以及原视频路径，任何错配都会停止绘图。

## 输入与图层

- `features.npz`：0.5 秒窗口级语音/视觉特征、三模态 mask、有效长度和窗口边界；
- `word_alignment.csv`：赛题转写经 WhisperX 强制对齐后的真实词级起止时间；
- `timeline.json`：原视频路径和实际 2 fps 视觉采样时刻；
- 原始 MP4：FFmpeg 直接解码真实单声道波形和展示帧。

图从上到下为：真实视频帧、词级时间区间、原音轨波形、窗口级 F0（若不存在则使用
Loudness）、窗口级 OpenFace `AU12_r`（若不存在则使用实际存在的其他 AU 强度）、三模态
有效覆盖。灰色斜线表示无效或缺失，浅蓝背景是一个三模态同时有效的 0.5 秒典型窗口。
所有横轴均为原视频绝对时间；窗口聚合特征没有画成逐帧值。

## 运行

```bash
cd /root/HuaWeiCup
conda activate myenv

# 自动从成功样本中选择三模态覆盖和词对齐质量较高的一条
python visualize_typical_alignment.py \
  --output outputs/typical_alignment_figure.png \
  --pdf outputs/typical_alignment_figure.pdf

# 指定样本和原始时间局部范围
python visualize_typical_alignment.py \
  --video-id=-3g5yACwYnA --clip-id=13 \
  --data-dir dataset --start 1.0 --end 6.0 \
  --output outputs/alignment_-3g5yACwYnA_13.png

# 也可显式指定视频，但必须与处理记录中的绝对路径完全一致
python visualize_typical_alignment.py \
  --video-id=-3g5yACwYnA --clip-id=13 \
  --video-file dataset/-3g5yACwYnA/13.mp4 \
  --output outputs/alignment_explicit_video.png
```

PNG 固定以 300 dpi 保存，同时生成同名 JSON，记录自动选择理由、全部来源文件、展示范围、
帧时间、实际声学/AU字段和字体。`--pdf` 可选。若环境没有中文字体，程序会在终端明确警告，
并使用英文标签，保证论文图片不出现乱码。
