#!/usr/bin/env python3
"""Paper-ready, traceable multimodal alignment visualization using real outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


COLORS = {"text": "#1f5f99", "audio": "#2a9d9f", "visual": "#bd5b5b"}


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path.resolve()


def auto_select(outputs: Path) -> Tuple[str, str, str]:
    rows = read_csv(require_file(outputs / "summary.csv", "summary table"))
    choices = []
    for row in rows:
        if row.get("status") != "success":
            continue
        sample = outputs / "samples" / row["sample_key"]
        try:
            words = read_csv(sample / "word_alignment.csv")
            valid_words = sum(bool(x.get("valid", "").lower() == "true" and x.get("start") and x.get("end")) for x in words)
            valid_length = max(1, int(row["valid_length"]))
            counts = [int(row[f"{m}_valid_windows"]) for m in ("text", "audio", "visual")]
            if not valid_words or min(counts) <= 0:
                continue
            word_ratio = valid_words / max(1, len(words))
            coverage = sum(min(1.0, n / valid_length) for n in counts) / 3
            # Prefer complete alignment and broad three-modality coverage; duration
            # only breaks ties in favor of a readable, information-rich example.
            score = 4 * word_ratio + 3 * coverage + min(float(row["duration_seconds"]), 15) / 100
            choices.append((score, row["video_id"], row["clip_id"], word_ratio, coverage))
        except (OSError, KeyError, ValueError):
            continue
    if not choices:
        raise RuntimeError("no successful sample has valid text, audio, visual and word timestamps")
    _, video_id, clip_id, wr, cov = max(choices)
    reason = f"auto-selected: all modalities valid; aligned-word completeness={wr:.1%}, mean modality coverage={cov:.1%}"
    return video_id, clip_id, reason


def choose_font(matplotlib) -> Optional[str]:
    from matplotlib import font_manager
    candidates = ["Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Micro Hei", "SimHei", "Microsoft YaHei"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    print("WARNING: no Chinese font found; using English labels to avoid missing glyphs", file=sys.stderr)
    return None


def decode_audio(ffmpeg: str, video: Path, sample_rate: int = 16000):
    import numpy as np
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wave = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    if not len(wave):
        raise RuntimeError(f"decoded audio is empty: {video}")
    return wave, sample_rate


def decode_frames(ffmpeg: str, video: Path, frame_times: Sequence[float], destination: Path) -> List[Path]:
    """Decode the pipeline's real 2-fps sequence; requested times must exist."""
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "sample_%04d.jpg"
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", "fps=2", "-vsync", "0", "-q:v", "2", str(pattern)], check=True)
    decoded = sorted(destination.glob("sample_*.jpg"))
    paths = []
    for t in frame_times:
        index = int(round(t * 2))
        if index < 0 or index >= len(decoded):
            raise RuntimeError(f"no real decoded 2-fps frame at t={t:.3f}s")
        paths.append(decoded[index])
    return paths


def shade_invalid(ax, mask, starts, ends, lo: float, hi: float) -> None:
    for valid, left, right in zip(mask, starts, ends):
        left, right = max(float(left), lo), min(float(right), hi)
        if right > left and not valid:
            ax.axvspan(left, right, facecolor="#dddddd", alpha=.65, hatch="///", edgecolor="#aaaaaa", linewidth=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-id"); p.add_argument("--clip-id")
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--video-file", type=Path, help="explicit source video; identity is still cross-checked")
    p.add_argument("--output", type=Path, default=Path("outputs/typical_alignment_figure.png"))
    p.add_argument("--pdf", type=Path, help="optional PDF path")
    p.add_argument("--start", type=float, default=0.0); p.add_argument("--end", type=float)
    p.add_argument("--frame-count", type=int, default=4)
    p.add_argument("--highlight-window", type=int, help="absolute 0.5-s window index")
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    if bool(args.video_id) != bool(args.clip_id):
        p.error("--video-id and --clip-id must be supplied together")
    outputs = args.outputs_dir.resolve(); data_dir = args.data_dir.resolve()
    selection_reason = "explicitly selected by command line"
    if not args.video_id:
        args.video_id, args.clip_id, selection_reason = auto_select(outputs)
    sample_key = f"{args.video_id}__{args.clip_id}"
    sample = outputs / "samples" / sample_key
    require_file(sample / "features.npz", "feature NPZ")
    metadata = json.loads(require_file(sample / "metadata.json", "metadata").read_text(encoding="utf-8"))
    timeline = json.loads(require_file(sample / "timeline.json", "timeline").read_text(encoding="utf-8"))
    status = json.loads(require_file(sample / "status.json", "status").read_text(encoding="utf-8"))
    words = read_csv(require_file(sample / "word_alignment.csv", "word alignment"))
    if status.get("video_id") != args.video_id or str(status.get("clip_id")) != str(args.clip_id) or timeline.get("sample_key") != sample_key:
        raise ValueError("sample identity mismatch among arguments, status.json and timeline.json")
    expected_video = (data_dir / args.video_id / f"{args.clip_id}.mp4").resolve()
    video = require_file(args.video_file.resolve() if args.video_file else expected_video, "source video")
    recorded = Path(timeline["video_path"]).resolve()
    if video != recorded:
        raise ValueError(f"video identity/path mismatch: chosen={video}, recorded={recorded}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    font = choose_font(matplotlib)
    with np.load(sample / "features.npz", allow_pickle=False) as z:
        if str(z["video_id"]) != args.video_id or str(z["clip_id"]) != str(args.clip_id):
            raise ValueError("NPZ sample identity mismatch")
        length = int(z["valid_length"]); duration = float(z["duration_seconds"])
        starts=z["window_start"][:length].copy(); ends=z["window_end"][:length].copy()
        features={m:z[m][:length].copy() for m in ("audio","visual")}
        masks={m:z[f"{m}_mask"][:length].copy() for m in ("text","audio","visual")}
        padding=z["padding_mask"].copy()
    if not padding[:length].all() or padding[length:].any():
        raise ValueError("invalid valid_length/padding_mask relationship")
    lo=max(0.0,args.start); hi=min(duration,args.end if args.end is not None else duration)
    if not (lo < hi): p.error(f"invalid time range [{lo}, {hi}] for duration {duration}")
    in_range=(ends > lo) & (starts < hi); centers=(starts+ends)/2

    audio_names=metadata["audio"]["feature_names"]; visual_names=metadata["visual"]["feature_names"]
    acoustic_name = "F0semitoneFrom27.5Hz_sma3nz__mean" if "F0semitoneFrom27.5Hz_sma3nz__mean" in audio_names else "Loudness_sma3__mean"
    if acoustic_name not in audio_names: raise RuntimeError("no interpretable F0 or Loudness field in saved audio features")
    au_name = "AU12_r" if "AU12_r" in visual_names else next((x for x in visual_names if x.startswith("AU") and x.endswith("_r")), None)
    if not au_name: raise RuntimeError("no OpenFace AU intensity field in saved visual features")
    acoustic=features["audio"][:,audio_names.index(acoustic_name)]
    au=features["visual"][:,visual_names.index(au_name)]

    wave,sr=decode_audio(args.ffmpeg,video); wave_t=np.arange(len(wave))/sr
    # Only use times explicitly recorded by the visual pipeline and available in the requested range.
    actual_times=[float(x["time"]) for x in timeline.get("frames",[]) if lo <= float(x["time"]) <= hi]
    if not actual_times: raise RuntimeError("no real visual sampling time in requested range")
    n=max(1,min(args.frame_count,len(actual_times)))
    chosen=[actual_times[i] for i in sorted(set(np.linspace(0,len(actual_times)-1,n,dtype=int)))]
    with tempfile.TemporaryDirectory(prefix="alignment_frames_") as tmp:
        frame_paths=decode_frames(args.ffmpeg,video,chosen,Path(tmp))
        frame_images=[plt.imread(x) for x in frame_paths]

        fig=plt.figure(figsize=(13,12),layout="constrained")
        gs=fig.add_gridspec(6,1,height_ratios=[2.1,1.45,1.25,1.15,1.15,.75])
        frame_grid=gs[0].subgridspec(1,len(chosen))
        for j,(t,img) in enumerate(zip(chosen,frame_images)):
            ax=fig.add_subplot(frame_grid[0,j]); ax.imshow(img); ax.axis("off")
            ax.set_title(f"t={t:.2f} s\n2-fps sample #{int(round(t*2))+1}",fontsize=9)
        axes=[fig.add_subplot(gs[i]) for i in range(1,6)]
        ax_words,ax_wave,ax_acoustic,ax_au,ax_cov=axes

        valid_words=[w for w in words if w.get("valid","").lower()=="true" and w.get("start") and w.get("end")
                     and float(w["end"])>lo and float(w["start"])<hi]
        lanes=4
        for i,w in enumerate(valid_words):
            s,e=float(w["start"]),float(w["end"]); lane=i%lanes
            ax_words.broken_barh([(s,e-s)],(lane+.12,.55),facecolors=COLORS["text"])
            ax_words.text((s+e)/2,lane+.72,w["word"],ha="center",va="bottom",fontsize=8,clip_on=True)
        ax_words.set_ylim(0,lanes+.35); ax_words.set_yticks([]); ax_words.set_ylabel("Aligned\nwords")

        keep=(wave_t>=lo)&(wave_t<=hi)
        ax_wave.plot(wave_t[keep],wave[keep],color=COLORS["audio"],lw=.35,rasterized=True)
        ax_wave.set_ylabel("Audio\nwaveform"); ax_wave.axhline(0,color="#777",lw=.4)

        av=in_range & masks["audio"]
        ax_acoustic.plot(centers[av],acoustic[av],"o-",color=COLORS["audio"],lw=1.4,ms=3)
        shade_invalid(ax_acoustic,masks["audio"],starts,ends,lo,hi)
        meaning="F0 (semitones re 27.5 Hz), window mean" if acoustic_name.startswith("F0") else "Loudness, window mean"
        ax_acoustic.set_ylabel(meaning); ax_acoustic.set_title(f"Saved acoustic field: {acoustic_name}",loc="left",fontsize=9)

        vv=in_range & masks["visual"]
        ax_au.plot(centers[vv],au[vv],"o-",color=COLORS["visual"],lw=1.4,ms=3)
        shade_invalid(ax_au,masks["visual"],starts,ends,lo,hi)
        ax_au.set_ylabel(f"{au_name} intensity\n(window-level)"); ax_au.set_title(f"Saved OpenFace field: {au_name}",loc="left",fontsize=9)

        for y,m in enumerate(("text","audio","visual")):
            for valid,s,e in zip(masks[m],starts,ends):
                left,right=max(float(s),lo),min(float(e),hi)
                if right>left:
                    ax_cov.broken_barh([(left,right-left)],(y+.15,.62),facecolors=COLORS[m] if valid else "#e2e2e2",
                                       hatch=None if valid else "///",edgecolor="#999" if not valid else "none")
        ax_cov.set_ylim(0,3); ax_cov.set_yticks([.46,1.46,2.46],labels=["Text","Audio","Visual"])
        ax_cov.set_ylabel("Coverage"); ax_cov.set_xlabel("Original video time (s)")

        highlight=args.highlight_window
        if highlight is None:
            candidates=np.flatnonzero(in_range & masks["text"] & masks["audio"] & masks["visual"])
            highlight=int(candidates[len(candidates)//2]) if len(candidates) else None
        for ax in axes:
            ax.set_xlim(lo,hi); ax.grid(axis="x",color="#dddddd",lw=.5)
            for edge in starts[(starts>=lo)&(starts<=hi)]: ax.axvline(edge,color="#c8d8e8",lw=.35,zorder=0)
            if highlight is not None and 0<=highlight<length:
                ax.axvspan(max(lo,float(starts[highlight])),min(hi,float(ends[highlight])),color="#dceaf7",alpha=.45,zorder=-1)
        for ax in axes[:-1]: ax.tick_params(labelbottom=False)
        invalid=Patch(facecolor="#e2e2e2",hatch="///",edgecolor="#999",label="invalid / missing")
        ax_cov.legend(handles=[Patch(color=COLORS[m],label=f"valid {m}") for m in COLORS]+[invalid],
                      ncol=4,loc="upper center",bbox_to_anchor=(.5,-.55),fontsize=8)
        local_note="full duration" if lo==0 and math.isclose(hi,duration,abs_tol=1e-3) else f"absolute-time excerpt [{lo:.2f}, {hi:.2f}] s"
        fig.suptitle(f"Multimodal temporal alignment — {sample_key}\nvideo duration={duration:.3f} s; {local_note}; 0.5-s feature windows",fontsize=14)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output,dpi=300,bbox_inches="tight")
        if args.pdf:
            args.pdf.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.pdf,bbox_inches="tight")
        plt.close(fig)

    report={"sample_key":sample_key,"selection_reason":selection_reason,"source_video":str(video),
            "feature_file":str((sample/"features.npz").resolve()),"word_alignment_file":str((sample/"word_alignment.csv").resolve()),
            "time_range_seconds":[lo,hi],"duration_seconds":duration,"frame_times_seconds":chosen,
            "frame_labels":"sequential samples from the same 2-fps rule as feature extraction",
            "acoustic_field":acoustic_name,"visual_field":au_name,"font":font or "DejaVu Sans / English labels",
            "output_png":str(args.output.resolve()),"output_pdf":str(args.pdf.resolve()) if args.pdf else None,
            "missing_elements":[]}
    report_path=args.output.with_suffix(".json"); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
