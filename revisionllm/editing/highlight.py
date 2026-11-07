"""Highlight extraction: pick the most important moments of the whole video and cut them into one short video."""
import os
import subprocess
import tempfile
from typing import Dict, List

import numpy as np

from revisionllm.editing.scoring import smooth


def target_seconds(duration: float, cfg, requested: float = None) -> float:
    t = requested if requested else float(np.clip(cfg.highlight_ratio * duration, cfg.highlight_min_sec, cfg.highlight_max_sec))
    return min(t, duration)


def plan_highlight(importance: np.ndarray, events: np.ndarray, classes, duration: float, cfg, target_sec: float = None) -> List[Dict]:
    """Greedy peak picking on the importance curve. Each pick grows around its peak while the curve stays above 60% of the
    peak value (between clip_min_sec and clip_max_sec); picks keep clip_gap_sec apart. Returned in time order, touching clips merged."""
    T = len(importance)
    budget = target_seconds(duration, cfg, target_sec)
    cmin, cmax = int(cfg.clip_min_sec), int(cfg.clip_max_sec)
    curve = smooth(importance, 3)
    taken = np.zeros(T, dtype=bool)
    clips, total = [], 0.0
    for p in np.argsort(-curve):
        if total >= budget:
            break
        p = int(p)
        if taken[max(p - int(cfg.clip_gap_sec), 0): p + int(cfg.clip_gap_sec) + 1].any():
            continue
        s = e = p
        while e - s + 1 < cmax:
            left = curve[s - 1] if s > 0 and not taken[s - 1] else -1
            right = curve[e + 1] if e + 1 < T and not taken[e + 1] else -1
            if max(left, right) < 0.6 * curve[p] and e - s + 1 >= cmin:
                break
            if max(left, right) < 0:
                break
            if left >= right:
                s -= 1
            else:
                e += 1
        e += 1                                                   # [s, e) in seconds
        e = min(e, int(np.ceil(duration)))
        if e - s < 1:
            continue
        taken[max(s - int(cfg.clip_gap_sec), 0): e + int(cfg.clip_gap_sec)] = True
        clips.append((s, e))
        total += e - s
    clips.sort()
    merged = []
    for s, e in clips:
        if merged and s - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        pm = events[s:e].mean(0)
        top = int(np.argmax(pm))
        out.append({"start": float(s), "end": float(min(e, duration)), "score": round(float(importance[s:e].mean()), 4),
                    "dominant_audio": classes[top], "audio_prob": round(float(pm[top]), 3)})
    return out


def render_highlight(video_path: str, clips: List[Dict], out_path: str, has_audio: bool = True) -> str:
    """Cut each clip (re-encoded so the pieces concatenate cleanly) and join them in time order with ffmpeg."""
    if not clips:
        raise ValueError("no clips to render")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, c in enumerate(clips):
            part = os.path.join(tmp, f"part{i:03d}.mp4")
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{c['start']:.3f}", "-i", video_path,
                   "-t", f"{c['end'] - c['start']:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                   "-r", "30"]
            cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2"] if has_audio else ["-an"]
            subprocess.run(cmd + [part], check=True)
            parts.append(part)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", out_path], check=True)
    return out_path
