"""
Case figure for the report: video frames, audio evidence, ground truth and the R1 / R2 / R4 predictions on one time axis.

    python scripts/audio/make_case_figure.py --qid c2473 --prompt "a person laughing" --out docs/figures/example.png
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
import lmdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz
from revisionllm.eval.dense_metrics import load_logs
from revisionllm.eval.qualitative_cases import top1

ap = argparse.ArgumentParser()
ap.add_argument("--qid", required=True)
ap.add_argument("--anno", default="data/charades/charades_test_anno.json")
ap.add_argument("--video_dir", default="data/charades/Charades_v1_480")
ap.add_argument("--eval_root", default="checkpoints/audio_charades_long/eval")
ap.add_argument("--rows", default="R1=R1_seed0,R2=R2_seed0,R4=R4")
ap.add_argument("--clap_lmdb", default="data/chapters_audio/clap_1s_lmdb")
ap.add_argument("--prompt", default=None, help="text for the CLAP audio-text similarity curve (default: the query)")
ap.add_argument("--n_thumbs", type=int, default=11)
ap.add_argument("--out", required=True)
a = ap.parse_args()

anno = json.load(open(a.anno))
d = anno[a.qid]
vid, dur, (gs, ge) = d["movie"], float(d["movie_duration"]), d["timestamps"]
path = os.path.join(a.video_dir, vid + ".mp4")

# --- thumbnails ---------------------------------------------------------------------------------------------------
n = a.n_thumbs
tt = (np.arange(n) + 0.5) * dur / n
thumbs = []
for t in tt:
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", "scale=240:-1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    thumbs.append(Image.open(io.BytesIO(png)).convert("RGB"))

# --- audio: RMS energy + CLAP audio-text similarity -----------------------------------------------------------------
wav = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                                   capture_output=True).stdout, dtype=np.float32)
hop = 4000   # 0.25 s
rms = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2) + 1e-12) for i in range(0, len(wav) - hop, hop)])
rms_t = (np.arange(len(rms)) + 0.5) * 0.25

env = lmdb.open(a.clap_lmdb, readonly=True, lock=False)
with env.begin() as txn:
    A = loads_npz(txn.get(vid.encode())).astype(np.float32)
import torch
from transformers import ClapModel, ClapProcessor
m = ClapModel.from_pretrained("laion/clap-htsat-fused").eval(); p = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
with torch.no_grad():
    te = torch.nn.functional.normalize(m.get_text_features(**p(text=[a.prompt or d["sentence"]], return_tensors="pt", padding=True)), dim=-1)[0].numpy()
sim = A @ te
sim_t = np.arange(len(sim)) + 0.5

# --- predictions --------------------------------------------------------------------------------------------------
rows = [r.split("=") for r in a.rows.split(",")]
pred = {}
for name, sub in rows:
    log = load_logs(os.path.join(a.eval_root, sub, "dense"))[a.qid]
    t1 = top1(log, dur)
    pred[name] = (t1[0], t1[1], t1[3]) if t1 else (None, None, 0.0)

# --- figure -------------------------------------------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 7.0))
gs_ = fig.add_gridspec(3, 1, height_rows := None) if False else fig.add_gridspec(3, 1, height_ratios=[0.62, 1.5, 2.1], hspace=0.28)
ax1, ax2, ax3 = (fig.add_subplot(gs_[i]) for i in range(3))
for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, dur)
w = dur / n
for t, im in zip(tt, thumbs):
    ax1.imshow(np.asarray(im), extent=[t - w / 2, t + w / 2, 0, 1], aspect="auto")
ax1.set_xlim(0, dur); ax1.set_ylim(0, 1); ax1.set_yticks([])
ax1.set_xticks(tt); ax1.set_xticklabels([f"{t:.0f}s" for t in tt], fontsize=8)
ax1.set_title(f'Video {vid}  ({dur:.1f}s)   query: "{d["sentence"]}"', fontsize=12, loc="left")

ax2.fill_between(rms_t, 0, rms / (rms.max() + 1e-9), color="#9aa5b8", alpha=0.6, step="mid", label="audio energy (RMS, normalized)")
ax2.set_ylim(0, 1.05); ax2.set_ylabel("energy", fontsize=9)
ax2b = ax2.twinx()
ax2b.plot(sim_t, sim, color="#d6336c", lw=2, marker="o", ms=3, label=f'CLAP audio-text similarity to "{a.prompt or d["sentence"]}"')
ax2b.set_ylabel("CLAP similarity", fontsize=9, color="#d6336c")
ax2.set_xticks(np.arange(0, dur, 5))
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

colors = {"GT": "#2f9e44", "R1": "#868e96", "R2": "#1c7ed6", "R4": "#f08c00"}
labels = {"GT": "ground truth", "R1": "R1  (no audio)", "R2": "R2  (audio)", "R4": "R4  (R2, audio off)"}
order = ["GT"] + [r[0] for r in rows]
for i, nme in enumerate(order):
    y = len(order) - 1 - i
    if nme == "GT":
        s, e, txt = gs, ge, f"{gs:.1f}-{ge:.1f}s"
    else:
        s, e, iou = pred[nme]
        txt = "no prediction" if s is None else f"{s:.1f}-{e:.1f}s   IoU {iou:.2f}"
    if s is not None:
        ax3.barh(y, e - s, left=s, height=0.55, color=colors[nme], alpha=0.9)
        ax3.text(min(e + 0.4, dur - 0.1) if e < dur * 0.72 else s - 0.4, y, txt, va="center",
                 ha="left" if e < dur * 0.72 else "right", fontsize=10, color="#222")
ax3.set_yticks(range(len(order))); ax3.set_yticklabels([labels[n_] for n_ in reversed(order)], fontsize=10)
ax3.set_ylim(-0.6, len(order) - 0.4); ax3.set_xlabel("time (s)")
for ax in (ax1, ax2, ax3):
    ax.axvspan(gs, ge, color="#2f9e44", alpha=0.12, lw=0)
for ax in (ax2, ax3):
    ax.grid(axis="x", alpha=0.25)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=150, bbox_inches="tight")
print("wrote", a.out)
