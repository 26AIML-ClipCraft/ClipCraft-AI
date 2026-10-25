"""Check semantic-cache accuracy: for query pairs of the same video, how often does a cache hit (cosine > threshold)
point at the same scene? A pair is 'correct' when the two queries' ground-truth segments overlap with IoU >= --iou.

  python revisionllm/multiquery/calibrate.py --anno data/charades/charades_test_anno.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)


def seg_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / (max(a[1], b[1]) - min(a[0], b[0]) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/charades/charades_test_anno.json")
    ap.add_argument("--clip_pt", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99])
    a = ap.parse_args()
    from revisionllm.multiquery.service import TextEncoder
    from revisionllm.multiquery.grounder import norm_sentence
    anno = json.load(open(a.anno))
    enc = TextEncoder(a.clip_pt)
    by_vid = defaultdict(list)
    for q in anno.values():
        by_vid[q["movie"]].append(q)
    sims, ok = [], []
    for vid, qs in by_vid.items():
        if len(qs) < 2:
            continue
        E = np.stack([TextEncoder.embedding(enc(norm_sentence(q["sentence"]))[1]) for q in qs])
        S = E @ E.T
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                sims.append(S[i, j]); ok.append(seg_iou(qs[i]["timestamps"], qs[j]["timestamps"]) >= a.iou)
    sims, ok = np.array(sims), np.array(ok)
    print(f"{len(sims)} same-video query pairs, {ok.mean()*100:.1f}% point at the same scene (GT IoU >= {a.iou})")
    print("threshold  hits   hit-rate  precision(same scene)  recall(of same-scene pairs)")
    for t in a.thresholds:
        h = sims > t
        print(f"{t:9.2f} {h.sum():6d} {h.mean()*100:8.1f}%  {ok[h].mean()*100 if h.any() else float('nan'):20.1f}%"
              f"  {h[ok].mean()*100:20.1f}%")


if __name__ == "__main__":
    main()
