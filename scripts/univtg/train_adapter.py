"""Align cached CLIP ViT-L/14 frame features (768, 2 fps) to UniVTG's CLIP-B/32 feature space (512, 1 clip/s).

Pairs come from Charades: ClipCraft cache (L/14, 2 fps) vs UniVTG-released Charades B/32 features (~1 clip/s).
Both are L2-normalised first (UniVTG normalises video features at load time). Cache frame index 2*i (+offset) <-> B/32 clip i.
Split by UniVTG's Charades train/test video lists. Nothing here touches the cache, the UniVTG repo or its weights.
"""
import json
import os
import sys
import time

import lmdb
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/minyoy/ClipCraft-AI")
from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz  # noqa: E402

A = "/home/minyoy/univtg_assets/"
CACHE = "/home/minyoy/ClipCraft-AI/data/chapters_audio/clip_l14_2fps_lmdb"
OUT = "/home/minyoy/univtg_work/"
dev = "cuda"
t0 = time.time()


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)


def vids(split):
    return sorted({json.loads(l)["vid"] for l in open(A + f"charades/metadata/charades_{split}.jsonl")})


splits = {"train": vids("train"), "test": vids("test")}
env = lmdb.open(CACHE, readonly=True, lock=False, readahead=False)


def load(vlist, offset, mode):
    X, Y, G = [], [], []
    with env.begin() as t:
        for gi, v in enumerate(vlist):
            buf = t.get(v.encode())
            if buf is None:
                continue
            x = l2(loads_npz(buf).astype(np.float32))
            y = l2(np.load(A + f"feat_vid/vid_clip/{v}.npz")["features"].astype(np.float32))
            n = min(len(y), (len(x) - offset) // 2)
            if n <= 0:
                continue
            if mode == "pick":
                xs = x[offset: offset + 2 * n: 2]
            else:  # mean of the two 2-fps frames inside each second
                xs = l2(x[: 2 * n].reshape(n, 2, -1).mean(1))
            X.append(xs); Y.append(y[:n]); G.append(np.full(n, gi))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def lstsq_val(Xtr, Ytr, Xva, Yva):
    W = np.linalg.lstsq(np.c_[Xtr, np.ones(len(Xtr))], Ytr, rcond=None)[0]
    P = np.c_[Xva, np.ones(len(Xva))] @ W
    return float((l2(P) * Yva).sum(-1).mean())


print("loading ...", flush=True)
cands = {}
for mode, off in [("pick", 0), ("pick", 1), ("mean", 0)]:
    Xtr, Ytr, _ = load(splits["train"][:1500], off, mode)
    Xva, Yva, _ = load(splits["test"][:500], off, mode)
    cands[(mode, off)] = lstsq_val(Xtr, Ytr, Xva, Yva)
    print(f"  mapping {mode} offset={off}: closed-form val cosine = {cands[(mode, off)]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
mode, off = max(cands, key=cands.get)
print("chosen mapping:", mode, off, flush=True)

Xtr, Ytr, Gtr = load(splits["train"], off, mode)
Xva, Yva, Gva = load(splits["test"], off, mode)
print(f"pairs train={len(Xtr)} val={len(Xva)}  videos train={len(set(Gtr))} val={len(set(Gva))}  ({time.time()-t0:.0f}s)", flush=True)
print("closed-form val cosine (all):", round(lstsq_val(Xtr, Ytr, Xva, Yva), 4), flush=True)

# --- gradient training of the Adapter (Linear 768 -> 512), cosine loss ---
torch.manual_seed(0)
Xt, Yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
Xv, Yv = torch.tensor(Xva, device=dev), torch.tensor(Yva, device=dev)
ad = nn.Linear(768, 512).to(dev)
opt = torch.optim.AdamW(ad.parameters(), lr=1e-3, weight_decay=1e-4)
E, B = 40, 4096
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E * ((len(Xt) + B - 1) // B))
best = -1
for ep in range(E):
    perm = torch.randperm(len(Xt), device=dev)
    for i in range(0, len(Xt), B):
        idx = perm[i:i + B]
        loss = 1 - nn.functional.cosine_similarity(ad(Xt[idx]), Yt[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        va = nn.functional.cosine_similarity(ad(Xv), Yv).mean().item()
    if va > best:
        best = va
        torch.save({"state_dict": ad.state_dict(), "val_cos": va, "mapping": (mode, off)}, OUT + "adapter_align.pt")
    if ep % 5 == 0 or ep == E - 1:
        print(f"ep{ep:02d} train_loss={loss.item():.4f} val_cos={va:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)

# within-video frame discrimination: does the adapted L/14 frame pick the matching B/32 frame (top-1 over frames of the same video)?
with torch.no_grad():
    P = nn.functional.normalize(ad(Xv), dim=-1).cpu().numpy()
hit, tot = 0, 0
for g in np.unique(Gva):
    m = Gva == g
    if m.sum() < 3:
        continue
    S = P[m] @ Yva[m].T
    hit += int((S.argmax(1) == np.arange(m.sum())).sum()); tot += int(m.sum())
print(f"within-video top-1 frame match (val): {hit/tot:.4f}  | total time {time.time()-t0:.0f}s", flush=True)
