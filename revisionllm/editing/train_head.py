"""Train the audio-visual UniVTG-style head on the cached VidChapters CLIP-L/14 + CLAP features. Reads LMDBs only.

  python revisionllm/editing/train_head.py \
      --train_anno data/chapters/chapters_vmr_train.jsonl --val_anno data/chapters/chapters_vmr_val.jsonl \
      --clip_lmdb data/chapters_audio/clip_l14_2fps_lmdb --clap_lmdb data/chapters_audio/clap_1s_lmdb \
      --out checkpoints/editing_head/head.pt
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.editing.dataset import ChapterAVDataset, collate, embed_titles
from revisionllm.editing.univtg_head import AVUniVTG, HeadConfig, save_head, univtg_loss
from revisionllm.multiquery.config import CacheConfig


def to_device(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot, n, parts = 0.0, 0, {}
    for b in loader:
        b = to_device(b, dev)
        loss, p = univtg_loss(model(b["vision"], b["audio"], b["query"], b["mask"]), b)
        tot, n = tot + float(loss), n + 1
        for k, x in p.items():
            parts[k] = parts.get(k, 0.0) + x
    return tot / max(n, 1), {k: x / max(n, 1) for k, x in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_anno", required=True)
    ap.add_argument("--val_anno", help="annotations of videos that have cached features; if omitted, --val_frac of the train videos is held out")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--clip_lmdb", required=True)
    ap.add_argument("--clap_lmdb", required=True)
    ap.add_argument("--clip_pt", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--out", default="checkpoints/editing_head/head.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=256, help="seconds per training crop")
    ap.add_argument("--samples_per_video", type=int, default=4)
    ap.add_argument("--query_prob", type=float, default=0.7, help="share of chapter-level queries; the rest use the joined chapter titles")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache_cfg = CacheConfig()

    def title_fn(split):
        cache = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"title_emb_{split}.npz")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        return lambda titles: embed_titles(titles, a.clip_pt, dev, cache)

    def make(anno, split, fixed, part=None):
        return ChapterAVDataset(anno, a.clip_lmdb, a.clap_lmdb, title_fn(split), a.crop, a.query_prob,
                                a.samples_per_video if not fixed else 1, cache_cfg.feature_fps, cache_cfg.audio_hop_sec,
                                a.seed, fixed, part=part, val_frac=a.val_frac if not a.val_anno else 0.0)

    # only part of the annotated videos have cached features, so without --val_anno a share of them is held out
    held_out = not a.val_anno and a.val_frac > 0
    train = make(a.train_anno, "train", False, "train" if held_out else None)
    val = make(a.val_anno, "val", True) if a.val_anno else (make(a.train_anno, "val", True, "val") if held_out else None)
    tl = DataLoader(train, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0)
    vl = DataLoader(val, a.bs, num_workers=a.workers, collate_fn=collate) if val else None

    model = AVUniVTG(HeadConfig()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    best, step = float("inf"), 0
    print(f"[head] {len(train.items)} videos, {len(train)} samples/epoch, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    for ep in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for b in tl:
            b = to_device(b, dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(b["vision"], b["audio"], b["query"], b["mask"])
            out = {k: v.float() for k, v in out.items()}
            loss, _ = univtg_loss(out, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += float(loss)
            step += 1
            if step % a.log_every == 0:
                print(f"[head] ep {ep + 1} step {step - ep * len(tl)}/{len(tl)} loss={float(loss):.4f} "
                      f"{(time.time() - t0) / (step - ep * len(tl)):.2f}s/step", flush=True)
        msg = f"[head] epoch {ep + 1}/{a.epochs} train_loss={run / len(tl):.4f} ({time.time() - t0:.0f}s)"
        score = run / len(tl)
        if vl is not None:
            score, parts = evaluate(model, vl, dev)
            msg += f" val_loss={score:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
        print(msg, flush=True)
        if score < best:
            best = score
            save_head(a.out, model, cache_cfg.feature_version, {"epoch": ep + 1, "score": score})
    print(f"[head] best={best:.4f} saved to {a.out}")


if __name__ == "__main__":
    main()
