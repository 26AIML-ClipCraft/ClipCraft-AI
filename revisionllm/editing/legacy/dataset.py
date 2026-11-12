"""Training samples for the UniVTG-style head from the cached VidChapters features (no feature extraction).

UniVTG's unified-label idea (get_saliency_labels_sub_as_query in the reference dataset.py): one annotation (a chapter
[start, end] with its title) gives all supervision. Foreground = seconds inside the chapter, boundary offsets = distances to
its start / end, saliency = pairs (a random second inside the chapter, a random second outside it) for the margin ranking.
A share of the samples (1 - query_prob) is video-level: the query is all chapter titles joined into one sentence, the
stand-in for a video title (the data has none), and the head must find the seconds that match it. UniVTG learns highlight /
summary from human saliency labels with the video title as query; there are none here, so the seconds that match ANY chapter
title best (top 30 % CLIP image-text cosine of the crop) are the foreground, the rest the background. That is a weak proxy
for "important", but unlike a query-free target it is observable from the input (the query is given). At inference the
video's title, or a one-sentence description written by a VLM, takes the place of the joined titles.
"""
import json
import os
import zlib

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz
from revisionllm.editing.shared import to_second_grid


_ENVS = {}


def open_env(path):
    """One read-only handle per (process, path): LMDB refuses a second open of the same path in a process, and a handle must
    not cross a fork, so DataLoader workers each open their own."""
    key = (os.getpid(), path)
    if key not in _ENVS:
        _ENVS[key] = lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=512)
    return _ENVS[key]


def joined_query(titles, max_titles=12, max_chars=200) -> str:
    """Video-level query used for training: the chapter titles of the video as one sentence (CLIP reads at most 77 tokens)."""
    return ", ".join(t.strip() for t in titles[:max_titles])[:max_chars]


def embed_titles(titles, clip_pt, device=None, cache=None):
    """CLIP ViT-L/14 text EOT features (the space of the frame features) -> (N, 768) unit vectors; cached as .npz."""
    key = np.array(titles, dtype=object)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if len(z["titles"]) == len(key) and all(a == b for a, b in zip(z["titles"], key)):
            return z["emb"]
    from revisionllm.data.feature_extraction.clip_extractor import ClipFeatureExtractor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ext = ClipFeatureExtractor(framerate=30, size=224, centercrop=True, model_name_or_path=clip_pt, device=device)
    _, eot = ext.encode_text(list(titles))
    emb = torch.stack(eot).float().cpu().numpy()
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    if cache:
        np.savez(cache, titles=key, emb=emb)
    return emb


class ChapterAVDataset(Dataset):
    """anno: chapters_vmr_*.jsonl rows {vid, duration, query:[titles], relevant_windows:[[[s,e]], ...]}."""

    def __init__(self, anno, clip_lmdb, clap_lmdb, title_emb_fn, crop=256, query_prob=0.7, samples_per_video=4,
                 feature_fps=2.0, audio_hop_sec=1.0, seed=0, fixed=False, n_pairs=2, null_top_frac=0.3,
                 part=None, val_frac=0.0):
        self.crop, self.query_prob, self.spv = crop, query_prob, samples_per_video
        self.fps, self.hop, self.seed, self.fixed = feature_fps, audio_hop_sec, seed, fixed
        self.n_pairs, self.null_top_frac = n_pairs, null_top_frac
        self.paths = (clip_lmdb, clap_lmdb)
        self._env = None
        rows = [json.loads(l) for l in open(anno)]
        probe = lmdb.open(clip_lmdb, readonly=True, lock=False, readahead=False)
        probe_txn = probe.begin()
        titles, self.items = [], []
        for r in rows:
            k = r["vid"].encode()
            if probe_txn.get(k) is None:
                continue
            if part is not None and (zlib.crc32(k) % 1000 < val_frac * 1000) != (part == "val"):
                continue                                  # deterministic video-level hold-out from the same file
            wins = [w[0] for w in r["relevant_windows"]]
            self.items.append({"vid": r["vid"], "wins": np.array(wins, dtype=np.float32), "first": len(titles)})
            titles += list(r["query"]) + [joined_query(r["query"])]       # chapter titles, then the video-level query
        if not self.items:
            raise RuntimeError("no annotated video has CLIP features in the given LMDB")
        self.emb = title_emb_fn(titles)
        del probe_txn
        probe.close()                                     # the real handles are opened lazily, once per worker process
        self._env = None

    def _open(self):
        if self._env is None:
            self._env = [open_env(p) for p in self.paths]
            self._txn = [e.begin() for e in self._env]

    def __len__(self):
        return len(self.items) * self.spv

    def _load(self, vid):
        self._open()
        v = loads_npz(self._txn[0].get(vid.encode()))
        ab = self._txn[1].get(vid.encode())
        a = loads_npz(ab) if ab is not None else None
        return to_second_grid(v, a, self.fps, self.hop)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i) if self.fixed else None)
        it = self.items[i % len(self.items)]
        v, a = self._load(it["vid"])
        T, L = len(v), min(self.crop, len(v))
        k = int(rng.integers(len(it["wins"])))
        s, e = map(float, it["wins"][k])
        c = float(np.clip((s + e) / 2, 0, T))
        lo, hi = max(0.0, min(e, T) - L), max(0.0, min(s, T - L))
        start = int(rng.uniform(lo, hi)) if hi > lo else int(np.clip(c - L / 2 + rng.uniform(-L / 4, L / 4), 0, T - L))
        v, a = v[start:start + L], a[start:start + L]
        t = np.arange(L, dtype=np.float32) + start + 0.5

        has_query = bool(rng.random() < self.query_prob)
        n_ch = len(it["wins"])
        all_t, video_q = self.emb[it["first"]: it["first"] + n_ch], self.emb[it["first"] + n_ch]
        fg = ((t >= s) & (t <= e)).astype(np.float32)
        if fg.sum() == 0:                                          # chapter shorter than a second
            fg[int(np.argmin(np.abs(t - c)))] = 1.0
        if not has_query:                                          # video-level query: fg = best match to any chapter title
            m = (v @ all_t.T).max(1)
            fg = (m >= np.quantile(m, 1 - self.null_top_frac)).astype(np.float32)
        pos_pool, neg_pool = np.flatnonzero(fg > 0), np.flatnonzero(fg == 0)
        valid = np.full(self.n_pairs, len(neg_pool) > 0)
        pos = rng.choice(pos_pool, self.n_pairs)
        neg = rng.choice(neg_pool, self.n_pairs) if len(neg_pool) else pos
        off = np.stack([np.maximum(t - s, 0), np.maximum(e - t, 0)], -1).astype(np.float32)
        return {"vision": torch.from_numpy(v), "audio": torch.from_numpy(a), "foreground": torch.from_numpy(fg),
                "offsets": torch.from_numpy(off), "query": torch.from_numpy((all_t[k] if has_query else video_q).astype(np.float32)),
                "has_query": has_query, "sal_pos": torch.from_numpy(pos.astype(np.int64)),
                "sal_neg": torch.from_numpy(neg.astype(np.int64)), "sal_valid": torch.from_numpy(valid)}


def collate(batch):
    B, T = len(batch), max(len(b["vision"]) for b in batch)
    out = {"mask": torch.zeros(B, T, dtype=torch.bool), "has_query": torch.tensor([b["has_query"] for b in batch])}
    for key in ("query", "sal_pos", "sal_neg", "sal_valid"):
        out[key] = torch.stack([b[key] for b in batch])
    for key in ("vision", "audio", "foreground", "offsets"):
        ref = batch[0][key]
        out[key] = torch.zeros((B, T) + tuple(ref.shape[1:]), dtype=ref.dtype)
    for n, b in enumerate(batch):
        L = len(b["vision"])
        out["mask"][n, :L] = True
        for key in ("vision", "audio", "foreground", "offsets"):
            out[key][n, :L] = b[key]
    return out
