"""
Zero-shot check: does the CLAP audio of a video carry information about WHERE a chapter query happens?   (no training)

For every Test-sub query we build per-second similarity curves over the video
  * audio  : CLAP text embedding of the query  .  CLAP audio embedding of each second   (same 512-d space)
  * visual : CLIP text embedding of the query  .  CLIP frame embeddings (2 fps, averaged to 1 s)  (same 768-d space)
  * combos : z(visual) + w * z(audio)
and measure, per query, the ROC-AUC of "second lies inside the GT chapter" (0.5 = chance), plus the share of queries
whose single best second falls inside the GT interval (compare with the chance level = GT fraction of the video).
Results are split into audio-cue queries (keyword rule of label_query_types.py) vs the rest, and by video length.

    python revisionllm/eval/clap_zeroshot.py --anno data/chapters_audio/subsets/test_sub_anno.json \
        --clip_lmdb data/chapters_audio/clip_l14_2fps_lmdb --clap_lmdb data/chapters_audio/clap_1s_lmdb \
        --qfeat data/chapters_audio/clip_L14_text_features_test --out clap_zeroshot.json
"""
import argparse
import io
import json
import math
import os
import re
import sys
from collections import defaultdict

import lmdb
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)
from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz

FALLBACK_KW = ["music", "song", "sing", "singing", "guitar", "piano", "drum", "beat", "melody", "concert", "laugh", "joke",
               "applause", "clap", "cheer", "crowd", "audience", "chant", "asmr", "voice", "vocal", "rap", "karaoke", "q&a"]
try:
    from revisionllm.data.audio_pipeline.label_query_types import KW_RE
except Exception:
    KW_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in FALLBACK_KW) + r")", re.IGNORECASE)

DUR_BUCKETS = (("<=500s", 0, 500), ("500s-25min", 500, 1500), (">25min", 1500, float("inf")))


def auc(x, pos):
    """ROC-AUC of scores x for boolean labels pos (rank statistic, average ranks for ties)."""
    n_pos = int(pos.sum()); n_neg = len(x) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]                                    # average ranks over ties
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def zs(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-8 else 1.0)


def smooth(x, k):
    if k <= 1:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def clap_text_embeddings(sentences, model_name, device, bs=128):
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(model_name).to(device).eval()
    proc = ClapProcessor.from_pretrained(model_name)
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), bs):
            inp = proc(text=sentences[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=77)
            e = model.get_text_features(**{k: v.to(device) for k, v in inp.items()})
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--clip_lmdb", required=True)
    ap.add_argument("--clap_lmdb", required=True)
    ap.add_argument("--qfeat", required=True, help="CLIP text features LMDB keyed by qid")
    ap.add_argument("--clap_model", default="laion/clap-htsat-fused")
    ap.add_argument("--smooth", default="1,5", help="comma list of moving-average widths (seconds); 1 = none")
    ap.add_argument("--weights", default="0.25,0.5,1.0", help="audio weights w in z(visual)+w*z(audio)")
    ap.add_argument("--text_cache", default=None, help=".npz cache of CLAP text embeddings")
    ap.add_argument("--max_queries", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    smooths = [int(x) for x in a.smooth.split(",")]
    weights = [float(x) for x in a.weights.split(",")]

    anno = json.load(open(a.anno))
    qids = list(anno.keys())
    if a.max_queries:
        qids = qids[:a.max_queries]
    sentences = [anno[q]["sentence"] for q in qids]

    if a.text_cache and os.path.exists(a.text_cache) and list(np.load(a.text_cache)["qids"]) == qids:
        tclap = np.load(a.text_cache)["emb"]
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tclap = clap_text_embeddings(sentences, a.clap_model, dev)
        if a.text_cache:
            np.savez(a.text_cache, qids=np.array(qids), emb=tclap)
    print("CLAP text embeddings:", tclap.shape, flush=True)

    clip_env = lmdb.open(a.clip_lmdb, readonly=True, lock=False, readahead=False)
    clap_env = lmdb.open(a.clap_lmdb, readonly=True, lock=False, readahead=False)
    q_env = lmdb.open(a.qfeat, readonly=True, lock=False, readahead=False)

    by_video = defaultdict(list)
    for i, q in enumerate(qids):
        by_video[anno[q]["movie"]].append(i)

    names = ["audio", "visual"] + [f"visual+{w}*audio" for w in weights]
    rec = {s: {n: {} for n in names} for s in smooths}      # smooth -> setting -> qid -> auc
    hit = {s: {n: {} for n in names} for s in smooths}
    chance = {}
    meta = {}
    n_done = 0
    with clip_env.begin() as ct, clap_env.begin() as at, q_env.begin() as qt:
        for vi, (vid, idxs) in enumerate(by_video.items()):
            vb, ab = ct.get(vid.encode()), at.get(vid.encode())
            if vb is None or ab is None:
                continue
            V = loads_npz(vb).astype(np.float32); A = loads_npz(ab).astype(np.float32)
            Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
            nsec_v = len(Vn) // 2
            if nsec_v == 0:
                continue
            Vs = Vn[:nsec_v * 2].reshape(nsec_v, 2, -1).mean(1)           # 1 s resolution
            T = min(len(A), nsec_v)
            An, Vs = A[:T], Vs[:T]
            for i in idxs:
                q = qids[i]; d = anno[q]
                s_, e_ = d["timestamps"]
                lo, hi = int(math.floor(s_)), int(math.ceil(e_))
                pos = np.zeros(T, dtype=bool); pos[max(lo, 0):min(hi, T)] = True
                if pos.sum() == 0 or pos.sum() == T:
                    continue
                qv = np.load(io.BytesIO(qt.get(q.encode())), allow_pickle=True)["cls_features"].astype(np.float32)
                qv = qv / max(np.linalg.norm(qv), 1e-8)
                curves = {"audio": An @ tclap[i], "visual": Vs @ qv}
                for s in smooths:
                    za = zs(smooth(curves["audio"], s)); zv = zs(smooth(curves["visual"], s))
                    cand = {"audio": za, "visual": zv}
                    for w in weights:
                        cand[f"visual+{w}*audio"] = zv + w * za
                    for n, c in cand.items():
                        rec[s][n][q] = auc(c, pos)
                        hit[s][n][q] = float(pos[int(np.argmax(c))])
                chance[q] = float(pos.mean())
                meta[q] = (bool(KW_RE.search(d["sentence"])), float(d["movie_duration"]))
            n_done += len(idxs)
            if (vi + 1) % 100 == 0:
                print(f"{vi + 1}/{len(by_video)} videos, {len(chance)} usable queries", flush=True)

    def agg(qs, s, n):
        v = np.array([rec[s][n][q] for q in qs if q in rec[s][n] and rec[s][n][q] is not None])
        h = np.array([hit[s][n][q] for q in qs if q in hit[s][n]])
        return {"n": int(len(v)), "AUC": float(v.mean()), "AUC_se": float(v.std() / math.sqrt(max(len(v), 1))),
                "top1_in_GT": float(h.mean())} if len(v) else {"n": 0}

    groups = {"all": list(chance), "audio_cue": [q for q in chance if meta[q][0]], "other": [q for q in chance if not meta[q][0]]}
    for nm, lo, hi in DUR_BUCKETS:
        groups[nm] = [q for q in chance if lo < meta[q][1] <= hi]
    result = {"chance_top1_in_GT": {g: float(np.mean([chance[q] for q in qs])) if qs else None for g, qs in groups.items()},
              "n_queries": {g: len(qs) for g, qs in groups.items()}, "results": {}}
    for s in smooths:
        result["results"][f"smooth{s}s"] = {g: {n: agg(qs, s, n) for n in names} for g, qs in groups.items()}
        # paired gain of adding audio to vision (same queries)
        gains = {}
        for g, qs in groups.items():
            gains[g] = {}
            for w in weights:
                n = f"visual+{w}*audio"
                dif = np.array([rec[s][n][q] - rec[s]["visual"][q] for q in qs if rec[s][n].get(q) is not None and rec[s]["visual"].get(q) is not None])
                gains[g][n] = {"dAUC": float(dif.mean()), "dAUC_ci95": float(1.96 * dif.std() / math.sqrt(max(len(dif), 1)))} if len(dif) else {}
        result["results"][f"smooth{s}s"]["paired_gain_over_visual"] = gains

    for s in smooths:
        print(f"\n=== smoothing {s}s ===  (AUC; 0.5 = chance)")
        for g in groups:
            if not result['n_queries'][g]:
                continue
            print(f"[{g}] n={result['n_queries'][g]}  chance top1-in-GT={result['chance_top1_in_GT'][g]:.3f}")
            for n in names:
                r = result["results"][f"smooth{s}s"][g][n]
                if r.get("n"):
                    print(f"   {n:22s} AUC {r['AUC']:.3f} +-{1.96 * r['AUC_se']:.3f}   top1-in-GT {r['top1_in_GT']:.3f}")
            for n, r in result["results"][f"smooth{s}s"]["paired_gain_over_visual"][g].items():
                if r:
                    print(f"   gain {n:17s} dAUC {r['dAUC']:+.4f} +-{r['dAUC_ci95']:.4f}")
    if a.out:
        json.dump(result, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
