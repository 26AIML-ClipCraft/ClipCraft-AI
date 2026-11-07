"""Does the trained head change the highlight / summary compared with the training-free curve? Runs on the held-out videos of
the cached LMDBs (no video files needed) and prints, per method, how the selected seconds look.

  python revisionllm/editing/compare_head.py --anno data/chapters/chapters_vmr_train.jsonl \
      --clip_lmdb data/chapters_audio/clip_l14_2fps_lmdb --clap_lmdb data/chapters_audio/clap_1s_lmdb
"""
import argparse
import json
import os
import sys

import lmdb
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz
from revisionllm.editing.audio_events import EVENT_PROMPTS, AudioEventTagger, prompt_embeddings
from revisionllm.editing.config import EditConfig
from revisionllm.editing.dataset import ChapterAVDataset, embed_titles
from revisionllm.editing.highlight import plan_highlight, target_seconds
from revisionllm.editing.scoring import importance_curve, rank_norm, smooth, visual_change
from revisionllm.editing.shared import AVRepresentation, to_second_grid
from revisionllm.editing.summarize import summarize
from revisionllm.editing.univtg_head import HeadScorer
from revisionllm.multiquery.config import CacheConfig


def mask_of(clips, T):
    m = np.zeros(T, bool)
    for c in clips:
        m[int(c["start"]):int(c["end"])] = True
    return m


def chapter_coverage(mask, wins):
    return float(np.mean([mask[int(s):int(e)].any() for s, e in wins if int(e) > int(s)] or [0.0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--clip_lmdb", required=True)
    ap.add_argument("--clap_lmdb", required=True)
    ap.add_argument("--clip_pt", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--head", default="checkpoints/editing_head/head.pt")
    ap.add_argument("--title_cache", default="checkpoints/editing_head/title_emb_val.npz")
    ap.add_argument("--val_frac", type=float, default=0.05, help="same hold-out as train_head.py")
    ap.add_argument("--max_videos", type=int, default=105)
    ap.add_argument("--out", default="logs/compare_head.json")
    a = ap.parse_args()

    cc, cfg = CacheConfig(), EditConfig()
    ds = ChapterAVDataset(a.anno, a.clip_lmdb, a.clap_lmdb, lambda t: embed_titles(t, a.clip_pt, None, a.title_cache),
                          fixed=True, part="val", val_frac=a.val_frac)
    scorer = HeadScorer(a.head)
    tagger = AudioEventTagger(prompt_embeddings(EVENT_PROMPTS, cc.clap_name, cc.cache_dir), list(EVENT_PROMPTS), cfg.event_temperature)
    ev_w = np.array([cfg.event_weights.get(c, 0.0) for c in tagger.classes])
    envs = [lmdb.open(p, readonly=True, lock=False, readahead=False) for p in (a.clip_lmdb, a.clap_lmdb)]
    txn = [e.begin() for e in envs]

    rows = []
    for it in ds.items[:a.max_videos]:
        vid = it["vid"]
        ab = txn[1].get(vid.encode())
        v, au = to_second_grid(loads_npz(txn[0].get(vid.encode())), loads_npz(ab) if ab is not None else None, cc.feature_fps, cc.audio_hop_sec)
        av = AVRepresentation(v, au, len(v), vid)
        q = ds.emb[it["first"] + len(it["wins"])]                      # the joined chapter titles = the video "title"
        events = tagger.probs(av)
        head = scorer.predict(av, q)
        T, dur = len(av), float(len(av))
        audio_score, motion = events @ ev_w, visual_change(av.vision)
        qrel = av.vision @ q

        curves = {"no_head": importance_curve(av, events, tagger.classes, cfg)[0],
                  "head": importance_curve(av, events, tagger.classes, cfg, head["saliency"] + head["foreground"])[0],
                  "head_only": rank_norm(smooth(head["saliency"] + head["foreground"], cfg.smooth_sec))}
        masks, row = {}, {"vid": vid, "T": T, "has_audio": av.has_audio}
        for name, imp in curves.items():
            clips = plan_highlight(imp, events, tagger.classes, dur, cfg)
            m = masks[name] = mask_of(clips, T)
            row[name] = {"sec": int(m.sum()), "chapter_cov": chapter_coverage(m, it["wins"]),
                         "query_lift": float(qrel[m].mean() - qrel.mean()), "audio_lift": float(audio_score[m].mean() - audio_score.mean()),
                         "motion_lift": float(motion[m].mean() - motion.mean()),
                         "edge_share": float((m[:max(T // 20, 1)].sum() + m[-max(T // 20, 1):].sum()) / max(m.sum(), 1))}
        row["jaccard_head_vs_no_head"] = float((masks["head"] & masks["no_head"]).sum() / max((masks["head"] | masks["no_head"]).sum(), 1))
        rho = np.corrcoef(rank_norm(curves["no_head"]), rank_norm(curves["head_only"]))[0, 1]
        row["curve_corr"] = float(rho) if np.isfinite(rho) else 0.0
        for name, imp in (("no_head", curves["no_head"]), ("head_f", rank_norm(smooth(head["foreground"], cfg.smooth_sec)))):
            segs = summarize(av, imp, events, tagger.classes, cfg)
            sm = np.zeros(T, bool)
            for s in segs:
                sm[int(s.start):int(s.end)] = True
            row[f"summary_{name}"] = {"n": len(segs), "chapter_cov": chapter_coverage(sm, it["wins"]),
                                      "query_lift": float(qrel[sm].mean() - qrel.mean()) if sm.any() else 0.0}
        rows.append(row)

    def mean(path):
        vals = []
        for r in rows:
            x = r
            for k in path:
                x = x[k]
            vals.append(x)
        return float(np.mean(vals))

    print(f"[compare] {len(rows)} held-out videos")
    print("highlight (budget = 10% of the video, 15-60 s):")
    print(f"  {'method':10s} {'chapter_cov':>11s} {'query_lift':>10s} {'audio_lift':>10s} {'motion_lift':>11s} {'edge_share':>10s}")
    for name in ("no_head", "head", "head_only"):
        print(f"  {name:10s} " + " ".join(f"{mean([name, k]):{w}.3f}" for k, w in (("chapter_cov", 11), ("query_lift", 10), ("audio_lift", 10), ("motion_lift", 11), ("edge_share", 10))))
    print(f"  selection overlap head vs no_head (Jaccard) = {np.mean([r['jaccard_head_vs_no_head'] for r in rows]):.3f}")
    print(f"  rank correlation of the two curves          = {np.mean([r['curve_corr'] for r in rows]):.3f}")
    print("summary (n_scenes segments):")
    for name in ("summary_no_head", "summary_head_f"):
        print(f"  {name:16s} chapter_cov={mean([name, 'chapter_cov']):.3f} query_lift={mean([name, 'query_lift']):.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"[compare] per-video numbers: {a.out}")


if __name__ == "__main__":
    main()
