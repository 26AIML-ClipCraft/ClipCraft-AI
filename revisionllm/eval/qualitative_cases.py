"""
Qualitative cases with an audio-ablation control (no training involved).

Inputs are the dense prediction dirs of the evaluated rows:
  R1 = visual-only,  R2 = audio-visual,  R4 = R2 with the audio zeroed at inference (DROP_AUDIO=1)
and optionally R1b = a second visual-only seed (noise baseline: how many queries flip with no audio at all).

A query is "correct" when its top-1 prediction (windows ranked by info.scores) has IoU > --iou.
Categories:
  audio_gain          R2 correct, R1 wrong, R4 wrong     -> attributable to the audio branch
  gain_without_audio  R2 correct, R1 wrong, R4 correct   -> better than R1, but not because of audio
  audio_loss          R2 wrong,  R4 correct              -> audio made it worse
  r2_loss_vs_r1       R2 wrong,  R1 correct
Counts, the R1-vs-R1b flip counts (noise floor) and a seeded random sample of every category are written, with the
predicted intervals in seconds (reconstructed exactly as eval_nlq_negative.iou does) and rough CLAP sound tags of the GT segment.

    python revisionllm/eval/qualitative_cases.py --anno .../test_sub_anno.json --r1 R1/dense --r2 R2/dense --r4 R4/dense \
        [--r1b R1_seed1/dense] [--clap_lmdb .../clap_1s_lmdb] --out cases.json --md cases.md
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)
from revisionllm.eval.dense_metrics import load_logs, ranked_ious

NUM_FRAMES, WINDOW = 250, 500
TAGS = ["speech", "a person talking", "music", "singing", "guitar", "piano", "drums", "applause", "laughter",
        "crowd cheering", "silence", "typing on a keyboard", "engine noise", "water", "wind", "birds",
        "dog barking", "explosion", "traffic", "sirens"]


def candidates(log, duration):
    """[(start_s, end_s, score, iou)] for the windows that predicted an interval (same rules as eval_nlq_negative.iou)."""
    nf_video = NUM_FRAMES if duration <= WINDOW else int(duration * NUM_FRAMES / WINDOW)
    info = log.get("info", {})
    ious, scores = info.get("iou", []), info.get("scores", [])
    out = []
    for i, a in enumerate(log.get("answer", [])):
        m = re.search(r"(\d+) (to|and) (\d+)", a)
        if not m:
            continue
        f, t = float(m.group(1)), float(m.group(3))
        if f == NUM_FRAMES - 1 and t == NUM_FRAMES - 1:
            continue
        if f == t:
            f, t = max(0, f - 1), min(nf_video, t + 1)
        gf, gt = int(i * NUM_FRAMES // 2 + f), int(i * NUM_FRAMES // 2 + t)
        out.append([gf / nf_video * duration, gt / nf_video * duration])
    k = len(out)
    if len(ious) != k:
        return None
    for j in range(k):
        out[j] += [scores[j] if len(scores) == k else 0.0, ious[j]]
    return out


def top1(log, duration):
    c = candidates(log, duration)
    if not c:
        return None
    return max(c, key=lambda x: x[2])     # (start, end, score, iou)


def clap_tags(clap_lmdb, vid, s, e, tag_emb, topn=3):
    import lmdb
    from revisionllm.data.audio_pipeline.lmdb_utils import loads_npz
    env = lmdb.open(clap_lmdb, readonly=True, lock=False)
    with env.begin() as t:
        buf = t.get(vid.encode())
    if buf is None:
        return []
    A = loads_npz(buf).astype(np.float32)
    seg = A[int(s):max(int(e), int(s) + 1)]
    if len(seg) == 0:
        return []
    m = seg.mean(0); m /= max(np.linalg.norm(m), 1e-8)
    sims = tag_emb @ m
    return [TAGS[i] for i in np.argsort(-sims)[:topn]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--r1", required=True); ap.add_argument("--r2", required=True); ap.add_argument("--r4", required=True)
    ap.add_argument("--r1b", default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n_show", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clap_lmdb", default=None)
    ap.add_argument("--clap_model", default="laion/clap-htsat-fused")
    ap.add_argument("--out", default=None); ap.add_argument("--md", default=None)
    a = ap.parse_args()

    anno = json.load(open(a.anno))
    logs = {k: load_logs(getattr(a, k)) for k in ("r1", "r2", "r4")}
    if a.r1b:
        logs["r1b"] = load_logs(a.r1b)
    rows = {}
    common = set(anno) & set.intersection(*[set(l) for l in logs.values()])
    for q in common:
        dur = float(anno[q]["movie_duration"])
        t = {k: top1(l[q], dur) for k, l in logs.items()}
        # a query with no candidate counts as wrong (IoU 0)
        rows[q] = {k: (v if v else [None, None, 0.0, 0.0]) for k, v in t.items()}
    ok = lambda q, k: rows[q][k][3] > a.iou
    cats = {
        "audio_gain": [q for q in rows if ok(q, "r2") and not ok(q, "r1") and not ok(q, "r4")],
        "gain_without_audio": [q for q in rows if ok(q, "r2") and not ok(q, "r1") and ok(q, "r4")],
        "audio_loss": [q for q in rows if not ok(q, "r2") and ok(q, "r4")],
        "r2_loss_vs_r1": [q for q in rows if not ok(q, "r2") and ok(q, "r1")],
    }
    summary = {"iou_threshold": a.iou, "n_queries": len(rows),
               "correct": {k: int(sum(ok(q, k) for q in rows)) for k in logs},
               "counts": {c: len(v) for c, v in cats.items()}}
    summary["r2_gain_total"] = summary["counts"]["audio_gain"] + summary["counts"]["gain_without_audio"]
    if summary["r2_gain_total"]:
        summary["share_of_gains_attributable_to_audio"] = summary["counts"]["audio_gain"] / summary["r2_gain_total"]
    if a.r1b:
        summary["noise_floor_R1_vs_R1b"] = {"r1_right_r1b_wrong": int(sum(ok(q, "r1") and not ok(q, "r1b") for q in rows)),
                                           "r1_wrong_r1b_right": int(sum(not ok(q, "r1") and ok(q, "r1b") for q in rows))}

    tag_emb = None
    if a.clap_lmdb:
        import torch
        from transformers import ClapModel, ClapProcessor
        m = ClapModel.from_pretrained(a.clap_model).eval(); p = ClapProcessor.from_pretrained(a.clap_model)
        with torch.no_grad():
            tag_emb = torch.nn.functional.normalize(m.get_text_features(**p(text=[f"This is a sound of {t}." for t in TAGS],
                                                    return_tensors="pt", padding=True)), dim=-1).numpy()
    rng = random.Random(a.seed)
    shown = {}
    for c, qs in cats.items():
        pick = rng.sample(sorted(qs), min(a.n_show, len(qs)))
        shown[c] = []
        for q in pick:
            d = anno[q]; s, e = d["timestamps"]
            pred = {k: {"start": rows[q][k][0], "end": rows[q][k][1], "iou": rows[q][k][3]} for k in ("r1", "r2", "r4")}
            shown[c].append({"query_id": q, "video": d["movie"], "query": d["sentence"], "gt": [s, e], "duration": d["movie_duration"],
                             "pred": pred, "gt_sound_tags": clap_tags(a.clap_lmdb, d["movie"], s, e, tag_emb) if tag_emb is not None else None})
    res = {"summary": summary, "cases": shown}
    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if a.md:
        L = [f"# Qualitative cases (correct = IoU > {a.iou})", "", "```", json.dumps(summary, indent=1), "```", ""]
        for c, items in shown.items():
            L += [f"## {c} ({summary['counts'][c]} queries, {len(items)} shown)", ""]
            for it in items:
                f = lambda k: ("-" if it["pred"][k]["start"] is None else f"{it['pred'][k]['start']:.0f}-{it['pred'][k]['end']:.0f}s") + f" (IoU {it['pred'][k]['iou']:.2f})"
                L.append(f"- **{it['query']}** [{it['video']}, {it['duration']:.0f}s] GT {it['gt'][0]:.0f}-{it['gt'][1]:.0f}s | R1 {f('r1')} | R2 {f('r2')} | R4 {f('r4')}"
                         + (f" | sound: {', '.join(it['gt_sound_tags'])}" if it["gt_sound_tags"] else ""))
            L.append("")
        open(a.md, "w").write("\n".join(L))
        print("wrote", a.md)


if __name__ == "__main__":
    main()
