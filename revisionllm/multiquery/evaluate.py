"""Report evaluation of the multi-query / cache system on Charades-STA test (R2 checkpoint).

  table    (1) Baseline / + Feature Cache / + Semantic Cache : mIoU, R1@0.3/0.5/0.7, avg latency per query
  scaling  (2) total time for 1/5/10/20 queries on one video  : Baseline / Feature Cache / Multi-Query (+batch)
  case     (3) one real Semantic-Cache hit (similar query -> stored segment)
  all      table + scaling + case

Configurations (all use the same R2 model, same windows, same grounding code):
  baseline       every query is an independent request: its features are extracted again, no cache
  +feature       features extracted once per video (SHA-256 cache), queries grounded one by one
  +semantic      +feature, and near-duplicate queries of the same video reuse the stored segment
  multi-query    (scaling only) +feature with all queries of the video batched into shared forward passes

  python revisionllm/multiquery/evaluate.py all --video_dir data/charades/Charades_v1_480 --max_videos 300
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.multiquery.cli import add_model_args, make_cfg
from revisionllm.multiquery.feature_cache import VideoFeatures, probe_duration
from revisionllm.multiquery.grounder import Grounder, model_args, norm_sentence
from revisionllm.multiquery.service import MultiQueryGrounder, TextEncoder, mmss, model_version_of


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def seg_iou(pred, gt):
    if pred is None or pred[0] is None:
        return 0.0
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    return inter / (max(pred[1], gt[1]) - min(pred[0], gt[0]) + 1e-9)


def metrics(ious):
    a = np.array(ious)
    return {"n": len(a), "mIoU": a.mean() * 100, **{f"R1@{t}": (a >= t).mean() * 100 for t in (0.3, 0.5, 0.7)}}


class Harness:
    def __init__(self, a):
        self.a = a
        self.cfg = make_cfg(a)
        self.cfg.cache_dir = a.eval_cache_dir
        g = Grounder(model_args(a.clip_pt, a.vicuna, a.stage2, a.stage1_proj, a.clip_adapter, self.cfg), self.cfg)
        self.svc = MultiQueryGrounder(self.cfg, a.clip_pt, g, model_version_of(a.stage2), query_batch=a.query_batch)
        anno = json.load(open(a.anno))
        self.by_vid = defaultdict(list)
        for qid, q in anno.items():
            q = dict(q, qid=qid)
            if os.path.exists(os.path.join(a.video_dir, q["movie"] + ".mp4")):
                self.by_vid[q["movie"]].append(q)
        vids = sorted(self.by_vid)
        random.Random(a.seed).shuffle(vids)
        self.vids = vids[:a.max_videos]
        print(f"[eval] {len(self.vids)} videos / {sum(len(self.by_vid[v]) for v in self.vids)} queries available")

    def path(self, vid):
        return os.path.join(self.a.video_dir, vid + ".mp4")

    def reset(self):
        """Empty caches (DB rows + feature files) so a run starts cold; models stay loaded."""
        self.svc.db.conn.execute("TRUNCATE video, query_cache RESTART IDENTITY CASCADE")
        shutil.rmtree(self.cfg.cache_dir, ignore_errors=True)
        self.svc.features.n_extractions = 0
        self.svc.n_grounding_calls = 0

    def warmup(self):
        v = self.vids[0]
        self.reset()
        self.svc.run(self.path(v), [self.by_vid[v][0]["sentence"]], semantic=False)
        self.reset()

    # ---- the three request styles ------------------------------------------------------------------------------
    def baseline_query(self, video_path, sentence):
        """Independent request: extract features again, then ground this single query."""
        v, a = self.svc.features._extract(video_path)
        self.svc.features.n_extractions += 1
        feats = VideoFeatures(0, "", v, a, probe_duration(video_path), False)
        sent = norm_sentence(sentence)
        tok, cls = self.svc.text(sent)
        pv = self.svc.grounder.prepare(feats)
        self.svc.n_grounding_calls += 1
        return self.svc.grounder.ground_many(pv, [(sent, tok, cls)], self.a.query_batch)[0]

    def run_video(self, mode, vid, sentences):
        """-> list of (start, end, source). Wall time is measured by the caller."""
        p = self.path(vid)
        if mode == "baseline":
            return [(*(self.baseline_query(p, s) or (None, None)), "grounding") for s in sentences]
        kw = dict(feature=dict(semantic=False, batched=False),
                  semantic=dict(semantic=True, batched=False), multiquery=dict(semantic=False, batched=True))[mode]
        return [(r.start_time, r.end_time, r.source) for r in self.svc.run(p, sentences, **kw)]


def cmd_table(h, a):
    h.warmup()
    qs = [(v, q) for v in h.vids for q in h.by_vid[v]]
    rows, per_query, hit_log = {}, {}, []
    for mode, label in (("baseline", "Baseline"), ("feature", "+ Feature Cache"), ("semantic", "+ Semantic Cache")):
        h.reset()
        ious, n_hit, t_total = {}, 0, 0.0
        if mode != "semantic":
            hit_log.clear()
        for v in h.vids:
            sents = [q["sentence"] for q in h.by_vid[v]]
            sync(); t = time.time()
            res = h.run_video(mode, v, sents)
            sync(); t_total += time.time() - t
            for q, (s, e, src) in zip(h.by_vid[v], res):
                ious[q["qid"]] = seg_iou((s, e), q["timestamps"])
                if src == "semantic_cache":
                    n_hit += 1
                    hit_log.append(q["qid"])
        m = metrics(list(ious.values()))
        m.update(avg_latency_s=t_total / len(qs), total_s=t_total, extractions=h.svc.features.n_extractions,
                 grounding_calls=h.svc.n_grounding_calls, semantic_hits=n_hit)
        rows[label], per_query[label] = m, ious
        print(label, m)
    # accuracy of the reused segments: IoU of the cache-hit queries, Semantic Cache vs. what the Baseline got for them
    hit_ids = list(hit_log)
    if hit_ids:
        rows["_hits"] = {"n": len(hit_ids), "mean_IoU_semantic": float(np.mean([per_query["+ Semantic Cache"][q] for q in hit_ids])),
                         "mean_IoU_baseline": float(np.mean([per_query["Baseline"][q] for q in hit_ids]))}
    rows["_note"] = (f"semantic hits: {rows['+ Semantic Cache']['semantic_hits']} of {len(qs)} queries"
                     + (f"; on those, mean IoU {rows['_hits']['mean_IoU_semantic']:.3f} (reused) vs "
                        f"{rows['_hits']['mean_IoU_baseline']:.3f} (Baseline grounding)" if hit_ids else ""))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "table.json"), "w"), indent=1)
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write("| 구성 | mIoU | R1@0.3 | R1@0.5 | R1@0.7 | 평균 지연(s/query) | feature 추출 | grounding 호출 | cache hit |\n|---|---|---|---|---|---|---|---|---|\n")
        for k, m in rows.items():
            if k.startswith("_"): continue
            f.write(f"| {k} | {m['mIoU']:.2f} | {m['R1@0.3']:.2f} | {m['R1@0.5']:.2f} | {m['R1@0.7']:.2f} | "
                    f"{m['avg_latency_s']:.2f} | {m['extractions']} | {m['grounding_calls']} | {m['semantic_hits']} |\n")
        f.write(f"\n{len(qs)} queries / {len(h.vids)} videos. {rows['_note']}\n")
    print(open(os.path.join(a.out, "table.md")).read())


def cmd_scaling(h, a):
    h.warmup()
    pool = [q["sentence"] for v in h.by_vid for q in h.by_vid[v]]
    rng = random.Random(a.seed)
    vids = h.vids[:a.scaling_videos]
    modes = [("baseline", "Baseline"), ("feature", "+ Feature Cache"), ("multiquery", "Multi-Query (+ batch)")]
    res = {lab: {n: [] for n in a.scaling_ns} for _, lab in modes}
    for v in vids:
        own = [q["sentence"] for q in h.by_vid[v]]
        others = [s for s in pool if s not in own]
        rng.shuffle(others)
        base = (own + others)[:max(a.scaling_ns)]
        for n in a.scaling_ns:
            for mode, lab in modes:
                for _ in range(a.repeats):
                    h.reset()
                    sync(); t = time.time()
                    h.run_video(mode, v, base[:n])
                    sync(); res[lab][n].append(time.time() - t)
        print("video", v, {lab: {n: round(float(np.mean(x)), 2) for n, x in d.items()} for lab, d in res.items()})
    os.makedirs(a.out, exist_ok=True)
    summary = {lab: {str(n): float(np.mean(x)) for n, x in d.items()} for lab, d in res.items()}
    json.dump({"mean_total_s": summary, "raw": {lab: {str(n): x for n, x in d.items()} for lab, d in res.items()},
               "videos": vids, "repeats": a.repeats}, open(os.path.join(a.out, "scaling.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for lab, d in summary.items():
        ax.plot(a.scaling_ns, [d[str(n)] for n in a.scaling_ns], marker="o", label=lab)
    ax.set_xlabel("number of queries"); ax.set_ylabel("total processing time (s)"); ax.set_xticks(a.scaling_ns)
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "scaling.png"), dpi=200)
    print(json.dumps(summary, indent=1))


def grab_frame(video, t):
    """RGB frame at time t (s) via ffmpeg -> numpy (H, W, 3)."""
    import subprocess
    import ffmpeg
    st = next(x for x in ffmpeg.probe(video)["streams"] if x["codec_type"] == "video")
    w, hh = int(st["width"]), int(st["height"])
    out = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(hh, w, 3)


def case_figure(path, video, duration, q1, q2, r1, r2):
    """Frames sampled inside the returned segment + a timeline of GT / returned segment for both queries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, e = r2.start_time, r2.end_time
    ts = np.linspace(s, max(s, e - 0.1), 4)
    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.3])
    for k, t in enumerate(ts):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(grab_frame(video, min(t, duration - 0.1))); ax.axis("off")
        ax.set_title(f"{t:.1f}s", fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    rows = [("Query 1 (grounding)", r1, q1), ("Query 2 (cache hit)", r2, q2)]
    for y, (name, r, q) in enumerate(rows):
        ax.barh(y + .17, q["timestamps"][1] - q["timestamps"][0], left=q["timestamps"][0], height=.3, color="#9aa5b1", label="GT" if y == 0 else None)
        ax.barh(y - .17, r.end_time - r.start_time, left=r.start_time, height=.3, color="#2563eb" if y == 0 else "#16a34a",
                label="returned" if y == 0 else None)
        ax.text(duration + .3, y, f"\"{q['sentence']}\"", va="center", fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels([n for n, _, _ in rows], fontsize=8); ax.set_xlim(0, duration)
    ax.set_xlabel("time (s)"); ax.legend(fontsize=7, loc="lower left"); ax.axvspan(s, e, color="#16a34a", alpha=.08)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def cmd_case(h, a):
    """Find a same-video pair (similar wording, overlapping GT) whose first result is good, run it, and show the hit."""
    thr = h.cfg.similarity_threshold
    cands = []
    for v in h.vids or list(h.by_vid):
        qs = h.by_vid[v]
        E = [TextEncoder.embedding(h.svc.text(norm_sentence(q["sentence"]))[1]) for q in qs]
        for i in range(len(qs)):
            for j in range(len(qs)):
                if i != j and float(E[i] @ E[j]) > thr and seg_iou(qs[i]["timestamps"], qs[j]["timestamps"]) >= .5:
                    cands.append((float(E[i] @ E[j]), v, qs[i], qs[j]))
    cands.sort(key=lambda c: -c[0])
    best = None
    for sim, v, q1, q2 in cands[:a.case_candidates]:   # keep a pair whose segment really matches the scene
        h.reset()
        r1 = h.svc.run(h.path(v), [q1["sentence"]])[0]
        r2 = h.svc.run(h.path(v), [q2["sentence"]])[0]
        score = min(seg_iou((r1.start_time, r1.end_time), q1["timestamps"]), seg_iou((r2.start_time, r2.end_time), q2["timestamps"]))
        if r2.source == "semantic_cache" and (best is None or score > best[0]):
            best = (score, v, q1, q2, r1, r2)
        if best and best[0] >= .6:
            break
    if best is None:
        print("no pair above the threshold among the available videos"); return
    score, v, q1, q2, r1, r2 = best
    dur = probe_duration(h.path(v))
    os.makedirs(a.out, exist_ok=True)
    case_figure(os.path.join(a.out, "case.png"), h.path(v), dur, q1, q2, r1, r2)
    with open(os.path.join(a.out, "case.md"), "w") as f:
        f.write(f"video `{v}` (duration {dur:.1f}s), threshold {thr}\n\n![case](case.png)\n\n")
        f.write("| 입력 쿼리 | 처리 | 유사도 | 반환 구간 | GT |\n|---|---|---|---|---|\n")
        for q, x in zip((q1, q2), (r1, r2)):
            sim = "-" if x.source == "grounding" else f"{x.similarity:.3f} (→ \"{x.matched_query}\")"
            f.write(f"| {q['sentence']} | {'Grounding 실행' if x.source == 'grounding' else 'Cache Hit'} | {sim} | "
                    f"{mmss(x.start_time)} ~ {mmss(x.end_time)} | {mmss(q['timestamps'][0])} ~ {mmss(q['timestamps'][1])} |\n")
    print(open(os.path.join(a.out, "case.md")).read())


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("what", choices=["table", "scaling", "case", "all"])
    ap.add_argument("--video_dir", default="data/charades/Charades_v1_480")
    ap.add_argument("--anno", default="data/charades/charades_test_anno.json")
    ap.add_argument("--max_videos", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query_batch", type=int, default=8)
    ap.add_argument("--scaling_ns", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--scaling_videos", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--case_candidates", type=int, default=20, help="pairs tried when looking for a good cache-hit example")
    ap.add_argument("--out", default="results/multiquery_eval")
    ap.add_argument("--eval_cache_dir", default="data/multiquery_eval_cache", help="wiped between runs")
    a = ap.parse_args()
    h = Harness(a)
    for w, fn in (("table", cmd_table), ("scaling", cmd_scaling), ("case", cmd_case)):
        if a.what in (w, "all"):
            fn(h, a)


if __name__ == "__main__":
    main()
