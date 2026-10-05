"""
Stage-1 (dense sliding-window) metrics over ALL evaluated queries.

The released two-stage evaluation (eval_nlq_retrieval_e2e2.py) only works for videos with >= ~100 windows
(~167 min) and silently drops every other query, so VidChapters numbers end up on a tiny, long-video-only subset.
This script scores the dense predictions that eval_nlq_negative.py writes instead:

  * per query, candidates are the windows that predicted an interval; they are ranked by ``info.scores``
  * R{r}@t = share of queries whose top-r candidates contain one with IoU > t   (same rule as
    metric_retrieval_forward_chapters.grounding_metrics_stream, strict ">")
  * mIoU = mean top-1 IoU; a query with no candidate counts as 0 and STAYS in the denominator
  * queries that were expected (from --anno / --video_ids) but have no log record (crashed) also count as 0
  * extra breakdown by video duration

Usage:
  python revisionllm/eval/dense_metrics.py --grounding_path <dir with predictions_streaming_*.txt> \
      --anno data/chapters_audio/subsets/test_sub_anno.json [--video_ids subsets/test_sub.json] [--out result_dense.json]
"""
import argparse
import glob
import json
import os

import numpy as np

THRS = (0.3, 0.5, 0.7, 0.9)
RANKS = (1, 5)
BUCKETS = (("<=500s", 0, 500), ("500s-25min", 500, 1500), (">25min", 1500, float("inf")))


def load_logs(path):
    logs = {}
    for p in sorted(glob.glob(os.path.join(path, "predictions_streaming_*.txt"))):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("task") == "grounding":
                    logs[str(r["query_id"])] = r
    return logs


def ranked_ious(log):
    info = log.get("info", {})
    iou, sc = info.get("iou", []), info.get("scores", [])
    if not isinstance(iou, list):
        return [float(iou)] if iou != -1 else []
    if len(sc) == len(iou) and len(iou) > 0:
        order = sorted(range(len(iou)), key=lambda k: sc[k], reverse=True)
        return [float(iou[k]) for k in order]
    return [float(x) for x in iou]


def summarize(per_query):
    """per_query: list of ranked-IoU lists (empty list = no candidate)."""
    n = len(per_query)
    if n == 0:
        return {"n": 0}
    top1 = np.array([q[0] if q else 0.0 for q in per_query])
    out = {"n": n, "mIoU": float(top1.mean() * 100)}
    for t in THRS:
        for r in RANKS:
            out[f"R{r}@{t}"] = float(100 * np.mean([any(x > t for x in q[:r]) for q in per_query]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounding_path", required=True)
    ap.add_argument("--anno", required=True, help="test_sub_anno.json ({qid: {movie, movie_duration, ...}})")
    ap.add_argument("--video_ids", default=None, help="optional JSON list restricting the evaluated videos")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    anno = json.load(open(a.anno))
    keep = set(json.load(open(a.video_ids))) if a.video_ids else None
    expected = {str(q): d for q, d in anno.items() if keep is None or d["movie"] in keep}
    logs = load_logs(a.grounding_path)

    per_q, rows, missing = {}, [], 0
    for q, d in expected.items():
        if q in logs:
            per_q[q] = ranked_ious(logs[q])
        else:
            per_q[q] = []
            missing += 1
        rows.append((q, float(d.get("movie_duration", 0))))
    extra = [q for q in logs if q not in expected]

    res = {"overall": summarize(list(per_q.values())),
           "coverage": {"expected_queries": len(expected), "with_log": len(expected) - missing,
                        "no_log_counted_as_0": missing, "no_candidate": sum(1 for q in per_q.values() if not q),
                        "logs_not_in_anno": len(extra)},
           "by_duration": {}}
    for name, lo, hi in BUCKETS:
        res["by_duration"][name] = summarize([per_q[q] for q, dur in rows if lo < dur <= hi])
    res["per_query_top1_iou"] = {q: (v[0] if v else 0.0) for q, v in per_q.items()}

    print(json.dumps({k: v for k, v in res.items() if k != "per_query_top1_iou"}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
