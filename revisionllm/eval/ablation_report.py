"""
Ablation report for the audio-visual ReVisionLLM experiment.

Reads per-query IoUs from prediction logs (predictions_stream_*.txt /
predictions_streaming_*.txt written by eval_nlq_negative.py, optionally after
metric_retrieval_forward_chapters.py has applied the hierarchical window
selection) for one or more runs and prints:

  * mIoU, R1@{0.3,0.5,0.7} per run (same formulas as metric_retrieval_forward_chapters.py)
  * paired bootstrap 95 % CI of (run_B - run_A) on R1@0.5 and mIoU over the
    common query set (design: 1,000 resamples, query level)
  * mean ± std over seeds when a run is given as several directories
  * H2 breakdown by query type from label_query_types.py (--labels)

Usage:
  python revisionllm/eval/ablation_report.py \
      --run R1=ckpt/stageC_R1_seed0,ckpt/stageC_R1_seed1 \
      --run R2=ckpt/stageC_R2_seed0,ckpt/stageC_R2_seed1 \
      --compare R1 R2 --labels data/subsets/test_query_types.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

THRESHOLDS = (0.3, 0.5, 0.7)


def load_query_ious(run_dir):
    """query_id -> best IoU (top-1 after score sort, like grounding_metrics_stream)."""
    logs = []
    merged = os.path.join(run_dir, "predictions_merged.txt")
    patterns = ["predictions_merged.txt"] if os.path.exists(merged) else \
        ["predictions_stream_*.txt", "predictions_streaming_*.txt", "predictions.txt"]
    for pat in patterns:
        for p in glob.glob(os.path.join(run_dir, pat)):
            with open(p) as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        pass
    out = {}
    for log in logs:
        if log.get("task") != "grounding":
            continue
        info = log.get("info", {})
        iou = info.get("iou", -1)
        if isinstance(iou, list):
            if len(iou) == 0:
                top = 0.0
            elif "scores" in info and len(info["scores"]) == len(iou):
                top = iou[int(np.argmax(info["scores"]))]
            else:
                top = iou[0]
        else:
            top = iou
        if top == -1:
            continue
        out[log["query_id"]] = float(top)
    return out


def metrics(ious):
    ious = np.asarray(ious, dtype=np.float64)
    m = {"n": int(len(ious)), "mIoU": float(ious.mean() * 100) if len(ious) else float("nan")}
    for t in THRESHOLDS:
        m[f"R1@{t}"] = float((ious >= t).mean() * 100) if len(ious) else float("nan")
    return m


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    """CI of mean(b - a) statistics over queries (paired: same query set)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    d_miou, d_r05 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_miou.append((b[idx].mean() - a[idx].mean()) * 100)
        d_r05.append(((b[idx] >= 0.5).mean() - (a[idx] >= 0.5).mean()) * 100)
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"delta_mIoU": float(np.mean(d_miou)), "ci_mIoU": ci(d_miou),
            "delta_R1@0.5": float(np.mean(d_r05)), "ci_R1@0.5": ci(d_r05), "n": int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=dir[,dir2,...] (several dirs = seeds)")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "PROPOSED"), default=None)
    ap.add_argument("--labels", default=None, help="query type labels from label_query_types.py")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    a = ap.parse_args()

    runs = {}
    for spec in a.run:
        name, dirs = spec.split("=", 1)
        runs[name] = [load_query_ious(d) for d in dirs.split(",")]

    report = {"runs": {}}
    print("=== per run (mean ± std over seeds) ===")
    for name, seeds in runs.items():
        per_seed = [metrics(list(s.values())) for s in seeds]
        agg = {}
        for k in per_seed[0]:
            vals = [m[k] for m in per_seed]
            agg[k] = (float(np.mean(vals)), float(np.std(vals)))
        report["runs"][name] = {"seeds": per_seed, "agg": agg}
        line = " ".join(f"{k}={v[0]:.2f}±{v[1]:.2f}" for k, v in agg.items() if k != "n")
        print(f"{name:10s} n={agg['n'][0]:.0f} seeds={len(seeds)} {line}")

    groups = None
    if a.labels:
        labels = json.load(open(a.labels))
        groups = {"audio_cue": {q for q, v in labels.items() if v.get("agree") and v.get("kw")},
                  "no_audio_cue": {q for q, v in labels.items() if v.get("agree") and not v.get("kw")}}

    if a.compare:
        base, prop = a.compare
        A, B = runs[base][0], runs[prop][0]  # seed 0 for the paired test
        common = sorted(set(A) & set(B))
        print(f"\n=== paired bootstrap {prop} - {base} (seed 0, n={len(common)}) ===")
        bs = paired_bootstrap([A[q] for q in common], [B[q] for q in common], a.n_boot)
        report["paired"] = bs
        for k in ("R1@0.5", "mIoU"):
            lo, hi = bs[f"ci_{k}"]
            verdict = "H1 holds" if lo > 0 else ("worse" if hi < 0 else "inconclusive (CI contains 0)")
            print(f"{k}: delta={bs[f'delta_{k}']:+.2f}  95% CI=[{lo:+.2f}, {hi:+.2f}]  -> {verdict}")
        if groups:
            print(f"\n=== H2 breakdown by query type ({prop} vs {base}) ===")
            report["breakdown"] = {}
            for g, qs in groups.items():
                qs = [q for q in common if q in qs]
                if not qs:
                    continue
                ma, mb = metrics([A[q] for q in qs]), metrics([B[q] for q in qs])
                bsg = paired_bootstrap([A[q] for q in qs], [B[q] for q in qs], a.n_boot)
                report["breakdown"][g] = {"base": ma, "proposed": mb, "paired": bsg}
                print(f"{g:14s} n={len(qs):5d}  R1@0.5 {ma['R1@0.5']:.2f} -> {mb['R1@0.5']:.2f} "
                      f"(delta {bsg['delta_R1@0.5']:+.2f}, CI [{bsg['ci_R1@0.5'][0]:+.2f}, {bsg['ci_R1@0.5'][1]:+.2f}])"
                      f"  mIoU {ma['mIoU']:.2f} -> {mb['mIoU']:.2f}")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
