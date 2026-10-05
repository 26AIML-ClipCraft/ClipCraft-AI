"""
Compare rows on query subsets (audio-cue keywords vs the rest) with paired bootstrap CIs, from the result_dense.json files
written by eval_row.sh (per-query top-1 IoU).  The keyword list is fixed BEFORE looking at any result.

    python revisionllm/eval/subset_compare.py --anno data/charades/charades_test_anno.json \
        --row R1=<eval>/R1_seed0/result_dense.json --row R2=<eval>/R2_seed0/result_dense.json --row R4=<eval>/R4/result_dense.json \
        --compare R1 R2 --compare R4 R2
"""
import argparse
import json
import re

import numpy as np

# actions that make a characteristic sound (Charades-STA queries)
AUDIO_KW = ["sneez", "cough", "laugh", "giggl", "smil", "talk", "speak", "sing", "whistl", "shout", "yell", "scream", "clap", "cheer",
            "call", "phone", "door", "knock", "slam", "cabinet", "cupboard", "drawer", "fridge", "refrigerator", "throw", "toss",
            "drop", "pour", "drink", "eat", "chew", "munch", "vacuum", "sweep", "wash", "water", "faucet", "flush", "brush",
            "turn on", "turn off", "switch", "close", "shut", "open", "tap", "type", "typing", "crinkl", "wrapper", "bag", "paper"]
# "smil" is a silent action: kept out on purpose (not a sound) -> removed below
AUDIO_KW = [k for k in AUDIO_KW if k != "smil"]
KW_RE = re.compile(r"(" + "|".join(re.escape(k) for k in AUDIO_KW) + ")", re.IGNORECASE)
THR = (0.3, 0.5, 0.7)


GE = False   # --ge: count IoU >= threshold (ablation_report.py / report.sh); default: IoU > threshold (dense_metrics.py)


def hit(t, x):
    return (t >= x) if GE else (t > x)


def metrics(top1):
    t = np.asarray(top1)
    return {"n": len(t), "mIoU": t.mean() * 100, **{f"R1@{x}": hit(t, x).mean() * 100 for x in THR}}


def boot(a, b, n_boot, seed=0):
    """paired bootstrap of (b - a) on R1@0.5 and mIoU"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    out = {}
    for name, f in (("R1@0.5", lambda x: hit(x, 0.5).mean() * 100), ("mIoU", lambda x: x.mean() * 100)):
        d = np.array([f(b[i]) - f(a[i]) for i in idx])
        out[name] = (f(b) - f(a), np.percentile(d, 2.5), np.percentile(d, 97.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", required=True)
    ap.add_argument("--row", action="append", required=True, help="NAME=path/to/result_dense.json")
    ap.add_argument("--compare", action="append", nargs=2, metavar=("BASE", "NEW"), default=[])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--ge", action="store_true", help="R1@t counts IoU >= t (same as report.sh); default is IoU > t")
    ap.add_argument("--keywords", default=None, help="comma list overriding the built-in audio-cue keywords (regex fragments)")
    a = ap.parse_args()
    global KW_RE, GE
    GE = a.ge
    if a.keywords:
        KW_RE = re.compile("(" + "|".join(k.strip() for k in a.keywords.split(",")) + ")", re.IGNORECASE)
    anno = json.load(open(a.anno))
    rows = {}
    for r in a.row:
        name, path = r.split("=", 1)
        rows[name] = json.load(open(path))["per_query_top1_iou"]
    qids = [q for q in anno if all(q in v for v in rows.values())]
    groups = {"all": qids,
              "audio_cue": [q for q in qids if KW_RE.search(anno[q]["sentence"])],
              "other": [q for q in qids if not KW_RE.search(anno[q]["sentence"])]}
    print(f"queries: all={len(groups['all'])}  audio_cue={len(groups['audio_cue'])}  other={len(groups['other'])}")
    for g, qs in groups.items():
        print(f"\n[{g}] n={len(qs)}")
        for name, v in rows.items():
            m = metrics([v[q] for q in qs])
            print("  %-4s mIoU %5.2f  R1@0.3 %5.2f  R1@0.5 %5.2f  R1@0.7 %5.2f" % (name, m["mIoU"], m["R1@0.3"], m["R1@0.5"], m["R1@0.7"]))
        for base, new in a.compare:
            r = boot([rows[base][q] for q in qs], [rows[new][q] for q in qs], a.n_boot)
            s = "   ".join("%s %+.2f [%+.2f, %+.2f]%s" % (k, d, lo, hi, " *" if lo > 0 or hi < 0 else "") for k, (d, lo, hi) in r.items())
            print(f"  {new} - {base}:  {s}")
    print("\n(* = 95% paired-bootstrap CI excludes 0)")
    ex = groups["audio_cue"][:0]


if __name__ == "__main__":
    main()
