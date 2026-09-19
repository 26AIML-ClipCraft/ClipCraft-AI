"""
Sample Train-sub / Val-sub / Test-sub from the OFFICIAL VidChapters-7M splits.

Why official splits: the released ReVisionLLM checkpoint saw the whole official
train set, so carving a "val" out of train would inflate the R0 / visual-only
baselines. See the experiment design doc (데이터 section).

Inputs  : chapters_vmr_{train,val,test}.jsonl  (VidChapters VMR format, one
          record per video with 'vid', 'duration', 'query'[], 'relevant_windows'[])
Outputs : <out_dir>/{train_sub,val_sub,test_sub}_candidates.json
          -> lists of video ids (candidate = before audio availability check).
          finalize_subsets.py turns candidates into the frozen final lists.

Train-sub is stratified by duration (design: <5 min 20 %, 5-20 min 40 %,
20-60 min 30 %, >60 min 10 %) and sized by TOTAL HOURS, not video count; extra
candidates (``--reserve_factor``) are kept per stratum so failed downloads can
be replaced from the same duration bucket.
"""
import argparse
import json
import os
import random
from collections import defaultdict

STRATA = [  # (name, min_sec, max_sec, share)
    ("lt5m", 0, 5 * 60, 0.20),
    ("5to20m", 5 * 60, 20 * 60, 0.40),
    ("20to60m", 20 * 60, 60 * 60, 0.30),
    ("gt60m", 60 * 60, float("inf"), 0.10),
]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stratum_of(duration):
    for name, lo, hi, _ in STRATA:
        if lo <= duration < hi:
            return name
    return STRATA[-1][0]


def sample_train(records, target_hours, reserve_factor, rng):
    by = defaultdict(list)
    for r in records:
        by[stratum_of(float(r["duration"]))].append(r)
    picked, reserve = [], {}
    for name, _, _, share in STRATA:
        pool = by[name]
        rng.shuffle(pool)
        budget = target_hours * 3600.0 * share
        acc, chosen = 0.0, []
        for r in pool:
            if acc >= budget:
                break
            chosen.append(r)
            acc += float(r["duration"])
        picked.extend(chosen)
        n_res = int(len(chosen) * reserve_factor)
        reserve[name] = [r["vid"] for r in pool[len(chosen):len(chosen) + n_res]]
        print(f"[train-sub] {name:8s} videos={len(chosen):6d} hours={acc / 3600:8.1f} reserve={len(reserve[name])}")
    return picked, reserve


def sample_uniform(records, n, rng):
    """Keep the split's own duration distribution (design: Test-sub adds no filter)."""
    recs = list(records)
    rng.shuffle(recs)
    return recs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_hours", type=float, default=10000.0)
    ap.add_argument("--val_videos", type=int, default=300)
    ap.add_argument("--test_videos", type=int, default=1000)
    ap.add_argument("--reserve_factor", type=float, default=0.5, help="extra candidates per stratum for failed downloads")
    ap.add_argument("--test_reserve", type=int, default=500, help="extra test/val candidates for failed downloads")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    train = read_jsonl(a.train_jsonl)
    val = read_jsonl(a.val_jsonl)
    test = read_jsonl(a.test_jsonl)
    print(f"official sizes: train={len(train)} val={len(val)} test={len(test)}")

    train_pick, train_reserve = sample_train(train, a.train_hours, a.reserve_factor, rng)
    val_pick = sample_uniform(val, a.val_videos + a.test_reserve, rng)
    test_pick = sample_uniform(test, a.test_videos + a.test_reserve, rng)

    train_ids = [r["vid"] for r in train_pick]
    test_ids = [r["vid"] for r in test_pick]
    val_ids = [r["vid"] for r in val_pick]
    assert not (set(train_ids) & set(test_ids)), "train/test overlap in official splits?!"
    assert not (set(train_ids) & set(val_ids)), "train/val overlap in official splits?!"

    meta = {r["vid"]: {"duration": r["duration"], "n_queries": len(r.get("query", []))} for r in train + val + test}
    out = {
        "train_sub_candidates.json": {"seed": a.seed, "target_hours": a.train_hours, "ids": train_ids,
                                      "reserve_by_stratum": train_reserve},
        "val_sub_candidates.json": {"seed": a.seed, "n_final": a.val_videos, "ids": val_ids},
        "test_sub_candidates.json": {"seed": a.seed, "n_final": a.test_videos, "ids": test_ids},
    }
    for name, payload in out.items():
        payload["durations"] = {v: meta[v]["duration"] for v in payload["ids"]}
        with open(os.path.join(a.out_dir, name), "w") as f:
            json.dump(payload, f)
        hours = sum(payload["durations"].values()) / 3600
        print(f"wrote {name}: {len(payload['ids'])} videos, {hours:.1f} h")


if __name__ == "__main__":
    main()
