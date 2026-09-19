"""
Freeze the final Train-sub / Val-sub / Test-sub lists.

* keep only ids present in BOTH LMDBs (CLIP + CLAP)
* Val/Test: take the first n_final surviving candidates (candidate order is the
  seeded shuffle, so this is deterministic)
* Train: report achieved hours per stratum
* assert Train ∩ Test = Train ∩ Val = ∅ on video ids
* write <out_dir>/{train_sub,val_sub,test_sub}.json  (plain id lists)
* also write the ReVisionLLM annotation JSONs restricted to those ids
  (chapters_to_activitynet.py output filtered by 'id') so the training and eval
  scripts can be pointed at the subsets directly.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(ROOT)
from revisionllm.data.audio_pipeline.lmdb_utils import lmdb_keys
from revisionllm.data.audio_pipeline.subsets import stratum_of


def filter_ids(cand, have, n_final=None):
    ids = [v for v in cand["ids"] if v in have]
    if n_final is not None:
        ids = ids[:n_final]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand_dir", required=True, help="dir with *_candidates.json from subsets.py")
    ap.add_argument("--clip_lmdb", required=True)
    ap.add_argument("--clap_lmdb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_anno", default=None, help="chapters train annotations (activitynet format) to filter")
    ap.add_argument("--test_anno", default=None, help="chapters test annotations (activitynet format) to filter")
    ap.add_argument("--val_anno", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    have = lmdb_keys(a.clip_lmdb) & lmdb_keys(a.clap_lmdb)
    print(f"{len(have)} videos have both CLIP and CLAP features")

    cand = {k: json.load(open(os.path.join(a.cand_dir, f"{k}_candidates.json"))) for k in ("train_sub", "val_sub", "test_sub")}
    final = {
        "train_sub": filter_ids(cand["train_sub"], have),
        "val_sub": filter_ids(cand["val_sub"], have, cand["val_sub"]["n_final"]),
        "test_sub": filter_ids(cand["test_sub"], have, cand["test_sub"]["n_final"]),
    }
    for k in ("val_sub", "test_sub"):
        want = cand[k]["n_final"]
        if len(final[k]) < want:
            print(f"WARNING: {k} has {len(final[k])} < {want} videos with audio; download more candidates")
    tr, va, te = map(set, (final["train_sub"], final["val_sub"], final["test_sub"]))
    assert not (tr & te), f"train/test overlap: {list(tr & te)[:5]}"
    assert not (tr & va), f"train/val overlap: {list(tr & va)[:5]}"

    durs = cand["train_sub"]["durations"]
    hours = {}
    for v in final["train_sub"]:
        s = stratum_of(float(durs[v]))
        hours[s] = hours.get(s, 0.0) + float(durs[v]) / 3600
    print("train-sub achieved hours per stratum:", {k: round(v, 1) for k, v in hours.items()},
          "total:", round(sum(hours.values()), 1))

    for k, ids in final.items():
        with open(os.path.join(a.out_dir, f"{k}.json"), "w") as f:
            json.dump(ids, f)
        print(f"wrote {k}.json ({len(ids)} videos)")

    for name, anno, key in (("train_sub", a.train_anno, "train_sub"), ("val_sub", a.val_anno, "val_sub"),
                            ("test_sub", a.test_anno, "test_sub")):
        if anno is None:
            continue
        keep = set(final[key])
        data = json.load(open(anno))
        if isinstance(data, dict):  # chapters_test.json style {qid: {...,'movie': vid}}
            out = {q: d for q, d in data.items() if d.get("movie") in keep}
        else:  # activitynet-style list with 'id'
            out = [d for d in data if d.get("id") in keep]
        path = os.path.join(a.out_dir, f"{name}_anno.json")
        with open(path, "w") as f:
            json.dump(out, f)
        print(f"wrote {path} ({len(out)} queries)")


if __name__ == "__main__":
    main()
