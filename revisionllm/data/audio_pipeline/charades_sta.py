"""
Charades-STA preparation for the ReVisionLLM audio experiment (all local files, nothing is downloaded from YouTube).

  anno        Charades_sta_{train,test}.txt  ->  train list (ReVisionLLM/activitynet-style, source='charades')
                                                 + test dict ({qid: {movie, sentence, timestamps, movie_duration, ...}})
                                                 + id lists; only videos whose CLIP + CLAP features exist are kept
  replay_ids  stored VidChapters Train videos that can be used as replay data (long videos only: duration >= 2*window,
              because the loader's short-video branch for VidChapters assumes features this pipeline did not produce)
  text        CLIP ViT-L/14 text features of the Charades queries, appended to the Train / Test query-feature LMDBs
              (keys '<vid>_<j>' for train, 'c<idx>' for test: no clash with the VidChapters keys)
  mix         Charades train + a random VidChapters replay sample (default 30 % of the mix) -> one training annotation file

    python charades_sta.py anno --sta_dir data/charades --clip_lmdb ... --clap_lmdb ... --out_dir data/charades
"""
import argparse
import io
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(ROOT)
from revisionllm.data.audio_pipeline.lmdb_utils import lmdb_keys

PROMPT = "<video>\nDuring which frames can we see {}?"


def read_sta(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or "##" not in line:
            continue
        head, sent = line.split("##", 1)
        vid, s, e = head.split()
        rows.append((vid, float(s), float(e), sent.strip()))
    return rows


def norm_sentence(sent):
    sent = sent.strip().lower()
    return sent[:-1] if sent.endswith(".") else sent


def cmd_anno(a):
    dur = json.load(open(os.path.join(a.sta_dir, "Charades_duration.json")))
    have = lmdb_keys(a.clip_lmdb) & lmdb_keys(a.clap_lmdb)
    out = {}
    stats = {}
    for split in ("train", "test"):
        rows = read_sta(os.path.join(a.sta_dir, f"Charades_sta_{split}.txt"))
        kept = [r for r in rows if r[0] in have and r[0] in dur]
        stats[split] = (len(rows), len(kept), len({r[0] for r in kept}))
        if split == "train":
            per_vid, lst = {}, []
            for vid, s, e, sent in rows:
                if vid not in have or vid not in dur:
                    continue
                j = per_vid.get(vid, 0); per_vid[vid] = j + 1
                s, e = max(0.0, s), min(e, dur[vid])
                if e <= s:
                    continue
                lst.append({"id": vid, "query_id": f"{vid}_{j}",
                            "conversations": [{"from": "human", "value": PROMPT.format(norm_sentence(sent))},
                                              {"from": "gpt", "value": "From <s0> to <e0>."}],
                            "meta": {"duration": dur[vid], "token": {"<s0>": round(s, 1), "<e0>": round(e, 1)}},
                            "source": "charades", "sentence": sent})
            json.dump(lst, open(os.path.join(a.out_dir, "charades_train_anno.json"), "w"))
            json.dump(sorted({d["id"] for d in lst}), open(os.path.join(a.out_dir, "charades_train_ids.json"), "w"))
            out["train"] = len(lst)
        else:
            dct = {}
            for idx, (vid, s, e, sent) in enumerate(rows):
                if vid not in have or vid not in dur:
                    continue
                dct[f"c{idx}"] = {"movie": vid, "sentence": sent, "timestamps": [s, e], "movie_duration": dur[vid],
                                  "vid": vid, "split": "test"}
            json.dump(dct, open(os.path.join(a.out_dir, "charades_test_anno.json"), "w"))
            json.dump(sorted({d["movie"] for d in dct.values()}), open(os.path.join(a.out_dir, "charades_test_ids.json"), "w"))
            out["test"] = len(dct)
    for k, (n, kept, nv) in stats.items():
        print(f"{k}: {n} queries in the txt, {kept} with features ({nv} videos)")
    print("wrote", out)


def cmd_replay_ids(a):
    cand = json.load(open(a.candidates))
    dur = cand["durations"]
    have = lmdb_keys(a.clip_lmdb) & lmdb_keys(a.clap_lmdb)
    ids = sorted(v for v in cand["ids"] if v in have and dur[v] >= a.min_duration)
    json.dump(ids, open(a.out, "w"))
    print(f"{len(ids)} replay videos (duration >= {a.min_duration}s) of {len(have & set(cand['ids']))} stored Train videos -> {a.out}")


def dumps_npz(d):
    with io.BytesIO() as w:
        np.savez_compressed(w, **d)
        return w.getvalue()


def cmd_text(a):
    import lmdb
    import torch
    from revisionllm.data.feature_extraction.clip_extractor import ClipFeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = ClipFeatureExtractor(framerate=30, size=224, centercrop=True, model_name_or_path=a.clip_path, device=dev)
    jobs = []
    train = json.load(open(os.path.join(a.out_dir, "charades_train_anno.json")))
    jobs.append((a.qfeat_train, [(d["query_id"], norm_sentence(d["sentence"])) for d in train]))
    test = json.load(open(os.path.join(a.out_dir, "charades_test_anno.json")))
    jobs.append((a.qfeat_test, [(q, norm_sentence(d["sentence"])) for q, d in test.items()]))
    for path, items in jobs:
        env = lmdb.open(path, map_size=1 << 40, lock=False)
        for i in range(0, len(items), 60):
            chunk = items[i:i + 60]
            tok, eot = ext.encode_text([t for _, t in chunk])
            with env.begin(write=True) as txn:
                for (key, _), tf, ef in zip(chunk, tok, eot):
                    txn.put(key.encode(), dumps_npz({"cls_features": ef.detach().cpu().numpy().astype(np.float32),
                                                     "token_features": tf.detach().cpu().numpy().astype(np.float32)}))
        env.sync(); env.close()
        print(f"wrote {len(items)} query features -> {path}")


def cmd_mix(a):
    rng = random.Random(a.seed)
    cha = json.load(open(os.path.join(a.out_dir, "charades_train_anno.json")))
    rep_all = json.load(open(a.replay_anno))
    n_rep = int(round(a.replay_share / (1 - a.replay_share) * len(cha)))
    if len(rep_all) >= n_rep:
        rep = rng.sample(rep_all, n_rep)
    else:   # small replay pool: repeat the queries (rehearsal data is regularisation, repeats are fine)
        reps, rep = n_rep // len(rep_all), []
        rep = rep_all * reps + rng.sample(rep_all, n_rep - reps * len(rep_all))
    mix = cha + rep
    rng.shuffle(mix)
    json.dump(mix, open(a.out, "w"))
    share = len(rep) / len(mix)
    print(f"mix: {len(cha)} Charades + {len(rep)} VidChapters replay queries (wanted {n_rep}, pool {len(rep_all)}, repeated x{len(rep)/max(len(rep_all),1):.1f}) "
          f"= {len(mix)}; replay share {share:.1%}; distinct replay videos: {len({d['id'] for d in rep})}")
    json.dump({"charades": len(cha), "replay": len(rep), "replay_share": share}, open(a.out + ".stats.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("anno"); p.add_argument("--sta_dir", required=True); p.add_argument("--clip_lmdb", required=True)
    p.add_argument("--clap_lmdb", required=True); p.add_argument("--out_dir", required=True); p.set_defaults(f=cmd_anno)
    p = sub.add_parser("replay_ids"); p.add_argument("--candidates", required=True); p.add_argument("--clip_lmdb", required=True)
    p.add_argument("--clap_lmdb", required=True); p.add_argument("--min_duration", type=float, default=1000)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_replay_ids)
    p = sub.add_parser("text"); p.add_argument("--out_dir", required=True); p.add_argument("--clip_path", required=True)
    p.add_argument("--qfeat_train", required=True); p.add_argument("--qfeat_test", required=True); p.set_defaults(f=cmd_text)
    p = sub.add_parser("mix"); p.add_argument("--out_dir", required=True); p.add_argument("--replay_anno", required=True)
    p.add_argument("--replay_share", type=float, default=0.3); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True); p.set_defaults(f=cmd_mix)
    a = ap.parse_args()
    a.f(a)
