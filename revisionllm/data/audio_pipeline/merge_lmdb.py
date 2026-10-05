"""
Merge shard LMDBs (one per extraction worker) into the main CLIP / CLAP LMDB.

Workers write to their own LMDB because LMDB's file locking is not reliable when several processes
write the same database over NFS. Values are copied as raw bytes (already npz-compressed), keys that
the destination already has are skipped, so the merge can be re-run at any time.

    python merge_lmdb.py --dst data/chapters_audio/clip_l14_2fps_lmdb --src_glob "data/chapters_audio/shards/clip_train_*"
"""
import argparse
import glob
import os

import lmdb

ONE_TB = 1 << 40


def merge(dst_path, src_paths, batch=64, delete_src=False):
    os.makedirs(dst_path, exist_ok=True)
    dst = lmdb.open(dst_path, map_size=ONE_TB, subdir=True, lock=True, readahead=False)
    added = skipped = 0
    for sp in src_paths:
        if not os.path.isdir(sp):
            continue
        src = lmdb.open(sp, readonly=True, lock=False, readahead=False)
        pending = []
        with src.begin() as stxn, stxn.cursor() as cur:
            for k, v in cur:
                pending.append((bytes(k), bytes(v)))
                if len(pending) >= batch:
                    a, s = _flush(dst, pending); added += a; skipped += s; pending = []
            a, s = _flush(dst, pending); added += a; skipped += s
        src.close()
    dst.sync()
    dst.close()
    print(f"merged into {dst_path}: +{added} new, {skipped} already present")
    if delete_src:
        import shutil
        for sp in src_paths:
            shutil.rmtree(sp, ignore_errors=True)


def _flush(dst, items):
    added = skipped = 0
    if not items:
        return 0, 0
    with dst.begin(write=True) as txn:
        for k, v in items:
            if txn.get(k) is None:
                txn.put(k, v); added += 1
            else:
                skipped += 1
    return added, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--src_glob", required=True)
    ap.add_argument("--delete_src", action="store_true", help="remove the shard dirs after a successful merge")
    a = ap.parse_args()
    srcs = sorted(glob.glob(a.src_glob))
    print(f"{len(srcs)} shard(s) -> {a.dst}")
    merge(a.dst, srcs, delete_src=a.delete_src)
