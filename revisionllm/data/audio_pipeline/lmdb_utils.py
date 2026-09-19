"""
LMDB helpers shared by the feature extractors and the training/eval loaders.

Layout (identical for CLIP and CLAP stores):
    key   = <video id>.encode()
    value = np.savez_compressed({'features': float16 (T, D)})
which is exactly what ``LazySupervisedDataset._get_video_appearance_feat_by_vid``
and ``AudioStore.get`` read back (``np.load(...)['features']``).
"""
import io
import os
import lmdb
import numpy as np

ONE_TB = 1 << 40


def dumps_npz(features: np.ndarray, compress: bool = True) -> bytes:
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, features=features)
        else:
            np.savez(writer, features=features)
        return writer.getvalue()


def loads_npz(buf) -> np.ndarray:
    with io.BytesIO(buf) as reader:
        return np.load(reader, allow_pickle=True)['features']


class LmdbWriter:
    """Append-only writer with existence checks (safe to re-run: existing keys are skipped)."""

    def __init__(self, path: str, map_size: int = ONE_TB):
        os.makedirs(path, exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size, subdir=True, lock=True, readahead=False)

    def has(self, key: str) -> bool:
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def put(self, key: str, features: np.ndarray, dtype=np.float16) -> None:
        feats = np.ascontiguousarray(features).astype(dtype)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), dumps_npz(feats))

    def keys(self):
        with self.env.begin() as txn:
            with txn.cursor() as cur:
                for k, _ in cur:
                    yield bytes(k).decode()

    def close(self):
        self.env.sync()
        self.env.close()


def lmdb_keys(path: str):
    """All keys of a read-only store (used by finalize_subsets.py / resume logic)."""
    if not os.path.isdir(path):
        return set()
    env = lmdb.open(path, readonly=True, lock=False, readahead=False)
    keys = set()
    with env.begin() as txn:
        with txn.cursor() as cur:
            for k, _ in cur:
                keys.add(bytes(k).decode())
    env.close()
    return keys
