"""
ClipCraft: shared helpers for reading CLAP features at evaluation time and
aligning them to the CLIP frame grid exactly like the training loader does
(``LazySupervisedDataset.align_audio_to_frames``).
"""
import io
import lmdb
import numpy as np
import torch


class AudioStore:
    """Read-only LMDB of CLAP features: key = video id, value = npz{'features': (T, D)}."""

    def __init__(self, path, audio_dim=512, hop_sec=1.0, feature_fps=2.0):
        self.path = path
        self.audio_dim = audio_dim
        self.hop_sec = float(hop_sec)
        self.feature_fps = float(feature_fps)
        self.env = None
        self.txn = None
        self._missing = set()
        if path is not None:
            self.env = lmdb.open(path, readonly=True, create=False, max_readers=4096 * 8, readahead=False, lock=False)
            self.txn = self.env.begin(buffers=True)

    @property
    def enabled(self):
        return self.txn is not None

    def get(self, vid):
        if not self.enabled or vid in self._missing:
            return None
        dump = self.txn.get(vid.encode())
        if dump is None:
            self._missing.add(vid)
            return None
        with io.BytesIO(dump) as reader:
            a = np.load(reader, allow_pickle=True)['features']
        return np.asarray(a, dtype=np.float32)

    def align(self, audio, frame_indices):
        """frame index (into the stored CLIP array) -> nearest 1-s audio row. Zeros if audio is None."""
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        if audio is None or len(audio) == 0:
            return np.zeros((len(frame_indices), self.audio_dim), dtype=np.float32)
        rows = np.floor((frame_indices.astype(np.float64) / self.feature_fps) / self.hop_sec).astype(np.int64)
        rows = np.clip(rows, 0, len(audio) - 1)
        return audio[rows]


def to_model_tensor(np_array, like: torch.Tensor):
    """Match the dtype/device of the visual feature tensor handed to the model."""
    t = torch.from_numpy(np.ascontiguousarray(np_array))
    return t.to(dtype=like.dtype, device=like.device)


def apply_drop_flags(features, audio_feats, drop_visual=False, drop_audio=False):
    """Ablation switches: zero a modality at inference time (R3 / R4 rows)."""
    if drop_visual and features is not None:
        features = torch.zeros_like(features)
    if drop_audio and audio_feats is not None:
        audio_feats = torch.zeros_like(audio_feats)
    return features, audio_feats
