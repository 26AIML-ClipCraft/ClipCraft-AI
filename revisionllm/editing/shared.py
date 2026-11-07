"""The one audio-visual representation all three features (grounding, recommendation, highlight) read.

It is a view of the cached CLIP / CLAP arrays of multiquery.FeatureCache on a common 1-second grid; nothing is encoded here.
"""
import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def to_second_grid(vision, audio, feature_fps: float = 2.0, audio_hop_sec: float = 1.0):
    """vision (Tv, 768) at feature_fps, audio (Ta, 512) at 1/audio_hop_sec or None -> L2-normalised (T, 768), (T, 512).

    T is the number of whole seconds covered by the vision features. Missing audio seconds are zero vectors."""
    vision = np.asarray(vision, dtype=np.float32)
    per = max(int(round(feature_fps)), 1)
    T = max(len(vision) // per, 1)
    if len(vision) < per:
        v = vision.mean(0, keepdims=True)
    else:
        v = vision[:T * per].reshape(T, per, -1).mean(1)
    a = np.zeros((T, 512 if audio is None else np.asarray(audio).shape[1]), dtype=np.float32)
    if audio is not None and len(audio):
        audio = np.asarray(audio, dtype=np.float32)
        rows = np.clip(np.floor(np.arange(T) / audio_hop_sec).astype(np.int64), 0, len(audio) - 1)
        a = audio[rows]
    return _unit(v), _unit(a) if audio is not None else a


class AVRepresentation:
    """vision (T, 768) and audio (T, 512), unit vectors, one row per second; has_audio=False -> audio rows are zero."""

    def __init__(self, vision, audio, duration, video_id=None, video_hash=None, feature_version=None):
        self.vision, self.audio = vision, audio
        self.duration = float(duration)
        self.video_id, self.video_hash, self.feature_version = video_id, video_hash, feature_version
        self.has_audio = bool(np.abs(audio).sum() > 0)

    def __len__(self):
        return len(self.vision)

    @classmethod
    def from_features(cls, feats, cfg) -> "AVRepresentation":
        """feats: multiquery.feature_cache.VideoFeatures, cfg: multiquery.config.CacheConfig."""
        v, a = to_second_grid(feats.vision, feats.audio, cfg.feature_fps, cfg.audio_hop_sec)
        return cls(v, a, feats.duration, feats.video_id, feats.video_hash, cfg.feature_version)
