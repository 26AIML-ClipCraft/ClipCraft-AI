"""Query-independent features (CLIP frames + CLAP audio) with a SHA-256 keyed file cache (Fig. 5, 3.1)."""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(ROOT)

from revisionllm.multiquery.config import CacheConfig
from revisionllm.multiquery.db import CacheDB


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probe_duration(path: str) -> float:
    import ffmpeg
    return float(ffmpeg.probe(path)["format"]["duration"])


class VideoFeatures:
    """vision: (T, 768) float16 at feature_fps; audio: (Ta, 512) float16 at 1/audio_hop_sec, or None (no audio)."""

    def __init__(self, video_id, video_hash, vision, audio, duration, cache_hit):
        self.video_id, self.video_hash = video_id, video_hash
        self.vision, self.audio, self.duration, self.cache_hit = vision, audio, duration, cache_hit


class FeatureCache:
    def __init__(self, cfg: CacheConfig, db: CacheDB, clip_pt: str, device: str = None):
        self.cfg, self.db, self.clip_pt, self.device = cfg, db, clip_pt, device
        self._clip = None
        self._clap = None
        self.n_extractions = 0          # incremented per video whose features were actually computed

    def _extract(self, video_path):
        from revisionllm.data.audio_pipeline.extract_clip import ClipFrameExtractor
        from revisionllm.data.audio_pipeline.extract_clap import ClapAudioExtractor
        if self._clip is None:
            self._clip = ClipFrameExtractor(self.clip_pt, fps=self.cfg.feature_fps, device=self.device)
            self._clap = ClapAudioExtractor(self.cfg.clap_name, hop_sec=self.cfg.audio_hop_sec, device=self.device)
        # transformers' fused CLAP front-end draws a random mel chunk per window; seed it so a re-extraction is reproducible
        np.random.seed(0)
        vision = self._clip(video_path)
        try:
            audio = self._clap(video_path)
        except Exception as e:           # no audio stream: the model then gets zero audio, like the eval loaders
            print(f"[multiquery] no audio features for {video_path}: {e}")
            audio = None
        return vision, audio

    def _paths(self, video_hash):
        d = os.path.join(self.cfg.cache_dir, video_hash)
        return d, os.path.join(d, "vision.npy"), os.path.join(d, "audio.npy")

    def _load(self, row):
        vision = np.load(row["vision_feature"])
        audio = np.load(row["audio_feature"]) if row["audio_feature"] else None
        return vision, audio

    def get(self, video_path: str) -> VideoFeatures:
        h = sha256_file(video_path)
        row = self.db.get_video(h)
        if row is not None and row["feature_version"] == self.cfg.feature_version \
                and os.path.exists(row["vision_feature"]) \
                and (row["audio_feature"] is None or os.path.exists(row["audio_feature"])):
            vision, audio = self._load(row)
            return VideoFeatures(row["video_id"], h, vision, audio, row["duration"], True)
        # miss, stale version, or files gone: extract, store, (re)register
        vision, audio = self._extract(video_path)
        self.n_extractions += 1
        d, vp, ap = self._paths(h)
        os.makedirs(d, exist_ok=True)
        np.save(vp + ".tmp.npy", vision); os.replace(vp + ".tmp.npy", vp)
        if audio is not None:
            np.save(ap + ".tmp.npy", audio); os.replace(ap + ".tmp.npy", ap)
        duration = probe_duration(video_path)
        vid = self.db.put_video(h, vp, ap if audio is not None else None, duration, self.cfg.feature_version)
        return VideoFeatures(vid, h, vision, audio, duration, False)
