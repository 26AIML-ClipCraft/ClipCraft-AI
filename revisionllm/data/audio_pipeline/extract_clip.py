"""
CLIP ViT-L/14 frame features at a fixed fps (2 fps for VidChapters, matching the
released ReVisionLLM chapters checkpoints and every scripts/chapters/*.sh).

Decoding uses ffmpeg (``VideoLoader`` from clip_extractor.py): the ``fps`` filter
emits one frame every 1/fps seconds starting at t=0, so stored row ``i`` is the
frame at ``i / fps`` seconds. The audio extractor uses the same time origin, so
``row / fps`` <-> ``floor(sec / hop)`` alignment in the loaders is exact.

Usage (single video, from download_and_extract.py):
    extractor = ClipFrameExtractor(clip_path, fps=2.0)
    feats = extractor(video_path)            # np.float16 (T, 768)
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.append(ROOT)

import clip  # openai/CLIP (pip install git+https://github.com/openai/CLIP.git)
from revisionllm.data.feature_extraction.clip_extractor import VideoLoader, Preprocessing


class ClipFrameExtractor:
    def __init__(self, clip_path: str = "ViT-L/14", fps: float = 2.0, batch_size: int = 256,
                 device: str = None, dtype=torch.float16):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(clip_path, device=self.device, jit=False)
        self.model.eval()
        self.loader = VideoLoader(framerate=fps, size=224, centercrop=True)
        self.preprocess = Preprocessing()
        self.batch_size = batch_size
        self.fps = fps
        self.dtype = dtype if self.device == "cuda" else torch.float32

    @torch.no_grad()
    def __call__(self, video_path: str) -> np.ndarray:
        frames = self.loader.read_video_from_file(video_path)  # (T, 3, 224, 224) float32 0..255
        if isinstance(frames, dict) or frames.dim() != 4 or frames.shape[0] == 0:
            raise RuntimeError(f"could not decode {video_path}")
        frames = self.preprocess(frames)
        out = []
        n_batch = int(math.ceil(frames.shape[0] / self.batch_size))
        for b in range(n_batch):
            x = frames[b * self.batch_size:(b + 1) * self.batch_size].to(self.device, dtype=self.dtype)
            out.append(self.model.encode_image(x).float().cpu())
        feats = torch.cat(out, dim=0).numpy().astype(np.float16)
        return feats  # (T, 768)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--clip_path", default="checkpoints/clip/ViT-L-14.pt")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="output .npy")
    a = ap.parse_args()
    f = ClipFrameExtractor(a.clip_path, fps=a.fps)(a.video)
    np.save(a.out, f)
    print(a.out, f.shape, f.dtype)
