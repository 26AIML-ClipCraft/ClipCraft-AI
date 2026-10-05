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


class _Uint8Loader(VideoLoader):
    """Same ffmpeg filter chain as VideoLoader.read_video_from_file (fps -> scale -> centre crop), so the
    decoded pixels are identical, but it returns the raw uint8 (T, H, W, 3) array instead of a float32
    copy (4x less RAM) and lets the caller cap the ffmpeg decoder threads (several workers share the CPU)."""

    def __init__(self, *args, threads: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads = threads

    def read_uint8(self, video_path):
        import ffmpeg
        try:
            info = self._get_video_info(video_path)
            h, w = info["height"], info["width"]
        except Exception:
            return None
        height, width = self._get_output_dim(h, w)
        fps = self.framerate
        try:
            duration = info["duration"]
            if duration > 0 and duration < 1 / fps + 0.1:
                fps = 2 / max(int(duration), 1)
        except Exception:
            fps = self.framerate
        src = ffmpeg.input(video_path, **({"threads": self.threads} if self.threads else {}))
        cmd = src.filter('fps', fps=fps).filter('scale', width, height)
        if self.centercrop:
            x = int((width - self.size) / 2.0)
            y = int((height - self.size) / 2.0)
            cmd = cmd.crop(x, y, self.size, self.size)
        out, _ = cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True, quiet=True)
        if self.centercrop and isinstance(self.size, int):
            height, width = self.size, self.size
        return np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])


class ClipFrameExtractor:
    def __init__(self, clip_path: str = "ViT-L/14", fps: float = 2.0, batch_size: int = 256,
                 device: str = None, dtype=torch.float16, ffmpeg_threads: int = 0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(clip_path, device=self.device, jit=False)
        self.model.eval()
        self.loader = _Uint8Loader(framerate=fps, size=224, centercrop=True, threads=ffmpeg_threads)
        self.preprocess = Preprocessing()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        self.batch_size = batch_size
        self.fps = fps
        self.dtype = dtype if self.device == "cuda" else torch.float32

    @torch.no_grad()
    def __call__(self, video_path: str) -> np.ndarray:
        frames = self.loader.read_uint8(video_path)  # (T, 224, 224, 3) uint8
        if frames is None or frames.ndim != 4 or frames.shape[0] == 0:
            raise RuntimeError(f"could not decode {video_path}")
        out = []
        for b in range(int(math.ceil(frames.shape[0] / self.batch_size))):
            x = torch.from_numpy(frames[b * self.batch_size:(b + 1) * self.batch_size]).to(self.device)
            x = x.permute(0, 3, 1, 2).float()
            x = (x / 255.0 - self.mean) / (self.std + 1e-8)   # same maths as Preprocessing, on the GPU
            out.append(self.model.encode_image(x.to(self.dtype)).float().cpu())
        return torch.cat(out, dim=0).numpy().astype(np.float16)  # (T, 768)

    @torch.no_grad()
    def reference(self, video_path: str) -> np.ndarray:
        """Original float32-on-CPU path (kept only to check that __call__ gives the same features)."""
        frames = VideoLoader(framerate=self.fps, size=224, centercrop=True).read_video_from_file(video_path)
        frames = self.preprocess(frames)
        out = []
        for b in range(int(math.ceil(frames.shape[0] / self.batch_size))):
            x = frames[b * self.batch_size:(b + 1) * self.batch_size].to(self.device, dtype=self.dtype)
            out.append(self.model.encode_image(x).float().cpu())
        return torch.cat(out, dim=0).numpy().astype(np.float16)


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
