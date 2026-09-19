"""
CLAP audio features with a fixed 1 s window / 1 s hop (ClipCraft fixed setting).

* Model: ``laion/clap-htsat-fused`` via transformers ``ClapModel`` /
  ``ClapProcessor``. The 512-d *projected* audio embedding
  (``get_audio_features``) is stored, i.e. the same space as CLAP's text
  encoder, which leaves the door open for a CLAP-text query path later.
* Audio is decoded with ffmpeg to 48 kHz mono float32 starting at t=0, so
  window ``i`` covers seconds ``[i, i+1)`` and aligns with CLIP row ``i * fps``.
* The processor pads/repeats each 1 s clip to CLAP's 10 s input (``repeatpad``);
  windows are batched to keep the GPU busy.
* Silent windows are stored as-is (their CLAP embedding is distinctive); the
  loader does not treat them specially.

Returns np.float16 (T_audio, 512) with T_audio = ceil(duration / hop).
"""
import math
import subprocess

import numpy as np
import torch

CLAP_SR = 48000


def decode_audio_ffmpeg(path: str, sr: int = CLAP_SR) -> np.ndarray:
    """Decode any container (mp4 / m4a / webm ...) to mono float32 PCM at ``sr``."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


class ClapAudioExtractor:
    def __init__(self, model_name: str = "laion/clap-htsat-fused", window_sec: float = 1.0,
                 hop_sec: float = 1.0, batch_size: int = 64, device: str = None):
        from transformers import ClapModel, ClapProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.window = int(round(window_sec * CLAP_SR))
        self.hop = int(round(hop_sec * CLAP_SR))
        self.hop_sec = hop_sec
        self.batch_size = batch_size
        self.dim = self.model.config.projection_dim

    def windows(self, wav: np.ndarray):
        n = max(1, int(math.ceil(len(wav) / self.hop)))
        for i in range(n):
            s = i * self.hop
            chunk = wav[s:s + self.window]
            if len(chunk) < self.window:
                chunk = np.pad(chunk, (0, self.window - len(chunk)))
            yield chunk

    @torch.no_grad()
    def from_waveform(self, wav: np.ndarray) -> np.ndarray:
        feats = []
        batch = []

        def flush():
            if not batch:
                return
            inputs = self.processor(audios=batch, sampling_rate=CLAP_SR, return_tensors="pt", padding="repeatpad")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            emb = self.model.get_audio_features(**inputs)  # (B, 512), already projected
            feats.append(emb.float().cpu().numpy())
            batch.clear()

        for chunk in self.windows(wav):
            batch.append(chunk)
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return np.concatenate(feats, axis=0).astype(np.float16)  # (T_audio, 512)

    def __call__(self, media_path: str) -> np.ndarray:
        wav = decode_audio_ffmpeg(media_path)
        if len(wav) == 0:
            raise RuntimeError(f"no audio stream in {media_path}")
        return self.from_waveform(wav)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True, help="video or audio file")
    ap.add_argument("--out", required=True, help="output .npy")
    ap.add_argument("--model", default="laion/clap-htsat-fused")
    a = ap.parse_args()
    f = ClapAudioExtractor(a.model)(a.media)
    np.save(a.out, f)
    print(a.out, f.shape, f.dtype)
