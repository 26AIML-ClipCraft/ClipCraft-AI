"""UniVTG-style unified heads on the cached CLIP-L/14 + CLAP features (1 s grid).

UniVTG (showlab/UniVTG) predicts, per clip, a foreground score, a boundary offset (distance to the start / end of the
target moment) and a saliency score from one shared encoder, so moment retrieval, highlight detection and summarisation
are read off the same outputs. The released weights expect CLIP-B/16 features and have no audio input, so the structure
is re-implemented here and trained on this repository's features (train_head.py). The losses and the way the outputs are
read follow the reference code (main/inference_hl.py, main/inference_qfvs.py): foreground = class-weighted cross entropy
(background weight eos_coef=0.1), saliency = margin ranking between a clip inside the target moment and one outside,
boundary = L1 + gIoU; highlight score = saliency + foreground logit, summary score = foreground logit.
Highlight extraction and summarisation give the head the video's title (or a VLM description of it) as the query; a learned
null query exists for query=None but is not trained on purpose, so do not rely on it.
"""
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    vis_dim: int = 768
    aud_dim: int = 512
    query_dim: int = 768
    d_model: int = 256
    heads: int = 4
    layers: int = 3
    ff: int = 1024
    dropout: float = 0.1
    audio_drop: float = 0.15          # training: share of samples whose audio is zeroed (robust to silent / audio-less video)
    max_offset: float = 256.0         # seconds


def sinusoid(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class AVUniVTG(nn.Module):
    def __init__(self, cfg: HeadConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or HeadConfig()
        d = cfg.d_model
        self.v_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.a_proj = nn.Sequential(nn.LayerNorm(cfg.aud_dim), nn.Linear(cfg.aud_dim, d))
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.q_proj = nn.Sequential(nn.LayerNorm(cfg.query_dim), nn.Linear(cfg.query_dim, d))
        self.null_query = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.saliency = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.foreground = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.offset = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, vision, audio, query=None, mask=None):
        """vision (B,T,768), audio (B,T,512), query (B,768) | None, mask (B,T) bool True = real frame.
        Returns saliency / foreground logits (B,T) and offsets (B,T,2) in seconds (distance back to the start, forward to the end)."""
        B, T, _ = vision.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=vision.device)
        if self.training and self.cfg.audio_drop > 0:
            keep = (torch.rand(B, 1, 1, device=vision.device) >= self.cfg.audio_drop).to(audio.dtype)
            audio = audio * keep
        x = self.fuse(torch.cat([self.v_proj(vision), self.a_proj(audio)], dim=-1))
        x = x + sinusoid(T, x.shape[-1], x.device)[None]
        q = self.null_query.expand(B, -1) if query is None else self.q_proj(query)
        seq = torch.cat([q[:, None], x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1)
        h = self.norm(self.encoder(seq, src_key_padding_mask=pad))[:, 1:]
        offsets = F.softplus(self.offset(h)).clamp(max=self.cfg.max_offset)
        return {"saliency": self.saliency(h).squeeze(-1), "foreground": self.foreground(h).squeeze(-1), "offsets": offsets}


def giou_1d(pred_l, pred_r, gt_l, gt_r, eps=1e-6):
    """Intervals [t - l, t + r]; returns generalised IoU per frame."""
    pl, pr, gl, gr = -pred_l, pred_r, -gt_l, gt_r
    inter = (torch.min(pr, gr) - torch.max(pl, gl)).clamp(min=0)
    union = (pr - pl) + (gr - gl) - inter
    enclose = torch.max(pr, gr) - torch.min(pl, gl)
    return inter / (union + eps) - (enclose - union) / (enclose + eps)


LOSS_COEF = {"f": 10.0, "s": 0.05, "b": 10.0, "g": 1.0}     # scripts/qvhl_pretrain.sh
EOS_COEF = 0.1                                                  # weight of the background class in the foreground loss
SAL_MARGIN = 1.0


def univtg_loss(out, tgt, coef=None):
    """tgt: mask (B,T) bool, has_query (B,) bool, foreground (B,T) {0,1}, offsets (B,T,2) seconds,
    sal_pos / sal_neg (B,P) long clip indices and sal_valid (B,P) bool (a pair is invalid when no clip lies outside).
    The offset term counts only for samples with a text query (a target moment exists); foreground and saliency count for all."""
    c = coef or LOSS_COEF
    m = tgt["mask"]
    zero = out["saliency"].sum() * 0.0

    if m.any():
        y = tgt["foreground"][m]
        w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, EOS_COEF))
        fg = (F.binary_cross_entropy_with_logits(out["foreground"][m], y, reduction="none") * w).sum() / w.sum()
    else:
        fg = zero

    valid = tgt["sal_valid"]
    if valid.any():
        sal_s = out["saliency"]
        pos, neg = sal_s.gather(1, tgt["sal_pos"]), sal_s.gather(1, tgt["sal_neg"])
        sal = (torch.clamp(SAL_MARGIN + neg - pos, min=0) * valid).sum() / valid.sum()
    else:
        sal = zero

    pos_fr = m & tgt["has_query"][:, None] & (tgt["foreground"] > 0.5)
    if pos_fr.any():
        po, to = out["offsets"][pos_fr], tgt["offsets"][pos_fr]
        l1 = F.l1_loss(po / 64.0, to / 64.0)
        giou = (1 - giou_1d(po[:, 0], po[:, 1], to[:, 0], to[:, 1])).mean()
    else:
        l1 = giou = zero
    total = c["f"] * fg + c["s"] * sal + c["b"] * l1 + c["g"] * giou
    return total, {"fg": float(fg), "sal": float(sal), "l1": float(l1), "giou": float(giou)}


# ---- checkpoint / inference ---------------------------------------------------------------------------------------

def save_head(path: str, model: AVUniVTG, feature_version: str, extra: dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "feature_version": feature_version,
                **(extra or {})}, path)


class HeadScorer:
    """Runs a trained head over a whole video with overlapping windows (the head is trained on crops)."""

    def __init__(self, ckpt_path: str, device: str = None, window: int = 256, stride: int = 128):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.model = AVUniVTG(HeadConfig(**ck["cfg"]))
        self.model.load_state_dict(ck["state"])
        self.feature_version = ck.get("feature_version")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.window, self.stride = window, stride
        st = os.stat(ckpt_path)
        self.version = f"{os.path.basename(ckpt_path)}|{st.st_size}|{int(st.st_mtime)}"

    @torch.no_grad()
    def predict(self, av, query: np.ndarray = None):
        """-> {'saliency': (T,), 'foreground': (T,)} raw scores (logits), averaged over overlapping windows. Raw, because the
        reference adds the two raw scores for highlight detection."""
        T = len(av)
        starts = list(range(0, max(T - self.window, 0) + 1, self.stride))
        if starts[-1] + self.window < T:
            starts.append(T - self.window)
        acc = {k: np.zeros(T) for k in ("saliency", "foreground")}
        cnt = np.zeros(T)
        q = None if query is None else torch.from_numpy(np.asarray(query, np.float32))[None].to(self.device)
        for s in starts:
            e = min(s + self.window, T)
            v = torch.from_numpy(av.vision[s:e])[None].to(self.device)
            a = torch.from_numpy(av.audio[s:e])[None].to(self.device)
            out = self.model(v, a, q)
            for k in acc:
                acc[k][s:e] += out[k][0].float().cpu().numpy()
            cnt[s:e] += 1
        return {k: acc[k] / np.maximum(cnt, 1) for k in acc}
