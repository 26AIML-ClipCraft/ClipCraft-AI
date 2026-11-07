"""UniVTG highlight / summarisation scores on ClipCraft's cached CLIP ViT-L/14 features (no new feature extraction).

    AVRepresentation.vision (T, 768, one unit vector per second, from the L/14 cache)
      -> pairs of seconds averaged, L2 -> 2 s clips   (UniVTG's clip length; TVSum features are 2 s clips)
      -> Adapter Linear(768 -> 512), L2          (trained to map L/14 into UniVTG's CLIP-B/32 space, checkpoints/univtg/README.md)
      -> frozen UniVTG (CLIP-B/32-only checkpoint) with the CLIP-B/32 text feature of the title as query
      -> one score per clip, repeated to the 1 s grid of the editing module.

Which UniVTG output is used: the saliency score (cosine of the projected clip and the pooled query). On TVSum (35 videos, no TVSum
training) it was at least as good as saliency + foreground logit on every metric (mAP 0.580 vs 0.551, F1@15% 0.258 vs 0.243,
rho 0.258 vs 0.238; the differences were not tested for significance). The same value is returned as "saliency" and "foreground" so the
EditingService, which reads highlight = saliency + foreground and summary = foreground, only ever sees ranks of that one score;
the raw foreground logit is returned as "logit" for inspection.

Limits (measured, see the repository notes): validated on whole videos of up to ~324 clips (~11 min). Longer videos are scored in
chunks of equal size with chunk-relative position features; that path is NOT validated. The query must be a text (title or a
description); UniVTG has no query-free mode.
"""
import hashlib
import os
import types
import warnings
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "univtg", "univtg_ft_b32.pt")
DEFAULT_ADAPTER = os.path.join(ROOT, "checkpoints", "univtg", "adapter_align.pt")
TEXT_MODEL = "openai/clip-vit-base-patch32"       # the encoder that produced UniVTG's released text features


def l2(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """UniVTG's l2_normalize_np_array (utils/basic_utils.py)."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class UniVTGScorer:
    uses_text = True          # EditingService passes the query text instead of the CLIP-L/14 vector

    def __init__(self, ckpt: str = DEFAULT_CKPT, adapter: str = DEFAULT_ADAPTER, text_model: str = TEXT_MODEL, device: str = None,
                 max_clips: int = 330, text_cache: int = 64):
        from revisionllm.editing.univtg.model import Model
        from revisionllm.editing.univtg.position_encoding import build_position_encoding
        from revisionllm.editing.univtg.transformer import build_transformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location="cpu")
        o = ck["opt"]
        args = types.SimpleNamespace(**o)
        self.model = Model(build_transformer(args), *build_position_encoding(args), txt_dim=o["t_feat_dim"], vid_dim=o["v_feat_dim"] + 2,
                           input_dropout=o["input_dropout"], span_loss_type=o["span_loss_type"], use_txt_pos=o["use_txt_pos"],
                           n_input_proj=o["n_input_proj"])
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.to(self.device).eval()
        ad = torch.load(adapter, map_location="cpu")
        self.adapter = nn.Linear(768, o["v_feat_dim"])
        self.adapter.load_state_dict(ad["state_dict"])
        self.adapter.to(self.device).eval()
        self.max_clips, self._text_model, self._tok, self._enc = max_clips, text_model, None, None
        self._cache, self._cache_n, self._warned = OrderedDict(), text_cache, False
        sig = [f"{os.path.basename(p)}|{os.path.getsize(p)}|{int(os.stat(p).st_mtime)}" for p in (ckpt, adapter)]
        self.version = "univtg-sal|" + hashlib.sha1("|".join(sig + [text_model, "pool-first"]).encode()).hexdigest()[:12]

    # ---- query ------------------------------------------------------------------------------------------------
    def text_feature(self, text: str) -> np.ndarray:
        """CLIP-B/32 token features (L, 512) of the text: the final-LayerNorm output without the projection, tokens up to EOT,
        L2-normalised per token. Identical to UniVTG's released extractor (token cosine 1.0000 on 100 queries)."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        if self._enc is None:
            from transformers import CLIPTextModel, CLIPTokenizer
            self._tok = CLIPTokenizer.from_pretrained(self._text_model)
            self._enc = CLIPTextModel.from_pretrained(self._text_model).to(self.device).eval()
        t = self._tok(text, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(self.device)
        with torch.no_grad():
            h = self._enc(**t).last_hidden_state[0].float().cpu().numpy()
        f = l2(h[: int(t["attention_mask"][0].sum())]).astype(np.float32)
        self._cache[text] = f
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return f

    # ---- video ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def clips(self, vision: np.ndarray) -> np.ndarray:
        """(T, 768) unit vectors per second -> (ceil(T/2), 512) UniVTG-space 2 s clips."""
        v = np.asarray(vision, np.float32)
        n = (len(v) + 1) // 2
        c = l2(np.stack([v[2 * i: 2 * i + 2].mean(0) for i in range(n)]))          # pool first: 2 s of L/14 features
        a = self.adapter(torch.from_numpy(c.astype(np.float32)).to(self.device)).float().cpu().numpy()
        return l2(a).astype(np.float32)

    @torch.no_grad()
    def _run(self, clips: np.ndarray, q: np.ndarray):
        n = len(clips)
        st = np.arange(n, dtype=np.float32) / n                                   # UniVTG's temporal-endpoint feature
        v = torch.from_numpy(np.concatenate([clips, np.stack([st, st + 1.0 / n], 1)], 1)).to(self.device)[None]
        qt = torch.from_numpy(q).to(self.device)[None]
        out = self.model(src_txt=qt, src_txt_mask=torch.ones(qt.shape[:2], dtype=torch.bool, device=self.device),
                         src_vid=v, src_vid_mask=torch.ones(v.shape[:2], dtype=torch.bool, device=self.device))
        return out["saliency_scores"][0].float().cpu().numpy(), out["pred_logits"][0, :, 0].float().cpu().numpy()

    def predict(self, av, query=None, text: str = None):
        """-> {'saliency': (T,), 'foreground': (T,), 'logit': (T,)} on av's 1 s grid. `query` (the L/14 vector) is ignored."""
        if not text or not text.strip():
            raise ValueError("UniVTG needs a text query (title or description); it has no query-free mode")
        T = len(av)
        clips = self.clips(av.vision)
        n = len(clips)
        q = self.text_feature(text.strip())
        if n <= self.max_clips:
            sal, lg = self._run(clips, q)
        else:
            if not self._warned:
                warnings.warn(f"video longer than {self.max_clips} clips ({2 * self.max_clips} s): scored in chunks, which has not been validated")
                self._warned = True
            k = -(-n // self.max_clips)
            size = -(-n // k)
            parts = [self._run(clips[i: i + size], q) for i in range(0, n, size)]
            sal, lg = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        sal_s, lg_s = np.repeat(sal, 2)[:T], np.repeat(lg, 2)[:T]
        return {"saliency": sal_s, "foreground": sal_s.copy(), "logit": lg_s}
